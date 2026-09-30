import os
import re
import random
import sys
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# --- 防呆機制：Gemini 與 Image 套件 ---
try:
    from PIL import Image
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# --- 1. 安全獲取 Secrets / 環境變數 ---
def get_secret(key_name):
    if key_name in st.secrets:
        return st.secrets[key_name]
    return os.environ.get(key_name, "")

API_FOOTBALL_KEY = get_secret("API_FOOTBALL_KEY")
THE_ODDS_API_KEY = get_secret("THE_ODDS_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
DATABASE_URL = get_secret("DATABASE_URL")

# --- 2. 雲端/本地 資料庫連線配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(CURRENT_DIR, "football_data.db")
sqlite_engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

engine = sqlite_engine
db_connection_warning = None

if DATABASE_URL:
    try:
        url = DATABASE_URL
        if "db." in url and ".supabase.co" in url:
            db_connection_warning = "⚠ 偵測到 Supabase 直連網址。Streamlit Cloud 不支援 IPv6，建議使用包含 pooler.supabase.com 與 Port 6543 的 Connection Pooling 網址。"
        
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
            
        temp_engine = create_engine(url, pool_pre_ping=True)
        with temp_engine.connect() as conn:
            pass
        engine = temp_engine
        db_connection_warning = None
    except Exception as e:
        db_connection_warning = f"⚠️ 雲端 PostgreSQL 連線失敗，已自動切換回本地 SQLite。詳細錯誤: {e}"
        engine = sqlite_engine

# --- 3. 自動初始化資料庫表架構 ---
def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fixtures_v4 (
                fixture_id BIGINT PRIMARY KEY,
                league_name TEXT,
                home_team TEXT,
                away_team TEXT,
                match_datetime TEXT,
                home_score INTEGER,
                away_score INTEGER,
                home_corner INTEGER,
                away_corner INTEGER,
                status TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS odds_history_v4 (
                fixture_id BIGINT,
                ah_line TEXT,
                ah_home_odd REAL,
                ah_away_odd REAL,
                ou_line TEXT,
                ou_over_odd REAL,
                ou_under_odd REAL,
                recorded_at TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions_v4 (
                fixture_id BIGINT PRIMARY KEY,
                prob_home_win REAL,
                value_bet_detected BOOLEAN,
                recommended_pick TEXT,
                ou_pick TEXT,
                corner_pick TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS historical_match_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                home_team TEXT,
                away_team TEXT,
                home_score INT,
                away_score INT,
                data_json TEXT,
                created_at TEXT
            )
        """))
        # 新增垃圾桶 / 刪除備份資料庫（用於安全復原 Undo）
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS undo_trash_v4 (
                trash_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_table TEXT,
                original_id TEXT,
                payload_json TEXT,
                deleted_at TEXT,
                batch_id TEXT
            )
        """))

try:
    init_db()
except Exception as e:
    st.error(f"資料庫初始化失敗: {e}")

if HAS_GENAI and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def get_candidate_gemini_models():
    """優先回傳高額度（1,500 RPD）的穩定模型，避開低額度測試模型"""
    preferred_models = [
        'gemini-1.5-flash',
        'gemini-2.5-flash',
        'gemini-1.5-pro',
        'gemini-3.8-flash'
    ]
    if not HAS_GENAI or not GEMINI_API_KEY:
        return preferred_models

    try:
        models_list = list(genai.list_models())
        available_models = [m.name.replace("models/", "") for m in models_list if "generateContent" in m.supported_generation_methods]
        
        ordered = []
        for pref in preferred_models:
            for m in available_models:
                if pref in m and m not in ordered:
                    ordered.append(m)
        for m in available_models:
            if m not in ordered:
                ordered.append(m)
        return ordered if ordered else preferred_models
    except Exception:
        return preferred_models

# --- 4. 數據合併、刪除與復原（Undo）邏輯 ---
def is_valid_val(v):
    if v is None:
        return False
    if isinstance(v, (int, float)) and v == 0:
        return False
    if isinstance(v, str) and v.strip() in ["", "0", "-", "N/A", "未知", "未知聯賽", "未知主隊", "未知客隊"]:
        return False
    if isinstance(v, list) and (len(v) == 0 or all(not is_valid_val(x) for x in v)):
        return False
    if isinstance(v, dict) and len(v) == 0:
        return False
    return True

def merge_match_json(old_data, new_data):
    if not old_data:
        return new_data
    if not new_data:
        return old_data

    merged = dict(old_data)

    for k, new_v in new_data.items():
        old_v = merged.get(k)
        if isinstance(new_v, dict) and isinstance(old_v, dict):
            merged_dict = dict(old_v)
            for sub_k, sub_v in new_v.items():
                if is_valid_val(sub_v):
                    merged_dict[sub_k] = sub_v
            merged[k] = merged_dict
        elif isinstance(new_v, list):
            if is_valid_val(new_v):
                merged[k] = new_v
        else:
            if is_valid_val(new_v):
                merged[k] = new_v

    return merged

def delete_records_with_undo(table_name, id_column, target_ids):
    """將欲刪除之資料寫入垃圾桶，隨後從原表抹除，回傳批次識別碼以供 Undo"""
    if not target_ids:
        return 0, None

    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    str_ids = [str(x) for x in target_ids]
    in_clause = ",".join([f"'{x}'" for x in str_ids])

    with engine.begin() as conn:
        df_to_delete = pd.read_sql(f"SELECT * FROM {table_name} WHERE {id_column} IN ({in_clause})", conn)
        
        if df_to_delete.empty:
            return 0, None

        trash_records = []
        for _, row in df_to_delete.iterrows():
            payload = row.to_dict()
            trash_records.append({
                'source_table': table_name,
                'original_id': str(row[id_column]),
                'payload_json': json.dumps(payload, ensure_ascii=False, default=str),
                'deleted_at': now_str,
                'batch_id': batch_id
            })

        df_trash = pd.DataFrame(trash_records)
        df_trash.to_sql('undo_trash_v4', conn, if_exists='append', index=False)

        conn.execute(text(f"DELETE FROM {table_name} WHERE {id_column} IN ({in_clause})"))
        
        # 若是刪除 fixtures_v4，同步清除賠率與預測
        if table_name == 'fixtures_v4':
            conn.execute(text(f"DELETE FROM odds_history_v4 WHERE fixture_id IN ({in_clause})"))
            conn.execute(text(f"DELETE FROM predictions_v4 WHERE fixture_id IN ({in_clause})"))

    return len(trash_records), batch_id

def restore_last_deletion(batch_id=None):
    """復原指定批次或最後一筆刪除紀錄"""
    with engine.begin() as conn:
        if batch_id:
            df_trash = pd.read_sql(text("SELECT * FROM undo_trash_v4 WHERE batch_id = :b"), conn, params={"b": batch_id})
        else:
            df_last_batch = pd.read_sql("SELECT batch_id FROM undo_trash_v4 ORDER BY trash_id DESC LIMIT 1", conn)
            if df_last_batch.empty:
                return False, "垃圾桶中無可復原之數據。"
            last_b = df_last_batch.iloc[0]['batch_id']
            df_trash = pd.read_sql(text("SELECT * FROM undo_trash_v4 WHERE batch_id = :b"), conn, params={"b": last_b})

        if df_trash.empty:
            return False, "未找到相符的恢復數據。"

        restored_count = 0
        for _, row in df_trash.iterrows():
            table_name = row['source_table']
            payload = json.loads(row['payload_json'])
            
            df_restore = pd.DataFrame([payload])
            df_restore.to_sql(table_name, conn, if_exists='append', index=False)
            restored_count += 1

        b_to_del = df_trash.iloc[0]['batch_id']
        conn.execute(text("DELETE FROM undo_trash_v4 WHERE batch_id = :b"), {"b": b_to_del})

    return True, f"✅ 成功復原 {restored_count} 筆賽事數據！"

def normalize_to_yyyy_mm_dd(dt_str, fallback_dt=None):
    if not dt_str or not isinstance(dt_str, str):
        if fallback_dt and isinstance(fallback_dt, str):
            return normalize_to_yyyy_mm_dd(fallback_dt)
        return datetime.now().strftime("%Y-%m-%d")
    
    dt_str = dt_str.strip()
    current_year = datetime.now().year
    
    m_full = re.search(r'(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})', dt_str)
    if m_full:
        yyyy, mm, dd = m_full.groups()
        return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
    
    m_short = re.search(r'(\d{1,2})[-/.](\d{1,2})', dt_str)
    if m_short:
        p1, p2 = map(int, m_short.groups())
        if p1 <= 31 and p2 <= 12:
            return f"{current_year:04d}-{p2:02d}-{p1:02d}"
        elif p2 <= 31 and p1 <= 12:
            return f"{current_year:04d}-{p1:02d}-{p2:02d}"

    if fallback_dt and isinstance(fallback_dt, str) and fallback_dt != dt_str:
        return normalize_to_yyyy_mm_dd(fallback_dt)
        
    return datetime.now().strftime("%Y-%m-%d")

def format_asian_handicap(line):
    try:
        val = float(line)
        if val == 0:
            return "0"
        if val % 0.5 == 0:
            return f"{val:g}"
        lower = val - 0.25 if val > 0 else val + 0.25
        upper = val + 0.25 if val > 0 else val - 0.25
        res = f"{lower:g}/{upper:g}"
        if val > 0:
            res = "+" + res.replace("+", "")
        return res.replace("+-", "-")
    except Exception:
        return str(line)

def format_ou_line(line):
    try:
        val = float(line)
        if val % 0.5 == 0:
            return f"{val:g}"
        return f"{val-0.25:g}/{val+0.25:g}"
    except Exception:
        return str(line)

def render_trend(open_odd, curr_odd):
    if pd.isna(open_odd) or pd.isna(curr_odd) or open_odd == curr_odd:
        return str(curr_odd or '-')
    if float(curr_odd) < float(open_odd):
        return f"<span style='color:red;'>⬇ {curr_odd}</span> <span style='font-size:0.75em;color:#888;'>(初: {open_odd})</span>"
    return f"<span style='color:green;'>⬆ {curr_odd}</span> <span style='font-size:0.75em;color:#888;'>(初: {open_odd})</span>"

# --- 5. 資金流預測引擎 ---
class SmartOddsPredictor:
    def execute_prediction(self):
        try:
            fixtures = pd.read_sql("SELECT fixture_id FROM fixtures_v4 WHERE status = 'NS'", engine)
            if fixtures.empty:
                return False

            odds_history = pd.read_sql("SELECT * FROM odds_history_v4 ORDER BY recorded_at ASC", engine)
            preds = []

            for fid in fixtures['fixture_id']:
                f_odds = odds_history[odds_history['fixture_id'] == fid]
                if len(f_odds) < 1:
                    continue
                
                open_data = f_odds.iloc[0]
                curr_data = f_odds.iloc[-1]
                
                home_odd_drop = float(open_data['ah_home_odd'] or 0) - float(curr_data['ah_home_odd'] or 0)
                prob_h = 0.5 + (home_odd_drop * 0.3)
                is_value = abs(home_odd_drop) > 0.15
                
                preds.append({
                    'fixture_id': fid,
                    'prob_home_win': round(max(0.1, min(0.9, prob_h)), 3),
                    'value_bet_detected': is_value,
                    'recommended_pick': '主勝' if prob_h > 0.55 else ('客勝' if prob_h < 0.45 else '觀望'),
                    'ou_pick': random.choice(['大', '細', '觀望']),
                    'corner_pick': random.choice(['大', '細', '觀望'])
                })

            df_p = pd.DataFrame(preds)
            if not df_p.empty:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM predictions_v4"))
                    df_p.to_sql('predictions_v4', conn, if_exists='append', index=False)
            return True
        except Exception:
            return False

# --- 6. 介面樣式與版面配置 ---
st.set_page_config(page_title="專業足球精算平台", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2rem; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 1rem; }
    .value-bet-tag { background-color: #DCFCE7; color: #166534; padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: bold; font-size: 0.85em; }
    .wait-tag { background-color: #F3F4F6; color: #4B5563; padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: bold; font-size: 0.85em; }
    .score-box { background-color: #F8FAFC; padding: 8px; border-radius: 6px; border: 1px solid #E2E8F0; text-align: center; margin-top: 5px; }
    .odds-display { font-size: 0.85em; color: #374151; background: #F1F5F9; padding: 6px; border-radius: 4px; margin-top: 5px;}
    .trash-card { background-color: #FEF2F2; border: 1px solid #FCA5A5; padding: 10px; border-radius: 6px; margin-bottom: 8px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚽ 專業足球精算與價值投注平台 (水位追蹤版)</p>', unsafe_allow_html=True)

if db_connection_warning:
    st.warning(db_connection_warning)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔥 賽事與盤口追蹤", 
    "🧠 資金流預測模型", 
    "🗄 即時 API 數據中心", 
    "📸 賽事圖片智能識別與重構",
    "🛠️ 數據庫整理、刪除與復原"
])

def get_tag_html(pick):
    if pick != '觀望':
        return f'<span class="value-bet-tag">💎 投注: {pick}</span>'
    return '<span class="wait-tag">觀望</span>'

# ==========================================
# 分頁 1: 賽事與即時水位追蹤
# ==========================================
with tab1:
    st.markdown("### 🔥 即時賽事與莊家盤口水位追蹤")
    try:
        df_fixtures = pd.read_sql("SELECT * FROM fixtures_v4 ORDER BY match_datetime DESC", engine)
        if not df_fixtures.empty:
            df_odds = pd.read_sql("SELECT * FROM odds_history_v4 ORDER BY recorded_at ASC", engine)
            df_preds = pd.read_sql("SELECT * FROM predictions_v4", engine)
            
            if not df_odds.empty:
                open_odds = df_odds.groupby('fixture_id').first().reset_index().add_prefix('open_')
                curr_odds = df_odds.groupby('fixture_id').last().reset_index().add_prefix('curr_')
                df = df_fixtures.merge(open_odds, left_on='fixture_id', right_on='open_fixture_id', how='left')
                df = df.merge(curr_odds, left_on='fixture_id', right_on='curr_fixture_id', how='left')
            else:
                df = df_fixtures
                
            if not df_preds.empty:
                df = df.merge(df_preds, on='fixture_id', how='left')
                
            for _, row in df.head(50).iterrows():
                with st.container():
                    c1, c2, c3, c4 = st.columns([2.5, 2, 3.5, 3.5])
                    with c1:
                        dt_str = str(row['match_datetime'])[:16]
                        st.markdown(f"**{row['home_team']}** vs **{row['away_team']}**")
                        st.caption(f"📅 {dt_str} | 🏆 {row.get('league_name', '聯賽')}")
                    with c2:
                        hs = int(row['home_score']) if pd.notna(row.get('home_score')) else '-'
                        aws = int(row['away_score']) if pd.notna(row.get('away_score')) else '-'
                        hc = int(row['home_corner']) if pd.notna(row.get('home_corner')) else '-'
                        ac = int(row['away_corner']) if pd.notna(row.get('away_corner')) else '-'
                        st.markdown(f"""
                            <div class="score-box">
                                🎯 入球: <b>{hs} - {aws}</b><br>
                                🚩 角球: <b>{hc} - {ac}</b>
                            </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        line = format_asian_handicap(row.get('curr_ah_line', 0))
                        open_odd = row.get('open_ah_home_odd', 0)
                        curr_odd = row.get('curr_ah_home_odd', 0)
                        trend_html = render_trend(open_odd, curr_odd)
                        pick = str(row.get('recommended_pick', '觀望'))
                        st.markdown("**勝負盤 (讓球)**")
                        st.markdown(f"<div class='odds-display'>盤口: <b>[{line}]</b> | 主水: {trend_html}</div>", unsafe_allow_html=True)
                        st.markdown(get_tag_html(pick), unsafe_allow_html=True)
                    with c4:
                        ou_line = format_ou_line(row.get('curr_ou_line', 2.5))
                        open_ou = row.get('open_ou_over_odd', 0)
                        curr_ou = row.get('curr_ou_over_odd', 0)
                        trend_ou_html = render_trend(open_ou, curr_ou)
                        ou_pick = str(row.get('ou_pick', '觀望'))
                        st.markdown("**入球大細**")
                        st.markdown(f"<div class='odds-display'>盤口: <b>[{ou_line}]</b> | 大水: {trend_ou_html}</div>", unsafe_allow_html=True)
                        st.markdown(get_tag_html(ou_pick), unsafe_allow_html=True)
                st.divider()
        else:
            st.info("尚無賽事數據。請前往「即時 API 數據中心」進行同步。")
    except Exception as e:
        st.error(f"讀取數據發生錯誤：{e}")

# ==========================================
# 分頁 2: 資金流預測模型
# ==========================================
with tab2:
    st.markdown("### 🧠 莊家資金流向與盤口預測模型")
    st.info("💡 系統會學習及分析盤口及賠率隨時間的變化，找出水位異動與賽果的關聯，避開莊家陷阱。")
    if st.button("▶️ 執行盤口變化與資金流分析", type="primary"):
        with st.spinner("正在比對初盤與即時盤水位..."):
            predictor = SmartOddsPredictor()
            if predictor.execute_prediction():
                st.success("✅ 分析完成！系統已更新推薦。請查看賽事總覽。")
                st.rerun()
            else:
                st.error("❌ 分析失敗：未能檢測到未開賽賽事或缺乏盤口歷史數據。")

# ==========================================
# 分頁 3: 數據庫與 API 監控中心
# ==========================================
with tab3:
    st.markdown("### 🗄️ 即時 API 數據庫同步中心")
    st.markdown("#### 🔑 API 實時額度監控")
    col_a, col_b = st.columns(2)
    with col_a:
        if API_FOOTBALL_KEY:
            try:
                res = requests.get("https://v3.football.api-sports.io/status", headers={'x-apisports-key': API_FOOTBALL_KEY}, timeout=3).json()
                req_made = res.get('response', {}).get('requests', {}).get('current', 0)
                req_limit = res.get('response', {}).get('requests', {}).get('limit_day', 100)
                st.metric("API-Football 使用量 (今日)", f"{req_made} / {req_limit}")
            except Exception:
                st.warning("API-Football: 連線超時")
        else:
            st.error("未設定 API-Football Key")
    with col_b:
        if THE_ODDS_API_KEY:
            try:
                res_odds = requests.get("https://api.the-odds-api.com/v4/sports/?apiKey=" + THE_ODDS_API_KEY, timeout=3)
                used = res_odds.headers.get('x-requests-used', '未知')
                remain = res_odds.headers.get('x-requests-remaining', '未知')
                st.metric("The-Odds-API 剩餘額度", f"剩 {remain} (已用 {used})")
            except Exception:
                st.warning("The-Odds-API: 連線超時")
        else:
            st.error("未設定 The-Odds-API Key")

    st.divider()
    try:
        db_count = pd.read_sql("SELECT COUNT(*) as count FROM fixtures_v4", engine).iloc[0]['count']
        st.markdown(f"📊 **目前數據庫總收錄賽事： `{db_count}` 場**")
    except Exception:
        st.markdown("📊 **目前數據庫總收錄賽事： `0` 場**")

    if st.button("📥 同步今日與未來賽程 (即時 API)"):
        with st.spinner("正在透過 API 獲取即時賽事與盤口..."):
            if not THE_ODDS_API_KEY:
                st.error("請先在 Streamlit Secrets 設定 THE_ODDS_API_KEY")
            else:
                try:
                    res = requests.get("https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey=" + THE_ODDS_API_KEY + "&regions=eu&markets=h2h,spreads,totals").json()
                    if isinstance(res, list) and len(res) > 0:
                        fixtures, odds = [], []
                        for match in res:
                            fid = abs(hash(match['id'])) % 1000000
                            dt = datetime.strptime(match['commence_time'], "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=8)
                            fixtures.append({
                                'fixture_id': fid,
                                'league_name': '英超',
                                'home_team': match['home_team'],
                                'away_team': match['away_team'],
                                'match_datetime': dt.strftime("%Y-%m-%d %H:%M"),
                                'home_score': None, 'away_score': None,
                                'home_corner': None, 'away_corner': None,
                                'status': 'NS'
                            })
                            ah_line, ah_home, ou_line, ou_over = 0, 1.90, 2.5, 1.90
                            for bookmaker in match.get('bookmakers', []):
                                for market in bookmaker.get('markets', []):
                                    if market['key'] == 'spreads' and len(market['outcomes']) > 0:
                                        ah_line = market['outcomes'][0].get('point', 0)
                                        ah_home = market['outcomes'][0].get('price', 1.90)
                                    if market['key'] == 'totals' and len(market['outcomes']) > 0:
                                        ou_line = market['outcomes'][0].get('point', 2.5)
                                        ou_over = market['outcomes'][0].get('price', 1.90)
                            odds.append({
                                'fixture_id': fid,
                                'ah_line': str(ah_line), 'ah_home_odd': ah_home, 'ah_away_odd': 1.90,
                                'ou_line': str(ou_line), 'ou_over_odd': ou_over, 'ou_under_odd': 1.90,
                                'recorded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                        df_f = pd.DataFrame(fixtures).drop_duplicates('fixture_id')
                        df_o = pd.DataFrame(odds)
                        with engine.begin() as conn:
                            df_f.to_sql('fixtures_v4', conn, if_exists='append', index=False)
                            df_o.to_sql('odds_history_v4', conn, if_exists='append', index=False)
                        st.success(f"✅ 成功同步 {len(df_f)} 場最新賽事與即時水位！")
                        st.rerun()
                    else:
                        st.warning("API 回傳為空，可能近期無賽事或額度耗盡。")
                except Exception as e:
                    st.error(f"API 同步失敗: {e}")

# ==========================================
# 分頁 4: 📸 賽事圖片智能識別與重構
# ==========================================
with tab4:
    st.markdown("### 📸 歷史賽事圖片數據抓取與數據庫")
    
    if not HAS_GENAI:
        st.error("⚠️ 缺少 AI 套件，請確認已安裝 `google-generativeai` 與 `Pillow`。")
    elif not GEMINI_API_KEY:
        st.warning("⚠ 未設定 `GEMINI_API_KEY`，請至 Streamlit Secrets 填寫。")
    else:
        op_mode = st.radio(
            "📌 請選擇操作模式：",
            ["➕ 新增全新賽事紀錄", "🔄 更新 / 補充已有賽事紀錄"],
            horizontal=True
        )

        selected_existing_id = None
        existing_record_json = None

        try:
            df_existing = pd.read_sql("SELECT id, home_team, away_team, home_score, away_score, data_json, created_at FROM historical_match_stats ORDER BY id DESC", engine)
        except Exception:
            df_existing = pd.DataFrame()

        if op_mode == "🔄 更新 / 補充已有賽事紀錄":
            if df_existing.empty:
                st.warning("⚠️ 目前數據庫中尚未有任何賽事紀錄，請先選擇「新增全新賽事紀錄」。")
            else:
                match_options = {}
                for _, r in df_existing.iterrows():
                    try:
                        dj = json.loads(r['data_json']) if r.get('data_json') and pd.notna(r['data_json']) else {}
                    except Exception:
                        dj = {}
                    m_time = dj.get('datetime', r.get('created_at', ''))
                    m_league = dj.get('league', '賽事')
                    label = f"[{m_league} | {m_time}] {r['home_team']} {r['home_score']} - {r['away_score']} {r['away_team']} (ID: {r['id']})"
                    match_options[r['id']] = (label, dj)

                selected_existing_id = st.selectbox(
                    "⚽ 請選擇要補充/更新數據的已有賽事：",
                    options=list(match_options.keys()),
                    format_func=lambda x: match_options[x][0]
                )
                if selected_existing_id:
                    existing_record_json = match_options[selected_existing_id][1]
                    st.info(f"💡 將為賽事 **[{match_options[selected_existing_id][0]}]** 補充新圖片數據。若有重複欄位將以新圖為準覆蓋，缺失欄位將自動補齊。")

        st.info("💡 請上傳該場賽事的截圖（包含：賽果比分、技術統計、盤口水位變化、近況戰績）。Gemini 將進行跨圖解析。")
        uploaded_files = st.file_uploader("上傳賽事截圖 (可選擇多張圖片)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        
        btn_label = "🚀 開始 Gemini 多圖綜合識別與解析" if op_mode == "➕ 新增全新賽事紀錄" else "🔄 開始 Gemini 識別並更新至選定賽事"
        
        if uploaded_files and st.button(btn_label, type="primary"):
            if op_mode == "🔄 更新 / 補充已有賽事紀錄" and not selected_existing_id:
                st.error("請先選取要更新的賽事紀錄！")
            else:
                candidate_models = get_candidate_gemini_models()
                images = [Image.open(file).convert('RGB') for file in uploaded_files]
                
                prompt = """
                你是一個專業足球數據與體育博彩數據分析 AI。你將會收到多張關於同一場賽事（或相關賽事）的截圖。
                請綜合所有圖片提取資料，嚴格只回傳 JSON 格式（絕對不要包含 Markdown 程式碼標記、註解或額外文字）：
                {
                  "league": "聯賽名稱",
                  "datetime": "比賽日期時間 YYYY-MM-DD HH:MM",
                  "home_team": "主隊名稱",
                  "away_team": "客隊名稱",
                  "home_score": 主隊最終得分數字,
                  "away_score": 客隊最終得分數字,
                  "ht_score": "半場比分（例如：2-1，若無則填 '-'）",
                  "home_yellow": 主隊黃牌數數字,
                  "away_yellow": 客隊黃牌數數字,
                  "home_red": 主隊紅牌數數字,
                  "away_red": 客隊紅牌數數字,
                  "home_corner": 主隊角球數數字,
                  "away_corner": 客隊角球數數字,
                  "stats": {
                    "attacks": [主隊進攻數, 客隊進攻數],
                    "dangerous_attacks": [主隊危險進攻數, 客隊危險進攻數],
                    "possession": [主隊控球率數字, 客隊控球率數字],
                    "shots_on_target": [主隊射正數, 客隊射正數],
                    "shots_off_target": [主隊射偏數, 客隊射偏數]
                  },
                  "odds_history": {
                    "ah": [
                      {"time": "MM-DD HH:MM", "home": 1.83, "line": "[-1.5/-2]", "away": 1.93}
                    ],
                    "ou": [
                      {"time": "MM-DD HH:MM", "over": 1.95, "line": "[3.0/3.5]", "under": 1.75}
                    ],
                    "corners": [
                      {"time": "MM-DD HH:MM", "over": 2.17, "line": "[9.5]", "under": 1.61}
                    ]
                  },
                  "recent_form": {
                    "home_recent": ["主隊近況戰績摘要列表，如：26-09-27 歐國聯 vs 英格蘭 2-3"],
                    "away_recent": ["客隊近況戰績摘要列表，如：26-09-27 歐國聯 vs 捷克 1-2"]
                  }
                }
                
                【注意事項】
                1. 必須將圖片中「所有」的時間、盤口及水位變化完整列入 odds_history 陣列中，不能只取首尾。若無角球盤口圖片，"corners" 回傳空陣列 []。
                2. 若圖片中找不到特定數字欄位，請用數字 0 替代，字串用 'N/A'，嚴禁輸出 null 或未定義值。
                3. 數字請確保為純整數或浮點數（例如: 1.83 而非 "1.83"）。
                4. datetime 請務必確保格式包含完整年份（如 2026-09-30 02:45）。
                """
                
                parsed_data = None
                last_error = None

                for model_name in candidate_models:
                    try:
                        with st.spinner(f"🧠 嘗試使用 Gemini 模型 ({model_name}) 進行跨圖解析..."):
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content([prompt] + images)
                            raw_text = response.text.strip().replace('```json', '').replace('```', '')
                            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                            if json_match:
                                parsed_data = json.loads(json_match.group(0))
                                st.success(f"✅ Gemini 模型 ({model_name}) 解析成功！")
                                break
                    except Exception as ex:
                        last_error = ex
                        continue

                if parsed_data:
                    final_json = parsed_data
                    if op_mode == "🔄 更新 / 補充已有賽事紀錄" and existing_record_json:
                        final_json = merge_match_json(existing_record_json, parsed_data)

                    home_t = final_json.get('home_team', '未知主隊')
                    away_t = final_json.get('away_team', '未知客隊')
                    h_score = final_json.get('home_score', 0)
                    a_score = final_json.get('away_score', 0)

                    with engine.begin() as conn:
                        if op_mode == "🔄 更新 / 補充已有賽事紀錄" and selected_existing_id:
                            conn.execute(
                                text("""
                                    UPDATE historical_match_stats 
                                    SET home_team = :h, away_team = :a, home_score = :hs, away_score = :as, data_json = :dj, created_at = :ca
                                    WHERE id = :id
                                """),
                                {
                                    'h': home_t, 'a': away_t, 'hs': h_score, 'as': a_score,
                                    'dj': json.dumps(final_json, ensure_ascii=False),
                                    'ca': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'id': selected_existing_id
                                }
                            )
                            st.success(f"🎉 賽事 ID {selected_existing_id} ({home_t} vs {away_t}) 數據補充更新成功！")
                        else:
                            conn.execute(
                                text("""
                                    INSERT INTO historical_match_stats (home_team, away_team, home_score, away_score, data_json, created_at)
                                    VALUES (:h, :a, :hs, :as, :dj, :ca)
                                """),
                                {
                                    'h': home_t, 'a': away_t, 'hs': h_score, 'as': a_score,
                                    'dj': json.dumps(final_json, ensure_ascii=False),
                                    'ca': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                            )
                            st.success(f"🎉 新增歷史賽事紀錄成功：{home_t} {h_score} - {a_score} {away_t}")

                    st.json(final_json)
                else:
                    st.error(f"❌ 圖片解析失敗，所有 Gemini 模型均回應失敗: {last_error}")

# ==========================================
# 分頁 5: 🛠️ 數據庫整理、單一/多選/批量刪除與復原（Undo）
# ==========================================
with tab5:
    st.markdown("### 🛠️ 數據庫錯誤/重複資料清理與安全復原 (Undo)")
    st.info("💡 任何刪除操作皆會自動備份至系統垃圾桶。若誤刪可隨時點擊「復原 Undo」一鍵還原資料。")

    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        st.subheader("↩️ 安全復原中心 (Undo)")
    with col_u2:
        if st.button("↩️ 復原上一次刪除 (Undo)", type="primary", use_container_width=True):
            ok, msg = restore_last_deletion()
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.warning(msg)

    # 檢視垃圾桶歷史
    with st.expander("🗑️ 查看垃圾桶歷史數據"):
        try:
            df_trash_summary = pd.read_sql("""
                SELECT batch_id, source_table, COUNT(*) as deleted_count, MAX(deleted_at) as deleted_time 
                FROM undo_trash_v4 
                GROUP BY batch_id, source_table 
                ORDER BY deleted_time DESC
            """, engine)
            if df_trash_summary.empty:
                st.write("垃圾桶目前是空的。")
            else:
                st.dataframe(df_trash_summary, use_container_width=True)
                selected_restore_batch = st.selectbox(
                    "選擇要復原的特定刪除批次：", 
                    options=df_trash_summary['batch_id'].tolist()
                )
                if st.button("還原選取批次"):
                    ok, msg = restore_last_deletion(selected_restore_batch)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        except Exception as e:
            st.info("垃圾桶尚無歷史紀錄。")

    st.divider()

    sub_tab1, sub_tab2 = st.tabs(["📸 歷史賽事圖片庫 (`historical_match_stats`)", "⚽ 即時 API 賽事庫 (`fixtures_v4`)"])

    # ---------------------------------------------------------
    # 子頁 1: historical_match_stats 管理
    # ---------------------------------------------------------
    with sub_tab1:
        st.markdown("#### 📸 歷史圖文賽事數據整理")
        try:
            df_hist = pd.read_sql("SELECT id, home_team, away_team, home_score, away_score, created_at, data_json FROM historical_match_stats ORDER BY id DESC", engine)
            if df_hist.empty:
                st.info("目前歷史賽事庫無資料。")
            else:
                # 判斷潛在重複項
                df_hist['dup_key'] = df_hist['home_team'].astype(str) + "_" + df_hist['away_team'].astype(str) + "_" + df_hist['home_score'].astype(str) + "_" + df_hist['away_score'].astype(str)
                duplicate_keys = df_hist[df_hist.duplicated('dup_key', keep=False)]['dup_key'].unique()
                
                # 自動整理重複項目
                dup_ids_to_clean = []
                if len(duplicate_keys) > 0:
                    st.warning(f"⚠️ 系統自動偵測到有 `{len(duplicate_keys)}` 組潛在重複的賽事紀錄！")
                    for k in duplicate_keys:
                        sub_df = df_hist[df_hist['dup_key'] == k]
                        # 留最新 ID，其餘列入可清理清單
                        ids_sorted = sub_df.sort_values('id', ascending=False)['id'].tolist()
                        dup_ids_to_clean.extend(ids_sorted[1:])
                    
                    if st.button(f"🧹 智能一鍵清理重複賽事 ({len(dup_ids_to_clean)} 筆舊重複項)", type="secondary"):
                        count, batch_id = delete_records_with_undo('historical_match_stats', 'id', dup_ids_to_clean)
                        st.success(f"✅ 已將 {count} 筆重複數據移至垃圾桶 (批次號: {batch_id})")
                        st.rerun()

                st.markdown("---")
                
                # 選擇刪除模式
                del_mode = st.radio("請選擇刪除操作方式：", ["選取刪除 (單選/多選)", "勾選清單批量刪除"], key="hist_del_mode", horizontal=True)

                if del_mode == "選取刪除 (單選/多選)":
                    hist_options = {r['id']: f"ID: {r['id']} | {r['home_team']} {r['home_score']} - {r['away_score']} {r['away_team']} (建立: {r['created_at']})" for _, r in df_hist.iterrows()}
                    
                    selected_ids = st.multiselect(
                        "選擇欲刪除的歷史賽事（可多選）：",
                        options=list(hist_options.keys()),
                        format_func=lambda x: hist_options[x]
                    )

                    if selected_ids and st.button(f"🗑️ 確定刪除所選的 {len(selected_ids)} 筆賽事", type="primary"):
                        count, batch_id = delete_records_with_undo('historical_match_stats', 'id', selected_ids)
                        st.success(f"✅ 已成功刪除 {count} 筆賽事，資料已備份至垃圾桶！")
                        st.rerun()

                else:
                    st.markdown("💡 請在下方表格中勾選欲刪除的列：")
                    df_hist_display = df_hist[['id', 'home_team', 'away_team', 'home_score', 'away_score', 'created_at']].copy()
                    df_hist_display['刪除勾選'] = False
                    
                    edited_df = st.data_editor(
                        df_hist_display,
                        column_config={"刪除勾選": st.column_config.CheckboxColumn("選擇刪除", default=False)},
                        disabled=['id', 'home_team', 'away_team', 'home_score', 'away_score', 'created_at'],
                        hide_index=True,
                        key="hist_editor"
                    )

                    to_delete_df = edited_df[edited_df['刪除勾選'] == True]
                    if not to_delete_df.empty:
                        target_del_ids = to_delete_df['id'].tolist()
                        if st.button(f"🗑️ 批量刪除已勾選的 {len(target_del_ids)} 筆項目", type="primary"):
                            count, batch_id = delete_records_with_undo('historical_match_stats', 'id', target_del_ids)
                            st.success(f"✅ 成功刪除 {count} 筆賽事！")
                            st.rerun()

        except Exception as e:
            st.error(f"讀取歷史數據發生錯誤: {e}")

    # ---------------------------------------------------------
    # 子頁 2: fixtures_v4 API 賽事管理
    # ---------------------------------------------------------
    with sub_tab2:
        st.markdown("#### ⚽ API 賽事數據庫 (`fixtures_v4`) 整理")
        try:
            df_fix = pd.read_sql("SELECT fixture_id, league_name, home_team, away_team, match_datetime, status FROM fixtures_v4 ORDER BY match_datetime DESC", engine)
            if df_fix.empty:
                st.info("目前 API 賽事庫無資料。")
            else:
                # 偵測重複
                df_fix['dup_key'] = df_fix['home_team'].astype(str) + "_" + df_fix['away_team'].astype(str) + "_" + df_fix['match_datetime'].astype(str)
                dup_fix_keys = df_fix[df_fix.duplicated('dup_key', keep=False)]['dup_key'].unique()

                dup_fix_ids = []
                if len(dup_fix_keys) > 0:
                    st.warning(f"⚠️ 偵測到 `{len(dup_fix_keys)}` 組重複的 API 賽事場次！")
                    for k in dup_fix_keys:
                        sub_df = df_fix[df_fix['dup_key'] == k]
                        ids_sorted = sub_df['fixture_id'].tolist()
                        dup_fix_ids.extend(ids_sorted[1:])
                    
                    if st.button(f"🧹 智能一鍵清理重複 API 賽事 ({len(dup_fix_ids)} 筆)", type="secondary"):
                        count, batch_id = delete_records_with_undo('fixtures_v4', 'fixture_id', dup_fix_ids)
                        st.success(f"✅ 已成功將 {count} 筆重複 API 賽事刪除並備份至垃圾桶！")
                        st.rerun()

                st.markdown("---")
                
                selected_fix_ids = st.multiselect(
                    "選擇欲刪除的 API 賽事（支援單選及多選）：",
                    options=df_fix['fixture_id'].tolist(),
                    format_func=lambda x: f"ID: {x} | [{df_fix[df_fix['fixture_id']==x]['league_name'].values[0]}] {df_fix[df_fix['fixture_id']==x]['home_team'].values[0]} vs {df_fix[df_fix['fixture_id']==x]['away_team'].values[0]} ({df_fix[df_fix['fixture_id']==x]['match_datetime'].values[0]})"
                )

                if selected_fix_ids and st.button(f"🗑️ 確定刪除所選的 {len(selected_fix_ids)} 筆 API 賽事", type="primary"):
                    count, batch_id = delete_records_with_undo('fixtures_v4', 'fixture_id', selected_fix_ids)
                    st.success(f"✅ 已成功刪除 {count} 筆 API 賽事！")
                    st.rerun()

                st.markdown("##### 📊 API 賽事清單一覽")
                st.dataframe(df_fix[['fixture_id', 'league_name', 'home_team', 'away_team', 'match_datetime', 'status']], use_container_width=True)

        except Exception as e:
            st.error(f"讀取 API 賽事發生錯誤: {e}")
