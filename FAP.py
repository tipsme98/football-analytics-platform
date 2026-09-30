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

# --- 4. 數據合併與正規化邏輯 (新增/覆蓋合併機制) ---
def is_valid_val(v):
    """判斷數據是否為有效值（非空、非零預設值）"""
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
    """將 Gemini 新解析的 JSON 與數據庫舊 JSON 進行智慧合併（補缺與覆蓋舊值）"""
    if not old_data:
        return new_data
    if not new_data:
        return old_data

    merged = dict(old_data)

    for k, new_v in new_data.items():
        old_v = merged.get(k)

        # 若是字典（如 stats, odds_trend, recent_form），進行深層合併
        if isinstance(new_v, dict) and isinstance(old_v, dict):
            merged_dict = dict(old_v)
            for sub_k, sub_v in new_v.items():
                if is_valid_val(sub_v):
                    merged_dict[sub_k] = sub_v
            merged[k] = merged_dict

        # 若是陣列 (如 recent_form 列表)
        elif isinstance(new_v, list):
            if is_valid_val(new_v):
                merged[k] = new_v

        # 一般欄位：若新資料有效，則進行更新或覆蓋
        else:
            if is_valid_val(new_v):
                merged[k] = new_v

    return merged

def normalize_to_yyyy_mm_dd(dt_str, fallback_dt=None):
    """將格式不同的日期統一為 YYYY-MM-DD"""
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
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚽ 專業足球精算與價值投注平台 (水位追蹤版)</p>', unsafe_allow_html=True)

if db_connection_warning:
    st.warning(db_connection_warning)

tab1, tab2, tab3, tab4 = st.tabs(["🔥 賽事與盤口追蹤", "🧠 資金流預測模型", "🗄 即時 API 數據中心", "📸 賽事圖片智能識別與重構"])

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
# 分頁 4: 📸 賽事圖片智能識別與歷史數據庫 (支援：新增賽事 / 選擇已有賽事覆蓋與增補)
# ==========================================
with tab4:
    st.markdown("### 📸 歷史賽事圖片數據抓取與數據庫")
    
    if not HAS_GENAI:
        st.error("⚠️ 缺少 AI 套件，請確認已安裝 `google-generativeai` 與 `Pillow`。")
    elif not GEMINI_API_KEY:
        st.warning("⚠ 未設定 `GEMINI_API_KEY`，請至 Streamlit Secrets 填寫。")
    else:
        # --- 操作模式選擇 (新增賽事 vs 補充已有賽事) ---
        op_mode = st.radio(
            "📌 請選擇操作模式：",
            ["➕ 新增全新賽事紀錄", "🔄 更新 / 補充已有賽事紀錄"],
            horizontal=True
        )

        selected_existing_id = None
        existing_record_json = None

        # 讀取目前數據庫中的所有紀錄以供選取
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
                你是一個專業足球數據與體育博彩數據分析 AI。你將會收到一張或多張關於同一場賽事（或相關賽事）的截圖，內容可能包含：
                1. 賽果比分與基本資訊（聯賽名稱、時間、主隊、客隊、半全場比分、黃/紅牌、角球）。
                2. 詳細技術統計（進攻、危險進攻、控球率、射正、射斜、點球、被擋射門等）。
                3. 莊家盤口與水位變化表（讓球盤口與初盤/即時水位、大細球盤口與初盤/即時水位、水位走勢紀錄）。
                4. 主隊與客隊近況戰績 / 歷史交鋒對賽紀錄。

                請綜合所有提供的截圖，精確結構化提取資料，並嚴格只回傳 JSON 格式（絕對不要包含 Markdown 程式碼標記、註解或額外文字）：
                {
                  "league": "聯賽名稱（例如：歐洲國家聯賽、英超、西甲等）",
                  "datetime": "比賽日期時間（格式為 YYYY-MM-DD HH:MM，例如：2026-09-30 02:45；若未顯示年份請預設為 2026 年）",
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
                  "odds_trend": {
                    "ah_line": "讓球盤口（例如：[-1.5/-2], [-2], 0，若未顯示填 '0'）",
                    "home_initial_odd": 主隊讓球初盤水位數字（例如 1.83，若無填 0）,
                    "home_current_odd": 主隊讓球即時/終盤水位數字（例如 1.98，若無填 0）,
                    "away_initial_odd": 客隊讓球初盤水位數字（例如 1.93，若無填 0）,
                    "away_current_odd": 客隊讓球即時/終盤水位數字（例如 1.82，若無填 0）,
                    "ou_line": "大細球盤口（例如：2.5, 2.75, 3，若未顯示填 '2.5'）",
                    "ou_over_initial": 大球初盤水位數字（例如 1.85，若無填 0）,
                    "ou_over_current": 大球即時水位數字（例如 1.95，若無填 0）,
                    "ou_under_initial": 細球初盤水位數字（例如 1.95，若無填 0）,
                    "ou_under_current": 細球即時水位數字（例如 1.85，若無填 0）
                  },
                  "recent_form": {
                    "home_recent": ["主隊近況戰績摘要列表，如：26-09-27 歐國聯 vs 英格蘭 2-3"],
                    "away_recent": ["客隊近況戰績摘要列表，如：26-09-27 歐國聯 vs 捷克 1-2"]
                  }
                }

                【注意事項】
                1. 請整合所有上傳圖片中的訊息。若有盤口變化截圖，初盤請採最早時間之水位，終盤/即時盤請採最晚時間之水位。
                2. 若圖片中找不到特定數字欄位，請用數字 0 替代，字串用 'N/A'，嚴禁輸出 null 或未定義值。
                3. 數字請確保為純整數或浮點數（例如: 1.83 而非 "1.83"）。
                4. datetime 請務必確保格式包含完整年份（如 2026-09-30 02:45）。
                """
                
                parsed_data = None
                last_error = None
                
                # 模型輪詢與自動降級備援機制
                for model_name in candidate_models:
                    try:
                        with st.spinner(f"🧠 嘗試使用 Gemini 模型 ({model_name}) 進行跨圖解析..."):
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content([prompt] + images)
                            raw_text = response.text.strip().replace('```json', '').replace('```', '').strip()
                            
                            start_idx = raw_text.find('{')
                            end_idx = raw_text.rfind('}')
                            if start_idx != -1 and end_idx != -1:
                                raw_text = raw_text[start_idx:end_idx+1]
                                
                            parsed_data = json.loads(raw_text)
                            if parsed_data:
                                st.toast(f"✅ 成功調用模型：{model_name}")
                                break
                    except Exception as e:
                        last_error = e
                        continue
                
                if parsed_data:
                    parsed_dt = parsed_data.get('datetime', '')
                    parsed_data['datetime'] = parsed_dt if parsed_dt else datetime.now().strftime("%Y-%m-%d %H:%M")
                    
                    if op_mode == "🔄 更新 / 補充已有賽事紀錄" and selected_existing_id:
                        # 執行數據合併：將 new_data 覆蓋/增補到 existing_record_json
                        merged_data = merge_match_json(existing_record_json, parsed_data)
                        
                        try:
                            with engine.begin() as conn:
                                conn.execute(text("""
                                    UPDATE historical_match_stats
                                    SET home_team = :h,
                                        away_team = :a,
                                        home_score = :hs,
                                        away_score = :aws,
                                        data_json = :dj
                                    WHERE id = :id
                                """), {
                                    'h': merged_data.get('home_team', '未知主隊'),
                                    'a': merged_data.get('away_team', '未知客隊'),
                                    'hs': int(merged_data.get('home_score') or 0),
                                    'aws': int(merged_data.get('away_score') or 0),
                                    'dj': json.dumps(merged_data, ensure_ascii=False),
                                    'id': selected_existing_id
                                })
                            st.success(f"✅ 已成功將 {len(uploaded_files)} 張新圖片的數據合併至該賽事紀錄（補齊缺項並更新舊數據）！")
                            st.rerun()
                        except Exception as e:
                            st.error(f"資料庫更新失敗: {e}")

                    else:
                        # 全新賽事紀錄新增
                        try:
                            with engine.begin() as conn:
                                conn.execute(text("""
                                    INSERT INTO historical_match_stats (home_team, away_team, home_score, away_score, data_json, created_at)
                                    VALUES (:h, :a, :hs, :aws, :dj, :ca)
                                """), {
                                    'h': parsed_data.get('home_team', '未知主隊'),
                                    'a': parsed_data.get('away_team', '未知客隊'),
                                    'hs': int(parsed_data.get('home_score') or 0),
                                    'aws': int(parsed_data.get('away_score') or 0),
                                    'dj': json.dumps(parsed_data, ensure_ascii=False),
                                    'ca': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                            st.success(f"✅ AI 成功識別並整合了 {len(uploaded_files)} 張截圖數據，已新增至數據庫！")
                            st.rerun()
                        except Exception as e:
                            st.error(f"資料庫寫入失敗: {e}")
                else:
                    st.error(f"所有 Gemini 模型嘗試皆失敗。最後錯誤訊息: {last_error}")

    st.divider()
    st.markdown("### 📅 歷史賽事紀錄查詢")
    
    try:
        df_history = pd.read_sql("SELECT id, home_team, away_team, home_score, away_score, data_json, created_at FROM historical_match_stats ORDER BY id DESC", engine)
    except Exception:
        df_history = pd.DataFrame()
        
    if not df_history.empty:
        record_list = []
        for _, row in df_history.iterrows():
            try:
                dj = json.loads(row['data_json']) if row.get('data_json') and pd.notna(row['data_json']) else {}
            except Exception:
                dj = {}
            
            raw_dt = dj.get('datetime')
            raw_ca = row.get('created_at')
            
            date_str = normalize_to_yyyy_mm_dd(raw_dt, fallback_dt=raw_ca)
            match_time_display = str(raw_dt).strip() if (raw_dt and pd.notna(raw_dt)) else str(raw_ca).strip()
            
            record_list.append({
                'id': row['id'],
                'record_date': date_str,
                'league': dj.get('league', '未知聯賽'),
                'match_time': match_time_display,
                'home_team': dj.get('home_team', row.get('home_team', '未知主隊')),
                'away_team': dj.get('away_team', row.get('away_team', '未知客隊')),
                'score': f"{dj.get('home_score', row.get('home_score', 0))} - {dj.get('away_score', row.get('away_score', 0))}",
                'data_json': dj
            })
            
        df_records = pd.DataFrame(record_list)
        available_dates = sorted(list(df_records['record_date'].unique()), reverse=True)
        
        col_date, _ = st.columns([1, 2])
        with col_date:
            selected_date = st.selectbox("📅 請選擇查詢日期：", available_dates)
            
        day_matches = df_records[df_records['record_date'] == selected_date]
        
        st.markdown(f"#### 📋 {selected_date} 已記錄賽事列表")
        summary_table = day_matches[['match_time', 'league', 'home_team', 'score', 'away_team']].copy()
        summary_table.columns = ['比賽時間', '聯賽名稱', '主隊', '比分', '客隊']
        st.dataframe(summary_table, use_container_width=True, hide_index=True)
        
        st.divider()
        
        match_options = {row['id']: f"[{row['league']}] {row['home_team']} vs {row['away_team']} ({row['score']})" for _, row in day_matches.iterrows()}
        selected_match_id = st.selectbox(
            "⚽ 請選擇要檢視詳細資料的賽事：",
            options=list(match_options.keys()),
            format_func=lambda x: match_options[x]
        )
        
        selected_record = day_matches[day_matches['id'] == selected_match_id].iloc[0]
        data = selected_record['data_json']
        
        sub1, sub2, sub3, sub4 = st.tabs(["📊 技術統計", "📈 盤口與水位變化表", "⚽ 賽果明細", "📜 近況戰績"])
        
        with sub1:
            st.markdown("##### 📊 團隊詳細技術統計")
            stats = data.get('stats', {}) if isinstance(data.get('stats'), dict) else {}
            att = stats.get('attacks', [0, 0])
            d_att = stats.get('dangerous_attacks', [0, 0])
            pos = stats.get('possession', [0, 0])
            shots_on = stats.get('shots_on_target', [0, 0])
            shots_off = stats.get('shots_off_target', [0, 0])
            
            df_stats_table = pd.DataFrame([
                {"技術指標": "進攻次數", "主隊": att[0] if len(att)>0 else 0, "客隊": att[1] if len(att)>1 else 0},
                {"技術指標": "危險進攻次數", "主隊": d_att[0] if len(d_att)>0 else 0, "客隊": d_att[1] if len(d_att)>1 else 0},
                {"技術指標": "控球率 (%)", "主隊": f"{pos[0]}%" if len(pos)>0 else "0%", "客隊": f"{pos[1]}%" if len(pos)>1 else "0%"},
                {"技術指標": "射正次數", "主隊": shots_on[0] if len(shots_on)>0 else 0, "客隊": shots_on[1] if len(shots_on)>1 else 0},
                {"技術指標": "射偏次數", "主隊": shots_off[0] if len(shots_off)>0 else 0, "客隊": shots_off[1] if len(shots_off)>1 else 0}
            ])
            st.table(df_stats_table)
            
        with sub2:
            st.markdown("##### 📈 莊家盤口與水位變化表")
            ot = data.get('odds_trend', {}) if isinstance(data.get('odds_trend'), dict) else {}
            
            df_odds_table = pd.DataFrame([
                {
                    "玩法類型": "讓球盤 (AH)",
                    "盤口": ot.get('ah_line', '0'),
                    "初盤水位 (主 / 客)": f"{ot.get('home_initial_odd', '-')} / {ot.get('away_initial_odd', '-')}",
                    "即時/終盤水位 (主 / 客)": f"{ot.get('home_current_odd', '-')} / {ot.get('away_current_odd', '-')}"
                },
                {
                    "玩法類型": "入球大細 (OU)",
                    "盤口": ot.get('ou_line', '2.5'),
                    "初盤水位 (大 / 細)": f"{ot.get('ou_over_initial', '-')} / {ot.get('ou_under_initial', '-')}",
                    "即時/終盤水位 (大 / 細)": f"{ot.get('ou_over_current', '-')} / {ot.get('ou_under_current', '-')}"
                }
            ])
            st.table(df_odds_table)
            
        with sub3:
            st.markdown("##### ⚽ 賽事數據與賽果總覽")
            df_result_table = pd.DataFrame([
                {"項目": "賽事聯賽 / 時間", "數據": f"{data.get('league', '-')} ({data.get('datetime', '-')})"},
                {"項目": "主隊 vs 客隊", "數據": f"{data.get('home_team', '主隊')} vs {data.get('away_team', '客隊')}"},
                {"項目": "全場最終比分", "數據": f"{data.get('home_score', 0)} - {data.get('away_score', 0)}"},
                {"項目": "半場比分 (HT)", "數據": str(data.get('ht_score', '-'))},
                {"項目": "黃牌數 (主隊 / 客隊)", "數據": f"🟨 {data.get('home_yellow', 0)} / 🟨 {data.get('away_yellow', 0)}"},
                {"項目": "紅牌數 (主隊 / 客隊)", "數據": f"🟥 {data.get('home_red', 0)} / 🟥 {data.get('away_red', 0)}"},
                {"項目": "角球數 (主隊 / 客隊)", "數據": f"🚩 {data.get('home_corner', 0)} / 🚩 {data.get('away_corner', 0)}"}
            ])
            st.table(df_result_table)

        with sub4:
            st.markdown("##### 📜 近況戰績數據摘要")
            rf = data.get('recent_form', {}) if isinstance(data.get('recent_form'), dict) else {}
            h_rec = rf.get('home_recent', [])
            a_rec = rf.get('away_recent', [])
            c_h, c_a = st.columns(2)
            with c_h:
                st.markdown(f"**主隊 ({data.get('home_team', '主隊')}) 近況戰績**")
                if h_rec:
                    for item in h_rec:
                        st.write(f"- {item}")
                else:
                    st.caption("無截圖近況紀錄")
            with c_a:
                st.markdown(f"**客隊 ({data.get('away_team', '客隊')}) 近況戰績**")
                if a_rec:
                    for item in a_rec:
                        st.write(f"- {item}")
                else:
                    st.caption("無截圖近況紀錄")
    else:
        st.info("目前數據庫尚無已記錄的歷史賽事截圖數據。")
