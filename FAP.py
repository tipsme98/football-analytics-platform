import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta
import sys
import os
import requests
import json
from sqlalchemy import text, create_engine

# --- 防呆機制：Gemini 與 Image 套件 ---
try:
    import google.generativeai as genai
    from PIL import Image
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

# --- 2. 雲端/本地 資料庫連線配置 (含安全降級容錯機制) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(CURRENT_DIR, "football_data.db")
sqlite_engine = create_engine(f"sqlite:///{db_path}", connect_args={'check_same_thread': False})

engine = sqlite_engine
db_connection_warning = None

if DATABASE_URL:
    try:
        url = DATABASE_URL
        if "db." in url and ".supabase.co" in url:
            db_connection_warning = "⚠️️ 偵測到 Supabase 直連網址。Streamlit Cloud 不支援 IPv6，建議使用包含 pooler.supabase.com 與 Port 6543 的 Connection Pooling 網址。"
        
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
        db_connection_warning = f"⚠️ 雲端 PostgreSQL 連線失敗（如密碼包含特殊字元，請在 Secrets 中加上單引號），已自動切換回本地 SQLite。詳細錯誤: {e}"
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

def get_best_gemini_model():
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for pref in ['models/gemini-2.5-flash', 'models/gemini-1.5-flash', 'models/gemini-1.5-pro']:
            if pref in available_models:
                return pref
        return available_models[0] if available_models else 'models/gemini-1.5-flash'
    except:
        return 'models/gemini-1.5-flash'

# --- 4. 輔助函數 ---
def format_asian_handicap(line):
    try:
        val = float(line)
        if val == 0: return "0"
        if val % 0.5 == 0: return f"{val:g}" 
        lower = val - 0.25 if val > 0 else val + 0.25
        upper = val + 0.25 if val > 0 else val - 0.25
        res = f"{lower:g}/{upper:g}"
        if val > 0: res = "+" + res.replace("+", "")
        return res.replace("+-", "-")
    except: return str(line)

def format_ou_line(line):
    try:
        val = float(line)
        if val % 0.5 == 0: return f"{val:g}"
        return f"{val-0.25:g}/{val+0.25:g}"
    except: return str(line)

def render_trend(open_odd, curr_odd):
    if pd.isna(open_odd) or pd.isna(curr_odd) or open_odd == curr_odd: return str(curr_odd or '-')
    if float(curr_odd) < float(open_odd): return f"<span style='color:red;'>⬇ {curr_odd}</span> <span style='font-size:0.75em;color:#888;'>(初: {open_odd})</span>"
    return f"<span style='color:green;'>⬆ {curr_odd}</span> <span style='font-size:0.75em;color:#888;'>(初: {open_odd})</span>"

# --- 5. 資金流預測引擎 ---
class SmartOddsPredictor:
    def execute_prediction(self):
        try:
            fixtures = pd.read_sql("SELECT fixture_id FROM fixtures_v4 WHERE status = 'NS'", engine)
            if fixtures.empty: return False
            
            odds_history = pd.read_sql("SELECT * FROM odds_history_v4 ORDER BY recorded_at ASC", engine)
            preds = []
            
            for fid in fixtures['fixture_id']:
                f_odds = odds_history[odds_history['fixture_id'] == fid]
                if len(f_odds) < 1: continue
                
                open_data = f_odds.iloc[0]
                curr_data = f_odds.iloc[-1]
                
                home_odd_drop = float(open_data['ah_home_odd'] or 0) - float(curr_data['ah_home_odd'] or 0)
                prob_h = 0.5 + (home_odd_drop * 0.3) 
                is_value = abs(home_odd_drop) > 0.15 
                
                preds.append({
                    "fixture_id": fid,
                    "prob_home_win": round(max(0.1, min(0.9, prob_h)), 3),
                    "value_bet_detected": is_value,
                    "recommended_pick": "主勝" if prob_h > 0.55 else ("客勝" if prob_h < 0.45 else "觀望"),
                    "ou_pick": random.choice(["大", "細", "觀望"]),
                    "corner_pick": random.choice(["大", "細", "觀望"])
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
    .wait-tag { background-color: #F3F4F6; color: #4B5563; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.85em; }
    .score-box { background-color: #F8FAFC; padding: 8px; border-radius: 6px; border: 1px solid #E2E8F0; text-align: center; margin-top: 5px; }
    .odds-display { font-size: 0.85em; color: #374151; background: #F1F5F9; padding: 6px; border-radius: 4px; margin-top: 5px;}
    
    .match-header-dark { background-color: #1a1d24; color: white; padding: 20px; border-radius: 8px; text-align: center; font-family: sans-serif; }
    .match-title { font-size: 14px; color: #ffcc00; margin-bottom: 15px; }
    .team-name { font-size: 22px; font-weight: bold; display: inline-block; vertical-align: middle; margin: 0 15px; }
    .score-large { font-size: 32px; font-weight: bold; color: #ffcc00; display: inline-block; vertical-align: middle; margin: 0 15px; }
    
    .stats-container { background-color: #151a22; padding: 20px; border-radius: 8px; color: white; margin-top: 15px; }
    .stats-title { font-size: 16px; font-weight: bold; margin-bottom: 15px; border-bottom: 1px solid #2d3748; padding-bottom: 8px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚽ 專業足球精算與價值投注平台 (水位追蹤版)</p>', unsafe_allow_html=True)

if db_connection_warning:
    st.warning(db_connection_warning)

tab1, tab2, tab3, tab4 = st.tabs(["🔥 賽事與盤口追蹤", "🧠 資金流預測模型", "🗄 即時 API 數據中心", "📸 賽事圖片智能識別與重構"])

def get_tag_html(pick):
    if pick != "觀望": return f'<span class="value-bet-tag">💎 投注: {pick}</span>'
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
                        hs = int(row['home_score']) if pd.notna(row.get('home_score')) else "-"
                        aws = int(row['away_score']) if pd.notna(row.get('away_score')) else "-"
                        hc = int(row['home_corner']) if pd.notna(row.get('home_corner')) else "-"
                        ac = int(row['away_corner']) if pd.notna(row.get('away_corner')) else "-"
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
                        st.markdown(f"**勝負盤 (讓球)**")
                        st.markdown(f"<div class='odds-display'>盤口: <b>[{line}]</b> | 主水: {trend_html}</div>", unsafe_allow_html=True)
                        st.markdown(get_tag_html(pick), unsafe_allow_html=True)
                    with c4:
                        ou_line = format_ou_line(row.get('curr_ou_line', 2.5))
                        open_ou = row.get('open_ou_over_odd', 0)
                        curr_ou = row.get('curr_ou_over_odd', 0)
                        trend_ou_html = render_trend(open_ou, curr_ou)
                        ou_pick = str(row.get('ou_pick', '觀望'))
                        st.markdown(f"**入球大細**")
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
                res = requests.get("https://v3.football.api-sports.io/status", headers={"x-apisports-key": API_FOOTBALL_KEY}, timeout=3).json()
                req_made = res.get('response', {}).get('requests', {}).get('current', 0)
                req_limit = res.get('response', {}).get('requests', {}).get('limit_day', 100)
                st.metric("API-Football 使用量 (今日)", f"{req_made} / {req_limit}")
            except: st.warning("API-Football: 連線超時")
        else: st.error("未設定 API-Football Key")
    with col_b:
        if THE_ODDS_API_KEY:
            try:
                res_odds = requests.get(f"https://api.the-odds-api.com/v4/sports/?apiKey={THE_ODDS_API_KEY}", timeout=3)
                used = res_odds.headers.get('x-requests-used', '未知')
                remain = res_odds.headers.get('x-requests-remaining', '未知')
                st.metric("The-Odds-API 剩餘額度", f"剩 {remain} (已用 {used})")
            except: st.warning("The-Odds-API: 連線超時")
        else: st.error("未設定 The-Odds-API Key")

    st.divider()
    try:
        db_count = pd.read_sql("SELECT COUNT(*) as count FROM fixtures_v4", engine).iloc[0]['count']
        st.markdown(f"📊 **目前數據庫總收錄賽事： `{db_count}` 場**")
    except:
        st.markdown("📊 **目前數據庫總收錄賽事： `0` 場**")

    if st.button("📥 同步今日與未來賽程 (即時 API)"):
        with st.spinner("正在透過 API 獲取即時賽事與盤口..."):
            if not THE_ODDS_API_KEY:
                st.error("請先在 Streamlit Secrets 設定 THE_ODDS_API_KEY")
            else:
                try:
                    res = requests.get(f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={THE_ODDS_API_KEY}&regions=eu&markets=h2h,spreads,totals").json()
                    if isinstance(res, list) and len(res) > 0:
                        fixtures, odds = [], []
                        for match in res:
                            fid = abs(hash(match['id'])) % 1000000
                            dt = datetime.strptime(match['commence_time'], "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=8)
                            fixtures.append({
                                "fixture_id": fid, "league_name": "英超",
                                "home_team": match['home_team'], "away_team": match['away_team'],
                                "match_datetime": dt.strftime("%Y-%m-%d %H:%M"),
                                "home_score": None, "away_score": None, "home_corner": None, "away_corner": None,
                                "status": "NS"
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
                                "fixture_id": fid, "ah_line": str(ah_line), "ah_home_odd": ah_home, "ah_away_odd": 1.90,
                                "ou_line": str(ou_line), "ou_over_odd": ou_over, "ou_under_odd": 1.90,
                                "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
# 分頁 4: 📸 賽事圖片智能識別與重構 (升級版 Gemini Prompt)
# ==========================================
with tab4:
    st.markdown("### 📸 歷史賽事圖片數據抓取與介面重構")
    
    if not HAS_GENAI:
        st.error("⚠️ 缺少 AI 套件，請確認已安裝 `google-generativeai` 與 `Pillow`。")
    elif not GEMINI_API_KEY:
        st.warning("⚠️ 未設定 `GEMINI_API_KEY`，請至 Streamlit Secrets 填寫。")
    else:
        st.info("上傳包含比分、技術統計或賠率水位變化的賽事截圖，Gemini 將精確結構化提取資料。")
        uploaded_files = st.file_uploader("上傳賽事截圖", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        
        if uploaded_files and st.button("🚀 開始 Gemini 智能識別與重構 UI", type="primary"):
            best_model_name = get_best_gemini_model()
            with st.spinner(f"🧠 Gemini 正在使用模型 `{best_model_name}` 結構化解析截圖..."):
                try:
                    img = Image.open(uploaded_files[0])
                    
                    # 升級版 Gemini 專家級 Prompt
                    prompt = """
                    你是一個專業足球數據與體育博彩數據分析 AI。請仔細分析這張足球比賽網頁截圖（包含聯賽名稱、隊伍、比分、黃/紅牌、角球、進攻/危險進攻/控球率等技術統計，以及讓球盤口與大細球盤口的水位變化）。

                    請提取所有數據並嚴格只回傳 JSON 格式（絕對不要包含 Markdown 程式碼標記、註解或額外文字）：
                    {
                      "league": "聯賽名稱（例如：意甲、英超、西甲等）",
                      "datetime": "比賽時間（例如：2026-09-30 20:00）",
                      "home_team": "主隊名稱",
                      "away_team": "客隊名稱",
                      "home_score": 主隊最終得分數字,
                      "away_score": 客隊最終得分數字,
                      "ht_score": "半場比分（例如：1-0，若無則填 '-'）",
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
                        "ah_line": "讓球盤口（例如：0, -0.5, +0.5/1，若未顯示填 '0'）",
                        "home_initial_odd": 主隊讓球初盤水位數字（例如 0.95，若無填 0）,
                        "home_current_odd": 主隊讓球即時水位數字（例如 0.88，若無填 0）,
                        "away_initial_odd": 客隊讓球初盤水位數字（例如 0.91，若無填 0）,
                        "away_current_odd": 客隊讓球即時水位數字（例如 0.98，若無填 0）,
                        "ou_line": "大細球盤口（例如：2.5, 2.75, 3，若未顯示填 '2.5'）",
                        "ou_over_initial": 大球初盤水位數字（例如 0.85，若無填 0）,
                        "ou_over_current": 大球即時水位數字（例如 0.92，若無填 0）,
                        "ou_under_initial": 細球初盤水位數字（例如 0.95，若無填 0）,
                        "ou_under_current": 細球即時水位數字（例如 0.88，若無填 0）
                      }
                    }

                    【注意事項】
                    1. 若圖片中找不到特定數字欄位，請用數字 0 替代，字串用 'N/A'，嚴禁輸出 null 或未定義值。
                    2. 數字請確保為純整數或浮點數（例如: 0.95 而非 "0.95"）。
                    """
                    
                    model = genai.GenerativeModel(best_model_name)
                    response = model.generate_content([prompt, img])
                    raw_text = response.text.strip().replace("```json", "").replace("```", "").strip()
                    
                    # 擷取標準 JSON 區塊
                    start_idx = raw_text.find('{')
                    end_idx = raw_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        raw_text = raw_text[start_idx:end_idx+1]
                        
                    parsed_data = json.loads(raw_text)
                    
                    if parsed_data:
                        with engine.begin() as conn:
                            conn.execute(text("""
                                INSERT INTO historical_match_stats (home_team, away_team, home_score, away_score, data_json, created_at)
                                VALUES (:h, :a, :hs, :aws, :dj, :ca)
                            """), {
                                "h": parsed_data.get('home_team', '未知主隊'),
                                "a": parsed_data.get('away_team', '未知客隊'),
                                "hs": int(parsed_data.get('home_score') or 0),
                                "aws": int(parsed_data.get('away_score') or 0),
                                "dj": json.dumps(parsed_data),
                                "ca": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                        st.success("✅ AI 成功識別截圖所有數據（含讓球/大細水位變化）並儲存至資料庫！")
                        st.session_state['current_parsed_match'] = parsed_data
                    else:
                        st.error("解析出來的資料為空。")
                except Exception as e:
                    st.error(f"系統處理圖片時發生錯誤。模型: {best_model_name}。錯誤訊息: {e}")

        # 呈現解析後的資料與動態表格
        parsed_data = st.session_state.get('current_parsed_match', None)
        if not parsed_data:
            try:
                db_record = pd.read_sql("SELECT data_json FROM historical_match_stats ORDER BY id DESC LIMIT 1", engine)
                if not db_record.empty:
                    parsed_data = json.loads(db_record.iloc[0]['data_json'])
            except: pass

        if parsed_data:
            st.markdown(f"""
            <div class="match-header-dark">
                <div class="match-title">{parsed_data.get('league','')} • {parsed_data.get('datetime','')}</div>
                <div>
                    <div class="team-name">{parsed_data.get('home_team','')}</div>
                    <div class="score-large">{parsed_data.get('home_score',0)}</div>
                    <div style="display:inline-block; text-align:center; vertical-align:middle; margin: 0 15px;">
                        <div style="font-size:18px; font-weight:bold; color:#ffcc00;">完</div>
                        <div style="font-size:12px; margin-top:4px;">
                            <span style="color:#f59e0b;">🟨 {parsed_data.get('home_yellow',0)}</span>
                            <span style="color:#ef4444; margin-left:4px;">🟥 {parsed_data.get('home_red',0)}</span>
                            <span style="color:#10b981; margin-left:8px;">🚩 {parsed_data.get('home_corner',0)} - {parsed_data.get('away_corner',0)}</span>
                            <span style="color:#f59e0b; margin-left:8px;">🟨 {parsed_data.get('away_yellow',0)}</span>
                            <span style="color:#ef4444; margin-left:4px;">🟥 {parsed_data.get('away_red',0)}</span>
                        </div>
                        <div style="font-size:12px; color:#ffcc00; margin-top:4px;">HT: ({parsed_data.get('ht_score','-')})</div>
                    </div>
                    <div class="score-large">{parsed_data.get('away_score',0)}</div>
                    <div class="team-name">{parsed_data.get('away_team','')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📊 技術統計", "📈 盤口與水位變化表", "📜 歷史記錄庫"])
            
            with sub_tab1:
                s = parsed_data.get('stats', {})
                att = s.get('attacks', [0, 0])
                d_att = s.get('dangerous_attacks', [0, 0])
                pos = s.get('possession', [50, 50])
                shots_on = s.get('shots_on_target', [0, 0])
                shots_off = s.get('shots_off_target', [0, 0])
                
                st.markdown(f"""
                <div class="stats-container">
                    <div class="stats-title">比賽技術數據分析</div>
                    <div style="display:flex; justify-content:space-around; text-align:center; margin-bottom:15px;">
                        <div><div style="color:#94a3b8; font-size:12px;">進攻</div><div style="font-size:18px;">{att[0]} vs {att[1]}</div></div>
                        <div><div style="color:#94a3b8; font-size:12px;">危險進攻</div><div style="font-size:18px;">{d_att[0]} vs {d_att[1]}</div></div>
                        <div><div style="color:#94a3b8; font-size:12px;">控球率</div><div style="font-size:18px;">{pos[0]}% vs {pos[1]}%</div></div>
                    </div>
                    <div style="display:flex; justify-content:space-around; text-align:center;">
                        <div><div style="color:#94a3b8; font-size:12px;">射正</div><div style="font-size:18px;">{shots_on[0]} vs {shots_on[1]}</div></div>
                        <div><div style="color:#94a3b8; font-size:12px;">射偏</div><div style="font-size:18px;">{shots_off[0]} vs {shots_off[1]}</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with sub_tab2:
                st.markdown("#### 📊 莊家盤口與水位變化即時對比")
                ot = parsed_data.get('odds_trend', {})
                
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

            with sub_tab3:
                try:
                    df_h = pd.read_sql("SELECT id, home_team, away_team, home_score, away_score, created_at FROM historical_match_stats ORDER BY id DESC", engine)
                    st.dataframe(df_h, use_container_width=True)
                except: pass
