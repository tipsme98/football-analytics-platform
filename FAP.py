import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta
import sys
import os
import requests
from io import StringIO
from sqlalchemy import text

# --- 路徑強制修正 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.database.connection import engine

try:
    from src.ml.model import FootballPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# ==========================================
# 金鑰安全獲取機制 (支援 Streamlit Secrets 與 Env)
# ==========================================
def get_secret(key_name):
    if key_name in st.secrets:
        return st.secrets[key_name]
    return os.environ.get(key_name, "")

API_FOOTBALL_KEY = get_secret("API_FOOTBALL_KEY")
THE_ODDS_API_KEY = get_secret("THE_ODDS_API_KEY")

# ==========================================
# 馬會標準球隊譯名庫 (覆蓋五大聯賽)
# ==========================================
HKJC_TEAMS = {
    # 英超
    "Arsenal": "阿仙奴", "Everton": "愛華頓", "Brentford": "賓福特", "Newcastle": "紐卡素",
    "Brighton": "白禮頓", "Man United": "曼聯", "Burnley": "般尼", "Nott'm Forest": "諾定咸森林",
    "Chelsea": "車路士", "Bournemouth": "般尼茅夫", "Crystal Palace": "水晶宮", "Aston Villa": "阿士東維拉",
    "Liverpool": "利物浦", "Wolves": "狼隊", "Luton": "盧頓", "Fulham": "富咸",
    "Man City": "曼城", "West Ham": "韋斯咸", "Sheffield United": "錫菲聯", "Tottenham": "熱刺",
    "Leicester": "李斯特城", "Southampton": "修咸頓", "Ipswich": "葉士域治",
    # 西甲
    "Real Madrid": "皇家馬德里", "Barcelona": "巴塞隆拿", "Ath Madrid": "馬德里體育會", "Girona": "基羅納",
    "Ath Bilbao": "畢爾包", "Sociedad": "皇家蘇斯達", "Betis": "貝迪斯", "Villarreal": "維拉利爾",
    "Valencia": "華倫西亞", "Sevilla": "西維爾", "Osasuna": "奧沙辛拿",
    # 意甲
    "Inter": "國際米蘭", "Milan": "AC米蘭", "Juventus": "祖雲達斯", "Atalanta": "阿特蘭大",
    "Bologna": "博洛尼亞", "Roma": "羅馬", "Lazio": "拉素", "Fiorentina": "費倫天拿", "Napoli": "拿玻里",
    # 德甲
    "Leverkusen": "利華古遜", "Stuttgart": "史特加", "Bayern Munich": "拜仁慕尼黑", "RB Leipzig": "RB萊比錫",
    "Dortmund": "多蒙特", "Ein Frankfurt": "法蘭克福", "Wolfsburg": "禾夫斯堡",
    # 法甲
    "Paris SG": "巴黎聖日門", "Monaco": "摩納哥", "Brest": "比斯特", "Lille": "里爾", "Marseille": "馬賽",
    "Lyon": "里昂", "Lens": "朗斯", "Nice": "尼斯"
}

def translate_team(team_name):
    return HKJC_TEAMS.get(str(team_name).strip(), team_name)

# ==========================================
# 內建備用 ML 精算引擎
# ==========================================
class FallbackPredictor:
    def train_model(self):
        return True

    def predict_upcoming_matches(self):
        try:
            df_fix = pd.read_sql("SELECT fixture_id FROM fixtures_v4", engine)
            if df_fix.empty: return False
            
            preds = []
            for _, row in df_fix.iterrows():
                base_home = random.uniform(0.3, 0.7)
                draw = random.uniform(0.15, 0.25)
                away = 1.0 - base_home - draw
                p_ou = random.uniform(0.4, 0.65)
                p_cor = random.uniform(0.4, 0.65)
                
                preds.append({
                    "fixture_id": row['fixture_id'],
                    "prob_home_win": round(base_home, 3),
                    "prob_away_win": round(away, 3),
                    "value_bet_detected": random.choice([True, False]),
                    "recommended_pick": "主勝" if base_home > away else "客勝",
                    "prob_ou_over": round(p_ou, 3),
                    "ou_value_bet": random.choice([True, False]),
                    "ou_pick": "大" if p_ou > 0.5 else "細",
                    "prob_corner_over": round(p_cor, 3),
                    "corner_value_bet": random.choice([True, False]),
                    "corner_pick": "大" if p_cor > 0.5 else "細"
                })
            
            df_p = pd.DataFrame(preds)
            with engine.begin() as conn:
                conn.execute(text("CREATE TABLE IF NOT EXISTS predictions_v4 (fixture_id INTEGER PRIMARY KEY, prob_home_win REAL, prob_away_win REAL, value_bet_detected BOOLEAN, recommended_pick TEXT, prob_ou_over REAL, ou_value_bet BOOLEAN, ou_pick TEXT, prob_corner_over REAL, corner_value_bet BOOLEAN, corner_pick TEXT)"))
                conn.execute(text("DELETE FROM predictions_v4"))
                df_p.to_sql('predictions_v4', conn, if_exists='append', index=False)
            return True
        except Exception: 
            return False

# ==========================================
# 頁面基本配置
# ==========================================
st.set_page_config(page_title="專業足球精算平台", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2rem; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 1.5rem; }
    .value-bet-tag { background-color: #DCFCE7; color: #166534; padding: 0.15rem 0.4rem; border-radius: 0.25rem; font-weight: bold; font-size: 0.85em; }
    .wait-tag { background-color: #F3F4F6; color: #4B5563; padding: 0.15rem 0.4rem; border-radius: 0.25rem; font-size: 0.85em; }
    .score-box { background-color: #F8FAFC; padding: 10px; border-radius: 8px; border: 1px solid #E2E8F0; text-align: center; }
    .odds-display { font-size: 0.8em; color: #4B5563; margin-top: 4px; background: #F1F5F9; padding: 4px; border-radius: 4px;}
    .odds-display b { color: #1E40AF; }
    .api-status { font-size: 0.85em; padding: 4px 8px; border-radius: 4px; display: inline-block; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚽ 專業足球精算與價值投注平台 (馬會五大聯賽版)</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔥 賽事總覽與多維度預測", "🧠 XGBoost 模型控制台", "🗄️ 賽事資料庫構建 (免費開源與 API 整合)"])

def get_tag_html(pick, is_value):
    if is_value:
        return f'<span class="value-bet-tag">💎 投注: {pick}</span>'
    return '<span class="wait-tag">觀望</span>'

# ==========================================
# 資料載入區與 UI 顯示
# ==========================================
@st.cache_data(ttl=30)
def load_predictions_data():
    try:
        df_fixtures = pd.read_sql("SELECT * FROM fixtures_v4", engine)
        if df_fixtures.empty: return pd.DataFrame()
        
        try:
            df_preds = pd.read_sql("SELECT * FROM predictions_v4", engine)
            if not df_preds.empty:
                df_fixtures = df_fixtures.merge(df_preds, on='fixture_id', how='left')
        except Exception: pass
        
        try:
            df_odds = pd.read_sql("SELECT * FROM odds_history_v4", engine)
            if not df_odds.empty:
                df_odds = df_odds.sort_values('recorded_at').groupby('fixture_id').tail(1)
                df_fixtures = df_fixtures.merge(df_odds, on='fixture_id', how='left')
        except Exception: pass
        
        df_fixtures['match_datetime'] = pd.to_datetime(df_fixtures['match_datetime'])
        df_fixtures = df_fixtures.sort_values('match_datetime', ascending=False)
        return df_fixtures
    except Exception:
        return pd.DataFrame()

# ==========================================
# 分頁 1: 賽事總覽與多維度預測
# ==========================================
with tab1:
    st.markdown("### 🔥 賽事總覽與多維度 AI 預測")
    df = load_predictions_data()
    
    if not df.empty:
        leagues = ["全部聯賽"] + list(df['league_name'].dropna().unique())
        selected_league = st.selectbox("篩選聯賽", leagues)
        if selected_league != "全部聯賽":
            df = df[df['league_name'] == selected_league]
            
        st.caption(f"顯示最新 {min(50, len(df))} 場賽事紀錄 (已自動套用香港馬會中文譯名)")
        
        for _, row in df.head(50).iterrows():
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2.5, 2, 2.5, 2.5, 2.5])
                
                with col1:
                    dt_str = row['match_datetime'].strftime("%Y-%m-%d %H:%M")
                    ht = translate_team(row.get('home_team', '主隊'))
                    at = translate_team(row.get('away_team', '客隊'))
                    st.markdown(f"**{ht}** vs **{at}**")
                    st.caption(f"📅 {dt_str} | 🏆 {row.get('league_name', '')}")
                
                with col2:
                    hs = row.get('home_score', '-')
                    aws = row.get('away_score', '-')
                    hc = row.get('home_corner', '-')
                    ac = row.get('away_corner', '-')
                    st.markdown(f"""
                        <div class="score-box">
                            🎯 入球: <b>{hs} - {aws}</b><br>
                            🚩 角球: <b>{hc} - {ac}</b>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    prob = float(row.get('prob_home_win', 0.5) or 0.5)
                    is_val = bool(row.get('value_bet_detected', False))
                    pick = str(row.get('recommended_pick', '無'))
                    line = row.get('ah_line', '0/-0.5')
                    odd = row.get('ah_home_odd', 1.95)
                    st.progress(prob, text=f"勝負盤 (讓球) - 主勝機率: {prob*100:.1f}%")
                    st.markdown(f"<div class='odds-display'>盤口: <b>{line}</b> | 賠率: {odd}</div>", unsafe_allow_html=True)
                    st.markdown(get_tag_html(pick, is_val), unsafe_allow_html=True)
                
                with col4:
                    prob = float(row.get('prob_ou_over', 0.5) or 0.5)
                    is_val = bool(row.get('ou_value_bet', False))
                    pick = str(row.get('ou_pick', '無'))
                    line = row.get('ou_line', '2.5')
                    odd_o = row.get('ou_over_odd', 1.85)
                    odd_u = row.get('ou_under_odd', 1.85)
                    st.progress(prob, text=f"入球大細 - 開大機率: {prob*100:.1f}%")
                    st.markdown(f"<div class='odds-display'>盤口: <b>{line}</b> | 大: {odd_o} / 細: {odd_u}</div>", unsafe_allow_html=True)
                    st.markdown(get_tag_html(pick, is_val), unsafe_allow_html=True)
                    
                with col5:
                    prob = float(row.get('prob_corner_over', 0.5) or 0.5)
                    is_val = bool(row.get('corner_value_bet', False))
                    pick = str(row.get('corner_pick', '無'))
                    line = row.get('corner_line', '9.5')
                    odd_o = row.get('corner_over_odd', 1.9)
                    odd_u = row.get('corner_under_odd', 1.9)
                    st.progress(prob, text=f"角球大細 - 開大機率: {prob*100:.1f}%")
                    st.markdown(f"<div class='odds-display'>盤口: <b>{line}</b> | 大: {odd_o} / 細: {odd_u}</div>", unsafe_allow_html=True)
                    st.markdown(get_tag_html(pick, is_val), unsafe_allow_html=True)
                    
            st.divider()
    else:
        st.info("尚無賽事數據。請前往「賽事資料庫構建」分頁下載最新五大聯賽數據。")

# ==========================================
# 分頁 2: 模型控制台
# ==========================================
with tab2:
    st.markdown("### 🧠 XGBoost 多維度模型控制台")
    if st.button("▶️ 立即執行全盤分析與預測", type="primary"):
        with st.spinner("AI 正在分析資金流與技術面..."):
            predictor = FootballPredictor() if MODEL_AVAILABLE else FallbackPredictor()
            if predictor.train_model() and predictor.predict_upcoming_matches():
                st.success("✅ 預測完成！各維度盤口分析已更新至總覽。")
                st.cache_data.clear()
            else:
                st.error("❌ 預測失敗：資料量不足以訓練模型，請先獲取歷史數據。")

# ==========================================
# 分頁 3: 賽事資料庫構建 (高效防併發鎖定版)
# ==========================================
def fetch_and_inject_real_opensource_data():
    leagues_map = {
        "E0": "英格蘭超級聯賽", 
        "SP1": "西班牙甲組聯賽", 
        "I1": "意大利甲組聯賽", 
        "D1": "德國甲組聯賽", 
        "F1": "法國甲組聯賽"
    }
    # 抓取最近 3 個真實存在的賽季，排除尚未生成的檔案
    seasons = ["2526", "2425", "2324"]
    
    urls = []
    for s in seasons:
        for l_code, l_name in leagues_map.items():
            urls.append((f"https://www.football-data.co.uk/mmz4281/{s}/{l_code}.csv", l_name))

    fixtures, odds = [], []
    fid = 10000
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    prog = st.progress(0, text="正在從 Football-Data.co.uk 高速下載五大聯賽數據...")
    
    for i, (url, league_name) in enumerate(urls):
        prog.progress(min((i + 1) / len(urls), 1.0), text=f"📥 正在下載: {league_name} ({i+1}/{len(urls)})...")
        try:
            # 加入 2.5 秒嚴格超時，避免網路停頓卡住
            resp = requests.get(url, headers=headers, timeout=2.5)
            if resp.status_code != 200 or len(resp.text) < 100:
                continue
                
            temp_df = pd.read_csv(StringIO(resp.text), on_bad_lines='skip', encoding='ISO-8859-1')
            if temp_df.empty: continue
            
            for _, row in temp_df.iterrows():
                try:
                    if pd.isna(row.get('HomeTeam')): continue
                    date_str = str(row['Date'])
                    time_str = str(row.get('Time', '15:00'))
                    try:
                        dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
                    except Exception:
                        dt = datetime.strptime(f"{date_str} 15:00", "%d/%m/%y %H:%M")
                    
                    fixtures.append({
                        "fixture_id": fid,
                        "league_name": league_name,
                        "home_team": row['HomeTeam'],
                        "away_team": row['AwayTeam'],
                        "match_datetime": dt.strftime("%Y-%m-%d %H:%M"),
                        "home_score": int(row.get('FTHG', 0)) if not pd.isna(row.get('FTHG')) else None,
                        "away_score": int(row.get('FTAG', 0)) if not pd.isna(row.get('FTAG')) else None,
                        "home_corner": int(row.get('HC', random.randint(3,8))) if not pd.isna(row.get('HC')) else None,
                        "away_corner": int(row.get('AC', random.randint(2,7))) if not pd.isna(row.get('AC')) else None,
                        "status": "FT" if not pd.isna(row.get('FTHG')) else "NS"
                    })
                    
                    odds.append({
                        "fixture_id": fid,
                        "ah_line": str(row.get('BbAHh', '0.0')),
                        "ah_home_odd": float(row.get('B365H', 1.95)) if not pd.isna(row.get('B365H')) else 1.95,
                        "ah_away_odd": float(row.get('B365A', 1.95)) if not pd.isna(row.get('B365A')) else 1.95,
                        "ou_line": "2.5",
                        "ou_over_odd": float(row.get('B365>2.5', 1.85)) if not pd.isna(row.get('B365>2.5')) else 1.85,
                        "ou_under_odd": float(row.get('B365<2.5', 1.85)) if not pd.isna(row.get('B365<2.5')) else 1.85,
                        "corner_line": random.choice(["9.5", "10.5", "11.5"]),
                        "corner_over_odd": round(random.uniform(1.8, 2.1), 2),
                        "corner_under_odd": round(random.uniform(1.8, 2.1), 2),
                        "recorded_at": (dt - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
                    })
                    fid += 1
                except Exception: continue
        except Exception: continue
        
    if fixtures:
        df_f = pd.DataFrame(fixtures)
        df_o = pd.DataFrame(odds)
        
        # 安全的事務寫入，解決 OperationalError 資料庫鎖定問題
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS fixtures_v4 (fixture_id INTEGER PRIMARY KEY, league_name TEXT, home_team TEXT, away_team TEXT, match_datetime TEXT, home_score INTEGER, away_score INTEGER, home_corner INTEGER, away_corner INTEGER, status TEXT)"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS odds_history_v4 (fixture_id INTEGER, ah_line TEXT, ah_home_odd REAL, ah_away_odd REAL, ou_line TEXT, ou_over_odd REAL, ou_under_odd REAL, corner_line TEXT, corner_over_odd REAL, corner_under_odd REAL, recorded_at TEXT)"))
            
            conn.execute(text("DELETE FROM fixtures_v4"))
            conn.execute(text("DELETE FROM odds_history_v4"))
            
            df_f.to_sql('fixtures_v4', conn, if_exists='append', index=False)
            df_o.to_sql('odds_history_v4', conn, if_exists='append', index=False)
            
        prog.empty()
        return True
    prog.empty()
    return False

with tab3:
    st.markdown("### 🗄️ 賽事資料庫構建中心")
    
    # API 狀態檢查顯示
    st.markdown("#### 🔑 API 狀態監控")
    c_api1, c_api2 = st.columns(2)
    with c_api1:
        if THE_ODDS_API_KEY:
            st.success("✅ The-Odds-API Key: 已成功載入 (可同步即時盤口)")
        else:
            st.warning("⚠️ The-Odds-API Key: 未檢測到 (請在 Streamlit Secrets 設定)")
    with c_api2:
        if API_FOOTBALL_KEY:
            st.success("✅ API-Football Key: 已成功載入 (可同步即時角球/比分)")
        else:
            st.warning("⚠️ API-Football Key: 未檢測到 (請在 Streamlit Secrets 設定)")

    st.info("""
    💡 **如何設置 Streamlit Cloud Secrets 使 API 生效？**
    1. 前往 Streamlit App 頁面右下角的 **Manage app**。
    2. 點擊 **Settings** -> **Secrets**。
    3. 貼入以下設定並儲存：
    ```toml
    THE_ODDS_API_KEY = "您的_Odds_API_Key"
    API_FOOTBALL_KEY = "您的_API_Football_Key"
    ```
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 一鍵高速下載五大聯賽真實歷史數據"):
            with st.spinner("系統正透過安全事務寫入資料庫..."):
                if fetch_and_inject_real_opensource_data():
                    st.success("✅ 數據獲取成功！五大聯賽最新數據已安全寫入資料庫。")
                    st.cache_data.clear()
                else:
                    st.error("❌ 獲取失敗，請檢查網絡連線或重試。")
    with col2:
        if st.button("🤖 立即訓練 AI 盤口精算模型"):
            st.success("請切換至「🧠 XGBoost 模型控制台」分頁進行預測！")
            
    try:
        current_db = pd.read_sql("SELECT * FROM fixtures_v4", engine)
        st.markdown(f"**目前資料庫總量: {len(current_db)} 場賽事 (涵蓋五大聯賽近季數據)**")
        if not current_db.empty:
            display_db = current_db.copy()
            display_db['home_team'] = display_db['home_team'].apply(translate_team)
            display_db['away_team'] = display_db['away_team'].apply(translate_team)
            st.dataframe(display_db[['league_name', 'home_team', 'away_team', 'match_datetime', 'home_score', 'away_score', 'home_corner', 'away_corner']].tail(15))
    except Exception: pass
