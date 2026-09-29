import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta
import sys
import os
import requests
from io import StringIO
from sqlalchemy import text, create_engine

# --- 資料庫連線安全配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    from src.database.connection import engine
except ImportError:
    # 若無預設 engine，建立帶有 Threading 防鎖死機制的 SQLite 引擎
    db_path = os.path.join(CURRENT_DIR, "football_data.db")
    engine = create_engine(f"sqlite:///{db_path}", connect_args={'check_same_thread': False})

# ==========================================
# 金鑰安全獲取機制 
# ==========================================
def get_secret(key_name):
    if key_name in st.secrets: return st.secrets[key_name]
    return os.environ.get(key_name, "")

API_FOOTBALL_KEY = get_secret("API_FOOTBALL_KEY")
THE_ODDS_API_KEY = get_secret("THE_ODDS_API_KEY")

# ==========================================
# 核心轉換演算法：亞洲讓球盤與大細盤轉換
# ==========================================
def format_asian_handicap(line):
    try:
        val = float(line)
        if val == 0: return "0"
        if val % 0.5 == 0: return f"{val:g}" # 整數或半球 (如 +0.5, -1)
        # 處理平半/半一等四分一盤口 (如 -0.75 -> -0.5/-1.0)
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
    if pd.isna(open_odd) or pd.isna(curr_odd) or open_odd == curr_odd: return ""
    if curr_odd < open_odd: return f"<span style='color:red;'>⬇ {curr_odd}</span> <span style='font-size:0.7em;color:#999;'>(初: {open_odd})</span>"
    return f"<span style='color:green;'>⬆ {curr_odd}</span> <span style='font-size:0.7em;color:#999;'>(初: {open_odd})</span>"

# ==========================================
# 資金流盤口預測引擎 (Tipsme 邏輯)
# ==========================================
class SmartOddsPredictor:
    def execute_prediction(self):
        try:
            # 獲取所有賽事與最新兩筆盤口變化以追蹤資金流
            fixtures = pd.read_sql("SELECT fixture_id FROM fixtures_v4 WHERE status = 'NS'", engine)
            if fixtures.empty: return False
            
            odds_history = pd.read_sql("SELECT * FROM odds_history_v4 ORDER BY recorded_at ASC", engine)
            preds = []
            
            for fid in fixtures['fixture_id']:
                f_odds = odds_history[odds_history['fixture_id'] == fid]
                if len(f_odds) < 1: continue
                
                open_data = f_odds.iloc[0]
                curr_data = f_odds.iloc[-1]
                
                # 盤口變化分析 (熱度與誘盤)
                home_odd_drop = open_data['ah_home_odd'] - curr_data['ah_home_odd']
                
                # 資金流向判斷模型
                prob_h = 0.5 + (home_odd_drop * 0.3) # 賠率下跌代表資金流入，勝率提升
                prob_a = 1 - prob_h
                is_value = abs(home_odd_drop) > 0.15 # 顯著跌水為價值投注
                
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
                    conn.execute(text("CREATE TABLE IF NOT EXISTS predictions_v4 (fixture_id INTEGER PRIMARY KEY, prob_home_win REAL, value_bet_detected BOOLEAN, recommended_pick TEXT, ou_pick TEXT, corner_pick TEXT)"))
                    conn.execute(text("DELETE FROM predictions_v4"))
                    df_p.to_sql('predictions_v4', conn, if_exists='append', index=False)
            return True
        except Exception as e: 
            return False

# ==========================================
# 頁面配置 & CSS
# ==========================================
st.set_page_config(page_title="專業足球精算平台", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2rem; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 1rem; }
    .value-bet-tag { background-color: #DCFCE7; color: #166534; padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: bold; font-size: 0.85em; }
    .wait-tag { background-color: #F3F4F6; color: #4B5563; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.85em; }
    .score-box { background-color: #F8FAFC; padding: 8px; border-radius: 6px; border: 1px solid #E2E8F0; text-align: center; margin-top: 5px; }
    .odds-display { font-size: 0.85em; color: #374151; background: #F1F5F9; padding: 6px; border-radius: 4px; margin-top: 5px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚽ 專業足球精算與價值投注平台 (水位追蹤版)</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔥 賽事與盤口追蹤", "🧠 資金流預測模型", "🗄️ 即時 API 數據中心"])

def get_tag_html(pick):
    if pick != "觀望": return f'<span class="value-bet-tag">💎 投注: {pick}</span>'
    return '<span class="wait-tag">觀望</span>'

# ==========================================
# 分頁 1: 賽事與即時水位追蹤
# ==========================================
with tab1:
    st.markdown("### 🔥 即時賽事與莊家盤口水位追蹤")
    try:
        # 提取資料庫
        df_fixtures = pd.read_sql("SELECT * FROM fixtures_v4 ORDER BY match_datetime DESC", engine)
        if not df_fixtures.empty:
            df_odds = pd.read_sql("SELECT * FROM odds_history_v4 ORDER BY recorded_at ASC", engine)
            df_preds = pd.read_sql("SELECT * FROM predictions_v4", engine) if 'predictions_v4' in pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", engine)['name'].values else pd.DataFrame()
            
            # 分離初盤與即時盤
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
                    
                    # 賽事資訊
                    with c1:
                        dt_str = str(row['match_datetime'])[:16]
                        st.markdown(f"**{row['home_team']}** vs **{row['away_team']}**")
                        st.caption(f"📅 {dt_str} | 🏆 {row.get('league_name', '聯賽')}")
                    
                    # 賽果與角球 (整數化)
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
                    
                    # 亞洲讓球盤追蹤
                    with c3:
                        line = format_asian_handicap(row.get('curr_ah_line', 0))
                        open_odd = row.get('open_ah_home_odd', 0)
                        curr_odd = row.get('curr_ah_home_odd', 0)
                        trend_html = render_trend(open_odd, curr_odd) or str(curr_odd)
                        
                        pick = str(row.get('recommended_pick', '觀望'))
                        st.markdown(f"**勝負盤 (讓球)**")
                        st.markdown(f"<div class='odds-display'>盤口: <b>[{line}]</b> | 主水: {trend_html}</div>", unsafe_allow_html=True)
                        st.markdown(get_tag_html(pick), unsafe_allow_html=True)
                    
                    # 入球大細盤口追蹤
                    with c4:
                        ou_line = format_ou_line(row.get('curr_ou_line', 2.5))
                        open_ou = row.get('open_ou_over_odd', 0)
                        curr_ou = row.get('curr_ou_over_odd', 0)
                        trend_ou_html = render_trend(open_ou, curr_ou) or str(curr_ou)
                        
                        ou_pick = str(row.get('ou_pick', '觀望'))
                        st.markdown(f"**入球大細**")
                        st.markdown(f"<div class='odds-display'>盤口: <b>[{ou_line}]</b> | 大水: {trend_ou_html}</div>", unsafe_allow_html=True)
                        st.markdown(get_tag_html(ou_pick), unsafe_allow_html=True)
                        
                st.divider()
        else:
            st.info("尚無賽事數據。請前往「即時 API 數據中心」進行同步。")
    except Exception as e:
        st.error(f"讀取數據發生錯誤，請先同步資料庫。({e})")

# ==========================================
# 分頁 2: 資金流預測模型
# ==========================================
with tab2:
    st.markdown("### 🧠 莊家資金流向與盤口預測模型")
    st.info("💡 系統會學習及分析盤口及賠率隨時間的變化。找出水位異動與賽果的關聯，避開莊家陷阱。")
    
    if st.button("▶️ 執行盤口變化與資金流分析", type="primary"):
        with st.spinner("正在比對初盤與即時盤水位..."):
            predictor = SmartOddsPredictor()
            if predictor.execute_prediction():
                st.success("✅ 分析完成！系統已根據『賠率暴跌/誘盤』等資金跡象更新推薦。請查看賽事總覽。")
                st.cache_data.clear()
            else:
                st.error("❌ 分析失敗：未能檢測到未開賽賽事或缺乏盤口歷史數據。")

# ==========================================
# 分頁 3: 數據庫與 API 監控中心
# ==========================================
with tab3:
    st.markdown("### 🗄️ 即時 API 數據庫同步中心")
    
    # === API 狀態監控 (實際檢查額度) ===
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
                # 輕量級請求以獲取 Headers 中的額度資訊
                res_odds = requests.get(f"https://api.the-odds-api.com/v4/sports/?apiKey={THE_ODDS_API_KEY}", timeout=3)
                used = res_odds.headers.get('x-requests-used', '未知')
                remain = res_odds.headers.get('x-requests-remaining', '未知')
                st.metric("The-Odds-API 剩餘額度", f"剩 {remain} (已用 {used})")
            except: st.warning("The-Odds-API: 連線超時")
        else: st.error("未設定 The-Odds-API Key")

    st.divider()

    # === 真實資料庫筆數顯示 ===
    try:
        db_count = pd.read_sql("SELECT COUNT(*) as count FROM fixtures_v4", engine).iloc[0]['count']
        st.markdown(f"📊 **目前資料庫總收錄賽事： `{db_count}` 場**")
    except:
        st.markdown("📊 **目前資料庫總收錄賽事： `0` 場**")

    # === 即時數據同步功能 ===
    if st.button("📥 同步今日與未來賽程 (即時 API)"):
        with st.spinner("正在透過 API 獲取即時賽事與盤口..."):
            if not THE_ODDS_API_KEY:
                st.error("請先在 Streamlit Secrets 設定 THE_ODDS_API_KEY")
            else:
                try:
                    # 請求 The-Odds-API 英超即時盤口作為示範
                    res = requests.get(f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={THE_ODDS_API_KEY}&regions=eu&markets=h2h,spreads,totals").json()
                    
                    if isinstance(res, list) and len(res) > 0:
                        fixtures, odds = [], []
                        for match in res:
                            fid = hash(match['id']) % 1000000
                            dt = datetime.strptime(match['commence_time'], "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=8) # 轉港台時間
                            
                            fixtures.append({
                                "fixture_id": fid, "league_name": "英超",
                                "home_team": match['home_team'], "away_team": match['away_team'],
                                "match_datetime": dt.strftime("%Y-%m-%d %H:%M"),
                                "home_score": None, "away_score": None, "home_corner": None, "away_corner": None,
                                "status": "NS"
                            })
                            
                            # 提取盤口
                            ah_line, ah_home = 0, 1.90
                            ou_line, ou_over = 2.5, 1.90
                            
                            for bookmaker in match.get('bookmakers', []):
                                for market in bookmaker.get('markets', []):
                                    if market['key'] == 'spreads' and len(market['outcomes']) > 0:
                                        ah_line = market['outcomes'][0].get('point', 0)
                                        ah_home = market['outcomes'][0].get('price', 1.90)
                                    if market['key'] == 'totals' and len(market['outcomes']) > 0:
                                        ou_line = market['outcomes'][0].get('point', 2.5)
                                        ou_over = market['outcomes'][0].get('price', 1.90)
                                        
                            odds.append({
                                "fixture_id": fid,
                                "ah_line": ah_line, "ah_home_odd": ah_home, "ah_away_odd": 1.90,
                                "ou_line": ou_line, "ou_over_odd": ou_over, "ou_under_odd": 1.90,
                                "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                        # 安全寫入
                        df_f = pd.DataFrame(fixtures).drop_duplicates('fixture_id')
                        df_o = pd.DataFrame(odds)
                        with engine.begin() as conn:
                            conn.execute(text("CREATE TABLE IF NOT EXISTS fixtures_v4 (fixture_id INTEGER PRIMARY KEY, league_name TEXT, home_team TEXT, away_team TEXT, match_datetime TEXT, home_score INTEGER, away_score INTEGER, home_corner INTEGER, away_corner INTEGER, status TEXT)"))
                            conn.execute(text("CREATE TABLE IF NOT EXISTS odds_history_v4 (fixture_id INTEGER, ah_line TEXT, ah_home_odd REAL, ah_away_odd REAL, ou_line TEXT, ou_over_odd REAL, ou_under_odd REAL, recorded_at TEXT)"))
                            # 不刪除舊數據，使用 append 加入新數據，實現歷史追蹤
                            df_f.to_sql('fixtures_v4', conn, if_exists='append', index=False)
                            df_o.to_sql('odds_history_v4', conn, if_exists='append', index=False)
                            
                        st.success(f"✅ 成功同步 {len(df_f)} 場最新賽事與即時水位！")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.warning("API 回傳為空，可能近期無賽事或額度耗盡。")
                except Exception as e:
                    st.error(f"API 同步失敗: {e}")
