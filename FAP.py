import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import sys
import os

# --- 路徑強制修正：確保雲端環境絕對能抓到 src 模組 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.database.connection import engine, SessionLocal
from src.database.models import League, Team, Fixture, Prediction

# 嘗試載入 ML 模型
try:
    from src.ml.model import FootballPredictor
    MODEL_AVAILABLE = True
except Exception as e:
    MODEL_AVAILABLE = False

# 1. 頁面基本配置
st.set_page_config(
    page_title="專業足球精算平台",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 自定義行動端響應式 CSS 樣式
st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .value-bet-tag {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚽ 專業足球精算與價值投注平台</p>', unsafe_allow_html=True)

# 3. 側邊欄導航
st.sidebar.title("導航選單")
page = st.sidebar.radio("選擇功能頁面", [
    "📊 賽事總覽與預測", 
    "🗃️ 歷史數據與資料庫總覽", 
    "📈 賠率與盤口追蹤", 
    "🤖 AI 賽前洞察報告"
])

# 4. 資料庫查詢輔助函數 (含防呆自動建立範例資料)
@st.cache_data(ttl=5)
def load_fixtures_data():
    """從 SQLite 資料庫讀取賽程，若為空則自動寫入測試範例供展示"""
    try:
        query = """
            SELECT 
                f.id as fixture_id,
                l.standard_name as league_name,
                t1.standard_name as home_team,
                t2.standard_name as away_team,
                f.match_datetime,
                f.status,
                f.home_score,
                f.away_score
            FROM fixtures f
            JOIN leagues l ON f.league_id = l.id
            JOIN teams t1 ON f.home_team_id = t1.id
            JOIN teams t2 ON f.away_team_id = t2.id
            ORDER BY f.match_datetime DESC
        """
        df = pd.read_sql(query, engine)
        
        # 若資料庫為空，自動寫入幾筆範例資料，確保介面與模型可以順利運作！
        if df.empty:
            db = SessionLocal()
            league = db.query(League).filter(League.standard_name == "Premier League").first()
            if not league:
                league = League(standard_name="Premier League", country="England")
                db.add(league)
                db.commit()
                db.refresh(league)
            
            t1 = Team(standard_name="Arsenal")
            t2 = Team(standard_name="Chelsea")
            db.add_all([t1, t2])
            db.commit()
            db.refresh(t1)
            db.refresh(t2)

            sample_fixture = Fixture(
                league_id=league.id,
                home_team_id=t1.id,
                away_team_id=t2.id,
                match_datetime=datetime.utcnow(),
                status="FT",
                home_score=2,
                away_score=1
            )
            db.add(sample_fixture)
            db.commit()
            db.close()
            # 重新讀取
            df = pd.read_sql(query, engine)
            
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=5)
def load_predictions_data():
    query = """
        SELECT fixture_id, prob_home_win, prob_draw, prob_away_win, value_bet_detected, recommended_pick
        FROM predictions
    """
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        return pd.DataFrame()

df_fixtures = load_fixtures_data()
df_preds = load_predictions_data()

if not df_fixtures.empty and not df_preds.empty:
    df_full = pd.merge(df_fixtures, df_preds, on="fixture_id", how="left")
else:
    df_full = df_fixtures

# --- 頁面一：賽事總覽與預測 ---
if page == "📊 賽事總覽與預測":
    st.subheader("🔥 賽事總覽與機器學習預測")
    
    if df_full.empty:
        st.warning("目前資料庫中尚無賽事資料。")
    else:
        league_filter = st.selectbox("篩選聯賽", options=["全部聯賽"] + list(df_full['league_name'].unique()))
        df_display = df_full if league_filter == "全部聯賽" else df_full[df_full['league_name'] == league_filter]

        for idx, row in df_display.head(50).iterrows():
            with st.container():
                cols = st.columns([3, 2, 3])
                with cols[0]:
                    st.markdown(f"**{row['home_team']}** vs **{row['away_team']}**")
                    st.caption(f"📅 {row['match_datetime']} | 🏆 {row['league_name']}")
                with cols[1]:
                    status = row['status']
                    if status == 'FT':
                        st.markdown(f"### 🎯 {row['home_score']} : {row['away_score']}")
                        st.caption("已完賽 (FT)")
                    else:
                        st.markdown(f"### ⏰ {status}")
                        st.caption("比賽狀態")
                with cols[2]:
                    if pd.notna(row.get('prob_home_win')):
                        h_prob = row['prob_home_win'] * 100
                        st.progress(row['prob_home_win'], text=f"主勝機率: {h_prob:.1f}%")
                        if row.get('value_bet_detected'):
                            st.markdown(f'<span class="value-bet-tag">💎 價值投注: {row["recommended_pick"]}</span>', unsafe_allow_html=True)
                    else:
                        st.caption("尚未生成預測 (請至歷史數據頁面點擊訓練)")
                st.divider()

# --- 頁面二：歷史數據與資料庫總覽 ---
elif page == "🗃️ 歷史數據與資料庫總覽":
    st.subheader("🗃️ 資料庫內容與歷史賽果總覽")
    st.write("在這裡你可以檢視資料庫中已同步的歷史賽事、比分，並手動執行模型訓練。")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🤖 立即訓練機器學習模型並生成預測"):
            if MODEL_AVAILABLE:
                with st.spinner("正在訓練 XGBoost 模型並進行預測..."):
                    try:
                        predictor = FootballPredictor()
                        if predictor.train_model():
                            predictor.predict_upcoming_matches()
                            st.success("模型訓練與預測完成！請重新整理頁面查看結果。")
                            st.rerun()
                        else:
                            st.warning("歷史完賽資料不足（需更多比分紀錄），已使用內建樣本進行處理。")
                    except Exception as e:
                        st.error(f"執行發生錯誤: {e}")
            else:
                st.error("系統偵測到未安裝 xgboost 套件，請確認 requirements.txt 內容。")
    
    st.markdown("---")
    st.subheader("📋 資料庫中的原始賽事與比分紀錄")
    if not df_fixtures.empty:
        st.dataframe(df_fixtures, use_container_width=True)
    else:
        st.warning("資料庫目前為空。")

# --- 頁面三：賠率與盤口追蹤 ---
elif page == "📈 賠率與盤口追蹤":
    st.subheader("📊 賠率變動趨勢與盤口分析")
    odds_query = """
        SELECT o.*, t1.standard_name as home_team, t2.standard_name as away_team
        FROM odds_history o
        JOIN fixtures f ON o.fixture_id = f.id
        JOIN teams t1 ON f.home_team_id = t1.id
        JOIN teams t2 ON f.away_team_id = t2.id
    """
    try:
        df_odds = pd.read_sql(odds_query, engine)
        if not df_odds.empty:
            match_names = df_odds['home_team'] + " vs " + df_odds['away_team']
            selected_match = st.selectbox("選擇比賽查看賠率走勢", options=match_names.unique())
            match_odds_df = df_odds[match_names == selected_match]
            fig = px.line(
                match_odds_df, x='recorded_at', y=['home_odds', 'draw_odds', 'away_odds'],
                labels={'value': 'Decimal Odds', 'recorded_at': '時間', 'variable': '選項'},
                title=f"{selected_match} - 1X2 賠率歷史走勢"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("目前尚無賠率歷史數據紀錄。")
    except Exception as e:
        st.info("尚無賠率資料表。")

# --- 頁面四：AI 賽前洞察報告 ---
elif page == "🤖 AI 賽前洞察報告":
    st.subheader("🤖 AI 智慧賽前精算報告")
    ai_query = """
        SELECT ar.*, t1.standard_name as home_team, t2.standard_name as away_team
        FROM ai_reports ar
        JOIN fixtures f ON ar.fixture_id = f.id
        JOIN teams t1 ON f.home_team_id = t1.id
        JOIN teams t2 ON f.away_team_id = t2.id
    """
    try:
        df_ai = pd.read_sql(ai_query, engine)
        if not df_ai.empty:
            report_titles = df_ai['report_title'] + " (" + df_ai['home_team'] + " vs " + df_ai['away_team'] + ")"
            selected_report = st.selectbox("選擇賽事報告", options=report_titles)
            report_row = df_ai[report_titles == selected_report].iloc[0]
            st.markdown(f"### {report_row['report_title']}")
            st.caption(f"生成時間: {report_row['created_at']}")
            st.markdown("---")
            st.markdown(report_row['report_content'])
        else:
            st.info("目前尚無 AI 生成的分析報告。")
    except Exception as e:
        st.info("AI 報告功能準備中...")
