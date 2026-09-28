import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import sys
import os

# 確保可以讀取 src 內的模組與資料庫
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.database.connection import engine

# 1. 頁面基本配置 (優化手機與網頁顯示)
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
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        text-align: center;
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

# 標題
st.markdown('<p class="main-header">⚽ 專業足球精算與價值投注平台</p>', unsafe_allow_html=True)

# 3. 側邊欄導航 (Mobile friendly)
st.sidebar.title("導航選單")
page = st.sidebar.radio("選擇功能頁面", ["📊 賽事總覽與預測", "📈 賠率與盤口追蹤", "🤖 AI 賽前洞察報告"])

# 4. 資料庫查詢輔助函數
@st.cache_data(ttl=60)
def load_fixtures_data():
    """從 SQLite 資料庫讀取賽程與球隊資訊"""
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
        ORDER BY f.match_datetime ASC
    """
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_predictions_data():
    """讀取機器學習預測與價值投注結果"""
    query = """
        SELECT 
            fixture_id,
            prob_home_win,
            prob_draw,
            prob_away_win,
            value_bet_detected,
            recommended_pick
        FROM predictions
    """
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        return pd.DataFrame()

# 載入資料
df_fixtures = load_fixtures_data()
df_preds = load_predictions_data()

if not df_fixtures.empty and not df_preds.empty:
    df_full = pd.merge(df_fixtures, df_preds, on="fixture_id", how="left")
else:
    df_full = df_fixtures

# --- 頁面一：賽事總覽與預測 ---
if page == "📊 賽事總覽與預測":
    st.subheader("🔥 今日焦點賽事與機器學習預測")
    
    if df_full.empty:
        st.info("目前資料庫中尚無賽事資料。請等待 GitHub Actions 自動排程爬蟲執行，或檢查 API 金鑰設定。")
    else:
        # 手機端自適應篩選器
        league_filter = st.selectbox("篩選聯賽", options=["全部聯賽"] + list(df_full['league_name'].unique()))
        if league_filter != "全部聯賽":
            df_display = df_full[df_full['league_name'] == league_filter]
        else:
            df_display = df_full

        for idx, row in df_display.iterrows():
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
                        st.markdown("### VS")
                        st.caption("尚未開賽 (NS)")
                
                with cols[2]:
                    if pd.notna(row.get('prob_home_win')):
                        h_prob = row['prob_home_win'] * 100
                        d_prob = row['prob_draw'] * 100
                        a_prob = row['prob_away_win'] * 100
                        st.progress(row['prob_home_win'], text=f"主勝機率: {h_prob:.1f}%")
                        
                        if row.get('value_bet_detected'):
                            st.markdown(f'<span class="value-bet-tag">💎 發現價值投注: {row["recommended_pick"]}</span>', unsafe_allow_html=True)
                    else:
                        st.caption("模型尚未生成預測")
                st.divider()

# --- 頁面二：賠率與盤口追蹤 ---
elif page == "📈 賠率與盤口追蹤":
    st.subheader("📊 賠率變動趨勢與盤口分析")
    st.info("此頁面將呈現賽前 48 小時內賠率的歷史波動曲線，幫助您捕捉莊家變盤的蛛絲馬跡。")
    
    # 讀取賠率歷史
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
            
            # 使用 Plotly 繪製賠率變動折線圖 (支援手機縮放)
            fig = px.line(
                match_odds_df, 
                x='recorded_at', 
                y=['home_odds', 'draw_odds', 'away_odds'],
                labels={'value': 'Decimal Odds', 'recorded_at': '時間', 'variable': '選項'},
                title=f"{selected_match} - 1X2 賠率歷史走勢"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("目前尚無賠率歷史數據紀錄。")
    except Exception as e:
        st.warning("尚無賠率資料表或資料為空。")

# --- 頁面三：AI 賽前洞察報告 ---
elif page == "🤖 AI 賽前洞察報告":
    st.subheader("🤖 AI 智慧賽前精算報告")
    st.write("結合量化數據與大型語言模型，自動生成的深度戰術分析與投資建議。")
    
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
