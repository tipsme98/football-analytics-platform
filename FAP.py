import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date
import sys
import os

# --- 路徑強制修正 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.database.connection import engine, SessionLocal
from src.database.models import League, Team, Fixture, Prediction
from src.api_clients.football_clients import FootballDataClient

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

# 2. 側邊欄導航與日曆工具
st.sidebar.title("導航選單")
page = st.sidebar.radio("選擇功能頁面", [
    "📊 賽事總覽與預測", 
    "🗃️ 歷史數據與日曆載入", 
    "📈 賠率與盤口追蹤", 
    "🤖 AI 賽前洞察報告"
])

st.sidebar.markdown("---")
st.sidebar.subheader("📅 賽事日期載入工具")
selected_date = st.sidebar.date_input("選擇要載入的比賽日期", value=date.today())

def fetch_and_store_matches_for_date(target_date: date):
    """透過 API 抓取指定日期的真實賽事，並提供詳細錯誤診斷"""
    date_str = target_date.strftime("%Y-%m-%d")
    client = FootballDataClient()
    
    # 檢查 API 金鑰是否存在
    if not client.api_key:
        st.error("⚠️ 系統未偵測到 FOOTBALL_DATA_API_KEY！請檢查 Streamlit Cloud 的 Secrets 設定。")
        return -1

    try:
        # 呼叫 API
        matches = client.get_matches(date_str, date_str)
        if not isinstance(matches, list):
            st.warning(f"API 回應異常或無權限存取該日期。回應內容: {matches}")
            return 0
            
        if len(matches) == 0:
            return 0
        
        db = SessionLocal()
        count = 0
        for m in matches:
            competition = m.get("competition", {})
            league_name = competition.get("name", "Unknown League")
            
            league = db.query(League).filter(League.standard_name == league_name).first()
            if not league:
                league = League(standard_name=league_name, country="International")
                db.add(league)
                db.commit()
                db.refresh(league)

            home_name = m.get("homeTeam", {}).get("name")
            away_name = m.get("awayTeam", {}).get("name")
            if not home_name or not away_name:
                continue

            home_team = db.query(Team).filter(Team.standard_name == home_name).first()
            if not home_team:
                home_team = Team(standard_name=home_name)
                db.add(home_team)
                db.commit()
                db.refresh(home_team)

            away_team = db.query(Team).filter(Team.standard_name == away_name).first()
            if not away_team:
                away_team = Team(standard_name=away_name)
                db.add(away_team)
                db.commit()
                db.refresh(away_team)

            match_utc_str = m.get("utcDate")
            match_datetime = datetime.fromisoformat(match_utc_str.replace("Z", "+00:00")).replace(tzinfo=None)

            status = m.get("status", "NS")
            score_data = m.get("score", {}).get("fullTime", {})
            home_score = score_data.get("home")
            away_score = score_data.get("away")

            existing = db.query(Fixture).filter(
                Fixture.league_id == league.id,
                Fixture.home_team_id == home_team.id,
                Fixture.away_team_id == away_team.id,
                Fixture.match_datetime == match_datetime
            ).first()

            if not existing:
                new_fix = Fixture(
                    league_id=league.id,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    match_datetime=match_datetime,
                    status=status,
                    home_score=home_score,
                    away_score=away_score
                )
                db.add(new_fix)
                count += 1
            else:
                existing.status = status
                existing.home_score = home_score
                existing.away_score = away_score
            db.commit()
        db.close()
        return count
    except Exception as e:
        st.error(f"API 連線發生例外錯誤: {e}")
        return -1

def inject_sample_historical_data():
    """一鍵注入多筆真實歷史賽事與比分，供機器學習與圖表測試"""
    db = SessionLocal()
    league = db.query(League).filter(League.standard_name == "Premier League").first()
    if not league:
        league = League(standard_name="Premier League", country="England")
        db.add(league)
        db.commit()
        db.refresh(league)

    teams_data = ["Manchester City", "Arsenal", "Liverpool", "Chelsea", "Manchester United", "Tottenham"]
    team_objs = {}
    for t_name in teams_data:
        t = db.query(Team).filter(Team.standard_name == t_name).first()
        if not t:
            t = Team(standard_name=t_name)
            db.add(t)
            db.commit()
            db.refresh(t)
        team_objs[t_name] = t

    samples = [
        ("Manchester City", "Arsenal", 2, 2, "FT"),
        ("Liverpool", "Chelsea", 2, 1, "FT"),
        ("Manchester United", "Tottenham", 0, 3, "FT"),
        ("Arsenal", "Chelsea", 1, 0, "FT"),
        ("Manchester City", "Liverpool", 1, 1, "FT"),
        ("Tottenham", "Chelsea", 0, 2, "FT")
    ]
    
    count = 0
    for h, a, hs, as_, st_ in samples:
        fix = Fixture(
            league_id=league.id,
            home_team_id=team_objs[h].id,
            away_team_id=team_objs[a].id,
            match_datetime=datetime.utcnow(),
            status=st_,
            home_score=hs,
            away_score=as_
        )
        db.add(fix)
        count += 1
    db.commit()
    db.close()
    return count

# 3. 資料庫查詢輔助函數
@st.cache_data(ttl=5)
def load_fixtures_data():
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
        return pd.read_sql(query, engine)
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
        st.warning("目前資料庫中尚無賽事資料。請至【🗃️ 歷史數據與日曆載入】頁面載入比賽。")
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
                        st.caption("尚未開賽")
                with cols[2]:
                    if pd.notna(row.get('prob_home_win')):
                        h_prob = row['prob_home_win'] * 100
                        st.progress(row['prob_home_win'], text=f"主勝機率: {h_prob:.1f}%")
                        if row.get('value_bet_detected'):
                            st.markdown(f'<span class="value-bet-tag">💎 價值投注: {row["recommended_pick"]}</span>', unsafe_allow_html=True)
                    else:
                        st.caption("模型尚未預測 (請至歷史數據頁面點擊訓練)")
                st.divider()

# --- 頁面二：歷史數據與日曆載入 ---
elif page == "🗃️ 歷史數據與日曆載入":
    st.subheader("🗃️ 歷史賽事數據與動態載入中心")
    st.write(f"目前選定的日曆日期: **{selected_date}**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(f"📥 從 API 載入 {selected_date} 賽事"):
            with st.spinner(f"正在向 API 查詢 {selected_date} 賽程..."):
                added_count = fetch_and_store_matches_for_date(selected_date)
                if added_count > 0:
                    st.success(f"成功同步！新增了 {added_count} 場賽事。")
                    st.rerun()
                elif added_count == 0:
                    st.info(f"API 回應正常，但該日期 ({selected_date}) 無排定賽事。")
                else:
                    st.error("API 同步失敗，請檢查金鑰設定。")
    with col2:
        if st.button("⚡ 快速注入歷史範例賽事 (推薦測試)"):
            c = inject_sample_historical_data()
            st.success(f"成功注入 {c} 場歷史經典賽事與比分！")
            st.rerun()
            
    with col3:
        if st.button("🤖 立即訓練機器學習模型"):
            if MODEL_AVAILABLE:
                with st.spinner("正在訓練 XGBoost 模型..."):
                    try:
                        predictor = FootballPredictor()
                        if predictor.train_model():
                            predictor.predict_upcoming_matches()
                            st.success("模型訓練與預測完成！")
                            st.rerun()
                        else:
                            st.warning("歷史完賽資料不足（請先點擊「快速注入歷史範例賽事」累積資料）。")
                    except Exception as e:
                        st.error(f"訓練發生錯誤: {e}")
            else:
                st.error("XGBoost 模組未就緒。")
    
    st.markdown("---")
    st.subheader("📋 資料庫中已儲存的賽事與比分紀錄")
    if not df_fixtures.empty:
        st.dataframe(df_fixtures, use_container_width=True)
    else:
        st.warning("資料庫目前為空。請點擊上方的「快速注入歷史範例賽事」或透過 API 載入。")

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
