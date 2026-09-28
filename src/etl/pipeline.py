import sys
import os
from datetime import datetime, timedelta

# 確保可以匯入 src 下的模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.database.connection import SessionLocal
from src.database.models import League, Team, Fixture, OddsHistory, ApiMapping
from src.api_clients.football_clients import FootballDataClient, ApiFootballClient, TheOddsApiClient

def normalize_name(name: str) -> str:
    """名稱標準化：去除大小寫、FC、常見縮寫與空格"""
    if not name:
        return ""
    return name.lower().replace("fc", "").replace("united", "utd").replace(" ", "").strip()

class FootballETLPipeline:
    def __init__(self):
        self.db = SessionLocal()
        self.fd_client = FootballDataClient()
        self.api_football_client = ApiFootballClient()
        self.odds_client = TheOddsApiClient()

    def get_or_create_team(self, external_name: str, provider: str) -> Team:
        """根據外部名稱與來源，透過 ApiMapping 尋找或建立內部標準 Team 實體"""
        norm_name = normalize_name(external_name)
        
        mapping = self.db.query(ApiMapping).filter(
            ApiMapping.entity_type == 'team',
            ApiMapping.provider == provider,
            ApiMapping.external_id == norm_name
        ).first()

        if mapping:
            team = self.db.query(Team).filter(Team.id == int(mapping.standard_id)).first()
            if team:
                return team

        all_teams = self.db.query(Team).all()
        matched_team = None
        for t in all_teams:
            if normalize_name(t.standard_name) == norm_name:
                matched_team = t
                break

        if not matched_team:
            matched_team = Team(standard_name=external_name)
            self.db.add(matched_team)
            self.db.commit()
            self.db.refresh(matched_team)

        new_mapping = ApiMapping(
            entity_type='team',
            standard_id=str(matched_team.id),
            provider=provider,
            external_id=norm_name
        )
        self.db.add(new_mapping)
        self.db.commit()

        return matched_team

    def sync_historical_and_upcoming_matches(self, days_back: int = 30, days_forward: int = 7):
        """
        擴大同步範圍：包含過去 N 天的歷史賽果（含比分）與未來 N 天的賽程，
        供機器學習模型進行特徵工程與盤口關聯分析。
        """
        today = datetime.utcnow().date()
        date_from = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        date_to = (today + timedelta(days=days_forward)).strftime("%Y-%m-%d")
        
        print(f"開始同步賽程資料，區間: {date_from} 至 {date_to} (含歷史完賽與未來賽事)...")
        
        matches = self.fd_client.get_matches(date_from, date_to)
        
        for m in matches:
            competition = m.get("competition", {})
            league_name = competition.get("name", "Unknown League")
            
            # 處理聯賽
            league = self.db.query(League).filter(League.standard_name == league_name).first()
            if not league:
                league = League(standard_name=league_name, country="International/Europe")
                self.db.add(league)
                self.db.commit()
                self.db.refresh(league)

            home_team_name = m.get("homeTeam", {}).get("name")
            away_team_name = m.get("awayTeam", {}).get("name")
            
            if not home_team_name or not away_team_name:
                continue

            home_team = self.get_or_create_team(home_team_name, "football_data")
            away_team = self.get_or_create_team(away_team_name, "football_data")

            match_utc_str = m.get("utcDate")
            match_datetime = datetime.fromisoformat(match_utc_str.replace("Z", "+00:00")).replace(tzinfo=None)

            # 提取比分資料 (若已完賽 FT)
            status = m.get("status", "NS")
            score_data = m.get("score", {}).get("fullTime", {})
            home_score = score_data.get("home")
            away_score = score_data.get("away")

            # 檢查賽程是否已存在
            existing_fixture = self.db.query(Fixture).filter(
                Fixture.league_id == league.id,
                Fixture.home_team_id == home_team.id,
                Fixture.away_team_id == away_team.id,
                Fixture.match_datetime == match_datetime
            ).first()

            if not existing_fixture:
                new_fixture = Fixture(
                    league_id=league.id,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    match_datetime=match_datetime,
                    status=status,
                    home_score=home_score,
                    away_score=away_score
                )
                self.db.add(new_fixture)
                self.db.commit()
                print(f"新增賽事: [{league_name}] {home_team_name} vs {away_team_name} (狀態: {status}, 比分: {home_score}:{away_score})")
            else:
                # 更新現有賽事的狀態與比分（例如完賽後的比分更新）
                existing_fixture.status = status
                existing_fixture.home_score = home_score
                existing_fixture.away_score = away_score
                self.db.commit()

        print("賽程與歷史數據同步完成！")

    def run_pipeline(self):
        print("--- 啟動擴充版 ETL 管線 ---")
        self.sync_historical_and_upcoming_matches(days_back=30, days_forward=7)
        self.db.close()
        print("--- ETL 管線執行結束 ---")

if __name__ == "__main__":
    pipeline = FootballETLPipeline()
    pipeline.run_pipeline()
