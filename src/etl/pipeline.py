import sys
import os
from datetime import datetime

# 確保可以匯入 src 下的模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.database.connection import SessionLocal
from src.database.models import League, Team, Fixture, OddsHistory, ApiMapping
from src.api_clients.football_clients import FootballDataClient, ApiFootballClient, TheOddsApiClient

def normalize_name(name: str) -> str:
    """
    名稱標準化：去除大小寫、FC、常見縮寫與空格，便於跨 API 進行初步比對
    """
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
        
        # 1. 檢查 Mapping 表是否已經存在該來源的對應關係
        mapping = self.db.query(ApiMapping).filter(
            ApiMapping.entity_type == 'team',
            ApiMapping.provider == provider,
            ApiMapping.external_id == norm_name
        ).first()

        if mapping:
            # 找到 Mapping，直接回傳對應的標準 Team
            team = self.db.query(Team).filter(Team.id == int(mapping.standard_id)).first()
            if team:
                return team

        # 2. 如果沒有 Mapping，則尋找是否已有相似標準名稱的球隊 (簡化版邏輯)
        all_teams = self.db.query(Team).all()
        matched_team = None
        for t in all_teams:
            if normalize_name(t.standard_name) == norm_name:
                matched_team = t
                break

        # 3. 若完全不存在符合的球隊，建立新球隊
        if not matched_team:
            matched_team = Team(standard_name=external_name)
            self.db.add(matched_team)
            self.db.commit()
            self.db.refresh(matched_team)

        # 4. 將新的對應關係寫入 Mapping 記錄
        new_mapping = ApiMapping(
            entity_type='team',
            standard_id=str(matched_team.id),
            provider=provider,
            external_id=norm_name
        )
        self.db.add(new_mapping)
        self.db.commit()

        return matched_team

    def sync_football_data_matches(self):
        """同步 Football-Data.org 的近期賽程 (作為範例)"""
        print("開始同步 Football-Data.org 賽程資料...")
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        
        matches = self.fd_client.get_matches(today_str, today_str)
        
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

            # 透過 Mapping 取得或建立球隊
            home_team = self.get_or_create_team(home_team_name, "football_data")
            away_team = self.get_or_create_team(away_team_name, "football_data")

            # 處理時間格式
            match_utc_str = m.get("utcDate")
            match_datetime = datetime.fromisoformat(match_utc_str.replace("Z", "+00:00")).replace(tzinfo=None)

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
                    status=m.get("status", "NS")
                )
                self.db.add(new_fixture)
                self.db.commit()
                print(f"新增賽事: [{league_name}] {home_team_name} vs {away_team_name}")
            else:
                # 這裡可以加入更新狀態的邏輯 (例如從 NS 變為 FT)
                pass

        print("Football-Data.org 賽程同步完成！")

    def run_pipeline(self):
        """執行完整的資料管線"""
        print("--- 啟動 ETL 管線 ---")
        self.sync_football_data_matches()
        # 未來可在此擴充 sync_api_football_stats() 和 sync_odds()
        self.db.close()
        print("--- ETL 管線執行結束 ---")

if __name__ == "__main__":
    pipeline = FootballETLPipeline()
    pipeline.run_pipeline()
