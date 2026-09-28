import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class FootballDataClient:
    """Football-Data.org API 客戶端 (輕量級備援與賽程來源)"""
    def __init__(self):
        self.api_key = os.getenv("FOOTBALL_DATA_API_KEY")
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {"X-Auth-Token": self.api_key} if self.api_key else {}

    def get_matches(self, date_from: str, date_to: str):
        """獲取指定日期區間的賽程"""
        url = f"{self.base_url}/matches?dateFrom={date_from}&dateTo={date_to}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json().get("matches", [])
            else:
                print(f"[Football-Data] 請求失敗: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"[Football-Data] 連線異常: {e}")
            return []


class ApiFootballClient:
    """API-Football (RapidAPI) 客戶端 (核心賽況與詳細數據來源)"""
    def __init__(self):
        self.api_key = os.getenv("API_FOOTBALL_KEY")
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "v3.football.api-sports.io"
        } if self.api_key else {}

    def get_fixtures_by_date(self, date_str: str):
        """獲取指定日期的即時賽況與賽程"""
        url = f"{self.base_url}/fixtures?date={date_str}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json().get("response", [])
            else:
                print(f"[API-Football] 請求失敗: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"[API-Football] 連線異常: {e}")
            return []


class TheOddsApiClient:
    """The Odds API 客戶端 (賽前盤口與賠率變化來源)"""
    def __init__(self):
        self.api_key = os.getenv("THE_ODDS_API_KEY")
        self.base_url = "https://api.the-odds-api.com/v4"

    def get_odds(self, sport_key: str = "soccer_epl", regions: str = "eu", markets: str = "h2h,spreads,totals"):
        """獲取指定聯賽的最新賠率與盤口"""
        url = f"{self.base_url}/sports/{sport_key}/odds/"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal"
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[The Odds API] 請求失敗: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"[The Odds API] 連線異常: {e}")
            return []
