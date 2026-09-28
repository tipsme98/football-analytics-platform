import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.database.connection import engine, Base
from src.database.models import League, Team, Fixture, OddsHistory, MatchStats, Prediction, AiReport, ApiMapping

def init_database():
    print("正在初始化足球精算平台資料庫結構...")
    Base.metadata.create_all(bind=engine)
    print("資料庫初始化完成！")

if __name__ == "__main__":
    init_database()
