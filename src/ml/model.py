import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

# 確保可以匯入 src 下的模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.database.connection import engine
from src.database.models import Prediction, SessionLocal

class FootballPredictor:
    def __init__(self):
        self.db = SessionLocal()
        # 建立資料庫連線引擎，供 pandas 讀取
        self.engine = engine
        self.model = None

    def fetch_training_data(self):
        """從資料庫中提取歷史賽果與賠率資料，作為訓練特徵"""
        # 這裡的 SQL 查詢是一個簡化示範，實務上會關聯更多特徵 (如控球率、傷病等)
        query = """
            SELECT 
                f.id as fixture_id,
                f.home_team_id,
                f.away_team_id,
                f.home_score,
                f.away_score,
                o.home_odds,
                o.draw_odds,
                o.away_odds
            FROM fixtures f
            LEFT JOIN odds_history o ON f.id = o.fixture_id
            WHERE f.status = 'FT' AND f.home_score IS NOT NULL
        """
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            print(f"讀取訓練資料失敗: {e}")
            return pd.DataFrame()

    def feature_engineering(self, df: pd.DataFrame):
        """特徵工程：將原始資料轉換為模型可學習的特徵"""
        if df.empty:
            return df

        # 定義目標變數 (Target): 1=主勝, 0=平局, 2=客勝
        conditions = [
            (df['home_score'] > df['away_score']),
            (df['home_score'] == df['away_score']),
            (df['home_score'] < df['away_score'])
        ]
        choices = [1, 0, 2]
        df['target'] = np.select(conditions, choices, default=np.nan)
        
        # 簡單特徵範例：利用賠率倒數作為莊家預估機率的特徵 (Implied Probability)
        # 實務上應加入如 "主隊近五場平均進球" 等更複雜的特徵
        df['prob_home_implied'] = 1 / df['home_odds']
        df['prob_draw_implied'] = 1 / df['draw_odds']
        df['prob_away_implied'] = 1 / df['away_odds']

        # 清除包含空值的資料
        df = df.dropna(subset=['target', 'prob_home_implied', 'prob_draw_implied', 'prob_away_implied'])
        return df

    def train_model(self):
        """訓練 XGBoost 模型"""
        print("開始訓練足球預測模型...")
        df = self.fetch_training_data()
        df = self.feature_engineering(df)

        if df.empty or len(df) < 50:
            print("訓練資料不足 (需至少 50 筆歷史賽事)，略過訓練。")
            return False

        # 定義特徵 (X) 與目標 (y)
        features = ['prob_home_implied', 'prob_draw_implied', 'prob_away_implied']
        X = df[features]
        y = df['target']

        # 初始化並訓練 XGBoost 分類器
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42
        )
        self.model.fit(X, y)
        print("模型訓練完成！")
        return True

    def predict_upcoming_matches(self):
        """預測未開打的賽事，尋找價值投注"""
        if not self.model:
            print("模型未訓練，無法進行預測。")
            return

        # 抓取尚未開打的賽程 (狀態為 NS)
        query = """
            SELECT 
                f.id as fixture_id,
                o.home_odds,
                o.draw_odds,
                o.away_odds
            FROM fixtures f
            JOIN odds_history o ON f.id = o.fixture_id
            WHERE f.status = 'NS'
        """
        upcoming_df = pd.read_sql(query, self.engine)
        if upcoming_df.empty:
            print("沒有找到即將開打且有賠率的賽事。")
            return

        # 計算特徵
        upcoming_df['prob_home_implied'] = 1 / upcoming_df['home_odds']
        upcoming_df['prob_draw_implied'] = 1 / upcoming_df['draw_odds']
        upcoming_df['prob_away_implied'] = 1 / upcoming_df['away_odds']
        
        # 進行預測，取得各結果的機率
        features = ['prob_home_implied', 'prob_draw_implied', 'prob_away_implied']
        X_predict = upcoming_df[features].fillna(0.33) # 若無賠率，假設機率均等
        
        # XGBoost output array of probabilities [draw_prob, home_win_prob, away_win_prob]
        probabilities = self.model.predict_proba(X_predict)

        print("開始進行預測並記錄至資料庫...")
        for i, row in upcoming_df.iterrows():
            fixture_id = int(row['fixture_id'])
            
            # XGBoost 的類別順序與我們訓練時設定的一致 (0:平, 1:主, 2:客)
            prob_draw = probabilities[i][0]
            prob_home = probabilities[i][1]
            prob_away = probabilities[i][2]

            # 價值投注 (Value Bet) 判定邏輯：
            # 若模型預測機率 > 莊家隱含機率 (加上一點安全邊際，例如 5%)，則視為價值投注
            value_detected = False
            pick = None
            if prob_home > (row['prob_home_implied'] + 0.05):
                value_detected = True
                pick = "Home Win"
            elif prob_away > (row['prob_away_implied'] + 0.05):
                value_detected = True
                pick = "Away Win"

            # 寫入預測結果至資料庫
            # 先檢查是否已有舊的預測紀錄
            existing_pred = self.db.query(Prediction).filter(Prediction.fixture_id == fixture_id).first()
            if existing_pred:
                existing_pred.prob_home_win = prob_home
                existing_pred.prob_draw = prob_draw
                existing_pred.prob_away_win = prob_away
                existing_pred.value_bet_detected = value_detected
                existing_pred.recommended_pick = pick
                existing_pred.model_version = "xgb_v1.0"
            else:
                new_pred = Prediction(
                    fixture_id=fixture_id,
                    model_version="xgb_v1.0",
                    prob_home_win=prob_home,
                    prob_draw=prob_draw,
                    prob_away_win=prob_away,
                    value_bet_detected=value_detected,
                    recommended_pick=pick
                )
                self.db.add(new_pred)
            
        self.db.commit()
        print("預測完成，並已存入資料庫！")

if __name__ == "__main__":
    predictor = FootballPredictor()
    # 訓練模型並執行預測
    if predictor.train_model():
        predictor.predict_upcoming_matches()
