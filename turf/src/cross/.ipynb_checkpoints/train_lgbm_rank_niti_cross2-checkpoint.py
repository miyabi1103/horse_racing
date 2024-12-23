import pickle
from pathlib import Path
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
import numpy as np

DATA_DIR = Path("..", "data")
INPUT_DIR = DATA_DIR / "02_features"
OUTPUT_DIR = DATA_DIR / "03_train"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

class Trainer_lightgbm_rank_niti_cross:
    def __init__(
        self,
        input_dir: Path = INPUT_DIR,
        features_filename: str = "features.csv",
        config_filepath: Path = "config_lightgbm_niti.yaml",
        output_dir: Path = OUTPUT_DIR,
    ):
        self.features = pd.read_csv(input_dir / features_filename, sep="\t")
        with open(config_filepath, "r") as f:
            config = yaml.safe_load(f)
            self.feature_cols = config["features"]
            self.params = config["params"]
        self.output_dir = output_dir

    def create_time_series_splits(self, n_splits: int = 5):
        """
        TimeSeriesSplitを作成し、学習データと検証データのインデックスを生成する。
        """
        # データを日付順に並べる
        self.features = self.features.sort_values("date")
        
        # インデックスを作成
        tscv = TimeSeriesSplit(n_splits=n_splits)
        self.splits = list(tscv.split(self.features))

    def train_with_cross_validation(self):
        """
        時系列クロスバリデーションを行い、モデルを訓練する。
        """
        all_metrics = []

        for fold, (train_idx, test_idx) in enumerate(self.splits):
            train_df = self.features.iloc[train_idx]
            test_df = self.features.iloc[test_idx]
            # 「馬の過去成績」がない馬を除外する
            test_df = test_df.query('rank_3races.notna()')
            
            lgb_train = lgb.Dataset(train_df[self.feature_cols], train_df["target"])
            lgb_valid = lgb.Dataset(test_df[self.feature_cols], test_df["target"], reference=lgb_train)
            
            # モデルの訓練
            model = lgb.train(
                params=self.params,
                train_set=lgb_train,
                num_boost_round=10000,
                valid_sets=[lgb_valid],
                callbacks=[
                    lgb.log_evaluation(100),
                    lgb.early_stopping(stopping_rounds=100),
                ],
            )
            
            # テストデータで予測
            pred = model.predict(test_df[self.feature_cols], num_iteration=model.best_iteration)
            # テストデータで予測
            test_df.loc[:, "pred"] = pred
            
            # 予測結果のバイナリ分類
            test_df.loc[:, "pred_binary"] = (test_df["pred"] >= 0.5).astype(int)

            
            # 評価指標の計算
            logloss = log_loss(test_df["target"], test_df["pred"])
            accuracy = accuracy_score(test_df["target"], test_df["pred_binary"])
            precision = precision_score(test_df["target"], test_df["pred_binary"])
            recall = recall_score(test_df["target"], test_df["pred_binary"])
            f1 = f1_score(test_df["target"], test_df["pred_binary"])
            roc_auc = roc_auc_score(test_df["target"], test_df["pred"])
            
            print(f"Fold {fold+1} - LogLoss: {logloss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            all_metrics.append({
                "fold": fold+1,
                "logloss": logloss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc
            })
        
        # すべてのfoldの評価結果を出力
        metrics_df = pd.DataFrame(all_metrics)
        print(metrics_df.mean(numeric_only=True))
        return metrics_df

    def run(
        self, 
        importance_filename: str = "importance_lightgbm_rank_niti",
        model_filename: str = "model_lightgbm_rank_niti_cross.pkl",
        evaluation_filename: str = "evaluation_lightgbm_rank_niti_cross.csv",
        n_splits=5):
        """
        時系列クロスバリデーションでモデルの評価を行う。
        """
        self.features["target"] = (self.features["rank"] == 1).astype(int)
        self.create_time_series_splits(n_splits=n_splits)
        metrics_df = self.train_with_cross_validation()
        metrics_df.to_csv(self.output_dir / evaluation_filename, index=False, sep="\t")
