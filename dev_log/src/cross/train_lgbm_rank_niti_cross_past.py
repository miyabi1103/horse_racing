import pickle
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from IPython.display import display

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
        output_dir: Path = OUTPUT_DIR
    ):
        self.features = pd.read_csv(input_dir / features_filename, sep="\t")
        with open(config_filepath, "r") as f:
            config = yaml.safe_load(f)
            self.feature_cols = config["features"]
            self.params = config["params"]
        self.output_dir = output_dir

    def create_dataset_for_cv(self, n_splits: int = 10):
        """
        時系列クロスバリデーションのため、データをn_splitsに分割
        学習データを徐々に増やし、評価データを次に進めていく
        """
        # 目的変数
        self.features["target"] = (self.features["rank"] == 1).astype(int)
        # 時系列でデータを並べ替え
        self.features = self.features.sort_values("date")
        
        # 時系列クロスバリデーションのインデックス
        n_samples = len(self.features)
        fold_size = n_samples // n_splits
        
        for i in range(n_splits):
            # 学習データの範囲
            train_df = self.features.iloc[:(i + 1) * fold_size]
            # 評価データの範囲
            valid_df = self.features.iloc[(i + 1) * fold_size: (i + 2) * fold_size]
            # 「馬の過去成績」の欠損率
            valid_df = valid_df.query('rank_3races.notna()')
            
            # 学習データと評価データを返す
            yield train_df, valid_df

    # def train(
    #     self, 
    #     train_df: pd.DataFrame, 
    #     valid_df: pd.DataFrame, 
    #     importance_filename: str, 
    #     model_filename: str
    # ) -> pd.DataFrame:
    #     # データセットの作成
    #     lgb_train = lgb.Dataset(train_df[self.feature_cols], train_df["target"])
    #     lgb_valid = lgb.Dataset(
    #         valid_df[self.feature_cols], 
    #         valid_df["target"], 
    #         reference=lgb_train
    #     )
        
    #     # 学習の実行
    #     model = lgb.train(
    #         params=self.params,
    #         train_set=lgb_train,
    #         num_boost_round=10000,
    #         valid_sets=[lgb_valid],
    #         callbacks=[
    #             lgb.log_evaluation(100),
    #             lgb.early_stopping(stopping_rounds=100),
    #         ],
    #     )
    #     self.best_params = model.params
    #     with open(self.output_dir / model_filename, "wb") as f:
    #         pickle.dump(model, f)
        
    #     # 特徴量重要度の可視化
    #     lgb.plot_importance(model, importance_type="gain", figsize=(30, 15), max_num_features=50)
    #     plt.savefig(self.output_dir / f"{importance_filename}.png")
    #     plt.close()

    #     importance_df = pd.DataFrame({
    #         "feature": model.feature_name(),
    #         "importance": model.feature_importance(importance_type="gain"),
    #     }).sort_values("importance", ascending=False)
    #     importance_df.to_csv(self.output_dir / f"{importance_filename}.csv", index=False, sep="\t")
        
    #     # 評価
    #     evaluation_df = valid_df[["race_id", "horse_id", "target", "rank", "tansho_odds", "popularity", "umaban"]].copy()
    #     evaluation_df["pred"] = model.predict(valid_df[self.feature_cols], num_iteration=model.best_iteration)
        
    #     # 0.5 を閾値にしてクラス分類
    #     evaluation_df["pred_binary"] = (evaluation_df["pred"] >= 0.07267).astype(int)
        
    #     # 評価指標の計算
    #     accuracy = accuracy_score(evaluation_df["target"], evaluation_df["pred_binary"])
    #     precision = precision_score(evaluation_df["target"], evaluation_df["pred_binary"])
    #     recall = recall_score(evaluation_df["target"], evaluation_df["pred_binary"])
    #     f1 = f1_score(evaluation_df["target"], evaluation_df["pred_binary"])
    #     roc_auc = roc_auc_score(evaluation_df["target"], evaluation_df["pred"])

    #     # 結果を出力
    #     print("-" * 20 + " Metrics " + "-" * 20)
    #     print(f"Accuracy: {accuracy:.4f}")
    #     print(f"Precision: {precision:.4f}")
    #     print(f"Recall: {recall:.4f}")
    #     print(f"F1 Score: {f1:.4f}")
    #     print(f"ROC AUC: {roc_auc:.4f}")
        
    #     return evaluation_df
    def train(
        self, 
        train_df: pd.DataFrame, 
        valid_df: pd.DataFrame, 
        importance_filename: str, 
        model_filename: str,
        boost_rounds: int = None  # 最適なブーストラウンド数を指定できるように変更
    ) -> pd.DataFrame:
        # データセットの作成
        lgb_train = lgb.Dataset(train_df[self.feature_cols], train_df["target"])
        lgb_valid = lgb.Dataset(
            valid_df[self.feature_cols], 
            valid_df["target"], 
            reference=lgb_train
        )
        
        # 学習の実行
        model = lgb.train(
            params=self.params,
            train_set=lgb_train,
            num_boost_round=boost_rounds if boost_rounds else 10000,  # boost_roundsが指定されたらそれを使う
            valid_sets=[lgb_valid],
            callbacks=[
                lgb.log_evaluation(100),
                lgb.early_stopping(stopping_rounds=100),
            ] if boost_rounds is None else []  # boost_roundsが指定されていない場合のみアーリーストッピング
        )
        
        self.best_params = model.params
        with open(self.output_dir / model_filename, "wb") as f:
            pickle.dump(model, f)
        
        # 特徴量重要度の可視化
        lgb.plot_importance(model, importance_type="gain", figsize=(30, 15), max_num_features=50)
        plt.savefig(self.output_dir / f"{importance_filename}.png")
        plt.close()
    
        importance_df = pd.DataFrame({
            "feature": model.feature_name(),
            "importance": model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(self.output_dir / f"{importance_filename}.csv", index=False, sep="\t")
        
        # 評価
        evaluation_df = valid_df[["race_id", "horse_id", "target", "rank", "tansho_odds", "popularity", "umaban"]].copy()
        evaluation_df["pred"] = model.predict(valid_df[self.feature_cols], num_iteration=model.best_iteration)
        
        # 0.5 を閾値にしてクラス分類
        evaluation_df["pred_binary"] = (evaluation_df["pred"] >= 0.07267).astype(int)
        
        # 評価指標の計算
        accuracy = accuracy_score(evaluation_df["target"], evaluation_df["pred_binary"])
        precision = precision_score(evaluation_df["target"], evaluation_df["pred_binary"])
        recall = recall_score(evaluation_df["target"], evaluation_df["pred_binary"])
        f1 = f1_score(evaluation_df["target"], evaluation_df["pred_binary"])
        roc_auc = roc_auc_score(evaluation_df["target"], evaluation_df["pred"])
    
        # 結果を出力
        print("-" * 20 + " Metrics " + "-" * 20)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # ここで num_boost_round を返す
        return evaluation_df, model.best_iteration  # 修正箇所


    
    def run(
        self, 
        n_splits: int = 10,  
        importance_filename: str = "importance_lightgbm_rank_niti_cross", 
        model_filename: str = "model_lightgbm_rank_niti_cross.pkl", 
        evaluation_filename: str = "evaluation_lightgbm_rank_niti_cross.csv",
        final_model_filename: str = "model_lightgbm_rank_niti_cross_full.pkl"
    ):
        """
        時系列クロスバリデーションを実行する
        最後に全データを使って学習したモデルを保存
        """
        evaluation_dfs = []
        num_boost_rounds = []  # 各foldでのnum_boost_roundを保持するリスト
    
        # クロスバリデーションをn_splits-1回だけ実行する
        for fold_idx, (train_df, valid_df) in enumerate(self.create_dataset_for_cv(n_splits)):
            if fold_idx == n_splits - 1:
                break  # 最後のfoldは実行しない
    
            print(f"Training fold {fold_idx + 1}/{n_splits}...")
            evaluation_df, num_boost_round = self.train(
                train_df, valid_df, 
                importance_filename=f"{importance_filename}_fold{fold_idx + 1}", 
                model_filename=f"{model_filename}_fold{fold_idx + 1}"
            )

            evaluation_dfs.append(evaluation_df)
            num_boost_rounds.append(num_boost_round)  # ブーストラウンド数を記録
    
        # クロスバリデーション後の評価データをまとめて保存
        full_evaluation_df = pd.concat(evaluation_dfs, axis=0)
        full_evaluation_df.to_csv(self.output_dir / evaluation_filename, sep="\t", index=False)
        
        # 最終モデルを学習（アーリーストッピングなし）
        print("Training final model using all data...")
    
        # 最後に使ったブーストラウンド数を取得
        last_num_boost_round = num_boost_rounds[-1] if num_boost_rounds else 1000  # デフォルト値は1000
        additional_boost_round = 300  # 最後に追加で学習するブーストラウンド回数
        total_boost_round = last_num_boost_round + additional_boost_round
    
        final_train_df = self.features
        lgb_train = lgb.Dataset(final_train_df[self.feature_cols], final_train_df["target"])
    
        # 最終モデルの学習
        final_model = lgb.train(
            params=self.params,
            train_set=lgb_train,
            num_boost_round=total_boost_round  # 最後のブーストラウンド数 + 追加分
        )
        
        # 最終モデルを保存
        with open(self.output_dir / final_model_filename, "wb") as f:
            pickle.dump(final_model, f)
    
        print(f"Final model saved to {final_model_filename}")
        return full_evaluation_df
    

