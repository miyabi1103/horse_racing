import pickle
from pathlib import Path
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import log_loss

# データディレクトリの設定
DATA_DIR = Path("..", "data")
INPUT_DIR = DATA_DIR / "02_features"
OUTPUT_DIR = DATA_DIR / "03_train"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

class Trainer:
    def __init__(
        self,
        input_dir: Path = INPUT_DIR,
        features_filepath: Path = "features.csv",
        config_filepath: Path = "config.yaml",
        output_dir: Path = OUTPUT_DIR,
    ):
        self.features = pd.read_csv(features_filepath, sep="\t")
        with open(config_filepath, "r") as f:
            self.feature_cols = yaml.safe_load(f)["features"]
        output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir = output_dir

    def create_dataset(self, test_start_date: str):
        # 目的変数
        self.features["target"] = (self.features["rank"] == 1).astype(int)
        
        # 学習データとテストデータに分割
        self.train_df = self.features.query("date < @test_start_date")
        self.test_df = self.features.query("date >= @test_start_date")
        
    def train(
        self,
        model_filename: str,
        importance_filename: str,
        evaluation_df_filename: str,
    ):
        # データセットの作成
        lgb_train = lgb.Dataset(self.train_df[self.feature_cols], self.train_df["target"])
        lgb_test = lgb.Dataset(self.test_df[self.feature_cols], self.test_df["target"])

        # パラメータの設定
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "random_state": 100,
            "verbosity": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "learning_rate": 0.01,
            "bagging_freq": 1
        }

        # 学習の実行
        model = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=900,
            valid_sets=[lgb_train, lgb_test],
            callbacks=[
                lgb.log_evaluation(100),
            ],
        )
        self.best_params = model.params
        
        # モデルを保存
        with open(self.output_dir / model_filename, "wb") as f:
            pickle.dump(model, f)
        
        # 特徴量重要度の可視化
        self.plot_feature_importance(model, importance_filename)

        # テストデータに対してスコアリング
        evaluation_df = self.evaluate_model(model)
        
        # 結果をCSVとして保存
        evaluation_df.to_csv(self.output_dir / f"{evaluation_df_filename}.csv", index=False, sep="\t")
        return evaluation_df

    def plot_feature_importance(self, model, importance_filename):
        lgb.plot_importance(
            model, importance_type="gain", figsize=(30, 15), max_num_features=50
        )
        plt.savefig(self.output_dir / f"{importance_filename}.png")
        plt.close()

    def evaluate_model(self, model):
        evaluation_df = self.test_df[
            [
                "race_id",
                "horse_id",
                "target",
                "rank",
                "tansho_odds",
                "popularity",
                "umaban",
            ]
        ].copy()
        evaluation_df["pred"] = model.predict(self.test_df[self.feature_cols], num_iteration=model.best_iteration)
        logloss = log_loss(evaluation_df["target"], evaluation_df["pred"])
        print("-" * 20 + "result" + "-" * 20)
        print(f"test_df's binary_logloss: {logloss}")

        # 予測した確率に基づいてベットする馬を選定
        bet_df = (
            evaluation_df
            .sort_values("pred", ascending=False)
            .groupby("race_id")
            .head(1)
        )
        
        # 的中率
        bet_accuracy = bet_df["target"].mean()
        print(f"Bet accuracy: {bet_accuracy:.2f}")

        # 帰ってくる金額
        returns = ((bet_df["target"] == 1) * bet_df["tansho_odds"]).sum()
        cost = len(bet_df)
        print(f"Total returns: {returns:.2f}, Total cost: {cost}, ROI: {returns / cost:.2f}")
        
        return evaluation_df

    def run(
        self,
        valid_start_date: str,
        test_start_date: str,
        importance_filename: str = "importance",
        model_filename: str = "model.pkl",
        evaluation_filename: str = "evaluation.csv"
    ):
        #学習処理を実行 test_start_dateの日付を指定すると、その日付以降をテストデータに、それより前のデータを学習データにする
        self.create_dataset(test_start_date)
        evaluation_df = self.train(
            model_filename, importance_filename, evaluation_filename
        )
        evaluation_df.to_csv(self.output_dif / evaluation_filename,sep="\t",index=False)
        return evaluation_df


# 実行部分
if __name__ == "__main__":
    trainer = Trainer()
    evaluation_df = trainer.run(
        valid_start_date='2022-12-01',
        test_start_date='2023-01-01'
    )
