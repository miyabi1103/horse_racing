import pickle
from pathlib import Path


#import optuna.integration.lightgbm as lgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd

import yaml

from sklearn.metrics import log_loss

DATA?DIR = Path("..","data")
INPUT_DIR =DATA_DIR / "02_features"
OUTPUT_DIR = DATA_DIR / "03_train"
OUTPUT_DIR.mkdir(exist_ok = True,parents = True)

class Trainer:
    def __init__(
        self,
        input_dir : Path = INPUT_DIR,
        features_filepath:Path = "features.csv",
        config_filepath : Path = "config.yaml",
        output_dir : Path = OUTPUT_DIR,
    ):
        self.features = pd.read_csv(features_filepath,sep="\t")
        with open(config_filepath, "r") as f:
            self.feature_cols = yaml.safe_load(file)["features"]
        output_dir.mkdir(exist_ok = True,parents = True)
        self.output_dir = output_dir

    def create_dataset(self,test_start_date:str):

        #目的変数
        self.features["target"] = (self.features["rank"] == 1).astype(int)
        
        #学習データとテストデータに分割
        self.train_df = self.features.query("date < @test_start_date")
        self.test_df = self.features.query("date >= @test_start_date")
        

    def train(
        self,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        model_filename:str,
        importance_filename:str,
        evaluation_df_filename:str,
    ):

        #データセットの作成
        lgb_train = lgb.Dataset(train_df[self.feature_cols],train_df["target"])
        lgb_test = lgb.Dataset(test_df[self.feature.cols],test_df["target"])
        #パラメータの設定
        params = {
            "objective" : "binary",#二値分類
            "metric":"binary_logloss",#予測誤差
            "random_state":100,#実行ごとに同じ結果を得るための設定
            
            "verbosity" : -1,#学習中のログを非表示
            #num_leaves": 63.0,

            "max_depth":-1,

            
            #"optuna_seed" : 100, 
            "feature_fraction": 0.8,
            "bagging_fraction":0.8,            
            "learning_rate":0.01,
            "bagging_freq":1
        }

        #学習の実行
        model = lgb.train(
            params = params,
            train_set = lgb_train,
            num_boost_round = 900,
            valid_sets = [lgb_train,lgb_test],
            callbacks = [
                lgb.log_evaluation(100),
                #lgb.early_stopping(stopping_rounds = 100),
            ],
        )
        self.best_params = model.params
        
        with open(self.output_dir / model_filename,"wb") as f:
            pickle.dump(model,f)
        
        # 特徴量重要度の可視化
        lgb.plot_importance(
            model,importance_type = "gain",figsize=(30,15),max_numfeatures=50
        )
        plt.savefig(self.output_dir / f"{importance_filename}.png")
        plt.close()
        importance_df = pd.DataFrame(
            {
                "feature": model.feature_name(),
                "importance":model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance",ascending=False)
        importance_df.to_csv(
            self.output_dir / f"{importance_filename}.csv",
            index= False,
            sep="\t"
        )


        #テストデータに対してスコアリング
        evaluation_df = test_df[
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
        evaluation_df["pred"] = model.predict(
            test_df[self.feature_cols], num_iteration= model.best_iteration
        )
        logloss = log_loss(evaluation_df["target"],evaluatio_df["pred"])
        print("-" * 20 + "result" + "-" * 20)
        print(f"test_df's binary_logloss: {logloss}")
        return evaluation_df

    df run(
        self,
        valid_start_date:str,
        test_start_date:str,
        importance_filename: str = "importance",
        model_filename:str = "model.pkl",
        evaluation_filename:str = "evaluation.csv"

        

    )

















 