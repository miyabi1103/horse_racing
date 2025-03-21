# import pickle
# from pathlib import Path
# import lightgbm as lgb
# import matplotlib.pyplot as plt
# import pandas as pd
# import yaml
# from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from IPython.display import display

# DATA_DIR = Path("..", "data")
# INPUT_DIR = DATA_DIR / "02_features"
# OUTPUT_DIR = DATA_DIR / "03_train"
# OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# #グリッドサーチを行うと死ぬほど時間がかかるので断念

# class Trainer_lightgbm_rank_niti_nested_cv:
#     def __init__(
#         self, 
#         input_dir: Path = INPUT_DIR, 
#         features_filename: str = "features.csv", 
#         config_filepath: Path = "config_lightgbm_niti.yaml", 
#         output_dir: Path = OUTPUT_DIR
#     ):
#         self.features = pd.read_csv(input_dir / features_filename, sep="\t")
#         with open(config_filepath, "r") as f:
#             config = yaml.safe_load(f)
#         self.feature_cols = config["features"]
#         self.params = config["params"]
#         self.output_dir = output_dir

#     def create_dataset_for_cv(
#         self, 
#         n_splits: int = 10
#     ):
#         """ 時系列クロスバリデーションのため、データをn_splitsに分割
#         学習データを徐々に増やし、評価データを次に進めていく
#         """
#         # 目的変数
#         self.features["target"] = (self.features["rank"] == 1).astype(int)

#         # 時系列でデータを並べ替え
#         self.features = self.features.sort_values("date")

#         # 時系列クロスバリデーションのインデックス
#         n_samples = len(self.features)
#         fold_size = n_samples // n_splits
#         for i in range(n_splits):
#             # 学習データの範囲
#             train_df = self.features.iloc[:(i + 1) * fold_size]
#             # 評価データの範囲
#             valid_df = self.features.iloc[(i + 1) * fold_size: (i + 2) * fold_size]
#             # 「馬の過去成績」の欠損率
#             valid_df = valid_df.query('rank_3races.notna()')
#             # 学習データと評価データを返す
#             yield train_df, valid_df

#     def train(
#         self, 
#         train_df: pd.DataFrame, 
#         valid_df: pd.DataFrame, 
#         importance_filename: str, 
#         model_filename: str, 
#         init_model=None, 
#         boost_rounds: int = None
#     ):
#         # データセットの作成
#         lgb_train = lgb.Dataset(train_df[self.feature_cols], train_df["target"])
#         lgb_valid = lgb.Dataset(valid_df[self.feature_cols], valid_df["target"], reference=lgb_train)

#         # 学習の実行
#         model = lgb.train(
#             params=self.params,
#             train_set=lgb_train,
#             num_boost_round=boost_rounds if boost_rounds else 10000,  # boost_roundsが指定されたらそれを使う
#             valid_sets=[lgb_valid],
#             callbacks=[
#                 lgb.log_evaluation(100),
#                 lgb.early_stopping(stopping_rounds=100),
#             ] if boost_rounds is None else [],  # boost_roundsが指定されていない場合のみアーリーストッピング
#             init_model=init_model  # init_modelをここで指定
#         )
#         self.best_params = model.params
#         with open(self.output_dir / model_filename, "wb") as f:
#             pickle.dump(model, f)

#         # 特徴量重要度の可視化
#         lgb.plot_importance(model, importance_type="gain", figsize=(30, 15), max_num_features=50)
#         plt.savefig(self.output_dir / f"{importance_filename}.png")
#         plt.close()
#         importance_df = pd.DataFrame({
#             "feature": model.feature_name(),
#             "importance": model.feature_importance(importance_type="gain"),
#         }).sort_values("importance", ascending=False)
#         importance_df.to_csv(self.output_dir / f"{importance_filename}.csv", index=False, sep="\t")

#         # 評価
#         evaluation_df = valid_df[["race_id", "horse_id", "target", "rank", "tansho_odds", "popularity", "umaban"]].copy()
#         evaluation_df["pred"] = model.predict(valid_df[self.feature_cols], num_iteration=model.best_iteration)

#         # 0.5 を閾値にしてクラス分類
#         evaluation_df["pred_binary"] = (evaluation_df["pred"] >= 0.07267).astype(int)

#         # 評価指標の計算
#         accuracy = accuracy_score(evaluation_df["target"], evaluation_df["pred_binary"])
#         precision = precision_score(evaluation_df["target"], evaluation_df["pred_binary"])
#         recall = recall_score(evaluation_df["target"], evaluation_df["pred_binary"])
#         f1 = f1_score(evaluation_df["target"], evaluation_df["pred_binary"])
#         roc_auc = roc_auc_score(evaluation_df["target"], evaluation_df["pred"])

#         # 結果を出力
#         print("-" * 20 + " Metrics " + "-" * 20)
#         print(f"Accuracy: {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall: {recall:.4f}")
#         print(f"F1 Score: {f1:.4f}")
#         print(f"ROC AUC: {roc_auc:.4f}")

#         # ここで num_boost_round を返す
#         return evaluation_df, model.best_iteration

#     def run(
#         self, 
#         n_splits: int = 10, 
#         importance_filename: str = "importance_lightgbm_rank_niti_nested_cv", 
#         model_filename: str = "model_lightgbm_rank_niti_nested_cv.pkl", 
#         evaluation_filename: str = "evaluation_lightgbm_rank_niti_nested_cv.csv", 
#         final_model_filename: str = "model_lightgbm_rank_niti_nested_cv_full.pkl"
#     ):
#         """ 時系列クロスバリデーションを実行し、ハイパーパラメータ調整を行う
#         最後に全データを使って学習したモデルを保存
#         """

    
#         # ハイパーパラメータのグリッドサーチ設定
#         param_grid = {
#             "learning_rate": [0.01,  0.1],
#             "feature_fraction": [0.65, 0.8],
#             "bagging_fraction": [0.7, 0.8],
#             "num_leaves": [31, 63, 255],
#             "max_depth": [-1, 16],
#             "bagging_freq": [1, 5],
#             "random_state": [100],
#             "verbosity": [-1]
#         }

#         # ハイパーパラメータ	デフォルト値	説明
#         # feature_fraction	1.0
#         # bagging_fraction	1.0
#         # random_state	None
#         # verbosity	1
#         # objective	'binary'	目的関数。バイナリ分類では 'binary'
#         # boosting_type	'gbdt'	ブースティングタイプ。gbdt（Gradient Boosting Decision Tree）が一般的
#         # num_leaves	31	各決定木の葉の数。大きな値にするとモデルが複雑になる
#         # max_depth	-1	木の最大深さ。-1 は制限なし
#         # learning_rate	0.1	学習率。小さくすると精度が高まるが、学習に時間がかかる
#         # n_estimators	100	学習する決定木の数
#         # subsample	1.0	サンプリング割合（データのサンプル割合）。1.0 は全データを使用
#         # colsample_bytree	1.0	各ツリーでサンプリングする特徴量の割合。1.0 は全特徴量を使用
#         # min_data_in_leaf	20	各葉に必要な最小データ数。小さくするとモデルが過学習しやすい


#         # 時系列クロスバリデーション用のTimeSeriesSplit
#         tscv = TimeSeriesSplit(n_splits=n_splits)

#         # 交差検証で最適なハイパーパラメータを調整する
#         gs = GridSearchCV(estimator=lgb.LGBMClassifier(objective="binary", metric="binary_logloss"), param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2, scoring="roc_auc")
        
#         # クロスバリデーションをn_splits-1回だけ実行する
#         evaluation_dfs = []
#         for fold_idx, (train_df, valid_df) in enumerate(self.create_dataset_for_cv(n_splits)):
#             if fold_idx == n_splits - 1:
#                 break  # 最後のfoldは実行しない
#             print(f"Training fold {fold_idx + 1}/{n_splits}...")

#             # 前回学習したモデルを指定（初回はNone）
#             init_model = None
#             if fold_idx > 0:
#                 prev_model_filename = self.output_dir / f"{model_filename}_fold{fold_idx}"
#                 with open(prev_model_filename, "rb") as f:
#                     init_model = pickle.load(f)

#             # モデルのハイパーパラメータ調整
#             gs.fit(train_df[self.feature_cols], train_df["target"])

#             # 最適なモデルを取得
#             best_model = gs.best_estimator_

#             # 評価
#             evaluation_df, num_boost_round = self.train(
#                 train_df, valid_df, importance_filename=f"{importance_filename}_fold{fold_idx + 1}", model_filename=f"{model_filename}_fold{fold_idx + 1}", init_model=init_model
#             )
#             evaluation_dfs.append(evaluation_df)

#         # クロスバリデーション後の評価データをまとめて保存
#         full_evaluation_df = pd.concat(evaluation_dfs, axis=0)
#         full_evaluation_df.to_csv(self.output_dir / evaluation_filename, sep="\t", index=False)

#         # 最終モデルを学習（アーリーストッピングなし）
#         print("Training final model using all data...")

#         # 最後に使ったブーストラウンド数を取得
#         final_train_df = self.features
#         lgb_train = lgb.Dataset(final_train_df[self.feature_cols], final_train_df["target"])
#         additional_boost_round = 300 
#         total_boost_round = num_boost_round+ additional_boost_round

#         # 最終モデルを学習（前回のモデルを引き継ぐ）
#         model = lgb.train(
#             self.params,
#             lgb_train,
#             num_boost_round=total_boost_round,
#             init_model=init_model
#         )

#         # 最終モデルの保存
#         with open(self.output_dir / final_model_filename, "wb") as f:
#             pickle.dump(model, f)
        
#         # モデル評価の結果を表示
#         print("Final model training completed.")



import pickle
from pathlib import Path
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from IPython.display import display

DATA_DIR = Path("..", "data")
INPUT_DIR = DATA_DIR / "02_features"
OUTPUT_DIR = DATA_DIR / "03_train"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

class Trainer_lightgbm_rank_niti_cv:
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

    def create_dataset_for_cv(
        self, 
        n_splits: int = 5
    ):
        """ 時系列クロスバリデーションのため、データをn_splitsに分割
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
            # 学習データと評価データを返す
            yield train_df, valid_df

    def train(
        self, 
        train_df: pd.DataFrame, 
        valid_df: pd.DataFrame, 
        importance_filename: str, 
        model_filename: str, 
        init_model=None, 
        boost_rounds: int = None
    ):
        # データセットの作成
        lgb_train = lgb.Dataset(train_df[self.feature_cols], train_df["target"])
        lgb_valid = lgb.Dataset(valid_df[self.feature_cols], valid_df["target"], reference=lgb_train)

        # 学習の実行
        model = lgb.train(
            params=self.params,
            train_set=lgb_train,
            num_boost_round=boost_rounds if boost_rounds else 10000,  # boost_roundsが指定されたらそれを使う
            valid_sets=[lgb_valid],
            callbacks=[
                lgb.log_evaluation(100),
                lgb.early_stopping(stopping_rounds=100),
            ] if boost_rounds is None else [],  # boost_roundsが指定されていない場合のみアーリーストッピング
            init_model=init_model  # init_modelをここで指定
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
        return evaluation_df, model.best_iteration

    def run(
        self, 
        n_splits: int = 5, 
        importance_filename: str = "importance_lightgbm_rank_niti_cv", 
        model_filename: str = "model_lightgbm_rank_niti_cv.pkl", 
        evaluation_filename: str = "evaluation_lightgbm_rank_niti_cv.csv", 
        final_model_filename: str = "model_lightgbm_rank_niti_cv_full.pkl"
    ):
        """ 時系列クロスバリデーションを実行し、最後に全データを使って学習したモデルを保存
        """

        # 時系列クロスバリデーション用のTimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # クロスバリデーションをn_splits-1回だけ実行する
        evaluation_dfs = []
        for fold_idx, (train_df, valid_df) in enumerate(self.create_dataset_for_cv(n_splits)):
            if fold_idx == n_splits - 1:
                break  # 最後のfoldは実行しない
            print(f"Training fold {fold_idx + 1}/{n_splits}...")

            # 前回学習したモデルを指定（初回はNone）
            init_model = None
            if fold_idx > 0:
                prev_model_filename = self.output_dir / f"{model_filename}_fold{fold_idx}"
                with open(prev_model_filename, "rb") as f:
                    init_model = pickle.load(f)

            # 評価
            evaluation_df, num_boost_round = self.train(
                train_df, valid_df, importance_filename=f"{importance_filename}_fold{fold_idx + 1}", model_filename=f"{model_filename}_fold{fold_idx + 1}", init_model=init_model
            )
            evaluation_dfs.append(evaluation_df)

        # クロスバリデーション後の評価データをまとめて保存
        full_evaluation_df = pd.concat(evaluation_dfs, axis=0)
        full_evaluation_df.to_csv(self.output_dir / evaluation_filename, sep="\t", index=False)

        # 最終モデルを学習（アーリーストッピングなし）
        print("Training final model using all data...")

        final_train_df = self.features
        lgb_train = lgb.Dataset(final_train_df[self.feature_cols], final_train_df["target"])
        additional_boost_round = 2000 
        total_boost_round = num_boost_round + additional_boost_round

        # 最終モデルを学習（前回のモデルを引き継ぐ）
        model = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=additional_boost_round,
            init_model=init_model
        )

        # 最終モデルの保存
        with open(self.output_dir / final_model_filename, "wb") as f:
            pickle.dump(model, f)
        
        # モデル評価の結果を表示
        print("Final model training completed.")
