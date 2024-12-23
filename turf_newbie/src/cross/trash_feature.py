from pathlib import Path

from tqdm.notebook import tqdm
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from tqdm.notebook import tqdm
from webdriver_manager.chrome import ChromeDriverManager
from io import StringIO 

DATA_DIR = Path("..", "data")
POPULATION_DIR = DATA_DIR / "00_population"
INPUT_DIR = DATA_DIR / "01_preprocessed"
OUTPUT_DIR = DATA_DIR / "02_features"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


#create_baselogで日付の管理をしている　
#特徴量作成
class FeatureCreator:
    def __init__(
        self,
        population_dir: Path = POPULATION_DIR,
        poplation_filename: str = "population.csv",
        input_dir: Path = INPUT_DIR,
        results_filename: str = "results.csv",
        race_info_filename: str = "race_info.csv",
        horse_results_filename: str = "horse_results.csv",
        jockey_leading_filename: str = "jockey_leading.csv",
        trainer_leading_filename: str = "trainer_leading.csv",
        peds_filename: str = "peds.csv",
        sire_leading_filename: str = "sire_leading.csv",
        output_dir: Path = OUTPUT_DIR,
        output_filename: str = "features.csv",
    ):
        self.population = pd.read_csv(population_dir / poplation_filename, sep="\t")
        self.results = pd.read_csv(input_dir / results_filename, sep="\t")
        self.race_info = pd.read_csv(input_dir / race_info_filename, sep="\t")
        self.horse_results = pd.read_csv(input_dir / horse_results_filename, sep="\t")
        self.jockey_leading = pd.read_csv(input_dir / jockey_leading_filename, sep="\t")
        self.trainer_leading = pd.read_csv(
            input_dir / trainer_leading_filename, sep="\t"
        )
        self.peds = pd.read_csv(input_dir / peds_filename, sep="\t")
        self.sire_leading = pd.read_csv(input_dir / sire_leading_filename, sep="\t")
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.agg_horse_per_group_cols_dfs = {}

    def create_baselog(self):
        """
        horse_resultsをレース結果テーブルの日付よりも過去に絞り、集計元のログを作成。
        """
        self.baselog = (
            self.population.merge(
                self.horse_results, on="horse_id", suffixes=("", "_horse")
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
    
    def agg_horse_n_races(self, n_races: list[int] = [2,3, 5, 10,1000]) -> None:
        """
        直近nレースの賞金の平均を集計する関数。
        """
        baselog = (
            self.population.merge(
                self.results[["race_id", "horse_id","time_relative","rank_relative"]], 
                on=["race_id","horse_id"],
            )
            .merge(
                self.race_info[["race_id", "course_len", "race_type"]], on="race_id"
            )
            .merge(
                self.horse_results,
                on=["horse_id", "course_len", "race_type"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        grouped_df = baselog.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        

        for n_race in n_races:
            df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["prize","rank_relative","time_relative"]]
                .mean()
                .add_suffix(f"_{n_race}races")
            )
            merged_df = merged_df.merge(df, on=["race_id", "horse_id"])

        
        # merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
        # merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})        
        
        self.agg_horse_n_races_df = merged_df


    

    def agg_horse_n_races_relative(
        self, n_races: list[int] = [1,2, 3, 5, 10]
    ) -> None:
        """
        直近nレースの馬の過去成績を集計し、相対値に変換する関数。
        """
        baselog = (
            self.population.merge(
                self.results[["race_id", "horse_id","time_relative","rank_relative","corner_1_relative","corner_2_relative","corner_3_relative","corner_4_relative"]], 
                on=["race_id","horse_id"],
            )
            .merge(
                self.race_info[["race_id", "course_len", "race_type"]], on="race_id"
            )
            .merge(
                self.horse_results,
                on=["horse_id", "course_len", "race_type"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        grouped_df = baselog.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        
        for n_race in tqdm(n_races, desc="agg_horse_n_races_relative"):
            df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [
                        "rank",
                        "rank_relative",
                        
                        "prize",
                        "rank_diff",
                        "course_len",
                        "race_class",
                        "n_horses",
                        
                        # "time",
                        "time_relative",
                        
                        "nobori",
                        
                        # "corner_1",
                        # "corner_2",
                        # "corner_3",
                        # "corner_4",
                        
                        "corner_1_relative",
                        "corner_2_relative",
                        "corner_3_relative",
                        "corner_4_relative",
                        
                        "pase_1",
                        "pase_2",
                    ]
                ]
                .agg(["mean",  "max", "min"])
            )
            df.columns = ["_".join(col) + f"_{n_race}races" for col in df.columns]
            # レースごとの相対値に変換
            tmp_df = df.groupby(["race_id"])
            relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            merged_df = merged_df.merge(
                relative_df, on=["race_id", "horse_id"], how="left"
            )
        
        # merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
        # merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})
        
        self.agg_horse_n_races_relative_df = merged_df


    
    def agg_horse_per_course_len(
        self, n_races: list[int] = [1, 2, 3, 5, 10]
    ) -> None:
        """
        直近nレースの馬の過去成績を距離・race_typeごとに集計し、相対値に変換する関数。（各値を芝ダートとかで分類したものを集めた特徴量）
        """
        baselog = (
            self.population.merge(
                self.race_info[["race_id", "course_len", "race_type"]], on="race_id"
            )
            .merge(
                self.results[["race_id", "horse_id","rank_relative"]], 
                on=["race_id","horse_id"],
            )
            .merge(
                self.horse_results,
                on=["horse_id", "course_len", "race_type"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        grouped_df = baselog.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        for n_race in tqdm(n_races, desc="agg_horse_per_course_len"):
            df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [
                        "rank",
                        "rank_relative",
                        "prize",
                        "rank_diff",
                        "time",
                        "nobori",
                        "corner_1",
                        "corner_2",
                        "corner_3",
                        "corner_4",
                        "pase_1",
                        "pase_2",                        
                        "win",
                        "show",
                    ]
                ]
                .agg(["mean", "min"])
            )
            df.columns = [
                "_".join(col) + f"_{n_race}races_per_course_len" for col in df.columns
            ]
            # レースごとの相対値に変換
            tmp_df = df.groupby(["race_id"])
            relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            merged_df = merged_df.merge(
                relative_df, on=["race_id", "horse_id"], how="left"
            )

        # merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
        # merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})        
        
        self.agg_horse_per_course_len_df = merged_df
        

    
    def agg_horse_per_group_cols(
        self,
        group_cols: list[str],
        df_name: str,
        n_races: list[int] = [1, 2, 3, 5, 10],
    ) -> None:
        """
        直近nレースの馬の過去成績をgroup_colsごとに集計し、相対値に変換する関数。
        """
        baselog = (
            self.population.merge(
                self.race_info[["race_id"] + group_cols], on="race_id"
            )
            .merge(
                self.horse_results,
                on=["horse_id"] + group_cols,
                suffixes=("", "_horse"),
            )
            .merge(self.results[["race_id", "horse_id","time_relative","rank_relative","corner_1_relative","corner_2_relative","corner_3_relative","corner_4_relative"]], 
                   on=["race_id","horse_id"], 
                   )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        grouped_df = baselog.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        for n_race in tqdm(n_races, desc=f"agg_horse_per_{df_name}"):
            df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [
                        "rank",
                        "rank_relative",
                        
                        "prize",
                        "rank_diff",
                        "time",
                        "time_relative",
                        
                        "nobori",
                        "corner_1",
                        "corner_2",
                        "corner_3",
                        "corner_4",

                        "corner_1_relative",
                        "corner_2_relative",
                        "corner_3_relative",
                        "corner_4_relative",                        
                        
                        "pase_1",
                        "pase_2",                        
                        "win",
                        "show",
                    ]
                ]
                .agg(["mean", "max", "min"])
            )
            df.columns = [
                "_".join(col) + f"_{n_race}races_per_{df_name}" for col in df.columns
            ]
            # レースごとの相対値に変換
            tmp_df = df.groupby(["race_id"])
            relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            merged_df = merged_df.merge(
                relative_df, on=["race_id", "horse_id"], how="left"
            )

        # merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
        # merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})

        self.agg_horse_per_group_cols_dfs[df_name] = merged_df

    def agg_jockey(self):
        """
        騎手の過去成績を紐付け、相対値に変換する関数。
        """
        print("running agg_jockey()...")
        df = self.population.merge(
            self.results[["race_id", "horse_id", "jockey_id"]],
            on=["race_id", "horse_id"],
        )
        df["year"] = pd.to_datetime(df["date"]).dt.year - 1
        df = (
            df.merge(self.jockey_leading, on=["jockey_id", "year"], how="left")
            .drop(["date", "jockey_id", "year"], axis=1)
            .set_index(["race_id", "horse_id"])
            .add_prefix("jockey_")
        )
        # レースごとの相対値に変換
        tmp_df = df.groupby(["race_id"])
        relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")

        # relative_df = relative_df.astype({col: 'float32' for col in relative_df.select_dtypes('float64').columns})
        # relative_df = relative_df.astype({col: 'int32' for col in relative_df.select_dtypes('int64').columns})        
        
        self.agg_jockey_df = relative_df
        print("running agg_jockey()...comp")
        
    def agg_trainer(self):
        """
        調教師の過去成績を紐付け、相対値に変換する関数。
        """
        print("running agg_trainer()...")
        df = self.population.merge(
            self.results[["race_id", "horse_id", "trainer_id"]],
            on=["race_id", "horse_id"],
        )
        df["year"] = pd.to_datetime(df["date"]).dt.year - 1
        df = (
            df.merge(self.trainer_leading, on=["trainer_id", "year"], how="left")
            .drop(["date", "trainer_id", "year"], axis=1)
            .set_index(["race_id", "horse_id"])
            .add_prefix("trainer_")
        )
        # レースごとの相対値に変換
        tmp_df = df.groupby(["race_id"])
        relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")

        # relative_df = relative_df.astype({col: 'float32' for col in relative_df.select_dtypes('float64').columns})
        # relative_df = relative_df.astype({col: 'int32' for col in relative_df.select_dtypes('int64').columns})        
        
        self.agg_trainer_df = relative_df
        print("running agg_trainer()...comp")

    def agg_sire(self):
        """
        種牡馬の過去成績を紐付け、相対値に変換する関数。
        """
        print("running agg_sire()...")
        df = self.population.merge(
            self.peds[["horse_id", "sire_id"]],
            on="horse_id",
        ).merge(
            self.race_info[["race_id", "race_type", "course_len"]],
        )
        df["year"] = pd.to_datetime(df["date"]).dt.year - 1
        df = df.merge(
            self.sire_leading,
            on=["sire_id", "year", "race_type"],
            suffixes=("", "_sire"),
        ).set_index(["race_id", "horse_id"])
        df["course_len_diff"] = df["course_len"] - df["course_len_sire"]
        df = df[["n_races", "n_wins", "winrate", "course_len_diff"]].add_prefix("sire_")
        # レースごとの相対値に変換
        tmp_df = df.groupby(["race_id"])
        relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")

        # relative_df = relative_df.astype({col: 'float32' for col in relative_df.select_dtypes('float64').columns})
        # relative_df = relative_df.astype({col: 'int32' for col in relative_df.select_dtypes('int64').columns})        
        
        self.agg_sire_df = relative_df
        print("running agg_sire()...comp")

        
    # def agg_interval(self):
    #     """
    #     前走からの出走間隔を集計する関数
    #     """        
    #     merged_df = self.population.copy()
        
    #     # 最新のレース結果を取得
    #     latest_df = (
    #         self.baselog
    #         .groupby(["race_id", "horse_id", "date"])["date_horse"]
    #         .max()
    #         .reset_index()
    #     )
        
    #     # 出走間隔（days）を計算
    #     latest_df["interval"] = (
    #         pd.to_datetime(latest_df["date"]) - pd.to_datetime(latest_df["date_horse"])
    #     ).dt.days
        
    #     # 'race_id', 'horse_id', 'intrerval' 列を指定してマージ
    #     # merged_df = merged_df.merge(
    #     #     latest_df[["race_id", "horse_id", "interval"]],                  
    #     #     on=["race_id", "horse_id"], 
    #     #     how="left"
    #     # )
    #     merged_df = merged_df.merge(
    #         latest_df[["race_id", "horse_id","interval"]],                  
    #         on=["race_id", "horse_id"], 
    #         how="left"
    #     )

        
    #     merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
    #     merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})        
        
    #     # 結果をインスタンス変数に保存
    #     self.agg_interval_df = merged_df
    #     print("running agg_interval()...comp")

    
    # def cross_features(self):
    #     """
    #     枠番、レースタイプ、直線か右左かの勝率比較_交互作用特徴量
    #     """
    #     merged_df = self.population.copy()        
    #     merged_df = merged_df.merge(
    #         self.results[["race_id", "horse_id","wakuban", "umaban", "sex"]],
    #         on=["race_id", "horse_id"],
    #     )
    #     df = merged_df.merge(
    #         self.race_info[["race_id", "race_type", "around"]],
    #         on=["race_id"],
    #     ) 
    
    #     # wakuban と race_type の交互作用特徴量
    #     df["wakuban_race_type"] = df["race_type"].map({0: 1, 1: -1, 2: 0}).fillna(0) * df["wakuban"]
    #     df["wakuban_around"] = df["around"].map({2: 1}).fillna(0) * df["wakuban"]
    #     df["umaban_race_type"] = df["race_type"].map({0: 1, 1: -1, 2: 0}).fillna(0) * df["umaban"]
    #     df["umaban_around"] = df["around"].map({2: 1}).fillna(0) * df["umaban"]
    
    #     # 季節 (日付) と性別に基づく交互作用特徴量
    #     df["date"] = pd.to_datetime(df["date"])
    #     df["sin_date"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365)
    #     df["cos_date"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365) + 1
    
    #     df["sin_date_sex"] = df["sex"].map({0: -1, 1: 1}) * df["sin_date"]
    #     df["cos_date_sex"] = df["sex"].map({0: -1, 1: 1}) * df["cos_date"]
    
    #     merged_df = df[["race_id", "horse_id", "wakuban_race_type", "date","wakuban_around","umaban_race_type","umaban_around", "sin_date_sex", "cos_date_sex"]]
        
    #     merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
    #     merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})
        
    #     self.agg_cross_features_df= merged_df
    #     print("running cross_features()...comp")

        
    def agg_interval(self):
        """
        前走からの出走間隔を集計する関数
        """        
        merged_df = self.population.copy()
        
        # 最新のレース結果を取得
        latest_df = (
            self.baselog
            .groupby(["race_id", "horse_id", "date"])["date_horse"]
            .max()
            .reset_index()
        )
        
        # 出走間隔（days）を計算
        latest_df["interval"] = (
            pd.to_datetime(latest_df["date"]) - pd.to_datetime(latest_df["date_horse"])
        ).dt.days
        
        # 'race_id', 'horse_id', 'intrerval' 列を指定してマージ
        # merged_df = merged_df.merge(
        #     latest_df[["race_id", "horse_id", "interval"]],                  
        #     on=["race_id", "horse_id"], 
        #     how="left"
        # )
        merged_df = merged_df.merge(
            latest_df[["race_id", "horse_id", "interval"]],                  
            on=["race_id", "horse_id"], 
            how="left"
        )

        
        # merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
        # merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})        
        
        # 結果をインスタンス変数に保存
        self.agg_interval_df = merged_df
        print("running agg_interval()...comp")
    
    
    def cross_features(self):
        """
        枠番、レースタイプ、直線か右左かの勝率比較_交互作用特徴量
        """
        merged_df = self.population.copy()        
        df = (
            self.results[["race_id", "horse_id", "wakuban", "umaban", "sex"]]
            .merge(self.race_info[["race_id", "race_type", "around"]], on="race_id")
            .merge(merged_df[["race_id", "horse_id", "date"]], on=["race_id", "horse_id"], how="left")
        )

    
        # wakuban と race_type の交互作用特徴量
        df["wakuban_race_type"] = df["race_type"].map({0: 1, 1: -1, 2: 0}).fillna(0) * df["wakuban"]
        df["wakuban_around"] = df["around"].map({2: 1}).fillna(0) * df["wakuban"]
        df["umaban_race_type"] = df["race_type"].map({0: 1, 1: -1, 2: 0}).fillna(0) * df["umaban"]
        df["umaban_around"] = df["around"].map({2: 1}).fillna(0) * df["umaban"]
    
        # 季節 (日付) と性別に基づく交互作用特徴量
        df["date_1"] = pd.to_datetime(df["date"])
        df["sin_date"] = np.sin(2 * np.pi * df["date_1"].dt.dayofyear / 365)
        df["cos_date"] = np.cos(2 * np.pi * df["date_1"].dt.dayofyear / 365) + 1
    
        df["sin_date_sex"] = df["sex"].map({0: -1, 1: 1}) * df["sin_date"]
        df["cos_date_sex"] = df["sex"].map({0: -1, 1: 1}) * df["cos_date"]
        
    
        merged_df = merged_df.merge(
            df[["race_id", "horse_id", "date","wakuban_race_type", "wakuban_around","umaban_race_type","umaban_around", "sin_date_sex", "cos_date_sex"]],
            on=["race_id","date", "horse_id"],
            how="left"
        )

        # merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
        # merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})
        
        self.agg_cross_features_df= merged_df
        print("running cross_features()...comp")

    

    def create_features(self) -> pd.DataFrame:
        """
        特徴量作成処理を実行し、populationテーブルに全ての特徴量を結合する。
        """
        # 馬の過去成績集計
        self.create_baselog()
        self.agg_horse_n_races()
        self.agg_horse_n_races_relative()
        self.agg_jockey()
        self.agg_trainer()
        self.agg_horse_per_course_len()
        self.cross_features()
        self.agg_interval()  


        self.agg_horse_per_group_cols(
            group_cols=["ground_state", "race_type"], df_name="ground_state_race_type"
        )
        self.agg_horse_per_group_cols(
            group_cols=["race_class"], df_name="race_class"
        )
        # self.agg_horse_per_group_cols(
        #     group_cols=["around","wakuban"], df_name="around_per_wakuban"
        # )
        self.agg_horse_per_group_cols(
            group_cols=["race_type"], df_name="race_type"
        )
        self.agg_horse_per_group_cols(
            group_cols=["place","course_len", "race_type"], df_name="race_place_len"
        )
        #self.agg_horse_per_group_cols(
        #    group_cols=["place", "race_type"], df_name="race_place_type"
        #)

        self.agg_horse_per_group_cols(
            group_cols=["weather"], df_name="weather"
        )
                
        self.agg_sire()
        # 全ての特徴量を結合
        print("merging all features...")
        features = (
            self.population.merge(self.results, on=["race_id", "horse_id"])
            .merge(self.race_info, on=["race_id", "date"])
            .merge(
                self.agg_horse_n_races_df,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_horse_n_races_relative_df,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_jockey_df,
                on=["race_id", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_trainer_df,
                on=["race_id", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_horse_per_course_len_df,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_horse_per_group_cols_dfs["ground_state_race_type"],
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_horse_per_group_cols_dfs["race_class"],
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_horse_per_group_cols_dfs["race_type"],
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_horse_per_group_cols_dfs["race_place_len"],
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_interval_df,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_cross_features_df,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )            
            # .merge(
            #     self.agg_horse_per_group_cols_dfs["around_per_wakuban"],
            #     on=["race_id", "date", "horse_id"],
            #     how="left",
            # )
            # .merge(
            #     self.agg_horse_per_group_cols_dfs["race_place_type"],
            #     on=["race_id", "date", "horse_id"],
            #     how="left",
            # )
            .merge(
                self.agg_horse_per_group_cols_dfs["weather"],
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )                     
            .merge(
                self.agg_sire_df,
                on=["race_id", "horse_id"],
                how="left",
                # copy=False,
            )
        )
        features.to_csv(self.output_dir / self.output_filename, sep="\t", index=False)
        print("merging all features...comp")
        return features
