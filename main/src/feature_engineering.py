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
# from webdriver_manager.chrome import ChromeDriverManager
from io import StringIO 

DATA_DIR = Path("..", "data")
POPULATION_DIR = DATA_DIR / "00_population"
INPUT_DIR = DATA_DIR / "01_preprocessed"
OUTPUT_DIR = DATA_DIR / "02_features"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


OUTPUT_INFO_DIR = Path("..", "data", "01_preprocessed")

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
        output_dir_info: Path = OUTPUT_INFO_DIR,
        output_filename_info: str = "race_info.csv",
        bms_leading_filename: str = "bms_leading.csv",  

        poplation_all_filename: str = "population_all.csv",    
        results_all_filename: str = "results_all.csv",
        race_info_all_filename: str = "race_info_all.csv",
        horse_results_all_filename: str = "horse_results_all.csv",  
        peds_all_filename: str = "peds_all.csv",      
    ):
        self.population = pd.read_csv(population_dir / poplation_filename, sep="\t")
        self.results = pd.read_csv(input_dir / results_filename, sep="\t")
        self.race_info_before = pd.read_csv(input_dir / race_info_filename, sep="\t")
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

        self.output_dir_info = output_dir_info
        self.output_filename_info = output_filename_info
        self.bms_leading = pd.read_csv(input_dir / bms_leading_filename, sep="\t")  

        self.all_population = pd.read_csv(population_dir / poplation_all_filename, sep="\t")
        self.all_results = pd.read_csv(input_dir / results_all_filename, sep="\t")
        self.all_race_info = pd.read_csv(input_dir / race_info_all_filename, sep="\t")
        self.all_horse_results = pd.read_csv(input_dir / horse_results_all_filename, sep="\t")
        self.all_peds = pd.read_csv(input_dir / peds_all_filename, sep="\t")

        self.race_info = pd.read_csv(input_dir / race_info_filename, sep="\t")

    # def create_race_grade(self):
    #     """
    #     horse_resultsをレース結果テーブルの日付よりも過去に絞り、集計元のログを作成。
    #     """
    #     df = (
    #         self.results[["race_id", "horse_id","mean_age_kirisute"]]
    #         .merge(self.race_info_before[["race_id", "race_type", "place", "season_level","race_class"]], on="race_id")
    #     )
    #     # self.race_info = self.race_info_before.copy()
    #     race_i = self.race_info_before


    #     """

    #     dfに新たな列、race_gradeを作成して欲しい
    #     作成ルールは以下の通りである
    #     'age_season'の条件に引っかかった場合、それを優先すること
    #     次点で"race_class"の条件にかかっても、'age_season'がある方を優先して変換すること


    #     "race_class"列が0は55

    #     "race_class"列が1は60

    #     "race_class"列が2は70
    #     2歳それ以外は68（20<='age_season'<30かつ、2<="race_class"列<5の行）
    #     2歳G2,G3,OPは73（20<='age_season'<30かつ、5<="race_class"列<8の行）

    #     "race_class"列が3は79
    #     2歳G1は79（20<='age_season'<30かつ、8<="race_class"の行）
    #     3歳春OPは80（30<='age_season'<33かつ、4<="race_class"列<6の行）
    #     3歳春G2.G3は81（30<='age_season'<33かつ、6<="race_class"列<8の行）

    #     "race_class"列が4は85
    #     3歳春G1は86（30<='age_season'<33かつ、8<="race_class"の行）
    #     3歳秋G2,G3は86（33<='age_season'<40かつ、5<="race_class"列<8の行）

    #     "race_class"列が5は89
    #     3歳秋G1は91（33<='age_season'<40かつ、8<="race_class"の行）

    #     "race_class"列が6は92

    #     "race_class"列が7は94

    #     "race_class"列が8は98




    #     これらを小さく（1/10 - 5）した列

    #     G1 8	100
    #     G2 7	95
    #     G3 6	92
    #     オープン5	89
    #     1600万4	86
    #     ２勝クラス3	80
    #     １勝クラス2	70
    #     未勝利1	60
    #     新馬0	55


    #     クラス	芝	ダート
    #     未勝利	６５（-１５）	６０（-２０）
    #     500万下
    #     Ｇ１を除く２歳ＯＰ	７５（-５）	７２（-８）
    #     1000万下
    #     ２歳Ｇ１
    #     Ｇ１を除く３歳春ＯＰ	８３（３）	８３（３）
    #     1600万下
    #     ３歳春Ｇ１
    #     ３歳秋重賞	８８（８）	９０（１０）
    #     ＯＰ（ただしダート重賞を除く）
    #     ３歳秋Ｇ１	９３（１３）	９５（１５）
    #     ダート重賞（３歳を除く）	－	１００（２０）
    #     古馬Ｇ１	９８（１８）	１０５（２５）
    #     """
    #     # "mean_age_kirisute"と"season"を文字列に変換して結合し、int型に変換して新しい列 "age_season" を作成
    #     df['age_season'] = (df['mean_age_kirisute'].astype(int).astype(str) + df["season_level"].astype(str)).astype(int)
    #     df["race_class"] = df["race_class"].astype(int)
    #     place_adjustment = {
    #         44: 0, 43: 0, 45: -1,30: -3,
    #         54: -4,50: -5,51: -5,42: -5,48: -7,
    #         35: -9,36: -10,46: -13,
    #         55: -16,47: -19,
            
    #     }
    #     # race_gradeの作成
    #     def calculate_race_grade(row):
    #         age_season = row['age_season']
    #         race_class = row['race_class']
    #         place = row['place']
    #         # 競馬場ごとの補正値を取得（該当しない場合は0）
    #         adjustment = place_adjustment.get(place, 0)

    #         # 'age_season' に基づく条件を優先してチェック
    #         if 20 <= age_season < 30:
    #             if 2 <= race_class < 5:
    #                 return 70
    #             elif 5 <= race_class < 8:
    #                 return 70
    #             elif 8 <= race_class:
    #                 return 79
    #         elif 30 <= age_season < 33:
    #             if 4 <= race_class < 6:
    #                 return 79
    #             elif 6 <= race_class < 8:
    #                 return 79
    #             elif 8 <= race_class:
    #                 return 85
    #         elif 33 <= age_season < 40:
    #             if 5 <= race_class < 8:
    #                 return 85
    #             elif 8 <= race_class:
    #                 return 91
            
    #         if race_class == 0:
    #             return 55
    #         elif race_class == 1:
    #             return 60
    #         elif race_class == 2:
    #             return 70
    #         elif race_class == 3:
    #             return 79
    #         elif race_class == 4:
    #             return 85
    #         elif race_class == 5:
    #             return 89
    #         elif race_class == 6:
    #             return 91
    #         elif race_class == 7:
    #             return 94
    #         elif race_class == 8:
    #             return 98
            

    #         elif race_class == -4:
    #             base_grade = 82
    #         elif race_class == -3:
    #             base_grade = 83
    #         elif race_class == -2:
    #             base_grade = 84
    #         elif race_class == -1:
    #             base_grade = 85
    #         elif race_class == -5:
    #             base_grade = 80
    #         elif race_class == -6:
    #             base_grade = 79
    #         elif race_class == -7:
    #             base_grade = 74
    #         elif race_class == -8:
    #             base_grade = 69
    #         elif race_class == -9:
    #             base_grade = 64
    #         elif race_class == -10:
    #             base_grade = 59
    #         elif race_class == -11:
    #             base_grade = 55
    #         elif race_class == -11.5:
    #             base_grade = 53
    #         elif race_class == -12.5:
    #             base_grade = 48
    #         elif race_class == -12:
    #             base_grade = 50
    #         elif race_class == -13:
    #             base_grade = 50
    #         elif race_class == -14:
    #             base_grade = 40
    #         elif race_class == -15:
    #             base_grade = 30
    #         else:
    #             return np.nan  


    #         # 競馬場ごとの補正値を適用
    #         return base_grade + adjustment



    #     # race_grade列を作成
    #     df['race_grade'] = df.apply(calculate_race_grade, axis=1)

    #     #race_grade_scaledの作成
    #     df['race_grade_scaled'] = df['race_grade'] / 10 - 5
    #     df_agg = df.groupby("race_id", as_index=False).agg({"race_grade": "first",'age_season':"first", "race_grade_scaled": "first"})

    #     # self.race_info[['age_season', 'race_grade', 'race_grade_scaled']] = df[['age_season', 'race_grade', 'race_grade_scaled']]
    #     self.race_info = (
    #         race_i
    #         .merge(df_agg[["race_id",'race_grade', 'age_season','race_grade_scaled']], on="race_id",how="left")
    #     )
        
    #     self.race_info.to_csv(self.output_dir_info/self.output_filename_info, sep="\t", index=False)



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
    
    def agg_horse_n_races(self, n_races: list[int] = [1,3, 5, 10,1000]) -> None:
        """
        直近nレースの賞金の平均を集計する関数。
        """
        # baselog = (
        #     self.population.merge(
        #         self.horse_results,
        #         on="horse_id",
        #         suffixes=("", "_horse"),
        #     )
        #     .query("date_horse < date")
        #     .sort_values("date_horse", ascending=False)
        # )
        # grouped_df = baselog.groupby(["race_id", "horse_id"])
        # merged_df = self.population.copy()

        grouped_df = self.baselog.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()

        for n_race in n_races:
            df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["rank","rank_per_horse","prize"]]
                .mean()
                .add_suffix(f"_{n_race}races")
            )
            merged_df = merged_df.merge(df, on=["race_id", "horse_id"])

        
        # merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
        # merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})        
        
        self.agg_horse_n_races_df = merged_df



    
    # def agg_horse_n_races_relative(
    #     self, n_races: list[int] = [1, 3, 5, 10]
    # ) -> None:
    #     """
    #     直近nレースの平均を集計して標準化した関数。
    #     """
    #     grouped_df = self.baselog.groupby(["race_id", "horse_id"])
    #     merged_df = self.population.copy()
    #     for n_race in tqdm(n_races, desc="agg_horse_n_races_relative"):
    #         df = (
    #             grouped_df.head(n_race)
    #             .groupby(["race_id", "horse_id"])[
    #                 [
    #                     "rank",
    #                     "rank_per_horse",
                        
    #                     "prize",
    #                     "rank_diff",
    #                     # "course_len",
    #                     "race_grade",
    #                     #"time",
                        
    #                     "nobori",
    #                     # "n_horses",       
                        
    #                     # "corner_1",
    #                     # "corner_2",
    #                     # "corner_3",
    #                     # "corner_4",
    #                     "corner_1_per_horse",
    #                     "corner_2_per_horse",
    #                     "corner_3_per_horse",
    #                     "corner_4_per_horse",                        
                        
    #                     "pace_1",
    #                     "pace_2",
    #                 ]
    #             ]
    #             .agg(["mean", "max", "min"])
    #         )
    #         df.columns = ["_".join(col) + f"_{n_race}races" for col in df.columns]
    #         # レースごとの相対値に変換

    #         tmp_df = df.groupby(["race_id"])
            
    #         relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
    #         merged_df = merged_df.merge(
    #             relative_df, on=["race_id", "horse_id"], how="left"
    #         )

        
        
    #     # merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
    #     # merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})        
        
    #     self.agg_horse_n_races_relative_df = merged_df




    def agg_course_len(
        self, n_races: list[int] = [1, 3, 5, 10]
    ) -> None:
        """
        直近nレースの平均を集計して標準化した関数。
        """
        baselog_df_info = self.baselog.merge(
                self.race_info[["race_id", "course_len"]], on="race_id", suffixes=("", "_info")
            )
        baselog_df = baselog_df_info.merge(
            self.results[["race_id", "n_horses"]], on="race_id", suffixes=("", "_results")
            )
                        
        baselog_df["course_len_diff"] = baselog_df["course_len_info"] - baselog_df["course_len"]
        baselog_df["n_horses_diff"] = baselog_df["n_horses_results"] - baselog_df["n_horses"]

        grouped_df = baselog_df.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()

        for n_race in tqdm(n_races, desc="course_len_relative"):
            df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [
                        "course_len",
                        "course_len_diff",
                        "n_horses",
                        "n_horses_diff",
                    ]
                ]
                .agg(["mean", "max", "min"])
            )
            df.columns = ["_".join(col) + f"_{n_race}races" for col in df.columns]
            # レースごとの相対値に変換
            original_df = df.copy()

            tmp_df = df.groupby(["race_id"])
            relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            merged_df = merged_df.merge(
                relative_df, on=["race_id", "horse_id"], how="left"
            )
            merged_df = merged_df.merge(original_df, on=["race_id", "horse_id"], how="left")

        
        
        # merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
        # merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})        
        
        self.course_len_df = merged_df




    def results_relative(
        self, n_races: list[int] = [1, 3, 5, 10]
    ) -> None:
        """
        直近nレースの平均を集計して標準化した関数。
        """
        merged_df = self.population.copy()
        results_normal = self.population.merge(
                self.results[["race_id","horse_id","umaban","wakuban","impost","age","weight","weight_diff","impost_percent"]], on=["race_id","horse_id"]
            )

        required_columns = ["umaban", "wakuban", "impost", "age", "weight", "weight_diff", "impost_percent"]


        tmp_df_results_normal= results_normal.groupby("race_id")[required_columns]
        mean_results_normal = tmp_df_results_normal.transform("mean")
        std_results_normal = tmp_df_results_normal.transform("std").replace(0, np.nan)  # 標準偏差が 0 の場合 NaN に置換
        relative_cols = [f"{col}_relative" for col in required_columns]
        results_normal[relative_cols] = (results_normal[required_columns] - mean_results_normal) / std_results_normal


        results_normal_1 = results_normal.merge(
            self.race_info[["race_id","start_point","ground_state"]],
            on="race_id", 
            how="left"
        )
        # 'start_point'が2の行だけに対して処理を行う
        results_normal_1.loc[
            (results_normal_1["start_point"] == 2) & (results_normal_1["ground_state"] == 0),
            "start_point_umaban"
        ] = results_normal_1["umaban_relative"]

        results_normal_1.loc[
            (results_normal_1["start_point"] == 2) & (results_normal_1["ground_state"] != 0),
            "start_point_umaban"
        ] = results_normal_1["umaban_relative"] * -1


        merged_df = merged_df.merge(
            results_normal_1[["race_id", "horse_id","umaban_relative","start_point_umaban", "wakuban_relative", "impost_relative", "age_relative", "weight_relative", "weight_diff_relative", "impost_percent_relative"]], 
            on=["race_id", "horse_id"], 
            how="left"
        )



        
        
        # merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
        # merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})        
        
        self.results_relative_df = merged_df


    
    def agg_horse_per_course_len(
        self, n_races: list[int] = [1, 3, 5, 10]
    ) -> None:
        """
        直近nレースの馬の過去成績を距離・race_typeごとに集計し、相対値に変換する関数。（各値を芝ダートとかで分類したものを集めた特徴量）
        タイプとコースの長さが同じものだけが過去レースとして集計される
        """
        baselog = (
            self.population.merge(
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
        for n_race in tqdm(n_races, desc="agg_horse_per_course_len"):
            df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [
                        # "rank",
                        # "rank_per_horse",
                        # "prize",
                        # "rank_diff",
                        "time",
                        # "time_courselen",
                        "nobori",
                        # # "corner_1",
                        # # "corner_2",
                        # # "corner_3",
                        # # "corner_4",
                        # "corner_1_per_horse",
                        # "corner_2_per_horse",
                        # "corner_3_per_horse",
                        # "corner_4_per_horse",                        
                        # "pace_1",
                        # "pace_2",                        
                        # "win",
                        # "show",
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
        n_races: list[int] = [1, 3, 5, 10],
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
                        # "rank",
                        # "rank_per_horse",
                        # "prize",
                        # "rank_diff",
                        "time",

                        "nobori",
                        # "corner_1",
                        # "corner_2",
                        # "corner_3",
                        "corner_1_per_horse",
                        "corner_2_per_horse",
                        "corner_3_per_horse",
                        "corner_4_per_horse",               
                        
                        # "pace_1",
                        # "pace_2",                        
                        # "win",
                        # "show",
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

    # def agg_jockey(self):
    #     """
    #     騎手の過去成績を紐付け、相対値に変換する関数。
    #     """
    #     print("running agg_jockey()...")
    #     df = self.population.merge(
    #         self.results[["race_id", "horse_id", "jockey_id"]],
    #         on=["race_id", "horse_id"],
    #     )
    #     df["year"] = pd.to_datetime(df["date"]).dt.year - 1
    #     df = (
    #         df.merge(self.jockey_leading, on=["jockey_id", "year"], how="left")
    #         .drop(["date", "jockey_id", "year"], axis=1)
    #         .set_index(["race_id", "horse_id"])
    #         .add_prefix("jockey_")
    #     )
    #     # レースごとの相対値に変換
    #     tmp_df = df.groupby(["race_id"])
    #     relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")

    #     # relative_df = relative_df.astype({col: 'float32' for col in relative_df.select_dtypes('float64').columns})
    #     # relative_df = relative_df.astype({col: 'int32' for col in relative_df.select_dtypes('int64').columns})        
        
    #     self.agg_jockey_df = relative_df
    #     print("running agg_jockey()...comp")
        
    # def agg_trainer(self):
    #     """
    #     調教師の過去成績を紐付け、相対値に変換する関数。
    #     """
    #     print("running agg_trainer()...")
    #     df = self.population.merge(
    #         self.results[["race_id", "horse_id", "trainer_id"]],
    #         on=["race_id", "horse_id"],
    #     )
    #     df["year"] = pd.to_datetime(df["date"]).dt.year - 1
    #     df = (
    #         df.merge(self.trainer_leading, on=["trainer_id", "year"], how="left")
    #         .drop(["date", "trainer_id", "year"], axis=1)
    #         .set_index(["race_id", "horse_id"])
    #         .add_prefix("trainer_")
    #     )
    #     # レースごとの相対値に変換
    #     tmp_df = df.groupby(["race_id"])
    #     relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")

    #     # relative_df = relative_df.astype({col: 'float32' for col in relative_df.select_dtypes('float64').columns})
    #     # relative_df = relative_df.astype({col: 'int32' for col in relative_df.select_dtypes('int64').columns})        
        
    #     self.agg_trainer_df = relative_df
    #     print("running agg_trainer()...comp")

    # def agg_sire(self):
    #     """
    #     種牡馬の過去成績を紐付け、相対値に変換する関数。
    #     """
    #     print("running agg_sire()...")
    #     df = self.population.merge(
    #         self.peds[["horse_id", "sire_id"]],
    #         on="horse_id",
    #     ).merge(
    #         self.race_info[["race_id", "race_type", "course_len"]],
    #     )
    #     df["year"] = pd.to_datetime(df["date"]).dt.year - 1
    #     df = df.merge(
    #         self.sire_leading,
    #         on=["sire_id", "year", "race_type"],
    #         suffixes=("", "_sire"),
    #     ).set_index(["race_id", "horse_id"])
    #     df["course_len_diff"] = df["course_len"] - df["course_len_sire"]
    #     df = df[["n_races", "n_wins", "winrate", "course_len_diff"]].add_prefix("sire_")
    #     # レースごとの相対値に変換
    #     tmp_df = df.groupby(["race_id"])
    #     relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")

    #     # relative_df = relative_df.astype({col: 'float32' for col in relative_df.select_dtypes('float64').columns})
    #     # relative_df = relative_df.astype({col: 'int32' for col in relative_df.select_dtypes('int64').columns})        
        
    #     self.agg_sire_df = relative_df
    #     print("running agg_sire()...comp")

    # def agg_bms(self):
    #     """
    #     bmsの過去成績を紐付け、相対値に変換する関数。
    #     """
    #     print("running agg_bms()...")
    #     df = self.population.merge(
    #         self.peds[["horse_id", "bms_id"]],
    #         on="horse_id",
    #     ).merge(
    #         self.race_info[["race_id", "race_type", "course_len"]],
    #     )
    #     df["year"] = pd.to_datetime(df["date"]).dt.year - 1
    #     df = df.merge(
    #         self.bms_leading,
    #         on=["bms_id", "year", "race_type"],
    #         suffixes=("", "_bms"),
    #     ).set_index(["race_id", "horse_id"])
    #     df["course_len_diff"] = df["course_len"] - df["course_len_bms"]
    #     df = df[["n_races", "n_wins", "winrate", "course_len_diff"]].add_prefix("bms_")
    #     # レースごとの相対値に変換
    #     tmp_df = df.groupby(["race_id"])
    #     relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")

    #     # relative_df = relative_df.astype({col: 'float32' for col in relative_df.select_dtypes('float64').columns})
    #     # relative_df = relative_df.astype({col: 'int32' for col in relative_df.select_dtypes('int64').columns})        
        
    #     self.agg_bms_df = relative_df
    #     print("running agg_bms()...comp")
        
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

    def dirt_weight_weather(
        self, n_races: list[int] = [1, 3, 5, 10],
    ):  
        
        # merged_df = self.population.copy()        
        # df = (
        #     self.population.merge(
        #         self.race_info[["race_id",  "race_type","ground_state"]], on="race_id"
        #     )
        #     .merge(
        #         self.results[["race_id", "horse_id","weight"]],on=["race_id","horse_id"]
        #     )
        # )
        # if df["race_type"] == 0 and df["ground_state"] > 1:
        #     df["weight_weather"] = df["weight"] / 440
        # else:
        #     df["weight_weather"] = 0

        # required_columns = ["weight_weather"]


        # tmp_df_results_normal= df.groupby("race_id")[required_columns]
        # mean_results_normal = tmp_df_results_normal.transform("mean")
        # std_results_normal = tmp_df_results_normal.transform("std").replace(0, np.nan)  # 標準偏差が 0 の場合 NaN に置換
        # relative_cols = [f"{col}_relative" for col in required_columns]
        # df[relative_cols] = (df[required_columns] - mean_results_normal) / std_results_normal

        # merged_df = merged_df.merge(
        #     df[["race_id", "horse_id", "weight_weather_relative"]], 
        #     on=["race_id", "horse_id"], 
        #     how="left"
        # )
                
        # self.agg_weight_weather = merged_df

        merged_df = self.population.copy()

        # レース情報と結果データをマージ
        df = (
            self.population.merge(
                self.race_info[["race_id", "race_type", "ground_state"]], on="race_id"
            )
            .merge(
                self.results[["race_id", "horse_id", "weight"]], on=["race_id", "horse_id"]
            )
        )

        # 条件に基づいて weight_weather を計算
        df["weight_weather"] = df.apply(
            lambda row: row["weight"] / 440 if row["race_type"] == 0 and row["ground_state"] > 1 else np.nan, 
            axis=1
        )

        # 必要な列を選択
        required_columns = ["weight_weather"]

        # グループごとの平均と標準偏差を計算
        tmp_df_results_normal = df.groupby("race_id")[required_columns]
        mean_results_normal = tmp_df_results_normal.transform("mean")
        std_results_normal = tmp_df_results_normal.transform("std").replace(0, np.nan)  # 標準偏差が 0 の場合 NaN に置換

        # 相対値を計算
        relative_cols = [f"{col}_relative" for col in required_columns]
        df[relative_cols] = (df[required_columns] - mean_results_normal) / std_results_normal

        # 結果をマージ
        merged_df = merged_df.merge(
            df[["race_id", "horse_id"] + relative_cols], 
            on=["race_id", "horse_id"], 
            how="left"
        )

        # 最終結果を保存
        self.agg_weight_weather = merged_df

        # {
        #     "良": 0,
        #     "重": 1,
        #     "稍": 2,
        #     "不": 3,
        #     "稍重": 2,
        #     "不良": 3
        # }
        # {
        #     "ダ": 0,
        #     "芝": 1,
        #     "障": 2
        # }
        print("running agg_weight_weather()...comp")

        



    def position_results(
        self, n_races: list[int] = [1, 3, 5, 10],
    ):  
        
        merged_df2 = self.population.copy()        


        all_df = (
            self.all_population.merge(
                self.all_race_info[["race_id",  "ground_state","course_type"]], on="race_id"
            )
            .merge(
                self.all_results[["race_id", "horse_id","wakuban","rank", "umaban","corner_1","corner_2","corner_3","corner_4",'n_horses']],on=["race_id","horse_id"]
            )
        )

            #脚質
        def determine_race_position(row):
            # 最終コーナーを決定（corner_1, 2, 3, 4 のいずれか）
            if pd.notna(row['corner_4']):
                final_corner = 'corner_4'
            elif pd.notna(row['corner_3']):
                final_corner = 'corner_3'
            elif pd.notna(row['corner_2']):
                final_corner = 'corner_2'
            elif pd.notna(row['corner_1']):
                final_corner = 'corner_1'
            else:
                return None  # すべて欠損値の場合
        
            # 逃げ判定：最終コーナー以外のコーナーを1位で通過した場合
            if (pd.notna(row['corner_1']) and row['corner_1'] == 1 or 
                pd.notna(row['corner_2']) and row['corner_2'] == 1 or 
                pd.notna(row['corner_3']) and row['corner_3'] == 1 or 
                pd.notna(row['corner_4']) and row['corner_4'] == 1) and pd.notna(row[final_corner]) and row[final_corner] != 1:
                return 1  # 逃げ
        
            # 先行判定：最終コーナーを4位以内で通過
            if pd.notna(row[final_corner]) and row[final_corner] <= 5:
                return 2  # 先行
        
            # 差し判定：出走頭数に応じて
            if row['n_horses'] >= 8 and pd.notna(row[final_corner]) and row[final_corner] <= (row['n_horses'] * 2) // 3:
                return 3  # 差し
            elif row['n_horses'] < 8:
                return 4  # 差しなし、追込
        
            # 追込判定：上記のいずれにも該当しない
            return 4  # 追込
        

        # all_dfの各レースについて脚質を決定
        all_df['race_position'] = all_df.apply(determine_race_position, axis=1)
        all_df = all_df.where(pd.notnull(all_df), np.nan)


        all_df["course_type_ground_position"] = (all_df["course_type"].astype(str)+ all_df["ground_state"].astype(str) + all_df['race_position'].astype(str)).astype(int)
        



        """
        過去nレースにおける脚質割合を計算し、当該レースのペースを脚質の割合から予想する
        """
        def calculate_race_position_percentage(group, n_race: int):
            """
            過去nレースにおける脚質割合を計算する
            """
            # 過去nレースに絞る
            past_races = group.head(n_race)
            
            # 各脚質のカウント
            counts = past_races['race_position'].value_counts(normalize=True).to_dict()
            
            # 結果を辞書形式で返す（割合が存在しない場合は0.0を補完）
            return {
                "escape": counts.get(1, 0.0),
                "taking_lead": counts.get(2, 0.0),
                "in_front": counts.get(3, 0.0),
                "pursuit": counts.get(4, 0.0),
            }
    
        # 過去nレースのリスト
        n_race_list = [1, 3, 5, 10]
        baselog = (
            self.population.merge(
                self.horse_results,
                on=["horse_id"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        
        # 集計用データフレームの初期化
        merged_df = self.population.copy()
        
        # grouped_dfを適用して計算
        grouped_df = baselog.groupby(["race_id", "horse_id"])
        
        # 各過去nレースの割合を計算して追加
        for n_race in n_race_list:
            
            position_percentage = grouped_df.apply(
                lambda group: calculate_race_position_percentage(group, n_race=n_race),
                include_groups=False,  # グループ列を除外
            )
            
            # 結果をデータフレームとして展開
            position_percentage_df = position_percentage.apply(pd.Series).reset_index()
            position_percentage_df.rename(
                columns={
                    "escape": f"escape_{n_race}races",
                    "taking_lead": f"taking_lead_{n_race}races",
                    "in_front": f"in_front_{n_race}races",
                    "pursuit": f"pursuit_{n_race}races",
                },
                inplace=True,
            )
            
            # 結果をマージ
            merged_df = merged_df.merge(position_percentage_df, on=["race_id", "horse_id"], how="left")


        # dominant_position_category を計算
        position_columns_5 = ["escape_5races", "taking_lead_5races", "in_front_5races", "pursuit_5races"]
        position_columns_3 = ["escape_3races", "taking_lead_3races", "in_front_3races", "pursuit_3races"]
        position_columns_1 = ["escape_1races", "taking_lead_1races", "in_front_1races", "pursuit_1races"]
        
        
        def determine_dominant_position(row):
            
            # 5racesの中から計算
            if not row[position_columns_5].isnull().all():
                max_column = row[position_columns_5].idxmax()
                return {
                    "escape_5races": 1,
                    "taking_lead_5races": 2,
                    "in_front_5races": 3,
                    "pursuit_5races": 4,
                }[max_column]
            # 3racesの中から計算
            elif not row[position_columns_3].isnull().all():
                max_column = row[position_columns_3].idxmax()
                return {
                    "escape_3races": 1,
                    "taking_lead_3races": 2,
                    "in_front_3races": 3,
                    "pursuit_3races": 4,
                }[max_column]
            # 1racesの中から計算
            elif not row[position_columns_1].isnull().all():
                max_column = row[position_columns_1].idxmax()
                return {
                    "escape_1races": 1,
                    "taking_lead_1races": 2,
                    "in_front_1races": 3,
                    "pursuit_1races": 4,
                }[max_column]
            # すべて欠損の場合は1を返す
            else:
                return 3
        
        # 各行に対して dominant_position_category を適用
        merged_df["dominant_position_category_s"] = merged_df.apply(determine_dominant_position, axis=1)
    
        df_x = (
            merged_df.merge(
                self.race_info[["race_id",  "ground_state","course_type"]], on="race_id"
            )
        )

        
        df_x["course_type_ground_position"] = (df_x["course_type"].astype(str)+ df_x["ground_state"].astype(str) + df_x["dominant_position_category_s"].astype(str)).astype(int)
        
        # ターゲットエンコーディングを計算（カテゴリごとの複勝率の平均）
        all_df["target_rank"] = (all_df["rank"] <= 3).astype(int)
        df2 = all_df[["course_type_ground_position", "target_rank"]].dropna().astype(int)
        
        # グループごとのカウントを作成
        group_counts = df2.groupby("course_type_ground_position").size()
        # 100未満のグループを除外
        valid_groups = group_counts[group_counts >= 5].index
        # 100以上のグループのみを使用して、平均複勝率を計算
        df2_filtered = df2[df2["course_type_ground_position"].isin(valid_groups)]
        # 平均複勝率を計算
        mean_fukusho_rate_position = df2_filtered.groupby("course_type_ground_position")["target_rank"].transform("mean")
        # 計算した平均複勝率を元のDataFrameに追加
        all_df["mean_fukusho_rate_position"] = mean_fukusho_rate_position
        
        all_df = all_df[["course_type_ground_position","mean_fukusho_rate_position"]]

        columns_to_copy = [
            "course_type_ground_position"
        ]

        # ループでコピー列を作成
        all_df = all_df.copy()

        for col in columns_to_copy:
            df_x.loc[:, f"{col}_copy"] = df_x[col]
            all_df.loc[:, f"{col}_copy"] = all_df[col]
        columns_to_merge = [
            ('course_type_ground_position_copy', "mean_fukusho_rate_position"),     
        ]
        df_x = df_x.copy()

        # 各ペアを順番に処理
        for original_col, encoded_col in columns_to_merge:
            df2_subset = all_df[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
            df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
            df_x = df_x.merge(df2_subset, on=original_col, how='left')  # dfにマージ
            
        merged_df2 = merged_df2.merge(
            df_x[[
                "race_id", "horse_id", "date", 
                'course_type_ground_position',"mean_fukusho_rate_position",
            ]],
            on=["race_id", "date", "horse_id"],
            how="left"
        )
        
        
        self.agg_position_results= merged_df2
        
        print("running agg_position_results()...comp")



    def cross_features(self):
        """
        枠番、レースタイプ、直線か右左かの勝率比較_交互作用特徴量
        """
        merged_df = self.population.copy()        
        df = (
            self.results[["race_id", "horse_id", "wakuban", "umaban", "sex"]]
            .merge(self.race_info[["race_id", "race_type", "around","weather","ground_state"]], on="race_id")
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

    def cross_features_2(self):
        """
        カテゴリ追加、過去成績なし
        """
        merged_df = self.population.copy()        
        df = (
            merged_df
            .merge(self.results[["race_id", "horse_id", "wakuban", "umaban", "sex"]], on=["race_id", "horse_id"])
            .merge(self.race_info[["race_id", "place","race_grade","around","weather","ground_state","course_len","race_type","course_type"]], on="race_id")
        )

        
        
        # 距離/タイプ
        df["distance_type"] = (df["course_len"].astype(str) + df["race_type"].astype(str)).astype(int)
        
        # 距離/競馬場/タイプ
        df["distance_place_type"] = (df["course_type"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク
        df["distance_place_type_race_grade"] = (df["course_type"].astype(str) + df["race_grade"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順
        df["distance_place_type_wakuban"] = (df["course_type"].astype(str) + df["wakuban"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番
        df["distance_place_type_umaban"] = (df["course_type"].astype(str) + df["umaban"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/レースランク
        df["distance_place_type_wakuban_race_grade"] = (df["course_type"].astype(str) + df["wakuban"].astype(str) + df["race_grade"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/レースランク
        df["distance_place_type_umaban_race_grade"] = (df["course_type"].astype(str)+ df["umaban"].astype(str) + df["race_grade"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/直線
        df["distance_place_type_wakuban_straight"] = (df["course_type"].astype(str) + df["wakuban"].astype(str) ).astype(int)
        # 距離/競馬場/タイプ/馬番/直線
        df["distance_place_type_umaban_straight"] = (df["course_type"].astype(str)+ df["umaban"].astype(str) ).astype(int)
        # 距離/競馬場/タイプ/枠順/馬場状態
        df["distance_place_type_wakuban_ground_state"] = (df["course_type"].astype(str) + df["wakuban"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/馬場状態
        df["distance_place_type_umaban_ground_state"] = (df["course_type"].astype(str) + df["umaban"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/馬場状態/直線
        df["distance_place_type_wakuban_ground_state_straight"] = (df["course_type"].astype(str) + df["wakuban"].astype(str) + df["ground_state"].astype(str) ).astype(int)
        # 距離/競馬場/タイプ/馬番/馬場状態/直線
        df["distance_place_type_umaban_ground_state_straight"] = (df["course_type"].astype(str) + df["umaban"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/タイプ/天気
        df["distance_type_weather"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/天気
        df["distance_place_type_weather"] = (df["course_type"].astype(str) + df["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/天気
        df["distance_place_type_race_grade_weather"] = (df["course_type"].astype(str)+ df["race_grade"].astype(str) + df["weather"].astype(str)).astype(int)
        # 距離/タイプ/馬場状態
        df["distance_type_ground_state"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬場状態
        df["distance_place_type_ground_state"] = (df["course_type"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/馬場状態
        df["distance_place_type_race_grade_ground_state"] = (df["course_type"].astype(str) + df["race_grade"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/タイプ/性別
        df["distance_type_sex"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/性別
        df["distance_place_type_sex"] = (df["course_type"].astype(str)  + df["sex"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/性別
        df["distance_place_type_race_grade_sex"] = (df["course_type"].astype(str) + df["race_grade"].astype(str) + df["sex"].astype(str)).astype(int)
        # 距離/タイプ/性別/天気
        df["distance_type_sex_weather"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str) + df["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/性別/天気
        df["distance_place_type_sex_weather"] = (df["course_type"].astype(str) + df["sex"].astype(str) + df["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/性別/天気
        df["distance_place_type_race_grade_sex_weather"] = (df["course_type"].astype(str) + df["race_grade"].astype(str) + df["sex"].astype(str) + df["weather"].astype(str)).astype(int)
        # 距離/タイプ/性別/馬場状態
        df["distance_type_sex_ground_state"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/性別/馬場状態
        df["distance_place_type_sex_ground_state"] = (df["course_type"].astype(str)  + df["sex"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/性別/馬場状態
        df["distance_place_type_race_grade_sex_ground_state"] = (df["course_type"].astype(str)  + df["race_grade"].astype(str) + df["sex"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/タイプ/直線
        df["distance_type_straight"] = (df["course_len"].astype(str) + df["race_type"].astype(str) ).astype(int)
        # 距離/競馬場/タイプ/直線
        df["distance_place_type_straight"] = (df["course_type"].astype(str)  ).astype(int)
        # 距離/競馬場/タイプ/レースランク/直線
        df["distance_place_type_race_grade_straight"] = (df["course_type"].astype(str) + df["race_grade"].astype(str) ).astype(int)
        # 距離/タイプ/直線/馬場状態
        df["distance_type_straight_ground_state"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["around"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/直線/馬場状態
        df["distance_place_type_straight_ground_state"] = (df["course_type"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/直線/馬場状態
        df["distance_place_type_race_grade_straight_ground_state"] = (df["course_type"].astype(str)  + df["race_grade"].astype(str) + df["ground_state"].astype(str)).astype(int)

        # 距離/競馬場/タイプ/馬番/レースランク/直線/天気/馬場状態
        df["distance_place_type_umaban_race_grade_around_weather_ground_state"] = (df["course_type"].astype(str) + df["umaban"].astype(str) + df["race_grade"].astype(str)+ df["ground_state"].astype(str)+ df["weather"].astype(str)).astype(int)        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df["distance_place_type_race_grade_around_weather_ground_state"] = (df["course_type"].astype(str)   + df["race_grade"].astype(str)+ df["ground_state"].astype(str)+ df["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬場状態/天気
        df["distance_place_type_ground_state_weather"] = (df["course_type"].astype(str)  + df["ground_state"].astype(str)+ df["weather"].astype(str)).astype(int)



                
        baselog = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type", "race_grade","ground_state","weather","place","around","course_type"]], on="race_id"
            )
            # .merge(
            #     self.horse_results,
            #     on=["horse_id", "course_len", "race_type"],
            #     suffixes=("", "_horse"),
            # )
            # .query("date_horse < date")
            # .sort_values("date_horse", ascending=False)
        )
             
        df_old = (
            baselog
            .merge(self.all_results[["race_id", "horse_id","nobori","time","wakuban", "umaban","rank", "sex"]], on=["race_id", "horse_id"])
        )
        df_old["nobori"] = df_old["nobori"].fillna(df_old["nobori"].mean())

        
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        df_old["around"] = df_old["around"].fillna(3).astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  

        
        # 距離/タイプ
        df_old["distance_type"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str)).astype(int)
        
        # 距離/競馬場/タイプ
        df_old["distance_place_type"] = (df_old["course_type"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク
        df_old["distance_place_type_race_grade"] = (df_old["course_type"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順
        df_old["distance_place_type_wakuban"] = (df_old["course_type"].astype(str)+ df_old["wakuban"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番
        df_old["distance_place_type_umaban"] = (df_old["course_type"].astype(str) + df_old["umaban"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/レースランク
        df_old["distance_place_type_wakuban_race_grade"] = (df_old["course_type"].astype(str)+ df_old["wakuban"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/レースランク
        df_old["distance_place_type_umaban_race_grade"] = (df_old["course_type"].astype(str) + df_old["umaban"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/直線
        df_old["distance_place_type_wakuban_straight"] = (df_old["course_type"].astype(str)+ df_old["wakuban"].astype(str) ).astype(int)
        # 距離/競馬場/タイプ/馬番/直線
        df_old["distance_place_type_umaban_straight"] = (df_old["course_type"].astype(str) + df_old["umaban"].astype(str) ).astype(int)
        # 距離/競馬場/タイプ/枠順/馬場状態
        df_old["distance_place_type_wakuban_ground_state"] = (df_old["course_type"].astype(str) + df_old["wakuban"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/馬場状態
        df_old["distance_place_type_umaban_ground_state"] = (df_old["course_type"].astype(str)+ df_old["umaban"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/馬場状態/直線
        df_old["distance_place_type_wakuban_ground_state_straight"] = (df_old["course_type"].astype(str) + df_old["wakuban"].astype(str) + df_old["ground_state"].astype(str) ).astype(int)
        # 距離/競馬場/タイプ/馬番/馬場状態/直線
        df_old["distance_place_type_umaban_ground_state_straight"] = (df_old["course_type"].astype(str)+ df_old["umaban"].astype(str) + df_old["ground_state"].astype(str) ).astype(int)
        # 距離/タイプ/天気
        df_old["distance_type_weather"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/天気
        df_old["distance_place_type_weather"] = (df_old["course_type"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/天気
        df_old["distance_place_type_race_grade_weather"] = (df_old["course_type"].astype(str)+ df_old["race_grade"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # 距離/タイプ/馬場状態
        df_old["distance_type_ground_state"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬場状態
        df_old["distance_place_type_ground_state"] = (df_old["course_type"].astype(str)+ df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/馬場状態
        df_old["distance_place_type_race_grade_ground_state"] = (df_old["course_type"].astype(str)+ df_old["race_grade"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/タイプ/性別
        df_old["distance_type_sex"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/性別
        df_old["distance_place_type_sex"] = (df_old["course_type"].astype(str)+ df_old["sex"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/性別
        df_old["distance_place_type_race_grade_sex"] = (df_old["course_type"].astype(str)+ df_old["race_grade"].astype(str) + df_old["sex"].astype(str)).astype(int)
        # 距離/タイプ/性別/天気
        df_old["distance_type_sex_weather"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/性別/天気
        df_old["distance_place_type_sex_weather"] = (df_old["course_type"].astype(str) + df_old["sex"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/性別/天気
        df_old["distance_place_type_race_grade_sex_weather"] = (df_old["course_type"].astype(str)+ df_old["race_grade"].astype(str) + df_old["sex"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # 距離/タイプ/性別/馬場状態
        df_old["distance_type_sex_ground_state"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/性別/馬場状態
        df_old["distance_place_type_sex_ground_state"] = (df_old["course_type"].astype(str)+ df_old["sex"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/性別/馬場状態
        df_old["distance_place_type_race_grade_sex_ground_state"] = (df_old["course_type"].astype(str) + df_old["race_grade"].astype(str) + df_old["sex"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/タイプ/直線
        df_old["distance_type_straight"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["around"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/直線
        df_old["distance_place_type_straight"] = (df_old["course_type"].astype(str) ).astype(int)
        # 距離/競馬場/タイプ/レースランク/直線
        df_old["distance_place_type_race_grade_straight"] = (df_old["course_type"].astype(str)+ df_old["race_grade"].astype(str)).astype(int)
        # 距離/タイプ/直線/馬場状態
        df_old["distance_type_straight_ground_state"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["around"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/直線/馬場状態
        df_old["distance_place_type_straight_ground_state"] = (df_old["course_type"].astype(str)+ df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/直線/馬場状態
        df_old["distance_place_type_race_grade_straight_ground_state"] = (df_old["course_type"].astype(str)+ df_old["race_grade"].astype(str)  + df_old["ground_state"].astype(str)).astype(int)
        

        # 距離/競馬場/タイプ/馬番/レースランク/直線/天気/馬場状態
        df_old["distance_place_type_umaban_race_grade_around_weather_ground_state"] = (df_old["course_type"].astype(str) + df_old["umaban"].astype(str) + df_old["race_grade"].astype(str)+  df_old["ground_state"].astype(str)+ df_old["weather"].astype(str)).astype(int)        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_race_grade_around_weather_ground_state"] = (df_old["course_type"].astype(str) + df_old["race_grade"].astype(str)+ df_old["ground_state"].astype(str)+ df_old["weather"].astype(str)).astype(int)    

        
        # 距離/競馬場/タイプ/馬場状態/天気
        df_old["distance_place_type_ground_state_weather"] = (df_old["course_type"].astype(str)+ df_old["ground_state"].astype(str)+ df_old["weather"].astype(str)).astype(int)



        df_old["course_type_ground_umaban"] = (df_old["course_type"].astype(str)+ df_old["ground_state"].astype(str) + df_old["umaban"].astype(str)).astype(int)
        df_old["course_type_ground_wakuban"] = (df_old["course_type"].astype(str)+ df_old["ground_state"].astype(str) + df_old["wakuban"].astype(str)).astype(int)
        df["course_type_ground_umaban"] = (df["course_type"].astype(str)+ df["ground_state"].astype(str) + df["umaban"].astype(str)).astype(int)
        df["course_type_ground_wakuban"] = (df["course_type"].astype(str)+ df["ground_state"].astype(str) + df["wakuban"].astype(str)).astype(int)
        # ターゲットエンコーディングを計算（カテゴリごとの複勝率の平均）
        df_old["target_rank"] = (df_old["rank"] <= 3).astype(int)
        df2 = df_old[["course_type_ground_umaban","course_type_ground_wakuban", "target_rank"]].dropna().astype(int)
        
        
        # グループごとのカウントを作成
        group_counts = df2.groupby("course_type_ground_umaban").size()
        # 100未満のグループを除外
        valid_groups = group_counts[group_counts >= 10].index
        # 100以上のグループのみを使用して、平均複勝率を計算
        df2_filtered = df2[df2["course_type_ground_umaban"].isin(valid_groups)]
        # 平均複勝率を計算
        mean_fukusho_rate = df2_filtered.groupby("course_type_ground_umaban")["target_rank"].transform("mean")
        # 計算した平均複勝率を元のDataFrameに追加
        df_old["mean_fukusho_rate_umaban"] = mean_fukusho_rate
        
        # グループごとのカウントを作成
        group_counts = df2.groupby("course_type_ground_wakuban").size()
        # 100未満のグループを除外
        valid_groups = group_counts[group_counts >= 10].index
        # 100以上のグループのみを使用して、平均複勝率を計算
        df2_filtered = df2[df2["course_type_ground_wakuban"].isin(valid_groups)]
        # 平均複勝率を計算
        mean_fukusho_rate = df2_filtered.groupby("course_type_ground_wakuban")["target_rank"].transform("mean")
        # 計算した平均複勝率を元のDataFrameに追加
        df_old["mean_fukusho_rate_wakuban"] = mean_fukusho_rate



        df_old = df_old.copy()   
        df_old = df_old[df_old['rank'].isin([1, 2, 3,4,5,6,7])] 
        df_old = df_old.copy()   
        # ターゲットエンコーディングを計算（カテゴリごとのtimeの平均）
        target_mean_1 = df_old.groupby("distance_type")["time"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_type_encoded"] = df_old["distance_type"].map(target_mean_1)
        
        
        # ターゲットエンコーディングを計算（カテゴリごとのtimeの平均）
        target_mean_1 = df_old.groupby("distance_place_type")["time"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_encoded"] = df_old["distance_place_type"].map(target_mean_1)
        
        # ターゲットエンコーディングを計算（カテゴリごとのtimeの平均）
        target_mean_1 = df_old.groupby("distance_place_type_race_grade")["time"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_race_grade_encoded"] = df_old["distance_place_type_race_grade"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/枠順
        target_mean_1 = df_old.groupby("distance_place_type_wakuban")["time"].mean()
        df_old["distance_place_type_wakuban_encoded"] = df_old["distance_place_type_wakuban"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/馬番
        target_mean_1 = df_old.groupby("distance_place_type_umaban")["time"].mean()
        df_old["distance_place_type_umaban_encoded"] = df_old["distance_place_type_umaban"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/枠順/レースランク
        target_mean_1 = df_old.groupby("distance_place_type_wakuban_race_grade")["time"].mean()
        df_old["distance_place_type_wakuban_race_grade_encoded"] = df_old["distance_place_type_wakuban_race_grade"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/馬番/レースランク
        target_mean_1 = df_old.groupby("distance_place_type_umaban_race_grade")["time"].mean()
        df_old["distance_place_type_umaban_race_grade_encoded"] = df_old["distance_place_type_umaban_race_grade"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/枠順/直線
        target_mean_1 = df_old.groupby("distance_place_type_wakuban_straight")["time"].mean()
        df_old["distance_place_type_wakuban_straight_encoded"] = df_old["distance_place_type_wakuban_straight"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/馬番/直線
        target_mean_1 = df_old.groupby("distance_place_type_umaban_straight")["time"].mean()
        df_old["distance_place_type_umaban_straight_encoded"] = df_old["distance_place_type_umaban_straight"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/枠順/馬場状態
        target_mean_1 = df_old.groupby("distance_place_type_wakuban_ground_state")["time"].mean()
        df_old["distance_place_type_wakuban_ground_state_encoded"] = df_old["distance_place_type_wakuban_ground_state"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/馬番/馬場状態
        target_mean_1 = df_old.groupby("distance_place_type_umaban_ground_state")["time"].mean()
        df_old["distance_place_type_umaban_ground_state_encoded"] = df_old["distance_place_type_umaban_ground_state"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/枠順/馬場状態/直線
        target_mean_1 = df_old.groupby("distance_place_type_wakuban_ground_state_straight")["time"].mean()
        df_old["distance_place_type_wakuban_ground_state_straight_encoded"] = df_old["distance_place_type_wakuban_ground_state_straight"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/馬番/馬場状態/直線
        target_mean_1 = df_old.groupby("distance_place_type_umaban_ground_state_straight")["time"].mean()
        df_old["distance_place_type_umaban_ground_state_straight_encoded"] = df_old["distance_place_type_umaban_ground_state_straight"].map(target_mean_1)
        
        # 距離/タイプ/天気
        target_mean_1 = df_old.groupby("distance_type_weather")["time"].mean()
        df_old["distance_type_weather_encoded"] = df_old["distance_type_weather"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/天気
        target_mean_1 = df_old.groupby("distance_place_type_weather")["time"].mean()
        df_old["distance_place_type_weather_encoded"] = df_old["distance_place_type_weather"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/レースランク/天気
        target_mean_1 = df_old.groupby("distance_place_type_race_grade_weather")["time"].mean()
        df_old["distance_place_type_race_grade_weather_encoded"] = df_old["distance_place_type_race_grade_weather"].map(target_mean_1)
        
        
        # 距離/タイプ/馬場状態
        target_mean_1 = df_old.groupby("distance_type_ground_state")["time"].mean()
        df_old["distance_type_ground_state_encoded"] = df_old["distance_type_ground_state"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/馬場状態
        target_mean_1 = df_old.groupby("distance_place_type_ground_state")["time"].mean()
        df_old["distance_place_type_ground_state_encoded"] = df_old["distance_place_type_ground_state"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/レースランク/馬場状態
        target_mean_1 = df_old.groupby("distance_place_type_race_grade_ground_state")["time"].mean()
        df_old["distance_place_type_race_grade_ground_state_encoded"] = df_old["distance_place_type_race_grade_ground_state"].map(target_mean_1)
        
        # 距離/タイプ/性別
        target_mean_1 = df_old.groupby("distance_type_sex")["time"].mean()
        df_old["distance_type_sex_encoded"] = df_old["distance_type_sex"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/性別
        target_mean_1 = df_old.groupby("distance_place_type_sex")["time"].mean()
        df_old["distance_place_type_sex_encoded"] = df_old["distance_place_type_sex"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/レースランク/性別
        target_mean_1 = df_old.groupby("distance_place_type_race_grade_sex")["time"].mean()
        df_old["distance_place_type_race_grade_sex_encoded"] = df_old["distance_place_type_race_grade_sex"].map(target_mean_1)
        
        # 距離/タイプ/性別/天気
        target_mean_1 = df_old.groupby("distance_type_sex_weather")["time"].mean()
        df_old["distance_type_sex_weather_encoded"] = df_old["distance_type_sex_weather"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/性別/天気
        target_mean_1 = df_old.groupby("distance_place_type_sex_weather")["time"].mean()
        df_old["distance_place_type_sex_weather_encoded"] = df_old["distance_place_type_sex_weather"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/レースランク/性別/天気
        target_mean_1 = df_old.groupby("distance_place_type_race_grade_sex_weather")["time"].mean()
        df_old["distance_place_type_race_grade_sex_weather_encoded"] = df_old["distance_place_type_race_grade_sex_weather"].map(target_mean_1)
        
        # 距離/タイプ/性別/馬場状態
        target_mean_1 = df_old.groupby("distance_type_sex_ground_state")["time"].mean()
        df_old["distance_type_sex_ground_state_encoded"] = df_old["distance_type_sex_ground_state"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/性別/馬場状態
        target_mean_1 = df_old.groupby("distance_place_type_sex_ground_state")["time"].mean()
        df_old["distance_place_type_sex_ground_state_encoded"] = df_old["distance_place_type_sex_ground_state"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/レースランク/性別/馬場状態
        target_mean_1 = df_old.groupby("distance_place_type_race_grade_sex_ground_state")["time"].mean()
        df_old["distance_place_type_race_grade_sex_ground_state_encoded"] = df_old["distance_place_type_race_grade_sex_ground_state"].map(target_mean_1)
        
        # 距離/タイプ/直線
        target_mean_1 = df_old.groupby("distance_type_straight")["time"].mean()
        df_old["distance_type_straight_encoded"] = df_old["distance_type_straight"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/直線
        target_mean_1 = df_old.groupby("distance_place_type_straight")["time"].mean()
        df_old["distance_place_type_straight_encoded"] = df_old["distance_place_type_straight"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/レースランク/直線
        target_mean_1 = df_old.groupby("distance_place_type_race_grade_straight")["time"].mean()
        df_old["distance_place_type_race_grade_straight_encoded"] = df_old["distance_place_type_race_grade_straight"].map(target_mean_1)
        
        # 距離/タイプ/直線/馬場状態
        target_mean_1 = df_old.groupby("distance_type_straight_ground_state")["time"].mean()
        df_old["distance_type_straight_ground_state_encoded"] = df_old["distance_type_straight_ground_state"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/直線/馬場状態
        target_mean_1 = df_old.groupby("distance_place_type_straight_ground_state")["time"].mean()
        df_old["distance_place_type_straight_ground_state_encoded"] = df_old["distance_place_type_straight_ground_state"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/レースランク/直線/馬場状態
        target_mean_1 = df_old.groupby("distance_place_type_race_grade_straight_ground_state")["time"].mean()
        df_old["distance_place_type_race_grade_straight_ground_state_encoded"] = df_old["distance_place_type_race_grade_straight_ground_state"].map(target_mean_1)


        
        target_mean_1 = df_old.groupby("distance_place_type_umaban_race_grade_around_weather_ground_state")["time"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_umaban_race_grade_around_weather_ground_state_encoded"] = df_old["distance_place_type_umaban_race_grade_around_weather_ground_state"].map(target_mean_1)

        target_mean_1 = df_old.groupby("distance_place_type_race_grade_around_weather_ground_state")["time"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_race_grade_around_weather_ground_state_encoded"] = df_old["distance_place_type_race_grade_around_weather_ground_state"].map(target_mean_1)
       
        target_mean_1 = df_old.groupby("distance_place_type_ground_state_weather")["time"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_ground_state_weather_encoded"] = df_old["distance_place_type_ground_state_weather"].map(target_mean_1)




        
        #noboriの平均
        target_mean_1 = df_old.groupby("distance_place_type_ground_state_weather")["nobori"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_ground_state_weather_nobori_encoded"] = df_old["distance_place_type_ground_state_weather"].map(target_mean_1)
        
        target_mean_1 = df_old.groupby("distance_place_type")["nobori"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_nobori_encoded"] = df_old["distance_place_type"].map(target_mean_1)
        

        #noboriの平均
        target_mean_1 = df_old.groupby("distance_place_type_umaban_race_grade_around_weather_ground_state")["nobori"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded"] = df_old["distance_place_type_umaban_race_grade_around_weather_ground_state"].map(target_mean_1)
        
        target_mean_1 = df_old.groupby("distance_place_type_race_grade")["nobori"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_race_grade_nobori_encoded"] = df_old["distance_place_type_race_grade"].map(target_mean_1)





        
        # 必要な列だけをdf2から取得
        df_old = df_old[["distance_type", "distance_place_type", 
                        "distance_place_type_race_grade", "distance_place_type_wakuban", "distance_place_type_umaban", 
                        "distance_place_type_wakuban_race_grade", "distance_place_type_umaban_race_grade", 
                        "distance_place_type_wakuban_straight", "distance_place_type_umaban_straight", 
                        "distance_place_type_wakuban_ground_state", "distance_place_type_umaban_ground_state", 
                        "distance_place_type_wakuban_ground_state_straight", "distance_place_type_umaban_ground_state_straight", 
                        "distance_type_weather", "distance_place_type_weather", "distance_place_type_race_grade_weather", 
                        "distance_type_ground_state", "distance_place_type_ground_state", "distance_place_type_race_grade_ground_state", 
                        "distance_type_sex", "distance_place_type_sex", "distance_place_type_race_grade_sex", 
                        "distance_type_sex_weather", "distance_place_type_sex_weather", "distance_place_type_race_grade_sex_weather", 
                        "distance_type_sex_ground_state", "distance_place_type_sex_ground_state", 
                        "distance_place_type_race_grade_sex_ground_state", "distance_type_straight", 
                        "distance_place_type_straight", "distance_place_type_race_grade_straight", 
                        "distance_type_straight_ground_state", "distance_place_type_straight_ground_state", "distance_place_type_race_grade_around_weather_ground_state",
                "distance_place_type_umaban_race_grade_around_weather_ground_state",
                        "distance_place_type_race_grade_straight_ground_state", 
                        "course_type_ground_umaban","course_type_ground_wakuban",

                         
                         
                         "distance_place_type_ground_state_weather", 


                         
                        'distance_type_encoded', 'distance_place_type_encoded', 'distance_place_type_race_grade_encoded', 'distance_place_type_wakuban_encoded', 'distance_place_type_umaban_encoded', 'distance_place_type_wakuban_race_grade_encoded', 'distance_place_type_umaban_race_grade_encoded', 'distance_place_type_wakuban_straight_encoded', 'distance_place_type_umaban_straight_encoded', 'distance_place_type_wakuban_ground_state_encoded', 'distance_place_type_umaban_ground_state_encoded', 'distance_place_type_wakuban_ground_state_straight_encoded', 'distance_place_type_umaban_ground_state_straight_encoded', 'distance_type_weather_encoded', 'distance_place_type_weather_encoded', 'distance_place_type_race_grade_weather_encoded', 'distance_type_ground_state_encoded', 'distance_place_type_ground_state_encoded', 'distance_place_type_race_grade_ground_state_encoded', 'distance_type_sex_encoded', 'distance_place_type_sex_encoded', 'distance_place_type_race_grade_sex_encoded', 'distance_type_sex_weather_encoded', 'distance_place_type_sex_weather_encoded', 'distance_place_type_race_grade_sex_weather_encoded', 'distance_type_sex_ground_state_encoded', 'distance_place_type_sex_ground_state_encoded', 'distance_place_type_race_grade_sex_ground_state_encoded', 'distance_type_straight_encoded', 'distance_place_type_straight_encoded', 'distance_place_type_race_grade_straight_encoded', 'distance_type_straight_ground_state_encoded', 'distance_place_type_straight_ground_state_encoded', 'distance_place_type_race_grade_straight_ground_state_encoded',"distance_place_type_race_grade_around_weather_ground_state_encoded","distance_place_type_umaban_race_grade_around_weather_ground_state_encoded","distance_place_type_ground_state_weather_encoded",
                         
                         "distance_place_type_ground_state_weather_nobori_encoded","distance_place_type_nobori_encoded","distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded","distance_place_type_race_grade_nobori_encoded",
                         
                         "mean_fukusho_rate_wakuban","mean_fukusho_rate_umaban"
                            ]]
        

        # コピーを作成したい列名をリストにまとめる
        columns_to_copy = [
            "distance_place_type",
            "distance_place_type_ground_state_weather",
            "distance_place_type_umaban_race_grade_around_weather_ground_state", 
            "distance_place_type_race_grade",
            "distance_place_type_umaban_ground_state_straight",
            "distance_place_type_wakuban_ground_state_straight",
            "course_type_ground_umaban",
            "course_type_ground_wakuban"
        ]
        
        # ループでコピー列を作成
        for col in columns_to_copy:
            df.loc[:, f"{col}_copy"] = df[col]
            df_old.loc[:, f"{col}_copy"] = df_old[col]

        
        columns_to_merge = [
            ("distance_type",'distance_type_encoded'),
            ("distance_place_type",'distance_place_type_encoded'),
            ("distance_place_type_race_grade",'distance_place_type_race_grade_encoded'),
            ("distance_place_type_wakuban", 'distance_place_type_wakuban_encoded'),
            ("distance_place_type_umaban", 'distance_place_type_umaban_encoded'),
            ("distance_place_type_wakuban_race_grade", 'distance_place_type_wakuban_race_grade_encoded'),
            ("distance_place_type_umaban_race_grade", 'distance_place_type_umaban_race_grade_encoded'),
            ("distance_place_type_wakuban_straight", 'distance_place_type_wakuban_straight_encoded'),
            ("distance_place_type_umaban_straight", 'distance_place_type_umaban_straight_encoded'),
            ("distance_place_type_wakuban_ground_state", 'distance_place_type_wakuban_ground_state_encoded'),
            ("distance_place_type_umaban_ground_state", 'distance_place_type_umaban_ground_state_encoded'),
            ("distance_place_type_wakuban_ground_state_straight", 'distance_place_type_wakuban_ground_state_straight_encoded'),
            ("distance_place_type_umaban_ground_state_straight", 'distance_place_type_umaban_ground_state_straight_encoded'),
            ("distance_type_weather", 'distance_type_weather_encoded'),
            ("distance_place_type_weather", 'distance_place_type_weather_encoded'),
            ("distance_place_type_race_grade_weather", 'distance_place_type_race_grade_weather_encoded'),
            ("distance_type_ground_state", 'distance_type_ground_state_encoded'),
            ("distance_place_type_ground_state", 'distance_place_type_ground_state_encoded'),
            ("distance_place_type_race_grade_ground_state", 'distance_place_type_race_grade_ground_state_encoded'),
            ("distance_type_sex", 'distance_type_sex_encoded'),
            ("distance_place_type_sex", 'distance_place_type_sex_encoded'),
            ("distance_place_type_race_grade_sex", 'distance_place_type_race_grade_sex_encoded'),
            ("distance_type_sex_weather", 'distance_type_sex_weather_encoded'),
            ("distance_place_type_sex_weather", 'distance_place_type_sex_weather_encoded'),
            ("distance_place_type_race_grade_sex_weather", 'distance_place_type_race_grade_sex_weather_encoded'),
            ("distance_type_sex_ground_state", 'distance_type_sex_ground_state_encoded'),
            ("distance_place_type_sex_ground_state", 'distance_place_type_sex_ground_state_encoded'),
            ("distance_place_type_race_grade_sex_ground_state", 'distance_place_type_race_grade_sex_ground_state_encoded'),
            ("distance_type_straight", 'distance_type_straight_encoded'),
            ("distance_place_type_straight", 'distance_place_type_straight_encoded'),
            ("distance_place_type_race_grade_straight", 'distance_place_type_race_grade_straight_encoded'),
            ("distance_type_straight_ground_state", 'distance_type_straight_ground_state_encoded'),
            ("distance_place_type_straight_ground_state", 'distance_place_type_straight_ground_state_encoded'),
            ("distance_place_type_race_grade_straight_ground_state", 'distance_place_type_race_grade_straight_ground_state_encoded'),
            
            ("distance_place_type_umaban_race_grade_around_weather_ground_state", "distance_place_type_umaban_race_grade_around_weather_ground_state_encoded"),
            ("distance_place_type_race_grade_around_weather_ground_state", "distance_place_type_race_grade_around_weather_ground_state_encoded"), 
            
            ("distance_place_type_ground_state_weather", "distance_place_type_ground_state_weather_encoded"), 

            
            ("distance_place_type_ground_state_weather_copy", "distance_place_type_ground_state_weather_nobori_encoded"), 
            ("distance_place_type_copy", "distance_place_type_nobori_encoded"), 
            ("distance_place_type_umaban_race_grade_around_weather_ground_state_copy", "distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded"), 
             
            ("distance_place_type_race_grade_copy", "distance_place_type_race_grade_nobori_encoded"), 

            ('course_type_ground_umaban_copy', "mean_fukusho_rate_umaban"), 
            ('course_type_ground_wakuban_copy', "mean_fukusho_rate_wakuban"),     
        ]
    
        # 各ペアを順番に処理
        for original_col, encoded_col in columns_to_merge:
            df2_subset = df_old[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
            df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
            df = df.merge(df2_subset, on=original_col, how='left')  # dfにマージ
    

        merged_df = merged_df.merge(
            df[[
                "race_id", "horse_id", "date", 
                # "distance_type", "distance_place_type", 
                # "distance_place_type_race_grade", "distance_place_type_wakuban", "distance_place_type_umaban", 
                # "distance_place_type_wakuban_race_grade", "distance_place_type_umaban_race_grade", 
                # "distance_place_type_wakuban_straight", "distance_place_type_umaban_straight", 
                # "distance_place_type_wakuban_ground_state", "distance_place_type_umaban_ground_state", 
                # "distance_place_type_wakuban_ground_state_straight", "distance_place_type_umaban_ground_state_straight", 
                # "distance_type_weather", "distance_place_type_weather", "distance_place_type_race_grade_weather", 
                # "distance_type_ground_state", "distance_place_type_ground_state", "distance_place_type_race_grade_ground_state", 
                # "distance_type_sex", "distance_place_type_sex", "distance_place_type_race_grade_sex", 
                # "distance_type_sex_weather", "distance_place_type_sex_weather", "distance_place_type_race_grade_sex_weather", 
                # "distance_type_sex_ground_state", "distance_place_type_sex_ground_state", 
                # "distance_place_type_race_grade_sex_ground_state", "distance_type_straight", 
                # "distance_place_type_straight", "distance_place_type_race_grade_straight", 
                # "distance_type_straight_ground_state", "distance_place_type_straight_ground_state", 
                # "distance_place_type_race_grade_straight_ground_state",
                # "distance_place_type_race_grade_around_weather_ground_state",
                # "distance_place_type_umaban_race_grade_around_weather_ground_state",
                # "distance_place_type_ground_state_weather",
                
                
                'distance_type_encoded', 'distance_place_type_encoded', 'distance_place_type_race_grade_encoded', 'distance_place_type_wakuban_encoded', 'distance_place_type_umaban_encoded', 'distance_place_type_wakuban_race_grade_encoded', 'distance_place_type_umaban_race_grade_encoded', 'distance_place_type_wakuban_straight_encoded', 'distance_place_type_umaban_straight_encoded', 'distance_place_type_wakuban_ground_state_encoded', 'distance_place_type_umaban_ground_state_encoded', 'distance_place_type_wakuban_ground_state_straight_encoded', 'distance_place_type_umaban_ground_state_straight_encoded', 'distance_type_weather_encoded', 'distance_place_type_weather_encoded', 'distance_place_type_race_grade_weather_encoded', 'distance_type_ground_state_encoded', 'distance_place_type_ground_state_encoded', 'distance_place_type_race_grade_ground_state_encoded', 'distance_type_sex_encoded', 'distance_place_type_sex_encoded', 'distance_place_type_race_grade_sex_encoded', 'distance_type_sex_weather_encoded', 'distance_place_type_sex_weather_encoded', 'distance_place_type_race_grade_sex_weather_encoded', 'distance_type_sex_ground_state_encoded', 'distance_place_type_sex_ground_state_encoded', 'distance_place_type_race_grade_sex_ground_state_encoded', 'distance_type_straight_encoded', 'distance_place_type_straight_encoded', 'distance_place_type_race_grade_straight_encoded', 'distance_type_straight_ground_state_encoded', 'distance_place_type_straight_ground_state_encoded', 'distance_place_type_race_grade_straight_ground_state_encoded',"distance_place_type_race_grade_around_weather_ground_state_encoded","distance_place_type_umaban_race_grade_around_weather_ground_state_encoded","distance_place_type_ground_state_weather_encoded",

                "distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded","distance_place_type_race_grade_nobori_encoded","distance_place_type_ground_state_weather_nobori_encoded","distance_place_type_nobori_encoded","mean_fukusho_rate_wakuban","mean_fukusho_rate_umaban"
            ]],
            on=["race_id", "date", "horse_id"],
            how="left"
        )
        
        
        self.agg_cross_features_df_2= merged_df
        print("running cross_features_2()...comp")






        
    def cross_features_3(
        self, n_races: list[int] = [1,  3, 5, 10]
    ):
        """
        カテゴリ・過去成績集計
        """

        baselog = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place"]], on="race_id"
            )
            # .merge(
            #     self.horse_results,
            #     on=["horse_id", "course_len", "race_type"],
            #     suffixes=("", "_horse"),
            # )
            # .query("date_horse < date")
            # .sort_values("date_horse", ascending=False)
        )
        merged_df = self.population.copy()
             
        df = (
            baselog
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "nobori","rank","time","umaban"]], on=["race_id", "horse_id"])
        )
        df["nobori"] = df["nobori"].fillna(df["nobori"].mean())

        df["place"] = df["place"].astype(int)
        df["race_grade"] = df["race_grade"].astype(int)
        df["ground_state"] = df["ground_state"].astype(int)
        df["weather"] = df["weather"].astype(int)  
        

         
        df_old = (
            self.population.merge(
                self.horse_results,
                on=["horse_id"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )



        df_old["nobori"] = df_old["nobori"].fillna(df_old["nobori"].mean())
        
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  
       
        
        
        # 距離/タイプ
        df["distance_type"] = (df["course_len"].astype(str) + df["race_type"].astype(str)).astype(int)
        
        # 距離/競馬場/タイプ
        df["distance_place_type"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク
        df["distance_place_type_race_grade"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["race_grade"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順
        df["distance_place_type_wakuban"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["wakuban"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番
        df["distance_place_type_umaban"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["umaban"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/レースランク
        df["distance_place_type_wakuban_race_grade"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["wakuban"].astype(str) + df["race_grade"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/レースランク
        df["distance_place_type_umaban_race_grade"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["umaban"].astype(str) + df["race_grade"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/枠順/直線
        # df["distance_place_type_wakuban_straight"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["wakuban"].astype(str) + df["around"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/馬番/直線
        # df["distance_place_type_umaban_straight"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["umaban"].astype(str) + df["around"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/馬場状態
        df["distance_place_type_wakuban_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["wakuban"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/馬場状態
        df["distance_place_type_umaban_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["umaban"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/枠順/馬場状態/直線
        # df["distance_place_type_wakuban_ground_state_straight"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["wakuban"].astype(str) + df["ground_state"].astype(str) + df["around"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/馬番/馬場状態/直線
        # df["distance_place_type_umaban_ground_state_straight"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["umaban"].astype(str) + df["ground_state"].astype(str) + df["around"].astype(str)).astype(int)
        # 距離/タイプ/天気
        df["distance_type_weather"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/天気
        df["distance_place_type_weather"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/天気
        df["distance_place_type_race_grade_weather"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["race_grade"].astype(str) + df["weather"].astype(str)).astype(int)
        # 距離/タイプ/馬場状態
        df["distance_type_ground_state"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬場状態
        df["distance_place_type_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/馬場状態
        df["distance_place_type_race_grade_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["race_grade"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # # 距離/タイプ/性別
        # df["distance_type_sex"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/性別
        # df["distance_place_type_sex"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/レースランク/性別
        # df["distance_place_type_race_grade_sex"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["race_grade"].astype(str) + df["sex"].astype(str)).astype(int)
        # # 距離/タイプ/性別/天気
        # df["distance_type_sex_weather"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str) + df["weather"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/性別/天気
        # df["distance_place_type_sex_weather"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str) + df["weather"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/レースランク/性別/天気
        # df["distance_place_type_race_grade_sex_weather"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["race_grade"].astype(str) + df["sex"].astype(str) + df["weather"].astype(str)).astype(int)
        # # 距離/タイプ/性別/馬場状態
        # df["distance_type_sex_ground_state"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/性別/馬場状態
        # df["distance_place_type_sex_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/レースランク/性別/馬場状態
        # df["distance_place_type_race_grade_sex_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["race_grade"].astype(str) + df["sex"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # # 距離/タイプ/直線
        # df["distance_type_straight"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["around"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/直線
        # df["distance_place_type_straight"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["around"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/レースランク/直線
        # df["distance_place_type_race_grade_straight"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["race_grade"].astype(str) + df["around"].astype(str)).astype(int)
        # # 距離/タイプ/直線/馬場状態
        # df["distance_type_straight_ground_state"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["around"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/直線/馬場状態
        # df["distance_place_type_straight_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["around"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/レースランク/直線/馬場状態
        # df["distance_place_type_race_grade_straight_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["race_grade"].astype(str) + df["around"].astype(str) + df["ground_state"].astype(str)).astype(int)
        

        # # 距離/競馬場/タイプ/馬番/レースランク/直線/天気/馬場状態
        # df["distance_place_type_umaban_race_grade_around_weather_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["umaban"].astype(str) + df["race_grade"].astype(str)+ df["around"].astype(str) + df["ground_state"].astype(str)+ df["weather"].astype(str)).astype(int)        
        # # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        # df["distance_place_type_race_grade_around_weather_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str)  + df["race_grade"].astype(str)+ df["around"].astype(str) + df["ground_state"].astype(str)+ df["weather"].astype(str)).astype(int)     
        
        # 距離/競馬場/タイプ/馬場状態/天気
        df["distance_place_type_ground_state_weather"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["ground_state"].astype(str)+ df["weather"].astype(str)).astype(int)
        
        

        df = df.copy()   
        df = df[df['rank'].isin([1, 2, 3,4,5,6,7])]  
        df = df.copy()   
        # ターゲットエンコーディングを計算（カテゴリごとのtimeの平均）
        target_mean_1 = df.groupby("distance_type")["time"].mean()
        # 平均値をカテゴリ変数にマッピング
        df["distance_type_encoded"] = df["distance_type"].map(target_mean_1)
        
        
        # ターゲットエンコーディングを計算（カテゴリごとのtimeの平均）
        target_mean_1 = df.groupby("distance_place_type")["time"].mean()
        # 平均値をカテゴリ変数にマッピング
        df["distance_place_type_encoded"] = df["distance_place_type"].map(target_mean_1)
        
        # ターゲットエンコーディングを計算（カテゴリごとのtimeの平均）
        target_mean_1 = df.groupby("distance_place_type_race_grade")["time"].mean()
        # 平均値をカテゴリ変数にマッピング
        df["distance_place_type_race_grade_encoded"] = df["distance_place_type_race_grade"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/枠順
        target_mean_1 = df.groupby("distance_place_type_wakuban")["time"].mean()
        df["distance_place_type_wakuban_encoded"] = df["distance_place_type_wakuban"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/馬番
        target_mean_1 = df.groupby("distance_place_type_umaban")["time"].mean()
        df["distance_place_type_umaban_encoded"] = df["distance_place_type_umaban"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/枠順/レースランク
        target_mean_1 = df.groupby("distance_place_type_wakuban_race_grade")["time"].mean()
        df["distance_place_type_wakuban_race_grade_encoded"] = df["distance_place_type_wakuban_race_grade"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/馬番/レースランク
        target_mean_1 = df.groupby("distance_place_type_umaban_race_grade")["time"].mean()
        df["distance_place_type_umaban_race_grade_encoded"] = df["distance_place_type_umaban_race_grade"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/枠順/直線
        # target_mean_1 = df.groupby("distance_place_type_wakuban_straight")["time"].mean()
        # df["distance_place_type_wakuban_straight_encoded"] = df["distance_place_type_wakuban_straight"].map(target_mean_1)
        
        # # 距離/競馬場/タイプ/馬番/直線
        # target_mean_1 = df.groupby("distance_place_type_umaban_straight")["time"].mean()
        # df["distance_place_type_umaban_straight_encoded"] = df["distance_place_type_umaban_straight"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/枠順/馬場状態
        target_mean_1 = df.groupby("distance_place_type_wakuban_ground_state")["time"].mean()
        df["distance_place_type_wakuban_ground_state_encoded"] = df["distance_place_type_wakuban_ground_state"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/馬番/馬場状態
        target_mean_1 = df.groupby("distance_place_type_umaban_ground_state")["time"].mean()
        df["distance_place_type_umaban_ground_state_encoded"] = df["distance_place_type_umaban_ground_state"].map(target_mean_1)
        
        # # 距離/競馬場/タイプ/枠順/馬場状態/直線
        # target_mean_1 = df.groupby("distance_place_type_wakuban_ground_state_straight")["time"].mean()
        # df["distance_place_type_wakuban_ground_state_straight_encoded"] = df["distance_place_type_wakuban_ground_state_straight"].map(target_mean_1)
        
        # # 距離/競馬場/タイプ/馬番/馬場状態/直線
        # target_mean_1 = df.groupby("distance_place_type_umaban_ground_state_straight")["time"].mean()
        # df["distance_place_type_umaban_ground_state_straight_encoded"] = df["distance_place_type_umaban_ground_state_straight"].map(target_mean_1)
        
        # 距離/タイプ/天気
        target_mean_1 = df.groupby("distance_type_weather")["time"].mean()
        df["distance_type_weather_encoded"] = df["distance_type_weather"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/天気
        target_mean_1 = df.groupby("distance_place_type_weather")["time"].mean()
        df["distance_place_type_weather_encoded"] = df["distance_place_type_weather"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/レースランク/天気
        target_mean_1 = df.groupby("distance_place_type_race_grade_weather")["time"].mean()
        df["distance_place_type_race_grade_weather_encoded"] = df["distance_place_type_race_grade_weather"].map(target_mean_1)
        
        
        # 距離/タイプ/馬場状態
        target_mean_1 = df.groupby("distance_type_ground_state")["time"].mean()
        df["distance_type_ground_state_encoded"] = df["distance_type_ground_state"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/馬場状態
        target_mean_1 = df.groupby("distance_place_type_ground_state")["time"].mean()
        df["distance_place_type_ground_state_encoded"] = df["distance_place_type_ground_state"].map(target_mean_1)
        
        # 距離/競馬場/タイプ/レースランク/馬場状態
        target_mean_1 = df.groupby("distance_place_type_race_grade_ground_state")["time"].mean()
        df["distance_place_type_race_grade_ground_state_encoded"] = df["distance_place_type_race_grade_ground_state"].map(target_mean_1)
        
        # # 距離/タイプ/性別
        # target_mean_1 = df.groupby("distance_type_sex")["time"].mean()
        # df["distance_type_sex_encoded"] = df["distance_type_sex"].map(target_mean_1)
        
        # # 距離/競馬場/タイプ/性別
        # target_mean_1 = df.groupby("distance_place_type_sex")["time"].mean()
        # df["distance_place_type_sex_encoded"] = df["distance_place_type_sex"].map(target_mean_1)
        
        # # 距離/競馬場/タイプ/レースランク/性別
        # target_mean_1 = df.groupby("distance_place_type_race_grade_sex")["time"].mean()
        # df["distance_place_type_race_grade_sex_encoded"] = df["distance_place_type_race_grade_sex"].map(target_mean_1)
        
        # # 距離/タイプ/性別/天気
        # target_mean_1 = df.groupby("distance_type_sex_weather")["time"].mean()
        # df["distance_type_sex_weather_encoded"] = df["distance_type_sex_weather"].map(target_mean_1)
        
        # # 距離/競馬場/タイプ/性別/天気
        # target_mean_1 = df.groupby("distance_place_type_sex_weather")["time"].mean()
        # df["distance_place_type_sex_weather_encoded"] = df["distance_place_type_sex_weather"].map(target_mean_1)
        
        # # 距離/競馬場/タイプ/レースランク/性別/天気
        # target_mean_1 = df.groupby("distance_place_type_race_grade_sex_weather")["time"].mean()
        # df["distance_place_type_race_grade_sex_weather_encoded"] = df["distance_place_type_race_grade_sex_weather"].map(target_mean_1)
        
        # # 距離/タイプ/性別/馬場状態
        # target_mean_1 = df.groupby("distance_type_sex_ground_state")["time"].mean()
        # df["distance_type_sex_ground_state_encoded"] = df["distance_type_sex_ground_state"].map(target_mean_1)
        
        # # 距離/競馬場/タイプ/性別/馬場状態
        # target_mean_1 = df.groupby("distance_place_type_sex_ground_state")["time"].mean()
        # df["distance_place_type_sex_ground_state_encoded"] = df["distance_place_type_sex_ground_state"].map(target_mean_1)
        
        # # 距離/競馬場/タイプ/レースランク/性別/馬場状態
        # target_mean_1 = df.groupby("distance_place_type_race_grade_sex_ground_state")["time"].mean()
        # df["distance_place_type_race_grade_sex_ground_state_encoded"] = df["distance_place_type_race_grade_sex_ground_state"].map(target_mean_1)
        
        # # 距離/タイプ/直線
        # target_mean_1 = df.groupby("distance_type_straight")["time"].mean()
        # df["distance_type_straight_encoded"] = df["distance_type_straight"].map(target_mean_1)
        
        # # 距離/競馬場/タイプ/直線
        # target_mean_1 = df.groupby("distance_place_type_straight")["time"].mean()
        # df["distance_place_type_straight_encoded"] = df["distance_place_type_straight"].map(target_mean_1)
        
        # # 距離/競馬場/タイプ/レースランク/直線
        # target_mean_1 = df.groupby("distance_place_type_race_grade_straight")["time"].mean()
        # df["distance_place_type_race_grade_straight_encoded"] = df["distance_place_type_race_grade_straight"].map(target_mean_1)
        
        # # 距離/タイプ/直線/馬場状態
        # target_mean_1 = df.groupby("distance_type_straight_ground_state")["time"].mean()
        # df["distance_type_straight_ground_state_encoded"] = df["distance_type_straight_ground_state"].map(target_mean_1)
        
        # # 距離/競馬場/タイプ/直線/馬場状態
        # target_mean_1 = df.groupby("distance_place_type_straight_ground_state")["time"].mean()
        # df["distance_place_type_straight_ground_state_encoded"] = df["distance_place_type_straight_ground_state"].map(target_mean_1)
        
        # # 距離/競馬場/タイプ/レースランク/直線/馬場状態
        # target_mean_1 = df.groupby("distance_place_type_race_grade_straight_ground_state")["time"].mean()
        # df["distance_place_type_race_grade_straight_ground_state_encoded"] = df["distance_place_type_race_grade_straight_ground_state"].map(target_mean_1)

        
        # target_mean_1 = df.groupby("distance_place_type_umaban_race_grade_around_weather_ground_state")["time"].mean()
        # # 平均値をカテゴリ変数にマッピング
        # df["distance_place_type_umaban_race_grade_around_weather_ground_state_encoded"] = df["distance_place_type_umaban_race_grade_around_weather_ground_state"].map(target_mean_1)

        # target_mean_1 = df.groupby("distance_place_type_race_grade_around_weather_ground_state")["time"].mean()
        # # 平均値をカテゴリ変数にマッピング
        # df["distance_place_type_race_grade_around_weather_ground_state_encoded"] = df["distance_place_type_race_grade_around_weather_ground_state"].map(target_mean_1)


        target_mean_1 = df.groupby("distance_place_type_ground_state_weather")["time"].mean()
        # 平均値をカテゴリ変数にマッピング
        df["distance_place_type_ground_state_weather_encoded"] = df["distance_place_type_ground_state_weather"].map(target_mean_1)



        
        #noboriの平均
        target_mean_1 = df.groupby("distance_place_type_ground_state_weather")["nobori"].mean()
        # 平均値をカテゴリ変数にマッピング
        df["distance_place_type_ground_state_weather_nobori_encoded"] = df["distance_place_type_ground_state_weather"].map(target_mean_1)
        
        target_mean_1 = df.groupby("distance_place_type")["nobori"].mean()
        # 平均値をカテゴリ変数にマッピング
        df["distance_place_type_nobori_encoded"] = df["distance_place_type"].map(target_mean_1)
        

        # #noboriの平均
        # target_mean_1 = df.groupby("distance_place_type_umaban_race_grade_around_weather_ground_state")["nobori"].mean()
        # # 平均値をカテゴリ変数にマッピング
        # df["distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded"] = df["distance_place_type_umaban_race_grade_around_weather_ground_state"].map(target_mean_1)
        
        target_mean_1 = df.groupby("distance_place_type_race_grade")["nobori"].mean()
        # 平均値をカテゴリ変数にマッピング
        df["distance_place_type_race_grade_nobori_encoded"] = df["distance_place_type_race_grade"].map(target_mean_1)



        
        # 距離/タイプ
        df_old["distance_type"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str)).astype(int)
        
        # 距離/競馬場/タイプ
        df_old["distance_place_type"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク
        df_old["distance_place_type_race_grade"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順
        df_old["distance_place_type_wakuban"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["wakuban"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番
        df_old["distance_place_type_umaban"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["umaban"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/レースランク
        df_old["distance_place_type_wakuban_race_grade"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["wakuban"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/レースランク
        df_old["distance_place_type_umaban_race_grade"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["umaban"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/枠順/直線
        # df_old["distance_place_type_wakuban_straight"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["wakuban"].astype(str) + df_old["around"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/馬番/直線
        # df_old["distance_place_type_umaban_straight"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["umaban"].astype(str) + df_old["around"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/馬場状態
        df_old["distance_place_type_wakuban_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["wakuban"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/馬場状態
        df_old["distance_place_type_umaban_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["umaban"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/枠順/馬場状態/直線
        # df_old["distance_place_type_wakuban_ground_state_straight"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["wakuban"].astype(str) + df_old["ground_state"].astype(str) + df_old["around"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/馬番/馬場状態/直線
        # df_old["distance_place_type_umaban_ground_state_straight"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["umaban"].astype(str) + df_old["ground_state"].astype(str) + df_old["around"].astype(str)).astype(int)
        # 距離/タイプ/天気
        df_old["distance_type_weather"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/天気
        df_old["distance_place_type_weather"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/天気
        df_old["distance_place_type_race_grade_weather"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # 距離/タイプ/馬場状態
        df_old["distance_type_ground_state"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬場状態
        df_old["distance_place_type_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/馬場状態
        df_old["distance_place_type_race_grade_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # # 距離/タイプ/性別
        # df_old["distance_type_sex"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/性別
        # df_old["distance_place_type_sex"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/レースランク/性別
        # df_old["distance_place_type_race_grade_sex"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str) + df_old["sex"].astype(str)).astype(int)
        # # 距離/タイプ/性別/天気
        # df_old["distance_type_sex_weather"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/性別/天気
        # df_old["distance_place_type_sex_weather"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/レースランク/性別/天気
        # df_old["distance_place_type_race_grade_sex_weather"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str) + df_old["sex"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # # 距離/タイプ/性別/馬場状態
        # df_old["distance_type_sex_ground_state"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/性別/馬場状態
        # df_old["distance_place_type_sex_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/レースランク/性別/馬場状態
        # df_old["distance_place_type_race_grade_sex_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str) + df_old["sex"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # # 距離/タイプ/直線
        # df_old["distance_type_straight"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["around"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/直線
        # df_old["distance_place_type_straight"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["around"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/レースランク/直線
        # df_old["distance_place_type_race_grade_straight"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str) + df_old["around"].astype(str)).astype(int)
        # # 距離/タイプ/直線/馬場状態
        # df_old["distance_type_straight_ground_state"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["around"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/直線/馬場状態
        # df_old["distance_place_type_straight_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["around"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # # 距離/競馬場/タイプ/レースランク/直線/馬場状態
        # df_old["distance_place_type_race_grade_straight_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str) + df_old["around"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        

        # # 距離/競馬場/タイプ/馬番/レースランク/直線/天気/馬場状態
        # df_old["distance_place_type_umaban_race_grade_around_weather_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["umaban"].astype(str) + df_old["race_grade"].astype(str)+ df_old["around"].astype(str) + df_old["ground_state"].astype(str)+ df_old["weather"].astype(str)).astype(int)        
        # # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        # df_old["distance_place_type_race_grade_around_weather_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str)  + df_old["race_grade"].astype(str)+ df_old["around"].astype(str) + df_old["ground_state"].astype(str)+ df_old["weather"].astype(str)).astype(int)     
        
        # 距離/競馬場/タイプ/馬場状態/天気
        df_old["distance_place_type_ground_state_weather"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["ground_state"].astype(str)+ df_old["weather"].astype(str)).astype(int)
        

        # 必要な列だけをdf2から取得
        df = df[["distance_type", "distance_place_type", 
                        "distance_place_type_race_grade", "distance_place_type_wakuban", "distance_place_type_umaban", 
                        "distance_place_type_wakuban_race_grade", "distance_place_type_umaban_race_grade", 
                        # "distance_place_type_wakuban_straight", "distance_place_type_umaban_straight", 
                        "distance_place_type_wakuban_ground_state", "distance_place_type_umaban_ground_state", 
                        # "distance_place_type_wakuban_ground_state_straight", "distance_place_type_umaban_ground_state_straight", 
                        "distance_type_weather", "distance_place_type_weather", "distance_place_type_race_grade_weather", 
                        "distance_type_ground_state", "distance_place_type_ground_state", "distance_place_type_race_grade_ground_state", 
                        # "distance_type_sex", "distance_place_type_sex", "distance_place_type_race_grade_sex", 
                        # "distance_type_sex_weather", "distance_place_type_sex_weather", "distance_place_type_race_grade_sex_weather", 
                        # "distance_type_sex_ground_state", "distance_place_type_sex_ground_state", 
                        # "distance_place_type_race_grade_sex_ground_state", "distance_type_straight", 
                        # "distance_place_type_straight", "distance_place_type_race_grade_straight", 
                        # "distance_type_straight_ground_state", "distance_place_type_straight_ground_state", 
                #         "distance_place_type_race_grade_around_weather_ground_state",
                # "distance_place_type_umaban_race_grade_around_weather_ground_state",
                        # "distance_place_type_race_grade_straight_ground_state", 
                        "distance_place_type_ground_state_weather", 
                         


                         
                        'distance_type_encoded', 'distance_place_type_encoded', 
                        'distance_place_type_race_grade_encoded', 'distance_place_type_wakuban_encoded', 
                        'distance_place_type_umaban_encoded', 'distance_place_type_wakuban_race_grade_encoded', 
                        'distance_place_type_umaban_race_grade_encoded', 
                        # 'distance_place_type_wakuban_straight_encoded', 
                        # 'distance_place_type_umaban_straight_encoded', 
                        'distance_place_type_wakuban_ground_state_encoded', 
                        'distance_place_type_umaban_ground_state_encoded', 
                        # 'distance_place_type_wakuban_ground_state_straight_encoded', 
                        # 'distance_place_type_umaban_ground_state_straight_encoded', 
                        'distance_type_weather_encoded', 
                        'distance_place_type_weather_encoded', 'distance_place_type_race_grade_weather_encoded', 
                        'distance_type_ground_state_encoded', 'distance_place_type_ground_state_encoded', 
                        'distance_place_type_race_grade_ground_state_encoded', 
                        # 'distance_type_sex_encoded', 
                        # 'distance_place_type_sex_encoded', 'distance_place_type_race_grade_sex_encoded', 
                        # 'distance_type_sex_weather_encoded', 'distance_place_type_sex_weather_encoded', 
                        # 'distance_place_type_race_grade_sex_weather_encoded', 'distance_type_sex_ground_state_encoded', 
                        # 'distance_place_type_sex_ground_state_encoded', 'distance_place_type_race_grade_sex_ground_state_encoded', 
                        # 'distance_type_straight_encoded', 'distance_place_type_straight_encoded', 
                        # 'distance_place_type_race_grade_straight_encoded', 'distance_type_straight_ground_state_encoded', 
                        # 'distance_place_type_straight_ground_state_encoded', 'distance_place_type_race_grade_straight_ground_state_encoded',
                        # "distance_place_type_race_grade_around_weather_ground_state_encoded","distance_place_type_umaban_race_grade_around_weather_ground_state_encoded",
                        "distance_place_type_ground_state_weather_encoded",


                        #  "distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded",
                         "distance_place_type_race_grade_nobori_encoded",
                         "distance_place_type_ground_state_weather_nobori_encoded",
                         "distance_place_type_nobori_encoded"
                         
                            ]]

        columns_to_copy = [
            "distance_place_type",
            "distance_place_type_ground_state_weather",
            # "distance_place_type_umaban_race_grade_around_weather_ground_state", 
            "distance_place_type_race_grade",
            # "distance_place_type_umaban_ground_state_straight",
            # "distance_place_type_wakuban_ground_state_straight"
        ]
        df_old = df_old.copy()
        df = df.copy()
        # ループでコピー列を作成
        for col in columns_to_copy:
            df_old.loc[:, f"{col}_copy"] = df_old[col]
            df.loc[:, f"{col}_copy"] = df[col]


    
        columns_to_merge = [
            ("distance_type",'distance_type_encoded'),
            ("distance_place_type",'distance_place_type_encoded'),
            ("distance_place_type_race_grade",'distance_place_type_race_grade_encoded'),
            ("distance_place_type_wakuban", 'distance_place_type_wakuban_encoded'),
            ("distance_place_type_umaban", 'distance_place_type_umaban_encoded'),
            ("distance_place_type_wakuban_race_grade", 'distance_place_type_wakuban_race_grade_encoded'),
            ("distance_place_type_umaban_race_grade", 'distance_place_type_umaban_race_grade_encoded'),
            # ("distance_place_type_wakuban_straight", 'distance_place_type_wakuban_straight_encoded'),
            # ("distance_place_type_umaban_straight", 'distance_place_type_umaban_straight_encoded'),
            ("distance_place_type_wakuban_ground_state", 'distance_place_type_wakuban_ground_state_encoded'),
            ("distance_place_type_umaban_ground_state", 'distance_place_type_umaban_ground_state_encoded'),
            # ("distance_place_type_wakuban_ground_state_straight", 'distance_place_type_wakuban_ground_state_straight_encoded'),
            # ("distance_place_type_umaban_ground_state_straight", 'distance_place_type_umaban_ground_state_straight_encoded'),
            ("distance_type_weather", 'distance_type_weather_encoded'),
            ("distance_place_type_weather", 'distance_place_type_weather_encoded'),
            ("distance_place_type_race_grade_weather", 'distance_place_type_race_grade_weather_encoded'),
            ("distance_type_ground_state", 'distance_type_ground_state_encoded'),
            ("distance_place_type_ground_state", 'distance_place_type_ground_state_encoded'),
            ("distance_place_type_race_grade_ground_state", 'distance_place_type_race_grade_ground_state_encoded'),
            # ("distance_type_sex", 'distance_type_sex_encoded'),
            # ("distance_place_type_sex", 'distance_place_type_sex_encoded'),
            # ("distance_place_type_race_grade_sex", 'distance_place_type_race_grade_sex_encoded'),
            # ("distance_type_sex_weather", 'distance_type_sex_weather_encoded'),
            # ("distance_place_type_sex_weather", 'distance_place_type_sex_weather_encoded'),
            # ("distance_place_type_race_grade_sex_weather", 'distance_place_type_race_grade_sex_weather_encoded'),
            # ("distance_type_sex_ground_state", 'distance_type_sex_ground_state_encoded'),
            # ("distance_place_type_sex_ground_state", 'distance_place_type_sex_ground_state_encoded'),
            # ("distance_place_type_race_grade_sex_ground_state", 'distance_place_type_race_grade_sex_ground_state_encoded'),
            # ("distance_type_straight", 'distance_type_straight_encoded'),
            # ("distance_place_type_straight", 'distance_place_type_straight_encoded'),
            # ("distance_place_type_race_grade_straight", 'distance_place_type_race_grade_straight_encoded'),
            # ("distance_type_straight_ground_state", 'distance_type_straight_ground_state_encoded'),
            # ("distance_place_type_straight_ground_state", 'distance_place_type_straight_ground_state_encoded'),
            # ("distance_place_type_race_grade_straight_ground_state", 'distance_place_type_race_grade_straight_ground_state_encoded'),
            
            # ("distance_place_type_umaban_race_grade_around_weather_ground_state", "distance_place_type_umaban_race_grade_around_weather_ground_state_encoded"),
            # ("distance_place_type_race_grade_around_weather_ground_state", "distance_place_type_race_grade_around_weather_ground_state_encoded"), 
            
            ("distance_place_type_ground_state_weather", "distance_place_type_ground_state_weather_encoded"), 


            
            ("distance_place_type_ground_state_weather_copy", "distance_place_type_ground_state_weather_nobori_encoded"), 
            ("distance_place_type_copy", "distance_place_type_nobori_encoded"), 
            # ("distance_place_type_umaban_race_grade_around_weather_ground_state_copy","distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded"),
            ("distance_place_type_race_grade_copy", "distance_place_type_race_grade_nobori_encoded"),             
        ]

             # 各ペアを順番に処理
        for original_col, encoded_col in columns_to_merge:
            df2_subset = df[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
            df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
            df_old = df_old.merge(df2_subset, on=original_col, how='left')  # dfにマージ
    
   


        # 計算結果を格納する辞書
        new_columns = {}

        # 各ターゲットエンコーディング列に対して計算
        for col in df_old.columns:
            if "_nobori_encoded" in col:
                new_columns[f"{col}_time_diff"] = df_old[col] - df_old["nobori"]
                
                if "_grade" in col:
                
                    # # rank と平均タイムの和
                    # new_columns[f"{col}_rank_sumprod"] = df_old[col] + df_old["rank"]
                    
                    # # rank と平均タイムの積
                    # new_columns[f"{col}_rank_multiprod"] = df_old[col] * df_old["rank"]
                    
                    # rankdiff + 1 と平均タイムの和
                    new_columns[f"{col}_rank_diff_sumprod"] = df_old[col] + (df_old["rank_diff"] + 1)
                    
                    # rankdiff + 1 と平均タイムの積
                    new_columns[f"{col}_rank_diff_multiprod"] = df_old[col] * (df_old["rank_diff"] + 1)/2

                elif "_wakuban" in col:
                    
                    # 頭数/rankdiff + 1 と平均タイムの和
                    new_columns[f"{col}_rank_diff_sumprod"] = df_old[col] + (1/(df_old["rank_diff"] + 1))
                    
                    # 頭数/rankdiff + 1 と平均タイムの積
                    new_columns[f"{col}_rank_diff_multiprod"] = df_old[col] * (0.5/(df_old["rank_diff"] + 1))

                

                elif "_umaban" in col:
                    
                    # 頭数/rankdiff + 1 と平均タイムの和
                    new_columns[f"{col}_rank_diff_sumprod"] = df_old[col] + (1/(df_old["rank_diff"] + 1))
                    
                    # 頭数/rankdiff + 1 と平均タイムの積
                    new_columns[f"{col}_rank_diff_multiprod"] = df_old[col] * (0.5/(df_old["rank_diff"] + 1))
                
                
            elif "_encoded" in col:  # ターゲットエンコーディング列を対象
                # 平均タイムとの差
                new_columns[f"{col}_time_diff"] = df_old[col] - df_old["time"]
                
                
                if "_grade" in col:
                
                    # # rank と平均タイムの和
                    # new_columns[f"{col}_rank_sumprod"] = df_old[col] + df_old["rank"]
                    
                    # # rank と平均タイムの積
                    # new_columns[f"{col}_rank_multiprod"] = df_old[col] * df_old["rank"]
                    
                    # rankdiff + 1 と平均タイムの和
                    new_columns[f"{col}_rank_diff_sumprod"] = df_old[col] + (df_old["rank_diff"] + 1)
                    
                    # rankdiff + 1 と平均タイムの積
                    new_columns[f"{col}_rank_diff_multiprod"] = df_old[col] * (df_old["rank_diff"] + 1)/2

                elif "_wakuban" in col:
                    
                    # 頭数/rankdiff + 1 と平均タイムの和
                    new_columns[f"{col}_rank_diff_sumprod"] = df_old[col] + (1/(df_old["rank_diff"] + 1))
                    
                    # 頭数/rankdiff + 1 と平均タイムの積
                    new_columns[f"{col}_rank_diff_multiprod"] = df_old[col] * (0.5/(df_old["rank_diff"] + 1))

                

                elif "_umaban" in col:
                    
                    # 頭数/rankdiff + 1 と平均タイムの和
                    new_columns[f"{col}_rank_diff_sumprod"] = df_old[col] + (1/(df_old["rank_diff"] + 1))
                    
                    # 頭数/rankdiff + 1 と平均タイムの積
                    new_columns[f"{col}_rank_diff_multiprod"] = df_old[col] * (0.5/(df_old["rank_diff"] + 1))



        # 新しい列を一括で追加
        df_old = pd.concat([df_old, pd.DataFrame(new_columns)], axis=1)



        # df_oldから必要なカラムだけを選択
        df_old = df_old[["race_id", "horse_id", 'distance_type_encoded_time_diff', 
                'distance_place_type_encoded_time_diff', 
                'distance_place_type_race_grade_encoded_time_diff',
                'distance_place_type_race_grade_encoded_rank_diff_sumprod', 
                'distance_place_type_race_grade_encoded_rank_diff_multiprod', 
                'distance_place_type_wakuban_encoded_time_diff', 
                'distance_place_type_wakuban_encoded_rank_diff_sumprod', 
                'distance_place_type_wakuban_encoded_rank_diff_multiprod', 
                'distance_place_type_umaban_encoded_time_diff', 
                'distance_place_type_umaban_encoded_rank_diff_sumprod', 
                'distance_place_type_umaban_encoded_rank_diff_multiprod', 
                'distance_place_type_wakuban_race_grade_encoded_time_diff', 
                'distance_place_type_wakuban_race_grade_encoded_rank_diff_sumprod', 
                'distance_place_type_wakuban_race_grade_encoded_rank_diff_multiprod', 
                'distance_place_type_umaban_race_grade_encoded_time_diff', 
                'distance_place_type_umaban_race_grade_encoded_rank_diff_sumprod', 
                'distance_place_type_umaban_race_grade_encoded_rank_diff_multiprod', 
                # 'distance_place_type_wakuban_straight_encoded_time_diff', 
                # 'distance_place_type_wakuban_straight_encoded_rank_diff_sumprod', 
                # 'distance_place_type_wakuban_straight_encoded_rank_diff_multiprod', 
                # 'distance_place_type_umaban_straight_encoded_time_diff', 
                # 'distance_place_type_umaban_straight_encoded_rank_diff_sumprod', 
                # 'distance_place_type_umaban_straight_encoded_rank_diff_multiprod', 
                'distance_place_type_wakuban_ground_state_encoded_time_diff', 
                'distance_place_type_wakuban_ground_state_encoded_rank_diff_sumprod', 
                'distance_place_type_wakuban_ground_state_encoded_rank_diff_multiprod', 
                'distance_place_type_umaban_ground_state_encoded_time_diff', 
                'distance_place_type_umaban_ground_state_encoded_rank_diff_sumprod', 
                'distance_place_type_umaban_ground_state_encoded_rank_diff_multiprod', 
                # 'distance_place_type_wakuban_ground_state_straight_encoded_time_diff', 
                # 'distance_place_type_wakuban_ground_state_straight_encoded_rank_diff_sumprod', 
                # 'distance_place_type_wakuban_ground_state_straight_encoded_rank_diff_multiprod', 
                # 'distance_place_type_umaban_ground_state_straight_encoded_time_diff', 
                # 'distance_place_type_umaban_ground_state_straight_encoded_rank_diff_sumprod', 
                # 'distance_place_type_umaban_ground_state_straight_encoded_rank_diff_multiprod', 
                'distance_type_weather_encoded_time_diff', 
                'distance_place_type_weather_encoded_time_diff', 
                'distance_place_type_race_grade_weather_encoded_time_diff', 
                'distance_place_type_race_grade_weather_encoded_rank_diff_sumprod', 
                'distance_place_type_race_grade_weather_encoded_rank_diff_multiprod', 
                'distance_type_ground_state_encoded_time_diff',
                'distance_place_type_ground_state_encoded_time_diff', 
                'distance_place_type_race_grade_ground_state_encoded_time_diff', 
                'distance_place_type_race_grade_ground_state_encoded_rank_diff_sumprod', 
                'distance_place_type_race_grade_ground_state_encoded_rank_diff_multiprod', 
                # 'distance_type_sex_encoded_time_diff', 
                # 'distance_place_type_sex_encoded_time_diff', 
                # 'distance_place_type_race_grade_sex_encoded_time_diff', 
                # 'distance_place_type_race_grade_sex_encoded_rank_diff_sumprod', 
                # 'distance_place_type_race_grade_sex_encoded_rank_diff_multiprod', 
                # 'distance_type_sex_weather_encoded_time_diff', 
                # 'distance_place_type_sex_weather_encoded_time_diff', 
                # 'distance_place_type_race_grade_sex_weather_encoded_time_diff', 
                # 'distance_place_type_race_grade_sex_weather_encoded_rank_diff_sumprod', 
                # 'distance_place_type_race_grade_sex_weather_encoded_rank_diff_multiprod', 
                # 'distance_type_sex_ground_state_encoded_time_diff', 
                # 'distance_place_type_sex_ground_state_encoded_time_diff', 
                # 'distance_place_type_race_grade_sex_ground_state_encoded_time_diff', 
                # 'distance_place_type_race_grade_sex_ground_state_encoded_rank_diff_sumprod', 
                # 'distance_place_type_race_grade_sex_ground_state_encoded_rank_diff_multiprod', 
                # 'distance_type_straight_encoded_time_diff', 
                # 'distance_place_type_straight_encoded_time_diff', 
                # 'distance_place_type_race_grade_straight_encoded_time_diff', 
                # 'distance_place_type_race_grade_straight_encoded_rank_diff_sumprod', 
                # 'distance_place_type_race_grade_straight_encoded_rank_diff_multiprod', 
                # 'distance_type_straight_ground_state_encoded_time_diff', 
                # 'distance_place_type_straight_ground_state_encoded_time_diff', 
                # 'distance_place_type_race_grade_straight_ground_state_encoded_time_diff', 
                
                # 'distance_place_type_race_grade_straight_ground_state_encoded_rank_diff_sumprod', 
                # 'distance_place_type_race_grade_straight_ground_state_encoded_rank_diff_multiprod',
                
        #         "distance_place_type_umaban_race_grade_around_weather_ground_state_encoded_time_diff",
        # "distance_place_type_race_grade_around_weather_ground_state_encoded_time_diff",
        # "distance_place_type_umaban_race_grade_around_weather_ground_state_encoded_rank_diff_sumprod",
        # "distance_place_type_race_grade_around_weather_ground_state_encoded_rank_diff_sumprod",
        # "distance_place_type_umaban_race_grade_around_weather_ground_state_encoded_rank_diff_multiprod",
        # "distance_place_type_race_grade_around_weather_ground_state_encoded_rank_diff_multiprod",
                "distance_place_type_ground_state_weather_encoded_time_diff",
                
                
                #  "distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded_time_diff",
                "distance_place_type_race_grade_nobori_encoded_time_diff",
                "distance_place_type_ground_state_weather_nobori_encoded_time_diff",
                "distance_place_type_nobori_encoded_time_diff",
                
                #  "distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded_rank_diff_sumprod",
                
                "distance_place_type_race_grade_nobori_encoded_rank_diff_sumprod",
                #  "distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded_rank_diff_multiprod",
                
                "distance_place_type_race_grade_nobori_encoded_rank_diff_multiprod",
        ]]

        baselog = pd.merge(df_old, baselog, on=["race_id", "horse_id"], how="left")        

        grouped_df_old = baselog.groupby(["race_id", "horse_id"])




        for n_race in tqdm(n_races, desc="agg_cross_encoded"):
            df_old = (
                grouped_df_old.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [
                        'distance_type_encoded_time_diff', 
                        'distance_place_type_encoded_time_diff', 
                        'distance_place_type_race_grade_encoded_time_diff',
                        # 'distance_place_type_race_grade_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_race_grade_encoded_rank_diff_multiprod', 
                        'distance_place_type_wakuban_encoded_time_diff', 
                        # 'distance_place_type_wakuban_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_wakuban_encoded_rank_diff_multiprod', 
                        'distance_place_type_umaban_encoded_time_diff', 
                        # 'distance_place_type_umaban_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_umaban_encoded_rank_diff_multiprod', 
                        'distance_place_type_wakuban_race_grade_encoded_time_diff', 
                        # 'distance_place_type_wakuban_race_grade_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_wakuban_race_grade_encoded_rank_diff_multiprod', 
                        'distance_place_type_umaban_race_grade_encoded_time_diff', 
                        # 'distance_place_type_umaban_race_grade_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_umaban_race_grade_encoded_rank_diff_multiprod', 
                        # 'distance_place_type_wakuban_straight_encoded_time_diff', 
                        # 'distance_place_type_wakuban_straight_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_wakuban_straight_encoded_rank_diff_multiprod', 
                        # 'distance_place_type_umaban_straight_encoded_time_diff', 
                        # 'distance_place_type_umaban_straight_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_umaban_straight_encoded_rank_diff_multiprod', 
                        'distance_place_type_wakuban_ground_state_encoded_time_diff', 
                        # 'distance_place_type_wakuban_ground_state_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_wakuban_ground_state_encoded_rank_diff_multiprod', 
                        'distance_place_type_umaban_ground_state_encoded_time_diff', 
                        # 'distance_place_type_umaban_ground_state_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_umaban_ground_state_encoded_rank_diff_multiprod', 
                        # 'distance_place_type_wakuban_ground_state_straight_encoded_time_diff', 
                        # 'distance_place_type_wakuban_ground_state_straight_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_wakuban_ground_state_straight_encoded_rank_diff_multiprod', 
                        # 'distance_place_type_umaban_ground_state_straight_encoded_time_diff', 
                        # 'distance_place_type_umaban_ground_state_straight_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_umaban_ground_state_straight_encoded_rank_diff_multiprod', 
                        'distance_type_weather_encoded_time_diff', 
                        'distance_place_type_weather_encoded_time_diff', 
                        'distance_place_type_race_grade_weather_encoded_time_diff', 
                        # 'distance_place_type_race_grade_weather_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_race_grade_weather_encoded_rank_diff_multiprod', 
                        'distance_type_ground_state_encoded_time_diff',
                        'distance_place_type_ground_state_encoded_time_diff', 
                        'distance_place_type_race_grade_ground_state_encoded_time_diff', 
                        # 'distance_place_type_race_grade_ground_state_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_race_grade_ground_state_encoded_rank_diff_multiprod', 
                        # 'distance_type_sex_encoded_time_diff', 
                        # 'distance_place_type_sex_encoded_time_diff', 
                        # 'distance_place_type_race_grade_sex_encoded_time_diff', 
                        # 'distance_place_type_race_grade_sex_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_race_grade_sex_encoded_rank_diff_multiprod', 
                        # 'distance_type_sex_weather_encoded_time_diff', 
                        # 'distance_place_type_sex_weather_encoded_time_diff', 
                        # 'distance_place_type_race_grade_sex_weather_encoded_time_diff', 
                        # 'distance_place_type_race_grade_sex_weather_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_race_grade_sex_weather_encoded_rank_diff_multiprod', 
                        # 'distance_type_sex_ground_state_encoded_time_diff', 
                        # 'distance_place_type_sex_ground_state_encoded_time_diff', 
                        # 'distance_place_type_race_grade_sex_ground_state_encoded_time_diff', 
                        # 'distance_place_type_race_grade_sex_ground_state_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_race_grade_sex_ground_state_encoded_rank_diff_multiprod', 
                        # 'distance_type_straight_encoded_time_diff', 
                        # 'distance_place_type_straight_encoded_time_diff', 
                        # 'distance_place_type_race_grade_straight_encoded_time_diff', 
                        # 'distance_place_type_race_grade_straight_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_race_grade_straight_encoded_rank_diff_multiprod', 
                        # 'distance_type_straight_ground_state_encoded_time_diff', 
                        # 'distance_place_type_straight_ground_state_encoded_time_diff', 
                        # 'distance_place_type_race_grade_straight_ground_state_encoded_time_diff', 
                        # 'distance_place_type_race_grade_straight_ground_state_encoded_rank_diff_sumprod', 
                        # 'distance_place_type_race_grade_straight_ground_state_encoded_rank_diff_multiprod',

                        # "distance_place_type_umaban_race_grade_around_weather_ground_state_encoded_time_diff",
                        # "distance_place_type_race_grade_around_weather_ground_state_encoded_time_diff",
                        # "distance_place_type_umaban_race_grade_around_weather_ground_state_encoded_rank_diff_sumprod",
                        # "distance_place_type_race_grade_around_weather_ground_state_encoded_rank_diff_sumprod",
                        # "distance_place_type_umaban_race_grade_around_weather_ground_state_encoded_rank_diff_multiprod",
                        # "distance_place_type_race_grade_around_weather_ground_state_encoded_rank_diff_multiprod",
                        
                        "distance_place_type_ground_state_weather_encoded_time_diff",
                        # "distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded_time_diff",
                        "distance_place_type_race_grade_nobori_encoded_time_diff",
                        "distance_place_type_ground_state_weather_nobori_encoded_time_diff",
                        "distance_place_type_nobori_encoded_time_diff", 
                        # "distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded_rank_diff_sumprod",
                        # "distance_place_type_race_grade_nobori_encoded_rank_diff_sumprod",
                        # "distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded_rank_diff_multiprod",
                        # "distance_place_type_race_grade_nobori_encoded_rank_diff_multiprod"
                    ]
                ]
                .agg(["mean", "min","max"])
            )
            

            df_old.columns = [
                "_".join(col) + f"_{n_race}races_cross_encoded" for col in df_old.columns
            ]
            # オリジナルの統計値（mean, min）を保持
            original_df_old = df_old.copy()
            
            # レースごとの相対値に変換
            tmp_df_old = df_old.groupby(["race_id"])
            relative_df = ((df_old - tmp_df_old.mean()) / tmp_df_old.std()).add_suffix("_relative")
            merged_df = merged_df.merge(
                relative_df, on=["race_id", "horse_id"], how="left"
            )
            merged_df = merged_df.merge(original_df_old, on=["race_id", "horse_id"], how="left")

        self.agg_cross_features_df_3 = merged_df
        print("running cross_features_3()...comp")








    

        
    def cross_features_4(
        self, n_races: list[int] = [1,  3, 5,8]
    ):
        """
        # 瞬発力、持続力指標の作成
        """
        #過去情報の交互特徴量
        
        merged_df = self.population.copy()    
        df = (
            self.population.merge(
                self.horse_results,
                on=["horse_id"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        

        df["place"] = df["place"].astype(int)
        df["race_grade"] = df["race_grade"].astype(int)
        df["ground_state"] = df["ground_state"].astype(int)
        df["weather"] = df["weather"].astype(int)   
        
        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df["distance_place_type_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str)  + df["ground_state"].astype(str)).astype(int)   
        
        
        baselog_old = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place"]], on="race_id"
            )
        )
        
             
        df_old = (
            baselog_old
            .merge(self.all_results[["race_id", "horse_id","wakuban", "umaban","nobori","time"]], on=["race_id", "horse_id"])
        )
        df_old["nobori"] = df_old["nobori"].fillna(df_old["nobori"].mean())
        
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        # df_old["around"] = df_old["around"].fillna(3).astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  
        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str)  + df_old["ground_state"].astype(str)).astype(int)   
        
        #noboriの平均
        target_mean_1 = df_old.groupby("distance_place_type_ground_state")["nobori"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_ground_state_nobori_encoded"] = df_old["distance_place_type_ground_state"].map(target_mean_1)
        
        # #pace_1の平均
        # target_mean_1 = df_old.groupby("distance_place_type_race_grade_around_weather_ground_state")["pace_1"].mean()
        # # 平均値をカテゴリ変数にマッピング
        # df_old["distance_place_type_race_grade_around_weather_ground_state_pace_1_encoded"] = df_old["distance_place_type_race_grade_around_weather_ground_state"].map(target_mean_1)
        
        df_old = df_old[["distance_place_type_ground_state","distance_place_type_ground_state_nobori_encoded"]]
        
        columns_to_merge = [("distance_place_type_ground_state","distance_place_type_ground_state_nobori_encoded")]
        
        
        for original_col, encoded_col in columns_to_merge:
            df2_subset = df_old[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
            df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
            df = df.merge(df2_subset, on=original_col, how='left')  # dfにマージ
            
        """
        【タイプ】
        タイプの列を作る
        タイプの指標を作り、数字が上位なほどそのタイプの適性がある
        
        ランクディフが高ければ（数値が少なければ）、そのレースランク応じてポイントがもらえる
        レースランクが多ければ増える
        それを以下二つ（瞬発力列と持続力列）に配分する形にする
        
        ・瞬発力
        おもに一瞬の末脚、ギアチェンジが極めて強い（noboriの単純な少なさ）
        コーナー順位と最終順位の差から見極める（コーナーの平均と、最終順位の差）
        
        ・持続力
        おもに持続力のある末脚
        
        
        ＿コードの文章
        使用する列は
        rank_diff（着差）、race_grade(レースグレード),corner_1_horse,corner_2_horse,corner_3_horse,corner_4_horse(コーナー順位/頭数,_3と_4は欠損値もある),race_id(レースのid),horse_id（出走馬のid）,nobori(のぼり3ハロン),"distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded_time_diff"（出走レースのnoboriの平均値）,course_len(距離)
        列は全て作成してあるものとするrace_gradeのカテゴリ数値は以下のようになっており、グレードが上がるほど、数値が上がるものとする
        
            * 		新馬0
            * 		未勝利1
            * 		1勝クラス2
            * 		１勝クラス2
            * 		2勝クラス3
            * 		２勝クラス3
            * 		3勝クラス4
            * 		３勝クラス4
            * 		オープン5
            * 		G36
            * 		G27
            * 		G18
            * 		GIII6
            * 		GII7
            * 		GI8
            * 		重賞6
            * 		GⅢ6
            * 		GⅡ7
            * 		GⅠ8
            * 		L5
            * 		OP5
            * 		特別5
            * 		500万下2
            * 		1000万下3
            * 		1600万下4
        
        これらの列からrank_diffとrace_gradeの二つを使い、当該レースの当該馬のスコアを決定する
        なお、rank_diffは数が少なければ強く、race_gradeは数が増えるほど強いことに留意
        rank_diffには0が入っていることもあるため、+1をして利用すると良い
        
        
        ＿条件分け
        1_そのレースで、当該馬のコーナー順位/頭数の平均が0.4を超えている場合、持続力にスコアを割り振る
        2_それ以外の場合、瞬発力にスコアを割り振る
        
        1_1ただし、1_で(出走レースのnoboriの平均値 - nobori)が0.4より多い場合、瞬発力にも持続力と同じだけのスコアを割り振る
        1_2持続力スコアは距離によって補正を行う、2000を基準とし、それより短いほど-に補正値、長いほど+に補正値をかける、わずかでよい
        
        
        過去レースの最大値,平均を取ること
        """
        
        
        # rank_diffに+1してスコア計算用の列を作成
        df["adjusted_rank_diff"] = df["rank_diff"] + 1
        

        # # rank_diffとrace_gradeを利用した基本スコアを計算
        # df["base_score"] = 0.5 / df["adjusted_rank_diff"] *(df['race_grade_scaled'] )*1
        df["base_score"] = ((df["race_grade"]) * ( 1/((((df["rank_diff"] + 1)+10)/10)*(((df["course_len"]*0.0025)+20)/20)*(((df["pace_diff"] * 1)+20)/20))))
        
        

        # 条件に基づく変換
        df = df.copy()
        df.loc[:, "place_season_condition_type_categori_processed"] = df["place_season_condition_type_categori"].apply(
            lambda x: (x + 2 if x == '-' else (x - 3)) if isinstance(x, str) else x
        )
        
        # その後で1/1.7で割る
        df["place_season_condition_type_categori_processed_1"] = (df["place_season_condition_type_categori_processed"]+20) / 20

        # コーナー順位の平均を計算
        df["corner_avg_ratio"] = (
            df[["corner_1_per_horse", "corner_2_per_horse", "corner_3_per_horse", "corner_4_per_horse"]]
            .mean(axis=1)
        )
        
        # 出走レースのnoboriとの差分を計算
        df["nobori_diff"] = df["distance_place_type_ground_state_nobori_encoded"] - df["nobori"]
        # df["pace_diff"]  = df["distance_place_type_race_grade_around_weather_ground_state_pace_1_encoded"] - df["pace_1"]
        
        # スコア分割用の列を作成
        df["syunpatu"] = 0.0
        df["zizoku"] = 0.0
        
        # 条件1: 持続力にスコアを割り振る
        is_sustain_condition = df["corner_avg_ratio"] < 0.45
        df.loc[is_sustain_condition, "zizoku"] = df["base_score"] * 1/df["place_season_condition_type_categori_processed_1"]
        
        # 条件1_1: 瞬発力にもスコアを割り振る（nobori_diff > 0.3 の場合）
        df.loc[(df["nobori_diff"] > 0.2), "syunpatu"] = df["base_score"] * df["place_season_condition_type_categori_processed_1"]* ((df["nobori_diff"]+10)/10)

        # 条件2: 瞬発力にスコアを割り振る（それ以外の場合）※未割り当ての場合のみ
        df.loc[df["syunpatu"].isna(), "syunpatu"] = df["base_score"] * df["place_season_condition_type_categori_processed_1"] * ((df["nobori_diff"]+20)/20)
        
        # 条件2:持続力にスコアを割り振る（それ以外の場合）※未割り当ての場合のみ
        df.loc[~is_sustain_condition, "zizoku"] = df["base_score"]/(df["corner_avg_ratio"]*2.3) * 1/df["place_season_condition_type_categori_processed_1"]
        
        # 瞬発力に補正を加える
        nobori_correction_factor = np.maximum(df["nobori_diff"] - 0.2, 0) * 0.1  # 補正を半分に減らす
        df["syunpatu"] += nobori_correction_factor * df["base_score"] * df["place_season_condition_type_categori_processed_1"]
        
        # # pace_1列に基づく補正
        
        # pace_correction_factor = 0.1 - (df["pace_1"] - 36) / 3.6  # pace_1が36より小さいほど補正を増やし、増えるほど補正を減らす
        # df["pace_correction_factor"] = pace_correction_factor
        pace_correction_factor =((df["pace_diff"])+10) /10
        df["pace_correction_factor"] = pace_correction_factor
        #スローのとき  
        df["zizoku"] *= 1/pace_correction_factor # 調整係数0.5は任意で調整可能
        #ハイのとき      
        # 瞬発力 (syunpatu) に補正を加える
        df["syunpatu"] *= pace_correction_factor *1  # 同様に調整係数を加えます
        
        # # 持続力スコアを距離で補正
        # distance_base = 1600
        # distance_correction_factor = 1  # 距離補正の影響を抑えるスケーリング係数
        # df["distance_correction"] = (df["course_len"] - distance_base) / 1500  # 分母を5000に変更し補正を小さく
        # df["zizoku"] += df["distance_correction"]  * distance_correction_factor
        # df["syunpatu_minus"] = df["syunpatu"] * -1
   
        
        df["zizoku"] *= (((df["course_len"]*0.0025) + 50) / 50)
        df["syunpatu_minus"] = df["syunpatu"] * -1

        
        # dfから必要なカラムだけを選択
        df = df[[
            "race_id", "horse_id",
            "syunpatu",
            "syunpatu_minus",
            "zizoku",
            "corner_avg_ratio",
            "place_season_condition_type_categori_processed"
                ]]


        grouped_df = df.groupby(["race_id", "horse_id"])
        
        for n_race in tqdm(n_races, desc="agg_cross_zizoku_syunpatu"):
            df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [
                        "syunpatu",
                        "zizoku",
                        "syunpatu_minus",
                        "corner_avg_ratio"
                    ]
                ]
                .agg(["mean", "min","max"])
            )
            

            # 列名をフラットにする
            df.columns = [
                f"{col[0]}_{col[1]}_{n_race}races_encoded" for col in df.columns
            ]
            # オリジナルの統計値（mean, min）を保持
            original_df = df.copy()
            
            # レースごとの相対値に変換
            tmp_df = df.groupby(["race_id"])
            relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            merged_df = merged_df.merge(
                relative_df, on=["race_id", "horse_id"], how="left"
            )
            merged_df = merged_df.merge(original_df, on=["race_id", "horse_id"], how="left")
            
        merged_df = merged_df.merge(
            self.race_info[["race_id", 'place_season_condition_type_categori']], on="race_id"
        )
        merged_df = merged_df.copy()
        merged_df.loc[:, "place_season_condition_type_categori_processed"] = merged_df["place_season_condition_type_categori"].apply(
            lambda x: (x + 2 if x == '-' else (x - 3)) if isinstance(x, str) else x
        )
        merged_df["place_season_condition_type_categori_processed"] = merged_df["place_season_condition_type_categori_processed"] / 1.5


        # print(merged_df.columns.tolist())

        # merged_df["syunpatu_minus_mean_1races_encoded","syunpatu_minus_mean_3races_encoded","syunpatu_minus_mean_5races_encoded","syunpatu_minus_mean_8races_encoded",
        #         "syunpatu_minus_min_1races_encoded","syunpatu_minus_min_3races_encoded","syunpatu_minus_min_5races_encoded","syunpatu_minus_min_8races_encoded",
        #         "syunpatu_minus_max_1races_encoded","syunpatu_minus_max_3races_encoded","syunpatu_minus_max_5races_encoded","syunpatu_minus_max_8races_encoded"]
        
        
                # 24種類の列名
        syunpatu_columns = [
            "syunpatu_minus_mean_1races_encoded", "syunpatu_minus_mean_3races_encoded", "syunpatu_minus_mean_5races_encoded", "syunpatu_minus_mean_8races_encoded",
            "syunpatu_minus_min_1races_encoded", "syunpatu_minus_min_3races_encoded", "syunpatu_minus_min_5races_encoded", "syunpatu_minus_min_8races_encoded",
            "syunpatu_minus_max_1races_encoded", "syunpatu_minus_max_3races_encoded", "syunpatu_minus_max_5races_encoded", "syunpatu_minus_max_8races_encoded",
            
            # zizokuバージョンの列を追加
            "zizoku_mean_1races_encoded", "zizoku_mean_3races_encoded", "zizoku_mean_5races_encoded", "zizoku_mean_8races_encoded",
            "zizoku_min_1races_encoded", "zizoku_min_3races_encoded", "zizoku_min_5races_encoded", "zizoku_min_8races_encoded",
            "zizoku_max_1races_encoded", "zizoku_max_3races_encoded", "zizoku_max_5races_encoded", "zizoku_max_8races_encoded"
        ]
        
        
        # "place_season_condition_type_categori_processed"を掛け合わせた列を新たに作成
        for col in syunpatu_columns:
            new_col_name = f"{col}_multiplied"
            merged_df[new_col_name] = merged_df[col] * merged_df["place_season_condition_type_categori_processed"]
        
        # 掛け算後の列名リスト
        syunpatu_columns_multiplied = [f"{col}_multiplied" for col in syunpatu_columns]
        

        # 相対化処理を行い、merged_dfと一緒に結合
        tmp_df = merged_df.groupby(["race_id"])
        
        # 相対化されたデータを生成
        relative_df = ((merged_df[syunpatu_columns_multiplied] - tmp_df[syunpatu_columns_multiplied].transform("mean")) /
                       tmp_df[syunpatu_columns_multiplied].transform("std")).add_suffix("_relative")
        
        # インデックスを元のmerged_dfに合わせてリセット
        relative_df = relative_df.set_index(merged_df.index)
        
        # インデックスを元に戻した上で必要な列を追加
        relative_df['race_id'] = merged_df['race_id']
        relative_df['horse_id'] = merged_df['horse_id']
        
        # print(relative_df.info())
        # print(relative_df.head())
        # print("relative_df columns:", relative_df.columns.tolist())
        
        
        # merged_dfとrelative_dfをrace_idとhorse_idでマージ（必要なキーを指定）
        merged_df = merged_df.merge(relative_df, on=["race_id", "horse_id"], how="left")

        syunpatu_columns_plus = [
            "syunpatu_mean_1races_encoded", "syunpatu_mean_3races_encoded", "syunpatu_mean_5races_encoded", "syunpatu_mean_8races_encoded",
            "syunpatu_min_1races_encoded", "syunpatu_min_3races_encoded", "syunpatu_min_5races_encoded", "syunpatu_min_8races_encoded",
            "syunpatu_max_1races_encoded", "syunpatu_max_3races_encoded", "syunpatu_max_5races_encoded", "syunpatu_max_8races_encoded",
            
            # zizokuバージョンの列を追加
            "zizoku_mean_1races_encoded", "zizoku_mean_3races_encoded", "zizoku_mean_5races_encoded", "zizoku_mean_8races_encoded",
            "zizoku_min_1races_encoded", "zizoku_min_3races_encoded", "zizoku_min_5races_encoded", "zizoku_min_8races_encoded",
            "zizoku_max_1races_encoded", "zizoku_max_3races_encoded", "zizoku_max_5races_encoded", "zizoku_max_8races_encoded"
        ]
        # 必要な列を指定
        selected_columns = syunpatu_columns_plus + ['race_id', 'horse_id', 'date']

        # merged_dfから指定した列だけを抽出
        self.syunpatu_zizoku_df = merged_df[selected_columns]


        self.agg_cross_features_df_4 = merged_df
        print("running cross_features_4()...comp")








        
    # def cross_features_5(
    #     self, n_races: list[int] = [1,  3, 5,8]
    # ):
            
    #     """
    #     過去nレースにおける脚質割合を計算し、当該レースのペースを脚質の割合から予想する
    #     """
    #     def calculate_race_position_percentage(group, n_races: list[int] = [1,  3, 5,8]):
    #         """
    #         過去nレースにおける脚質割合を計算する
    #         """
    #         # 過去nレースに絞る
    #         past_races = group.head(n_race)
            
    #         # 各脚質のカウント
    #         counts = past_races['race_position'].value_counts(normalize=True).to_dict()
            
    #         # 結果を辞書形式で返す（割合が存在しない場合は0.0を補完）
    #         return {
    #             "escape": counts.get(1, 0.0),
    #             "taking_lead": counts.get(2, 0.0),
    #             "in_front": counts.get(3, 0.0),
    #             "pursuit": counts.get(4, 0.0),
    #         }
        
    #     # 過去nレースのリスト
    #     n_race_list = [1, 3, 5, 10]
    #     baselog = (
    #         self.population.merge(
    #             self.race_info[["race_id", "course_len", "race_type"]], on="race_id"
    #         )
    #         .merge(
    #             self.horse_results,
    #             on=["horse_id", "course_len", "race_type"],
    #             suffixes=("", "_horse"),
    #         )
    #         .query("date_horse < date")
    #         .sort_values("date_horse", ascending=False)
    #     )
        
    #     # 集計用データフレームの初期化
    #     merged_df = self.population.copy()
    #     # grouped_dfを適用して計算
    #     grouped_df = baselog.groupby(["race_id", "horse_id"])
    #     # 各過去nレースの割合を計算して追加
    #     for n_race in n_race_list:
            
    #         position_percentage = grouped_df.apply(
    #             lambda group: calculate_race_position_percentage(group, n_race=n_race),
    #             include_groups=False,  # グループ列を除外
    #         )
            
    #         # 結果をデータフレームとして展開
    #         position_percentage_df = position_percentage.apply(pd.Series).reset_index()
    #         position_percentage_df.rename(
    #             columns={
    #                 "escape": f"escape_{n_race}races",
    #                 "taking_lead": f"taking_lead_{n_race}races",
    #                 "in_front": f"in_front_{n_race}races",
    #                 "pursuit": f"pursuit_{n_race}races",
    #             },
    #             inplace=True,
    #         )
            
    #         # 結果をマージ
    #         merged_df = merged_df.merge(position_percentage_df, on=["race_id", "horse_id"], how="left")
    
    def cross_features_5(self, n_races: list[int] = [1,  3, 5, 8]):
        """
        過去nレースにおける脚質割合を計算し、当該レースのペースを脚質の割合から予想する
        """
        def calculate_race_position_percentage(group, n_race: int):
            """
            過去nレースにおける脚質割合を計算する
            """
            # 過去nレースに絞る
            past_races = group.head(n_race)
            
            # 各脚質のカウント
            counts = past_races['race_position'].value_counts(normalize=True).to_dict()
            
            # 結果を辞書形式で返す（割合が存在しない場合は0.0を補完）
            return {
                "escape": counts.get(1, 0.0),
                "taking_lead": counts.get(2, 0.0),
                "in_front": counts.get(3, 0.0),
                "pursuit": counts.get(4, 0.0),
            }
    
        # 過去nレースのリスト
        n_race_list = [1, 3, 5, 10]
        baselog = (
            self.population.merge(
                self.horse_results,
                on=["horse_id"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        
        # 集計用データフレームの初期化
        merged_df = self.population.copy()
        
        # grouped_dfを適用して計算
        grouped_df = baselog.groupby(["race_id", "horse_id"])
        
        # 各過去nレースの割合を計算して追加
        for n_race in n_race_list:
            
            position_percentage = grouped_df.apply(
                lambda group: calculate_race_position_percentage(group, n_race=n_race),
                include_groups=False,  # グループ列を除外
            )
            
            # 結果をデータフレームとして展開
            position_percentage_df = position_percentage.apply(pd.Series).reset_index()
            position_percentage_df.rename(
                columns={
                    "escape": f"escape_{n_race}races",
                    "taking_lead": f"taking_lead_{n_race}races",
                    "in_front": f"in_front_{n_race}races",
                    "pursuit": f"pursuit_{n_race}races",
                },
                inplace=True,
            )
            
            # 結果をマージ
            merged_df = merged_df.merge(position_percentage_df, on=["race_id", "horse_id"], how="left")


        # dominant_position_category を計算
        position_columns_5 = ["escape_5races", "taking_lead_5races", "in_front_5races", "pursuit_5races"]
        position_columns_3 = ["escape_3races", "taking_lead_3races", "in_front_3races", "pursuit_3races"]
        position_columns_1 = ["escape_1races", "taking_lead_1races", "in_front_1races", "pursuit_1races"]
        
        
        def determine_dominant_position(row):
            
            # 5racesの中から計算
            if not row[position_columns_5].isnull().all():
                max_column = row[position_columns_5].idxmax()
                return {
                    "escape_5races": 1,
                    "taking_lead_5races": 2,
                    "in_front_5races": 3,
                    "pursuit_5races": 4,
                }[max_column]
            # 3racesの中から計算
            elif not row[position_columns_3].isnull().all():
                max_column = row[position_columns_3].idxmax()
                return {
                    "escape_3races": 1,
                    "taking_lead_3races": 2,
                    "in_front_3races": 3,
                    "pursuit_3races": 4,
                }[max_column]
            # 1racesの中から計算
            elif not row[position_columns_1].isnull().all():
                max_column = row[position_columns_1].idxmax()
                return {
                    "escape_1races": 1,
                    "taking_lead_1races": 2,
                    "in_front_1races": 3,
                    "pursuit_1races": 4,
                }[max_column]
            # すべて欠損の場合は1を返す
            else:
                return 3
        
        # 各行に対して dominant_position_category を適用
        merged_df["dominant_position_category"] = merged_df.apply(determine_dominant_position, axis=1)
    
    
    
        
            #ペース作成
        """
        merged_df にはrace_id（レースのid）と、そのレースに出走するhorse_id(馬のid)の列があり、
        さらに dominant_position_categoryというその馬の脚質を表す列がある
        脚質は1-4の四種類の数字カテゴリが入っており
        レースごとに、出走する馬全体における、その脚質がいる割合と絶対数を求め、
        それぞれ四つの列を新たに作成
        """
        # dominant_position_category の脚質マッピング
        category_mapping_per = {1: "escape_per", 2: "taking_lead_per", 3: "in_front_per", 4: "pursuit_per"}
        category_mapping_count = {1: "escape_count", 2: "taking_lead_count", 3: "in_front_count", 4: "pursuit_count"}
        
        # 各レースで dominant_position_category の割合を計算
        position_ratios = (
            merged_df.groupby("race_id")["dominant_position_category"]
            .value_counts(normalize=True)  # 割合を計算
            .unstack(fill_value=0)         # 行: race_id, 列: dominant_position_category の形に
            .rename(columns=category_mapping_per)  # 列名を割合用に置き換え
        )
        
        # 各レースで dominant_position_category の絶対数を計算
        position_counts = (
            merged_df.groupby("race_id")["dominant_position_category"]
            .value_counts()                # 絶対数を計算
            .unstack(fill_value=0)         # 行: race_id, 列: dominant_position_category の形に
            .rename(columns=category_mapping_count)  # 列名を絶対数用に置き換え
        )
        
        # 割合と絶対数を結合
        position_data = position_ratios.join(position_counts)
        columns_to_remove = list(position_data.columns)
        merged_df = merged_df.drop(columns=columns_to_remove, errors="ignore")
        
        # 元の DataFrame にマージ
        merged_df = merged_df.merge(position_data, how="left", on="race_id")
    
        
        if 'pace_category' in merged_df.columns:
            merged_df = merged_df.drop(columns=['pace_category'])
        
        # ペースカテゴリを計算する関数
        def calculate_pace_category(row):
            escape_count = row.get("escape_count", 0)
            taking_lead_per = row.get("taking_lead_per", 0)
        
            if escape_count >= 3:
                return 4  # ハイペース
            elif escape_count == 2:
                if taking_lead_per > 0.5:
                    return 4  # ハイペース
                else:
                    return 3  
            elif escape_count == 1:
                if taking_lead_per > 0.6:
                    return 4  # ハイペース
                elif 0.5 < taking_lead_per <= 0.6:
                    return 3  # ハイミドルペース
                elif 0.4 < taking_lead_per <= 0.5:
                    return 2  # スローミドルペース
                else:
                    return 1  # スローペース
            elif escape_count == 0:
                # 逃げ馬が0の場合、先行馬の割合に基づく
                if taking_lead_per > 0.7:
                    return 4  # ハイペース
                elif 0.6 < taking_lead_per <= 0.7:
                    return 3  # ハイミドルペース
                elif 0.5 < taking_lead_per <= 0.6:
                    return 2  # スローミドルペース
                else:
                    return 1  # スローペース
            return None  # 予期しない場合は None
        merged_df["pace_category"] = merged_df.apply(calculate_pace_category, axis=1)
        # escape_per と escape_count が存在しない場合、列を作成して 0 を代入
        if "escape_per" not in merged_df.columns:
            merged_df["escape_per"] = 0
        if "escape_count" not in merged_df.columns:
            merged_df["escape_count"] = 0        
        merged_df = merged_df.drop(columns=["pursuit_per", "pursuit_count"], errors="ignore")
        # 除外対象の列を正規表現で特定
        columns_to_remove = [
            col for col in merged_df.columns if any(f"pursuit_{n}races" in col for n in n_race_list)
        ]
        
        # 対象の列を削除
        merged_df = merged_df.drop(columns=columns_to_remove, errors="ignore")
        self.agg_cross_features_df_5 = merged_df
        print("running cross_features_5()...comp")
    
    

    def cross_features_6(
        self, n_races: list[int] = [1,  3, 5,8]
    ):   
        #直近レースから、馬場状態調査、race_resultsで行う
        """
        1/競馬場xタイプxランクx距離ごとの平均タイムと直近のレースとの差異の平均を特徴量に、そのまま数値として入れれる
        2/芝以外を除去
        3/競馬場x芝タイプで集計(グループバイ)
        4/"race_date_day_count"の当該未満かつ、800以内が一週間以内のデータになるはず
        5/その平均を取る
        6/+なら軽く、-なら重く、それぞれ5段階のカテゴリに入れる
        """
        
        merged_df = self.population.copy()        
        df = (
            merged_df
            .merge(self.race_info[["race_id", "place","weather","ground_state","course_len","race_type","race_date_day_count","course_type"]], on="race_id")
        )
        df["place"] = df["place"].astype(int)
        
        
        
        #predictの時にはold_poplationに変える
        old_merged_df = self.population.copy()      
        
        #ここで、直近レースのtimeを知りたい
        df_old2 = (
            old_merged_df
            .merge(self.results[["race_id", "horse_id","time","rank","rank_per_horse"]], on=["race_id", "horse_id"])
            .merge(self.race_info[["race_id", "place","weather","course_type","ground_state","race_grade","course_len","race_type","race_date_day_count"]], on="race_id")
        )
        df_old2["place"] = df_old2["place"].astype(int)
        df_old2["race_grade"] = df_old2["race_grade"].astype(int)
        # 距離/競馬場/タイプ/レースランク
        df_old2["distance_place_type_race_grade"] = (df_old2["course_type"].astype(str) + df_old2["race_grade"].astype(str)).astype(int)
        
        
        baselog_old = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around","course_type"]], on="race_id"
            )
            # .merge(
            #     self.horse_results,
            #     on=["horse_id", "course_len", "race_type"],
            #     suffixes=("", "_horse"),
            # )
            # .query("date_horse < date")
            # .sort_values("date_horse", ascending=False)
        )
        df_old = (
            baselog_old
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "umaban","nobori","rank","time","sex"]], on=["race_id", "horse_id"])
        )
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        df_old["around"] = df_old["around"].fillna(3).astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  
                     
        
        df_old["distance_place_type_race_grade"] = (df_old["course_type"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
        df_old_copy = df_old
        # rank列が1, 2, 3の行だけを抽出
        df_old = df_old[df_old['rank'].isin([1, 2, 3])]
        target_mean_1 = df_old.groupby("distance_place_type_race_grade")["time"].mean()
        # スライスをコピーしてから処理
        df_old = df_old.copy()

        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_race_grade_encoded"] = df_old["distance_place_type_race_grade"].map(target_mean_1)
        
        
        df_old = df_old[["distance_place_type_race_grade",'distance_place_type_race_grade_encoded']]
        
        
        columns_to_merge = [
            ("distance_place_type_race_grade",'distance_place_type_race_grade_encoded'),
        ]
        
        # 各ペアを順番に処理
        for original_col, encoded_col in columns_to_merge:
            df2_subset = df_old[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
            df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
            df_old2 = df_old2.merge(df2_subset, on=original_col, how='left')  # dfにマージ
        df_old2 = df_old2[df_old2['rank'].isin([1, 2, 3])]
        df_old2["distance_place_type_race_grade_encoded_time_diff"] = df_old2['distance_place_type_race_grade_encoded'] - df_old2["time"]

        # df_old2= df_old2[df_old2["race_type"] != 2]
        # df_old2_1 = df_old2[df_old2["race_type"] != 0]
        df_old2_1 = df_old2
        # 2. df の各行について処理
        def compute_mean_for_row(row, df_old2_1):
            # race_type == 0 の場合は NaN を返す
            # if row["race_type"] == 0:
            #     return np.nan
                
            target_day_count = row["race_date_day_count"]  # df の各行の race_date_day_count
        
            # 3. df_old2_1 から条件に一致する行をフィルタリング
            filtered_df_old2_1 = df_old2_1[
                (df_old2_1["race_date_day_count"] >= (target_day_count - 400)) &  # race_date_day_count が target_day_count-1200 以上
                (df_old2_1["race_date_day_count"] <= (target_day_count - 1)) &  # race_date_day_count が target_day_count-1 以下
                (df_old2_1["place"] == row["place"]) &  # place が一致
                (df_old2_1["race_type"] == row["race_type"])  
                # (df_old2_1["weather"].isin([0, 1, 2])) &  # weather が 0, 1, 2 のいずれか
                # (df_old2_1["ground_state"] == 0)   # ground_state が 0
                # &
                # (df_old2_1["rank_per_horse"] < 0.87)  # rank_per_horse が 0.87 未満
            ]

            filtered_df_old2_2 = df_old2_1[
                (df_old2_1["race_date_day_count"] >= (target_day_count - 1200)) &  # race_date_day_count が target_day_count-1200 以上
                (df_old2_1["race_date_day_count"] <= (target_day_count - 1)) &  # race_date_day_count が target_day_count-1 以下
                (df_old2_1["place"] == row["place"]) &  # place が一致
                (df_old2_1["race_type"] == row["race_type"]) & 
                (df_old2_1["weather"].isin([0, 1, 2])) &  # weather が 0, 1, 2 のいずれか
                (df_old2_1["ground_state"] == 0)   # ground_state が 0                    
                # &
                # (df_old2_1["rank_per_horse"] < 0.87)  # rank_per_horse が 0.87 未満
            ]
            # 4. フィルタリングした行の "distance_place_type_race_grade_encoded_time_diff" の平均を計算
            mean_time_diff = filtered_df_old2_1["distance_place_type_race_grade_encoded_time_diff"].mean()

            # 5. 計算結果を返す（NaNの場合も考慮）
            # return mean_time_diff if not np.isnan(mean_time_diff) else filtered_df_old2_2["distance_place_type_race_grade_encoded_time_diff"].mean()
            return np.nan_to_num(mean_time_diff, nan=np.nan_to_num(filtered_df_old2_2["distance_place_type_race_grade_encoded_time_diff"].mean()))


        
            # # 4. フィルタリングした行の "distance_place_type_race_grade_encoded_time_diff" の平均を計算
            # mean_time_diff = filtered_df_old2_1["distance_place_type_race_grade_encoded_time_diff"].mean()
        
            # # 5. 計算結果を返す（NaNの場合も考慮）
            # return mean_time_diff if not np.isnan(mean_time_diff) else np.nan
        
        # 6. df の各行に対して、計算した平均値を新しい列に追加
        df["mean_ground_state_time_diff"] = df.apply(compute_mean_for_row, axis=1, df_old2_1=df_old2_1)

        
        
        def assign_value(row):
            # weather が 3, 4, 5 または ground_state が 0 以外の場合は 5 を設定
            if row["ground_state"] in [3]:
                return 0
            if row["weather"] in [3, 4, 5]:
                return 1

            if row["ground_state"] in [1]:
                return 1
            if row["ground_state"] in [2]:
                if row["race_type"] in [0]:
                    # mean_ground_state_time_diff に基づいて分類
                    if 2.0 <= row["mean_ground_state_time_diff"]:
                        return 3.7  #　超高速馬場1
                    elif 1 <= row["mean_ground_state_time_diff"] < 2.0:
                        return 4.5  # 高速馬場2
                    
                    elif 0.4 <= row["mean_ground_state_time_diff"] < 1:
                        return 5  # 軽い馬場3
                        
                    elif -0.4 <= row["mean_ground_state_time_diff"] < 0.4:
                        return 5.5  # 標準的な馬場4
                    elif -1 <= row["mean_ground_state_time_diff"] < -0.4:
                        return 6.2  # やや重い馬場5
                    
                    elif -2 <= row["mean_ground_state_time_diff"] < -1:
                        return 7  # 重い馬場5
                    
                    elif row["mean_ground_state_time_diff"] < -2:
                        return 8  # 道悪7
                if row["race_type"] in [1,2]:
                    return 2.7
            # mean_ground_state_time_diff に基づいて分類
            if 2.0 <= row["mean_ground_state_time_diff"]:
                return 2.7  #　超高速馬場1
            elif 1 <= row["mean_ground_state_time_diff"] < 2.0:
                return 3.5  # 高速馬場2
            
            elif 0.4 <= row["mean_ground_state_time_diff"] < 1:
                return 3.8   # 軽い馬場3
                
            elif -0.4 <= row["mean_ground_state_time_diff"] < 0.4:
                return 4  # 標準的な馬場4
        
            
            elif -1 <= row["mean_ground_state_time_diff"] < -0.4:
                return 4.5  # やや重い馬場5
            
            elif -2 <= row["mean_ground_state_time_diff"] < -1:
                return 5.2  # 重い馬場5
            
            elif row["mean_ground_state_time_diff"] < -2:
                return 6  # 道悪7
            # 該当しない場合は NaN を設定
            return 3.5
        
        
        # 新しい列を追加
        df["ground_state_level"] = df.apply(assign_value, axis=1)
        merged_df = merged_df.merge(df[["race_id","date","horse_id","mean_ground_state_time_diff","ground_state_level"]],on=["race_id","date","horse_id"])	

        self.agg_cross_features_df_6 = merged_df
        print("running cross_features_6()...comp")


    def cross_features_7(
        self, n_races: list[int] = [1,  3, 5,8]
    ):  
        """
        "pace_category"にある1234という数字カテゴリごとにwin,rentai,showを
        
        baselogの"pace_category"==が1,2,3,4ごとにそれぞれ分ける
        baselog1,2,3,4ごとに、win,rentai,showのn_racesごとの平均を算出
        最終的にbaselog1,2,3,4をconcatする
        
        脚質から未来のペースを求めたpace_categoryの数値と、同一のpace_categoryの過去データのみを参考にする
        
        
        「今回のレースと同じ展開での実績」として交互作用特徴量できる
        
        過去どんなレースを得意としているかがわかる
        """
        
        baselog = (
            self.population.merge(
                self.horse_results,
                on=["horse_id"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        grouped_df = baselog.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        
        """
        "pace_category"にある1234という数字カテゴリごとにwin,rentai,showを
        新たな列が必要
        
        baselogの"pace_category"==が1,2,3,4ごとにそれぞれ分ける
        baselog1,2,3,4ごとに、win,rentai,showのn_racesごとの平均を算出
        最終的にbaselog1,2,3,4をconcatする
        
        """
        # pace_category ごとにデータをフィルタリング
        baselog1 = baselog[baselog["pace_category"] == 1]
        baselog2 = baselog[baselog["pace_category"] == 2]
        baselog3 = baselog[baselog["pace_category"] == 3]
        baselog4 = baselog[baselog["pace_category"] == 4]
        n_races: list[int] = [1, 3, 5, 10]
        # 結果を格納するためのリスト
        result_dfs = []
        
        # 各pace_categoryに対してn_racesごとの平均を計算
        for pace_category, df1 in zip([1, 2, 3, 4], [baselog1, baselog2, baselog3, baselog4]):
            grouped_df = df1.groupby(["race_id", "horse_id"])
            
            for n_race in tqdm(n_races, desc=f"pace_category_win_{pace_category}"):
                # n_raceに基づいて集計
                agg_df = (
                    grouped_df.head(n_race)
                    .groupby(["race_id", "horse_id"])[["win", "rentai", "show"]]
                    .agg("mean")
                )
                
                # 列名を修正
                agg_df.columns = [
                    f"{col}_{n_race}races_per_pace_category{pace_category}" for col in agg_df.columns
                ]
                
                # # 平均と標準偏差を基に相対値を計算
                # tmp_df = agg_df.groupby(["race_id"])
                # relative_df = ((agg_df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
                
                # # 生成したデータフレームをresult_dfsに追加
                # result_dfs.append(relative_df)
                result_dfs.append(agg_df)
        
        
        # 結果をconcatして1つのデータフレームにまとめる
        final_agg_df = pd.concat(result_dfs, axis=1)
        
        # merge_dfとマージ
        merged_df = merged_df.merge(final_agg_df, on=["race_id", "horse_id"], how="left")
                
        
        """
        過去nレースにおける脚質割合を計算し、当該レースのペースを脚質の割合から予想する
        """
        def calculate_race_position_percentage(group, n_race: int):
            """
            過去nレースにおける脚質割合を計算する
            """
            # 過去nレースに絞る
            past_races = group.head(n_race)
            
            # 各脚質のカウント
            counts = past_races['race_position'].value_counts(normalize=True).to_dict()
            
            # 結果を辞書形式で返す（割合が存在しない場合は0.0を補完）
            return {
                "escape": counts.get(1, 0.0),
                "taking_lead": counts.get(2, 0.0),
                "in_front": counts.get(3, 0.0),
                "pursuit": counts.get(4, 0.0),
            }
        
        # 過去nレースのリスト
        n_race_list = [1, 3, 5, 10]
        baselog2 = (
            self.population.merge(
                self.horse_results,
                on=["horse_id"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        
        # 集計用データフレームの初期化
        merged_df2 = self.population.copy()
        # grouped_dfを適用して計算
        grouped_df2 = baselog2.groupby(["race_id", "horse_id"])
        # 各過去nレースの割合を計算して追加
        for n_race in n_race_list:
            # `n_race`を整数として関数に渡す
            position_percentage = grouped_df2.apply(
                lambda group: calculate_race_position_percentage(group, n_race=n_race),
                include_groups=False,  # グループ列を除外
            )
            
            # 結果をデータフレームとして展開
            position_percentage_df = position_percentage.apply(pd.Series).reset_index()
            position_percentage_df.rename(
                columns={
                    "escape": f"escape_{n_race}races",
                    "taking_lead": f"taking_lead_{n_race}races",
                    "in_front": f"in_front_{n_race}races",
                    "pursuit": f"pursuit_{n_race}races",
                },
                inplace=True,
            )
            
            # 結果をマージ
            merged_df2 = merged_df2.merge(position_percentage_df, on=["race_id", "horse_id"], how="left")
        
        
        
        # dominant_position_category を計算
        position_columns_5 = ["escape_5races", "taking_lead_5races", "in_front_5races", "pursuit_5races"]
        position_columns_3 = ["escape_3races", "taking_lead_3races", "in_front_3races", "pursuit_3races"]
        position_columns_1 = ["escape_1races", "taking_lead_1races", "in_front_1races", "pursuit_1races"]
        
        
        def determine_dominant_position(row):
            
            # 5racesの中から計算
            if not row[position_columns_5].isnull().all():
                max_column = row[position_columns_5].idxmax()
                return {
                    "escape_5races": 1,
                    "taking_lead_5races": 2,
                    "in_front_5races": 3,
                    "pursuit_5races": 4,
                }[max_column]
            # 3racesの中から計算
            elif not row[position_columns_3].isnull().all():
                max_column = row[position_columns_3].idxmax()
                return {
                    "escape_3races": 1,
                    "taking_lead_3races": 2,
                    "in_front_3races": 3,
                    "pursuit_3races": 4,
                }[max_column]
            # 1racesの中から計算
            elif not row[position_columns_1].isnull().all():
                max_column = row[position_columns_1].idxmax()
                return {
                    "escape_1races": 1,
                    "taking_lead_1races": 2,
                    "in_front_1races": 3,
                    "pursuit_1races": 4,
                }[max_column]
            # すべて欠損の場合は1を返す
            else:
                return 3
        
        # 各行に対して dominant_position_category を適用
        merged_df2["dominant_position_category"] = merged_df2.apply(determine_dominant_position, axis=1)
        
        
        
        
            #ペース作成
        """
        merged_df にはrace_id（レースのid）と、そのレースに出走するhorse_id(馬のid)の列があり、
        さらに dominant_position_categoryというその馬の脚質を表す列がある
        脚質は1-4の四種類の数字カテゴリが入っており
        レースごとに、出走する馬全体における、その脚質がいる割合と絶対数を求め、
        それぞれ四つの列を新たに作成
        """
        # dominant_position_category の脚質マッピング
        category_mapping_per = {1: "escape_per", 2: "taking_lead_per", 3: "in_front_per", 4: "pursuit_per"}
        category_mapping_count = {1: "escape_count", 2: "taking_lead_count", 3: "in_front_count", 4: "pursuit_count"}
        
        # 各レースで dominant_position_category の割合を計算
        position_ratios = (
            merged_df2.groupby("race_id")["dominant_position_category"]
            .value_counts(normalize=True)  # 割合を計算
            .unstack(fill_value=0)         # 行: race_id, 列: dominant_position_category の形に
            .rename(columns=category_mapping_per)  # 列名を割合用に置き換え
        )
        
        # 各レースで dominant_position_category の絶対数を計算
        position_counts = (
            merged_df2.groupby("race_id")["dominant_position_category"]
            .value_counts()                # 絶対数を計算
            .unstack(fill_value=0)         # 行: race_id, 列: dominant_position_category の形に
            .rename(columns=category_mapping_count)  # 列名を絶対数用に置き換え
        )
        
        # 割合と絶対数を結合
        position_data = position_ratios.join(position_counts)
        columns_to_remove = list(position_data.columns)
        merged_df2 = merged_df2.drop(columns=columns_to_remove, errors="ignore")
        
        # 元の DataFrame にマージ
        merged_df2 = merged_df2.merge(position_data, how="left", on="race_id")
        
        
        if 'pace_category' in merged_df2.columns:
            merged_df2 = merged_df2.drop(columns=['pace_category'])
        
        # ペースカテゴリを計算する関数
        def calculate_pace_category(row):
            escape_count = row.get("escape_count", 0)
            taking_lead_per = row.get("taking_lead_per", 0)
        
            if escape_count >= 3:
                return 4  # ハイペース
            elif escape_count == 2:
                if taking_lead_per > 0.5:
                    return 4  # ハイペース
                else:
                    return 3  
            elif escape_count == 1:
                if taking_lead_per > 0.6:
                    return 4  # ハイペース
                elif 0.5 < taking_lead_per <= 0.6:
                    return 3  # ハイミドルペース
                elif 0.4 < taking_lead_per <= 0.5:
                    return 2  # スローミドルペース
                else:
                    return 1  # スローペース
            elif escape_count == 0:
                # 逃げ馬が0の場合、先行馬の割合に基づく
                if taking_lead_per > 0.7:
                    return 4  # ハイペース
                elif 0.6 < taking_lead_per <= 0.7:
                    return 3  # ハイミドルペース
                elif 0.5 < taking_lead_per <= 0.6:
                    return 2  # スローミドルペース
                else:
                    return 1  # スローペース
            return None  # 予期しない場合は None
        merged_df2["pace_category"] = merged_df2.apply(calculate_pace_category, axis=1)
        merged_df = merged_df.merge(merged_df2[["race_id","date","horse_id","pace_category"]],
                            on=["race_id","date","horse_id"],)
        
        


        new_columns = []

        # merged_df に対して、1, 3, 5, 10 の各nに対して処理
        for n in [1, 3, 5, 10]:  # nの値を1, 3, 5, 10とする
            for result_type in ["win", "rentai", "show"]:  # win, rentai, showの各列を処理
                # 新しい列名を設定
                new_col_name = f"{result_type}_for_pace_category_n{n}"
                new_columns.append(new_col_name)
        
                # nとresult_typeに基づいて、該当する列を選択
                for category in [1, 2, 3, 4]:  # pace_categoryに基づいて1から4のカテゴリを処理
                    # 該当するpace_categoryとnを持つ元の列を動的に選択
                    col_name = f"{result_type}_{n}races_per_pace_category{category}"
                    
                    # merged_dfに新しい列を代入（pace_categoryに関わらず同じ列に結果を格納）
                    merged_df.loc[merged_df["pace_category"] == category, new_col_name] = merged_df[col_name]

        merged_df = merged_df[["race_id","horse_id",
            'win_for_pace_category_n1',
         'rentai_for_pace_category_n1',
         'show_for_pace_category_n1',
         'win_for_pace_category_n3',
         'rentai_for_pace_category_n3',
         'show_for_pace_category_n3',
         'win_for_pace_category_n5',
         'rentai_for_pace_category_n5',
         'show_for_pace_category_n5',
         'win_for_pace_category_n10',
         'rentai_for_pace_category_n10',
         'show_for_pace_category_n10']]
        # race_id ごとに標準化（相対値）を計算
        tmp_df = merged_df.groupby("race_id")
        
        # 各レースごとに、平均と標準偏差を計算
        # .transform を使って、同じサイズのデータフレームに変換
        relative_df = (merged_df - tmp_df.transform("mean")) / tmp_df.transform("std")
        
        # カラム名をわかりやすくするために "_relative" を追加
        relative_df = relative_df.add_suffix("_relative")
        
        # 元のデータフレームに標準化されたデータを結合
        merged_df1000 = merged_df.join(relative_df, how="left")
        merged_df1000 = merged_df1000[['race_id', 'horse_id', 
               #                         'win_for_pace_category_n1',
               # 'rentai_for_pace_category_n1', 'show_for_pace_category_n1',
               # 'win_for_pace_category_n3', 'rentai_for_pace_category_n3',
               # 'show_for_pace_category_n3', 'win_for_pace_category_n5',
               # 'rentai_for_pace_category_n5', 'show_for_pace_category_n5',
               # 'win_for_pace_category_n10', 'rentai_for_pace_category_n10',
               # 'show_for_pace_category_n10', 
               'rentai_for_pace_category_n1_relative',
               'rentai_for_pace_category_n10_relative',
               'rentai_for_pace_category_n3_relative',
               'rentai_for_pace_category_n5_relative',
               'show_for_pace_category_n1_relative',
               'show_for_pace_category_n10_relative',
               'show_for_pace_category_n3_relative',
               'show_for_pace_category_n5_relative',
               'win_for_pace_category_n1_relative',
               'win_for_pace_category_n10_relative',
               'win_for_pace_category_n3_relative',
               'win_for_pace_category_n5_relative']]
        merged_df4 = self.population.copy()
        merged_df1000 = merged_df4.merge(merged_df1000, on=["race_id", "horse_id"], how="left")

        self.agg_cross_features_df_7 = merged_df1000
        print("running cross_features_7()...comp")

    def cross_features_8(
        self, n_races: list[int] = [1,  3, 5,8]
    ):  
        """
        特筆見る
        
        過去レースで
        ◎スローペースで勝った差し・追い込み馬は瞬発力のある証拠
        ◎ハイペースで勝った逃げ・先行馬は高く評価しましょう
        をつくる、それの方が数字が多くなるように
        
        
        脚質がある
        ペースの速さがある
        以下のときで、showに入っているとき
        +レースクラス+1、でいいかも
        入ってなかったら0
        過去3.5.10.100raceの最大だけ見る
        それを標準化としないver
        
        ＿＿ハイペース1→4
        逃げ先行がバテるため、差し追い込みが有利
        
        ミドルハイペース2→3
        
        ミドルローペース3→2
        
        ＿＿ローペース4→1
        後続を寄せ付けないため逃げ先行が有利

        
        逃げ1
        234
        
        """
        n_races: list[int] = [1, 3, 5, 10]
        
        baselog = (
            self.population
            .merge(
                self.horse_results,
                on="horse_id",
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        def calculate_scores(baselog):
            # スコア初期化
            baselog["score_pace_potision"] = 0.0
        
            # スローペースで差し・追い込み馬がshowに入った場合
            condition1 = (baselog["pace_category"] == 1) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition1, "score_pace_potision"] += baselog['race_grade_scaled'] + 1
        
            # ハイペースで逃げ・先行馬がshowに入った場合
            condition2 = (baselog["pace_category"] == 4) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            baselog.loc[condition2, "score_pace_potision"] += baselog['race_grade_scaled'] + 1
        
            # ミドルスローペースで差し・追い込み馬がshowに入った場合
            condition3 = (baselog["pace_category"] == 2) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition3, "score_pace_potision"] += (baselog['race_grade_scaled'] + 1) / 1.8
        
            # ミドルハイペースで逃げ・先行馬がshowに入った場合
            condition4 = (baselog["pace_category"] == 3) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            baselog.loc[condition4, "score_pace_potision"] += (baselog['race_grade_scaled'] + 1) / 1.8
        
            # 逃げ・先行馬がshowに入った場合
            condition5 =baselog["pace_category"].isin([1, 2]) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            baselog.loc[condition5, "score_pace_potision"] += (baselog['race_grade_scaled'] + 1) / 3.6
            # 差し・追い込み馬showに入った場合
            condition6 =baselog["pace_category"].isin([3, 4]) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition6, "score_pace_potision"] += (baselog['race_grade_scaled'] + 1) / 3.6    
        
             # showに入りらなかった場合 
            condition7 = (baselog["show"] != 1)
            baselog.loc[condition7, "score_pace_potision"] += 0 
        
            return baselog
        result_df = calculate_scores(baselog)
        
        grouped_df = result_df.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        
        for n_race in tqdm(n_races, desc=f"agg_pace_per_score"):
            # スコアの集計（標準化しない版）
            raw_df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["score_pace_potision"]]
                .agg(["mean", "max", "min"])
            )
            raw_df.columns = [
                "_".join(col) + f"_{n_race}races_per_score_raw" for col in raw_df.columns
            ]
            
            # スコアの集計（標準化する版）
            std_df = raw_df.copy()  # 同じデータを使用して標準化
            tmp_df = raw_df.groupby(["race_id"])
            std_df = ((raw_df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            
            # merged_df に追加
            # merged_df = merged_df.merge(raw_df, on=["race_id", "horse_id"], how="left")
            merged_df = merged_df.merge(std_df, on=["race_id", "horse_id"], how="left")
    

        self.agg_cross_features_df_8 = merged_df
        print("running cross_features_8()...comp")




    
    def cross_features_9(
        self, n_races: list[int] = [1, 3, 5, 10]
    ):  
        """
        特筆見る
        
        直線が短い差しは不利
        長いと先行不利、というわけではない
        
        
        """

        
        baselog = (
            self.population
            .merge(
                self.horse_results,
                on="horse_id",
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        def calculate_scores(baselog):
            # スコア初期化
            baselog["score_goal_range"] = 0.0
        
            # 直線が短いで差し・追い込み馬がshowに入った場合
            condition1 = (baselog["goal_range_100"] < 3) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition1, "score_goal_range"] += (baselog['race_grade_scaled'] + 1) / 2
        
            # 直線が短いで差し・追い込み馬がshowに入った場合
            condition2 = (baselog["goal_range_100"] < 4) & (baselog["goal_range_100"] >= 3) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition2, "score_goal_range"] +=(baselog['race_grade_scaled'] + 1) / 2.7
        
            # 直線が短いで差し・追い込み馬がshowに入った場合
            condition3 = (baselog["goal_range_100"] < 4.2) & (baselog["goal_range_100"] >= 4) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition3, "score_goal_range"] +=(baselog['race_grade_scaled'] + 1) / 3.5
        
            # 先行馬がshowに入った場合
            condition4 = (baselog["goal_range_100"] >= 0) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            baselog.loc[condition4, "score_goal_range"] += (baselog['race_grade_scaled'] + 1) / 4.5
        
            # 直線が短いで差し・追い込み馬がshowに入った場合
            condition5 = (baselog["goal_range_100"] >= 4.2) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition5, "score_goal_range"] +=(baselog['race_grade_scaled'] + 1) / 4.5
        
             # showに入りらなかった場合 
            condition7 = (baselog["show"] != 1)
            baselog.loc[condition7, "score_goal_range"] += 0 
        
            
            return baselog
        result_df = calculate_scores(baselog)
        
        grouped_df = result_df.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        
        for n_race in tqdm(n_races, desc=f"agg_goal_per_score"):
            # スコアの集計（標準化しない版）
            raw_df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["score_goal_range"]]
                .agg(["mean", "max", "min"])
            )
            raw_df.columns = [
                "_".join(col) + f"_{n_race}races_per_goal_raw" for col in raw_df.columns
            ]
            
            # スコアの集計（標準化する版）
            std_df = raw_df.copy()  # 同じデータを使用して標準化
            tmp_df = raw_df.groupby(["race_id"])
            std_df = ((raw_df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            
            # merged_df に追加
            # merged_df = merged_df.merge(raw_df, on=["race_id", "horse_id"], how="left")
            merged_df = merged_df.merge(std_df, on=["race_id", "horse_id"], how="left")
        self.agg_cross_features_df_9 = merged_df
        print("running cross_features_9()...comp")



    def cross_features_10(
        self, n_races: list[int] = [1, 3, 5, 10]
    ):  
        
        """
        特筆見る
        
        急カーブは小回りよりさらにきつい
        小回りで買った差し？　差し不利
        小スパイラル　小回りほどではないが、先行有利
        スパイラル　小回りほどではないが、先行有利
        
        大回り・複合で勝った先行？　先行不利
        
        "急": 1,
        "小回": 2,
        "小スパ": 3,
        "スパ": 4,
        "複合": 5,
        None: None 
        """

        
        baselog = (
            self.population
            .merge(
                self.horse_results,
                on="horse_id",
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        def calculate_scores(baselog):
            # スコア初期化
            baselog["score_curve_range"] = 0.0
        
            # # 急カーブで差し・追い込み馬がshowに入った場合
            # condition1 = (baselog["curve"] == 1) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            # baselog.loc[condition1, "score_curve_range"] += (baselog["race_grade"] + 1)
        
            # # 小回りカーブで差し・追い込み馬がshowに入った場合
            # condition2 = (baselog["curve"] == 2) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            # baselog.loc[condition2, "score_curve_range"] += (baselog["race_grade"] + 1) / 2
            
            # # 小スパカーブで差し・追い込み馬がshowに入った場合
            # condition3 = (baselog["curve"] == 3) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            # baselog.loc[condition3, "score_curve_range"] += (baselog["race_grade"] + 1) / 2.3
        
            # # スパカーブで差し・追い込み馬がshowに入った場合
            # condition4 = (baselog["curve"] == 4) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            # baselog.loc[condition4, "score_curve_range"] += (baselog["race_grade"] + 1) / 2.6
        
            # # 複合カーブで差し・追い込み馬がshowに入った場合
            # condition5 = (baselog["curve"] == 5) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            # baselog.loc[condition5, "score_curve_range"] += (baselog["race_grade"] + 1)/ 5   
            
            # # 複合カーブで先行がshowに入った場合
            # condition6 = (baselog["curve"] == 5) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            # baselog.loc[condition6, "score_curve_range"] += (baselog["race_grade"] + 1)   
            
            # # カーブで先行がshowに入った場合
            # condition8 = (baselog["curve"].isin([1, 2,3,4])) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            # baselog.loc[condition8, "score_curve_range"] += (baselog["race_grade"] + 1)/  5
        
            
            # 急カーブで差し・追い込み馬がshowに入った場合
            condition1 = (baselog["curve"] == 1) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition1, "score_curve_range"] += (baselog['race_grade_scaled'] + 1) / 1.7
        
            # 小回りカーブで差し・追い込み馬がshowに入った場合
            condition2 = (baselog["curve"] == 2) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition2, "score_curve_range"] += (baselog['race_grade_scaled'] + 1) / 2.1
            
            # 小スパカーブで差し・追い込み馬がshowに入った場合
            condition3 = (baselog["curve"] == 3) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition3, "score_curve_range"] += (baselog['race_grade_scaled'] + 1) / 2.4
        
            # スパカーブで差し・追い込み馬がshowに入った場合
            condition4 = (baselog["curve"] == 4) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition4, "score_curve_range"] += (baselog['race_grade_scaled'] + 1) / 2.7
        
            # 複合カーブで差し・追い込み馬がshowに入った場合
            condition5 = (baselog["curve"] == 5) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition5, "score_curve_range"] += (baselog['race_grade_scaled'] + 1)/ 5   
            
            # 複合カーブで先行がshowに入った場合
            condition6 = (baselog["curve"] == 5) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            baselog.loc[condition6, "score_curve_range"] += (baselog['race_grade_scaled'] + 1)/ 1.8    
            
            # カーブで先行がshowに入った場合
            condition8 = (baselog["curve"].isin([1, 2,3,4])) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            baselog.loc[condition8, "score_curve_range"] += (baselog['race_grade_scaled'] + 1)/  5
        
            
             # showに入りらなかった場合 
            condition7 = (baselog["show"] != 1)
            baselog.loc[condition7, "score_curve_range"] += 0 
            
             # showに入りらなかった場合 
            condition9 = (~baselog["curve"].isin([1, 2, 3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition9, "score_curve_range"] += 0 
            
        
            return baselog
        result_df = calculate_scores(baselog)
        
        grouped_df = result_df.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        
        for n_race in tqdm(n_races, desc=f"agg_curve_per_score"):
            # スコアの集計（標準化しない版）
            raw_df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["score_curve_range"]]
                .agg(["mean", "max", "min"])
            )
            raw_df.columns = [
                "_".join(col) + f"_{n_race}races_per_curve_raw" for col in raw_df.columns
            ]
            
            # スコアの集計（標準化する版）
            std_df = raw_df.copy()  # 同じデータを使用して標準化
            tmp_df = raw_df.groupby(["race_id"])
            std_df = ((raw_df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            
            # merged_df に追加
            # merged_df = merged_df.merge(raw_df, on=["race_id", "horse_id"], how="left")
            merged_df = merged_df.merge(std_df, on=["race_id", "horse_id"], how="left")

        self.agg_cross_features_df_10 = merged_df
        print("running cross_features_10()...comp")



    def cross_features_11(
        self, n_races: list[int] = [1, 3, 5, 10]
    ):  


        """
        特筆見る
        
        坂のあるところ
        あるなら差しが有利、ただしペースが早いと先行有利
        ないならどっちも
        だから、
        坂があってスローペースや普通のとき、差しが有利、先行評価
        坂があって、ハイペースのとき、先行有利、差し評価
        ゆるさかは/1.5
        ないなら同評価
        "平坦": 1,        
        "緩坂": 2,
        "急坂": 3
        """

        
        baselog = (
            self.population
            .merge(
                self.horse_results,
                on="horse_id",
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        def calculate_scores(baselog):
            # スコア初期化
            baselog["score_goal_slope"] = 0.0
        
        
            # 平坦の時、特に評価なし
            condition1 = (baselog["goal_slope"] == 1) & (baselog["race_position"].isin([1,2,3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition1, "score_goal_slope"] += (baselog['race_grade_scaled'] + 1) /5
        
            # ゆる坂ペースが早いとき
            condition2 = (baselog["goal_slope"] == 2) & (baselog["race_position"].isin([3, 4])) & baselog["pace_category"].isin([4]) & (baselog["show"] == 1)
            baselog.loc[condition2, "score_goal_slope"] += (baselog['race_grade_scaled'] + 1) / 2.2
            
            # ゆる坂ペースが遅いとき
            condition3 = (baselog["goal_slope"] == 2) & (baselog["race_position"].isin([1,2])) & baselog["pace_category"].isin([1]) & (baselog["show"] == 1)
            baselog.loc[condition3, "score_goal_slope"] += (baselog['race_grade_scaled'] + 1) / 2.2
        
            # ゆる坂ペースが早いとき
            condition4 = (baselog["goal_slope"] == 2) & (baselog["race_position"].isin([3, 4])) & baselog["pace_category"].isin([3]) & (baselog["show"] == 1)
            baselog.loc[condition4, "score_goal_slope"] += (baselog['race_grade_scaled'] + 1) / 3.0
            
            # ゆる坂ペースが遅いとき
            condition5 = (baselog["goal_slope"] == 2) & (baselog["race_position"].isin([1,2])) & baselog["pace_category"].isin([2]) & (baselog["show"] == 1)
            baselog.loc[condition5, "score_goal_slope"] += (baselog['race_grade_scaled'] + 1) / 3.0
        
        
            # ゆる坂ペースそれ以外
            condition6 = (baselog["goal_slope"] == 2) & (baselog["race_position"].isin([1,2])) & baselog["pace_category"].isin([3,4]) & (baselog["show"] == 1)
            baselog.loc[condition6, "score_goal_slope"] += (baselog['race_grade_scaled'] + 1) /4.5
        
            # ゆる坂ペースそれ以外
            condition7 = (baselog["goal_slope"] == 2) & (baselog["race_position"].isin([3, 4])) & baselog["pace_category"].isin([1,2]) & (baselog["show"] == 1)
            baselog.loc[condition7, "score_goal_slope"] += (baselog['race_grade_scaled'] + 1) /4.5
            
            
            
            # 厳しい坂でペースが早いとき
            condition8 = (baselog["goal_slope"] == 3) & (baselog["race_position"].isin([3, 4])) & baselog["pace_category"].isin([4]) & (baselog["show"] == 1)
            baselog.loc[condition8, "score_goal_slope"] += (baselog['race_grade_scaled'] + 1) /1.5
            
            # 厳しい坂ペースが遅いとき
            condition9 = (baselog["goal_slope"] == 3) & (baselog["race_position"].isin([1,2])) & baselog["pace_category"].isin([1]) & (baselog["show"] == 1)
            baselog.loc[condition9, "score_goal_slope"] += (baselog['race_grade_scaled'] + 1) /1.5
        
            # 厳しい坂でペースが早いとき
            condition10 = (baselog["goal_slope"] == 3) & (baselog["race_position"].isin([3, 4])) & baselog["pace_category"].isin([3]) & (baselog["show"] == 1)
            baselog.loc[condition10, "score_goal_slope"] += (baselog['race_grade_scaled'] + 1) / 2.0
            
            # 厳しい坂ペースが遅いとき
            condition11 = (baselog["goal_slope"] == 3) & (baselog["race_position"].isin([1,2])) & baselog["pace_category"].isin([2]) & (baselog["show"] == 1)
            baselog.loc[condition11, "score_goal_slope"] += (baselog['race_grade_scaled'] + 1) / 2.0
        
            
            # 厳しい坂ペースそれ以外
            condition12 = (baselog["goal_slope"] == 3) & (baselog["race_position"].isin([1,2])) & baselog["pace_category"].isin([3,4]) & (baselog["show"] == 1)
            baselog.loc[condition12, "score_goal_slope"] += (baselog['race_grade_scaled'] + 1) / 3.0
        
            # 厳しい坂ペースそれ以外
            condition13 = (baselog["goal_slope"] == 3) & (baselog["race_position"].isin([3, 4])) & baselog["pace_category"].isin([1,2]) & (baselog["show"] == 1)
            baselog.loc[condition13, "score_goal_slope"] += (baselog['race_grade_scaled'] + 1) / 3.0
        
            
        
             # showに入りらなかった場合 
            condition14 = (baselog["show"] != 1)
            baselog.loc[condition14, "score_goal_slope"] += 0 
            
             # showに入りらなかった場合 
            condition15 = (~baselog["goal_slope"].isin([1, 2, 3])) & (baselog["show"] == 1)
            baselog.loc[condition15, "score_goal_slope"] += 0 
            
        
            return baselog
        result_df = calculate_scores(baselog)
        
        grouped_df = result_df.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        
        for n_race in tqdm(n_races, desc=f"agg_goal_slope_per_score"):
            # スコアの集計（標準化しない版）
            raw_df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["score_goal_slope"]]
                .agg(["mean", "max", "min"])
            )
            raw_df.columns = [
                "_".join(col) + f"_{n_race}races_per_goal_slope_raw" for col in raw_df.columns
            ]
            
            # スコアの集計（標準化する版）
            std_df = raw_df.copy()  # 同じデータを使用して標準化
            tmp_df = raw_df.groupby(["race_id"])
            std_df = ((raw_df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            
            # merged_df に追加
            # merged_df = merged_df.merge(raw_df, on=["race_id", "horse_id"], how="left")
            merged_df = merged_df.merge(std_df, on=["race_id", "horse_id"], how="left")
        
        self.agg_cross_features_df_11 = merged_df
        print("running cross_features_11()...comp")
    


    

    def cross_features_12(
        self, n_races: list[int] = [1, 3, 5, 10]
    ):  
        """
        特筆見る
        
        スタミナ指標
        かっているとして
        長ければ+
        小回りで-
        大回りでさらに+
        ⑤大回りのコースは、カーブで息が入りにくいので、距離に不安のある馬には厳しい
        
        
        'race_grade_scaled'*コースの長さ/2000
        
        こいつは相対値以外も出していい
        
        """
        
        baselog = (
            self.population
            .merge(
                self.horse_results,
                on="horse_id",
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        def calculate_scores(baselog):
            # スコア初期化
            baselog["score_stamina"] = 0.0
        
        
        
            # 急カーブ場合
            condition1 = (baselog["curve"] == 1) & (baselog["show"] == 1)
            baselog.loc[condition1, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *(((baselog["course_len"]*0.0025)+20)/20)*(((baselog["pace_diff"] * 1)+20)/20)/ 2.5 
        
            # 小回りカーブで場合
            condition2 = (baselog["curve"] == 2)  & (baselog["show"] == 1)
            baselog.loc[condition2, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *(((baselog["course_len"]*0.0025)+20)/20)*(((baselog["pace_diff"] * 1)+20)/20)/ 2.1
            
            # 小スパカーブ場合
            condition3 = (baselog["curve"] == 3) & (baselog["show"] == 1)
            baselog.loc[condition3, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *(((baselog["course_len"]*0.0025)+20)/20)*(((baselog["pace_diff"] * 1)+20)/20)/ 1.8
        
            # スパカーブ場合
            condition4 = (baselog["curve"] == 4)  & (baselog["show"] == 1)
            baselog.loc[condition4, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *(((baselog["course_len"]*0.0025)+20)/20)*(((baselog["pace_diff"] * 1)+20)/20)/ 1.5
        
            # 複合カーブで場合
            condition5 = (baselog["curve"] == 5)  & (baselog["show"] == 1)
            baselog.loc[condition5, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori']))*(((baselog["course_len"]*0.0025)+20)/20)*(((baselog["pace_diff"] * 1)+20)/20)
            
            
             # showに入りらなかった場合 
            condition7 = (baselog["show"] != 1)
            baselog.loc[condition7, "score_stamina"] += 0 
            
             # showに入りらなかった場合 
            condition8 = (~baselog["curve"].isin([1, 2, 3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition8, "score_stamina"] += 0 
            
        
            return baselog
        result_df = calculate_scores(baselog)
        
        grouped_df = result_df.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        
        for n_race in tqdm(n_races, desc=f"agg_stamina_per_score"):
            # スコアの集計（標準化しない版）
            raw_df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["score_stamina"]]
                .agg(["mean", "max", "min"])
            )
            raw_df.columns = [
                "_".join(col) + f"_{n_race}races_per_stamina_raw" for col in raw_df.columns
            ]
            
            # スコアの集計（標準化する版）
            std_df = raw_df.copy()  # 同じデータを使用して標準化
            tmp_df = raw_df.groupby(["race_id"])
            std_df = ((raw_df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            
            # merged_df に追加
            merged_df = merged_df.merge(raw_df, on=["race_id", "horse_id"], how="left")
            merged_df = merged_df.merge(std_df, on=["race_id", "horse_id"], how="left")
        
        # merged_df2 = merged_df.merge(self.race_info[["race_id","course_len"]], on="race_id")
        # merged_df2["course_len_relative"] = (merged_df2["course_len"] / 1600/7) +1
        # merged_df2.loc[merged_df2["course_len_relative"] < 0, "course_len_relative"] = 0
        # merged_df2["course_len_relative"] = merged_df2["course_len_relative"] 
        # # 対象の列名リスト
        # columns_to_multiply = [
        #     'score_stamina_mean_1races_per_stamina_raw',
        #     'score_stamina_max_1races_per_stamina_raw',
        #     'score_stamina_min_1races_per_stamina_raw',
        #     'score_stamina_mean_1races_per_stamina_raw_relative',
        #     'score_stamina_max_1races_per_stamina_raw_relative',
        #     'score_stamina_min_1races_per_stamina_raw_relative',
        #     'score_stamina_mean_3races_per_stamina_raw',
        #     'score_stamina_max_3races_per_stamina_raw',
        #     'score_stamina_min_3races_per_stamina_raw',
        #     'score_stamina_mean_3races_per_stamina_raw_relative',
        #     'score_stamina_max_3races_per_stamina_raw_relative',
        #     'score_stamina_min_3races_per_stamina_raw_relative',
        #     'score_stamina_mean_5races_per_stamina_raw',
        #     'score_stamina_max_5races_per_stamina_raw',
        #     'score_stamina_min_5races_per_stamina_raw',
        #     'score_stamina_mean_5races_per_stamina_raw_relative',
        #     'score_stamina_max_5races_per_stamina_raw_relative',
        #     'score_stamina_min_5races_per_stamina_raw_relative',
        #     'score_stamina_mean_10races_per_stamina_raw',
        #     'score_stamina_max_10races_per_stamina_raw',
        #     'score_stamina_min_10races_per_stamina_raw',
        #     'score_stamina_mean_10races_per_stamina_raw_relative',
        #     'score_stamina_max_10races_per_stamina_raw_relative',
        #     'score_stamina_min_10races_per_stamina_raw_relative'
        # ]
        
        # # 'course_len_relative'を各列に掛け算
        # for col in columns_to_multiply:
        #     merged_df2[col] = merged_df2[col] * merged_df2['course_len_relative']
        # # 'course_len'と'course_len_relative'の列を削除
        # merged_df2 = merged_df2.drop(columns=['course_len', 'course_len_relative'])

        self.agg_cross_features_df_12 = merged_df
        print("running cross_features_12()...comp")

    
    
    def cross_features_13(
        self, n_races: list[int] = [1, 3, 5, 10]
    ):  
        """
        特筆見る
        オールスコア
        コースの長さが長い時だけ、スタミナ指標を追加したやつも入れる
        """
        n_races: list[int] = [1, 3, 5, 10]
        
        baselog = (
            self.population
            .merge(
                self.horse_results,
                on="horse_id",
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        def calculate_scores(baselog):
            # スコア初期化
            baselog["score_pace_potision"] = 0.0
        
            # スローペースで差し・追い込み馬がshowに入った場合
            condition1 = (baselog["pace_category"] == 1) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition1, "score_pace_potision"] += baselog["race_grade_scaled"] + 1
        
            # ハイペースで逃げ・先行馬がshowに入った場合
            condition2 = (baselog["pace_category"] == 4) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            baselog.loc[condition2, "score_pace_potision"] += baselog["race_grade_scaled"] + 1
        
            # ミドルスローペースで差し・追い込み馬がshowに入った場合
            condition3 = (baselog["pace_category"] == 2) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition3, "score_pace_potision"] += (baselog["race_grade_scaled"] + 1) / 1.8
        
            # ミドルハイペースで逃げ・先行馬がshowに入った場合
            condition4 = (baselog["pace_category"] == 3) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            baselog.loc[condition4, "score_pace_potision"] += (baselog["race_grade_scaled"] + 1) / 1.8
        
            # 逃げ・先行馬がshowに入った場合
            condition5 =baselog["pace_category"].isin([1, 2]) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            baselog.loc[condition5, "score_pace_potision"] += (baselog["race_grade_scaled"] + 1) / 3.6
            # 差し・追い込み馬showに入った場合
            condition6 =baselog["pace_category"].isin([3, 4]) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition6, "score_pace_potision"] += (baselog["race_grade_scaled"] + 1) / 3.6    
        
             # showに入りらなかった場合 
            condition7 = (baselog["show"] != 1)
            baselog.loc[condition7, "score_pace_potision"] += 0 
           
            # # その他のペースでshowに入った場合
            # condition5 = ~(
            #     (baselog["pace_category"].isin([1, 2, 3, 4])) & 
            #     (baselog["race_position"].isin([1, 2, 3, 4])) & 
            #     (baselog["show"] == 1)
            # )
            # baselog.loc[condition5 & (baselog["show"] == 1), "score_pace_potision"] += (baselog["race_grade"] + 1) / 3.6
        
            return baselog
        result_df1 = calculate_scores(baselog)
        
        
        grouped_df = result_df1.groupby(["race_id", "horse_id"])
        merged_df1 = self.population.copy()
        
        for n_race in tqdm(n_races, desc=f"agg_pace_per_score"):
            # スコアの集計（標準化しない版）
            raw_df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["score_pace_potision"]]
                .agg(["mean", "max", "min"])
            )
            raw_df.columns = [
                "_".join(col) + f"_{n_race}races_per_score_raw" for col in raw_df.columns
            ]
            
            # merged_df に追加
            merged_df1 = merged_df1.merge(raw_df, on=["race_id", "horse_id"], how="left")
        
        
        """
        特筆見る
        
        直線が短い差しは不利
        長いと先行不利、というわけではない
        
        
        """
        
        
        baselog = (
            self.population
            .merge(
                self.horse_results,
                on="horse_id",
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        def calculate_scores(baselog):
            # スコア初期化
            baselog["score_goal_range"] = 0.0
        
            # 直線が短いで差し・追い込み馬がshowに入った場合
            condition1 = (baselog["goal_range_100"] < 3) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition1, "score_goal_range"] += (baselog["race_grade_scaled"] + 1) / 2
        
            # 直線が短いで差し・追い込み馬がshowに入った場合
            condition2 = (baselog["goal_range_100"] < 4) & (baselog["goal_range_100"] >= 3) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition2, "score_goal_range"] +=(baselog["race_grade_scaled"] + 1) / 2.7
        
            # 直線が短いで差し・追い込み馬がshowに入った場合
            condition3 = (baselog["goal_range_100"] < 4.2) & (baselog["goal_range_100"] >= 4) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition3, "score_goal_range"] +=(baselog["race_grade_scaled"] + 1) / 3.5
        
            # 先行馬がshowに入った場合
            condition4 = (baselog["goal_range_100"] >= 0) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            baselog.loc[condition4, "score_goal_range"] += (baselog["race_grade_scaled"] + 1) / 4.5
        
            # 直線が短いで差し・追い込み馬がshowに入った場合
            condition5 = (baselog["goal_range_100"] >= 4.2) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition5, "score_goal_range"] +=(baselog["race_grade_scaled"] + 1) / 4.5
        
             # showに入りらなかった場合 
            condition7 = (baselog["show"] != 1)
            baselog.loc[condition7, "score_goal_range"] += 0 
        
        
            
            return baselog
        result_df2 = calculate_scores(baselog)
        
        grouped_df = result_df2.groupby(["race_id", "horse_id"])
        merged_df2 = self.population.copy()
        
        for n_race in tqdm(n_races, desc=f"agg_goal_per_score"):
            # スコアの集計（標準化しない版）
            raw_df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["score_goal_range"]]
                .agg(["mean", "max", "min"])
            )
            raw_df.columns = [
                "_".join(col) + f"_{n_race}races_per_goal_raw" for col in raw_df.columns
            ]
        
            # merged_df に追加
            merged_df2 = merged_df2.merge(raw_df, on=["race_id", "horse_id"], how="left")
        
        
        
        
        """
        特筆見る
        
        急カーブは小回りよりさらにきつい
        小回りで買った差し？　差し不利
        小スパイラル　小回りほどではないが、先行有利
        スパイラル　小回りほどではないが、先行有利
        
        大回り・複合で勝った先行？　先行不利
        
        "急": 1,
        "小回": 2,
        "小スパ": 3,
        "スパ": 4,
        "複合": 5,
        None: None 
        """
        
        
        baselog = (
            self.population
            .merge(
                self.horse_results,
                on="horse_id",
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        def calculate_scores(baselog):
            # スコア初期化
            baselog["score_curve_range"] = 0.0
        
            # # 急カーブで差し・追い込み馬がshowに入った場合
            # condition1 = (baselog["curve"] == 1) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            # baselog.loc[condition1, "score_curve_range"] += (baselog["race_grade"] + 1)
        
            # # 小回りカーブで差し・追い込み馬がshowに入った場合
            # condition2 = (baselog["curve"] == 2) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            # baselog.loc[condition2, "score_curve_range"] += (baselog["race_grade"] + 1) / 2
            
            # # 小スパカーブで差し・追い込み馬がshowに入った場合
            # condition3 = (baselog["curve"] == 3) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            # baselog.loc[condition3, "score_curve_range"] += (baselog["race_grade"] + 1) / 2.3
        
            # # スパカーブで差し・追い込み馬がshowに入った場合
            # condition4 = (baselog["curve"] == 4) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            # baselog.loc[condition4, "score_curve_range"] += (baselog["race_grade"] + 1) / 2.6
        
            # # 複合カーブで差し・追い込み馬がshowに入った場合
            # condition5 = (baselog["curve"] == 5) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            # baselog.loc[condition5, "score_curve_range"] += (baselog["race_grade"] + 1)/ 5   
            
            # # 複合カーブで先行がshowに入った場合
            # condition6 = (baselog["curve"] == 5) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            # baselog.loc[condition6, "score_curve_range"] += (baselog["race_grade"] + 1)   
            
            # # カーブで先行がshowに入った場合
            # condition8 = (baselog["curve"].isin([1, 2,3,4])) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            # baselog.loc[condition8, "score_curve_range"] += (baselog["race_grade"] + 1)/  5
        
            
            # 急カーブで差し・追い込み馬がshowに入った場合
            condition1 = (baselog["curve"] == 1) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition1, "score_curve_range"] += (baselog["race_grade_scaled"] + 1) / 1.8
        
            # 小回りカーブで差し・追い込み馬がshowに入った場合
            condition2 = (baselog["curve"] == 2) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition2, "score_curve_range"] += (baselog["race_grade_scaled"] + 1) / 2.1
            
            # 小スパカーブで差し・追い込み馬がshowに入った場合
            condition3 = (baselog["curve"] == 3) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition3, "score_curve_range"] += (baselog["race_grade_scaled"] + 1) / 2.4
        
            # スパカーブで差し・追い込み馬がshowに入った場合
            condition4 = (baselog["curve"] == 4) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition4, "score_curve_range"] += (baselog["race_grade_scaled"] + 1) / 2.7
        
            # 複合カーブで差し・追い込み馬がshowに入った場合
            condition5 = (baselog["curve"] == 5) & (baselog["race_position"].isin([3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition5, "score_curve_range"] += (baselog["race_grade_scaled"] + 1)/ 5   
            
            # 複合カーブで先行がshowに入った場合
            condition6 = (baselog["curve"] == 5) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            baselog.loc[condition6, "score_curve_range"] += (baselog["race_grade_scaled"] + 1)/ 1.8    
            
            # カーブで先行がshowに入った場合
            condition8 = (baselog["curve"].isin([1, 2,3,4])) & (baselog["race_position"].isin([1, 2])) & (baselog["show"] == 1)
            baselog.loc[condition8, "score_curve_range"] += (baselog["race_grade_scaled"] + 1)/  5
        
            
            
             # showに入りらなかった場合 
            condition7 = (baselog["show"] != 1)
            baselog.loc[condition7, "score_curve_range"] += 0 
            
             # showに入りらなかった場合 
            condition9 = (~baselog["curve"].isin([1, 2, 3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition9, "score_curve_range"] += 0 
            
        
            return baselog
        result_df3 = calculate_scores(baselog)
        
        grouped_df = result_df3.groupby(["race_id", "horse_id"])
        merged_df3 = self.population.copy()
        
        for n_race in tqdm(n_races, desc=f"agg_curve_per_score"):
            # スコアの集計（標準化しない版）
            raw_df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["score_curve_range"]]
                .agg(["mean", "max", "min"])
            )
            raw_df.columns = [
                "_".join(col) + f"_{n_race}races_per_curve_raw" for col in raw_df.columns
            ]
            
            # merged_df に追加
            merged_df3 = merged_df3.merge(raw_df, on=["race_id", "horse_id"], how="left")
        
        
        
        """
        特筆見る
        
        坂のあるところ
        あるなら差しが有利、ただしペースが早いと先行有利
        ないならどっちも
        だから、
        坂があってスローペースや普通のとき、差しが有利、先行評価
        坂があって、ハイペースのとき、先行有利、差し評価
        ゆるさかは/1.5
        ないなら同評価
        "平坦": 1,        
        "緩坂": 2,
        "急坂": 3
        """
        
        
        baselog = (
            self.population
            .merge(
                self.horse_results,
                on="horse_id",
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        def calculate_scores(baselog):
            # スコア初期化
            baselog["score_goal_slope"] = 0.0
        
        
            # 平坦の時、特に評価なし
            condition1 = (baselog["goal_slope"] == 1) & (baselog["race_position"].isin([1,2,3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition1, "score_goal_slope"] += (baselog["race_grade_scaled"] + 1) /5
        
            # ゆる坂ペースが早いとき
            condition2 = (baselog["goal_slope"] == 2) & (baselog["race_position"].isin([3, 4])) & baselog["pace_category"].isin([4]) & (baselog["show"] == 1)
            baselog.loc[condition2, "score_goal_slope"] += (baselog["race_grade_scaled"] + 1) / 2.2
            
            # ゆる坂ペースが遅いとき
            condition3 = (baselog["goal_slope"] == 2) & (baselog["race_position"].isin([1,2])) & baselog["pace_category"].isin([1]) & (baselog["show"] == 1)
            baselog.loc[condition3, "score_goal_slope"] += (baselog["race_grade_scaled"] + 1) / 2.2
        
            # ゆる坂ペースが早いとき
            condition4 = (baselog["goal_slope"] == 2) & (baselog["race_position"].isin([3, 4])) & baselog["pace_category"].isin([3]) & (baselog["show"] == 1)
            baselog.loc[condition4, "score_goal_slope"] += (baselog["race_grade_scaled"] + 1) / 3.0
            
            # ゆる坂ペースが遅いとき
            condition5 = (baselog["goal_slope"] == 2) & (baselog["race_position"].isin([1,2])) & baselog["pace_category"].isin([2]) & (baselog["show"] == 1)
            baselog.loc[condition5, "score_goal_slope"] += (baselog["race_grade_scaled"] + 1) / 3.0
        
        
            # ゆる坂ペースそれ以外
            condition6 = (baselog["goal_slope"] == 2) & (baselog["race_position"].isin([1,2])) & baselog["pace_category"].isin([3,4]) & (baselog["show"] == 1)
            baselog.loc[condition6, "score_goal_slope"] += (baselog["race_grade_scaled"] + 1) /4.5
        
            # ゆる坂ペースそれ以外
            condition7 = (baselog["goal_slope"] == 2) & (baselog["race_position"].isin([3, 4])) & baselog["pace_category"].isin([1,2]) & (baselog["show"] == 1)
            baselog.loc[condition7, "score_goal_slope"] += (baselog["race_grade_scaled"] + 1) /4.5
            
            
            
            # 厳しい坂でペースが早いとき
            condition8 = (baselog["goal_slope"] == 3) & (baselog["race_position"].isin([3, 4])) & baselog["pace_category"].isin([4]) & (baselog["show"] == 1)
            baselog.loc[condition8, "score_goal_slope"] += (baselog["race_grade_scaled"] + 1) /1.5
            
            # 厳しい坂ペースが遅いとき
            condition9 = (baselog["goal_slope"] == 3) & (baselog["race_position"].isin([1,2])) & baselog["pace_category"].isin([1]) & (baselog["show"] == 1)
            baselog.loc[condition9, "score_goal_slope"] += (baselog["race_grade_scaled"] + 1) /1.5
        
            # 厳しい坂でペースが早いとき
            condition10 = (baselog["goal_slope"] == 3) & (baselog["race_position"].isin([3, 4])) & baselog["pace_category"].isin([3]) & (baselog["show"] == 1)
            baselog.loc[condition10, "score_goal_slope"] += (baselog["race_grade_scaled"] + 1) / 2.0
            
            # 厳しい坂ペースが遅いとき
            condition11 = (baselog["goal_slope"] == 3) & (baselog["race_position"].isin([1,2])) & baselog["pace_category"].isin([2]) & (baselog["show"] == 1)
            baselog.loc[condition11, "score_goal_slope"] += (baselog["race_grade_scaled"] + 1) / 2.0
        
            
            # 厳しい坂ペースそれ以外
            condition12 = (baselog["goal_slope"] == 3) & (baselog["race_position"].isin([1,2])) & baselog["pace_category"].isin([3,4]) & (baselog["show"] == 1)
            baselog.loc[condition12, "score_goal_slope"] += (baselog["race_grade_scaled"] + 1) / 3.0
        
            # 厳しい坂ペースそれ以外
            condition13 = (baselog["goal_slope"] == 3) & (baselog["race_position"].isin([3, 4])) & baselog["pace_category"].isin([1,2]) & (baselog["show"] == 1)
            baselog.loc[condition13, "score_goal_slope"] += (baselog["race_grade_scaled"] + 1) / 3.0
        
            
        
            
        
             # showに入りらなかった場合 
            condition14 = (baselog["show"] != 1)
            baselog.loc[condition14, "score_goal_slope"] += 0 
            
             # showに入りらなかった場合 
            condition15 = (~baselog["goal_slope"].isin([1, 2, 3])) & (baselog["show"] == 1)
            baselog.loc[condition15, "score_goal_slope"] += 0 
            
        
            return baselog
        result_df4 = calculate_scores(baselog)
        
        
        
            
        grouped_df = result_df4.groupby(["race_id", "horse_id"])
        merged_df4 = self.population.copy()
        
        for n_race in tqdm(n_races, desc=f"agg_goal_slope_per_score"):
            # スコアの集計（標準化しない版）
            raw_df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["score_goal_slope"]]
                .agg(["mean", "max", "min"])
            )
            raw_df.columns = [
                "_".join(col) + f"_{n_race}races_per_goal_slope_raw" for col in raw_df.columns
            ]
            
        
            # merged_df に追加
            merged_df4 = merged_df4.merge(raw_df, on=["race_id", "horse_id"], how="left")
        
        
        
        
        """
        特筆見る
        
        スタミナ指標
        かっているとして
        長ければ+
        小回りで-
        大回りでさらに+
        ⑤大回りのコースは、カーブで息が入りにくいので、距離に不安のある馬には厳しい
        
        
        race_grade*コースの長さ/2000
        
        こいつは相対値以外も出していい
        
        """
        
        
        baselog = (
            self.population
            .merge(
                self.horse_results,
                on="horse_id",
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        def calculate_scores(baselog):
            # スコア初期化
            baselog["score_stamina"] = 0.0
        
        
            # 急カーブ場合
            condition1 = (baselog["curve"] == 1) & (baselog["show"] == 1)
            baselog.loc[condition1, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *(((baselog["course_len"]*0.0025)+20)/20)*(((baselog["pace_diff"] * 1)+20)/20)/ 2.5
        
            # 小回りカーブで場合
            condition2 = (baselog["curve"] == 2)  & (baselog["show"] == 1)
            baselog.loc[condition2, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *(((baselog["course_len"]*0.0025)+20)/20)*(((baselog["pace_diff"] * 1)+20)/20)/ 2.1
            
            # 小スパカーブ場合
            condition3 = (baselog["curve"] == 3) & (baselog["show"] == 1)
            baselog.loc[condition3, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *(((baselog["course_len"]*0.0025)+20)/20)*(((baselog["pace_diff"] * 1)+20)/20)/ 1.8
        
            # スパカーブ場合
            condition4 = (baselog["curve"] == 4)  & (baselog["show"] == 1)
            baselog.loc[condition4, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori']))*(((baselog["course_len"]*0.0025)+20)/20)*(((baselog["pace_diff"] * 1)+20)/20)/ 1.5
        
            # 複合カーブで場合
            condition5 = (baselog["curve"] == 5)  & (baselog["show"] == 1)
            baselog.loc[condition5, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori']))*(((baselog["course_len"]*0.0025)+20)/20)*(((baselog["pace_diff"] * 1)+20)/20)
            
            
            
             # showに入りらなかった場合 
            condition7 = (baselog["show"] != 1)
            baselog.loc[condition7, "score_stamina"] += 0 
            
             # showに入りらなかった場合 
            condition8 = (~baselog["curve"].isin([1, 2, 3, 4])) & (baselog["show"] == 1)
            baselog.loc[condition8, "score_stamina"] += 0 
            
        
            return baselog
        result_df5 = calculate_scores(baselog)
        
        grouped_df = result_df5.groupby(["race_id", "horse_id"])
        merged_df5 = self.population.copy()
        
        for n_race in tqdm(n_races, desc=f"agg_stamina_per_score"):
            # スコアの集計（標準化しない版）
            raw_df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["score_stamina"]]
                .agg(["mean", "max", "min"])
            )
            raw_df.columns = [
                "_".join(col) + f"_{n_race}races_per_stamina_raw" for col in raw_df.columns
            ]
            
          
            # merged_df に追加
            merged_df5 = merged_df5.merge(raw_df, on=["race_id", "horse_id"], how="left")
            
        
        # merged_df5 = merged_df5.merge(self.race_info[["race_id","course_len"]], on="race_id")
        # merged_df5["course_len_relative"] = (merged_df5["course_len"] / 1600/7) +1
        # merged_df5.loc[merged_df5["course_len_relative"] < 0, "course_len_relative"] = 0
        # merged_df5["course_len_relative"] = merged_df5["course_len_relative"] 
        # # 対象の列名リスト
        # columns_to_multiply = [
        #     'score_stamina_mean_1races_per_stamina_raw',
        #     'score_stamina_max_1races_per_stamina_raw',
        #     'score_stamina_min_1races_per_stamina_raw',
        
        #     'score_stamina_mean_3races_per_stamina_raw',
        #     'score_stamina_max_3races_per_stamina_raw',
        #     'score_stamina_min_3races_per_stamina_raw',
        
        #     'score_stamina_mean_5races_per_stamina_raw',
        #     'score_stamina_max_5races_per_stamina_raw',
        #     'score_stamina_min_5races_per_stamina_raw',
        
        #     'score_stamina_mean_10races_per_stamina_raw',
        #     'score_stamina_max_10races_per_stamina_raw',
        #     'score_stamina_min_10races_per_stamina_raw',
        
        # ]
        
        # # 'course_len_relative'を各列に掛け算
        # for col in columns_to_multiply:
        #     merged_df5[col] = merged_df5[col] * merged_df5['course_len_relative']
        # # 'course_len'と'course_len_relative'の列を削除
        # merged_df5 = merged_df5.drop(columns=['course_len', 'course_len_relative'])
        
        
        results_df = (
            merged_df1
            .merge(
                merged_df2,
                on=("race_id","date","horse_id"),
            )
            .merge(
                merged_df3,
                on=("race_id","date","horse_id"),
            )
            .merge(
                merged_df4,
                on=("race_id","date","horse_id"),
            )
            .merge(
                merged_df5,
                on=("race_id","date","horse_id"),
            )
        )
        
        
        
        
        # 合計列を作成するレース数のリスト
        n_races = [1, 3, 5, 10]
        
        # "mean", "min", "max" のそれぞれに対して処理
        for agg_type in ["mean", "min", "max"]:
            for n_race in n_races:
                # 対象となる列をフィルタリング
                target_columns = [
                    col for col in results_df.columns if f"{agg_type}_{n_race}races" in col
                ]
                
                # 合計列を作成
                results_df[f"{agg_type}_{n_race}races_sum"] = results_df[target_columns].sum(axis=1)
        # 'column1' と 'column2' を削除
        results_df = results_df.drop(columns=["date",'score_pace_potision_mean_1races_per_score_raw', 'score_pace_potision_max_1races_per_score_raw', 'score_pace_potision_min_1races_per_score_raw', 'score_pace_potision_mean_3races_per_score_raw', 'score_pace_potision_max_3races_per_score_raw', 'score_pace_potision_min_3races_per_score_raw', 'score_pace_potision_mean_5races_per_score_raw', 'score_pace_potision_max_5races_per_score_raw', 'score_pace_potision_min_5races_per_score_raw', 'score_pace_potision_mean_10races_per_score_raw', 'score_pace_potision_max_10races_per_score_raw', 'score_pace_potision_min_10races_per_score_raw', 'score_goal_range_mean_1races_per_goal_raw', 'score_goal_range_max_1races_per_goal_raw', 'score_goal_range_min_1races_per_goal_raw', 'score_goal_range_mean_3races_per_goal_raw', 'score_goal_range_max_3races_per_goal_raw', 'score_goal_range_min_3races_per_goal_raw', 'score_goal_range_mean_5races_per_goal_raw', 'score_goal_range_max_5races_per_goal_raw', 'score_goal_range_min_5races_per_goal_raw', 'score_goal_range_mean_10races_per_goal_raw', 'score_goal_range_max_10races_per_goal_raw', 'score_goal_range_min_10races_per_goal_raw', 'score_curve_range_mean_1races_per_curve_raw', 'score_curve_range_max_1races_per_curve_raw', 'score_curve_range_min_1races_per_curve_raw', 'score_curve_range_mean_3races_per_curve_raw', 'score_curve_range_max_3races_per_curve_raw', 'score_curve_range_min_3races_per_curve_raw', 'score_curve_range_mean_5races_per_curve_raw', 'score_curve_range_max_5races_per_curve_raw', 'score_curve_range_min_5races_per_curve_raw', 'score_curve_range_mean_10races_per_curve_raw', 'score_curve_range_max_10races_per_curve_raw', 'score_curve_range_min_10races_per_curve_raw', 'score_goal_slope_mean_1races_per_goal_slope_raw', 'score_goal_slope_max_1races_per_goal_slope_raw', 'score_goal_slope_min_1races_per_goal_slope_raw', 'score_goal_slope_mean_3races_per_goal_slope_raw', 'score_goal_slope_max_3races_per_goal_slope_raw', 'score_goal_slope_min_3races_per_goal_slope_raw', 'score_goal_slope_mean_5races_per_goal_slope_raw', 'score_goal_slope_max_5races_per_goal_slope_raw', 'score_goal_slope_min_5races_per_goal_slope_raw', 'score_goal_slope_mean_10races_per_goal_slope_raw', 'score_goal_slope_max_10races_per_goal_slope_raw', 'score_goal_slope_min_10races_per_goal_slope_raw', 'score_stamina_mean_1races_per_stamina_raw', 'score_stamina_max_1races_per_stamina_raw', 'score_stamina_min_1races_per_stamina_raw', 'score_stamina_mean_3races_per_stamina_raw', 'score_stamina_max_3races_per_stamina_raw', 'score_stamina_min_3races_per_stamina_raw', 'score_stamina_mean_5races_per_stamina_raw', 'score_stamina_max_5races_per_stamina_raw', 'score_stamina_min_5races_per_stamina_raw', 'score_stamina_mean_10races_per_stamina_raw', 'score_stamina_max_10races_per_stamina_raw', 'score_stamina_min_10races_per_stamina_raw'])
        
        
        
        # population をコピーして merged_df を作成
        merged_df = self.population.copy()
        
        
        # スコアの集計（標準化）
        tmp_df = results_df.groupby(["race_id"])
        
        # 標準化されたデータを計算
        std_df = (
            (results_df.set_index(["race_id", "horse_id"]) - tmp_df.mean()) / tmp_df.std()
        ).add_suffix("_relative").reset_index()
        
        # merged_df に追加
        merged_df = merged_df.merge(results_df, on=["race_id", "horse_id"], how="left")
        merged_df = merged_df.merge(std_df, on=["race_id", "horse_id"], how="left")
        
        # 結果を確認
        merged_df = merged_df.drop(columns="horse_id_relative")

        self.agg_cross_features_df_13 = merged_df
        print("running cross_features_13()...comp")



    
    
    def cross_features_14(
        self, n_races: list[int] = [1, 3, 5, 10]
    ):  
        
        def calculate_race_position_percentage(group, n_race):
            """
            過去nレースにおける脚質割合を計算する
            """
            # 過去nレースに絞る
            past_races = group.head(n_race)
            
            # 各脚質のカウント
            counts = past_races['race_position'].value_counts(normalize=True).to_dict()
            
            # 結果を辞書形式で返す（割合が存在しない場合は0.0を補完）
            return {
                "escape": counts.get(1, 0.0),
                "taking_lead": counts.get(2, 0.0),
                "in_front": counts.get(3, 0.0),
                "pursuit": counts.get(4, 0.0),
            }
        
        # 過去nレースのリスト
        n_race_list = [1, 3, 5, 10]
        baselog = (
            self.population.merge(
                self.horse_results,
                on=["horse_id"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        
        # 集計用データフレームの初期化
        merged_df = self.population.copy()
        # grouped_dfを適用して計算
        grouped_df = baselog.groupby(["race_id", "horse_id"])
        # 各過去nレースの割合を計算して追加
        for n_race in n_race_list:
            
            position_percentage = grouped_df.apply(
                lambda group: calculate_race_position_percentage(group, n_race=n_race),
                include_groups=False,  # グループ列を除外
            )
            
            # 結果をデータフレームとして展開
            position_percentage_df = position_percentage.apply(pd.Series).reset_index()
            position_percentage_df.rename(
                columns={
                    "escape": f"escape_{n_race}races",
                    "taking_lead": f"taking_lead_{n_race}races",
                    "in_front": f"in_front_{n_race}races",
                    "pursuit": f"pursuit_{n_race}races",
                },
                inplace=True,
            )
            
            # 結果をマージ
            merged_df = merged_df.merge(position_percentage_df, on=["race_id", "horse_id"], how="left")
        
        
        # dominant_position_category を計算
        position_columns_5 = ["escape_5races", "taking_lead_5races", "in_front_5races", "pursuit_5races"]
        position_columns_3 = ["escape_3races", "taking_lead_3races", "in_front_3races", "pursuit_3races"]
        position_columns_1 = ["escape_1races", "taking_lead_1races", "in_front_1races", "pursuit_1races"]
        
        
        
        """
        未来の脚質を予想
        """
        def determine_dominant_position(row):
            # 5racesの中から計算
            if not row[position_columns_5].isnull().all():
                max_column = row[position_columns_5].idxmax()
                return {
                    "escape_5races": 1,
                    "taking_lead_5races": 2,
                    "in_front_5races": 3,
                    "pursuit_5races": 4,
                }[max_column]
            # 3racesの中から計算
            elif not row[position_columns_3].isnull().all():
                max_column = row[position_columns_3].idxmax()
                return {
                    "escape_3races": 1,
                    "taking_lead_3races": 2,
                    "in_front_3races": 3,
                    "pursuit_3races": 4,
                }[max_column]
            # 1racesの中から計算
            elif not row[position_columns_1].isnull().all():
                max_column = row[position_columns_1].idxmax()
                return {
                    "escape_1races": 1,
                    "taking_lead_1races": 2,
                    "in_front_1races": 3,
                    "pursuit_1races": 4,
                }[max_column]
            # すべて欠損の場合は1を返す
            else:
                return 3
        
        # 各行に対して dominant_position_category を適用
        merged_df["dominant_position_category"] = merged_df.apply(determine_dominant_position, axis=1)
        merged_df1 = merged_df
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #ペース作成
        """
        merged_df にはrace_id（レースのid）と、そのレースに出走するhorse_id(馬のid)の列があり、
        さらに dominant_position_categoryというその馬の脚質を表す列がある
        脚質は1-4の四種類の数字カテゴリが入っており
        レースごとに、出走する馬全体における、その脚質がいる割合と絶対数を求め、
        それぞれ四つの列を新たに作成
        """
        # dominant_position_category の脚質マッピング
        category_mapping_per = {1: "escape_per", 2: "taking_lead_per", 3: "in_front_per", 4: "pursuit_per"}
        category_mapping_count = {1: "escape_count", 2: "taking_lead_count", 3: "in_front_count", 4: "pursuit_count"}
        
        # 各レースで dominant_position_category の割合を計算
        position_ratios = (
            merged_df.groupby("race_id")["dominant_position_category"]
            .value_counts(normalize=True)  # 割合を計算
            .unstack(fill_value=0)         # 行: race_id, 列: dominant_position_category の形に
            .rename(columns=category_mapping_per)  # 列名を割合用に置き換え
        )
        
        # 各レースで dominant_position_category の絶対数を計算
        position_counts = (
            merged_df.groupby("race_id")["dominant_position_category"]
            .value_counts()                # 絶対数を計算
            .unstack(fill_value=0)         # 行: race_id, 列: dominant_position_category の形に
            .rename(columns=category_mapping_count)  # 列名を絶対数用に置き換え
        )
        
        # 割合と絶対数を結合
        position_data = position_ratios.join(position_counts)
        columns_to_remove = list(position_data.columns)
        merged_df = merged_df.drop(columns=columns_to_remove, errors="ignore")
        
        # 元の DataFrame にマージ
        merged_df = merged_df.merge(position_data, how="left", on="race_id")
        
        if 'pace_category' in merged_df.columns:
            merged_df = merged_df.drop(columns=['pace_category'])
        
        # ペースカテゴリを計算する関数
        def calculate_pace_category(row):
            escape_count = row.get("escape_count", 0)
            taking_lead_per = row.get("taking_lead_per", 0)
        
            if escape_count >= 3:
                return 4  # ハイペース
            elif escape_count == 2:
                if taking_lead_per > 0.5:
                    return 4  # ハイペース
                else:
                    return 3  
            elif escape_count == 1:
                if taking_lead_per > 0.6:
                    return 4  # ハイペース
                elif 0.5 < taking_lead_per <= 0.6:
                    return 3  # ハイミドルペース
                elif 0.4 < taking_lead_per <= 0.5:
                    return 2  # スローミドルペース
                else:
                    return 1  # スローペース
            elif escape_count == 0:
                # 逃げ馬が0の場合、先行馬の割合に基づく
                if taking_lead_per > 0.7:
                    return 4  # ハイペース
                elif 0.6 < taking_lead_per <= 0.7:
                    return 3  # ハイミドルペース
                elif 0.5 < taking_lead_per <= 0.6:
                    return 2  # スローミドルペース
                else:
                    return 1  # スローペース
            return None  # 予期しない場合は None
        merged_df["pace_category"] = merged_df.apply(calculate_pace_category, axis=1)
        
        merged_df2 = merged_df
        
        
        
        # merged_df = merged_df.merge(merged_df2[["race_id","date","horse_id","pace_category2"]],
        #                             on=["race_id","date","horse_id"],)
        
        
        # # merged_df のコピーを作成
        # new_columns = []
        
        # # merged_df に対して、1, 3, 5, 10 の各nに対して処理
        # for n in [1, 3, 5, 10]:  # nの値を1, 3, 5, 10とする
        #     for result_type in ["win", "rentai", "show"]:  # win, rentai, showの各列を処理
        #         # 新しい列名を設定
        #         new_col_name = f"{result_type}_for_pace_category_n{n}"
        #         new_columns.append(new_col_name)
        
        #         # nとresult_typeに基づいて、該当する列を選択
        #         for category in [1, 2, 3, 4]:  # pace_categoryに基づいて1から4のカテゴリを処理
        #             # 該当するpace_categoryとnを持つ元の列を動的に選択
        #             col_name = f"{result_type}_{n}races_per_pace_category{category}"
                    
        #             # merged_dfに新しい列を代入（pace_categoryに関わらず同じ列に結果を格納）
        #             merged_df.loc[merged_df["pace_category"] == category, new_col_name] = merged_df[col_name]
        # merged_df = merged_df[["race_id","horse_id",
        #     'win_for_pace_category_n1',
        #  'rentai_for_pace_category_n1',
        #  'show_for_pace_category_n1',
        #  'win_for_pace_category_n3',
        #  'rentai_for_pace_category_n3',
        #  'show_for_pace_category_n3',
        #  'win_for_pace_category_n5',
        #  'rentai_for_pace_category_n5',
        #  'show_for_pace_category_n5',
        #  'win_for_pace_category_n10',
        #  'rentai_for_pace_category_n10',
        #  'show_for_pace_category_n10']]
        
        # # race_id ごとに標準化（相対値）を計算
        # tmp_df = merged_df.groupby("race_id")
        
        # # 各レースごとに、平均と標準偏差を計算
        # # .transform を使って、同じサイズのデータフレームに変換
        # relative_df = (merged_df - tmp_df.transform("mean")) / tmp_df.transform("std")
        
        # # カラム名をわかりやすくするために "_relative" を追加
        # relative_df = relative_df.add_suffix("_relative")
        
        # # 元のデータフレームに標準化されたデータを結合
        # merged_df1000 = merged_df.join(relative_df, how="left")
        
        # merged_df1000.columns
        # merged_df1000 = merged_df1000[['race_id', 'horse_id', 'win_for_pace_category_n1',
        #        'rentai_for_pace_category_n1', 'show_for_pace_category_n1',
        #        'win_for_pace_category_n3', 'rentai_for_pace_category_n3',
        #        'show_for_pace_category_n3', 'win_for_pace_category_n5',
        #        'rentai_for_pace_category_n5', 'show_for_pace_category_n5',
        #        'win_for_pace_category_n10', 'rentai_for_pace_category_n10',
        #        'show_for_pace_category_n10', 
        #        'rentai_for_pace_category_n1_relative',
        #        'rentai_for_pace_category_n10_relative',
        #        'rentai_for_pace_category_n3_relative',
        #        'rentai_for_pace_category_n5_relative',
        #        'show_for_pace_category_n1_relative',
        #        'show_for_pace_category_n10_relative',
        #        'show_for_pace_category_n3_relative',
        #        'show_for_pace_category_n5_relative',
        #        'win_for_pace_category_n1_relative',
        #        'win_for_pace_category_n10_relative',
        #        'win_for_pace_category_n3_relative',
        #        'win_for_pace_category_n5_relative']]
        # merged_df4 = population.copy()
        # merged_df1000 = merged_df4.merge(merged_df1000, on=["race_id", "horse_id"], how="left")
        
        
        
        
        #直近レースから、馬場状態調査、race_resultsで行う
        """
        1/競馬場xタイプxランクx距離ごとの平均タイムと直近のレースとの差異の平均を特徴量に、そのまま数値として入れれる
        2/芝以外を除去
        3/競馬場x芝タイプで集計(グループバイ)
        4/"race_date_day_count"の当該未満かつ、800以内が一週間以内のデータになるはず
        5/その平均を取る
        6/+なら軽く、-なら重く、それぞれ5段階のカテゴリに入れる
        """
        
        merged_df = self.population.copy()        
        # df1 = (
        #     merged_df
        #     .merge(race_info[["race_id", "place","weather","ground_state","course_len","race_type","race_date_day_count"]], on="race_id",how="left")
        # )
        
        df = (
            merged_df
            .merge(self.race_info[["race_id", "place","weather","ground_state","course_len","race_type","race_date_day_count","course_type"]], on="race_id")
        )
        df["place"] = df["place"].astype(int)
        
        
        old_merged_df = self.population.copy()      
        
        
        df_old2 = (
            old_merged_df
            .merge(self.results[["race_id", "horse_id","time","rank","rank_per_horse"]], on=["race_id", "horse_id"])
            .merge(self.race_info[["race_id", "place","weather","ground_state","race_grade","course_len","race_type","race_date_day_count","course_type"]], on="race_id")
        )
        df_old2["place"] = df_old2["place"].astype(int)
        df_old2["race_grade"] = df_old2["race_grade"].astype(int)
        # 距離/競馬場/タイプ/レースランク
        df_old2["distance_place_type_race_grade"] = (df_old2["course_type"].astype(str) + df_old2["race_grade"].astype(str)).astype(int)
        
        
        baselog_old = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around","course_type"]], on="race_id"
            )
            # .merge(
            #     self.horse_results,
            #     on=["horse_id", "course_len", "race_type"],
            #     suffixes=("", "_horse"),
            # )
            # .query("date_horse < date")
            # .sort_values("date_horse", ascending=False)
        )
        df_old = (
            baselog_old
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "umaban","nobori","rank","time","sex"]], on=["race_id", "horse_id"])
        )
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        df_old["around"] = df_old["around"].fillna(3).astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  
                     
        
        df_old["distance_place_type_race_grade"] = (df_old["course_type"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
        df_old_copy = df_old
        # rank列が1, 2, 3の行だけを抽出
        df_old = df_old[df_old['rank'].isin([1, 2, 3,4,5])]
        
        target_mean_1 = df_old.groupby("distance_place_type_race_grade")["time"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old = df_old.copy()
        df_old["distance_place_type_race_grade_encoded"] = df_old["distance_place_type_race_grade"].map(target_mean_1)
        
        
        df_old = df_old[["distance_place_type_race_grade",'distance_place_type_race_grade_encoded']]
        
        
        columns_to_merge = [
            ("distance_place_type_race_grade",'distance_place_type_race_grade_encoded'),
        ]
        
        # 各ペアを順番に処理
        for original_col, encoded_col in columns_to_merge:
            df2_subset = df_old[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
            df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
            df_old2 = df_old2.merge(df2_subset, on=original_col, how='left')  # dfにマージ
        df_old2 = df_old2[df_old2['rank'].isin([1, 2, 3,4,5])]    
        df_old2["distance_place_type_race_grade_encoded_time_diff"] = df_old2['distance_place_type_race_grade_encoded'] - df_old2["time"]
        
        # df_old2= df_old2[df_old2["race_type"] != 2]
        # df_old2_1 = df_old2[df_old2["race_type"] != 0]
        df_old2_1 = df_old2 
        # 2. df の各行について処理
        def compute_mean_for_row(row, df_old2_1):
            # race_type == 0 の場合は NaN を返す
            # if row["race_type"] == 0:
            #     return np.nan
                
            target_day_count = row["race_date_day_count"]  # df の各行の race_date_day_count
        
            # 3. df_old2_1 から条件に一致する行をフィルタリング
            filtered_df_old2_1 = df_old2_1[
                (df_old2_1["race_date_day_count"] >= (target_day_count - 400)) &  # race_date_day_count が target_day_count-1200 以上
                (df_old2_1["race_date_day_count"] <= (target_day_count - 1)) &  # race_date_day_count が target_day_count-1 以下
                (df_old2_1["place"] == row["place"]) &  # place が一致
                (df_old2_1["race_type"] == row["race_type"]) & 
                (df_old2_1["weather"].isin([0, 1, 2])) &  # weather が 0, 1, 2 のいずれか
                (df_old2_1["ground_state"] == 0)   # ground_state が 0
                # &
                # (df_old2_1["rank_per_horse"] < 0.87)  # rank_per_horse が 0.87 未満
            ]
        
            filtered_df_old2_2 = df_old2_1[
                (df_old2_1["race_date_day_count"] >= (target_day_count - 1200)) &  # race_date_day_count が target_day_count-1200 以上
                (df_old2_1["race_date_day_count"] <= (target_day_count - 1 )) &  # race_date_day_count が target_day_count-1 以下
                (df_old2_1["place"] == row["place"]) &  # place が一致
                (df_old2_1["race_type"] == row["race_type"])  &
                (df_old2_1["weather"].isin([0, 1, 2])) &  # weather が 0, 1, 2 のいずれか
                (df_old2_1["ground_state"] == 0)   # ground_state が 0                    
                # &
                # (df_old2_1["rank_per_horse"] < 0.87)  # rank_per_horse が 0.87 未満
            ]
            # 4. フィルタリングした行の "distance_place_type_race_grade_encoded_time_diff" の平均を計算
            mean_time_diff = filtered_df_old2_1["distance_place_type_race_grade_encoded_time_diff"].mean()

            # 5. 計算結果を返す（NaNの場合も考慮）
            # return mean_time_diff if not np.isnan(mean_time_diff) else filtered_df_old2_2["distance_place_type_race_grade_encoded_time_diff"].mean()
            return np.nan_to_num(mean_time_diff, nan=np.nan_to_num(filtered_df_old2_2["distance_place_type_race_grade_encoded_time_diff"].mean()))

            # # 4. フィルタリングした行の "distance_place_type_race_grade_encoded_time_diff" の平均を計算
            # mean_time_diff = filtered_df_old2_1["distance_place_type_race_grade_encoded_time_diff"].mean()
        
            # # 5. 計算結果を返す（NaNの場合も考慮）
            # return mean_time_diff if not np.isnan(mean_time_diff) else np.nan
        
        # 6. df の各行に対して、計算した平均値を新しい列に追加
        df["mean_ground_state_time_diff"] = df.apply(compute_mean_for_row, axis=1, df_old2_1=df_old2_1)
        
        
        
        def assign_value(row):
            # weather が 3, 4, 5 または ground_state が 0 以外の場合は 5 を設定
            if row["ground_state"] in [3]:
                return 0
            if row["weather"] in [3, 4, 5]:
                return 1

            if row["ground_state"] in [1]:
                return 1
            if row["ground_state"] in [2]:
                if row["race_type"] in [0]:
                    # mean_ground_state_time_diff に基づいて分類
                    if 2.0 <= row["mean_ground_state_time_diff"]:
                        return 3.7  #　超高速馬場1
                    elif 1 <= row["mean_ground_state_time_diff"] < 2.0:
                        return 4.5  # 高速馬場2
                    
                    elif 0.4 <= row["mean_ground_state_time_diff"] < 1:
                        return 5  # 軽い馬場3
                        
                    elif -0.4 <= row["mean_ground_state_time_diff"] < 0.4:
                        return 5.5  # 標準的な馬場4
                    elif -1 <= row["mean_ground_state_time_diff"] < -0.4:
                        return 6.2  # やや重い馬場5
                    
                    elif -2 <= row["mean_ground_state_time_diff"] < -1:
                        return 7  # 重い馬場5
                    
                    elif row["mean_ground_state_time_diff"] < -2:
                        return 8  # 道悪7
                if row["race_type"] in [1,2]:
                    return 2.7
            if row["ground_state"] in [2]:
                return 2.7
            # mean_ground_state_time_diff に基づいて分類
            if 2.0 <= row["mean_ground_state_time_diff"]:
                return 2.7  #　超高速馬場1
            elif 1 <= row["mean_ground_state_time_diff"] < 2.0:
                return 3.5  # 高速馬場2
            
            elif 0.4 <= row["mean_ground_state_time_diff"] < 1:
                return 3.8  # 軽い馬場3
                
            elif -0.4 <= row["mean_ground_state_time_diff"] < 0.4:
                return 4  # 標準的な馬場4
        
            
            elif -1 <= row["mean_ground_state_time_diff"] < -0.4:
                return 4.5  # やや重い馬場5
            
            elif -2 <= row["mean_ground_state_time_diff"] < -1:
                return 5.2  # 重い馬場5
            
            elif row["mean_ground_state_time_diff"] < -2:
                return 6  # 道悪7
            # 該当しない場合は NaN を設定
            return 3.5
        
        # 新しい列を追加
        df["ground_state_level"] = df.apply(assign_value, axis=1)
        
        merged_df = merged_df.merge(df[["race_id","date","horse_id","mean_ground_state_time_diff","ground_state_level"]],on=["race_id","date","horse_id"])	
        merged_df3 = merged_df
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # merged_df = population.copy()
        
        # """
        # "pace_category"にある1234という数字カテゴリごとにwin,rentai,showを
        # 新たな列が必要
        
        # baselogの"pace_category"==が1,2,3,4ごとにそれぞれ分ける
        # baselog1,2,3,4ごとに、win,rentai,showのn_racesごとの平均を算出
        # 最終的にbaselog1,2,3,4をconcatする
        
        # """
        # # pace_category ごとにデータをフィルタリング
        # baselog1 = baselog[baselog["pace_category"] == 1]
        # baselog2 = baselog[baselog["pace_category"] == 2]
        # baselog3 = baselog[baselog["pace_category"] == 3]
        # baselog4 = baselog[baselog["pace_category"] == 4]
        # n_races: list[int] = [1, 3, 5, 10]
        # # 結果を格納するためのリスト
        # result_dfs = []
        
        # # 各pace_categoryに対してn_racesごとの平均を計算
        # for pace_category, df1 in zip([1, 2, 3, 4], [baselog1, baselog2, baselog3, baselog4]):
        #     grouped_df = df1.groupby(["race_id", "horse_id"])
            
        #     for n_race in tqdm(n_races, desc=f"pace_category_win_{pace_category}"):
        #         # n_raceに基づいて集計
        #         agg_df = (
        #             grouped_df.head(n_race)
        #             .groupby(["race_id", "horse_id"])[["win", "rentai", "show"]]
        #             .agg("mean")
        #         )
                
        #         # 列名を修正
        #         agg_df.columns = [
        #             f"{col}_{n_race}races_per_pace_category{pace_category}" for col in agg_df.columns
        #         ]
                
        #         # # 平均と標準偏差を基に相対値を計算
        #         # tmp_df = agg_df.groupby(["race_id"])
        #         # relative_df = ((agg_df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
                
        #         # # 生成したデータフレームをresult_dfsに追加
        #         # result_dfs.append(relative_df)
        #         result_dfs.append(agg_df)
        
        
        # # 結果をconcatして1つのデータフレームにまとめる
        # final_agg_df = pd.concat(result_dfs, axis=1)
        
        # # merge_dfとマージ
        # merged_df = merged_df.merge(final_agg_df, on=["race_id", "horse_id"], how="left")
        # merged_df4 = merged_df
        


        merged_df100 = merged_df2.merge(
            merged_df3,
            on=["race_id","date","horse_id"],
        )
        merged_df_all = merged_df100[["race_id","date","horse_id","dominant_position_category","pace_category","ground_state_level"]]
        self.pace_category = merged_df_all 



        """
        dominant_position_categoryが当該レースの脚質予想
        pace_categoryが当該レースのペース予想
        ground_state_levelが当日の馬場状態
        
        
        ＿＿ハイペース1→4
        
        ミドルハイペース2→3
        
        ミドルローペース3→2
        
        ＿＿ローペース4→1
        
        
        馬場レベル	タイム差
        超高速馬場1	−2.0以上
        高速馬場2
        	−1.0以上
        やや軽い3
        	−0.40〜−0.99
        標準的な馬場4	0〜−0.39
        標準的な馬場4
        	0〜+0.39
        やや重い馬場5	+0.40〜+0.99
        重い馬場6	+1.0秒以上
        道悪馬場7	+2.0秒以上
        
        逃げ、1
        追い込み4



        軽い芝、先行有利
        重い芝、差し有利

        オール野芝 ▶︎最も軽い芝で、速い時計が出やすい
        軽い芝 ▶︎時計面では野芝に劣るものの、こちらも速い時計が出やすい
        オーバーシード ▶︎ケースバイケース
        重い芝 ▶︎馬場が重く、時計もかかりやすい
        非常に重い芝 ▶︎JRA10場の中で最も重く、時計もかかりやすい
        5
        4
        3
        2
        1
        これを逆転させて、

        # 条件に基づく変換
        df = df.copy()
        df.loc[:, "place_season_condition_type_categori_processed"] = df["place_season_condition_type_categori"].apply(
            lambda x: (x + 2 if x == '-' else (x - 3)) if isinstance(x, str) else x
        )
        
        # その後で1/1.7で割る
        df["place_season_condition_type_categori_processed_1"] = (df["place_season_condition_type_categori_processed"]+20) / 20

        
        高速馬場の方が逃げ先行が有利なので、ground_state_level列に-4を行い、先行有利を-側へ
        ground_state_levelには欠損値があるため、それは4扱いにし、-4で0になるようにすること
        ローペースの方が逃げ先行有利なので、pace_category列に-2.5を行い、先行有利を-側へ
        先行逃げを-側にしたいため、dominant_position_categoryが1,2のものを-2に、3を2に、4を1.8にそれぞれ変換する
        
        まずこれだけの列を作成する
        影響を大きくしたい場合、それぞれの列に倍率を調整できるようにする
        
        "goal_range_100"は-3.5を行い、+の場合は全て0に変換する
        
        "curve"は-4.5を行い、+の場合は数値を8倍する
        
        "goal_slope"は-1を行う、pace_categoryに-をかける、pace_categoryと掛け合わせる
        #     "平坦": 1,        
        #     "緩坂": 2,
        #     "急坂": 3

        ③内枠は基本的に、先行馬にとって有利で、差し馬には不利である
        ④外枠は基本的に、差し馬に有利で、先行馬には不利である
        ⑤中枠は内と外の特徴をマイルドにしたもので、どの脚質でも柔軟に対応可能

        距離が長いほど、影響を受けやすい
        """
        
        merged_df_all = merged_df100[["race_id","date","horse_id","dominant_position_category","pace_category","ground_state_level"]]
        merged_df_all = merged_df_all.merge(
                self.race_info[["race_id", "goal_range_100","curve","goal_slope","course_len","place_season_condition_type_categori","start_slope","start_range","race_grade","race_type"]], 
                on="race_id",
            )
        

        merged_df_all = merged_df_all.merge(
                self.results[["race_id","horse_id","umaban","umaban_odd"]], 
                on=["horse_id","race_id"],
            )

        # 必要に応じてコピーを作成
        merged_df_all = merged_df_all.copy()
        merged_df_all_umaban = merged_df_all[["race_id","umaban"]]

        tmp_df_umaban = merged_df_all_umaban.groupby("race_id")
        mean_umaban = tmp_df_umaban["umaban"].transform("mean")
        std_umaban = tmp_df_umaban["umaban"].transform("std")

        merged_df_all_umaban = merged_df_all_umaban.copy()
        merged_df_all_umaban.loc[:, "umaban_relative"] = (
            (merged_df_all_umaban["umaban"] - mean_umaban) / std_umaban
        )


        merged_df_all = merged_df_all.merge(
            merged_df_all_umaban[["race_id","umaban", "umaban_relative"]], on=["race_id","umaban"], how="left"
        )

        merged_df_all = merged_df_all.copy()
        merged_df_all.loc[:, "place_season_condition_type_categori_processed_z"] = (
            merged_df_all["place_season_condition_type_categori"]
            .replace({5: -0.43, 4: -0.19, 3: 0, 2: 0.19,1: 0.43, -1: -0.23, -2: 0, -3: 0.23,-4:0.43,-10000:0})
        ).astype(float)



        merged_df_all["start_range_processed"] = ((merged_df_all["start_range"] /360)-1)*5
        # 1600m未満ならそのまま、1600m以上なら緩やかに上昇 
        merged_df_all["start_range_processed"] = merged_df_all["start_range_processed"].apply(
            lambda x: x * 1.3 if x <= 0 else x*0.3
        )


        merged_df_all["start_slope_processed"] = (merged_df_all["start_slope"] * -1)/1.2


        merged_df_all["race_grade_processed"] = (merged_df_all["race_grade"] - 75)/20
        merged_df_all["umaban_odd_processed"] = (merged_df_all["umaban_odd"] - 1)/2
        merged_df_all["race_type_processed"] = (merged_df_all["race_type"] - 1)



        # # 1600で正規化
        merged_df_all["course_len_processed"] = (merged_df_all["course_len"] / 1800)

        # 1600m未満ならそのまま、1600m以上なら緩やかに上昇 
        merged_df_all["course_len_processed"] = merged_df_all["course_len_processed"].apply(
            lambda x: ((x+1)/2) if x <= 1 else (((1 + ((x - 1) / 5))+2)/3)
        )


        # 1600で正規化
        merged_df_all["course_len_processed_0"] = (merged_df_all["course_len"] / 1800)-1

        # 1600m未満ならそのまま、1600m以上なら緩やかに上昇 
        merged_df_all["course_len_processed_0"] = merged_df_all["course_len_processed_0"].apply(
            lambda x: x * 2 if x <= 0 else x*0.7
        )

        
        # ground_state_level_processed 列の処理
        merged_df_all.loc[:, "ground_state_level_processed"] = (
            merged_df_all["ground_state_level"].fillna(4) - 4
        ).astype(float)
        
        # pace_category_processed 列の処理
        merged_df_all.loc[:, "pace_category_processed"] = (
            merged_df_all["pace_category"] - 2.5
        ).astype(float)
        
        # dominant_position_category_processed 列の処理
        merged_df_all.loc[:, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.93, 2: -2, 3: 1.87, 4: 1.57})
        ).astype(float)
        
        # goal_range_100 に -3.5 を行う
        merged_df_all["goal_range_100_processed"] = merged_df_all["goal_range_100"] - 3.6
        # # プラスの値をすべて 0 に変換
        # merged_df_all["goal_range_100_processed"] = merged_df_all["goal_range_100_processed"].clip(upper=0)
        merged_df_all.loc[merged_df_all["goal_range_100_processed"] > 0, "goal_range_100_processed"] *= 0.7



        # -4.5 を行う
        merged_df_all["curve_processed"] = merged_df_all["curve"] - 4.5
        # +の場合は数値を8倍する
        merged_df_all["curve_processed"] = merged_df_all["curve_processed"].apply(
            lambda x: x * 8 if x > 0 else x
        )
        
        
        # goal_slope に -1 を行う
        merged_df_all["goal_slope_processed"] = merged_df_all["goal_slope"]
        
        # pace_category に - をかけて符号反転
        merged_df_all["pace_category_processed"] = (merged_df_all["pace_category"] * -1)*3
        
        # goal_slope_processed と pace_category_processed を掛け合わせる
        merged_df_all["goal_slope_processed"] = merged_df_all["goal_slope_processed"] * merged_df_all["pace_category_processed"]
        
        
        
        # それぞれの列に倍率を適用
        ground_state_multiplier = 4.37 # 必要に応じて調整
        pace_multiplier = 7  # 必要に応じて調整
        dominant_multiplier = 2.8  # 必要に応じて調整
        goal_range_multiplier = 21
        curve_multiplier = 5
        goal_slope_multiplier = 1.8

        
        merged_df_all["dominant_position_category_processed"] = merged_df_all["dominant_position_category_processed"].astype(float)
        merged_df_all["goal_range_100_processed"] = merged_df_all["goal_range_100_processed"].astype(float)
        merged_df_all["goal_slope_processed"] = merged_df_all["goal_slope_processed"].astype(float)
        merged_df_all["pace_category_processed"] = merged_df_all["pace_category_processed"].astype(float)


        merged_df_all["umaban"] = merged_df_all["umaban"]-8
        merged_df_all["umaban"] = merged_df_all["umaban"].apply(
            lambda x: x * 1/2.1 if x < 0 else x
        )
        merged_df_all["umaban_relative"] = merged_df_all["umaban"] + (merged_df_all["umaban_relative"]*3.8)
        merged_df_all.loc[:, "dominant_position_category_processed"] += (merged_df_all["umaban_relative"]/20)
        # merged_df_all.loc[:, "dominant_position_category_processed"] += merged_df_all["course_len_processed_0"]



        # merged_df_all.loc[:, "dominant_position_category_processed"] *= merged_df_all["course_len_processed"]



        merged_df_all.loc[:, "ground_state_level_processed"]+= merged_df_all["start_range_processed"]
        merged_df_all.loc[:, "ground_state_level_processed"]+= merged_df_all["start_slope_processed"]
        merged_df_all.loc[:, "ground_state_level_processed"]+= merged_df_all["race_grade_processed"] 
        merged_df_all.loc[:, "ground_state_level_processed"]+=  merged_df_all["umaban_odd_processed"] 
        merged_df_all.loc[:, "ground_state_level_processed"]+=   merged_df_all["race_type_processed"]

        merged_df_all.loc[:, "ground_state_level_processed"]+= merged_df_all["course_len_processed_0"]
        merged_df_all.loc[:, "ground_state_level_processed"] += merged_df_all["place_season_condition_type_categori_processed_z"]
        merged_df_all.loc[:, "ground_state_level_processed"] *= merged_df_all["course_len_processed"]

        merged_df_all.loc[:, "ground_state_level_processed"] *= ground_state_multiplier
        merged_df_all.loc[:, "pace_category_processed"] *= pace_multiplier
        merged_df_all.loc[:, "dominant_position_category_processed"] *= dominant_multiplier
        merged_df_all.loc[:, "goal_range_100_processed"] *= goal_range_multiplier
        merged_df_all.loc[:, "curve_processed"] *= curve_multiplier
        merged_df_all.loc[:, "goal_slope_processed"] *= goal_slope_multiplier
        

        # 加算と乗算による列の計算
        merged_df_all.loc[:, "tenkai_sumed"] = (
            merged_df_all["ground_state_level_processed"]
            + merged_df_all["pace_category_processed"]
        )
        
        merged_df_all.loc[:, "tenkai_combined"] = (
            merged_df_all["tenkai_sumed"]
            * merged_df_all["dominant_position_category_processed"]
        )
        
        
        
        
        #直線の長さを追加
        merged_df_all.loc[:, "tenkai_goal_range_sumed"] = (
            merged_df_all["tenkai_sumed"]
            + merged_df_all["goal_range_100_processed"]
        )
        
        merged_df_all.loc[:, "tenkai_goal_range_combined"] = (
            merged_df_all["tenkai_goal_range_sumed"]
            * merged_df_all["dominant_position_category_processed"]
        )
        
        
        #カーブの長さを追加
        merged_df_all.loc[:, "tenkai_curve_sumed"] = (
            merged_df_all["tenkai_sumed"]
            + merged_df_all["curve_processed"]
        )
        
        merged_df_all.loc[:, "tenkai_curve_combined"] = (
            merged_df_all["tenkai_curve_sumed"]
            * merged_df_all["dominant_position_category_processed"]
        )
        
        
        
        #坂を追加
        merged_df_all.loc[:, "tenkai_goal_slope_sumed"] = (
            merged_df_all["tenkai_sumed"]
            + merged_df_all["goal_slope_processed"]
        )
        
        merged_df_all.loc[:, "tenkai_goal_slope_combined"] = (
            merged_df_all["tenkai_goal_slope_sumed"]
            * merged_df_all["dominant_position_category_processed"]
        )
        
        
        
        #直線_カーブ
        merged_df_all.loc[:, "tenkai_goal_range_curve_sumed"] = (
            merged_df_all["tenkai_goal_range_sumed"]
            + merged_df_all["curve_processed"]
        )
        merged_df_all.loc[:, "tenkai_goal_range_curve_combined"] = (
            merged_df_all["tenkai_goal_range_curve_sumed"]
            * merged_df_all["dominant_position_category_processed"]
        )
        
        
        #直線_坂
        merged_df_all.loc[:, "tenkai_goal_range_goal_slope_sumed"] = (
            merged_df_all["tenkai_goal_range_sumed"]
            + merged_df_all["goal_slope_processed"]
        )
        merged_df_all.loc[:, "tenkai_goal_range_goal_slope_combined"] = (
            merged_df_all["tenkai_goal_range_goal_slope_sumed"]
            * merged_df_all["dominant_position_category_processed"]
        )
        
        
        
        
        #坂_カーブ
        merged_df_all.loc[:, "tenkai_curve_goal_slope_sumed"] = (
            merged_df_all["tenkai_curve_sumed"]
            + merged_df_all["goal_slope_processed"]
        )
        merged_df_all.loc[:, "tenkai_curve_goal_slope_combined"] = (
            merged_df_all["tenkai_curve_goal_slope_sumed"]
            * merged_df_all["dominant_position_category_processed"]
        )
        
        
        #全部載せ
        
        merged_df_all.loc[:, "tenkai_all_sumed"] = (
            merged_df_all["tenkai_curve_goal_slope_sumed"]
            + merged_df_all["goal_range_100_processed"]
        )
        merged_df_all.loc[:, "tenkai_all_combined"] = (
            merged_df_all["tenkai_all_sumed"]
            * merged_df_all["dominant_position_category_processed"]
        )
        
        # 各列に定数を掛ける
        merged_df_all["tenkai_combined"] *= 1/60
        merged_df_all["tenkai_goal_range_combined"] *= 1/90
        merged_df_all["tenkai_curve_combined"] *= 1/90
        merged_df_all["tenkai_goal_slope_combined"] *= 1/90
        merged_df_all["tenkai_goal_range_curve_combined"] *= 1/120
        merged_df_all["tenkai_goal_range_goal_slope_combined"] *= 1/120
        merged_df_all["tenkai_curve_goal_slope_combined"] *= 1/120
        merged_df_all["tenkai_all_combined"] *= 1/180        
        
        
        # _combined系の列をリストで指定
        combined_columns = [
            "tenkai_combined","tenkai_goal_range_combined", "tenkai_curve_combined", "tenkai_goal_slope_combined", 
            "tenkai_goal_range_curve_combined", "tenkai_goal_range_goal_slope_combined", 
            "tenkai_curve_goal_slope_combined", "tenkai_all_combined"
        ]
        
        # 各_combined列を標準化
        for col in combined_columns:
            merged_df_all[f"{col}_standardized"] = (
                merged_df_all[col] - merged_df_all.groupby("race_id")[col].transform("mean")
            ) / merged_df_all.groupby("race_id")[col].transform("std")
        
        
        
        
        
        df_x = (
            self.population.merge(
                self.horse_results,
                on=["horse_id"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        
        df_x = df_x[["race_id","date","horse_id","date_horse","race_grade","rank_diff","course_len","pace_diff"]]
        
        
        df_x = df_x.copy()  # df_xのコピーを作成
        df_x.loc[:, "race_grade_rank_diff_multi"] = ((df_x["race_grade"]) * ( 1/((((df_x["rank_diff"] + 1)+10)/10)*(((df_x["course_len"]*0.0025)+20)/20)*(((df_x["pace_diff"] * 1)+20)/20))))
        df_x.loc[:, "race_grade_rank_diff_sum"] = (((df_x["race_grade"])/10) +( 1/((((df_x["rank_diff"] + 1)+10)/10)*(((df_x["course_len"]*0.0025)+20)/20)*(((df_x["pace_diff"] * 1)+20)/20))))*10
        
        n_races: list[int] = [1, 3, 5, 10]
        grouped_df = df_x.groupby(["race_id", "horse_id"])
        merged_x = self.population.copy() 
        for n_race in tqdm(n_races, desc=f"agg_raceclass_rankdiff_per_score"):
            raw_df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["race_grade_rank_diff_sum","race_grade_rank_diff_multi"]]
                .agg(["mean", "max", "min"])
            )
            raw_df.columns = [
                "_".join(col) + f"_{n_race}races_grade_rankdiff_score" for col in raw_df.columns
            ]
            std_df = raw_df.copy()  # 同じデータを使用して標準化
            tmp_df = raw_df.groupby(["race_id"])
            std_df = ((raw_df - tmp_df.mean()) / tmp_df.std()).add_suffix("_standardized")
            
            # merged_df に追加
            merged_x = merged_x.merge(raw_df, on=["race_id", "horse_id"], how="left")
            merged_x = merged_x.merge(std_df, on=["race_id", "horse_id"], how="left")
        
        merge_all_ex = merged_df_all.merge(
            merged_x,
            on = ["horse_id","date","race_id"],
        )
        # 対象となる列をリストに格納
        columns_to_multiply = [
            'race_grade_rank_diff_sum_mean_1races_grade_rankdiff_score', 
            'race_grade_rank_diff_sum_max_1races_grade_rankdiff_score', 
            'race_grade_rank_diff_sum_min_1races_grade_rankdiff_score', 
            'race_grade_rank_diff_sum_mean_3races_grade_rankdiff_score', 
            'race_grade_rank_diff_sum_max_3races_grade_rankdiff_score', 
            'race_grade_rank_diff_sum_min_3races_grade_rankdiff_score', 
            'race_grade_rank_diff_sum_mean_5races_grade_rankdiff_score', 
            'race_grade_rank_diff_sum_max_5races_grade_rankdiff_score', 
            'race_grade_rank_diff_sum_min_5races_grade_rankdiff_score', 
            'race_grade_rank_diff_sum_mean_10races_grade_rankdiff_score', 
            'race_grade_rank_diff_sum_max_10races_grade_rankdiff_score', 
            'race_grade_rank_diff_sum_min_10races_grade_rankdiff_score',

            'race_grade_rank_diff_multi_mean_1races_grade_rankdiff_score', 
            'race_grade_rank_diff_multi_max_1races_grade_rankdiff_score', 
            'race_grade_rank_diff_multi_min_1races_grade_rankdiff_score', 
            'race_grade_rank_diff_multi_mean_3races_grade_rankdiff_score', 
            'race_grade_rank_diff_multi_max_3races_grade_rankdiff_score', 
            'race_grade_rank_diff_multi_min_3races_grade_rankdiff_score', 
            'race_grade_rank_diff_multi_mean_5races_grade_rankdiff_score', 
            'race_grade_rank_diff_multi_max_5races_grade_rankdiff_score', 
            'race_grade_rank_diff_multi_min_5races_grade_rankdiff_score', 
            'race_grade_rank_diff_multi_mean_10races_grade_rankdiff_score', 
            'race_grade_rank_diff_multi_max_10races_grade_rankdiff_score', 
            'race_grade_rank_diff_multi_min_10races_grade_rankdiff_score'

        ]
        
        # # `tenkai_combined` と `tenkai_all_combined` を掛け算し、新しい列を作成
        # for col in columns_to_multiply:
        #     # 'tenkai_combined' と 'tenkai_all_combined' を掛け合わせる
        #     merge_all_ex[f"{col}_tenkai_combined"] = merge_all_ex[col] + merge_all_ex["tenkai_combined"]
        #     merge_all_ex[f"{col}_tenkai_all_combined"] = merge_all_ex[col] + merge_all_ex["tenkai_all_combined"]
        
        # # 標準化
        # combined_columns = [
        #     f"{col}_tenkai_combined" for col in columns_to_multiply
        # ] + [
        #     f"{col}_tenkai_all_combined" for col in columns_to_multiply
        # ]
        
        # for col in combined_columns:
        #     merge_all_ex[f"{col}_standardized"] = (
        #         merge_all_ex[col] - merge_all_ex.groupby("race_id")[col].transform("mean")
        #     ) / merge_all_ex.groupby("race_id")[col].transform("std")
        
        
        
        # # ドロップする列をリストに格納
        # columns_to_drop = [
        #     'dominant_position_category', 'pace_category', 'ground_state_level', 
        #     'goal_range_100', 'curve', 'goal_slope', 
        #     'ground_state_level_processed', 'pace_category_processed', 
        #     'dominant_position_category_processed', 'goal_range_100_processed', 
        #     'curve_processed', 'goal_slope_processed'
        # ]
        
        # # 指定された列を削除
        # merge_df_all = merge_all_ex.drop(columns=columns_to_drop)
        
        # # 'race_id', 'date', 'horse_id' を除いた列のリストを作成
        # columns_to_drop = [
        #     col for col in merge_all_ex.columns 
        #     if not col.endswith('_standardized') and col not in ['race_id', 'date', 'horse_id']
        # ]
        
        # # 指定された列を削除
        # merge_all_ex = merge_all_ex.drop(columns=columns_to_drop)
        
        
        # # 'race_id', 'date', 'horse_id' を除いた列のリストを作成
        # columns_to_drop = [
        #     col for col in merge_all_ex.columns 
        #     if not col.endswith('_standardized') and col not in ['race_id', 'date', 'horse_id']
        # ]
        
        # # 指定された列を削除
        # merge_all_ex = merge_all_ex.drop(columns=columns_to_drop)
        

        
        # self.agg_cross_features_df_14 = merge_all_ex 
        # print("running cross_features_14()...comp")

        # `tenkai_combined` と `tenkai_all_combined` を掛け算し、新しい列を作成
        for col in columns_to_multiply:
            merge_all_ex[f"{col}_plus_tenkai_combined"] = merge_all_ex[col] + (merge_all_ex["tenkai_combined"]*13)
            merge_all_ex[f"{col}_plus_tenkai_all_combined"] = merge_all_ex[col] + (merge_all_ex["tenkai_all_combined"]*13)
            merge_all_ex[f"{col}_px_tenkai_combined"] = merge_all_ex[col]*(((merge_all_ex["tenkai_combined"]+7))/7)
            merge_all_ex[f"{col}_px_tenkai_all_combined"] = merge_all_ex[col]  *  (((merge_all_ex["tenkai_all_combined"]+7))/7)

        # 標準化
        combined_columns = [
            f"{col}_plus_tenkai_combined" for col in columns_to_multiply
        ] + [
            f"{col}_plus_tenkai_all_combined" for col in columns_to_multiply
        ]+ [
            f"{col}_px_tenkai_combined" for col in columns_to_multiply
        ]+ [
            f"{col}_px_tenkai_all_combined" for col in columns_to_multiply
        ]

        # 統計量の事前計算
        stats = merge_all_ex.groupby("race_id")[combined_columns].agg(['mean', 'std']).reset_index()
        stats.columns = ['race_id'] + [f"{col}_{stat}" for col, stat in stats.columns[1:]]
        merge_all_ex = merge_all_ex.merge(stats, on='race_id', how='left')

        # 標準化
        for col in combined_columns:
            merge_all_ex[f"{col}_standardized"] = (
                merge_all_ex[col] - merge_all_ex[f"{col}_mean"]
            ) / merge_all_ex[f"{col}_std"]

        # 不要な列を削除
        columns_to_drop = [
            'dominant_position_category', 'pace_category', 'ground_state_level', 
            'goal_range_100', 'curve', 'goal_slope', 
            'ground_state_level_processed', 'pace_category_processed', 
            'dominant_position_category_processed', 'goal_range_100_processed', 
            'curve_processed', 'goal_slope_processed'
        ]
        merge_all_ex = merge_all_ex.drop(columns=columns_to_drop)

        # '_standardized', 'race_id', 'date', 'horse_id' を除くすべての列を削除
        merge_all_ex = merge_all_ex.filter(regex='_standardized|race_id|date|horse_id')

        self.agg_cross_features_df_14 = merge_all_ex 
        print("running cross_features_14()...comp")






    def cross_features_15(
        self, n_races: list[int] = [1, 3, 5, 10]
    ):  
        
        #speed_index
        horse_results_baselog = (
            self.population.merge(
                self.horse_results, on="horse_id", suffixes=("", "_horse")
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        #基準タイムの選定
        """

        （	基準
        タイム	－	走破
        タイム	）×	距離
        指数	＋（	斥量	－	５５	）×	２	＋	馬場
        指数	＋	８０	＝	スピード指数


        １０００万クラスのタイムを基準
        ・3歳上 / 4歳以上
        ・1勝クラス
        ・良 / 稍重
        ・入線順位1～3着馬
        上記条件の走破タイムの平均を出す
        (牝馬限定戦を除く)
        ・3歳上 / 4歳以上
        ・2勝クラス
        ・良 / 稍重
        ・入線順位1～3着馬
        上記条件の走破タイムの平均を出す
        (牝馬限定戦を除く)
        それらを更に平均する

        データ数が少ないコースでは、存在するクラスの平均走破タイムを
        「クラス指数の指数差 ÷ 距離指数 ÷ 10」で秒換算し、基準タイムを補正する。
        """
        baselog = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","course_type"]], on="race_id"
            )
        )

        df = (
            baselog
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "nobori","time","umaban","rank"]], on=["race_id", "horse_id"])
        )
        df["nobori"] = df["nobori"].fillna(df["nobori"].mean())

        df["place"] = df["place"].astype(int)
        df["race_grade"] = df["race_grade"].astype(int)
        df["ground_state"] = df["ground_state"].astype(int)
        df["weather"] = df["weather"].astype(int)  
        df["distance_place_type"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str)).astype(int)
        df2 = df
        # 1. 計算したいrace_gradeのリスト
        grades = [55,60,70,79,85,  89,  91,  94,  98]

        # 2. 各race_gradeごとにtimeとnoboriの平均を計算し、結合
        for grade in grades:
            # race_gradeが指定された値のときのtimeとnoboriの平均を計算
            time_nobori_avg = (
                df[
                (df['race_grade'] == grade) &  # race_gradeが指定のgrade
                (df['ground_state'].isin([0, 2])) &  # ground_stateが0または2
                (df['rank'].isin([1, 2, 3,4,5]))  # rankが1, 2, 3
                ]
                .groupby(["course_type"])[['time', 'nobori']]  # 3つのカテゴリごとにtimeとnoboriを集計
                .mean()
                .reset_index()  # インデックスをリセット
                .rename(columns={'time': f'time_avg_{grade}', 'nobori': f'nobori_avg_{grade}'})  # 列名を変更
            )
            
            # 元のDataFrameにマージ（left join）
            horse_results_baselog = pd.merge(horse_results_baselog, time_nobori_avg, on=["course_type"], how='left')
        # 1. 補完する順番を指定
        grades = [70, 79,85, 89, 60, 91,  94, 55, 98]

        # 2. まずhorse_results_baselog内の補完処理を行う
        horse_results_baselog['final_time_avg'] = np.nan
        horse_results_baselog['final_nobori_avg'] = np.nan

        # 3. 最初に85で補完
        horse_results_baselog['final_time_avg'] = np.where(
            horse_results_baselog['final_time_avg'].isna(), 
            horse_results_baselog['time_avg_70'], 
            horse_results_baselog['final_time_avg']
        )
        horse_results_baselog['final_nobori_avg'] = np.where(
            horse_results_baselog['final_nobori_avg'].isna(), 
            horse_results_baselog['nobori_avg_70'], 
            horse_results_baselog['final_nobori_avg']
        )


        """
        距離指数をかける
        距離	芝	ダート
        1000m	1.8	1.7
        1150m	1.52	1.45
        1200m	1.45	1.39
        1300m	1.34	1.27
        1400m	1.23	1.18
        1500m	1.12	1.08
        1600m	1.06	1.02
        1700m	1.00	0.94
        1800m	0.93	0.88
        1900m	0.88	0.83
        2000m	0.83	0.79
        2100m	0.79	0.75
        2200m	0.75	0.7
        2300m	0.71	0.67
        2400m	0.68	0.64
        2500m	0.64	0.61
        2600m	0.62	0.59
        2800m	0.56	0.53
        3000m	0.53	0.5
        3200m	0.50	0.47
        3400m	0.47	0.44
        3600m	0.45	0.42
        """

        # 距離と芝ダートの対応表を辞書として定義
        conversion_table = {
            800: {1: 2.1, 0: 2},
            820: {1: 2.05, 0: 1.97},
            850: {1: 2, 0: 1.9},
            900: {1: 1.94, 0: 1.85},
            920: {1: 1.92, 0: 1.83},
            1000: {1: 1.8, 0: 1.7},
            1100: {1: 1.6, 0: 1.5},
            1150: {1: 1.52, 0: 1.45},
            1160: {1: 1.5, 0: 1.43},
            1170: {1: 1.48, 0: 1.42},
            1200: {1: 1.45, 0: 1.39},
            1230: {1: 1.41, 0: 1.36},
            1300: {1: 1.34, 0: 1.27},
            1400: {1: 1.23, 0: 1.18},
            1500: {1: 1.12, 0: 1.08},
            1600: {1: 1.06, 0: 1.02},
            1700: {1: 1.00, 0: 0.94},
            1750: {1: 0.96, 0: 0.91},
            1800: {1: 0.93, 0: 0.88},
            1860: {1: 0.91, 0: 0.86},
            1870: {1: 0.90, 0: 0.85},
            1900: {1: 0.88, 0: 0.83},
            2000: {1: 0.83, 0: 0.79},
            2100: {1: 0.79, 0: 0.75},
            2200: {1: 0.75, 0: 0.7},
            2300: {1: 0.71, 0: 0.67},
            2400: {1: 0.68, 0: 0.64},
            2500: {1: 0.64, 0: 0.61},
            2600: {1: 0.62, 0: 0.59},
            2800: {1: 0.56, 0: 0.53},
            3000: {1: 0.53, 0: 0.5},
            3200: {1: 0.5, 0: 0.47},
            3400: {1: 0.47, 0: 0.44},
            3600: {1: 0.45, 0: 0.42}
        }

        # 新しい列に変換後の値を格納
        def map_conversion(row):
            course_len = row['course_len']
            race_type = row['race_type']
            if course_len in conversion_table and race_type in conversion_table[course_len]:
                return conversion_table[course_len][race_type]
            else:
                return None  # 該当しない場合はNoneを返す

        # 適用
        horse_results_baselog['converted_value'] = horse_results_baselog.apply(map_conversion, axis=1)


        # 4. 残りのグレードで補完（順番に）
        for grade in grades:
            if grade != 70:
                time_col = f'time_avg_{grade}'
                nobori_col = f'nobori_avg_{grade}'
                
                # timeの欠損を補完
                horse_results_baselog['final_time_avg'] = np.where(
                    horse_results_baselog['final_time_avg'].isna(), 
                    horse_results_baselog[time_col] + ((grade - 70) / 1.2/horse_results_baselog['converted_value']/10),
                    horse_results_baselog['final_time_avg']
                )
                # noboriの欠損を補完
                horse_results_baselog['final_nobori_avg'] = np.where(
                    horse_results_baselog['final_nobori_avg'].isna(), 
                    horse_results_baselog[nobori_col], 
                    horse_results_baselog['final_nobori_avg']
                )



        # 1. final_time_avgからtimeを引いた数値（秒）を計算
        horse_results_baselog['time_diff_sp'] = horse_results_baselog['final_time_avg'] - horse_results_baselog['time']

        # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['time_points'] = (horse_results_baselog['time_diff_sp'])*10

        # 1. final_nobori_avgからnoboriを引いた数値（秒）を計算
        horse_results_baselog['nobori_diff_sp'] = horse_results_baselog['final_nobori_avg'] - horse_results_baselog['nobori']

        # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['nobori_points'] = (horse_results_baselog['nobori_diff_sp'])*10



        # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['time_points_course_index'] = horse_results_baselog['time_points'] *horse_results_baselog['converted_value'] 

        # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['nobori_points_course_index'] = horse_results_baselog['nobori_points'] *horse_results_baselog['converted_value'] 


        """
        ＋（	斥量	－	５５	）馬体重の12%から斤量がかなりくるらしい
        """
        # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['time_points_impost'] = (horse_results_baselog['time_points_course_index'] +(((horse_results_baselog["impost"]-(55- ((55 - (horse_results_baselog["weight"] *(12/100)))/7))) *1.7)*(((horse_results_baselog["course_len"]*0.0025)+20)/20)*(((horse_results_baselog["race_type"])+10)/10)))

        # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['nobori_points_impost'] = (horse_results_baselog['nobori_points_course_index']+(((horse_results_baselog["impost"]-(55- ((55 - (horse_results_baselog["weight"] *(12/100)))/7))) *1.7)*(((horse_results_baselog["course_len"]*0.0025)+20)/20)*(((horse_results_baselog["race_type"])+10)/10)))



        """
        暫定馬場指数＝（馬場指数用基準タイム－該当レース上位３頭の平均タイム）× 距離指数
        馬場指数用基準タイム ＝ 基準タイム － (クラス指数 × 距離指数)　＋pase_diff

        ハイペースなら低く（早く見える）でてしまう
        スローペースなら高く（遅く見える）でてしまう
        pase_diffは+だとハイペース
        -だとスローペースなので
        そのまま+してよいかも
        """

        # # 2. 各race_gradeごとにtimeとnoboriの平均を計算し、結合
        # for grade in grades:
        #     # race_gradeが指定された値のときのtimeとnoboriの平均を計算
        #     time_nobori_avg_top = (
        #         df2[
        #         (df2['race_grade'] == grade) &  # race_gradeが指定のgrade
        #         (df2['ground_state'].isin([0])) &  # ground_stateが0または2
        #         (df2['rank'].isin([1]))  # rankが1, 2, 3
        #         ]
        #         .groupby(['course_len', 'place', 'race_type'])[['time', 'nobori']]  # 3つのカテゴリごとにtimeとnoboriを集計
        #         .mean()
        #         .reset_index()  # インデックスをリセット
        #         .rename(columns={'time': f'time_avg_{grade}_top', 'nobori': f'nobori_avg_{grade}_top'})  # 列名を変更
        #     )
            
        #     # 元のDataFrameにマージ（left join）
        #     horse_results_baselog = pd.merge(horse_results_baselog, time_nobori_avg_top, on=['course_len', 'place', 'race_type'], how='left')
        # # 1. 補完する順番を指定
        # grades = [70, 79,85, 89, 60, 91,  94, 55, 98]

        # # 2. まずhorse_results_baselog内の補完処理を行う
        # horse_results_baselog['final_time_avg_top'] = np.nan
        # horse_results_baselog['final_nobori_avg_top'] = np.nan

        # # 3. 最初に85で補完
        # horse_results_baselog['final_time_avg_top'] = np.where(
        #     horse_results_baselog['final_time_avg_top'].isna(), 
        #     horse_results_baselog['time_avg_70_top'], 
        #     horse_results_baselog['final_time_avg_top']
        # )
        # horse_results_baselog['final_nobori_avg_top'] = np.where(
        #     horse_results_baselog['final_nobori_avg_top'].isna(), 
        #     horse_results_baselog['nobori_avg_70_top'], 
        #     horse_results_baselog['final_nobori_avg_top']
        # )

        # # 4. 残りのグレードで補完（順番に）
        # for grade in grades:
        #     if grade != 70:
        #         time_col = f'time_avg_{grade}_top'
        #         nobori_col = f'nobori_avg_{grade}_top'
                
        #         # timeの欠損を補完
        #         horse_results_baselog['final_time_avg_top'] = np.where(
        #             horse_results_baselog['final_time_avg_top'].isna(), 
        #             horse_results_baselog[time_col] + ((grade - 70) / 1.2/horse_results_baselog['converted_value']/10),
        #             horse_results_baselog['final_time_avg_top']
        #         )
        #         # noboriの欠損を補完
        #         horse_results_baselog['final_nobori_avg_top'] = np.where(
        #             horse_results_baselog['final_nobori_avg_top'].isna(), 
        #             horse_results_baselog[nobori_col], 
        #             horse_results_baselog['final_nobori_avg_top']
        #         )
        # 2. 各race_gradeごとにtimeとnoboriの平均を計算し、結合
        for grade in grades:
            # race_gradeが指定された値のときのtimeとnoboriの平均を計算
            time_nobori_avg_top = (
                df2[
                (df2['race_grade'] == grade) &  # race_gradeが指定のgrade
                (df2['ground_state'].isin([0])) &  # ground_stateが0または2
                (df2['rank'].isin([1]))  # rankが1, 2, 3
                ]
                .groupby(["course_type"])[['time', 'nobori']]  # 3つのカテゴリごとにtimeとnoboriを集計
                .mean()
                .reset_index()  # インデックスをリセット
                .rename(columns={'time': f'time_avg_{grade}_top', 'nobori': f'nobori_avg_{grade}_top'})  # 列名を変更
            )
            
            # 元のDataFrameにマージ（left join）
            horse_results_baselog = pd.merge(horse_results_baselog, time_nobori_avg_top, on=["course_type"], how='left')
        # 1. 補完する順番を指定
        grades = [70, 79,85, 89, 60, 91,  94, 55, 98]

        # 2. まずhorse_results_baselog内の補完処理を行う
        horse_results_baselog['final_time_avg_top'] = np.nan
        horse_results_baselog['final_nobori_avg_top'] = np.nan

        # 1. race_grade に基づいて time_avg_〇_top から補完
        horse_results_baselog['final_time_avg_top'] = horse_results_baselog.apply(
            lambda row: row[f'time_avg_{int(row["race_grade"])}_top'] 
            if pd.isna(row['final_time_avg_top']) and f'time_avg_{int(row["race_grade"])}_top' in horse_results_baselog.columns 
            else row['final_time_avg_top'], 
            axis=1
        )

        # 2. race_grade に基づいて nobori_avg_〇_top から補完
        horse_results_baselog['final_nobori_avg_top'] = horse_results_baselog.apply(
            lambda row: row[f'nobori_avg_{int(row["race_grade"])}_top'] 
            if pd.isna(row['final_nobori_avg_top']) and f'nobori_avg_{int(row["race_grade"])}_top' in horse_results_baselog.columns 
            else row['final_nobori_avg_top'], 
            axis=1
        )

        horse_results_baselog['time_condition_index_shaft'] = horse_results_baselog['final_time_avg_top']
        horse_results_baselog['nobori_condition_index_shaft'] = horse_results_baselog['final_nobori_avg_top']
        # rankが1の場合はそのまま、rankが2以上の場合はtimeからrank_diffを引く
        horse_results_baselog['adjusted_time'] = horse_results_baselog.apply(
            lambda row: row['time'] if row['rank'] == 1 else row['time'] - (row['rank_diff']-2),
            axis=1
        )


        # 1. final_time_avgからtimeを引いた数値（秒）を計算
        horse_results_baselog['time_diff_grade'] = horse_results_baselog['time_condition_index_shaft'] - horse_results_baselog['adjusted_time']

        # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['time_points_grade'] = (horse_results_baselog['time_diff_grade'] )*10

        # 1. final_nobori_avgからnoboriを引いた数値（秒）を計算
        horse_results_baselog['nobori_diff_grade'] = horse_results_baselog['nobori_condition_index_shaft'] 

        # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['nobori_points_grade'] = (horse_results_baselog['time_diff_grade'] ) *10

        """
        距離指数をかける
        """

        # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['time_points_grade_index'] = (horse_results_baselog['time_points_grade'] *horse_results_baselog['converted_value'])

        # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['nobori_points_grade_index'] = (horse_results_baselog['nobori_points_grade'] *horse_results_baselog['converted_value'])

        horse_results_baselog['time_condition_index'] = horse_results_baselog['time_points_grade_index'] -horse_results_baselog['pace_diff']*5
        horse_results_baselog['nobori_condition_index'] = horse_results_baselog['nobori_points_grade_index'] +horse_results_baselog['pace_diff']*5



        # 新しい列を作成
        horse_results_baselog['race_grade_transformed'] = (horse_results_baselog['race_grade'] - 80) / 40 + 80


        horse_results_baselog['speed_index'] = horse_results_baselog['time_points_impost'] + horse_results_baselog['time_condition_index'] + horse_results_baselog['race_grade_transformed'] 
        horse_results_baselog['nobori_index'] = horse_results_baselog['nobori_points_impost'] + horse_results_baselog['nobori_condition_index'] + horse_results_baselog['race_grade_transformed'] 



        # 5. 不要な中間列を削除し、必要な列だけ残す
        columns_to_drop = [f'time_avg_{grade}' for grade in grades] + [f'nobori_avg_{grade}' for grade in grades]
        horse_results_baselog = horse_results_baselog.drop(columns=columns_to_drop)



        horse_results_baselog = horse_results_baselog[['race_id',
        'date',
        'horse_id',
        'time_diff_sp',
        'nobori_diff_sp',
        'time_points_course_index',
        'nobori_points_course_index',
        'time_points_impost',
        'nobori_points_impost',
        'time_diff_grade',
        'nobori_diff_grade',
        'time_points_grade_index',
        'nobori_points_grade_index',
        'time_condition_index',
        'nobori_condition_index',
        'speed_index',
        'nobori_index']]

        grouped_df = horse_results_baselog.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        for n_race in tqdm(n_races, desc=f"speed_index"):
            df_speed = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [
                        'time_diff_sp',
                        'nobori_diff_sp',
                        'time_points_course_index',
                        'nobori_points_course_index',
                        'time_points_impost',
                        'nobori_points_impost',
                        'time_diff_grade',
                        'nobori_diff_grade',
                        'time_points_grade_index',
                        'nobori_points_grade_index',
                        'time_condition_index',
                        'nobori_condition_index',
                        'speed_index',
                        'nobori_index',
                    ]
                ]
                .agg(["mean", "max", "min"])
            )
            original_df = df_speed.copy()
            df_speed.columns = [
                "_".join(col) + f"_{n_race}races" for col in df_speed.columns
            ]
            # レースごとの相対値に変換
            tmp_df = df_speed.groupby(["race_id"])
            relative_df = ((df_speed - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            merged_df = merged_df.merge(
                relative_df, on=["race_id", "horse_id"], how="left"
            )
            # 相対値を付けない元の列をそのまま追加
            original_df.columns = [
                "_".join(col) + f"_{n_race}races" for col in original_df.columns
            ]  # 列名変更
            
            merged_df = merged_df.merge(
                original_df, on=["race_id", "horse_id"], how="left"
            )


        self.agg_cross_features_df_15 = merged_df
        print("running cross_features_15()...comp")











    def cross_features_16(
        self, n_races: list[int] = [1, 3, 5, 10]
    ):  
                
        merged_df = self.population.copy()    
        df_syunpatu = self.syunpatu_zizoku_df
        df_pace = self.pace_category

        df = df_syunpatu.merge(
            df_pace,
            on=["race_id","date","horse_id"],
        )
        """
        syunpatuは道悪がよく、ローペースが得意、内枠が良い
        zizokuは高速馬場がよく、長距離がよく、ミドルハイペース・ハイペースが良い、外枠が得意である
        dfにはumaban,course_len,"pace_category","ground_state_level"
        と以下のsyunpatu_columns_plus列がある
        "pace_category"はペース
        "ground_state_level"は馬場状態である
                ＿＿ハイペース4
                
                ミドルハイペース3
                
                ミドルローペース2
                
                ＿＿ローペース1

        #　超高速馬場1
        # 高速馬場2
        # 軽い馬場3
                    # 標準的な馬場4
                
                    # やや重い馬場5
            # 重い馬場5
        # 道悪7
        である

        このとき、syunpatu_columns_plusの各値とペース、馬場状態、距離、枠をそれぞれかけ合わせ、
        「        syunpatuは道悪がよく、ローハイペースが得意、内枠が良い
                zizokuは高速馬場がよく、長距離がよく、ミドルハイペース・ハイペースが良い、外枠が得意である
        」
        の条件を満たすように、補正をかける
        """
        df = df.merge(
            self.results[["race_id","horse_id","umaban"]],
            on=["race_id","horse_id"],
        )

        df = df.merge(
            self.race_info[["race_id","course_len"]],
            on=["race_id"],
        )

        syunpatu_columns_plus = [
            "syunpatu_mean_1races_encoded", "syunpatu_mean_3races_encoded", "syunpatu_mean_5races_encoded", "syunpatu_mean_8races_encoded",
            "syunpatu_min_1races_encoded", "syunpatu_min_3races_encoded", "syunpatu_min_5races_encoded", "syunpatu_min_8races_encoded",
            "syunpatu_max_1races_encoded", "syunpatu_max_3races_encoded", "syunpatu_max_5races_encoded", "syunpatu_max_8races_encoded",
        
        ]
        zizoku_columns_plus = [      
            # zizokuバージョンの列を追加
            "zizoku_mean_1races_encoded", "zizoku_mean_3races_encoded", "zizoku_mean_5races_encoded", "zizoku_mean_8races_encoded",
            "zizoku_min_1races_encoded", "zizoku_min_3races_encoded", "zizoku_min_5races_encoded", "zizoku_min_8races_encoded",
            "zizoku_max_1races_encoded", "zizoku_max_3races_encoded", "zizoku_max_5races_encoded", "zizoku_max_8races_encoded"
        
        ]

        # for col in syunpatu_columns_plus:
        #     df[f"{col}_index"] = df[col] * ((df["pace_category"]+20)/20) * (1/((df["ground_state_level"]+20)/20)) * (1/((df["umaban"]+60)/60))

        # for col in zizoku_columns_plus:
        #     df[f"{col}_index"] = df[col] * (1/((df["pace_category"]+20)/20)) * (((df["ground_state_level"]+20)/20)) * (((df["umaban"]+70)/70)) * (((df["course_len"]/100)+35)/35)

        # オリジナルの統計値（mean, min）を保持
        df2 = df.copy()

        # syunpatu_columns_plusの補正値を計算
        for col in syunpatu_columns_plus:
            # 補正後の列を計算
            df[f"{col}_index"] = df[col] * ((df["pace_category"] + 15) / 15) * (1 / ((df["ground_state_level"] + 20) / 20)) * (1 / ((df["umaban"] + 50) / 50))
            # レースごとの相対値に変換
            tmp_df = df.groupby(["race_id"])[f"{col}_index"].transform(lambda x: (x - x.mean()) / x.std())
            df[f"{col}_index_relative"] = tmp_df

        # zizoku_columns_plusの補正値を計算
        for col in zizoku_columns_plus:
            # 補正後の列を計算
            df[f"{col}_index"] = df[col] * (1 / ((df["pace_category"] + 15) / 15)) * (((df["ground_state_level"] + 20) / 20)) * (((df["umaban"] + 50) / 50)) * (((df["course_len"] / 100) + 35) / 35)
            # レースごとの相対値に変換
            # レースごとの相対値に変換
            tmp_df = df.groupby(["race_id"])[f"{col}_index"].transform(lambda x: (x - x.mean()) / x.std())
            df[f"{col}_index_relative"] = tmp_df


        """                
        ＿＿ハイペース4
                
                ミドルハイペース3
                
                ミドルローペース2
                
                ＿＿ローペース1

        #　超高速馬場1
        # 高速馬場2
        # 軽い馬場3
                    # 標準的な馬場4
                
                    # やや重い馬場5
            # 重い馬場5
        # 道悪7
        "pace_category","ground_state_level"
        ローペース、道悪のとき、瞬発指数を
        ハイペース、高速のとき、持続指数を
        それぞれ参照する新たな列を作成
        """
        n_values = [1, 3, 5, 8]
        for n in n_values:
            # 各条件の定義
            condition1 = (df["pace_category"] < 2.5) & (df["ground_state_level"] >= 3)
            condition2 = (df["pace_category"] > 2.5) & (df["ground_state_level"] <= 5)
            condition3 = (~condition1 & ~condition2) & (df["course_len"] > 2000)
            condition4 = (~condition1 & ~condition2) & (df["course_len"] <= 2000)

            # 条件リストと対応する値
            conditions = [condition1, condition2, condition3, condition4]
            choices = [
                df[f"syunpatu_mean_{n}races_encoded_index"],  # condition1
                df[f"zizoku_mean_{n}races_encoded_index"],    # condition2
                df[f"zizoku_mean_{n}races_encoded_index"],    # condition3
                df[f"syunpatu_mean_{n}races_encoded_index"]   # condition4
            ]

            # 条件ごとに列を適用
            df[f'advantage_mean_{n}_index'] = np.select(conditions, choices, default=np.nan)



        for n in n_values:
            # 各条件の定義
            condition1 = (df["pace_category"] < 2.5) & (df["ground_state_level"] >= 3)
            condition2 = (df["pace_category"] > 2.5) & (df["ground_state_level"] <= 5)
            condition3 = (~condition1 & ~condition2) & (df["course_len"] > 2000)
            condition4 = (~condition1 & ~condition2) & (df["course_len"] <= 2000)

            # 条件リストと対応する値
            conditions = [condition1, condition2, condition3, condition4]
            choices = [
                df[f"syunpatu_min_{n}races_encoded_index"],  # condition1
                df[f"zizoku_min_{n}races_encoded_index"],    # condition2
                df[f"zizoku_min_{n}races_encoded_index"],    # condition3
                df[f"syunpatu_min_{n}races_encoded_index"]   # condition4
            ]

            # 条件ごとに列を適用
            df[f'advantage_min_{n}_index'] = np.select(conditions, choices, default=np.nan)


        for n in n_values:
            # 各条件の定義
            condition1 = (df["pace_category"] < 2.5) & (df["ground_state_level"] >= 3)
            condition2 = (df["pace_category"] > 2.5) & (df["ground_state_level"] <= 5)
            condition3 = (~condition1 & ~condition2) & (df["course_len"] > 2000)
            condition4 = (~condition1 & ~condition2) & (df["course_len"] <= 2000)

            # 条件リストと対応する値
            conditions = [condition1, condition2, condition3, condition4]
            choices = [
                df[f"syunpatu_max_{n}races_encoded_index"],  # condition1
                df[f"zizoku_max_{n}races_encoded_index"],    # condition2
                df[f"zizoku_max_{n}races_encoded_index"],    # condition3
                df[f"syunpatu_max_{n}races_encoded_index"]   # condition4
            ]

            # 条件ごとに列を適用
            df[f'advantage_max_{n}_index'] = np.select(conditions, choices, default=np.nan)

        advantage_row = [
            "advantage_max_1_index", "advantage_max_3_index", "advantage_max_5_index", "advantage_max_8_index",
            "advantage_min_1_index", "advantage_min_3_index", "advantage_min_5_index", "advantage_min_8_index",
            "advantage_mean_1_index", "advantage_mean_3_index", "advantage_mean_5_index", "advantage_mean_8_index"
            ]


        for col in advantage_row:
            tmp_df = df.groupby(["race_id"])[f"{col}"].transform(lambda x: (x - x.mean()) / x.std())
            df[f"{col}_relative"] = tmp_df



        select_row = [
            # "syunpatu_mean_1races_encoded_index", "syunpatu_mean_3races_encoded_index", "syunpatu_mean_5races_encoded_index", "syunpatu_mean_8races_encoded_index",
            # "syunpatu_min_1races_encoded_index", "syunpatu_min_3races_encoded_index", "syunpatu_min_5races_encoded_index", "syunpatu_min_8races_encoded_index",
            # "syunpatu_max_1races_encoded_index", "syunpatu_max_3races_encoded_index", "syunpatu_max_5races_encoded_index", "syunpatu_max_8races_encoded_index",
            
            # # zizokuバージョンの列を追加
            # "zizoku_mean_1races_encoded_index", "zizoku_mean_3races_encoded_index", "zizoku_mean_5races_encoded_index", "zizoku_mean_8races_encoded_index",
            # "zizoku_min_1races_encoded_index", "zizoku_min_3races_encoded_index", "zizoku_min_5races_encoded_index", "zizoku_min_8races_encoded_index",
            # "zizoku_max_1races_encoded_index", "zizoku_max_3races_encoded_index", "zizoku_max_5races_encoded_index", "zizoku_max_8races_encoded_index",

            # "advantage_max_1_index", "advantage_max_3_index", "advantage_max_5_index", "advantage_max_8_index",
            # "advantage_min_1_index", "advantage_min_3_index", "advantage_min_5_index", "advantage_min_8_index",
            # "advantage_mean_1_index", "advantage_mean_3_index", "advantage_mean_5_index", "advantage_mean_8_index",


            "syunpatu_mean_1races_encoded_index_relative", "syunpatu_mean_3races_encoded_index_relative", "syunpatu_mean_5races_encoded_index_relative", "syunpatu_mean_8races_encoded_index_relative",
            "syunpatu_min_1races_encoded_index_relative", "syunpatu_min_3races_encoded_index_relative", "syunpatu_min_5races_encoded_index_relative", "syunpatu_min_8races_encoded_index_relative",
            "syunpatu_max_1races_encoded_index_relative", "syunpatu_max_3races_encoded_index_relative", "syunpatu_max_5races_encoded_index_relative", "syunpatu_max_8races_encoded_index_relative",
            
            # zizokuバージョンの列を追加
            "zizoku_mean_1races_encoded_index_relative", "zizoku_mean_3races_encoded_index_relative", "zizoku_mean_5races_encoded_index_relative", "zizoku_mean_8races_encoded_index_relative",
            "zizoku_min_1races_encoded_index_relative", "zizoku_min_3races_encoded_index_relative", "zizoku_min_5races_encoded_index_relative", "zizoku_min_8races_encoded_index_relative",
            "zizoku_max_1races_encoded_index_relative", "zizoku_max_3races_encoded_index_relative", "zizoku_max_5races_encoded_index_relative", "zizoku_max_8races_encoded_index_relative",

            "advantage_max_1_index_relative", "advantage_max_3_index_relative", "advantage_max_5_index_relative", "advantage_max_8_index_relative",
            "advantage_min_1_index_relative", "advantage_min_3_index_relative", "advantage_min_5_index_relative", "advantage_min_8_index_relative",
            "advantage_mean_1_index_relative", "advantage_mean_3_index_relative", "advantage_mean_5_index_relative", "advantage_mean_8_index_relative",

            'race_id', 'horse_id', 'date'


        ]

        # merged_dfから指定した列だけを抽出
        merged_df2 = df[select_row]


        self.agg_cross_features_df_16 = merged_df2
        print("running cross_features_16()...comp")

































        
    def cross_features_17(
        self
    ):  
                    
                
            #基準タイムの選定
            """

            （	基準
            タイム	－	走破
            タイム	）×	距離
            指数	＋（	斥量	－	５５	）×	２	＋	馬場
            指数	＋	８０	＝	スピード指数


            １０００万クラスのタイムを基準
            ・3歳上 / 4歳以上
            ・1勝クラス
            ・良 / 稍重
            ・入線順位1～3着馬
            上記条件の走破タイムの平均を出す
            (牝馬限定戦を除く)
            ・3歳上 / 4歳以上
            ・2勝クラス
            ・良 / 稍重
            ・入線順位1～3着馬
            上記条件の走破タイムの平均を出す
            (牝馬限定戦を除く)
            それらを更に平均する

            データ数が少ないコースでは、存在するクラスの平均走破タイムを
            「クラス指数の指数差 ÷ 距離指数 ÷ 10」で秒換算し、基準タイムを補正する。
            """
            baselog = (
                self.population.merge(
                    self.race_info[["race_id", "course_len", "race_class","race_type","race_grade","ground_state", "weather","place","race_date_day_count","course_type"]], on="race_id"
                )
            )

            df = (
                baselog
                .merge(self.results[["race_id", "impost","horse_id","weight", "wakuban", "nobori","time","umaban","rank"]], on=["race_id", "horse_id"])
            )
            df["nobori"] = df["nobori"].fillna(df["nobori"].mean())

            df["place"] = df["place"].astype(int)
            df["race_grade"] = df["race_grade"].astype(int)
            df["ground_state"] = df["ground_state"].astype(int)
            df["weather"] = df["weather"].astype(int)  
            df["distance_place_type"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str)).astype(int)
            df2 = df
            # 1. 計算したいrace_gradeのリスト
            grades = [55,60,70,79,85,  89,  91,  94,  98]

            # 2. 各race_gradeごとにtimeとnoboriの平均を計算し、結合
            for grade in grades:
                # race_gradeが指定された値のときのtimeとnoboriの平均を計算
                time_nobori_avg = (
                    df[
                    (df['race_grade'] == grade) &  # race_gradeが指定のgrade
                    (df['ground_state'].isin([0, 2])) &  # ground_stateが0または2
                    (df['rank'].isin([1, 2, 3,4,5]))  # rankが1, 2, 3
                    ]
                    .groupby(["course_type"])[['time', 'nobori']]  # 3つのカテゴリごとにtimeとnoboriを集計
                    .mean()
                    .reset_index()  # インデックスをリセット
                    .rename(columns={'time': f'time_avg_{grade}', 'nobori': f'nobori_avg_{grade}'})  # 列名を変更
                )
                
                # 元のDataFrameにマージ（left join）
                df = pd.merge(df, time_nobori_avg, on=["course_type"], how='left')
            # 1. 補完する順番を指定
            grades = [70, 79,85, 89, 60, 91,  94, 55, 98]

            # 2. まずhorse_results_baselog内の補完処理を行う
            df['final_time_avg'] = np.nan
            df['final_nobori_avg'] = np.nan

            # 3. 最初に85で補完
            df['final_time_avg'] = np.where(
                df['final_time_avg'].isna(), 
                df['time_avg_70'], 
                df['final_time_avg']
            )
            df['final_nobori_avg'] = np.where(
                df['final_nobori_avg'].isna(), 
                df['nobori_avg_70'], 
                df['final_nobori_avg']
            )


            """
            距離指数をかける
            距離	芝	ダート
            1000m	1.8	1.7
            1150m	1.52	1.45
            1200m	1.45	1.39
            1300m	1.34	1.27
            1400m	1.23	1.18
            1500m	1.12	1.08
            1600m	1.06	1.02
            1700m	1.00	0.94
            1800m	0.93	0.88
            1900m	0.88	0.83
            2000m	0.83	0.79
            2100m	0.79	0.75
            2200m	0.75	0.7
            2300m	0.71	0.67
            2400m	0.68	0.64
            2500m	0.64	0.61
            2600m	0.62	0.59
            2800m	0.56	0.53
            3000m	0.53	0.5
            3200m	0.50	0.47
            3400m	0.47	0.44
            3600m	0.45	0.42
            """
            # 距離と芝ダートの対応表を辞書として定義
            conversion_table = {
                800: {1: 2.1, 0: 2},
                820: {1: 2.05, 0: 1.97},
                850: {1: 2, 0: 1.9},
                900: {1: 1.94, 0: 1.85},
                920: {1: 1.92, 0: 1.83},
                1000: {1: 1.8, 0: 1.7},
                1100: {1: 1.6, 0: 1.5},
                1150: {1: 1.52, 0: 1.45},
                1160: {1: 1.5, 0: 1.43},
                1170: {1: 1.48, 0: 1.42},
                1200: {1: 1.45, 0: 1.39},
                1230: {1: 1.41, 0: 1.36},
                1300: {1: 1.34, 0: 1.27},
                1400: {1: 1.23, 0: 1.18},
                1500: {1: 1.12, 0: 1.08},
                1600: {1: 1.06, 0: 1.02},
                1700: {1: 1.00, 0: 0.94},
                1750: {1: 0.96, 0: 0.91},
                1800: {1: 0.93, 0: 0.88},
                1860: {1: 0.91, 0: 0.86},
                1870: {1: 0.90, 0: 0.85},
                1900: {1: 0.88, 0: 0.83},
                2000: {1: 0.83, 0: 0.79},
                2100: {1: 0.79, 0: 0.75},
                2200: {1: 0.75, 0: 0.7},
                2300: {1: 0.71, 0: 0.67},
                2400: {1: 0.68, 0: 0.64},
                2500: {1: 0.64, 0: 0.61},
                2600: {1: 0.62, 0: 0.59},
                2800: {1: 0.56, 0: 0.53},
                3000: {1: 0.53, 0: 0.5},
                3200: {1: 0.5, 0: 0.47},
                3400: {1: 0.47, 0: 0.44},
                3600: {1: 0.45, 0: 0.42}
            }

            # 新しい列に変換後の値を格納
            def map_conversion(row):
                course_len = row['course_len']
                race_type = row['race_type']
                if course_len in conversion_table and race_type in conversion_table[course_len]:
                    return conversion_table[course_len][race_type]
                else:
                    return None  # 該当しない場合はNoneを返す

            # 適用
            df['converted_value'] = df.apply(map_conversion, axis=1)


            # 4. 残りのグレードで補完（順番に）
            for grade in grades:
                if grade != 70:
                    time_col = f'time_avg_{grade}'
                    nobori_col = f'nobori_avg_{grade}'
                    
                    # timeの欠損を補完
                    df['final_time_avg'] = np.where(
                        df['final_time_avg'].isna(), 
                        df[time_col] + ((grade - 70) / 1.2/df['converted_value']/10),
                        df['final_time_avg']
                    )
                    # noboriの欠損を補完
                    df['final_nobori_avg'] = np.where(
                        df['final_nobori_avg'].isna(), 
                        df[nobori_col], 
                        df['final_nobori_avg']
                    )



            # 1. final_time_avgからtimeを引いた数値（秒）を計算
            df['time_diff_sp'] = df['final_time_avg'] - df['time']

            # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
            df['time_points'] = (df['time_diff_sp'])*10

            # 1. final_nobori_avgからnoboriを引いた数値（秒）を計算
            df['nobori_diff_sp'] = df['final_nobori_avg'] - df['nobori']

            # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
            df['nobori_points'] = (df['nobori_diff_sp'])*10



            # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
            df['time_points_course_index'] = df['time_points'] *df['converted_value'] 

            # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
            df['nobori_points_course_index'] = df['nobori_points'] *df['converted_value'] 


            """
            ＋（	斥量	－	５５	）
            """
            # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
            df['time_points_impost'] = (df['time_points_course_index']+(((df["impost"]-(55- ((55 - (df["weight"] *(12/100)))/7))) *1.7)*(((df["course_len"]*0.0025)+20)/20)*(((df["race_type"])+10)/10)))

            # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
            df['nobori_points_impost'] = (df['nobori_points_course_index']+(((df["impost"]-(55- ((55 - (df["weight"] *(12/100)))/7))) *1.7)*(((df["course_len"]*0.0025)+20)/20)*(((df["race_type"])+10)/10)))



            """
            暫定馬場指数＝（馬場指数用基準タイム－該当レース上位３頭の平均タイム）× 距離指数
            馬場指数用基準タイム ＝ 基準タイム － (クラス指数 × 距離指数)　＋pase_diff

            ハイペースなら低く（早く見える）でてしまう
            スローペースなら高く（遅く見える）でてしまう
            pase_diffは+だとハイペース
            -だとスローペースなので
            そのまま+してよいかも
            """
            #predictの時にはold_poplationに変える
            old_merged_df = self.population.copy()     
            #ここで、直近レースのtimeを知りたい
            df_old2 = (
                old_merged_df
                .merge(self.results[["race_id", "horse_id","time","rank","rank_per_horse"]], on=["race_id", "horse_id"])
                .merge(self.race_info[["race_id", "place","weather","ground_state","race_grade","course_len","race_type","course_type","race_date_day_count"]], on="race_id")
            )
            df_old2["place"] = df_old2["place"].astype(int)
            df_old2["race_grade"] = df_old2["race_grade"].astype(int)
            # 距離/競馬場/タイプ/レースランク
            df_old2["distance_place_type_race_grade"] = (df_old2["course_type"].astype(str) + df_old2["race_grade"].astype(str)).astype(int)


            baselog_old = (
                self.population.merge(
                    self.race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "course_type","weather","place","around"]], on="race_id"
                )
                # .merge(
                #     self.horse_results,
                #     on=["horse_id", "course_len", "race_type"],
                #     suffixes=("", "_horse"),
                # )
                # .query("date_horse < date")
                # .sort_values("date_horse", ascending=False)
            )
            df_old = (
                baselog_old
                .merge(self.results[["race_id", "horse_id", "wakuban", "umaban","nobori","rank","time","sex"]], on=["race_id", "horse_id"])
            )
            df_old["place"] = df_old["place"].astype(int)
            df_old["race_grade"] = df_old["race_grade"].astype(int)
            df_old["ground_state"] = df_old["ground_state"].astype(int)
            df_old["around"] = df_old["around"].fillna(3).astype(int)
            df_old["weather"] = df_old["weather"].astype(int)  
                        

            df_old["distance_place_type_race_grade"] = (df_old["course_type"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
            df_old_copy = df_old
            # rank列が1, 2, 3の行だけを抽出
            df_old = df_old[df_old['rank'].isin([1, 2, 3,4,5])]
            target_mean_1 = df_old.groupby("distance_place_type_race_grade")["time"].mean()
            df_old = df_old.copy()
            # 平均値をカテゴリ変数にマッピング
            df_old["distance_place_type_race_grade_encoded"] = df_old["distance_place_type_race_grade"].map(target_mean_1)


            df_old = df_old[["distance_place_type_race_grade",'distance_place_type_race_grade_encoded']]


            columns_to_merge = [
                ("distance_place_type_race_grade",'distance_place_type_race_grade_encoded'),
            ]

            # 各ペアを順番に処理
            for original_col, encoded_col in columns_to_merge:
                df2_subset = df_old[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
                df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
                df_old2 = df_old2.merge(df2_subset, on=original_col, how='left')  # dfにマージ
            df_old2 = df_old2[df_old2['rank'].isin([1, 2, 3,4,5])]
            df_old2["distance_place_type_race_grade_encoded_time_diff"] = df_old2['distance_place_type_race_grade_encoded'] - df_old2["time"]

            # df_old2= df_old2[df_old2["race_type"] != 2]
            # df_old2_1 = df_old2[df_old2["race_type"] != 0]
            df_old2_1 = df_old2
            # 2. df の各行について処理
            def compute_mean_for_row(row, df_old2_1):
                # race_type == 0 の場合は NaN を返す
                # if row["race_type"] == 0:
                #     return np.nan
                    
                target_day_count = row["race_date_day_count"]  # df の各行の race_date_day_count

                # 3. df_old2_1 から条件に一致する行をフィルタリング
                filtered_df_old2_1 = df_old2_1[
                    (df_old2_1["race_date_day_count"] >= (target_day_count - 400)) &  # race_date_day_count が target_day_count-1200 以上
                    (df_old2_1["race_date_day_count"] <= (target_day_count - 1)) &  # race_date_day_count が target_day_count-1 以下
                    (df_old2_1["place"] == row["place"]) &  # place が一致
                    (df_old2_1["race_type"] == row["race_type"])  
                    # (df_old2_1["weather"].isin([0, 1, 2])) &  # weather が 0, 1, 2 のいずれか
                    # (df_old2_1["ground_state"] == 0)   # ground_state が 0
                    # &
                    # (df_old2_1["rank_per_horse"] < 0.87)  # rank_per_horse が 0.87 未満
                ]
                filtered_df_old2_2 = df_old2_1[
                    (df_old2_1["race_date_day_count"] >= (target_day_count - 400)) &  # race_date_day_count が target_day_count-1200 以上
                    (df_old2_1["race_date_day_count"] <= (target_day_count + 100)) &  # race_date_day_count が target_day_count-1 以下
                    (df_old2_1["place"] == row["place"]) &  # place が一致
                    (df_old2_1["race_type"] == row["race_type"]) 
                    # (df_old2_1["weather"].isin([0, 1, 2])) &  # weather が 0, 1, 2 のいずれか
                    # (df_old2_1["ground_state"] == 0)   # ground_state が 0                    
                    # &
                    # (df_old2_1["rank_per_horse"] < 0.87)  # rank_per_horse が 0.87 未満
                ]
                # 4. フィルタリングした行の "distance_place_type_race_grade_encoded_time_diff" の平均を計算
                mean_time_diff = filtered_df_old2_1["distance_place_type_race_grade_encoded_time_diff"].mean()

                # 5. 計算結果を返す（NaNの場合も考慮）
                # return mean_time_diff if not np.isnan(mean_time_diff) else filtered_df_old2_2["distance_place_type_race_grade_encoded_time_diff"].mean()
                return np.nan_to_num(mean_time_diff, nan=np.nan_to_num(filtered_df_old2_2["distance_place_type_race_grade_encoded_time_diff"].mean(), nan=-1))

            # 6. df の各行に対して、計算した平均値を新しい列に追加
            df['time_diff_grade'] = df.apply(compute_mean_for_row, axis=1, df_old2_1=df_old2_1)


            # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
            df['time_points_grade'] = (df['time_diff_grade'] )*10

            # # 1. final_nobori_avgからnoboriを引いた数値（秒）を計算
            # df['nobori_diff_grade'] = df['nobori_condition_index_shaft'] 

            # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
            df['nobori_points_grade'] = (df['time_diff_grade'] )*10

            """
            距離指数をかける
            """

            # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
            df['time_points_grade_index'] = (df['time_points_grade'] *df['converted_value'])

            # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
            df['nobori_points_grade_index'] = (df['nobori_points_grade'] *df['converted_value'])

            df['time_condition_index'] = df['time_points_grade_index']
            df['nobori_condition_index'] = df['nobori_points_grade_index']



            # 新しい列を作成
            df['race_grade_transformed'] = (df['race_grade'] - 80) / 40 + 80


            df['speed_index'] = df['time_points_impost'] + df['time_condition_index'] +df['race_grade_transformed']
            df['nobori_index'] = df['nobori_points_impost'] + df['nobori_condition_index'] + df['race_grade_transformed']


            # 5. 不要な中間列を削除し、必要な列だけ残す
            columns_to_drop = [f'time_avg_{grade}' for grade in grades] + [f'nobori_avg_{grade}' for grade in grades]
            df = df.drop(columns=columns_to_drop)

            # df = df.dropna(subset=["speed_index"])
            # df["speed_index"] = df["speed_index"].astype(int)
            # df = df.dropna(subset=["nobori_index"])
            # df["nobori_index"] = df["nobori_index"].astype(int)




            df = df[['race_id',
            'date',
            'horse_id','speed_index',
            'nobori_index']]

            merged_df = self.population.copy()
            merged_df = merged_df.merge(df, on=["race_id",'date', "horse_id"], how="left")

            self.agg_cross_features_df_17 = merged_df
            print("running cross_features_17()...comp")
                    





    def umaban_good(
        self, n_races: list[int] = [1, 3, 5, 10]
    ):  
        merged_df = self.population.copy()  
        
        merged_df_umaban =  merged_df.merge(
                self.race_info[["race_id", "course_len","start_slope","start_range","flont_slope","curve"]], 
                on="race_id",
            )
        merged_df_umaban = merged_df_umaban.merge(
                self.results[["race_id","horse_id","umaban","umaban_odd"]], 
                on=["horse_id","race_id"],
            )
        #1（奇数）または 0（偶数）
        
        merged_df_umaban["umaban_processed"] = merged_df_umaban["umaban"].apply(
            lambda x: ((x*-1))-1/4 if x < 4 else (x-8)/4
        ).astype(float)
        merged_df_umaban.loc[:, "umaban_odd_processed"] = (
            (merged_df_umaban["umaban_odd"]-1)
        ).astype(float)

        merged_df_umaban["course_len_processed"] = (((merged_df_umaban["course_len"] - 2000)/200)+6)/4
        merged_df_umaban["start_slope_processed"] = ((merged_df_umaban["start_slope"])*2)

        merged_df_umaban["start_range_processed"] = (((merged_df_umaban["start_range"])-360)/18)
        merged_df_umaban["start_range_processed"] = merged_df_umaban["start_range_processed"].apply(
            lambda x: 3 if x > 3 else x
        )

        merged_df_umaban["flont_slope_processed"] = ((merged_df_umaban["flont_slope"])*-1)
        
        # -4.5 を行う
        merged_df_umaban["curve_processed"] = (merged_df_umaban["curve"] - 4.5)/2
        # +の場合は数値を8倍する
        merged_df_umaban["curve_processed"] = merged_df_umaban["curve_processed"].apply(
            lambda x: x * 8 if x > 0 else x
        )
        merged_df_umaban["all_umaban_processed"] = ((merged_df_umaban["start_slope_processed"] + merged_df_umaban["start_range_processed"] + merged_df_umaban["flont_slope_processed"] + merged_df_umaban["curve_processed"])/merged_df_umaban["course_len_processed"])*merged_df_umaban["umaban_processed"]
        merged_df_umaban["all_umaban_odd_processed"] = ((merged_df_umaban["start_slope_processed"] + merged_df_umaban["start_range_processed"]*2 + merged_df_umaban["flont_slope_processed"] + merged_df_umaban["curve_processed"])/merged_df_umaban["course_len_processed"])*merged_df_umaban["umaban_odd_processed"]

        merged_df_umaban["all_umaban_odd_processed"] = merged_df_umaban["all_umaban_odd_processed"].apply(
            lambda x: 0 if x < 0 else x
        )

        merged_df_umaban = merged_df_umaban[["race_id","horse_id","date","all_umaban_odd_processed","all_umaban_processed"]]
        self.agg_umaban_good = merged_df_umaban


        print("running agg_umaban_good()...comp")
  

    # def cross_features_15(
    #     self, n_races: list[int] = [1, 3, 5, 10]
    # ):  

    #     self.agg_cross_features_df_15 = merged_df
    #     print("running cross_features_15()...comp")
        





    def cross_rank_diff(
        self, n_races: list[int] = [1, 3,5,8]
    ):  
        

        base_2 = (
            self.population
            .merge(
                self.horse_results,
                on="horse_id",
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )


        merged_df = self.population.copy()  
        # # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # #



        # # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # #
        """
        持続瞬発の策定
        芝コース	
        瞬発戦	32.0〜34.4
        消耗戦	35.7〜
        ダート短距離	
        瞬発戦	〜35.9
        消耗戦	37.3〜
        ダート中距離以上	
        瞬発戦	〜36.4
        消耗戦	37.8〜

        ・持続
        コーナー回数が少ない、コーナーゆるい、ラストの直線が短い、ラストの下り坂、ミドルorハイ、外枠、高低差+
        ・瞬発
        コーナー回数が多い、コーナーきつい、ラストの直線が長い、ラストの上り坂or平坦、スローペース、内枠、高低差なしがいい
        瞬発系はタフなレースきつい

        horse_resultsなので、これは判別だけ
        普通のnobori、33未満なら絶瞬発

        "nobori_pace_diff_slope_range_groundstate_position_umaban"のnoboriが
        33未満なら絶瞬発
        〜33.9など出せるなら超瞬発
        〜34.5など出せるなら瞬発
       
        普通のnoboriがグレード込みの平均より0.8以下の場合、瞬発にいれる
        1.2以下なら超瞬発

        それ以外のあぶれでtop<5以内の場合
        34.6〜なら問答無用で持続
        35.7〜なら問答無用で超持続
        それ以外はどっちもなし
        -6絶瞬発、-4超瞬発、-2.5瞬発、2.5持続、4超持続、0どっちもなし
        
        これをカウントじゃなくて平均で出す

        平均で出すとどっちも行ける系がきつい
        瞬発だけ集計-
        持続だけ集計+
        で、当該レースが持続なら+の値だけを集計、カウントがなかった場合,逆の値を代入する
        +の値は、あったものの平均だから、+の集計でマイナスだったものは割り算に含めない、0は含める


        ここでは持続判定と瞬発判定の列を作成

        同じ指標で見たいところ
        rank_diffの真の値を作るのに使う
        rank_diff*gradeは_いくつかのバージョン作る
        """
                


        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        base_2["distance_place_type_ground_state"] = (base_2["course_type"].astype(str)+ base_2["ground_state"].astype(str)).astype(int)   
        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        base_2["distance_place_type_ground_state_grade"] = (base_2["distance_place_type_ground_state"].astype(str)+ base_2["race_grade"].astype(str)).astype(int)   
        


        base_2_old = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","season","course_type"]], on="race_id"
            )
        )

        df_old = (
            base_2_old
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "umaban","nobori","time","rank"]], on=["race_id", "horse_id"])
        )

        df_old["nobori"] = df_old["nobori"].fillna(df_old["nobori"].mean())
        
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        # df_old["around"] = df_old["around"].fillna(3).astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  
        df_old["rank"] = df_old["rank"].astype(int)

        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_ground_state"] = (df_old["course_type"].astype(str)+ df_old["ground_state"].astype(str)).astype(int)   
        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_ground_state_grade"] = (df_old["distance_place_type_ground_state"].astype(str)+ df_old["race_grade"].astype(str)).astype(int)   
        
        target_mean_1= (
            df_old[
            (df_old['rank'].isin([1,2,3,4,5,6,7]))  # rankが1, 2, 3
            ]
            .groupby("distance_place_type_ground_state_grade")["nobori"]
            .mean()
        )
        # #noboriの平均
        # target_mean_1 = df_old.groupby("distance_place_type_ground_state_grade")["nobori"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_ground_state_grade_nobori_encoded"] = df_old["distance_place_type_ground_state_grade"].map(target_mean_1).fillna(0)
        

        df_old_rush_type = df_old[["distance_place_type_ground_state_grade","distance_place_type_ground_state_grade_nobori_encoded"]]
        
        columns_to_merge = [("distance_place_type_ground_state_grade","distance_place_type_ground_state_grade_nobori_encoded")]
        
        
        for original_col, encoded_col in columns_to_merge:
            df2_subset = df_old_rush_type[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
            df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
            base_2 = base_2.merge(df2_subset, on=original_col, how='left')  # dfにマージ
            
        base_2["nobori_diff"] = base_2["distance_place_type_ground_state_grade_nobori_encoded"] - base_2["nobori"]

        base_2 = base_2.copy()
        def calculate_rush_type(row):
            if row["nobori"] < 33 and row["race_type"] == 1:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 33 and row["race_type"] == 1:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 33.9 and row["race_type"] == 1:
                return -4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34.5 and row["race_type"] == 1:
                return -2.5
            if row["nobori"] < 34.5 and row["race_type"] == 1:
                return -2

            if row["nobori"] < 33.5 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 33.5 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34.4 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 35 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -2.5
            if row["nobori"] < 35 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -2

            if row["nobori"] < 34 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34.9 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -2.5
            if row["nobori"] < 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -2

            if row["nobori_diff"] >= 1:
                return -4
            if row["nobori_diff"] >= 0.6:
                return -2.5

            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 35.6 and row["race_type"] == 1 and row["rank"] <= 6:
                return 4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 34.5 and row["race_type"] == 1 and row["rank"] <= 6:
                return 2.5
            if row["nobori"] >= 34.5 and row["race_type"] == 1 and row["rank"] <= 6:
                return 2.5

            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 36.1 and row["race_type"] == 0 and row["course_len"] < 1600 and row["rank"] <= 6:
                return 4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 35 and row["race_type"] == 0 and row["course_len"] < 1600 and row["rank"] <= 6:
                return 2.5
            if row["nobori"] >= 35 and row["race_type"] == 0 and row["course_len"] < 1600 and row["rank"] <= 6:
                return 2.5

            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 36.6 and row["race_type"] == 0 and row["course_len"] >= 1600 and row["rank"] <= 6:
                return 4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600 and row["rank"] <= 6:
                return 2.5
            if row["nobori"] >= 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600 and row["rank"] <= 6:
                return 2.5
            else:
                return 0

        # DataFrame に適用
        base_2["rush_type"] = base_2.apply(calculate_rush_type, axis=1)



        # if base_2["nobori"] < 33 and base_2["race_type"] == 1:
        #     base_2["rush_type"] = -6
        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 33 and base_2["race_type"] == 1:
        #     base_2["rush_type"] = -6
        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 33.9 and base_2["race_type"] == 1:
        #     base_2["rush_type"] = -4
        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34.5 and base_2["race_type"] == 1:
        #     base_2["rush_type"] = -2.5
        # if base_2["nobori"] < 34.5 and base_2["race_type"] == 1:
        #     base_2["rush_type"] = -2

        # if base_2["nobori"] < 33.5 and base_2["race_type"] == 0 and base_2["course_len"] < 1600:
        #     base_2["rush_type"] = -6
        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 33.5 and base_2["race_type"] == 0 and base_2["course_len"] < 1600:
        #     base_2["rush_type"] = -6
        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34.4 and base_2["race_type"] == 0 and base_2["course_len"] < 1600:
        #     base_2["rush_type"] = -4
        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 35 and base_2["race_type"] == 0 and base_2["course_len"] < 1600:
        #     base_2["rush_type"] = -2.5
        # if base_2["nobori"] < 35 and base_2["race_type"] == 0 and base_2["course_len"] < 1600:
        #     base_2["rush_type"] = -2

        # if base_2["nobori"] < 34 and base_2["race_type"] == 0 and base_2["course_len"] >= 1600:
        #     base_2["rush_type"] = -6
        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34 and base_2["race_type"] == 0 and base_2["course_len"] >= 1600:
        #     base_2["rush_type"] = -6
        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34.9 and base_2["race_type"] == 0 and base_2["course_len"] >= 1600:
        #     base_2["rush_type"] = -4
        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 35.5 and base_2["race_type"] == 0 and base_2["course_len"] >= 1600:
        #     base_2["rush_type"] = -2.5
        # if base_2["nobori"] < 35.5 and base_2["race_type"] == 0 and base_2["course_len"] >= 1600:
        #     base_2["rush_type"] = -2


        # if base_2["nobori_diff"] >= 1.2:
        #     base_2["rush_type"] = -4
        # if base_2["nobori_diff"] >= 0.8:
        #     base_2["rush_type"] = -2.5

        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 35.6 and base_2["race_type"] == 1 and base_2["rank"] <= 6:
        #     base_2["rush_type"] = 4
        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 34.5 and base_2["race_type"] == 1 and base_2["rank"] <= 6:
        #     base_2["rush_type"] = 2.5
        # if base_2["nobori"] >= 34.5 and base_2["race_type"] == 1 base_2["rank"] <= 6:
        #     base_2["rush_type"] = 2.5

        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 36.1 and base_2["race_type"] == 0 and base_2["course_len"] < 1600 and base_2["rank"] <= 6:
        #     base_2["rush_type"] = 4
        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 35 and base_2["race_type"] == 0 and base_2["course_len"] < 1600 base_2["rank"] <= 6:
        #     base_2["rush_type"] = 2.5
        # if base_2["nobori"] >= 35 and base_2["race_type"] == 0 and base_2["course_len"] < 1600 and base_2["rank"] <= 6:
        #     base_2["rush_type"] = 2.5

        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 36.6 and base_2["race_type"] == 0 and base_2["course_len"] >=  1600 and base_2["rank"] <= 6:
        #     base_2["rush_type"] = 4
        # if base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 35.5 and base_2["race_type"] == 0 and base_2["course_len"] >=  1600 and base_2["rank"] <= 6:
        #     base_2["rush_type"] = 2.5
        # if base_2["nobori"] >= 35.5 and base_2["race_type"] == 0 and base_2["course_len"] >= 1600 and base_2["rank"] <= 6:
        #     base_2["rush_type"] = 2.5
        # else:
        #     base_2["rush_type"] = 0

        # # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # #
            
        """
        距離指標
        ハイペース+
        ハイローセット	
        レースグレードによる距離のハイペース	
        最初の直線が長いほど	（倍率は下げる）ハイペースになる
        コーナーの数が4以上だと、先行争い諦めることがあるので	若干スローに
        コーナーの数が4以上、直線の合計が1000を超えてくると	若干スローに
        短距離は	基本ハイペース
        最初直線＿上り平坦	スローペース
        最初下り坂	ペースが上がり
            
        12コーナーがきつい	スロー
        コーナーがきつい	スロー
        向正面上り坂	スロー
        タフパック	
        ハイペース	タフ
        コーナー種類	ゆるいと遅くならないのでタフ
        コーナーR	大きいとタフ
        コーナーの数	少ないほうがタフ
        高低差がある	タフ
        馬場状態、天気が悪い	タフ
        芝によって	タフ
        直線合計/コーナー合計	多いほどタフ
        距離	長いほどタフ
        脚質パック	
        季節ごとの馬場状態馬場でも-+補正をいれる	
        スタートが上りorくだり	逃げ先行有利
        偶数ほどよいは先行系	逃げ先行有利
        コーナーまで短いが	逃げ先行有利
            
        4コーナーくだり坂	差し追い込み有利
        ダートの方が	かなり先行優位
        雨	先行有利
        雨のない稍重	差し有利
        内外セット	
        ダートで内枠は不利	外枠
        3,4コーナーが急	圧倒的内枠有利になる
        3,4が下り坂	圧倒的内枠有利になる
        距離が長いほど関係なくなる	関係なくなる
        スタートからコーナーまでの距離が短い	ポジションが取りづらいため、内枠有利特に偶数有利
        スタートが上りorくだり坂	ポジションが取りづらいため、内枠有利特に偶数有利
        コーナーがきついは内枠	内枠
        向正面上り坂は内枠	内枠
        芝スタートだと外が有利	外枠
        芝スタート系は道悪で逆転する	内枠
        """

        # #最大200前後
        # base_2["course_len_pace_diff"] = base_2["course_len"] + (base_2["pace_diff"] * 70)

        # #グレード100前後
        # base_2["course_len_diff_grade"] = base_2["course_len_pace_diff"] + (((base_2['race_grade']/70)-1)*200)

        # #100前後
        # base_2["course_len_diff_grade_slope"] = np.where(
        #     (base_2["season"] == 1) | (base_2["season"] == 4),
        #     base_2["course_len_diff_grade"] + (base_2["goal_slope"] * 30),
        #     base_2["course_len_diff_grade"] + (base_2["goal_slope"] * 13)
        # )

        # #最初の直線の長さ、長いほどきつい、50前後くらい
        # base_2["start_range_processed_1"] = (((base_2["start_range"])-360)/150)
        # base_2["start_range_processed_1"] = base_2["start_range_processed_1"].apply(
        #     lambda x: x if x < 0 else x*0.5
        # )

        # base_2["start_range_processed_course"] = base_2["start_range_processed_1"]*30
        # base_2["course_len_pace_diff_grade_slope_range"] = base_2["course_len_diff_grade_slope"] + (base_2["start_range_processed_course"])

        # # 条件ごとに処理を適用
        # base_2["course_len_diff_grade_slope_range_pace"] = np.where(
        #     ((base_2['race_position'] == 1) | (base_2['race_position'] == 2)) & (base_2["pace_diff"] >= 0),
        #     base_2["course_len_pace_diff_grade_slope_range"] + ((base_2["pace_diff"] / 8) * 100),
            
        #     np.where(
        #         ((base_2['race_position'] == 1) | (base_2['race_position'] == 2)) & (base_2["pace_diff"] < 0),
        #         base_2["course_len_pace_diff_grade_slope_range"] + ((base_2["pace_diff"] / 22)*-100),
                
        #         np.where(
        #             ((base_2['race_position'] == 3) | (base_2['race_position'] == 4))  & (base_2["pace_diff"] < 0),
        #             base_2["course_len_pace_diff_grade_slope_range"] + ((base_2["pace_diff"] / 28) * -100),
                    
        #             np.where(
        #                 ((base_2['race_position'] == 3) | (base_2['race_position'] == 4)) & (base_2["pace_diff"] >= 0),
        #                 base_2["course_len_pace_diff_grade_slope_range"] + ((base_2["pace_diff"] / 13) * 100),
        #                 base_2["course_len_pace_diff_grade_slope_range"]  # どの条件にも当てはまらない場合は元の値を保持
        #             )
        #         )
        #     )
        # )


        # # -4.5 を行う
        # base_2["curve_processed"] = base_2["curve"] - 4.5
        # # +の場合は数値を8倍する
        # base_2["curve_processed"] = base_2["curve_processed"].apply(
        #     lambda x: x * 8 if x > 0 else x
        # )
        # #12コーナーがきついと、ゆるい、-
        # base_2["course_len_diff_grade_slope_range_pace_12curve"] = base_2["course_len_diff_grade_slope_range_pace"] + (base_2["curve_processed"] * 25)

        # #向正面上り坂、ゆるい、-
        # base_2["course_len_diff_grade_slope_range_pace_12curve_front"] = base_2["course_len_diff_grade_slope_range_pace_12curve"] - (base_2["flont_slope"] * 25)



        # #最大0.02*n
        # def calculate_course_len_pace_diff(row):
        #     if row["curve_amount"] == 0:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"]
        #     elif row["curve_amount"] <= 2:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R34"]-100)/3)
        #     elif row["curve_amount"] <= 3:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 / 2 + (row["curve_R34"]-100)/4)
        #     elif row["curve_amount"] <= 4:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 + ((row["curve_R34"]-100)/4))
        #     elif row["curve_amount"] <= 5:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 + ((row["curve_R34"]-100)/4) * 3 / 2)
        #     elif row["curve_amount"] <= 6:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"] +((row["curve_R12"]-100)/4 + ((row["curve_R34"]-100)/4) * 2)
        #     elif row["curve_amount"] <= 7:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 * 3 / 2 + ((row["curve_R34"]-100)/4) * 2)
        #     else:  # curve_amount <= 8
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 * 2 +((row["curve_R34"]-100)/4) * 2)

        # base_2["course_len_diff_grade_slope_range_pace_12curve_front_R"] = base_2.apply(calculate_course_len_pace_diff, axis=1)

        # #最大0.09くらい
        # base_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] = base_2["course_len_diff_grade_slope_range_pace_12curve_front_R"] + (base_2["height_diff"]*50)


        # # 条件ごとに適用
        # base_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate"] = np.where(
        #     ((base_2["ground_state"] == 1) | (base_2["ground_state"] == 3)) & (base_2["race_type"] == 1),
        #     base_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] + 400,

        #     np.where(
        #         (base_2["ground_state"] == 2) & (base_2["race_type"] == 1),
        #         base_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] + 120,

        #         np.where(
        #             ((base_2["ground_state"] == 1) | (base_2["ground_state"] == 3)) & (base_2["race_type"] == 0),
        #             base_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] - 100,

        #             np.where(
        #                 (base_2["ground_state"] == 2) & (base_2["race_type"] == 0),
        #                 base_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] - 50,
                        
        #                 # どの条件にも当てはまらない場合は元の値を保持
        #                 base_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height"]
        #             )
        #         )
        #     )
        # )

        # base_2 = base_2.copy()
        # base_2.loc[:, "place_season_condition_type_categori_processed"] = (
        #     base_2["place_season_condition_type_categori"]
        #     .replace({5: -0.3, 4: -0.17, 3: 0, 2: 0.17,1: 0.3, -1: -0.18, -2: 0, -3: 0.18,-4:0.3,-10000:0})
        # ).astype(float)

        # base_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place"] = (
        #     base_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate"] + (base_2["place_season_condition_type_categori_processed"]*-500)
        #     )

        # #最大0.05くらい
        # base_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight"] = (
        #     base_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place"] - (((base_2["straight_total"]/ base_2["course_len"])-0.5)*400)
        #     )







        # # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # #

        """
        ・rank_diffの策定
        rank_diffは
        _その他補正のみ（内外、着差のつきやすさ）
        その他補正のみ+脚質補正（向いていない展開なら展開なら評価）
        その他補正のみ+タイプ補正（向いていない展開なら展開なら評価）
        両方補正
        の４つ分作る



        ❶スローペースは着差がつきにくく
        ❷ハイペースは着差がつきやすい
        ❸短距離戦は着差がつきにくい
        ❹道悪は着差がつきやすい

        内外セット	
        ダートで内枠は不利	外枠
        3,4コーナーが急	圧倒的内枠有利になる
        3,4が下り坂	圧倒的内枠有利になる
        距離が長いほど関係なくなる	関係なくなる
        スタートからコーナーまでの距離が短い	ポジションが取りづらいため、内枠有利特に偶数有利
        スタートが上りorくだり坂	ポジションが取りづらいため、内枠有利特に偶数有利
        コーナーがきついは内枠	内枠
        向正面上り坂は内枠	内枠
        芝スタートだと外が有利	外枠
        芝スタート系は道悪で逆転する	内枠
            

        レベル評価
        馬場状態、シーズン、グレード込みの平均とdiffする
        著しく離れている場合、
        (1200m未勝利で0.4秒以上)
        重賞で0,7
        2000 で1.5秒以上
        重賞で1.8秒以上
        離れていた場合
        top1平均とtop1のタイムがどれだけ離れているか
        それ以上離れていたら、すごかったら評価し、遅かったら評価しない
        範囲内ならそのまま
        範囲外で評価する
        """

        # #最大前後0.2、ハイペースは+
        # base_2["rank_diff_pace_diff"] = base_2["rank_diff"] + (base_2["pace_diff"] /25)


        # # 1600で正規化
        # base_2["course_len_processed_rd"] = (base_2["course_len"] / 1600) - 1

        # # 1600m未満ならそのまま、1600m以上なら緩やかに上昇 
        # base_2["course_len_processed_rd"] = base_2["course_len_processed_rd"].apply(
        #     lambda x: x/2 if x <= 0 else x/3
        # )
        # #-0.25
        # #長距離のほうが着差の価値がなくなる

        # base_2["rank_diff_pace_course_len"] = base_2["rank_diff_pace_diff"] * ((base_2["course_len_processed_rd"]+25)/25)



        # # 条件ごとに適用,馬場状態が悪い状態ほど評価（-）
        # base_2["rank_diff_pace_course_len_ground_state"] = np.where(
        #     ((base_2["ground_state"] == 1) | (base_2["ground_state"] == 3)) & (base_2["race_type"] == 1),
        #     base_2["rank_diff_pace_course_len"] - 0.3,

        #     np.where(
        #         (base_2["ground_state"] == 2) & (base_2["race_type"] == 1),
        #         base_2["rank_diff_pace_course_len"] - 0.12,

        #         np.where(
        #             ((base_2["ground_state"] == 1) | (base_2["ground_state"] == 3)) & (base_2["race_type"] == 0),
        #             base_2["rank_diff_pace_course_len"] + 0.1,

        #             np.where(
        #                 (base_2["ground_state"] == 2) & (base_2["race_type"] == 0),
        #                 base_2["rank_diff_pace_course_len"] + 0.05,
                        
        #                 # どの条件にも当てはまらない場合は元の値を保持
        #                 base_2["rank_diff_pace_course_len"]
        #             )
        #         )
        #     )
        # )




        # #0,01-0.01,内がプラス(内枠が有利を受けるとしたら、rank_diffは+にして、有利ポジはマイナス補正)
        # base_2["umaban_rank_diff_processed"] = base_2["umaban"].apply(
        #     lambda x: ((x*-1.5)+1.5) if x < 4 else ((x-8)/1.5)-1
        # ).astype(float)
        # #0,-0.1,-0.3,-0.36,-0.3,-0.23,（-1/10のとき）
        # base_2["umaban_rank_diff_processed"] = base_2["umaban_rank_diff_processed"] * (1/10)
        # #0 , -0.05
        # #1（奇数）または 0（偶数）,偶数が有利
        # base_2.loc[:, "umaban_odd_rank_diff_processed"] = (
        #     (base_2["umaban_odd"]-1)/10
        # ).astype(float)

        # #rdが-0.25,,0.25が0.5に
        # base_2["umaban_rank_diff_processed_2"] = base_2["umaban_rank_diff_processed"] / ((base_2["course_len_processed_rd"]*2) + 1)
        # base_2["umaban_odd_rank_diff_processed_2"] = base_2["umaban_odd_rank_diff_processed"] / ((base_2["course_len_processed_rd"]*2)+1)

        # #不利が-,ダートは外枠有利,0.06
        # base_2["rank_diff_pace_course_len_ground_state_type"] = np.where(
        #     base_2["race_type"] == 0,
        #     base_2["rank_diff_pace_course_len_ground_state"] / ((base_2["umaban_rank_diff_processed_2"]+12)/12),
        #     base_2["rank_diff_pace_course_len_ground_state"] * ((base_2["umaban_rank_diff_processed_2"]+12)/12)
        # )

        # base_2["rank_diff_pace_course_len_ground_state_type_odd"]= np.where(
        #     base_2["race_type"] == 0,
        #     base_2["rank_diff_pace_course_len_ground_state_type"] * ((base_2["umaban_odd_rank_diff_processed_2"]+7)/7),
        #     base_2["rank_diff_pace_course_len_ground_state_type"] * ((base_2["umaban_odd_rank_diff_processed_2"]+7)/7)
        # )

        # # -4.5 を行う
        # base_2["curve_processed"] = base_2["curve"] - 4.5
        # # +の場合は数値を8倍する
        # base_2["curve_processed"] = base_2["curve_processed"].apply(
        #     lambda x: x * 8 if x > 0 else x
        # )
        # #最初の直線の長さ、長いほどきつい、50前後くらい
        # base_2["start_range_processed"] = (((base_2["start_range"])-360)/150)
        # base_2["start_range_processed"] = base_2["start_range_processed"].apply(
        #     lambda x: x if x < 0 else x*0.5
        # )
        # base_2["start_slope_abs"] = base_2["start_slope"].abs()
        # base_2["start_slope_abs_processed"] = base_2["start_slope_abs"] /4

        # #last急カーブ,フリ評価
        # base_2["rank_diff_pace_course_len_ground_state_type_odd_curve"] = (
        #     base_2["rank_diff_pace_course_len_ground_state_type_odd"] * ((((base_2["umaban_rank_diff_processed_2"]*((base_2["curve_processed"]/4))))+12)/12)
        # )

        # #3カーブ下り坂,フリ評価
        # base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope"] = (
        #     base_2["rank_diff_pace_course_len_ground_state_type_odd_curve"] * ((((base_2["umaban_rank_diff_processed_2"]*(base_2["last_curve_slope"]/3)))+12)/12)
        # )

        # #スタートからコーナー
        # base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start"] = (
        #     base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope"] / (((base_2["umaban_rank_diff_processed_2"]*(((base_2["start_range_processed"]))*-1/1.2)- ((base_2["umaban_odd_rank_diff_processed_2"]*(base_2["start_range_processed"]))*-1/1.2))+12)/12)
        # )


        # #+-0.06,スタートからコーナー、坂,上り下り両方
        # base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] = (
        #     base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start"] / (((((base_2["umaban_rank_diff_processed_2"]*(base_2["start_slope_abs_processed"]))*-1)- ((base_2["umaban_odd_rank_diff_processed_2"]*( base_2["start_slope_abs_processed"]))*-1))+12)/12)
        # )

        # #最大0.3*nコーナーがきついは内枠
        # def calculate_rank_diff_pace_diff_2(row):
        #     if row["curve_amount"] == 0:
        #         return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"]
        #     elif row["curve_amount"] <= 2:
        #         return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] / (((row["umaban_rank_diff_processed_2"]*(((row["curve_R34"]-100)/120)*-1))+20)/20)
        #     elif row["curve_amount"] <= 3:
        #         return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] / (((row["umaban_rank_diff_processed_2"]*(((row["curve_R12"]-100)/120 / 2 + (row["curve_R34"]-100)/120)*-1))+20)/20)
        #     elif row["curve_amount"] <= 4:
        #         return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] / (((row["umaban_rank_diff_processed_2"]*(((row["curve_R12"]-100)/120 + ((row["curve_R34"]-100)/120))*-1))+20)/20)
        #     elif row["curve_amount"] <= 5:
        #         return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] / (((row["umaban_rank_diff_processed_2"]*(((row["curve_R12"]-100)/120 + ((row["curve_R34"]-100)/120) * 3 / 2)*-1))+20)/20)
        #     elif row["curve_amount"] <= 6:
        #         return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] /((((row["umaban_rank_diff_processed_2"]*((row["curve_R12"]-100)/120 + ((row["curve_R34"]-100)/120) * 2)*-1))+20)/20)
        #     elif row["curve_amount"] <= 7:
        #         return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] / (((row["umaban_rank_diff_processed_2"]*(((row["curve_R12"]-100)/120 * 3 / 2 + ((row["curve_R34"]-100)/120) * 2)*-1))+20)/20)
        #     else:  # curve_amount <= 8
        #         return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] /((((row["umaban_rank_diff_processed_2"]*((row["curve_R12"]-100)/120 * 2 +((row["curve_R34"]-100)/120) * 2)*-1))+20)/20)



        # base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff"] = base_2.apply(calculate_rank_diff_pace_diff_2, axis=1)


        # #最大1*向正面
        # base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont"] = (
        #     base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff"] / ((((base_2["umaban_rank_diff_processed_2"]*(base_2["flont_slope"]/8)))+20)/20)
        # )

        # #芝スタートかつ良馬場、芝スタートかつ良以外、どっちでもない場合,外評価

        # condition_rank_diff = (base_2["start_point"] == 2) & (base_2["ground_state"] == 0)

        # base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] = np.where(
        #     condition_rank_diff,
        #     base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont"] / (((base_2["umaban_rank_diff_processed_2"] * -1.5)+20)/20),
        #     base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont"] / (((base_2["umaban_rank_diff_processed_2"])+30)/30)
        # )













        # ❶スローペースは着差がつきにくく
        # ❷ハイペースは着差がつきやすい
        # ❸短距離戦は着差がつきにくい
        # ❹道悪は着差がつきやすい
        base_2 = base_2.copy()
        base_2.loc[:, "place_season_condition_type_categori_processed"] = (
            base_2["place_season_condition_type_categori"]
            .replace({5: -0.3, 4: -0.17, 3: 0, 2: 0.17,1: 0.3, -1: -0.18, -2: 0, -3: 0.18,-4:0.3,-10000:0})
        ).astype(float)

        #高速馬場だと差がつく
        #着差がつかないのを、１，１倍で掛け算して0.4引く

        base_2["rank_diff_pace_diff"] = base_2["rank_diff"]+ 0.5



        # 条件リスト
        conditions_1 = [
            (base_2["pace_diff"] >= 1),  # pace_diff が 1 以上
            (base_2["place_season_condition_type_categori_processed"] <= -0.1),  # place_season_condition_type_categori_processed が -0.1 以下
            (base_2["course_len"] >= 2400),  # course_len が 2400 以上
            (base_2["ground_state"] != 0),  # ground_state が 0 以外
        ]

        conditions_2 = [
            (base_2["pace_diff"] <= -1),  # pace_diff が -1 以下
            (base_2["rank_diff"] >= 0.1),  # rank_diff が 0.1 以上
            (base_2["course_len"] <= 1600),  # course_len が 1600 以下
        ]

        # それぞれの条件に該当する場合の計算
        for cond in conditions_1:
            base_2.loc[cond, "rank_diff_pace_diff"] = base_2.loc[cond, "rank_diff_pace_diff"] * 0.9 + 0.25

        for cond in conditions_2:
            base_2.loc[cond, "rank_diff_pace_diff"] = base_2.loc[cond, "rank_diff_pace_diff"] * 1.1 - 0.25





        base_2["goal_range_processed_1"] = (((base_2["goal_range"])-360))
        base_2["goal_range_processed_1"] = base_2["goal_range_processed_1"].apply(
            lambda x: x*2 if x < 0 else x*0.5
        )

        """
        ハイスロー、脚質修正
        コーナー順位が前（先行）で、ハイペースの場合、rank_diffをさらに-0.5する（不利条件）
        前でローペースの場合、rank_diffを+0.2する(ペースによる)有利
        後ろで、ローペースの場合、rank_diffを-0.1する（作っておけばrankdiffで使える）不利
        後ろでハイの場合、rank_diffを+0.3する(ハイスローの分を相殺する)有利
        #+だとハイペース、ーだとスローペース
        """
        #着差がつかない,不利を-側へ
        # 条件ごとに処理を適用
        base_2["rank_diff_pace_diff_slope_range_pace"] = np.where(
            ((base_2['race_position'] == 1)),
            base_2["rank_diff_pace_diff"] - (base_2["pace_diff"] / 17),
            np.where(
                    ((base_2['race_position'] == 2)),
                    base_2["rank_diff_pace_diff"] - (base_2["pace_diff"] / 30),
                
                    np.where(
                        (base_2['race_position'] == 3),
                        base_2["rank_diff_pace_diff"] - ((base_2["pace_diff"] / 20) * -1),
                        
                        np.where(
                            ((base_2['race_position'] == 4)),
                            base_2["rank_diff_pace_diff"] - ((base_2["pace_diff"] / 18) * -1),
                            base_2["rank_diff_pace_diff"]  # どの条件にも当てはまらない場合は元の値を保持
                        )
                    
                )
            )
        )
        # # ❹道悪は着差がつきやすい
        # # 条件ごとに適用
        # base_2["rank_diff_pace_diff_slope_range_groundstate"] = np.where(
        #     ((base_2["ground_state"] == 1) | (base_2["ground_state"] == 3)) & (base_2["race_type"] == 1),#芝で道悪
        #     ((base_2["rank_diff_pace_diff_slope_range_pace"] * 8/10)+0.2),

        #     np.where(
        #         (base_2["ground_state"] == 2) & (base_2["race_type"] == 1),
        #         ((base_2["rank_diff_pace_diff_slope_range_pace"] * 8.5/10)+0.15),

        #         np.where(
        #             ((base_2["ground_state"] == 1) | (base_2["ground_state"] == 3)) & (base_2["race_type"] == 0),
        #             base_2["rank_diff_pace_diff_slope_range_pace"] * (41/40),

        #             np.where(
        #                 (base_2["ground_state"] == 2) & (base_2["race_type"] == 0),
        #                 base_2["rank_diff_pace_diff_slope_range_pace"] * (81/80),
                        
        #                 # どの条件にも当てはまらない場合は元の値を保持
        #                 base_2["rank_diff_pace_diff_slope_range_pace"]
        #             )
        #         )
        #     )
        # )

        base_2["rank_diff_pace_diff_slope_range_groundstate"] = base_2["rank_diff_pace_diff_slope_range_pace"]

        # 1600で正規化,-0.5 - 1
        base_2["course_len_processed"] = (base_2["course_len"] / 1700)-1

        # ,-1.5 - 4
        base_2["course_len_processed"] = base_2["course_len_processed"].apply(
            lambda x: x*3 if x <= 0 else x*4
        )
        base_2["course_len_processed_rank_diff"] = ((base_2["course_len_processed"] + 10)/10)

        # # ❸短距離戦は着差がつきにくい
        # base_2["rank_diff_pace_diff_slope_range_groundstate"] = np.where(
        #     (base_2["course_len_processed"] <= 0),#芝で道悪
        #     ((base_2["rank_diff_pace_diff_slope_range_groundstate"] * 11.2/10)-(0.2*base_2["course_len_processed_rank_diff"])),

        #     np.where(
        #         (base_2["course_len_processed"] > 0),
        #         ((base_2["rank_diff_pace_diff_slope_range_pace"] * 8/10)+(0.2*base_2["course_len_processed_rank_diff"])),
        #         base_2["rank_diff_pace_diff_slope_range_groundstate"]
        #     )

        # )

        # conditions = [
        #     (base_2["course_len_processed"] <= 0),  # 芝で道悪
        #     (base_2["course_len_processed"] > 0)
        # ]

        # choices = [
        #     (base_2["rank_diff_pace_diff_slope_range_groundstate"] * 11.2 / 10) - (0.2 * base_2["course_len_processed_rank_diff"]),
        #     (base_2["rank_diff_pace_diff_slope_range_pace"] * 8 / 10) + (0.2 * base_2["course_len_processed_rank_diff"])
        # ]

        # base_2["rank_diff_pace_diff_slope_range_groundstate"] = np.select(conditions, choices, default=base_2["rank_diff_pace_diff_slope_range_groundstate"])



        base_2["start_range_processed_1"] = (((base_2["start_range"])-360))
        base_2["start_range_processed_1"] = base_2["start_range_processed_1"].apply(
            lambda x: x if x < 0 else x*0.8
        )


        # -4.5 を行う
        base_2["curve_processed"] = base_2["curve"] - 4.5
        # +の場合は数値を8倍する
        base_2["curve_processed"] = base_2["curve_processed"].apply(
            lambda x: x * 8 if x > 0 else x
        )



        # ペースに関係ある要素は弱体化
        base_2["rank_diff_pace_diff_slope_range_groundstate_position"] = np.where(
            ((base_2['race_position'] == 1) | (base_2['race_position'] == 2)),
            base_2["rank_diff_pace_diff_slope_range_groundstate"] 
            / ((2000 + base_2["start_range_processed_1"]) / 2000) 
            * ((100 + base_2["start_slope"]) / 100) 
            * ((1000 + base_2["curve_processed"]) / 1000) 
            / ((base_2["goal_range_processed_1"] + 1000) / 1000) 
            / ((base_2["goal_slope"] + 45) / 45) 
            / ((base_2["place_season_condition_type_categori_processed"] + 9) / 9) 
            / ((base_2["race_type"] + 59) / 60)
            / ((base_2["course_len_processed"] + 400)/400),  # ここでカンマ

            np.where(
                ((base_2['race_position'] == 3) | (base_2['race_position'] == 4)),
                base_2["rank_diff_pace_diff_slope_range_groundstate"] 
                * ((2000 + base_2["start_range_processed_1"]) / 2000) 
                / ((100 + base_2["start_slope"]) / 100) 
                / ((1000 + base_2["curve_processed"]) / 1000) 
                * ((base_2["goal_range_processed_1"]+ 1000) / 1000) 
                * ((base_2["goal_slope"] + 45) / 45) 
                * ((base_2["place_season_condition_type_categori_processed"] + 9) / 9) 
                * ((base_2["race_type"] +59) / 60)
                * ((base_2["course_len_processed"] + 400)/400), 

                base_2["rank_diff_pace_diff_slope_range_groundstate"]
            )
        )






        # # 月を抽出して開催シーズンを判定
        # def determine_season_turf(month):
        #     if 6 <= month <= 8:
        #         return "4" #"夏開催"
        #     elif month == 12 or 1 <= month <= 2:
        #         return "2" #"冬開催"
        #     elif 3 <= month <= 5:
        #         return "3" #"春開催"
        #     elif 9 <= month <= 11:
        #         return "1" #"秋開催"    
        
        # base_2["season_turf"] = base_2["date"].dt.month.map(determine_season_turf)
        # base_2["day"] = base_2["day"].astype(str)

        # base_2["day_season_turf"] =  base_2["day"] + base_2["season_turf"]
        # base_2["day_season_turf"] =  base_2["day_season_turf"].astype(int)
        # base_2["day"] = base_2["day"].astype(int)

        # #umaban

        # base_2["season_turf_condition"] = np.where(
        #     base_2["season"] == 1, base_2["day"],
        #     np.where(
        #         base_2["season"] == 2, (base_2["day"] + 1.5) * 1.5,
        #         np.where(
        #             base_2["season"] == 3, base_2["day"] + 3,
        #             np.where(
        #                 base_2["season"] == 4, base_2["day"] + 4,
        #                 base_2["day"]  # それ以外のとき NaN
        #             )
        #         )
        #     )
        # )



        #-2-2,内がマイナス
        base_2["umaban_processed"] = base_2["umaban"].apply(
            lambda x: ((x*-1)) if x < 4 else ((x-8)/3)-1
        ).astype(float)
        #0-0.005

        base_2["umaban_judge"] = (base_2["umaban"].astype(float)/base_2["n_horses"].astype(float))-0.55

        #1（奇数）または 0（偶数）
        base_2.loc[:, "umaban_odd_processed"] = (
            (base_2["umaban_odd"]-1)
        ).astype(float)

        # # 1600で正規化,-0.5 - 1
        # base_2["course_len_processed"] = (base_2["course_len"] / 1700)-1

        # # ,-1.5 - 4
        # base_2["course_len_processed"] = base_2["course_len_processed"].apply(
        #     lambda x: x*3 if x <= 0 else x*4
        # )
        base_2["course_len_processed_2"] = ((base_2["course_len_processed"] + 3)/3)


        base_2["umaban_processed_2"] = base_2["umaban_processed"] / base_2["course_len_processed_2"]
        base_2["umaban_odd_processed_2"] = base_2["umaban_odd_processed"] / base_2["course_len_processed_2"]



        base_2["first_corner"] = np.where(
            (base_2["curve_amount"] == 2) | (base_2["curve_amount"] == 6),
            base_2["curve_R34"],
            np.where(
                (base_2["curve_amount"] == 4) | (base_2["curve_amount"] == 8),
                base_2["curve_R12"],
                0  # それ以外のとき 0
            )
        )



        # # 内が小さい
        # base_2["rank_diff_pace_diff_slope_range_groundstate_position_umaban"] = np.where(
        #     (base_2["umaban_judge"] < 0),
        #     base_2["rank_diff_pace_diff_slope_range_groundstate_position"] /
        #     (
        #         ((base_2["umaban_processed_2"] + 80) / 80)  # 少ないほうがrank_diffが増える
        #         * ((base_2["umaban_odd_processed_2"] + 40) / 40)  # 奇数不利なので分母を増やして総合を減らす
        #         * (((base_2["start_point"] - 1) + 40) / 40)  # 外枠が有利なので分母を増やして総合を減らす
        #         * ((base_2["curve_processed"] + 100) / 100)  # ラストカーブきついほど数値が減る
        #         * ((base_2["last_curve_slope"] + 100) / 100)  # ラストカーブくだりほど数値が減る
        #         * (((base_2["season_turf_condition"] - 7) + 40) / 40)  # 馬場状態が良いほど数値が減る
        #         * (((base_2["race_type"] - 0.5) + 20) / 20)  # 芝ほど数値が減る
        #         / (((base_2["first_corner"] - 100) + 10000) / 10000)  # 最初のコーナーがでかいほど数値が減る
        #     ),

        #     np.where(
        #         (base_2["umaban_judge"] >= 0),
        #         base_2["rank_diff_pace_diff_slope_range_groundstate_position"] /
        #         (
        #             ((base_2["umaban_processed_2"] + 80) / 80)  # 少ないほうがrank_diffが増える
        #             * ((base_2["umaban_odd_processed_2"] + 40) / 40)  # 奇数不利なので分母を増やして総合を減らす
        #             / (((base_2["start_point"] - 1) + 40) / 40)  # 外枠が有利なので分母を増やして総合を減らす
        #             / ((base_2["curve_processed"] + 100) / 100)  # ラストカーブきついほど数値が減る
        #             / ((base_2["last_curve_slope"] + 100) / 100)  # ラストカーブくだりほど数値が減る
        #             / (((base_2["season_turf_condition"] - 7) + 40) / 40)  # 馬場状態が良いほど数値が減る
        #             / (((base_2["race_type"] - 0.5) + 20) / 20)  # 芝ほど数値が減る
        #             * (((base_2["first_corner"] - 100) + 10000) / 10000)  # 最初のコーナーがでかいほど数値が減る
        #         ),


        #         base_2["rank_diff_pace_diff_slope_range_groundstate_position"]
        #     )
        # )

        base_2["umaban_processed_abs2"] = base_2["umaban_processed_2"].abs()

        # 内が小さい,最大50くらいになってしまう
        base_2["rank_diff_pace_diff_slope_range_groundstate_position_umaban"] = np.where(
            (base_2["umaban_judge"] < 0),
            base_2["rank_diff_pace_diff_slope_range_groundstate_position"] /
            ((
                ((base_2["umaban_processed_abs2"]) # 少ないほうがtimeが増える-4.5 から3
                * (
                    base_2["umaban_odd_processed_2"]# 奇数不利なので分母を増やして総合を減らす 1
                        +  (base_2["start_point"] - 1)# 外枠が有利なので分母を増やして総合を減らす 1
                        +  (base_2["curve_processed"]*1.3)# ラストカーブきついほど数値が減る6
                        +  (base_2["last_curve_slope"]*1.5)# ラストカーブくだりほど数値が減る4
                        +  ((base_2["season_turf_condition"] - 7)*1.5)# 馬場状態が良いほど数値が減る 7-7
                        +  (base_2["race_type"] - 0.5)*4# 芝ほど数値が減る 2
                        -  ((base_2["first_corner"] - 100)/50)# 最初のコーナーがでかいほど数値が減る1
                ) 
            ) +500) /500)
            ,
            

            np.where(
                (base_2["umaban_judge"] >= 0),
                base_2["rank_diff_pace_diff_slope_range_groundstate_position"] /
                ((
                    ((base_2["umaban_processed_abs2"]) # 少ないほうがtimeが増える-4.5 から3
                    * (
                        base_2["umaban_odd_processed_2"]# 奇数不利なので分母を増やして総合を減らす 1
                        -  (base_2["start_point"] - 1)# 外枠が有利なので分母を増やして総合を減らす 1
                        -  (base_2["curve_processed"]*1.3)# ラストカーブきついほど数値が減る4
                        -  (base_2["last_curve_slope"]*1.5)# ラストカーブくだりほど数値が減る2
                        -  ((base_2["season_turf_condition"] - 7)*1.5)# 馬場状態が良いほど数値が減る 7-7
                        -  (base_2["race_type"] - 0.5)*4# 芝ほど数値が減る 2
                        +  ((base_2["first_corner"] - 100)/50)# 最初のコーナーがでかいほど数値が減る1
                    ) 
                ) +500) / 500),

                base_2["rank_diff_pace_diff_slope_range_groundstate_position"]
            )
        )





        # シーズンを追加
        base_2["distance_place_type_ground_state_grade_season"] = base_2["distance_place_type_ground_state_grade"].astype(int)

        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_ground_state_grade_season"] = df_old["distance_place_type_ground_state_grade"].astype(int)
        
        target_mean_1= (
            df_old[
            (df_old['rank'].isin([1]))  # rankが1, 2, 3
            ]
            .groupby("distance_place_type_ground_state_grade_season")["time"]
            .mean()
        )
        df_old["distance_place_type_ground_state_grade_season_time_encoded"] = df_old["distance_place_type_ground_state_grade_season"].map(target_mean_1).fillna(0)
        

        df_old_rank_diff = df_old[["distance_place_type_ground_state_grade_season","distance_place_type_ground_state_grade_season_time_encoded"]]
        
        columns_to_merge = [("distance_place_type_ground_state_grade_season","distance_place_type_ground_state_grade_season_time_encoded")]
        
        
        for original_col, encoded_col in columns_to_merge:
            df2_subset = df_old_rank_diff[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
            df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
            base_2 = base_2.merge(df2_subset, on=original_col, how='left')  # dfにマージ
            

        base_2['no1_time'] = base_2.apply(
            lambda row: row['time'] if row['rank'] == 1 else row['time'] - (row['rank_diff']-2),
            axis=1
        )

        base_2["time_class"] = base_2["distance_place_type_ground_state_grade_season_time_encoded"] - base_2['no1_time']

        base_2["time_class_abs"] = base_2["time_class"].abs()

        # if base_2["course_len"] < 1600:
        #     if base_2["race_grade"] < 80 and base_2["time_class_abs"] >0.7 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if base_2["race_grade"] < 80 and base_2["time_class_abs"] >0.7 and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 80 <= base_2["race_grade"] <= 87 and base_2["time_class_abs"] >1 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 80 <= base_2["race_grade"] <= 87 and base_2["time_class_abs"] >1 and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 88 <= base_2["race_grade"] and base_2["time_class_abs"] >1.2 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 88 <= base_2["race_grade"]and base_2["time_class_abs"] >1.2 and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        # if 1600 <= base_2["course_len"] and base_2["race_type"] == 1:
        #     if base_2["race_grade"] < 80 and base_2["time_class_abs"] >1.2 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if base_2["race_grade"] < 80 and base_2["time_class_abs"]>1.2  and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 80 <= base_2["race_grade"] <= 87 and base_2["time_class_abs"] >1.5 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 80 <= base_2["race_grade"] <= 87 and base_2["time_class_abs"] >1.5 and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 88 <= base_2["race_grade"] and base_2["time_class_abs"]  >1.7 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 88 <= base_2["race_grade"]and base_2["time_class_abs"] >1.7 and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )

        # if 1600 <= base_2["course_len"] and base_2["race_type"] == 0:
        #     if base_2["race_grade"] < 80 and base_2["time_class_abs"] >1 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if base_2["race_grade"] < 80 and base_2["time_class_abs"]>1  and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 80 <= base_2["race_grade"] <= 87 and base_2["time_class_abs"] >1.3 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 80 <= base_2["race_grade"] <= 87 and base_2["time_class_abs"] >1.3 and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 88 <= base_2["race_grade"] and base_2["time_class_abs"]  >1.5 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 88 <= base_2["race_grade"]and base_2["time_class_abs"] >1.5 and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )


        # if 2400 <= base_2["course_len"] and base_2["race_type"] == 1:
        #     if base_2["race_grade"] < 80 and base_2["time_class_abs"] >1.8 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if base_2["race_grade"] < 80 and base_2["time_class_abs"]>1.8  and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 80 <= base_2["race_grade"] <= 87 and base_2["time_class_abs"] >2.1 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 80 <= base_2["race_grade"] <= 87 and base_2["time_class_abs"] >2.1 and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 88 <= base_2["race_grade"] and base_2["time_class_abs"]  >2.3 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 88 <= base_2["race_grade"]and base_2["time_class_abs"] >2.3 and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )

        # if 2400 <= base_2["course_len"] and base_2["race_type"] == 0:
        #     if base_2["race_grade"] < 80 and base_2["time_class_abs"] >1.5 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if base_2["race_grade"] < 80 and base_2["time_class_abs"]>1.5  and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 80 <= base_2["race_grade"] <= 87 and base_2["time_class_abs"] >1.8 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 80 <= base_2["race_grade"] <= 87 and base_2["time_class_abs"] >1.8 and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 88 <= base_2["race_grade"] and base_2["time_class_abs"]  >2 and base_2["time_class"] > 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 88 <= base_2["race_grade"]and base_2["time_class_abs"] >2 and base_2["time_class"] < 0:
        #         base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             base_2["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )


        def calculate_course_len_pace_diff(row):
            # 初期値として元のrank_diffを設定
            result = row["rank_diff_pace_diff_slope_range_groundstate_position"]
            
            if row["course_len"] < 1600:
                if row["race_grade"] < 80 and row["time_class_abs"] > 0.7 and row["time_class"] > 0:
                    result -= 0.15
                elif row["race_grade"] < 80 and row["time_class_abs"] > 0.7 and row["time_class"] < 0:
                    result += 0.15
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1 and row["time_class"] > 0:
                    result -= 0.15
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1 and row["time_class"] < 0:
                    result += 0.15
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 1.2 and row["time_class"] > 0:
                    result -= 0.15
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 1.2 and row["time_class"] < 0:
                    result += 0.15

            if 1600 <= row["course_len"] and row["race_type"] == 1:
                if row["race_grade"] < 80 and row["time_class_abs"] > 1.2 and row["time_class"] > 0:
                    result -= 0.15
                elif row["race_grade"] < 80 and row["time_class_abs"] > 1.2 and row["time_class"] < 0:
                    result += 0.15
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1.5 and row["time_class"] > 0:
                    result -= 0.15
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1.5 and row["time_class"] < 0:
                    result += 0.15
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 1.7 and row["time_class"] > 0:
                    result -= 0.15
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 1.7 and row["time_class"] < 0:
                    result += 0.15

            if 1600 <= row["course_len"] and row["race_type"] == 0:
                if row["race_grade"] < 80 and row["time_class_abs"] > 1 and row["time_class"] > 0:
                    result -= 0.15
                elif row["race_grade"] < 80 and row["time_class_abs"] > 1 and row["time_class"] < 0:
                    result += 0.15
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1.3 and row["time_class"] > 0:
                    result -= 0.15
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1.3 and row["time_class"] < 0:
                    result += 0.15
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 1.5 and row["time_class"] > 0:
                    result -= 0.15
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 1.5 and row["time_class"] < 0:
                    result += 0.15

            if 2400 <= row["course_len"] and row["race_type"] == 1:
                if row["race_grade"] < 80 and row["time_class_abs"] > 1.8 and row["time_class"] > 0:
                    result -= 0.15
                elif row["race_grade"] < 80 and row["time_class_abs"] > 1.8 and row["time_class"] < 0:
                    result += 0.15
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 2.1 and row["time_class"] > 0:
                    result -= 0.15
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 2.1 and row["time_class"] < 0:
                    result += 0.15
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 2.3 and row["time_class"] > 0:
                    result -= 0.15
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 2.3 and row["time_class"] < 0:
                    result += 0.15

            if 2400 <= row["course_len"] and row["race_type"] == 0:
                if row["race_grade"] < 80 and row["time_class_abs"] > 1.5 and row["time_class"] > 0:
                    result -= 0.15
                elif row["race_grade"] < 80 and row["time_class_abs"] > 1.5 and row["time_class"] < 0:
                    result += 0.15
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1.8 and row["time_class"] > 0:
                    result -= 0.15
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1.8 and row["time_class"] < 0:
                    result += 0.15
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 2 and row["time_class"] > 0:
                    result -= 0.15
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 2 and row["time_class"] < 0:
                    result += 0.15

            return result

        base_2["rank_diff_correction"] = base_2.apply(calculate_course_len_pace_diff, axis=1)


        # """
        # その他補正のみ+脚質補正（向いていない展開なら展開なら評価）
        # 脚質パック	
        # 季節ごとの馬場状態馬場でも-+補正をいれる	
        # 軽い芝、先行有利
        # 重い芝、差し有利

        # スタートが上りorくだり	逃げ先行有利
        # 偶数ほどよいは先行系	逃げ先行有利
        # コーナーまで短いが	逃げ先行有利
        #     レースグレードがあがると	差し有利

        # 4コーナーくだり坂	差し追い込み有利
        # ダートの方が	かなり先行優位
        # 雨	先行有利
        # 雨のない稍重	差し有利

        # ハイペースが+ローが-(pace_diff)
        # """
        # base_2 = base_2.copy()
        # base_2.loc[:, "place_season_condition_type_categori_processed_rank_diff"] = (
        #     base_2["place_season_condition_type_categori"]
        #     .replace({5: -0.09, 4: -0.05, 3: 0, 2: 0.05,1: 0.09, -1: -0.08, -2: 0, -3: 0.08,-4:0.11,-10000:0})
        # ).astype(float)

        # # #向いていない場合は-
        # # if base_2['race_position'] == 1 or 2:
        # #     if base_2['weather'] == 1 and base_2['ground_state'] == 2:
        # #         base_2["rank_diff_correction_position"] = (
        # #             base_2["rank_diff_correction"] - (base_2["pace_diff"]/12) - base_2["place_season_condition_type_categori_processed_rank_diff"] - base_2["place_season_condition_type_categori_processed_rank_diff"] - (base_2["start_slope_abs_processed"]*0.1) - base_2["start_range_processed_1"]- base_2["umaban_odd_rank_diff_processed"] - (((base_2['race_grade']/70)-1)/2) + (base_2["last_curve_slope"]/15) - 0.12
        # #         )
        # #     if base_2['weather'] != 1 or base_2['ground_state'] == 1 or base_2['ground_state'] == 3:
        # #         base_2["rank_diff_correction_position"] = (
        # #             base_2["rank_diff_correction"] - (base_2["pace_diff"]/12) - base_2["place_season_condition_type_categori_processed_rank_diff"] - base_2["place_season_condition_type_categori_processed_rank_diff"] - (base_2["start_slope_abs_processed"]*0.1) - base_2["start_range_processed_1"]- base_2["umaban_odd_rank_diff_processed"] - (((base_2['race_grade']/70)-1)/2) + (base_2["last_curve_slope"]/15) + 0.12
        # #         )
        # #     base_2["rank_diff_correction_position"] = (
        # #         base_2["rank_diff_correction"] - (base_2["pace_diff"]/12) - base_2["place_season_condition_type_categori_processed_rank_diff"] - base_2["place_season_condition_type_categori_processed_rank_diff"] - (base_2["start_slope_abs_processed"]*0.1) - base_2["start_range_processed_1"]- base_2["umaban_odd_rank_diff_processed"] - (((base_2['race_grade']/70)-1)/2) + (base_2["last_curve_slope"]/15)
        # #     )

        # # if base_2['race_position'] == 3 or 4:
        # #     if base_2['weather'] == 1 and base_2['ground_state'] == 2:
        # #         base_2["rank_diff_correction_position"] = (
        # #             base_2["rank_diff_correction"] + (base_2["pace_diff"]/12) + base_2["place_season_condition_type_categori_processed_rank_diff"] + base_2["place_season_condition_type_categori_processed_rank_diff"] + (base_2["start_slope_abs_processed"]*0.1) + base_2["start_range_processed_1"]+ base_2["umaban_odd_rank_diff_processed"] + (((base_2['race_grade']/70)-1)/2) - (base_2["last_curve_slope"]/15) + 0.12
        # #         )
        # #     if base_2['weather'] != 1 or base_2['ground_state'] == 1 or base_2['ground_state'] == 3:
        # #         base_2["rank_diff_correction_position"] = (
        # #             base_2["rank_diff_correction"] + (base_2["pace_diff"]/12) + base_2["place_season_condition_type_categori_processed_rank_diff"] + base_2["place_season_condition_type_categori_processed_rank_diff"] + (base_2["start_slope_abs_processed"]*0.1) + base_2["start_range_processed_1"]+ base_2["umaban_odd_rank_diff_processed"] + (((base_2['race_grade']/70)-1)/2) - (base_2["last_curve_slope"]/15) - 0.12
        # #         )
        # #     base_2["rank_diff_correction_position"] = (
        # #         base_2["rank_diff_correction"] + (base_2["pace_diff"]/12) + base_2["place_season_condition_type_categori_processed_rank_diff"] + base_2["place_season_condition_type_categori_processed_rank_diff"] + (base_2["start_slope_abs_processed"]*0.1) + base_2["start_range_processed_1"]+ base_2["umaban_odd_rank_diff_processed"] + (((base_2['race_grade']/70)-1)/2) - (base_2["last_curve_slope"]/15)
        # #     )

        # base_2["goal_range_100_processed"] = base_2["goal_range_100"] - 3.6
        # # # プラスの値をすべて 0 に変換
        # # base_2["goal_range_100_processed"] = base_2["goal_range_100_processed"].clip(upper=0)
        # base_2.loc[base_2["goal_range_100_processed"] > 0, "goal_range_100_processed"] *= 0.7

        # # base_2["goal_range_processed_1"] = (((base_2["goal_range"])-360)/150)
        # # base_2["goal_range_processed_1"] = base_2["goal_range_processed_1"].apply(
        # #     lambda x: x*2 if x < 0 else x*0.7
        # # )
        # def calculate_rank_diff_correction_position(row):
        #     # 共通の部分の計算
        #     rank_diff_correction_position = (
        #         - (row["pace_diff"] / 10)
        #         - row["place_season_condition_type_categori_processed_rank_diff"]
        #         - (row["start_slope_abs_processed"] * 0.1)
        #         - row["start_range_processed"]*1.1
        #         - row["umaban_odd_rank_diff_processed"]
        #         - (((row['race_grade'] / 70) - 1) / 4)
        #         + (row["last_curve_slope"] / 12)
        #         - (row["curve_processed"] / 25)
        #         - (row["goal_range_100_processed"] / 4)   
        #         - (row["goal_slope"] / 15)

        #     )

        #     # race_positionが1または2の場合の処理
        #     if row['race_position'] in [1, 2]:
        #         if row['weather']  in [1, 2] and row['ground_state'] == 2:
        #             row["rank_diff_correction_position"] = row["rank_diff_correction"] + rank_diff_correction_position - 0.12
        #         elif row['weather'] not in [1, 2] or row['ground_state'] in [1, 2,3]:
        #             row["rank_diff_correction_position"] = row["rank_diff_correction"] + rank_diff_correction_position + 0.12
        #         elif row['weather'] in [1, 2]or row['ground_state'] in [1,3]:
        #             row["rank_diff_correction_position"] = row["rank_diff_correction"] - rank_diff_correction_position + 0.12
        #         else:
        #             row["rank_diff_correction_position"] = row["rank_diff_correction"] + rank_diff_correction_position

        #     # race_positionが3または4の場合の処理
        #     elif row['race_position'] in [3, 4]:
        #         if row['weather']  in [1, 2] and row['ground_state'] == 2:
        #             row["rank_diff_correction_position"] = row["rank_diff_correction"] - rank_diff_correction_position + 0.12
        #         elif row['weather'] not in [1, 2]or row['ground_state'] in [1, 2,3]:
        #             row["rank_diff_correction_position"] = row["rank_diff_correction"] - rank_diff_correction_position - 0.12
        #         elif row['weather'] in [1, 2]or row['ground_state'] in [1,3]:
        #             row["rank_diff_correction_position"] = row["rank_diff_correction"] - rank_diff_correction_position - 0.12
        #         else:
        #             row["rank_diff_correction_position"] = row["rank_diff_correction"] - rank_diff_correction_position

        #     return row

        # # DataFrameに適用
        # base_2 = base_2.apply(calculate_rank_diff_correction_position, axis=1)


        # """
        # ・持続
        # コーナー回数が少ない、コーナーゆるい、ラストの直線が短い、ラストの下り坂、ミドルorハイ、外枠、高低差+
        # ・瞬発
        # コーナー回数が多い、コーナーきつい、ラストの直線が長い、ラストの上り坂or平坦、スローペース、内枠、高低差なしがいい
        # 瞬発系はタフなレースきつい
        # """

        # # if base_2["rush_type"] < 0:
        # #     base_2["rank_diff_correction_rush"] =(
        # #         base_2["rank_diff_correction"] 
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["curve_amount"]-4)/8)) 
        # #         - (((base_2["rush_type"]+0.1)/30)*(base_2["curve_processed"] /-4))
        # #         - (((base_2["rush_type"]+0.1)/30)*(base_2["goal_range_processed_1"] /1.2))
        # #         - (((base_2["rush_type"]+0.1)/30)*(base_2["goal_slope"] /4))
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["pace_diff"]+0.6) / -3))
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["umaban_rank_diff_processed_2"]) * -10))
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["height_diff"]/-2)))
        # #     )
        # # if base_2["rush_type"] >= 0:
        # #     base_2["rank_diff_correction_rush"] =(
        # #         base_2["rank_diff_correction"] 
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["curve_amount"]-4)/8)) 
        # #         - (((base_2["rush_type"]+0.1)/30)*(base_2["curve_processed"] /-4))
        # #         - (((base_2["rush_type"]+0.1)/30)*(base_2["goal_range_processed_1"] /1.2))
        # #         - (((base_2["rush_type"]+0.1)/30)*(base_2["goal_slope"] /4))
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["pace_diff"]+0.6) / -3))
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["umaban_rank_diff_processed_2"]) * -10))
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["height_diff"]/-2)))
        # #     )

        # # 共通の計算
        # correction_factor = (base_2["rush_type"] + 0.1) / 30
        # base_2 = base_2.copy() 

        # # `rank_diff_correction_rush` の計算
        # base_2.loc[:, "rank_diff_correction_rush"] = (
        #     base_2["rank_diff_correction"]
        #     - (correction_factor * ((base_2["curve_amount"] - 4) / 10))
        #     - (correction_factor * (base_2["curve_processed"] / -6))
        #     - (correction_factor * (base_2["goal_range_100_processed"] / 1.2))
        #     - (correction_factor * (base_2["goal_slope"] / 4))
        #     - (correction_factor * ((base_2["pace_diff"] + 0.6) / -3))
        #     - (correction_factor * (base_2["umaban_rank_diff_processed_2"] * -10))
        #     - (correction_factor * (base_2["height_diff"] / -2)) 
        # )
        # # `rank_diff_correction_rush` の計算
        # base_2.loc[:, "rank_diff_correction_position_rush"] = (
        #     base_2["rank_diff_correction_position"]
        #     - (correction_factor * ((base_2["curve_amount"] - 4) / 10))
        #     - (correction_factor * (base_2["curve_processed"] / -6))
        #     - (correction_factor * (base_2["goal_range_100_processed"] / 1.2))
        #     - (correction_factor * (base_2["goal_slope"] / 4))
        #     - (correction_factor * ((base_2["pace_diff"] + 0.6) / -3))
        #     - (correction_factor * (base_2["umaban_rank_diff_processed_2"] * -10))
        #     - (correction_factor * (base_2["height_diff"] / -2)) 
        # )

        """
        rank_diff*race_grade
        どんな数字になる？
        """
        
        # base_2["rank_diff_correction_position_rush_xxx_race_grade_multi"] = (
        #     (((base_2["race_grade"]-10)) *(1/((base_2["rank_diff_correction_position_rush"]+5)/6)))
        # )
        base_2["rank_diff_position_xxx_race_grade_multi"] = (
            (((base_2["race_grade"]-10))  * (1/((base_2["rank_diff_pace_diff_slope_range_groundstate_position"]+7)/8)))
        )
        # base_2["rank_diff_correction_rush_xxx_race_grade_multi"] = (
        #     (((base_2["race_grade"]-10) ) *(1/((base_2["rank_diff_correction_rush"]+5)/6)))
        # )
        base_2["rank_diff_all_xxx_race_grade_multi"] = (
           (((base_2["race_grade"]-10))  * (1/((base_2["rank_diff_correction"]+7)/8)))
        )
        #


        # base_2["rank_diff_correction_position_rush_xxx_race_grade_sum"] = (
        #     ((base_2["race_grade"]) + ((base_2["rank_diff_correction_position_rush"])*-8))
        # )
        base_2["rank_diff_position_xxx_race_grade_sum"] = (
            ((base_2["race_grade"]+25)  + ((base_2["rank_diff_pace_diff_slope_range_groundstate_position"])*-10))
        )
        # base_2["rank_diff_correction_rush_xxx_race_grade_sum"] = (
        #     ((base_2["race_grade"]) + ((base_2["rank_diff_correction_rush"])*-8))
        # )
        base_2["rank_diff_all_xxx_race_grade_sum"] = (
            ((base_2["race_grade"]+25)  + ((base_2["rank_diff_correction"])*-10))
        )
        #90max

        base_2 = base_2.copy() 




        merged_df = self.population.copy()  
        # # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # #




        a_base_2 = base_2[['race_id',
        'date',
        'horse_id',

        "rush_type",


        "rank_diff_pace_diff_slope_range_groundstate_position",
        "rank_diff_correction",

        "time_class",
        "rank_diff_correction",
        
        "rank_diff_position_xxx_race_grade_multi",
        "rank_diff_all_xxx_race_grade_multi",
        "rank_diff_position_xxx_race_grade_sum",
        "rank_diff_all_xxx_race_grade_sum",
        ]]

        grouped_df = a_base_2.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        for n_race in tqdm(n_races, desc=f"rank_diff"):
            df_speed = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [
 
                        "rush_type",

                        "rank_diff_pace_diff_slope_range_groundstate_position",
                        "rank_diff_correction",

                        "time_class",
                        "rank_diff_correction",
                        
                        "rank_diff_position_xxx_race_grade_multi",
                        "rank_diff_all_xxx_race_grade_multi",
                        "rank_diff_position_xxx_race_grade_sum",
                        "rank_diff_all_xxx_race_grade_sum",
                    ]
                ]
                .agg(["min","max","mean"])
            )
            original_df = df_speed.copy()
            df_speed.columns = [
                "_".join(col) + f"_{n_race}races" for col in df_speed.columns
            ]
            # レースごとの相対値に変換
            tmp_df = df_speed.groupby(["race_id"])
            relative_df = ((df_speed - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            merged_df = merged_df.merge(
                relative_df, on=["race_id", "horse_id"], how="left"
            )
            # 相対値を付けない元の列をそのまま追加
            original_df.columns = [
                "_".join(col) + f"_{n_race}races" for col in original_df.columns
            ]  # 列名変更
            
            merged_df = merged_df.merge(
                original_df, on=["race_id", "horse_id"], how="left"
            )


        self.agg_rank_diff = merged_df
        print("agg_rank_diff ...comp")
        






    def cross_time(
        self, n_races: list[int] = [1, 3,5,8]
    ):  
        



        base_2  = (
            self.population.merge(
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



        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        base_2["distance_place_type_ground_state"] = (base_2["course_type"].astype(str)+ base_2["ground_state"].astype(str)).astype(int)   
        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        base_2["distance_place_type_ground_state_grade"] = (base_2["distance_place_type_ground_state"].astype(str)+ base_2["race_grade"].astype(str)).astype(int)   
        


        base_2_old = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","season","course_type"]], on="race_id"
            )
        )

        df_old = (
            base_2_old
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "umaban","nobori","time","rank"]], on=["race_id", "horse_id"])
        )

        df_old["nobori"] = df_old["nobori"].fillna(df_old["nobori"].mean())
        
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        # df_old["around"] = df_old["around"].fillna(3).astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  
        df_old["rank"] = df_old["rank"].astype(int)

        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_ground_state"] = (df_old["course_type"].astype(str)+ df_old["ground_state"].astype(str)).astype(int)   
        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_ground_state_grade"] = (df_old["distance_place_type_ground_state"].astype(str)+ df_old["race_grade"].astype(str)).astype(int)   
        
        target_mean_1= (
            df_old[
            (df_old['rank'].isin([1,2,3,4,5,6,7]))  # rankが1, 2, 3
            ]
            .groupby("distance_place_type_ground_state_grade")["nobori"]
            .mean()
        )
        # #noboriの平均
        # target_mean_1 = df_old.groupby("distance_place_type_ground_state_grade")["nobori"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_ground_state_grade_nobori_encoded"] = df_old["distance_place_type_ground_state_grade"].map(target_mean_1).fillna(0)
        

        df_old_rush_type = df_old[["distance_place_type_ground_state_grade","distance_place_type_ground_state_grade_nobori_encoded"]]
        
        columns_to_merge = [("distance_place_type_ground_state_grade","distance_place_type_ground_state_grade_nobori_encoded")]
        
        
        for original_col, encoded_col in columns_to_merge:
            df2_subset = df_old_rush_type[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
            df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
            base_2 = base_2.merge(df2_subset, on=original_col, how='left')  # dfにマージ
            
        base_2["nobori_diff"] = base_2["distance_place_type_ground_state_grade_nobori_encoded"] - base_2["nobori"]

        base_2 = base_2.copy()
        def calculate_rush_type(row):
            if row["nobori"] < 33 and row["race_type"] == 1:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 33 and row["race_type"] == 1:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 33.9 and row["race_type"] == 1:
                return -4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34.5 and row["race_type"] == 1:
                return -2.5
            if row["nobori"] < 34.5 and row["race_type"] == 1:
                return -2

            if row["nobori"] < 33.5 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 33.5 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34.4 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 35 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -2.5
            if row["nobori"] < 35 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -2

            if row["nobori"] < 34 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34.9 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -2.5
            if row["nobori"] < 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -2

            if row["nobori_diff"] >= 1:
                return -4
            if row["nobori_diff"] >= 0.6:
                return -2.5

            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 35.6 and row["race_type"] == 1 and row["rank"] <= 6:
                return 4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 34.5 and row["race_type"] == 1 and row["rank"] <= 6:
                return 2.5
            if row["nobori"] >= 34.5 and row["race_type"] == 1 and row["rank"] <= 6:
                return 2.5

            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 36.1 and row["race_type"] == 0 and row["course_len"] < 1600 and row["rank"] <= 6:
                return 4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 35 and row["race_type"] == 0 and row["course_len"] < 1600 and row["rank"] <= 6:
                return 2.5
            if row["nobori"] >= 35 and row["race_type"] == 0 and row["course_len"] < 1600 and row["rank"] <= 6:
                return 2.5

            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 36.6 and row["race_type"] == 0 and row["course_len"] >= 1600 and row["rank"] <= 6:
                return 4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600 and row["rank"] <= 6:
                return 2.5
            if row["nobori"] >= 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600 and row["rank"] <= 6:
                return 2.5

            return 0

        # DataFrame に適用
        base_2["rush_type"] = base_2.apply(calculate_rush_type, axis=1)




        # #最大前後0.2、ハイペースは+
        # base_2["time_pace_diff"] = base_2["time"] + (base_2["pace_diff"] /2)


        # # # 1600で正規化
        # # base_2["course_len_processed_rd"] = (base_2["course_len"] / 1600) - 1

        # # # 1600m未満ならそのまま、1600m以上なら緩やかに上昇 
        # # base_2["course_len_processed_rd"] = base_2["course_len_processed_rd"].apply(
        # #     lambda x: x/2 if x <= 0 else x/3
        # # )
        # #-0.25
        # #長距離のほうが着差の価値がなくなる


        # # 1600で正規化
        # base_2["course_len_processed_rd"] = (base_2["course_len"] / 1600) - 1

        # # 1600m未満ならそのまま、1600m以上なら緩やかに上昇 
        # base_2["course_len_processed_rd"] = base_2["course_len_processed_rd"].apply(
        #     lambda x: x/2 if x <= 0 else x/3
        # )

        # base_2["time_pace_course_len"] = base_2["time_pace_diff"]



        # # 条件ごとに適用,馬場状態が悪い状態ほど評価（-）
        # base_2["time_pace_course_len_ground_state"] = np.where(
        #     ((base_2["ground_state"] == 1) | (base_2["ground_state"] == 3)) & (base_2["race_type"] == 1),
        #     base_2["time_pace_course_len"] * 0.92,

        #     np.where(
        #         (base_2["ground_state"] == 2) & (base_2["race_type"] == 1),
        #         base_2["time_pace_course_len"] * 0.95,

        #         np.where(
        #             ((base_2["ground_state"] == 1) | (base_2["ground_state"] == 3)) & (base_2["race_type"] == 0),
        #             base_2["time_pace_course_len"]  * 1.01,

        #             np.where(
        #                 (base_2["ground_state"] == 2) & (base_2["race_type"] == 0),
        #                 base_2["time_pace_course_len"]  * 1.015,
                        
        #                 # どの条件にも当てはまらない場合は元の値を保持
        #                 base_2["time_pace_course_len"]
        #             )
        #         )
        #     )
        # )


        # # -4.5 を行う
        # base_2["curve_processed"] = base_2["curve"] - 4.5
        # # +の場合は数値を8倍する
        # base_2["curve_processed"] = base_2["curve_processed"].apply(
        #     lambda x: x * 8 if x > 0 else x
        # )
        # #最初の直線の長さ、長いほどきつい、50前後くらい
        # base_2["start_range_processed"] = (((base_2["start_range"])-360)/150)
        # base_2["start_range_processed"] = base_2["start_range_processed"].apply(
        #     lambda x: x if x < 0 else x*0.5
        # )
        # base_2["start_slope_abs"] = base_2["start_slope"].abs()
        # base_2["start_slope_abs_processed"] = base_2["start_slope_abs"] /4


        # #0,01-0.01,内がプラス(内枠が有利を受けるとしたら、timeは+にして、有利ポジはマイナス補正)
        # base_2["umaban_time_processed"] = base_2["umaban"].apply(
        #     lambda x: ((x*-1.5)+1.5) if x < 4 else ((x-8)/1.5)-1
        # ).astype(float)
        # #0,-0.1,-0.3,-0.36,-0.3,-0.23,（-1/10のとき）
        # base_2["umaban_time_processed"] = base_2["umaban_time_processed"] * (1/10)
        # #0 , -0.05
        # #1（奇数）または 0（偶数）,偶数が有利
        # base_2.loc[:, "umaban_odd_time_processed"] = (
        #     (base_2["umaban_odd"]-1)/10
        # ).astype(float)

        # #rdが-0.25,,0.25が0.5に
        # base_2["umaban_time_processed_2"] = base_2["umaban_time_processed"] / ((base_2["course_len_processed_rd"]*2) + 1)
        # base_2["umaban_odd_time_processed_2"] = base_2["umaban_odd_time_processed"] / ((base_2["course_len_processed_rd"]*2)+1)

        # #不利が-,ダートは外枠有利,0.06
        # base_2["time_pace_course_len_ground_state_type"] = np.where(
        #     base_2["race_type"] == 0,
        #     base_2["time_pace_course_len_ground_state"] / ((base_2["umaban_time_processed_2"]+120)/120),
        #     base_2["time_pace_course_len_ground_state"] * ((base_2["umaban_time_processed_2"]+120)/120)
        # )

        # base_2["time_pace_course_len_ground_state_type_odd"]= np.where(
        #     base_2["race_type"] == 0,
        #     base_2["time_pace_course_len_ground_state_type"] * ((base_2["umaban_odd_time_processed_2"]+70)/70),
        #     base_2["time_pace_course_len_ground_state_type"] * ((base_2["umaban_odd_time_processed_2"]+70)/70)
        # )

        # #last急カーブ,フリ評価
        # base_2["time_pace_course_len_ground_state_type_odd_curve"] = (
        #     base_2["time_pace_course_len_ground_state_type_odd"] * ((((base_2["umaban_time_processed_2"]*((base_2["curve_processed"]/4))))+120)/120)
        # )

        # #3カーブ下り坂,フリ評価
        # base_2["time_pace_course_len_ground_state_type_odd_curve_slope"] = (
        #     base_2["time_pace_course_len_ground_state_type_odd_curve"] * ((((base_2["umaban_time_processed_2"]*(base_2["last_curve_slope"]/3)))+120)/120)
        # )

        # #スタートからコーナー
        # base_2["time_pace_course_len_ground_state_type_odd_curve_slope_start"] = (
        #     base_2["time_pace_course_len_ground_state_type_odd_curve_slope"] / (((base_2["umaban_time_processed_2"]*(((base_2["start_range_processed"]))*-1/1.2)- ((base_2["umaban_odd_time_processed_2"]*(base_2["start_range_processed"]))*-1/1.2))+120)/120)
        # )


        # #+-0.06,スタートからコーナー、坂,上り下り両方
        # base_2["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] = (
        #     base_2["time_pace_course_len_ground_state_type_odd_curve_slope_start"] / (((((base_2["umaban_time_processed_2"]*(base_2["start_slope_abs_processed"]))*-1)- ((base_2["umaban_odd_time_processed_2"]*( base_2["start_slope_abs_processed"]))*-1))+120)/120)
        # )

        # #最大0.3*nコーナーがきついは内枠
        # def calculate_time_pace_diff_2(row):
        #     if row["curve_amount"] == 0:
        #         return row["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope"]
        #     elif row["curve_amount"] <= 2:
        #         return row["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] / (((row["umaban_time_processed_2"]*(((row["curve_R34"]-100)/120)*-1))+200)/200)
        #     elif row["curve_amount"] <= 3:
        #         return row["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] / (((row["umaban_time_processed_2"]*(((row["curve_R12"]-100)/120 / 2 + (row["curve_R34"]-100)/120)*-1))+200)/200)
        #     elif row["curve_amount"] <= 4:
        #         return row["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] / (((row["umaban_time_processed_2"]*(((row["curve_R12"]-100)/120 + ((row["curve_R34"]-100)/120))*-1))+200)/200)
        #     elif row["curve_amount"] <= 5:
        #         return row["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] / (((row["umaban_time_processed_2"]*(((row["curve_R12"]-100)/120 + ((row["curve_R34"]-100)/120) * 3 / 2)*-1))+200)/200)
        #     elif row["curve_amount"] <= 6:
        #         return row["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] /((((row["umaban_time_processed_2"]*((row["curve_R12"]-100)/120 + ((row["curve_R34"]-100)/120) * 2)*-1))+200)/200)
        #     elif row["curve_amount"] <= 7:
        #         return row["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] / (((row["umaban_time_processed_2"]*(((row["curve_R12"]-100)/120 * 3 / 2 + ((row["curve_R34"]-100)/120) * 2)*-1))+200)/200)
        #     else:  # curve_amount <= 8
        #         return row["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] /((((row["umaban_time_processed_2"]*((row["curve_R12"]-100)/120 * 2 +((row["curve_R34"]-100)/120) * 2)*-1))+200)/200)



        # base_2["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope_time"] = base_2.apply(calculate_time_pace_diff_2, axis=1)


        # #最大1*向正面
        # base_2["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope_time_flont"] = (
        #     base_2["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope_time"] / ((((base_2["umaban_time_processed_2"]*(base_2["flont_slope"]/8)))+200)/200)
        # )

        # #芝スタートかつ良馬場、芝スタートかつ良以外、どっちでもない場合,外評価

        # condition_time = (base_2["start_point"] == 2) & (base_2["ground_state"] == 0)

        # base_2["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope_time_flont_point"] = np.where(
        #     condition_time,
        #     base_2["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope_time_flont"] / (((base_2["umaban_time_processed_2"] * -1.5)+200)/200),
        #     base_2["time_pace_course_len_ground_state_type_odd_curve_slope_start_slope_time_flont"] / (((base_2["umaban_time_processed_2"])+300)/300)
        # )





        base_2 = base_2.copy()
        base_2.loc[:, "place_season_condition_type_categori_processed"] = (
            base_2["place_season_condition_type_categori"]
            .replace({5: -0.3, 4: -0.17, 3: 0, 2: 0.17,1: 0.3, -1: -0.18, -2: 0, -3: 0.18,-4:0.3,-10000:0})
        ).astype(float)


        base_2["time_pace_diff"] = base_2["time"] + (base_2["pace_diff"]/2) - (base_2["place_season_condition_type_categori"]*7)

        base_2["time_pace_diff_slope"] = np.where(
            (base_2["season"] == 1) | (base_2["season"] == 4),
            base_2["time_pace_diff"] / ((base_2["goal_slope"] + 200)/200),
            base_2["time_pace_diff"] / ((base_2["goal_slope"] + 400)/400),
        )


        base_2["goal_range_processed_1"] = (((base_2["goal_range"])-360))
        base_2["goal_range_processed_1"] = base_2["goal_range_processed_1"].apply(
            lambda x: x*2 if x < 0 else x*0.5
        )

        #ゴールが短いと上りの分が入らないので上りが少し遅くなる
        base_2["time_pace_diff_slope_range"] = base_2["time_pace_diff_slope"] * ((base_2["goal_range_processed_1"] + 50000) / 50000)/((base_2["height_diff"]+500)/500)


        """
        ハイスロー、脚質修正
        コーナー順位が前（先行）で、ハイペースの場合、timeをさらに-0.5する（不利条件）
        前でローペースの場合、timeを+0.2する(ペースによる)有利
        後ろで、ローペースの場合、timeを-0.1する（作っておけばrankdiffで使える）不利
        後ろでハイの場合、timeを+0.3する(ハイスローの分を相殺する)有利
        #+だとハイペース、ーだとスローペース
        """
        # 条件ごとに処理を適用
        base_2["time_pace_diff_slope_range_pace"] = np.where(
            ((base_2['race_position'] == 1) | (base_2['race_position'] == 2)) & (base_2["pace_diff"] >= 0),
            base_2["time_pace_diff_slope_range"] - (base_2["pace_diff"] / 4),
            
            np.where(
                    ((base_2['race_position'] == 2)) & (base_2["pace_diff"] >= 0),
                    base_2["time_pace_diff_slope_range"] - (base_2["pace_diff"] / 40),
                
                np.where(
                    ((base_2['race_position'] == 1) | (base_2['race_position'] == 2)) & (base_2["pace_diff"] < 0),
                    base_2["time_pace_diff_slope_range"] - (base_2["pace_diff"] / 10),
                    
                    np.where(
                        ((base_2['race_position'] == 3) | (base_2['race_position'] == 4)) & (base_2["pace_diff"] < 0),
                        base_2["time_pace_diff_slope_range"] - ((base_2["pace_diff"] / 10) * -1),
                        

                        np.where(
                            ((base_2['race_position'] == 3) | (base_2['race_position'] == 4)) & (base_2["pace_diff"] >= 0),
                            base_2["time_pace_diff_slope_range"] - ((base_2["pace_diff"] / 8) * -1),
                            base_2["time_pace_diff_slope_range"]  # どの条件にも当てはまらない場合は元の値を保持
                            )
                    )
                )
            )
        )

        # 条件ごとに適用
        base_2["time_pace_diff_slope_range_groundstate"] = np.where(
            ((base_2["ground_state"] == 1) | (base_2["ground_state"] == 3)) & (base_2["race_type"] == 1),
            base_2["time_pace_diff_slope_range_pace"] * (29/30),

            np.where(
                (base_2["ground_state"] == 2) & (base_2["race_type"] == 1),
                base_2["time_pace_diff_slope_range_pace"] * (119/120),

                np.where(
                    ((base_2["ground_state"] == 1) | (base_2["ground_state"] == 3)) & (base_2["race_type"] == 0),
                    base_2["time_pace_diff_slope_range_pace"] * (81/80),

                    np.where(
                        (base_2["ground_state"] == 2) & (base_2["race_type"] == 0),
                        base_2["time_pace_diff_slope_range_pace"] * (121/120),
                        
                        # どの条件にも当てはまらない場合は元の値を保持
                        base_2["time_pace_diff_slope_range_pace"]
                    )
                )
            )
        )

        base_2["start_range_processed_1"] = (((base_2["start_range"])-360))
        base_2["start_range_processed_1"] = base_2["start_range_processed_1"].apply(
            lambda x: x if x < 0 else x*1
        )
        # -4.5 を行う
        base_2["curve_processed"] = base_2["curve"] - 4.5
        # +の場合は数値を8倍する
        base_2["curve_processed"] = base_2["curve_processed"].apply(
            lambda x: x * 8 if x > 0 else x
        )


        # ペースに関係ある要素は弱体化
        base_2["time_pace_diff_slope_range_groundstate_position"] = np.where(
            ((base_2['race_position'] == 1) | (base_2['race_position'] == 2)),
            base_2["time_pace_diff_slope_range_groundstate"] 
            / ((100000 + base_2["start_range_processed_1"]) / 100000) 
            * ((1600 + base_2["start_slope"]) / 1600) 
            * ((2000 + base_2["curve_processed"]) / 2000) 
            / ((base_2["goal_range_processed_1"] + 100000) / 100000) 
            / ((base_2["goal_slope"] + 1600) / 1600) 
            / ((base_2["place_season_condition_type_categori_processed"] + 500) / 500) 
            / ((base_2["race_type"] + 799) / 800),  # ここでカンマ

            np.where(
                ((base_2['race_position'] == 3) | (base_2['race_position'] == 4)),
                base_2["time_pace_diff_slope_range_groundstate"] 
                * ((100000 + base_2["start_range_processed_1"]) / 100000) 
                / ((1600 + base_2["start_slope"]) / 1600) 
                / ((2000 + base_2["curve_processed"]) / 2000) 
                * ((base_2["goal_range_processed_1"] + 100000) / 100000) 
                * ((base_2["goal_slope"] + 1600) / 1600) 
                * ((base_2["place_season_condition_type_categori_processed"] + 500) / 500) 
                * ((base_2["race_type"] + 799) / 800), 

                base_2["time_pace_diff_slope_range_groundstate"] # それ以外の場合は NaN
            )
        )





        # # 月を抽出して開催シーズンを判定
        # def determine_season_turf(month):
        #     if 6 <= month <= 8:
        #         return "4" #"夏開催"
        #     elif month == 12 or 1 <= month <= 2:
        #         return "2" #"冬開催"
        #     elif 3 <= month <= 5:
        #         return "3" #"春開催"
        #     elif 9 <= month <= 11:
        #         return "1" #"秋開催"    
        
        # base_2["season_turf"] = base_2["date"].dt.month.map(determine_season_turf)
        # base_2["day"] = base_2["day"].astype(str)

        # base_2["day_season_turf"] =  base_2["day"] + base_2["season_turf"]
        # base_2["day_season_turf"] =  base_2["day_season_turf"].astype(int)
        # base_2["day"] = base_2["day"].astype(int)

        # #umaban

        # base_2["season_turf_condition"] = np.where(
        #     base_2["season"] == 1, base_2["day"],
        #     np.where(
        #         base_2["season"] == 2, (base_2["day"] + 1.5) * 1.5,
        #         np.where(
        #             base_2["season"] == 3, base_2["day"] + 3,
        #             np.where(
        #                 base_2["season"] == 4, base_2["day"] + 4,
        #                 base_2["day"]  # それ以外のとき NaN
        #             )
        #         )
        #     )
        # )



        #0,01-0.01,内がマイナス
        base_2["umaban_processed"] = base_2["umaban"].apply(
            lambda x: ((x*-1)) if x < 4 else ((x-8)/3)-1
        ).astype(float)
        #0-0.005

        base_2["umaban_judge"] = (base_2["umaban"].astype(float)/base_2["n_horses"].astype(float))-0.55

        #1（奇数）または 0（偶数）
        base_2.loc[:, "umaban_odd_processed"] = (
            (base_2["umaban_odd"]-1)
        ).astype(float)

        # 1600で正規化,-0.5 - 1
        base_2["course_len_processed"] = (base_2["course_len"] / 1700)-1

        # ,-1.5 - 4
        base_2["course_len_processed"] = base_2["course_len_processed"].apply(
            lambda x: x*3 if x <= 0 else x*4
        )
        base_2["course_len_processed_2"] = ((base_2["course_len_processed"] + 3)/3)

        base_2["umaban_processed_2"] = base_2["umaban_processed"] / base_2["course_len_processed_2"]
        base_2["umaban_odd_processed_2"] = base_2["umaban_odd_processed"] / base_2["course_len_processed_2"]



        base_2["first_corner"] = np.where(
            (base_2["curve_amount"] == 2) | (base_2["curve_amount"] == 6),
            base_2["curve_R34"],
            np.where(
                (base_2["curve_amount"] == 4) | (base_2["curve_amount"] == 8),
                base_2["curve_R12"],
                0  # それ以外のとき 0
            )
        )



        base_2["umaban_processed_abs2"] = base_2["umaban_processed_2"].abs()

        # 内が小さい,最大50くらいになってしまう
        base_2["time_pace_diff_slope_range_groundstate_position_umaban"] = np.where(
            (base_2["umaban_judge"] < 0),
            base_2["time_pace_diff_slope_range_groundstate_position"] /
            ((
                ((base_2["umaban_processed_abs2"]) # 少ないほうがtimeが増える-4.5 から3
                * (
                    base_2["umaban_odd_processed_2"]# 奇数不利なので分母を増やして総合を減らす 1
                      +  (base_2["start_point"] - 1)# 外枠が有利なので分母を増やして総合を減らす 1
                      +  base_2["curve_processed"]# ラストカーブきついほど数値が減る4
                      +  base_2["last_curve_slope"]# ラストカーブくだりほど数値が減る2
                      +  (base_2["season_turf_condition"] - 7)# 馬場状態が良いほど数値が減る 7-7
                      +  (base_2["race_type"] - 0.5)*4# 芝ほど数値が減る 2
                      -  ((base_2["first_corner"] - 100)/50)# 最初のコーナーがでかいほど数値が減る1
                ) 
            ) + 2500) / 2500)
            ,
            

            np.where(
                (base_2["umaban_judge"] >= 0),
                base_2["time_pace_diff_slope_range_groundstate_position"] /
                ((
                    ((base_2["umaban_processed_abs2"]) # 少ないほうがtimeが増える-4.5 から3
                    * (
                        base_2["umaban_odd_processed_2"]# 奇数不利なので分母を増やして総合を減らす 1
                        -  (base_2["start_point"] - 1)# 外枠が有利なので分母を増やして総合を減らす 1
                        -  base_2["curve_processed"]# ラストカーブきついほど数値が減る4
                        -  base_2["last_curve_slope"]# ラストカーブくだりほど数値が減る2
                        -  (base_2["season_turf_condition"] - 7)# 馬場状態が良いほど数値が減る 7-7
                        -  (base_2["race_type"] - 0.5)*4# 芝ほど数値が減る 2
                        +  ((base_2["first_corner"] - 100)/50)# 最初のコーナーがでかいほど数値が減る1
                    ) 
                ) + 2500) / 2500),

                base_2["time_pace_diff_slope_range_groundstate_position"]
            )
        )






        
        # # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        # base_2["distance_place_type_ground_state"] = (base_2["course_type"].astype(str)+ base_2["ground_state"].astype(str)).astype(int)   
        
        # # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        # base_2["distance_place_type_ground_state_grade"] = (base_2["distance_place_type_ground_state"].astype(str)+ base_2["race_grade"].astype(str)).astype(int)   
        


        base_2_old = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","season","course_type"]], on="race_id"
            )
        )

        df_old = (
            base_2_old
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "umaban","nobori","time","rank"]], on=["race_id", "horse_id"])
        )

        df_old["nobori"] = df_old["nobori"].fillna(df_old["nobori"].mean())
        
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        # df_old["around"] = df_old["around"].fillna(3).astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  
        df_old["rank"] = df_old["rank"].astype(int)

        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_ground_state"] = (df_old["course_type"].astype(str)+ df_old["ground_state"].astype(str)).astype(int)   
        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_ground_state_grade"] = (df_old["distance_place_type_ground_state"].astype(str)+ df_old["race_grade"].astype(str)).astype(int)   
        

        # シーズンを追加
        base_2["distance_place_type_ground_state_grade_season"] = base_2["distance_place_type_ground_state_grade"].astype(int)   

        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_ground_state_grade_season"] = df_old["distance_place_type_ground_state_grade"].astype(int)   
        
        target_mean_1= (
            df_old[
            (df_old['rank'].isin([1]))  # rankが1, 2, 3
            ]
            .groupby("distance_place_type_ground_state_grade_season")["time"]
            .mean()
        )
        df_old["distance_place_type_ground_state_grade_season_time_encoded"] = df_old["distance_place_type_ground_state_grade_season"].map(target_mean_1).fillna(0)
        

        df_old_rank_diff = df_old[["distance_place_type_ground_state_grade_season","distance_place_type_ground_state_grade_season_time_encoded"]]
        
        columns_to_merge = [("distance_place_type_ground_state_grade_season","distance_place_type_ground_state_grade_season_time_encoded")]
        
        
        for original_col, encoded_col in columns_to_merge:
            df2_subset = df_old_rank_diff[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
            df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
            base_2 = base_2.merge(df2_subset, on=original_col, how='left')  # dfにマージ
            

        base_2['no1_time'] = base_2.apply(
            lambda row: row['time'] if row['rank'] == 1 else row['time'] - (row['rank_diff']-2),
            axis=1
        )

        base_2["time_class"] = base_2["distance_place_type_ground_state_grade_season_time_encoded"] - base_2['no1_time']

        base_2["time_class_abs"] = base_2["time_class"].abs()




        def calculate_time_course_len_pace_diff(row):
            # 初期値として元のtimeを設定
            result = row["time_pace_diff_slope_range_groundstate_position_umaban"]
            
            if row["course_len"] < 1600:
                if row["race_grade"] < 80 and row["time_class_abs"] > 0.7 and row["time_class"] > 0:
                    result /= 1.025
                elif row["race_grade"] < 80 and row["time_class_abs"] > 0.7 and row["time_class"] < 0:
                    result *= 1.025
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1 and row["time_class"] > 0:
                    result /= 1.025
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1 and row["time_class"] < 0:
                    result *= 1.0325
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 1.2 and row["time_class"] > 0:
                    result /= 1.025
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 1.2 and row["time_class"] < 0:
                    result *= 1.025

            if 1600 <= row["course_len"] and row["race_type"] == 1:
                if row["race_grade"] < 80 and row["time_class_abs"] > 1.2 and row["time_class"] > 0:
                    result /= 1.025
                elif row["race_grade"] < 80 and row["time_class_abs"] > 1.2 and row["time_class"] < 0:
                    result *= 1.025
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1.5 and row["time_class"] > 0:
                    result /= 1.025
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1.5 and row["time_class"] < 0:
                    result *= 1.025
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 1.7 and row["time_class"] > 0:
                    result /= 1.025
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 1.7 and row["time_class"] < 0:
                    result *= 1.025

            if 1600 <= row["course_len"] and row["race_type"] == 0:
                if row["race_grade"] < 80 and row["time_class_abs"] > 1 and row["time_class"] > 0:
                    result /= 1.025
                elif row["race_grade"] < 80 and row["time_class_abs"] > 1 and row["time_class"] < 0:
                    result *= 1.025
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1.3 and row["time_class"] > 0:
                    result /= 1.025
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1.3 and row["time_class"] < 0:
                    result *= 1.025
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 1.5 and row["time_class"] > 0:
                    result /= 1.025
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 1.5 and row["time_class"] < 0:
                    result *= 1.025

            if 2400 <= row["course_len"] and row["race_type"] == 1:
                if row["race_grade"] < 80 and row["time_class_abs"] > 1.8 and row["time_class"] > 0:
                    result /= 1.025
                elif row["race_grade"] < 80 and row["time_class_abs"] > 1.8 and row["time_class"] < 0:
                    result *= 1.025
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 2.1 and row["time_class"] > 0:
                    result /= 1.025
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 2.1 and row["time_class"] < 0:
                    result *= 1.025
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 2.3 and row["time_class"] > 0:
                    result /= 1.025
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 2.3 and row["time_class"] < 0:
                    result *= 1.025

            if 2400 <= row["course_len"] and row["race_type"] == 0:
                if row["race_grade"] < 80 and row["time_class_abs"] > 1.5 and row["time_class"] > 0:
                    result /= 1.025
                elif row["race_grade"] < 80 and row["time_class_abs"] > 1.5 and row["time_class"] < 0:
                    result *= 1.025
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1.8 and row["time_class"] > 0:
                    result /= 1.025
                elif 80 <= row["race_grade"] <= 87 and row["time_class_abs"] > 1.8 and row["time_class"] < 0:
                    result *= 1.025
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 2 and row["time_class"] > 0:
                    result /= 1.025
                elif 88 <= row["race_grade"] and row["time_class_abs"] > 2 and row["time_class"] < 0:
                    result *= 1.025


            return result

        base_2["time_correction"] = base_2.apply(calculate_time_course_len_pace_diff, axis=1)


        """
        その他補正のみ+脚質補正（向いていない展開なら展開なら評価）
        脚質パック	
        季節ごとの馬場状態馬場でも-+補正をいれる	
        軽い芝、先行有利
        重い芝、差し有利

        スタートが上りorくだり	逃げ先行有利
        偶数ほどよいは先行系	逃げ先行有利
        コーナーまで短いが	逃げ先行有利
            レースグレードがあがると	差し有利

        4コーナーくだり坂	差し追い込み有利
        ダートの方が	かなり先行優位
        雨	先行有利
        雨のない稍重	差し有利

        ハイペースが+ローが-(pace_diff)
        """
        # base_2 = base_2.copy()
        # base_2.loc[:, "place_season_condition_type_categori_processed_time"] = (
        #     base_2["place_season_condition_type_categori"]
        #     .replace({5: -0.09, 4: -0.05, 3: 0, 2: 0.05,1: 0.09, -1: -0.08, -2: 0, -3: 0.08,-4:0.11,-10000:0})
        # ).astype(float)

        # # #向いていない場合は-
        # # if base_2['race_position'] == 1 or 2:
        # #     if base_2['weather'] == 1 and base_2['ground_state'] == 2:
        # #         base_2["time_correction_position"] = (
        # #             base_2["time_correction"] - (base_2["pace_diff"]/12) - base_2["place_season_condition_type_categori_processed_time"] - base_2["place_season_condition_type_categori_processed_time"] - (base_2["start_slope_abs_processed"]*0.1) - base_2["start_range_processed_1"]- base_2["umaban_odd_time_processed"] - (((base_2['race_grade']/70)-1)/2) + (base_2["last_curve_slope"]/15) - 0.12
        # #         )
        # #     if base_2['weather'] != 1 or base_2['ground_state'] == 1 or base_2['ground_state'] == 3:
        # #         base_2["time_correction_position"] = (
        # #             base_2["time_correction"] - (base_2["pace_diff"]/12) - base_2["place_season_condition_type_categori_processed_time"] - base_2["place_season_condition_type_categori_processed_time"] - (base_2["start_slope_abs_processed"]*0.1) - base_2["start_range_processed_1"]- base_2["umaban_odd_time_processed"] - (((base_2['race_grade']/70)-1)/2) + (base_2["last_curve_slope"]/15) + 0.12
        # #         )
        # #     base_2["time_correction_position"] = (
        # #         base_2["time_correction"] - (base_2["pace_diff"]/12) - base_2["place_season_condition_type_categori_processed_time"] - base_2["place_season_condition_type_categori_processed_time"] - (base_2["start_slope_abs_processed"]*0.1) - base_2["start_range_processed_1"]- base_2["umaban_odd_time_processed"] - (((base_2['race_grade']/70)-1)/2) + (base_2["last_curve_slope"]/15)
        # #     )

        # # if base_2['race_position'] == 3 or 4:
        # #     if base_2['weather'] == 1 and base_2['ground_state'] == 2:
        # #         base_2["time_correction_position"] = (
        # #             base_2["time_correction"] + (base_2["pace_diff"]/12) + base_2["place_season_condition_type_categori_processed_time"] + base_2["place_season_condition_type_categori_processed_time"] + (base_2["start_slope_abs_processed"]*0.1) + base_2["start_range_processed_1"]+ base_2["umaban_odd_time_processed"] + (((base_2['race_grade']/70)-1)/2) - (base_2["last_curve_slope"]/15) + 0.12
        # #         )
        # #     if base_2['weather'] != 1 or base_2['ground_state'] == 1 or base_2['ground_state'] == 3:
        # #         base_2["time_correction_position"] = (
        # #             base_2["time_correction"] + (base_2["pace_diff"]/12) + base_2["place_season_condition_type_categori_processed_time"] + base_2["place_season_condition_type_categori_processed_time"] + (base_2["start_slope_abs_processed"]*0.1) + base_2["start_range_processed_1"]+ base_2["umaban_odd_time_processed"] + (((base_2['race_grade']/70)-1)/2) - (base_2["last_curve_slope"]/15) - 0.12
        # #         )
        # #     base_2["time_correction_position"] = (
        # #         base_2["time_correction"] + (base_2["pace_diff"]/12) + base_2["place_season_condition_type_categori_processed_time"] + base_2["place_season_condition_type_categori_processed_time"] + (base_2["start_slope_abs_processed"]*0.1) + base_2["start_range_processed_1"]+ base_2["umaban_odd_time_processed"] + (((base_2['race_grade']/70)-1)/2) - (base_2["last_curve_slope"]/15)
        # #     )

        # base_2["goal_range_100_processed"] = base_2["goal_range_100"] - 3.6
        # # # プラスの値をすべて 0 に変換
        # # base_2["goal_range_100_processed"] = base_2["goal_range_100_processed"].clip(upper=0)
        # base_2.loc[base_2["goal_range_100_processed"] > 0, "goal_range_100_processed"] *= 0.7


        # def calculate_time_correction_position(row):
        #     # 共通の部分の計算
        #     time_correction_position = (
        #         - (row["pace_diff"] / 2)
        #         - row["place_season_condition_type_categori_processed_time"]
        #         - (row["start_slope_abs_processed"] * 0.8)
        #         - row["start_range_processed"]*3
        #         - row["umaban_odd_time_processed"]*3
        #         - (((row['race_grade'] / 70) - 1))
        #         + (row["last_curve_slope"] / 3)
        #         - (row["curve_processed"] / 8)
        #         - (row["goal_range_100_processed"] / 10)   
        #         - (row["goal_slope"] / 10)

        #     )

        #     # race_positionが1または2の場合の処理
        #     if row['race_position'] in [1, 2]:
        #         if row['weather']  in [1, 2] and row['ground_state'] == 2:
        #             row["time_correction_position"] = row["time_correction"] + time_correction_position - 1.2
        #         elif row['weather'] not in [1, 2] or row['ground_state'] in [1, 2,3]:
        #             row["time_correction_position"] = row["time_correction"] + time_correction_position +1.2
        #         elif row['weather'] in [1, 2]or row['ground_state'] in [1,3]:
        #             row["time_correction_position"] = row["time_correction"] - time_correction_position + 1.2
        #         else:
        #             row["time_correction_position"] = row["time_correction"] + time_correction_position

        #     # race_positionが3または4の場合の処理
        #     elif row['race_position'] in [3, 4]:
        #         if row['weather']  in [1, 2] and row['ground_state'] == 2:
        #             row["time_correction_position"] = row["time_correction"] - time_correction_position + 1.2
        #         elif row['weather'] not in [1, 2]or row['ground_state'] in [1, 2,3]:
        #             row["time_correction_position"] = row["time_correction"] - time_correction_position - 1.2
        #         elif row['weather'] in [1, 2]or row['ground_state'] in [1,3]:
        #             row["time_correction_position"] = row["time_correction"] - time_correction_position - 1.2
        #         else:
        #             row["time_correction_position"] = row["time_correction"] - time_correction_position

        #     return row

        # # DataFrameに適用
        # base_2 = base_2.apply(calculate_time_correction_position, axis=1)


        # """
        # ・持続
        # コーナー回数が少ない、コーナーゆるい、ラストの直線が短い、ラストの下り坂、ミドルorハイ、外枠、高低差+
        # ・瞬発
        # コーナー回数が多い、コーナーきつい、ラストの直線が長い、ラストの上り坂or平坦、スローペース、内枠、高低差なしがいい
        # 瞬発系はタフなレースきつい
        # """

        # # if base_2["rush_type"] < 0:
        # #     base_2["time_correction_rush"] =(
        # #         base_2["time_correction"] 
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["curve_amount"]-4)/8)) 
        # #         - (((base_2["rush_type"]+0.1)/30)*(base_2["curve_processed"] /-4))
        # #         - (((base_2["rush_type"]+0.1)/30)*(base_2["goal_range_processed_1"] /1.2))
        # #         - (((base_2["rush_type"]+0.1)/30)*(base_2["goal_slope"] /4))
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["pace_diff"]+0.6) / -3))
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["umaban_time_processed_2"]) * -10))
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["height_diff"]/-2)))
        # #     )
        # # if base_2["rush_type"] >= 0:
        # #     base_2["time_correction_rush"] =(
        # #         base_2["time_correction"] 
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["curve_amount"]-4)/8)) 
        # #         - (((base_2["rush_type"]+0.1)/30)*(base_2["curve_processed"] /-4))
        # #         - (((base_2["rush_type"]+0.1)/30)*(base_2["goal_range_processed_1"] /1.2))
        # #         - (((base_2["rush_type"]+0.1)/30)*(base_2["goal_slope"] /4))
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["pace_diff"]+0.6) / -3))
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["umaban_time_processed_2"]) * -10))
        # #         - (((base_2["rush_type"]+0.1)/30)*((base_2["height_diff"]/-2)))
        # #     )

        # # 共通の計算
        # correction_factor = (base_2["rush_type"] + 0.1) / 20
        # base_2 = base_2.copy() 

        # # `time_correction_rush` の計算
        # base_2.loc[:, "time_correction_rush"] = (
        #     base_2["time_correction"]
        #     - (correction_factor * ((base_2["curve_amount"] - 4) / 3))
        #     - (correction_factor * (base_2["curve_processed"] / -2))
        #     - (correction_factor * (base_2["goal_range_100_processed"]))
        #     - (correction_factor * (base_2["goal_slope"] ))
        #     - (correction_factor * ((base_2["pace_diff"] + 0.6) / -1))
        #     - (correction_factor * (base_2["umaban_time_processed_2"] * -5))
        #     - (correction_factor * (base_2["height_diff"] / -1)) 
        # )
        # # `time_correction_rush` の計算
        # base_2.loc[:, "time_correction_position_rush"] = (
        #     base_2["time_correction_position"]
        #     - (correction_factor * ((base_2["curve_amount"] - 4) / 3))
        #     - (correction_factor * (base_2["curve_processed"] / -2))
        #     - (correction_factor * (base_2["goal_range_100_processed"] / 1))
        #     - (correction_factor * (base_2["goal_slope"] / 2))
        #     - (correction_factor * ((base_2["pace_diff"] + 0.6) / -1))
        #     - (correction_factor * (base_2["umaban_time_processed_2"] * -5))
        #     - (correction_factor * (base_2["height_diff"] / -1)) 
        # )



        merged_df = self.population.copy()  
        # # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # #




        a_base_2 = base_2[['race_id',
        'date',
        'horse_id',
        "time_pace_diff_slope_range_groundstate_position",
        "time_correction",
        ]]

        grouped_df = a_base_2.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        for n_race in tqdm(n_races, desc=f"rank_diff"):
            df_speed = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [
                        "time_pace_diff_slope_range_groundstate_position",
                        "time_correction",
                    ]
                ]
                .agg(["min","max","mean"])
            )
            # original_df = df_speed.copy()
            df_speed.columns = [
                "_".join(col) + f"_{n_race}races" for col in df_speed.columns
            ]
            # レースごとの相対値に変換
            tmp_df = df_speed.groupby(["race_id"])
            relative_df = ((df_speed - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            merged_df = merged_df.merge(
                relative_df, on=["race_id", "horse_id"], how="left"
            )
            # 相対値を付けない元の列をそのまま追加
            # original_df.columns = [
                # "_".join(col) + f"_{n_race}races" for col in original_df.columns
            # ]  # 列名変更
            
            # merged_df = merged_df.merge(
                # original_df, on=["race_id", "horse_id"], how="left"
            # )


        self.agg_cross_time = merged_df
        print("agg_cross_time ...comp")
        





    def cross2(
        self, n_races: list[int] = [1,3,5,8]
    ):  
    
        


        # # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # #



        merged_df = self.population.copy()  

        def calculate_race_position_percentage(group, n_race: int):
            """
            過去nレースにおける脚質割合を計算する
            """
            # 過去nレースに絞る
            past_races = group.head(n_race)
            
            # 各脚質のカウント
            counts = past_races['race_position'].value_counts(normalize=True).to_dict()
            
            # 結果を辞書形式で返す（割合が存在しない場合は0.0を補完）
            return {
                "escape": counts.get(1, 0.0),
                "taking_lead": counts.get(2, 0.0),
                "in_front": counts.get(3, 0.0),
                "pursuit": counts.get(4, 0.0),
            }
    
        # 過去nレースのリスト
        n_race_list = [1, 3, 5, 8]

        baselog = (
            self.population.merge(
                self.horse_results,
                on=["horse_id"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
       # 集計用データフレームの初期化
        merged_df = self.population.copy()
        
        # grouped_dfを適用して計算
        grouped_df = baselog.groupby(["race_id", "horse_id"])
        
        # 各過去nレースの割合を計算して追加
        for n_race in n_race_list:
            
            position_percentage = grouped_df.apply(
                lambda group: calculate_race_position_percentage(group, n_race=n_race),
                include_groups=False,  # グループ列を除外
            )
            
            # 結果をデータフレームとして展開
            position_percentage_df = position_percentage.apply(pd.Series).reset_index()
            position_percentage_df.rename(
                columns={
                    "escape": f"escape_{n_race}races",
                    "taking_lead": f"taking_lead_{n_race}races",
                    "in_front": f"in_front_{n_race}races",
                    "pursuit": f"pursuit_{n_race}races",
                },
                inplace=True,
            )
            
            # 結果をマージ
            merged_df = merged_df.merge(position_percentage_df, on=["race_id", "horse_id"], how="left")


        # dominant_position_category を計算
        position_columns_5 = ["escape_5races", "taking_lead_5races", "in_front_5races", "pursuit_5races"]
        position_columns_3 = ["escape_3races", "taking_lead_3races", "in_front_3races", "pursuit_3races"]
        position_columns_1 = ["escape_1races", "taking_lead_1races", "in_front_1races", "pursuit_1races"]
        
        
        def determine_dominant_position(row):
            
            # 5racesの中から計算
            if not row[position_columns_5].isnull().all():
                max_column = row[position_columns_5].idxmax()
                return {
                    "escape_5races": 1,
                    "taking_lead_5races": 2,
                    "in_front_5races": 3,
                    "pursuit_5races": 4,
                }[max_column]
            # 3racesの中から計算
            elif not row[position_columns_3].isnull().all():
                max_column = row[position_columns_3].idxmax()
                return {
                    "escape_3races": 1,
                    "taking_lead_3races": 2,
                    "in_front_3races": 3,
                    "pursuit_3races": 4,
                }[max_column]
            # 1racesの中から計算
            elif not row[position_columns_1].isnull().all():
                max_column = row[position_columns_1].idxmax()
                return {
                    "escape_1races": 1,
                    "taking_lead_1races": 2,
                    "in_front_1races": 3,
                    "pursuit_1races": 4,
                }[max_column]
            # すべて欠損の場合は1を返す
            else:
                return 3
        
        # 各行に対して dominant_position_category を適用
        merged_df["dominant_position_category"] = merged_df.apply(determine_dominant_position, axis=1)
        merged_df1 = merged_df
    
    
       
        #ペース作成
        """
        merged_df にはrace_id（レースのid）と、そのレースに出走するhorse_id(馬のid)の列があり、
        さらに dominant_position_categoryというその馬の脚質を表す列がある
        脚質は1-4の四種類の数字カテゴリが入っており
        レースごとに、出走する馬全体における、その脚質がいる割合と絶対数を求め、
        それぞれ四つの列を新たに作成
        """
        # dominant_position_category の脚質マッピング
        category_mapping_per = {1: "escape_per", 2: "taking_lead_per", 3: "in_front_per", 4: "pursuit_per"}
        category_mapping_count = {1: "escape_count", 2: "taking_lead_count", 3: "in_front_count", 4: "pursuit_count"}
        
        # 各レースで dominant_position_category の割合を計算
        position_ratios = (
            merged_df.groupby("race_id")["dominant_position_category"]
            .value_counts(normalize=True)  # 割合を計算
            .unstack(fill_value=0)         # 行: race_id, 列: dominant_position_category の形に
            .rename(columns=category_mapping_per)  # 列名を割合用に置き換え
        )
        
        # 各レースで dominant_position_category の絶対数を計算
        position_counts = (
            merged_df.groupby("race_id")["dominant_position_category"]
            .value_counts()                # 絶対数を計算
            .unstack(fill_value=0)         # 行: race_id, 列: dominant_position_category の形に
            .rename(columns=category_mapping_count)  # 列名を絶対数用に置き換え
        )
        
        # 割合と絶対数を結合
        position_data = position_ratios.join(position_counts)
        columns_to_remove = list(position_data.columns)
        merged_df = merged_df.drop(columns=columns_to_remove, errors="ignore")
        
        # 元の DataFrame にマージ
        merged_df = merged_df.merge(position_data, how="left", on="race_id")
    
        
        if 'pace_category' in merged_df.columns:
            merged_df = merged_df.drop(columns=['pace_category'])
        
        # ペースカテゴリを計算する関数
        def calculate_pace_category(row):
            escape_count = row.get("escape_count", 0)
            taking_lead_per = row.get("taking_lead_per", 0)
        
            if escape_count >= 3:
                return 4  # ハイペース
            elif escape_count == 2:
                if taking_lead_per > 0.5:
                    return 4  # ハイペース
                else:
                    return 3  
            elif escape_count == 1:
                if taking_lead_per > 0.6:
                    return 4  # ハイペース
                elif 0.5 < taking_lead_per <= 0.6:
                    return 3  # ハイミドルペース
                elif 0.4 < taking_lead_per <= 0.5:
                    return 2  # スローミドルペース
                else:
                    return 1  # スローペース
            elif escape_count == 0:
                # 逃げ馬が0の場合、先行馬の割合に基づく
                if taking_lead_per > 0.7:
                    return 4  # ハイペース
                elif 0.6 < taking_lead_per <= 0.7:
                    return 3  # ハイミドルペース
                elif 0.5 < taking_lead_per <= 0.6:
                    return 2  # スローミドルペース
                else:
                    return 1  # スローペース
            return None  # 予期しない場合は None
        merged_df["pace_category"] = merged_df.apply(calculate_pace_category, axis=1)
        
        merged_df2 = merged_df




        """
        1/競馬場xタイプxランクx距離ごとの平均タイムと直近のレースとの差異の平均を特徴量に、そのまま数値として入れれる
        2/芝以外を除去
        3/競馬場x芝タイプで集計(グループバイ)
        4/"race_date_day_count"の当該未満かつ、800以内が一週間以内のデータになるはず
        5/その平均を取る
        6/+なら軽く、-なら重く、それぞれ5段階のカテゴリに入れる
        """    
        merged_df_ground = self.population.copy()    
        df = (
            merged_df_ground
            .merge(self.race_info[["race_id", "place","weather","ground_state","course_len","race_type","race_date_day_count","course_type"]], on="race_id")
        )
        df["place"] = df["place"].astype(int)
        

        #当該レース近辺のレース、馬場状態
        df_old2 = (
            self.results[["race_id", "horse_id","time","rank","rank_per_horse"]]
            .merge(self.race_info[["race_id", "place","weather","ground_state","race_grade","course_len","race_type","course_type","race_date_day_count"]], on="race_id")
        )

        df_old2["place"] = df_old2["place"].astype(int)
        df_old2["race_grade"] = df_old2["race_grade"].astype(int)
        # 距離/競馬場/タイプ/レースランク
        df_old2["distance_place_type_race_grade"] = (df_old2["course_type"].astype(str)+ df_old2["race_grade"].astype(str)).astype(int)
        

        df_old = (
            self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around","course_type"]]
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "umaban","nobori","rank","time","sex"]], on="race_id")
        )
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  
                     
        # 距離/競馬場/タイプ/レースランク
        df_old["distance_place_type_race_grade"] = (df_old["course_type"].astype(str)+ df_old["race_grade"].astype(str)).astype(int)
        
        df_old_copy = df_old
        # rank列が1, 2, 3の行だけを抽出
        df_old = df_old[df_old['rank'].isin([1, 2, 3,4,5])]

        target_mean_1 = df_old.groupby("distance_place_type_race_grade")["time"].mean()
        # 平均値をカテゴリ変数にマッピング
        # スライスをコピーしてから処理
        df_old = df_old.copy()

        df_old["distance_place_type_race_grade_encoded"] = df_old["distance_place_type_race_grade"].map(target_mean_1)
        
        
        df_old = df_old[["distance_place_type_race_grade",'distance_place_type_race_grade_encoded']]
        
        columns_to_merge = [
            ("distance_place_type_race_grade",'distance_place_type_race_grade_encoded'),
        ]
        
        # 各ペアを順番に処理
        for original_col, encoded_col in columns_to_merge:
            df2_subset = df_old[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
            df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
            df_old2 = df_old2.merge(df2_subset, on=original_col, how='left')  # dfにマージ
        df_old2 = df_old2[df_old2['rank'].isin([1, 2, 3,4,5])]
        df_old2["distance_place_type_race_grade_encoded_time_diff"] = df_old2['distance_place_type_race_grade_encoded'] - df_old2["time"]

        # df_old2= df_old2[df_old2["race_type"] != 2]
        # df_old2_1 = df_old2[df_old2["race_type"] != 0]
        df_old2_1 = df_old2.copy()
        # 2. df の各行について処理

        def compute_mean_for_row(row, df_old2_1):
            # race_type == 0 の場合は NaN を返す
            # if row["race_type"] == 0:
                # return np.nan
            df_old2_1["race_date_day_count"] = df_old2_1["race_date_day_count"].astype(int)
            target_day_count = int(row["race_date_day_count"])  # df の各行の race_date_day_count

            # target_day_count = row["race_date_day_count"]  # df の各行の race_date_day_count

                
            # 3. df_old2_1 から条件に一致する行をフィルタリング
            filtered_df_old2_1 = df_old2_1[
                (df_old2_1["race_date_day_count"] >= (target_day_count - 400)) &  # race_date_day_count が target_day_count-1200 以上
                (df_old2_1["race_date_day_count"] <= (target_day_count - 1)) &  # race_date_day_count が target_day_count-1 以下
                (df_old2_1["place"] == row["place"]) &  # place が一致
                (df_old2_1["race_type"] == row["race_type"])  & 
                (df_old2_1["weather"].isin([0, 1, 2])) &  # weather が 0, 1, 2 のいずれか
                (df_old2_1["ground_state"] == 0)   # ground_state が 0
                # &
                # (df_old2_1["rank_per_horse"] < 0.87)  # rank_per_horse が 0.87 未満
            ]

            filtered_df_old2_2 = df_old2_1[
                (df_old2_1["race_date_day_count"] >= (target_day_count - 1200)) &  # race_date_day_count が target_day_count-1200 以上
                (df_old2_1["race_date_day_count"] <= (target_day_count - 1)) &  # race_date_day_count が target_day_count-1 以下
                (df_old2_1["place"] == row["place"]) &  # place が一致
                (df_old2_1["race_type"] == row["race_type"]) & 
                (df_old2_1["weather"].isin([0, 1, 2])) &  # weather が 0, 1, 2 のいずれか
                (df_old2_1["ground_state"] == 0)   # ground_state が 0                    
                # &
                # (df_old2_1["rank_per_horse"] < 0.87)  # rank_per_horse が 0.87 未満
            ]
            # 4. フィルタリングした行の "distance_place_type_race_grade_encoded_time_diff" の平均を計算
            mean_time_diff = filtered_df_old2_1["distance_place_type_race_grade_encoded_time_diff"].mean()

            # 5. 計算結果を返す（NaNの場合も考慮）
            # return mean_time_diff if not np.isnan(mean_time_diff) else filtered_df_old2_2["distance_place_type_race_grade_encoded_time_diff"].mean()
            return np.nan_to_num(mean_time_diff, nan=np.nan_to_num(filtered_df_old2_2["distance_place_type_race_grade_encoded_time_diff"].mean()))

            # # 4. フィルタリングした行の "distance_place_type_race_grade_encoded_time_diff" の平均を計算
            # mean_time_diff = filtered_df_old2_1["distance_place_type_race_grade_encoded_time_diff"].mean()
            # # 5. 計算結果を返す（NaNの場合も考慮）
            # return mean_time_diff if not np.isnan(mean_time_diff) else np.nan
        


        # 6. df の各行に対して、計算した平均値を新しい列に追加
        df["mean_ground_state_time_diff"] = df.apply(compute_mean_for_row, axis=1, df_old2_1=df_old2_1)

        def assign_value(row):
            # weather が 3, 4, 5 または ground_state が 0 以外の場合は 5 を設定
            if row["ground_state"] in [3]:
                return 0
            if row["weather"] in [3, 4, 5]:
                return 1

            if row["ground_state"] in [1]:
                return 1
            if row["ground_state"] in [2]:
                if row["race_type"] in [0]:
                    # mean_ground_state_time_diff に基づいて分類
                    if 2.0 <= row["mean_ground_state_time_diff"]:
                        return 3.7  #　超高速馬場1
                    elif 1 <= row["mean_ground_state_time_diff"] < 2.0:
                        return 4.5  # 高速馬場2
                    
                    elif 0.4 <= row["mean_ground_state_time_diff"] < 1:
                        return 5  # 軽い馬場3
                        
                    elif -0.4 <= row["mean_ground_state_time_diff"] < 0.4:
                        return 5.5  # 標準的な馬場4
                    elif -1 <= row["mean_ground_state_time_diff"] < -0.4:
                        return 6.2  # やや重い馬場5
                    
                    elif -2 <= row["mean_ground_state_time_diff"] < -1:
                        return 7  # 重い馬場5
                    
                    elif row["mean_ground_state_time_diff"] < -2:
                        return 8  # 道悪7
                if row["race_type"] in [1,2]:
                    return 2.7
            # mean_ground_state_time_diff に基づいて分類
            if 2.0 <= row["mean_ground_state_time_diff"]:
                return 2.7  #　超高速馬場1
            elif 1 <= row["mean_ground_state_time_diff"] < 2.0:
                return 3.5  # 高速馬場2
            
            elif 0.4 <= row["mean_ground_state_time_diff"] < 1:
                return 3.8  # 軽い馬場3
                
            elif -0.4 <= row["mean_ground_state_time_diff"] < 0.4:
                return 4  # 標準的な馬場4
        
            
            elif -1 <= row["mean_ground_state_time_diff"] < -0.4:
                return 4.5  # やや重い馬場5
            
            elif -2 <= row["mean_ground_state_time_diff"] < -1:
                return 5.2  # 重い馬場5
            
            elif row["mean_ground_state_time_diff"] < -2:
                return 6  # 道悪7
            # 該当しない場合は NaN を設定
            return 3.5
        
        
        # 新しい列を追加
        df["ground_state_level"] = df.apply(assign_value, axis=1)

        
        merged_df = merged_df.merge(df[["race_id","date","horse_id","mean_ground_state_time_diff","ground_state_level"]],on=["race_id","date","horse_id"])	

        merged_df3 = merged_df
        # merged_df100 = merged_df2.merge(
        #     merged_df3,
        #     on=["race_id","date","horse_id"],
        # )
        # print(merged_df100.columns)
        # print(merged_df2.columns)
        # print(merged_df3.columns)

        merged_df_3rd = merged_df3[["race_id","date","horse_id","dominant_position_category","pace_category","mean_ground_state_time_diff","ground_state_level"]]


        # merged_df_all = merged_df_all.merge(
        #         self.race_info[["race_id", "goal_range_100","curve","goal_slope","course_len","place_season_condition_type_categori","start_slope","start_range","race_grade","race_type"]], 
        #         on="race_id",
        #     )
        





        # # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # #

        """
        まずは過去のタイプからいまのタイプを推定する列を作成
        現在のマージ先で、どんなタイプが優勢なのかを設定
        それに沿った情報を列にしてまとめる
        現在のタイプをマージ
        それをマージする
        """
        merged_df = self.population
        
        base_2 = (
            self.population.merge(
                self.horse_results,
                on=["horse_id"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        base_2["distance_place_type_ground_state"] = (base_2["course_type"].astype(str)+ base_2["ground_state"].astype(str)).astype(int)   
        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        base_2["distance_place_type_ground_state_grade"] = (base_2["distance_place_type_ground_state"].astype(str)+ base_2["race_grade"].astype(str)).astype(int)   
        


        base_2_old = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","season","course_type"]], on="race_id"
            )
        )

        df_old = (
            base_2_old
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "umaban","nobori","time","rank"]], on=["race_id", "horse_id"])
        )

        df_old["nobori"] = df_old["nobori"].fillna(df_old["nobori"].mean())
        
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        # df_old["around"] = df_old["around"].fillna(3).astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  
        df_old["rank"] = df_old["rank"].astype(int)

        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_ground_state"] = (df_old["course_type"].astype(str)+ df_old["ground_state"].astype(str)).astype(int)   
        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_ground_state_grade"] = (df_old["distance_place_type_ground_state"].astype(str)+ df_old["race_grade"].astype(str)).astype(int)   
        
        target_mean_1= (
            df_old[
            (df_old['rank'].isin([1,2,3,4,5,6,7]))  # rankが1, 2, 3
            ]
            .groupby("distance_place_type_ground_state_grade")["nobori"]
            .mean()
        )
        # #noboriの平均
        # target_mean_1 = df_old.groupby("distance_place_type_ground_state_grade")["nobori"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_ground_state_grade_nobori_encoded"] = df_old["distance_place_type_ground_state_grade"].map(target_mean_1).fillna(0)
        

        df_old_rush_type = df_old[["distance_place_type_ground_state_grade","distance_place_type_ground_state_grade_nobori_encoded"]]
        
        columns_to_merge = [("distance_place_type_ground_state_grade","distance_place_type_ground_state_grade_nobori_encoded")]
        
        
        for original_col, encoded_col in columns_to_merge:
            df2_subset = df_old_rush_type[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
            df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
            base_2 = base_2.merge(df2_subset, on=original_col, how='left')  # dfにマージ
            
        base_2["nobori_diff"] = base_2["distance_place_type_ground_state_grade_nobori_encoded"] - base_2["nobori"]

        base_2 = base_2.copy()
        def calculate_rush_type(row):
            if row["nobori"] < 33 and row["race_type"] == 1:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 33 and row["race_type"] == 1:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 33.9 and row["race_type"] == 1:
                return -4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34.5 and row["race_type"] == 1:
                return -2.5
            if row["nobori"] < 34.5 and row["race_type"] == 1:
                return -2

            if row["nobori"] < 33.5 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 33.5 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34.4 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 35 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -2.5
            if row["nobori"] < 35 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -2

            if row["nobori"] < 34 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -6
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 34.9 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] < 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -2.5
            if row["nobori"] < 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -2

            if row["nobori_diff"] >= 1:
                return -4
            if row["nobori_diff"] >= 0.6:
                return -2.5

            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 35.6 and row["race_type"] == 1 and row["rank"] <= 6:
                return 4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 34.5 and row["race_type"] == 1 and row["rank"] <= 6:
                return 2.5
            if row["nobori"] >= 34.5 and row["race_type"] == 1 and row["rank"] <= 6:
                return 2.5

            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 36.1 and row["race_type"] == 0 and row["course_len"] < 1600 and row["rank"] <= 6:
                return 4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 35 and row["race_type"] == 0 and row["course_len"] < 1600 and row["rank"] <= 6:
                return 2.5
            if row["nobori"] >= 35 and row["race_type"] == 0 and row["course_len"] < 1600 and row["rank"] <= 6:
                return 2.5

            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 36.6 and row["race_type"] == 0 and row["course_len"] >= 1600 and row["rank"] <= 6:
                return 4
            if row["nobori_pace_diff_slope_range_groundstate_position_umaban"] >= 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600 and row["rank"] <= 6:
                return 2.5
            if row["nobori"] >= 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600 and row["rank"] <= 6:
                return 2.5

            return 0

        # DataFrame に適用
        base_2["rush_type"] = base_2.apply(calculate_rush_type, axis=1)





        n_races: list[int] = [1, 3, 5, 8]
        """
        ・どのタイプが有利かを判断する列
        -は瞬発有利の競馬場
        +は持続有利
        """

        merged_df = self.population
        df_rush_type = (
            merged_df
            .merge(self.race_info[["race_id","curve" ,"curve_amount","goal_range","goal_slope","height_diff"]], on=["race_id"])
            .merge(merged_df_3rd[["race_id","date","horse_id","pace_category"]],on=["race_id","date","horse_id"])   
            .merge(self.results[["race_id","horse_id","umaban"]], on=["race_id","horse_id"])
        )

        df_rush_type["goal_range_processed"] = (((df_rush_type["goal_range"])-360)/100)
        df_rush_type["goal_range_processed"] = df_rush_type["goal_range_processed"].apply(
            lambda x: x*2 if x < 0 else x*0.9
        )

        df_rush_type["curve_processed"] = df_rush_type["curve"] - 4.5
        # +の場合は数値を8倍する
        df_rush_type["curve_processed"] = df_rush_type["curve_processed"].apply(
            lambda x: x * 8 if x > 0 else x
        )
        """
        持続よし
        2
        4
        4
        1
        6
        1
        7
        """

        #4ハイペース、1スローペース
        df_rush_type["rush_type_Advantages"] = (
            0 - (((df_rush_type["curve_amount"] - 4) / 1))
            - ((df_rush_type["curve_processed"] / -1))
            - ((df_rush_type["goal_range_processed"] / 1.2))
            - ((df_rush_type["goal_slope"] / 0.8))
            - (((df_rush_type["pace_category"] - 2.5) / -0.7))
            - (((df_rush_type["umaban"]-8) / -6))
            - ((df_rush_type["height_diff"] / -0.7)) 
        )
        




        def calculate_rush_type_averages(group, n_race):
            """
            各 horse_id に対して、過去 n_race レース分の rush_type の負・正の平均を計算する。
            """
            past_races = group.head(n_race)  # 過去 n_race レース分を取得

            negative_rush = past_races[past_races["rush_type"] < 0]["rush_type"]
            positive_rush = past_races[past_races["rush_type"] > 0]["rush_type"]

            return pd.Series({
                f"rush_type_avg_syunpatu_{n_race}races": negative_rush.mean() if not negative_rush.empty else 0,
                f"rush_type_avg_zizoku_{n_race}races": positive_rush.mean() if not positive_rush.empty else 0
            })

        # 計算対象の過去 n レースのリスト
        n_race_list = [1, 3, 5, 8]

        # 結果を格納する DataFrame
        merged_df_rush2 = base_2.copy()
        # race_id, horse_id でグループ化
        grouped_df = merged_df_rush2.groupby(["race_id", "horse_id"])

        # 各 n_race について計算
        for n_race in tqdm(n_race_list, desc="計算中"):
            rush_type_avg_df = grouped_df.apply(lambda group: calculate_rush_type_averages(group, n_race)).reset_index()

            # df にマージ
            df_rush_type= df_rush_type.merge(rush_type_avg_df, on=["race_id", "horse_id"], how="left")




        def calculate_rush_type_advantage(df, n_race_list=[1,3,5,8]):
            # 新しい列を格納するリスト
            new_column_values = []

            # 各 row に対して処理を行う
            for _, row in df.iterrows():
                selected_value = None

                # n_race の順に参照（8, 5, 3, 1 の順番）
                for n_race in n_race_list:
                    col_syunpatu = f"rush_type_avg_syunpatu_{n_race}races"
                    col_zizoku = f"rush_type_avg_zizoku_{n_race}races"

                    # rush_type_Advantages が 0 未満の時
                    if row["rush_type_Advantages"] < 0:
                        if row[col_syunpatu] != 0:
                            selected_value = row[col_syunpatu]
                        elif row[col_zizoku] != 0:
                            selected_value = row[col_zizoku]
                        else:
                            selected_value = 0  # 両方とも0の場合、デフォルトで0を設定

                    # rush_type_Advantages が 0 以上の時
                    elif row["rush_type_Advantages"] >= 0:      
                        if row[col_zizoku] != 0:
                            selected_value = row[col_zizoku]
                        elif row[col_syunpatu] != 0:
                            selected_value = row[col_syunpatu]
                        else:
                            selected_value = 0  # 両方とも0の場合、デフォルトで0を設定
                    # 値が見つかった場合はそのままループを抜ける
                    if selected_value is not None:
                        break

                # 選ばれた値を新しい列に追加
                new_column_values.append(selected_value if selected_value is not None else 0)

            # 新しい列を df に追加
            df["rush_type_final"] = new_column_values

            return df

        # 新しい列を追加
        df_rush_type = calculate_rush_type_advantage(df_rush_type)




        merged_df_all = merged_df_3rd.merge(
                df_rush_type[["race_id","date","horse_id","rush_type_Advantages","rush_type_final"]], 
                on=["race_id","date","horse_id"],
            )


        merged_df_all = merged_df_all.merge(
                self.race_info[["race_id", "goal_range_100","curve","weather",'start_point','ground_state','curve_R12', 'curve_R34',"curve_amount","season_turf_condition","season","day" ,'flont_slope',"goal_slope",'first_curve_slope', 'goal_range',"course_len","place_season_condition_type_categori","start_slope","start_range","race_grade","race_type","last_curve_slope"]], 
                on="race_id",
            )

        merged_df_all = merged_df_all.merge(
                self.results[["race_id","horse_id","umaban","n_horses",'umaban_odd']], 
                on=["race_id","horse_id"],
            )

        merged_df_all["rush_advantages_cross"] = merged_df_all["rush_type_Advantages"] * merged_df_all["rush_type_final"]
        merged_df_all["rush_advantages_cross_plus"] = merged_df_all["rush_type_Advantages"] + merged_df_all["rush_type_final"]

        self.all_position = merged_df_all

        """
        展開
        馬場状態
        2が道悪、-2が超高速
        4前後だと思う

        脚質
        1逃げ
        4追い込み

        ペース
        ハイ:4
        ロー:1

        9前後かな

        """



        # merged_df_all.loc[(merged_df_all['race_type'] == 0) & (merged_df_all['weather'].isin([1, 2])) & (merged_df_all['ground_state'] == 2), "ground_state_level_processed"] =  12
        # merged_df_all.loc[(merged_df_all['ground_state'] == 2), "ground_state_level_processed"] = -14
        # merged_df_all.loc[merged_df_all['ground_state'].isin([1, 3]), "ground_state_level_processed"] = -24

        # merged_df_all.loc[merged_df_all["mean_ground_state_time_diff"].isna(), "ground_state_level_processed"] = (merged_df_all["ground_state_level"] - 4) * 8

        # merged_df_all.loc[~merged_df_all['ground_state'].isin([1, 2, 3]), "ground_state_level_processed"] = merged_df_all["mean_ground_state_time_diff"] * -7

        # 初期化（すべてNaNにする）
        merged_df_all["ground_state_level_processed"] = np.nan  

        # 条件を適用
        merged_df_all.loc[(merged_df_all['race_type'] == 0) & (merged_df_all['weather'].isin([1, 2])) & (merged_df_all['ground_state'] == 2), "ground_state_level_processed"] = 12

        merged_df_all.loc[(merged_df_all['race_type'] == 0) & (merged_df_all['ground_state'] == 2) & merged_df_all["ground_state_level_processed"].isna(), "ground_state_level_processed"] = -6

        merged_df_all.loc[(merged_df_all['race_type'] == 0) & merged_df_all['ground_state'].isin([1, 3]) & merged_df_all["ground_state_level_processed"].isna(), "ground_state_level_processed"] = -10


        merged_df_all.loc[(merged_df_all['ground_state'] == 2) & merged_df_all["ground_state_level_processed"].isna(), "ground_state_level_processed"] = -14

        merged_df_all.loc[merged_df_all['ground_state'].isin([1, 3]) & merged_df_all["ground_state_level_processed"].isna(), "ground_state_level_processed"] = -24

        # `NaN` の場合のみ計算
        merged_df_all.loc[merged_df_all["ground_state_level_processed"].isna() & merged_df_all["mean_ground_state_time_diff"].isna(), "ground_state_level_processed"] = (merged_df_all["ground_state_level"] - 4) * 8

        merged_df_all.loc[merged_df_all["ground_state_level_processed"].isna() & ~merged_df_all['ground_state'].isin([1, 2, 3]), "ground_state_level_processed"] = merged_df_all["mean_ground_state_time_diff"] * -7



        merged_df_all["pace_category_processed"]  = (merged_df_all["pace_category"]  - 2.5) *20
        # dominant_position_category_processed 列の処理
        merged_df_all["ground_state_level_processed"] = merged_df_all["ground_state_level_processed"] + ((merged_df_all["season_turf_condition"] - 7)*1/1)


        # tenkai_sumed の計算
        merged_df_all["tenkai_sumed"] = (
            merged_df_all["ground_state_level_processed"] + merged_df_all["pace_category_processed"]
        )

        # 条件に応じて dominant_position_category_processed を更新
        merged_df_all.loc[merged_df_all["tenkai_sumed"] < 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.93, 2: -2, 3: -0.7, 4: 1.67})
            .astype(float)
        )

        merged_df_all.loc[merged_df_all["tenkai_sumed"] >= 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1})
            .astype(float)
        )

        # tenkai_combined の計算
        merged_df_all["tenkai_combined_2"] = merged_df_all["tenkai_sumed"] * merged_df_all["dominant_position_category_processed"]




        # place_season_condition_type_categori の処理
        merged_df_all["place_season_condition_type_categori_processed_z"] = merged_df_all["place_season_condition_type_categori"].replace(
            {5: -2.3, 4: -1.4, 3: 0, 2: 1.4, 1: 2.3, -1: -1.4, -2: 0, -3: 1.4, -4: 1.8, -10000: 0}
        ).astype(float)

        # tenkai_place_sumed の計算
        merged_df_all["tenkai_place_sumed"] = merged_df_all["tenkai_sumed"] + merged_df_all["place_season_condition_type_categori_processed_z"]*2

        # 条件に応じて dominant_position_category_processed を更新
        merged_df_all.loc[merged_df_all["tenkai_place_sumed"] < 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.93, 2: -2,3: 0.1, 4: 1.67})
            .astype(float)
        )

        merged_df_all.loc[merged_df_all["tenkai_place_sumed"] >= 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1})
            .astype(float)
        )

        # tenkai_place_combined の計算
        merged_df_all["tenkai_place_combined"] = merged_df_all["tenkai_place_sumed"] * merged_df_all["dominant_position_category_processed"]





        # start_slope の処理
        merged_df_all["start_slope_abs_processed"] = merged_df_all["start_slope"] * -2

        # tenkai_place_start_slope_sumed の計算
        merged_df_all["tenkai_place_start_slope_sumed"] = merged_df_all["tenkai_place_sumed"] + merged_df_all["start_slope_abs_processed"]

        # 条件に応じて dominant_position_category_processed を更新
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_sumed"] < 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.93, 2: -2, 3: -0.7, 4: 1.67})
            .astype(float)
        )

        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_sumed"] >= 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1})
            .astype(float)
        )

        # tenkai_place_start_slope_combined の計算
        merged_df_all["tenkai_place_start_slope_combined"] = merged_df_all["tenkai_place_start_slope_sumed"] * merged_df_all["dominant_position_category_processed"]




        # start_range_processed_1 の計算
        merged_df_all["start_range_processed_1"] = ((merged_df_all["start_range"] - 360) / 30).apply(lambda x: x * 4 if x < 0 else x * 0.7)

        # tenkai_place_start_slope_range_sumed の計算
        merged_df_all["tenkai_place_start_slope_range_sumed"] = merged_df_all["tenkai_place_start_slope_sumed"] + merged_df_all["start_range_processed_1"]

        # 条件に応じて dominant_position_category_processed を更新
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_sumed"] < 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.93, 2: -2, 3: -0.7, 4: 1.67})
            .astype(float)
        )

        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_sumed"] >= 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1})
            .astype(float)
        )

        # tenkai_place_start_slope_range_combined の計算
        merged_df_all["tenkai_place_start_slope_range_combined"] = merged_df_all["tenkai_place_start_slope_range_sumed"] * merged_df_all["dominant_position_category_processed"]





        # tenkai_place_start_slope_range_grade_sumed の計算
        merged_df_all["tenkai_place_start_slope_range_grade_sumed"] = merged_df_all["tenkai_place_start_slope_range_sumed"] + ((merged_df_all["race_grade"] - 70) / 10)
        # 条件に応じて dominant_position_category_processed を更新
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_sumed"] < 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.93, 2: -2, 3: -0.7, 4: 1.67})
            .astype(float)
        )

        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_sumed"] >= 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1})
            .astype(float)
        )

        # tenkai_place_start_slope_range_grade_combined の計算
        merged_df_all["tenkai_place_start_slope_range_grade_combined"] = merged_df_all["tenkai_place_start_slope_range_grade_sumed"] * merged_df_all["dominant_position_category_processed"]





        # tenkai_place_start_slope_range_grade_lcurve_slope_sumed の計算
        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_sumed"] = merged_df_all["tenkai_place_start_slope_range_grade_sumed"] + (merged_df_all["last_curve_slope"] * 2.5)

        # 条件に応じて dominant_position_category_processed を更新
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_sumed"] < 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.93, 2: -2,3: 0.1, 4: 1.67})
            .astype(float)
        )

        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_sumed"] >= 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1})
            .astype(float)
        )

        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_combined"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_sumed"] * merged_df_all["dominant_position_category_processed"]




        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_sumed"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_sumed"] + (merged_df_all["race_type"]-1)*4

        # 条件に応じて dominant_position_category_processed を更新
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_sumed"] < 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.93, 2: -2, 3: -0.7, 4: 1.67})
            .astype(float)
        )

        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_sumed"] >= 0, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1})
            .astype(float)
        )

        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_combined"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_sumed"] * merged_df_all["dominant_position_category_processed"]




        # -4.5 を行う
        merged_df_all["curve_processed"] = merged_df_all["curve"] - 4.5
        # +の場合は数値を8倍する
        merged_df_all["curve_processed"] = merged_df_all["curve_processed"].apply(
            lambda x: x * 8 if x > 0 else x
        )

        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_sumed"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_sumed"] + (merged_df_all["curve_processed"]*2.5)

        # dominant_position_category_processed の更新 (再利用)
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_sumed"] < 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: -0.7, 4: 1.67}
        )
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_sumed"] >= 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_combined"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_sumed"] * merged_df_all["dominant_position_category_processed"]




        merged_df_all["goal_range_processed_1"] = (((merged_df_all["goal_range"])-360)/100)
        merged_df_all["goal_range_processed_1"] = merged_df_all["goal_range_processed_1"].apply(
            lambda x: x*3 if x < 0 else x*1.7
        )


        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_sumed"] + (merged_df_all["goal_range_processed_1"]*3)

        # dominant_position_category_processed の更新 (再利用)
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] < 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: -0.7, 4: 1.67}
        )
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] >= 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] * merged_df_all["dominant_position_category_processed"]





        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_sumed"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] + (merged_df_all["goal_slope"]*5)

        # dominant_position_category_processed の更新 (再利用)
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_sumed"] < 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: -0.7, 4: 1.67}
        )
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_sumed"] >= 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_combined"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_sumed"] * merged_df_all["dominant_position_category_processed"]







        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_sumed"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_sumed"] + (merged_df_all["first_curve_slope"]*-5)

        # dominant_position_category_processed の更新 (再利用)
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_sumed"] < 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: -0.7, 4: 1.67}
        )
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_sumed"] >= 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_combined"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_sumed"] * merged_df_all["dominant_position_category_processed"]





        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_sumed"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_sumed"] + (merged_df_all["flont_slope"]*-1)

        # dominant_position_category_processed の更新 (再利用)
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_sumed"] < 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: -0.7, 4: 1.67}
        )
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_sumed"] >= 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_combined"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_sumed"] * merged_df_all["dominant_position_category_processed"]





        merged_df_all["course_len_processed_11"] = (merged_df_all["course_len"] - 1600) /1000
        merged_df_all["course_len_processed_11"] = merged_df_all["course_len_processed_11"].apply(
            lambda x: x*4 if x < 0 else x
        )

        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_sumed"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_sumed"] + merged_df_all["course_len_processed_11"]

        # dominant_position_category_processed の更新 (再利用)
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_sumed"] < 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: -0.7, 4: 1.67}
        )
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_sumed"] >= 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_combined"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_sumed"] * merged_df_all["dominant_position_category_processed"]



        #0,01-0.01,内がプラス(内枠が有利を受けるとしたら、rank_diffは+にして、有利ポジはマイナス補正)
        merged_df_all["umaban_processed2"] = merged_df_all["umaban"].apply(
            lambda x: ((x*-1.5)+1.5) if x < 4 else ((x-8)/1.5)-1
        ).astype(float)
        #0,-0.1,-0.3,-0.36,-0.3,-0.23,（-1/10のとき）
        merged_df_all["umaban_processed2"] = merged_df_all["umaban_processed2"]*2
        #0 , -0.05
        #1（奇数）または 0（偶数）,偶数が有利
        merged_df_all.loc[:, "umaban_odd_processed2"] = (
            (merged_df_all["umaban_odd"]-1)*2
        ).astype(float)

        #rdが-0.25,,0.25が0.5に
        merged_df_all["umaban_processed2"] = merged_df_all["umaban_processed2"] / ((merged_df_all["course_len_processed_11"]) + 4)
        merged_df_all["umaban_odd_processed2"] = merged_df_all["umaban_odd_processed2"] / ((merged_df_all["course_len_processed_11"])+4)




        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_sumed"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_sumed"] + merged_df_all["umaban_odd_processed2"] + merged_df_all["umaban_processed2"]

        # dominant_position_category_processed の更新 (再利用)
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_sumed"] < 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: -0.7, 4: 1.67}
        )
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_sumed"] >= 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_sumed"] * merged_df_all["dominant_position_category_processed"]




        merged_df_all2 = merged_df_all.copy()

        


        # """
        # タイプ適性
        # 7*4
        # rankgradeと+してもよい
        # sumと+するなら1/10する
        # multiとするならそのまま
        # 掛け算するなら+200/200する
        # """
        # df["rush_advantages_cross"] = df["rush_type_Advantages"] * df["rush_type_final"]



        # """
        # umabanのみ補正(rank_diff)
        # "rank_diff_pace_course_len_ground_state_type",
        # "rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope",
        # "rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point",
        # "rank_diff_correction",
        # "rank_diff_correction_position",
        # "rank_diff_correction_rush",
        # "rank_diff_correction_position_rush",
        # "rank_diff_correction_position_rush_xxx_race_grade_multi",
        # "rank_diff_correction_position_xxx_race_grade_multi",
        # "rank_diff_correction_rush_xxx_race_grade_multi",
        # "rank_diff_correction_xxx_race_grade_multi",
        # "rank_diff_correction_position_rush_xxx_race_grade_sum",
        # "rank_diff_correction_position_xxx_race_grade_sum",
        # "rank_diff_correction_rush_xxx_race_grade_sum",
        # "rank_diff_correction_xxx_race_grade_sum",

        # """
        # # -4.5 を行う
        # merged_df_all["curve_processed_umaban"] = merged_df_all["curve"] - 4.5
        # # +の場合は数値を8倍する
        # merged_df_all["curve_processed_umaban"] = merged_df_all["curve_processed_umaban"].apply(
        #     lambda x: x * 4 if x > 0 else x
        # )

        # merged_df_all["curve_umaban"] = (merged_df_all["curve_processed_umaban"] + merged_df_all["umaban_processed2"] * 1/2)*-1

        # # `race_type` に基づいて `umaban_p_type` を設定する
        # merged_df_all["umaban_p_type"] = merged_df_all.apply(lambda row: -1/2 * row["umaban_processed2"] if row["race_type"] == 1 else (1/2 * row["umaban_processed2"] if row["race_type"] == 0 else None), axis=1)

        # merged_df_all["umaban_odd_2"] = merged_df_all["umaban_odd_processed2"] * -1

        # merged_df_all["last_curve_umaban"] = ((merged_df_all["last_curve_slope"]) + merged_df_all["umaban_processed2"] * 1/2) * -1

        # merged_df_all["start_range_umaban"] = ((merged_df_all["start_range_processed_1"] / 2) + merged_df_all["umaban_processed2"] * 1 + merged_df_all["umaban_odd_processed2"]) * -1

        # merged_df_all["start_slope_umaban"] = ((merged_df_all["start_slope_abs_processed"] * -1) + merged_df_all["umaban_processed2"] * 1/2 + merged_df_all["umaban_odd_processed2"] * 1/2) * -1

        # # 最終的な補正値の計算
        # umaban_correction_position = (
        #     merged_df_all["curve_umaban"] + merged_df_all["umaban_p_type"] + merged_df_all["umaban_odd_2"] + merged_df_all["last_curve_umaban"] + merged_df_all["start_range_umaban"] + merged_df_all["start_slope_umaban"]
        # )



        #-3,2内がマイナス
        merged_df_all["umaban_processed_100"] = merged_df_all["umaban"].apply(
            lambda x: ((x*-1)) if x < 4 else ((x-8)/3)-1
        ).astype(float)
        #0-0.005

        merged_df_all["umaban_judge"] = (merged_df_all["umaban"].astype(float)/merged_df_all["n_horses"].astype(float))-0.55

        #1（偶数）または 0（奇数）偶数有利
        merged_df_all.loc[:, "umaban_odd_processed_100"] = (
            (merged_df_all["umaban_odd"]-1)*-1
        ).astype(float)

        # 1600で正規化,-0.5 - 1
        merged_df_all["course_len_processed_22"] = (merged_df_all["course_len"] / 1700)-1

        # ,-1.5 - 10
        merged_df_all["course_len_processed_100"] = merged_df_all["course_len_processed_22"].apply(
            lambda x: x*3 if x <= 0 else x*10
        )
        merged_df_all["course_len_processed_100"] = ((merged_df_all["course_len_processed_22"] + 3)/3)

        merged_df_all["umaban_processed_100"] = merged_df_all["umaban_processed_100"] / merged_df_all["course_len_processed_100"]
        merged_df_all["umaban_odd_processed_100"] = merged_df_all["umaban_odd_processed_100"] / merged_df_all["course_len_processed_100"]



        merged_df_all["first_corner"] = np.where(
            (merged_df_all["curve_amount"] == 2) | (merged_df_all["curve_amount"] == 6),
            merged_df_all["curve_R34"],
            np.where(
                (merged_df_all["curve_amount"] == 4) | (merged_df_all["curve_amount"] == 8),
                merged_df_all["curve_R12"],
                0  # それ以外のとき 0
            )
        )

        """
        内枠有利ならマイナス方面とumaban_processed_100を足し算する、そこにoddを足し算する
        約-4.5から3までが最大
        """
        # 'start_point'が2の行だけに対して処理を行う
        merged_df_all.loc[
            (merged_df_all["start_point"] == 2) & (merged_df_all["ground_state"] != 0),
            "start_point"
        ] = -1

        #2
        merged_df_all["umaban_combined"] = (
            (merged_df_all["umaban_processed_100"]* (
                (merged_df_all["start_point"] - 1.5)*3
                )
                )
            /2
            + merged_df_all["umaban_odd_processed_100"]
            )

        #5
        merged_df_all["umaban_curve_combined"] = (
            (merged_df_all["umaban_processed_100"]* (
                (merged_df_all["start_point"] - 1.5)*3#1.5
                +((merged_df_all["curve"]-4.5)*2) #8
                )
                )
            /2
            + merged_df_all["umaban_odd_processed_100"]
            )

        #5
        merged_df_all["umaban_curve_lcurves_combined"] = (
            (merged_df_all["umaban_processed_100"]* (
                (merged_df_all["start_point"] - 1.5)*4#2
                +((merged_df_all["curve"]-4.5)*2) #8
                + (merged_df_all["last_curve_slope"])*4 #8
                )
                )
            /2
            + merged_df_all["umaban_odd_processed_100"]
            )
        
        #5
        merged_df_all["umaban_curve_lcurves_condition_combined"] = (
            (merged_df_all["umaban_processed_100"]* (
                (merged_df_all["start_point"] - 1.5)*4#2
                +((merged_df_all["curve"]-4.5)*2) #8
                + (merged_df_all["last_curve_slope"])*4 #8
                + (merged_df_all["season_turf_condition"]-7)*3.2 #14
                )
                )
            /2
            + merged_df_all["umaban_odd_processed_100"]
            )
        
        #2
        merged_df_all["umaban_curve_lcurves_condition_type_combined"] = (
            (merged_df_all["umaban_processed_100"]* (
                (merged_df_all["start_point"] - 1.5)*4#2
                +((merged_df_all["curve"]-4.5)*2) #8
                + (merged_df_all["last_curve_slope"])*4 #8
                + (merged_df_all["season_turf_condition"]-7)*3.2 #14
                + (merged_df_all["race_type"] - 0.5)*-6
                )
                )
            /2
            + merged_df_all["umaban_odd_processed_100"]
            )
        #2
        merged_df_all["umaban_curve_lcurves_condition_type_corner_combined"] = (
            (merged_df_all["umaban_processed_100"]* (
                (merged_df_all["start_point"] - 1.5)*4#2
                +((merged_df_all["curve"]-4.5)*2) #8
                + (merged_df_all["last_curve_slope"])*4 #8
                + (merged_df_all["season_turf_condition"]-7)*3.2 #14
                + (merged_df_all["race_type"] - 0.5)*-6
                + (merged_df_all["first_corner"] - 100)/-20

                )
                )
            /2
            + merged_df_all["umaban_odd_processed_100"]
            )
        



        # _combined系の列をリストで指定
        combined_columns = [
            "rush_type_final","rush_type_Advantages","rush_advantages_cross","rush_advantages_cross_plus","tenkai_combined_2","tenkai_place_combined","tenkai_place_start_slope_combined","tenkai_place_start_slope_range_combined","tenkai_place_start_slope_range_grade_combined","tenkai_place_start_slope_range_grade_lcurve_slope_combined","tenkai_place_start_slope_range_grade_lcurve_slope_type_combined",
            "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_combined","tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined","tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_combined","tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_combined",
            "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_combined","tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_combined","tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined","umaban_combined",
            "umaban_curve_combined","umaban_curve_lcurves_combined","umaban_curve_lcurves_condition_combined","umaban_curve_lcurves_condition_type_combined","umaban_curve_lcurves_condition_type_corner_combined"
        ]
        # 各_combined列を標準化
        for col in combined_columns:
            merged_df_all[f"{col}_standardized"] = (
                merged_df_all[col] - merged_df_all.groupby("race_id")[col].transform("mean")
            ) / merged_df_all.groupby("race_id")[col].transform("std")
        

        merged_df_all= merged_df_all.merge(
            self.agg_rank_diff[['race_id',
            'date',
            'horse_id',

            "rank_diff_position_xxx_race_grade_multi_mean_1races",
            "rank_diff_all_xxx_race_grade_multi_mean_1races",
            "rank_diff_position_xxx_race_grade_sum_mean_1races",
            "rank_diff_all_xxx_race_grade_sum_mean_1races",

            "rank_diff_position_xxx_race_grade_multi_mean_3races",
            "rank_diff_all_xxx_race_grade_multi_mean_3races",
            "rank_diff_position_xxx_race_grade_sum_mean_3races",
            "rank_diff_all_xxx_race_grade_sum_mean_3races",

            "rank_diff_position_xxx_race_grade_multi_mean_5races",
            "rank_diff_all_xxx_race_grade_multi_mean_5races",
            "rank_diff_position_xxx_race_grade_sum_mean_5races",
            "rank_diff_all_xxx_race_grade_sum_mean_5races",

            "rank_diff_position_xxx_race_grade_multi_mean_8races",
            "rank_diff_all_xxx_race_grade_multi_mean_8races",
            "rank_diff_position_xxx_race_grade_sum_mean_8races",
            "rank_diff_all_xxx_race_grade_sum_mean_8races",


            "rank_diff_position_xxx_race_grade_multi_min_1races",
            "rank_diff_all_xxx_race_grade_multi_min_1races",
            "rank_diff_position_xxx_race_grade_sum_min_1races",
            "rank_diff_all_xxx_race_grade_sum_min_1races",

            "rank_diff_position_xxx_race_grade_multi_min_3races",
            "rank_diff_all_xxx_race_grade_multi_min_3races",
            "rank_diff_position_xxx_race_grade_sum_min_3races",
            "rank_diff_all_xxx_race_grade_sum_min_3races",

            "rank_diff_position_xxx_race_grade_multi_min_5races",
            "rank_diff_all_xxx_race_grade_multi_min_5races",
            "rank_diff_position_xxx_race_grade_sum_min_5races",
            "rank_diff_all_xxx_race_grade_sum_min_5races",

            "rank_diff_position_xxx_race_grade_multi_min_8races",
            "rank_diff_all_xxx_race_grade_multi_min_8races",
            "rank_diff_position_xxx_race_grade_sum_min_8races",
            "rank_diff_all_xxx_race_grade_sum_min_8races",

            "rank_diff_position_xxx_race_grade_multi_max_1races",
            "rank_diff_all_xxx_race_grade_multi_max_1races",
            "rank_diff_position_xxx_race_grade_sum_max_1races",
            "rank_diff_all_xxx_race_grade_sum_max_1races",

            "rank_diff_position_xxx_race_grade_multi_max_3races",
            "rank_diff_all_xxx_race_grade_multi_max_3races",
            "rank_diff_position_xxx_race_grade_sum_max_3races",
            "rank_diff_all_xxx_race_grade_sum_max_3races",

            "rank_diff_position_xxx_race_grade_multi_max_5races",
            "rank_diff_all_xxx_race_grade_multi_max_5races",
            "rank_diff_position_xxx_race_grade_sum_max_5races",
            "rank_diff_all_xxx_race_grade_sum_max_5races",

            "rank_diff_position_xxx_race_grade_multi_max_8races",
            "rank_diff_all_xxx_race_grade_multi_max_8races",
            "rank_diff_position_xxx_race_grade_sum_max_8races",
            "rank_diff_all_xxx_race_grade_sum_max_8races",


            ]],
            on=["race_id","date","horse_id"],
        )




        merged_df_all = merged_df_all.copy()
        # 100種類のカラム名のリスト
        columns = [           
                    "rank_diff_position_xxx_race_grade_multi_mean_1races",
                    "rank_diff_all_xxx_race_grade_multi_mean_1races",
                    "rank_diff_position_xxx_race_grade_sum_mean_1races",
                    "rank_diff_all_xxx_race_grade_sum_mean_1races",

                    "rank_diff_position_xxx_race_grade_multi_mean_3races",
                    "rank_diff_all_xxx_race_grade_multi_mean_3races",
                    "rank_diff_position_xxx_race_grade_sum_mean_3races",
                    "rank_diff_all_xxx_race_grade_sum_mean_3races",

                    "rank_diff_position_xxx_race_grade_multi_mean_5races",
                    "rank_diff_all_xxx_race_grade_multi_mean_5races",
                    "rank_diff_position_xxx_race_grade_sum_mean_5races",
                    "rank_diff_all_xxx_race_grade_sum_mean_5races",

                    "rank_diff_position_xxx_race_grade_multi_mean_8races",
                    "rank_diff_all_xxx_race_grade_multi_mean_8races",
                    "rank_diff_position_xxx_race_grade_sum_mean_8races",
                    "rank_diff_all_xxx_race_grade_sum_mean_8races",


                    "rank_diff_position_xxx_race_grade_multi_min_1races",
                    "rank_diff_all_xxx_race_grade_multi_min_1races",
                    "rank_diff_position_xxx_race_grade_sum_min_1races",
                    "rank_diff_all_xxx_race_grade_sum_min_1races",

                    "rank_diff_position_xxx_race_grade_multi_min_3races",
                    "rank_diff_all_xxx_race_grade_multi_min_3races",
                    "rank_diff_position_xxx_race_grade_sum_min_3races",
                    "rank_diff_all_xxx_race_grade_sum_min_3races",

                    "rank_diff_position_xxx_race_grade_multi_min_5races",
                    "rank_diff_all_xxx_race_grade_multi_min_5races",
                    "rank_diff_position_xxx_race_grade_sum_min_5races",
                    "rank_diff_all_xxx_race_grade_sum_min_5races",

                    "rank_diff_position_xxx_race_grade_multi_min_8races",
                    "rank_diff_all_xxx_race_grade_multi_min_8races",
                    "rank_diff_position_xxx_race_grade_sum_min_8races",
                    "rank_diff_all_xxx_race_grade_sum_min_8races",

                    "rank_diff_position_xxx_race_grade_multi_max_1races",
                    "rank_diff_all_xxx_race_grade_multi_max_1races",
                    "rank_diff_position_xxx_race_grade_sum_max_1races",
                    "rank_diff_all_xxx_race_grade_sum_max_1races",

                    "rank_diff_position_xxx_race_grade_multi_max_3races",
                    "rank_diff_all_xxx_race_grade_multi_max_3races",
                    "rank_diff_position_xxx_race_grade_sum_max_3races",
                    "rank_diff_all_xxx_race_grade_sum_max_3races",

                    "rank_diff_position_xxx_race_grade_multi_max_5races",
                    "rank_diff_all_xxx_race_grade_multi_max_5races",
                    "rank_diff_position_xxx_race_grade_sum_max_5races",
                    "rank_diff_all_xxx_race_grade_sum_max_5races",

                    "rank_diff_position_xxx_race_grade_multi_max_8races",
                    "rank_diff_all_xxx_race_grade_multi_max_8races",
                    "rank_diff_position_xxx_race_grade_sum_max_8races",
                    "rank_diff_all_xxx_race_grade_sum_max_8races",
        ]  # ここに100種類の列名をリストとして入れる

        # 追加する列をリストで保持
        new_columns = []

        # 計算する変数
        t1 = merged_df_all["tenkai_combined_2"] / 40
        t2 = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] / 100
        t3 = merged_df_all["umaban_curve_lcurves_condition_type_corner_combined"] / 20

        m1 = (merged_df_all["tenkai_combined_2"] + 1200) / 1200
        m2 = (merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 3000) / 3000
        m3 = (merged_df_all["umaban_curve_lcurves_condition_type_corner_combined"] + 600) / 600

        # 100種類の列との組み合わせ
        for col in columns:
            merged_df_all[f"{col}_tenkai_plus"] = merged_df_all[col] + (t1)
            merged_df_all[f"{col}_tenkai_all_plus"] = merged_df_all[col] + (t2)
            merged_df_all[f"{col}_umaban_plus"] = merged_df_all[col] + (t3)
            merged_df_all[f"{col}_umaban_tenkai_plus"] = merged_df_all[col] + (t3 + t1)
            merged_df_all[f"{col}_umaban_tenkai_all_plus"] = merged_df_all[col] + (t3 + t2)

            merged_df_all[f"{col}_tenkai_com"] = merged_df_all[col] * (m1)
            merged_df_all[f"{col}_tenkai_all_com"] = merged_df_all[col] * (m2)
            merged_df_all[f"{col}_umaban_com"] = merged_df_all[col] * (m3)
            merged_df_all[f"{col}_umaban_tenkai_com"] = merged_df_all[col] * (m3 * m1)
            merged_df_all[f"{col}_umaban_tenkai_all_com"] = merged_df_all[col] * (m3 * m2)

            new_columns.extend([
                f"{col}_tenkai_plus", f"{col}_tenkai_all_plus", f"{col}_umaban_plus",
                f"{col}_umaban_tenkai_plus", f"{col}_umaban_tenkai_all_plus",
                f"{col}_tenkai_com", f"{col}_tenkai_all_com", f"{col}_umaban_com",
                f"{col}_umaban_tenkai_com", f"{col}_umaban_tenkai_all_com"
            ])
            merged_df_all = merged_df_all.copy()

        # `race_id` ごとの統計量（平均・標準偏差）を計算
        stats = merged_df_all.groupby("race_id")[new_columns].agg(['mean', 'std']).reset_index()
        stats.columns = ['race_id'] + [f"{col}_{stat}" for col, stat in stats.columns[1:]]

        # `merged_df_all` に統計量をマージ
        merged_df_all = pd.merge(merged_df_all, stats, on='race_id', how='left')

        # 標準化
        for col in new_columns:
            merged_df_all[f"{col}_standardized"] = (
                (merged_df_all[col] - merged_df_all[f"{col}_mean"]) / merged_df_all[f"{col}_std"]
            )
            merged_df_all = merged_df_all.copy()

        # `merged_df_all` をコピー（変更が遅くなるのを防ぐ）
        merged_df_all = merged_df_all.copy()

        merged_df_all = merged_df_all.filter(regex='_standardized|race_id|date|horse_id|horse_id|rush_type_final|rush_type_Advantages|rush_advantages_cross|rush_advantages_cross_plus')

        # merged_df_all = merged_df_all.drop(columns=["goal_range_100","curve","dominant_position_category","pace_category","ground_state_level","goal_slope","course_len","place_season_condition_type_categori","start_slope","start_range","race_grade","race_type"])

        merged_df = merged_df.merge(merged_df_all, on=["race_id","date" ,"horse_id"], how="left")
        self.agg_cross2 = merged_df


        # # df["rank_diff_pace_course_len_ground_state_type_mean_1races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_pace_course_len_ground_state_type_mean_3races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_pace_course_len_ground_state_type_mean_5races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_pace_course_len_ground_state_type_mean_8races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_1races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_3races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_5races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_8races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_1races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_3races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_5races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_8races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # # df["rank_diff_correction_mean_1races_umaban"] = df["rank_diff_correction_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_mean_3races_umaban"] = df["rank_diff_correction_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_mean_5races_umaban"] = df["rank_diff_correction_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_mean_8races_umaban"] = df["rank_diff_correction_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # # df["rank_diff_correction_position_mean_1races_umaban"] = df["rank_diff_correction_position_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_mean_3races_umaban"] = df["rank_diff_correction_position_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_mean_5races_umaban"] = df["rank_diff_correction_position_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_mean_8races_umaban"] = df["rank_diff_correction_position_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # # df["rank_diff_correction_rush_mean_1races_umaban"] = df["rank_diff_correction_rush_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_rush_mean_3races_umaban"] = df["rank_diff_correction_rush_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_rush_mean_5races_umaban"] = df["rank_diff_correction_rush_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_rush_mean_8races_umaban"] = df["rank_diff_correction_rush_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # # df["rank_diff_correction_position_rush_mean_1races_umaban"] = df["rank_diff_correction_position_rush_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_rush_mean_3races_umaban"] = df["rank_diff_correction_position_rush_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_rush_mean_5races_umaban"] = df["rank_diff_correction_position_rush_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_rush_mean_8races_umaban"] = df["rank_diff_correction_position_rush_mean_8races"] * ((300 + umaban_correction_position) / 300)
        # # df = df.copy()  # 新しいコピーを作成

        # # df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races"] * ((300 + umaban_correction_position) / 300)
        # # df = df.copy()  # 新しいコピーを作成

        # # df["rank_diff_correction_position_xxx_race_grade_multi_mean_1races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_xxx_race_grade_multi_mean_3races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_xxx_race_grade_multi_mean_5races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_xxx_race_grade_multi_mean_8races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_8races"] * ((300 + umaban_correction_position) / 300)
        # # df = df.copy()  # 新しいコピーを作成

        # # df["rank_diff_correction_rush_xxx_race_grade_multi_mean_1races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_rush_xxx_race_grade_multi_mean_3races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_rush_xxx_race_grade_multi_mean_5races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_rush_xxx_race_grade_multi_mean_8races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_8races"] * ((300 + umaban_correction_position) / 300)
        # # df = df.copy()  # 新しいコピーを作成

        # # df["rank_diff_correction_xxx_race_grade_multi_mean_1races_umaban"] = df["rank_diff_correction_xxx_race_grade_multi_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_xxx_race_grade_multi_mean_3races_umaban"] = df["rank_diff_correction_xxx_race_grade_multi_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_xxx_race_grade_multi_mean_5races_umaban"] = df["rank_diff_correction_xxx_race_grade_multi_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_xxx_race_grade_multi_mean_8races_umaban"] = df["rank_diff_correction_xxx_race_grade_multi_mean_8races"] * ((300 + umaban_correction_position) / 300)
        # # df = df.copy()  # 新しいコピーを作成

        # # df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races"] * ((300 + umaban_correction_position) / 300)
        # # df = df.copy()  # 新しいコピーを作成

        # # df["rank_diff_correction_position_xxx_race_grade_sum_mean_1races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_xxx_race_grade_sum_mean_3races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_xxx_race_grade_sum_mean_5races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_position_xxx_race_grade_sum_mean_8races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # # df["rank_diff_correction_rush_xxx_race_grade_sum_mean_1races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_rush_xxx_race_grade_sum_mean_3races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_rush_xxx_race_grade_sum_mean_5races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_rush_xxx_race_grade_sum_mean_8races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # # df["rank_diff_correction_xxx_race_grade_sum_mean_1races_umaban"] = df["rank_diff_correction_xxx_race_grade_sum_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_xxx_race_grade_sum_mean_3races_umaban"] = df["rank_diff_correction_xxx_race_grade_sum_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_xxx_race_grade_sum_mean_5races_umaban"] = df["rank_diff_correction_xxx_race_grade_sum_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # # df["rank_diff_correction_xxx_race_grade_sum_mean_8races_umaban"] = df["rank_diff_correction_xxx_race_grade_sum_mean_8races"] * ((300 + umaban_correction_position) / 300)
        # umaban_correction_position_factor = (300 + umaban_correction_position) / 300
        # columns_to_process = [
        #     "rank_diff_correction_position_rush_xxx_race_grade_multi_mean",
        #     "rank_diff_correction_position_xxx_race_grade_multi_mean",
        #     "rank_diff_correction_rush_xxx_race_grade_multi_mean",
        #     "rank_diff_correction_xxx_race_grade_multi_mean",
        #     "rank_diff_correction_position_rush_xxx_race_grade_sum_mean",
        #     "rank_diff_correction_position_xxx_race_grade_sum_mean",
        #     "rank_diff_correction_rush_xxx_race_grade_sum_mean",
        #     "rank_diff_correction_xxx_race_grade_sum_mean"
        # ]

        # # 1〜8レースの異なる値を処理
        # races = [1,3,5,8]

        # # 各列に対して一括処理
        # for column_prefix in columns_to_process:
        #     for race in races:
        #         new_column_name = f"{column_prefix}_{race}races_umaban"
        #         original_column_name = f"{column_prefix}_{race}races"
                
        #         # 列を計算して新しい列を追加
        #         merged_df_all[new_column_name] = merged_df_all[original_column_name] * umaban_correction_position_factor

        # # 新しいコピーを作成
        # merged_df_all = merged_df_all.copy()




        # """
        # それぞれのrank_gradeかけ
        # """



        # # df["rank_diff_pace_course_len_ground_state_type_mean_1races_umaban"]
        # # df["rank_diff_pace_course_len_ground_state_type_mean_3races_umaban"]
        # # df["rank_diff_pace_course_len_ground_state_type_mean_5races_umaban"]
        # # df["rank_diff_pace_course_len_ground_state_type_mean_8races_umaban"]

        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_1races_umaban"]
        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_3races_umaban"]
        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_5races_umaban"]
        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_8races_umaban"]

        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_1races_umaban"]
        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_3races_umaban"]
        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_5races_umaban"]
        # # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_8races_umaban"]

        # # df["rank_diff_correction_mean_1races_umaban"]
        # # df["rank_diff_correction_mean_3races_umaban"]
        # # df["rank_diff_correction_mean_5races_umaban"]
        # # df["rank_diff_correction_mean_8races_umaban"]

        # # df["rank_diff_correction_position_mean_1races_umaban"]
        # # df["rank_diff_correction_position_mean_3races_umaban"]
        # # df["rank_diff_correction_position_mean_5races_umaban"]
        # # df["rank_diff_correction_position_mean_8races_umaban"]

        # # df["rank_diff_correction_rush_mean_1races_umaban"]
        # # df["rank_diff_correction_rush_mean_3races_umaban"]
        # # df["rank_diff_correction_rush_mean_5races_umaban"]
        # # df["rank_diff_correction_rush_mean_8races_umaban"]

        # # df["rank_diff_correction_position_rush_mean_1races_umaban"]
        # # df["rank_diff_correction_position_rush_mean_3races_umaban"]
        # # df["rank_diff_correction_position_rush_mean_5races_umaban"]
        # # df["rank_diff_correction_position_rush_mean_8races_umaban"]




        # # df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races_umaban"] + df["rush_advantages_cross"]/6
        
        # # df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_position_xxx_race_grade_multi_mean_1races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_xxx_race_grade_multi_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_1races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_position_xxx_race_grade_multi_mean_3races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_xxx_race_grade_multi_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_position_xxx_race_grade_multi_mean_5races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_xxx_race_grade_multi_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_position_xxx_race_grade_multi_mean_8races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_xxx_race_grade_multi_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_rush_xxx_race_grade_multi_mean_1races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_rush_xxx_race_grade_multi_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_1races_umaban"] + df["rush_advantages_cross"]/6
        # # df = df.copy()  # 新しいコピーを作成

        # # df["rank_diff_correction_rush_xxx_race_grade_multi_mean_3races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_rush_xxx_race_grade_multi_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_rush_xxx_race_grade_multi_mean_5races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_rush_xxx_race_grade_multi_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_rush_xxx_race_grade_multi_mean_8races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_rush_xxx_race_grade_multi_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_xxx_race_grade_multi_mean_1races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_xxx_race_grade_multi_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_1races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_xxx_race_grade_multi_mean_3races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_xxx_race_grade_multi_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_xxx_race_grade_multi_mean_5races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_xxx_race_grade_multi_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_xxx_race_grade_multi_mean_8races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_xxx_race_grade_multi_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races_umaban"] + df["rush_advantages_cross"]/6
        # # df = df.copy()  # 新しいコピーを作成

        # # df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_position_xxx_race_grade_sum_mean_1races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_xxx_race_grade_sum_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_1races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_position_xxx_race_grade_sum_mean_3races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_xxx_race_grade_sum_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_position_xxx_race_grade_sum_mean_5races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_xxx_race_grade_sum_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_position_xxx_race_grade_sum_mean_8races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_position_xxx_race_grade_sum_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_rush_xxx_race_grade_sum_mean_1races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_rush_xxx_race_grade_sum_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_1races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_rush_xxx_race_grade_sum_mean_3races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_rush_xxx_race_grade_sum_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_3races_umaban"] + df["rush_advantages_cross"]/6
        # # df = df.copy()  # 新しいコピーを作成

        # # df["rank_diff_correction_rush_xxx_race_grade_sum_mean_5races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_rush_xxx_race_grade_sum_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_rush_xxx_race_grade_sum_mean_8races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_rush_xxx_race_grade_sum_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_xxx_race_grade_sum_mean_1races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_xxx_race_grade_sum_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_1races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_xxx_race_grade_sum_mean_3races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_xxx_race_grade_sum_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_xxx_race_grade_sum_mean_5races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_xxx_race_grade_sum_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        # # df["rank_diff_correction_xxx_race_grade_sum_mean_8races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        # # df["rank_diff_correction_xxx_race_grade_sum_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        # # df = df.copy()  # 新しいコピーを作成


        # # 一度、カラム名のリストを作成
        # columns = [
        #     # "rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races_umaban",
        #     # "rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races_umaban",
        #     # "rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races_umaban",
        #     # "rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races_umaban",
        #     # "rank_diff_correction_position_xxx_race_grade_multi_mean_1races_umaban",
        #     # "rank_diff_correction_position_xxx_race_grade_multi_mean_3races_umaban",
        #     # "rank_diff_correction_position_xxx_race_grade_multi_mean_5races_umaban",
        #     # "rank_diff_correction_position_xxx_race_grade_multi_mean_8races_umaban",
        #     # "rank_diff_correction_rush_xxx_race_grade_multi_mean_1races_umaban",
        #     # "rank_diff_correction_rush_xxx_race_grade_multi_mean_3races_umaban",
        #     # "rank_diff_correction_rush_xxx_race_grade_multi_mean_5races_umaban",
        #     # "rank_diff_correction_rush_xxx_race_grade_multi_mean_8races_umaban",
        #     # "rank_diff_correction_xxx_race_grade_multi_mean_1races_umaban",
        #     # "rank_diff_correction_xxx_race_grade_multi_mean_3races_umaban",
        #     # "rank_diff_correction_xxx_race_grade_multi_mean_5races_umaban",
        #     # "rank_diff_correction_xxx_race_grade_multi_mean_8races_umaban",
        #     # "rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races_umaban",
        #     # "rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races_umaban",
        #     # "rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races_umaban",
        #     # "rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races_umaban",
        #     # "rank_diff_correction_position_xxx_race_grade_sum_mean_1races_umaban",
        #     # "rank_diff_correction_position_xxx_race_grade_sum_mean_3races_umaban",
        #     # "rank_diff_correction_position_xxx_race_grade_sum_mean_5races_umaban",
        #     # "rank_diff_correction_position_xxx_race_grade_sum_mean_8races_umaban",
        #     # "rank_diff_correction_rush_xxx_race_grade_sum_mean_1races_umaban",
        #     # "rank_diff_correction_rush_xxx_race_grade_sum_mean_3races_umaban",
        #     # "rank_diff_correction_rush_xxx_race_grade_sum_mean_5races_umaban",
        #     # "rank_diff_correction_rush_xxx_race_grade_sum_mean_8races_umaban",
        #     # "rank_diff_correction_xxx_race_grade_sum_mean_1races_umaban",
        #     # "rank_diff_correction_xxx_race_grade_sum_mean_3races_umaban",
        #     # "rank_diff_correction_xxx_race_grade_sum_mean_5races_umaban",
        #     # "rank_diff_correction_xxx_race_grade_sum_mean_8races_umaban",
        #     "rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races_umaban",
        #     "rank_diff_correction_position_rush_xxx_race_grade_multi_mean_4races_umaban",
        #     "rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races_umaban",
        #     "rank_diff_correction_position_xxx_race_grade_multi_mean_1races_umaban",
        #     "rank_diff_correction_position_xxx_race_grade_multi_mean_4races_umaban",
        #     "rank_diff_correction_position_xxx_race_grade_multi_mean_8races_umaban",
        #     "rank_diff_correction_rush_xxx_race_grade_multi_mean_1races_umaban",
        #     "rank_diff_correction_rush_xxx_race_grade_multi_mean_4races_umaban",
        #     "rank_diff_correction_rush_xxx_race_grade_multi_mean_8races_umaban",
        #     "rank_diff_correction_xxx_race_grade_multi_mean_1races_umaban",
        #     "rank_diff_correction_xxx_race_grade_multi_mean_4races_umaban",
        #     "rank_diff_correction_xxx_race_grade_multi_mean_8races_umaban",
        #     "rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races_umaban",
        #     "rank_diff_correction_position_rush_xxx_race_grade_sum_mean_4races_umaban",
        #     "rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races_umaban",
        #     "rank_diff_correction_position_xxx_race_grade_sum_mean_1races_umaban",
        #     "rank_diff_correction_position_xxx_race_grade_sum_mean_4races_umaban",
        #     "rank_diff_correction_position_xxx_race_grade_sum_mean_8races_umaban",
        #     "rank_diff_correction_rush_xxx_race_grade_sum_mean_1races_umaban",
        #     "rank_diff_correction_rush_xxx_race_grade_sum_mean_4races_umaban",
        #     "rank_diff_correction_rush_xxx_race_grade_sum_mean_8races_umaban",
        #     "rank_diff_correction_xxx_race_grade_sum_mean_1races_umaban",
        #     "rank_diff_correction_xxx_race_grade_sum_mean_4races_umaban",
        #     "rank_diff_correction_xxx_race_grade_sum_mean_8races_umaban",
        # ]

        # # それぞれのカラム名に対応する処理をまとめてループで適用
        # for col in columns:
        #     # "_and_rush"と"_plus_rush"の2つの新しいカラムを作成
        #     merged_df_all[f"{col}_and_rush"] = merged_df_all[col] * ((merged_df_all["rush_advantages_cross"] + 200) / 200)
        #     merged_df_all[f"{col}_plus_rush"] = merged_df_all[col] + merged_df_all["rush_advantages_cross"] / 6

        # merged_df_all = merged_df_all.copy()  # 新しいコピーを作成




        # # # レース数のバリエーション
        # # race_counts = [1, 3, 5, 8]

        # # # 各レース数に対応するカラムの作成
        # # for r in race_counts:
        # #     # rank_diff_correction_position_rush_xxx_race_grade_multi_mean_xraces_umaban
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)
        # #     df = df.copy()
        # #     # rank_diff_correction_position_xxx_race_grade_multi_mean_xraces_umaban
        # #     df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
        # #     df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
        # #     df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
        # #     df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

        # #     df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
        # #     df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)
        # #     df = df.copy()
        # #     # rank_diff_correction_rush_xxx_race_grade_multi_mean_xraces_umaban
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

        # #     df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)
        # #     df = df.copy()
        # #     # rank_diff_correction_xxx_race_grade_multi_mean_xraces_umaban
        # #     df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
        # #     df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
        # #     df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
        # #     df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

        # #     df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
        # #     df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)

        # #     # rank_diff_correction_position_rush_xxx_race_grade_sum_mean_xraces_umaban
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)
        # #     df = df.copy()
        # #     # rank_diff_correction_position_xxx_race_grade_sum_mean_xraces_umaban
        # #     df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
        # #     df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
        # #     df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
        # #     df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

        # #     df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
        # #     df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)

        # #     df = df.copy()
        # #     # rank_diff_correction_rush_xxx_race_grade_sum_mean_xraces_umaban
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

        # #     df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)
        # #     df = df.copy()
        # #     # rank_diff_correction_xxx_race_grade_sum_mean_xraces_umaban
        # #     df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
        # #     df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)

        # #     df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
        # #     df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

        # #     df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
        # #     df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)

        # race_counts = [1,3,5,8]
        # # レース数の対応に基づいて、必要なカラムを作成
        # def create_columns_for_race(merged_df_all, race_counts):
        #     # カラム名のパターンを定義
        #     column_patterns = [
        #         ("rank_diff_correction_position_rush", "rank_diff_correction_position_rush_xxx_race_grade_multi_mean"),
        #         ("rank_diff_correction_position", "rank_diff_correction_position_xxx_race_grade_multi_mean"),
        #         ("rank_diff_correction_rush", "rank_diff_correction_rush_xxx_race_grade_multi_mean"),
        #         ("rank_diff_correction", "rank_diff_correction_xxx_race_grade_multi_mean")
        #     ]

        #     # 各レース数に対するループ
        #     for r in race_counts:
        #         # 各パターンに対して処理
        #         for base_name, column_name in column_patterns:
        #             # 同じ操作を繰り返す部分を関数化して処理
        #             create_column_set(merged_df_all, r, base_name, column_name)

        #     return merged_df_all


        # def create_column_set(merged_df_all, r, base_name, column_name):
        #     # 共通のカラム作成処理を一度で行う
        #     merged_df_all[f"{column_name}_{r}races_umaban_and_tenkai_place"] = merged_df_all[f"{column_name}_{r}races_umaban"] * ((merged_df_all["tenkai_place_combined"] + 400) / 400)
        #     merged_df_all[f"{column_name}_{r}races_umaban_plus_tenkai_place"] = merged_df_all[f"{column_name}_{r}races_umaban"] + (merged_df_all["tenkai_place_combined"] / 60)
            
        #     merged_df_all[f"{column_name}_{r}races_umaban_and_tenkai_half"] = merged_df_all[f"{column_name}_{r}races_umaban"] * ((merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500) / 500)
        #     merged_df_all[f"{column_name}_{r}races_umaban_plus_tenkai_half"] = merged_df_all[f"{column_name}_{r}races_umaban"] + (merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] / 80)
            
        #     merged_df_all[f"{column_name}_{r}races_umaban_and_tenkai_all"] = merged_df_all[f"{column_name}_{r}races_umaban"] * ((merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600) / 600)
        #     merged_df_all[f"{column_name}_{r}races_umaban_plus_tenkai_all"] = merged_df_all[f"{column_name}_{r}races_umaban"] + (merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] / 100)


            
        # merged_df_all = merged_df_all.copy()
        # # 使用例
        # merged_df_all = create_columns_for_race(merged_df_all, race_counts)




        # """
        # 全合わせ
        # """

        # # レース数のバリエーション
        # race_counts = [1,3,5,8]

        # # # 各レース数に対応するカラムの作成
        # # for r in race_counts:
        # #     # rank_diff_correction_position_rush_xxx_race_grade_multi_mean_xraces_umaban
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6
        # #     df = df.copy()
        # #     # rank_diff_correction_position_xxx_race_grade_multi_mean_xraces_umaban
        # #     df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
        # #     df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

        # #     df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6
        # #     df = df.copy()
        # #     # rank_diff_correction_rush_xxx_race_grade_multi_mean_xraces_umaban
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

        # #     df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6
        # #     df = df.copy()
        # #     # rank_diff_correction_xxx_race_grade_multi_mean_xraces_umaban
        # #     df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
        # #     df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

        # #     df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6
        # #     df = df.copy()
        # #     # rank_diff_correction_position_rush_xxx_race_grade_sum_mean_xraces_umaban
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6
        # #     df = df.copy()
        # #     # rank_diff_correction_position_xxx_race_grade_sum_mean_xraces_umaban
        # #     df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
        # #     df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

        # #     df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6
        # #     df = df.copy()

        # #     # rank_diff_correction_rush_xxx_race_grade_sum_mean_xraces_umaban
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

        # #     df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6
        # #     df = df.copy()
        # #     # rank_diff_correction_xxx_race_grade_sum_mean_xraces_umaban
        # #     df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
        # #     df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

        # #     df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
        # #     df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6

        # #     df = df.copy()  # 新しいコピーを作成









        # # def create_column(merged_df_all, r, prefix, cols, div_values):
        # #     # 列名のパターンを生成
        # #     for col, div in zip(cols, div_values):
        # #         col_fix = f"{prefix}_all_fix"
        # #         col_sum = f"{prefix}_all_sum"
                
        # #         # 修正列の計算
        # #         merged_df_all[col_fix] = merged_df_all[f"{prefix}_umaban"] * ((merged_df_all[col] + div) / div) * ((merged_df_all["rush_advantages_cross"] + 200) / 200)
                
        # #         # 合計列の計算
        # #         merged_df_all[col_sum] = merged_df_all[f"{prefix}_umaban"] + (merged_df_all[col] / div) + merged_df_all["rush_advantages_cross"] / 6

        # #     return merged_df_all

        # # # race_counts の各値に対して処理を行う
        # # for r in race_counts:
        # #     # 各列の計算
        # #     merged_df_all = create_column(merged_df_all, r, f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races", 
        # #                     ["tenkai_place_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"], 
        # #                     [400, 500, 600])

        # #     merged_df_all = create_column(merged_df_all, r, f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races", 
        # #                     ["tenkai_place_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"], 
        # #                     [400, 500, 600])

        # #     merged_df_all = create_column(merged_df_all, r, f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races", 
        # #                     ["tenkai_place_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"], 
        # #                     [400, 500, 600])

        # #     merged_df_all = create_column(merged_df_all, r, f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races", 
        # #                     ["tenkai_place_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"], 
        # #                     [400, 500, 600])

        # #     merged_df_all = create_column(merged_df_all, r, f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races", 
        # #                     ["tenkai_place_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"], 
        # #                     [400, 500, 600])

        # #     merged_df_all = create_column(merged_df_all, r, f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races", 
        # #                     ["tenkai_place_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"], 
        # #                     [400, 500, 600])

        # #     merged_df_all = create_column(merged_df_all, r, f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races", 
        # #                     ["tenkai_place_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"], 
        # #                     [400, 500, 600])

        # #     merged_df_all = create_column(merged_df_all, r, f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races", 
        # #                     ["tenkai_place_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"], 
        # #                     [400, 500, 600])


        # # merged_df_all = merged_df_all.drop(columns=["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races",
        # #     "rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races",
        # #     "rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races",
        # #     "rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races",
        # #     "rank_diff_correction_position_xxx_race_grade_multi_mean_1races",
        # #     "rank_diff_correction_position_xxx_race_grade_multi_mean_3races",
        # #     "rank_diff_correction_position_xxx_race_grade_multi_mean_5races",
        # #     "rank_diff_correction_position_xxx_race_grade_multi_mean_8races",
        # #     "rank_diff_correction_rush_xxx_race_grade_multi_mean_1races",
        # #     "rank_diff_correction_rush_xxx_race_grade_multi_mean_3races",
        # #     "rank_diff_correction_rush_xxx_race_grade_multi_mean_5races",
        # #     "rank_diff_correction_rush_xxx_race_grade_multi_mean_8races",
        # #     "rank_diff_correction_xxx_race_grade_multi_mean_1races",
        # #     "rank_diff_correction_xxx_race_grade_multi_mean_3races",
        # #     "rank_diff_correction_xxx_race_grade_multi_mean_5races",
        # #     "rank_diff_correction_xxx_race_grade_multi_mean_8races",
        # #     "rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races",
        # #     "rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races",
        # #     "rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races",
        # #     "rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races",
        # #     "rank_diff_correction_position_xxx_race_grade_sum_mean_1races",
        # #     "rank_diff_correction_position_xxx_race_grade_sum_mean_3races",
        # #     "rank_diff_correction_position_xxx_race_grade_sum_mean_5races",
        # #     "rank_diff_correction_position_xxx_race_grade_sum_mean_8races",
        # #     "rank_diff_correction_rush_xxx_race_grade_sum_mean_1races",
        # #     "rank_diff_correction_rush_xxx_race_grade_sum_mean_3races",
        # #     "rank_diff_correction_rush_xxx_race_grade_sum_mean_5races",
        # #     "rank_diff_correction_rush_xxx_race_grade_sum_mean_8races",
        # #     "rank_diff_correction_xxx_race_grade_sum_mean_1races",
        # #     "rank_diff_correction_xxx_race_grade_sum_mean_3races",
        # #     "rank_diff_correction_xxx_race_grade_sum_mean_5races",
        # #     "rank_diff_correction_xxx_race_grade_sum_mean_8races",
        # #     ])


        # # # "1races", "3races", "5races", "8races" などを含む列名を抽出
        # # race_columns = [col for col in merged_df_all.columns if any(f"{n}races" in col for n in [1, 3, 5, 8])]

        # # # 統計量の事前計算
        # # stats = merged_df_all.groupby("race_id")[race_columns].agg(['mean', 'std']).reset_index()
        # # stats.columns = ['race_id'] + [f"{col}_{stat}" for col, stat in stats.columns[1:]]
        # # merged_df_all = merged_df_all.merge(stats, on='race_id', how='left')
        # # 事前に統計量を計算してからマージ


        # def create_column(merged_df_all, r, prefix, cols, div_values):
        #     for col, div in zip(cols, div_values):
        #         col_fix = f"{prefix}_all_fix"
        #         col_sum = f"{prefix}_all_sum"

        #         merged_df_all[col_fix] = merged_df_all[f"{prefix}_umaban"] * ((merged_df_all[col] + div) / div) * ((merged_df_all["rush_advantages_cross"] + 200) / 200)
        #         merged_df_all[col_sum] = merged_df_all[f"{prefix}_umaban"] + (merged_df_all[col] / div) + merged_df_all["rush_advantages_cross"] / 6
        #         merged_df_all = merged_df_all.copy()
        #     return merged_df_all

        # # 動的に処理を行う
        # prefixes = [
        #     "rank_diff_correction_position_rush_xxx_race_grade_multi_mean",
        #     "rank_diff_correction_position_xxx_race_grade_multi_mean",
        #     "rank_diff_correction_rush_xxx_race_grade_multi_mean",
        #     "rank_diff_correction_xxx_race_grade_multi_mean"
        # ]

        # for r in race_counts:
        #     for prefix in prefixes:
        #         merged_df_all = create_column(
        #             merged_df_all, r, f"{prefix}_{r}races",
        #             ["tenkai_place_combined", "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined", 
        #             "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"], 
        #             [400, 500, 600]
        #         )
        #         merged_df_all = merged_df_all.copy()
        # race_columns = [col for col in merged_df_all.columns if any(f"{n}races" in col for n in [1,3,5,8])]





        # stats = merged_df_all.groupby("race_id")[race_columns].agg(['mean', 'std']).reset_index()
        # stats.columns = ['race_id'] + [f"{col}_{stat}" for col, stat in stats.columns[1:]]
        # merged_df_all = pd.merge(merged_df_all, stats, on='race_id', how='left')

        # # 標準化
        # for col in race_columns:
        #     merged_df_all[f"{col}_standardized"] = (
        #         merged_df_all[col] - merged_df_all[f"{col}_mean"]
        #     ) / merged_df_all[f"{col}_std"]
        #     merged_df_all = merged_df_all.copy()
        # # '_standardized', 'race_id', 'date', 'horse_id' を除くすべての列を削除
        # merged_df_all = merged_df_all.filter(regex='_standardized|race_id|date|horse_id|horse_id|rush_type_final|rush_type_Advantages|rush_advantages_cross|rush_advantages_cross_plus')

        # # merged_df_all = merged_df_all.drop(columns=["goal_range_100","curve","dominant_position_category","pace_category","ground_state_level","goal_slope","course_len","place_season_condition_type_categori","start_slope","start_range","race_grade","race_type"])

        # merged_df = merged_df.merge(merged_df_all, on=["race_id","date" ,"horse_id"], how="left")
        # self.agg_cross2 = merged_df




    def race_type_True(
        self, n_races: list[int] = [1,  3, 5, 8]
    ) -> None:
        """
        直近nレースの馬の過去成績を距離・race_typeごとに集計し、相対値に変換する関数。
        """


        merged_df = self.population.copy()  

        def calculate_race_position_percentage(group, n_race: int):
            """
            過去nレースにおける脚質割合を計算する
            """
            # 過去nレースに絞る
            past_races = group.head(n_race)
            
            # 各脚質のカウント
            counts = past_races['race_position'].value_counts(normalize=True).to_dict()
            
            # 結果を辞書形式で返す（割合が存在しない場合は0.0を補完）
            return {
                "escape": counts.get(1, 0.0),
                "taking_lead": counts.get(2, 0.0),
                "in_front": counts.get(3, 0.0),
                "pursuit": counts.get(4, 0.0),
            }
    
        # 過去nレースのリスト
        n_race_list = [1, 3, 5, 8]

        baselog = (
            self.population.merge(
                self.horse_results,
                on=["horse_id"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )
       # 集計用データフレームの初期化
        merged_df = self.population.copy()
        
        # grouped_dfを適用して計算
        grouped_df = baselog.groupby(["race_id", "horse_id"])
        
        # 各過去nレースの割合を計算して追加
        for n_race in n_race_list:
            
            position_percentage = grouped_df.apply(
                lambda group: calculate_race_position_percentage(group, n_race=n_race),
                include_groups=False,  # グループ列を除外
            )
            
            # 結果をデータフレームとして展開
            position_percentage_df = position_percentage.apply(pd.Series).reset_index()
            position_percentage_df.rename(
                columns={
                    "escape": f"escape_{n_race}races",
                    "taking_lead": f"taking_lead_{n_race}races",
                    "in_front": f"in_front_{n_race}races",
                    "pursuit": f"pursuit_{n_race}races",
                },
                inplace=True,
            )
            
            # 結果をマージ
            merged_df = merged_df.merge(position_percentage_df, on=["race_id", "horse_id"], how="left")


        # dominant_position_category を計算
        position_columns_5 = ["escape_5races", "taking_lead_5races", "in_front_5races", "pursuit_5races"]
        position_columns_3 = ["escape_3races", "taking_lead_3races", "in_front_3races", "pursuit_3races"]
        position_columns_1 = ["escape_1races", "taking_lead_1races", "in_front_1races", "pursuit_1races"]
        
        
        def determine_dominant_position(row):
            
            # 5racesの中から計算
            if not row[position_columns_5].isnull().all():
                max_column = row[position_columns_5].idxmax()
                return {
                    "escape_5races": 1,
                    "taking_lead_5races": 2,
                    "in_front_5races": 3,
                    "pursuit_5races": 4,
                }[max_column]
            # 3racesの中から計算
            elif not row[position_columns_3].isnull().all():
                max_column = row[position_columns_3].idxmax()
                return {
                    "escape_3races": 1,
                    "taking_lead_3races": 2,
                    "in_front_3races": 3,
                    "pursuit_3races": 4,
                }[max_column]
            # 1racesの中から計算
            elif not row[position_columns_1].isnull().all():
                max_column = row[position_columns_1].idxmax()
                return {
                    "escape_1races": 1,
                    "taking_lead_1races": 2,
                    "in_front_1races": 3,
                    "pursuit_1races": 4,
                }[max_column]
            # すべて欠損の場合は1を返す
            else:
                return 3
        
        # 各行に対して dominant_position_category を適用
        merged_df["dominant_position_category"] = merged_df.apply(determine_dominant_position, axis=1)
        merged_df1 = merged_df
    
    
       
        #ペース作成
        """
        merged_df にはrace_id（レースのid）と、そのレースに出走するhorse_id(馬のid)の列があり、
        さらに dominant_position_categoryというその馬の脚質を表す列がある
        脚質は1-4の四種類の数字カテゴリが入っており
        レースごとに、出走する馬全体における、その脚質がいる割合と絶対数を求め、
        それぞれ四つの列を新たに作成
        """
        # dominant_position_category の脚質マッピング
        category_mapping_per = {1: "escape_per", 2: "taking_lead_per", 3: "in_front_per", 4: "pursuit_per"}
        category_mapping_count = {1: "escape_count", 2: "taking_lead_count", 3: "in_front_count", 4: "pursuit_count"}
        
        # 各レースで dominant_position_category の割合を計算
        position_ratios = (
            merged_df.groupby("race_id")["dominant_position_category"]
            .value_counts(normalize=True)  # 割合を計算
            .unstack(fill_value=0)         # 行: race_id, 列: dominant_position_category の形に
            .rename(columns=category_mapping_per)  # 列名を割合用に置き換え
        )
        
        # 各レースで dominant_position_category の絶対数を計算
        position_counts = (
            merged_df.groupby("race_id")["dominant_position_category"]
            .value_counts()                # 絶対数を計算
            .unstack(fill_value=0)         # 行: race_id, 列: dominant_position_category の形に
            .rename(columns=category_mapping_count)  # 列名を絶対数用に置き換え
        )
        
        # 割合と絶対数を結合
        position_data = position_ratios.join(position_counts)
        columns_to_remove = list(position_data.columns)
        merged_df = merged_df.drop(columns=columns_to_remove, errors="ignore")
        
        # 元の DataFrame にマージ
        merged_df = merged_df.merge(position_data, how="left", on="race_id")

        
        if 'pace_category' in merged_df.columns:
            merged_df = merged_df.drop(columns=['pace_category'])
        
        # ペースカテゴリを計算する関数
        def calculate_pace_category(row):
            escape_count = row.get("escape_count", 0)
            taking_lead_per = row.get("taking_lead_per", 0)
        
            if escape_count >= 3:
                return 4  # ハイペース
            elif escape_count == 2:
                if taking_lead_per > 0.5:
                    return 4  # ハイペース
                else:
                    return 3  
            elif escape_count == 1:
                if taking_lead_per > 0.6:
                    return 4  # ハイペース
                elif 0.5 < taking_lead_per <= 0.6:
                    return 3  # ハイミドルペース
                elif 0.4 < taking_lead_per <= 0.5:
                    return 2  # スローミドルペース
                else:
                    return 1  # スローペース
            elif escape_count == 0:
                # 逃げ馬が0の場合、先行馬の割合に基づく
                if taking_lead_per > 0.7:
                    return 4  # ハイペース
                elif 0.6 < taking_lead_per <= 0.7:
                    return 3  # ハイミドルペース
                elif 0.5 < taking_lead_per <= 0.6:
                    return 2  # スローミドルペース
                else:
                    return 1  # スローペース
            return None  # 予期しない場合は None
        merged_df["pace_category"] = merged_df.apply(calculate_pace_category, axis=1)
        merged_df2 = merged_df


        baselog_1 = (
            self.population.merge(
                self.race_info, on=["race_id","date"]
            )
            .merge(
                self.results[["race_id","umaban","n_horses","horse_id",'umaban_odd']], on=["race_id","horse_id"]
            )
            .merge(
                merged_df2[["horse_id", "date","race_id","pace_category","dominant_position_category"]],
                on=["race_id","date","horse_id"],
            )

        )


        baselog_1["pace_diff"] = baselog_1["pace_category"] -2.5


        #最大200前後1.1倍くらいならok
        baselog_1["course_len_pace_diff"] = baselog_1["course_len"] * ((baselog_1["pace_diff"] +40)/40)

        #グレード100前後
        baselog_1["course_len_diff_grade"] = baselog_1["course_len_pace_diff"] *  (((baselog_1['race_grade']+300)/375))

        #100前後
        baselog_1["course_len_diff_grade_slope"] = np.where(
            (baselog_1["season"] == 1) | (baselog_1["season"] == 4),
            baselog_1["course_len_diff_grade"]  *  ((baselog_1["goal_slope"] +40)/40),
            baselog_1["course_len_diff_grade"]  * ((baselog_1["goal_slope"] +55)/55)
        )

        #最初の直線の長さ、長いほどきつい、50前後くらい
        baselog_1["start_range_processed_2"] = (((baselog_1["start_range"])-360)/150)
        baselog_1["start_range_processed_2"] = baselog_1["start_range_processed_2"].apply(
            lambda x: x if x < 0 else x*0.5
        )

        baselog_1["course_len_pace_diff_grade_slope_range"] = baselog_1["course_len_diff_grade_slope"] *  ((baselog_1["start_range_processed_2"]+30)/30)

        # 条件ごとに処理を適用
        baselog_1["course_len_diff_grade_slope_range_pace"] = np.where(
            ((baselog_1["dominant_position_category"] == 1) & (baselog_1["pace_diff"] >= 0)),
            baselog_1["course_len_pace_diff_grade_slope_range"] * ((baselog_1["pace_diff"] +80)/80),
            np.where(
                (baselog_1["dominant_position_category"] == 2) & (baselog_1["pace_diff"] >= 0),
                baselog_1["course_len_pace_diff_grade_slope_range"] * ((baselog_1["pace_diff"] +200)/200),

                np.where(
                    ((baselog_1["dominant_position_category"] == 1) | (baselog_1["dominant_position_category"] == 2)) & (baselog_1["pace_diff"] < 0),
                    baselog_1["course_len_pace_diff_grade_slope_range"] / ((baselog_1["pace_diff"] +100)/100),
                    
                    np.where(
                        ((baselog_1["dominant_position_category"] == 3) | (baselog_1["dominant_position_category"] == 4))  & (baselog_1["pace_diff"] < 0),
                        baselog_1["course_len_pace_diff_grade_slope_range"]  / ((baselog_1["pace_diff"] +300)/300),
                        
                        np.where(
                            ((baselog_1["dominant_position_category"] == 3) | (baselog_1["dominant_position_category"] == 4)) & (baselog_1["pace_diff"] >= 0),
                            baselog_1["course_len_pace_diff_grade_slope_range"] * ((baselog_1["pace_diff"] +120)/120),
                            baselog_1["course_len_pace_diff_grade_slope_range"]  # どの条件にも当てはまらない場合は元の値を保持
                        )
                    )
                )
            )  
        )



        # -4.5 を行う
        baselog_1["curve_processed"] = baselog_1["curve"] - 4.5
        # +の場合は数値を8倍する
        baselog_1["curve_processed"] = baselog_1["curve_processed"].apply(
            lambda x: x * 8 if x > 0 else x
        )
        #12コーナーがきついと、ゆるい、-
        baselog_1["course_len_diff_grade_slope_range_pace_12curve"] = baselog_1["course_len_diff_grade_slope_range_pace"] * ((baselog_1["curve_processed"] + 100) / 100)

        #向正面上り坂、ゆるい、-
        baselog_1["course_len_diff_grade_slope_range_pace_12curve_front"] = baselog_1["course_len_diff_grade_slope_range_pace_12curve"] / ((baselog_1["flont_slope"] + 200) / 200)



        # #最大0.02*n
        # def calculate_course_len_pace_diff(row):
        #     if row["curve_amount"] == 0:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"]
        #     elif row["curve_amount"] <= 2:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"] * (((row["curve_R34"] + 2000) / 2100))
        #     elif row["curve_amount"] <= 3:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"] *(((row["curve_R12"] /2 + row["curve_R34"])+ 2000) / 2100)
        #     elif row["curve_amount"] <= 4:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"] * ((row["curve_R12"]+ row["curve_R34"]+ 2000) / 2100)
        #     elif row["curve_amount"] <= 5:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"]* ((row["curve_R12"] + (row["curve_R34"]*3/2)+ 2000) / 2100)
        #     elif row["curve_amount"] <= 6:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"] *((row["curve_R12"]+ (row["curve_R34"]* 2)+ 2000) / 2100)
        #     elif row["curve_amount"] <= 7:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"] * (((row["curve_R12"]* 3 / 2) + (row["curve_R34"]* 2)+ 2000) / 2100)
        #     else:  # curve_amount <= 8
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front"] * (((row["curve_R12"]* 2)+(row["curve_R34"] * 2)+ 2000) / 2100)

        # baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R"] = baselog_1.apply(calculate_course_len_pace_diff, axis=1)

        #最大0.09くらい
        baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] = baselog_1["course_len_diff_grade_slope_range_pace_12curve_front"] * ((baselog_1["height_diff"]+ 12) / 12)


        # 条件ごとに適用
        baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate"] = np.where(
            ((baselog_1["ground_state"] == 1) | (baselog_1["ground_state"] == 3)) & (baselog_1["race_type"] == 1),
            baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] + 300,

            np.where(
                (baselog_1["ground_state"] == 2) & (baselog_1["race_type"] == 1),
                baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] + 120,

                np.where(
                    ((baselog_1["ground_state"] == 1) | (baselog_1["ground_state"] == 3)) & (baselog_1["race_type"] == 0),
                    baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] - 100,

                    np.where(
                        (baselog_1["ground_state"] == 2) & (baselog_1["race_type"] == 0),
                        baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] - 50,
                        
                        # どの条件にも当てはまらない場合は元の値を保持
                        baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height"]
                    )
                )
            )
        )
        baselog_1 = baselog_1.copy()
        baselog_1.loc[:, "place_season_condition_type_categori_processed"] = (
            baselog_1["place_season_condition_type_categori"]
            .replace({5: -0.3, 4: -0.17, 3: 0, 2: 0.17,1: 0.3, -1: -0.18, -2: 0, -3: 0.18,-4:0.3,-10000:0})
        ).astype(float)

        baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place"] = (
            baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate"] * ((baselog_1["place_season_condition_type_categori_processed"]+3)/3)
            )

        #最大0.05くらい
        baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight"] = (
            baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place"] + (((baselog_1["straight_total"]/ baselog_1["course_len"])-0.5)*400)
            )


        baselog_1["course_len_allfix"] = (
            baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight"] * (((baselog_1["season_turf_condition"] - 7) + 100) / 100) 
            )

        #0,01-0.01,内がマイナス
        baselog_1["umaban_processed"] = baselog_1["umaban"].apply(
            lambda x: ((x*-1)) if x < 4 else ((x-8)/3)-1
        ).astype(float)
        #0-0.005

        baselog_1["umaban_judge"] = (baselog_1["umaban"].astype(float)/baselog_1["n_horses"].astype(float))-0.55

        #1（奇数）または 0（偶数）
        baselog_1.loc[:, "umaban_odd_processed"] = (
            (baselog_1["umaban_odd"]-1)
        ).astype(float)

        # 1600で正規化,-0.5 - 1
        baselog_1["course_len_processed"] = (baselog_1["course_len"] / 1700)-1

        # ,-1.5 - 4
        baselog_1["course_len_processed"] = baselog_1["course_len_processed"].apply(
            lambda x: x*3 if x <= 0 else x*4
        )
        baselog_1["course_len_processed_2"] = ((baselog_1["course_len_processed"] + 3)/3)

        baselog_1["umaban_processed_2"] = baselog_1["umaban_processed"] / baselog_1["course_len_processed_2"]
        baselog_1["umaban_odd_processed_2"] = baselog_1["umaban_odd_processed"] / baselog_1["course_len_processed_2"]




        baselog_1["umaban_processed_abs2"] = baselog_1["umaban_processed_2"].abs()

        baselog_1["first_corner"] = np.where(
            (baselog_1["curve_amount"] == 2) | (baselog_1["curve_amount"] == 6),
            baselog_1["curve_R34"],
            np.where(
                (baselog_1["curve_amount"] == 4) | (baselog_1["curve_amount"] == 8),
                baselog_1["curve_R12"],
                0  # それ以外のとき 0
            )
        )
        # 内が小さい,最大50くらいになってしまう
        baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight"] = np.where(
            (baselog_1["umaban_judge"] < 0),
            baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight"] /
            ((
                ((baselog_1["umaban_processed_abs2"]) # 少ないほうがtimeが増える-4.5 から3
                * (
                    baselog_1["umaban_odd_processed_2"]# 奇数不利なので分母を増やして総合を減らす 1
                        -  (baselog_1["start_point"] - 1)# 外枠が有利なので分母を増やして総合を減らす 1
                        -  baselog_1["curve_processed"]# ラストカーブきついほど数値が減る4
                        -  baselog_1["last_curve_slope"]# ラストカーブくだりほど数値が減る2
                        -  (baselog_1["season_turf_condition"] - 7)# 馬場状態が良いほど数値が減る 7-7
                        -  (baselog_1["race_type"] - 0.5)*4# 芝ほど数値が減る 2
                        +  ((baselog_1["first_corner"] - 100)/50)# 最初のコーナーがでかいほど数値が減る1
                ) 
            ) +500) / 500)
            ,
            

            np.where(
                (baselog_1["umaban_judge"] >= 0),
                baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight"] /
                ((
                    ((baselog_1["umaban_processed_abs2"]) # 少ないほうがtimeが増える-4.5 から3
                    * (
                        baselog_1["umaban_odd_processed_2"]# 奇数不利なので分母を増やして総合を減らす 1
                        +  (baselog_1["start_point"] - 1)# 外枠が有利なので分母を増やして総合を減らす 1
                        +  baselog_1["curve_processed"]# ラストカーブきついほど数値が減る4
                        +  baselog_1["last_curve_slope"]# ラストカーブくだりほど数値が減る2
                        +  (baselog_1["season_turf_condition"] - 7)# 馬場状態が良いほど数値が減る 7-7
                        + (baselog_1["race_type"] - 0.5)*4# 芝ほど数値が減る 2
                        -  ((baselog_1["first_corner"] - 100)/50)# 最初のコーナーがでかいほど数値が減る1
                    ) 
                ) + 500) / 500),

                baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight"]
            )
        )






        # #最大200前後
        # baselog_1["course_len_pace_diff_info"] = baselog_1["course_len"] * ((baselog_1["pace_diff"] +150)/150)

        # #グレード100前後
        # baselog_1["course_len_diff_grade_info"] = baselog_1["course_len_pace_diff_info"] *  (((baselog_1['race_grade']+300)/375))

        # #100前後
        # baselog_1["course_len_diff_grade_slope_info"] = np.where(
        #     (baselog_1["season"] == 1) | (baselog_1["season"] == 4),
        #     baselog_1["course_len_diff_grade_info"]  *  ((baselog_1["goal_slope"] +40)/40),
        #     baselog_1["course_len_diff_grade_info"]  * ((baselog_1["goal_slope"] +55)/55)
        # )

        # #最初の直線の長さ、長いほどきつい、50前後くらい
        # baselog_1["start_range_processed_2"] = (((baselog_1["start_range"])-360)/150)
        # baselog_1["start_range_processed_2"] = baselog_1["start_range_processed_2"].apply(
        #     lambda x: x if x < 0 else x*0.5
        # )

        # baselog_1["course_len_pace_diff_grade_slope_range_info"] = baselog_1["course_len_diff_grade_slope_info"] *  ((baselog_1["start_range_processed_2"]+30)/30)

        # # 条件ごとに処理を適用
        # baselog_1["course_len_diff_grade_slope_range_pace_info"] = np.where(
        #     ((baselog_1["dominant_position_category"] == 1) | (baselog_1["dominant_position_category"] == 2)) & (baselog_1["pace_diff"] >= 0),
        #     baselog_1["course_len_pace_diff_grade_slope_range_info"] * ((baselog_1["pace_diff"] +100)/100),
            
        #     np.where(
        #         ((baselog_1["dominant_position_category"] == 1) | (baselog_1["dominant_position_category"] == 2)) & (baselog_1["pace_diff"] < 0),
        #         baselog_1["course_len_pace_diff_grade_slope_range_info"] / ((baselog_1["pace_diff"] +120)/120),
                
        #         np.where(
        #             ((baselog_1["dominant_position_category"] == 3) | (baselog_1["dominant_position_category"] == 4))  & (baselog_1["pace_diff"] < 0),
        #             baselog_1["course_len_pace_diff_grade_slope_range_info"]  / ((baselog_1["pace_diff"] +160)/160),
                    
        #             np.where(
        #                 ((baselog_1["dominant_position_category"] == 3) | (baselog_1["dominant_position_category"] == 4)) & (baselog_1["pace_diff"] >= 0),
        #                 baselog_1["course_len_pace_diff_grade_slope_range_info"] * ((baselog_1["pace_diff"] +120)/120),
        #                 baselog_1["course_len_pace_diff_grade_slope_range_info"]  # どの条件にも当てはまらない場合は元の値を保持
        #             )
        #         )
        #     )
        # )


        # # # -4.5 を行う
        # baselog_1["curve_processed"] = baselog_1["curve"] - 4.5
        # # +の場合は数値を8倍する
        # baselog_1["curve_processed"] = baselog_1["curve_processed"].apply(
        #     lambda x: x * 8 if x > 0 else x
        # )
        # #12コーナーがきついと、ゆるい、-
        # baselog_1["course_len_diff_grade_slope_range_pace_12curve_info"] = baselog_1["course_len_diff_grade_slope_range_pace_info"] * ((baselog_1["curve_processed"] + 100) / 100)

        # #向正面上り坂、ゆるい、-
        # baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_info"] = baselog_1["course_len_diff_grade_slope_range_pace_12curve_info"] / ((baselog_1["flont_slope"] + 200) / 200)



        # #最大0.02*n
        # def calculate_course_len_pace_diff(row):
        #     if row["curve_amount"] == 0:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front_info"]
        #     elif row["curve_amount"] <= 2:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front_info"] * (((row["curve_R34"] + 2000) / 2100))
        #     elif row["curve_amount"] <= 3:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front_info"] *(((row["curve_R12"] /2 + row["curve_R34"])+ 2000) / 2100)
        #     elif row["curve_amount"] <= 4:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front_info"] * ((row["curve_R12"]+ row["curve_R34"]+ 2000) / 2100)
        #     elif row["curve_amount"] <= 5:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front_info"]* ((row["curve_R12"] + (row["curve_R34"]*3/2)+ 2000) / 2100)
        #     elif row["curve_amount"] <= 6:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front_info"] *((row["curve_R12"]+ (row["curve_R34"]* 2)+ 2000) / 2100)
        #     elif row["curve_amount"] <= 7:
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front_info"] * (((row["curve_R12"]* 3 / 2) + (row["curve_R34"]* 2)+ 2000) / 2100)
        #     else:  # curve_amount <= 8
        #         return row["course_len_diff_grade_slope_range_pace_12curve_front_info"] * (((row["curve_R12"]* 2)+(row["curve_R34"] * 2)+ 2000) / 2100)

        # baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_info"] = baselog_1.apply(calculate_course_len_pace_diff, axis=1)

        # #最大0.09くらい
        # baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_info"] = baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_info"] * ((baselog_1["height_diff"]+ 15) / 15)


        # # 条件ごとに適用
        # baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_info"] = np.where(
        #     ((baselog_1["ground_state"] == 1) | (baselog_1["ground_state"] == 3)) & (baselog_1["race_type"] == 1),
        #     baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_info"] + 250,

        #     np.where(
        #         (baselog_1["ground_state"] == 2) & (baselog_1["race_type"] == 1),
        #         baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_info"] + 120,

        #         np.where(
        #             ((baselog_1["ground_state"] == 1) | (baselog_1["ground_state"] == 3)) & (baselog_1["race_type"] == 0),
        #             baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_info"] - 80,

        #             np.where(
        #                 (baselog_1["ground_state"] == 2) & (baselog_1["race_type"] == 0),
        #                 baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_info"] - 40,
                        
        #                 # どの条件にも当てはまらない場合は元の値を保持
        #                 baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_info"]
        #             )
        #         )
        #     )
        # )

        # baselog_1.loc[:, "place_season_condition_type_categori_processed"] = (
        #     baselog_1["place_season_condition_type_categori"]
        #     .replace({5: -0.3, 4: -0.17, 3: 0, 2: 0.17,1: 0.3, -1: -0.18, -2: 0, -3: 0.18,-4:0.3,-10000:0})
        # ).astype(float)


        # baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_info"] = (
        #     baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_info"] * ((baselog_1["place_season_condition_type_categori_processed"]+5)/5)
        #     )

        # #最大0.05くらい
        # baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight_info"] = (
        #     baselog_1["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_info"] + (((baselog_1["straight_total"]/ baselog_1["course_len"])-0.5)*400)
        #     )




        baselog_2 = (
            baselog_1.merge(
                self.horse_results,
                on=["horse_id"],
                suffixes=("", "_horse"),
            )
            .query("date_horse < date")
            .sort_values("date_horse", ascending=False)
        )




        merged_df = self.population.copy()  


        # baselog_2["course_len_diff_pace_diff"] =baselog_2["course_len_pace_diff"] -  baselog_2["course_len"]
        # baselog_2["course_len_diff_grade_diff"] = baselog_2["course_len_diff_grade"] - baselog_2["course_len"]
        # # baselog_2["course_len_diff_grade_slope_diff"] =baselog_2["course_len_diff_grade_slope"] -  baselog_2["course_len"]
        # baselog_2["course_len_diff_grade_slope_range_pace_diff"] = baselog_2["course_len_diff_grade_slope_range_pace"] - baselog_2["course_len"]
        # baselog_2["course_len_diff_grade_slope_range_pace_12curve_diff"] = baselog_2["course_len_diff_grade_slope_range_pace_12curve"] - baselog_2["course_len"]
        # # baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_diff"] = baselog_2["course_len_diff_grade_slope_range_pace_12curve_front"] - baselog_2["course_len"]
        # # baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_diff"] = baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R"] - baselog_2["course_len"]
        # # baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_diff"] =baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] -  baselog_2["course_len"]
        # baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_diff"] = baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate"] - baselog_2["course_len"]
        # baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_diff"] = baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place"] - baselog_2["course_len"]
        # baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight_diff"] = baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight"] - baselog_2["course_len"]



        # baselog_2["course_len_diff_pace_diff_info"] = baselog_2["course_len_pace_diff"] - baselog_2["course_len_pace_diff_info"]
        # baselog_2["course_len_diff_grade_diff_info"] = baselog_2["course_len_diff_grade"] -baselog_2["course_len_diff_grade_info"] 
        # baselog_2["course_len_diff_grade_slope_range_pace_diff_info"] = baselog_2["course_len_diff_grade_slope_range_pace"] - baselog_2["course_len_diff_grade_slope_range_pace_info"]
        # baselog_2["course_len_diff_grade_slope_range_pace_12curve_diff_info"] = baselog_2["course_len_diff_grade_slope_range_pace_12curve"] - baselog_2["course_len_diff_grade_slope_range_pace_12curve_info"]

        # baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_diff_info"] = baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate"] - baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_info"]
        # baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_diff_info"] = baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place"] - baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_info"]
        # baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight_diff_info"] = baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight"] - baselog_2["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight_info"]
        baselog_2["course_len_allfix_diff"] = baselog_2["course_len_allfix"] - baselog_2["course_len_allfix_horse"]
        baselog_2["course_len_allfix_normal_diff"] = baselog_2["course_len_allfix"] - baselog_2["course_len_horse"]


        n_race_list = [1, 3,5, 8]
        grouped_df = baselog_2.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        for n_race in tqdm(n_races, desc="agg_race_type_True"):
            df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [
                        "nobori_pace_diff_slope_range_groundstate_position",
                        "nobori_pace_diff_slope_range_groundstate_position_umaban",
                        "course_len_allfix_diff",
                        "course_len_allfix_normal_diff",
                        "course_len_allfix_horse",
                    ]
                ]
                .agg(["mean", "max","min"])
            )
            df.columns = [
                "_".join(col) + f"_{n_race}races" for col in df.columns
            ]
            original_df = df.copy()
            # レースごとの相対値に変換
            tmp_df = df.groupby(["race_id"])
            relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            merged_df = merged_df.merge(
                relative_df, on=["race_id", "horse_id"], how="left"
            )
            merged_df = merged_df.merge(original_df, on=["race_id", "horse_id"], how="left")

        

        



        """
        noboriと
        nobori修正
        **これ、の交互特徴量
        上がりが早いほうが有利なのは

        直線が長い
        上り坂
        最終コーナーがゆるい、早くこれる
        """




        merged_df = merged_df.merge(
            self.race_info[["race_id", "goal_range","goal_slope","curve"]], on="race_id"
        )

        merged_df["cross_nobori_range"] = merged_df["nobori_pace_diff_slope_range_groundstate_position_umaban_min_3races"] * (20/(((merged_df["goal_range"]-360)/90)+20))
        merged_df["cross_nobori_slope"] = merged_df["nobori_pace_diff_slope_range_groundstate_position_umaban_min_3races"] * (40/(merged_df["goal_slope"]+40))
        merged_df["cross_nobori_corner"] = merged_df["nobori_pace_diff_slope_range_groundstate_position_umaban_min_3races"] * (30/((merged_df["curve"]-3)+30))
        merged_df["cross_nobori_all"] = merged_df["nobori_pace_diff_slope_range_groundstate_position_umaban_min_3races"] * (30/((merged_df["curve"]-3)+30))* (40/(merged_df["goal_slope"]+40))* ((20/((merged_df["goal_range"]-360)/90)+20))
        merged_df = merged_df.drop(columns=["goal_range", "goal_slope", "curve"])


        """
        タフは距離短縮
        ２種類作る
        ＋がタフ
        """
        merged_df = (
            merged_df
            .merge(self.all_position[["race_id","date","horse_id","rush_type_Advantages"]], on=["race_id","date","horse_id"])
                )



        merged_df["course_len_shorter"] = (merged_df["course_len_allfix_diff_mean_3races"]/10) * merged_df["rush_type_Advantages"]
        merged_df = merged_df.drop(columns=["rush_type_Advantages"])


        # 統計量の事前計算
        combined_columns = ["cross_nobori_range", "cross_nobori_slope", "cross_nobori_corner", "cross_nobori_all", "course_len_shorter"]

        merged_df_course = merged_df.copy()

        # race_id ごとに統計量を計算
        stats = merged_df.groupby("race_id")[combined_columns].agg(['mean', 'std']).reset_index()

        # 新しい列名を設定
        stats.columns = ['race_id'] + [f"{col}_{stat}" for col, stat in stats.columns[1:]]

        # 統計量を元のデータにマージ
        merged_df = merged_df.merge(stats, on='race_id', how='left')

        # 標準化の実行
        for col in combined_columns:
            merged_df[f"{col}_standardized"] = (
                merged_df[col] - merged_df[f"{col}_mean"]
            ) / merged_df[f"{col}_std"]

        merged_df_course = merged_df_course.merge(
            merged_df[["race_id","date","horse_id","cross_nobori_range_standardized","cross_nobori_slope_standardized",	"cross_nobori_corner_standardized",	"cross_nobori_all_standardized",	"course_len_shorter_standardized"]], 
            on=["race_id","date","horse_id"]
        )
        self.race_type_True_df = merged_df_course



    

    def create_features(self) -> pd.DataFrame:
        """
        特徴量作成処理を実行し、populationテーブルに全ての特徴量を結合する。
        """
        # 馬の過去成績集計
        self.create_baselog()
        # self.create_race_grade()
        self.agg_horse_n_races()
        # self.agg_horse_n_races_relative()
        self.agg_course_len()
        self.results_relative()

        self.cross_rank_diff()
        self.cross_time()
        self.cross2()
        self.race_type_True()
        
        # self.agg_jockey()
        # self.agg_trainer()
        self.agg_horse_per_course_len()
        self.cross_features()
        self.agg_interval()  
        self.cross_features_2()
        self.cross_features_3()
        self.cross_features_4()

        self.cross_features_5()
        self.cross_features_6()
        self.cross_features_7()
        self.cross_features_8()
        self.cross_features_9()
        self.cross_features_10()
        self.cross_features_11()
        self.cross_features_12()
        self.cross_features_13()
        self.cross_features_14()
        self.cross_features_15()       
        # self.cross_features_16()  
        self.cross_features_17()  
        self.position_results()
        self.dirt_weight_weather()
        self.umaban_good()


        self.agg_horse_per_group_cols(
            group_cols=["ground_state", "race_type"], df_name="ground_state_race_type"
        )
        # self.agg_horse_per_group_cols(
        #     group_cols=["race_grade"], df_name="race_grade"
        # )
        # self.agg_horse_per_group_cols(
        #     group_cols=["around","wakuban"], df_name="around_per_wakuban"
        # )
        self.agg_horse_per_group_cols(
            group_cols=["race_type"], df_name="race_type"
        )
        self.agg_horse_per_group_cols(
            group_cols=["place","course_len", "race_type"], df_name="race_place_len"
        )
        # self.agg_horse_per_group_cols(
        #     group_cols=["place", "race_type"], df_name="race_place"
        # )    

        # self.agg_horse_per_group_cols(
        #     group_cols=["weather"], df_name="weather"
        # )
                
        # self.agg_sire()
        # self.agg_bms()  
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
            # .merge(
            #     self.agg_horse_n_races_relative_df,
            #     on=["race_id", "date", "horse_id"],
            #     how="left",
            #     # copy=False,
            # )
            .merge(
                self.course_len_df,
                on=["race_id","date","horse_id"],
                how="left",
                # copy=False,
            )     
            .merge(
                self.results_relative_df,
                on=["race_id","date","horse_id"],
                how="left",
                # copy=False,
            )                 
            # .merge(
            #     self.agg_jockey_df,
            #     on=["race_id", "horse_id"],
            #     how="left",
            #     # copy=False,
            # )
            # .merge(
            #     self.agg_trainer_df,
            #     on=["race_id", "horse_id"],
            #     how="left",
            #     # copy=False,
            # )
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
            # .merge(
            #     self.agg_horse_per_group_cols_dfs["race_grade"],
            #     on=["race_id", "date", "horse_id"],
            #     how="left",
            #     # copy=False,
            # )
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
            .merge(
                self.agg_cross_features_df_2,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )    
            .merge(
                self.agg_cross_features_df_3,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )   
            .merge(
                self.agg_cross_features_df_4,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )    
            .merge(
                self.agg_cross_features_df_5,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )    
            .merge(
                self.agg_cross_features_df_6,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )    
            .merge(
                self.agg_cross_features_df_7,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )    
            .merge(
                self.agg_cross_features_df_8,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )    
            .merge(
                self.agg_cross_features_df_9,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )    
            .merge(
                self.agg_cross_features_df_10,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )    
            .merge(
                self.agg_cross_features_df_11,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )    
            .merge(
                self.agg_cross_features_df_12,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )    
            .merge(
                self.agg_cross_features_df_13,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )    
            .merge(
                self.agg_cross_features_df_14,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )    
            .merge(
                self.agg_cross_features_df_15,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )     
            # .merge(
            #     self.agg_cross_features_df_16,
            #     on=["race_id", "date", "horse_id"],
            #     how="left",
            #     # copy=False,
            # )         
            .merge(
                self.agg_cross_features_df_17,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )                 
            .merge(
                self.agg_position_results,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )   
            .merge(
                self.agg_weight_weather,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )   
            .merge(
                self.agg_umaban_good,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )                         
            .merge(
                self.agg_rank_diff,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_cross2,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )       
            .merge(
                self.race_type_True_df,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )       
            .merge(
                self.agg_cross_time,
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
            #     self.agg_horse_per_group_cols_dfs["race_place"],
            #     on=["race_id", "date", "horse_id"],
            #     how="left",
            # )
            # .merge(
            #     self.agg_horse_per_group_cols_dfs["weather"],
            #     on=["race_id", "date", "horse_id"],
            #     how="left",
            #     # copy=False,
            # )                     
            # .merge(
            #     self.agg_sire_df,
            #     on=["race_id", "horse_id"],
            #     how="left",
            #     # copy=False,
            # )
            # .merge(
            #     self.agg_bms_df,
            #     on=["race_id", "horse_id"],
            #     how="left",
            #     # copy=False,
            # )            
        )
        # features.drop(columns=['place_season_condition_type_categori_x'], inplace=True)        
        features.to_csv(self.output_dir / self.output_filename, sep="\t", index=False)
        print("merging all features...comp")
        return features
