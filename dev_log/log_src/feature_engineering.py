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

    def create_race_grade(self):
        """
        horse_resultsをレース結果テーブルの日付よりも過去に絞り、集計元のログを作成。
        """
        df = (
            self.results[["race_id", "horse_id","mean_age_kirisute"]]
            .merge(self.race_info_before[["race_id", "race_type", "season","race_class"]], on="race_id")
        )
        self.race_info = self.race_info_before.copy()


        """

        dfに新たな列、race_gradeを作成して欲しい
        作成ルールは以下の通りである
        'age_season'の条件に引っかかった場合、それを優先すること
        次点で"race_class"の条件にかかっても、'age_season'がある方を優先して変換すること


        "race_class"列が0は55

        "race_class"列が1は60

        "race_class"列が2は70
        2歳それ以外は68（20<='age_season'<30かつ、2<="race_class"列<5の行）
        2歳G2,G3,OPは73（20<='age_season'<30かつ、5<="race_class"列<8の行）

        "race_class"列が3は79
        2歳G1は79（20<='age_season'<30かつ、8<="race_class"の行）
        3歳春OPは80（30<='age_season'<33かつ、4<="race_class"列<6の行）
        3歳春G2.G3は81（30<='age_season'<33かつ、6<="race_class"列<8の行）

        "race_class"列が4は85
        3歳春G1は86（30<='age_season'<33かつ、8<="race_class"の行）
        3歳秋G2,G3は86（33<='age_season'<40かつ、5<="race_class"列<8の行）

        "race_class"列が5は89
        3歳秋G1は91（33<='age_season'<40かつ、8<="race_class"の行）

        "race_class"列が6は92

        "race_class"列が7は94

        "race_class"列が8は98




        これらを小さく（1/10 - 5）した列

        G1 8	100
        G2 7	95
        G3 6	92
        オープン5	89
        1600万4	86
        ２勝クラス3	80
        １勝クラス2	70
        未勝利1	60
        新馬0	55


        クラス	芝	ダート
        未勝利	６５（-１５）	６０（-２０）
        500万下
        Ｇ１を除く２歳ＯＰ	７５（-５）	７２（-８）
        1000万下
        ２歳Ｇ１
        Ｇ１を除く３歳春ＯＰ	８３（３）	８３（３）
        1600万下
        ３歳春Ｇ１
        ３歳秋重賞	８８（８）	９０（１０）
        ＯＰ（ただしダート重賞を除く）
        ３歳秋Ｇ１	９３（１３）	９５（１５）
        ダート重賞（３歳を除く）	－	１００（２０）
        古馬Ｇ１	９８（１８）	１０５（２５）
        """
        # "mean_age_kirisute"と"season"を文字列に変換して結合し、int型に変換して新しい列 "age_season" を作成
        df['age_season'] = (df['mean_age_kirisute'].astype(str) + df['season'].astype(str)).astype(int)

        # race_gradeの作成
        def calculate_race_grade(row):
            age_season = row['age_season']
            race_class = row['race_class']

            # 'age_season' に基づく条件を優先してチェック
            if 20 <= age_season < 30:
                if 2 <= race_class < 5:
                    return 70
                elif 5 <= race_class < 8:
                    return 70
                elif 8 <= race_class:
                    return 79
            elif 30 <= age_season < 33:
                if 4 <= race_class < 6:
                    return 79
                elif 6 <= race_class < 8:
                    return 79
                elif 8 <= race_class:
                    return 85
            elif 33 <= age_season < 40:
                if 5 <= race_class < 8:
                    return 85
                elif 8 <= race_class:
                    return 91
            
            if race_class == 0:
                return 55
            elif race_class == 1:
                return 60
            elif race_class == 2:
                return 70
            elif race_class == 3:
                return 79
            elif race_class == 4:
                return 85
            elif race_class == 5:
                return 89
            elif race_class == 6:
                return 91
            elif race_class == 7:
                return 94
            elif race_class == 8:
                return 98
            else:
                return np.nan  

        # race_grade列を作成
        df['race_grade'] = df.apply(calculate_race_grade, axis=1)

        #race_grade_scaledの作成
        df['race_grade_scaled'] = df['race_grade'] / 10 - 5

        self.race_info[['age_season', 'race_grade', 'race_grade_scaled']] = df[['age_season', 'race_grade', 'race_grade_scaled']]
        self.race_info.to_csv(self.output_dir_info/self.output_filename_info, sep="\t", index=False)



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



    
    def agg_horse_n_races_relative(
        self, n_races: list[int] = [1, 3, 5, 10]
    ) -> None:
        """
        直近nレースの平均を集計して標準化した関数。
        """
        grouped_df = self.baselog.groupby(["race_id", "horse_id"])
        merged_df = self.population.copy()
        for n_race in tqdm(n_races, desc="agg_horse_n_races_relative"):
            df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [
                        "rank",
                        "rank_per_horse",
                        
                        "prize",
                        "rank_diff",
                        # "course_len",
                        "race_grade",
                        #"time",
                        
                        "nobori",
                        # "n_horses",       
                        
                        # "corner_1",
                        # "corner_2",
                        # "corner_3",
                        # "corner_4",
                        "corner_1_per_horse",
                        "corner_2_per_horse",
                        "corner_3_per_horse",
                        "corner_4_per_horse",                        
                        
                        "pace_1",
                        "pace_2",
                    ]
                ]
                .agg(["mean", "max", "min"])
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





        merged_df = merged_df.merge(
            results_normal[["race_id", "horse_id","umaban_relative", "wakuban_relative", "impost_relative", "age_relative", "weight_relative", "weight_diff_relative", "impost_percent_relative"]], 
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
                        "rank",
                        "rank_per_horse",
                        "prize",
                        "rank_diff",
                        "time",
                        "time_courselen",
                        "nobori",
                        # "corner_1",
                        # "corner_2",
                        # "corner_3",
                        # "corner_4",
                        "corner_1_per_horse",
                        "corner_2_per_horse",
                        "corner_3_per_horse",
                        "corner_4_per_horse",                        
                        "pace_1",
                        "pace_2",                        
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
                        "rank_per_horse",
                        "prize",
                        "rank_diff",
                        "time",

                        "nobori",
                        # "corner_1",
                        # "corner_2",
                        # "corner_3",
                        "corner_1_per_horse",
                        "corner_2_per_horse",
                        "corner_3_per_horse",
                        "corner_4_per_horse",               
                        
                        "pace_1",
                        "pace_2",                        
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

    def agg_bms(self):
        """
        bmsの過去成績を紐付け、相対値に変換する関数。
        """
        print("running agg_bms()...")
        df = self.population.merge(
            self.peds[["horse_id", "bms_id"]],
            on="horse_id",
        ).merge(
            self.race_info[["race_id", "race_type", "course_len"]],
        )
        df["year"] = pd.to_datetime(df["date"]).dt.year - 1
        df = df.merge(
            self.bms_leading,
            on=["bms_id", "year", "race_type"],
            suffixes=("", "_bms"),
        ).set_index(["race_id", "horse_id"])
        df["course_len_diff"] = df["course_len"] - df["course_len_bms"]
        df = df[["n_races", "n_wins", "winrate", "course_len_diff"]].add_prefix("bms_")
        # レースごとの相対値に変換
        tmp_df = df.groupby(["race_id"])
        relative_df = ((df - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")

        # relative_df = relative_df.astype({col: 'float32' for col in relative_df.select_dtypes('float64').columns})
        # relative_df = relative_df.astype({col: 'int32' for col in relative_df.select_dtypes('int64').columns})        
        
        self.agg_bms_df = relative_df
        print("running agg_bms()...comp")
        
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

    def cross_features_2(self):
        """
        カテゴリ追加、過去成績なし
        """
        merged_df = self.population.copy()        
        df = (
            merged_df
            .merge(self.results[["race_id", "horse_id", "wakuban", "umaban", "sex"]], on=["race_id", "horse_id"])
            .merge(self.race_info[["race_id", "place","race_grade","around","weather","ground_state","course_len","race_type"]], on="race_id")
        )

        
        
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
        # 距離/競馬場/タイプ/枠順/直線
        df["distance_place_type_wakuban_straight"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["wakuban"].astype(str) + df["around"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/直線
        df["distance_place_type_umaban_straight"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["umaban"].astype(str) + df["around"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/馬場状態
        df["distance_place_type_wakuban_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["wakuban"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/馬場状態
        df["distance_place_type_umaban_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["umaban"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/馬場状態/直線
        df["distance_place_type_wakuban_ground_state_straight"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["wakuban"].astype(str) + df["ground_state"].astype(str) + df["around"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/馬場状態/直線
        df["distance_place_type_umaban_ground_state_straight"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["umaban"].astype(str) + df["ground_state"].astype(str) + df["around"].astype(str)).astype(int)
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
        # 距離/タイプ/性別
        df["distance_type_sex"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/性別
        df["distance_place_type_sex"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/性別
        df["distance_place_type_race_grade_sex"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["race_grade"].astype(str) + df["sex"].astype(str)).astype(int)
        # 距離/タイプ/性別/天気
        df["distance_type_sex_weather"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str) + df["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/性別/天気
        df["distance_place_type_sex_weather"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str) + df["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/性別/天気
        df["distance_place_type_race_grade_sex_weather"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["race_grade"].astype(str) + df["sex"].astype(str) + df["weather"].astype(str)).astype(int)
        # 距離/タイプ/性別/馬場状態
        df["distance_type_sex_ground_state"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/性別/馬場状態
        df["distance_place_type_sex_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["sex"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/性別/馬場状態
        df["distance_place_type_race_grade_sex_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["race_grade"].astype(str) + df["sex"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/タイプ/直線
        df["distance_type_straight"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["around"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/直線
        df["distance_place_type_straight"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["around"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/直線
        df["distance_place_type_race_grade_straight"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["race_grade"].astype(str) + df["around"].astype(str)).astype(int)
        # 距離/タイプ/直線/馬場状態
        df["distance_type_straight_ground_state"] = (df["course_len"].astype(str) + df["race_type"].astype(str) + df["around"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/直線/馬場状態
        df["distance_place_type_straight_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["around"].astype(str) + df["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/直線/馬場状態
        df["distance_place_type_race_grade_straight_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["race_grade"].astype(str) + df["around"].astype(str) + df["ground_state"].astype(str)).astype(int)

        # 距離/競馬場/タイプ/馬番/レースランク/直線/天気/馬場状態
        df["distance_place_type_umaban_race_grade_around_weather_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["umaban"].astype(str) + df["race_grade"].astype(str)+ df["around"].astype(str) + df["ground_state"].astype(str)+ df["weather"].astype(str)).astype(int)        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df["distance_place_type_race_grade_around_weather_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str)  + df["race_grade"].astype(str)+ df["around"].astype(str) + df["ground_state"].astype(str)+ df["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬場状態/天気
        df["distance_place_type_ground_state_weather"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["ground_state"].astype(str)+ df["weather"].astype(str)).astype(int)



                
        baselog = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type", "race_grade","ground_state","weather","place","around"]], on="race_id"
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
        # 距離/競馬場/タイプ/枠順/直線
        df_old["distance_place_type_wakuban_straight"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["wakuban"].astype(str) + df_old["around"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/直線
        df_old["distance_place_type_umaban_straight"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["umaban"].astype(str) + df_old["around"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/馬場状態
        df_old["distance_place_type_wakuban_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["wakuban"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/馬場状態
        df_old["distance_place_type_umaban_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["umaban"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/枠順/馬場状態/直線
        df_old["distance_place_type_wakuban_ground_state_straight"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["wakuban"].astype(str) + df_old["ground_state"].astype(str) + df_old["around"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/馬番/馬場状態/直線
        df_old["distance_place_type_umaban_ground_state_straight"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["umaban"].astype(str) + df_old["ground_state"].astype(str) + df_old["around"].astype(str)).astype(int)
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
        # 距離/タイプ/性別
        df_old["distance_type_sex"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/性別
        df_old["distance_place_type_sex"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/性別
        df_old["distance_place_type_race_grade_sex"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str) + df_old["sex"].astype(str)).astype(int)
        # 距離/タイプ/性別/天気
        df_old["distance_type_sex_weather"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/性別/天気
        df_old["distance_place_type_sex_weather"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/性別/天気
        df_old["distance_place_type_race_grade_sex_weather"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str) + df_old["sex"].astype(str) + df_old["weather"].astype(str)).astype(int)
        # 距離/タイプ/性別/馬場状態
        df_old["distance_type_sex_ground_state"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/性別/馬場状態
        df_old["distance_place_type_sex_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["sex"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/性別/馬場状態
        df_old["distance_place_type_race_grade_sex_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str) + df_old["sex"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/タイプ/直線
        df_old["distance_type_straight"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["around"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/直線
        df_old["distance_place_type_straight"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["around"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/直線
        df_old["distance_place_type_race_grade_straight"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str) + df_old["around"].astype(str)).astype(int)
        # 距離/タイプ/直線/馬場状態
        df_old["distance_type_straight_ground_state"] = (df_old["course_len"].astype(str) + df_old["race_type"].astype(str) + df_old["around"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/直線/馬場状態
        df_old["distance_place_type_straight_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["around"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        # 距離/競馬場/タイプ/レースランク/直線/馬場状態
        df_old["distance_place_type_race_grade_straight_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str) + df_old["around"].astype(str) + df_old["ground_state"].astype(str)).astype(int)
        

        # 距離/競馬場/タイプ/馬番/レースランク/直線/天気/馬場状態
        df_old["distance_place_type_umaban_race_grade_around_weather_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["umaban"].astype(str) + df_old["race_grade"].astype(str)+ df_old["around"].astype(str) + df_old["ground_state"].astype(str)+ df_old["weather"].astype(str)).astype(int)        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_race_grade_around_weather_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str)  + df_old["race_grade"].astype(str)+ df_old["around"].astype(str) + df_old["ground_state"].astype(str)+ df_old["weather"].astype(str)).astype(int)    

        
        # 距離/競馬場/タイプ/馬場状態/天気
        df_old["distance_place_type_ground_state_weather"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["ground_state"].astype(str)+ df_old["weather"].astype(str)).astype(int)


        df_old["target_rank"] = (df_old["rank"] <= 3).astype(int)
        df2 = df_old[["distance_place_type_umaban_ground_state_straight","distance_place_type_wakuban_ground_state_straight", "target_rank"]].dropna().astype(int)
        # グループごとのカウントを作成
        group_counts = df2.groupby("distance_place_type_umaban_ground_state_straight").size()
        # 100未満のグループを除外
        valid_groups = group_counts[group_counts >= 10].index
        # 100以上のグループのみを使用して、平均複勝率を計算
        df2_filtered = df2[df2["distance_place_type_umaban_ground_state_straight"].isin(valid_groups)]
        # 平均複勝率を計算
        mean_fukusho_rate = df2_filtered.groupby("distance_place_type_umaban_ground_state_straight")["target_rank"].transform("mean")
        # 計算した平均複勝率を元のDataFrameに追加
        df_old["mean_fukusho_rate_umaban"] = mean_fukusho_rate
        
        # グループごとのカウントを作成
        group_counts = df2.groupby("distance_place_type_wakuban_ground_state_straight").size()
        # 100未満のグループを除外
        valid_groups = group_counts[group_counts >= 10].index
        # 100以上のグループのみを使用して、平均複勝率を計算
        df2_filtered = df2[df2["distance_place_type_wakuban_ground_state_straight"].isin(valid_groups)]
        # 平均複勝率を計算
        mean_fukusho_rate = df2_filtered.groupby("distance_place_type_wakuban_ground_state_straight")["target_rank"].transform("mean")
        # 計算した平均複勝率を元のDataFrameに追加
        df_old["mean_fukusho_rate_wakuban"] = mean_fukusho_rate



        
        
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
            "distance_place_type_wakuban_ground_state_straight"
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

            ('distance_place_type_umaban_ground_state_straight_copy', "mean_fukusho_rate_umaban"), 
            ('distance_place_type_wakuban_ground_state_straight_copy', "mean_fukusho_rate_wakuban"), 
        ]
    
        # 各ペアを順番に処理
        for original_col, encoded_col in columns_to_merge:
            df2_subset = df_old[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
            df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
            df = df.merge(df2_subset, on=original_col, how='left')  # dfにマージ
    

        merged_df = merged_df.merge(
            df[[
                "race_id", "horse_id", "date", "distance_type", "distance_place_type", 
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
                "distance_type_straight_ground_state", "distance_place_type_straight_ground_state", 
                "distance_place_type_race_grade_straight_ground_state",
                "distance_place_type_race_grade_around_weather_ground_state",
                "distance_place_type_umaban_race_grade_around_weather_ground_state",
                "distance_place_type_ground_state_weather",
                
                
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
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "nobori","time","umaban"]], on=["race_id", "horse_id"])
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
                        # 'distance_place_type_wakuban_ground_state_encoded_rank_diff_sumprod', 
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
                        "distance_place_type_race_grade_nobori_encoded_rank_diff_sumprod",
                        # "distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded_rank_diff_multiprod",
                        "distance_place_type_race_grade_nobori_encoded_rank_diff_multiprod"
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
                return 2
        
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
                if taking_lead_per > 0.7:
                    return 4  # ハイペース
                else:
                    return 3  
            elif escape_count == 1:
                if taking_lead_per > 0.75:
                    return 4  # ハイペース
                elif 0.65 < taking_lead_per <= 0.75:
                    return 3  # ハイミドルペース
                elif 0.55 < taking_lead_per <= 0.65:
                    return 2  # スローミドルペース
                else:
                    return 1  # スローペース
            elif escape_count == 0:
                # 逃げ馬が0の場合、先行馬の割合に基づく
                if taking_lead_per > 0.88:
                    return 4  # ハイペース
                elif 0.75 < taking_lead_per <= 0.88:
                    return 3  # ハイミドルペース
                elif 0.65 < taking_lead_per <= 0.75:
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
            .merge(self.race_info[["race_id", "place","weather","ground_state","course_len","race_type","race_date_day_count"]], on="race_id")
        )
        df["place"] = df["place"].astype(int)
        
        
        
        #predictの時にはold_poplationに変える
        old_merged_df = self.population.copy()      
        
        #ここで、直近レースのtimeを知りたい
        df_old2 = (
            old_merged_df
            .merge(self.results[["race_id", "horse_id","time","rank","rank_per_horse"]], on=["race_id", "horse_id"])
            .merge(self.race_info[["race_id", "place","weather","ground_state","race_grade","course_len","race_type","race_date_day_count"]], on="race_id")
        )
        df_old2["place"] = df_old2["place"].astype(int)
        df_old2["race_grade"] = df_old2["race_grade"].astype(int)
        # 距離/競馬場/タイプ/レースランク
        df_old2["distance_place_type_race_grade"] = (df_old2["course_len"].astype(str) + df_old2["place"].astype(str) + df_old2["race_type"].astype(str) + df_old2["race_grade"].astype(str)).astype(int)
        
        
        baselog_old = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around"]], on="race_id"
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
                     
        
        df_old["distance_place_type_race_grade"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
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
                (df_old2_1["race_date_day_count"] >= (target_day_count - 1200)) &  # race_date_day_count が target_day_count-1200 以上
                (df_old2_1["race_date_day_count"] <= (target_day_count - 1)) &  # race_date_day_count が target_day_count-1 以下
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
            return mean_time_diff if not np.isnan(mean_time_diff) else np.nan
        
        # 6. df の各行に対して、計算した平均値を新しい列に追加
        df["mean_ground_state_time_diff"] = df.apply(compute_mean_for_row, axis=1, df_old2_1=df_old2_1)

        def assign_value(row):
            # weather が 3, 4, 5 または ground_state が 0 以外の場合は 5 を設定
            if row["weather"] in [3, 4, 5] or row["ground_state"] != 0:
                return 7
            # mean_ground_state_time_diff に基づいて分類
            if 2.0 <= row["mean_ground_state_time_diff"]:
                return 1  #　超高速馬場1
            elif 1 <= row["mean_ground_state_time_diff"] < 2.0:
                return 2  # 高速馬場2
            
            elif 0.4 <= row["mean_ground_state_time_diff"] < 1:
                return 3  # 軽い馬場3
                
            elif -0.4 <= row["mean_ground_state_time_diff"] < 0.4:
                return 4  # 標準的な馬場4

            
            elif -1 <= row["mean_ground_state_time_diff"] < -0.4:
                return 5  # やや重い馬場5
            
            elif -2 <= row["mean_ground_state_time_diff"] < -1:
                return 6  # 重い馬場5
            
            elif row["mean_ground_state_time_diff"] < -2:
                return 7  # 道悪7
            # 該当しない場合は NaN を設定
            return 3
        
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
                return 2
        
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
                if taking_lead_per > 0.7:
                    return 4  # ハイペース
                else:
                    return 3  
            elif escape_count == 1:
                if taking_lead_per > 0.75:
                    return 4  # ハイペース
                elif 0.65 < taking_lead_per <= 0.75:
                    return 3  # ハイミドルペース
                elif 0.55 < taking_lead_per <= 0.65:
                    return 2  # スローミドルペース
                else:
                    return 1  # スローペース
            elif escape_count == 0:
                # 逃げ馬が0の場合、先行馬の割合に基づく
                if taking_lead_per > 0.88:
                    return 4  # ハイペース
                elif 0.75 < taking_lead_per <= 0.88:
                    return 3  # ハイミドルペース
                elif 0.65 < taking_lead_per <= 0.75:
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
                return 2
        
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
                if taking_lead_per > 0.7:
                    return 4  # ハイペース
                else:
                    return 3  
            elif escape_count == 1:
                if taking_lead_per > 0.75:
                    return 4  # ハイペース
                elif 0.65 < taking_lead_per <= 0.75:
                    return 3  # ハイミドルペース
                elif 0.55 < taking_lead_per <= 0.65:
                    return 2  # スローミドルペース
                else:
                    return 1  # スローペース
            elif escape_count == 0:
                # 逃げ馬が0の場合、先行馬の割合に基づく
                if taking_lead_per > 0.88:
                    return 4  # ハイペース
                elif 0.75 < taking_lead_per <= 0.88:
                    return 3  # ハイミドルペース
                elif 0.65 < taking_lead_per <= 0.75:
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
            .merge(self.race_info[["race_id", "place","weather","ground_state","course_len","race_type","race_date_day_count"]], on="race_id")
        )
        df["place"] = df["place"].astype(int)
        
        
        old_merged_df = self.population.copy()      
        
        
        df_old2 = (
            old_merged_df
            .merge(self.results[["race_id", "horse_id","time","rank","rank_per_horse"]], on=["race_id", "horse_id"])
            .merge(self.race_info[["race_id", "place","weather","ground_state","race_grade","course_len","race_type","race_date_day_count"]], on="race_id")
        )
        df_old2["place"] = df_old2["place"].astype(int)
        df_old2["race_grade"] = df_old2["race_grade"].astype(int)
        # 距離/競馬場/タイプ/レースランク
        df_old2["distance_place_type_race_grade"] = (df_old2["course_len"].astype(str) + df_old2["place"].astype(str) + df_old2["race_type"].astype(str) + df_old2["race_grade"].astype(str)).astype(int)
        
        
        baselog_old = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around"]], on="race_id"
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
                     
        
        df_old["distance_place_type_race_grade"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
        df_old_copy = df_old
        # rank列が1, 2, 3の行だけを抽出
        df_old = df_old[df_old['rank'].isin([1, 2, 3])]
        
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
                (df_old2_1["race_date_day_count"] >= (target_day_count - 1200)) &  # race_date_day_count が target_day_count-1200 以上
                (df_old2_1["race_date_day_count"] <= (target_day_count - 1)) &  # race_date_day_count が target_day_count-1 以下
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
            return mean_time_diff if not np.isnan(mean_time_diff) else np.nan
        
        # 6. df の各行に対して、計算した平均値を新しい列に追加
        df["mean_ground_state_time_diff"] = df.apply(compute_mean_for_row, axis=1, df_old2_1=df_old2_1)
        
        
        def assign_value(row):
            # weather が 3, 4, 5 または ground_state が 0 以外の場合は 5 を設定
            if row["ground_state"] in [3]:
                return 1
            if row["weather"] in [3, 4, 5]:
                return 2

            if row["ground_state"] in [1]:
                return 2
            if row["ground_state"] in [2]:
                return 5
            # mean_ground_state_time_diff に基づいて分類
            if 2.0 <= row["mean_ground_state_time_diff"]:
                return 1  #　超高速馬場1
            elif 1 <= row["mean_ground_state_time_diff"] < 2.0:
                return 2  # 高速馬場2
            
            elif 0.4 <= row["mean_ground_state_time_diff"] < 1:
                return 3  # 軽い馬場3
                
            elif -0.4 <= row["mean_ground_state_time_diff"] < 0.4:
                return 4  # 標準的な馬場4
        
            
            elif -1 <= row["mean_ground_state_time_diff"] < -0.4:
                return 5  # やや重い馬場5
            
            elif -2 <= row["mean_ground_state_time_diff"] < -1:
                return 6  # 重い馬場5
            
            elif row["mean_ground_state_time_diff"] < -2:
                return 7  # 道悪7
            # 該当しない場合は NaN を設定
            return 3
        
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
                self.race_info[["race_id", "goal_range_100","curve","goal_slope","course_len","place_season_condition_type_categori"]], 
                on="race_id",
            )
        

        merged_df_all = merged_df_all.merge(
                self.results[["race_id","horse_id","umaban"]], 
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
            .replace({5: -0.43, 4: -0.19, 3: 0, 2: 0.19,1: 0.43, -1: -0.23, -2: 0, -3: 0.23,-10000:0})
        ).astype(float)





        # 1600で正規化
        merged_df_all["course_len_processed"] = (merged_df_all["course_len"] / 1600)

        # 1600m未満ならそのまま、1600m以上なら緩やかに上昇 
        merged_df_all["course_len_processed"] = merged_df_all["course_len_processed"].apply(
            lambda x: ((x+2)/3) if x <= 1 else (((1 + ((x - 1) / 5))+3)/4)
        )


        # 1600で正規化
        merged_df_all["course_len_processed_0"] = (merged_df_all["course_len"] / 1700)-1

        # 1600m未満ならそのまま、1600m以上なら緩やかに上昇 
        merged_df_all["course_len_processed_0"] = merged_df_all["course_len_processed_0"].apply(
            lambda x: x * 1.7 if x <= 0 else 0
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
        merged_df_all["goal_range_100_processed"] = merged_df_all["goal_range_100"] - 3.5
        # プラスの値をすべて 0 に変換
        merged_df_all["goal_range_100_processed"] = merged_df_all["goal_range_100_processed"].clip(upper=0)
        
        
        # -4.5 を行う
        merged_df_all["curve_processed"] = merged_df_all["curve"] - 4.5
        # +の場合は数値を8倍する
        merged_df_all["curve_processed"] = merged_df_all["curve_processed"].apply(
            lambda x: x * 10 if x > 0 else x
        )
        
        
        # goal_slope に -1 を行う
        merged_df_all["goal_slope_processed"] = merged_df_all["goal_slope"] - 1
        
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
        merged_df_all.loc[:, "dominant_position_category_processed"] += merged_df_all["course_len_processed_0"]

        # merged_df_all.loc[:, "dominant_position_category_processed"] *= merged_df_all["course_len_processed"]





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
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place"]], on="race_id"
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
                (df['rank'].isin([1, 2, 3]))  # rankが1, 2, 3
                ]
                .groupby(['course_len', 'place', 'race_type'])[['time', 'nobori']]  # 3つのカテゴリごとにtimeとnoboriを集計
                .mean()
                .reset_index()  # インデックスをリセット
                .rename(columns={'time': f'time_avg_{grade}', 'nobori': f'nobori_avg_{grade}'})  # 列名を変更
            )
            
            # 元のDataFrameにマージ（left join）
            horse_results_baselog = pd.merge(horse_results_baselog, time_nobori_avg, on=['course_len', 'place', 'race_type'], how='left')
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
            1000: {1: 1.8, 0: 1.7},
            1150: {1: 1.52, 0: 1.45},
            1200: {1: 1.45, 0: 1.39},
            1300: {1: 1.34, 0: 1.27},
            1400: {1: 1.23, 0: 1.18},
            1500: {1: 1.12, 0: 1.08},
            1600: {1: 1.06, 0: 1.02},
            1700: {1: 1.00, 0: 0.94},
            1800: {1: 0.93, 0: 0.88},
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
                .groupby(['course_len', 'place', 'race_type'])[['time', 'nobori']]  # 3つのカテゴリごとにtimeとnoboriを集計
                .mean()
                .reset_index()  # インデックスをリセット
                .rename(columns={'time': f'time_avg_{grade}_top', 'nobori': f'nobori_avg_{grade}_top'})  # 列名を変更
            )
            
            # 元のDataFrameにマージ（left join）
            horse_results_baselog = pd.merge(horse_results_baselog, time_nobori_avg_top, on=['course_len', 'place', 'race_type'], how='left')
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
            lambda row: row['time'] if row['rank'] == 1 else row['time'] - row['rank_diff']+1,
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
        horse_results_baselog['race_grade_transformed'] = (horse_results_baselog['race_grade'] - 80) / 10 + 80


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
            "syunpatu_mean_1races_encoded_index", "syunpatu_mean_3races_encoded_index", "syunpatu_mean_5races_encoded_index", "syunpatu_mean_8races_encoded_index",
            "syunpatu_min_1races_encoded_index", "syunpatu_min_3races_encoded_index", "syunpatu_min_5races_encoded_index", "syunpatu_min_8races_encoded_index",
            "syunpatu_max_1races_encoded_index", "syunpatu_max_3races_encoded_index", "syunpatu_max_5races_encoded_index", "syunpatu_max_8races_encoded_index",
            
            # zizokuバージョンの列を追加
            "zizoku_mean_1races_encoded_index", "zizoku_mean_3races_encoded_index", "zizoku_mean_5races_encoded_index", "zizoku_mean_8races_encoded_index",
            "zizoku_min_1races_encoded_index", "zizoku_min_3races_encoded_index", "zizoku_min_5races_encoded_index", "zizoku_min_8races_encoded_index",
            "zizoku_max_1races_encoded_index", "zizoku_max_3races_encoded_index", "zizoku_max_5races_encoded_index", "zizoku_max_8races_encoded_index",

            "advantage_max_1_index", "advantage_max_3_index", "advantage_max_5_index", "advantage_max_8_index",
            "advantage_min_1_index", "advantage_min_3_index", "advantage_min_5_index", "advantage_min_8_index",
            "advantage_mean_1_index", "advantage_mean_3_index", "advantage_mean_5_index", "advantage_mean_8_index",


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






































        
    # def cross_features_16(
    #     self
    # ):  
                    
                
    #         #基準タイムの選定
    #         """

    #         （	基準
    #         タイム	－	走破
    #         タイム	）×	距離
    #         指数	＋（	斥量	－	５５	）×	２	＋	馬場
    #         指数	＋	８０	＝	スピード指数


    #         １０００万クラスのタイムを基準
    #         ・3歳上 / 4歳以上
    #         ・1勝クラス
    #         ・良 / 稍重
    #         ・入線順位1～3着馬
    #         上記条件の走破タイムの平均を出す
    #         (牝馬限定戦を除く)
    #         ・3歳上 / 4歳以上
    #         ・2勝クラス
    #         ・良 / 稍重
    #         ・入線順位1～3着馬
    #         上記条件の走破タイムの平均を出す
    #         (牝馬限定戦を除く)
    #         それらを更に平均する

    #         データ数が少ないコースでは、存在するクラスの平均走破タイムを
    #         「クラス指数の指数差 ÷ 距離指数 ÷ 10」で秒換算し、基準タイムを補正する。
    #         """
    #         baselog = (
    #             self.population.merge(
    #                 self.race_info[["race_id", "course_len", "race_class","race_type","race_grade","ground_state", "weather","place","race_date_day_count"]], on="race_id"
    #             )
    #         )

    #         df = (
    #             baselog
    #             .merge(self.results[["race_id", "impost","horse_id", "wakuban", "nobori","time","umaban","rank"]], on=["race_id", "horse_id"])
    #         )
    #         df["nobori"] = df["nobori"].fillna(df["nobori"].mean())

    #         df["place"] = df["place"].astype(int)
    #         df["race_grade"] = df["race_grade"].astype(int)
    #         df["ground_state"] = df["ground_state"].astype(int)
    #         df["weather"] = df["weather"].astype(int)  
    #         df["distance_place_type"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str)).astype(int)
    #         df2 = df
    #         # 1. 計算したいrace_gradeのリスト
    #         grades = [55,60,70,79,85,  89,  91,  94,  98]

    #         # 2. 各race_gradeごとにtimeとnoboriの平均を計算し、結合
    #         for grade in grades:
    #             # race_gradeが指定された値のときのtimeとnoboriの平均を計算
    #             time_nobori_avg = (
    #                 df[
    #                 (df['race_grade'] == grade) &  # race_gradeが指定のgrade
    #                 (df['ground_state'].isin([0, 2])) &  # ground_stateが0または2
    #                 (df['rank'].isin([1, 2, 3]))  # rankが1, 2, 3
    #                 ]
    #                 .groupby(['course_len', 'place', 'race_type'])[['time', 'nobori']]  # 3つのカテゴリごとにtimeとnoboriを集計
    #                 .mean()
    #                 .reset_index()  # インデックスをリセット
    #                 .rename(columns={'time': f'time_avg_{grade}', 'nobori': f'nobori_avg_{grade}'})  # 列名を変更
    #             )
                
    #             # 元のDataFrameにマージ（left join）
    #             df = pd.merge(df, time_nobori_avg, on=['course_len', 'place', 'race_type'], how='left')
    #         # 1. 補完する順番を指定
    #         grades = [70, 79,85, 89, 60, 91,  94, 55, 98]

    #         # 2. まずhorse_results_baselog内の補完処理を行う
    #         df['final_time_avg'] = np.nan
    #         df['final_nobori_avg'] = np.nan

    #         # 3. 最初に85で補完
    #         df['final_time_avg'] = np.where(
    #             df['final_time_avg'].isna(), 
    #             df['time_avg_70'], 
    #             df['final_time_avg']
    #         )
    #         df['final_nobori_avg'] = np.where(
    #             df['final_nobori_avg'].isna(), 
    #             df['nobori_avg_70'], 
    #             df['final_nobori_avg']
    #         )


    #         """
    #         距離指数をかける
    #         距離	芝	ダート
    #         1000m	1.8	1.7
    #         1150m	1.52	1.45
    #         1200m	1.45	1.39
    #         1300m	1.34	1.27
    #         1400m	1.23	1.18
    #         1500m	1.12	1.08
    #         1600m	1.06	1.02
    #         1700m	1.00	0.94
    #         1800m	0.93	0.88
    #         1900m	0.88	0.83
    #         2000m	0.83	0.79
    #         2100m	0.79	0.75
    #         2200m	0.75	0.7
    #         2300m	0.71	0.67
    #         2400m	0.68	0.64
    #         2500m	0.64	0.61
    #         2600m	0.62	0.59
    #         2800m	0.56	0.53
    #         3000m	0.53	0.5
    #         3200m	0.50	0.47
    #         3400m	0.47	0.44
    #         3600m	0.45	0.42
    #         """
    #         # 距離と芝ダートの対応表を辞書として定義
    #         conversion_table = {
    #             1000: {1: 1.8, 0: 1.7},
    #             1150: {1: 1.52, 0: 1.45},
    #             1200: {1: 1.45, 0: 1.39},
    #             1300: {1: 1.34, 0: 1.27},
    #             1400: {1: 1.23, 0: 1.18},
    #             1500: {1: 1.12, 0: 1.08},
    #             1600: {1: 1.06, 0: 1.02},
    #             1700: {1: 1.00, 0: 0.94},
    #             1800: {1: 0.93, 0: 0.88},
    #             1900: {1: 0.88, 0: 0.83},
    #             2000: {1: 0.83, 0: 0.79},
    #             2100: {1: 0.79, 0: 0.75},
    #             2200: {1: 0.75, 0: 0.7},
    #             2300: {1: 0.71, 0: 0.67},
    #             2400: {1: 0.68, 0: 0.64},
    #             2500: {1: 0.64, 0: 0.61},
    #             2600: {1: 0.62, 0: 0.59},
    #             2800: {1: 0.56, 0: 0.53},
    #             3000: {1: 0.53, 0: 0.5},
    #             3200: {1: 0.5, 0: 0.47},
    #             3400: {1: 0.47, 0: 0.44},
    #             3600: {1: 0.45, 0: 0.42}
    #         }

    #         # 新しい列に変換後の値を格納
    #         def map_conversion(row):
    #             course_len = row['course_len']
    #             race_type = row['race_type']
    #             if course_len in conversion_table and race_type in conversion_table[course_len]:
    #                 return conversion_table[course_len][race_type]
    #             else:
    #                 return None  # 該当しない場合はNoneを返す

    #         # 適用
    #         df['converted_value'] = df.apply(map_conversion, axis=1)


    #         # 4. 残りのグレードで補完（順番に）
    #         for grade in grades:
    #             if grade != 70:
    #                 time_col = f'time_avg_{grade}'
    #                 nobori_col = f'nobori_avg_{grade}'
                    
    #                 # timeの欠損を補完
    #                 df['final_time_avg'] = np.where(
    #                     df['final_time_avg'].isna(), 
    #                     df[time_col] + ((grade - 70) / 1.2/df['converted_value']/10),
    #                     df['final_time_avg']
    #                 )
    #                 # noboriの欠損を補完
    #                 df['final_nobori_avg'] = np.where(
    #                     df['final_nobori_avg'].isna(), 
    #                     df[nobori_col], 
    #                     df['final_nobori_avg']
    #                 )



    #         # 1. final_time_avgからtimeを引いた数値（秒）を計算
    #         df['time_diff_sp'] = df['final_time_avg'] - df['time']

    #         # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
    #         df['time_points'] = (df['time_diff_sp'])*10

    #         # 1. final_nobori_avgからnoboriを引いた数値（秒）を計算
    #         df['nobori_diff_sp'] = df['final_nobori_avg'] - df['nobori']

    #         # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
    #         df['nobori_points'] = (df['nobori_diff_sp'])*10



    #         # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
    #         df['time_points_course_index'] = df['time_points'] *df['converted_value'] 

    #         # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
    #         df['nobori_points_course_index'] = df['nobori_points'] *df['converted_value'] 


    #         """
    #         ＋（	斥量	－	５５	）
    #         """
    #         # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
    #         df['time_points_impost'] = (df['time_points_course_index'] +(df["impost"]-55) *1.7)

    #         # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
    #         df['nobori_points_impost'] = (df['nobori_points_course_index']+(df["impost"]-55) *1.7)



    #         """
    #         暫定馬場指数＝（馬場指数用基準タイム－該当レース上位３頭の平均タイム）× 距離指数
    #         馬場指数用基準タイム ＝ 基準タイム － (クラス指数 × 距離指数)　＋pase_diff

    #         ハイペースなら低く（早く見える）でてしまう
    #         スローペースなら高く（遅く見える）でてしまう
    #         pase_diffは+だとハイペース
    #         -だとスローペースなので
    #         そのまま+してよいかも
    #         """
    #         #predictの時にはold_poplationに変える
    #         old_merged_df = self.population.copy()     
    #         #ここで、直近レースのtimeを知りたい
    #         df_old2 = (
    #             old_merged_df
    #             .merge(self.results[["race_id", "horse_id","time","rank","rank_per_horse"]], on=["race_id", "horse_id"])
    #             .merge(self.race_info[["race_id", "place","weather","ground_state","race_grade","course_len","race_type","race_date_day_count"]], on="race_id")
    #         )
    #         df_old2["place"] = df_old2["place"].astype(int)
    #         df_old2["race_grade"] = df_old2["race_grade"].astype(int)
    #         # 距離/競馬場/タイプ/レースランク
    #         df_old2["distance_place_type_race_grade"] = (df_old2["course_len"].astype(str) + df_old2["place"].astype(str) + df_old2["race_type"].astype(str) + df_old2["race_grade"].astype(str)).astype(int)


    #         baselog_old = (
    #             self.population.merge(
    #                 self.race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around"]], on="race_id"
    #             )
    #             # .merge(
    #             #     self.horse_results,
    #             #     on=["horse_id", "course_len", "race_type"],
    #             #     suffixes=("", "_horse"),
    #             # )
    #             # .query("date_horse < date")
    #             # .sort_values("date_horse", ascending=False)
    #         )
    #         df_old = (
    #             baselog_old
    #             .merge(self.results[["race_id", "horse_id", "wakuban", "umaban","nobori","rank","time","sex"]], on=["race_id", "horse_id"])
    #         )
    #         df_old["place"] = df_old["place"].astype(int)
    #         df_old["race_grade"] = df_old["race_grade"].astype(int)
    #         df_old["ground_state"] = df_old["ground_state"].astype(int)
    #         df_old["around"] = df_old["around"].fillna(3).astype(int)
    #         df_old["weather"] = df_old["weather"].astype(int)  
                        

    #         df_old["distance_place_type_race_grade"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
    #         df_old_copy = df_old
    #         # rank列が1, 2, 3の行だけを抽出
    #         df_old = df_old[df_old['rank'].isin([1, 2, 3])]
    #         target_mean_1 = df_old.groupby("distance_place_type_race_grade")["time"].mean()
    #         df_old = df_old.copy()
    #         # 平均値をカテゴリ変数にマッピング
    #         df_old["distance_place_type_race_grade_encoded"] = df_old["distance_place_type_race_grade"].map(target_mean_1)


    #         df_old = df_old[["distance_place_type_race_grade",'distance_place_type_race_grade_encoded']]


    #         columns_to_merge = [
    #             ("distance_place_type_race_grade",'distance_place_type_race_grade_encoded'),
    #         ]

    #         # 各ペアを順番に処理
    #         for original_col, encoded_col in columns_to_merge:
    #             df2_subset = df_old[[original_col, encoded_col]].drop_duplicates()  # 重複を削除
    #             df2_subset = df2_subset.reset_index(drop=True)  # インデックスをリセット
    #             df_old2 = df_old2.merge(df2_subset, on=original_col, how='left')  # dfにマージ
    #         df_old2 = df_old2[df_old2['rank'].isin([1, 2, 3])]
    #         df_old2["distance_place_type_race_grade_encoded_time_diff"] = df_old2['distance_place_type_race_grade_encoded'] - df_old2["time"]

    #         # df_old2= df_old2[df_old2["race_type"] != 2]
    #         # df_old2_1 = df_old2[df_old2["race_type"] != 0]
    #         df_old2_1 = df_old2
    #         # 2. df の各行について処理
    #         def compute_mean_for_row(row, df_old2_1):
    #             # race_type == 0 の場合は NaN を返す
    #             # if row["race_type"] == 0:
    #             #     return np.nan
                    
    #             target_day_count = row["race_date_day_count"]  # df の各行の race_date_day_count

    #             # 3. df_old2_1 から条件に一致する行をフィルタリング
    #             filtered_df_old2_1 = df_old2_1[
    #                 (df_old2_1["race_date_day_count"] >= (target_day_count - 1200)) &  # race_date_day_count が target_day_count-1200 以上
    #                 (df_old2_1["race_date_day_count"] <= (target_day_count - 1)) &  # race_date_day_count が target_day_count-1 以下
    #                 (df_old2_1["place"] == row["place"]) &  # place が一致
    #                 (df_old2_1["race_type"] == row["race_type"])  & 
    #                 (df_old2_1["weather"].isin([0, 1, 2])) &  # weather が 0, 1, 2 のいずれか
    #                 (df_old2_1["ground_state"] == 0)   # ground_state が 0
    #                 # &
    #                 # (df_old2_1["rank_per_horse"] < 0.87)  # rank_per_horse が 0.87 未満
    #             ]
    #             filtered_df_old2_2 = df_old2_1[
    #                 (df_old2_1["race_date_day_count"] >= (target_day_count - 1200)) &  # race_date_day_count が target_day_count-1200 以上
    #                 (df_old2_1["race_date_day_count"] <= (target_day_count + 1000)) &  # race_date_day_count が target_day_count-1 以下
    #                 (df_old2_1["place"] == row["place"]) &  # place が一致
    #                 (df_old2_1["race_type"] == row["race_type"]) & 
    #                 (df_old2_1["weather"].isin([0, 1, 2])) &  # weather が 0, 1, 2 のいずれか
    #                 (df_old2_1["ground_state"] == 0)   # ground_state が 0                    
    #                 # &
    #                 # (df_old2_1["rank_per_horse"] < 0.87)  # rank_per_horse が 0.87 未満
    #             ]
    #             # 4. フィルタリングした行の "distance_place_type_race_grade_encoded_time_diff" の平均を計算
    #             mean_time_diff = filtered_df_old2_1["distance_place_type_race_grade_encoded_time_diff"].mean()

    #             # 5. 計算結果を返す（NaNの場合も考慮）
    #             return mean_time_diff if not np.isnan(mean_time_diff) else filtered_df_old2_2["distance_place_type_race_grade_encoded_time_diff"].mean()

    #         # 6. df の各行に対して、計算した平均値を新しい列に追加
    #         df['time_diff_grade'] = df.apply(compute_mean_for_row, axis=1, df_old2_1=df_old2_1)


    #         # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
    #         df['time_points_grade'] = (df['time_diff_grade'] )*10

    #         # # 1. final_nobori_avgからnoboriを引いた数値（秒）を計算
    #         # df['nobori_diff_grade'] = df['nobori_condition_index_shaft'] 

    #         # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
    #         df['nobori_points_grade'] = (df['time_diff_grade'] )*10

    #         """
    #         距離指数をかける
    #         """

    #         # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
    #         df['time_points_grade_index'] = (df['time_points_grade'] *df['converted_value'])

    #         # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
    #         df['nobori_points_grade_index'] = (df['nobori_points_grade'] *df['converted_value'])

    #         df['time_condition_index'] = df['time_points_grade_index']
    #         df['nobori_condition_index'] = df['nobori_points_grade_index']



    #         # 新しい列を作成
    #         df['race_grade_transformed'] = (df['race_grade'] - 80) / 3 + 80


    #         df['speed_index'] = df['time_points_impost'] + df['time_condition_index'] +df['race_grade_transformed']
    #         df['nobori_index'] = df['nobori_points_impost'] + df['nobori_condition_index'] + df['race_grade_transformed']


    #         # 5. 不要な中間列を削除し、必要な列だけ残す
    #         columns_to_drop = [f'time_avg_{grade}' for grade in grades] + [f'nobori_avg_{grade}' for grade in grades]
    #         df = df.drop(columns=columns_to_drop)

    #         # df = df.dropna(subset=["speed_index"])
    #         # df["speed_index"] = df["speed_index"].astype(int)
    #         # df = df.dropna(subset=["nobori_index"])
    #         # df["nobori_index"] = df["nobori_index"].astype(int)




    #         df = df[['race_id',
    #         'date',
    #         'horse_id','speed_index',
    #         'nobori_index']]

    #         merged_df = self.population.copy()
    #         merged_df = merged_df.merge(df, on=["race_id",'date', "horse_id"], how="left")

    #         self.agg_cross_features_df_16 = merged_df
    #         print("running cross_features_16()...comp")
                    

    # def cross_features_15(
    #     self, n_races: list[int] = [1, 3, 5, 10]
    # ):  

    #     self.agg_cross_features_df_15 = merged_df
    #     print("running cross_features_15()...comp")
        

    
    def create_features(self) -> pd.DataFrame:
        """
        特徴量作成処理を実行し、populationテーブルに全ての特徴量を結合する。
        """
        # 馬の過去成績集計
        self.create_baselog()
        self.create_race_grade()
        self.agg_horse_n_races()
        self.agg_horse_n_races_relative()
        self.agg_course_len()
        self.results_relative()
        self.agg_jockey()
        self.agg_trainer()
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
        self.cross_features_16()  


        self.agg_horse_per_group_cols(
            group_cols=["ground_state", "race_type"], df_name="ground_state_race_type"
        )
        self.agg_horse_per_group_cols(
            group_cols=["race_grade"], df_name="race_grade"
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
        self.agg_horse_per_group_cols(
            group_cols=["place", "race_type"], df_name="race_place"
        )    

        self.agg_horse_per_group_cols(
            group_cols=["weather"], df_name="weather"
        )
                
        self.agg_sire()
        self.agg_bms()  
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
                self.agg_horse_per_group_cols_dfs["race_grade"],
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
            .merge(
                self.agg_cross_features_df_16,
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
            .merge(
                self.agg_sire_df,
                on=["race_id", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_bms_df,
                on=["race_id", "horse_id"],
                how="left",
                # copy=False,
            )            
        )
        # features.drop(columns=['place_season_condition_type_categori_x'], inplace=True)        
        features.to_csv(self.output_dir / self.output_filename, sep="\t", index=False)
        print("merging all features...comp")
        return features
