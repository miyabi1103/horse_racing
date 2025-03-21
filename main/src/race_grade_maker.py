

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
COMMON_DATA_DIR = Path("..", "..", "common", "data")
DATA_DIR = Path("..", "data")
POPULATION_DIR = DATA_DIR / "00_population"
INPUT_DIR = DATA_DIR / "01_preprocessed"
OUTPUT_DIR = DATA_DIR / "02_features"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MAPPING_DIR = COMMON_DATA_DIR / "mapping"

OUTPUT_INFO_DIR = Path("..", "data", "01_preprocessed")

# カテゴリ変数を数値に変換するためのマッピング
with open(MAPPING_DIR / "sex.json", "r") as f:
    sex_mapping = json.load(f)
with open(MAPPING_DIR / "race_type.json", "r") as f:
    race_type_mapping = json.load(f)
with open(MAPPING_DIR / "around.json", "r") as f:
    around_mapping = json.load(f)
with open(MAPPING_DIR / "weather.json", "r") as f:
    weather_mapping = json.load(f)
with open(MAPPING_DIR / "ground_state.json", "r") as f:
    ground_state_mapping = json.load(f)
with open(MAPPING_DIR / "race_class.json", "r") as f:
    race_class_mapping = json.load(f)
with open(MAPPING_DIR / "place.json", "r") as f:
    place_mapping = json.load(f)



def create_race_grade(
    population_dir: Path = POPULATION_DIR,
    poplation_filename: str = "population.csv",
    input_dir: Path = INPUT_DIR,
    results_filename: str = "results.csv",
    race_info_filename: str = "race_info_before.csv",
    output_dir_info: Path = OUTPUT_INFO_DIR,
    output_filename_info: str = "race_info.csv",
) -> pd.DataFrame:
    population = pd.read_csv(population_dir / poplation_filename, sep="\t")
    results = pd.read_csv(input_dir / results_filename, sep="\t")
    race_info_before = pd.read_csv(input_dir / race_info_filename, sep="\t")
    output_dir_info = output_dir_info
    output_filename_info = output_filename_info


    df = (
        results[["race_id", "horse_id","mean_age_kirisute"]]
        .merge(race_info_before[["race_id", "race_type","place",  "season_level","race_class"]], on="race_id")
    )
    # self.race_info = self.race_info_before.copy()

    race_i = race_info_before

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
    df['age_season'] = (df['mean_age_kirisute'].astype(str) + df['season_level'].astype(str)).astype(int)
    df['race_class'] = df['race_class'].astype(int)
    df['place'] = df['place'].astype(int)
    place_adjustment = {
        44: 0, 43: 0, 45: -1,30: -3,
        54: -4,50: -5,51: -5,42: -5,48: -7,
        35: -9,36: -10,46: -13,
        55: -16,47: -19,
        
    }
    # race_gradeの作成
    def calculate_race_grade(row):
        age_season = row['age_season']
        race_class = row['race_class']
        place = row['place']
        # 競馬場ごとの補正値を取得（該当しない場合は0）
        adjustment = place_adjustment.get(place, 0)

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
        

        elif race_class == -4:
            base_grade = 82
        elif race_class == -3:
            base_grade = 83
        elif race_class == -2:
            base_grade = 84
        elif race_class == -1:
            base_grade = 85
        elif race_class == -5:
            base_grade = 80
        elif race_class == -6:
            base_grade = 79
        elif race_class == -7:
            base_grade = 74
        elif race_class == -8:
            base_grade = 69
        elif race_class == -9:
            base_grade = 64
        elif race_class == -10:
            base_grade = 59
        elif race_class == -11:
            base_grade = 55
        elif race_class == -11.5:
            base_grade = 53
        elif race_class == -12.5:
            base_grade = 48
        elif race_class == -12:
            base_grade = 50
        elif race_class == -13:
            base_grade = 50
        elif race_class == -14:
            base_grade = 40
        elif race_class == -15:
            base_grade = 30
        else:
            return np.nan  

        # 競馬場ごとの補正値を適用
        return base_grade + adjustment

    # race_grade列を作成
    df['race_grade'] = df.apply(calculate_race_grade, axis=1)

    #race_grade_scaledの作成
    df['race_grade_scaled'] = df['race_grade'] / 10
    df_agg = df.groupby("race_id", as_index=False).agg({"race_grade": "first",'age_season':"first", "race_grade_scaled": "first"})

    # self.race_info[['age_season', 'race_grade', 'race_grade_scaled']] = df[['age_season', 'race_grade', 'race_grade_scaled']]
    race_info = (
        race_i
        .merge(df_agg[["race_id",'race_grade','age_season', 'race_grade_scaled']], on="race_id",how="left")
    )
    race_info = race_info.dropna(subset=['race_grade'])
    race_info['race_grade'] = race_info['race_grade'].astype(int)
    race_info.to_csv(output_dir_info/output_filename_info, sep="\t", index=False)

    return race_info