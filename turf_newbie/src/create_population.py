from pathlib import Path
import json
import re
import pandas as pd
import numpy as np


#除外していないデータ


COMMON_DATA_DIR = Path("..", "..", "common", "data")
RAWDF_DIR = COMMON_DATA_DIR / "rawdf"
OUTPUT_DIR = Path("..", "data", "00_population")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAPPING_DIR = COMMON_DATA_DIR / "mapping"

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

def create(
    from_: str,
    to_: str,
    input_dir: Path = RAWDF_DIR,
    race_info_filename: str = "race_info.csv",
    results_filename: str = "results.csv",
    output_dir: Path = OUTPUT_DIR,
    output_filename: str = "population.csv",
) -> pd.DataFrame:
    """
    from_に開始日、to_に終了日を指定（yyyy-mm-dd形式）して、その期間に絞って
    学習母集団である（race_id, date, horse_id）の組み合わせを作成する。
    """
    race_info = pd.read_csv(input_dir / race_info_filename, sep="\t")
    race_info["date"] = pd.to_datetime(
        race_info["info2"].map(lambda x: eval(x)[0]), format="%Y年%m月%d日"
    )
    
    race_info["tmp"] = race_info["info1"].map(lambda x: eval(x)[0])
    race_info["race_type"] = race_info["tmp"].str[0].map(race_type_mapping)
    race_info = race_info[race_info["race_type"] != 2]
    race_info = race_info[race_info["race_type"] != 0]


    results = pd.read_csv(input_dir / results_filename, sep="\t")
    population = (
        race_info.query("@from_ <= date <= @to_")[["race_id", "date"]]
        .merge(results[["race_id", "horse_id"]], on="race_id")
        .sort_values(["date", "race_id"])
    )


    population.to_csv(output_dir / output_filename, sep="\t", index=False)
    return population

