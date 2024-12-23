

#過去3レースのない馬のいるレースを除外

from pathlib import Path
import json
import re
import pandas as pd
import numpy as np




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
    
    horse_results_filename = "horse_results.csv",    
) -> pd.DataFrame:
    """
    from_に開始日、to_に終了日を指定（yyyy-mm-dd形式）して、その期間に絞って
    学習母集団である（race_id, date, horse_id）の組み合わせを作成する。
    """
    race_info = pd.read_csv(input_dir / race_info_filename, sep="\t")
    race_info["date"] = pd.to_datetime(
        race_info["info2"].map(lambda x: eval(x)[0]), format="%Y年%m月%d日"
    )
    results = pd.read_csv(input_dir / results_filename, sep="\t")
    population = (
        race_info.query("@from_ <= date <= @to_")[["race_id", "date"]]
        .merge(results[["race_id", "horse_id"]], on="race_id")
        .sort_values(["date", "race_id"])
    )


    
    
    #ホース結果を加工し、rankの過去がないやつを探す 
    
    population_horse_ids = population['horse_id'].tolist()
    
    # クエリでリストを直接使用
    df = pd.read_csv(input_dir / horse_results_filename, sep="\t").query(
        "horse_id in @population_horse_ids"
    )

    df["rank"] = pd.to_numeric(df["着順"], errors="coerce")
    df.dropna(subset=["rank"], inplace=True)
    df["date"] = pd.to_datetime(df["日付"])  
    # df["place"] = df["開催"].str.extract(r"(\D+)")[0].map(place_mapping)
    # df.dropna(subset=["place"], inplace=True)   

    # #障害レースを排除
    # # ダートor芝or障害2
    # df["race_type"] = df["距離"].str[0].map(race_type_mapping)
    # df = df[df["race_type"] != 2]

    # NoneをNaNに置き換え
    df = df.where(pd.notnull(df), np.nan)
    # 使用する列を選択
    df = df[
        [
            "horse_id",
            "date",
            "rank",
            # "place",
            # "race_type",
        ] 
    ]
    baselog = (
        population.merge
        (
            df,
            on="horse_id",
            suffixes=("", "_horse"),
        )  
        .query("date_horse < date")
        .sort_values("date_horse", ascending=False)
        )
    grouped_df = baselog.groupby(["race_id", "horse_id"])
    merged_df = population.copy()
    
    df2 = (
        grouped_df.head(3)
        .groupby(["race_id", "horse_id"])[["rank"]]
        .mean()
        .add_suffix(f"_3races")
    )
    merged_df = merged_df.merge(df2, on=["race_id", "horse_id"])
    population = merged_df[["race_id","date","horse_id"]]
    
    population.to_csv(output_dir / output_filename, sep="\t", index=False)
    return population






