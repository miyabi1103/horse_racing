import json
from pathlib import Path


import re

import pandas as pd
import numpy as np
import ast
COMMON_DATA_DIR = Path("..", "..", "common", "data")
RAWDF_DIR = COMMON_DATA_DIR / "rawdf"
MAPPING_DIR = COMMON_DATA_DIR / "mapping"
POPULATION_DIR = Path("..", "data", "00_population")
OUTPUT_DIR = Path("..", "data", "01_preprocessed")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

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


# 特徴量に使うレースリザルトを加工してアウトプットする関数
def process_results(
    population_dir: Path = POPULATION_DIR,
    populaton_filename: str = "population.csv",
    input_dir: Path = RAWDF_DIR,
    output_dir: Path = OUTPUT_DIR,
    input_filename: str = "results.csv",
    output_filename: str = "results.csv",
    sex_mapping: dict = sex_mapping,
) -> pd.DataFrame:
    """
    未加工のレース結果テーブルをinput_dirから読み込んで加工し、
    output_dirに保存する関数。
    """
    population = pd.read_csv(population_dir / populaton_filename, sep="\t")

    # df = pd.read_csv(input_dir / input_filename, sep="\t").query(
    #     "race_id in @population['race_id']"
    # )
    # `race_id`のリストを作成
    population_race_ids = population['race_id'].tolist()
    
    # クエリでリストを直接使用
    df = pd.read_csv(input_dir / input_filename, sep="\t").query(
        "race_id in @population_race_ids"
    )


    df["rank"] = pd.to_numeric(df["着順"], errors="coerce")
    df.dropna(subset=["rank"], inplace=True)
    df["rank"] = df["rank"].astype(int)
    
    # 時間を秒に変換
    df["time"] = pd.to_datetime(df["タイム"], format="%M:%S.%f", errors="coerce")
    df.dropna(subset=["time"], inplace=True)
    df["time"] = (
        df["time"].dt.minute * 60
        + df["time"].dt.second
        + df["time"].dt.microsecond / 1000000
    )
    df["time"] = df["time"].astype(float)

    
    # その他の列を整形
    df["nobori"] = df["上り"].astype(float)
    df["umaban"] = df["馬番"].astype(int)
    df["tansho_odds"] = df["単勝"].astype(float)
    df["popularity"] = df["人気"].astype(int)
    df["impost"] = df["斤量"].astype(float)
    df["wakuban"] = df["枠番"].astype(int)
    df["sex"] = df["性齢"].str[0].map(sex_mapping)
    df["age"] = df["性齢"].str[1:].astype(int)
    df["weight"] = df["馬体重"].str.extract(r"(\d+)").astype(int)
    df["weight_diff"] = df["馬体重"].str.extract(r"\((.+)\)").astype(int)
    df["n_horses"] = df.groupby("race_id")["race_id"].transform("count")
    
    # コーナー通過順を分割して列を作成
    corner_cols = df['通過'].str.split('-', expand=True)
    corner_cols.columns = [f'corner_{i+1}' for i in range(corner_cols.shape[1])]
    # オブジェクト型のデータを整数型に変換する
    corner_cols = corner_cols.apply(pd.to_numeric, errors='coerce').astype('Int64')  
    # nullable int型を指定
    

    # # time列の相対化
    # tmp_df = df.groupby("race_id")["time"]
    # df["time_relative"] = ((df["time"] - tmp_df.transform("mean")) / tmp_df.transform("std"))
    # tmp_df = df.groupby("race_id")["rank"]
    # df["rank_relative"] = ((df["rank"] - tmp_df.transform("mean")) / tmp_df.transform("std"))

    
    # 元のデータフレームと結合
    result_df = pd.concat([df, corner_cols], axis=1)
    # NoneをNaNに置き換え
    result_df = result_df.where(pd.notnull(result_df), np.nan)
    df = result_df

    # rank / n_horses の特徴量を作成（欠損値を含む行はNaNに設定）
    df["rank_per_horse"] = df["rank"].where(df["rank"].notna(), np.nan) / df["n_horses"].where(df["n_horses"].notna(), np.nan)
    
    # corner_1 / n_horses の特徴量を作成（欠損値を含む行はNaNに設定）
    df["corner_1_per_horse"] = df["corner_1"].where(df["corner_1"].notna(), np.nan) / df["n_horses"].where(df["n_horses"].notna(), np.nan)
    
    df["corner_2_per_horse"] = df["corner_2"].where(df["corner_2"].notna(), np.nan) / df["n_horses"].where(df["n_horses"].notna(), np.nan)
    df["corner_3_per_horse"] = df["corner_3"].where(df["corner_3"].notna(), np.nan) / df["n_horses"].where(df["n_horses"].notna(), np.nan)
    df["corner_4_per_horse"] = df["corner_4"].where(df["corner_4"].notna(), np.nan) / df["n_horses"].where(df["n_horses"].notna(), np.nan)
    # NoneをNaNに置き換え
    df = df.where(pd.notnull(df), np.nan)   
    result_df = df
    
    # for col in corner_cols.columns:
    #     # ここでは result_df を使う
    #     tmp_df = result_df.groupby("race_id")[col]
    #     result_df[f"{col}_relative"] = ((result_df[col] - tmp_df.transform("mean")) / tmp_df.transform("std"))
    # result_df = result_df.apply(lambda col: col.apply(lambda x: np.nan if pd.isna(x) else x))


    """
    レース平均年齢、中央値、平均年齢切り捨て
    """

    # 1. race_idごとの年齢の平均、中央値、平均（小数点以下切り捨て）を計算
    race_age_stats = result_df.groupby('race_id').agg(
        mean_age=('age', 'mean'),
        median_age=('age', 'median'),
        mean_age_kirisute=('age', lambda x: int(x.mean()))  # 小数点以下切り捨て
    ).reset_index()
    
    # 2. result_dfに統合（merge）し、_xや_yを回避する
    result_df = result_df.merge(
        race_age_stats, 
        on='race_id', 
        how='left', 
        suffixes=('', '_drop')  # _drop を付けることで重複を回避
    )
    
    # 不要な列を削除（この場合は、"age_drop" などの列があれば削除する）
    result_df = result_df.loc[:, ~result_df.columns.str.endswith('_drop')]



    # データが着順に並んでいることによるリーク防止のため、各レースを馬番順にソートする
    result_df = result_df.sort_values(["race_id", "umaban"])

    # 使用する列を選択
    result_df = result_df[
        [
            "race_id",
            "horse_id",
            "jockey_id",
            "trainer_id",
            "owner_id",
            "rank", 
            "rank_per_horse",
            "time",
            "nobori",
            "umaban",
            "wakuban",
            "tansho_odds",
            "popularity",
            "impost",
            "sex",
            "age",
            "weight",
            "weight_diff",
            "n_horses",
            "corner_1_per_horse",
            "corner_2_per_horse",
            "corner_3_per_horse",
            "corner_4_per_horse",
            "mean_age",
            "median_age",
            "mean_age_kirisute",
            # "time_relative",  # timeの相対化列
            # "rank_relative",  # rankの相対化列
        ]  + list(corner_cols.columns)  # コーナー通過列はここで自動的に含まれます
    # + [f"{col}_relative" for col in corner_cols.columns] 
    ]

    # 結果を出力
    result_df.to_csv(output_dir / output_filename, sep="\t", index=False)
    
    return result_df




def process_race_info(
    population_dir: Path = POPULATION_DIR,
    populaton_filename: str = "population.csv",
    input_dir: Path = RAWDF_DIR,
    output_dir: Path = OUTPUT_DIR,
    input_filename: str = "race_info.csv",
    output_filename: str = "race_info.csv",
    race_type_mapping: dict = race_type_mapping,
    around_mapping: dict = around_mapping,
    weather_mapping: dict = weather_mapping,
    ground_state_mapping: dict = ground_state_mapping,
    race_class_mapping: dict = race_class_mapping,
) -> pd.DataFrame:
    """
    未加工のレース情報テーブルをinput_dirから読み込んで加工し、
    output_dirに保存する関数。
    """
    population = pd.read_csv(population_dir / populaton_filename, sep="\t")
    population_race_ids = population['race_id'].tolist()
    
    # クエリでリストを直接使用
    df = pd.read_csv(input_dir / input_filename, sep="\t").query(
        "race_id in @population_race_ids"
    )
    
    # df = pd.read_csv(input_dir / input_filename, sep="\t").query(
    #     "race_id in @population['race_id']"
    # )
    
    # evalで文字列型の列をリスト型に変換し、一時的な列を作成
    df["tmp"] = df["info1"].map(lambda x: eval(x)[0])

    # info1 列からコースの長さを取り出して tmp 列を作成
    df["tmp2"] = df["info1"].map(lambda x: ast.literal_eval(x))
    
    # tmp 列から距離の部分を抽出
    def extract_course_len(info_list):
        for item in info_list:
            match = re.search(r"(\d+)m", item)
            if match:
                return match.group(1)  # マッチした数字部分を返す
        return None  # 該当がなければ None を返す
    
    # コース長を新しい列に追加
    df["course_len_1"] = df["tmp2"].apply(extract_course_len)
    df["course_len_2"] = df["tmp2"].map(lambda x: x[1]).str.extract(r"(\d+)")
    df["combined_course_len"] = df["course_len_1"].fillna(df["course_len_2"])
  

    # ダートor芝or障害
    df["race_type"] = df["tmp"].str[0].map(race_type_mapping)

    # 右or左or直線
    df["around"] = df["tmp"].str[1].map(around_mapping)
    df["course_len"] = df["course_len_1"].fillna(df["course_len_2"])
    # df["course_len"] = df["tmp"].str.extract(r"(\d+)")
    # df["course_len"] = df["tmp"].str.extract(r"(\d+)(?=m)")

    df["weather"] = df["info1"].str.extract(r"天候:(\w+)")[0].map(weather_mapping)
    df["ground_state"] = (
        df["info1"].str.extract(r"(芝|ダート|障害):(\w+)")[1].map(ground_state_mapping)
    )
    df["date"] = pd.to_datetime(
        df["info2"].map(lambda x: eval(x)[0]), format="%Y年%m月%d日"
    )
    regex_race_class = "|".join(race_class_mapping)
    df["race_class"] = (
        df["title"]
        .str.extract(rf"({regex_race_class})")
        # タイトルからレース階級情報が取れない場合はinfo2から取得
        .fillna(df["info2"].str.extract(rf"({regex_race_class})"))[0]
        .map(race_class_mapping)
    )
    df["place"] = df["race_id"].astype(str).str[4:6].astype(int)
    df.dropna(subset=["place"], inplace=True)

    # 年、月、日をそれぞれ抽出
    df["date_year"] = df["date"].dt.year
    df["date_month"] = df["date"].dt.month
    df["date_day"] = df["date"].dt.day

    df["date_year"] = df["date_year"] - 1
    # 各月の累積日数を計算する関数
    def get_cumulative_days(month):
        # 各月の累積日数（平年を仮定）
        cumulative_days = {
            1: 0,               # 1月
            2: 31,              # 2月
            3: 31 + 28,         # 3月
            4: 31 + 28 + 31,    # 4月
            5: 31 + 28 + 31 + 30,  # 5月
            6: 31 + 28 + 31 + 30 + 31,  # 6月
            7: 31 + 28 + 31 + 30 + 31 + 30,  # 7月
            8: 31 + 28 + 31 + 30 + 31 + 30 + 31,  # 8月
            9: 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31,  # 9月
            10: 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30,  # 10月
            11: 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31,  # 11月
            12: 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30  # 12月
        }
        return cumulative_days.get(month, 0)
    
    # 月の累積日数を計算
    df["month_cumulative_days"] = df["date_month"].apply(get_cumulative_days)
    
    # 年×365 + 月の累積日数 + 日を計算
    df["custom_date_value"] = df["date_year"] * 365 + df["month_cumulative_days"] + df["date_day"]
    
    # 必要に応じて中間列を削除
    df = df.drop(columns=["month_cumulative_days"])  # 中間列が不要な場合
    
        
    df["race_day_count"] = df['race_id'].astype(str).str[-2:]
    
    
    df["race_date_day_count"] = df["custom_date_value"].astype(str) + df["race_day_count"]
    
    
    
    df["race_day_count"].astype(int)
    df["race_date_day_count"].astype(int)


    # 月を抽出して開催シーズンを判定
    def determine_season(month):
        if 6 <= month <= 8:
            return "2" #"夏開催"
        elif month == 12 or 1 <= month <= 2:
            return "4" #"冬開催"
        elif 3 <= month <= 5:
            return "1" #"春開催"
        elif 9 <= month <= 11:
            return "3" #"秋開催"    
    
    
    df["race_type"] = df["race_type"].astype(str)
    
    df["season"] = df["date"].dt.month.map(determine_season)
    
    # 開催地とシーズンを組み合わせた新しいカテゴリを作成
    df["place_season"] = df["place"].astype(str) + df["season"]
    #芝/ダートを紐付け、ダ0芝1障2
    df["place_season_type"] = df["place_season"] + df["race_type"]
    
    
    
    # 開催レースが何回かを表示
    df["kaisai_race"] = df['race_id'].astype(str).str[-2:]
    #芝/ダートを紐付け、ダ0芝1障2
    df["kaisai_race_type"] = df["kaisai_race"] + df["race_type"]
    
    # 何日目かを表示
    df["day"] = df['race_id'].astype(str).str[-4:-2]
    #芝/ダートを紐付け、ダ0芝1障2
    df["day_type"] = df["day"] + df["race_type"]
    
    # 開催回数を表示
    df["kaisai_count"] = df['race_id'].astype(str).str[-6:-4]
    #芝/ダートを紐付け、ダ0芝1障2
    df["kaisai_count_type"] = df["kaisai_count"] + df["race_type"]
    
    
    # 開催_季節_開催回数を表示
    df["place_season_day_type"] = df["place_season"] + df["day_type"]
    
    
    # condition
    df['day_condition'] = np.where(df['day'].astype(int) < 4, 1, 2)
    df['day_condition'] = df['day_condition'].astype(str)
    
    # 開催_季節_開催コンディション_レースタイプを表示
    df["place_season_condition_type"] = df["place_season"] + df['day_condition'] + df["race_type"]
    
    
    # コース別の馬場レベルを対応させる
    """
    オール野芝 ▶︎最も軽い芝で、速い時計が出やすい
    軽い芝 ▶︎時計面では野芝に劣るものの、こちらも速い時計が出やすい
    オーバーシード ▶︎ケースバイケース
    重い芝 ▶︎馬場が重く、時計もかかりやすい
    非常に重い芝 ▶︎JRA10場の中で最も重く、時計もかかりやすい
    
    オール野芝5	 '10111','10211',   小倉(夏)	'6311',中山(秋)	 '9311',   阪神(秋)	 '4111',  '4211',  '4311', '4411', 
    新潟	 '7311',   中京(秋)		
    軽い芝4	 '5111', '5211',  '5311',  '5411', 
     東京	 '8111',  '8211', '8311',   京都	'4121','4221','4321', '4421',  新潟	'10121','10221','小倉(夏)	 '6321', ' 中山(秋)	'9321', 阪神(秋)	'7321', 中京(秋)
    オーバーシード3	'5121','5221','5321','5421',  東京	'8121','8221', '8321',京都	 '9111',  '9211', '9411', 阪神	 '3111', '3211',  '3311', '3411', 
    福島	'7111',  '7211',  '7411', 中京	'6111',  '6211',  '6411',  中山	'10311','10411'小倉(冬)
    重い芝2	'9121','9221','9421',阪神	'6121','6221','6421',中山	'7121','7221','7421',中京	'3121', '3221','3321', '3421',  福島	'8411', '8421'京都(冬)	 '10321', '10421'小倉(冬)	
    非常に重い芝1	 '2111', '2121', '2211', '2221', '2311', '2321', '2411', '2421',  
    函館	'1111', '1121', '1211', '1221', '1311', '1321', '1411', '1421',  札幌					
    
    令和版ダートコースの馬場レベル表
    軽いダート-1	 '5110', '5120', '5210', '5220', '5310', '5320', '5410', '5420',東京	 '8110', '8120', '8210', '8220', '8310', '8320', '8410', '8420', 京都				
    重いダート-2	'1110', '1120', '1210', '1220', '1310', '1320', '1410', '1420', 札幌	'2110', '2120', '2210', '2220', '2310', '2320', '2410', '2420',函館	   '10110', '10120', '10210', '10220', '10310', '10320', '10410', '10420'小倉	'3110', '3120', '3210', '3220', '3310', '3320', '3410', '3420',福島	'4110', '4120', '4210', '4220', '4310', '4320', '4410', '4420',新潟	 '9110', '9120', '9210', '9220', '9310', '9320', '9410', '9420', 阪神
    非常に重いダート-3	'7110', '7120', '7210', '7220', '7310', '7320', '7410', '7420', 中京	'6110', '6120', '6210', '6220', '6310', '6320', '6410', '6420', 中山				
    障害レースはその他0
    """
    
    
    conversion_map = {
        '10111': 5, '10211': 5, '6311': 5, '9311': 5, '4111': 5, '4211': 5, '4311': 5, '4411': 5, '7311': 5,
        '5111': 4, '5211': 4, '5311': 4, '5411': 4, '8111': 4, '8211': 4, '8311': 4, '4121': 4, '4221': 4, '4321': 4, '4421': 4,
        '10121': 4, '10221': 4, '6321': 4, '9321': 4, '7321': 4,
        '5121': 3, '5221': 3, '5321': 3, '5421': 3, '8121': 3, '8221': 3, '8321': 3, '9111': 3, '9211': 3, '9411': 3,
        '3111': 3, '3211': 3, '3311': 3, '3411': 3, '7111': 3, '7211': 3, '7411': 3, '6111': 3, '6211': 3, '6411': 3,
        '10311': 3, '10411': 3,
        '9121': 2, '9221': 2, '9421': 2, '6121': 2, '6221': 2, '6421': 2, '7121': 2, '7221': 2, '7421': 2, '3121': 2, 
        '3221': 2, '3321': 2, '3421': 2, '8411': 2, '8421': 2, '10321': 2, '10421': 2,
        '2111': 1, '2121': 1, '2211': 1, '2221': 1, '2311': 1, '2321': 1, '2411': 1, '2421': 1,
        '1111': 1, '1121': 1, '1211': 1, '1221': 1, '1311': 1, '1321': 1, '1411': 1, '1421': 1,
        '5110': -1, '5120': -1, '5210': -1, '5220': -1, '5310': -1, '5320': -1, '5410': -1, '5420': -1,
        '8110': -1, '8120': -1, '8210': -1, '8220': -1, '8310': -1, '8320': -1, '8410': -1, '8420': -1,
        '1110': -2, '1120': -2, '1210': -2, '1220': -2, '1310': -2, '1320': -2, '1410': -2, '1420': -2,
        '2110': -2, '2120': -2, '2210': -2, '2220': -2, '2310': -2, '2320': -2, '2410': -2, '2420': -2,
        '10110': -2, '10120': -2, '10210': -2, '10220': -2, '10310': -2, '10320': -2, '10410': -2, '10420': -2,
        '3110': -2, '3120': -2, '3210': -2, '3220': -2, '3310': -2, '3320': -2, '3410': -2, '3420': -2,
        '4110': -2, '4120': -2, '4210': -2, '4220': -2, '4310': -2, '4320': -2, '4410': -2, '4420': -2,
        '9110': -2, '9120': -2, '9210': -2, '9220': -2, '9310': -2, '9320': -2, '9410': -2, '9420': -2,
        '7110': -3, '7120': -3, '7210': -3, '7220': -3, '7310': -3, '7320': -3, '7410': -3, '7420': -3,
        '6110': -3, '6120': -3, '6210': -3, '6220': -3, '6310': -3, '6320': -3, '6410': -3, '6420': -3
    }
    
    df['place_season_condition_type_categori'] = df['place_season_condition_type'].map(conversion_map).fillna(-10000).astype(int)
    df['place_season_condition_type_categori'] = df['place_season_condition_type_categori'].replace(-10000, np.nan)


    
    
    df["place"] = df["place"].astype(int)
    df["course_len"] = df["course_len"].astype(int)
    
    df["race_type"] = df["race_type"].astype(int)
    
    #芝のコース詳細データ
    classification_map = {
        9: {  # 阪神
            "内": [1200, 1400, 2000, 2200, 3000, 3200],
            "外": [1400, 1600, 1800, 2400, 2600],
        },
        8: {  # 京都
            "内": [1100, 1200],
            "外": [1800, 2200, 2400, 3000, 3200],
            "内外": [1400, 1600, 2000],
        },
        4: {  # 新潟
            "内": [1200, 1400, 2000, 2200, 2400],
            "外": [1400, 1600, 1800, 2000, 3000, 3200],
            "内外": [],
            "直線": [1000],
        },
        6: {  # 中山
            "内": [1800, 2000, 2500, 3600],
            "外": [1200, 1600, 2200, 2600, 4000],
            "内外": [3200],
        }
    }
    """
    阪神内
    1,200m、1,400m、2,000m
    2,200m、3,000m、3,200m
    
    阪神外
    1,400m、1,600m、1,800m
    2,400m、2,600m
    
    京都内
    1,100m(内)、1,200m(内)
    
    京都内・外
    1,400m(内・外)、1,600m(内・外)
    2,000m(内・外)
    
    京都外
    1,800m(外)
    2,200m(外)、2,400m(外)
    3,000m(外)、3,200m(外)
    
    新潟直線
    1,000m
    
    新潟内
    1,200m、1,400m
    2,000m、2,200m、2,400m
    
    新潟外
    1,400m、1,600m
    1,800m、2,000m、3,000m、3,200m
    
    中山内外
    3,200m（外・内）
    
    中山内
    1,800m（内）、2,000m（内）
    2,500m（内）、3,600m（内）
    
    中山外
    1,200m（外）、1,600m（外）
    2,200m（外）、2,600m（外）、4,000m（外）
    """
    # 修正された分類ロジック
    def classify_place_course(row):
        place = row["place"]
        course_len = row["course_len"]
        race_type = row["race_type"]  # 0: ダート, 1: 芝, 2: 障害
        
        # ダートまたは障害の場合は NaN を返す
        if race_type in [0, 2]:
            return None
        
        place_map = classification_map.get(place, {})
        for category, distances in place_map.items():
            if course_len in distances:
                category_index = list(place_map.keys()).index(category) + 1
                return int(f"{place}{category_index:02d}")  # 整数で返す
        
        # 分類ルールに該当しない場合は競馬場データをそのまま返す
        return place
    df["place_course_category"] = df.apply(classify_place_course, axis=1)
    #-1を欠損値として扱いintに直す
    df["place_course_category"] = df["place_course_category"].fillna(-1).astype(int)
    
    
    """
    コース別タフ度（芝）
    軽い3	新潟	東京	京都外	阪神外
    中間2	京都内	小倉	福島	中京
    タフ1	阪神内	函館	中山	札幌
    """
    # コース別タフ度（芝）_変換ルール
    conversion_map = {
        401: 3, 402: 3, 404: 3, 5: 3, 802: 3, 902: 3,
        801: 2, 803: 2, 10: 2, 3: 2, 7: 2,
        901: 1, 2: 1, 6: 1, 601: 1, 602: 1, 1: 1
    }
    # 新しい列を追加
    df["place_course_tough"] = df["place_course_category"].map(conversion_map).fillna(-1).astype(int)
    
    """
    競馬場	直線/m	カーブ	ゴール前
    新潟直線	1000	nan	平坦
    新潟外	658.7	急	平坦
    東京	525.9	複合	緩坂
    阪神外	473.6	複合	急坂
    中京	412.5	スパ	急坂
    京都外	403.7	複合	平坦
    京都内・外	345	複合	平坦
    新潟内	358.7	急	平坦
    阪神内	356.5	複合	急坂
    京都内	328.4	複合	平坦
    中山内	310	小回	急坂
    中山外	310	複合	急坂
    中山内外	310	複合	急坂
    札幌	266.1	大回	平坦
    函館	262.1	小回	平坦
    福島	292	小スパ	平坦
    小倉	293	小スパ	平坦
    """
    
    # # 競馬場カテゴリと対応する直線、カーブ、ゴール前情報
    # conversion_map = {
    #     401: {"直線": 1000, "カーブ": "直線", "ゴール前": "平坦"},
    #     402: {"直線": 658.7, "カーブ": "急", "ゴール前": "平坦"},
    #     404: {"直線": 525.9, "カーブ": "複合", "ゴール前": "緩坂"},
    #     902: {"直線": 473.6, "カーブ": "複合", "ゴール前": "急坂"},
    #     801: {"直線": 412.5, "カーブ": "スパ", "ゴール前": "急坂"},
    #     802: {"直線": 403.7, "カーブ": "複合", "ゴール前": "平坦"},
    #     803: {"直線": 345, "カーブ": "複合", "ゴール前": "平坦"},
    #     901: {"直線": 358.7, "カーブ": "急", "ゴール前": "平坦"},
    #     601: {"直線": 310, "カーブ": "小回", "ゴール前": "急坂"},
    #     602: {"直線": 310, "カーブ": "複合", "ゴール前": "急坂"},
    #     603: {"直線": 310, "カーブ": "複合", "ゴール前": "急坂"},
    #     1: {"直線": 266.1, "カーブ": "大回", "ゴール前": "平坦"},
    #     2: {"直線": 262.1, "カーブ": "小回", "ゴール前": "平坦"},
    #     3: {"直線": 292, "カーブ": "小スパ", "ゴール前": "平坦"},
    #     10: {"直線": 293, "カーブ": "小スパ", "ゴール前": "平坦"}
    # }

    # # カーブとゴール前を数値に変換する辞書
    # curve_map = {
    #     "急": 1,
    #     "小回": 2,
    #     "小スパ": 3,
    #     "スパ": 4,
    #     "複合": 5,
    #     "直線": 0  # Noneのまま
    # }
    
    # goal_map = {
    #     "平坦": 1,        
    #     "緩坂": 2,
    #     "急坂": 3
    #     # None: None  # Noneのまま
    # }
    conversion_map = {
        404: {"直線": 1000, "カーブ": 4, "ゴール前": 1},
        402: {"直線": 658.7, "カーブ": 1, "ゴール前": 1},
        5: {"直線": 525.9, "カーブ": 5, "ゴール前": 2},
        902: {"直線": 473.6, "カーブ": 5, "ゴール前": 3},
        7: {"直線": 412.5, "カーブ": 4, "ゴール前": 3},
        802: {"直線": 403.7, "カーブ": 5, "ゴール前": 1},
        803: {"直線": 345, "カーブ": 5, "ゴール前": 1},
        401: {"直線": 358.7, "カーブ": 1, "ゴール前": 1},
        901: {"直線": 356.5, "カーブ": 5, "ゴール前": 3},    
        801: {"直線": 328.4, "カーブ": 5, "ゴール前": 1},    
        601: {"直線": 310, "カーブ": 2, "ゴール前": 3},
        602: {"直線": 310, "カーブ": 5, "ゴール前": 3},
        603: {"直線": 310, "カーブ": 5, "ゴール前": 3},
        1: {"直線": 266.1, "カーブ": 5, "ゴール前": 1},
        2: {"直線": 262.1, "カーブ": 2, "ゴール前": 1},
        3: {"直線": 292, "カーブ": 3, "ゴール前": 1},
        10: {"直線": 293, "カーブ": 3, "ゴール前": 1}
    }

    # データフレームに変換情報を適用する関数
    def convert_course(row):
        place_code = row["place_course_category"]  # 競馬場の数値コード
        if place_code in conversion_map:
            # 直線、カーブ、ゴール前の情報を取得
            course_info = conversion_map[place_code]
            # 列名を変更
            return pd.Series({
                "goal_range": course_info["直線"], 
                "curve": course_info["カーブ"], 
                "goal_slope": course_info["ゴール前"]
            })
        else:
            return pd.Series({"goal_range": None, "curve": None, "goal_slope": None})
    
    # 競馬場カテゴリに基づく変換を追加
    df[['goal_range', 'curve', 'goal_slope']] = df.apply(convert_course, axis=1)
    
    df["goal_range_100"] = df["goal_range"]/100
    
    
    #競馬場_季節_芝ダート障害_長さ
    df["place_season_type_course_len"] = df["place_season_type"] + df["course_len"].astype(str)
    
    """
    ラップの緩急差が小さいコース
    中山芝1600
    阪神芝1400
    東京芝1400
    東京芝1600
    
    緩急のあるラップで脚を溜めやすいコース
    京都芝1200
    京都芝1400外
    中山(秋)芝1200
    中山芝2000
    阪神芝1200
    阪神芝1600外
    阪神芝1800外
    阪神芝2400外
    新潟芝1200、1400内
    札幌芝1200、1800、2000
    東京芝1800、2000、2400
    
    ロングスパート戦になりやすいコース
    東京芝1600、2400
    中山芝1800、2000、2200
    京都芝1600、1800、2200、2400外
    京都芝2000内
    阪神芝2000、2200内、2400外
    福島芝1800、2000、1700
    新潟芝1600、1800、2000外、2200内
    小倉芝1800、2000
    中京芝1600、2000、2200
    札幌芝1500、2000、2600
    
    
    失速ラップになりやすいコース
    中山芝1200
    福島芝1200
    小倉芝1200
    新潟芝1000
    函館芝1200、1800、2000
    中京芝1200、1400、1600、2000
    """
    
    # マッピングを定義
    mapping = {
        # ラップの緩急差が小さいコース → 4
        "6111600": 4, "6211600": 4, "6311600": 4, "6411600": 4,  # 中山(春/夏/秋/冬)芝1600
        "9111400": 4, "9211400": 4, "9311400": 4, "9411400": 4,  # 阪神(春/夏/秋/冬)芝1400
        "5111400": 4, "5211400": 4, "5311400": 4, "5411400": 4,  # 東京(春/夏/秋/冬)芝1400
        "5111600": 4, "5211600": 4, "5311600": 4, "5411600": 4,  # 東京(春/夏/秋/冬)芝1600
    
        # 緩急のあるラップで脚を溜めやすいコース → 3
        "8111200": 3, "8211200": 3, "8311200": 3, "8411200": 3,  # 京都(春/夏/秋/冬)芝1200
        "8111400": 3, "8211400": 3, "8311400": 3, "8411400": 3,  # 京都(春/夏/秋/冬)芝1400外
        "6311200": 3,  # 中山(秋)芝1200
        "6112000": 3, "6212000": 3, "6312000": 3, "6412000": 3,  # 中山(春/夏/秋/冬)芝2000
        "9111200": 3, "9211200": 3, "9311200": 3, "9411200": 3,  # 阪神(春/夏/秋/冬)芝1200
        "9111600": 3, "9211600": 3, "9311600": 3, "9411600": 3,  # 阪神(春/夏』/秋/冬)芝1600外
        "9111800": 3, "9211800": 3, "9311800": 3, "9411800": 3,  # 阪神(春/夏/秋/冬)芝1800外
        "9112400": 3, "9212400": 3, "9312400": 3, "9412400": 3,  # 阪神(春/夏/秋/冬)芝2400外
    
        # ロングスパート戦になりやすいコース → 2
        "5111600": 2, "5211600": 2, "5311600": 2, "5411600": 2,  # 東京(春/夏/秋/冬)芝1600
        "5112400": 2, "5212400": 2, "5312400": 2, "5412400": 2,  # 東京(春/夏/秋/冬)芝2400
        "6111800": 2, "6211800": 2, "6311800": 2, "6411800": 2,  # 中山(春/夏/秋/冬)芝1800
        "6112000": 2, "6212000": 2, "6312000": 2, "6412000": 2,  # 中山(春/夏/秋/冬)芝2000
        "6112200": 2, "6212200": 2, "6312200": 2, "6412200": 2,  # 中山(春/夏/秋/冬)芝2200
        "8111600": 2, "8211600": 2, "8311600": 2, "8411600": 2,  # 京都(春/夏/秋/冬)芝1600外
        "8111800": 2, "8211800": 2, "8311800": 2, "8411800": 2,  # 京都(春/夏/秋/冬)芝1800外
        "8112200": 2, "8212200": 2, "8312200": 2, "8412200": 2,  # 京都(春/夏/秋/冬)芝2200外
        "8112400": 2, "8212400": 2, "8312400": 2, "8412400": 2,  # 京都(春/夏/秋/冬)芝2400外
    
        # 失速ラップになりやすいコース → 1
        "6111200": 1, "6211200": 1, "6411200": 1,  # 中山(春/夏/冬)芝1200
        "3111200": 1, "3211200": 1, "3311200": 1, "3411200": 1,  # 福島(春/夏/秋/冬)芝1200
        "10111200": 1, "11111200": 1, "12111200": 1, "13111200": 1,  # 小倉(春/夏/秋/冬)芝1200
        "4111000": 1, "4211000": 1, "4311000": 1, "4411000": 1,  # 新潟(春/夏/秋/冬)芝1000
        "2111200": 1, "2211200": 1, "2311200": 1, "2411200": 1,  # 函館(春/夏/秋/冬)芝1200
        "7111200": 1, "7211200": 1, "7311200": 1, "7411200": 1,  # 中京(春/夏/秋/冬)芝1200
    }
    
    # `place_season_type_course_len` をマッピングに基づいて変換
    df["lap_type"] = df["place_season_type_course_len"].map(mapping)
    # マッピングできなかったものを NaN にする
    df['lap_type'] = df['lap_type'].fillna(-1)
    df['lap_type'] = df['lap_type'].astype(int)
    
    
    
    
    # 各列をint型に変換
    
    df["race_class"] = df["race_class"].astype(int)
    df["ground_state"] = df["ground_state"].astype(int)
    df["around"] = df["around"].fillna(3).astype(int)
    df["weather"] = df["weather"].astype(int)
    
    
    
    # df[['goal_range', 'curve', 'goal_slope']] = df[['goal_range', 'curve', 'goal_slope']].fillna(-10000).astype(int)
    # df["place_season_type_course_len"] = df["place_season_type_course_len"].fillna(-10000).astype(int)
    # df['lap_type'] = df['lap_type'].fillna(-10000).astype(int)
    
    
    df["course_len"] = df["course_len"].astype(int)
    df["season"] = df["season"].astype(int)
    df["place_season"] = df["place_season"].astype(int)
    df["place_season_type"] = df["place_season_type"].astype(int)
    df["kaisai_race"] = df["kaisai_race"].astype(int)
    df["kaisai_race_type"] = df["kaisai_race_type"].astype(int)
    df["day"] = df["day"].astype(int)
    df["day_type"] = df["day_type"].astype(int)
    df["kaisai_count"] = df["kaisai_count"].astype(int)
    df["kaisai_count_type"] = df["kaisai_count_type"].astype(int)
    df["place_season_day_type"] = df["place_season_day_type"].astype(int)
    df["day_condition"] = df["day_condition"].astype(int)
    df["place_season_condition_type"] = df["place_season_condition_type"].astype(int)
    
    # df['goal_range'] = df['goal_range'].fillna(-10000).astype(int)
    # df['curve'] = df['curve'].fillna(-10000).astype(int)
    # df['goal_slope'] = df['goal_slope'].fillna(-10000).astype(int)
    
    

    # 使用する列を選択
    df = df[
        [
            "race_id",
            "date",
            "race_type",
            "around",
            "course_len",
            "weather",
            "ground_state",
            "race_class",
            "place",
            
            "season",
            "place_season",
            "place_season_type",
            "kaisai_race",
            "kaisai_race_type",
            "day",
            "day_type",
            "kaisai_count",
            "kaisai_count_type",
            "place_season_day_type",
            "day_condition",
            "place_season_condition_type",
            'place_season_condition_type_categori',
            
            "place_course_category",
            "place_course_tough",
            'goal_range',
            'curve',
            'goal_slope',
            "place_season_type_course_len",
            "lap_type",
            "race_day_count",
            "race_date_day_count",

            "goal_range_100",


        ]
    ]
    df.to_csv(output_dir / output_filename, sep="\t", index=False)
    return df

def process_return_tables(  
    population_dir: Path = POPULATION_DIR,
    populaton_filename: str = "population.csv",
    input_dir: Path = RAWDF_DIR,
    input_filename: str = "return_tables.csv",
    output_dir: Path = OUTPUT_DIR,
    output_filename: str = "return_tables.pickle",
    ):
    #未加工の払い戻しーブルをinpuｔ_drから読み込んで加工し、output_dirに保存する
    population = pd.read_csv(population_dir / populaton_filename, sep="\t")
    # df = pd.read_csv(input_dir / input_filename, sep="\t", index_col=0).query(
    #     "race_id in @population['race_id']"
    # )
    population_race_ids = population['race_id'].tolist()
    # クエリでリストを直接使用
    df = pd.read_csv(input_dir / input_filename, sep="\t", index_col=0).query(
        "race_id in @population_race_ids"
    )
    
    df = (
        df[["0","1","2"]]
        .replace(" (-|→) ","-",regex = True)
        .replace(",","",regex= True)
        .apply(lambda x: x.str.split())
        .explode(["1","2"])
        .explode("0")
        .apply(lambda x: x.str.split("-"))
        .explode(["0","2"])
        )
    #列名の変更
    df.columns = ["bet_type","win_umaban","return"]
    # "枠連" を含む行を除外
    df = df.query("bet_type != '枠連'").reset_index()
    #払い戻しのreturnを整数型に変換
    df["return"] = df["return"].astype(int)
    #pickleだとｐythonのオブジェクトとして保存してくれる（文ではなく）,リスト型を維持
    df.to_pickle(output_dir / output_filename)
    return df





def process_horse_results(
    population_dir: Path = POPULATION_DIR,
    populaton_filename: str = "population.csv",
    input_dir: Path = RAWDF_DIR,
    output_dir: Path = OUTPUT_DIR,
    input_filename: str = "horse_results.csv",
    output_filename: str = "horse_results.csv",
    race_type_mapping: dict = race_type_mapping,
    weather_mapping: dict = weather_mapping,
    ground_state_mapping: dict = ground_state_mapping,
    race_class_mapping: dict = race_class_mapping,
    place_mapping: dict = place_mapping,
) -> pd.DataFrame:
    """
    未加工の馬の過去成績テーブルをinput_dirから読み込んで加工し、
    output_dirに保存する関数。
    """
    population = pd.read_csv(population_dir / populaton_filename, sep="\t")
    # df = pd.read_csv(input_dir / input_filename, sep="\t").query(
    #     "horse_id in @population['horse_id']"
    # )
    population_horse_ids = population['horse_id'].tolist()
    
    # クエリでリストを直接使用
    df = pd.read_csv(input_dir / input_filename, sep="\t").query(
        "horse_id in @population_horse_ids"
    )

    
    df["rank"] = pd.to_numeric(df["着順"], errors="coerce")
    df.dropna(subset=["rank"], inplace=True)
    df["date"] = pd.to_datetime(df["日付"])
    df["weather"] = df["天気"].map(weather_mapping)
    df["race_type"] = df["距離"].str[0].map(race_type_mapping)
    df = df[df["race_type"] != 2]
    df["course_len"] = df["距離"].str.extract(r"(\d+)").astype(int)
    df["ground_state"] = df["馬場"].map(ground_state_mapping)
    df["rank_diff"] = df["着差"].map(lambda x: 0 if x < 0 else x)
    df["prize"] = df["賞金"].fillna(0)
    df["umaban"] = df["馬番"].astype(int)
    df["impost"] = df["斤量"].astype(float)
    df["wakuban"] = df["枠番"].fillna(0).astype(int)

    
    df["nobori"] = df["上り"].astype(float)
    
    regex_race_class = "|".join(race_class_mapping)
    df["race_class"] = (
        df["レース名"].str.extract(rf"({regex_race_class})")[0].map(race_class_mapping)
    )
    df["time"] = pd.to_datetime(df["タイム"], format="%M:%S.%f", errors="coerce")
    df["time"] = (
        df["time"].dt.minute * 60
        + df["time"].dt.second
        + df["time"].dt.microsecond / 1000000
    )
    df["win"] = (df["rank"] == 1).astype(int)
    df["rentai"] = (df["rank"] <= 2).astype(int)
    df["show"] = (df["rank"] <= 3).astype(int)
    df["place"] = df["開催"].str.extract(r"(\D+)")[0].map(place_mapping)
    df.dropna(subset=["place"], inplace=True)    
    df.rename(columns={"頭数": "n_horses"}, inplace=True)

    pace_cols = df['ペース'].str.split('-', expand=True)
    pace_cols.columns = [f'pace_{i+1}' for i in range(pace_cols.shape[1])]
    
    # データ型変換と型確認
    pace_cols = pace_cols.apply(pd.to_numeric, errors='coerce').astype('float64')

    # 元のデータフレームと結合
    df = pd.concat([df, pace_cols], axis=1)
    
    # pace_2 - pace_1 の計算
    df['pace_diff'] = df['pace_2'] - df['pace_1']
    # course_len の上2桁を取得して奇数かどうかを判定
    df['course_len_prefix'] = df['course_len'].astype(str).str[:2].astype(float)  # 上2桁を取得
    
    # 奇数の場合に pace_diff を NaN に設定
    df.loc[df['course_len_prefix'] % 2 != 0, 'pace_diff'] = None
    # pace_diff をカテゴリに分ける関数
    def categorize_pace_diff(value):
        if value < -1.0:
            return 4  # ハイペース
        elif -1.0 <= value <= 0.0:
            return 3  # ハイミドルペース
        elif 0.0 < value <= 1.0:
            return 2  # スローミドルペース
        elif value > 1.0:
            return 1  # スローペース
        return None  # 値が NaN の場合など
        
    
    # 新しい列にカテゴリを適用
    df["pace_category"] = df["pace_diff"].apply(categorize_pace_diff)    

    # NoneをNaNに置き換え
    # df = df.where(pd.notnull(df), np.nan)
    df["place"] = df["place"].astype(int)
    df["race_class"] = df["race_class"].astype(int)
    df["ground_state"] = df["ground_state"].astype(int)
    df["weather"] = df["weather"].astype(int)   

        
    """
    ホースリザルトは、シーズンデータ入れなくてもいいかも
    年齢は入ってるからそれで
    
    
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
    # race_gradeの作成
    def calculate_race_grade(row):
        race_class = row['race_class']
        
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

    
        
    # コーナー通過順を分割して列を作成
    corner_cols = df['通過'].str.split('-', expand=True)
    corner_cols.columns = [f'corner_{i+1}' for i in range(corner_cols.shape[1])]
    # オブジェクト型のデータを整数型に変換する
    corner_cols = corner_cols.apply(pd.to_numeric, errors='coerce').astype('Int64')  
    # nullable int型を指定
    # 元のデータフレームと結合
    df = pd.concat([df, corner_cols], axis=1)

    # NoneをNaNに置き換え
    df = df.where(pd.notnull(df), np.nan)







    

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
        if pd.notna(row[final_corner]) and row[final_corner] <= 4:
            return 2  # 先行
    
        # 差し判定：出走頭数に応じて
        if row['n_horses'] >= 8 and pd.notna(row[final_corner]) and row[final_corner] <= (row['n_horses'] * 2) // 3:
            return 3  # 差し
        elif row['n_horses'] < 8:
            return 4  # 差しなし、追込
    
        # 追込判定：上記のいずれにも該当しない
        return 4  # 追込
    
    # dfの各レースについて脚質を決定
    df['race_position'] = df.apply(determine_race_position, axis=1)
    df = df.where(pd.notnull(df), np.nan)

    






    


    # rank / n_horses の特徴量を作成（欠損値を含む行はNaNに設定）
    df["rank_per_horse"] = df["rank"].where(df["rank"].notna(), np.nan) / df["n_horses"].where(df["n_horses"].notna(), np.nan)
    
    # corner_1 / n_horses の特徴量を作成（欠損値を含む行はNaNに設定）
    df["corner_1_per_horse"] = df["corner_1"].where(df["corner_1"].notna(), np.nan) / df["n_horses"].where(df["n_horses"].notna(), np.nan)
    
    df["corner_2_per_horse"] = df["corner_2"].where(df["corner_2"].notna(), np.nan) / df["n_horses"].where(df["n_horses"].notna(), np.nan)
    df["corner_3_per_horse"] = df["corner_3"].where(df["corner_3"].notna(), np.nan) / df["n_horses"].where(df["n_horses"].notna(), np.nan)
    df["corner_4_per_horse"] = df["corner_4"].where(df["corner_4"].notna(), np.nan) / df["n_horses"].where(df["n_horses"].notna(), np.nan)
    df["time_courselen"] = df["time"] / df["course_len"]


    # NoneをNaNに置き換え
    df = df.where(pd.notnull(df), np.nan)
    # 使用する列を選択
    df = df[
        [
            "horse_id",
            "rank_per_horse",
            "date",
            "rank",
            "prize",
            "rank_diff",
            "umaban",
            "wakuban",
            "weather",
            "race_type",
            "course_len",
            "impost",
            "ground_state",
            "race_class",
            "n_horses",
            "time",
            "time_courselen",
            "nobori",
            "win",
            "rentai",
            "show",
            "place",
            "corner_1_per_horse",
            "corner_2_per_horse",
            "corner_3_per_horse",
            "corner_4_per_horse",
            'race_position',
            'race_grade',
            'race_grade_scaled',
            'pace_diff',
            "pace_category",
        ] + list(corner_cols.columns) + list(pace_cols.columns)
    ]
    df.to_csv(output_dir / output_filename, sep="\t", index=False)
    return df


    
def process_jockey_leading(
    input_dir: Path = RAWDF_DIR,
    output_dir: Path = OUTPUT_DIR,
    input_filename: str = "jockey_leading.csv",
    output_filename: str = "jockey_leading.csv",
) -> pd.DataFrame:
    """
    未加工の騎手成績テーブルをinput_dirから読み込んで加工し、
    output_dirに保存する関数。
    """
    df = pd.read_csv(input_dir / input_filename, sep="\t")
    df["year"] = df["page_id"].str[:4].astype(int)
    df["n_races"] = df["1着"] + df["2着"] + df["3着"] + df["着外"]
    df["winrate_graded"] = df["重賞_勝利"] / df["重賞_出走"]
    df["winrate_special"] = df["特別_勝利"] / df["特別_出走"]
    df["winrate_ordinal"] = df["平場_勝利"] / df["平場_出走"]
    df["winrate_turf"] = df["芝_勝利"] / df["芝_出走"]
    df["winrate_dirt"] = df["ダート_勝利"] / df["ダート_出走"]
    df.rename(
        columns={
            "順位": "rank",
            "重賞_出走": "n_races_graded",
            "特別_出走": "n_races_special",
            "平場_出走": "n_races_ordinal",
            "芝_出走": "n_races_turf",
            "ダート_出走": "n_races_dirt",
            "勝率": "winrate",
            "連対率": "placerate",
            "複勝率": "showrate",
            "収得賞金(万円)": "prize",
        },
        inplace=True,
    )
    # 使用する列を選択
    df = df[
        [
            "jockey_id",
            "year",
            "rank",
            "n_races",
            "n_races_graded",
            "winrate_graded",
            "n_races_special",
            "winrate_special",
            "n_races_ordinal",
            "winrate_ordinal",
            "n_races_turf",
            "winrate_turf",
            "n_races_dirt",
            "winrate_dirt",
            "winrate",
            "placerate",
            "showrate",
            "prize",
        ]
    ]
    df.to_csv(output_dir / output_filename, sep="\t", index=False)
    return df


def process_trainer_leading(
    input_dir: Path = RAWDF_DIR,
    output_dir: Path = OUTPUT_DIR,
    input_filename: str = "trainer_leading.csv",
    output_filename: str = "trainer_leading.csv",
) -> pd.DataFrame:
    """
    未加工の騎手成績テーブルをinput_dirから読み込んで加工し、output_dirに保存する関数。
    """
    df = pd.read_csv(input_dir / input_filename, sep="\t")
    df["year"] = df["page_id"].str[:4].astype(int)
    df["n_races"] = df["1着"] + df["2着"] + df["3着"] + df["着外"]
    df["winrate_graded"] = df["重賞_勝利"] / df["重賞_出走"]
    df["winrate_special"] = df["特別_勝利"] / df["特別_出走"]
    df["winrate_ordinal"] = df["平場_勝利"] / df["平場_出走"]
    df["winrate_turf"] = df["芝_勝利"] / df["芝_出走"]
    df["winrate_dirt"] = df["ダート_勝利"] / df["ダート_出走"]
    df.rename(
        columns={
            "順位": "rank",
            "重賞_出走": "n_races_graded",
            "特別_出走": "n_races_special",
            "平場_出走": "n_races_ordinal",
            "芝_出走": "n_races_turf",
            "ダート_出走": "n_races_dirt",
            "勝率": "winrate",
            "連対率": "placerate",
            "複勝率": "showrate",
            "収得賞金(万円)": "prize",
        },
        inplace=True,
    )
    # 使用する列を選択
    df = df[
        [
            "trainer_id",
            "year",
            "rank",
            "n_races",
            "n_races_graded",
            "winrate_graded",
            "n_races_special",
            "winrate_special",
            "n_races_ordinal",
            "winrate_ordinal",
            "n_races_turf",
            "winrate_turf",
            "n_races_dirt",
            "winrate_dirt",
            "winrate",
            "placerate",
            "showrate",
            "prize",
        ]
    ]
    df.to_csv(output_dir / output_filename, sep="\t", index=False)
    return df


def process_peds(
    population_dir: Path = POPULATION_DIR,
    populaton_filename: str = "population.csv",
    input_dir: Path = RAWDF_DIR,
    output_dir: Path = OUTPUT_DIR,
    input_filename: str = "peds.csv",
    output_filename: str = "peds.csv",
):
    """
    未加工の血統テーブルをinput_dirから読み込んで加工し、output_dirに保存する関数。
    """
    population = pd.read_csv(population_dir / populaton_filename, sep="\t")
    # df = pd.read_csv(input_dir / input_filename, sep="\t").query(
    #     "horse_id in @population['horse_id']"
    # )
    population_horse_ids = population['horse_id'].tolist()
    
    # クエリでリストを直接使用
    df = pd.read_csv(input_dir / input_filename, sep="\t").query(
        "horse_id in @population_horse_ids"
    )

    
    # 種牡馬とBMSに絞る
    df = df[["horse_id", "ped_0", "ped_32"]]
    df.columns = [["horse_id", "sire_id", "bms_id"]]
    df.to_csv(output_dir / output_filename, sep="\t", index=False)
    return df


def process_sire_leading(
    input_dir: Path = RAWDF_DIR,
    output_dir: Path = OUTPUT_DIR,
    input_filename: str = "sire_leading.csv",
    output_filename: str = "sire_leading.csv",
    race_type_mapping: dict = race_type_mapping,
    id_col: str = "sire_id",
) -> pd.DataFrame:
    """
    未加工の種牡馬リーディングテーブルをinput_dirから読み込んで加工し、
    output_dirに保存する関数。
    """
    df = pd.read_csv(input_dir / input_filename, sep="\t")
    df["year"] = df["page_id"].str[:4].astype(int)
    key_cols = ["page_id", id_col, "year"]
    target_cols = [
        "芝_出走",
        "芝_勝利",
        "ダート_出走",
        "ダート_勝利",
        "平均距離(芝)",
        "平均距離(ダ)",
    ]
    df = df[key_cols + target_cols]
    df = df.melt(
        id_vars=key_cols,
        value_vars=target_cols,
        var_name="category",
        value_name="value",
    )
    splitted_df = (
        df["category"]
        .str.replace("平均距離(芝)", "芝_平均距離")
        .str.replace("平均距離(ダ)", "ダート_平均距離")
        .str.split("_", expand=True)
    )
    df["race_type"] = splitted_df[0]
    df["category"] = splitted_df[1]
    df["race_type"] = df["race_type"].str.replace("ダート", "ダ").map(race_type_mapping)
    df = df.pivot_table(
        index=key_cols + ["race_type"], columns="category", values="value"
    ).reset_index()
    df["winrate"] = df["勝利"] / df["出走"]
    df.rename(
        columns={"出走": "n_races", "勝利": "n_wins", "平均距離": "course_len"},
        inplace=True,
    )
    df.to_csv(output_dir / output_filename, sep="\t", index=False)
    return df
