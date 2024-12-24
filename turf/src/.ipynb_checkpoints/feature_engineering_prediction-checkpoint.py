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


# commonディレクトリのパス
COMMON_DATA_DIR = Path("..", "..", "common", "data")
POPULATION_DIR = COMMON_DATA_DIR / "prediction_population"
MAPPING_DIR = COMMON_DATA_DIR / "mapping"


DATA_DIR = Path("..", "data")
OLD_POPULATION_DIR = DATA_DIR / "00_population"
INPUT_DIR = DATA_DIR / "01_preprocessed"
OUTPUT_DIR = DATA_DIR / "02_features"
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

#出走馬の過去のレースとかのデータを、学習したデータに合わせる

class PredictionFeatureCreator:
    def __init__(
        self,
        population_dir: Path = POPULATION_DIR,
        population_filename: str = "population.csv",
        
        old_population_dir: Path = OLD_POPULATION_DIR,
        old_population_filename: str = "population.csv",
        old_results_filename: str = "results.csv",
        old_race_info_filename: str = "race_info.csv",
        old_horse_results_filename: str = "horse_results.csv",
        
        input_dir: Path = INPUT_DIR,
        
        horse_results_filename: Path = "horse_results_prediction.csv",
        jockey_leading_filename: Path = "jockey_leading.csv",
        trainer_leading_filename: Path = "trainer_leading.csv",
        peds_filename: str = "peds_prediction.csv",
        sire_leading_filename: str = "sire_leading.csv",
        output_dir: Path = OUTPUT_DIR,
        output_filename: str = "features_prediction.csv",


        old_results_condition_filename: str = "results_prediction.csv",
        old_race_info_condition_filename: str = "race_info_prediction.csv",   
        bms_leading_filename: str = "bms_leading.csv",     
    ):
        self.population = pd.read_csv(population_dir / population_filename, sep="\t")
        self.old_results = pd.read_csv(input_dir / old_results_filename, sep="\t")
        self.old_race_info = pd.read_csv(input_dir / old_race_info_filename, sep="\t")
        self.old_population = pd.read_csv(old_population_dir / old_population_filename, sep="\t")
        self.old_horse_results = pd.read_csv(input_dir / old_horse_results_filename, sep="\t")

        self.old_results_condition = pd.read_csv(input_dir / old_results_condition_filename, sep="\t")
        self.old_race_info_condition = pd.read_csv(input_dir / old_race_info_condition_filename, sep="\t")
        
        
        self.horse_results = pd.read_csv(input_dir / horse_results_filename, sep="\t")
        self.jockey_leading = pd.read_csv(input_dir / jockey_leading_filename, sep="\t")
        self.trainer_leading = pd.read_csv(
            input_dir / trainer_leading_filename, sep="\t"
        )
        self.peds = pd.read_csv(input_dir / peds_filename, sep="\t")
        self.sire_leading = pd.read_csv(input_dir / sire_leading_filename, sep="\t")
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.htmls = {}
        self.agg_horse_per_group_cols_dfs = {}

        self.bms_leading = pd.read_csv(input_dir / bms_leading_filename, sep="\t")       



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
        出走馬が確定した時点で先に実行しておいても良い。
        """
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
                        "course_len",
                        "race_grade",
                        "n_horses",
                        
                        #"time",
                        
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

    
    def fetch_shutuba_page_html(self, race_id: str) -> None:
        """
        レースidを指定すると、出馬表ページのhtmlをスクレイピングする関数。
        """
        print("fetching shutuba page html...")
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        # chrome driverをインストール
        driver_path = ChromeDriverManager().install()
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
            driver.implicitly_wait(10000)
            driver.get(url)
            self.htmls[race_id] = driver.page_source
        print("fetching shutuba page html...comp")

    def fetch_results(
        self, race_id: str, html: str, sex_mapping: dict = sex_mapping
    ) -> None:
        """
        出馬表ページのhtmlを受け取ると、
        「レース結果テーブル」を取得して、学習時と同じ形式に前処理する関数。
        """
        df = pd.read_html(StringIO(html))[0]
        df.columns = df.columns.get_level_values(1)
        soup = BeautifulSoup(html, "lxml").find("table", class_="Shutuba_Table")
        horse_id_list = []
        a_list = soup.find_all("a", href=re.compile(r"/horse/"))
        for a in a_list:
            horse_id = re.findall(r"\d{10}", a["href"])[0]
            horse_id_list.append(int(horse_id))
        df["horse_id"] = horse_id_list
        jockey_id_list = []
        a_list = soup.find_all("a", href=re.compile(r"/jockey/"))
        for a in a_list:
            jockey_id = re.findall(r"\d{5}", a["href"])[0]
            jockey_id_list.append(int(jockey_id))
        df["jockey_id"] = jockey_id_list
        trainer_id_list = []
        a_list = soup.find_all("a", href=re.compile(r"/trainer/"))
        for a in a_list:
            trainer_id = re.findall(r"\d{5}", a["href"])[0]
            trainer_id_list.append(int(trainer_id))
        df["trainer_id"] = trainer_id_list
        # 前処理
        df["wakuban"] = df.iloc[:, 0].astype(int)
        df["umaban"] = df.iloc[:, 1].astype(int)
        df["sex"] = df.iloc[:, 4].str[0].map(sex_mapping)
        df["age"] = df.iloc[:, 4].str[1:].astype(int)
        df["impost"] = df.iloc[:, 5].astype(float)
        # df["weight"] = df.iloc[:, 8].str.extract(r"(\d+)").astype(int)
        # df["weight_diff"] = df.iloc[:, 8].str.extract(r"\((.+)\)").astype(int)
        df["weight"] = df.iloc[:, 8].astype(str).str.extract(r"(\d+)").astype(float)
        df["weight_diff"] = df.iloc[:, 8].astype(str).str.extract(r"\((.+)\)").astype(float)

        df = df[df.iloc[:, 9] != '--']
        df["tansho_odds"] = df.iloc[:, 9].astype(float)
        df["popularity"] = df.iloc[:, 10].astype(int)
        df["race_id"] = int(race_id)
        df["n_horses"] = df.groupby("race_id")["race_id"].transform("count")
        
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

        df = result_df
        
        # 使用する列を選択
        df = df[
            [
                "race_id",
                "horse_id",
                "jockey_id",
                "trainer_id",
                # "rank", 
                # "rank_per_horse",                
                # "time",
                # "nobori",
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
                "mean_age",
                "median_age",
                "mean_age_kirisute", 
            ] 
        ]

        # df = df.astype({col: 'float32' for col in df.select_dtypes('float64').columns})
        # df = df.astype({col: 'int32' for col in df.select_dtypes('int64').columns})
        
        self.results = df

    def fetch_race_info(
        self,
        race_id: str,
        date_content_a: str,
        html: str,
        race_type_mapping: dict = race_type_mapping,
        around_mapping: dict = around_mapping,
        weather_mapping: dict = weather_mapping,
        ground_state_mapping: dict = ground_state_mapping,
        race_class_mapping: dict = race_class_mapping,
    ):
        """
        出馬表ページのhtmlを受け取ると、
        「レース情報テーブル」を取得して、学習時と同じ形式に前処理する関数。
        """
        info_dict = {}
        info_dict["race_id"] = int(race_id)
        soup = BeautifulSoup(html, "lxml").find("div", class_="RaceList_Item02")
        title = soup.find("h1").text.strip()
        divs = soup.find_all("div")
        div1 = divs[0].text.replace(" ", "")
        info1 = re.findall(r"[\w:]+", div1)
        info_dict["race_type"] = race_type_mapping[info1[1][0]]
        info_dict["around"] = (
            around_mapping[info1[2][0]] if info_dict["race_type"] != 2 else np.nan
        )
        info_dict["course_len"] = int(re.findall(r"\d+", info1[1])[0])
        info_dict["weather"] = weather_mapping[re.findall(r"天候:(\w+)", div1)[0]]
        info_dict["ground_state"] = ground_state_mapping[
            re.findall(r"馬場:(\w+)", div1)[0]
        ]
        # レース階級情報の取得
        regex_race_class = "|".join(race_class_mapping)
        race_class_title = re.findall(regex_race_class, title)
        # タイトルからレース階級情報が取れない場合
        race_class = re.findall(regex_race_class, divs[1].text)
        if len(race_class_title) != 0:
            info_dict["race_class"] = race_class_mapping[race_class_title[0]]
        elif len(race_class) != 0:
            info_dict["race_class"] = race_class_mapping[race_class[0]]
        else:
            info_dict["race_class"] = None
        info_dict["place"] = int(race_id[4:6])

            # 渡された日付をdatetimeに変換して格納
        try:
            info_dict["date"] = pd.to_datetime(date_content_a, format="%Y年%m月%d日")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_content_a}. Expected '%Y年%m月%d日'.")
        

        # try:
        #     # div1 から日付を正規表現で抽出
        #     date_text = re.findall(r"日付:(\d{4}年\d{2}月\d{2}日)", div1)[0]
        #     info_dict["date"] = pd.to_datetime(date_text, format="%Y年%m月%d日", errors="coerce")  # 直接日付に変換
        # except IndexError:
        #     info_dict["date"] = pd.NaT  # 日付情報が取得できない場合はNaTを設定
                

        
        # info_dictをDataFrameに変換
        df = pd.DataFrame(info_dict, index=[0])

        # df = df[df["race_type"] != 2]
        
        # place列のNaNを削除
        df.dropna(subset=["place"], inplace=True)
        df["place"] = df["place"].astype(int)
        df["race_class"] = df["race_class"].astype(int)
        df["ground_state"] = df["ground_state"].astype(int)
        df["around"] = df["around"].fillna(3).astype(int)
        df["weather"] = df["weather"].astype(int)       
        # df = df.astype({col: 'float32' for col in df.select_dtypes('float64').columns})

            # 年、月、日をそれぞれ抽出
        df["date_year"] = df["date"].dt.year
        df["date_month"] = df["date"].dt.month
        df["date_day"] = df["date"].dt.day

    
        df["date_year"] = df["date_year"] - 1
        print(df["date_year"])
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
        
        
        df["race_day_count"] = df['race_id'].astype(str).str[-2:]
        
        
        df["race_date_day_count"] = df["custom_date_value"].astype(str) + df["race_day_count"]
        
        # 必要に応じて中間列を削除
        df = df.drop(columns=["month_cumulative_days"])  # 中間列が不要な場合
        df = df.drop(columns=["date_year", "date_month","date_day","custom_date_value"])
        df["race_day_count"].astype(int)
        df["race_date_day_count"].astype(int)

    
        # df = df.astype({col: 'int32' for col in df.select_dtypes('int64').columns})     
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
        df['place_season_condition_type_categori'] = df['place_season_condition_type'].map(conversion_map).fillna(-100).astype(int)
        df['place_season_condition_type_categori'] = df['place_season_condition_type_categori'].replace(-100, np.nan)
    
    
        
        
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
        
        
        
        df[['goal_range', 'curve', 'goal_slope']] = df[['goal_range', 'curve', 'goal_slope']].fillna(-1).astype(int)
        df["place_season_type_course_len"] = df["place_season_type_course_len"].fillna(-1).astype(int)
        df['lap_type'] = df['lap_type'].fillna(-1).astype(int)
        
        
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

        # 数値以外の値をNaNに変換
        df["race_day_count"] = pd.to_numeric(df["race_day_count"], errors="coerce").fillna(0).astype(int)
        df["race_date_day_count"] = pd.to_numeric(df["race_date_day_count"], errors="coerce").fillna(0).astype(int)
        
                
        df['goal_range'] = df['goal_range'].fillna(-1).astype(int)
        df['curve'] = df['curve'].fillna(-1).astype(int)
        df['goal_slope'] = df['goal_slope'].fillna(-1).astype(int)
        
        df.drop(columns=["date"], inplace=True)  # inplace=Trueを使うと元のdfが更新される


    
            
        # race_infoに設定
        self.race_info_before = df

        # df.dropna(subset=["place"], inplace=True)        
        # self.race_info = pd.DataFrame(info_dict, index=[0])


        
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
        # latest_df = latest_df[["race_id", "horse_id", "interval"]]  # リストで列を選択
        merged_df = merged_df.merge(
            latest_df[["race_id", "horse_id", "interval"]], 
            on=["race_id", "horse_id"], 
            how="left")

        # merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
        # merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})
        
        # 結果をインスタンス変数に保存（変数名を変更）
        self.agg_interval_df = merged_df
        print("running agg_interval()...comp")


        
    def agg_horse_per_course_len(
        self, n_races: list[int] = [1,  3, 5, 10]
    ) -> None:
        """
        直近nレースの馬の過去成績を距離・race_typeごとに集計し、相対値に変換する関数。
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
        n_races: list[int] = [1,  3, 5, 10],
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
            df[["race_id", "horse_id","date","wakuban_race_type", "wakuban_around","umaban_race_type","umaban_around", "sin_date_sex", "cos_date_sex"]],
            on=["race_id", "date","horse_id"],
            how="left"
        )

        # merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
        # merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})
        
        self.agg_cross_features_df = merged_df
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
        df["place"] = df["place"].astype(int)
        df["race_grade"] = df["race_grade"].astype(int)
        df["ground_state"] = df["ground_state"].astype(int)
        df["around"] = df["around"].fillna(3).astype(int)
        df["weather"] = df["weather"].astype(int)   
        
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



        baselog_old = (
            self.old_population.merge(
                self.old_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around"]], on="race_id"
            )
            # .merge(
            #     self.old_horse_results,
            #     on=["horse_id", "course_len", "race_type"],
            #     suffixes=("", "_horse"),
            # )
            # .query("date_horse < date")
            # .sort_values("date_horse", ascending=False)
        )

             
        df_old = (
            baselog_old
            .merge(self.old_results[["race_id", "horse_id", "nobori","time","wakuban", "umaban","rank","sex"]], on=["race_id", "horse_id"])
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



        
        # ターゲットエンコーディングを計算（カテゴリごとの複勝率の平均）
        df_old["target_rank"] = (df_old["rank"] <= 3).astype(int)
        df2 = df_old[["distance_place_type_umaban_ground_state_straight","distance_place_type_wakuban_ground_state_straight", "target_rank"]].dropna().astype(int)
        # グループごとのカウントを作成
        group_counts = df2.groupby("distance_place_type_umaban_ground_state_straight").size()
        # 100未満のグループを除外
        valid_groups = group_counts[group_counts >= 100].index
        # 100以上のグループのみを使用して、平均複勝率を計算
        df2_filtered = df2[df2["distance_place_type_umaban_ground_state_straight"].isin(valid_groups)]
        # 平均複勝率を計算
        mean_fukusho_rate = df2_filtered.groupby("distance_place_type_umaban_ground_state_straight")["target_rank"].transform("mean")
        # 計算した平均複勝率を元のDataFrameに追加
        df_old["mean_fukusho_rate_umaban"] = mean_fukusho_rate
        
        # グループごとのカウントを作成
        group_counts = df2.groupby("distance_place_type_wakuban_ground_state_straight").size()
        # 100未満のグループを除外
        valid_groups = group_counts[group_counts >= 100].index
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
                        "distance_place_type_race_grade_straight_ground_state", "distance_place_type_ground_state_weather", 
                    


                         
                        'distance_type_encoded', 'distance_place_type_encoded', 'distance_place_type_race_grade_encoded', 'distance_place_type_wakuban_encoded', 'distance_place_type_umaban_encoded', 'distance_place_type_wakuban_race_grade_encoded', 'distance_place_type_umaban_race_grade_encoded', 'distance_place_type_wakuban_straight_encoded', 'distance_place_type_umaban_straight_encoded', 'distance_place_type_wakuban_ground_state_encoded', 'distance_place_type_umaban_ground_state_encoded', 'distance_place_type_wakuban_ground_state_straight_encoded', 'distance_place_type_umaban_ground_state_straight_encoded', 'distance_type_weather_encoded', 'distance_place_type_weather_encoded', 'distance_place_type_race_grade_weather_encoded', 'distance_type_ground_state_encoded', 'distance_place_type_ground_state_encoded', 'distance_place_type_race_grade_ground_state_encoded', 'distance_type_sex_encoded', 'distance_place_type_sex_encoded', 'distance_place_type_race_grade_sex_encoded', 'distance_type_sex_weather_encoded', 'distance_place_type_sex_weather_encoded', 'distance_place_type_race_grade_sex_weather_encoded', 'distance_type_sex_ground_state_encoded', 'distance_place_type_sex_ground_state_encoded', 'distance_place_type_race_grade_sex_ground_state_encoded', 'distance_type_straight_encoded', 'distance_place_type_straight_encoded', 'distance_place_type_race_grade_straight_encoded', 'distance_type_straight_ground_state_encoded', 'distance_place_type_straight_ground_state_encoded', 'distance_place_type_race_grade_straight_ground_state_encoded',"distance_place_type_race_grade_around_weather_ground_state_encoded","distance_place_type_umaban_race_grade_around_weather_ground_state_encoded",
                        "distance_place_type_ground_state_weather_encoded","distance_place_type_ground_state_weather_nobori_encoded","distance_place_type_nobori_encoded","distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded","distance_place_type_race_grade_nobori_encoded",
                         
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
                "distance_place_type_race_grade_straight_ground_state","distance_place_type_race_grade_around_weather_ground_state",
                "distance_place_type_umaban_race_grade_around_weather_ground_state",
                "distance_place_type_ground_state_weather",
                'distance_type_encoded', 'distance_place_type_encoded', 'distance_place_type_race_grade_encoded', 'distance_place_type_wakuban_encoded', 'distance_place_type_umaban_encoded', 'distance_place_type_wakuban_race_grade_encoded', 'distance_place_type_umaban_race_grade_encoded', 'distance_place_type_wakuban_straight_encoded', 'distance_place_type_umaban_straight_encoded', 'distance_place_type_wakuban_ground_state_encoded', 'distance_place_type_umaban_ground_state_encoded', 'distance_place_type_wakuban_ground_state_straight_encoded', 'distance_place_type_umaban_ground_state_straight_encoded', 'distance_type_weather_encoded', 'distance_place_type_weather_encoded', 'distance_place_type_race_grade_weather_encoded', 'distance_type_ground_state_encoded', 'distance_place_type_ground_state_encoded', 'distance_place_type_race_grade_ground_state_encoded', 'distance_type_sex_encoded', 'distance_place_type_sex_encoded', 'distance_place_type_race_grade_sex_encoded', 'distance_type_sex_weather_encoded', 'distance_place_type_sex_weather_encoded', 'distance_place_type_race_grade_sex_weather_encoded', 'distance_type_sex_ground_state_encoded', 'distance_place_type_sex_ground_state_encoded', 'distance_place_type_race_grade_sex_ground_state_encoded', 'distance_type_straight_encoded', 'distance_place_type_straight_encoded', 'distance_place_type_race_grade_straight_encoded', 'distance_type_straight_ground_state_encoded', 'distance_place_type_straight_ground_state_encoded', 'distance_place_type_race_grade_straight_ground_state_encoded',"distance_place_type_race_grade_around_weather_ground_state_encoded","distance_place_type_umaban_race_grade_around_weather_ground_state_encoded","distance_place_type_ground_state_weather_encoded",



                "distance_place_type_umaban_race_grade_around_weather_ground_state_nobori_encoded","distance_place_type_race_grade_nobori_encoded","distance_place_type_ground_state_weather_nobori_encoded","distance_place_type_nobori_encoded",

                "mean_fukusho_rate_wakuban","mean_fukusho_rate_umaban"
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
        平均をold_resultsからだし
        過去成績のtime_diffをhorse_resultsからだし
        現在のmerged_dfに紐づける
        """

        baselog = (
            self.old_population.merge(
                self.old_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place"]], on="race_id"
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
            .merge(self.old_results[["race_id", "horse_id", "wakuban", "nobori","time","umaban"]], on=["race_id", "horse_id"])
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
        df["distance_place_type_race_grade_weather_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str)  + df["race_grade"].astype(str) + df["ground_state"].astype(str)+ df["weather"].astype(str)).astype(int)   
        
        
        baselog_old = (
            self.old_population.merge(
                self.old_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place"]], on="race_id"
            )
            # .merge(
            #     self.old_horse_results,
            #     on=["horse_id", "course_len", "race_type"],
            #     suffixes=("", "_horse"),
            # )
            # .query("date_horse < date")
            # .sort_values("date_horse", ascending=False)
        )
        
             
        df_old = (
            baselog_old
            .merge(self.old_results[["race_id", "horse_id", "wakuban", "umaban","nobori","time"]], on=["race_id", "horse_id"])
        )
        df_old["nobori"] = df_old["nobori"].fillna(df_old["nobori"].mean())
        
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        # df_old["around"] = df_old["around"].fillna(3).astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  
        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_race_grade_weather_ground_state"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str)  + df_old["race_grade"].astype(str) + df_old["ground_state"].astype(str)+ df_old["weather"].astype(str)).astype(int)   
        
        #noboriの平均
        target_mean_1 = df_old.groupby("distance_place_type_race_grade_weather_ground_state")["nobori"].mean()
        # 平均値をカテゴリ変数にマッピング
        df_old["distance_place_type_race_grade_weather_ground_state_nobori_encoded"] = df_old["distance_place_type_race_grade_weather_ground_state"].map(target_mean_1)
        
        # #pace_1の平均
        # target_mean_1 = df_old.groupby("distance_place_type_race_grade_around_weather_ground_state")["pace_1"].mean()
        # # 平均値をカテゴリ変数にマッピング
        # df_old["distance_place_type_race_grade_around_weather_ground_state_pace_1_encoded"] = df_old["distance_place_type_race_grade_around_weather_ground_state"].map(target_mean_1)
        
        df_old = df_old[["distance_place_type_race_grade_weather_ground_state","distance_place_type_race_grade_weather_ground_state_nobori_encoded"]]
        
        columns_to_merge = [("distance_place_type_race_grade_weather_ground_state","distance_place_type_race_grade_weather_ground_state_nobori_encoded")]
        
        
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
        
        # rank_diffとrace_gradeを利用した基本スコアを計算
        df["base_score"] = 0.5 / df["adjusted_rank_diff"] *(df['race_grade_scaled'] )*1
        
        # 条件に基づく変換
        df = df.copy()
        df.loc[:, "place_season_condition_type_categori_processed"] = df["place_season_condition_type_categori"].apply(
            lambda x: (x + 2 if x == '-' else (x - 3)) if isinstance(x, str) else x
        )
        
        # その後で1/1.7で割る
        df["place_season_condition_type_categori_processed_1"] = df["place_season_condition_type_categori_processed"] / 4

        # コーナー順位の平均を計算
        df["corner_avg_ratio"] = (
            df[["corner_1_per_horse", "corner_2_per_horse", "corner_3_per_horse", "corner_4_per_horse"]]
            .mean(axis=1)
        )
        
        # 出走レースのnoboriとの差分を計算
        df["nobori_diff"] = df["distance_place_type_race_grade_weather_ground_state_nobori_encoded"] - df["nobori"]
        # df["pace_diff"]  = df["distance_place_type_race_grade_around_weather_ground_state_pace_1_encoded"] - df["pace_1"]
        
        # スコア分割用の列を作成
        df["syunpatu"] = 0.0
        df["zizoku"] = 0.0
        
        # 条件1: 持続力にスコアを割り振る
        is_sustain_condition = df["corner_avg_ratio"] < 0.4
        df.loc[is_sustain_condition, "zizoku"] = df["base_score"] + df["place_season_condition_type_categori_processed_1"]
        
        # 条件1_1: 瞬発力にもスコアを割り振る（nobori_diff > 0.3 の場合）
        df.loc[is_sustain_condition & (df["nobori_diff"] > 0.3), "syunpatu"] = df["base_score"] - df["place_season_condition_type_categori_processed_1"]
        
        # 条件2: 瞬発力にスコアを割り振る（それ以外の場合）※未割り当ての場合のみ
        df.loc[~is_sustain_condition & df["syunpatu"].isna(), "syunpatu"] = df["base_score"] - df["place_season_condition_type_categori_processed_1"]
        
        
        # 瞬発力に補正を加える
        nobori_correction_factor = np.maximum(df["nobori_diff"] - 0.3, 0) * 0.4  # 補正を半分に減らす
        df["syunpatu"] += nobori_correction_factor * df["base_score"] - df["place_season_condition_type_categori_processed_1"]
        
        # # pace_1列に基づく補正
        
        # pace_correction_factor = 0.1 - (df["pace_1"] - 36) / 3.6  # pace_1が36より小さいほど補正を増やし、増えるほど補正を減らす
        # df["pace_correction_factor"] = pace_correction_factor
        pace_correction_factor =(df["pace_diff"]) * 1
        df["pace_correction_factor"] = pace_correction_factor
        
        # 持続力 (zizoku) に補正を加える
        df["zizoku"] += pace_correction_factor *df["base_score"] *0.3 # 調整係数0.5は任意で調整可能
        
        # 瞬発力 (syunpatu) に補正を加える
        df["syunpatu"] += pace_correction_factor *df["base_score"]* 0.4  # 同様に調整係数を加えます
        
        # 持続力スコアを距離で補正
        distance_base = 2000
        distance_correction_factor = 1  # 距離補正の影響を抑えるスケーリング係数
        df["distance_correction"] = (df["course_len"] - distance_base) / 1500  # 分母を5000に変更し補正を小さく
        df["zizoku"] += df["distance_correction"]  * distance_correction_factor
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



        self.agg_cross_features_df_4 = merged_df
        print("running cross_features_4()...comp")



    
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
        
        df_old2 = (
            self.old_results_condition[["race_id", "horse_id","time","rank_per_horse"]]
            .merge(self.old_race_info_condition[["race_id", "place","weather","ground_state","race_grade","course_len","race_type","race_date_day_count"]], on="race_id")
        )

        df_old2["place"] = df_old2["place"].astype(int)
        df_old2["race_grade"] = df_old2["race_grade"].astype(int)
        # 距離/競馬場/タイプ/レースランク
        df_old2["distance_place_type_race_grade"] = (df_old2["course_len"].astype(str) + df_old2["place"].astype(str) + df_old2["race_type"].astype(str) + df_old2["race_grade"].astype(str)).astype(int)
        
        
        # baselog_old = (
        #     self.old_population.merge(
        #         self.old_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around"]], on="race_id"
        #     )
        #     # .merge(
        #     #     self.old_horse_results,
        #     #     on=["horse_id", "course_len", "race_type"],
        #     #     suffixes=("", "_horse"),
        #     # )
        #     # .query("date_horse < date")
        #     # .sort_values("date_horse", ascending=False)
        # )
        df_old = (
            self.old_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around"]]
            .merge(self.old_results[["race_id", "horse_id", "wakuban", "umaban","nobori","time","sex"]], on="race_id")
        )
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        df_old["around"] = df_old["around"].fillna(3).astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  
                     
        
        df_old["distance_place_type_race_grade"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
        target_mean_1 = df_old.groupby("distance_place_type_race_grade")["time"].mean()
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
        df_old2["distance_place_type_race_grade_encoded_time_diff"] = df_old2['distance_place_type_race_grade_encoded'] - df_old2["time"]

        df_old2= df_old2[df_old2["race_type"] != 2]
        df_old2_1 = df_old2[df_old2["race_type"] != 0]
        # 2. df の各行について処理

        def compute_mean_for_row(row, df_old2_1):
            # race_type == 0 の場合は NaN を返す
            if row["race_type"] == 0:
                return np.nan
            df_old2_1["race_date_day_count"] = df_old2_1["race_date_day_count"].astype(int)
            target_day_count = int(row["race_date_day_count"])  # df の各行の race_date_day_count

            # target_day_count = row["race_date_day_count"]  # df の各行の race_date_day_count

                
            # 3. df_old2_1 から条件に一致する行をフィルタリング
            filtered_df_old2_1 = df_old2_1[
                (df_old2_1["race_date_day_count"] >= (target_day_count - 1200)) &  # race_date_day_count が target_day_count-1200 以上
                (df_old2_1["race_date_day_count"] <= (target_day_count - 1)) &  # race_date_day_count が target_day_count-1 以下
                (df_old2_1["place"] == row["place"]) &  # place が一致
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
            if row["weather"] in [3, 4, 5] or row["ground_state"] in [1, 2, 3]:
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
            return float("nan")
        
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
            baselog.loc[condition1, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *baselog["course_len"]/2000/ 2.5 
        
            # 小回りカーブで場合
            condition2 = (baselog["curve"] == 2)  & (baselog["show"] == 1)
            baselog.loc[condition2, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *baselog["course_len"]/2000/ 2.1
            
            # 小スパカーブ場合
            condition3 = (baselog["curve"] == 3) & (baselog["show"] == 1)
            baselog.loc[condition3, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *baselog["course_len"]/2000/ 1.8
        
            # スパカーブ場合
            condition4 = (baselog["curve"] == 4)  & (baselog["show"] == 1)
            baselog.loc[condition4, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *baselog["course_len"]/2000/ 1.5
        
            # 複合カーブで場合
            condition5 = (baselog["curve"] == 5)  & (baselog["show"] == 1)
            baselog.loc[condition5, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori']))*baselog["course_len"]/2000
            
            
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
        
        merged_df2 = merged_df.merge(self.race_info[["race_id","course_len"]], on="race_id")
        merged_df2["course_len_relative"] = (merged_df2["course_len"] / 1800) -1
        merged_df2.loc[merged_df2["course_len_relative"] < 0, "course_len_relative"] = 0
        merged_df2["course_len_relative"] = merged_df2["course_len_relative"] * 5
        # 対象の列名リスト
        columns_to_multiply = [
            'score_stamina_mean_1races_per_stamina_raw',
            'score_stamina_max_1races_per_stamina_raw',
            'score_stamina_min_1races_per_stamina_raw',
            'score_stamina_mean_1races_per_stamina_raw_relative',
            'score_stamina_max_1races_per_stamina_raw_relative',
            'score_stamina_min_1races_per_stamina_raw_relative',
            'score_stamina_mean_3races_per_stamina_raw',
            'score_stamina_max_3races_per_stamina_raw',
            'score_stamina_min_3races_per_stamina_raw',
            'score_stamina_mean_3races_per_stamina_raw_relative',
            'score_stamina_max_3races_per_stamina_raw_relative',
            'score_stamina_min_3races_per_stamina_raw_relative',
            'score_stamina_mean_5races_per_stamina_raw',
            'score_stamina_max_5races_per_stamina_raw',
            'score_stamina_min_5races_per_stamina_raw',
            'score_stamina_mean_5races_per_stamina_raw_relative',
            'score_stamina_max_5races_per_stamina_raw_relative',
            'score_stamina_min_5races_per_stamina_raw_relative',
            'score_stamina_mean_10races_per_stamina_raw',
            'score_stamina_max_10races_per_stamina_raw',
            'score_stamina_min_10races_per_stamina_raw',
            'score_stamina_mean_10races_per_stamina_raw_relative',
            'score_stamina_max_10races_per_stamina_raw_relative',
            'score_stamina_min_10races_per_stamina_raw_relative'
        ]
        
        # 'course_len_relative'を各列に掛け算
        for col in columns_to_multiply:
            merged_df2[col] = merged_df2[col] * merged_df2['course_len_relative']
        # 'course_len'と'course_len_relative'の列を削除
        merged_df2 = merged_df2.drop(columns=['course_len', 'course_len_relative'])

        self.agg_cross_features_df_12 = merged_df2
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
            baselog.loc[condition1, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *baselog["course_len"]/2000/ 2.5
        
            # 小回りカーブで場合
            condition2 = (baselog["curve"] == 2)  & (baselog["show"] == 1)
            baselog.loc[condition2, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *baselog["course_len"]/2000/ 2.1
            
            # 小スパカーブ場合
            condition3 = (baselog["curve"] == 3) & (baselog["show"] == 1)
            baselog.loc[condition3, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *baselog["course_len"]/2000/ 1.8
        
            # スパカーブ場合
            condition4 = (baselog["curve"] == 4)  & (baselog["show"] == 1)
            baselog.loc[condition4, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori'])) *baselog["course_len"]/2000/ 1.5
        
            # 複合カーブで場合
            condition5 = (baselog["curve"] == 5)  & (baselog["show"] == 1)
            baselog.loc[condition5, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori']))*baselog["course_len"]/2000
            
            
            
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
            
        
        merged_df5 = merged_df5.merge(self.race_info[["race_id","course_len"]], on="race_id")
        merged_df5["course_len_relative"] = (merged_df5["course_len"] / 2000) -1
        merged_df5.loc[merged_df5["course_len_relative"] < 0, "course_len_relative"] = 0
        merged_df5["course_len_relative"] = merged_df5["course_len_relative"] * 2.2
        # 対象の列名リスト
        columns_to_multiply = [
            'score_stamina_mean_1races_per_stamina_raw',
            'score_stamina_max_1races_per_stamina_raw',
            'score_stamina_min_1races_per_stamina_raw',
        
            'score_stamina_mean_3races_per_stamina_raw',
            'score_stamina_max_3races_per_stamina_raw',
            'score_stamina_min_3races_per_stamina_raw',
        
            'score_stamina_mean_5races_per_stamina_raw',
            'score_stamina_max_5races_per_stamina_raw',
            'score_stamina_min_5races_per_stamina_raw',
        
            'score_stamina_mean_10races_per_stamina_raw',
            'score_stamina_max_10races_per_stamina_raw',
            'score_stamina_min_10races_per_stamina_raw',
        
        ]
        
        # 'course_len_relative'を各列に掛け算
        for col in columns_to_multiply:
            merged_df5[col] = merged_df5[col] * merged_df5['course_len_relative']
        # 'course_len'と'course_len_relative'の列を削除
        merged_df5 = merged_df5.drop(columns=['course_len', 'course_len_relative'])
        
        
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
                return 1  # ハイペース
            elif escape_count == 2:
                if taking_lead_per > 0.7:
                    return 1  # ハイペース
                else:
                    return 2  
            elif escape_count == 1:
                if taking_lead_per > 0.75:
                    return 1  # ハイペース
                elif 0.65 < taking_lead_per <= 0.75:
                    return 2  # ハイミドルペース
                elif 0.55 < taking_lead_per <= 0.65:
                    return 3  # スローミドルペース
                else:
                    return 4  # スローペース
            elif escape_count == 0:
                # 逃げ馬が0の場合、先行馬の割合に基づく
                if taking_lead_per > 0.88:
                    return 1  # ハイペース
                elif 0.75 < taking_lead_per <= 0.88:
                    return 2  # ハイミドルペース
                elif 0.65 < taking_lead_per <= 0.75:
                    return 3  # スローミドルペース
                else:
                    return 4  # スローペース
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
        

        merged_df = self.population.copy()        
        df = (
            merged_df
            .merge(self.race_info[["race_id", "place","weather","ground_state","course_len","race_type","race_date_day_count"]], on="race_id")
        )
        df["place"] = df["place"].astype(int)
        
        df_old2 = (
            self.old_results_condition[["race_id", "horse_id","time","rank_per_horse"]]
            .merge(self.old_race_info_condition[["race_id", "place","weather","ground_state","race_grade","course_len","race_type","race_date_day_count"]], on="race_id")
        )

        df_old2["place"] = df_old2["place"].astype(int)
        df_old2["race_grade"] = df_old2["race_grade"].astype(int)
        # 距離/競馬場/タイプ/レースランク
        df_old2["distance_place_type_race_grade"] = (df_old2["course_len"].astype(str) + df_old2["place"].astype(str) + df_old2["race_type"].astype(str) + df_old2["race_grade"].astype(str)).astype(int)
        
        
        # baselog_old = (
        #     self.old_population.merge(
        #         self.old_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around"]], on="race_id"
        #     )
        #     # .merge(
        #     #     self.old_horse_results,
        #     #     on=["horse_id", "course_len", "race_type"],
        #     #     suffixes=("", "_horse"),
        #     # )
        #     # .query("date_horse < date")
        #     # .sort_values("date_horse", ascending=False)
        # )
        df_old = (
            self.old_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around"]]
            .merge(self.old_results[["race_id", "horse_id", "wakuban", "umaban","nobori","time","sex"]], on="race_id")
        )
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        df_old["around"] = df_old["around"].fillna(3).astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  
                     
        
        df_old["distance_place_type_race_grade"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
        target_mean_1 = df_old.groupby("distance_place_type_race_grade")["time"].mean()
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
        df_old2["distance_place_type_race_grade_encoded_time_diff"] = df_old2['distance_place_type_race_grade_encoded'] - df_old2["time"]
        
        df_old2= df_old2[df_old2["race_type"] != 2]
        df_old2_1 = df_old2[df_old2["race_type"] != 0]
        
        # 2. df の各行について処理
        def compute_mean_for_row(row, df_old2_1):
            # race_type == 0 の場合は NaN を返す
            if row["race_type"] == 0:
                return np.nan
            df_old2_1["race_date_day_count"] = df_old2_1["race_date_day_count"].astype(int)
            target_day_count = int(row["race_date_day_count"])  # df の各行の race_date_day_count
    
            # target_day_count = row["race_date_day_count"]  # df の各行の race_date_day_count
        
            # 3. df_old2_1 から条件に一致する行をフィルタリング
            filtered_df_old2_1 = df_old2_1[
                (df_old2_1["race_date_day_count"] >= (target_day_count - 1200)) &  # race_date_day_count が target_day_count-1200 以上
                (df_old2_1["race_date_day_count"] <= (target_day_count - 1)) &  # race_date_day_count が target_day_count-1 以下
                (df_old2_1["place"] == row["place"]) &  # place が一致
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
            if row["weather"] in [3, 4, 5] or row["ground_state"] in [1, 2, 3]:
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
            return float("nan")
        
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
        
        高速馬場の方が逃げ先行が有利なので、ground_state_level列に-4を行い、先行有利を-側へ
        ground_state_levelには欠損値があるため、それは4扱いにし、-4で0になるようにすること
        ローペースの方が逃げ先行有利なので、pace_category列に-2.5を行い、先行有利を-側へ
        先行逃げを-側にしたいため、dominant_position_categoryが1,2のものを-2に、3を2に、4を1.8にそれぞれ変換する
        
        まずこれだけの列を作成する
        影響を大きくしたい場合、それぞれの列に倍率を調整できるようにする
        
        "goal_range_100"は-3.5を行い、+の場合は全て0に変換する
        
        "curve"は-4.5を行い、+の場合は数値を8倍する
        
        "goal_slope"は-1を行う、pace_categoryに-をかける、pace_categoryと掛け合わせる
        
        """        
        
        merged_df_all = merged_df100[["race_id","date","horse_id","dominant_position_category","pace_category","ground_state_level"]]
        merged_df_all = merged_df_all.merge(
                self.race_info[["race_id", "goal_range_100","curve","goal_slope"]], 
                on="race_id",
            )
        
        # 必要に応じてコピーを作成
        merged_df_all = merged_df_all.copy()
        
        
        
        # ground_state_level_processed 列の処理
        merged_df_all.loc[:, "ground_state_level_processed"] = (
            merged_df_all["ground_state_level"].fillna(4) - 4
        )
        
        # pace_category_processed 列の処理
        merged_df_all.loc[:, "pace_category_processed"] = (
            merged_df_all["pace_category"] - 2.5
        )
        
        # dominant_position_category_processed 列の処理
        merged_df_all.loc[:, "dominant_position_category_processed"] = (
            merged_df_all["dominant_position_category"]
            .replace({1: -2, 2: -2, 3: 2, 4: 1.8})
        )
        
        # goal_range_100 に -3.5 を行う
        merged_df_all["goal_range_100_processed"] = merged_df_all["goal_range_100"] - 3.5
        # プラスの値をすべて 0 に変換
        merged_df_all["goal_range_100_processed"] = merged_df_all["goal_range_100_processed"].clip(upper=0)
        
        
        # -4.5 を行う
        merged_df_all["curve_processed"] = merged_df_all["curve"] - 4.5
        # +の場合は数値を8倍する
        merged_df_all["curve_processed"] = merged_df_all["curve_processed"].apply(
            lambda x: x * 8 if x > 0 else x
        )
        
        
        # goal_slope に -1 を行う
        merged_df_all["goal_slope_processed"] = merged_df_all["goal_slope"] - 1
        
        # pace_category に - をかけて符号反転
        merged_df_all["pace_category_processed"] = merged_df_all["pace_category"] * -1
        
        # goal_slope_processed と pace_category_processed を掛け合わせる
        merged_df_all["goal_slope_processed"] = merged_df_all["goal_slope_processed"] * merged_df_all["pace_category_processed"]
        
        
        
        # それぞれの列に倍率を適用
        ground_state_multiplier = 4.5 # 必要に応じて調整
        pace_multiplier = 3.5  # 必要に応じて調整
        dominant_multiplier = 2.8  # 必要に応じて調整
        goal_range_multiplier = 18
        curve_multiplier = 2.25
        goal_slope_multiplier = 2.5

        
        merged_df_all["dominant_position_category_processed"] = merged_df_all["dominant_position_category_processed"].astype(float)
        merged_df_all["goal_range_100_processed"] = merged_df_all["goal_range_100_processed"].astype(float)
        merged_df_all["goal_slope_processed"] = merged_df_all["goal_slope_processed"].astype(float)
        merged_df_all["pace_category_processed"] = merged_df_all["pace_category_processed"].astype(float)


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
        
        df_x = df_x[["race_id","date","horse_id","date_horse","race_grade_scaled","rank_diff"]]
        
        df_x = df_x.copy()  # df_xのコピーを作成
        df_x.loc[:, "race_grade_rank_diff_multi"] = ((df_x["race_grade_scaled"] + 1)) * (0.5 / (df_x["rank_diff"] + 1)) * 48
        df_x.loc[:, "race_grade_rank_diff_sum"] = ((df_x["race_grade_scaled"] + 1) ) + (0.5 / (df_x["rank_diff"] + 1)) * 60
        
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
            merge_all_ex[f"{col}_plus_tenkai_combined"] = merge_all_ex[col] + merge_all_ex["tenkai_combined"]
            merge_all_ex[f"{col}_plus_tenkai_all_combined"] = merge_all_ex[col] + merge_all_ex["tenkai_all_combined"]
            merge_all_ex[f"{col}_px_tenkai_combined"] = merge_all_ex[col] * merge_all_ex["tenkai_combined"]
            merge_all_ex[f"{col}_px_tenkai_all_combined"] = merge_all_ex[col] * merge_all_ex["tenkai_all_combined"]


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
        85
        """
        baselog = (
            self.old_population.merge(
                self.old_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place"]], on="race_id"
            )
        )

        df = (
            baselog
            .merge(self.old_results[["race_id", "horse_id", "wakuban", "nobori","time","umaban"]], on=["race_id", "horse_id"])
        )
        df["nobori"] = df["nobori"].fillna(df["nobori"].mean())

        df["place"] = df["place"].astype(int)
        df["race_grade"] = df["race_grade"].astype(int)
        df["ground_state"] = df["ground_state"].astype(int)
        df["weather"] = df["weather"].astype(int)  
        df["distance_place_type"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str)).astype(int)


        # 1. 計算したいrace_gradeのリスト
        grades = [55, 60, 70, 79, 85, 89, 91, 94, 98]

        # 2. 各race_gradeごとにtimeとnoboriの平均を計算し、結合
        for grade in grades:
            # race_gradeが指定された値のときのtimeとnoboriの平均を計算
            time_nobori_avg = (
                df[df['race_grade'] == grade]  # race_gradeが指定のgradeの行を抽出
                .groupby(['course_len', 'place', 'race_type'])[['time', 'nobori']]  # 3つのカテゴリごとにtimeとnoboriを集計
                .mean()
                .reset_index()  # インデックスをリセット
                .rename(columns={'time': f'time_avg_{grade}', 'nobori': f'nobori_avg_{grade}'})  # 列名を変更
            )
            
            # 元のDataFrameにマージ（left join）
            horse_results_baselog = pd.merge(horse_results_baselog, time_nobori_avg, on=['course_len', 'place', 'race_type'], how='left')



        # 1. 補完する順番を指定
        grades = [85, 79, 89, 70, 91, 60, 94, 55, 98]

        # 2. まずhorse_results_baselog内の補完処理を行う
        horse_results_baselog['final_time_avg'] = np.nan
        horse_results_baselog['final_nobori_avg'] = np.nan

        # 3. 最初に85で補完
        horse_results_baselog['final_time_avg'] = np.where(
            horse_results_baselog['final_time_avg'].isna(), 
            horse_results_baselog['time_avg_85'], 
            horse_results_baselog['final_time_avg']
        )
        horse_results_baselog['final_nobori_avg'] = np.where(
            horse_results_baselog['final_nobori_avg'].isna(), 
            horse_results_baselog['nobori_avg_85'], 
            horse_results_baselog['final_nobori_avg']
        )

        # 4. 残りのグレードで補完（順番に）
        for grade in grades:
            # `85`はすでに補完されているので、以降のグレードのみで補完
            if grade != 85:
                time_col = f'time_avg_{grade}'
                nobori_col = f'nobori_avg_{grade}'
                
                # timeの欠損を補完
                horse_results_baselog['final_time_avg'] = np.where(
                    horse_results_baselog['final_time_avg'].isna(), 
                    horse_results_baselog[time_col], 
                    horse_results_baselog['final_time_avg']
                )
                
                # noboriの欠損を補完
                horse_results_baselog['final_nobori_avg'] = np.where(
                    horse_results_baselog['final_nobori_avg'].isna(), 
                    horse_results_baselog[nobori_col], 
                    horse_results_baselog['final_nobori_avg']
                )

        # 5. 不要な中間列を削除し、必要な列だけ残す
        columns_to_drop = [f'time_avg_{grade}' for grade in grades] + [f'nobori_avg_{grade}' for grade in grades]
        horse_results_baselog = horse_results_baselog.drop(columns=columns_to_drop)



        # 1. final_time_avgからtimeを引いた数値（秒）を計算
        horse_results_baselog['time_diff_sp'] = horse_results_baselog['final_time_avg'] - horse_results_baselog['time']

        # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['time_points'] = (horse_results_baselog['time_diff_sp'] *10)

        # 1. final_nobori_avgからnoboriを引いた数値（秒）を計算
        horse_results_baselog['nobori_diff_sp'] = horse_results_baselog['final_nobori_avg'] - horse_results_baselog['nobori']

        # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['nobori_points'] = (horse_results_baselog['nobori_diff_sp'] *10)



        """
        距離指数をかける
        """

        # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['time_points_course_index'] = (horse_results_baselog['time_points'] *(1600/horse_results_baselog["course_len"]))

        # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['nobori_points_course_index'] = (horse_results_baselog['nobori_points'] *(1600/horse_results_baselog["course_len"]))



        """
        ＋（	斥量	－	５５	）
        """
        # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['time_points_impost'] = (horse_results_baselog['time_points_course_index'] +(horse_results_baselog["impost"]-55) *1.7)

        # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['nobori_points_impost'] = (horse_results_baselog['nobori_points_course_index']+(horse_results_baselog["impost"]-55) *1.7)


        """
        （	走破
        タイム	－	基準
        タイム（グレード別）	）×	距離
        指数	＋	クラス値＋pase_diff

        ハイペースなら低く（早く見える）でてしまう
        スローペースなら高く（遅く見える）でてしまう
        pase_diffは+だとハイペース
        -だとスローペースなので
        そのまま+してよいかも
        """

        # race_gradeが指定された値のときのtimeとnoboriの平均を計算
        time_nobori_grade_avg = (
            df.groupby(['course_len', 'place', 'race_type',"race_grade"])[['time', 'nobori']]# 3つのカテゴリごとにtimeとnoboriを集計
            .mean()
            .reset_index()  # インデックスをリセット
            .rename(columns={'time': f'time_grade_avg', 'nobori': f'nobori_grade_avg'})  # 列名を変更
        )

        # 元のDataFrameにマージ（left join）
        horse_results_baselog = pd.merge(horse_results_baselog, time_nobori_grade_avg, on=['course_len', 'place', 'race_type',"race_grade"], how='left')


        # 1. final_time_avgからtimeを引いた数値（秒）を計算
        horse_results_baselog['time_diff_grade'] = horse_results_baselog['time'] - horse_results_baselog["time_grade_avg"] 

        # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['time_points_grade'] = (horse_results_baselog['time_diff_grade'] *10)

        # 1. final_nobori_avgからnoboriを引いた数値（秒）を計算
        horse_results_baselog['nobori_diff_grade'] = horse_results_baselog['nobori'] - horse_results_baselog['nobori_grade_avg']

        # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['nobori_points_grade'] = (horse_results_baselog['nobori_diff_grade'] *10)

        """
        距離指数をかける
        """

        # 2. time_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['time_points_grade_index'] = (horse_results_baselog['time_points_grade'] *(1600/horse_results_baselog["course_len"]))

        # 2. nobori_diffを0.1秒ごとのポイントに変換（1ポイント = 0.1秒）
        horse_results_baselog['nobori_points_grade_index'] = (horse_results_baselog['nobori_points_grade'] *(1600/horse_results_baselog["course_len"]))

        horse_results_baselog['time_condition_index'] = horse_results_baselog['nobori_points_grade_index'] +horse_results_baselog['pace_diff'] +horse_results_baselog['race_grade']
        horse_results_baselog['nobori_condition_index'] = horse_results_baselog['nobori_points_grade_index'] +horse_results_baselog['pace_diff'] +horse_results_baselog['race_grade']


        horse_results_baselog['speed_index'] = horse_results_baselog['time_points_impost'] + horse_results_baselog['time_condition_index'] + 80
        horse_results_baselog['nobori_index'] = horse_results_baselog['nobori_points_impost'] + horse_results_baselog['nobori_condition_index'] + 80
        
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


    
    # def cross_features_15(
    #     self, n_races: list[int] = [1, 3, 5, 10]
    # ):  

    #     self.agg_cross_features_df_15 = merged_df
    #     print("running cross_features_15()...comp")
        
            
    def create_features(
        self, race_id: str, date_content_a: str,skip_agg_horse: bool = False
    ) -> pd.DataFrame:
        """
        特徴量作成処理を実行し、populationテーブルに全ての特徴量を結合する。
        先に馬の過去成績集計を実行しておいた場合は、
        skip_agg_horse=Trueとすればスキップできる。
        """
        # 馬の過去成績集計（先に実行しておいた場合は、スキップできる）
        if not skip_agg_horse:
            self.create_baselog()
            self.agg_horse_n_races()
            self.agg_horse_n_races_relative()
        # 各種テーブルの取得
        self.fetch_shutuba_page_html(race_id)
        self.fetch_results(race_id, self.htmls[race_id])
        self.fetch_race_info(race_id, date_content_a,self.htmls[race_id])
        self.create_race_grade()
        # グループごとの馬の過去成績集計（race_infoのカラムが必要なため、ここで実行）
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
        
        self.agg_horse_per_course_len()
        self.agg_horse_per_group_cols(
            group_cols=["ground_state", "race_type"], df_name="ground_state_race_type"
        )
        self.agg_horse_per_group_cols(group_cols=["race_grade"], df_name="race_grade")
        self.agg_horse_per_group_cols(group_cols=["race_type"], df_name="race_type")
        self.agg_horse_per_group_cols(
            group_cols=["place","course_len", "race_type"], df_name="race_place_len"
        )
        self.agg_horse_per_group_cols(
            group_cols=["place", "race_type"], df_name="race_place"
        )        
        self.agg_horse_per_group_cols(
            group_cols=["weather"], df_name="weather"
        )
        # リーディングデータの紐付け
        self.agg_jockey()
        self.agg_trainer()
        self.agg_sire()
        self.agg_bms()
        # 全ての特徴量を結合
        
        print("merging all features...")
        features = (
            self.population.merge(self.results, on=["race_id", "horse_id"])
            .merge(self.race_info, on=["race_id"])
            .merge(
                self.agg_horse_n_races_df,
                on=["race_id", "date", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_horse_n_races_relative_df,
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
                on=["race_id","date", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_horse_per_group_cols_dfs["ground_state_race_type"],
                on=["race_id",  "date","horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_horse_per_group_cols_dfs["race_grade"],
                on=["race_id",  "date","horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_horse_per_group_cols_dfs["race_type"],
                on=["race_id","date", "horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_horse_per_group_cols_dfs["race_place_len"],
                on=["race_id", "date","horse_id"],
                how="left",
                # copy=False,
            )
            .merge(
                self.agg_interval_df,
                on=["race_id","date",  "horse_id"],
                how="left",  
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
            )    
            .merge(
                self.agg_cross_features_df_4,
                on=["race_id", "date", "horse_id"],
                how="left",
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
            #     self.agg_horse_per_group_cols_dfs["around_per_wakuban"],
            #     on=["race_id", "horse_id"],
            #     how="left",
            # )
            .merge(
                self.agg_horse_per_group_cols_dfs["race_place"],
                on=["race_id", "date", "horse_id"],
                how="left",
            )
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
            .merge(
                self.agg_bms_df,
                on=["race_id", "horse_id"],
                how="left",
                # copy=False,
            )
        )
        features.drop(columns=["date"], inplace=True)
        # features.drop(columns=['place_season_condition_type_categori_x'], inplace=True)

        features.to_csv(self.output_dir / self.output_filename, sep="\t", index=False)
        print("merging all features...comp")
        return features