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

from urllib.request import Request, urlopen
import time
import chardet
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

        population_all_filename: str = "population_all.csv",    
        results_all_filename: str = "results_all.csv",
        race_info_all_filename: str = "race_info_all.csv",
        horse_results_all_filename: str = "horse_results_all.csv",  
        # peds_all_filename: str = "peds_all.csv",      

        
        old_results_condition_filename: str = "results_prediction.csv",
        old_race_info_condition_filename: str = "race_info_prediction.csv",   
        bms_leading_filename: str = "bms_leading.csv",     
    ):
        self.population = pd.read_csv(population_dir / population_filename, sep="\t")



        self.all_population = pd.read_csv(old_population_dir / population_all_filename, sep="\t")
        self.all_results = pd.read_csv(input_dir / results_all_filename, sep="\t")
        self.all_race_info = pd.read_csv(input_dir / race_info_all_filename, sep="\t")
        self.all_horse_results = pd.read_csv(input_dir / horse_results_all_filename, sep="\t")
       


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

        

    
    # def fetch_shutuba_page_html(self, race_id: str) -> None:
    #     """
    #     レースidを指定すると、出馬表ページのhtmlをスクレイピングする関数。
    #     """
    #     print("fetching shutuba page html...")
    #     options = Options()
    #     options.add_argument("--headless")
    #     options.add_argument("--no-sandbox")
    #     options.add_argument("--disable-dev-shm-usage")
    #     # chrome driverをインストール
    #     driver_path = ChromeDriverManager().install()
    #     url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    #     with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
    #         driver.implicitly_wait(10000)
    #         driver.get(url)
    #         self.htmls[race_id] = driver.page_source
    #     print("fetching shutuba page html...comp")

    def fetch_shutuba_page_html(self, race_id: str) -> None:
        """
        レースidを指定すると、出馬表ページのhtmlをスクレイピングする関数。
        """
        print("fetching shutuba page html...")
        
        # ヘッダーを設定
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}
        
        # URLを作成
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        
        # リクエストを作成し、HTMLを取得
        request = Request(url, headers=headers)
        html = urlopen(request).read()  # HTMLを取得
        
        # 文字エンコーディングを判別
        result = chardet.detect(html)
        
        # エンコーディングを取得
        encoding = result.get('encoding', 'utf-8')  # chardetがエンコーディングを判別できなかった場合、デフォルトで'utf-8'を使用
        
        self.htmls[race_id] = html.decode(encoding)  # 判別したエンコーディングでHTMLを文字列にデコードして保存

        
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

        a_list = soup.find_all("a", href=re.compile(r"/jockey/"))
        jockey_id_list = []
        for a in a_list:
            match = re.findall(r"\d{5}", a.get("href", ""))
            if match:  # 5桁の数字が見つかった場合
                jockey_id_list.append(int(match[0]))
            else:
                jockey_id_list.append(np.nan)  # 見つからない場合は NaN を追加

        df["jockey_id"] = jockey_id_list


        # jockey_id_list = []
        # a_list = soup.find_all("a", href=re.compile(r"/jockey/"))
        # for a in a_list:
        #     jockey_id = re.findall(r"\d{5}", a["href"])[0]
        #     jockey_id_list.append(int(jockey_id))
        # df["jockey_id"] = jockey_id_list


        trainer_id_list = []
        a_list = soup.find_all("a", href=re.compile(r"/trainer/"))
        # for a in a_list:
        #     trainer_id = re.findall(r"\d{5}", a["href"])[0]
        #     trainer_id_list.append(int(trainer_id))
        for a in a_list:
            matches = re.findall(r"\d{5}", a["href"])
            if matches:
                trainer_id = matches[0]
                trainer_id_list.append(int(trainer_id))
            else:
                trainer_id_list.append(0) 
                                
        df["trainer_id"] = trainer_id_list


        # df = df[df.iloc[:, 9] != '--']
        # df = df[df.iloc[:, 9] != '---.-']
        # df["tansho_odds"] = df.iloc[:, 9].astype(float)
        # 前処理
        df["wakuban"] = df.iloc[:, 0].astype(int)
        df["umaban"] = df.iloc[:, 1].astype(int)
        df["umaban_odd"] = (df["umaban"] % 2 == 1).astype(int)
        df["sex"] = df.iloc[:, 4].str[0].map(sex_mapping)
        df["age"] = df.iloc[:, 4].str[1:].astype(int)
        df["impost"] = df.iloc[:, 5].astype(float)
        # # df["weight"] = df.iloc[:, 8].str.extract(r"(\d+)").astype(int)
        # # df["weight_diff"] = df.iloc[:, 8].str.extract(r"\((.+)\)").astype(int)
        
        # # df["weight_diff"] = df.iloc[:, 8].astype(str).str.extract(r"\((.+)\)").astype(float)
        # # 増減部分を抽出し、'前計不' を NaN に置き換える
        # df["weight_diff"] = df.iloc[:, 8].str.extract(r"\((.+)\)")

        # # '前計不'を NaN に置き換え、残りの部分を float 型に変換
        # df["weight_diff"] = df["weight_diff"].replace("前計不", np.nan).astype(float)
        # 列が存在しない、またはすべて空の場合に備える
        if df.iloc[:, 8].isnull().all():  # 全てがNaNの場合
            df["weight"] = np.nan  # 新しい列をNaNで埋める
            df["weight_diff"] = np.nan  # 新しい列をNaNで埋める
        else:
            # 必要に応じて文字列型に変換
            df.iloc[:, 8] = df.iloc[:, 8].astype(str)

            df["weight"] = df.iloc[:, 8].str.extract(r"(\d+)").astype(float)
            # 体重増減部分を抽出
            df["weight_diff"] = df.iloc[:, 8].str.extract(r"\((.+)\)")

            # '前計不'を NaN に置き換え、残りを float 型に変換
            df["weight_diff"] = df["weight_diff"].replace("前計不", np.nan).astype(float)



        # 改行や不要な空白を完全に削除して、整数に変換
        # 数値型の場合、文字列に変換してから改行を削除
        # df["popularity"] = df.iloc[:, 10].astype(str).str.replace(r'\\n', '', regex=True).str.replace(r'\n', '', regex=True).str.strip().astype(int)


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

        result_df["impost_percent"] = result_df["impost"] / result_df["weight"]

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
                # "tansho_odds",
                # "popularity",
                "impost",
                "impost_percent",
                "sex",
                "age",
                "weight",
                "weight_diff",
                "n_horses",
                "mean_age",
                "median_age",
                "mean_age_kirisute", 
                "umaban_odd",
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
        info_dict["course_len_type"] = (
            1 if len(info1[2]) > 1 and "内" in str(info1[2][1]) 
            else 2 if len(info1[2]) > 1 and "外" in str(info1[2][1]) 
            else 1
        )
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
        elif len(race_class) != 0 and race_class != ['オープン']:
            info_dict["race_class"] = race_class_mapping[race_class[0]]
        elif len(race_class) != 0 and race_class ==['オープン']:
            #オープンの場合
            #賞金
            #2900未満でオープン
            #2900-5000G3
            #5000-10000G2
            #10000-G1
            # 本賞金部分の抽出
            prize_text = divs[1].find("span", string=lambda text: text and "本賞金:" in text).text
            prize_amount = int(prize_text.split(":")[1].split(",")[0])  # 最初の金額を取得
            # 本賞金に基づいてレースクラスを決定
            if prize_amount < 2900:
                race_grade = "オープン"
                info_dict["race_class"] = race_class_mapping[race_grade]
            elif 2900 <= prize_amount <= 5000:
                race_grade = "GⅢ"
                info_dict["race_class"] = race_class_mapping[race_grade]
            elif 5000 < prize_amount < 10000:
                race_grade = "GⅡ"
                info_dict["race_class"] = race_class_mapping[race_grade]
            elif 10000 <= prize_amount:
                race_grade = "GⅠ"
                info_dict["race_class"] = race_class_mapping[race_grade]
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

        df["course_type"] = df["place"].astype(str)+ df["race_type"].astype(str) + df["course_len"].astype(str) + df["course_len_type"].astype(str) 

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
        
        # 月を抽出して開催シーズンを判定
        def determine_season_2(month):
            if 4 <= month <= 6:
                return "2" #"夏開催"
            elif 1 <= month <= 3:
                return "1" #"冬開催"
            elif 7 <= month <= 9:
                return "3" #"春開催"
            elif 10 <= month <= 12:
                return "4" #"秋開催"    
        df["season_level"] = df["date"].dt.month.map(determine_season_2)
        df["season_level"] = df["season_level"].astype(int)
        

        # 月を抽出して開催シーズンを判定
        def determine_season_turf(month):
            if 6 <= month <= 8:
                return "4" #"夏開催"
            elif month == 12 or 1 <= month <= 2:
                return "2" #"冬開催"
            elif 3 <= month <= 5:
                return "3" #"春開催"
            elif 9 <= month <= 11:
                return "1" #"秋開催"    
        
        df["season_turf"] = df["date"].dt.month.map(determine_season_turf)
        df["day"] = df["day"].astype(str)

        df["day_season_turf"] =  df["day"] + df["season_turf"]
        df["day_season_turf"] =  df["day_season_turf"].astype(int)
        df["day"] = df["day"].astype(int)
        df["season_turf"] = df["season_turf"].astype(float)


        df["season_turf_condition"] = np.where(
            df["season_turf"] == 1, df["day"],
            np.where(
                df["season_turf"] == 2, (df["day"] + 1.5) * 1.5,
                np.where(
                    df["season_turf"] == 3, df["day"] + 3,
                    np.where(
                        df["season_turf"] == 4, df["day"] + 4,
                        df["day"]  # それ以外のとき NaN
                    )
                )
            )
        )
        df["season_turf_condition"] = df["season_turf_condition"].fillna(7)

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
            '5110': -1, '5120': -1, '5210': -1, '5220': -1, '5310': -1, '5320': -1, '5410': -2, '5420': -2,
            '8110': -1, '8120': -1, '8210': -1, '8220': -1, '8310': -1, '8320': -1, '8410': -2, '8420': -2,
            '1110': -2, '1120': -2, '1210': -2, '1220': -2, '1310': -2, '1320': -2, '1410': -3, '1420': -3,
            '2110': -2, '2120': -2, '2210': -2, '2220': -2, '2310': -2, '2320': -2, '2410': -3, '2420': -3,
            '10110': -2, '10120': -2, '10210': -2, '10220': -2, '10310': -2, '10320': -2, '10410': -3, '10420': -3,
            '3110': -2, '3120': -2, '3210': -2, '3220': -2, '3310': -2, '3320': -2, '3410': -3, '3420': -3,
            '4110': -2, '4120': -2, '4210': -2, '4220': -2, '4310': -2, '4320': -2, '4410': -3, '4420': -3,
            '9110': -2, '9120': -2, '9210': -2, '9220': -2, '9310': -2, '9320': -2, '9410': -3, '9420': -3,
            '7110': -3, '7120': -3, '7210': -3, '7220': -3, '7310': -3, '7320': -3, '7410': -4, '7420': -4,
            '6110': -3, '6120': -3, '6210': -3, '6220': -3, '6310': -3, '6320': -3, '6410': -4, '6420': -4
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
            if race_type in [2]:
                return None
            if race_type in [0]:
                return place + 1000 
                
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
        # {
        # "札幌": 1,
        # "函館": 2,
        # "福島": 3,
        # "新潟": 4,
        # "東京": 5,
        # "中山": 6,
        # "中京": 7,
        # "京都": 8,
        # "阪神": 9,
        # "小倉": 10
        #    
        # conversion_map = {
        #     404: {"直線": 1000, "カーブ": 4, "ゴール前": 1},
        #     402: {"直線": 658.7, "カーブ": 1, "ゴール前": 1},
        #     5: {"直線": 525.9, "カーブ": 5, "ゴール前": 2},
        #     902: {"直線": 473.6, "カーブ": 5, "ゴール前": 3},
        #     7: {"直線": 412.5, "カーブ": 4, "ゴール前": 3},
        #     802: {"直線": 403.7, "カーブ": 5, "ゴール前": 1},
        #     803: {"直線": 345, "カーブ": 5, "ゴール前": 1},
        #     401: {"直線": 358.7, "カーブ": 1, "ゴール前": 1},
        #     901: {"直線": 356.5, "カーブ": 5, "ゴール前": 3},    
        #     801: {"直線": 328.4, "カーブ": 5, "ゴール前": 1},    
        #     601: {"直線": 310, "カーブ": 2, "ゴール前": 3},
        #     602: {"直線": 310, "カーブ": 5, "ゴール前": 3},
        #     603: {"直線": 310, "カーブ": 5, "ゴール前": 3},
        #     1: {"直線": 266.1, "カーブ": 5, "ゴール前": 1},
        #     2: {"直線": 262.1, "カーブ": 2, "ゴール前": 1},
        #     3: {"直線": 292, "カーブ": 3, "ゴール前": 1},
        #     10: {"直線": 293, "カーブ": 3, "ゴール前": 1},
        #     1001: {"直線": 264.3, "カーブ": 5, "ゴール前": 1},
        #     1002: {"直線": 256.0, "カーブ": 2, "ゴール前": 1},
        #     1003: {"直線": 267.3, "カーブ": 3, "ゴール前": 1},
        #     1004: {"直線": 353.9, "カーブ": 1, "ゴール前": 1},
        #     1005: {"直線": 501.6, "カーブ": 5, "ゴール前": 2},
        #     1006: {"直線": 308.0, "カーブ": 2, "ゴール前": 3},
        #     1007: {"直線": 410.7, "カーブ": 4, "ゴール前": 3},
        #     1008: {"直線": 329.1, "カーブ": 5, "ゴール前": 1},
        #     1009: {"直線": 325.5, "カーブ": 5, "ゴール前": 3},
        #     1010: {"直線": 291.3, "カーブ": 3, "ゴール前": 1}
        # }

        # # データフレームに変換情報を適用する関数
        # def convert_course(row):
        #     place_code = row["place_course_category"]  # 競馬場の数値コード
        #     if place_code in conversion_map:
        #         # 直線、カーブ、ゴール前の情報を取得
        #         course_info = conversion_map[place_code]
        #         # 列名を変更
        #         return pd.Series({
        #             "goal_range": course_info["直線"], 
        #             "curve": course_info["カーブ"], 
        #             "goal_slope": course_info["ゴール前"]
        #         })
        #     else:
        #         return pd.Series({"goal_range": None, "curve": None, "goal_slope": None})
        
        # # 競馬場カテゴリに基づく変換を追加
        # df[['goal_range', 'curve', 'goal_slope']] = df.apply(convert_course, axis=1)
        
        # df["goal_range_100"] = df["goal_range"]/100
        
        
        """
        コーナータイプ
        緩やかで大きなカーブ:5
        小回:2
        普通:3
        膨らみ小回り:4
        膨らみ形状:5
        df["course_type"] 
        = df["place"].astype(str)+ df["race_type"].astype(str) + df["course_len"].astype(str) + df["course_len_type"].astype(str) 

        スタート位置、ダート1 芝2

        """
        df["course_len_type"] = df["course_len_type"].astype(int)
        
        df["course_type"] = df["course_type"].astype(int)
    
        conversion_map_course_type = {
            #ダート
            #札幌
            1010001:{"コーナー数": 2, "最終直線": 264, "ゴール前坂": 0,  "スタート位置": 1,"最初直線": 284.00,"直線合計": 548, "コーナー合計m":452,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":147,"高低差":0.9,"幅":20.0,"最初坂":0.4,"向正面坂":0.4,"最初コーナー坂":0,"最終コーナー坂":0.2},
            1010002:{"コーナー数": 2, "最終直線": 264, "ゴール前坂": 0,  "スタート位置": 1,"最初直線": 284.00,"直線合計": 548, "コーナー合計m":452,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":147,"高低差":0.9,"幅":20.0,"最初坂":0.4,"向正面坂":0.4,"最初コーナー坂":0,"最終コーナー坂":0.2},

            1017001:{"コーナー数": 4, "最終直線": 264, "ゴール前坂": 0,  "スタート位置": 1,"最初直線": 240.00,"直線合計": 796, "コーナー合計m":904,"コーナータイプ":4,"コーナーR12":147,"コーナーR34":147,"高低差":0.9,"幅":20.0,   "最初坂":0,"向正面坂":0.4,"最初コーナー坂":0.5,"最終コーナー坂":0.2},
            1017002:{"コーナー数": 4, "最終直線": 264, "ゴール前坂": 0,  "スタート位置": 1,"最初直線": 240.00,"直線合計": 796, "コーナー合計m":904,"コーナータイプ":4,"コーナーR12":147,"コーナーR34":147,"高低差":0.9,"幅":20.0,   "最初坂":0,"向正面坂":0.4,"最初コーナー坂":0.5,"最終コーナー坂":0.2},

            1024001:{"コーナー数": 6, "最終直線": 264, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線": 180.00,"直線合計": 1044, "コーナー合計m":1356,"コーナータイプ":4,"コーナーR12":147,"コーナーR34":147,"高低差":0.9,"幅":20.0,   "最初坂":0.3,"向正面坂":0.5,"最初コーナー坂":0,"最終コーナー坂":0.2},
            1024002:{"コーナー数": 6, "最終直線": 264, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線": 180.00,"直線合計": 1044, "コーナー合計m":1356,"コーナータイプ":4,"コーナーR12":147,"コーナーR34":147,"高低差":0.9,"幅":20.0,   "最初坂":0.3,"向正面坂":0.5,"最初コーナー坂":0,"最終コーナー坂":0.2},
            
            #函館
            2010001:{"コーナー数": 2, "最終直線": 260, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線": 366.4,"直線合計": 626, "コーナー合計m":374,"コーナータイプ":2,"コーナーR12":0,"コーナーR34":147,"高低差":3.4,"幅":20.0,   "最初坂":2,"向正面坂":2,"最初コーナー坂":1.25,"最終コーナー坂":-0.25},
            2010002:{"コーナー数": 2, "最終直線": 260, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線": 366.4,"直線合計": 626, "コーナー合計m":374,"コーナータイプ":2,"コーナーR12":0,"コーナーR34":147,"高低差":3.4,"幅":20.0,   "最初坂":2,"向正面坂":2,"最初コーナー坂":1.25,"最終コーナー坂":-0.25},

            2017001:{"コーナー数": 4, "最終直線": 260, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線": 328.5,"直線合計": 952, "コーナー合計m":748,"コーナータイプ":2,"コーナーR12":107,"コーナーR34":147,"高低差":3.4,"幅":20.0,   "最初坂":-1.5,"向正面坂":2,"最初コーナー坂":-1.4,"最終コーナー坂":-0.25},
            2017002:{"コーナー数": 4, "最終直線": 260, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線": 328.5,"直線合計": 952, "コーナー合計m":748,"コーナータイプ":2,"コーナーR12":107,"コーナーR34":147,"高低差":3.4,"幅":20.0,   "最初坂":-1.5,"向正面坂":2,"最初コーナー坂":-1.4,"最終コーナー坂":-0.25},

            2024001:{"コーナー数": 6, "最終直線": 260, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線": 290.6,"直線合計": 1278, "コーナー合計m":1122,"コーナータイプ":2,"コーナーR12":107,"コーナーR34":147,"高低差":3.4,"幅":20.0,   "最初坂":1.5,"向正面坂":2,"最初コーナー坂":1.25,"最終コーナー坂":-0.25},
            2024002:{"コーナー数": 6, "最終直線": 260, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線": 290.6,"直線合計": 1278, "コーナー合計m":1122,"コーナータイプ":2,"コーナーR12":107,"コーナーR34":147,"高低差":3.4,"幅":20.0,   "最初坂":1.5,"向正面坂":2,"最初コーナー坂":1.25,"最終コーナー坂":-0.25},

            #福島
            3010001:{"コーナー数": 2, "最終直線": 296, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 384.00,"直線合計": 780, "コーナー合計m":370,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":147,"高低差":2.1,"幅":20.0,   "最初坂":1.5,"向正面坂":1.7,"最初コーナー坂":0,"最終コーナー坂":-0.8},
            3010002:{"コーナー数": 2, "最終直線": 296, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 384.00,"直線合計": 780, "コーナー合計m":370,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":147,"高低差":2.1,"幅":20.0,   "最初坂":1.5,"向正面坂":1.7,"最初コーナー坂":0,"最終コーナー坂":-0.8},

            3011501:{"コーナー数": 2, "最終直線": 296, "ゴール前坂": 1,  "スタート位置": 2,    "最初直線": 484.00,"直線合計": 680, "コーナー合計m":370,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":147,"高低差":2.1,"幅":20.0,   "最初坂":1.5,"向正面坂":1.7,"最初コーナー坂":0,"最終コーナー坂":-0.8},
            3011502:{"コーナー数": 2, "最終直線": 296, "ゴール前坂": 1,  "スタート位置": 2,    "最初直線": 484.00,"直線合計": 680, "コーナー合計m":370,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":147,"高低差":2.1,"幅":20.0,   "最初坂":1.5,"向正面坂":1.7,"最初コーナー坂":0,"最終コーナー坂":-0.8},

            3017001:{"コーナー数": 4, "最終直線": 296, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 338.5,"直線合計": 960, "コーナー合計m":740,"コーナータイプ":3,"コーナーR12":113,"コーナーR34":147,"高低差":2.1,"幅":20.0,   "最初坂":1,"向正面坂":1.7,"最初コーナー坂":-2,"最終コーナー坂":-0.8},
            3017002:{"コーナー数": 4, "最終直線": 296, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 338.5,"直線合計": 960, "コーナー合計m":740,"コーナータイプ":3,"コーナーR12":113,"コーナーR34":147,"高低差":2.1,"幅":20.0,   "最初坂":1,"向正面坂":1.7,"最初コーナー坂":-2,"最終コーナー坂":-0.8},

            3024001:{"コーナー数": 6, "最終直線": 296, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 325.9,"直線合計": 1290, "コーナー合計m":1110,"コーナータイプ":3,"コーナーR12":113,"コーナーR34":147,"高低差":2.1,"幅":20.0,   "最初坂":1.7,"向正面坂":1.7,"最初コーナー坂":0,"最終コーナー坂":-0.8},
            3024002:{"コーナー数": 6, "最終直線": 296, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 325.9,"直線合計": 1290, "コーナー合計m":1110,"コーナータイプ":3,"コーナーR12":113,"コーナーR34":147,"高低差":2.1,"幅":20.0,   "最初坂":1.7,"向正面坂":1.7,"最初コーナー坂":0,"最終コーナー坂":-0.8},


            #新潟
            4012001:{"コーナー数": 2, "最終直線": 354, "ゴール前坂": 0,  "スタート位置": 2,    "最初直線": 524.9,"直線合計": 879, "コーナー合計m":321,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":104,"高低差":0.5,"幅":20.0,   "最初坂":0,"向正面坂":0,"最初コーナー坂":-0.5,"最終コーナー坂":0},
            4012002:{"コーナー数": 2, "最終直線": 354, "ゴール前坂": 0,  "スタート位置": 2,    "最初直線": 524.9,"直線合計": 879, "コーナー合計m":321,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":104,"高低差":0.5,"幅":20.0,   "最初坂":0,"向正面坂":0,"最初コーナー坂":-0.5,"最終コーナー坂":0},

            4018001:{"コーナー数": 4, "最終直線": 354, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線": 388.7,"直線合計": 1157, "コーナー合計m":643,"コーナータイプ":1,"コーナーR12":104,"コーナーR34":104,"高低差":0.5,"幅":20.0,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0.5,"最終コーナー坂":0},
            4018002:{"コーナー数": 4, "最終直線": 354, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線": 388.7,"直線合計": 1157, "コーナー合計m":643,"コーナータイプ":1,"コーナーR12":104,"コーナーR34":104,"高低差":0.5,"幅":20.0,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0.5,"最終コーナー坂":0},

            #東京
            5013001:{"コーナー数": 2, "最終直線": 501, "ゴール前坂": 1.25,  "スタート位置": 1,    "最初直線":341.9,"直線合計": 844, "コーナー合計m":456,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":0.9,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
            5013002:{"コーナー数": 2, "最終直線": 501, "ゴール前坂": 1.25,  "スタート位置": 1,    "最初直線":341.9,"直線合計": 844, "コーナー合計m":456,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":0.9,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
            
            5014001:{"コーナー数": 2, "最終直線": 501, "ゴール前坂": 1.25,  "スタート位置": 1,    "最初直線":441.9,"直線合計": 944, "コーナー合計m":456,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":-0.6,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
            5014002:{"コーナー数": 2, "最終直線": 501, "ゴール前坂": 1.25,  "スタート位置": 1,    "最初直線":441.9,"直線合計": 944, "コーナー合計m":456,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":-0.6,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},

            5016001:{"コーナー数": 2, "最終直線": 501, "ゴール前坂": 1.25, "スタート位置": 2,    "最初直線":641.9,"直線合計": 1144, "コーナー合計m":456,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":-1,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
            5016002:{"コーナー数": 2, "最終直線": 501, "ゴール前坂": 1.25,  "スタート位置": 2,    "最初直線":641.9,"直線合計": 1144, "コーナー合計m":456,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":-1,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},

            5021001:{"コーナー数": 4, "最終直線": 501, "ゴール前坂": 1.25,  "スタート位置": 1,    "最初直線":236.1,"直線合計": 1188, "コーナー合計m":912,"コーナータイプ":5,"コーナーR12":162,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":0,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
            5021002:{"コーナー数": 4, "最終直線": 501, "ゴール前坂": 1.25,  "スタート位置": 1,    "最初直線":236.1,"直線合計": 1188, "コーナー合計m":912,"コーナータイプ":5,"コーナーR12":162,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":0,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},

            5024001:{"コーナー数": 4, "最終直線": 501, "ゴール前坂": 1.25,  "スタート位置": 1,    "最初直線":536.1,"直線合計": 1488, "コーナー合計m":912,"コーナータイプ":5,"コーナーR12":162,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":2.5,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
            5024002:{"コーナー数": 4, "最終直線": 501, "ゴール前坂": 1.25,  "スタート位置": 1,    "最初直線":536.1,"直線合計": 1488, "コーナー合計m":912,"コーナータイプ":5,"コーナーR12":162,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":2.5,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},


            #中山
            6012001:{"コーナー数": 2, "最終直線": 308, "ゴール前坂": 2.3,  "スタート位置": 2,    "最初直線":502.6,"直線合計": 811, "コーナー合計m":389,"コーナータイプ":2,"コーナーR12":0,"コーナーR34":134,"高低差":4.4,"幅":20.0,   "最初坂":-3.5,"向正面坂":-3.5,"最初コーナー坂":-0.5,"最終コーナー坂":-0.1},
            6012002:{"コーナー数": 2, "最終直線": 308, "ゴール前坂": 2.3,  "スタート位置": 2,    "最初直線":502.6,"直線合計": 811, "コーナー合計m":389,"コーナータイプ":2,"コーナーR12":0,"コーナーR34":134,"高低差":4.4,"幅":20.0,   "最初坂":-3.5,"向正面坂":-3.5,"最初コーナー坂":-0.5,"最終コーナー坂":-0.1},

            6018001:{"コーナー数": 4, "最終直線": 308, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":375.0,"直線合計": 1022, "コーナー合計m":778,"コーナータイプ":2,"コーナーR12":134,"コーナーR34":134,"高低差":4.4,"幅":20.0,   "最初坂":3.5,"向正面坂":-3.5,"最初コーナー坂":0.7,"最終コーナー坂":-0.1},
            6018002:{"コーナー数": 4, "最終直線": 308, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":375.0,"直線合計": 1022, "コーナー合計m":778,"コーナータイプ":2,"コーナーR12":134,"コーナーR34":134,"高低差":4.4,"幅":20.0,   "最初坂":3.5,"向正面坂":-3.5,"最初コーナー坂":0.7,"最終コーナー坂":-0.1},

            6024001:{"コーナー数": 6, "最終直線": 308, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":209.6,"直線合計": 1233, "コーナー合計m":1167,"コーナータイプ":2,"コーナーR12":134,"コーナーR34":134,"高低差":4.4,"幅":20.0,   "最初坂":-1,"向正面坂":-3.5,"最初コーナー坂":-0.5,"最終コーナー坂":-0.1},
            6024002:{"コーナー数": 6, "最終直線": 308, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":209.6,"直線合計": 1233, "コーナー合計m":1167,"コーナータイプ":2,"コーナーR12":134,"コーナーR34":134,"高低差":4.4,"幅":20.0,   "最初坂":-1,"向正面坂":-3.5,"最初コーナー坂":-0.5,"最終コーナー坂":-0.1},

            6025001:{"コーナー数": 6, "最終直線": 308, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":309.6,"直線合計": 1333, "コーナー合計m":1167,"コーナータイプ":2,"コーナーR12":134,"コーナーR34":134,"高低差":4.4,"幅":20.0,   "最初坂":-3.3,"向正面坂":-3.5,"最初コーナー坂":-0.5,"最終コーナー坂":-0.1},
            6025002:{"コーナー数": 6, "最終直線": 308, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":309.6,"直線合計": 1333, "コーナー合計m":1167,"コーナータイプ":2,"コーナーR12":134,"コーナーR34":134,"高低差":4.4,"幅":20.0,   "最初坂":-3.3,"向正面坂":-3.5,"最初コーナー坂":-0.5,"最終コーナー坂":-0.1},


            #中京       
            7012001:{"コーナー数": 2, "最終直線": 410.7, "ゴール前坂": 1.2,  "スタート位置": 1,    "最初直線":407.7,"直線合計": 818, "コーナー合計m":382,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":1.2,"向正面坂":-1.2,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
            7012002:{"コーナー数": 2, "最終直線": 410.7, "ゴール前坂": 1.2,  "スタート位置": 1,    "最初直線":407.7,"直線合計": 818, "コーナー合計m":382,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":1.2,"向正面坂":-1.2,"最初コーナー坂":-1.4,"最終コーナー坂":-1},

            7014001:{"コーナー数": 2, "最終直線": 410.7, "ゴール前坂": 1.2, "スタート位置": 2,    "最初直線":607.7,"直線合計": 1018, "コーナー合計m":382,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":1.2,"向正面坂":-1.2,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
            7014002:{"コーナー数": 2, "最終直線": 410.7, "ゴール前坂": 1.2,  "スタート位置": 2,    "最初直線":607.7,"直線合計": 1018, "コーナー合計m":382,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":1.2,"向正面坂":-1.2,"最初コーナー坂":-1.4,"最終コーナー坂":-1},

            7018001:{"コーナー数": 4, "最終直線": 410.7, "ゴール前坂": 1.2, "スタート位置": 1,    "最初直線":291.8,"直線合計": 1103, "コーナー合計m":697,"コーナータイプ":3,"コーナーR12":95,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":1.5,"向正面坂":-1.2,"最初コーナー坂":0,"最終コーナー坂":-1},
            7018001:{"コーナー数": 4, "最終直線": 410.7, "ゴール前坂": 1.2,  "スタート位置": 1,    "最初直線":291.8,"直線合計": 1103, "コーナー合計m":697,"コーナータイプ":3,"コーナーR12":95,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":1.5,"向正面坂":-1.2,"最初コーナー坂":0,"最終コーナー坂":-1},

            7019001:{"コーナー数": 4, "最終直線": 410.7, "ゴール前坂": 1.2,  "スタート位置": 1,    "最初直線":391.80,"直線合計": 1203, "コーナー合計m":697,"コーナータイプ":3,"コーナーR12":95,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":2,"向正面坂":-1.2,"最初コーナー坂":0,"最終コーナー坂":-1},
            7019002:{"コーナー数": 4, "最終直線": 410.7, "ゴール前坂": 1.2, "スタート位置": 1,    "最初直線":391.80,"直線合計": 1203, "コーナー合計m":697,"コーナータイプ":3,"コーナーR12":95,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":2,"向正面坂":-1.2,"最初コーナー坂":0,"最終コーナー坂":-1},


            #京都
            8012001:{"コーナー数": 2, "最終直線": 329, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":409.6,"直線合計": 739, "コーナー合計m":461,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":122,"高低差":3.0,"幅":25.0,   "最初坂":2.2,"向正面坂":2.2,"最初コーナー坂":-2.2,"最終コーナー坂":0},
            8012002:{"コーナー数": 2, "最終直線": 329, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":409.6,"直線合計": 739, "コーナー合計m":461,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":122,"高低差":3.0,"幅":25.0,   "最初坂":2.2,"向正面坂":2.2,"最初コーナー坂":-2.2,"最終コーナー坂":0},

            8014001:{"コーナー数": 2, "最終直線": 329, "ゴール前坂": 0,  "スタート位置": 2,    "最初直線":609.6,"直線合計": 939, "コーナー合計m":461,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":122,"高低差":3.0,"幅":25.0,   "最初坂":0,"向正面坂":2.2,"最初コーナー坂":-2.2,"最終コーナー坂":0},
            8014002:{"コーナー数": 2, "最終直線": 329, "ゴール前坂": 0,  "スタート位置": 2,    "最初直線":609.6,"直線合計": 939, "コーナー合計m":461,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":122,"高低差":3.0,"幅":25.0,   "最初坂":0,"向正面坂":2.2,"最初コーナー坂":-2.2,"最終コーナー坂":0},
        
            8018001:{"コーナー数": 4, "最終直線": 329, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":285.8,"直線合計": 1055, "コーナー合計m":745,"コーナータイプ":4,"コーナーR12":102,"コーナーR34":122,"高低差":3.0,"幅":25.0,   "最初坂":0,"向正面坂":2.2,"最初コーナー坂":0.2,"最終コーナー坂":0},
            8018002:{"コーナー数": 4, "最終直線": 329, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":285.8,"直線合計": 1055, "コーナー合計m":745,"コーナータイプ":4,"コーナーR12":102,"コーナーR34":122,"高低差":3.0,"幅":25.0,   "最初坂":0,"向正面坂":2.2,"最初コーナー坂":0.2,"最終コーナー坂":0},
        
            8019001:{"コーナー数": 4, "最終直線": 329, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":385.8,"直線合計": 1155, "コーナー合計m":745,"コーナータイプ":4,"コーナーR12":102,"コーナーR34":122,"高低差":3.0,"幅":25.0,   "最初坂":0,"向正面坂":2.2,"最初コーナー坂":0.2,"最終コーナー坂":0},
            8019002:{"コーナー数": 4, "最終直線": 329, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":385.8,"直線合計": 1155, "コーナー合計m":745,"コーナータイプ":4,"コーナーR12":102,"コーナーR34":122,"高低差":3.0,"幅":25.0,   "最初坂":0,"向正面坂":2.2,"最初コーナー坂":0.2,"最終コーナー坂":0},
        

            #阪神
            9012001:{"コーナー数": 2, "最終直線": 353, "ゴール前坂": 1.2,  "スタート位置": 1,    "最初直線":343.6,"直線合計": 696, "コーナー合計m":504,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":159,"高低差":1.5,"幅":22.0,   "最初坂":0,"向正面坂":-0.4,"最初コーナー坂":-0.7,"最終コーナー坂":-0.5},
            9012002:{"コーナー数": 2, "最終直線": 353, "ゴール前坂": 1.2,  "スタート位置": 1,    "最初直線":343.6,"直線合計": 696, "コーナー合計m":504,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":159,"高低差":1.5,"幅":22.0,   "最初坂":0,"向正面坂":-0.4,"最初コーナー坂":-0.7,"最終コーナー坂":-0.5},
        
            9014001:{"コーナー数": 2, "最終直線": 353, "ゴール前坂": 1.2,  "スタート位置": 2,    "最初直線":543.6,"直線合計": 896, "コーナー合計m":504,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":159,"高低差":1.5,"幅":22.0,   "最初坂":0,"向正面坂":-0.4,"最初コーナー坂":-0.7,"最終コーナー坂":-0.5},
            9014002:{"コーナー数": 2, "最終直線": 353, "ゴール前坂": 1.2,  "スタート位置": 2,    "最初直線":543.6,"直線合計": 896, "コーナー合計m":504,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":159,"高低差":1.5,"幅":22.0,   "最初坂":0,"向正面坂":-0.4,"最初コーナー坂":-0.7,"最終コーナー坂":-0.5},
        
            9018001:{"コーナー数": 4, "最終直線": 353, "ゴール前坂": 1.2,  "スタート位置": 1,    "最初直線":298.2,"直線合計": 896, "コーナー合計m":504,"コーナータイプ":4,"コーナーR12":112,"コーナーR34":159,"高低差":1.5,"幅":22.0,   "最初坂":1.2,"向正面坂":-0.4,"最初コーナー坂":-0,"最終コーナー坂":-0.5},
            9018002:{"コーナー数": 4, "最終直線": 353, "ゴール前坂": 1.2,  "スタート位置": 1,    "最初直線":298.2,"直線合計": 896, "コーナー合計m":504,"コーナータイプ":4,"コーナーR12":112,"コーナーR34":159,"高低差":1.5,"幅":22.0,   "最初坂":1.2,"向正面坂":-0.4,"最初コーナー坂":-0,"最終コーナー坂":-0.5},

            9020001:{"コーナー数": 4, "最終直線": 353, "ゴール前坂": 1.2,  "スタート位置": 2,    "最初直線":498.2,"直線合計": 1221, "コーナー合計m":779,"コーナータイプ":4,"コーナーR12":112,"コーナーR34":159,"高低差":1.5,"幅":22.0,   "最初坂":1.2,"向正面坂":-0.4,"最初コーナー坂":-0,"最終コーナー坂":-0.5},
            9020002:{"コーナー数": 4, "最終直線": 353, "ゴール前坂": 1.2,  "スタート位置": 2,    "最初直線":498.2,"直線合計": 1221, "コーナー合計m":779,"コーナータイプ":4,"コーナーR12":112,"コーナーR34":159,"高低差":1.5,"幅":22.0,   "最初坂":1.2,"向正面坂":-0.4,"最初コーナー坂":-0,"最終コーナー坂":-0.5},


            #小倉
            10010001:{"コーナー数": 2, "最終直線": 291, "ゴール前坂": 0.5,  "スタート位置": 1,    "最初直線":370.0,"直線合計": 661, "コーナー合計m":339,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":112,"高低差":2.9,"幅":24.0,   "最初坂":-0.7,"向正面坂":-0.7,"最初コーナー坂":-1.2,"最終コーナー坂":-0.3},
            10010002:{"コーナー数": 2, "最終直線": 291, "ゴール前坂": 0.5,  "スタート位置": 1,    "最初直線":370.0,"直線合計": 661, "コーナー合計m":339,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":112,"高低差":2.9,"幅":24.0,   "最初坂":-0.7,"向正面坂":-0.7,"最初コーナー坂":-1.2,"最終コーナー坂":-0.3},
        
            10017001:{"コーナー数": 4, "最終直線": 291, "ゴール前坂": 0.5,  "スタート位置": 1,    "最初直線":343.0,"直線合計": 1022, "コーナー合計m":678,"コーナータイプ":1,"コーナーR12":112,"コーナーR34":112,"高低差":2.9,"幅":24.0,   "最初坂":0.4,"向正面坂":-0.7,"最初コーナー坂":2.2,"最終コーナー坂":-0.3},
            10017002:{"コーナー数": 4, "最終直線": 291, "ゴール前坂": 0.5,  "スタート位置": 1,    "最初直線":343.0,"直線合計": 1022, "コーナー合計m":678,"コーナータイプ":1,"コーナーR12":112,"コーナーR34":112,"高低差":2.9,"幅":24.0,   "最初坂":0.4,"向正面坂":-0.7,"最初コーナー坂":2.2,"最終コーナー坂":-0.3},
        
            10024001:{"コーナー数": 6, "最終直線": 291, "ゴール前坂": 0.5,  "スタート位置": 1,    "最初直線":320.0,"直線合計": 1383, "コーナー合計m":1017,"コーナータイプ":1,"コーナーR12":112,"コーナーR34":112,"高低差":2.9,"幅":24.0,   "最初坂":0,"向正面坂":-0.7,"最初コーナー坂":-1.2,"最終コーナー坂":-0.3},
            10024002:{"コーナー数": 6, "最終直線": 291, "ゴール前坂": 0.5,  "スタート位置": 1,    "最初直線":320.0,"直線合計": 1383, "コーナー合計m":1017,"コーナータイプ":1,"コーナーR12":112,"コーナーR34":112,"高低差":2.9,"幅":24.0,   "最初坂":0,"向正面坂":-0.7,"最初コーナー坂":-1.2,"最終コーナー坂":-0.3},
        



            # "門別": 30,
            # "盛岡": 35,
            # "水沢": 36,
            # "浦和": 42,
            # "船橋": 43,
            # "大井": 44,
            # "川崎": 45,
            # "金沢": 46,
            # "笠松": 47,
            # "名古屋": 48,
            # "園田": 50,
            # "高知": 54,
            # "姫路": 51,
            # "佐賀": 55


            #門別
            30010001:{"コーナー数": 2, "最終直線": 330, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":290.0,"直線合計": 720, "コーナー合計m":280,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":90,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            30010002:{"コーナー数": 2, "最終直線": 330, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":290.0,"直線合計": 720, "コーナー合計m":280,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":90,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            30011001:{"コーナー数": 2, "最終直線": 330, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":390.0,"直線合計": 820, "コーナー合計m":370,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":90,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            30011002:{"コーナー数": 2, "最終直線": 330, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":390.0,"直線合計": 820, "コーナー合計m":370,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":90,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            30012001:{"コーナー数": 2, "最終直線": 330, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":490.0,"直線合計": 920, "コーナー合計m":370,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":97,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            30012002:{"コーナー数": 2, "最終直線": 330, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":490.0,"直線合計": 920, "コーナー合計m":370,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":97,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
        
            30015001:{"コーナー数": 4, "最終直線": 218, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":168.0,"直線合計": 674, "コーナー合計m":736,"コーナータイプ":3,"コーナーR12":88,"コーナーR34":97,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            30015002:{"コーナー数": 4, "最終直線": 218, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":168.0,"直線合計": 674, "コーナー合計m":736,"コーナータイプ":3,"コーナーR12":88,"コーナーR34":97,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            30016001:{"コーナー数": 4, "最終直線": 218, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":268.0,"直線合計": 774, "コーナー合計m":736,"コーナータイプ":3,"コーナーR12":88,"コーナーR34":97,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            30016002:{"コーナー数": 4, "最終直線": 218, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":268.0,"直線合計": 774, "コーナー合計m":736,"コーナータイプ":3,"コーナーR12":88,"コーナーR34":97,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            30017001:{"コーナー数": 4, "最終直線": 330, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":258.0,"直線合計": 988, "コーナー合計m":738,"コーナータイプ":4,"コーナーR12":88,"コーナーR34":97,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            30017002:{"コーナー数": 4, "最終直線": 330, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":258.0,"直線合計": 988, "コーナー合計m":738,"コーナータイプ":4,"コーナーR12":88,"コーナーR34":97,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            30018001:{"コーナー数": 4, "最終直線": 330, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":358.0,"直線合計": 1088, "コーナー合計m":738,"コーナータイプ":4,"コーナーR12":88,"コーナーR34":97,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            30018002:{"コーナー数": 4, "最終直線": 330, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":358.0,"直線合計": 1088, "コーナー合計m":738,"コーナータイプ":4,"コーナーR12":88,"コーナーR34":97,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            30020001:{"コーナー数": 4, "最終直線": 530, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":358.0,"直線合計": 1088, "コーナー合計m":738,"コーナータイプ":4,"コーナーR12":88,"コーナーR34":97,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            30020002:{"コーナー数": 4, "最終直線": 530, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":358.0,"直線合計": 1088, "コーナー合計m":738,"コーナータイプ":4,"コーナーR12":88,"コーナーR34":97,"高低差":1.54,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},


            #盛岡

            35010001:{"コーナー数": 2, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":280.0,"直線合計": 580, "コーナー合計m":420,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":1.25,"向正面坂":0,"最初コーナー坂":-3.5,"最終コーナー坂":-2},
            35010002:{"コーナー数": 2, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":280.0,"直線合計": 580, "コーナー合計m":420,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":1.25,"向正面坂":0,"最初コーナー坂":-3.5,"最終コーナー坂":-2},

            35012001:{"コーナー数": 2, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":480.0,"直線合計": 780, "コーナー合計m":420,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":2,"向正面坂":0,"最初コーナー坂":-3.5,"最終コーナー坂":-2},
            35012002:{"コーナー数": 2, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":480.0,"直線合計": 780, "コーナー合計m":420,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":2,"向正面坂":0,"最初コーナー坂":-3.5,"最終コーナー坂":-2},

            35014001:{"コーナー数": 2, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":680.0,"直線合計": 980, "コーナー合計m":420,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":2.5,"向正面坂":0,"最初コーナー坂":-3.5,"最終コーナー坂":-2},
            35014002:{"コーナー数": 2, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":680.0,"直線合計": 980, "コーナー合計m":420,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":2.5,"向正面坂":0,"最初コーナー坂":-3.5,"最終コーナー坂":-2},

            35016001:{"コーナー数": 2, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":880.0,"直線合計": 1180, "コーナー合計m":420,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":2.5,"向正面坂":0,"最初コーナー坂":-3.5,"最終コーナー坂":-2},
            35016002:{"コーナー数": 2, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":880.0,"直線合計": 1180, "コーナー合計m":420,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":2.5,"向正面坂":0,"最初コーナー坂":-3.5,"最終コーナー坂":-2},
            
            35018001:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 1000, "コーナー合計m":800,"コーナータイプ":4,"コーナーR12":88,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":1.5,"向正面坂":2,"最初コーナー坂":0,"最終コーナー坂":-2},
            35018002:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 1000, "コーナー合計m":800,"コーナータイプ":4,"コーナーR12":88,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":1.5,"向正面坂":2,"最初コーナー坂":0,"最終コーナー坂":-2},

            35020001:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":500.0,"直線合計": 1200, "コーナー合計m":800,"コーナータイプ":4,"コーナーR12":88,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":2,"向正面坂":2,"最初コーナー坂":0,"最終コーナー坂":-2},
            35020002:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":500.0,"直線合計": 1200, "コーナー合計m":800,"コーナータイプ":4,"コーナーR12":88,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":2,"向正面坂":2,"最初コーナー坂":0,"最終コーナー坂":-2},

            35025001:{"コーナー数": 6, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":180.0,"直線合計": 1280, "コーナー合計m":1220,"コーナータイプ":4,"コーナーR12":88,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":1,"向正面坂":2,"最初コーナー坂":-3.5,"最終コーナー坂":-2},
            35025002:{"コーナー数": 6, "最終直線": 300, "ゴール前坂": 1.5,  "スタート位置": 1,    "最初直線":180.0,"直線合計": 1280, "コーナー合計m":1220,"コーナータイプ":4,"コーナーR12":88,"コーナーR34":88,"高低差":4.4,"幅":25,   "最初坂":1,"向正面坂":2,"最初コーナー坂":-3.5,"最終コーナー坂":-2},

            #盛岡芝
            35110001:{"コーナー数": 2, "最終直線": 300, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":400.0,"直線合計": 700, "コーナー合計m":300,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":80,"高低差":4.6,"幅":25,   "最初坂":1.5,"向正面坂":0,"最初コーナー坂":-4.5,"最終コーナー坂":-2},
            35110002:{"コーナー数": 2, "最終直線": 300, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":400.0,"直線合計": 700, "コーナー合計m":300,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":80,"高低差":4.6,"幅":25,   "最初坂":1.5,"向正面坂":0,"最初コーナー坂":-4.5,"最終コーナー坂":-2},
            
            35116001:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 1000, "コーナー合計m":600,"コーナータイプ":3,"コーナーR12":80,"コーナーR34":80,"高低差":4.6,"幅":25,   "最初坂":2.5,"向正面坂":2.0,"最初コーナー坂":-0.5,"最終コーナー坂":-2.5},
            35116002:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 1000, "コーナー合計m":600,"コーナータイプ":3,"コーナーR12":80,"コーナーR34":80,"高低差":4.6,"幅":25,   "最初坂":2.5,"向正面坂":2.0,"最初コーナー坂":-0.5,"最終コーナー坂":-2.5},

            35117001:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":400.0,"直線合計": 1100, "コーナー合計m":600,"コーナータイプ":3,"コーナーR12":80,"コーナーR34":80,"高低差":4.6,"幅":25,   "最初坂":2.5,"向正面坂":2.0,"最初コーナー坂":-0.5,"最終コーナー坂":-2.5},
            35117002:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":400.0,"直線合計": 1100, "コーナー合計m":600,"コーナータイプ":3,"コーナーR12":80,"コーナーR34":80,"高低差":4.6,"幅":25,   "最初坂":2.5,"向正面坂":2.0,"最初コーナー坂":-0.5,"最終コーナー坂":-2.5},
        
            35124001:{"コーナー数": 6, "最終直線": 300, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":400.0,"直線合計": 1500, "コーナー合計m":900,"コーナータイプ":3,"コーナーR12":80,"コーナーR34":80,"高低差":4.6,"幅":25,   "最初坂":1.5,"向正面坂":2.0,"最初コーナー坂":-4.5,"最終コーナー坂":-2},
            35124002:{"コーナー数": 6, "最終直線": 300, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":400.0,"直線合計": 1500, "コーナー合計m":900,"コーナータイプ":3,"コーナーR12":80,"コーナーR34":80,"高低差":4.6,"幅":25,   "最初坂":1.5,"向正面坂":2.0,"最初コーナー坂":-4.5,"最終コーナー坂":-2},


            #水沢
            3608501:{"コーナー数": 2, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":317.0,"直線合計": 562, "コーナー合計m":288,"コーナータイプ":2,"コーナーR12":0,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            3608502:{"コーナー数": 2, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":317.0,"直線合計": 562, "コーナー合計m":288,"コーナータイプ":2,"コーナーR12":0,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
                    
            36013001:{"コーナー数": 4, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":200.0,"直線合計": 762, "コーナー合計m":576,"コーナータイプ":2,"コーナーR12":76,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            36013002:{"コーナー数": 4, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":200.0,"直線合計": 762, "コーナー合計m":576,"コーナータイプ":2,"コーナーR12":76,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            36014001:{"コーナー数": 4, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 862, "コーナー合計m":576,"コーナータイプ":2,"コーナーR12":76,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            36014002:{"コーナー数": 4, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 862, "コーナー合計m":576,"コーナータイプ":2,"コーナーR12":76,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
        
            36016001:{"コーナー数": 4, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":500.0,"直線合計": 1062, "コーナー合計m":576,"コーナータイプ":2,"コーナーR12":76,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            36016002:{"コーナー数": 4, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":500.0,"直線合計": 1062, "コーナー合計m":576,"コーナータイプ":2,"コーナーR12":76,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
        
            36018001:{"コーナー数": 6, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":67.0,"直線合計": 929, "コーナー合計m":864,"コーナータイプ":2,"コーナーR12":76,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            36018002:{"コーナー数": 6, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":67.0,"直線合計": 929, "コーナー合計m":864,"コーナータイプ":2,"コーナーR12":76,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
        
            36019001:{"コーナー数": 6, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":167.0,"直線合計": 1029, "コーナー合計m":864,"コーナータイプ":2,"コーナーR12":76,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            36019002:{"コーナー数": 6, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":167.0,"直線合計": 1029, "コーナー合計m":864,"コーナータイプ":2,"コーナーR12":76,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
        
            36020001:{"コーナー数": 6, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":267.0,"直線合計": 1129, "コーナー合計m":864,"コーナータイプ":2,"コーナーR12":76,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            36020002:{"コーナー数": 6, "最終直線": 245, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":267.0,"直線合計": 1129, "コーナー合計m":864,"コーナータイプ":2,"コーナーR12":76,"コーナーR34":76,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
        

            #浦和
            4208001:{"コーナー数": 2, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 520, "コーナー合計m":280,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            4208001:{"コーナー数": 2, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 520, "コーナー合計m":280,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            42013001:{"コーナー数": 4, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":220.0,"直線合計": 740, "コーナー合計m":560,"コーナータイプ":3,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            42013002:{"コーナー数": 4, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":220.0,"直線合計": 740, "コーナー合計m":560,"コーナータイプ":3,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            42014001:{"コーナー数": 4, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":320.0,"直線合計": 840, "コーナー合計m":560,"コーナータイプ":3,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            42014002:{"コーナー数": 4, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":320.0,"直線合計": 840, "コーナー合計m":560,"コーナータイプ":3,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            42015001:{"コーナー数": 4, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":420.0,"直線合計": 940, "コーナー合計m":560,"コーナータイプ":3,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            42015002:{"コーナー数": 4, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":420.0,"直線合計": 940, "コーナー合計m":560,"コーナータイプ":3,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            42016001:{"コーナー数": 5, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":180.0,"直線合計": 940, "コーナー合計m":660,"コーナータイプ":3,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            42016002:{"コーナー数": 5, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":180.0,"直線合計": 940, "コーナー合計m":660,"コーナータイプ":3,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            42019001:{"コーナー数": 6, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":200.0,"直線合計": 1060, "コーナー合計m":840,"コーナータイプ":3,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            42019002:{"コーナー数": 6, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":200.0,"直線合計": 1060, "コーナー合計m":840,"コーナータイプ":3,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            42020001:{"コーナー数": 6, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 1160, "コーナー合計m":840,"コーナータイプ":3,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            42020002:{"コーナー数": 6, "最終直線": 220, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 1160, "コーナー合計m":840,"コーナータイプ":3,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            

            #船橋
            4308001:{"コーナー数": 2, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":140.0,"直線合計": 448, "コーナー合計m":352,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":99,"高低差":0,"幅":25,  "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            4308002:{"コーナー数": 2, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":140.0,"直線合計": 448, "コーナー合計m":352,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            43010001:{"コーナー数": 2, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":340.0,"直線合計": 648, "コーナー合計m":352,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            43010002:{"コーナー数": 2, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":340.0,"直線合計": 648, "コーナー合計m":352,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            43012001:{"コーナー数": 2, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":540.0,"直線合計": 848, "コーナー合計m":352,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            43012002:{"コーナー数": 2, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":540.0,"直線合計": 848, "コーナー合計m":352,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            43014001:{"コーナー数": 4, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":126.0,"直線合計": 796, "コーナー合計m":604,"コーナータイプ":4,"コーナーR12":91,"コーナーR34":91,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            43014002:{"コーナー数": 4, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":126.0,"直線合計": 796, "コーナー合計m":604,"コーナータイプ":4,"コーナーR12":91,"コーナーR34":91,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            43015001:{"コーナー数": 4, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":226.0,"直線合計": 896, "コーナー合計m":604,"コーナータイプ":4,"コーナーR12":91,"コーナーR34":91,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            43015002:{"コーナー数": 4, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":126.0,"直線合計": 796, "コーナー合計m":704,"コーナータイプ":4,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            43016001:{"コーナー数": 4, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":226.0,"直線合計": 896, "コーナー合計m":704,"コーナータイプ":4,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            43016002:{"コーナー数": 4, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":226.0,"直線合計": 896, "コーナー合計m":704,"コーナータイプ":4,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            43017001:{"コーナー数": 4, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":326.0,"直線合計": 996, "コーナー合計m":704,"コーナータイプ":4,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            43017002:{"コーナー数": 4, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":326.0,"直線合計": 996, "コーナー合計m":704,"コーナータイプ":4,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            43018001:{"コーナー数": 4, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":426.0,"直線合計": 1096, "コーナー合計m":704,"コーナータイプ":4,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            43018002:{"コーナー数": 4, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":426.0,"直線合計": 1096, "コーナー合計m":704,"コーナータイプ":4,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            43020001:{"コーナー数": 5, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":492.0,"直線合計": 1262, "コーナー合計m":870,"コーナータイプ":4,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            43020002:{"コーナー数": 5, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":492.0,"直線合計": 1262, "コーナー合計m":870,"コーナータイプ":4,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            43022001:{"コーナー数": 6, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":140.0,"直線合計": 1222, "コーナー合計m":1178,"コーナータイプ":4,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            43022002:{"コーナー数": 6, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":140.0,"直線合計": 1222, "コーナー合計m":1178,"コーナータイプ":4,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            43024001:{"コーナー数": 6, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":340.0,"直線合計": 1422, "コーナー合計m":1178,"コーナータイプ":4,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            43024002:{"コーナー数": 6, "最終直線": 308, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":340.0,"直線合計": 1422, "コーナー合計m":1178,"コーナータイプ":4,"コーナーR12":99,"コーナーR34":99,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            


            #大井

            44010001:{"コーナー数": 2, "最終直線": 386, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":243.0,"直線合計": 629, "コーナー合計m":371,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            44010002:{"コーナー数": 2, "最終直線": 386, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":243.0,"直線合計": 629, "コーナー合計m":371,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            44012001:{"コーナー数": 2, "最終直線": 386, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":443.0,"直線合計": 829, "コーナー合計m":371,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            44012002:{"コーナー数": 2, "最終直線": 386, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":443.0,"直線合計": 829, "コーナー合計m":371,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            44014001:{"コーナー数": 3, "最終直線": 386, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":200.0,"直線合計": 1029, "コーナー合計m":371,"コーナータイプ":4,"コーナーR12":51,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            44014002:{"コーナー数": 3, "最終直線": 386, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":200.0,"直線合計": 1029, "コーナー合計m":371,"コーナータイプ":4,"コーナーR12":51,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            44015001:{"コーナー数": 4, "最終直線": 286, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":240.0,"直線合計": 912, "コーナー合計m":588,"コーナータイプ":5,"コーナーR12":123,"コーナーR34":123,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            44015002:{"コーナー数": 4, "最終直線": 286, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":240.0,"直線合計": 912, "コーナー合計m":588,"コーナータイプ":5,"コーナーR12":123,"コーナーR34":123,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            44016001:{"コーナー数": 4, "最終直線": 286, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":340.0,"直線合計": 1012, "コーナー合計m":588,"コーナータイプ":5,"コーナーR12":123,"コーナーR34":123,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            44016002:{"コーナー数": 4, "最終直線": 286, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":340.0,"直線合計": 1012, "コーナー合計m":588,"コーナータイプ":5,"コーナーR12":123,"コーナーR34":123,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            44016501:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":243.0,"直線合計": 1029, "コーナー合計m":674,"コーナータイプ":5,"コーナーR12":102,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            44016502:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":243.0,"直線合計": 1029, "コーナー合計m":674,"コーナータイプ":5,"コーナーR12":102,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            44017001:{"コーナー数": 4, "最終直線": 386, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":240.0,"直線合計": 1026, "コーナー合計m":674,"コーナータイプ":4,"コーナーR12":102,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            44017002:{"コーナー数": 4, "最終直線": 386, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":240.0,"直線合計": 1026, "コーナー合計m":674,"コーナータイプ":4,"コーナーR12":102,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            44018001:{"コーナー数": 4, "最終直線": 386, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":340.0,"直線合計": 1126, "コーナー合計m":674,"コーナータイプ":4,"コーナーR12":102,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            44018002:{"コーナー数": 4, "最終直線": 386, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":340.0,"直線合計": 1126, "コーナー合計m":674,"コーナータイプ":4,"コーナーR12":102,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            44020001:{"コーナー数": 4, "最終直線": 386, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":540.0,"直線合計": 1326, "コーナー合計m":674,"コーナータイプ":4,"コーナーR12":102,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            44020002:{"コーナー数": 4, "最終直線": 386, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":540.0,"直線合計": 1326, "コーナー合計m":674,"コーナータイプ":4,"コーナーR12":102,"コーナーR34":102,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},


            #川崎
            4509001:{"コーナー数": 2, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":400.0,"直線合計": 700, "コーナー合計m":200,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":81,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            4509002:{"コーナー数": 2, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":400.0,"直線合計": 700, "コーナー合計m":200,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":81,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
        
            45014001:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 1000, "コーナー合計m":400,"コーナータイプ":1,"コーナーR12":81,"コーナーR34":81,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            45014002:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 1000, "コーナー合計m":400,"コーナータイプ":1,"コーナーR12":81,"コーナーR34":81,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            45015001:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":400.0,"直線合計": 1100, "コーナー合計m":400,"コーナータイプ":1,"コーナーR12":81,"コーナーR34":81,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            45015002:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":400.0,"直線合計": 1100, "コーナー合計m":400,"コーナータイプ":1,"コーナーR12":81,"コーナーR34":81,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
        
            45016001:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":500.0,"直線合計": 1200, "コーナー合計m":400,"コーナータイプ":1,"コーナーR12":81,"コーナーR34":81,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            45016002:{"コーナー数": 4, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":500.0,"直線合計": 1200, "コーナー合計m":400,"コーナータイプ":1,"コーナーR12":81,"コーナーR34":81,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
        
            45020001:{"コーナー数": 6, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 1400, "コーナー合計m":600,"コーナータイプ":1,"コーナーR12":81,"コーナーR34":81,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            45020002:{"コーナー数": 6, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 1400, "コーナー合計m":600,"コーナータイプ":1,"コーナーR12":81,"コーナーR34":81,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            45021001:{"コーナー数": 6, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":400.0,"直線合計": 1500, "コーナー合計m":600,"コーナータイプ":1,"コーナーR12":81,"コーナーR34":81,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            45021001:{"コーナー数": 6, "最終直線": 300, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":400.0,"直線合計": 1500, "コーナー合計m":600,"コーナータイプ":1,"コーナーR12":81,"コーナーR34":81,"高低差":0,"幅":25,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            #金沢
            4609001:{"コーナー数": 2, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":286.0,"直線合計": 522, "コーナー合計m":378,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            4609002:{"コーナー数": 2, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":286.0,"直線合計": 522, "コーナー合計m":378,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            46013001:{"コーナー数": 4, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":166.0,"直線合計": 688, "コーナー合計m":612,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            46013001:{"コーナー数": 4, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":166.0,"直線合計": 688, "コーナー合計m":612,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            46014001:{"コーナー数": 4, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":266.0,"直線合計": 788, "コーナー合計m":612,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            46014002:{"コーナー数": 4, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":266.0,"直線合計": 788, "コーナー合計m":612,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            46015001:{"コーナー数": 4, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":366.0,"直線合計": 888, "コーナー合計m":612,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            46015002:{"コーナー数": 4, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":366.0,"直線合計": 888, "コーナー合計m":612,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            46017001:{"コーナー数": 5, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":161.0,"直線合計": 949, "コーナー合計m":751,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            46017002:{"コーナー数": 5, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":161.0,"直線合計": 949, "コーナー合計m":751,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            46019001:{"コーナー数": 6, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":86.0,"直線合計": 894, "コーナー合計m":1006,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            46019001:{"コーナー数": 6, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":86.0,"直線合計": 894, "コーナー合計m":1006,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            46020001:{"コーナー数": 6, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":186.0,"直線合計": 994, "コーナー合計m":1006,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            46020002:{"コーナー数": 6, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":186.0,"直線合計": 994, "コーナー合計m":1006,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            46021001:{"コーナー数": 6, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":286.0,"直線合計": 1094, "コーナー合計m":1006,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            46021002:{"コーナー数": 6, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":286.0,"直線合計": 1094, "コーナー合計m":1006,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            46023001:{"コーナー数": 7, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":161.0,"直線合計": 1105, "コーナー合計m":1195,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            46023002:{"コーナー数": 7, "最終直線": 236, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":161.0,"直線合計": 1105, "コーナー合計m":1195,"コーナータイプ":1,"コーナーR12":77,"コーナーR34":77,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},


            
            #笠松

            4708001:{"コーナー数": 2, "最終直線": 201, "ゴール前坂": -1.92,  "スタート位置": 1,    "最初直線":251.0,"直線合計": 452, "コーナー合計m":348,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":70,"高低差":1.92,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            4708002:{"コーナー数": 2, "最終直線": 201, "ゴール前坂": -1.92,  "スタート位置": 1,    "最初直線":251.0,"直線合計": 452, "コーナー合計m":348,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":70,"高低差":1.92,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            47014001:{"コーナー数": 4, "最終直線": 201, "ゴール前坂": -1.92,  "スタート位置": 1,    "最初直線":252.0,"直線合計": 704, "コーナー合計m":696,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1.92,"幅":20,   "最初坂":-1.92,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            47014002:{"コーナー数": 4, "最終直線": 201, "ゴール前坂": -1.92,  "スタート位置": 1,    "最初直線":252.0,"直線合計": 704, "コーナー合計m":696,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1.92,"幅":20,   "最初坂":-1.92,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            47016001:{"コーナー数": 5, "最終直線": 201, "ゴール前坂": -1.92,  "スタート位置": 1,    "最初直線":140.0,"直線合計": 730, "コーナー合計m":870,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1.92,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            47016002:{"コーナー数": 5, "最終直線": 201, "ゴール前坂": -1.92,  "スタート位置": 1,    "最初直線":140.0,"直線合計": 730, "コーナー合計m":870,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1.92,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            47018001:{"コーナー数": 6, "最終直線": 201, "ゴール前坂": -1.92,  "スタート位置": 1,    "最初直線":151.0,"直線合計": 756, "コーナー合計m":1044,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1.92,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            47018002:{"コーナー数": 6, "最終直線": 201, "ゴール前坂": -1.92,  "スタート位置": 1,    "最初直線":151.0,"直線合計": 756, "コーナー合計m":1044,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1.92,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            47019001:{"コーナー数": 6, "最終直線": 201, "ゴール前坂": -1.92,  "スタート位置": 1,    "最初直線":251.0,"直線合計": 856, "コーナー合計m":1044,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1.92,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            47019002:{"コーナー数": 6, "最終直線": 201, "ゴール前坂": -1.92,  "スタート位置": 1,    "最初直線":251.0,"直線合計": 856, "コーナー合計m":1044,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1.92,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            47025001:{"コーナー数": 8, "最終直線": 201, "ゴール前坂": -1.92,  "スタート位置": 1,    "最初直線":252.0,"直線合計": 1108, "コーナー合計m":1392,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1.92,"幅":20,   "最初坂":-1.92,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            47025002:{"コーナー数": 8, "最終直線": 201, "ゴール前坂": -1.92,  "スタート位置": 1,    "最初直線":252.0,"直線合計": 1108, "コーナー合計m":1392,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1.92,"幅":20,   "最初坂":-1.92,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            


            #名古屋
            4809001:{"コーナー数": 2, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":370.0,"直線合計": 610, "コーナー合計m":290,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            4809002:{"コーナー数": 2, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":370.0,"直線合計": 610, "コーナー合計m":290,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            4809201:{"コーナー数": 2, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":390.0,"直線合計": 630, "コーナー合計m":290,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            4809202:{"コーナー数": 2, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":390.0,"直線合計": 630, "コーナー合計m":290,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            48014001:{"コーナー数": 4, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":280.0,"直線合計": 820, "コーナー合計m":580,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            48014002:{"コーナー数": 4, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":280.0,"直線合計": 820, "コーナー合計m":580,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            48015001:{"コーナー数": 4, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":380.0,"直線合計": 920, "コーナー合計m":580,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            48015002:{"コーナー数": 4, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":380.0,"直線合計": 920, "コーナー合計m":580,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            48017001:{"コーナー数": 5, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":135.0,"直線合計": 975, "コーナー合計m":725,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            48017002:{"コーナー数": 5, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":135.0,"直線合計": 975, "コーナー合計m":725,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            48020001:{"コーナー数": 6, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":290.0,"直線合計": 1130, "コーナー合計m":870,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            48020002:{"コーナー数": 6, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":290.0,"直線合計": 1130, "コーナー合計m":870,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            48021001:{"コーナー数": 6, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":390.0,"直線合計": 1230, "コーナー合計m":870,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            48021002:{"コーナー数": 6, "最終直線": 240, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":390.0,"直線合計": 1230, "コーナー合計m":870,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":0,"幅":30,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            

            #園田
            5008201:{"コーナー数": 2, "最終直線": 213, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":325.0,"直線合計": 538, "コーナー合計m":282,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":66,"高低差":1.2,"幅":20,   "最初坂":1.23,"向正面坂":0,"最初コーナー坂":1.23,"最終コーナー坂":-1.23},
            5008202:{"コーナー数": 2, "最終直線": 213, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":325.0,"直線合計": 538, "コーナー合計m":282,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":66,"高低差":1.2,"幅":20,   "最初坂":1.23,"向正面坂":0,"最初コーナー坂":1.23,"最終コーナー坂":-1.23},
            
            50012301:{"コーナー数": 4, "最終直線": 213, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":207.0,"直線合計": 666, "コーナー合計m":564,"コーナータイプ":1,"コーナーR12":66,"コーナーR34":66,"高低差":1.2,"幅":20,   "最初坂":0,"向正面坂":1.23,"最初コーナー坂":0,"最終コーナー坂":-1.23},
            50012302:{"コーナー数": 4, "最終直線": 213, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":207.0,"直線合計": 666, "コーナー合計m":564,"コーナータイプ":1,"コーナーR12":66,"コーナーR34":66,"高低差":1.2,"幅":20,   "最初坂":0,"向正面坂":1.23,"最初コーナー坂":0,"最終コーナー坂":-1.23},
            
            50014001:{"コーナー数": 4, "最終直線": 213, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":407.0,"直線合計": 866, "コーナー合計m":564,"コーナータイプ":1,"コーナーR12":66,"コーナーR34":66,"高低差":1.2,"幅":20,   "最初坂":0,"向正面坂":1.23,"最初コーナー坂":0,"最終コーナー坂":-1.23},
            50014002:{"コーナー数": 4, "最終直線": 213, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":407.0,"直線合計": 866, "コーナー合計m":564,"コーナータイプ":1,"コーナーR12":66,"コーナーR34":66,"高低差":1.2,"幅":20,   "最初坂":0,"向正面坂":1.23,"最初コーナー坂":0,"最終コーナー坂":-1.23},
            
            50017001:{"コーナー数": 6, "最終直線": 213, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":155.0,"直線合計": 854, "コーナー合計m":846,"コーナータイプ":1,"コーナーR12":66,"コーナーR34":66,"高低差":1.2,"幅":20,   "最初坂":1.23,"向正面坂":0,"最初コーナー坂":1.23,"最終コーナー坂":-1.23},
            50017002:{"コーナー数": 6, "最終直線": 213, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":155.0,"直線合計": 854, "コーナー合計m":846,"コーナータイプ":1,"コーナーR12":66,"コーナーR34":66,"高低差":1.2,"幅":20,   "最初坂":1.23,"向正面坂":0,"最初コーナー坂":1.23,"最終コーナー坂":-1.23},
            
            50018701:{"コーナー数": 6, "最終直線": 213, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":325.0,"直線合計": 1024, "コーナー合計m":846,"コーナータイプ":1,"コーナーR12":66,"コーナーR34":66,"高低差":1.2,"幅":20,   "最初坂":1.23,"向正面坂":0,"最初コーナー坂":1.23,"最終コーナー坂":-1.23},
            50018702:{"コーナー数": 6, "最終直線": 213, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":325.0,"直線合計": 1024, "コーナー合計m":846,"コーナータイプ":1,"コーナーR12":66,"コーナーR34":66,"高低差":1.2,"幅":20,   "最初坂":1.23,"向正面坂":0,"最初コーナー坂":1.23,"最終コーナー坂":-1.23},
            


            #高知
            5408001:{"コーナー数": 2, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":240.0,"直線合計": 440, "コーナー合計m":360,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":-0.79,"最終コーナー坂":-0.79},
            5408002:{"コーナー数": 2, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":240.0,"直線合計": 440, "コーナー合計m":360,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":-0.79,"最終コーナー坂":-0.79},
            
            54010001:{"コーナー数": 3, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 500, "コーナー合計m":500,"コーナータイプ":4,"コーナーR12":35,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0.6,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":-0.79},
            54010002:{"コーナー数": 3, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 500, "コーナー合計m":500,"コーナータイプ":4,"コーナーR12":35,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0.6,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":-0.79},
            
            54013001:{"コーナー数": 4, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":210.0,"直線合計": 640, "コーナー合計m":660,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0.79,"最終コーナー坂":-0.79},
            54013002:{"コーナー数": 4, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":210.0,"直線合計": 640, "コーナー合計m":660,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0.79,"最終コーナー坂":-0.79},
            
            54014001:{"コーナー数": 4, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":310.0,"直線合計": 740, "コーナー合計m":660,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0.79,"最終コーナー坂":-0.79},
            54014002:{"コーナー数": 4, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":310.0,"直線合計": 740, "コーナー合計m":660,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0.79,"最終コーナー坂":-0.798},
            
            54016001:{"コーナー数": 5, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":100.0,"直線合計": 760, "コーナー合計m":840,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":-0.79,"向正面坂":0,"最初コーナー坂":-0.79,"最終コーナー坂":-0.79},
            54016002:{"コーナー数": 5, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":100.0,"直線合計": 760, "コーナー合計m":840,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":-0.79,"向正面坂":0,"最初コーナー坂":-0.79,"最終コーナー坂":-0.79},

            54018001:{"コーナー数": 6, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":140.0,"直線合計": 800, "コーナー合計m":1000,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":-0.79,"最終コーナー坂":-0.79},
            54018002:{"コーナー数": 6, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":140.0,"直線合計": 800, "コーナー合計m":1000,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":-0.79,"最終コーナー坂":-0.79},
            
            54019001:{"コーナー数": 6, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":240.0,"直線合計": 900, "コーナー合計m":1000,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":-0.79,"最終コーナー坂":-0.79},
            54019002:{"コーナー数": 6, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":240.0,"直線合計": 900, "コーナー合計m":1000,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":-0.79,"最終コーナー坂":-0.79},
            
            54021001:{"コーナー数": 7, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 960, "コーナー合計m":1140,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0.6,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":-0.79},
            54021002:{"コーナー数": 7, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":300.0,"直線合計": 960, "コーナー合計m":1140,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0.6,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":-0.79},
            
            54024001:{"コーナー数": 8, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":210.0,"直線合計": 1080, "コーナー合計m":1320,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0.79,"最終コーナー坂":-0.79},
            54024002:{"コーナー数": 8, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":210.0,"直線合計": 1080, "コーナー合計m":1320,"コーナータイプ":4,"コーナーR12":70,"コーナーR34":80,"高低差":1.58,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0.79,"最終コーナー坂":-0.79},
            

            #姫路
            5108001:{"コーナー数": 2, "最終直線": 230, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":320.0,"直線合計": 550, "コーナー合計m":250,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":80,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            5108002:{"コーナー数": 2, "最終直線": 230, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":320.0,"直線合計": 550, "コーナー合計m":250,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":80,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            51014001:{"コーナー数": 4, "最終直線": 230, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":350.0,"直線合計": 900, "コーナー合計m":500,"コーナータイプ":3,"コーナーR12":85,"コーナーR34":80,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            51014002:{"コーナー数": 4, "最終直線": 230, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":350.0,"直線合計": 900, "コーナー合計m":500,"コーナータイプ":3,"コーナーR12":85,"コーナーR34":80,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
        
            51015001:{"コーナー数": 4, "最終直線": 230, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":450.0,"直線合計": 1000, "コーナー合計m":500,"コーナータイプ":3,"コーナーR12":85,"コーナーR34":80,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            51015002:{"コーナー数": 4, "最終直線": 230, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":450.0,"直線合計": 1000, "コーナー合計m":500,"コーナータイプ":3,"コーナーR12":85,"コーナーR34":80,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            51018001:{"コーナー数": 6, "最終直線": 230, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":120.0,"直線合計": 1050, "コーナー合計m":750,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":80,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            51018002:{"コーナー数": 6, "最終直線": 230, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":120.0,"直線合計": 1050, "コーナー合計m":750,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":80,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            51020001:{"コーナー数": 6, "最終直線": 230, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":320.0,"直線合計": 1250, "コーナー合計m":750,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":80,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            51020002:{"コーナー数": 6, "最終直線": 230, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":320.0,"直線合計": 1250, "コーナー合計m":750,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":80,"高低差":0,"幅":20,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
        
            #佐賀
            5509001:{"コーナー数": 2, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":390.0,"直線合計": 590, "コーナー合計m":310,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            5509002:{"コーナー数": 2, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":390.0,"直線合計": 590, "コーナー合計m":310,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            55013001:{"コーナー数": 4, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":230.0,"直線合計": 680, "コーナー合計m":620,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            55013002:{"コーナー数": 4, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":230.0,"直線合計": 680, "コーナー合計m":620,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            55014001:{"コーナー数": 4, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":330.0,"直線合計": 780, "コーナー合計m":620,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            55014002:{"コーナー数": 4, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":330.0,"直線合計": 780, "コーナー合計m":620,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            55017501:{"コーナー数": 6, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":120.0,"直線合計": 820, "コーナー合計m":930,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            55017501:{"コーナー数": 6, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":120.0,"直線合計": 820, "コーナー合計m":930,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            55018001:{"コーナー数": 6, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":170.0,"直線合計": 870, "コーナー合計m":930,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            55018002:{"コーナー数": 6, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":170.0,"直線合計": 870, "コーナー合計m":930,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            55018601:{"コーナー数": 6, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":230.0,"直線合計": 930, "コーナー合計m":930,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            55018602:{"コーナー数": 6, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":230.0,"直線合計": 930, "コーナー合計m":930,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            
            55020001:{"コーナー数": 6, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":370.0,"直線合計": 1070, "コーナー合計m":930,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            55020002:{"コーナー数": 6, "最終直線": 200, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":370.0,"直線合計": 1070, "コーナー合計m":930,"コーナータイプ":1,"コーナーR12":70,"コーナーR34":70,"高低差":1,"幅":19.2,   "最初坂":0,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            




            #芝
            #札幌
            1112001:{"コーナー数": 2, "最終直線": 266, "ゴール前坂": 0,  "スタート位置": 1,   "最初直線": 410.0,"直線合計": 670, "コーナー合計m":530,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":167,"高低差":0.6,"幅":20.0,      "最初坂":-0.2,"向正面坂":0.3,"最初コーナー坂":-0.2,"最終コーナー坂":-0.1},
            1112002:{"コーナー数": 2, "最終直線": 266, "ゴール前坂": 0,  "スタート位置": 1,   "最初直線": 410.0,"直線合計": 670, "コーナー合計m":530,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":167,"高低差":0.6,"幅":20.0,      "最初坂":-0.2,"向正面坂":0.3,"最初コーナー坂":-0.2,"最終コーナー坂":-0.1},

            1115001:{"コーナー数": 3, "最終直線": 266, "ゴール前坂": 0,  "スタート位置": 1,   "最初直線": 170.0,"直線合計": 670, "コーナー合計m":830,"コーナータイプ":5,"コーナーR12":167,"コーナーR34":167,"高低差":0.6,"幅":20.0,      "最初坂":0,"向正面坂":0.3,"最初コーナー坂":0,"最終コーナー坂":-0.1},
            1115002:{"コーナー数": 3, "最終直線": 266, "ゴール前坂": 0,  "スタート位置": 1,   "最初直線": 170.0,"直線合計": 670, "コーナー合計m":830,"コーナータイプ":5,"コーナーR12":167,"コーナーR34":167,"高低差":0.6,"幅":20.0,      "最初坂":0,"向正面坂":0.3,"最初コーナー坂":0,"最終コーナー坂":-0.1},

            1118001:{"コーナー数": 4, "最終直線": 266, "ゴール前坂": 0,  "スタート位置": 1,   "最初直線": 180.0,"直線合計": 740, "コーナー合計m":1060,"コーナータイプ":5,"コーナーR12":167,"コーナーR34":167,"高低差":0.6,"幅":20.0,      "最初坂":0,"向正面坂":0.3,"最初コーナー坂":-0.3,"最終コーナー坂":-0.1},
            1118002:{"コーナー数": 4, "最終直線": 266, "ゴール前坂": 0,  "スタート位置": 1,   "最初直線": 180.0,"直線合計": 740, "コーナー合計m":1060,"コーナータイプ":5,"コーナーR12":167,"コーナーR34":167,"高低差":0.6,"幅":20.0,      "最初坂":0,"向正面坂":0.3,"最初コーナー坂":-0.3,"最終コーナー坂":-0.1},
        
            1120001:{"コーナー数": 4, "最終直線": 266, "ゴール前坂": 0,  "スタート位置": 1,   "最初直線": 380.0,"直線合計": 940, "コーナー合計m":1060,"コーナータイプ":5,"コーナーR12":167,"コーナーR34":167,"高低差":0.6,"幅":20.0,      "最初坂":0.2,"向正面坂":0.3,"最初コーナー坂":-0.3,"最終コーナー坂":-0.1},
            1120002:{"コーナー数": 4, "最終直線": 266, "ゴール前坂": 0,  "スタート位置": 1,   "最初直線": 380.0,"直線合計": 940, "コーナー合計m":1060,"コーナータイプ":5,"コーナーR12":167,"コーナーR34":167,"高低差":0.6,"幅":20.0,      "最初坂":0.2,"向正面坂":0.3,"最初コーナー坂":-0.3,"最終コーナー坂":-0.1},
        
            1126001:{"コーナー数": 6, "最終直線": 266, "ゴール前坂": 0,  "スタート位置": 1,   "最初直線": 160.0,"直線合計": 1010, "コーナー合計m":1590,"コーナータイプ":5,"コーナーR12":167,"コーナーR34":167,"高低差":0.6,"幅":20.0,      "最初坂":0.3,"向正面坂":0.3,"最初コーナー坂":-0.3,"最終コーナー坂":-0.1},
            1126002:{"コーナー数": 6, "最終直線": 266, "ゴール前坂": 0,  "スタート位置": 1,   "最初直線": 160.0,"直線合計": 1010, "コーナー合計m":1590,"コーナータイプ":5,"コーナーR12":167,"コーナーR34":167,"高低差":0.6,"幅":20.0,      "最初坂":0.3,"向正面坂":0.3,"最初コーナー坂":-0.3,"最終コーナー坂":-0.1},
        

            #函館
            2110001:{"コーナー数": 2, "最終直線": 262, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線":289.1,"直線合計": 552, "コーナー合計m":448,"コーナータイプ":2,"コーナーR12":0,"コーナーR34":167,"高低差":3.4,"幅":29.0,   "最初坂":1.8,"向正面坂":2,"最初コーナー坂":1.4,"最終コーナー坂":-0.4},
            2110002:{"コーナー数": 2, "最終直線": 262, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線":289.1,"直線合計": 552, "コーナー合計m":448,"コーナータイプ":2,"コーナーR12":0,"コーナーR34":167,"高低差":3.4,"幅":29.0,   "最初坂":1.8,"向正面坂":2,"最初コーナー坂":1.4,"最終コーナー坂":-0.4},

            2112001:{"コーナー数": 2, "最終直線": 262, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線":489.10,"直線合計": 752, "コーナー合計m":448,"コーナータイプ":2,"コーナーR12":0,"コーナーR34":167,"高低差":3.4,"幅":29.0,   "最初坂":2.1,"向正面坂":2,"最初コーナー坂":1.4,"最終コーナー坂":-0.4},
            2112002:{"コーナー数": 2, "最終直線": 262, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線":489.10,"直線合計": 752, "コーナー合計m":448,"コーナータイプ":2,"コーナーR12":0,"コーナーR34":167,"高低差":3.4,"幅":29.0,   "最初坂":2.1,"向正面坂":2,"最初コーナー坂":1.4,"最終コーナー坂":-0.4},

            2118001:{"コーナー数": 4, "最終直線": 262, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線":275.80,"直線合計": 904, "コーナー合計m":896,"コーナータイプ":2,"コーナーR12":127,"コーナーR34":167,"高低差":3.4,"幅":29.0,   "最初坂":-0.3,"向正面坂":2,"最初コーナー坂":-2,"最終コーナー坂":-0.4},
            2118002:{"コーナー数": 4, "最終直線": 262, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線":275.80,"直線合計": 904, "コーナー合計m":896,"コーナータイプ":2,"コーナーR12":127,"コーナーR34":167,"高低差":3.4,"幅":29.0,   "最初坂":-0.3,"向正面坂":2,"最初コーナー坂":-2,"最終コーナー坂":-0.4},

            2120001:{"コーナー数": 4, "最終直線": 262, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線":475.8,"直線合計": 1104, "コーナー合計m":896,"コーナータイプ":2,"コーナーR12":127,"コーナーR34":167,"高低差":3.4,"幅":29.0,   "最初坂":-0.7,"向正面坂":2,"最初コーナー坂":-2,"最終コーナー坂":-0.4},
            2120002:{"コーナー数": 4, "最終直線": 262, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線":475.8,"直線合計": 1104, "コーナー合計m":896,"コーナータイプ":2,"コーナーR12":127,"コーナーR34":167,"高低差":3.4,"幅":29.0,   "最初坂":-0.7,"向正面坂":2,"最初コーナー坂":-2,"最終コーナー坂":-0.4},

            2126001:{"コーナー数": 6, "最終直線": 262, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線":262.5,"直線合計": 1256, "コーナー合計m":1344,"コーナータイプ":2,"コーナーR12":127,"コーナーR34":167,"高低差":3.4,"幅":29.0,   "最初坂":1,"向正面坂":2,"最初コーナー坂":1.4,"最終コーナー坂":-0.4},
            2126002:{"コーナー数": 6, "最終直線": 262, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線":262.5,"直線合計": 1256, "コーナー合計m":1344,"コーナータイプ":2,"コーナーR12":127,"コーナーR34":167,"高低差":3.4,"幅":29.0,   "最初坂":1,"向正面坂":2,"最初コーナー坂":1.4,"最終コーナー坂":-0.4},


            #福島
            3110001:{"コーナー数": 2, "最終直線": 292, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 211.7,"直線合計": 502, "コーナー合計m":498,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":167,"高低差":1.8,"幅":25.0,   "最初坂":1.3,"向正面坂":1.7,"最初コーナー坂":0,"最終コーナー坂":-0.8},
            3110002:{"コーナー数": 2, "最終直線": 292, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 211.7,"直線合計": 502, "コーナー合計m":498,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":167,"高低差":1.8,"幅":25.0,   "最初坂":1.3,"向正面坂":1.7,"最初コーナー坂":0,"最終コーナー坂":-0.8},

            3112001:{"コーナー数": 2, "最終直線": 292, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 411.7,"直線合計": 702, "コーナー合計m":498,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":167,"高低差":1.8,"幅":25.0,   "最初坂":1.7,"向正面坂":1.7,"最初コーナー坂":0,"最終コーナー坂":-0.8},
            3112002:{"コーナー数": 2, "最終直線": 292, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 411.7,"直線合計": 702, "コーナー合計m":498,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":167,"高低差":1.8,"幅":25.0,   "最初坂":1.7,"向正面坂":1.7,"最初コーナー坂":0,"最終コーナー坂":-0.8},
        
            3117001:{"コーナー数":4, "最終直線": 292, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 205.3,"直線合計": 825, "コーナー合計m":875,"コーナータイプ":3,"コーナーR12":133,"コーナーR34":167,"高低差":1.8,"幅":25.0,   "最初坂":0.5,"向正面坂":1.7,"最初コーナー坂":-2,"最終コーナー坂":-0.8},
            3117002:{"コーナー数":4, "最終直線": 292, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 205.3,"直線合計": 825, "コーナー合計m":875,"コーナータイプ":3,"コーナーR12":133,"コーナーR34":167,"高低差":1.8,"幅":25.0,   "最初坂":0.5,"向正面坂":1.7,"最初コーナー坂":-2,"最終コーナー坂":-0.8},
            
            3118001:{"コーナー数":4, "最終直線": 292, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 305.3,"直線合計": 925, "コーナー合計m":875,"コーナータイプ":3,"コーナーR12":133,"コーナーR34":167,"高低差":1.8,"幅":25.0,   "最初坂":1,"向正面坂":1.7,"最初コーナー坂":-2,"最終コーナー坂":-0.8},
            3118002:{"コーナー数":4, "最終直線": 292, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 305.3,"直線合計": 925, "コーナー合計m":875,"コーナータイプ":3,"コーナーR12":133,"コーナーR34":167,"高低差":1.8,"幅":25.0,   "最初坂":1,"向正面坂":1.7,"最初コーナー坂":-2,"最終コーナー坂":-0.8},
        
            3120001:{"コーナー数":4, "最終直線": 292, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 505.3,"直線合計": 1125, "コーナー合計m":875,"コーナータイプ":3,"コーナーR12":133,"コーナーR34":167,"高低差":1.8,"幅":25.0,   "最初坂":1,"向正面坂":1.7,"最初コーナー坂":-2,"最終コーナー坂":-0.8},
            3120002:{"コーナー数":4, "最終直線": 292, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 505.3,"直線合計": 1125, "コーナー合計m":875,"コーナータイプ":3,"コーナーR12":133,"コーナーR34":167,"高低差":1.8,"幅":25.0,   "最初坂":1,"向正面坂":1.7,"最初コーナー坂":-2,"最終コーナー坂":-0.8},
        
            3126001:{"コーナー数":6, "最終直線": 292, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 211.7,"直線合計": 1227, "コーナー合計m":1373,"コーナータイプ":3,"コーナーR12":133,"コーナーR34":167,"高低差":1.8,"幅":25.0,   "最初坂":1.3,"向正面坂":1.7,"最初コーナー坂":0,"最終コーナー坂":-0.8},
            3126002:{"コーナー数":6, "最終直線": 292, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 211.7,"直線合計": 1227, "コーナー合計m":1373,"コーナータイプ":3,"コーナーR12":133,"コーナーR34":167,"高低差":1.8,"幅":25.0,   "最初坂":1.3,"向正面坂":1.7,"最初コーナー坂":0,"最終コーナー坂":-0.8},
        

            #新潟
            4110001:{"コーナー数": 0, "最終直線": 1000, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 1000,"直線合計": 1000, "コーナー合計m":1,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":0,"高低差":1,"幅":25.0,   "最初坂":1.5,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},
            4110002:{"コーナー数": 0, "最終直線": 1000, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線": 1000,"直線合計": 1000, "コーナー合計m":1,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":0,"高低差":1,"幅":25.0,   "最初坂":1.5,"向正面坂":0,"最初コーナー坂":0,"最終コーナー坂":0},

            4112001:{"コーナー数": 2, "最終直線": 359, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線": 444.9,"直線合計": 804, "コーナー合計m":396,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":124,"高低差":0.7,"幅":25.0,   "最初坂":-0.2,"向正面坂":0,"最初コーナー坂":-0.3,"最終コーナー坂":0},
            4112002:{"コーナー数": 2, "最終直線": 359, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線": 444.9,"直線合計": 804, "コーナー合計m":396,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":124,"高低差":0.7,"幅":25.0,   "最初坂":-0.2,"向正面坂":0,"最初コーナー坂":-0.3,"最終コーナー坂":0},
                
            4114001:{"コーナー数": 2, "最終直線": 359, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線": 644.9,"直線合計": 1004, "コーナー合計m":396,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":124,"高低差":0.7,"幅":25.0,   "最初坂":-0.2,"向正面坂":0,"最初コーナー坂":-0.3,"最終コーナー坂":0},
            4114002:{"コーナー数": 2, "最終直線": 359, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線": 644.9,"直線合計": 1004, "コーナー合計m":396,"コーナータイプ":1,"コーナーR12":0,"コーナーR34":124,"高低差":0.7,"幅":25.0,   "最初坂":-0.2,"向正面坂":0,"最初コーナー坂":-0.3,"最終コーナー坂":0},
                
            4116001:{"コーナー数": 2, "最終直線": 659, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線": 547.9,"直線合計": 1207, "コーナー合計m":393,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":124,"高低差":2.2,"幅":25.0,   "最初坂":2,"向正面坂":2,"最初コーナー坂":-1.6,"最終コーナー坂":-0.3},
            4116002:{"コーナー数": 2, "最終直線": 659, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線": 547.9,"直線合計": 1207, "コーナー合計m":393,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":124,"高低差":2.2,"幅":25.0,   "最初坂":2,"向正面坂":2,"最初コーナー坂":-1.6,"最終コーナー坂":-0.3},
                
            4118001:{"コーナー数": 2, "最終直線": 659, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線": 747.9,"直線合計": 1407, "コーナー合計m":393,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":124,"高低差":2.2,"幅":25.0,   "最初坂":0.2,"向正面坂":2,"最初コーナー坂":-1.6,"最終コーナー坂":-0.3},
            4118002:{"コーナー数": 2, "最終直線": 659, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線": 747.9,"直線合計": 1407, "コーナー合計m":393,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":124,"高低差":2.2,"幅":25.0,   "最初坂":0.2,"向正面坂":2,"最初コーナー坂":-1.6,"最終コーナー坂":-0.3},
            
            4120001:{"コーナー数": 4, "最終直線": 359, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線": 436.4,"直線合計": 1208, "コーナー合計m":792,"コーナータイプ":1,"コーナーR12":124,"コーナーR34":124,"高低差":0.7,"幅":25.0,   "最初坂":0,"向正面坂":0.3,"最初コーナー坂":0.3,"最終コーナー坂":-0.3},
            
            4120002:{"コーナー数": 2, "最終直線": 659, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線": 947.9,"直線合計": 1607, "コーナー合計m":393,"コーナータイプ":3,"コーナーR12":124,"コーナーR34":124,"高低差":2.2,"幅":25.0,   "最初坂":0.2,"向正面坂":2,"最初コーナー坂":-1.6,"最終コーナー坂":-0.3},
        
            4122001:{"コーナー数": 4, "最終直線": 359, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線": 636.4,"直線合計": 1408, "コーナー合計m":792,"コーナータイプ":1,"コーナーR12":124,"コーナーR34":124,"高低差":0.7,"幅":25.0,   "最初坂":-0.2,"向正面坂":0.3,"最初コーナー坂":0.1,"最終コーナー坂":-0.3},
            4122002:{"コーナー数": 4, "最終直線": 359, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線": 636.4,"直線合計": 1408, "コーナー合計m":792,"コーナータイプ":1,"コーナーR12":124,"コーナーR34":124,"高低差":0.7,"幅":25.0,   "最初坂":-0.2,"向正面坂":0.3,"最初コーナー坂":0.1,"最終コーナー坂":-0.3},
            
            4124001:{"コーナー数": 4, "最終直線": 359, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線": 836.4,"直線合計": 1608, "コーナー合計m":792,"コーナータイプ":1,"コーナーR12":124,"コーナーR34":124,"高低差":0.7,"幅":25.0,   "最初坂":-1.7,"向正面坂":0.3,"最初コーナー坂":0.1,"最終コーナー坂":-0.3},
            4124002:{"コーナー数": 4, "最終直線": 359, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線": 836.4,"直線合計": 1608, "コーナー合計m":792,"コーナータイプ":1,"コーナーR12":124,"コーナーR34":124,"高低差":0.7,"幅":25.0,   "最初坂":-1.7,"向正面坂":0.3,"最初コーナー坂":0.1,"最終コーナー坂":-0.3},
            

            #東京
            5114001:{"コーナー数": 2, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":342.7,"直線合計": 869, "コーナー合計m":531,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":1.7,"向正面坂":1.7,"最初コーナー坂":-1.7,"最終コーナー坂":0.3},
            5114002:{"コーナー数": 2, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":342.7,"直線合計": 869, "コーナー合計m":531,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":1.7,"向正面坂":1.7,"最初コーナー坂":-1.7,"最終コーナー坂":0.3},
            
            5116001:{"コーナー数": 2, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":542.7,"直線合計":1069, "コーナー合計m":531,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":1.7,"向正面坂":1.7,"最初コーナー坂":-1.7,"最終コーナー坂":0.3},
            5116002:{"コーナー数": 2, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":542.7,"直線合計":1069, "コーナー合計m":531,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":1.7,"向正面坂":1.7,"最初コーナー坂":-1.7,"最終コーナー坂":0.3},
            
            5118001:{"コーナー数": 3, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":156.6,"直線合計":1226, "コーナー合計m":574,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":-0.7,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            5118002:{"コーナー数": 3, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":156.6,"直線合計":1226, "コーナー合計m":574,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":-0.7,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            
            5120001:{"コーナー数": 3, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":126.2,"直線合計":1195, "コーナー合計m":805,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":-0.7,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            5120002:{"コーナー数": 3, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":126.2,"直線合計":1195, "コーナー合計m":805,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":-0.7,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            
            5123001:{"コーナー数": 4, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":249.5,"直線合計":1321, "コーナー合計m":979,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":0,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            5123002:{"コーナー数": 4, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":249.5,"直線合計":1321, "コーナー合計m":979,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":0,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            
            5124001:{"コーナー数": 4, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":349.5,"直線合計":1421, "コーナー合計m":979,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":0,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            5124002:{"コーナー数": 4, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":349.5,"直線合計":1421, "コーナー合計m":979,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":0,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            
            5125001:{"コーナー数": 4, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":449.5,"直線合計":1521, "コーナー合計m":979,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":2,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            5125002:{"コーナー数": 4, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":449.5,"直線合計":1521, "コーナー合計m":979,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":2,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            
            5134001:{"コーナー数": 6, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":259.8,"直線合計":1890, "コーナー合計m":1510,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":1.5,"向正面坂":1.7,"最初コーナー坂":-1.7,"最終コーナー坂":0.3},
            5134002:{"コーナー数": 6, "最終直線": 526, "ゴール前坂": 1,  "スタート位置": 1,    "最初直線":259.8,"直線合計":1890, "コーナー合計m":1510,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":1.5,"向正面坂":1.7,"最初コーナー坂":-1.7,"最終コーナー坂":0.3},
            


            #中山
            6112001:{"コーナー数": 2, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":275.1,"直線合計": 585, "コーナー合計m":615,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":190,"高低差":5.3,"幅":24.0,   "最初坂":-3.7,"向正面坂":-2,"最初コーナー坂":-1,"最終コーナー坂":-0.4},
            6112002:{"コーナー数": 2, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":275.1,"直線合計": 585, "コーナー合計m":615,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":190,"高低差":5.3,"幅":24.0,   "最初坂":-3.7,"向正面坂":-2,"最初コーナー坂":-1,"最終コーナー坂":-0.4},
            
            6116001:{"コーナー数": 3, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":239.8,"直線合計": 825, "コーナー合計m":775,"コーナータイプ":5,"コーナーR12":190,"コーナーR34":190,"高低差":5.3,"幅":24.0,   "最初坂":2,"向正面坂":-2,"最初コーナー坂":2,"最終コーナー坂":-0.4},
            6116002:{"コーナー数": 3, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":239.8,"直線合計": 825, "コーナー合計m":775,"コーナータイプ":5,"コーナーR12":190,"コーナーR34":190,"高低差":5.3,"幅":24.0,   "最初坂":2,"向正面坂":-2,"最初コーナー坂":2,"最終コーナー坂":-0.4},
            
            6118001:{"コーナー数": 4, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":204.9,"直線合計": 885, "コーナー合計m":915,"コーナータイプ":2,"コーナーR12":154,"コーナーR34":154,"高低差":5.3,"幅":20.0,   "最初坂":2,"向正面坂":-2,"最初コーナー坂":2,"最終コーナー坂":-0.4},
            6118002:{"コーナー数": 4, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":204.9,"直線合計": 885, "コーナー合計m":915,"コーナータイプ":2,"コーナーR12":154,"コーナーR34":154,"高低差":5.3,"幅":20.0,   "最初坂":2,"向正面坂":-2,"最初コーナー坂":2,"最終コーナー坂":-0.4},
            
            6120001:{"コーナー数": 4, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":404.9,"直線合計": 1085, "コーナー合計m":915,"コーナータイプ":2,"コーナーR12":154,"コーナーR34":154,"高低差":5.3,"幅":20.0,   "最初坂":2,"向正面坂":-2,"最初コーナー坂":2,"最終コーナー坂":-0.4},
            6120002:{"コーナー数": 4, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":404.9,"直線合計": 1085, "コーナー合計m":915,"コーナータイプ":2,"コーナーR12":154,"コーナーR34":154,"高低差":5.3,"幅":20.0,   "最初坂":2,"向正面坂":-2,"最初コーナー坂":2,"最終コーナー坂":-0.4},
            
            6122001:{"コーナー数": 5, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":432.3,"直線合計": 1017, "コーナー合計m":1183,"コーナータイプ":5,"コーナーR12":190,"コーナーR34":190,"高低差":5.3,"幅":24.0,   "最初坂":2,"向正面坂":-2,"最初コーナー坂":2,"最終コーナー坂":-0.4},
            6122002:{"コーナー数": 5, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":432.3,"直線合計": 1017, "コーナー合計m":1183,"コーナータイプ":5,"コーナーR12":190,"コーナーR34":190,"高低差":5.3,"幅":24.0,   "最初坂":2,"向正面坂":-2,"最初コーナー坂":2,"最終コーナー坂":-0.4},
            
            6125001:{"コーナー数": 6, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":192.0,"直線合計": 1277, "コーナー合計m":1223,"コーナータイプ":2,"コーナーR12":154,"コーナーR34":154,"高低差":5.3,"幅":20.0,   "最初坂":0,"向正面坂":-2,"最初コーナー坂":0,"最終コーナー坂":-0.4},
            6125002:{"コーナー数": 6, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":192.0,"直線合計": 1277, "コーナー合計m":1223,"コーナータイプ":2,"コーナーR12":154,"コーナーR34":154,"高低差":5.3,"幅":20.0,   "最初坂":0,"向正面坂":-2,"最初コーナー坂":0,"最終コーナー坂":-0.4},
            
            6136001:{"コーナー数": 8, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":337.7,"直線合計": 1776, "コーナー合計m":1824,"コーナータイプ":2,"コーナーR12":154,"コーナーR34":154,"高低差":5.3,"幅":20.0,   "最初坂":2,"向正面坂":-2,"最初コーナー坂":2,"最終コーナー坂":-0.4},
            6136002:{"コーナー数": 8, "最終直線": 310, "ゴール前坂": 3,  "スタート位置": 1,    "最初直線":337.7,"直線合計": 1776, "コーナー合計m":1824,"コーナータイプ":2,"コーナーR12":154,"コーナーR34":154,"高低差":5.3,"幅":20.0,   "最初坂":2,"向正面坂":-2,"最初コーナー坂":2,"最終コーナー坂":-0.4},


            #中京
            7112001:{"コーナー数": 2, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":315.5,"直線合計": 728, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":-0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
            7112002:{"コーナー数": 2, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":315.5,"直線合計": 728, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":-0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},

            7113001:{"コーナー数": 2, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":415.5,"直線合計": 828, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
            7113002:{"コーナー数": 2, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":415.5,"直線合計": 828, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},

            7114001:{"コーナー数": 2, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":515.5,"直線合計": 928, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
            7114002:{"コーナー数": 2, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":515.5,"直線合計": 928, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},

            7116001:{"コーナー数": 3, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":199.0,"直線合計": 1028, "コーナー合計m":572,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":0.5,"最終コーナー坂":-1},
            7116002:{"コーナー数": 3, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":199.0,"直線合計": 1028, "コーナー合計m":572,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":0.5,"最終コーナー坂":-1},

            7120001:{"コーナー数": 4, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":314.1,"直線合計": 1142, "コーナー合計m":858,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":2,"向正面坂":-0.6,"最初コーナー坂":0.2,"最終コーナー坂":-1},
            7120002:{"コーナー数": 4, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":314.1,"直線合計": 1142, "コーナー合計m":858,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":2,"向正面坂":-0.6,"最初コーナー坂":0.2,"最終コーナー坂":-1},

            7122001:{"コーナー数": 4, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":514.1,"直線合計": 1342, "コーナー合計m":858,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":2,"向正面坂":-0.6,"最初コーナー坂":0.2,"最終コーナー坂":-1},
            7122002:{"コーナー数": 4, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":514.1,"直線合計": 1342, "コーナー合計m":858,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":2,"向正面坂":-0.6,"最初コーナー坂":0.2,"最終コーナー坂":-1},

            7130001:{"コーナー数": 6, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":415.5,"直線合計": 828, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
            7130002:{"コーナー数": 6, "最終直線": 412.5, "ゴール前坂": 1.6,  "スタート位置": 1,    "最初直線":415.5,"直線合計": 828, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},


            #京都
            8112001:{"コーナー数": 2, "最終直線": 328, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":316.2,"直線合計": 644, "コーナー合計m":556,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":150,"高低差":3.1,"幅":28.0,   "最初坂":3,"向正面坂":3,"最初コーナー坂":-3,"最終コーナー坂":-0.3},
            8112002:{"コーナー数": 2, "最終直線": 328, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":316.2,"直線合計": 644, "コーナー合計m":556,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":150,"高低差":3.1,"幅":28.0,   "最初坂":3,"向正面坂":3,"最初コーナー坂":-3,"最終コーナー坂":-0.3},
            
            8114001:{"コーナー数": 2, "最終直線": 328, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":516.2,"直線合計": 844, "コーナー合計m":556,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":150,"高低差":3.1,"幅":28.0,   "最初坂":0,"向正面坂":3,"最初コーナー坂":-3,"最終コーナー坂":-0.3},
            
            8114002:{"コーナー数": 2, "最終直線": 404.0, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":511.7,"直線合計": 916, "コーナー合計m":484,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":190,"高低差":4.3,"幅":28.0,   "最初坂":0,"向正面坂":4,"最初コーナー坂":-3.7,"最終コーナー坂":-0.4},
            
            8116001:{"コーナー数": 2, "最終直線": 328, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":716.2,"直線合計": 1044, "コーナー合計m":556,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":150,"高低差":3.1,"幅":28.0,   "最初坂":0,"向正面坂":3,"最初コーナー坂":-3,"最終コーナー坂":-0.3},
            
            8116002:{"コーナー数": 2, "最終直線": 404.0, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":711.7,"直線合計":1116, "コーナー合計m":484,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":190,"高低差":4.3,"幅":28.0,   "最初坂":0,"向正面坂":4,"最初コーナー坂":-3.7,"最終コーナー坂":-0.4},

            8118001:{"コーナー数": 2, "最終直線": 404.0, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":911.7,"直線合計":1316, "コーナー合計m":484,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":190,"高低差":4.3,"幅":28.0,   "最初坂":0,"向正面坂":4,"最初コーナー坂":-3.7,"最終コーナー坂":-0.4},
            8118002:{"コーナー数": 2, "最終直線": 404.0, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":911.7,"直線合計":1316, "コーナー合計m":484,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":190,"高低差":4.3,"幅":28.0,   "最初坂":0,"向正面坂":4,"最初コーナー坂":-3.7,"最終コーナー坂":-0.4},

            8120001:{"コーナー数": 4, "最終直線": 328, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":308.7,"直線合計": 1067, "コーナー合計m":933,"コーナータイプ":4,"コーナーR12":130,"コーナーR34":150,"高低差":3.1,"幅":28.0,   "最初坂":0,"向正面坂":3,"最初コーナー坂":-3,"最終コーナー坂":-0.3},
            8120002:{"コーナー数": 4, "最終直線": 328, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":308.7,"直線合計": 1067, "コーナー合計m":933,"コーナータイプ":4,"コーナーR12":130,"コーナーR34":150,"高低差":3.1,"幅":28.0,   "最初坂":0,"向正面坂":3,"最初コーナー坂":-3,"最終コーナー坂":-0.3},

            8122001:{"コーナー数": 4, "最終直線": 404.0, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":397.3,"直線合計":1339, "コーナー合計m":861,"コーナータイプ":5,"コーナーR12":130,"コーナーR34":190,"高低差":4.3,"幅":28.0,   "最初坂":0.2,"向正面坂":4,"最初コーナー坂":0,"最終コーナー坂":-0.4},
            8122002:{"コーナー数": 4, "最終直線": 404.0, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":397.3,"直線合計":1339, "コーナー合計m":861,"コーナータイプ":5,"コーナーR12":130,"コーナーR34":190,"高低差":4.3,"幅":28.0,   "最初坂":0.2,"向正面坂":4,"最初コーナー坂":0,"最終コーナー坂":-0.4},

            8124001:{"コーナー数": 4, "最終直線": 404.0, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":597.3,"直線合計":1539, "コーナー合計m":861,"コーナータイプ":5,"コーナーR12":130,"コーナーR34":190,"高低差":4.3,"幅":28.0,   "最初坂":0.2,"向正面坂":4,"最初コーナー坂":0,"最終コーナー坂":-0.4},
            8124002:{"コーナー数": 4, "最終直線": 404.0, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":597.3,"直線合計":1539, "コーナー合計m":861,"コーナータイプ":5,"コーナーR12":130,"コーナーR34":190,"高低差":4.3,"幅":28.0,   "最初坂":0.2,"向正面坂":4,"最初コーナー坂":0,"最終コーナー坂":-0.4},

            8130001:{"コーナー数": 6, "最終直線": 404.0, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":217.4,"直線合計":1655, "コーナー合計m":1345,"コーナータイプ":5,"コーナーR12":130,"コーナーR34":190,"高低差":4.3,"幅":28.0,   "最初坂":3,"向正面坂":4,"最初コーナー坂":-3,"最終コーナー坂":-0.4},
            8130002:{"コーナー数": 6, "最終直線": 404.0, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":217.4,"直線合計":1655, "コーナー合計m":1345,"コーナータイプ":5,"コーナーR12":130,"コーナーR34":190,"高低差":4.3,"幅":28.0,   "最初坂":3,"向正面坂":4,"最初コーナー坂":-3,"最終コーナー坂":-0.4},

            8132001:{"コーナー数": 6, "最終直線": 404.0, "ゴール前坂": 0.2,  "スタート位置": 1,    "最初直線":417.4,"直線合計":1855, "コーナー合計m":1345,"コーナータイプ":5,"コーナーR12":130,"コーナーR34":190,"高低差":4.3,"幅":28.0,   "最初坂":4,"向正面坂":4,"最初コーナー坂":-3,"最終コーナー坂":-0.4},
            

            #阪神
            9112001:{"コーナー数": 2, "最終直線": 357, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":258.2,"直線合計": 615, "コーナー合計m":585,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":170,"高低差":1.8,"幅":24.0,   "最初坂":-0.2,"向正面坂":-0.4,"最初コーナー坂":-0.7,"最終コーナー坂":-0.5},
            9112002:{"コーナー数": 2, "最終直線": 357, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":258.2,"直線合計": 615, "コーナー合計m":585,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":170,"高低差":1.8,"幅":24.0,   "最初坂":-0.2,"向正面坂":-0.4,"最初コーナー坂":-0.7,"最終コーナー坂":-0.5},
                        
            9114001:{"コーナー数": 2, "最終直線": 357, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":458.2,"直線合計": 815, "コーナー合計m":585,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":170,"高低差":1.8,"幅":24.0,   "最初坂":0,"向正面坂":-0.4,"最初コーナー坂":-0.7,"最終コーナー坂":-0.5},
            9114002:{"コーナー数": 2, "最終直線": 357, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":458.2,"直線合計": 815, "コーナー合計m":585,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":170,"高低差":1.8,"幅":24.0,   "最初坂":0,"向正面坂":-0.4,"最初コーナー坂":-0.7,"最終コーナー坂":-0.5},
                
            9116001:{"コーナー数": 2, "最終直線": 474, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":444.4,"直線合計": 918, "コーナー合計m":682,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":192,"高低差":2.3,"幅":24.0,   "最初坂":-0.2,"向正面坂":0.4,"最初コーナー坂":-0.2,"最終コーナー坂":-2.2},
            9116002:{"コーナー数": 2, "最終直線": 474, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":444.4,"直線合計": 918, "コーナー合計m":682,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":192,"高低差":2.3,"幅":24.0,   "最初坂":-0.2,"向正面坂":0.4,"最初コーナー坂":-0.2,"最終コーナー坂":-2.2},

            9118001:{"コーナー数": 2, "最終直線": 474, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":644.4,"直線合計": 1118, "コーナー合計m":682,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":192,"高低差":2.3,"幅":24.0,   "最初坂":0,"向正面坂":0.4,"最初コーナー坂":-0.2,"最終コーナー坂":-2.2},
            9118002:{"コーナー数": 2, "最終直線": 474, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":644.4,"直線合計": 1118, "コーナー合計m":682,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":192,"高低差":2.3,"幅":24.0,   "最初坂":0,"向正面坂":0.4,"最初コーナー坂":-0.2,"最終コーナー坂":-2.2},

            9120001:{"コーナー数": 4, "最終直線": 357, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":330.5,"直線合計": 1067, "コーナー合計m":933,"コーナータイプ":5,"コーナーR12":133,"コーナーR34":170,"高低差":1.8,"幅":24.0,   "最初坂":1.9,"向正面坂":-0.4,"最初コーナー坂":0,"最終コーナー坂":-1.8},
            9120002:{"コーナー数": 4, "最終直線": 357, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":330.5,"直線合計": 1067, "コーナー合計m":933,"コーナータイプ":5,"コーナーR12":133,"コーナーR34":170,"高低差":1.8,"幅":24.0,   "最初坂":1.9,"向正面坂":-0.4,"最初コーナー坂":0,"最終コーナー坂":-1.8},
                
            9122001:{"コーナー数": 4, "最終直線": 357, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":530.5,"直線合計": 1267, "コーナー合計m":933,"コーナータイプ":5,"コーナーR12":133,"コーナーR34":170,"高低差":1.8,"幅":24.0,   "最初坂":-1.9,"向正面坂":-0.4,"最初コーナー坂":0,"最終コーナー坂":-1.8},
            9122002:{"コーナー数": 4, "最終直線": 357, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":530.5,"直線合計": 1267, "コーナー合計m":933,"コーナータイプ":5,"コーナーR12":133,"コーナーR34":170,"高低差":1.8,"幅":24.0,   "最初坂":-1.9,"向正面坂":-0.4,"最初コーナー坂":0,"最終コーナー坂":-1.8},
            
            9124001:{"コーナー数": 4, "最終直線": 474, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":309.0,"直線合計": 1370, "コーナー合計m":1030,"コーナータイプ":5,"コーナーR12":133,"コーナーR34":192,"高低差":2.3,"幅":24.0,   "最初坂":1.9,"向正面坂":0.4,"最初コーナー坂":0,"最終コーナー坂":-2.2},
            9124002:{"コーナー数": 4, "最終直線": 474, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":309.0,"直線合計": 1370, "コーナー合計m":1030,"コーナータイプ":5,"コーナーR12":133,"コーナーR34":192,"高低差":2.3,"幅":24.0,   "最初坂":1.9,"向正面坂":0.4,"最初コーナー坂":0,"最終コーナー坂":-2.2},

            9130001:{"コーナー数": 6, "最終直線": 357, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":369.2,"直線合計": 1482, "コーナー合計m":1518,"コーナータイプ":5,"コーナーR12":133,"コーナーR34":170,"高低差":1.8,"幅":24.0,   "最初坂":0,"向正面坂":-0.4,"最初コーナー坂":-0.7,"最終コーナー坂":-1.8},
            9130002:{"コーナー数": 6, "最終直線": 357, "ゴール前坂": 1.9,  "スタート位置": 1,    "最初直線":369.2,"直線合計": 1482, "コーナー合計m":1518,"コーナータイプ":5,"コーナーR12":133,"コーナーR34":170,"高低差":1.8,"幅":24.0,   "最初坂":0,"向正面坂":-0.4,"最初コーナー坂":-0.7,"最終コーナー坂":-1.8},
        
            
            #小倉
            10112001:{"コーナー数": 2, "最終直線": 293, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":	479.0,"直線合計":772, "コーナー合計m":428,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":136,"高低差":3.0,"幅":30.0,   "最初坂":-1,"向正面坂":-0.7,"最初コーナー坂":-0.9,"最終コーナー坂":-0.3},
            10112002:{"コーナー数": 2, "最終直線": 293, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":	479.0,"直線合計":772, "コーナー合計m":428,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":136,"高低差":3.0,"幅":30.0,   "最初坂":-1,"向正面坂":-0.7,"最初コーナー坂":-0.9,"最終コーナー坂":-0.3},

            10117001:{"コーナー数": 4, "最終直線": 293, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":	172.0,"直線合計":844, "コーナー合計m":856,"コーナータイプ":3,"コーナーR12":136,"コーナーR34":136,"高低差":3.0,"幅":30.0,   "最初坂":0,"向正面坂":-0.7,"最初コーナー坂":3,"最終コーナー坂":-0.3},
            10117002:{"コーナー数": 4, "最終直線": 293, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":	172.0,"直線合計":844, "コーナー合計m":856,"コーナータイプ":3,"コーナーR12":136,"コーナーR34":136,"高低差":3.0,"幅":30.0,   "最初坂":0,"向正面坂":-0.7,"最初コーナー坂":3,"最終コーナー坂":-0.3},

            10118001:{"コーナー数": 4, "最終直線": 293, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":	272.0,"直線合計":944, "コーナー合計m":856,"コーナータイプ":3,"コーナーR12":136,"コーナーR34":136,"高低差":3.0,"幅":30.0,   "最初坂":0,"向正面坂":-0.7,"最初コーナー坂":3,"最終コーナー坂":-0.3},
            10118002:{"コーナー数": 4, "最終直線": 293, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":	272.0,"直線合計":944, "コーナー合計m":856,"コーナータイプ":3,"コーナーR12":136,"コーナーR34":136,"高低差":3.0,"幅":30.0,   "最初坂":0,"向正面坂":-0.7,"最初コーナー坂":3,"最終コーナー坂":-0.3},

            10120001:{"コーナー数": 4, "最終直線": 293, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":	472.0,"直線合計":1144, "コーナー合計m":856,"コーナータイプ":3,"コーナーR12":136,"コーナーR34":136,"高低差":3.0,"幅":30.0,   "最初坂":0,"向正面坂":-0.7,"最初コーナー坂":3,"最終コーナー坂":-0.3},
            10120002:{"コーナー数": 4, "最終直線": 293, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":	472.0,"直線合計":1144, "コーナー合計m":856,"コーナータイプ":3,"コーナーR12":136,"コーナーR34":136,"高低差":3.0,"幅":30.0,   "最初坂":0,"向正面坂":-0.7,"最初コーナー坂":3,"最終コーナー坂":-0.3},

            10126001:{"コーナー数": 6, "最終直線": 293, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":	240.0,"直線合計":1326, "コーナー合計m":1274,"コーナータイプ":3,"コーナーR12":136,"コーナーR34":136,"高低差":3.0,"幅":30.0,   "最初坂":0,"向正面坂":-0.7,"最初コーナー坂":-0.9,"最終コーナー坂":-0.3},
            10126002:{"コーナー数": 6, "最終直線": 293, "ゴール前坂": 0,  "スタート位置": 1,    "最初直線":	240.0,"直線合計":1326, "コーナー合計m":1274,"コーナータイプ":3,"コーナーR12":136,"コーナーR34":136,"高低差":3.0,"幅":30.0,   "最初坂":0,"向正面坂":-0.7,"最初コーナー坂":-0.9,"最終コーナー坂":-0.3}
        }



        # データフレームに変換情報を適用する関数
        def convert_course(row):
            place_code = row["course_type"]  # 競馬場の数値コード
            if place_code in conversion_map_course_type:
                # 直線、カーブ、ゴール前の情報を取得
                course_info = conversion_map_course_type[place_code]
                # 列名を変更
                return pd.Series({
                    "goal_range": course_info["最終直線"], 

                    "curve": course_info["コーナータイプ"], 
                    "goal_slope": course_info["ゴール前坂"],
                    "curve_amount": course_info["コーナー数"], 

                    "start_point": course_info["スタート位置"], 
                    "start_range": course_info["最初直線"], 
                    "straight_total": course_info["直線合計"], 
                    "curve_total": course_info["コーナー合計m"],
                    "curve_R12": course_info["コーナーR12"],
                    "curve_R34": course_info["コーナーR34"],
                    "height_diff": course_info["高低差"],
                    "width": course_info["幅"],
                    "start_slope": course_info["最初坂"],
                    "flont_slope": course_info["向正面坂"],
                    "first_curve_slope": course_info["最初コーナー坂"],
                    "last_curve_slope": course_info["最終コーナー坂"]
                })
            else:
                return pd.Series({"goal_range": None, "curve": None, "goal_slope": None,"curve_amount":None,"start_point":None,"start_range":None,"straight_total":None,"curve_total":None,"curve_R12":None,"curve_R34":None,"height_diff":None,"width":None,"start_slope":None,"flont_slope":None,"first_curve_slope":None,"last_curve_slope":None})
        
        # 競馬場カテゴリに基づく変換を追加
        df[['goal_range', 'curve', 'goal_slope',"curve_amount","start_point","start_range","straight_total","curve_total","curve_R12","curve_R34","height_diff","width","start_slope","flont_slope","first_curve_slope","last_curve_slope"]] = df.apply(convert_course, axis=1)
        
        df["goal_range_100"] = df["goal_range"].astype(float)/100

        # # `course_type` に基づく情報を取得する関数
        # def convert_course(place_code):
        #     if place_code in conversion_map_course_type:
        #         # コース情報を取得
        #         course_info = conversion_map_course_type[place_code]
        #     else:
        #         # すべてのカラムを `None` にする辞書を作成
        #         keys = ["最終直線", "コーナー数", "コーナータイプ", "ゴール前坂", 
        #                 "スタート位置", "最初直線", "直線合計", "コーナー合計m", 
        #                 "コーナーR12", "コーナーR34", "高低差", "幅", "最初坂", 
        #                 "向正面坂", "最初コーナー坂", "最終コーナー坂"]
        #         course_info = dict.fromkeys(keys, None)
            
        #     # DataFrame 用に列名を統一
        #     return pd.Series({
        #         "goal_range": course_info["最終直線"], 
        #         "curve_amount": course_info["コーナー数"], 
        #         "curve": course_info["コーナータイプ"], 
        #         "goal_slope": course_info["ゴール前坂"],

        #         "start_point": course_info["スタート位置"], 
        #         "start_range": course_info["最初直線"], 
        #         "straight_total": course_info["直線合計"], 
        #         "curve_total": course_info["コーナー合計m"],
        #         "curve_R12": course_info["コーナーR12"],
        #         "curve_R34": course_info["コーナーR34"],
        #         "height_diff": course_info["高低差"],
        #         "width": course_info["幅"],
        #         "start_slope": course_info["最初坂"],
        #         "flont_slope": course_info["向正面坂"],
        #         "first_curve_slope": course_info["最初コーナー坂"],
        #         "last_curve_slope": course_info["最終コーナー坂"]
        #     })

        # # `map(pd.Series)` を使って DataFrame にマージ（高速化）
        # df = df.join(df["course_type"].map(convert_course))

        # # `goal_range` の値を100で割る（`NaN` に対応）
        # df["goal_range_100"] = pd.to_numeric(df["goal_range"], errors="coerce") / 100






        df["goal_range"] = df["goal_range"].astype(str)
        
        
        
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
        
        
        
        # df[['goal_range', 'curve', 'goal_slope']] = df[['goal_range', 'curve', 'goal_slope']].fillna(-1).astype(int)
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
        
                
        # df['goal_range'] = df['goal_range'].fillna(-1).astype(int)
        # df['curve'] = df['curve'].fillna(-1).astype(int)
        # df['goal_slope'] = df['goal_slope'].fillna(-1).astype(int)

        df.drop(columns=["date"], inplace=True)  # inplace=Trueを使うと元のdfが更新される

        df = df.replace('None', pd.NA)


        columns_to_convert_x = [
            "goal_range", "curve", "goal_slope", "curve_amount", "start_point",
            "start_range", "straight_total", "curve_total", "curve_R12", "curve_R34",
            "height_diff", "width", "start_slope", "flont_slope",
            "first_curve_slope", "last_curve_slope"
        ]

        df[columns_to_convert_x] = df[columns_to_convert_x].fillna(-10000).astype(float)



            
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
            .merge(self.race_info_before[["race_id", "race_type", "place", "season_level","race_class"]], on="race_id")
        )
        # self.race_info = self.race_info_before.copy()
        race_i = self.race_info_before

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
        df['age_season'] = (df['mean_age_kirisute'].astype(str) + df["season_level"].astype(str)).astype(int)
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
        df['race_grade_scaled'] = df['race_grade'] / 10 - 5
        df_agg = df.groupby("race_id", as_index=False).agg({"race_grade": "first",'age_season':"first", "race_grade_scaled": "first"})

        # self.race_info[['age_season', 'race_grade', 'race_grade_scaled']] = df[['age_season', 'race_grade', 'race_grade_scaled']]
        self.race_info = (
            race_i
            .merge(df_agg[["race_id",'race_grade', 'age_season','race_grade_scaled']], on="race_id",how="left")
        )
        

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

        
        
        # merged_df = merged_df.astype({col: 'float32' for col in merged_df.select_dtypes('float64').columns})
        # merged_df = merged_df.astype({col: 'int32' for col in merged_df.select_dtypes('int64').columns})        
        
        # self.agg_horse_n_races_relative_df = merged_df




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
            .merge(self.race_info[["race_id", "place","race_grade","around","weather","ground_state","course_len","race_type","course_type"]], on="race_id")
        )
        df["place"] = df["place"].astype(int)
        df["race_grade"] = df["race_grade"].astype(int)
        df["ground_state"] = df["ground_state"].astype(int)
        df["around"] = df["around"].fillna(3).astype(int)
        df["weather"] = df["weather"].astype(int)   
    
        

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


        baselog_old = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around","course_type"]], on="race_id"
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
            .merge(self.all_results[["race_id", "horse_id", "nobori","time","wakuban", "umaban","rank","sex"]], on=["race_id", "horse_id"])
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
        df_old["distance_place_type_umaban_race_grade_around_weather_ground_state"] = (df_old["course_type"].astype(str) + df_old["umaban"].astype(str) + df_old["race_grade"].astype(str)+ df_old["ground_state"].astype(str)+ df_old["weather"].astype(str)).astype(int)        
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
                        "distance_place_type_race_grade_straight_ground_state", "distance_place_type_ground_state_weather", 
                        "course_type_ground_umaban","course_type_ground_wakuban",
                    


                         
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
                # "distance_place_type_race_grade_straight_ground_state","distance_place_type_race_grade_around_weather_ground_state",
                # "distance_place_type_umaban_race_grade_around_weather_ground_state",
                # "distance_place_type_ground_state_weather",
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
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "nobori","time","rank","umaban"]], on=["race_id", "horse_id"])
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
        df["distance_place_type_ground_state"] = (df["course_len"].astype(str) + df["place"].astype(str) + df["race_type"].astype(str) + df["ground_state"].astype(str)).astype(int)   
        
        
        baselog_old = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place"]], on="race_id"
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
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "umaban","nobori","time"]], on=["race_id", "horse_id"])
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
        self,date_condition_a: int, n_races: list[int] = [1,  3, 5,8],
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
        #当該レース
        merged_df = self.population.copy()        
        df = (
            merged_df
            .merge(self.race_info[["race_id", "place","weather","ground_state","course_len","race_type","race_date_day_count","course_type"]], on="race_id")
        )
        df["place"] = df["place"].astype(int)

        #当該レース近辺のレース、馬場状態
        df_old2 = (
            self.old_results_condition[["race_id", "horse_id","time","rank","rank_per_horse"]]
            .merge(self.old_race_info_condition[["race_id", "place","weather","ground_state","race_grade","course_len","race_type","race_date_day_count","course_type"]], on="race_id")
        )

        df_old2["place"] = df_old2["place"].astype(int)
        df_old2["race_grade"] = df_old2["race_grade"].astype(int)
        # 距離/競馬場/タイプ/レースランク
        df_old2["distance_place_type_race_grade"] = (df_old2["course_type"].astype(str) + df_old2["race_grade"].astype(str)).astype(int)
        
        
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
        #平均タイム、芝のみ
        df_old = (
            self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around","course_type"]]
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "rank","umaban","nobori","time","sex"]], on="race_id")
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
        df_old2 = df_old2[df_old2['rank'].isin([1, 2, 3])]
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

        # df["mean_ground_state_time_diff"] = df.apply(
        #     lambda row: compute_mean_for_row(row, df_old2_1=df_old2_1), axis=1
        # )


        
        
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
        if date_condition_a != 0:
            df["ground_state_level"] = date_condition_a
        else:
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
            baselog.loc[condition1, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori']))*(((baselog["course_len"]*0.0025)+20)/20)*(((baselog["pace_diff"] * 1)+20)/20)/ 2.5
        
            # 小回りカーブで場合
            condition2 = (baselog["curve"] == 2)  & (baselog["show"] == 1)
            baselog.loc[condition2, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori']))*(((baselog["course_len"]*0.0025)+20)/20)*(((baselog["pace_diff"] * 1)+20)/20)/ 2.1
            
            # 小スパカーブ場合
            condition3 = (baselog["curve"] == 3) & (baselog["show"] == 1)
            baselog.loc[condition3, "score_stamina"] += (baselog["race_grade_scaled"] + 1+(2/baselog['place_season_condition_type_categori']))*(((baselog["course_len"]*0.0025)+20)/20)*(((baselog["pace_diff"] * 1)+20)/20)/ 1.8
        
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
        self, date_condition_a:int,n_races: list[int] = [1, 3, 5, 10],
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
        

        merged_df = self.population.copy()        
        df = (
            merged_df
            .merge(self.race_info[["race_id", "place","weather","ground_state","course_len","race_type","race_date_day_count"]], on="race_id")
        )
        df["place"] = df["place"].astype(int)
        
        df_old2 = (
            self.old_results_condition[["race_id", "horse_id","time","rank","rank_per_horse"]]
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
            self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","around"]]
            .merge(self.all_results[["race_id", "horse_id", "wakuban", "umaban","nobori","rank","time","sex"]], on="race_id")
        )
        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        df_old["around"] = df_old["around"].fillna(3).astype(int)
        df_old["weather"] = df_old["weather"].astype(int)  
                     
        
        df_old["distance_place_type_race_grade"] = (df_old["course_len"].astype(str) + df_old["place"].astype(str) + df_old["race_type"].astype(str) + df_old["race_grade"].astype(str)).astype(int)
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
            df_old2_1["race_date_day_count"] = df_old2_1["race_date_day_count"].astype(int)
            target_day_count = int(row["race_date_day_count"])  # df の各行の race_date_day_count
    
            # target_day_count = row["race_date_day_count"]  # df の各行の race_date_day_count
        
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
        if date_condition_a != 0:
            df["ground_state_level"] = date_condition_a
        else:
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

        雨、稍重などの場合、全体が濡れるので前の残りになる
        
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
        ＋（	斥量	－	５５	）
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
        #         (df2['ground_state'].isin([0,2])) &  # ground_stateが0または2
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
        horse_results_baselog['time_points_grade'] = (horse_results_baselog['time_diff_grade'] ) *10

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

        horse_results_baselog['time_condition_index'] = horse_results_baselog['time_points_grade_index'] -horse_results_baselog['pace_diff'] *5
        horse_results_baselog['nobori_condition_index'] = horse_results_baselog['nobori_points_grade_index'] +horse_results_baselog['pace_diff'] *5


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






    def umaban_good(
        self, n_races: list[int] =[1, 3, 5, 8]
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
        
            

            





    # def cross1(
    #     self, date_condition_a: int,n_races: list[int] = [1,3,5,8]
    # ):  
        

    #     base_2 = (
    #         self.population
    #         .merge(
    #             self.horse_results,
    #             on="horse_id",
    #             suffixes=("", "_horse"),
    #         )
    #         .query("date_horse < date")
    #         .sort_values("date_horse", ascending=False)
    #     )


    #     merged_df = self.population.copy()  
    #     # # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # #




        # """
        # ・noboriの策定
        # ハイペースなら-,スローペースなら+


        # のぼりやタイムは0.1秒単位で大事
        # +-のほうがいいかも

        # ハイローセット	
        # レースグレードによる距離のハイペース	
        # 最初の直線が長いほど	（倍率は下げる）ハイペースになる
        # コーナーの数が4以上だと、先行争い諦めることがあるので	若干スローに
        # コーナーの数が4以上、直線の合計が1000を超えてくると	若干スローに
        # 短距離は	基本ハイペース
        # 最初直線＿上り平坦	スローペース
        # 最初下り坂	ペースが上がり
            
        # 12コーナーがきつい	スロー
        # コーナーがきつい	スロー
        # 向正面上り坂	スロー
        # 内外セット	
        # 3,4コーナーが急	圧倒的内枠有利になる
        # 3,4が下り坂	圧倒的内枠有利になる
        # スタートからコーナーまでの距離が短い	ポジションが取りづらいため、内枠有利特に偶数有利
        # スタートが上りorくだり坂	ポジションが取りづらいため、内枠有利特に偶数有利
        # コーナーがきついは内枠	内枠
        # 向正面上り坂は内枠	内枠
        # 芝スタートだと外が有利	外枠
        # 芝スタート系は道悪で逆転する	内枠
        # 距離が長いほど関係なくなる	関係なくなる
        # ダートで内枠は不利	外枠

        # """



        # """
        # ハイペースローペース自体の影響は0.5前後にまとまるよう下げる

        # コーナー順位が前（先行）で、ハイペースの場合、noboriをさらに-0.5する（不利条件）
        # 前でローペースの場合、noboriを+0.2する(ペースによる)有利
        # 後ろで、ローペースの場合、noboriを-0.1する（作っておけばrankdiffで使える）不利
        # 後ろでハイの場合、noboriを+0.3する(ハイスローの分を相殺する)有利
        # #+だとハイペース、ーだとスローペース
        # """

        # #最大0.5前後
        # base_2["nobori_pace_diff"] = base_2["nobori"] - (base_2["pace_diff"] / 12)

        # #ハイペースが不利、だから補正する、最大0.05くらい
        # base_2["nobori_pace_diff_grade"] = base_2["nobori_pace_diff"] - (((base_2['race_grade']/70)-1)/8)


        # #坂、0.2くらい
        # #芝が傷んでくる冬から春には、坂は効く

        # base_2["nobori_pace_diff_grade_slope"] = np.where(
        #     (base_2["season"] == 1) | (base_2["season"] == 4),
        #     base_2["nobori_pace_diff_grade"] - (base_2["goal_slope"] / 12),
        #     base_2["nobori_pace_diff_grade"] - (base_2["goal_slope"] / 18)
        # )

        # #直線の長さ0.2くらい
        # base_2["start_range_processed_1"] = (((base_2["start_range"])-360)/150)
        # base_2["start_range_processed_1"] = base_2["start_range_processed_1"].apply(
        #     lambda x: x if x < 0 else x*0.5
        # )
        # base_2["goal_range_processed_1"] = (((base_2["goal_range"])-360)/150)
        # base_2["goal_range_processed_1"] = base_2["goal_range_processed_1"].apply(
        #     lambda x: x*2 if x < 0 else x*0.7
        # )

        # base_2["nobori_pace_diff_grade_slope_range"] = base_2["nobori_pace_diff_grade_slope"] + (base_2["goal_range_processed_1"]/10)

        # """
        # ハイスロー、脚質修正
        # コーナー順位が前（先行）で、ハイペースの場合、noboriをさらに-0.5する（不利条件）
        # 前でローペースの場合、noboriを+0.2する(ペースによる)有利
        # 後ろで、ローペースの場合、noboriを-0.1する（作っておけばrankdiffで使える）不利
        # 後ろでハイの場合、noboriを+0.3する(ハイスローの分を相殺する)有利
        # #+だとハイペース、ーだとスローペース
        # """
        # # 条件ごとに処理を適用
        # base_2["nobori_pace_diff_grade_slope_range_pace"] = np.where(
        #     ((base_2['race_position'] == 1) | (base_2['race_position'] == 2)) & (base_2["pace_diff"] >= 0),
        #     base_2["nobori_pace_diff_grade_slope_range"] - (base_2["pace_diff"] / 8),
            
        #     np.where(
        #         ((base_2['race_position'] == 1) | (base_2['race_position'] == 2)) & (base_2["pace_diff"] < 0),
        #         base_2["nobori_pace_diff_grade_slope_range"] - (base_2["pace_diff"] / 16),
                
        #         np.where(
        #             (base_2['race_position'] == 4) & (base_2["pace_diff"] < 0),
        #             base_2["nobori_pace_diff_grade_slope_range"] - ((base_2["pace_diff"] / 28) * -1),
                    
        #             np.where(
        #                 ((base_2['race_position'] == 3) | (base_2['race_position'] == 4)) & (base_2["pace_diff"] >= 0),
        #                 base_2["nobori_pace_diff_grade_slope_range"] - ((base_2["pace_diff"] / 13) * -1),
        #                 base_2["nobori_pace_diff_grade_slope_range"]  # どの条件にも当てはまらない場合は元の値を保持
        #             )
        #         )
        #     )
        # )

        # """
        # 馬場状態
        # 普通に不良馬場なら-1.5
        # 稍重くらいなら-0.3

        # ダートなら逆で倍率は変わらず、
        # +1.2と+0.6くらいに
        # """

        # # 条件ごとに適用
        # base_2["nobori_pace_diff_grade_slope_range_pace_groundstate"] = np.where(
        #     ((base_2["ground_state"] == 1) | (base_2["ground_state"] == 3)) & (base_2["race_type"] == 1),
        #     base_2["nobori_pace_diff_grade_slope_range_pace"] - 1.2,

        #     np.where(
        #         (base_2["ground_state"] == 2) & (base_2["race_type"] == 1),
        #         base_2["nobori_pace_diff_grade_slope_range_pace"] - 0.3,

        #         np.where(
        #             ((base_2["ground_state"] == 1) | (base_2["ground_state"] == 3)) & (base_2["race_type"] == 0),
        #             base_2["nobori_pace_diff_grade_slope_range_pace"] + 0.8,

        #             np.where(
        #                 (base_2["ground_state"] == 2) & (base_2["race_type"] == 0),
        #                 base_2["nobori_pace_diff_grade_slope_range_pace"] + 0.4,
                        
        #                 # どの条件にも当てはまらない場合は元の値を保持
        #                 base_2["nobori_pace_diff_grade_slope_range_pace"]
        #             )
        #         )
        #     )
        # )

        # #タフが不利、だから補正する
        # """
        # タフパック	
        # ハイペース	タフ
        # コーナー種類	ゆるいと遅くならないのでタフ,だけどゆるいほうが早く出る  カーブが緩い、複合だと早いまま入れる

        # コーナーR	大きいとタフ
        # コーナーの数	少ないほうがタフ
        # 高低差がある	タフ
        # 馬場状態、天気が悪い	タフ
        # 芝によって	タフ
        # 直線合計/コーナー合計	多いほどタフ
        # """

        # # -4.5 を行う
        # base_2["curve_processed"] = base_2["curve"] - 4.5
        # # +の場合は数値を8倍する
        # base_2["curve_processed"] = base_2["curve_processed"].apply(
        #     lambda x: x * 8 if x > 0 else x
        # )
        # #最大0.12くらい
        # base_2["nobori_pace_diff_grade_curve"] = base_2["nobori_pace_diff_grade_slope_range_pace_groundstate"] + (base_2["curve_processed"]/60)






        # """"
        # "curve_amount"を2以下のとき"curve_R34"を
        # "curve_amount"を3以下のとき"curve_R12"/2と"curve_R34"を
        # "curve_amount"を4以下のとき"curve_R12"と"curve_R34"を
        # "curve_amount"を5以下のとき"curve_R12"と"curve_R34"*3/2を
        # "curve_amount"を6以下のとき"curve_R12"と"curve_R34"*2を
        # "curve_amount"を7以下のとき"curve_R12"*3/2と"curve_R34"*2を
        # "curve_amount"を8以下のとき"curve_R12"*2と"curve_R34"*2を
        # """
        # #最大0.02*n
        # def calculate_nobori_pace_diff(row):
        #     if row["curve_amount"] == 0:
        #         return row["nobori_pace_diff_grade_curve"]
        #     elif row["curve_amount"] <= 2:
        #         return row["nobori_pace_diff_grade_curve"] + ((row["curve_R34"]-100)/1200)
        #     elif row["curve_amount"] <= 3:
        #         return row["nobori_pace_diff_grade_curve"] + ((row["curve_R12"]-100)/1200 / 2 + (row["curve_R34"]-100)/1200)
        #     elif row["curve_amount"] <= 4:
        #         return row["nobori_pace_diff_grade_curve"] + ((row["curve_R12"]-100)/1200 + ((row["curve_R34"]-100)/1200))
        #     elif row["curve_amount"] <= 5:
        #         return row["nobori_pace_diff_grade_curve"] + ((row["curve_R12"]-100)/1200 + ((row["curve_R34"]-100)/1200) * 3 / 2)
        #     elif row["curve_amount"] <= 6:
        #         return row["nobori_pace_diff_grade_curve"] +((row["curve_R12"]-100)/1200 + ((row["curve_R34"]-100)/1200) * 2)
        #     elif row["curve_amount"] <= 7:
        #         return row["nobori_pace_diff_grade_curve"] + ((row["curve_R12"]-100)/1200 * 3 / 2 + ((row["curve_R34"]-100)/1200) * 2)
        #     else:  # curve_amount <= 8
        #         return row["nobori_pace_diff_grade_curve"] + ((row["curve_R12"]-100)/1200 * 2 +((row["curve_R34"]-100)/1200) * 2)

        # base_2["nobori_pace_diff_grade_curveR"] = base_2.apply(calculate_nobori_pace_diff, axis=1)

        # #最大0.09くらい
        # base_2["nobori_pace_diff_grade_curveR_height_diff"] = base_2["nobori_pace_diff_grade_curveR"] - ((base_2["height_diff"]/30)-0.02)

        # #芝の質で一秒くらい違う
        # #最大と最小で0,4:-0,4

        # base_2 = base_2.copy()
        # base_2.loc[:, "place_season_condition_type_categori_processed"] = (
        #     base_2["place_season_condition_type_categori"]
        #     .replace({5: -0.3, 4: -0.17, 3: 0, 2: 0.17,1: 0.3, -1: -0.18, -2: 0, -3: 0.18,-4:0.3,-10000:0})
        # ).astype(float)



        # #最大0.5くらい
        # base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] = base_2["nobori_pace_diff_grade_curveR_height_diff"] - base_2['place_season_condition_type_categori_processed']

        # #最大0.05くらい
        # base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight"] = base_2["nobori_pace_diff_slope_range_groundstate_position_umaban"] - (((base_2["straight_total"]/ base_2["course_len"])/10)-0.05)

        # # 1600で正規化
        # base_2["course_len_processed"] = (base_2["course_len"] / 1800)-1

        # # 1600m未満ならそのまま、1600m以上なら緩やかに上昇 
        # base_2["course_len_processed_1"] = base_2["course_len_processed"].apply(
        #     lambda x: x*0.2 if x <= 0 else x*0.1
        # )

        # #最大0.1くらい
        # base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len"] = base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight"] - base_2["course_len_processed_1"]

        # """
        # 内外セット	
        # 3,4コーナーが急	圧倒的内枠有利になる
        # 3,4が下り坂	圧倒的内枠有利になる
        # スタートからコーナーまでの距離が短い	ポジションが取りづらいため、内枠有利特に偶数有利
        # スタートが上りorくだり坂	ポジションが取りづらいため、内枠有利特に偶数有利
        # コーナーがきついは内枠	内枠
        # 向正面上り坂は内枠	内枠
        # 芝スタートだと外が有利	外枠
        # 芝スタート系は道悪で逆転する	内枠
        # 距離が長いほど関係なくなる	関係なくなる
        # ダートで内枠は不利	外枠
        # """

        # #0,01-0.01,内がマイナス
        # base_2["umaban_processed"] = base_2["umaban"].apply(
        #     lambda x: ((x*-1/200)) if x < 4 else (x-8)/1250
        # ).astype(float)
        # #0-0.005
        # base_2.loc[:, "umaban_odd_processed"] = (
        #     (base_2["umaban_odd"]-1)/200
        # ).astype(float)

        # # 1600m未満ならそのまま、1600m以上なら緩やかに上昇 
        # base_2["course_len_processed_2"] = base_2["course_len_processed"].apply(
        #     lambda x: x+1 if x <= 0 else x+1
        # )

        # base_2["umaban_processed_2"] = base_2["umaban_processed"] / base_2["course_len_processed_2"]
        # base_2["umaban_odd_processed_2"] = base_2["umaban_odd_processed"] / base_2["course_len_processed_2"]


        # #最大0.03くらい、不利が+,ダートは外枠有利
        # base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban"] = np.where(
        #     base_2["race_type"] == 0,
        #     base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len"] + base_2["umaban_processed_2"],
        #     base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len"] - base_2["umaban_processed_2"]
        # )

        # base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds"]= np.where(
        #     base_2["race_type"] == 0,
        #     base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban"] - base_2["umaban_odd_processed_2"],
        #     base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban"] - base_2["umaban_odd_processed_2"]
        # )



        # #+-0.03急カーブ,フリ評価
        # base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve"] = (
        #     base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds"] - ((base_2["umaban_processed_2"]*(base_2["curve_processed"]/4))*-1)
        # )


        # #+-0.03カーブ下り坂,フリ評価
        # base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope"] = (
        #     base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve"] - ((base_2["umaban_processed_2"]*(base_2["last_curve_slope"]/2))*-1)
        # )



        # base_2["start_range_processed"] = (((base_2["start_range"])-360)/150)
        # base_2["start_range_processed"] = base_2["start_range_processed"].apply(
        #     lambda x: x if x < 0 else x*0.5
        # )

        # #+-0.06,スタートからコーナー
        # base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range"] = (
        #     base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope"] - ((base_2["umaban_processed_2"]*(base_2["start_range_processed"]))*-1)- ((base_2["umaban_odd_processed_2"]*(base_2["start_range_processed"]))*-1)
        # )



        # base_2["start_slope_abs"] = base_2["start_slope"].abs()
        # base_2["start_slope_abs_processed"] = base_2["start_slope_abs"] /4

        # #+-0.06,スタートからコーナー、坂,上り下り両方
        # base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] = (
        #     base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range"] - ((base_2["umaban_processed_2"]*(base_2["start_slope_abs_processed"]))*-1)- ((base_2["umaban_odd_processed_2"]*( base_2["start_slope_abs_processed"]))*-1)
        # )




        # #最大0.3*nコーナーがきついは内枠
        # def calculate_nobori_pace_diff_2(row):
        #     if row["curve_amount"] == 0:
        #         return row["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"]
        #     elif row["curve_amount"] <= 2:
        #         return row["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] - (row["umaban_processed_2"]*(((row["curve_R34"]-100)/450)*-1))
        #     elif row["curve_amount"] <= 3:
        #         return row["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] - (row["umaban_processed_2"]*(((row["curve_R12"]-100)/450 / 2 + (row["curve_R34"]-100)/450)*-1))
        #     elif row["curve_amount"] <= 4:
        #         return row["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] - (row["umaban_processed_2"]*(((row["curve_R12"]-100)/450 + ((row["curve_R34"]-100)/450))*-1))
        #     elif row["curve_amount"] <= 5:
        #         return row["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] - (row["umaban_processed_2"]*(((row["curve_R12"]-100)/450 + ((row["curve_R34"]-100)/450) * 3 / 2)*-1))
        #     elif row["curve_amount"] <= 6:
        #         return row["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] -((row["umaban_processed_2"]*((row["curve_R12"]-100)/450 + ((row["curve_R34"]-100)/450) * 2)*-1))
        #     elif row["curve_amount"] <= 7:
        #         return row["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] - (row["umaban_processed_2"]*(((row["curve_R12"]-100)/450 * 3 / 2 + ((row["curve_R34"]-100)/450) * 2)*-1))
        #     else:  # curve_amount <= 8
        #         return row["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] - ((row["umaban_processed_2"]*((row["curve_R12"]-100)/450 * 2 +((row["curve_R34"]-100)/450) * 2)*-1))


        # base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner"] = base_2.apply(calculate_nobori_pace_diff_2, axis=1)


        # #最大1*向正面
        # base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont"] = (
        #     base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner"] - ((base_2["umaban_processed_2"]*(base_2["flont_slope"]/4)))
        # )


        # #芝スタートかつ良馬場、芝スタートかつ良以外、どっちでもない場合,外評価

        # condition = (base_2["start_point"] == 2) & (base_2["ground_state"] == 0)

        # base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont_point"] = np.where(
        #     condition,
        #     base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont"] - (base_2["umaban_processed_2"] * -1),
        #     base_2["nobori_pace_diff_slope_range_groundstate_position_umaban_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont"] - base_2["umaban_processed_2"]
        # )






    def cross_rank_diff(
        self, date_condition_a: int,n_races: list[int] = [1, 3,5,8]
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

        # #高速馬場だと差がつく
        # #着差がつかない、フリを-側へ
        # base_2["rank_diff_pace_diff"] = (
        #     (base_2["rank_diff"] + 1) 
        #     * ((base_2["pace_diff"]+20) / 20) 
        #     * (((base_2["place_season_condition_type_categori"]*-1)+3)/3)
        #     )


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
        #     base_2["season_turf"] == 1, base_2["day"],
        #     np.where(
        #         base_2["season_turf"] == 2, (base_2["day"] + 1.5) * 1.5,
        #         np.where(
        #             base_2["season_turf"] == 3, base_2["day"] + 3,
        #             np.where(
        #                 base_2["season_turf"] == 4, base_2["day"] + 4,
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
                        -  (base_2["race_type"] - 0.5)*4# 芝ほど数値が減る 2
                        -  ((base_2["first_corner"] - 100)/50)# 最初のコーナーがでかいほど数値が減る1
                ) 
            ) +500) / 500)
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
                        +  (base_2["race_type"] - 0.5)*4# 芝ほど数値が減る 2
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












        # if base_2["rush_type"] < 0:
        #     base_2["rank_diff_correction_rush"] =(
        #         base_2["rank_diff_correction"] 
        #         - (((base_2["rush_type"]+0.1)/30)*((base_2["curve_amount"]-4)/8)) 
        #         - (((base_2["rush_type"]+0.1)/30)*(base_2["curve_processed"] /-4))
        #         - (((base_2["rush_type"]+0.1)/30)*(base_2["goal_range_processed_1"] /1.2))
        #         - (((base_2["rush_type"]+0.1)/30)*(base_2["goal_slope"] /4))
        #         - (((base_2["rush_type"]+0.1)/30)*((base_2["pace_diff"]+0.6) / -3))
        #         - (((base_2["rush_type"]+0.1)/30)*((base_2["umaban_rank_diff_processed_2"]) * -10))
        #         - (((base_2["rush_type"]+0.1)/30)*((base_2["height_diff"]/-2)))
        #     )
        # if base_2["rush_type"] >= 0:
        #     base_2["rank_diff_correction_rush"] =(
        #         base_2["rank_diff_correction"] 
        #         - (((base_2["rush_type"]+0.1)/30)*((base_2["curve_amount"]-4)/8)) 
        #         - (((base_2["rush_type"]+0.1)/30)*(base_2["curve_processed"] /-4))
        #         - (((base_2["rush_type"]+0.1)/30)*(base_2["goal_range_processed_1"] /1.2))
        #         - (((base_2["rush_type"]+0.1)/30)*(base_2["goal_slope"] /4))
        #         - (((base_2["rush_type"]+0.1)/30)*((base_2["pace_diff"]+0.6) / -3))
        #         - (((base_2["rush_type"]+0.1)/30)*((base_2["umaban_rank_diff_processed_2"]) * -10))
        #         - (((base_2["rush_type"]+0.1)/30)*((base_2["height_diff"]/-2)))
        #     )








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
        40前後
        
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
        self, date_condition_a: int,n_races: list[int] = [1, 3,5,8]
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


        base_2["time_pace_diff"] = base_2["time"] - (base_2["pace_diff"]/4) - (base_2["place_season_condition_type_categori"]*5)

        base_2["time_pace_diff_slope"] = np.where(
            (base_2["season"] == 1) | (base_2["season"] == 4),
            base_2["time_pace_diff"] / ((base_2["goal_slope"] + 700)/700),
            base_2["time_pace_diff"] / ((base_2["goal_slope"] + 1200)/1200),
        )


        base_2["goal_range_processed_1"] = (((base_2["goal_range"])-360))
        base_2["goal_range_processed_1"] = base_2["goal_range_processed_1"].apply(
            lambda x: x*2 if x < 0 else x*0.5
        )

        #ゴールが短いと上りの分が入らないので上りが少し遅くなる
        base_2["time_pace_diff_slope_range"] = base_2["time_pace_diff_slope"] * ((base_2["goal_range_processed_1"] + 80000) / 80000)/((base_2["height_diff"]+200)/200)


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
            ((base_2['race_position'] == 1) & (base_2["pace_diff"] >= 0)),
            base_2["time_pace_diff_slope_range"] - (base_2["pace_diff"] / 5),
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
            base_2["time_pace_diff_slope_range_pace"] * (28/30),

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
        #     base_2["season_turf"] == 1, base_2["day"],
        #     np.where(
        #         base_2["season_turf"] == 2, (base_2["day"] + 1.5) * 1.5,
        #         np.where(
        #             base_2["season_turf"] == 3, base_2["day"] + 3,
        #             np.where(
        #                 base_2["season_turf"] == 4, base_2["day"] + 4,
        #                 base_2["day"]  # それ以外のとき NaN
        #             )
        #         )
        #     )
        # )



        #-3 から2
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



        #-4.5 から3
        # # 内が小さい
        # base_2["time_pace_diff_slope_range_groundstate_position_umaban"] = np.where(
        #     (base_2["umaban_judge"] < 0),
        #     base_2["time_pace_diff_slope_range_groundstate_position"] /
        #     (
        #         ((base_2["umaban_processed_2"] + 400) / 400)  # 少ないほうがtimeが増える
        #         * ((base_2["umaban_odd_processed_2"] + 300) / 300)  # 奇数不利なので分母を増やして総合を減らす
        #         * (((base_2["start_point"] - 1) + 200) / 200)  # 外枠が有利なので分母を増やして総合を減らす
        #         * ((base_2["curve_processed"] + 500) / 500)  # ラストカーブきついほど数値が減る
        #         * ((base_2["last_curve_slope"] + 700) / 700)  # ラストカーブくだりほど数値が減る
        #         * (((base_2["season_turf_condition"] - 7) + 500) / 500)  # 馬場状態が良いほど数値が減る
        #         * (((base_2["race_type"] - 0.5) + 200) / 200)  # 芝ほど数値が減る
        #         / (((base_2["first_corner"] - 100) + 50000) / 50000)  # 最初のコーナーがでかいほど数値が減る
        #     ),

        #     np.where(
        #         (base_2["umaban_judge"] >= 0),
        #         base_2["time_pace_diff_slope_range_groundstate_position"] /
        #         (
        #             ((base_2["umaban_processed_2"] + 400) / 400)
        #             * ((base_2["umaban_odd_processed_2"] + 300) / 300)
        #             / (((base_2["start_point"] - 1) + 200) / 200)  # 外枠が有利
        #             / ((base_2["curve_processed"] + 500) / 500)
        #             / ((base_2["last_curve_slope"] + 700) / 700)
        #             / (((base_2["season_turf_condition"] - 7) + 500) / 500)
        #             / (((base_2["race_type"] - 0.5) + 200) / 200)
        #             * (((base_2["first_corner"] - 100) + 50000) / 50000)
        #         ),

        #         base_2["time_pace_diff_slope_range_groundstate_position"]
        #     )
        # )

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
                      -  (base_2["race_type"] - 0.5)*4# 芝ほど数値が減る 2
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
                        +  (base_2["race_type"] - 0.5)*4# 芝ほど数値が減る 2
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
        self, date_condition_a: int,n_races: list[int] = [1,3,5,8]
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
            self.old_results_condition[["race_id", "horse_id","time","rank","rank_per_horse"]]
            .merge(self.old_race_info_condition[["race_id", "place","weather","ground_state","race_grade","course_len","race_type","course_type","race_date_day_count"]], on="race_id")
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
        if date_condition_a != 0:
            df["ground_state_level"] = date_condition_a
        else:
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
                self.race_info[["race_id", "goal_range_100","curve","weather",'ground_state','start_point','curve_R12','curve_R34', "curve_amount","season_turf_condition",'flont_slope',"goal_slope",'first_curve_slope', 'goal_range',"course_len","place_season_condition_type_categori","start_slope","start_range","race_grade","race_type","last_curve_slope","season","day"]], 
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


        馬場状態が16前後
        ペースが30前後

        """

        if date_condition_a != 0:
            merged_df_all["ground_state_level_processed"] = ((date_condition_a)-4)*8
        else:
            # merged_df_all.loc[(merged_df_all['race_type'] == 0) & (merged_df_all['weather'].isin([1, 2])) & (merged_df_all['ground_state'] == 2), "ground_state_level_processed"] = 12
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
        merged_df_all["start_range_processed_1"] = ((merged_df_all["start_range"] - 360) / 30).apply(lambda x: x * 3 if x < 0 else x * 0.6)

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


        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_sumed"] + (merged_df_all["goal_range_processed_1"]*2.5)

        # dominant_position_category_processed の更新 (再利用)
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] < 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: -0.7, 4: 1.67}
        )
        merged_df_all.loc[merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] >= 0, "dominant_position_category_processed"] = merged_df_all["dominant_position_category"].replace(
            {1: -1.03, 2: 0.7, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] * merged_df_all["dominant_position_category_processed"]





        merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_sumed"] = merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] + (merged_df_all["goal_slope"]*4)

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



        """
        umabanのみ補正(rank_diff)
        "rank_diff_pace_course_len_ground_state_type",
        "rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope",
        "rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point",
        "rank_diff_correction",
        "rank_diff_correction_position",
        "rank_diff_correction_rush",
        "rank_diff_correction_position_rush",
        "rank_diff_correction_position_rush_xxx_race_grade_multi",
        "rank_diff_correction_position_xxx_race_grade_multi",
        "rank_diff_correction_rush_xxx_race_grade_multi",
        "rank_diff_correction_xxx_race_grade_multi",
        "rank_diff_correction_position_rush_xxx_race_grade_sum",
        "rank_diff_correction_position_xxx_race_grade_sum",
        "rank_diff_correction_rush_xxx_race_grade_sum",
        "rank_diff_correction_xxx_race_grade_sum",

        """
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

        """
        レースグレードは40-100
        馬番の補正は最大45から-45なので/10くらいしてsumと+80くらいして/80のmulti
        展開は両方の最大80と200くらい、/20と/80してsumと+800/800のmultiと+2000して/2000のmulti
        """
        # + merged_df_all["tenkai_combined_2"]/80
        # + merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/200
        # + merged_df_all["umaban_curve_lcurves_condition_type_corner_combined"]/10

        # * ((merged_df_all["tenkai_combined_2"]+800)/800)
        # * ((merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]+2000)/2000)
        # * ((merged_df_all["umaban_curve_lcurves_condition_type_corner_combined"]+80)/80)


        # + merged_df_all["umaban_curve_lcurves_condition_type_corner_combined"]/10 + merged_df_all["tenkai_combined_2"]/80
        # + merged_df_all["umaban_curve_lcurves_condition_type_corner_combined"]/10 + merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/200

        # * ((merged_df_all["umaban_curve_lcurves_condition_type_corner_combined"]+80)/80) * ((merged_df_all["tenkai_combined_2"]+800)/800)
        # * ((merged_df_all["umaban_curve_lcurves_condition_type_corner_combined"]+80)/80) * ((merged_df_all["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]+2000)/2000)


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

        """
        かけるなら1.05、+なら2前後
        レースグレードは40-100
        馬番の補正は最大45から-45なので/10くらいしてsumと+80くらいして/80のmulti
        展開は両方の最大80と200くらい、/20と/80してsumと+800/800のmultiと+2000して/2000のmulti
        """
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





        # umaban_correction_position_factor = (300 + umaban_correction_position) / 300
        #         columns_to_process = [
        #             "rank_diff_correction_position_rush_xxx_race_grade_multi_mean",
        #             "rank_diff_correction_position_xxx_race_grade_multi_mean",
        #             "rank_diff_correction_rush_xxx_race_grade_multi_mean",
        #             "rank_diff_correction_xxx_race_grade_multi_mean",
        #             "rank_diff_correction_position_rush_xxx_race_grade_sum_mean",
        #             "rank_diff_correction_position_xxx_race_grade_sum_mean",
        #             "rank_diff_correction_rush_xxx_race_grade_sum_mean",
        #             "rank_diff_correction_xxx_race_grade_sum_mean"
        #         ]

        #         # 1〜8レースの異なる値を処理
        #         races = [1,3,5,8]

        #         # 各列に対して一括処理
        #         for column_prefix in columns_to_process:
        #             for race in races:
        #                 new_column_name = f"{column_prefix}_{race}races_umaban"
        #                 original_column_name = f"{column_prefix}_{race}races"
                        
        #                 # 列を計算して新しい列を追加
        #                 merged_df_all[new_column_name] = merged_df_all[original_column_name] * umaban_correction_position_factor

        #         # 新しいコピーを作成
        #         merged_df_all = merged_df_all.copy()




        #         """
        #         それぞれのrank_gradeかけ
        #         """



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
        # '_standardized', 'race_id', 'date', 'horse_id' を除くすべての列を削除
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
                self.race_info, on="race_id"
            )
            .merge(
                self.results[["race_id","umaban","n_horses","horse_id",'umaban_odd']], on=["race_id","horse_id"]
            )
            .merge(
                merged_df2[["horse_id","date", "race_id","pace_category","dominant_position_category"]],
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
                        +  (baselog_1["race_type"] - 0.5)*4# 芝ほど数値が減る 2
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
                        - (baselog_1["race_type"] - 0.5)*4# 芝ほど数値が減る 2
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



    #     print(df.columns.tolist())
    #     merged_df_final = self.population.copy()

    #     merged_df_final2 = merged_df_final.merge(
    #             df, on=["race_id", "date", "horse_id"], how="left"
    #         )
        
    #     print(df.columns)
    #     self.agg_cross = merged_df_final2
    #     print("cross...comp")
        


    def create_features(
        self, race_id: str, date_content_a: str, date_condition_a: int,skip_agg_horse: bool = False
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
            
        # 各種テーブルの取得
        self.fetch_shutuba_page_html(race_id)
        self.fetch_results(race_id, self.htmls[race_id])
        self.fetch_race_info(race_id, date_content_a,self.htmls[race_id])
        self.create_race_grade()
        # グループごとの馬の過去成績集計（race_infoのカラムが必要なため、ここで実行）
        # self.agg_horse_n_races_relative()
        self.cross_features()
        self.agg_interval() 
        self.agg_course_len()
        self.results_relative()

        self.cross_rank_diff(date_condition_a)
        self.cross_time(date_condition_a)
        self.cross2(date_condition_a)
        self.race_type_True()


        self.cross_features_2()
        self.cross_features_3()
        self.cross_features_4()
        self.cross_features_5()
        self.cross_features_6(date_condition_a)
        self.cross_features_7()
        self.cross_features_8()
        self.cross_features_9()
        self.cross_features_10()
        self.cross_features_11()
        self.cross_features_12()
        self.cross_features_13()
        self.cross_features_14(date_condition_a)
        self.cross_features_15()
        # self.cross_features_16()
        self.position_results()
        self.dirt_weight_weather()
        self.umaban_good()



        self.agg_horse_per_course_len()

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
        # リーディングデータの紐付け
        # self.agg_jockey()
        # self.agg_trainer()
        # self.agg_sire()
        # self.agg_bms()
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
            # .merge(
            #     self.agg_horse_n_races_relative_df,
            #     on=["race_id","date","horse_id"],
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
            # .merge(
            #     self.agg_horse_per_group_cols_dfs["race_grade"],
            #     on=["race_id",  "date","horse_id"],
            #     how="left",
            #     # copy=False,
            # )
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
            #     self.agg_cross_features_df_16,
            #     on=["race_id", "date", "horse_id"],
            #     how="left",
            #     # copy=False,
            # )   
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
            #     self.agg_cross1,
            #     on=["race_id","date","horse_id"],
            #     how="left",
            #     # copy=False,
            # )   
            # .merge(
            #     self.agg_cross2,
            #     on=["race_id","date","horse_id"],
            #     how="left",
            #     # copy=False,
            # )                                  
            # .merge(
            #     self.agg_horse_per_group_cols_dfs["around_per_wakuban"],
            #     on=["race_id", "horse_id"],
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
        features.drop(columns=["date"], inplace=True)
        # features.drop(columns=['place_season_condition_type_categori_x'], inplace=True)

        features.to_csv(self.output_dir / self.output_filename, sep="\t", index=False)
        print("merging all features...comp")
        return features