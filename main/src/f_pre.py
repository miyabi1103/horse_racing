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
            2010002:{"コーナー数": 2, "最終直線": 260, "ゴール前坂": -1,  "スタート位置": 1,    "最初直線": 366.4,"直線合計": 626, "コーナー合計m":374,"コーナータイプ":2,"コーナーR12":0,"コーナーR34":147,"幅":20.0,   "最初坂":2,"向正面坂":2,"最初コーナー坂":1.25,"最終コーナー坂":-0.25},

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
            5013001:{"コーナー数": 2, "最終直線": 501, "ゴール前坂": 2.5,  "スタート位置": 1,    "最初直線":341.9,"直線合計": 844, "コーナー合計m":456,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":0.9,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
            5013002:{"コーナー数": 2, "最終直線": 501, "ゴール前坂": 2.5,  "スタート位置": 1,    "最初直線":341.9,"直線合計": 844, "コーナー合計m":456,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":0.9,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
            
            5014001:{"コーナー数": 2, "最終直線": 501, "ゴール前坂": 2.5,  "スタート位置": 1,    "最初直線":441.9,"直線合計": 944, "コーナー合計m":456,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":-0.6,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
            5014002:{"コーナー数": 2, "最終直線": 501, "ゴール前坂": 2.5,  "スタート位置": 1,    "最初直線":441.9,"直線合計": 944, "コーナー合計m":456,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":-0.6,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},

            5016001:{"コーナー数": 2, "最終直線": 501, "ゴール前坂": 2.5,  "スタート位置": 2,    "最初直線":641.9,"直線合計": 1144, "コーナー合計m":456,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":-1,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
            5016002:{"コーナー数": 2, "最終直線": 501, "ゴール前坂": 2.5,  "スタート位置": 2,    "最初直線":641.9,"直線合計": 1144, "コーナー合計m":456,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":-1,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},

            5021001:{"コーナー数": 4, "最終直線": 501, "ゴール前坂": 2.5,  "スタート位置": 1,    "最初直線":236.1,"直線合計": 1188, "コーナー合計m":912,"コーナータイプ":5,"コーナーR12":162,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":0,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
            5021002:{"コーナー数": 4, "最終直線": 501, "ゴール前坂": 2.5,  "スタート位置": 1,    "最初直線":236.1,"直線合計": 1188, "コーナー合計m":912,"コーナータイプ":5,"コーナーR12":162,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":0,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
    
            5024001:{"コーナー数": 4, "最終直線": 501, "ゴール前坂": 2.5,  "スタート位置": 1,    "最初直線":536.1,"直線合計": 1488, "コーナー合計m":912,"コーナータイプ":5,"コーナーR12":162,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":2.5,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
            5024002:{"コーナー数": 4, "最終直線": 501, "ゴール前坂": 2.5,  "スタート位置": 1,    "最初直線":536.1,"直線合計": 1488, "コーナー合計m":912,"コーナータイプ":5,"コーナーR12":162,"コーナーR34":162,"高低差":2.5,"幅":25.0,   "最初坂":2.5,"向正面坂":0.9,"最初コーナー坂":-1.7,"最終コーナー坂":0},
    

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
            7012001:{"コーナー数": 2, "最終直線": 410.7, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":407.7,"直線合計": 818, "コーナー合計m":382,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":1.2,"向正面坂":-1.2,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
            7012002:{"コーナー数": 2, "最終直線": 410.7, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":407.7,"直線合計": 818, "コーナー合計m":382,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":1.2,"向正面坂":-1.2,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
    
            7014001:{"コーナー数": 2, "最終直線": 410.7, "ゴール前坂": 2,  "スタート位置": 2,    "最初直線":607.7,"直線合計": 1018, "コーナー合計m":382,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":1.2,"向正面坂":-1.2,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
            7014002:{"コーナー数": 2, "最終直線": 410.7, "ゴール前坂": 2,  "スタート位置": 2,    "最初直線":607.7,"直線合計": 1018, "コーナー合計m":382,"コーナータイプ":3,"コーナーR12":0,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":1.2,"向正面坂":-1.2,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
    
            7018001:{"コーナー数": 4, "最終直線": 410.7, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":291.8,"直線合計": 1103, "コーナー合計m":697,"コーナータイプ":3,"コーナーR12":95,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":1.5,"向正面坂":-1.2,"最初コーナー坂":0,"最終コーナー坂":-1},
            7018001:{"コーナー数": 4, "最終直線": 410.7, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":291.8,"直線合計": 1103, "コーナー合計m":697,"コーナータイプ":3,"コーナーR12":95,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":1.5,"向正面坂":-1.2,"最初コーナー坂":0,"最終コーナー坂":-1},

            7019001:{"コーナー数": 4, "最終直線": 410.7, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":391.80,"直線合計": 1203, "コーナー合計m":697,"コーナータイプ":3,"コーナーR12":95,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":2,"向正面坂":-1.2,"最初コーナー坂":0,"最終コーナー坂":-1},
            7019002:{"コーナー数": 4, "最終直線": 410.7, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":391.80,"直線合計": 1203, "コーナー合計m":697,"コーナータイプ":3,"コーナーR12":95,"コーナーR34":115,"高低差":3.4,"幅":25.0,   "最初坂":2,"向正面坂":-1.2,"最初コーナー坂":0,"最終コーナー坂":-1},


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
            5114001:{"コーナー数": 2, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":342.7,"直線合計": 869, "コーナー合計m":531,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":1.7,"向正面坂":1.7,"最初コーナー坂":-1.7,"最終コーナー坂":0.3},
            5114002:{"コーナー数": 2, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":342.7,"直線合計": 869, "コーナー合計m":531,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":1.7,"向正面坂":1.7,"最初コーナー坂":-1.7,"最終コーナー坂":0.3},
            
            5116001:{"コーナー数": 2, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":542.7,"直線合計":1069, "コーナー合計m":531,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":1.7,"向正面坂":1.7,"最初コーナー坂":-1.7,"最終コーナー坂":0.3},
            5116002:{"コーナー数": 2, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":542.7,"直線合計":1069, "コーナー合計m":531,"コーナータイプ":5,"コーナーR12":0,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":1.7,"向正面坂":1.7,"最初コーナー坂":-1.7,"最終コーナー坂":0.3},
            
            5118001:{"コーナー数": 3, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":156.6,"直線合計":1226, "コーナー合計m":574,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":-0.7,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            5118002:{"コーナー数": 3, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":156.6,"直線合計":1226, "コーナー合計m":574,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":-0.7,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            
            5120001:{"コーナー数": 3, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":126.2,"直線合計":1195, "コーナー合計m":805,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":-0.7,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            5120002:{"コーナー数": 3, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":126.2,"直線合計":1195, "コーナー合計m":805,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":-0.7,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            
            5123001:{"コーナー数": 4, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":249.5,"直線合計":1321, "コーナー合計m":979,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":0,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            5123002:{"コーナー数": 4, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":249.5,"直線合計":1321, "コーナー合計m":979,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":0,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            
            5124001:{"コーナー数": 4, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":349.5,"直線合計":1421, "コーナー合計m":979,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":0,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            5124002:{"コーナー数": 4, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":349.5,"直線合計":1421, "コーナー合計m":979,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":0,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            
            5125001:{"コーナー数": 4, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":449.5,"直線合計":1521, "コーナー合計m":979,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":2,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            5125002:{"コーナー数": 4, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":449.5,"直線合計":1521, "コーナー合計m":979,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":2,"向正面坂":1.7,"最初コーナー坂":-0.7,"最終コーナー坂":0.3},
            
            5134001:{"コーナー数": 6, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":259.8,"直線合計":1890, "コーナー合計m":1510,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":1.5,"向正面坂":1.7,"最初コーナー坂":-1.7,"最終コーナー坂":0.3},
            5134002:{"コーナー数": 6, "最終直線": 526, "ゴール前坂": 2,  "スタート位置": 1,    "最初直線":259.8,"直線合計":1890, "コーナー合計m":1510,"コーナータイプ":5,"コーナーR12":187,"コーナーR34":187,"高低差":2.7,"幅":31.0,   "最初坂":1.5,"向正面坂":1.7,"最初コーナー坂":-1.7,"最終コーナー坂":0.3},
            


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
            7112001:{"コーナー数": 2, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":315.5,"直線合計": 728, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":-0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
            7112002:{"コーナー数": 2, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":315.5,"直線合計": 728, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":-0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},

            7113001:{"コーナー数": 2, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":415.5,"直線合計": 828, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
            7113002:{"コーナー数": 2, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":415.5,"直線合計": 828, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},

            7114001:{"コーナー数": 2, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":515.5,"直線合計": 928, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
            7114002:{"コーナー数": 2, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":515.5,"直線合計": 928, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":0,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},

            7116001:{"コーナー数": 3, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":199.0,"直線合計": 1028, "コーナー合計m":572,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":0.5,"最終コーナー坂":-1},
            7116002:{"コーナー数": 3, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":199.0,"直線合計": 1028, "コーナー合計m":572,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":0.5,"最終コーナー坂":-1},

            7120001:{"コーナー数": 4, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":314.1,"直線合計": 1142, "コーナー合計m":858,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":2,"向正面坂":-0.6,"最初コーナー坂":0.2,"最終コーナー坂":-1},
            7120002:{"コーナー数": 4, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":314.1,"直線合計": 1142, "コーナー合計m":858,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":2,"向正面坂":-0.6,"最初コーナー坂":0.2,"最終コーナー坂":-1},

            7122001:{"コーナー数": 4, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":514.1,"直線合計": 1342, "コーナー合計m":858,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":2,"向正面坂":-0.6,"最初コーナー坂":0.2,"最終コーナー坂":-1},
            7122002:{"コーナー数": 4, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":514.1,"直線合計": 1342, "コーナー合計m":858,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":2,"向正面坂":-0.6,"最初コーナー坂":0.2,"最終コーナー坂":-1},

            7130001:{"コーナー数": 6, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":415.5,"直線合計": 828, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},
            7130002:{"コーナー数": 6, "最終直線": 412.5, "ゴール前坂": 2.3,  "スタート位置": 1,    "最初直線":415.5,"直線合計": 828, "コーナー合計m":472,"コーナータイプ":4,"コーナーR12":120,"コーナーR34":140,"高低差":3.5,"幅":28.0,   "最初坂":0.5,"向正面坂":-0.6,"最初コーナー坂":-1.4,"最終コーナー坂":-1},


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
                    "corve_amount": course_info["コーナー数"], 
                    "curve": course_info["コーナータイプ"], 
                    "goal_slope": course_info["ゴール前坂"],

                    "start_point": course_info["スタート位置"], 
                    "start_range": course_info["最初直線"], 
                    "straight_total": course_info["直線合計"], 
                    "corve_total": course_info["コーナー合計m"],
                    "corve_R12": course_info["コーナーR12"],
                    "corve_R34": course_info["コーナーR34"],
                    "height_diff": course_info["高低差"],
                    "width": course_info["幅"],
                    "start_slope": course_info["最初坂"],
                    "flont_slope": course_info["向正面坂"],
                    "first_curve_slope": course_info["最初コーナー坂"],
                    "last_curve_slope": course_info["最終コーナー坂"]
                })
            else:
                return pd.Series({"goal_range": None, "curve": None, "goal_slope": None,"corve_amount":None,"start_point":None,"start_range":None,"straight_total":None,"corve_total":None,"corve_R12":None,"corve_R34":None,"height_diff":None,"width":None,"start_slope":None,"flont_slope":None,"first_curve_slope":None,"last_curve_slope":None})
        
        # 競馬場カテゴリに基づく変換を追加
        df[['goal_range', 'curve', 'goal_slope',"corve_amount","start_point","start_range","straight_total","corve_total","corve_R12","corve_R34","height_diff","width","start_slope","flont_slope","first_curve_slope","last_curve_slope"]] = df.apply(convert_course, axis=1)
        
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
        #         "corve_amount": course_info["コーナー数"], 
        #         "curve": course_info["コーナータイプ"], 
        #         "goal_slope": course_info["ゴール前坂"],

        #         "start_point": course_info["スタート位置"], 
        #         "start_range": course_info["最初直線"], 
        #         "straight_total": course_info["直線合計"], 
        #         "corve_total": course_info["コーナー合計m"],
        #         "corve_R12": course_info["コーナーR12"],
        #         "corve_R34": course_info["コーナーR34"],
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
            "goal_range", "curve", "goal_slope", "corve_amount", "start_point",
            "start_range", "straight_total", "corve_total", "corve_R12", "corve_R34",
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



    def cross(
        self, date_condition_a: int,n_races: list[int] = [1, 3, 5, 8]
    ):  
        
        merged_df = self.population.copy()  
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



        # # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # #




        """
        ・noboriの策定
        ハイペースなら-,スローペースなら+


        のぼりやタイムは0.1秒単位で大事
        +-のほうがいいかも

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
        内外セット	
        3,4コーナーが急	圧倒的内枠有利になる
        3,4が下り坂	圧倒的内枠有利になる
        スタートからコーナーまでの距離が短い	ポジションが取りづらいため、内枠有利特に偶数有利
        スタートが上りorくだり坂	ポジションが取りづらいため、内枠有利特に偶数有利
        コーナーがきついは内枠	内枠
        向正面上り坂は内枠	内枠
        芝スタートだと外が有利	外枠
        芝スタート系は道悪で逆転する	内枠
        距離が長いほど関係なくなる	関係なくなる
        ダートで内枠は不利	外枠

        """



        """
        ハイペースローペース自体の影響は0.5前後にまとまるよう下げる

        コーナー順位が前（先行）で、ハイペースの場合、noboriをさらに-0.5する（不利条件）
        前でローペースの場合、noboriを+0.2する(ペースによる)有利
        後ろで、ローペースの場合、noboriを-0.1する（作っておけばrankdiffで使える）不利
        後ろでハイの場合、noboriを+0.3する(ハイスローの分を相殺する)有利
        #+だとハイペース、ーだとスローペース
        """

        #最大0.5前後
        baselog["nobori_pace_diff"] = baselog["nobori"] - (baselog["pace_diff"] / 12)

        #ハイペースが不利、だから補正する、最大0.05くらい
        baselog["nobori_pace_diff_grade"] = baselog["nobori_pace_diff"] - (((baselog['race_grade']/70)-1)/8)


        #坂、0.2くらい
        #芝が傷んでくる冬から春には、坂は効く

        baselog["nobori_pace_diff_grade_slope"] = np.where(
            (baselog["season"] == 1) | (baselog["season"] == 4),
            baselog["nobori_pace_diff_grade"] - (baselog["goal_slope"] / 12),
            baselog["nobori_pace_diff_grade"] - (baselog["goal_slope"] / 18)
        )

        #直線の長さ0.2くらい
        baselog["start_range_processed_1"] = (((baselog["start_range"])-360)/150)
        baselog["start_range_processed_1"] = baselog["start_range_processed_1"].apply(
            lambda x: x if x < 0 else x*0.5
        )
        baselog["goal_range_processed_1"] = (((baselog["goal_range"])-360)/150)
        baselog["goal_range_processed_1"] = baselog["start_range_processed_1"].apply(
            lambda x: x*2 if x < 0 else x*0.4
        )

        baselog["nobori_pace_diff_grade_slope_range"] = baselog["nobori_pace_diff_grade_slope"] + (baselog["goal_range_processed_1"]/10)

        """
        ハイスロー、脚質修正
        コーナー順位が前（先行）で、ハイペースの場合、noboriをさらに-0.5する（不利条件）
        前でローペースの場合、noboriを+0.2する(ペースによる)有利
        後ろで、ローペースの場合、noboriを-0.1する（作っておけばrankdiffで使える）不利
        後ろでハイの場合、noboriを+0.3する(ハイスローの分を相殺する)有利
        #+だとハイペース、ーだとスローペース
        """
        # 条件ごとに処理を適用
        baselog["nobori_pace_diff_grade_slope_range_pace"] = np.where(
            ((baselog['race_position'] == 1) | (baselog['race_position'] == 2)) & (baselog["pace_diff"] >= 0),
            baselog["nobori_pace_diff_grade_slope_range"] - (baselog["pace_diff"] / 8),
            
            np.where(
                ((baselog['race_position'] == 1) | (baselog['race_position'] == 2)) & (baselog["pace_diff"] < 0),
                baselog["nobori_pace_diff_grade_slope_range"] - (baselog["pace_diff"] / 16),
                
                np.where(
                    (baselog['race_position'] == 4) & (baselog["pace_diff"] < 0),
                    baselog["nobori_pace_diff_grade_slope_range"] - ((baselog["pace_diff"] / 28) * -1),
                    
                    np.where(
                        ((baselog['race_position'] == 3) | (baselog['race_position'] == 4)) & (baselog["pace_diff"] >= 0),
                        baselog["nobori_pace_diff_grade_slope_range"] - ((baselog["pace_diff"] / 13) * -1),
                        baselog["nobori_pace_diff_grade_slope_range"]  # どの条件にも当てはまらない場合は元の値を保持
                    )
                )
            )
        )

        """
        馬場状態
        普通に不良馬場なら-1.5
        稍重くらいなら-0.3

        ダートなら逆で倍率は変わらず、
        +1.2と+0.6くらいに
        """

        # 条件ごとに適用
        baselog["nobori_pace_diff_grade_slope_range_pace_groundstate"] = np.where(
            ((baselog["ground_state"] == 1) | (baselog["ground_state"] == 3)) & (baselog["race_type"] == 1),
            baselog["nobori_pace_diff_grade_slope_range_pace"] - 1.2,

            np.where(
                (baselog["ground_state"] == 2) & (baselog["race_type"] == 1),
                baselog["nobori_pace_diff_grade_slope_range_pace"] - 0.3,

                np.where(
                    ((baselog["ground_state"] == 1) | (baselog["ground_state"] == 3)) & (baselog["race_type"] == 0),
                    baselog["nobori_pace_diff_grade_slope_range_pace"] + 0.8,

                    np.where(
                        (baselog["ground_state"] == 2) & (baselog["race_type"] == 0),
                        baselog["nobori_pace_diff_grade_slope_range_pace"] + 0.4,
                        
                        # どの条件にも当てはまらない場合は元の値を保持
                        baselog["nobori_pace_diff_grade_slope_range_pace"]
                    )
                )
            )
        )

        #タフが不利、だから補正する
        """
        タフパック	
        ハイペース	タフ
        コーナー種類	ゆるいと遅くならないのでタフ,だけどゆるいほうが早く出る  カーブが緩い、複合だと早いまま入れる

        コーナーR	大きいとタフ
        コーナーの数	少ないほうがタフ
        高低差がある	タフ
        馬場状態、天気が悪い	タフ
        芝によって	タフ
        直線合計/コーナー合計	多いほどタフ
        """

        # -4.5 を行う
        baselog["curve_processed"] = baselog["curve"] - 4.5
        # +の場合は数値を8倍する
        baselog["curve_processed"] = baselog["curve_processed"].apply(
            lambda x: x * 8 if x > 0 else x
        )
        #最大0.12くらい
        baselog["nobori_pace_diff_grade_curve"] = baselog["nobori_pace_diff_grade_slope_range_pace_groundstate"] + (baselog["curve_processed"]/60)






        """"
        "curve_amount"を2以下のとき"curve_R34"を
        "curve_amount"を3以下のとき"curve_R12"/2と"curve_R34"を
        "curve_amount"を4以下のとき"curve_R12"と"curve_R34"を
        "curve_amount"を5以下のとき"curve_R12"と"curve_R34"*3/2を
        "curve_amount"を6以下のとき"curve_R12"と"curve_R34"*2を
        "curve_amount"を7以下のとき"curve_R12"*3/2と"curve_R34"*2を
        "curve_amount"を8以下のとき"curve_R12"*2と"curve_R34"*2を
        """
        #最大0.02*n
        def calculate_nobori_pace_diff(row):
            if row["curve_amount"] == 0:
                return row["nobori_pace_diff_grade_curve"]
            elif row["curve_amount"] <= 2:
                return row["nobori_pace_diff_grade_curve"] + ((row["curve_R34"]-100)/1200)
            elif row["curve_amount"] <= 3:
                return row["nobori_pace_diff_grade_curve"] + ((row["curve_R12"]-100)/1200 / 2 + (row["curve_R34"]-100)/1200)
            elif row["curve_amount"] <= 4:
                return row["nobori_pace_diff_grade_curve"] + ((row["curve_R12"]-100)/1200 + ((row["curve_R34"]-100)/1200))
            elif row["curve_amount"] <= 5:
                return row["nobori_pace_diff_grade_curve"] + ((row["curve_R12"]-100)/1200 + ((row["curve_R34"]-100)/1200) * 3 / 2)
            elif row["curve_amount"] <= 6:
                return row["nobori_pace_diff_grade_curve"] +((row["curve_R12"]-100)/1200 + ((row["curve_R34"]-100)/1200) * 2)
            elif row["curve_amount"] <= 7:
                return row["nobori_pace_diff_grade_curve"] + ((row["curve_R12"]-100)/1200 * 3 / 2 + ((row["curve_R34"]-100)/1200) * 2)
            else:  # curve_amount <= 8
                return row["nobori_pace_diff_grade_curve"] + ((row["curve_R12"]-100)/1200 * 2 +((row["curve_R34"]-100)/1200) * 2)

        baselog["nobori_pace_diff_grade_curveR"] = baselog.apply(calculate_nobori_pace_diff, axis=1)



        #最大0.09くらい
        baselog["nobori_pace_diff_grade_curveR_height_diff"] = baselog["nobori_pace_diff_grade_curveR"] - ((baselog["height_diff"]/30)-0.02)

        #芝の質で一秒くらい違う
        #最大と最小で0,4:-0,4
        baselog = baselog.copy()
        baselog.loc[:, "place_season_condition_type_categori_processed"] = (
            baselog["place_season_condition_type_categori"]
            .replace({5: -0.3, 4: -0.17, 3: 0, 2: 0.17,1: 0.3, -1: -0.18, -2: 0, -3: 0.18,-4:0.3,-10000:0})
        ).astype(float)


        #最大0.5くらい
        baselog["nobori_pace_diff_grade_curveR_height_diff_season"] = baselog["nobori_pace_diff_grade_curveR_height_diff"] - baselog['place_season_condition_type_categori_processed']

        #最大0.05くらい
        baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight"] = baselog["nobori_pace_diff_grade_curveR_height_diff_season"] - (((baselog["straight_total"]/ baselog["course_len"])/10)-0.05)

        # 1600で正規化
        baselog["course_len_processed"] = (baselog["course_len"] / 1800)-1

        # 1600m未満ならそのまま、1600m以上なら緩やかに上昇 
        baselog["course_len_processed_1"] = baselog["course_len_processed"].apply(
            lambda x: x*0.2 if x <= 0 else x*0.1
        )

        #最大0.1くらい
        baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len"] = baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight"] - baselog["course_len_processed_1"]

        """
        内外セット	
        3,4コーナーが急	圧倒的内枠有利になる
        3,4が下り坂	圧倒的内枠有利になる
        スタートからコーナーまでの距離が短い	ポジションが取りづらいため、内枠有利特に偶数有利
        スタートが上りorくだり坂	ポジションが取りづらいため、内枠有利特に偶数有利
        コーナーがきついは内枠	内枠
        向正面上り坂は内枠	内枠
        芝スタートだと外が有利	外枠
        芝スタート系は道悪で逆転する	内枠
        距離が長いほど関係なくなる	関係なくなる
        ダートで内枠は不利	外枠
        """

        #0,01-0.01,内がマイナス
        baselog["umaban_processed"] = baselog["umaban"].apply(
            lambda x: ((x*-1/200)) if x < 4 else (x-8)/1250
        ).astype(float)
        #0-0.005
        baselog.loc[:, "umaban_odd_processed"] = (
            (baselog["umaban_odd"]-1)/200
        ).astype(float)

        # 1600m未満ならそのまま、1600m以上なら緩やかに上昇 
        baselog["course_len_processed_2"] = baselog["course_len_processed"].apply(
            lambda x: x+1 if x <= 0 else x+1
        )

        baselog["umaban_processed_2"] = baselog["umaban_processed"] / baselog["course_len_processed_2"]
        baselog["umaban_odd_processed_2"] = baselog["umaban_odd_processed"] / baselog["course_len_processed_2"]


        #最大0.03くらい、不利が+,ダートは外枠有利
        baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban"] = np.where(
            baselog["race_type"] == 0,
            baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len"] + baselog["umaban_processed_2"],
            baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len"] - baselog["umaban_processed_2"]
        )

        baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds"]= np.where(
            baselog["race_type"] == 0,
            baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban"] - baselog["umaban_odd_processed_2"],
            baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban"] - baselog["umaban_odd_processed_2"]
        )



        #+-0.03急カーブ,フリ評価
        baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve"] = (
            baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds"] - ((baselog["umaban_processed_2"]*(baselog["curve_processed"]/4))*-1)
        )


        #+-0.03カーブ下り坂,フリ評価
        baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope"] = (
            baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve"] - ((baselog["umaban_processed_2"]*(baselog["last_curve_slope"]/2))*-1)
        )



        baselog["start_range_processed"] = (((baselog["start_range"])-360)/150)
        baselog["start_range_processed"] = baselog["start_range_processed"].apply(
            lambda x: x if x < 0 else x*0.5
        )

        #+-0.06,スタートからコーナー
        baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range"] = (
            baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope"] - ((baselog["umaban_processed_2"]*(baselog["start_range_processed"]))*-1)- ((baselog["umaban_odd_processed_2"]*(baselog["start_range_processed"]))*-1)
        )



        baselog["start_slope_abs"] = baselog["start_slope"].abs()
        baselog["start_slope_abs_processed"] = baselog["start_slope_abs"]/4

        #+-0.06,スタートからコーナー、坂,上り下り両方
        baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] = (
            baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range"] - ((baselog["umaban_processed_2"]*(baselog["start_slope_abs_processed"]))*-1)- ((baselog["umaban_odd_processed_2"]*( baselog["start_slope_abs_processed"]))*-1)
        )




        #最大0.3*nコーナーがきついは内枠
        def calculate_nobori_pace_diff_2(row):
            if row["curve_amount"] == 0:
                return row["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"]
            elif row["curve_amount"] <= 2:
                return row["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] - (row["umaban_processed_2"]*(((row["curve_R34"]-100)/450)*-1))
            elif row["curve_amount"] <= 3:
                return row["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] - (row["umaban_processed_2"]*(((row["curve_R12"]-100)/450 / 2 + (row["curve_R34"]-100)/450)*-1))
            elif row["curve_amount"] <= 4:
                return row["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] - (row["umaban_processed_2"]*(((row["curve_R12"]-100)/450 + ((row["curve_R34"]-100)/450))*-1))
            elif row["curve_amount"] <= 5:
                return row["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] - (row["umaban_processed_2"]*(((row["curve_R12"]-100)/450 + ((row["curve_R34"]-100)/450) * 3 / 2)*-1))
            elif row["curve_amount"] <= 6:
                return row["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] -((row["umaban_processed_2"]*((row["curve_R12"]-100)/450 + ((row["curve_R34"]-100)/450) * 2)*-1))
            elif row["curve_amount"] <= 7:
                return row["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] - (row["umaban_processed_2"]*(((row["curve_R12"]-100)/450 * 3 / 2 + ((row["curve_R34"]-100)/450) * 2)*-1))
            else:  # curve_amount <= 8
                return row["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope"] - ((row["umaban_processed_2"]*((row["curve_R12"]-100)/450 * 2 +((row["curve_R34"]-100)/450) * 2)*-1))


        baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner"] = baselog.apply(calculate_nobori_pace_diff_2, axis=1)


        #最大1*向正面
        baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont"] = (
            baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner"] - ((baselog["umaban_processed_2"]*(baselog["flont_slope"]/4)))
        )


        #芝スタートかつ良馬場、芝スタートかつ良以外、どっちでもない場合,外評価

        condition = (baselog["start_point"] == 2) & (baselog["ground_state"] == 0)

        baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont_point"] = np.where(
            condition,
            baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont"] - (baselog["umaban_processed_2"] * -1),
            baselog["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont"] - baselog["umaban_processed_2"]
        )






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

        "nobori_pace_diff_grade_curveR_height_diff_season"のnoboriが
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
        baselog["distance_place_type_ground_state"] = (baselog["course_type"].astype(str)+ baselog["ground_state"].astype(str)).astype(int)   
        
        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        baselog["distance_place_type_ground_state_grade"] = (baselog["distance_place_type_ground_state"].astype(str)+ baselog["race_grade"].astype(str)).astype(int)   
        


        baselog_old = (
            self.all_population.merge(
                self.all_race_info[["race_id", "course_len", "race_type","race_grade","ground_state", "weather","place","season","course_type"]], on="race_id"
            )
        )

        df_old = (
            baselog_old
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
            baselog = baselog.merge(df2_subset, on=original_col, how='left')  # dfにマージ
            
        baselog["nobori_diff"] = baselog["distance_place_type_ground_state_grade_nobori_encoded"] - baselog["nobori"]

        baselog = baselog.copy()
        def calculate_rush_type(row):
            if row["nobori"] < 33 and row["race_type"] == 1:
                return -6
            if row["nobori_pace_diff_grade_curveR_height_diff_season"] < 33 and row["race_type"] == 1:
                return -6
            if row["nobori_pace_diff_grade_curveR_height_diff_season"] < 33.9 and row["race_type"] == 1:
                return -4
            if row["nobori_pace_diff_grade_curveR_height_diff_season"] < 34.5 and row["race_type"] == 1:
                return -2.5
            if row["nobori"] < 34.5 and row["race_type"] == 1:
                return -2

            if row["nobori"] < 33.5 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -6
            if row["nobori_pace_diff_grade_curveR_height_diff_season"] < 33.5 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -6
            if row["nobori_pace_diff_grade_curveR_height_diff_season"] < 34.4 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -4
            if row["nobori_pace_diff_grade_curveR_height_diff_season"] < 35 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -2.5
            if row["nobori"] < 35 and row["race_type"] == 0 and row["course_len"] < 1600:
                return -2

            if row["nobori"] < 34 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -6
            if row["nobori_pace_diff_grade_curveR_height_diff_season"] < 34 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -6
            if row["nobori_pace_diff_grade_curveR_height_diff_season"] < 34.9 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -4
            if row["nobori_pace_diff_grade_curveR_height_diff_season"] < 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -2.5
            if row["nobori"] < 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600:
                return -2

            if row["nobori_diff"] >= 1:
                return -4
            if row["nobori_diff"] >= 0.6:
                return -2.5

            if row["nobori_pace_diff_grade_curveR_height_diff_season"] >= 35.6 and row["race_type"] == 1 and row["rank"] <= 6:
                return 4
            if row["nobori_pace_diff_grade_curveR_height_diff_season"] >= 34.5 and row["race_type"] == 1 and row["rank"] <= 6:
                return 2.5
            if row["nobori"] >= 34.5 and row["race_type"] == 1 and row["rank"] <= 6:
                return 2.5

            if row["nobori_pace_diff_grade_curveR_height_diff_season"] >= 36.1 and row["race_type"] == 0 and row["course_len"] < 1600 and row["rank"] <= 6:
                return 4
            if row["nobori_pace_diff_grade_curveR_height_diff_season"] >= 35 and row["race_type"] == 0 and row["course_len"] < 1600 and row["rank"] <= 6:
                return 2.5
            if row["nobori"] >= 35 and row["race_type"] == 0 and row["course_len"] < 1600 and row["rank"] <= 6:
                return 2.5

            if row["nobori_pace_diff_grade_curveR_height_diff_season"] >= 36.6 and row["race_type"] == 0 and row["course_len"] >= 1600 and row["rank"] <= 6:
                return 4
            if row["nobori_pace_diff_grade_curveR_height_diff_season"] >= 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600 and row["rank"] <= 6:
                return 2.5
            if row["nobori"] >= 35.5 and row["race_type"] == 0 and row["course_len"] >= 1600 and row["rank"] <= 6:
                return 2.5

            return 0

        # DataFrame に適用
        baselog["rush_type"] = baselog.apply(calculate_rush_type, axis=1)



        # if baselog["nobori"] < 33 and baselog["race_type"] == 1:
        #     baselog["rush_type"] = -6
        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] < 33 and baselog["race_type"] == 1:
        #     baselog["rush_type"] = -6
        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] < 33.9 and baselog["race_type"] == 1:
        #     baselog["rush_type"] = -4
        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] < 34.5 and baselog["race_type"] == 1:
        #     baselog["rush_type"] = -2.5
        # if baselog["nobori"] < 34.5 and baselog["race_type"] == 1:
        #     baselog["rush_type"] = -2

        # if baselog["nobori"] < 33.5 and baselog["race_type"] == 0 and baselog["course_len"] < 1600:
        #     baselog["rush_type"] = -6
        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] < 33.5 and baselog["race_type"] == 0 and baselog["course_len"] < 1600:
        #     baselog["rush_type"] = -6
        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] < 34.4 and baselog["race_type"] == 0 and baselog["course_len"] < 1600:
        #     baselog["rush_type"] = -4
        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] < 35 and baselog["race_type"] == 0 and baselog["course_len"] < 1600:
        #     baselog["rush_type"] = -2.5
        # if baselog["nobori"] < 35 and baselog["race_type"] == 0 and baselog["course_len"] < 1600:
        #     baselog["rush_type"] = -2

        # if baselog["nobori"] < 34 and baselog["race_type"] == 0 and baselog["course_len"] >= 1600:
        #     baselog["rush_type"] = -6
        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] < 34 and baselog["race_type"] == 0 and baselog["course_len"] >= 1600:
        #     baselog["rush_type"] = -6
        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] < 34.9 and baselog["race_type"] == 0 and baselog["course_len"] >= 1600:
        #     baselog["rush_type"] = -4
        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] < 35.5 and baselog["race_type"] == 0 and baselog["course_len"] >= 1600:
        #     baselog["rush_type"] = -2.5
        # if baselog["nobori"] < 35.5 and baselog["race_type"] == 0 and baselog["course_len"] >= 1600:
        #     baselog["rush_type"] = -2


        # if baselog["nobori_diff"] >= 1.2:
        #     baselog["rush_type"] = -4
        # if baselog["nobori_diff"] >= 0.8:
        #     baselog["rush_type"] = -2.5

        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] >= 35.6 and baselog["race_type"] == 1 and baselog["rank"] <= 6:
        #     baselog["rush_type"] = 4
        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] >= 34.5 and baselog["race_type"] == 1 and baselog["rank"] <= 6:
        #     baselog["rush_type"] = 2.5
        # if baselog["nobori"] >= 34.5 and baselog["race_type"] == 1 baselog["rank"] <= 6:
        #     baselog["rush_type"] = 2.5

        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] >= 36.1 and baselog["race_type"] == 0 and baselog["course_len"] < 1600 and baselog["rank"] <= 6:
        #     baselog["rush_type"] = 4
        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] >= 35 and baselog["race_type"] == 0 and baselog["course_len"] < 1600 baselog["rank"] <= 6:
        #     baselog["rush_type"] = 2.5
        # if baselog["nobori"] >= 35 and baselog["race_type"] == 0 and baselog["course_len"] < 1600 and baselog["rank"] <= 6:
        #     baselog["rush_type"] = 2.5

        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] >= 36.6 and baselog["race_type"] == 0 and baselog["course_len"] >=  1600 and baselog["rank"] <= 6:
        #     baselog["rush_type"] = 4
        # if baselog["nobori_pace_diff_grade_curveR_height_diff_season"] >= 35.5 and baselog["race_type"] == 0 and baselog["course_len"] >=  1600 and baselog["rank"] <= 6:
        #     baselog["rush_type"] = 2.5
        # if baselog["nobori"] >= 35.5 and baselog["race_type"] == 0 and baselog["course_len"] >= 1600 and baselog["rank"] <= 6:
        #     baselog["rush_type"] = 2.5
        # else:
        #     baselog["rush_type"] = 0

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

        #最大200前後
        baselog["course_len_pace_diff"] = baselog["course_len"] + (baselog["pace_diff"] * 70)

        #グレード100前後
        baselog["course_len_diff_grade"] = baselog["course_len_pace_diff"] + (((baselog['race_grade']/70)-1)*200)

        #100前後
        baselog["course_len_diff_grade_slope"] = np.where(
            (baselog["season"] == 1) | (baselog["season"] == 4),
            baselog["course_len_diff_grade"] + (baselog["goal_slope"] * 30),
            baselog["course_len_diff_grade"] + (baselog["goal_slope"] * 13)
        )

        #最初の直線の長さ、長いほどきつい、50前後くらい
        baselog["start_range_processed_1"] = (((baselog["start_range"])-360)/150)
        baselog["start_range_processed_1"] = baselog["start_range_processed_1"].apply(
            lambda x: x if x < 0 else x*0.5
        )

        baselog["start_range_processed_course"] = baselog["start_range_processed_1"]*30
        baselog["course_len_pace_diff_grade_slope_range"] = baselog["course_len_diff_grade_slope"] + (baselog["start_range_processed_course"])

        # 条件ごとに処理を適用
        baselog["course_len_diff_grade_slope_range_pace"] = np.where(
            ((baselog['race_position'] == 1) | (baselog['race_position'] == 2)) & (baselog["pace_diff"] >= 0),
            baselog["course_len_pace_diff_grade_slope_range"] + ((baselog["pace_diff"] / 8) * 100),
            
            np.where(
                ((baselog['race_position'] == 1) | (baselog['race_position'] == 2)) & (baselog["pace_diff"] < 0),
                baselog["course_len_pace_diff_grade_slope_range"] + ((baselog["pace_diff"] / 16)*100),
                
                np.where(
                    (baselog['race_position'] == 4) & (baselog["pace_diff"] < 0),
                    baselog["course_len_pace_diff_grade_slope_range"] + ((baselog["pace_diff"] / 28) * -100),
                    
                    np.where(
                        ((baselog['race_position'] == 3) | (baselog['race_position'] == 4)) & (baselog["pace_diff"] >= 0),
                        baselog["course_len_pace_diff_grade_slope_range"] + ((baselog["pace_diff"] / 13) * -100),
                        baselog["course_len_pace_diff_grade_slope_range"]  # どの条件にも当てはまらない場合は元の値を保持
                    )
                )
            )
        )


        # # -4.5 を行う
        # baselog["curve_processed"] = baselog["curve"] - 4.5
        # # +の場合は数値を8倍する
        # baselog["curve_processed"] = baselog["curve_processed"].apply(
        #     lambda x: x * 8 if x > 0 else x
        # )
        #12コーナーがきついと、ゆるい、-
        baselog["course_len_diff_grade_slope_range_pace_12curve"] = baselog["course_len_diff_grade_slope_range_pace"] + (baselog["curve_processed"] * 25)

        #向正面上り坂、ゆるい、-
        baselog["course_len_diff_grade_slope_range_pace_12curve_front"] = baselog["course_len_diff_grade_slope_range_pace_12curve"] - (baselog["flont_slope"] * 25)



        #最大0.02*n
        def calculate_course_len_pace_diff(row):
            if row["curve_amount"] == 0:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"]
            elif row["curve_amount"] <= 2:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R34"]-100)/3)
            elif row["curve_amount"] <= 3:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 / 2 + (row["curve_R34"]-100)/4)
            elif row["curve_amount"] <= 4:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 + ((row["curve_R34"]-100)/4))
            elif row["curve_amount"] <= 5:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 + ((row["curve_R34"]-100)/4) * 3 / 2)
            elif row["curve_amount"] <= 6:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] +((row["curve_R12"]-100)/4 + ((row["curve_R34"]-100)/4) * 2)
            elif row["curve_amount"] <= 7:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 * 3 / 2 + ((row["curve_R34"]-100)/4) * 2)
            else:  # curve_amount <= 8
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 * 2 +((row["curve_R34"]-100)/4) * 2)

        baselog["course_len_diff_grade_slope_range_pace_12curve_front_R"] = baselog.apply(calculate_course_len_pace_diff, axis=1)

        #最大0.09くらい
        baselog["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] = baselog["course_len_diff_grade_slope_range_pace_12curve_front_R"] + (baselog["height_diff"]/50)


        # 条件ごとに適用
        baselog["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate"] = np.where(
            ((baselog["ground_state"] == 1) | (baselog["ground_state"] == 3)) & (baselog["race_type"] == 1),
            baselog["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] + 400,

            np.where(
                (baselog["ground_state"] == 2) & (baselog["race_type"] == 1),
                baselog["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] + 120,

                np.where(
                    ((baselog["ground_state"] == 1) | (baselog["ground_state"] == 3)) & (baselog["race_type"] == 0),
                    baselog["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] - 100,

                    np.where(
                        (baselog["ground_state"] == 2) & (baselog["race_type"] == 0),
                        baselog["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] - 50,
                        
                        # どの条件にも当てはまらない場合は元の値を保持
                        baselog["course_len_diff_grade_slope_range_pace_12curve_front_R_height"]
                    )
                )
            )
        )

        baselog = baselog.copy()
        baselog.loc[:, "place_season_condition_type_categori_processed"] = (
            baselog["place_season_condition_type_categori"]
            .replace({5: -0.3, 4: -0.17, 3: 0, 2: 0.17,1: 0.3, -1: -0.18, -2: 0, -3: 0.18,-4:0.3,-10000:0})
        ).astype(float)

        baselog["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place"] = (
            baselog["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate"] + (baselog["place_season_condition_type_categori_processed"]*-500)
            )

        #最大0.05くらい
        baselog["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight"] = (
            baselog["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place"] - (((baselog["straight_total"]/ baselog["course_len"])-0.5)*400)
            )











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

        #最大前後0.2、ハイペースは+
        baselog["rank_diff_pace_diff"] = baselog["rank_diff"] + (baselog["pace_diff"] /25)


        # 1600で正規化
        baselog["course_len_processed_rd"] = (baselog["course_len"] / 1600) - 1

        # 1600m未満ならそのまま、1600m以上なら緩やかに上昇 
        baselog["course_len_processed_rd"] = baselog["course_len_processed_rd"].apply(
            lambda x: x/2 if x <= 0 else x/4
        )

        baselog["rank_diff_pace_course_len"] = baselog["rank_diff_pace_diff"] + baselog["course_len_processed_rd"]



        # 条件ごとに適用,馬場状態が悪い状態ほど評価（-）
        baselog["rank_diff_pace_course_len_ground_state"] = np.where(
            ((baselog["ground_state"] == 1) | (baselog["ground_state"] == 3)) & (baselog["race_type"] == 1),
            baselog["rank_diff_pace_course_len"] - 0.3,

            np.where(
                (baselog["ground_state"] == 2) & (baselog["race_type"] == 1),
                baselog["rank_diff_pace_course_len"] - 0.12,

                np.where(
                    ((baselog["ground_state"] == 1) | (baselog["ground_state"] == 3)) & (baselog["race_type"] == 0),
                    baselog["rank_diff_pace_course_len"] + 0.1,

                    np.where(
                        (baselog["ground_state"] == 2) & (baselog["race_type"] == 0),
                        baselog["rank_diff_pace_course_len"] + 0.05,
                        
                        # どの条件にも当てはまらない場合は元の値を保持
                        baselog["rank_diff_pace_course_len"]
                    )
                )
            )
        )




        #0,01-0.01,内がプラス(内枠が有利を受けるとしたら、rank_diffは+にして、有利ポジはマイナス補正)
        baselog["umaban_rank_diff_processed"] = baselog["umaban"].apply(
            lambda x: ((x*-1.5)+1.5) if x < 4 else ((x-8)/1.5)-1
        ).astype(float)
        #0,-0.1,-0.3,-0.36,-0.3,-0.23,（-1/10のとき）
        baselog["umaban_rank_diff_processed"] = baselog["umaban_rank_diff_processed"] * (1/44)
        #0 , -0.05
        #1（奇数）または 0（偶数）,偶数が有利
        baselog.loc[:, "umaban_odd_rank_diff_processed"] = (
            (baselog["umaban_odd"]-1)/20
        ).astype(float)

        #rdが-0.25,,0.25が0.5に
        baselog["umaban_rank_diff_processed_2"] = baselog["umaban_rank_diff_processed"] / ((baselog["course_len_processed_rd"]*2) + 1)
        baselog["umaban_odd_rank_diff_processed_2"] = baselog["umaban_odd_rank_diff_processed"] / ((baselog["course_len_processed_rd"]*2)+1)

        #不利が-,ダートは外枠有利,0.06
        baselog["rank_diff_pace_course_len_ground_state_type"] = np.where(
            baselog["race_type"] == 0,
            baselog["rank_diff_pace_course_len_ground_state"] + (((baselog["umaban"]-8)/150)/ ((baselog["course_len_processed_rd"]*2) + 1)),
            baselog["rank_diff_pace_course_len_ground_state"] - (baselog["umaban_rank_diff_processed_2"]/2)
        )

        baselog["rank_diff_pace_course_len_ground_state_type_odd"]= np.where(
            baselog["race_type"] == 0,
            baselog["rank_diff_pace_course_len_ground_state_type"] - baselog["umaban_odd_rank_diff_processed_2"],
            baselog["rank_diff_pace_course_len_ground_state_type"] - baselog["umaban_odd_rank_diff_processed_2"]
        )

        #last急カーブ,フリ評価
        baselog["rank_diff_pace_course_len_ground_state_type_odd_curve"] = (
            baselog["rank_diff_pace_course_len_ground_state_type_odd"] + ((baselog["umaban_rank_diff_processed_2"]*((baselog["curve_processed"]/4))))
        )

        #3カーブ下り坂,フリ評価
        baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope"] = (
            baselog["rank_diff_pace_course_len_ground_state_type_odd_curve"] + ((baselog["umaban_rank_diff_processed_2"]*(baselog["last_curve_slope"]/3)))
        )

        #スタートからコーナー
        baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start"] = (
            baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope"] - ((baselog["umaban_rank_diff_processed_2"]*(baselog["start_range_processed"]))*-1/1.2)- ((baselog["umaban_odd_rank_diff_processed_2"]*(baselog["start_range_processed"]))*-1/1.2)
        )


        #+-0.06,スタートからコーナー、坂,上り下り両方
        baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] = (
            baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start"] - ((baselog["umaban_rank_diff_processed_2"]*(baselog["start_slope_abs_processed"]))*-1)- ((baselog["umaban_odd_rank_diff_processed_2"]*( baselog["start_slope_abs_processed"]))*-1)
        )

        #最大0.3*nコーナーがきついは内枠
        def calculate_rank_diff_pace_diff_2(row):
            if row["curve_amount"] == 0:
                return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"]
            elif row["curve_amount"] <= 2:
                return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] - (row["umaban_rank_diff_processed_2"]*(((row["curve_R34"]-100)/120)*-1))
            elif row["curve_amount"] <= 3:
                return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] - (row["umaban_rank_diff_processed_2"]*(((row["curve_R12"]-100)/120 / 2 + (row["curve_R34"]-100)/120)*-1))
            elif row["curve_amount"] <= 4:
                return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] - (row["umaban_rank_diff_processed_2"]*(((row["curve_R12"]-100)/120 + ((row["curve_R34"]-100)/120))*-1))
            elif row["curve_amount"] <= 5:
                return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] - (row["umaban_rank_diff_processed_2"]*(((row["curve_R12"]-100)/120 + ((row["curve_R34"]-100)/120) * 3 / 2)*-1))
            elif row["curve_amount"] <= 6:
                return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] -((row["umaban_rank_diff_processed_2"]*((row["curve_R12"]-100)/120 + ((row["curve_R34"]-100)/120) * 2)*-1))
            elif row["curve_amount"] <= 7:
                return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] - (row["umaban_rank_diff_processed_2"]*(((row["curve_R12"]-100)/120 * 3 / 2 + ((row["curve_R34"]-100)/120) * 2)*-1))
            else:  # curve_amount <= 8
                return row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope"] - ((row["umaban_rank_diff_processed_2"]*((row["curve_R12"]-100)/120 * 2 +((row["curve_R34"]-100)/120) * 2)*-1))



        baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff"] = baselog.apply(calculate_rank_diff_pace_diff_2, axis=1)


        #最大1*向正面
        baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont"] = (
            baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff"] - ((baselog["umaban_rank_diff_processed_2"]*(baselog["flont_slope"]/8)))
        )

        #芝スタートかつ良馬場、芝スタートかつ良以外、どっちでもない場合,外評価

        condition_rank_diff = (baselog["start_point"] == 2) & (baselog["ground_state"] == 0)

        baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] = np.where(
            condition_rank_diff,
            baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont"] - (baselog["umaban_rank_diff_processed_2"] * -1.5),
            baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont"] - baselog["umaban_rank_diff_processed_2"]
        )




        # シーズンを追加
        baselog["distance_place_type_ground_state_grade_season"] = (baselog["distance_place_type_ground_state_grade"].astype(str)+ baselog["season"].astype(str)).astype(int)   

        # 距離/競馬場/タイプレースランク/直線/天気/馬場状態
        df_old["distance_place_type_ground_state_grade_season"] = (df_old["distance_place_type_ground_state_grade"].astype(str)+ df_old["season"].astype(str)).astype(int)   
        
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
            baselog = baselog.merge(df2_subset, on=original_col, how='left')  # dfにマージ
            

        baselog['no1_time'] = baselog.apply(
            lambda row: row['time'] if row['rank'] == 1 else row['time'] - (row['rank_diff']-1),
            axis=1
        )

        baselog["time_class"] = baselog["distance_place_type_ground_state_grade_season_time_encoded"] - baselog['no1_time']

        baselog["time_class_abs"] = baselog["time_class"].abs()

        # if baselog["course_len"] < 1600:
        #     if baselog["race_grade"] < 80 and baselog["time_class_abs"] >0.7 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if baselog["race_grade"] < 80 and baselog["time_class_abs"] >0.7 and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 80 <= baselog["race_grade"] <= 87 and baselog["time_class_abs"] >1 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 80 <= baselog["race_grade"] <= 87 and baselog["time_class_abs"] >1 and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 88 <= baselog["race_grade"] and baselog["time_class_abs"] >1.2 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 88 <= baselog["race_grade"]and baselog["time_class_abs"] >1.2 and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        # if 1600 <= baselog["course_len"] and baselog["race_type"] == 1:
        #     if baselog["race_grade"] < 80 and baselog["time_class_abs"] >1.2 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if baselog["race_grade"] < 80 and baselog["time_class_abs"]>1.2  and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 80 <= baselog["race_grade"] <= 87 and baselog["time_class_abs"] >1.5 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 80 <= baselog["race_grade"] <= 87 and baselog["time_class_abs"] >1.5 and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 88 <= baselog["race_grade"] and baselog["time_class_abs"]  >1.7 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 88 <= baselog["race_grade"]and baselog["time_class_abs"] >1.7 and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )

        # if 1600 <= baselog["course_len"] and baselog["race_type"] == 0:
        #     if baselog["race_grade"] < 80 and baselog["time_class_abs"] >1 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if baselog["race_grade"] < 80 and baselog["time_class_abs"]>1  and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 80 <= baselog["race_grade"] <= 87 and baselog["time_class_abs"] >1.3 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 80 <= baselog["race_grade"] <= 87 and baselog["time_class_abs"] >1.3 and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 88 <= baselog["race_grade"] and baselog["time_class_abs"]  >1.5 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 88 <= baselog["race_grade"]and baselog["time_class_abs"] >1.5 and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )


        # if 2400 <= baselog["course_len"] and baselog["race_type"] == 1:
        #     if baselog["race_grade"] < 80 and baselog["time_class_abs"] >1.8 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if baselog["race_grade"] < 80 and baselog["time_class_abs"]>1.8  and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 80 <= baselog["race_grade"] <= 87 and baselog["time_class_abs"] >2.1 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 80 <= baselog["race_grade"] <= 87 and baselog["time_class_abs"] >2.1 and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 88 <= baselog["race_grade"] and baselog["time_class_abs"]  >2.3 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 88 <= baselog["race_grade"]and baselog["time_class_abs"] >2.3 and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )

        # if 2400 <= baselog["course_len"] and baselog["race_type"] == 0:
        #     if baselog["race_grade"] < 80 and baselog["time_class_abs"] >1.5 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if baselog["race_grade"] < 80 and baselog["time_class_abs"]>1.5  and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 80 <= baselog["race_grade"] <= 87 and baselog["time_class_abs"] >1.8 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 80 <= baselog["race_grade"] <= 87 and baselog["time_class_abs"] >1.8 and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )
        #     if 88 <= baselog["race_grade"] and baselog["time_class_abs"]  >2 and baselog["time_class"] > 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] - 0.15
        #         )           
        #     if 88 <= baselog["race_grade"]and baselog["time_class_abs"] >2 and baselog["time_class"] < 0:
        #         baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_class"] =(
        #             baselog["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"] + 0.15
        #         )


        def calculate_course_len_pace_diff(row):
            # 初期値として元のrank_diffを設定
            result = row["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point"]
            
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
            
        baselog["rank_diff_correction"] = baselog.apply(calculate_course_len_pace_diff, axis=1)


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
        baselog = baselog.copy()
        baselog.loc[:, "place_season_condition_type_categori_processed_rank_diff"] = (
            baselog["place_season_condition_type_categori"]
            .replace({5: -0.09, 4: -0.05, 3: 0, 2: 0.05,1: 0.09, -1: -0.08, -2: 0, -3: 0.08,-4:0.11,-10000:0})
        ).astype(float)

        # #向いていない場合は-
        # if baselog['race_position'] == 1 or 2:
        #     if baselog['weather'] == 1 and baselog['ground_state'] == 2:
        #         baselog["rank_diff_correction_position"] = (
        #             baselog["rank_diff_correction"] - (baselog["pace_diff"]/12) - baselog["place_season_condition_type_categori_processed_rank_diff"] - baselog["place_season_condition_type_categori_processed_rank_diff"] - (baselog["start_slope_abs_processed"]*0.1) - baselog["start_range_processed_1"]- baselog["umaban_odd_rank_diff_processed"] - (((baselog['race_grade']/70)-1)/2) + (baselog["last_curve_slope"]/15) - 0.12
        #         )
        #     if baselog['weather'] != 1 or baselog['ground_state'] == 1 or baselog['ground_state'] == 3:
        #         baselog["rank_diff_correction_position"] = (
        #             baselog["rank_diff_correction"] - (baselog["pace_diff"]/12) - baselog["place_season_condition_type_categori_processed_rank_diff"] - baselog["place_season_condition_type_categori_processed_rank_diff"] - (baselog["start_slope_abs_processed"]*0.1) - baselog["start_range_processed_1"]- baselog["umaban_odd_rank_diff_processed"] - (((baselog['race_grade']/70)-1)/2) + (baselog["last_curve_slope"]/15) + 0.12
        #         )
        #     baselog["rank_diff_correction_position"] = (
        #         baselog["rank_diff_correction"] - (baselog["pace_diff"]/12) - baselog["place_season_condition_type_categori_processed_rank_diff"] - baselog["place_season_condition_type_categori_processed_rank_diff"] - (baselog["start_slope_abs_processed"]*0.1) - baselog["start_range_processed_1"]- baselog["umaban_odd_rank_diff_processed"] - (((baselog['race_grade']/70)-1)/2) + (baselog["last_curve_slope"]/15)
        #     )

        # if baselog['race_position'] == 3 or 4:
        #     if baselog['weather'] == 1 and baselog['ground_state'] == 2:
        #         baselog["rank_diff_correction_position"] = (
        #             baselog["rank_diff_correction"] + (baselog["pace_diff"]/12) + baselog["place_season_condition_type_categori_processed_rank_diff"] + baselog["place_season_condition_type_categori_processed_rank_diff"] + (baselog["start_slope_abs_processed"]*0.1) + baselog["start_range_processed_1"]+ baselog["umaban_odd_rank_diff_processed"] + (((baselog['race_grade']/70)-1)/2) - (baselog["last_curve_slope"]/15) + 0.12
        #         )
        #     if baselog['weather'] != 1 or baselog['ground_state'] == 1 or baselog['ground_state'] == 3:
        #         baselog["rank_diff_correction_position"] = (
        #             baselog["rank_diff_correction"] + (baselog["pace_diff"]/12) + baselog["place_season_condition_type_categori_processed_rank_diff"] + baselog["place_season_condition_type_categori_processed_rank_diff"] + (baselog["start_slope_abs_processed"]*0.1) + baselog["start_range_processed_1"]+ baselog["umaban_odd_rank_diff_processed"] + (((baselog['race_grade']/70)-1)/2) - (baselog["last_curve_slope"]/15) - 0.12
        #         )
        #     baselog["rank_diff_correction_position"] = (
        #         baselog["rank_diff_correction"] + (baselog["pace_diff"]/12) + baselog["place_season_condition_type_categori_processed_rank_diff"] + baselog["place_season_condition_type_categori_processed_rank_diff"] + (baselog["start_slope_abs_processed"]*0.1) + baselog["start_range_processed_1"]+ baselog["umaban_odd_rank_diff_processed"] + (((baselog['race_grade']/70)-1)/2) - (baselog["last_curve_slope"]/15)
        #     )

        baselog["goal_range_100_processed"] = baselog["goal_range_100"] - 3.6
        # # プラスの値をすべて 0 に変換
        # baselog["goal_range_100_processed"] = baselog["goal_range_100_processed"].clip(upper=0)
        baselog.loc[baselog["goal_range_100_processed"] > 0, "goal_range_100_processed"] *= 0.7


        def calculate_rank_diff_correction_position(row):
            # 共通の部分の計算
            rank_diff_correction_position = (
                - (row["pace_diff"] / 12)
                - row["place_season_condition_type_categori_processed_rank_diff"]
                - (row["start_slope_abs_processed"] * 0.06)
                - row["start_range_processed_1"]
                - row["umaban_odd_rank_diff_processed"]
                - (((row['race_grade'] / 70) - 1) / 3)
                + (row["last_curve_slope"] / 15)
                - (row["curve_processed"] / 40)
                - (row["goal_range_processed_1"] / 15)   
                - (row["goal_slope"] / 18)

            )

            # race_positionが1または2の場合の処理
            if row['race_position'] in [1, 2]:
                if row['weather'] == 1 and row['ground_state'] == 2:
                    row["rank_diff_correction_position"] = row["rank_diff_correction"] + rank_diff_correction_position - 0.12
                elif row['weather'] != 1 or row['ground_state'] in [1, 3]:
                    row["rank_diff_correction_position"] = row["rank_diff_correction"] + rank_diff_correction_position + 0.12
                else:
                    row["rank_diff_correction_position"] = row["rank_diff_correction"] + rank_diff_correction_position

            # race_positionが3または4の場合の処理
            elif row['race_position'] in [3, 4]:
                if row['weather'] == 1 and row['ground_state'] == 2:
                    row["rank_diff_correction_position"] = row["rank_diff_correction"] - rank_diff_correction_position + 0.12
                elif row['weather'] != 1 or row['ground_state'] in [1, 3]:
                    row["rank_diff_correction_position"] = row["rank_diff_correction"] - rank_diff_correction_position - 0.12
                else:
                    row["rank_diff_correction_position"] = row["rank_diff_correction"] - rank_diff_correction_position

            return row

        # DataFrameに適用
        baselog = baselog.apply(calculate_rank_diff_correction_position, axis=1)


        """
        ・持続
        コーナー回数が少ない、コーナーゆるい、ラストの直線が短い、ラストの下り坂、ミドルorハイ、外枠、高低差+
        ・瞬発
        コーナー回数が多い、コーナーきつい、ラストの直線が長い、ラストの上り坂or平坦、スローペース、内枠、高低差なしがいい
        瞬発系はタフなレースきつい
        """

        # if baselog["rush_type"] < 0:
        #     baselog["rank_diff_correction_rush"] =(
        #         baselog["rank_diff_correction"] 
        #         - (((baselog["rush_type"]+0.1)/30)*((baselog["curve_amount"]-4)/8)) 
        #         - (((baselog["rush_type"]+0.1)/30)*(baselog["curve_processed"] /-4))
        #         - (((baselog["rush_type"]+0.1)/30)*(baselog["goal_range_processed_1"] /1.2))
        #         - (((baselog["rush_type"]+0.1)/30)*(baselog["goal_slope"] /4))
        #         - (((baselog["rush_type"]+0.1)/30)*((baselog["pace_diff"]+0.6) / -3))
        #         - (((baselog["rush_type"]+0.1)/30)*((baselog["umaban_rank_diff_processed_2"]) * -10))
        #         - (((baselog["rush_type"]+0.1)/30)*((baselog["height_diff"]/-2)))
        #     )
        # if baselog["rush_type"] >= 0:
        #     baselog["rank_diff_correction_rush"] =(
        #         baselog["rank_diff_correction"] 
        #         - (((baselog["rush_type"]+0.1)/30)*((baselog["curve_amount"]-4)/8)) 
        #         - (((baselog["rush_type"]+0.1)/30)*(baselog["curve_processed"] /-4))
        #         - (((baselog["rush_type"]+0.1)/30)*(baselog["goal_range_processed_1"] /1.2))
        #         - (((baselog["rush_type"]+0.1)/30)*(baselog["goal_slope"] /4))
        #         - (((baselog["rush_type"]+0.1)/30)*((baselog["pace_diff"]+0.6) / -3))
        #         - (((baselog["rush_type"]+0.1)/30)*((baselog["umaban_rank_diff_processed_2"]) * -10))
        #         - (((baselog["rush_type"]+0.1)/30)*((baselog["height_diff"]/-2)))
        #     )
        baselog = baselog.copy() 

        # 共通の計算
        correction_factor = (baselog["rush_type"] + 0.1) / 30

        # `rank_diff_correction_rush` の計算
        baselog.loc[:, "rank_diff_correction_rush"] = (
            baselog["rank_diff_correction"]
            - (correction_factor * ((baselog["curve_amount"] - 4) / 10))
            - (correction_factor * (baselog["curve_processed"] / -6))
            - (correction_factor * (baselog["goal_range_processed_1"] / 1.2))
            - (correction_factor * (baselog["goal_slope"] / 4))
            - (correction_factor * ((baselog["pace_diff"] + 0.6) / -3))
            - (correction_factor * (baselog["umaban_rank_diff_processed_2"] * -10))
            - (correction_factor * (baselog["height_diff"] / -2)) 
        )
        # `rank_diff_correction_rush` の計算
        baselog.loc[:, "rank_diff_correction_position_rush"] = (
            baselog["rank_diff_correction_position"]
            - (correction_factor * ((baselog["curve_amount"] - 4) / 10))
            - (correction_factor * (baselog["curve_processed"] / -6))
            - (correction_factor * (baselog["goal_range_processed_1"] / 1.2))
            - (correction_factor * (baselog["goal_slope"] / 4))
            - (correction_factor * ((baselog["pace_diff"] + 0.6) / -3))
            - (correction_factor * (baselog["umaban_rank_diff_processed_2"] * -10))
            - (correction_factor * (baselog["height_diff"] / -2)) 
        )

        """
        rank_diff*race_grade
        どんな数字になる？
        """
        
        baselog["rank_diff_correction_position_rush_xxx_race_grade_multi"] = (
            ((baselog["race_grade"]-120) *(4 + baselog["rank_diff_correction_position_rush"]))+1000
        )
        baselog["rank_diff_correction_position_xxx_race_grade_multi"] = (
            ((baselog["race_grade"]-120)  * (4 + baselog["rank_diff_correction_position"]))+1000
        )
        baselog["rank_diff_correction_rush_xxx_race_grade_multi"] = (
            ((baselog["race_grade"]-120)  * (4 + baselog["rank_diff_correction_rush"]))+1000
        )
        baselog["rank_diff_correction_xxx_race_grade_multi"] = (
           ((baselog["race_grade"]-120)  * (4 + baselog["rank_diff_correction"]))+1000
        )


        baselog["rank_diff_correction_position_rush_xxx_race_grade_sum"] = (
            ((baselog["race_grade"]-120)/10  + (4 + baselog["rank_diff_correction_position_rush"]))+100
        )
        baselog["rank_diff_correction_position_xxx_race_grade_sum"] = (
            ((baselog["race_grade"]-120)/10  + (4 + baselog["rank_diff_correction_position"]))+100
        )
        baselog["rank_diff_correction_rush_xxx_race_grade_sum"] = (
            ((baselog["race_grade"]-120)/10 + (4 + baselog["rank_diff_correction_rush"]))+100
        )
        baselog["rank_diff_correction_xxx_race_grade_sum"] = (
            ((baselog["race_grade"]-120)/10  + (4 + baselog["rank_diff_correction"]))+100
        )





        baselog = baselog.copy() 




        # # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # #



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





        """
        1/競馬場xタイプxランクx距離ごとの平均タイムと直近のレースとの差異の平均を特徴量に、そのまま数値として入れれる
        2/芝以外を除去
        3/競馬場x芝タイプで集計(グループバイ)
        4/"race_date_day_count"の当該未満かつ、800以内が一週間以内のデータになるはず
        5/その平均を取る
        6/+なら軽く、-なら重く、それぞれ5段階のカテゴリに入れる
        """    
        df = (
            merged_df
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
        

        df_old["place"] = df_old["place"].astype(int)
        df_old["race_grade"] = df_old["race_grade"].astype(int)
        df_old["ground_state"] = df_old["ground_state"].astype(int)
        df_old["around"] = df_old["around"].fillna(3).astype(int)
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
                return 1
            if row["weather"] in [3, 4, 5]:
                return 2

            if row["ground_state"] in [1]:
                return 2
            if row["ground_state"] in [2]:
                # mean_ground_state_time_diff に基づいて分類
                if 2.0 <= row["mean_ground_state_time_diff"]:
                    return 3  #　超高速馬場1
                elif 1 <= row["mean_ground_state_time_diff"] < 2.0:
                    return 3.7  # 高速馬場2
                
                elif 0.4 <= row["mean_ground_state_time_diff"] < 1:
                    return 4.5  # 軽い馬場3
                    
                elif -0.4 <= row["mean_ground_state_time_diff"] < 0.4:
                    return 5  # 標準的な馬場4
                elif -1 <= row["mean_ground_state_time_diff"] < -0.4:
                    return 5.5  # やや重い馬場5
                
                elif -2 <= row["mean_ground_state_time_diff"] < -1:
                    return 6.2  # 重い馬場5
                
                elif row["mean_ground_state_time_diff"] < -2:
                    return 7  # 道悪7

            # mean_ground_state_time_diff に基づいて分類
            if 2.0 <= row["mean_ground_state_time_diff"]:
                return 2  #　超高速馬場1
            elif 1 <= row["mean_ground_state_time_diff"] < 2.0:
                return 2.7  # 高速馬場2
            
            elif 0.4 <= row["mean_ground_state_time_diff"] < 1:
                return 3.5  # 軽い馬場3
                
            elif -0.4 <= row["mean_ground_state_time_diff"] < 0.4:
                return 4  # 標準的な馬場4
        
            
            elif -1 <= row["mean_ground_state_time_diff"] < -0.4:
                return 4.5  # やや重い馬場5
            
            elif -2 <= row["mean_ground_state_time_diff"] < -1:
                return 5.2  # 重い馬場5
            
            elif row["mean_ground_state_time_diff"] < -2:
                return 6  # 道悪7
            # 該当しない場合は NaN を設定
            return 2.7
        
        # 新しい列を追加
        if date_condition_a != 0:
            df["ground_state_level"] = date_condition_a
        else:
            df["ground_state_level"] = df.apply(assign_value, axis=1)

        
        merged_df = merged_df.merge(df[["race_id","date","horse_id","mean_ground_state_time_diff","ground_state_level"]],on=["race_id","date","horse_id"])	




        grouped_df = baselog.groupby(["race_id", "horse_id"])


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
                

                result_dfs.append(agg_df)
        
        
        # 結果をconcatして1つのデータフレームにまとめる
        final_agg_df = pd.concat(result_dfs, axis=1)
        
        # merge_dfとマージ
        merged_df = merged_df.merge(final_agg_df, on=["race_id", "horse_id"], how="left")
                


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

        merged_df00 = merged_df[["race_id","horse_id",
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
        tmp_df = merged_df00.groupby("race_id")
        
        # 各レースごとに、平均と標準偏差を計算
        # .transform を使って、同じサイズのデータフレームに変換
        relative_df = (merged_df00 - tmp_df.transform("mean")) / tmp_df.transform("std")
        
        # カラム名をわかりやすくするために "_relative" を追加
        relative_df = relative_df.add_suffix("_relative")
        
        # 元のデータフレームに標準化されたデータを結合
        merged_df1000 = merged_df00.join(relative_df, how="left")
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
        merged_df= merged_df.merge(merged_df1000, on=["race_id", "horse_id"], how="left")


        """
        ・どのタイプが有利かを判断する列
        -は瞬発有利の競馬場
        +は持続有利
        """
        df = (
            merged_df
            .merge(self.race_info, on=["race_id","date"])
            .merge(self.results, on=["race_id","horse_id"])
        )

        df["goal_range_processed"] = (((df["goal_range"])-360)/100)
        df["goal_range_processed"] = df["start_range_processed"].apply(
            lambda x: x*2 if x < 0 else x*0.7
        )

        df["curve_processed"] = df["curve"] - 4.5
        # +の場合は数値を8倍する
        df["curve_processed"] = df["curve_processed"].apply(
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
        df["rush_type_Advantages"] = (
            0 - (((df["curve_amount"] - 4) / 1))
            - ((df["curve_processed"] / -1))
            - ((df["goal_range_processed"] / 1.2))
            - ((df["goal_slope"] / 1.2))
            - (((df["pace_category"] - 2.5) / -0.75))
            - (((df["umaban"]-8) / -6))
            - ((df["height_diff"] / -0.7)) 
        )
        



        # b_baselog = baselog[['race_id',
        # 'date',
        # 'horse_id',

        # "rush_type",

        # ]]

        # grouped_df = b_baselog.groupby(["race_id", "horse_id"])
        # for n_race in tqdm(n_races, desc=f"kari"):
        #     df_speed = (
        #         grouped_df.head(n_race)
        #         .groupby(["race_id", "horse_id"])[
        #             [
        #                 "rush_type",
        #             ]
        #         ]
        #         .agg(["min"])
        #     )
        #     original_df = df_speed.copy()

        #     # 相対値を付けない元の列をそのまま追加
        #     original_df.columns = [
        #         "_".join(col) + f"_{n_race}races" for col in original_df.columns
        #     ]  # 列名変更
            
        #     df = df.merge(
        #         original_df, on=["race_id", "horse_id"], how="left"
        #     )
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
        merged_df_rush = df.copy()

        # race_id, horse_id でグループ化
        grouped_df = baselog.groupby(["race_id", "horse_id"])

        # 各 n_race について計算
        for n_race in tqdm(n_race_list, desc="計算中"):
            rush_type_avg_df = grouped_df.apply(lambda group: calculate_rush_type_averages(group, n_race)).reset_index()

            # df にマージ
            df = merged_df_rush.merge(rush_type_avg_df, on=["race_id", "horse_id"], how="left")



        def calculate_rush_type_advantage(df, n_race_list=[8, 5, 3, 1]):
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
        df = calculate_rush_type_advantage(df)




        """
        真の距離を作成、真の距離との差を作成

        """

        #最大200前後
        df["course_len_pace_diff"] = df["course_len"] + (((df["pace_category"] - 2.5) * 100))

        #グレード100前後
        df["course_len_diff_grade"] = df["course_len_pace_diff"] + (((df['race_grade']/70)-1)*200)

        #100前後
        df["course_len_diff_grade_slope"] = np.where(
            (df["season"] == 1) | (df["season"] == 4),
            df["course_len_diff_grade"] + (df["goal_slope"] * 30),
            df["course_len_diff_grade"] + (df["goal_slope"] * 13)
        )

        #最初の直線の長さ、長いほどきつい、50前後くらい
        df["start_range_processed_1"] = (((df["start_range"])-360)/150)
        df["start_range_processed_1"] = df["start_range_processed_1"].apply(
            lambda x: x if x < 0 else x*0.5
        )

        df["start_range_processed_course"] = df["start_range_processed_1"]*30
        df["course_len_pace_diff_grade_slope_range"] = df["course_len_diff_grade_slope"] + (df["start_range_processed_course"])

        # 条件ごとに処理を適用
        df["course_len_diff_grade_slope_range_pace"] = np.where(
            ((df['dominant_position_category'] == 1) | (df['dominant_position_category'] == 2)) & (df["pace_category"] >= 2.5),
            df["course_len_pace_category_grade_slope_range"] + (((df["pace_category"] - 2.5) / 6) * 100),
            
            np.where(
                ((df['dominant_position_category'] == 1) | (df['dominant_position_category'] == 2)) & (df["pace_category"] < 2.5),
                df["course_len_pace_category_grade_slope_range"] + (((df["pace_category"] - 2.5) / 12)*-100),
                
                np.where(
                    ((baselog['race_position'] == 3) | (baselog['race_position'] == 4)) & (df["pace_category"] < 2.5),
                    df["course_len_pace_category_grade_slope_range"] + (((df["pace_category"] - 2.5) / 20) * -100),
                    
                    np.where(
                        ((df['dominant_position_category'] == 3) | (df['dominant_position_category'] == 4)) & (df["pace_category"] >= 2.5),
                        df["course_len_pace_category_grade_slope_range"] + (((df["pace_category"] - 2.5) / 10) * 100),
                        df["course_len_pace_category_grade_slope_range"]  # どの条件にも当てはまらない場合は元の値を保持
                    )
                )
            )
        )



        #12コーナーがきついと、ゆるい、-
        df["course_len_diff_grade_slope_range_pace_12curve"] = df["course_len_diff_grade_slope_range_pace"] + (df["curve_processed"] * 25)

        #向正面上り坂、ゆるい、-
        df["course_len_diff_grade_slope_range_pace_12curve_front"] = df["course_len_diff_grade_slope_range_pace_12curve"] - (df["flont_slope"] * 25)



        #最大0.02*n
        def calculate_course_len_pace_diff(row):
            if row["curve_amount"] == 0:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"]
            elif row["curve_amount"] <= 2:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R34"]-100)/3)
            elif row["curve_amount"] <= 3:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 / 2 + (row["curve_R34"]-100)/4)
            elif row["curve_amount"] <= 4:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 + ((row["curve_R34"]-100)/4))
            elif row["curve_amount"] <= 5:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 + ((row["curve_R34"]-100)/4) * 3 / 2)
            elif row["curve_amount"] <= 6:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] +((row["curve_R12"]-100)/4 + ((row["curve_R34"]-100)/4) * 2)
            elif row["curve_amount"] <= 7:
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 * 3 / 2 + ((row["curve_R34"]-100)/4) * 2)
            else:  # curve_amount <= 8
                return row["course_len_diff_grade_slope_range_pace_12curve_front"] + ((row["curve_R12"]-100)/4 * 2 +((row["curve_R34"]-100)/4) * 2)

        df["course_len_diff_grade_slope_range_pace_12curve_front_R"] = df.apply(calculate_course_len_pace_diff, axis=1)

        #最大0.09くらい
        df["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] = df["course_len_diff_grade_slope_range_pace_12curve_front_R"] + (df["height_diff"]*50)


        # 条件ごとに適用
        df["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate"] = np.where(
            ((df["ground_state"] == 1) | (df["ground_state"] == 3)) & (df["race_type"] == 1),
            df["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] + 400,

            np.where(
                (df["ground_state"] == 2) & (df["race_type"] == 1),
                df["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] + 120,

                np.where(
                    ((df["ground_state"] == 1) | (df["ground_state"] == 3)) & (df["race_type"] == 0),
                    df["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] - 100,

                    np.where(
                        (df["ground_state"] == 2) & (df["race_type"] == 0),
                        df["course_len_diff_grade_slope_range_pace_12curve_front_R_height"] - 50,
                        
                        # どの条件にも当てはまらない場合は元の値を保持
                        df["course_len_diff_grade_slope_range_pace_12curve_front_R_height"]
                    )
                )
            )
        )

        df = df.copy()
        df.loc[:, "place_season_condition_type_categori_processed_courselen"] = (
            df["place_season_condition_type_categori"]
            .replace({5: -0.3, 4: -0.17, 3: 0, 2: 0.17,1: 0.3, -1: -0.18, -2: 0, -3: 0.18,-4:0.3,-10000:0})
        ).astype(float)

        df["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place"] = (
            df["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate"] + (df["place_season_condition_type_categori_processed_courselen"]*-500)
            )

        #最大0.05くらい
        df["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight"] = (
            df["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place"] - (((df["straight_total"]/ df["course_len"])-0.5)*400)
            )


        baselog_df_info = baselog.merge(
                df[["race_id", "course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight"]], on="race_id", suffixes=("", "_info")
            )


        baselog_df_info["course_len_diff_R"] = baselog_df_info["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight_info"] - baselog_df_info["course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight"]


        grouped_df = baselog_df_info.groupby(["race_id", "horse_id"])
        merged_df_len = self.population.copy()

        for n_race in tqdm(n_races, desc="course_len_relative"):
            df_len = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [
                        "course_len_diff_R"
                    ]
                ]
                .agg(["mean", "max", "min"])
            )
            df_len.columns = ["_".join(col) + f"_{n_race}races" for col in df_len.columns]
            # レースごとの相対値に変換
            original_df = df_len.copy()

            tmp_df = df_len.groupby(["race_id"])
            relative_df = ((df_len - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            merged_df_len = merged_df_len.merge(
                relative_df, on=["race_id", "horse_id"], how="left"
            )
            merged_df_len = merged_df_len.merge(original_df, on=["race_id", "horse_id"], how="left")
        df = df.merge(merged_df_len , on=["race_id", "horse_id"], how="left")



        
        df_baselog =(
                df.merge(
                    base_2,
                    on=["horse_id", "course_len", "race_type"],
                    suffixes=("", "_horse"),
                    .query("date_horse < date")
                    .sort_values("date_horse", ascending=False)
            )
        )

        grouped_df =  df_baselog.groupby(["race_id", "horse_id"])
        merged_df_pop = self.population.copy()
        for n_race in tqdm(n_races, desc="agg_horse_per_course_len"):
            df_relative = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [

                        "nobori_pace_diff_grade_slope_range_pace",
                        "nobori_pace_diff_grade_slope_range_pace_groundstate",
                        "nobori_pace_diff_grade_curveR_height_diff_season_straight",
                        "nobori_pace_diff_grade_curveR_height_diff",
                        "nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len",
                        "nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont_point",
                        "course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight",
                        "rank_diff_pace_course_len_ground_state_type",
                        "rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope",
                        "rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point",
                        "time_class",
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
                        
                    ]
                ]
                .agg(["mean", "min"])
            )
            df_relative.columns = [
                "_".join(col) + f"_{n_race}races_per_course_len" for col in df_relative.columns
            ]
            # レースごとの相対値に変換
            tmp_df = df_relative.groupby(["race_id"])
            relative_df = ((df_relative - tmp_df.mean()) / tmp_df.std()).add_suffix("_relative")
            merged_df_pop = merged_df_pop.merge(
                relative_df, on=["race_id", "horse_id"], how="left"
            )
            
        df = df.merge(
                merged_df_pop, on=["race_id", "horse_id"], how="left"
            )


        df_baselog2 =(
                df.merge(
                    base_2,
                    on=["horse_id"],
                    suffixes=("", "_horse"),
                    .query("date_horse < date")
                    .sort_values("date_horse", ascending=False)
            )
        )

        grouped_df =  df_baselog2.groupby(["race_id", "horse_id"])
        merged_df_pop2 = self.population.copy()
        for n_race in tqdm(n_races, desc="2"):
            df_relative2 = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[
                    [

                        "nobori_pace_diff_grade_slope_range_pace",
                        "nobori_pace_diff_grade_slope_range_pace_groundstate",
                        "nobori_pace_diff_grade_curveR_height_diff_season_straight",
                        "nobori_pace_diff_grade_curveR_height_diff",
                        "nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len",
                        "nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont_point",
                        "course_len_diff_grade_slope_range_pace_12curve_front_R_height_groundstate_place_straight",
                        "rank_diff_pace_course_len_ground_state_type",
                        "rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope",
                        "rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point",
                        "time_class",
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
                        
                    ]
                ]
                .agg(["mean", "max"])
            )
            df_relative2.columns = [
                "_".join(col) + f"_{n_race}races" for col in df_relative2.columns
            ]
            # レースごとの相対値に変換
            tmp_df2 = df_relative2.groupby(["race_id"])
            relative_df2 = ((df_relative2 - tmp_df.mean()) / tmp_df2.std()).add_suffix("_relative")
            merged_df_pop2 = merged_df_pop2.merge(
                relative_df2, on=["race_id", "horse_id"], how="left"
            )
            
        df = df.merge(
                merged_df_pop2, on=["race_id", "horse_id"], how="left"
            )
        


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
        if date_condition_a != 0:
            df["ground_state_level_processed"] = date_condition_a
        else:
            df.loc[(df['weather'].isin([1, 2])) & (df['ground_state'] == 2), "ground_state_level_processed"] = 4
            df.loc[(~df['weather'].isin([1, 2])) & (df['ground_state'] == 2), "ground_state_level_processed"] = -5
            df.loc[df['ground_state'].isin([1, 3]), "ground_state_level_processed"] = -9
            df.loc[~df['ground_state'].isin([1, 2, 3]), "ground_state_level_processed"] = df["mean_ground_state_time_diff"] * -5


        df["pace_category_processed"]  = (df["pace_category"]  - 2.5) *8
        # dominant_position_category_processed 列の処理



        # tenkai_sumed の計算
        df["tenkai_sumed"] = (
            df["ground_state_level_processed"] + df["pace_category_processed"]
        )

        # 条件に応じて dominant_position_category_processed を更新
        df.loc[df["tenkai_sumed"] < 0, "dominant_position_category_processed"] = (
            df["dominant_position_category"]
            .replace({1: -1.93, 2: -2, 3: 0.5, 4: 1.87})
            .astype(float)
        )

        df.loc[df["tenkai_sumed"] >= 0, "dominant_position_category_processed"] = (
            df["dominant_position_category"]
            .replace({1: -1.93, 2: -0.1, 3: 1.87, 4: 1.1})
            .astype(float)
        )

        # tenkai_combined の計算
        df["tenkai_combined"] = df["tenkai_sumed"] * df["dominant_position_category_processed"]




        # place_season_condition_type_categori の処理
        df["place_season_condition_type_categori_processed_z"] = df["place_season_condition_type_categori"].replace(
            {5: -1.3, 4: -0.7, 3: 0, 2: 0.7, 1: 1.3, -1: -0.9, -2: 0, -3: 0.9, -4: 1.2, -10000: 0}
        ).astype(float)

        # tenkai_place_sumed の計算
        df["tenkai_place_sumed"] = df["tenkai_sumed"] + df["place_season_condition_type_categori_processed_z"]

        # dominant_position_category_processed の更新
        df["dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: 0.1, 4: 1.67}
        )
        df.loc[df["tenkai_place_sumed"] >= 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.73, 2: -0.1, 3: 1.87, 4: 1.1}
        )

        # tenkai_place_combined の計算
        df["tenkai_place_combined"] = df["tenkai_place_sumed"] * df["dominant_position_category_processed"]





        # start_slope の処理
        df["start_slope_abs_processed"] = df["start_slope"].abs() / -2

        # tenkai_place_start_slope_sumed の計算
        df["tenkai_place_start_slope_sumed"] = df["tenkai_place_sumed"] + df["start_slope_abs_processed"]

        # dominant_position_category_processed の更新 (再利用)
        df.loc[df["tenkai_place_start_slope_sumed"] < 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: 0.1, 4: 1.67}
        )
        df.loc[df["tenkai_place_start_slope_sumed"] >= 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.73, 2: -0.1, 3: 1.87, 4: 1.1}
        )

        # tenkai_place_start_slope_combined の計算
        df["tenkai_place_start_slope_combined"] = df["tenkai_place_start_slope_sumed"] * df["dominant_position_category_processed"]




        # start_range_processed_1 の計算
        df["start_range_processed_1"] = ((df["start_range"] - 360) / 100).apply(lambda x: x * 3 if x < 0 else x * 0.5)

        # tenkai_place_start_slope_range_sumed の計算
        df["tenkai_place_start_slope_range_sumed"] = df["tenkai_place_start_slope_sumed"] + df["start_range_processed_1"]

        # dominant_position_category_processed の更新 (再利用)
        df.loc[df["tenkai_place_start_slope_range_sumed"] < 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: 0.1, 4: 1.67}
        )
        df.loc[df["tenkai_place_start_slope_range_sumed"] >= 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.73, 2: -0.1, 3: 1.87, 4: 1.1}
        )

        # tenkai_place_start_slope_range_combined の計算
        df["tenkai_place_start_slope_range_combined"] = df["tenkai_place_start_slope_range_sumed"] * df["dominant_position_category_processed"]





        # tenkai_place_start_slope_range_grade_sumed の計算
        df["tenkai_place_start_slope_range_grade_sumed"] = df["tenkai_place_start_slope_range_sumed"] + ((df["race_grade"] - 70) / 10)

        # dominant_position_category_processed の更新 (再利用)
        df.loc[df["tenkai_place_start_slope_range_grade_sumed"] < 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: 0.1, 4: 1.67}
        )
        df.loc[df["tenkai_place_start_slope_range_grade_sumed"] >= 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.73, 2: -0.1, 3: 1.87, 4: 1.1}
        )

        # tenkai_place_start_slope_range_grade_combined の計算
        df["tenkai_place_start_slope_range_grade_combined"] = df["tenkai_place_start_slope_range_grade_sumed"] * df["dominant_position_category_processed"]





        # tenkai_place_start_slope_range_grade_lcurve_slope_sumed の計算
        df["tenkai_place_start_slope_range_grade_lcurve_slope_sumed"] = df["tenkai_place_start_slope_range_grade_sumed"] + (df["last_curve_slope"] * -2)

        # dominant_position_category_processed の更新 (再利用)
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_sumed"] < 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: 0.1, 4: 1.67}
        )
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_sumed"] >= 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.73, 2: -0.1, 3: 1.87, 4: 1.1}
        )

        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        df["tenkai_place_start_slope_range_grade_lcurve_slope_combined"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_sumed"] * df["dominant_position_category_processed"]




        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_sumed"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_sumed"] + (df["race_type"]-1)*2

        # dominant_position_category_processed の更新 (再利用)
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_sumed"] < 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: 0.1, 4: 1.67}
        )
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_sumed"] >= 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.73, 2: -0.1, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_combined"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_sumed"] * df["dominant_position_category_processed"]




        # -4.5 を行う
        df["curve_processed"] = df["curve"] - 4.5
        # +の場合は数値を8倍する
        df["curve_processed"] = df["curve_processed"].apply(
            lambda x: x * 8 if x > 0 else x
        )

        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_sumed"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_sumed"] + (df["curve_processed"])*0.8

        # dominant_position_category_processed の更新 (再利用)
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_sumed"] < 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: 0.1, 4: 1.67}
        )
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_sumed"] >= 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.73, 2: -0.1, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_combined"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_sumed"] * df["dominant_position_category_processed"]




        df["goal_range_processed_1"] = (((df["goal_range"])-360)/100)
        df["goal_range_processed_1"] = df["goal_range_processed_1"].apply(
            lambda x: x*3 if x < 0 else x*0.8
        )


        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_sumed"] + (df["goal_range_processed_1"]*1.2)

        # dominant_position_category_processed の更新 (再利用)
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] < 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: 0.1, 4: 1.67}
        )
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] >= 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.73, 2: -0.1, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] * df["dominant_position_category_processed"]





        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_sumed"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_sumed"] + (df["goal_slope"]*1.8)

        # dominant_position_category_processed の更新 (再利用)
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_sumed"] < 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: 0.1, 4: 1.67}
        )
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_sumed"] >= 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.73, 2: -0.1, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_combined"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_sumed"] * df["dominant_position_category_processed"]







        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_sumed"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_sumed"] + (df["first_curve_slope"]*-1.5)

        # dominant_position_category_processed の更新 (再利用)
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_sumed"] < 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: 0.1, 4: 1.67}
        )
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_sumed"] >= 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.73, 2: -0.1, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_combined"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_sumed"] * df["dominant_position_category_processed"]





        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_sumed"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_sumed"] + (df["flont_slope"]*-1.5)

        # dominant_position_category_processed の更新 (再利用)
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_sumed"] < 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: 0.1, 4: 1.67}
        )
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_sumed"] >= 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.73, 2: -0.1, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_combined"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_sumed"] * df["dominant_position_category_processed"]





        df["course_len_processed_11"] = (df["course_len"] - 1600) /1000
        df["course_len_processed_11"] = df["course_len_processed_11"].apply(
            lambda x: x*4 if x < 0 else x
        )

        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_sumed"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_sumed"] + df["course_len_processed_11"]

        # dominant_position_category_processed の更新 (再利用)
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_sumed"] < 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: 0.1, 4: 1.67}
        )
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_sumed"] >= 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.73, 2: -0.1, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_combined"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_sumed"] * df["dominant_position_category_processed"]



        #0,01-0.01,内がプラス(内枠が有利を受けるとしたら、rank_diffは+にして、有利ポジはマイナス補正)
        df["umaban_processed2"] = df["umaban"].apply(
            lambda x: ((x*-1.5)+1.5) if x < 4 else ((x-8)/1.5)-1
        ).astype(float)
        #0,-0.1,-0.3,-0.36,-0.3,-0.23,（-1/10のとき）
        df["umaban_processed2"] = df["umaban_processed2"]*2
        #0 , -0.05
        #1（奇数）または 0（偶数）,偶数が有利
        df.loc[:, "umaban_odd_processed2"] = (
            (df["umaban_odd"]-1)*2
        ).astype(float)

        #rdが-0.25,,0.25が0.5に
        df["umaban_processed2"] = df["umaban_processed2"] / ((df["course_len_processed_11"]) + 4)
        df["umaban_odd_processed2"] = df["umaban_odd_processed2"] / ((df["course_len_processed_11"])+4)




        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_sumed"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_sumed"] + df["umaban_odd_processed2"] + df["umaban_processed2"]

        # dominant_position_category_processed の更新 (再利用)
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_sumed"] < 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.93, 2: -2, 3: 0.1, 4: 1.67}
        )
        df.loc[df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_sumed"] >= 0, "dominant_position_category_processed"] = df["dominant_position_category"].replace(
            {1: -1.73, 2: -0.1, 3: 1.87, 4: 1.1}
        )
        # tenkai_place_start_slope_range_grade_lcurve_slope_combined の計算
        df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] = df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_sumed"] * df["dominant_position_category_processed"]





        """
        タイプ適性
        7*4
        rankgradeと+してもよい
        sumと+するなら1/10する
        multiとするならそのまま
        掛け算するなら+200/200する
        """
        df["rush_advantages_cross"] = df["rush_type_Advantages"] * df["rush_type_final"]




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
        # -4.5 を行う
        df["curve_processed_umaban"] = df["curve"] - 4.5
        # +の場合は数値を8倍する
        df["curve_processed_umaban"] = df["curve_processed_umaban"].apply(
            lambda x: x * 4 if x > 0 else x
        )

        df["curve_umaban"] = (df["curve_processed_umaban"] + df["umaban_processed2"] * 1/2)*-1

        # `race_type` に基づいて `umaban_p_type` を設定する
        df["umaban_p_type"] = df.apply(lambda row: -1/2 * row["umaban_processed2"] if row["race_type"] == 1 else (1/2 * row["umaban_processed2"] if row["race_type"] == 0 else None), axis=1)

        df["umaban_odd_2"] = df["umaban_odd_processed2"] * -1

        df["last_curve_umaban"] = ((df["last_curve_slope"]) + df["umaban_processed2"] * 1/2) * -1

        df["start_range_umaban"] = ((df["start_range_processed_1"] / 2) + df["umaban_processed2"] * 1 + df["umaban_odd_processed2"]) * -1

        df["start_slope_umaban"] = ((df["start_slope_abs_processed"] * -1) + df["umaban_processed2"] * 1/2 + df["umaban_odd_processed2"] * 1/2) * -1

        # 最終的な補正値の計算
        umaban_correction_position = (
            df["curve_umaban"] + df["umaban_p_type"] + df["umaban_odd_2"] + df["last_curve_umaban"] + df["start_range_umaban"] + df["start_slope_umaban"]
        )



        # df["rank_diff_pace_course_len_ground_state_type_mean_1races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_pace_course_len_ground_state_type_mean_3races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_pace_course_len_ground_state_type_mean_5races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_pace_course_len_ground_state_type_mean_8races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_1races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_3races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_5races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_8races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_1races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_3races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_5races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_8races_umaban"] = df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # df["rank_diff_correction_mean_1races_umaban"] = df["rank_diff_correction_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_correction_mean_3races_umaban"] = df["rank_diff_correction_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_correction_mean_5races_umaban"] = df["rank_diff_correction_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_correction_mean_8races_umaban"] = df["rank_diff_correction_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # df["rank_diff_correction_position_mean_1races_umaban"] = df["rank_diff_correction_position_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_correction_position_mean_3races_umaban"] = df["rank_diff_correction_position_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_correction_position_mean_5races_umaban"] = df["rank_diff_correction_position_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_correction_position_mean_8races_umaban"] = df["rank_diff_correction_position_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # df["rank_diff_correction_rush_mean_1races_umaban"] = df["rank_diff_correction_rush_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_correction_rush_mean_3races_umaban"] = df["rank_diff_correction_rush_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_correction_rush_mean_5races_umaban"] = df["rank_diff_correction_rush_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_correction_rush_mean_8races_umaban"] = df["rank_diff_correction_rush_mean_8races"] * ((300 + umaban_correction_position) / 300)

        # df["rank_diff_correction_position_rush_mean_1races_umaban"] = df["rank_diff_correction_position_rush_mean_1races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_correction_position_rush_mean_3races_umaban"] = df["rank_diff_correction_position_rush_mean_3races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_correction_position_rush_mean_5races_umaban"] = df["rank_diff_correction_position_rush_mean_5races"] * ((300 + umaban_correction_position) / 300)
        # df["rank_diff_correction_position_rush_mean_8races_umaban"] = df["rank_diff_correction_position_rush_mean_8races"] * ((300 + umaban_correction_position) / 300)

        df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races"] * ((300 + umaban_correction_position) / 300)

        df["rank_diff_correction_position_xxx_race_grade_multi_mean_1races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_1races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_position_xxx_race_grade_multi_mean_3races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_3races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_position_xxx_race_grade_multi_mean_5races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_5races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_position_xxx_race_grade_multi_mean_8races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_8races"] * ((300 + umaban_correction_position) / 300)

        df["rank_diff_correction_rush_xxx_race_grade_multi_mean_1races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_1races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_rush_xxx_race_grade_multi_mean_3races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_3races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_rush_xxx_race_grade_multi_mean_5races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_5races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_rush_xxx_race_grade_multi_mean_8races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_8races"] * ((300 + umaban_correction_position) / 300)

        df["rank_diff_correction_xxx_race_grade_multi_mean_1races_umaban"] = df["rank_diff_correction_xxx_race_grade_multi_mean_1races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_xxx_race_grade_multi_mean_3races_umaban"] = df["rank_diff_correction_xxx_race_grade_multi_mean_3races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_xxx_race_grade_multi_mean_5races_umaban"] = df["rank_diff_correction_xxx_race_grade_multi_mean_5races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_xxx_race_grade_multi_mean_8races_umaban"] = df["rank_diff_correction_xxx_race_grade_multi_mean_8races"] * ((300 + umaban_correction_position) / 300)

        df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races_umaban"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races"] * ((300 + umaban_correction_position) / 300)

        df["rank_diff_correction_position_xxx_race_grade_sum_mean_1races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_1races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_position_xxx_race_grade_sum_mean_3races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_3races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_position_xxx_race_grade_sum_mean_5races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_5races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_position_xxx_race_grade_sum_mean_8races_umaban"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_8races"] * ((300 + umaban_correction_position) / 300)

        df["rank_diff_correction_rush_xxx_race_grade_sum_mean_1races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_1races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_rush_xxx_race_grade_sum_mean_3races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_3races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_rush_xxx_race_grade_sum_mean_5races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_5races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_rush_xxx_race_grade_sum_mean_8races_umaban"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_8races"] * ((300 + umaban_correction_position) / 300)

        df["rank_diff_correction_xxx_race_grade_sum_mean_1races_umaban"] = df["rank_diff_correction_xxx_race_grade_sum_mean_1races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_xxx_race_grade_sum_mean_3races_umaban"] = df["rank_diff_correction_xxx_race_grade_sum_mean_3races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_xxx_race_grade_sum_mean_5races_umaban"] = df["rank_diff_correction_xxx_race_grade_sum_mean_5races"] * ((300 + umaban_correction_position) / 300)
        df["rank_diff_correction_xxx_race_grade_sum_mean_8races_umaban"] = df["rank_diff_correction_xxx_race_grade_sum_mean_8races"] * ((300 + umaban_correction_position) / 300)


        """
        それぞれのrank_gradeかけ
        """



        # df["rank_diff_pace_course_len_ground_state_type_mean_1races_umaban"]
        # df["rank_diff_pace_course_len_ground_state_type_mean_3races_umaban"]
        # df["rank_diff_pace_course_len_ground_state_type_mean_5races_umaban"]
        # df["rank_diff_pace_course_len_ground_state_type_mean_8races_umaban"]

        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_1races_umaban"]
        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_3races_umaban"]
        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_5races_umaban"]
        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_mean_8races_umaban"]

        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_1races_umaban"]
        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_3races_umaban"]
        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_5races_umaban"]
        # df["rank_diff_pace_course_len_ground_state_type_odd_curve_slope_start_slope_rank_diff_flont_point_mean_8races_umaban"]

        # df["rank_diff_correction_mean_1races_umaban"]
        # df["rank_diff_correction_mean_3races_umaban"]
        # df["rank_diff_correction_mean_5races_umaban"]
        # df["rank_diff_correction_mean_8races_umaban"]

        # df["rank_diff_correction_position_mean_1races_umaban"]
        # df["rank_diff_correction_position_mean_3races_umaban"]
        # df["rank_diff_correction_position_mean_5races_umaban"]
        # df["rank_diff_correction_position_mean_8races_umaban"]

        # df["rank_diff_correction_rush_mean_1races_umaban"]
        # df["rank_diff_correction_rush_mean_3races_umaban"]
        # df["rank_diff_correction_rush_mean_5races_umaban"]
        # df["rank_diff_correction_rush_mean_8races_umaban"]

        # df["rank_diff_correction_position_rush_mean_1races_umaban"]
        # df["rank_diff_correction_position_rush_mean_3races_umaban"]
        # df["rank_diff_correction_position_rush_mean_5races_umaban"]
        # df["rank_diff_correction_position_rush_mean_8races_umaban"]

        df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_1races_umaban"] + df["rush_advantages_cross"]/6
        
        df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_multi_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_xxx_race_grade_multi_mean_1races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_xxx_race_grade_multi_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_1races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_xxx_race_grade_multi_mean_3races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_xxx_race_grade_multi_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_xxx_race_grade_multi_mean_5races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_xxx_race_grade_multi_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_xxx_race_grade_multi_mean_8races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_xxx_race_grade_multi_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_multi_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_rush_xxx_race_grade_multi_mean_1races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_rush_xxx_race_grade_multi_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_1races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_rush_xxx_race_grade_multi_mean_3races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_rush_xxx_race_grade_multi_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_rush_xxx_race_grade_multi_mean_5races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_rush_xxx_race_grade_multi_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_rush_xxx_race_grade_multi_mean_8races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_rush_xxx_race_grade_multi_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_multi_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_xxx_race_grade_multi_mean_1races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_xxx_race_grade_multi_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_1races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_xxx_race_grade_multi_mean_3races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_xxx_race_grade_multi_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_xxx_race_grade_multi_mean_5races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_xxx_race_grade_multi_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_xxx_race_grade_multi_mean_8races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_xxx_race_grade_multi_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_multi_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_1races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races_umaban_and_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_position_rush_xxx_race_grade_sum_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_xxx_race_grade_sum_mean_1races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_xxx_race_grade_sum_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_1races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_xxx_race_grade_sum_mean_3races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_xxx_race_grade_sum_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_xxx_race_grade_sum_mean_5races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_xxx_race_grade_sum_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_position_xxx_race_grade_sum_mean_8races_umaban_and_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_position_xxx_race_grade_sum_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_position_xxx_race_grade_sum_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_rush_xxx_race_grade_sum_mean_1races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_rush_xxx_race_grade_sum_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_1races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_rush_xxx_race_grade_sum_mean_3races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_rush_xxx_race_grade_sum_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_rush_xxx_race_grade_sum_mean_5races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_rush_xxx_race_grade_sum_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_rush_xxx_race_grade_sum_mean_8races_umaban_and_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_rush_xxx_race_grade_sum_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_rush_xxx_race_grade_sum_mean_8races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_xxx_race_grade_sum_mean_1races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_1races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_xxx_race_grade_sum_mean_1races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_1races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_xxx_race_grade_sum_mean_3races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_3races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_xxx_race_grade_sum_mean_3races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_3races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_xxx_race_grade_sum_mean_5races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_5races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_xxx_race_grade_sum_mean_5races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_5races_umaban"] + df["rush_advantages_cross"]/6

        df["rank_diff_correction_xxx_race_grade_sum_mean_8races_umaban_and_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_8races_umaban"] * ((df["rush_advantages_cross"] + 200)/200)
        df["rank_diff_correction_xxx_race_grade_sum_mean_8races_umaban_plus_rush"] = df["rank_diff_correction_xxx_race_grade_sum_mean_8races_umaban"] + df["rush_advantages_cross"]/6



        # レース数のバリエーション
        race_counts = [1, 3, 5, 8]

        # 各レース数に対応するカラムの作成
        for r in race_counts:
            # rank_diff_correction_position_rush_xxx_race_grade_multi_mean_xraces_umaban
            df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
            df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
            df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
            df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

            df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
            df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)

            # rank_diff_correction_position_xxx_race_grade_multi_mean_xraces_umaban
            df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
            df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
            df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
            df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

            df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
            df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)

            # rank_diff_correction_rush_xxx_race_grade_multi_mean_xraces_umaban
            df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
            df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
            df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
            df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

            df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
            df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)

            # rank_diff_correction_xxx_race_grade_multi_mean_xraces_umaban
            df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
            df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
            df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
            df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

            df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
            df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)

            # rank_diff_correction_position_rush_xxx_race_grade_sum_mean_xraces_umaban
            df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
            df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
            df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
            df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

            df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
            df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)

            # rank_diff_correction_position_xxx_race_grade_sum_mean_xraces_umaban
            df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
            df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
            df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
            df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

            df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
            df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)


            # rank_diff_correction_rush_xxx_race_grade_sum_mean_xraces_umaban
            df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
            df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
            df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
            df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

            df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
            df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)

            # rank_diff_correction_xxx_race_grade_sum_mean_xraces_umaban
            df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_place"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400)
            df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_place"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)
            
            df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_half"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500)
            df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_half"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)

            df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban_and_tenkai_all"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600)
            df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban_plus_tenkai_all"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)

        """
        全合わせ
        """

        # レース数のバリエーション
        race_counts = [1, 3, 5, 8]

        # 各レース数に対応するカラムの作成
        for r in race_counts:
            # rank_diff_correction_position_rush_xxx_race_grade_multi_mean_xraces_umaban
            df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

            df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6

            # rank_diff_correction_position_xxx_race_grade_multi_mean_xraces_umaban
            df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
            df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

            df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6

            # rank_diff_correction_rush_xxx_race_grade_multi_mean_xraces_umaban
            df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
            df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

            df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_rush_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6

            # rank_diff_correction_xxx_race_grade_multi_mean_xraces_umaban
            df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
            df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

            df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_all_fix"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_all_sum"] = df[f"rank_diff_correction_xxx_race_grade_multi_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6

            # rank_diff_correction_position_rush_xxx_race_grade_sum_mean_xraces_umaban
            df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
            df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

            df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6

            # rank_diff_correction_position_xxx_race_grade_sum_mean_xraces_umaban
            df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
            df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

            df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_position_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6


            # rank_diff_correction_rush_xxx_race_grade_sum_mean_xraces_umaban
            df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
            df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

            df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_rush_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6

            # rank_diff_correction_xxx_race_grade_sum_mean_xraces_umaban
            df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_combined"] + 400)/400) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_combined"]/60)+ df["rush_advantages_cross"]/6
            
            df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"] + 500)/500) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_combined"]/80)+ df["rush_advantages_cross"]/6

            df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_all_fix"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] * ((df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"] + 600)/600) * ((df["rush_advantages_cross"] + 200)/200)
            df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_all_sum"] = df[f"rank_diff_correction_xxx_race_grade_sum_mean_{r}races_umaban"] + (df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined"]/100)+ df["rush_advantages_cross"]/6


            print(df.columns.tolist())


            """
            noboriと
            nobori修正
            **これ、の交互特徴量
            上がりが早いほうが有利なのは

            直線が長い
            上り坂
            最終コーナーがゆるい、早くこれる
            """


            df["cross_nobori_range"] = df["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont_point_min_3races"] * (20/(df["goal_range_processed_1"]+20))
            df["cross_nobori_slope"] = df["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont_point_min_3races"] * (10/(df["goal_slope"]+10))
            df["cross_nobori_corner"] = df["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont_point_min_3races"] * (10/(df["curve_processed"]+10))
            df["cross_nobori_all"] = df["nobori_pace_diff_grade_curveR_height_diff_season_straight_course_len_umaban_odds_curve_slope_start_range_start_slope_corner_flont_point_min_3races"] * (10/(df["curve_processed"]+10))* (20/(df["goal_range_processed_1"]+20))* (10/(df["goal_slope"]+10))

            """
            タフは距離短縮
            ２種類作る
            ＋がタフ
            """
            df["course_len_shorter"] = (df["course_len_diff_R_mean_3races"]/10) * df["rush_type_Advantages"]



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
        self.cross_features_16()
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
            .merge(
                self.agg_cross_features_df_16,
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