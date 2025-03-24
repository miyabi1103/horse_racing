import re
import time
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import scraping
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm

POPULATION_DIR = Path("..","..", "..","common","data", "prediction_population")
POPULATION_DIR.mkdir(exist_ok=True, parents=True)


# def scrape_horse_id_list(race_id: str) -> list[str]:
#     """
#     レースidを指定すると、出走馬id一覧が返ってくる関数。
#     """
#     url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
#     html = urlopen(url).read()
#     soup = BeautifulSoup(html, "lxml")
#     td_list = soup.find_all("td", class_="HorseInfo")
#     horse_id_list = []
#     for td in td_list:
#         horse_id = re.findall(r"\d{10}", td.find("a")["href"])[0]
#         horse_id_list.append(horse_id)
#     return horse_id_list


# def create(
#     kaisai_date: str,
#     save_dir: Path = POPULATION_DIR,
#     save_filename: str = "population.csv",
# ) -> pd.DataFrame:
#     """
#     開催日（yyyymmdd形式）を指定すると、予測母集団である
#     (date, race_id, horse_id)のDataFrameが返ってくる関数。
#     """
#     print("scraping race_id_list...")
#     race_id_list = scraping.scrape_race_id_list([kaisai_date])
#     dfs = {}
#     print("scraping horse_id_list...")
#     for race_id in tqdm(race_id_list):
#         horse_id_list = scrape_horse_id_list(race_id)
#         time.sleep(1)
#         df = pd.DataFrame(
#             {"date": kaisai_date, "race_id": race_id, "horse_id": horse_id_list}
#         )
#         dfs[race_id] = df
#     #これをやるとデータがどんどん縦に繋がってくれる
#     concat_df = pd.concat(dfs.values())
#     concat_df["date"] = pd.to_datetime(concat_df["date"])
#     concat_df.to_csv(save_dir / save_filename, index=False, sep="\t")
#     return concat_df

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import re
import time
import random

def scrape_horse_id_list(race_id: str) -> list[str]:
    """
    レースidを指定すると、出走馬id一覧が返ってくる関数。
    """
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

    # User-Agentを設定してリクエストを送信
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
    try:
        html = urlopen(req).read()
        soup = BeautifulSoup(html, "lxml")
        td_list = soup.find_all("td", class_="HorseInfo")
        horse_id_list = []
        for td in td_list:
            horse_id = re.findall(r"\d{10}", td.find("a")["href"])[0]
            horse_id_list.append(horse_id)
        return horse_id_list
    except Exception as e:
        print(f"Error scraping race {race_id}: {e}")
        return []  # エラーが発生した場合、空のリストを返す

def create(
    kaisai_date: str,
    save_dir: Path = POPULATION_DIR,
    save_filename: str = "population.csv",
) -> pd.DataFrame:
    """
    開催日（yyyymmdd形式）を指定すると、予測母集団である
    (date, race_id, horse_id)のDataFrameが返ってくる関数。
    """
    print("scraping race_id_list...")
    race_id_list = scraping.scrape_race_id_list([kaisai_date])
    dfs = {}
    print("scraping horse_id_list...")
    for race_id in tqdm(race_id_list):
        horse_id_list = scrape_horse_id_list(race_id)
        
        # リクエスト間隔をランダムに設定（1秒から3秒の間）
        time.sleep(random.uniform(1, 3)) 
        
        df = pd.DataFrame(
            {"date": kaisai_date, "race_id": race_id, "horse_id": horse_id_list}
        )
        dfs[race_id] = df
    
    # データを縦に結合
    concat_df = pd.concat(dfs.values())
    concat_df["date"] = pd.to_datetime(concat_df["date"])
    concat_df.to_csv(save_dir / save_filename, index=False, sep="\t")
    return concat_df






# def old_create(
#     kaisai_date: str,
#     save_dir: Path = POPULATION_DIR,
#     save_filename: str = "population_old.csv",
# ) -> pd.DataFrame:
#     """
#     開催日（yyyymmdd形式）を指定すると、予測母集団である
#     (date, race_id, horse_id)のDataFrameが返ってくる関数。
#     """
#     print("scraping race_id_list...")
#     for race_id in tqdm(race_id_list):
#         horse_id_list = scrape_horse_id_list(race_id)
        
#         # リクエスト間隔をランダムに設定（1秒から3秒の間）
#         time.sleep(random.uniform(1, 3)) 
        
#         df = pd.DataFrame(
#             {"date": kaisai_date, "race_id": race_id, "horse_id": horse_id_list}
#         )
#         dfs[race_id] = df
        
#     race_id_list = scraping.scrape_race_id_list([kaisai_date_list_prediction])
#     dfs = {}
    
#     # print("scraping horse_id_list...")
#     # for race_id in tqdm(race_id_list):
#     #     horse_id_list = scrape_horse_id_list(race_id)
        
#     #     # リクエスト間隔をランダムに設定（1秒から3秒の間）
#     #     time.sleep(random.uniform(1, 3)) 
        
#     #     df = pd.DataFrame(
#     #         {"date": kaisai_date, "race_id": race_id, "horse_id": horse_id_list}
#     #     )
#     #     dfs[race_id] = df
    
#     # データを縦に結合
#     concat_df = pd.concat(dfs.values())
#     concat_df["date"] = pd.to_datetime(concat_df["date"])
#     concat_df.to_csv(save_dir / save_filename, index=False, sep="\t")
#     return concat_df





















