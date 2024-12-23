import re
import time
import traceback
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from tqdm.notebook import tqdm
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import random

import requests
import time
from pathlib import Path
from tqdm import tqdm


DATA_DIR = Path("..", "data")
HTML_RACE_DIR = DATA_DIR / "html" / "race"
HTML_HORSE_DIR = DATA_DIR / "html" / "horse"
HTML_PED_DIR = DATA_DIR / "html" / "ped"
HTML_LEADING_DIR = DATA_DIR / "html" / "leading"


# def scrape_kaisai_date(from_: str, to_: str, save_dir: Path = None) -> list[str]:
#     """
#     from_とto_をyyyy-mmの形で指定すると、間の開催日一覧を取得する関数。
#     save_dirを指定すると、取得結果がkaisai_date_list.txtとして保存される。
#     """

#     kaisai_date_list = []
#     for date in tqdm(pd.date_range(from_, to_, freq="MS")):
#         year = date.year
#         month = date.month
#         url = f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"
#         html = urlopen(url).read()  # スクレイピング
#         time.sleep(1)  # 絶対忘れないように
#         soup = BeautifulSoup(html, "lxml")
#         a_list = soup.find("table", class_="Calendar_Table").find_all("a")
#         for a in a_list:
#             kaisai_date = re.findall(r"kaisai_date=(\d{8})", a["href"])[0]
#             kaisai_date_list.append(kaisai_date)
#     if save_dir:
#         save_dir.mkdir(parents=True, exist_ok=True)
#         with open(save_dir / "kaisai_date_list.txt", "w") as f:
#             f.write("\n".join(kaisai_date_list))
#     return kaisai_date_list


def scrape_kaisai_date(from_: str, to_: str, save_dir: Path = None) -> list[str]:
    """
    指定された期間の Netkeiba 開催日一覧を取得する関数。

    Args:
        from_ (str): 開始年月 (yyyy-mm)。
        to_ (str): 終了年月 (yyyy-mm)。
        save_dir (Path, optional): 保存先ディレクトリ。指定しない場合は保存しない。

    Returns:
        list[str]: 開催日 (yyyyMMdd) のリスト。
    """
    kaisai_date_list = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}

    for date in tqdm(pd.date_range(from_, to_, freq="MS")):
        year = date.year
        month = date.month
        url = f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"
        try:
            # ヘッダーを指定してリクエスト
            req = Request(url, headers=headers)
            html = urlopen(req).read()
        except (HTTPError, URLError) as e:
            print(f"Error fetching {url}: {e}")
            continue  # 次の月に進む

        time.sleep(1)
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", class_="Calendar_Table")
        if not table:
            print(f"No table found on {url}")
            continue

        a_list = table.find_all("a")
        for a in a_list:
            match = re.search(r"kaisai_date=(\d{8})", a.get("href", ""))
            if match:
                kaisai_date = match.group(1)
                kaisai_date_list.append(kaisai_date)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "kaisai_date_list.txt", "w") as f:
            f.write("\n".join(kaisai_date_list))

    return kaisai_date_list



# def scrape_race_id_list(
#     kaisai_date_list: list[str], save_dir: Path = None
# ) -> list[str]:
#     """
#     開催日（yyyymmdd形式）をリストで入れると、レースid一覧が返ってくる関数。
#     save_dirを指定すると、取得結果がrace_id_list.txtとして保存される。
#     """
#     options = Options()
#     # ヘッドレスモード（バックグラウンド）で起動
#     options.add_argument("--headless")
#     # その他のクラッシュ対策
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")
#     driver_path = ChromeDriverManager().install()
#     race_id_list = []

#     with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
#         # 要素を取得できない時、最大10秒待つ
#         driver.implicitly_wait(10)
#         for kaisai_date in tqdm(kaisai_date_list):
#             url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={kaisai_date}"
#             try:
#                 driver.get(url)
#                 time.sleep(1)
#                 li_list = driver.find_elements(By.CLASS_NAME, "RaceList_DataItem")
#                 for li in li_list:
#                     href = li.find_element(By.TAG_NAME, "a").get_attribute("href")
#                     race_id = re.findall(r"race_id=(\d{12})", href)[0]
#                     race_id_list.append(race_id)
#             except:
#                 print(f"stopped at {url}")
#                 print(traceback.format_exc())
#                 break
#     if save_dir:
#         save_dir.mkdir(parents=True, exist_ok=True)
#         with open(save_dir / "race_id_list.txt", "w") as f:
#             f.write("\n".join(race_id_list))
#     return race_id_list











# def scrape_race_id_list(
#     kaisai_date_list: list[str], save_dir: Path = None
# ) -> list[str]:
#     """
#     開催日（yyyymmdd形式）をリストで入れると、レースid一覧が返ってくる関数。
#     save_dirを指定すると、取得結果がrace_id_list.txtとして保存される。
#     再実行時、途中まで取得済みのデータがあれば再開して取得する。
#     """
#     options = Options()
#     options.add_argument("--headless")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")
#     driver_path = ChromeDriverManager().install()
#     race_id_list = []
    
#     # 保存ファイルの設定と既存データの読み込み
#     if save_dir:
#         save_dir.mkdir(parents=True, exist_ok=True)
#         save_file = save_dir / "race_id_list.txt"
#         if save_file.exists():
#             with open(save_file, "r") as f:
#                 race_id_list = f.read().splitlines()
#         else:
#             save_file.touch()  # ファイルが存在しない場合は新規作成

#     # 処理済みの日付を特定し、未処理の日付のみを対象にする
#     processed_dates = {race_id[:8] for race_id in race_id_list}  # race_idの最初8文字（yyyyMMdd形式）で処理済みの日付を特定
#     remaining_dates = [date for date in kaisai_date_list if date not in processed_dates]
    
#     with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
#         wait = WebDriverWait(driver, 10000)
#         for kaisai_date in tqdm(remaining_dates):
#             url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={kaisai_date}"
#             try:
#                 driver.get(url)
#                 wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "RaceList_DataItem")))
#                 li_list = driver.find_elements(By.CLASS_NAME, "RaceList_DataItem")
                
#                 for li in li_list:
#                     href = li.find_element(By.TAG_NAME, "a").get_attribute("href")
#                     race_id = re.findall(r"race_id=(\d{12})", href)[0]
#                     if race_id not in race_id_list:  # 重複チェック
#                         race_id_list.append(race_id)
                        
#                 # 途中結果をファイルに保存
#                 if save_dir:
#                     with open(save_file, "w") as f:
#                         f.write("\n".join(race_id_list))
#             except Exception:
#                 print(f"stopped at {url}")
#                 print(traceback.format_exc())
#                 break

#     return race_id_list





def scrape_race_id_list(
    kaisai_date_list_2: list[str], save_dir: Path = None, skip: bool = True
) -> list[str]:
    """
    開催日（yyyymmdd形式）をリストで入れると、レースid一覧が返ってくる関数。
    save_dirを指定すると、取得結果がrace_id_list.txtとして保存される。
    再実行時、途中まで取得済みのデータがあれば再開して取得する。
    skip=Trueにすると、既に取得済みのrace_idをスキップする。
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver_path = ChromeDriverManager().install()
    race_id_list = []
    
    # 保存ファイルの設定と既存データの読み込み
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / "race_id_list.txt"
        if save_file.exists():
            with open(save_file, "r") as f:
                race_id_list = f.read().splitlines()
        else:
            save_file.touch()  # ファイルが存在しない場合は新規作成

    # 処理済みの日付を特定し、未処理の日付のみを対象にする
    processed_dates = {race_id[:8] for race_id in race_id_list}  # race_idの最初8文字（yyyyMMdd形式）で処理済みの日付を特定
    remaining_dates = [date for date in kaisai_date_list_2 if date not in processed_dates]
    
    with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
        wait = WebDriverWait(driver, 10000)
        driver.set_page_load_timeout(10000)  # タイムアウトを10000秒に設定
        for kaisai_date in tqdm(remaining_dates):
            url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={kaisai_date}"
            try:
                driver.get(url)
                wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "RaceList_DataItem")))
                li_list = driver.find_elements(By.CLASS_NAME, "RaceList_DataItem")
                
                for li in li_list:
                    href = li.find_element(By.TAG_NAME, "a").get_attribute("href")
                    race_id = re.findall(r"race_id=(\d{12})", href)[0]
                    
                    # スキップ処理
                    if skip and race_id in race_id_list:
                        print(f"skipped: {race_id}")
                        continue
                    
                    if race_id not in race_id_list:  # 重複チェック
                        race_id_list.append(race_id)
                        
                # 途中結果をファイルに保存
                if save_dir:
                    with open(save_file, "w") as f:
                        f.write("\n".join(race_id_list))
            except Exception:
                print(f"stopped at {url}")
                print(traceback.format_exc())
                break

    return race_id_list







# def scrape_html_race(
#     race_id_list: list[str], save_dir: Path = HTML_RACE_DIR, skip: bool = True
# ) -> list[Path]:
#     """
#     netkeiba.comのraceページのhtmlをスクレイピングしてsave_dirに保存する関数。
#     skip=Trueにすると、すでにhtmlが存在する場合はスキップされる。
#     逆に上書きしたい場合は、skip=Falseにする。
#     スキップされたhtmlのパスは返り値に含まれない。
#     """
#     updated_html_path_list = []
#     save_dir.mkdir(parents=True, exist_ok=True)
#     for race_id in tqdm(race_id_list):
#         filepath = save_dir / f"{race_id}.bin"
#         # skipがTrueで、かつファイルがすでに存在する場合は飛ばす
#         if skip and filepath.is_file():
#             print(f"skipped: {race_id}")
#         else:
#             url = f"https://db.netkeiba.com/race/{race_id}"
#             html = urlopen(url).read()
#             time.sleep(1)
#             with open(filepath, "wb") as f:
#                 f.write(html)
#             updated_html_path_list.append(filepath)
#     return updated_html_path_list

from urllib.request import urlopen, Request
from urllib.error import HTTPError

def scrape_html_race(
    race_id_list: list[str], save_dir: Path = HTML_RACE_DIR, skip: bool = True
) -> list[Path]:
    """
    netkeiba.comのraceページのhtmlをスクレイピングしてsave_dirに保存する関数。
    skip=Trueにすると、すでにhtmlが存在する場合はスキップされる。
    逆に上書きしたい場合は、skip=Falseにする。
    スキップされたhtmlのパスは返り値に含まれない。
    """
    updated_html_path_list = []
    save_dir.mkdir(parents=True, exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}

    for race_id in tqdm(race_id_list):
        filepath = save_dir / f"{race_id}.bin"
        # skipがTrueで、かつファイルがすでに存在する場合は飛ばす
        if skip and filepath.is_file():
            print(f"skipped: {race_id}")
        else:
            try:
                url = f"https://db.netkeiba.com/race/{race_id}"
                request = Request(url, headers=headers)
                html = urlopen(request).read()
                time.sleep(1)
                with open(filepath, "wb") as f:
                    f.write(html)
                updated_html_path_list.append(filepath)
            except HTTPError as e:
                print(f"Error fetching {race_id}: {e}")

    return updated_html_path_list






# def scrape_html_horse(
#     horse_id_list: list[str], save_dir: Path = HTML_HORSE_DIR, skip: bool = True
# ) -> list[Path]:
#     """
#     netkeiba.comのhorseページのhtmlをスクレイピングしてsave_dirに保存する関数。
#     skip=Trueにすると、すでにhtmlが存在する場合はスキップされる。
#     逆に上書きしたい場合は、skip=Falseにする。
#     スキップされたhtmlのパスは返り値に含まれない。
#     """
#     updated_html_path_list = []
#     save_dir.mkdir(parents=True, exist_ok=True)
#     for horse_id in tqdm(horse_id_list):
#         filepath = save_dir / f"{horse_id}.bin"
#         # skipがTrueで、かつファイルがすでに存在する場合は飛ばす
#         if skip and filepath.is_file():
#             print(f"skipped: {horse_id}")
#         else:
#             url = f"https://db.netkeiba.com/horse/{horse_id}"
#             html = urlopen(url).read()
#             time.sleep(1)
#             with open(filepath, "wb") as f:
#                 f.write(html)
#             updated_html_path_list.append(filepath)
#     return updated_html_path_list

def scrape_html_horse(
    horse_id_list: list[str], save_dir: Path = HTML_HORSE_DIR, skip: bool = True
) -> list[Path]:
    """
    netkeiba.comのhorseページのhtmlをスクレイピングしてsave_dirに保存する関数。
    skip=Trueにすると、すでにhtmlが存在する場合はスキップされる。
    逆に上書きしたい場合は、skip=Falseにする。
    スキップされたhtmlのパスは返り値に含まれない。
    """
    updated_html_path_list = []
    save_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'ja-JP,ja;q=0.9',
        'Connection': 'keep-alive'
    }

    for horse_id in tqdm(horse_id_list):
        filepath = save_dir / f"{horse_id}.bin"
        
        # skipがTrueで、かつファイルがすでに存在する場合は飛ばす
        if skip and filepath.is_file():
            print(f"skipped: {horse_id}")
        else:
            url = f"https://db.netkeiba.com/horse/{horse_id}"
            req = Request(url, headers=headers)
            
            try:
                html = urlopen(req).read()
                time.sleep(1)  # アクセス頻度を抑えるため、リクエスト間隔を2秒に設定
                
                with open(filepath, "wb") as f:
                    f.write(html)
                updated_html_path_list.append(filepath)
            
            except Exception as e:
                print(f"An error occurred for horse_id {horse_id}: {e}")
    
    return updated_html_path_list


# def scrape_html_horse(
#     horse_id_list: list[str], save_dir: Path = HTML_HORSE_DIR, skip: bool = True
# ) -> list[Path]:
#     """
#     netkeiba.comのhorseページのhtmlをスクレイピングしてsave_dirに保存する関数。
#     skip=Trueにすると、すでにhtmlが存在する場合はスキップされる。
#     逆に上書きしたい場合は、skip=Falseにする。
#     スキップされたhtmlのパスは返り値に含まれない。
#     """
#     updated_html_path_list = []
#     save_dir.mkdir(parents=True, exist_ok=True)
    
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
#         'Accept-Language': 'ja-JP,ja;q=0.9',
#         'Connection': 'keep-alive'
#     }


#     for horse_id in tqdm(horse_id_list):
#         filepath = save_dir / f"{horse_id}.bin"
        
#         # skipがTrueで、かつファイルがすでに存在する場合は飛ばす
#         if skip and filepath.is_file():
#             print(f"skipped: {horse_id}")
#         else:
#             url = f"https://db.netkeiba.com/horse/{horse_id}"
            
#             try:
#                 # requestsでGETリクエスト
#                 response = requests.get(url, headers=headers, timeout=10)
                
#                 # ステータスコードが200の場合のみ処理
#                 if response.status_code == 200:
#                     with open(filepath, "wb") as f:
#                         f.write(response.content)
#                     updated_html_path_list.append(filepath)
#                     print(f"Downloaded: {horse_id}")
#                 else:
#                     print(f"Failed to retrieve horse_id {horse_id} with status code {response.status_code}")
                
#                 # アクセス頻度を抑えるため、リクエスト間隔を1秒に設定
#                 time.sleep(1)
                
#             except requests.exceptions.RequestException as e:
#                 # リクエストに関する例外をキャッチ
#                 print(f"An error occurred for horse_id {horse_id}: {e}")
#                 time.sleep(5)  # エラー時は5秒待機して次のリクエストへ
                
#             except Exception as e:
#                 # その他のエラーをキャッチ
#                 print(f"Unexpected error for horse_id {horse_id}: {e}")
#                 time.sleep(5)  # エラー時は5秒待機して次のリクエストへ
    
#     return updated_html_path_list



# def scrape_html_leading(
#     leading_type: str,
#     year: int,
#     pages: list[int],
#     save_dir: Path = HTML_LEADING_DIR,
#     skip: bool = True,
# ) -> list[Path]:
#     """
#     リーディングページのhtmlをスクレイピングしてsave_dirに保存する関数。
#     skip=Trueにすると、すでにhtmlが存在する場合はスキップされる。
#     逆に上書きしたい場合は、skip=Falseにする。
#     スキップされたhtmlのパスは返り値に含まれない。
#     """

#     updated_html_path_list = []
#     save_dir = save_dir / leading_type
#     save_dir.mkdir(parents=True, exist_ok=True)
#     for page in tqdm(pages):
#         filepath = save_dir / f"{year}_{str(page).zfill(2)}.bin"
#         # skipがTrueで、かつファイルがすでに存在する場合は飛ばす
#         if skip and filepath.is_file():
#             print(f"skipped: {filepath}")
#         else:
#             url = (
#                 f"https://db.netkeiba.com//?pid={leading_type}&year={year}&page={page}"
#             )
#             html = urlopen(url).read()
#             time.sleep(1)
#             with open(filepath, "wb") as f:
#                 f.write(html)
#             updated_html_path_list.append(filepath)
#     return updated_html_path_list



def scrape_html_leading(
    leading_type: str,
    year: int,
    pages: list[int],
    save_dir: Path = HTML_LEADING_DIR,
    skip: bool = True,
) -> list[Path]:
    """
    リーディングページのhtmlをスクレイピングしてsave_dirに保存する関数。
    skip=Trueにすると、すでにhtmlが存在する場合はスキップされる。
    逆に上書きしたい場合は、skip=Falseにする。
    スキップされたhtmlのパスは返り値に含まれない。
    """

    updated_html_path_list = []
    save_dir = save_dir / leading_type
    save_dir.mkdir(parents=True, exist_ok=True)
    for page in tqdm(pages):
        filepath = save_dir / f"{year}_{str(page).zfill(2)}.bin"
        # skipがTrueで、かつファイルがすでに存在する場合は飛ばす
        if skip and filepath.is_file():
            print(f"skipped: {filepath}")
        else:
            url = f"https://db.netkeiba.com/?pid={leading_type}&year={year}&page={page}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.3945.79 Safari/537.36"}
            request = Request(url, headers=headers)
            html = urlopen(request).read()
            time.sleep(1)
            with open(filepath, "wb") as f:
                f.write(html)
            updated_html_path_list.append(filepath)
    return updated_html_path_list





# def scrape_html_ped(
#     horse_id_list: list[str], save_dir: Path = HTML_PED_DIR, skip: bool = True
# ) -> list[Path]:
#     """
#     netkeiba.comのpedページのhtmlをスクレイピングしてsave_dirに保存する関数。
#     skip=Trueにすると、すでにhtmlが存在する場合はスキップされる。
#     逆に上書きしたい場合は、skip=Falseにする。
#     スキップされたhtmlのパスは返り値に含まれない。
#     """
#     updated_html_path_list = []
#     save_dir.mkdir(parents=True, exist_ok=True)
    
#     for horse_id in tqdm(horse_id_list):
#         filepath = save_dir / f"{horse_id}.bin"
        
#         # skipがTrueで、かつファイルがすでに存在する場合は飛ばす
#         if skip and filepath.is_file():
#             print(f"skipped: {horse_id}")
#             continue
        
#         url = f"https://db.netkeiba.com/horse/ped/{horse_id}"
#         req = Request(
#             url,
#             headers={
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
#                 'Referer': 'https://db.netkeiba.com/'
#             }
#         )
        
#         try:
#             html = urlopen(req).read()
#             time.sleep(2)  # リクエスト間の待機時間
            
#             with open(filepath, "wb") as f:
#                 f.write(html)
#             updated_html_path_list.append(filepath)
        
#         except Exception as e:
#             print(f"Error fetching {horse_id}: {e}")  # エラーメッセージを出力
    
#     return updated_html_path_list



# import time
# from pathlib import Path
# from urllib.request import Request, urlopen
# from urllib.error import HTTPError, URLError
# from tqdm import tqdm



# def scrape_html_ped(
#     horse_id_list: list[str], save_dir: Path, skip: bool = True
# ) -> list[Path]:
#     """
#     netkeiba.comのpedページのhtmlをスクレイピングしてsave_dirに保存する関数。
#     skip=Trueにすると、すでにhtmlが存在する場合はスキップされる。
#     逆に上書きしたい場合は、skip=Falseにする。
#     スキップされたhtmlのパスは返り値に含まれない。
#     """
#     updated_html_path_list = []
#     save_dir.mkdir(parents=True, exist_ok=True)
    
#     for horse_id in tqdm(horse_id_list):
#         filepath = save_dir / f"{horse_id}.bin"
        
#         # skipがTrueで、かつファイルがすでに存在する場合は飛ばす
#         if skip and filepath.is_file():
#             print(f"skipped: {horse_id}")
#             continue
        
#         url = f"https://db.netkeiba.com/horse/ped/{horse_id}"
#         req = Request(
#             url,
#             headers={
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
#                 'Referer': 'https://db.netkeiba.com/'
#             }
#         )
        
#         try:
#             # 最大3回リトライ
#             for attempt in range(3):
#                 try:
#                     html = urlopen(req).read()
#                     with open(filepath, "wb") as f:
#                         f.write(html)
#                     updated_html_path_list.append(filepath)
#                     break  # 正常に完了したらループを抜ける
#                 except HTTPError as e:
#                     print(f"HTTP Error fetching {horse_id} (Attempt {attempt+1}): {e}")
#                     if e.code == 400:
#                         # 特定のhorse_idが存在しない、または無効である場合
#                         print(f"Invalid horse ID {horse_id} - Skipping.")
#                         break
#                     elif attempt < 2:
#                         time.sleep(2)  # リトライのための待機時間
#                     else:
#                         print(f"Failed to fetch {horse_id} after 3 attempts.")
#                 except URLError as e:
#                     print(f"URL Error fetching {horse_id} (Attempt {attempt+1}): {e}")
#                     time.sleep(2)  # リトライのための待機時間

#         except Exception as e:
#             print(f"Unexpected error for {horse_id}: {e}")
    
#     return updated_html_path_list





# def scrape_html_ped(
#     horse_id_list: list[str], save_dir: Path, skip: bool = True
# ) -> list[Path]:
#     updated_html_path_list = []
#     error_horse_ids = []  # エラーログ用
#     save_dir.mkdir(parents=True, exist_ok=True)
    
#     for horse_id in tqdm(horse_id_list):
#         filepath = save_dir / f"{horse_id}.bin"
        
#         if skip and filepath.is_file():
#             print(f"skipped: {horse_id}")
#             continue
        
#         url = f"https://db.netkeiba.com/horse/ped/{horse_id}"
#         req = Request(
#             url,
#             headers={
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
#                 'Referer': 'https://db.netkeiba.com/'
#             }
#         )
        
#         try:
#             for attempt in range(3):
#                 try:
#                     html = urlopen(req).read()
#                     with open(filepath, "wb") as f:
#                         f.write(html)
#                     updated_html_path_list.append(filepath)
#                     break
#                 except HTTPError as e:
#                     print(f"HTTP Error fetching {horse_id} (Attempt {attempt+1}): {e}")
#                     if e.code == 400:
#                         error_horse_ids.append(horse_id)  # 無効なhorse_idをリストに追加
#                         break
#                     elif attempt < 2:
#                         time.sleep(1)
#                         error_horse_ids.append(horse_id)
#                 except URLError as e:
#                     print(f"URL Error fetching {horse_id} (Attempt {attempt+1}): {e}")
#                     error_horse_ids.append(horse_id)
#                     time.sleep(2)
#         except Exception as e:
#             print(f"Unexpected error for {horse_id}: {e}")
#             error_horse_ids.append(horse_id)  # その他のエラーもエラーログに追加
    
#     # エラーログをCSVに保存
#     error_log_path = save_dir / "error_horse_ids.csv"
#     with open(error_log_path, "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["horse_id"])
#         for error_id in error_horse_ids:
#             writer.writerow([error_id])
    
#     return updated_html_path_list, error_horse_ids
















from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import csv
import time
from tqdm import tqdm

# def scrape_html_ped(
#     horse_id_list: list[str], save_dir: Path, skip: bool = True
# ) -> tuple[list[Path], list[str]]:
#     updated_html_path_list = []
#     error_horse_ids = []  # エラーログ用
#     save_dir.mkdir(parents=True, exist_ok=True)
    
#     for horse_id in tqdm(horse_id_list):
#         filepath = save_dir / f"{horse_id}.bin"
        
#         if skip and filepath.is_file():
#             print(f"skipped: {horse_id}")
#             continue
        
#         url = f"https://db.netkeiba.com/horse/ped/{horse_id}"
#         req = Request(
#             url,
#             headers={
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
#                 'Referer': 'https://db.netkeiba.com/'
#             }
#         )
        
#         success = False  # 各 horse_id の試行で成功したかを追跡
#         for attempt in range(3):
#             try:
#                 html = urlopen(req).read()
#                 with open(filepath, "wb") as f:
#                     f.write(html)
#                 updated_html_path_list.append(filepath)
#                 success = True  # 成功フラグを立てる
#                 break  # 成功した場合はループを抜ける
#             except HTTPError as e:
#                 print(f"HTTP Error fetching {horse_id} (Attempt {attempt+1}): {e}")
#                 if e.code == 400:  # 400エラーの場合は再試行せず終了
#                     break
#                 time.sleep(1)
#             except URLError as e:
#                 print(f"URL Error fetching {horse_id} (Attempt {attempt+1}): {e}")
#                 time.sleep(2)
#             except Exception as e:
#                 print(f"Unexpected error for {horse_id} (Attempt {attempt+1}): {e}")
#                 time.sleep(1)

#         if not success:  # 全試行が失敗した場合にエラーログに追加
#             error_horse_ids.append(horse_id)

#     # エラーログをCSVに保存
#     error_log_path = save_dir / "error_horse_ids.csv"
#     with open(error_log_path, "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["horse_id"])
#         for error_id in error_horse_ids:
#             writer.writerow([error_id])
    
#     return updated_html_path_list, error_horse_ids


def scrape_html_ped(
    horse_id_list: list[str], save_dir: Path, skip: bool = True
) -> tuple[list[Path], list[str]]:
    updated_html_path_list = []
    error_horse_ids = []  # エラーログ用
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for horse_id in tqdm(horse_id_list):
        filepath = save_dir / f"{horse_id}.bin"
        
        if skip and filepath.is_file():
            print(f"skipped: {horse_id}")
            continue
        
        url = f"https://db.netkeiba.com/horse/ped/{horse_id}"
        req = Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
                'Referer': 'https://db.netkeiba.com/'
            }
        )
        
        success = False
        attempt = 0  # リトライ回数を追跡
        while not success:
            try:
                html = urlopen(req).read()
                with open(filepath, "wb") as f:
                    f.write(html)
                updated_html_path_list.append(filepath)
                success = True
            except HTTPError as e:
                print(f"HTTP Error fetching {horse_id} (Attempt {attempt+1}): {e}")
                if e.code != 400:  # 400以外のエラーでは3回までリトライ
                    attempt += 1
                    if attempt >= 3:
                        break
                time.sleep(1)
            except URLError as e:
                print(f"URL Error fetching {horse_id} (Attempt {attempt+1}): {e}")
                attempt += 1
                if attempt >= 3:
                    break
                time.sleep(2)
            except Exception as e:
                print(f"Unexpected error for {horse_id} (Attempt {attempt+1}): {e}")
                attempt += 1
                if attempt >= 3:
                    break
                time.sleep(1)

        if not success:  # 全試行が失敗した場合にエラーログに追加
            error_horse_ids.append(horse_id)

    # エラーログをCSVに保存
    error_log_path = save_dir / "error_horse_ids.csv"
    with open(error_log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["horse_id"])
        for error_id in error_horse_ids:
            writer.writerow([error_id])
    
    return updated_html_path_list, error_horse_ids



