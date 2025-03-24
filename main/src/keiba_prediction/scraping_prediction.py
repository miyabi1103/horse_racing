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





DATA_DIR = Path("..", "..", "..","common", "data")
HTML_RACE_DIR = DATA_DIR / "html" / "race2"
HTML_HORSE_DIR = DATA_DIR / "html" / "horse2"
HTML_PED_DIR = DATA_DIR / "html" / "ped2"
HTML_LEADING_DIR = DATA_DIR / "html" / "leading2"


RAWDF_DIR = DATA_DIR / "rawdf2"


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










def create_results(
    html_path_list: list[Path],
    save_dir: Path = RAWDF_DIR,
    save_filename: str = "results.csv",
) -> pd.DataFrame:
    """
    保存されているraceページのhtmlを読み込んで、レース結果テーブルに加工する関数。
    """
    dfs = {}
    for html_path in tqdm(html_path_list):
        with open(html_path, "rb") as f:
            try:
                race_id = html_path.stem
                html = (
                    f.read()
                    .replace(b"<diary_snap_cut>", b"")
                    .replace(b"</diary_snap_cut>", b"")
                )
                soup = BeautifulSoup(html, "lxml").find(
                    "table", class_="race_table_01 nk_tb_common"
                )
                df = pd.read_html(html)[0]

                # horse_id列追加
                horse_id_list = []
                a_list = soup.find_all("a", href=re.compile(r"^/horse/"))
                for a in a_list:
                    horse_id = re.findall(r"\d{10}", a["href"])[0]
                    horse_id_list.append(horse_id)
                df["horse_id"] = horse_id_list

                # jockey_id列追加
                jockey_id_list = []
                a_list = soup.find_all("a", href=re.compile(r"^/jockey/"))
                for a in a_list:
                    jockey_id = re.findall(r"\d{5}", a["href"])[0]
                    jockey_id_list.append(jockey_id)
                df["jockey_id"] = jockey_id_list

                # trainer_id列追加
                trainer_id_list = []
                a_list = soup.find_all("a", href=re.compile(r"^/trainer/"))
                for a in a_list:
                    trainer_id = re.findall(r"\d{5}", a["href"])[0]
                    trainer_id_list.append(trainer_id)
                df["trainer_id"] = trainer_id_list

                # owner_id列追加
                owner_id_list = []
                a_list = soup.find_all("a", href=re.compile(r"^/owner/"))
                for a in a_list:
                    owner_id = re.findall(r"\d{6}", a["href"])[0]
                    owner_id_list.append(owner_id)
                df["owner_id"] = owner_id_list

                # 最初の列にrace_idを挿入
                df.insert(0, "race_id", race_id)
                dfs[race_id] = df
            except IndexError as e:
                print(f"table not found at {race_id}")
                continue
    concat_df = pd.concat(dfs.values())
    concat_df.columns = concat_df.columns.str.replace(" ", "")
    save_dir.mkdir(parents=True, exist_ok=True)
    update_rawdf(
        concat_df,
        key="race_id",
        save_filename=save_filename,
        save_dir=save_dir,
    )
    return concat_df


def create_race_info(
    html_path_list: list[Path],
    save_dir: Path = RAWDF_DIR,
    save_filename: str = "race_info.csv",
) -> pd.DataFrame:
    """
    保存されているraceページのhtmlを読み込んで、レース情報テーブルに加工する関数。
    """
    dfs = {}
    for html_path in tqdm(html_path_list):
        with open(html_path, "rb") as f:
            try:
                # ファイル名からrace_idを取得
                race_id = html_path.stem
                html = f.read()
                soup = BeautifulSoup(html, "lxml").find("div", class_="data_intro")
                info_dict = {}
                info_dict["title"] = soup.find("h1").text
                p_list = soup.find_all("p")
                info_dict["info1"] = re.findall(
                    r"[\w:]+", p_list[0].text.replace(" ", "")
                )
                info_dict["info2"] = re.findall(r"\w+", p_list[1].text)
                df = pd.DataFrame().from_dict(info_dict, orient="index").T
                # 最初の列にrace_idを挿入
                df.insert(0, "race_id", race_id)
                dfs[race_id] = df
            except IndexError as e:
                print(f"table not found at {race_id}")
                continue
            except AttributeError as e:
                print(f"{e} at {race_id}")
                continue
    concat_df = pd.concat(dfs.values())
    concat_df.columns = concat_df.columns.str.replace(" ", "")
    save_dir.mkdir(exist_ok=True, parents=True)
    update_rawdf(
        concat_df,
        key="race_id",
        save_filename=save_filename,
        save_dir=save_dir,
    )
    return concat_df


def update_rawdf(
    new_df: pd.DataFrame,
    key: str,
    save_filename: str,
    save_dir: Path = RAWDF_DIR,
) -> None:
    """
    既存のrawdfに新しいデータを追加して保存する関数。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / save_filename).exists():
        old_df = pd.read_csv(save_dir / save_filename, sep="\t", dtype={f"{key}": str})
        # 念の為、key列をstr型に変換
        new_df[key] = new_df[key].astype(str)
        df = pd.concat([old_df[~old_df[key].isin(new_df[key])], new_df])
        df.to_csv(save_dir / save_filename, sep="\t", index=False)
    else:
        # ファイルが存在しない場合は単にそのまま保存
        new_df.to_csv(save_dir / save_filename, sep="\t", index=False)

