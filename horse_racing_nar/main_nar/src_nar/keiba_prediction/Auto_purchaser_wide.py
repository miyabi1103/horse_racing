import asyncio
import re
from playwright.async_api import Playwright, async_playwright, expect
import pandas as pd
from io import StringIO
from collections import defaultdict
from bs4 import BeautifulSoup
import numpy as np
import ast

import sys
from pathlib import Path


import matplotlib
matplotlib.use("Agg")  # GUIバックエンドを無効化
import matplotlib.pyplot as plt
import seaborn as sns
DATA_DIR = Path("..", "..", "data_nar")
SAVE_DIR = DATA_DIR / "05_prediction_results"
from dotenv import load_dotenv
# .envファイルを読み込む
load_dotenv()
import os
# DiscordのWebhook URLを環境変数から取得
SPAT_KANYUSYA_URL = os.getenv("SPAT_KANYUSYA_NUM")
USE_URL = os.getenv("USER_ID")
PASSWORD = os.getenv("SPAT_PASSWORD")
if not USE_URL:
    raise ValueError("USE_URLが設定されていません。'.env'ファイルを確認してください。")
if not SPAT_KANYUSYA_URL:
    raise ValueError("SPAT_KANYUSYA_URLが設定されていません。'.env'ファイルを確認してください。")
if not PASSWORD:
    raise ValueError("PASSWORDが設定されていません。'.env'ファイルを確認してください。")

csv_file_path = Path("../../data_nar/05_prediction_results/prediction_result.csv")


# if not INET_URL:
#     raise ValueError("INET_IDが設定されていません。'.env'ファイルを確認してください。")
# if not KANYUSYA_URL:
#     raise ValueError("KANYUSYA_NOが設定されていません。'.env'ファイルを確認してください。")
# if not PASSWORD_URL:
#     raise ValueError("PASSWORD_PATが設定されていません。'.env'ファイルを確認してください。")
# if not PARS_URL:
#     raise ValueError("PARS_NOが設定されていません。'.env'ファイルを確認してください。")

async def Auto_purchase_wide(race_id:str,top_n: int = 2,amount: str = "100",amount_num: str = "1"):
    csv_path = Path("../../data_nar/05_prediction_results/prediction_result.csv")
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(csv_path, sep="\t")

        # umaban列の上位N個を抽出
        top_umaban = df["umaban"].head(top_n).tolist()

    except FileNotFoundError:
        print(f"Error: ファイルが見つかりません: {csv_path}")

    except KeyError:
        print(f"Error: umaban列が存在しません: {csv_path}")

    except Exception as e:
        print(f"Error: {e}")
    # ここでtop_umabanを返す
    print(f"抽出されたumaban: {top_umaban}")

    # top_umabanの一番手前の数字を文字列に変換
    first_umaban = str(top_umaban[0])  # 1番目の要素
    second_umaban = str(top_umaban[1]) if len(top_umaban) > 1 else None  # 2番目の要素（存在する場合）
    third_umaban = str(top_umaban[2]) if len(top_umaban) > 2 else None  # 3番目の要素（存在する場合）


    race_type_mapping = {
        "ダ": 0
    }
    place_mapping = {
        30: "門別",
        35: "盛岡",
        36: "水沢",
        42: "浦和",
        43: "船橋",
        44: "大井",
        45: "川崎",
        46: "金沢",
        47: "笠松",
        48: "名古屋",
        50: "園田",
        51: "姫路",
        54: "高知",
        55: "佐賀"
    }


    async with async_playwright() as playwright:
        # playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        place_name = f"{place_mapping[int(race_id[4:6])]}"
        place_count = f"{int(race_id[10:12])}"

        await page.goto("https://www.spat4.jp/keiba/pc")
        await page.locator("#MEMBERNUMR").click()
        await page.locator("#MEMBERNUMR").fill(SPAT_KANYUSYA_URL)
        await page.locator("#MEMBERIDR").click()
        await page.locator("#MEMBERIDR").fill(USE_URL)
        await page.get_by_text("ログイン", exact=True).click()
        await asyncio.sleep(2)
        close_buttons = page.get_by_role("button", name="閉じる")
        if await close_buttons.count() > 0:
            if await close_buttons.nth(0).is_visible():
                await close_buttons.nth(0).click()
        else:
            await asyncio.sleep(2)
        await asyncio.sleep(1)
        try:
            # ここでplace_countRをクリック（一定時間だけ待つ）
            await asyncio.wait_for(
                page.get_by_role("link", name=place_name, exact=True).click(),
                timeout=10  # 秒
            )
        except asyncio.TimeoutError:
            # タイムアウトしたら代わりに「照会」と「開催要領」ボタンをクリック
            await page.get_by_role("button", name="照会").click()
            await page.get_by_role("button", name="開催要領").click()
        await page.get_by_role("link", name=f"{place_count}R").click()
        async with page.expect_popup() as page2_info:
            await page.get_by_role("button", name="マークカード投票").click()
        page2 = await page2_info.value
        await page2.get_by_text("ボックス").click()
        await page2.get_by_text("ワイド").click()
        await page2.get_by_role("button", name=first_umaban, exact=True).click()
        await page2.get_by_role("button", name=second_umaban, exact=True).click()
        await page2.get_by_text("投票金額入力へ").click()
        await page2.get_by_role("cell", name="各 00円 0円").get_by_role("textbox").click()
        await page2.get_by_role("cell", name="各 00円 0円").get_by_role("textbox").fill(amount_num)
        await page2.locator("#gotoCfm-buy").click()
        await page2.locator("input[name=\"cfm_ansho\"]").click()
        await page2.locator("input[name=\"cfm_ansho\"]").fill(PASSWORD)
        await asyncio.sleep(1)
        await page2.locator("input[name=\"cfm_amount\"]").click()
        await asyncio.sleep(1)
        await page2.locator("input[name=\"cfm_amount\"]").fill(amount)
        await asyncio.sleep(2)
        try:
            # ここでplace_countRをクリック（一定時間だけ待つ）
            await asyncio.wait_for(
                page2.get_by_text("投票する").click(),
                timeout=10  # 秒
            )
        except asyncio.TimeoutError:
            await page2.locator("div").filter(has_text=place_name).first.click()
            await asyncio.sleep(2)
            await page2.get_by_text("投票する").click()
        await asyncio.sleep(2)
        # ---------------------
        await context.close()
        await browser.close()
    print("ワイド投票が完了しました")

# if __name__ == "__main__":
#     asyncio.run(Auto_purchase_wide())
