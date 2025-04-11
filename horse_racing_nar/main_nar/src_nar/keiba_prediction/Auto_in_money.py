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


# if not INET_URL:
#     raise ValueError("INET_IDが設定されていません。'.env'ファイルを確認してください。")
# if not KANYUSYA_URL:
#     raise ValueError("KANYUSYA_NOが設定されていません。'.env'ファイルを確認してください。")
# if not PASSWORD_URL:
#     raise ValueError("PASSWORD_PATが設定されていません。'.env'ファイルを確認してください。")
# if not PARS_URL:
#     raise ValueError("PARS_NOが設定されていません。'.env'ファイルを確認してください。")

async def auto_in_money():

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
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        # print("ブラウザを起動します")

        await page.goto("https://www.spat4.jp/keiba/pc")
        await page.locator("#MEMBERNUMR").click()
        await page.locator("#MEMBERNUMR").fill(SPAT_KANYUSYA_URL)
        await page.locator("#MEMBERIDR").click()
        await page.locator("#MEMBERIDR").fill(USE_URL)
        await page.get_by_text("ログイン", exact=True).click()
        async with page.expect_popup() as page1_info:
            await page.get_by_role("button", name="入金").click()
        page1 = await page1_info.value
        await page1.locator("#ENTERR").click()
        await page1.locator("#ENTERR").fill("1000")
        await page1.get_by_role("button", name="入金指示確認へ").click()
        await page1.locator("#MEMBERPASSR").click()
        await page1.locator("#MEMBERPASSR").fill(PASSWORD)
        await page1.get_by_role("button", name="入金指示する").click()
        await page1.close()

        # ---------------------
        await context.close()
        await browser.close()
    print("入金が終了しました")

# if __name__ == "__main__":
#     asyncio.run(auto_in_money())
