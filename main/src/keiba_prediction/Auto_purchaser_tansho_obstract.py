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

from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # GUIバックエンドを無効化
import matplotlib.pyplot as plt
import seaborn as sns
DATA_DIR = Path("..", "..", "data")
SAVE_DIR = DATA_DIR / "05_prediction_results"
from dotenv import load_dotenv

load_dotenv(override=True)
# .envファイルを読み込む
load_dotenv()
import os
# DiscordのWebhook URLを環境変数から取得
INET_URL = os.getenv("INET_ID")
KANYUSYA_URL = os.getenv("KANYUSYA_NO")
PASSWORD_URL = os.getenv("PASSWORD_PAT")
PARS_URL = os.getenv("PARS_NO")

if not INET_URL:
    raise ValueError("INET_IDが設定されていません。'.env'ファイルを確認してください。")
if not KANYUSYA_URL:
    raise ValueError("KANYUSYA_NOが設定されていません。'.env'ファイルを確認してください。")
if not PASSWORD_URL:
    raise ValueError("PASSWORD_PATが設定されていません。'.env'ファイルを確認してください。")
if not PARS_URL:
    raise ValueError("PARS_NOが設定されていません。'.env'ファイルを確認してください。")


csv_file_path = Path("../../data_nar/05_prediction_results/prediction_result.csv")


# if not INET_URL:
#     raise ValueError("INET_IDが設定されていません。'.env'ファイルを確認してください。")
# if not KANYUSYA_URL:
#     raise ValueError("KANYUSYA_NOが設定されていません。'.env'ファイルを確認してください。")
# if not PASSWORD_URL:
#     raise ValueError("PASSWORD_PATが設定されていません。'.env'ファイルを確認してください。")
# if not PARS_URL:
#     raise ValueError("PARS_NOが設定されていません。'.env'ファイルを確認してください。")

async def Auto_purchase_tansho_obstract(race_id:str,csv_path: Path,amount: str = "100",amount_num: str = "1"):
    csv_path = Path("../../data/05_prediction_results/prediction_result.csv")
    """
    【障害】条件を満たすumaban列の値を抽出
    条件: pred > 0.1, tansho_odds < 100, 期待値(Ex_value) > 1
    """
    try:
        df = pd.read_csv(csv_path, sep="\t")
        filtered = df[(df["pred"] > 0.1) & (df["tansho_odds"] < 100) & (df["Ex_value"] > 1)]
        umaban_list = filtered["umaban"].tolist()

    except Exception as e:
        return 
    
    place_mapping = {
        1: '札幌',
        2: '函館',
        3: '福島',
        4: '新潟',
        5: '東京',
        6: '中山',
        7: '中京',
        8: '京都',
        9: '阪神',
        10: '小倉'
    }

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        place_count = f"{int(race_id[10:12])}"
        place_count_minus = int(race_id[10:12])- 1
        place_name = f"{place_mapping[int(race_id[4:6])]}"
        weekday_map = ["月","火", "水", "木", "金", "土", "日"]
        today_weekday = weekday_map[datetime.now().weekday()]
        button_name = f"{place_name}（{today_weekday}）"
        race_button_name = f"{place_count}R"
        await page.goto("https://jra.jp/")

        async with page.expect_popup() as page1_info:
            await page.get_by_role("link", name="JRAの馬券をネット投票 即 PAT A-PAT").click()
        page1 = await page1_info.value
        await page1.get_by_role("textbox").click()
        await page1.get_by_role("textbox").fill(INET_URL)
        await page1.get_by_role("link", name="ログイン").click()
        await page1.locator("input[name=\"i\"]").fill(KANYUSYA_URL)
        await page1.locator("input[name=\"i\"]").click()
        await page1.locator("input[name=\"p\"]").fill(PASSWORD_URL)
        await page1.locator("input[name=\"r\"]").click()
        await page1.locator("input[name=\"r\"]").fill(PARS_URL)
        await page1.get_by_role("link", name="ネット投票メニューへ").click()
        await page1.get_by_role("button", name="マークカード投票").click()
        await page1.get_by_role("button", name=button_name, exact=False).click()
        await page1.locator('button:has-text("R")').nth(place_count_minus).click()
        #単勝
        await page1.locator(".btn-mark").first.click()
        for count in umaban_list:
            if count == 1:
                await page1.locator(".btn-mark-num").first.click()
            elif 2 <= count <= 9:
                await page1.locator(f"div:nth-child({count}) > .btn-mark-num").first.click()
            elif count == 10:
                await page1.locator("div:nth-child(2) > div > .btn-mark-num").first.click()
            else:
                one_digit = count % 10
                nth = one_digit + 1
                await page1.locator(f"div:nth-child(2) > div:nth-child({nth}) > .btn-mark-num").first.click()

            await page1.get_by_role("button", name="セット").click()

            await page1.get_by_role("button", name="購入予定リスト").click()
            await page1.get_by_label("円").first.click()
            await page1.get_by_label("円").first.fill(amount_num)
            await page1.get_by_role("row", name="1").get_by_label("円").dblclick()
            await page1.get_by_role("row", name="1").get_by_label("円").fill(amount_num)
            await page1.get_by_role("cell", name="合計金額入力： 円 金額セット用テンキ―").get_by_role("textbox").click()
            await page1.get_by_role("cell", name="合計金額入力： 円 金額セット用テンキ―").get_by_role("textbox").fill(amount)
            await page1.get_by_role("button", name="購入する").click()
            await page1.get_by_role("button", name="OK").click()
            await page1.get_by_role("button", name="閉じる").click()
            await page1.close()
        # ---------------------
        await context.close()
        await browser.close()
    print("単勝投票が完了しました")

# if __name__ == "__main__":
#     asyncio.run(Auto_purchase_sanrenpuku())
