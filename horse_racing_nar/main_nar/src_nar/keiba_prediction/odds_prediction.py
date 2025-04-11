import asyncio
import re
from playwright.async_api import Playwright, async_playwright, expect
import pandas as pd
from io import StringIO

async def update_odds_and_popularity(features,race_id:str):
        
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
        10: '小倉',
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
        place_count = f"{int(race_id[10:12])}"
        place_name = f"{place_mapping[int(race_id[4:6])]}"
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/TodayRaceInfoTop")
        await page.get_by_role("cell", name=place_name, exact=True).click()
        async with page.expect_navigation():
            # await page.get_by_role("row", name="1R 14:15 Ｃ３三 左1200m 晴 重 12").get_by_role("link").nth(1).click()
            await page.locator("tr").filter(has_text=re.compile(fr"\b{place_count}R\b")).locator("a").nth(1).click()
            # ページ内のリンクテキストをすべて取得して、何があるのか確認する
        html = await page.content()
        html = StringIO(html)
        odds = (
            pd.read_html(html)[0][["馬番","単勝 オッズ"]]
            .set_index("馬番")["単勝 オッズ"]
            .to_dict()
        )
        # ---------------------
        await context.close()
        await browser.close()


    df = features.copy()
    df["tansho_odds"] = df["umaban"].map(odds)
    df["tansho_odds"] = pd.to_numeric(df["tansho_odds"], errors="coerce")

    df["popularity"] = df["tansho_odds"].rank(ascending = True,method ="min")

    df_reload = df
    return df_reload

# import asyncio
# import re
# from playwright.async_api import Playwright, async_playwright, expect
# import pandas as pd
# from io import StringIO
# place_mapping = {
#     1: '札幌',
#     2: '函館',
#     3: '福島',
#     4: '新潟',
#     5: '東京',
#     6: '中山',
#     7: '中京',
#     8: '京都',
#     9: '阪神',
#     10: '小倉'
# }
# race_id = "202506010401"
# playwright = await async_playwright().start()
# async def run(race_id : str,playwright: Playwright) -> None:
#     kaisai_name = f"{int(race_id[6:8])}回{place_mapping[int(race_id[4:6])]}{int(race_id[8:10])}日"
#     race_name = f"{int(race_id[10:12])}レース"
#     browser = await playwright.chromium.launch(headless=False)
#     context = await browser.new_context()
#     page = await context.new_page()
#     await page.goto("https://jra.jp/keiba/")
#     await page.get_by_role("link", name="オッズ", exact=True).click()
#     await page.get_by_role("link", name=kaisai_name).click()
#     async with page.expect_navigation():
#         await page.get_by_role("link", name=race_name, exact=True).click()
#         # ページ内のリンクテキストをすべて取得して、何があるのか確認する
#     html = await page.content()
#     html = StringIO(html)
#     odds = pd.read_html(html)[0][["馬番","単勝"]]
    
#     # ---------------------
#     await context.close()
#     await browser.close()

#     return odds

# async with async_playwright() as playwright:
#     odds = await run(race_id, playwright)