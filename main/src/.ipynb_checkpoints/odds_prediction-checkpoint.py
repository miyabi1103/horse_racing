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
        10: '小倉'
    }
    async with async_playwright() as playwright:
        kaisai_name = f"{int(race_id[6:8])}回{place_mapping[int(race_id[4:6])]}{int(race_id[8:10])}日"
        race_name = f"{int(race_id[10:12])}レース"
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("https://jra.jp/keiba/")
        await page.get_by_role("link", name="オッズ", exact=True).click()
        await page.get_by_role("link", name=kaisai_name).click()
        async with page.expect_navigation():
            await page.get_by_role("link", name=race_name, exact=True).click()
            # ページ内のリンクテキストをすべて取得して、何があるのか確認する
        html = await page.content()
        html = StringIO(html)
        odds = (
            pd.read_html(html)[0][["馬番","単勝"]]
            .set_index("馬番")["単勝"]
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