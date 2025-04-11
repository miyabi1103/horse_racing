import asyncio
import re
from playwright.async_api import Playwright, async_playwright, expect
import pandas as pd
from io import StringIO
from collections import defaultdict
from bs4 import BeautifulSoup
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from keiba_prediction import predict_exe_turf
from keiba_prediction import predict_exe_turf_nowin
from keiba_prediction import predict_exe_dirt
from keiba_prediction import predict_exe_dirt_nowin
from keiba_prediction import predict_exe_obstract


async def Create_time_table(kaisai_data:str):
    race_type_mapping = {
        "ダ": 0,
        "芝": 1,
        "障": 2,
        "ジャンプ": 2,
        "JS": 2
    }
    race_type_mapping2 = {
        "芝": 1,
        "障": 2,
        "ジャンプ": 2,
        "JS": 2
    }
    race_class_mapping = {
        "新馬": 0,
        "未勝利": 1,
        "1勝クラス": 2,
        "１勝クラス": 2,
        "2勝クラス": 3,
        "２勝クラス": 3,
        "3勝クラス": 4,
        "３勝クラス": 4,
        "オープン": 5,
        "G3": 6,
        "G2": 7,
        "G1": 8,
        "GIII": 6,
        "GII": 7,
        "GI": 8,
        "重賞": 6,    
        "GⅢ": 6,
        "GⅡ": 7,
        "GⅠ": 8,
        "L":5,
        "OP": 5,
        "特別":5,
        "500万下": 2,
        "1000万下": 3,
        "1600万下": 4
        }

    async with async_playwright() as playwright:
        # playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(
            f"https://race.netkeiba.com/top/race_list.html?kaisai_date={kaisai_data}",
            wait_until="domcontentloaded",
            # wait_until= "load",
           
            )
        await page.wait_for_selector("li.RaceList_DataItem", timeout=60000) 
        
        # html = await page.content()
        # html = StringIO(html)
        race_list = page.locator("li.RaceList_DataItem")

        time_table_dict = defaultdict(list)
        
        count = await race_list.count()
        print(count)
        for i in range(await race_list.count()):
            race = race_list.nth(i)
            #race_idの取得
            href = await race.locator("a").first.get_attribute("href")
            race_id = re.findall(r"race_id=(\d{12})",href)[0]
            time_table_dict["race_id"].append(race_id)
            #距離の取得
            # long = await race.locator("span.RaceList_ItemLong").inner_text()
            # time_table_dict["long_type"].append(long.strip())
            # time_table_dict["long"] = (
            #     re.search(r"(\d+)m")
            #     .astype(int)
            # )
            # long_text = await race.locator("span.RaceList_ItemLong").inner_text()
            # time_table_dict["long_type"].append(long_text.strip())

            # long_match = re.search(r"(\d+)m", long_text)
            # long_value = int(long_match.group(1)) if long_match else None
            # time_table_dict["long"].append(long_value)

            span_tags = await race.locator("div.RaceData").locator("span").all_text_contents()
            
            for long_text in span_tags:
                # 長さ情報が含まれるテキストを取り出す
                long_text = long_text.strip()
                
                # "m"が含まれるテキストを抽出
                if 'm' in long_text:
                    time_table_dict["long_type"].append(long_text)
            
                    # 正規表現を使って距離部分（m前の数値）を取得
                    long_match = re.search(r"(\d+)m", long_text)
                    
                    # 見つかった場合はその数値を整数として取得
                    long_value = int(long_match.group(1)) if long_match else None
                    time_table_dict["long"].append(long_value)
            
                    
            # #タイプ,グレードの取得
            title = await race.locator("span.ItemTitle").inner_text()
            time_table_dict["title"].append(title.strip())
            
            # time_table_dict["type"] = (
            #     time_table_dict["title"]
            #     .str.extract(rf"({race_type_mapping})")
            #     .fillna(time_table_dict["long_type"].str.extract(rf"({race_type_mapping})"))
            # )
            # time_table_dict["class"] = (
            #     time_table_dict["title"]
            #     .str.extract(rf"({race_class_mapping})")
            #     .fillna(3)
            # )
            
            # レースの種類 (ダート、芝など)
            race_type_pattern = "|".join(race_type_mapping2.keys())
            race_type_match = re.search(race_type_pattern, title)
            race_type = race_type_mapping2.get(race_type_match.group(0), None) if race_type_match else None


            time_table_dict["type"].append(race_type)

            # レースクラスの取得
            race_class_pattern = "|".join(race_class_mapping.keys())
            race_class_match = re.search(race_class_pattern, title)
            race_class = race_class_mapping.get(race_class_match.group(0), 3) if race_class_match else 3
            time_table_dict["class"].append(race_class)
            
            #発走時刻の取得
            post_time = await race.locator("span.RaceList_Itemtime").inner_text()
            time_table_dict["post_time"].append(post_time.strip())
           


        # ---------------------
        await context.close()
        await browser.close()
    df = pd.DataFrame(time_table_dict)
    # type列がNoneの行に対して処理
    for index, row in df[df["type"].isna()].iterrows():
        # long_type列の文字列からレース種類を抽出
        long_text = row["long_type"]
        race_type_match = re.search(r"(ダ|芝|障|ジャンプ|JS)", long_text)
        
        if race_type_match:
            race_type = race_type_mapping.get(race_type_match.group(0), None)
            # type列に数字を割り振る
            df.at[index, "type"] = race_type
    df['type'] = df['type'].astype('Int64') 
    df['class'] = df['class'].astype('Int64') 
    df['long'] = df['long'].astype('Int64') 
    
    df = df.sort_values(by="post_time", ascending=True)

    #ここで、新馬とダートを抜いても良い
    # df = df[df["type"] != 0]
    df = df[df["class"] != 0]

    # typeが0かつclassが1かつlongが1900未満の行を除外
    df = df[~((df["type"] == 0) & (df["class"] == 1) & (df["long"] < 1900))]


    place_mapping = {
        '01': 'Sapporo',
        '02': 'Hakodate',
        '03': 'Fukushima',
        '04': 'Niigata',
        '05': 'Tokyo',
        '06': 'Nakayama',
        '07': 'Chukyo',
        '08': 'Kyoto',
        '09': 'Hanshin',
        '10': 'Kokura'
    }
    from datetime import datetime

    # 曜日を取得するためのマッピング
    weekday_mapping = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    # RACE_INFO列を作成
    df["RACE_INFO"] = df["race_id"].apply(
        lambda race_id: f"{datetime.now().strftime('%Y/%m/%d')}({weekday_mapping[datetime.now().weekday()]}) "
                        f"{place_mapping[str(race_id[4:6])]} No.{int(str(race_id[10:12]))}"
    )
    # RACE_INFO列を一番左に移動
    columns = ["RACE_INFO"] + [col for col in df.columns if col != "RACE_INFO"]
    df = df[columns]
    # prediction_df = prediction_df.drop(columns=["race_id"])


    # データフレームをPNG形式で保存する関数
    def save_csv_as_image(dataframe: pd.DataFrame, output_file: Path):
        import matplotlib
        matplotlib.use("Agg")  # GUIバックエンドを無効化
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.rcParams["font.family"] = "IPAexGothic"

        # 背景色とスタイルを設定
        plt.rcParams.update({
            "figure.facecolor": "black",  # 背景色を黒に設定
            "axes.facecolor": "black",   # 軸の背景色を黒に設定
            "text.color": "white",       # テキストの色を白に設定
            "axes.labelcolor": "white",  # 軸ラベルの色を白に設定
            "xtick.color": "white",      # x軸目盛りの色を白に設定
            "ytick.color": "white",      # y軸目盛りの色を白に設定
        })

        # Seaborn のスタイル適用
        sns.set_style("dark")

        # プロットの作成
        fig, ax = plt.subplots(figsize=(len(dataframe.columns) * 1.2, len(dataframe) * 0.6))  # サイズ調整
        ax.axis("tight")
        ax.axis("off")

        # テーブルを描画
        table = ax.table(
            cellText=dataframe.values,
            colLabels=dataframe.columns,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1]  # テーブル全体を中央配置
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # 列幅を適切に調整
        for i in range(len(dataframe.columns)):
            table.auto_set_column_width([i])

        # テーブルのデザイン（完全ダークモード）
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#555555")  # 枠線をダークグレーに
            cell.set_linewidth(0.5)  # 線を細く

            if row == 0:  # ヘッダー部分
                cell.set_facecolor("#1F354D")  # ヘッダーの背景色をダークグレーに設定
                cell.set_text_props(weight="bold", color="white")  # ヘッダーの文字色を白に設定
            else:
                # 偶数・奇数行で色を変える
                if row % 2 == 0:
                    cell.set_facecolor("#333333")  # 偶数行の背景色を濃いグレーに設定
                else:
                    cell.set_facecolor("#222222")  # 奇数行の背景色をさらに暗いグレーに設定
                cell.set_text_props(color="white")  # テキストを白に設定

        # 画像を保存
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0, facecolor="black")
        plt.close()

    # 保存先ディレクトリとファイル名
    DATA_DIR = Path("..", "..", "data")
    SAVE_DIR = DATA_DIR / "05_prediction_results"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)  # ディレクトリを作成
    output_image = SAVE_DIR / "prediction_result.png"

    # データフレームをPNG形式で保存
    save_csv_as_image(df, output_image)
    print(f"データフレームを画像として保存しました: {output_image}")
    
    return df




    # async with async_playwright() as playwright:
    #     # playwright = await async_playwright().start()
    #     browser = await playwright.chromium.launch(headless=True)
    #     context = await browser.new_context()
    #     page = await context.new_page()

    #     await page.goto(
    #         f"https://race.netkeiba.com/top/race_list.html?kaisai_date={kaisai_data}",
    #         wait_until="domcontentloaded",
    #         )
    #     html = await page.content()
    #     # print(html)
    #     # html = StringIO(html)
    #     await page.wait_for_selector("li.RaceList_DataItem", timeout=60000) 
        

    #     # ---------------------
    #     await context.close()
    #     await browser.close()
    # soup = BeautifulSoup(html, 'html.parser')
    # # print(soup.prettify())
        
    # # レースリストをHTMLから取得
    # race_list = soup.select("li.RaceList_DataItem")
    # print(race_list)

    # time_table_dict = defaultdict(list)

    # for race in race_list:
    #     # race_idの取得
    #     href = race.find("a")["href"]
    #     race_id = re.findall(r"race_id=(\d{12})", href)[0]
    #     time_table_dict["race_id"].append(race_id)

    #     # 距離の取得
    #     long_text = race.find("span", class_="RaceList_ItemLong").get_text(strip=True)
    #     time_table_dict["long_type"].append(long_text)

    #     long_match = re.search(r"(\d+)m", long_text)
    #     long_value = int(long_match.group(1)) if long_match else None
    #     time_table_dict["long"].append(long_value)

    #     # レースタイトルの取得
    #     title = race.find("span", class_="ItemTitle").get_text(strip=True)
    #     time_table_dict["title"].append(title)

    #     # レースの種類 (ダート、芝など)
    #     race_type_pattern = "|".join(race_type_mapping.keys())
    #     race_type_match = re.search(race_type_pattern, title)
    #     race_type = race_type_mapping.get(race_type_match.group(0), None) if race_type_match else None
    #     time_table_dict["type"].append(race_type)

    #     # レースクラスの取得
    #     race_class_pattern = "|".join(race_class_mapping.keys())
    #     race_class_match = re.search(race_class_pattern, title)
    #     race_class = race_class_mapping.get(race_class_match.group(0), 3) if race_class_match else 3
    #     time_table_dict["class"].append(race_class)

    #     # 発走時刻の取得
    #     post_time = race.find("span", class_="RaceList_Itemtime").get_text(strip=True)
    #     time_table_dict["post_time"].append(post_time)
    # # pandasのDataFrameに変換
    # df = pd.DataFrame(time_table_dict)

    # return df







    