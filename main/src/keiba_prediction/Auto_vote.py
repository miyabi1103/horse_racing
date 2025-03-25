import asyncio
import argparse
from datetime import datetime,timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
import Auto_prediction
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from keiba_prediction import pre_predict_exe
from keiba_prediction import predict_exe_turf
from keiba_prediction import predict_exe_turf_nowin
from keiba_prediction import predict_exe_dirt
from keiba_prediction import predict_exe_dirt_nowin
from keiba_prediction import predict_exe_obstract
import pandas as pd
from keiba_notify import discord
#ここに投票を行う遷移の関数をいれる
# def scrape_job(race_id:str,scraper:Auto_prediction):
#     print(f"scraping auto {race_id}")
#     asyncio.run(scraper.Create_time_table(race_id=race_id))

def scrape_job(race_id:str, row: dict):
    #ここに予測の実行、投票の実行のプログラムをいれる
    #その前に、pre_predict_exeを実行したいが、それは一日に一度で良い,10分前にやるとか
    #まずは普通に開発し、そのあとにifでレースタイプごとに分ける
    # 条件分岐を追加

    if row["type"] == 1 and row["class"] >= 2:
        predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord()

    elif row["type"] == 1 and row["class"] < 2:
        predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord()

    elif row["type"] == 0 and row["class"] >= 2:
        predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord()

    elif row["type"] == 0 and row["class"] < 2:
        predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord()

    elif row["type"] == 2:
        predict = predict_exe_obstract.def_predict_exe_obstract(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord()

    elif row["type"] == "none":
        predict = pre_predict_exe.prepredict(kaisai_date=args.kaisai_date)


    else:
        predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord() 
    print(f"scraping auto {race_id}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kaisai_date",
        type=str,
        required=True,
    )
    args = parser.parse_args()


    # Vote = 



    # scraper = Create_time_table(kaisai_date=args.kaisai_date)
    time_table = asyncio.run(Auto_prediction.Create_time_table(kaisai_data=args.kaisai_date))
    time_table_dev = time_table.copy()

    # #開発用
    # time_table_dev["post_time"] = [
    #     (datetime.now() + timedelta(minutes=1 * i + 20)).strftime("%H:%M")
    #     for i in range(len(time_table))
    # ]


    # 一番手前のpost_timeを取得
    first_post_time = datetime.strptime(time_table_dev.iloc[0]["post_time"], "%H:%M")

    # 1時間前の時間を計算
    new_post_time = (first_post_time - timedelta(minutes=120)).strftime("%H:%M")

    # 新しい行を作成
    new_row = {col: "none" for col in time_table_dev.columns}  # すべての列に100を設定
    new_row["post_time"] = new_post_time  # post_timeだけ1時間前の時間を設定

    # 新しい行をDataFrameに追加
    time_table_dev = pd.concat([pd.DataFrame([new_row]), time_table_dev], ignore_index=True)


    print(time_table_dev)


    scheduler = BlockingScheduler()

    
    for idx, row in time_table_dev.iterrows():
        race_id = row["race_id"]
        post_time = datetime.strptime(row["post_time"],"%H:%M").time()
        run_at = (
            datetime.combine(datetime.now(),post_time)
            - timedelta(minutes = 6)
            - timedelta(seconds = 0)
            # それより何分前に実行させるか
        )
        scheduler.add_job(
            func=scrape_job,
            trigger="date",
            run_date=run_at,
            args=[race_id, row]
        )

    try:
        scheduler.start()
    except (KeyboardInterrupt,SystemExit):
        scheduler.shutdown()
        print("停止しました")









        # for i in range(-5,2):
        #     run_at = (
        #         datetime.combine(datetime.now(),post_time)
        #         + timedelta(minutes=i) #
        #         + timedelta(seconds=20)
        #     )