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
from Auto_in_money import TicketsPurchaser
from keiba_prediction import Auto_purchaser_sanrenpuku
from keiba_prediction import Auto_purchaser_sanrentan
from keiba_prediction import Auto_purchaser_umaren
from keiba_prediction import Auto_purchaser_wide
from keiba_prediction import Auto_purchaser_tansho_dirt
from keiba_prediction import Auto_purchaser_tansho_obstract
from keiba_prediction import Auto_purchaser_tansho_turf
from keiba_prediction import Auto_purchaser_tansho_turf_nowin
#ここに投票を行う遷移の関数をいれる
# def scrape_job(race_id:str,scraper:Auto_prediction):
#     print(f"scraping auto {race_id}")
#     asyncio.run(scraper.Create_time_table(race_id=race_id))

def scrape_job(race_id:str, row: dict):
    #ここに予測の実行、投票の実行のプログラムをいれる
    #その前に、pre_predict_exeを実行したいが、それは一日に一度で良い,10分前にやるとか
    #まずは普通に開発し、そのあとにifでレースタイプごとに分ける
    # 条件分岐を追加
    

    if row['EARLY'] == 1 and row["type"] == 1 and row["class"] >= 2:
        predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord()
        #購入手続き
        money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "800",amount_num = "8"))
        money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))

        predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord()
        #購入手続き
        money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
        money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))



    elif row['EARLY'] == 1 and row["type"] == 1 and row["class"] < 2:
        predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord()
        #購入手続き
        money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
        money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))


    elif row['EARLY'] == 1 and row["type"] == 0 and row["class"] >= 2:
        predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord()
        money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "200",amount_num = "2"))
        # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))


    elif row['EARLY'] == 1 and row["type"] == 0 and row["class"] < 2:
        predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord()
        money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "200",amount_num = "2"))
        # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))


    elif row['EARLY'] == 1 and row["type"] == 2:
        predict = predict_exe_obstract.def_predict_exe_obstract(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord()
        money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "900",amount_num = "9"))
        money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "400",amount_num = "2"))



    elif row['EARLY'] == 0 and row["type"] == 1 and row["class"] >= 2:
        predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
        money = asyncio.run(Auto_purchaser_tansho_turf.Auto_purchase_tansho_turf(race_id=race_id,amount = "300",amount_num = "3"))
        print("demo")


    elif row['EARLY'] == 0 and row["type"] == 1 and row["class"] < 2:
        predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
        money = asyncio.run(Auto_purchaser_tansho_turf_nowin.Auto_purchase_tansho_turf_nowin(race_id=race_id,amount = "200",amount_num = "2"))
        print("demo")


    elif row['EARLY'] == 0 and row["type"] == 0 and row["class"] >= 2:
        predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
        money = asyncio.run(Auto_purchaser_tansho_dirt.Auto_purchase_tansho_dirt(race_id=race_id,amount = "200",amount_num = "2"))
        print("demo")


    elif row['EARLY'] == 0 and row["type"] == 0 and row["class"] < 2:
        print("dirt_nowin")


    elif row['EARLY'] == 0 and row["type"] == 2:
        predict = predict_exe_obstract.def_predict_exe_obstract(kaisai_date=args.kaisai_date, race_id=race_id)
        money = asyncio.run(Auto_purchaser_tansho_obstract.Auto_purchase_tansho_obstract(race_id=race_id,amount = "900",amount_num = "9"))
        print("demo")

    elif row["type"] == "none":
        
        predict = pre_predict_exe.prepredict(kaisai_date=args.kaisai_date)
        discord_notify = discord.post_discord() 
        purchaser = TicketsPurchaser()
        money = purchaser.auto_in_money()
        # print("demo")
        
    elif row["type"] == "end":
        print("スケジューラを停止します...")
        scheduler.shutdown(wait=False)  # スケジューラを停止
        return
        

    else:
        predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord() 
    print(f"scraping auto {race_id}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kaisai_date",
        type=str,
        default=datetime.now().strftime("%Y%m%d"),  
    )
    args = parser.parse_args()

    print(f"指定された開催日: {args.kaisai_date}")


    # Vote = 



    # scraper = Create_time_table(kaisai_date=args.kaisai_date)
    time_table = asyncio.run(Auto_prediction.Create_time_table(kaisai_data=args.kaisai_date))
    time_table_dev = time_table.copy()


    # コピー
    time_table_dev = time_table.copy()

    # 新しい行を作る：post_timeを11分前にして、EARLY=1
    early_rows = time_table_dev.copy()
    # 1. 文字列を datetime に変換
    early_rows['post_time'] = pd.to_datetime(early_rows['post_time'])

    early_rows['post_time'] = early_rows['post_time'] - pd.Timedelta(minutes=8)

    # 3. str に戻す（フォーマットは例として "HH:MM" にしてます）
    early_rows['post_time'] = early_rows['post_time'].dt.strftime('%H:%M')
    
    early_rows['EARLY'] = 1

    # 元の行にはEARLY=0を設定
    time_table_dev['EARLY'] = 0

    # 結合して行数を2倍に
    time_table_dev = pd.concat([time_table_dev, early_rows], ignore_index=True)

    # post_timeで昇順にソート
    time_table_dev = time_table_dev.sort_values(by='post_time').reset_index(drop=True)


    # ###########################################
    # #開発用
    # time_table_dev["post_time"] = [
    #     (datetime.now() + timedelta(minutes=3 * i + 7)).strftime("%H:%M")
    #     for i in range(len(time_table))
    # ]







    # 一番手前のpost_timeを取得
    first_post_time = datetime.strptime(time_table_dev.iloc[0]["post_time"], "%H:%M")

    # # 1時間前の時間を計算
    # new_post_time = (first_post_time - timedelta(minutes=60)).strftime("%H:%M")

    # 現在時刻を取得
    current_time = datetime.now()

    # 現在時刻の1分後を計算
    new_post_time = (current_time + timedelta(minutes=7)).strftime("%H:%M")


    # 新しい行を作成
    new_row = {col: "none" for col in time_table_dev.columns}  # すべての列に100を設定
    new_row["post_time"] = new_post_time  # post_timeだけ1時間前の時間を設定

    # 新しい行をDataFrameに追加
    time_table_dev = pd.concat([pd.DataFrame([new_row]), time_table_dev], ignore_index=True)

    # 最後の行から20分後のpost_timeを持つ行を追加
    last_post_time = datetime.strptime(time_table_dev.iloc[-1]["post_time"], "%H:%M")
    end_post_time = (last_post_time + timedelta(minutes=20)).strftime("%H:%M")
    end_row = {col: "none" for col in time_table_dev.columns}
    end_row["post_time"] = end_post_time
    end_row["type"] = "end"
    time_table_dev = pd.concat([time_table_dev, pd.DataFrame([end_row])], ignore_index=True)


    print(time_table_dev)


    scheduler = BlockingScheduler()

    
    for idx, row in time_table_dev.iterrows():
        race_id = row["race_id"]
        post_time = datetime.strptime(row["post_time"],"%H:%M").time()
        run_at = (
            datetime.combine(datetime.now(),post_time)
            - timedelta(minutes = 3)
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