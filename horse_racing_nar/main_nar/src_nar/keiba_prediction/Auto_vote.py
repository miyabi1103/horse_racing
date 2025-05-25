import asyncio
import argparse
from datetime import datetime,timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src_nar"))
import pandas as pd

import predict_exe_funabasi
import predict_exe_monbetu
import predict_exe_kanazawa
import pre_predict_exe
import predict_exe_kasamatu
import predict_exe_kawasaki
import predict_exe_kouti
import predict_exe_mizusawa
import predict_exe_nagoya
import predict_exe_morioka
import predict_exe_sonoda
import predict_exe_urawa
import predict_exe_ooi
import predict_exe_saga

from keiba_notify import discord
from keiba_notify import discord_2


import Auto_prediction
import Auto_in_money
import Auto_purchaser_sanrenpuku
import Auto_purchaser_wide
import Auto_purchaser_umaren
#ここに投票を行う遷移の関数をいれる
# def scrape_job(race_id:str,scraper:Auto_prediction):
#     print(f"scraping auto {race_id}")
#     asyncio.run(scraper.Create_time_table(race_id=race_id))

def scrape_job(race_id:str, row: dict):
    #ここに予測の実行、投票の実行のプログラムをいれる
    # Tier1 = dict(race_id=race_id,amount = "4000",amount_num = "40")
    # Tier2 = dict(race_id=race_id,amount = "3000",amount_num = "30")
    # Tier3 = dict(race_id=race_id, amount="2500", amount_num="25")
    # Tier4 = dict(race_id=race_id,amount = "2000",amount_num = "20")
    # Tier5 = dict(race_id=race_id,amount = "1200",amount_num = "12")
    # Tier6 = dict(race_id=race_id, amount="1000", amount_num="10")
    # Tier7 = dict(race_id=race_id,amount = "800",amount_num = "8")
    # Tier8 = dict(race_id=race_id,amount = "600",amount_num = "6")
    # Tier9 = dict(race_id=race_id,amount = "400",amount_num = "4")

    #まずは一年やってから増やすこと
    Tier1 = dict(race_id=race_id,amount = "5000",amount_num = "50")
    Tier2 = dict(race_id=race_id,amount = "4000",amount_num = "40")
    Tier3 = dict(race_id=race_id, amount="3000", amount_num="30")
    Tier4 = dict(race_id=race_id,amount = "2500",amount_num = "25")
    Tier5 = dict(race_id=race_id,amount = "1700",amount_num = "17")
    Tier6 = dict(race_id=race_id, amount="1300", amount_num="13")
    Tier7 = dict(race_id=race_id,amount = "1000",amount_num = "10")
    Tier8 = dict(race_id=race_id,amount = "800",amount_num = "8")
    Tier9 = dict(race_id=race_id,amount = "600",amount_num = "6")


    # "札幌": 1,
    # "函館": 2,
    # "福島": 3,
    # "新潟": 4,
    # "東京": 5,
    # "中山": 6,
    # "中京": 7,
    # "京都": 8,
    # "阪神": 9,
    # "小倉": 10,
    # "門別": 30,
    # "盛岡": 35,
    # "水沢": 36,
    # "浦和": 42,
    # "船橋": 43,
    # "大井": 44,
    # "川崎": 45,
    # "金沢": 46,
    # "笠松": 47,
    # "名古屋": 48,
    # "園田": 50,
    # "高知": 54,
    # "姫路": 51,
    # "佐賀": 55

    if row["place"] == 30:  
        predict = predict_exe_monbetu.def_predict_exe_monbetu(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier2))

    elif row["place"] == 35:  
        predict = predict_exe_morioka.def_predict_exe_morioka(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier2))


    elif row["place"] == 36:  
        predict = predict_exe_mizusawa.def_predict_exe_mizusawa(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier6))


    elif row["place"] == 42:  
        predict = predict_exe_urawa.def_predict_exe_urawa(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier4))

    elif row["place"] == 43:  
        predict = predict_exe_funabasi.def_predict_exe_funabasi(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier3))


    elif row["place"] == 44:  
        predict = predict_exe_ooi.def_predict_exe_ooi(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier1))


    elif row["place"] == 45:  
        predict = predict_exe_kawasaki.def_predict_exe_kawasaki(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier5))


    elif row["place"] == 46:  
        predict = predict_exe_kanazawa.def_predict_exe_kanazawa(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier6))



    elif row["place"] == 47:  
        predict = predict_exe_kasamatu.def_predict_exe_kasamatu(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier7))


    elif row["place"] == 48:  
        predict = predict_exe_nagoya.def_predict_exe_nagoya(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier2))



    elif row["place"] == 50:  
        predict = predict_exe_sonoda.def_predict_exe_sonoda(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier3))


    elif row["place"] == 51:  
        predict = predict_exe_sonoda.def_predict_exe_sonoda(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier3))


    elif row["place"] == 54:  
        predict = predict_exe_kouti.def_predict_exe_kouti(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier8))

    elif row["place"] == 55:  
        predict = predict_exe_saga.def_predict_exe_saga(kaisai_date=args.kaisai_date, race_id=race_id)  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        #購入手続き
        money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier9))


    elif row["type"] == "none":  
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        predict = pre_predict_exe.prepredict(kaisai_date=args.kaisai_date)  
        money = asyncio.run(Auto_in_money.auto_in_money())
        # print("demo")
        
    elif row["type"] == "end":
        print("スケジューラを停止します...")
        scheduler.shutdown(wait=False)  # スケジューラを停止
        return
    

    else:
        print("error")
        

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


    # ###########################################
    # #開発用
    # time_table_dev["post_time"] = [
    #     (datetime.now() + timedelta(minutes=1 * i + 21)).strftime("%H:%M")
    #     for i in range(len(time_table))
    # ]
    # ###########################################


    # 一番手前のpost_timeを取得
    first_post_time = datetime.strptime(time_table_dev.iloc[0]["post_time"], "%H:%M")


    # 現在時刻を取得
    current_time = datetime.now()

    # 現在時刻の1分後を計算
    new_post_time = (current_time + timedelta(minutes=21)).strftime("%H:%M")


    # 新しい行を作成
    new_row = {col: "none" for col in time_table_dev.columns}  # すべての列に100を設定
    new_row["post_time"] = new_post_time  # post_timeだけ1時間前の時間を設定

    # 新しい行をDataFrameに追加
    time_table_dev = pd.concat([pd.DataFrame([new_row]), time_table_dev], ignore_index=True)


    # 最後の行から20分後のpost_timeを持つ行を追加
    last_post_time = datetime.strptime(time_table_dev.iloc[-1]["post_time"], "%H:%M")
    end_post_time = (last_post_time + timedelta(minutes=10)).strftime("%H:%M")
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
            - timedelta(minutes = 20)
            # - timedelta(seconds = 10)
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