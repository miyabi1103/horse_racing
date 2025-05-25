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
from keiba_notify import discord_2
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
    
    #まずは一年やってから増やすこと
    #     	1	3	5	7	9	11	13	15
    # 1	    0	0	0	0	1	1	2	2
    # 1.2	0	1	1	1	2	3	3	5
    # 1.4	1	2	3	2	3	7	8	12
    # 1.6	1	3	4	4	7	10	15	20
    # 1.8	2	4	5	7	10	15	20	30
    # 2	    3	5	7	10	15	20	30	40


    Tier1 = dict(race_id=race_id,amount = "4000",amount_num = "40")
    Tier2 = dict(race_id=race_id,amount = "3000",amount_num = "30")
    Tier3 = dict(race_id=race_id, amount="2000", amount_num="20")
    Tier4 = dict(race_id=race_id,amount = "1500",amount_num = "15")
    Tier5 = dict(race_id=race_id,amount = "1200",amount_num = "12")
    Tier6 = dict(race_id=race_id, amount="1000", amount_num="10")
    Tier7 = dict(race_id=race_id,amount = "800",amount_num = "8")
    Tier8 = dict(race_id=race_id,amount = "700",amount_num = "7")
    Tier9 = dict(race_id=race_id,amount = "500",amount_num = "5")
    Tier10 = dict(race_id=race_id, amount="400", amount_num="4")
    Tier11 = dict(race_id=race_id,amount = "300",amount_num = "3")
    Tier12 = dict(race_id=race_id,amount = "200",amount_num = "2")
    Tier13 = dict(race_id=race_id,amount = "100",amount_num = "1")

    Tier1_tan = dict(race_id=race_id,amount = "600",amount_num = "3")
    Tier2_tan = dict(race_id=race_id,amount = "400",amount_num = "2")
    Tier3_tan = dict(race_id=race_id,amount = "200",amount_num = "1")
    # **Tier1)



    # 芝					芝展開3,展開ex3



    # 芝未勝利			芝nowin_noweight3,展開3



    # ダート				ダート展開,ダート展開weight/in3



    # ダート未勝利		未勝利展開weight/in3,nowin/noweigt





    if row['EARLY'] == 1 and row["type"] == 1 and row["class"] >= 2:
        if  row['long'] <= 1500:
            if row['place'] == 1:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))

                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier8))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier5))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier13 ))




            elif row['place'] == 2:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "800",amount_num = "8"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))

                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier8))






            elif row['place'] == 3:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())





            elif row['place'] == 4:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier8))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier2_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier5))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier8))


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier10))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier12))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())





            elif row['place'] == 5:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))

                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())






            elif row['place'] == 6:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier3))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier7))

                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier9))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier5))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())





            elif row['place'] == 7:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier3))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier1_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier1))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier4))

                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier5))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier2_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier5))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier5))




            elif row['place'] == 8:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())



                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier9))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier6))





            elif row['place'] == 9:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier5))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier13))

                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku( **Tier9 ))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())





            else:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier4))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier1_tan))                
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier3))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier9))


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku( **Tier9 ))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier10))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren( **Tier4))





        elif 1500 < row['long'] < 1900:
            if row['place'] == 1:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier9))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())




            elif row['place'] == 2:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier2))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier1_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier1))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier13))


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier4))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier1_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier3))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())





            elif row['place'] == 3:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier8))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier2_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier11))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())



            elif row['place'] == 4:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier13))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier13))


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier13))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier13))





            elif row['place'] == 5:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier6))            


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())






            elif row['place'] == 6:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier6))


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())






            elif row['place'] == 7:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier8))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier2_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren( **Tier8))



                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier11))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier11))







            elif row['place'] == 8:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()                
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier11))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier5))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier11))


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier11))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier2_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier11))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier11))








            elif row['place'] == 9:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier4))


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier13))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier13))





            else:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier3))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier11))
                

                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier12))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier12))







        else:
            if row['place'] == 1:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier11))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier7))

                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier12))





            elif row['place'] == 2:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier4))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier2_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier4))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier13))






            elif row['place'] == 3:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier9))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier5))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())

                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())





            elif row['place'] == 4:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier13))







            elif row['place'] == 5:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())








            elif row['place'] == 6:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide( **Tier9))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier13))





            elif row['place'] == 7:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier9))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier12))


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier12))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier13))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())





            elif row['place'] == 8:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier12))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier11))






            elif row['place'] == 9:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier13))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren())

                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier12))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide())
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier13))





            else:
                predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier8))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier8))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier8))


                predict = predict_exe_turf.def_predict_exe_turf_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()

                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier10))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier3_tan))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier8))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier13))




























    elif row['EARLY'] == 1 and row["type"] == 1 and row["class"] < 2:       


        if  row['long'] <= 1500:
            if row['place'] == 1:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "600",amount_num = "6"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "700",amount_num = "7"))



                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "1000",amount_num = "10"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "3"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "400",amount_num = "4"))




            elif row['place'] == 2:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "700",amount_num = "7"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "400",amount_num = "4"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "400",amount_num = "4"))


                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "400",amount_num = "4"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "3"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "400",amount_num = "4"))





            elif row['place'] == 3:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "400",amount_num = "4"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "400",amount_num = "4"))


                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "400",amount_num = "4"))




            elif row['place'] == 4:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))



                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "400",amount_num = "4"))







            elif row['place'] == 5:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))



                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "400",amount_num = "4"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))







            elif row['place'] == 6:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))



                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "400",amount_num = "4"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))






            elif row['place'] == 7:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "500",amount_num = "5"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))



                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "400",amount_num = "4"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))






            elif row['place'] == 8:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "1200",amount_num = "12"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))



                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "400",amount_num = "4"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))






            elif row['place'] == 9:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))




                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "400",amount_num = "4"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))





            else:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))


                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "400",amount_num = "4"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))









        elif 1500 < row['long'] < 1900:
            if row['place'] == 1:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "600",amount_num = "6"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "700",amount_num = "7"))


                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "500",amount_num = "5"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))






            elif row['place'] == 2:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "400",amount_num = "4"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "600",amount_num = "6"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "700",amount_num = "7"))

                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "700",amount_num = "7"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))






            elif row['place'] == 3:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "7"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "700",amount_num = "7"))


                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "500",amount_num = "5"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))





            elif row['place'] == 4:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "7"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "1000",amount_num = "10"))


                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))








            elif row['place'] == 5:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "7"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "1000",amount_num = "10"))


                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))






            elif row['place'] == 6:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "7"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "1000",amount_num = "10"))


                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))





            elif row['place'] == 7:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "1200",amount_num = "12"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "7"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "1000",amount_num = "10"))

                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "700",amount_num = "7"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))






            elif row['place'] == 8:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "7"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "700",amount_num = "7"))


                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))





            elif row['place'] == 9:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "7"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "700",amount_num = "7"))

                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))





            else:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "900",amount_num = "9"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "7"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "700",amount_num = "7"))




                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "1000",amount_num = "10"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "200",amount_num = "2"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))








        else:
            if row['place'] == 1:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "7"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "700",amount_num = "7"))



                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "700",amount_num = "7"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "1200",amount_num = "12"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))






            elif row['place'] == 2:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "500",amount_num = "5"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "900",amount_num = "9"))



                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "1000",amount_num = "10"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "800",amount_num = "8"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))






            elif row['place'] == 3:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "500",amount_num = "5"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))



                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "800",amount_num = "8"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))






            elif row['place'] == 4:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "900",amount_num = "9"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "400",amount_num = "4"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))

                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "800",amount_num = "8"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "400",amount_num = "4"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))






            elif row['place'] == 5:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "400",amount_num = "4"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))


                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "400",amount_num = "4"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))







            elif row['place'] == 6:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))



                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "600",amount_num = "6"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))





            elif row['place'] == 7:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "500",amount_num = "5"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))



                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "400",amount_num = "4"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))





            elif row['place'] == 8:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "500",amount_num = "5"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))


                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "1000",amount_num = "10"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))






            elif row['place'] == 9:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "500",amount_num = "5"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))



                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))







            else:
                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "500",amount_num = "5"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))


                predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin_ex(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                #購入手続き
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))
























    elif row['EARLY'] == 1 and row["type"] == 0 and row["class"] >= 2:

        if  row['long'] <= 1500:
            if row['place'] == 1:
                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))




                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "700",amount_num = "7"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))






            elif row['place'] == 2:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "500",amount_num = "5"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "1500",amount_num = "1500"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))




                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "1000",amount_num = "10"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "1500",amount_num = "1500"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))






            elif row['place'] == 3:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "500",amount_num = "5"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "1500",amount_num = "1500"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))



                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "400",amount_num = "4"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "1000",amount_num = "1000"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





            elif row['place'] == 4:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "500",amount_num = "5"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "1500",amount_num = "1500"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))




                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





            elif row['place'] == 5:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "1500",amount_num = "1500"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))



                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "1000",amount_num = "1000"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





            elif row['place'] == 6:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))



                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "1000",amount_num = "1000"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))




            elif row['place'] == 7:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))



                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "200",amount_num = "200"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





            elif row['place'] == 8:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))



                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "1000",amount_num = "1000"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))






            elif row['place'] == 9:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "700",amount_num = "7"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))




                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "200",amount_num = "200"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))




            else:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "700",amount_num = "7"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))




                # predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                # discord_notify = discord.post_discord()
                # discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "1000",amount_num = "1000"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





        elif 1500 < row['long'] < 1900:
            if row['place'] == 1:
                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))



                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "800",amount_num = "8"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "900",amount_num = "900"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





            elif row['place'] == 2:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))




                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "700",amount_num = "7"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "400",amount_num = "4"))






            elif row['place'] == 3:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))



                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))




            elif row['place'] == 4:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))




                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))






            elif row['place'] == 5:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "500",amount_num = "5"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))



                # predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                # discord_notify = discord.post_discord()
                # discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "800",amount_num = "8"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))




            elif row['place'] == 6:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "800",amount_num = "8"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))




                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))




            elif row['place'] == 7:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))



                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))




            elif row['place'] == 8:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))





                # predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                # discord_notify = discord.post_discord()
                # discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "800",amount_num = "8"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))




            elif row['place'] == 9:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))




                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "400",amount_num = "4"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





            else:

                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "600",amount_num = "6"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))




                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





        else:
            if row['place'] == 1:
                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))


                # predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                # discord_notify = discord.post_discord()
                # discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "800",amount_num = "8"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





            elif row['place'] == 2:
                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))



                # predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                # discord_notify = discord.post_discord()
                # discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "800",amount_num = "8"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))



            elif row['place'] == 3:
                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "500",amount_num = "5"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))


                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "800",amount_num = "8"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "1000",amount_num = "1000"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





            elif row['place'] == 4:
                # predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                # discord_notify = discord.post_discord()
                # discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "500",amount_num = "5"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))


                # predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                # discord_notify = discord.post_discord()
                # discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))



            elif row['place'] == 5:
                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "600",amount_num = "6"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))



                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "400",amount_num = "4"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





            elif row['place'] == 6:
                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "600",amount_num = "6"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))




                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "400",amount_num = "4"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





            elif row['place'] == 7:
                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "600",amount_num = "6"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))




                # predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                # discord_notify = discord.post_discord()
                # discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "800",amount_num = "8"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





            elif row['place'] == 8:
                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))




                # predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                # discord_notify = discord.post_discord()
                # discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "800",amount_num = "8"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





            elif row['place'] == 9:
                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))




                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "300"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "300",amount_num = "3"))





            else:
                predict = predict_exe_dirt.def_predict_exe_dirt(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))




                predict = predict_exe_dirt.def_predict_exe_dirt_tenkai_in3(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "800",amount_num = "8"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "300"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "1500",amount_num = "15"))










    elif row['EARLY'] == 1 and row["type"] == 0 and row["class"] < 2:
        if  row['long'] <= 1500:
            if row['place'] == 1:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "200",amount_num = "2"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "300"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "1200",amount_num = "12"))

            elif row['place'] == 2:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "900",amount_num = "9"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "300"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))

            elif row['place'] == 3:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "300"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))

            elif row['place'] == 4:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "900",amount_num = "9"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "700",amount_num = "700"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))



            elif row['place'] == 5:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "600",amount_num = "6"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "300"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))

            elif row['place'] == 6:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "500",amount_num = "5"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "300"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "100",amount_num = "1"))



            elif row['place'] == 7:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "900",amount_num = "9"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))


            elif row['place'] == 8:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "500",amount_num = "5"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "300"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))

            elif row['place'] == 9:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "300"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))

            else:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "900",amount_num = "9"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))





        elif 1500 < row['long'] < 1900:
            if row['place'] == 1:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))

            elif row['place'] == 2:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "100",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))

            elif row['place'] == 3:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "400",amount_num = "4"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))


            elif row['place'] == 4:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "400",amount_num = "4"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))


            elif row['place'] == 5:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "400",amount_num = "4"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))


            elif row['place'] == 6:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "400",amount_num = "4"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "200",amount_num = "2"))



            elif row['place'] == 7:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))

            elif row['place'] == 8:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))

            elif row['place'] == 9:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "700",amount_num = "7"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))



            else:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))





        else:
            if row['place'] == 1:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))



            elif row['place'] == 2:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))


            elif row['place'] == 3:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))

            elif row['place'] == 4:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))


            elif row['place'] == 5:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))



            elif row['place'] == 6:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))


            elif row['place'] == 7:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "900",amount_num = "9"))

            elif row['place'] == 8:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "900",amount_num = "9"))
                money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "300",amount_num = "300"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))


            elif row['place'] == 9:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))


            else:

                predict = predict_exe_dirt_nowin.def_predict_exe_dirt_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
                discord_notify = discord.post_discord()
                discord_notify_2 = discord_2.post_discord_2()
                # money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(race_id=race_id,amount = "300",amount_num = "3"))
                # money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(race_id=race_id,amount = "200",amount_num = "1"))
                # money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(race_id=race_id,amount = "100",amount_num = "100"))
                # money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(race_id=race_id,amount = "800",amount_num = "8"))






    elif row['EARLY'] == 1 and row["type"] == 2:
        predict = predict_exe_obstract.def_predict_exe_obstract(kaisai_date=args.kaisai_date, race_id=race_id)
        discord_notify = discord.post_discord()
        discord_notify_2 = discord_2.post_discord_2()
        money = asyncio.run(Auto_purchaser_sanrenpuku.Auto_purchase_sanrenpuku(**Tier6))
        money = asyncio.run(Auto_purchaser_sanrentan.Auto_purchase_sanrentan(**Tier2_tan))
        money = asyncio.run(Auto_purchaser_wide.Auto_purchase_wide(**Tier6))
        money = asyncio.run(Auto_purchaser_umaren.Auto_purchase_umaren(**Tier9))


    elif row['EARLY'] == 0 and row["type"] == 1 and row["class"] >= 2:
        predict = predict_exe_turf.def_predict_exe_turf(kaisai_date=args.kaisai_date, race_id=race_id)
        money = asyncio.run(Auto_purchaser_tansho_turf.Auto_purchase_tansho_turf(**Tier5))
        print("demo")


    elif row['EARLY'] == 0 and row["type"] == 1 and row["class"] < 2:
        predict = predict_exe_turf_nowin.def_predict_exe_turf_nowin(kaisai_date=args.kaisai_date, race_id=race_id)
        money = asyncio.run(Auto_purchaser_tansho_turf_nowin.Auto_purchase_tansho_turf_nowin(**Tier5))
        print("demo")


    elif row['EARLY'] == 0 and row["type"] == 0 and row["class"] >= 2:

        print("demo")


    elif row['EARLY'] == 0 and row["type"] == 0 and row["class"] < 2:
        print("dirt_nowin")


    elif row['EARLY'] == 0 and row["type"] == 2:
        predict = predict_exe_obstract.def_predict_exe_obstract(kaisai_date=args.kaisai_date, race_id=race_id)
        money = asyncio.run(Auto_purchaser_tansho_obstract.Auto_purchase_tansho_obstract(**Tier5))
        print("demo")

    elif row["type"] == "none":
        
        predict = pre_predict_exe.prepredict(kaisai_date=args.kaisai_date)
        discord_notify = discord.post_discord() 
        discord_notify_2 = discord_2.post_discord_2()
        purchaser = TicketsPurchaser()
        money = purchaser.auto_in_money()
        print("demo")
        
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
    # #開発用.exのみ用
    # time_table_dev["post_time"] = [
    #     (datetime.now() + timedelta(minutes=3 * i + 100)).strftime("%H:%M")
    #     for i in range(len(time_table))
    # ]







    # 一番手前のpost_timeを取得
    first_post_time = datetime.strptime(time_table_dev.iloc[0]["post_time"], "%H:%M")

    # # 1時間前の時間を計算
    # new_post_time = (first_post_time - timedelta(minutes=60)).strftime("%H:%M")

    # 現在時刻を取得
    current_time = datetime.now()

    # 現在時刻の1分後を計算
    new_post_time = (current_time + timedelta(minutes=4)).strftime("%H:%M")


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