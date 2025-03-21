import asyncio
import argparse
from datetime import datetime,timedelta
from apscheduler.schedulers.blocking import BlockingScheduler

from Auto_prediction import Create_time_table

#ここに投票を行う遷移の関数をいれる
# def scrape_job(race_id:str,scraper:Auto_prediction):
#     print(f"scraping auto {race_id}")
#     asyncio.run(scraper.Create_time_table(race_id=race_id))

def scrape_job(race_id:str):
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
    time_table = asyncio.run(Create_time_table(kaisai_data=args.kaisai_date))
    time_table_dev = time_table.copy()


    #開発用
    time_table_dev["post_time"] = [
        (datetime.now() + timedelta(minutes=1 * i + 1)).strftime("%H:%M")
        for i in range(len(time_table))
    ]

    print(time_table_dev)
    scheduler = BlockingScheduler()

    
    for idx, row in time_table_dev.iterrows():
        race_id = row["race_id"]
        post_time = datetime.strptime(row["post_time"],"%H:%M").time()
        run_at = (
            datetime.combine(datetime.now(),post_time)
            # - timedelta(minutes = 2)
            # - timedelta(seconds = 10)
        )
        scheduler.add_job(
            func=scrape_job,
            trigger="date",
            run_date=run_at,
            args=[race_id]
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