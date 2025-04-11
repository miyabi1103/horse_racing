#!/usr/bin/env python
# coding: utf-8

# # インポート

# In[1]:


import pandas as pd
import pickle

import numpy as np

import scraping
import create_rawdf
import create_prediction_population
from pathlib import Path
import requests

import condition_prediction

import scraping_prediction

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import preprocessing_nar

from feature_engineering_prediction import PredictionFeatureCreator

import shutil


DATA_DIR = Path("..", "..","..","common_nar", "data_nar")
HTML_RACE_DIR = DATA_DIR / "html" / "race"
HTML_HORSE_DIR = DATA_DIR / "html" / "horse"
HTML_PED_DIR = DATA_DIR / "html" / "ped"
HTML_LEADING_DIR = DATA_DIR / "html" / "leading"
POPULATION_DIR_NEW = DATA_DIR / "prediction_population"

# ### 実際に予測する際の事前準備

# - 実際の予測時の処理
# - 予測したいレース前日などに出走馬が発表されたら、以下の事前準備を行う

def prepredict(kaisai_date:str):

    # 予測母集団の作成
    #予測したい日を入力してください
    prediction_population = create_prediction_population.create(kaisai_date=kaisai_date)


    print(prediction_population)

    # 当日出走馬のhorse_idリスト
    horse_id_list = prediction_population["horse_id"].unique()


    ## 馬の過去成績取得 更新している可能性が高いため
    html_paths_horse = scraping.scrape_html_horse(horse_id_list=horse_id_list, skip=False)





    # 当日出走馬の過去成績テーブル作成
    horse_results_prediction = create_rawdf.create_horse_results(
        html_path_list=html_paths_horse,
        save_filename="horse_results_prediction.csv",
    )


    #血統データ取得
    # 当日出走馬の血統をスクレイピング
    html_paths_peds = scraping.scrape_html_ped(
        horse_id_list=horse_id_list,
        save_dir=Path(HTML_PED_DIR)
    )


    # skipされたものも含めて全てのhtmlファイルのパスを取得
    html_paths_peds = [scraping.HTML_PED_DIR / f"{horse_id}.bin" for horse_id in horse_id_list]


    # 当日出走馬の血統テーブル作成
    peds_prediction = create_rawdf.create_peds(
        html_path_list=html_paths_peds,
        save_filename="peds_prediction.csv",
    )




    TMP_DIR2 = scraping.DATA_DIR / "tmp_predict"


    # In[14]:



    # 年月を抽出して "YYYY-MM" の形式に変換
    kaisai_month = datetime.strptime(kaisai_date, "%Y%m%d").strftime("%Y-%m")

    print(kaisai_month)  # 出力: "2025-03"
    # 文字列をdatetimeオブジェクトに変換
    kaisai_date_obj = datetime.strptime(kaisai_month, "%Y-%m")

    # 1か月前を計算
    previous_month = kaisai_date_obj - relativedelta(months=1)

    # "YYYY-MM"形式に変換
    previous_month_str = previous_month.strftime("%Y-%m")

    print(previous_month_str)  # 出力: "2025-02"


    # 開催日一覧の取得、行うレースの月を含む1ヶ月前をとっておく、実際に使用するのは12日前のレースのみ
    kaisai_date_list_prediction = scraping.scrape_kaisai_date(
        from_=previous_month_str, to_=kaisai_month, save_dir=TMP_DIR2
    )

    cut_index = kaisai_date_list_prediction.index(kaisai_date)

    # リストをスライスして削除
    kaisai_date_list_prediction = kaisai_date_list_prediction[:cut_index]

    # kaisai_dateをdatetimeオブジェクトに変換
    kaisai_date_obj2 = datetime.strptime(kaisai_date, "%Y%m%d")

    # 14日前の日付を計算
    cutoff_date = kaisai_date_obj2 - timedelta(days=10)

    # リスト内の日付をフィルタリング
    kaisai_date_list_prediction = [
        date for date in kaisai_date_list_prediction
        if datetime.strptime(date, "%Y%m%d") >= cutoff_date
    ]
    print(kaisai_date_list_prediction)

    TMP_DIR3 = scraping.DATA_DIR / "tmp_predict2"
    # TMP_DIR3の中身を削除して再作成
    if TMP_DIR3.exists():
        shutil.rmtree(TMP_DIR3)  # フォルダを削除
    TMP_DIR3.mkdir(parents=True, exist_ok=True)  # フォルダを再作成
    # スクレイピング対象レースのid取得
    race_id_list_prediction = scraping.scrape_race_id_list(
        kaisai_date_list_prediction, save_dir=TMP_DIR3
    )


    # raceページのhtmlをスクレイピング
    html_paths_race_prediction = scraping_prediction.scrape_html_race(race_id_list=race_id_list_prediction, skip=False)


    # 途中で処理が途切れるなどした場合は、直接htmlのファイルパスを取得
    html_paths_race_prediction = [
        scraping_prediction.HTML_RACE_DIR / f"{race_id}.bin" for race_id in race_id_list_prediction
    ]



    results_pre = scraping_prediction.create_results(html_path_list=html_paths_race_prediction)


    race_info_pre = scraping_prediction.create_race_info(html_path_list=html_paths_race_prediction)


    # 当日出走馬の過去成績テーブルの前処理_そのままで使えない未加工のデータを加工する
    horse_results_preprocessed = preprocessing_nar.process_horse_results(
        input_filename="horse_results_prediction.csv",
        population_dir = POPULATION_DIR_NEW,
        output_filename="horse_results_prediction_dirt.csv"
    )
    # レース結果テーブルの前処理
    results_preprocessed = condition_prediction.process_results()
    # レース情報テーブルの前処理
    race_info_preprocessed = condition_prediction.process_race_info()
    create_race_grade_preprocessed = condition_prediction.create_race_grade()



    print("pre_predict...comp")

