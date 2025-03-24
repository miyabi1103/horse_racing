
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import preprocessing
from feature_engineering_prediction import PredictionFeatureCreator

import prediction
import pandas as pd
import asyncio
import condition_prediction

import odds_prediction



COMMON_DATA_DIR = Path("..", "..", "..","common", "data")
POPULATION_DIR_NEW = COMMON_DATA_DIR / "prediction_population"


def def_predict_exe_turf(kaisai_date:str,race_id:str):
    pfc_turf = PredictionFeatureCreator(
        horse_results_filename = "horse_results_prediction_turf.csv",
        peds_filename = "peds_prediction_turf.csv",
        population_all_filename="population_turf.csv",
        results_all_filename="results_turf.csv",
        race_info_all_filename="race_info_turf.csv",
        horse_results_all_filename="horse_results_turf.csv",
    )
    # 過去成績集計は事前に行うことができる
    pfc_turf.create_baselog()
    pfc_turf.agg_horse_n_races()



    race_id=race_id  # 予測するレースidを指定
    date_content_a = kaisai_date #"%Y年%m月%d日"形式で該当レース当日の日付を入れてください    
    date_condition_a=0

    # 特徴量の更新
    features = pfc_turf.create_features(
        race_id=race_id,  # 予測するレースidを指定
        date_content_a = date_content_a, #"%Y年%m月%d日"形式で該当レース当日の日付を入れてください
        date_condition_a=date_condition_a,#前週から今日までで雨が降っている、または途中から重馬場になりそうなとき、記載
        #　特にない場合は0を指定する
        # 標準的な
        # 道悪7
        skip_agg_horse=True  # 事前に集計した場合はスキップできる
    )


    # # 体重あり

    # features = await odds_prediction.update_odds_and_popularity(
    #     features = features,
    #     race_id = race_id,
    # )
    # prediction.predict(
    #     features,
    #     model_filename="model_lightgbm_rank_niti_cv_full_dev_turf_in3.pkl",
    #     config_filepath="config_lightgbm_niti_dev_turf.yaml"
    # )


    # In[232]:


    # 体重あり、展開

    features = asyncio.run(odds_prediction.update_odds_and_popularity(
        features = features,
        race_id = race_id,
    ))
    prediction.predict(
        features,
        model_filename="model_lightgbm_rank_niti_cv_full_dev_turf_only_tenkai_in3.pkl",
        config_filepath="config_lightgbm_niti_dev_turf_only_tenkai.yaml"
    )

    print("turf")