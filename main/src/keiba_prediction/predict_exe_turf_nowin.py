

import preprocessing
from feature_engineering_prediction import PredictionFeatureCreator

import prediction
import pandas as pd
import asyncio
import condition_prediction

import odds_prediction


from pathlib import Path
COMMON_DATA_DIR = Path("..", "..", "..","common", "data")
POPULATION_DIR_NEW = COMMON_DATA_DIR / "prediction_population"


def def_predict_exe_turf_nowin(kaisai_date:str,race_id:str):

    pfc_turf_nowin = PredictionFeatureCreator(
        horse_results_filename = "horse_results_prediction_turf.csv",
        peds_filename = "peds_prediction_turf.csv",
        old_population_filename="population_turf.csv",
        old_results_filename="results_turf.csv",
        old_race_info_filename="race_info_turf.csv",
        old_horse_results_filename="horse_results_turf.csv",
    )
    # 過去成績集計は事前に行うことができる
    pfc_turf_nowin.create_baselog()
    pfc_turf_nowin.agg_horse_n_races()


    race_id=race_id  # 予測するレースidを指定
    date_content_a = kaisai_date #"%Y年%m月%d日"形式で該当レース当日の日付を入れてください    
    date_condition_a=0

    # ### 芝未勝利

    # 特徴量の更新
    features = pfc_turf_nowin.create_features(
        race_id=race_id,  # 予測するレースidを指定
        date_content_a = date_content_a, #"%Y年%m月%d日"形式で該当レース当日の日付を入れてください
        date_condition_a=date_condition_a,#前週から今日までで雨が降っている、または途中から重馬場になりそうなとき、記載
        #　特にない場合は0を指定する
        # 標準的な馬場4
        # 道悪7
        skip_agg_horse=True  # 事前に集計した場合はスキップできる
    )


    # 体重あり

    features = asyncio.run(odds_prediction.update_odds_and_popularity(
        features = features,
        race_id = race_id,
    ))
    prediction.predict(
        features,
        model_filename="model_lightgbm_rank_niti_cv_full_dev_turf_nowin_only_tenkai.pkl",
        config_filepath="config_lightgbm_niti_dev_turf_only_tenkai_nowin.yaml"
    )
    print("turf_nowin")


    # In[ ]:


    # # 体重あり、展開
    # #　傑出はノーマルで判断可能
    # features = await odds_prediction.update_odds_and_popularity(
    #     features = features,
    #     race_id = race_id,
    # )
    # prediction.predict(
    #     features,
    #     model_filename="model_lightgbm_rank_niti_cv_full_dev_turf_nowin_only_tenkai_in3.pkl",
    #     config_filepath="config_lightgbm_niti_dev_turf_only_tenkai_nowin.yaml"
    # )


    # In[ ]:





    # In[ ]:





    # In[357]:


    # # 体重なし
    # features = await odds_prediction.update_odds_and_popularity(
    #     features = features,
    #     race_id = race_id,
    # )
    # prediction.predict(
    #     features,
    #     model_filename="model_lightgbm_rank_niti_cv_full_dev_turf_nowin_noweight.pkl",
    #     config_filepath="config_lightgbm_niti_dev_turf_nowin_noweight.yaml"
    # )


    # # In[322]:


    # # 体重なし展開

    # features = await odds_prediction.update_odds_and_popularity(
    #     features = features,
    #     race_id = race_id,
    # )
    # prediction.predict(
    #     features,
    #     model_filename="model_lightgbm_rank_niti_cv_full_dev_turf_nowin_only_tenkai_noweight.pkl",
    #     config_filepath="config_lightgbm_niti_dev_turf_only_tenkai_noweight_nowin.yaml"
    # )

