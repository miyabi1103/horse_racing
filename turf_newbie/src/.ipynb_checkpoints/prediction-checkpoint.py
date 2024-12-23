import pickle
from pathlib import Path

import pandas as pd
import yaml

DATA_DIR = Path("..", "data")
MODEL_DIR = DATA_DIR / "03_train"


def predict(
    features: pd.DataFrame,
    model_dir: Path = MODEL_DIR,
    model_filename: Path = "model_lightgbm_rank_niti.pkl",
    config_filepath: Path = "config_lightgbm_niti.yaml",
):
    with open(config_filepath, "r") as f:
        feature_cols = yaml.safe_load(f)["features"]
    with open(model_dir / model_filename, "rb") as f:
        model = pickle.load(f)
    prediction_df = features[["race_id", "umaban", "tansho_odds", "popularity","mean_fukusho_rate_wakuban","mean_fukusho_rate_umaban",
                              
                              # "distance_place_type_ground_state_weather_encoded_time_diff_mean_5_cross_encoded",  
                              # "distance_place_type_race_class_around_weather_ground_state_encoded_rank_diff_sumprod_mean_5_cross_encoded",
                              
                              # "distance_place_type_umaban_around_weather_ground_state_encoded_time_diff_mean_5_cross_encoded",
                              # "distance_place_type_ground_state_weather_nobori_encoded_time_diff_mean_5_cross_encoded"
                              
                              #その競馬場、長さ、タイプでの枠番の複勝率、馬番の複勝率
                              #距離、競馬場、タイプ、馬場状態、天気での平均タイム誤差
                              #距離、競馬場、タイプ、馬場状態、天気_umabanでの平均タイム誤差
                              #距離、競馬場、タイプ、馬場状態、天気での登りタイム誤差の和の平均、
                              
                              #距離、競馬場、タイプ、馬場状態、天気、レースランクでの平均タイム誤差
                              #距離、競馬場、タイプ、馬場状態、天気、レースランクでのrankdiffの和の平均、
                              
                              # "distance_place_type_race_class_around_weather_ground_state_encoded_time_diff_mean_5_cross_encoded",   
                              
                              # "distance_place_type_umaban_race_class_around_weather_ground_state_encoded_rank_diff_sumprod_mean_5_cross_encoded",
                              
                             ]].copy()
    prediction_df["pred"] = model.predict(features[feature_cols])
    return prediction_df.sort_values("pred", ascending=False)
