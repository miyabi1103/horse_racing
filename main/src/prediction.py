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
    prediction_df = features[[
        "race_id", 
        "umaban",               
        "tansho_odds", 
        "popularity",
        "mean_fukusho_rate_wakuban",
        # "mean_fukusho_rate_umaban",
        #ここから展開予想
        "dominant_position_category",
        "pace_category",
        "ground_state_level",       
        #展開の有利不利
        # "tenkai_combined_standardized",
        "tenkai_all_combined_standardized",
        # "race_grade_rank_diff_sum_mean_3races_grade_rankdiff_score_plus_tenkai_combined_standardized",
        # "race_grade_rank_diff_multi_mean_3races_grade_rankdiff_score_plus_tenkai_combined_standardized",
        "race_grade_rank_diff_sum_mean_3races_grade_rankdiff_score_plus_tenkai_all_combined_standardized",
        # "race_grade_rank_diff_multi_mean_3races_grade_rankdiff_score_plus_tenkai_all_combined_standardized",
        # "race_grade_rank_diff_sum_mean_3races_grade_rankdiff_score_px_tenkai_combined_standardized",
        # "race_grade_rank_diff_multi_mean_3races_grade_rankdiff_score_px_tenkai_combined_standardized",
        # "race_grade_rank_diff_sum_mean_3races_grade_rankdiff_score_px_tenkai_all_combined_standardized",
        # "race_grade_rank_diff_multi_mean_3races_grade_rankdiff_score_px_tenkai_all_combined_standardized",

        #各馬番、枠番の勝率がわかる
        "distance_place_type_ground_state_encoded_time_diff_mean_3races_cross_encoded_relative",
        # "distance_place_type_ground_state_encoded_time_diff_min_3races_cross_encoded_relative",
        "syunpatu_mean_3races_encoded",
        "zizoku_mean_3races_encoded",
        #全ての特筆
        # "rentai_for_pace_category_n5_relative",
        # "score_pace_potision_mean_3races_per_score_raw_relative",
        "mean_5races_sum_relative",

        #スピード指数
        "speed_index_mean_3races",
        "nobori_index_mean_3races",
        # "speed_index_min_5races",
        # "nobori_index_min_5races",
        # "speed_index_mean_3races_relative",
        # "nobori_index_mean_3races_relative",

                             ]].copy()
    prediction_df["pred"] = model.predict(features[feature_cols])
    return prediction_df.sort_values("pred", ascending=False)
