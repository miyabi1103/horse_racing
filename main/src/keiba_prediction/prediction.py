import pickle
from pathlib import Path

import pandas as pd
import yaml

DATA_DIR = Path("..","..", "data")
MODEL_DIR = DATA_DIR / "03_train"
SAVE_DIR = DATA_DIR / "05_prediction_results"
def predict(
    features: pd.DataFrame,
    model_dir: Path = MODEL_DIR,
    model_filename: Path = "model_lightgbm_rank_niti.pkl",
    config_filepath: Path = "config_lightgbm_niti.yaml",
    save_dir: Path = SAVE_DIR,
    save_filename: Path = "prediction_result.csv",
):
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(config_filepath, "r") as f:
        feature_cols = yaml.safe_load(f)["features"]
    with open(model_dir / model_filename, "rb") as f:
        model = pickle.load(f)
    features = features.dropna(subset=["tansho_odds"])
    
    # features["popularity"] =  fuatures["popularity"].astype(int)
    # features = features[features["tansho_odds"] != "取消"]
    prediction_df = features[[
        "race_id", 
        "umaban",               
        "tansho_odds", 
        "popularity",
    ]].copy()

    prediction_df["pred"] = model.predict(features[feature_cols])
    # race_id ごとに pred の合計が 1 になるように正規化
    prediction_df["pred"] = prediction_df.groupby("race_id")["pred"].transform(lambda x: x / x.sum())

    prediction_df["Ex_value"] = prediction_df["pred"] * prediction_df["tansho_odds"]
    prediction_df["popularity"] =  prediction_df["popularity"].astype(str) + "番人気"

    prediction_df = prediction_df.join(features[[
        "weight_diff",
        "mean_fukusho_rate_wakuban",
        # "mean_fukusho_rate_umaban",
        #ここから展開予想
        "pace_category",
        "ground_state_level",
        "dominant_position_category",
        "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined_standardized",     
        # "place_season_condition_type_categori_x",

        # #展開の有利不利
        # "tenkai_combined",
        # "tenkai_all_combined",
        # "tenkai_goal_range_combined",
        # "tenkai_curve_combined",
        # "tenkai_goal_slope_combined",
        # "tenkai_goal_range_curve_combined",
        # "tenkai_goal_range_goal_slope_combined",
        # "tenkai_all_combined",
        # "umaban_relative",
        
        # "tenkai_combined_standardized",
        # "tenkai_all_combined_standardized",

        
        # "course_len_diff_mean_1races",
        # "race_grade_rank_diff_sum_mean_3races_grade_rankdiff_score_plus_tenkai_combined_standardized",
        # "race_grade_rank_diff_multi_mean_3races_grade_rankdiff_score_plus_tenkai_combined_standardized",
        # "race_grade_rank_diff_sum_mean_3races_grade_rankdiff_score_plus_tenkai_all_combined_standardized",
        # "race_grade_rank_diff_multi_mean_3races_grade_rankdiff_score_plus_tenkai_all_combined_standardized",
        # "race_grade_rank_diff_sum_mean_3races_grade_rankdiff_score_px_tenkai_combined_standardized",
        # "race_grade_rank_diff_multi_mean_3races_grade_rankdiff_score_px_tenkai_combined_standardized",
        # "race_grade_rank_diff_sum_mean_3races_grade_rankdiff_score_px_tenkai_all_combined_standardized",
        # "race_grade_rank_diff_multi_mean_3races_grade_rankdiff_score_px_tenkai_all_combined_standardized",

        #各馬番、枠番の勝率がわかる
        # "goal_range",
        # "goal_range_100",
        # "sire_n_wins_relative",
        # "bms_n_wins_relative",
        # "interval",
        # "sin_date_sex",
        # "distance_place_type_ground_state_encoded_time_diff_mean_1races_cross_encoded_relative",
        # "distance_place_type_ground_state_encoded_time_diff_min_1races_cross_encoded_relative",
        # "distance_place_type_ground_state_encoded_time_diff_mean_3races_cross_encoded_relative",
        # "distance_place_type_ground_state_encoded_time_diff_min_3races_cross_encoded_relative",
        # "syunpatu_mean_3races_encoded",
        # "zizoku_mean_3races_encoded",
        # "syunpatu_mean_1races_encoded",
        # "zizoku_mean_1races_encoded",        
        #全ての特筆
        # "rentai_for_pace_category_n5_relative",
        # "score_pace_potision_mean_3races_per_score_raw_relative",
        # "mean_5races_sum_relative",

        #スピード指数
        # "speed_index_mean_1races",
        # # "nobori_index_mean_1races",
        # "speed_index_mean_3races",
        # "nobori_index_mean_3races",


        # "syunpatu_mean_3races_encoded_index_relative",
        # "zizoku_mean_3races_encoded_index_relative",
        # "advantage_mean_3_index", 
        # "advantage_mean_3_index_relative",
        # "speed_index_min_5races",
        # "nobori_index_min_5races",
        # "speed_index_mean_3races_relative",
        # "nobori_index_mean_3races_relative",
        
        # 'time_diff_sp_mean_3races',
        # 'nobori_diff_sp_mean_3races',
        # 'time_points_course_index_mean_3races',
        # 'nobori_points_course_index_mean_3races',
        # 'time_points_impost_mean_3races',
        # 'nobori_points_impost_mean_3races',
        # 'time_diff_grade_mean_3races',
        # 'nobori_diff_grade_mean_3races',
        # 'time_points_grade_index_mean_3races',
        # 'nobori_points_grade_index_mean_3races',
        # 'time_condition_index_mean_3races',
        # 'nobori_condition_index_mean_3races',
        # 'speed_index_mean_3races',
        # 'nobori_index_mean_3races',
                             ]].copy())
    prediction_df["tenkai_Advantage"] = prediction_df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined_standardized"]
    prediction_df = prediction_df.drop(columns=["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined_standardized"])
    prediction_df = prediction_df.sort_values("pred", ascending=False)

    prediction_df.to_csv(save_dir / save_filename, sep="\t", index=False)
    print("prediction...comp")
    return prediction_df.sort_values("pred", ascending=False)






    
