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
    prediction_df["Ex_value"] = prediction_df["pred"] * prediction_df["tansho_odds"]
    prediction_df["popularity"] =  prediction_df["popularity"].astype(int)


    prediction_df = prediction_df.join(features[[
        "mean_fukusho_rate_wakuban",
        # "mean_fukusho_rate_umaban",
        #ここから展開予想
        "dominant_position_category",
        "pace_category",
        "ground_state_level",     
        "place_season_condition_type_categori_x",
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
        
        "tenkai_combined_standardized",
        "tenkai_all_combined_standardized",
        "weight_diff",
        # "course_len_diff_mean_1races",
        "race_grade_rank_diff_sum_mean_3races_grade_rankdiff_score_plus_tenkai_combined_standardized",
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
        "interval",
        "sin_date_sex",
        # "distance_place_type_ground_state_encoded_time_diff_mean_1races_cross_encoded_relative",
        "distance_place_type_ground_state_encoded_time_diff_min_1races_cross_encoded_relative",
        "distance_place_type_ground_state_encoded_time_diff_mean_3races_cross_encoded_relative",
        # "distance_place_type_ground_state_encoded_time_diff_min_3races_cross_encoded_relative",
        "syunpatu_mean_3races_encoded",
        "zizoku_mean_3races_encoded",
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



    return prediction_df.sort_values("Ex_value", ascending=False)














# """
# 障害はtop4(相手３)頭top1軸馬連(15%),top3boxのワイド
# ダートはtop１軸３頭馬連（5%）,top1一頭軸３連単
# 芝はtop3三連単,top1軸（2%）

# 強馬がいたらオッズが高い場合は、単複top1
# 低い場合はtop1軸ワイド（相手はときによる）で保険をかけつつ、勝負に出る
# (ダートは一頭軸連単、連複＿芝は一頭軸三連複)

# また、タイム差がない場合、オッズが高いモノは穴馬の可能性があるため、単複しても良い。
# 複勝は単勝の1/5と考えると、10になるためには50倍以上はほしい
# 10あたりなら単勝onlyが良い
# """


# """
# 短距離ほどタイム差はシビアになる
# タイム差があまりない場合、オッズが低いものを単複ワイドで買っても良い(基本は1600m/0.2秒以内)
# 短距離なら0.12
# 長距離なら長距離なら0.3
# """




# """
# オッズあり、ではかわない、あくまで乖離を見る
# """








    
