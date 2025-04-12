import pickle
from pathlib import Path

import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


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
    prediction_df["pop"] =  prediction_df["popularity"].astype(int).astype(str)  + "pop"
    prediction_df = prediction_df.join(features[[
        "weight_diff",
        # "mean_fukusho_rate_wakuban",
        "mean_fukusho_rate_umaban",
        "race_type",
        "race_class",
        "place",
        # #ここから展開予想
        "pace_category",
        "ground_state_level",
        "dominant_position_category",
        "tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined_standardized",     
        # # "place_season_condition_type_categori_x",
        "start_time",

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


    prediction_df["waku_rate"] = prediction_df["mean_fukusho_rate_umaban"]
    prediction_df["type"] = prediction_df["race_type"]
    prediction_df["class"] = prediction_df["race_class"]
    prediction_df["ground"] = prediction_df["ground_state_level"]
    prediction_df["pace"] = prediction_df["pace_category"]
    prediction_df["position"] = prediction_df["dominant_position_category"]
    prediction_df = prediction_df.drop(columns=["popularity","mean_fukusho_rate_umaban","race_type","race_class","pace_category","ground_state_level","dominant_position_category"])
    
    prediction_df["tenkai_Advantage"] = prediction_df["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined_standardized"]
    prediction_df = prediction_df.drop(columns=["tenkai_place_start_slope_range_grade_lcurve_slope_type_curve_goal_range_slope_first_curve_front_len_umaban_combined_standardized"])
    prediction_df = prediction_df.sort_values("pred", ascending=False)
    prediction_df = prediction_df.round(5)
    
    place_mapping = {
        '01': 'Sapporo',
        '02': 'Hakodate',
        '03': 'Fukushima',
        '04': 'Niigata',
        '05': 'Tokyo',
        '06': 'Nakayama',
        '07': 'Chukyo',
        '08': 'Kyoto',
        '09': 'Hanshin',
        '10': 'Kokura'
    }
    from datetime import datetime

    # 曜日を取得するためのマッピング
    weekday_mapping = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    prediction_df["race_id"] = prediction_df["race_id"].astype(str)
    # RACE_INFO列を作成
    prediction_df["________RACE_INFO________"] = prediction_df["race_id"].apply(
        lambda race_id: f"{datetime.now().strftime('%Y/%m/%d')}({weekday_mapping[datetime.now().weekday()]}) "
                        f"{place_mapping[str(race_id[4:6])]} No.{int(str(race_id[10:12]))}"
    )
    # RACE_INFO列を一番左に移動
    columns = ["________RACE_INFO________"] + [col for col in prediction_df.columns if col != "________RACE_INFO________"]
    prediction_df = prediction_df[columns]
    prediction_df = prediction_df.drop(columns=["race_id"])

    prediction_df.to_csv(save_dir / save_filename, sep="\t", index=False)
    # def save_csv_as_image(dataframe, output_file):
    #     import matplotlib
    #     matplotlib.use("Agg")  # GUIバックエンドを無効化
    #     import matplotlib.pyplot as plt
    #     plt.rcParams["font.family"] = "DejaVu Sans"  # 英語用フォント
    #     plt.rcParams["font.family"] = "IPAexGothic"


    #     # 背景色を完全ダークモードに
    #     plt.rcParams["figure.facecolor"] = "black"  # 背景黒
    #     plt.rcParams["axes.facecolor"] = "black"    # 軸も黒

    #     # Seaborn のスタイル適用
    #     sns.set_style("dark")

    #     fig, ax = plt.subplots(figsize=(len(dataframe.columns) * 1.2, len(dataframe) * 0.6))

    #     # 軸を非表示にする
    #     ax.xaxis.set_visible(False)
    #     ax.yaxis.set_visible(False)
    #     ax.set_frame_on(False)

    #     # テーブルの作成
    #     table = ax.table(
    #         cellText=dataframe.values,
    #         colLabels=dataframe.columns,
    #         cellLoc="center",
    #         loc="center",
    #         bbox=[0, 0, 1, 1]  # テーブル全体を中央配置
    #     )

    #     # フォントサイズ調整
    #     table.auto_set_font_size(False)
    #     table.set_fontsize(10)

    #     # 列幅を適切に調整
    #     for i in range(len(dataframe.columns)):
    #         table.auto_set_column_width([i])

    #     # テーブルのデザイン（完全ダークモード）
    #     for (row, col), cell in table.get_celld().items():
    #         cell.set_edgecolor("#555555")  # 枠線をダークグレーに
    #         cell.set_linewidth(0.5)  # 線を細く
            
    #         if row == 0:  # ヘッダー部分
    #             cell.set_facecolor("#222222")  # Jupyter の青色 → ダークグレーに変更
    #             cell.set_text_props(weight="bold", color="white")
    #         else:
    #             # 偶数・奇数行で色を変える
    #             if row % 2 == 0:
    #                 cell.set_facecolor("#333333")  # 濃いグレー
    #             else:
    #                 cell.set_facecolor("#222222")  # さらに暗いグレー
    #             cell.set_text_props(color="white")  # テキストを白に

    #     plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0, facecolor="black")
    #     plt.close()
    #     # 画像として保存
    # output_image = save_dir / "prediction_result.png"
    # save_csv_as_image(prediction_df, output_image)

    # データフレームをPNG形式で保存する関数
    def save_csv_as_image(dataframe: pd.DataFrame, output_file: Path):
        import matplotlib
        matplotlib.use("Agg")  # GUIバックエンドを無効化
        import matplotlib.pyplot as plt
        import seaborn as sns

        from matplotlib import rcParams


        # 背景色とスタイルを設定
        plt.rcParams.update({
            "figure.facecolor": "black",  # 背景色を黒に設定
            "axes.facecolor": "black",   # 軸の背景色を黒に設定
            "text.color": "white",       # テキストの色を白に設定
            "axes.labelcolor": "white",  # 軸ラベルの色を白に設定
            "xtick.color": "white",      # x軸目盛りの色を白に設定
            "ytick.color": "white",      # y軸目盛りの色を白に設定
        })

        # Seaborn のスタイル適用
        sns.set_style("dark")

        # プロットの作成
        fig, ax = plt.subplots(figsize=(len(dataframe.columns) * 1.2, len(dataframe) * 0.6))  # サイズ調整
        ax.axis("tight")
        ax.axis("off")

        # テーブルを描画
        table = ax.table(
            cellText=dataframe.values,
            colLabels=dataframe.columns,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1]  # テーブル全体を中央配置
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # 列幅を適切に調整
        for i in range(len(dataframe.columns)):
            table.auto_set_column_width([i])

        # テーブルのデザイン（完全ダークモード）
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#555555")  # 枠線をダークグレーに
            cell.set_linewidth(0.5)  # 線を細く

            if row == 0:  # ヘッダー部分
                cell.set_facecolor("#1F354D")  # ヘッダーの背景色をダークグレーに設定
                cell.set_text_props(weight="bold", color="white")  # ヘッダーの文字色を白に設定
            else:
                # 偶数・奇数行で色を変える
                if row % 2 == 0:
                    cell.set_facecolor("#333333")  # 偶数行の背景色を濃いグレーに設定
                else:
                    cell.set_facecolor("#222222")  # 奇数行の背景色をさらに暗いグレーに設定
                cell.set_text_props(color="white")  # テキストを白に設定

        # 画像を保存
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0, facecolor="black")
        plt.close()


    output_image = save_dir / "prediction_result.png"
    save_csv_as_image(prediction_df, output_image)

    print("prediction...comp")
    return prediction_df.sort_values("pred", ascending=False)





    
