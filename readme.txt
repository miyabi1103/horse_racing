
使用方法は後半に記述

binファイル、csvファイルはありません
common/src/mainにてスクレイピングも可能です
したくない方は後藤にlineか、alicetowa0315@gmail.comまで
送られてきたファイルを各フォルダに入れてください

discordにてスクレイピング結果を通知しているので、興味があればお知らせください

_narは地方競馬verです

ソースコードの実行環境
・OS: Mac OS 13.6.9
・言語: Python 3.11.8

main/src/keiba_prediction/Auto_voteにて自動投票
Auto_predictionでスケジュール把握
Auto_in_moneyで自動入金

Auto_purcaserで自動投票
prediction_exeでモデルによる予測を行っています

それ以外の細かなファイル、フォルダの説明は割愛。興味ある方は連絡を


mainの中身ではそれぞれ、芝、ダート、障害を予想する用で分けられています
_newbieはそれぞれのタイプで新馬を見越した予測版です、学習時に過去戦績がない馬に最適化するようにしています。精度は悪いです
nowinは未勝利戦です。

yamlファイルには特徴量とハイパーパラメータが記述されています
変更はそちらのファイルで行ってください


■ディレクトリ構成
.
├── common
│   ├── data
│   │   ├── html
│   │   │   ├── horse
│   │   │   │   └── horse_readme.txt
│   │   │   ├── leading
│   │   │   │   ├── bms_leading
│   │   │   │   │   └── bms_leading_readme.txt
│   │   │   │   ├── jockey_leading
│   │   │   │   │   └── jockey_leading_readme.txt
│   │   │   │   ├── sire_leading
│   │   │   │   │   └── sire_leading_readme.txt
│   │   │   │   └── trainer_leading
│   │   │   │       └── trainer_leading_readme.txt
│   │   │   ├── ped
│   │   │   │   └── ped_readme.txt
│   │   │   ├── race
│   │   │   │   └── race_readme.txt
│   │   │   └── race2
│   │   │       └── race2_readme.txt
│   │   ├── mapping
│   │   │   ├── around.json
│   │   │   ├── ground_state.json
│   │   │   ├── place.json
│   │   │   ├── race_class.json
│   │   │   ├── race_type.json
│   │   │   ├── sex.json
│   │   │   └── weather.json
│   │   ├── prediction_population
│   │   │   └── population_readme.txt
│   │   ├── rawdf
│   │   │   └── rawdf_readme.txt
│   │   ├── rawdf2
│   │   │   └── rawdf2_readme.txt
│   │   ├── tmp
│   │   │   ├── kaisai_date_list.txt
│   │   │   ├── kaisai_date_list_2.txt
│   │   │   ├── race_id_list.txt
│   │   │   ├── race_id_list2.txt
│   │   │   └── tmp_readme.txt
│   │   ├── tmp_new
│   │   │   ├── kaisai_date_list.txt
│   │   │   └── race_id_list.txt
│   │   ├── tmp_predict
│   │   │   ├── kaisai_date_list.txt
│   │   │   ├── race_id_list-Copy1.txt
│   │   │   ├── race_id_list.txt
│   │   │   └── tmp_readme.txt
│   │   └── tmp_predict2
│   │       └── race_id_list.txt
│   ├── src
│   │   ├── __pycache__
│   │   ├── create_prediction_population.py
│   │   ├── create_rawdf.py
│   │   ├── race_id_list.pickle
│   │   ├── scraping.py
│   │   └── scraping_prediction.py
│   └── src_log
├── conda_requirements.txt
├── dev_log
│   ├── data
│   │   ├── 00_population
│   │   │   └── poplation_readme.txt
│   │   ├── 01_preprocessed
│   │   │   ├── preprocessed_readme.txt
│   │   │   └── return_tables.pickle
│   │   ├── 02_features
│   │   │   └── features_readme.txt
│   │   ├── 03_train
│   │   │   └── train_readme.txt
│   │   └── 04_evaluation
│   │       └── evaluation_readme.txt
│   ├── log_src
│   │   ├── evaluation_lightgbm_index.py
│   │   ├── evaluation_shaft_index.py
│   │   ├── feature_engineering.py
│   │   ├── feature_engineering_prediction.py
│   │   └── train_lgbm_index_cross.py
│   └── src
│       ├── __pycache__
│       ├── condition_prediction.py
│       ├── config_lightgbm_kaiki.yaml
│       ├── config_lightgbm_kaiki_nopast.yaml
│       ├── config_lightgbm_kaiki_nopast_odds.yaml
│       ├── config_lightgbm_kaiki_odds_removed.yaml
│       ├── config_lightgbm_kaiki_only.yaml
│       ├── config_lightgbm_niti.yaml
│       ├── config_lightgbm_niti_new.yaml
│       ├── config_lightgbm_niti_nopast.yaml
│       ├── config_lightgbm_niti_nopast_odds.yaml
│       ├── config_lightgbm_niti_odds_removed.yaml
│       ├── config_lightgbm_niti_only.yaml
│       ├── create_population.py
│       ├── create_population3.py
│       ├── create_population_3age.py
│       ├── cross
│       │   ├── evaluation_lgbm_rank_niti_cross2.py
│       │   ├── evaluation_lightgbm_rank_diff.py
│       │   ├── feature_engineering-Copy1.py
│       │   ├── feature_engineering_prediction-Copy1.py
│       │   ├── train_lgbm_rank_niti_cross2.py
│       │   ├── train_lgbm_rank_niti_cross_past.py
│       │   ├── train_lightgbm_rank_diff.py
│       │   └── trash_feature.py
│       ├── evaluation_lgbm_rank_niti_cross.py
│       ├── evaluation_lightgbm_rank_kaiki.py
│       ├── evaluation_lightgbm_rank_niti.py
│       ├── evaluation_lightgbm_time_kaiki.py
│       ├── evaluation_pop.py
│       ├── evaluation_shaft_time_kaiki_cross.py
│       ├── feature_engineering.py
│       ├── feature_engineering_prediction.py
│       ├── pivot_table.xlsx
│       ├── pivot_table_styled_ground_state.xlsx
│       ├── pivot_table_styled_weather.xlsx
│       ├── prediction.py
│       ├── preprocessing.py
│       ├── styled_pivot_table.xlsx
│       ├── train_lgbm_rank_kaiki_cross.py
│       ├── train_lgbm_rank_niti_cross.py
│       ├── train_lgbm_time_cross.py
│       ├── train_lightgbm_rank_kaiki.py
│       ├── train_lightgbm_rank_niti.py
│       ├── train_lightgbm_time.py
│       └── yaml
│           ├── config-time_old.yaml
│           ├── config_new_old.yaml
│           ├── config_odds_removed_old.yaml
│           ├── config_odds_removed_racetype.yaml
│           ├── config_odds_removed_racetype2.yaml
│           ├── config_odds_removed_racetype3.yaml
│           ├── config_odds_removed_sire.yaml
│           ├── config_odds_removed_sire2.yaml
│           ├── config_old.yaml
│           └── config_time_odds_removed_old.yaml
├── horse
│   └── bin
│       ├── Activate.ps1
│       ├── activate
│       ├── activate.csh
│       ├── activate.fish
│       ├── pip
│       ├── pip3
│       ├── pip3.12
│       ├── python -> /opt/anaconda3/bin/python
│       ├── python3 -> python
│       └── python3.12 -> python
├── horse_racing_nar
│   ├── common_nar
│   │   ├── data_nar
│   │   │   ├── html
│   │   │   │   ├── horse
│   │   │   │   ├── leading
│   │   │   │   ├── ped
│   │   │   │   ├── race
│   │   │   │   └── race2
│   │   │   ├── mapping
│   │   │   │   ├── around.json
│   │   │   │   ├── ground_state.json
│   │   │   │   ├── place.json
│   │   │   │   ├── race_class.json
│   │   │   │   ├── race_class_2.json
│   │   │   │   ├── race_type.json
│   │   │   │   ├── sex.json
│   │   │   │   └── weather.json
│   │   │   ├── prediction_population
│   │   │   │   └── population_readme.txt
│   │   │   ├── rawdf
│   │   │   ├── rawdf2
│   │   │   ├── tmp
│   │   │   │   ├── kaisai_date_list.txt
│   │   │   │   └── race_id_list.txt
│   │   │   ├── tmp2
│   │   │   │   ├── kaisai_date_list.txt
│   │   │   │   └── race_id_list.txt
│   │   │   ├── tmp_new
│   │   │   ├── tmp_predict
│   │   │   │   ├── kaisai_date_list.txt
│   │   │   │   └── race_id_list.txt
│   │   │   └── tmp_predict2
│   │   │       └── race_id_list.txt
│   │   └── src_nar
│   │       ├── __pycache__
│   │       ├── create_prediction_population.py
│   │       ├── create_rawdf.py
│   │       ├── race_id_list.pickle
│   │       ├── scraping.py
│   │       └── scraping_prediction.py
│   └── main_nar
│       ├── data_nar
│       │   ├── 00_population
│       │   │   └── poplation_readme.txt
│       │   ├── 01_preprocessed
│       │   │   └── return_tables_all.pickle
│       │   ├── 02_features
│       │   ├── 03_train
│       │   │   └── train_readme.txt
│       │   ├── 04_evaluation
│       │   └── 05_prediction_results
│       └── src_nar
│           ├── __pycache__
│           ├── keiba_betting
│           │   └── tickets_purchaser.py
│           ├── keiba_exp
│           │   ├── __pycache__
│           │   ├── config_lightgbm_kaiki_dev_dirt10.yaml
│           │   ├── config_lightgbm_kaiki_dev_dirt4.yaml
│           │   ├── config_lightgbm_kaiki_dev_dirt7.yaml
│           │   ├── config_lightgbm_kaiki_dev_dirt_only_tenkai_weight10.yaml
│           │   ├── config_lightgbm_kaiki_dev_dirt_only_tenkai_weight4.yaml
│           │   ├── config_lightgbm_kaiki_dev_dirt_only_tenkai_weight7.yaml
│           │   ├── config_lightgbm_niti_dev_dirt10.yaml
│           │   ├── config_lightgbm_niti_dev_dirt4.yaml
│           │   ├── config_lightgbm_niti_dev_dirt7.yaml
│           │   ├── config_lightgbm_niti_dev_dirt_noweight10.yaml
│           │   ├── config_lightgbm_niti_dev_dirt_noweight4.yaml
│           │   ├── config_lightgbm_niti_dev_dirt_noweight7.yaml
│           │   ├── config_lightgbm_niti_dev_dirt_only_tenkai10.yaml
│           │   ├── config_lightgbm_niti_dev_dirt_only_tenkai4.yaml
│           │   ├── config_lightgbm_niti_dev_dirt_only_tenkai7.yaml
│           │   ├── config_lightgbm_niti_dev_dirt_only_tenkai_weight10.yaml
│           │   ├── config_lightgbm_niti_dev_dirt_only_tenkai_weight4.yaml
│           │   ├── config_lightgbm_niti_dev_dirt_only_tenkai_weight7.yaml
│           │   ├── create_population.py
│           │   ├── create_population_dirt_funabasi.py
│           │   ├── create_population_dirt_kanazawa.py
│           │   ├── create_population_dirt_kasamatu.py
│           │   ├── create_population_dirt_kawasaki.py
│           │   ├── create_population_dirt_kouti.py
│           │   ├── create_population_dirt_mizusawa.py
│           │   ├── create_population_dirt_monbetu.py
│           │   ├── create_population_dirt_morioka.py
│           │   ├── create_population_dirt_nagoya.py
│           │   ├── create_population_dirt_ooi.py
│           │   ├── create_population_dirt_saga.py
│           │   ├── create_population_dirt_sonoda.py
│           │   ├── create_population_dirt_urawa.py
│           │   ├── evaluation_lgbm_rank_niti_cross.py
│           │   ├── evaluation_lgbm_rank_niti_cross_ex_value.py
│           │   ├── evaluation_lgbm_rank_niti_cross_in3.py
│           │   ├── evaluation_lightgbm_time_kaiki.py
│           │   ├── evaluation_shaft_rank_cross.py
│           │   ├── evaluation_shaft_time_kaiki_cross.py
│           │   ├── feature_engineering.py
│           │   ├── preprocessing_nar.py
│           │   ├── race_grade_maker.py
│           │   ├── text_to_text.text
│           │   ├── train_lgbm_rank_niti_cross.py
│           │   ├── train_lgbm_rank_niti_cross_in3.py
│           │   ├── train_lgbm_time_cross.py
│           │   └── train_lightgbm_time.py
│           ├── keiba_log
│           │   ├── Auto_csv_to_number.py
│           │   └── config.txt
│           ├── keiba_notify
│           │   ├── __pycache__
│           │   └── discord.py
│           └── keiba_prediction
│               ├── Auto_in_money.py
│               ├── Auto_prediction.py
│               ├── Auto_purchaser_sanrenpuku.py
│               ├── Auto_purchaser_umaren.py
│               ├── Auto_purchaser_wide.py
│               ├── Auto_vote.py
│               ├── __pycache__
│               ├── condition_prediction.py
│               ├── config_lightgbm_kaiki_dev_dirt10.yaml
│               ├── config_lightgbm_kaiki_dev_dirt4.yaml
│               ├── config_lightgbm_kaiki_dev_dirt7.yaml
│               ├── config_lightgbm_kaiki_dev_dirt_only_tenkai_weight10.yaml
│               ├── config_lightgbm_kaiki_dev_dirt_only_tenkai_weight4.yaml
│               ├── config_lightgbm_kaiki_dev_dirt_only_tenkai_weight7.yaml
│               ├── config_lightgbm_niti_dev_dirt10.yaml
│               ├── config_lightgbm_niti_dev_dirt4.yaml
│               ├── config_lightgbm_niti_dev_dirt7.yaml
│               ├── config_lightgbm_niti_dev_dirt_noweight10.yaml
│               ├── config_lightgbm_niti_dev_dirt_noweight4.yaml
│               ├── config_lightgbm_niti_dev_dirt_noweight7.yaml
│               ├── config_lightgbm_niti_dev_dirt_only_tenkai10.yaml
│               ├── config_lightgbm_niti_dev_dirt_only_tenkai4.yaml
│               ├── config_lightgbm_niti_dev_dirt_only_tenkai7.yaml
│               ├── config_lightgbm_niti_dev_dirt_only_tenkai_weight10.yaml
│               ├── config_lightgbm_niti_dev_dirt_only_tenkai_weight4.yaml
│               ├── config_lightgbm_niti_dev_dirt_only_tenkai_weight7.yaml
│               ├── create_prediction_population.py
│               ├── create_rawdf.py
│               ├── crontab.txt
│               ├── feature_engineering_prediction.py
│               ├── odds_prediction.py
│               ├── pre_predict_exe.py
│               ├── predict_exe_funabasi.py
│               ├── predict_exe_kanazawa.py
│               ├── predict_exe_kasamatu.py
│               ├── predict_exe_kawasaki.py
│               ├── predict_exe_kouti.py
│               ├── predict_exe_mizusawa.py
│               ├── predict_exe_monbetu.py
│               ├── predict_exe_morioka.py
│               ├── predict_exe_nagoya.py
│               ├── predict_exe_ooi.py
│               ├── predict_exe_saga.py
│               ├── predict_exe_sonoda.py
│               ├── predict_exe_urawa.py
│               ├── prediction.py
│               ├── preprocessing_nar.py
│               ├── race_grade_maker.py
│               ├── scraping.py
│               └── scraping_prediction.py
├── main
│   ├── data
│   │   ├── 00_population
│   │   │   └── poplation_readme.txt
│   │   ├── 01_preprocessed
│   │   │   ├── preprocessed_readme.txt
│   │   │   ├── return_tables_all.pickle
│   │   │   ├── return_tables_dirt.pickle
│   │   │   ├── return_tables_dirt_newbie.pickle
│   │   │   ├── return_tables_dirt_nowin.pickle
│   │   │   ├── return_tables_obstract.pickle
│   │   │   ├── return_tables_turf.pickle
│   │   │   ├── return_tables_turf_newbie.pickle
│   │   │   └── return_tables_turf_nowin.pickle
│   │   ├── 02_features
│   │   │   └── features_readme.txt
│   │   ├── 03_train
│   │   │   └── train_readme.txt
│   │   ├── 04_evaluation
│   │   │   └── evaluation_readme.txt
│   │   └── 05_prediction_results
│   └── src
│       ├── GitHub.code-workspace
│       ├── __pycache__
│       ├── keiba_betting
│       │   ├── __pycache__
│       │   └── tickets_purchaser.py
│       ├── keiba_exp
│       │   ├── __pycache__
│       │   ├── config_lightgbm_kaiki_dev_dirt.yaml
│       │   ├── config_lightgbm_kaiki_dev_dirt_only_tenkai_weight.yaml
│       │   ├── config_lightgbm_kaiki_dev_turf.yaml
│       │   ├── config_lightgbm_kaiki_dev_turf_only_tenkai.yaml
│       │   ├── config_lightgbm_niti_dev_dirt.yaml
│       │   ├── config_lightgbm_niti_dev_dirt_noweight.yaml
│       │   ├── config_lightgbm_niti_dev_dirt_nowin.yaml
│       │   ├── config_lightgbm_niti_dev_dirt_nowin_noweight.yaml
│       │   ├── config_lightgbm_niti_dev_dirt_only_tenkai.yaml
│       │   ├── config_lightgbm_niti_dev_dirt_only_tenkai_weight.yaml
│       │   ├── config_lightgbm_niti_dev_obstract.yaml
│       │   ├── config_lightgbm_niti_dev_obstract_noweight.yaml
│       │   ├── config_lightgbm_niti_dev_turf.yaml
│       │   ├── config_lightgbm_niti_dev_turf_noweight.yaml
│       │   ├── config_lightgbm_niti_dev_turf_nowin.yaml
│       │   ├── config_lightgbm_niti_dev_turf_nowin_noweight.yaml
│       │   ├── config_lightgbm_niti_dev_turf_only_tenkai.yaml
│       │   ├── config_lightgbm_niti_dev_turf_only_tenkai_ex.yaml
│       │   ├── config_lightgbm_niti_dev_turf_only_tenkai_noweight.yaml
│       │   ├── config_lightgbm_niti_dev_turf_only_tenkai_noweight_nowin.yaml
│       │   ├── config_lightgbm_niti_dev_turf_only_tenkai_nowin.yaml
│       │   ├── config_lightgbm_niti_new.yaml
│       │   ├── create_population.py
│       │   ├── create_population_dirt.py
│       │   ├── create_population_dirt_newbie.py
│       │   ├── create_population_dirt_nowin.py
│       │   ├── create_population_obstract.py
│       │   ├── create_population_turf.py
│       │   ├── create_population_turf_newbie.py
│       │   ├── create_population_turf_nowin.py
│       │   ├── evaluation_lgbm_rank_niti_cross.py
│       │   ├── evaluation_lgbm_rank_niti_cross_ex_value.py
│       │   ├── evaluation_lgbm_rank_niti_cross_in3.py
│       │   ├── evaluation_lightgbm_time_kaiki.py
│       │   ├── evaluation_shaft_rank_cross.py
│       │   ├── evaluation_shaft_time_kaiki_cross.py
│       │   ├── feature_engineering.py
│       │   ├── preprocessing.py
│       │   ├── race_grade_maker.py
│       │   ├── text_to_text.text
│       │   ├── train_lgbm_rank_niti_cross.py
│       │   ├── train_lgbm_rank_niti_cross_in3.py
│       │   ├── train_lgbm_time_cross.py
│       │   └── train_lightgbm_time.py
│       ├── keiba_log
│       │   ├── Auto_csv_to_number.py
│       │   ├── Auto_csv_to_number_top.py
│       │   ├── config.txt
│       │   └── prediction_ex.py
│       ├── keiba_notify
│       │   ├── __pycache__
│       │   └── discord.py
│       └── keiba_prediction
│           ├── Auto_in_money.py
│           ├── Auto_prediction.py
│           ├── Auto_purchaser_sanrenpuku.py
│           ├── Auto_purchaser_sanrentan.py
│           ├── Auto_purchaser_tansho_dirt.py
│           ├── Auto_purchaser_tansho_obstract.py
│           ├── Auto_purchaser_tansho_turf.py
│           ├── Auto_purchaser_tansho_turf_nowin.py
│           ├── Auto_purchaser_umaren.py
│           ├── Auto_purchaser_wide.py
│           ├── Auto_vote.py
│           ├── __pycache__
│           ├── condition_prediction.py
│           ├── config_lightgbm_niti_dev_dirt.yaml
│           ├── config_lightgbm_niti_dev_dirt_noweight.yaml
│           ├── config_lightgbm_niti_dev_dirt_nowin.yaml
│           ├── config_lightgbm_niti_dev_dirt_nowin_noweight.yaml
│           ├── config_lightgbm_niti_dev_dirt_only_tenkai.yaml
│           ├── config_lightgbm_niti_dev_dirt_only_tenkai_weight.yaml
│           ├── config_lightgbm_niti_dev_obstract.yaml
│           ├── config_lightgbm_niti_dev_obstract_noweight.yaml
│           ├── config_lightgbm_niti_dev_turf.yaml
│           ├── config_lightgbm_niti_dev_turf_noweight.yaml
│           ├── config_lightgbm_niti_dev_turf_nowin.yaml
│           ├── config_lightgbm_niti_dev_turf_nowin_noweight.yaml
│           ├── config_lightgbm_niti_dev_turf_only_tenkai.yaml
│           ├── config_lightgbm_niti_dev_turf_only_tenkai_ex.yaml
│           ├── config_lightgbm_niti_dev_turf_only_tenkai_noweight.yaml
│           ├── config_lightgbm_niti_dev_turf_only_tenkai_noweight_nowin.yaml
│           ├── config_lightgbm_niti_dev_turf_only_tenkai_nowin.yaml
│           ├── create_prediction_population.py
│           ├── create_rawdf.py
│           ├── crontab_main.txt
│           ├── feature_engineering_prediction.py
│           ├── odds_prediction.py
│           ├── pre_predict_exe.py
│           ├── predict_exe_dirt.py
│           ├── predict_exe_dirt_nowin.py
│           ├── predict_exe_obstract.py
│           ├── predict_exe_turf.py
│           ├── predict_exe_turf_nowin.py
│           ├── prediction.py
│           ├── preprocessing.py
│           ├── race_grade_maker.py
│           ├── scraping.py
│           └── scraping_prediction.py
├── readme.txt
└── requirements.txt








________________________________________________________________________________________________________________________

[特徴量一覧（ざっくり）]
ここには書いていないものも
yamlファイルにはあるので、読み取ってください

n races系にはそれぞれの平均、最大、最小を取ったバージョンがあります
また、過去の成績などはレースごとに標準化されています。


race_id: レースごとに一意に割り振られた識別ID。
horse_id: 出走馬ごとに一意に割り振られた識別ID。
jockey_id: 騎手ごとに一意に割り振られた識別ID。
trainer_id: 調教師ごとに一意に割り振られた識別ID。
umaban: 出走馬の馬番（レースごとの馬の番号）。
wakuban: 枠番（同じ枠内の馬は複数存在する可能性あり）。
tansho_odds: 単勝オッズ（その馬が1着になる確率の倍率）。
popularity: 人気順（その馬のファン投票の順位、1が最も人気）。
impost: 斤量（その馬が背負う重量）。
sex: 馬の性別（牡、牝、騸のいずれか）。
age: 馬の年齢（競走馬の年齢）。
weight: レース当日の馬体重（馬の体重）。
weight_diff: 前回出走時からの馬体重の増減。
n_horses: そのレースに出走する馬の総数。
mean_age: 出走馬の年齢の平均値。
race_type: レースの種類（芝、ダート、障害などの区分）。
around: コースの周回方向（右回り、左回り）。
course_len: コースの全長（レースの距離）。
weather: 当日の天候（晴れ、雨、曇りなど）。
ground_state: 馬場状態（良、稍重、重、不良の4段階）。
race_class: レースのクラス（G1、G2、G3、オープン、1勝クラスなど）。
place: 開催場所（札幌、中山、東京、京都などの競馬場）。
season: 季節（春、夏、秋、冬のいずれか）。
day: 開催日（その開催の何日目かを表す）。

place_course_category: 競馬場のコースカテゴリ（平坦コース、坂のあるコースなどの区分）。
place_course_tough: コースのタフさを表す指標（ペースや消耗度に影響）。
goal_range: ゴール前の直線の長さ（ゴールまでの直線距離）。
curve: コースのカーブの数や特徴（コーナーの回数など）。
goal_slope: ゴール前の坂の有無（坂の傾斜の有無）。
lap_type: レースのラップ傾向（前半が速い、後半が速いなどのパターン）。
race_grade: レースのグレードを数値化した情報（G1=100, G2=97, G3=94など）。
rank_nraces: その馬の過去nレースにおける順位（前走の順位）。
rank_per_horse_nraces: その馬の過去nレースの成績を馬ごとに標準化したもの。
prize_nraces: その馬の過去nレースにおける獲得賞金。

条件ごとの平均タイムと当該馬のタイムの差
それにrankdiffを掛け合わせたもの

前走からどのくらい月日が立っているか

wakuban/umabanごとの複勝率

脚質
ペース
馬場状態（直近レースのタイムと、平均タイムの誤差から導き出したもの）
展開予想

カーブの種類
直線の長さ
坂の有無
と、それらが展開に与える影響

その他、坂の高さ、高低差、カーブの角度、スタートからの長さ、第一コーナーの坂など
競馬場ごとの情報などを追記



血統データ（父、母父の勝率や得意な長さ、タイプなど）
騎手データ
調教師データ

スピード指数
上り指数




____________________________________________________________


・使用方法
スクレイピングがまだの場合や、追加でスクレイピングを行いたい場合は
commonフォルダのsrc/mainを上から実行する


スクレイピングが終わっている場合
mainフォルダにあるsrc/mainを上から実行すること
特徴量作成処理にはかなり時間がかかります、場合によっては落ちます


特徴量作成処理が終わっている場合
そのままmainを進み、モデルの学習を行なってください

モデルの学習の種類は
目的変数がrank二値、回帰、time回帰
そのホールドアウトverかクロスバリデーションver
があり
※mainの方ではクロスバリデーションver、time回帰のみ設定されています。他のものを指定したい場合はdev_log/main2のコードを参考にしながら設定してください。わからなければlineをば

特徴量はyamlファイルの選択で変更できます
既存のyamlファイルには
・全部載せオッズありver
・全部載せオッズなしver
・過去の成績なしver
・展開予想やスピード指数のみのver
があります

既存の特徴量を簡単に説明したものは、一番下に記載しています

新しい特徴量ファイルを作成したい場合は、既存のものをコピペして、お好みで変更（コメントアウトなどを）してください
ハイパーパラメータも記載されているので、お好みで変更してください


評価はお好みで行ってください



モデルの学習が終わっており、予測を行いたい場合
commonフォルダのsrc/main2の下の方にある、事前情報集計を実行（該当レースの日時をnetkeibaのurlから手動で取得してきてください、サンプルを参考に手動で入れても良いです、サンプルのidを検索にかければそのレースがわかります）
turf/dirt/obstaclesフォルダにあるいずれかのsrc/mainの下の方にある、予測処理を行うことで予測が可能


※既存の特徴量を減らしたり増やしたりしたい場合（こちらの方が手軽です）
学習の際、すでにある特徴量を操作したい場合、同フォルダ同階層のyamlファイルにある特徴量をコピーし、自分好みに設定（コメントアウト）し、あらたなyamlファイルを作成してください


※自身で特徴量などを作成したい場合
future_engineeringは特徴量作成の関数が詰まったファイルです。好みの特徴量を作成する際は、ここに記述してください
future_engineeringにて特徴量を変更したり作成した場合、同階層のfuture_engineering_predictionを変更してください
また、特徴量を使用するにはyamlファイルを書き換える必要があります
作成したデータフレームの列をもとに、自動でnew.yamlファイルを作成するコードがmain2内のセルにあるので、実行したあと、既存のファイルにコピペするか、ハイパーパラメータなどを記述して、新たにファイルを作成してください


*注意*
自身の環境で特徴量作成処理や学習を行う場合
データフレームの行列が多いため、処理が落ちてカーネルが止まることがあります
自身の環境で作動しない場合、future_engineeringにて列名を減らすか、mainの最初の処理で期間を指定するする際に、作成する年月を減らしてください

また、後藤に連絡してくれれば、処理済みのcsvやpklファイルを送ります





