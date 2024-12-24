
使用方法は後半に記述

ソースコードの実行環境
・OS: Mac OS 13.6.9
・言語: Python 3.11.8

beautifulsoup4==4.12.3
lightgbm==4.5.0

numpy==2.2.1
pandas==2.2.2
matplotlib                3.9.2  
pyyaml                    6.0.2 
scikit-learn              1.5.2 
selenium                  4.24.0  
tqdm                      4.66.5
webdriver_manager==4.0.2

■ディレクトリ構成
.
├── requirements.txt                ・・・必要なライブラリを記載
├── common（スクレイピングなどの処理）
│   ├── data
│   │   ├── html
│   │   │   ├── race
│   │   │   │   └── {race_id}.bin   ・・・スクレイピングしたraceページのhtml
│   │   │   ├── horse
│   │   │   │   └── {horse_id}.bin  ・・・スクレイピングしたhorseページのhtml
│   │   │   ├── ped
│   │   │   │   └── {horse_id}.bin  ・・・スクレイピングしたpedページのhtml
│   │   │   └── leading
│   │   │       ├── jockey_leading  ・・・スクレイピングした騎手リーディングページのhtml
│   │   │       ├── trainer_leading ・・・スクレイピングした調教師リーディングページのhtml
│   │   │       ├── sire_leading    ・・・スクレイピングした種牡馬リーディングページのhtml
│   │   │       └── bms_leading     ・・・スクレイピングしたBMSリーディングページのhtml
│   │   ├── rawdf                   ・・・Pandas.DataFrameのrawデータを保存するディレクトリ
│   │   │   ├── results.csv
│   │   │   ├── horse_results.csv
│   │   │   ├── horse_results_prediction.csv
│   │   │   ├── peds.csv
│   │   │   ├── jockey_leading.csv
│   │   │   ├── trainer_leading.csv
│   │   │   ├── sire_leading.csv
│   │   │   ├── bms_leading.csv
│   │   │   ├── return_tables.csv
│   │   │   └── race_info.csv
│   │   ├── mapping                 ・・・カテゴリ変数から整数へのマッピング
│   │   │   ├── around.json
│   │   │   ├── ground_state.json
│   │   │   ├── race_class.json
│   │   │   ├── race_type.json
│   │   │   ├── place.json
│   │   │   ├── sex.json
│   │   │   └── weather.json
│   │   └── prediction_population   ・・・予測母集団（開催日, race_id, horse_id）
│   │       └── population.csv
│   └── src
│       ├── create_rawdf.py                     ・・・htmlをDataFrameに変換する関数を定義
│       ├── scraping_prediction_population.py   ・・・直近馬場状態をスクレイピングするための関数を定義
│       ├── main.ipynb                          ・・・コードを実行するnotebook
│       ├── dev.ipynb                           ・・・開発用notebook
│       ├── create_prediction_population.py     ・・・予測母集団を作成する関数を定義
│       └── scraping.py                         ・・・スクレイピングする関数を定義
└── turf/dirt/obstract/etc...
    ├── data
    │   ├── 00_population       ・・・学習母集団を保存するディレクトリ
    │   │   └── population.csv
    │   ├── 01_preprocessed     ・・・前処理済みのデータを保存するディレクトリ
    │   │   ├── horse_results.csv
    │   │   ├── horse_results_prediction.csv
    │   │   ├── peds.csv
    │   │   ├── peds_prediction.csv
    │   │   ├── return_tables.pickle
    │   │   ├── results.csv
    │   │   ├── jockey_leading.csv
    │   │   ├── trainer_leading.csv
    │   │   ├── sire_leading.csv
    │   │   ├── bms_leading.csv
    │   │   └── race_info.csv
    │   ├── 02_features         ・・・全てのテーブルを集計・結合した特徴量を保存するディレクトリ
    │   │   └── features.csv
    │   ├── 03_train            ・・・学習結果を保存するディレクトリ
    │   │   ├── model.pkl           ・・・学習済みモデル
    │   │   ├── evaluation.csv      ・・・検証データに対する予測結果
    │   │   ├── importance.csv      ・・・特徴量重要度（一覧）
    │   │   └── importance.png      ・・・特徴量重要度（上位を可視化）
    │   │   ├── model_odds_removed.pkl           ・・・学習済みモデル（オッズと人気を抜いたモデル）
    │   │   ├── evaluation_odds_removed.csv      ・・・検証データに対する予測結果（オッズと人気を抜いたモデル）
    │   │   ├── importance_odds_removed.csv      ・・・特徴量重要度（オッズと人気を抜いたモデル）
    │   │   └── importance_odds_removed.png      ・・・特徴量重要度（オッズと人気を抜いたモデル）
    │   └── 04_evaluation       ・・・検証データに対する精度評価結果を保存するディレクトリ
    └── src
        ├── dev.ipynb               ・・・開発用notebook
        ├── main.ipynb              ・・・コードを実行するnotebook
        ├── create_population.py    ・・・学習母集団を作成する関数を定義
        ├── preprocessing.py        ・・・/common/rawdf/のデータを前処理する関数を定義
        ├── feature_engineering.py  ・・・機械学習モデルにインプットする特徴量を作成するクラスを定義（学習時）
        ├── feature_engineering_prediction.py ・・・機械学習モデルにインプットする特徴量を作成するクラスを定義（予測時）
        ├── config.yaml             ・・・学習に用いる特徴量一覧
        ├── config_odds_removed.yaml・・・学習に用いる特徴量一覧（オッズと人気を抜いたモデル）
        ├── train.py                ・・・学習処理を行うクラスを定義
        ├── evaluation.py           ・・・モデルの精度評価を行うクラスを定義
        └── prediction.py           ・・・予測処理を行う関数を定義




これ以外にもファイルを追加していますが、タイトルから読み取ってください
_edaでは分析
_kitaitiでは単勝における期待値の最適を考えています

_crossや_cv系ファイルはクロスバリデーションを行ったものです


turf/dirt/obstaclesはそれぞれ、芝、ダート、障害を予想する用で分けられています
_newbieはそれぞれのタイプで新馬や過去レースが少ない馬が走る場合を見越した予測版です、学習時に過去戦績がない馬に最適化するようにしています


yamlファイルには特徴量とハイパーパラメータが記述されています
変更はそちらのファイルで行ってください




____________________________________________________________


・使用方法
スクレイピングがまだの場合や、追加でスクレイピングを行いたい場合は
commonフォルダのsrc/main2を上から実行する


スクレイピングが終わっている場合
turf/dirt/obstaclesフォルダにあるいずれかのsrc/main2を上から実行すること
特徴量作成処理にはかなり時間がかかります、場合によっては落ちます



特徴量作成処理が終わっている場合
そのままmain2を進み、モデルの学習を行なってください

モデルの学習の種類は
目的変数がrank二値、回帰、time回帰
そのホールドアウトverかクロスバリデーションver
があり

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
turf/dirt/obstaclesフォルダにあるいずれかのsrc/main2の下の方にある、予測処理を行うことで予測が可能


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
自身の環境で作動しない場合、future_engineeringにて列名を減らすか、main2の最初の処理で期間を指定するする際に、作成する年月を減らしてください

また、後藤に連絡してくれれば、処理済みのcsvやpklファイルを送ります






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



血統データ（父、母父の勝率や得意な長さ、タイプなど）
騎手データ
調教師データ

スピード指数
上り指数







