import math
from pathlib import Path

import pandas as pd

DATA_DIR = Path("..","..", "data_nar")
PREPROCESSED_DIR = DATA_DIR / "01_preprocessed"
TRAIN_DIR = DATA_DIR / "03_train"
OUTPUT_DIR = DATA_DIR / "04_evaluation"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class Evaluator_lightgbm_rank_niti_cross:
    def __init__(
        self,
        return_tables_filename: str = "return_tables.pickle",
        preprocessed_dir: Path = PREPROCESSED_DIR,
        train_dir: Path = TRAIN_DIR,
        evaluation_filename: str = "evaluation_lightgbm_rank_niti_cross.csv",
        output_dir: Path = OUTPUT_DIR,
    ):
        self.return_tables_filepath = preprocessed_dir / return_tables_filename
        
        self.return_tables = pd.read_pickle(self.return_tables_filepath)        
        self.evaluation_df = pd.read_csv(train_dir / evaluation_filename, sep="\t")
        self.output_dir = output_dir

    def box_top_n(
        self,
        sort_col: str = "pred",
        ascending: bool = False,
        n: int = 5,
        exp_name: str = "model",
    ):
        """
        sort_colで指定した列でソートし、上位n件のBOX馬券の的中率・回収率を
        シミュレーションする関数。
        """

        """
        sortする場合は個々で記入
        """
        # self.evaluation_df = self.evaluation_df.query("course_len <= 1500")
        # self.evaluation_df = self.evaluation_df.query("1500 < course_len <= 2200")
        # self.evaluation_df = self.evaluation_df.query("1900 <= course_len")

        # self.evaluation_df = self.evaluation_df.query("weather <= 2")
        # self.evaluation_df = self.evaluation_df.query("2 <  weather")

        # self.evaluation_df = self.evaluation_df.query("ground_state <= 0")
        # self.evaluation_df = self.evaluation_df.query("1 <=  ground_state")

        # self.evaluation_df = self.evaluation_df.query("race_class <= -10")
        # self.evaluation_df = self.evaluation_df.query("3 <= race_class")
        # self.evaluation_df = self.evaluation_df.query("-5 <= race_class")

        # self.evaluation_df = self.evaluation_df.query("mean_age <= 3.4")
        # self.evaluation_df = self.evaluation_df.query("3.4 < mean_age")


        # self.evaluation_df = self.evaluation_df.query("n_horses <=8")
        # self.evaluation_df = self.evaluation_df.query("n_horses <=10")
        # self.evaluation_df = self.evaluation_df.query("12 <= n_horses")
        # self.evaluation_df = self.evaluation_df.query("10 <= n_horses <=15")

        # ・距離1500以下	
        # ・距離1500～2200	
        # ・距離2200～	
            
        # ・天気2以下	
        # ・天気3以上	
            
        # ・馬場状態0	
        # ・馬場状態1以上	
            
        # ・レースクラス4以下	
        # ・５以上	
            
        # ・年齢3歳以下	
        # ・4歳以上	


        #距離、1400未満、1400～1700,1800～2200m,2200～2800m,2800m-
        #天気2以下、2以上
        #馬場状態0、それ以外
        #race_class、4(３勝)以下、以上
        #年齢三歳以下、以上

        # "course_len","weather","ground_state","race_class","age","place",
        # "小雨": 0,
        # "晴": 1,
        # "曇": 2,
        # "雨": 3,
        # "小雪": 4,
        # "雪": 5

        #    "良": 0,
        # "重": 1,
        # "稍": 2,
        # "不": 3,
        # "稍重": 2,
        # "不良": 3



        bet_df = (
            self.evaluation_df.sort_values(sort_col, ascending=ascending)
            .groupby("race_id")
            .head(n)
            .groupby("race_id")["umaban"]
            # 払い戻しテーブルの馬番は文字列型なので合わせる
            .apply(lambda x: list(x.astype(str)))
            .reset_index()
        )
        df = bet_df.merge(self.return_tables, on="race_id")


        # 的中判定。BOXで全通り賭けるシミュレーションなので、集合の包含関係で判定できる。
        df["hit"] = df.apply(
            lambda x: set(x["win_umaban"]).issubset(set(x["umaban"])), axis=1
        )

        # 馬券種ごとの的中率
        agg_hitrate = (
            df.groupby(["race_id", "bet_type"])["hit"]
            .max()
            .groupby("bet_type")
            .mean()
            .rename(f"hitrate_{exp_name}")
            .to_frame()
        )
        # 馬券種ごとの回収率
        df["hit_return"] = df["return"] * df["hit"]
        n_bets_dict = {
            "単勝": n,
            "複勝": n,
            "馬連": math.comb(n, 2),
            "ワイド": math.comb(n, 2),
            "馬単": math.perm(n, 2),
            "三連複": math.comb(n, 3),
            "三連単": math.perm(n, 3),
        }
        agg_df = df.groupby(["race_id", "bet_type"])["hit_return"].sum().reset_index()
        agg_df["n_bets"] = agg_df["bet_type"].map(n_bets_dict)
        agg_df = (
            agg_df.query("n_bets > 0")
            .groupby("bet_type")[["hit_return", "n_bets"]]
            .sum()
        )
        agg_returnrate = (
            (agg_df["hit_return"] / agg_df["n_bets"] / 100)
            .rename(f"returnrate_{exp_name}")
            .to_frame()
            .reset_index()
        )
        output_df = pd.merge(agg_hitrate, agg_returnrate, on="bet_type")
        output_df.insert(0, "topn", n)
        return output_df

    def summarize_box_top_n(
        self,
        sort_col: str = "pred",
        ascending: bool = False,
        n: int = 5,
        exp_name: str = "model",
        save_filename: str = "box_summary_lightgbm_rank_niti.csv",
    ) -> pd.DataFrame:
        """
        topnの的中率・回収率を人気順モデルと比較してまとめる関数。
        """
        summary_df = pd.merge(
            self.box_top_n("popularity", True, n, "pop"),  # 人気順モデル
            self.box_top_n(sort_col, ascending, n, exp_name),  # 実験モデル
            on=["topn", "bet_type"],
        )
        summary_df.to_csv(self.output_dir / save_filename, sep="\t")
        return summary_df

    def summarize_box_exp(
        self,
        exp_name: str = "model"
        
    ) -> pd.DataFrame:
        """
        top1~5の的中率・回収率をまとめる関数。実験モデルの比較に使用。
        """
        summary_df = pd.concat(
            [
                self.box_top_n(n=1, exp_name=exp_name),
                self.box_top_n(n=2, exp_name=exp_name),
                self.box_top_n(n=3, exp_name=exp_name),
                self.box_top_n(n=4, exp_name=exp_name),
                self.box_top_n(n=5, exp_name=exp_name),
            ]
        )
        summary_df2 =pd.concat(
            [
                self.box_top_n("popularity", True,1,"pop"),
                self.box_top_n("popularity", True,2, "pop"),
                self.box_top_n("popularity", True,3, "pop"),
                self.box_top_n("popularity", True,4,"pop"),
                self.box_top_n("popularity", True,5, "pop"),
            ]
        )
        
        summary_df =summary_df.merge(
            summary_df2,
            on=["topn", "bet_type"]
        )  # 人気順モデル
            
        save_filename = f"box_summary_{exp_name}.csv"
        summary_df.to_csv(self.output_dir / save_filename, sep="\t", index=False)
        return summary_df
