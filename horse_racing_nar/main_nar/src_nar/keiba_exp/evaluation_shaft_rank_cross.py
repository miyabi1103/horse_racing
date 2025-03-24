import math
from pathlib import Path

import pandas as pd

DATA_DIR = Path("..","..", "data_nar")
PREPROCESSED_DIR = DATA_DIR / "01_preprocessed"
TRAIN_DIR = DATA_DIR / "03_train"
OUTPUT_DIR = DATA_DIR / "04_evaluation"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class Evaluator_lightgbm_rank_niti_shaft:
    def __init__(
        self,
        return_tables_filename: str = "return_tables.pickle",
        preprocessed_dir: Path = PREPROCESSED_DIR,
        train_dir: Path = TRAIN_DIR,
        evaluation_filename: str = "evaluation_lightgbm_rank_niti.csv",
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
        # 一頭軸の的中判定
        def one_head_hit(row):
            # 一頭軸の馬番（先頭1つ目の数字）
            head = row["umaban"][0]
            # 残りの馬番
            other_horses = row["umaban"][1:]
            # win_umabanが一頭軸として的中しているかを判定
            if head in row["win_umaban"]:
                # 残りの馬番のうち、win_umabanに含まれるものを探す
                remaining_hits = [h for h in row["win_umaban"] if h in other_horses]
                # 必要な的中数を満たしているか
                if len(remaining_hits) >= len(row["win_umaban"]) - 1:
                    return True
            return False
        
        
        df["one_head_hit_trio"] = df.apply(one_head_hit, axis=1)
        
        # 二頭軸の的中判定
        def two_head_hit(row):
            # 'umaban'の長さを確認
            umaban_value = row["umaban"]
            if len(umaban_value) < 2:
                return False  # もしくは適切なデフォルト値を返す
            # 二頭軸の馬番（先頭2つの数字）
            head1, head2 = row["umaban"][:2]
            # 残りの馬番
            other_horses = row["umaban"][2:]
            # win_umabanが二頭軸として的中しているかを判定
            if head1 in row["win_umaban"] and head2 in row["win_umaban"]:
                # 残りの馬番のうち、win_umabanに含まれるものを探す
                remaining_hits = [h for h in row["win_umaban"] if h in other_horses]
                # 必要な的中数を満たしているか
                if len(remaining_hits) >= len(row["win_umaban"]) - 2:
                    return True
            return False
        
        df["two_head_hit_trio"] = df.apply(two_head_hit, axis=1)
        
        # 一頭軸の的中判定（軸馬は完全一致、他は順不同）連単系
        def one_head_hit_tai(row):
            # 一頭軸の馬番（先頭1つ目の数字）
            head = row["umaban"][0]
            # 残りの馬番
            other_horses = row["umaban"][1:]
            # 軸馬がwin_umabanの先頭と一致し、残りが順不同で含まれている場合
            if row["win_umaban"][0] == head:
                return set(row["win_umaban"][1:]).issubset(set(other_horses))
            return False
        
        df["one_head_hit_trifecta"] = df.apply(one_head_hit_tai, axis=1)
        
        # ２頭軸の的中判定（軸馬は完全一致、他は順不同）連単系
        def two_head_hit_tai(row):
            # 軸馬（umabanの先頭2つ）
            head_horses = set(row["umaban"][:2])
            # 残りの馬（umabanの残り）
            other_horses = set(row["umaban"][2:])
            # win_umaban の軸部分と残り部分
            win_heads = set(row["win_umaban"][:2])
            win_others = set(row["win_umaban"][2:])
            # 条件判定
            return head_horses == win_heads and win_others.issubset(other_horses)
        
        df["two_head_hit_trifecta"] = df.apply(two_head_hit_tai, axis=1)

                # 馬券種ごとの的中率
        agg_hitrate_one_head = (
            df.groupby(["race_id", "bet_type"])["one_head_hit_trio"]
            .max()
            .groupby("bet_type")
            .mean()
            .rename(f"hit_1head_{exp_name}")
            .to_frame()
        )
        agg_hitrate_two_head = (
            df.groupby(["race_id", "bet_type"])["two_head_hit_trio"]
            .max()
            .groupby("bet_type")
            .mean()
            .rename(f"hit_2head_{exp_name}")
            .to_frame()
        )

        agg_hitrate_one_head_tai = (
            df.groupby(["race_id", "bet_type"])["one_head_hit_trifecta"]
            .max()
            .groupby("bet_type")
            .mean()
            .rename(f"hit_1head_tai_{exp_name}")
            .to_frame()
        )
        agg_hitrate_two_head_tai = (
            df.groupby(["race_id", "bet_type"])["two_head_hit_trifecta"]
            .max()
            .groupby("bet_type")
            .mean()
            .rename(f"hit_2head_tai_{exp_name}")
            .to_frame()
        )

        # 馬券種ごとの回収率
        df["one_head_hit_return"] = df["return"] * df["one_head_hit_trio"]
        df["two_head_hit_return"] = df["return"] * df["two_head_hit_trio"]
        df["one_head_hit_tai_return"] = df["return"] * df["one_head_hit_trifecta"]
        df["two_head_hit_tai_return"] = df["return"] * df["two_head_hit_trifecta"]

        n_bets_dict_one_head = {
            # "単勝": 1,
            # "複勝": 1,
            # "ワイド": n-1,
            
         
            
            "馬連": n-1,
            "三連複": ((n-1)*(n-2))/2,
            "馬単": n-1,
            "三連単": (n-1)*(n-2),   
        }
        n_bets_dict_two_head = {
            # "単勝": 1,
            # "複勝": 1,
            # "ワイド": 1,
            
            "馬連": n-1,
            "三連複": ((n-1)*(n-2))/2,
            "馬単": n-1,
            "三連単": (n-1)*(n-2),
        }
        n_bets_dict_one_head_tai = {
            # "単勝": 1,
            # "複勝": 1,
            # "ワイド": n-1,
            
            "馬連": n-1,
            "三連複": ((n-1)*(n-2))/2,
            "馬単": n-1,
            "三連単": (n-1)*(n-2),
        }
        n_bets_dict_two_head_tai = {
            # "単勝": 1,
            # "複勝": 1,
            # "ワイド": 1,
            
            "馬連": n-1,
            "三連複": ((n-1)*(n-2))/2,
            "馬単": n-1,
            "三連単": (n-1)*(n-2),
        }

        #回収率計算
        agg_df_one_head = df.groupby(["race_id", "bet_type"])["one_head_hit_return"].sum().reset_index()
        agg_df_two_head = df.groupby(["race_id", "bet_type"])["two_head_hit_return"].sum().reset_index()
        agg_df_one_head_tai = df.groupby(["race_id", "bet_type"])["one_head_hit_tai_return"].sum().reset_index()
        agg_df_two_head_tai = df.groupby(["race_id", "bet_type"])["two_head_hit_tai_return"].sum().reset_index()


        agg_df_one_head["n_bets_one"] = agg_df_one_head["bet_type"].map(n_bets_dict_one_head)
        agg_df_two_head["n_bets_two"] = agg_df_two_head["bet_type"].map(n_bets_dict_two_head)
        agg_df_one_head_tai["n_bets_one"] = agg_df_one_head["bet_type"].map(n_bets_dict_one_head_tai)
        agg_df_two_head_tai["n_bets_two"] = agg_df_two_head["bet_type"].map(n_bets_dict_two_head_tai)

        agg_df_one_head  = (
                    agg_df_one_head.query("n_bets_one > 0")
                    .groupby("bet_type")[["one_head_hit_return", "n_bets_one"]]
                    .sum()
        )
        agg_returnrate_one_head = (
            (agg_df_one_head["one_head_hit_return"] / agg_df_one_head["n_bets_one"] / 100)
            .rename(f"return_1head_{exp_name}")
            .to_frame()
            .reset_index()
        )
        
        agg_df_two_head = (
                    agg_df_two_head.query("n_bets_two > 0")
                    .groupby("bet_type")[["two_head_hit_return", "n_bets_two"]]
                    .sum()
        )
        agg_returnrate_two_head = (
            (agg_df_two_head["two_head_hit_return"] / agg_df_two_head["n_bets_two"] / 100)
            .rename(f"return_2head_{exp_name}")
            .to_frame()
            .reset_index()
        )





        agg_df_one_head_tai  = (
                    agg_df_one_head_tai.query("n_bets_one > 0")
                    .groupby("bet_type")[["one_head_hit_tai_return", "n_bets_one"]]
                    .sum()
        )
        agg_returnrate_one_head_tai = (
            (agg_df_one_head_tai["one_head_hit_tai_return"] / agg_df_one_head_tai["n_bets_one"] / 100)
            .rename(f"return_1head_tai_{exp_name}")
            .to_frame()
            .reset_index()
        )
        
        agg_df_two_head_tai = (
                    agg_df_two_head_tai.query("n_bets_two > 0")
                    .groupby("bet_type")[["two_head_hit_tai_return", "n_bets_two"]]
                    .sum()
        )
        agg_returnrate_two_head_tai = (
            (agg_df_two_head_tai["two_head_hit_tai_return"] / agg_df_two_head_tai["n_bets_two"] / 100)
            .rename(f"return_2head_tai_{exp_name}")
            .to_frame()
            .reset_index()
        )




        # 1つ目と2つ目のデータフレームを結合
        merged_1 = pd.merge(agg_hitrate_one_head, agg_returnrate_one_head, on="bet_type")
        
        # その結果を3つ目と4つ目のデータフレームと結合
        output = pd.merge(merged_1, agg_hitrate_two_head,on="bet_type")
        output_1 = pd.merge(output, agg_returnrate_two_head, on="bet_type")


        output_2 = pd.merge(output_1, agg_hitrate_one_head_tai, on="bet_type")
        output_3 = pd.merge(output_2, agg_returnrate_one_head_tai, on="bet_type")


        output_4 = pd.merge(output_3, agg_hitrate_two_head_tai, on="bet_type")
        output_df = pd.merge(output_4, agg_returnrate_two_head_tai, on="bet_type")


        output_df.insert(0, "topn", n)
        return output_df



    def summarize_box_top_n(
        self,
        sort_col: str = "pred",
        ascending: bool = False,
        n: int = 5,
        exp_name: str = "model",
        save_filename: str = "box_summary_lightgbm_rank_shaft.csv",
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
            
        save_filename = f"summary_{exp_name}_shaft.csv"
        summary_df.to_csv(self.output_dir / save_filename, sep="\t", index=False)
        return summary_df
