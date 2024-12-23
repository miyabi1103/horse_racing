import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = Path("..", "data")
PREPROCESSED_DIR = DATA_DIR / "01_preprocessed"
TRAIN_DIR = DATA_DIR / "03_train"
OUTPUT_DIR = DATA_DIR / "04_evaluation"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class Evaluator_lightgbm_rank_niti_cross:
    def __init__(
        self,
        return_tables_filepath: Path = PREPROCESSED_DIR / "return_tables.pickle",
        train_dir: Path = TRAIN_DIR,
        evaluation_filename: str = "evaluation_lightgbm_rank_niti_cross.csv",
        output_dir: Path = OUTPUT_DIR,
    ):
        self.return_tables = pd.read_pickle(return_tables_filepath)
        self.evaluation_df = pd.read_csv(train_dir / evaluation_filename, sep="\t")
        self.output_dir = output_dir

    def box_top_n(
        self,
        evaluation_fold_df: pd.DataFrame,
        sort_col: str = "pred",
        ascending: bool = False,
        n: int = 5,
        exp_name: str = "model",
    ) -> pd.DataFrame:
        """
        Sort the evaluation DataFrame by `sort_col`, take the top N predictions, and calculate hit rate and return rate.
        """
        bet_df = (
            evaluation_fold_df.sort_values(sort_col, ascending=ascending)
            .groupby("race_id")
            .head(n)
            .groupby("race_id")["umaban"]
            .apply(lambda x: list(x.astype(str)))  # Ensure umaban is a string to match with return_tables
            .reset_index()
        )
        df = bet_df.merge(self.return_tables, on="race_id")
        
        df["hit"] = df.apply(
            lambda x: set(x["win_umaban"]).issubset(set(x["umaban"])), axis=1
        )
        
        # Calculate hit rate
        agg_hitrate = (
            df.groupby(["race_id", "bet_type"])["hit"]
            .max()
            .groupby("bet_type")
            .mean()
            .rename(f"hitrate_{exp_name}")
            .to_frame()
        )
        
        # Calculate return rate
        df["hit_return"] = df["return"] * df["hit"]
        n_bets_dict = {
            "単勝": n,
            "複勝": n,
            "馬連": math.comb(n, 2),
            "ワイド": math.comb(n, 2),
            "馬喫": math.perm(n, 2),
            "三連複": math.comb(n, 3),
            "三連喫": math.perm(n, 3),
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

    def evaluate_cross_validation(
        self, 
        n_splits: int = 5, 
        exp_name: str = "model", 
        save_filename: str = "box_summary_lightgbm_rank_niti_cross.csv"
    ) -> pd.DataFrame:
        """
        Perform cross-validation evaluation for top 1-5 predictions.
        """
        # サンプル数を取得
        n_samples = len(self.evaluation_df)
        
        # n_splitsをサンプル数に合わせる
        n_splits = min(n_splits, n_samples - 1)  # 少なくとも1サンプルがテストセットに必要
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(self.evaluation_df)):
            fold_df = self.evaluation_df.iloc[test_idx]
            print(f"Evaluating fold {fold_idx + 1} / {n_splits}")
            for n in range(1, 6):
                result_df = self.box_top_n(fold_df, n=n, exp_name=f"{exp_name}_fold{fold_idx+1}")
                result_df.insert(0, "fold", fold_idx + 1)
                fold_results.append(result_df)
        
        summary_df = pd.concat(fold_results, ignore_index=True)
        summary_df.to_csv(self.output_dir / save_filename, sep="\t", index=False)
        return summary_df

    def summarize_box_exp(
        self, 
        exp_name: str = "model", 
        n_splits: int = 5
    ) -> pd.DataFrame:
        """
        Summarize the hit rate and return rate for top N (1 to 5) across all folds.
        """
        summary_df = self.evaluate_cross_validation(n_splits=n_splits, exp_name=exp_name)
        agg_summary_df = (
            summary_df.groupby(["topn", "bet_type"])[[f"hitrate_{exp_name}", f"returnrate_{exp_name}"]]
            .mean()
            .reset_index()
        )
        
        save_filename = f"box_summary_{exp_name}_crossval.csv"
        agg_summary_df.to_csv(self.output_dir / save_filename, sep="\t", index=False)
        return agg_summary_df