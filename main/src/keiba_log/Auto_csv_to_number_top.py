import pandas as pd
from pathlib import Path
csv_file_path = Path("../../data/05_prediction_results/prediction_result.csv")

def extract_umaban_turf(csv_path: Path):
    """
    【芝】条件を満たすumaban列の値を抽出
    条件: pred > 0.1, tansho_odds < 100, 期待値(Ex_value) >= 7
    """
    try:
        df = pd.read_csv(csv_path, sep="\t")
        filtered = df[(df["pred"] > 0.1) & (df["tansho_odds"] < 100) & (df["Ex_value"] >= 7)]
        return filtered["umaban"].tolist()
    except Exception as e:
        return 

def extract_umaban_turf_nowin(csv_path: Path):
    """
    【芝未勝利】条件を満たすumaban列の値を抽出
    条件: pred > 0.1, tansho_odds < 100, 期待値(Ex_value) > 6.2
    """
    try:
        df = pd.read_csv(csv_path, sep="\t")
        filtered = df[(df["pred"] > 0.1) & (df["tansho_odds"] < 100) & (df["Ex_value"] > 6.2)]
        return filtered["umaban"].tolist()
    except Exception as e:
        print(f"Error: {e}")
        return []

def extract_umaban_obstract(csv_path: Path):
    """
    【障害】条件を満たすumaban列の値を抽出
    条件: pred > 0.1, tansho_odds < 100, 期待値(Ex_value) > 1
    """
    try:
        df = pd.read_csv(csv_path, sep="\t")
        filtered = df[(df["pred"] > 0.1) & (df["tansho_odds"] < 100) & (df["Ex_value"] > 1)]
        return filtered["umaban"].tolist()
    except Exception as e:
        print(f"Error: {e}")
        return []

def extract_umaban_dirt(csv_path: Path):
    """
    【ダート】条件を満たすumaban列の値を抽出
    条件: pred > 0.1, tansho_odds < 100
    """
    try:
        df = pd.read_csv(csv_path, sep="\t")
        filtered = df[(df["pred"] > 0.1) & (df["tansho_odds"] < 100) & (df["Ex_value"] > 6)]
        return filtered["umaban"].tolist()
    except Exception as e:
        print(f"Error: {e}")
        return []