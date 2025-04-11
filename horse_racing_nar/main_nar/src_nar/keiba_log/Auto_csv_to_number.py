import pandas as pd
from pathlib import Path
# from utils import extract_top_umaban

# CSVファイルのパスを指定
csv_file_path = Path("../../data_nar/05_prediction_results/prediction_result.csv")

# # umaban列の上位3つを取得
# top_umaban = extract_top_umaban(csv_file_path, top_n=3)

# print("上位3つのumaban:", top_umaban)

def extract_top_umaban(csv_path: Path, top_n: int = 3):
    """
    指定されたCSVファイルを読み込み、umaban列の上から指定された数値を抽出する関数。

    Args:
        csv_path (Path): 読み込むCSVファイルのパス。
        top_n (int): 抽出する上位の行数（デフォルトは3）。

    Returns:
        list: umaban列の上位N個の値をリストとして返す。
    """
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(csv_path, sep="\t")

        # umaban列の上位N個を抽出
        top_umaban = df["umaban"].head(top_n).tolist()

        return top_umaban
    except FileNotFoundError:
        print(f"Error: ファイルが見つかりません: {csv_path}")
        return []
    except KeyError:
        print(f"Error: umaban列が存在しません: {csv_path}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []