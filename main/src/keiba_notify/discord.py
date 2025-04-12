import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from urllib.request import Request, urlopen
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import requests
# ディレクトリ設定
DATA_DIR = Path("..", "..", "data")
WATCH_DIR = DATA_DIR / "05_prediction_results"

import re
from dotenv import load_dotenv  # dotenvをインポート

# .envファイルを読み込む
load_dotenv()

# DiscordのWebhook URLを環境変数から取得
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

if not WEBHOOK_URL:
    raise ValueError("Discord Webhook URLが設定されていません。'.env'ファイルを確認してください。")


def post_discord_message(message: str):
    data = {"content": message}
    try:
        response = requests.post(WEBHOOK_URL, json=data)
        if response.status_code in [200, 204]:
            print("Discordにメッセージを送信しました")
        else:
            print(f"メッセージ送信失敗: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"メッセージ送信中にエラーが発生: {e}")
        
def send_race_info_before_image():
    csv_path = Path("../../data/05_prediction_results/prediction_result.csv")
    if not csv_path.exists():
        print("CSVファイルが存在しません")
        return

    df = pd.read_csv(csv_path, sep="\t")

    if df.empty:
        print("CSVが空です")
        return

    # 最初の行から情報を取得
    first_row = df.iloc[0]

    # ローマ字→日本語のマッピングを逆引き辞書に
    place_mapping = {
        'Sapporo': '札幌',
        'Hakodate': '函館',
        'Fukushima': '福島',
        'Niigata': '新潟',
        'Tokyo': '東京',
        'Nakayama': '中山',
        'Chukyo': '中京',
        'Kyoto': '京都',
        'Hanshin': '阪神',
        'Kokura': '小倉'
    }

    try:
        start_time_str = first_row["start_time"]  # 例: '10:30'
        race_time = datetime.strptime(start_time_str, "%H:%M")
        deadline_time = (race_time - timedelta(minutes=1)).strftime("%H:%M")
    except Exception as e:
        print(f"start_timeの処理に失敗: {e}")
        return

    try:
        race_info = first_row["________RACE_INFO________"]

        # 正規表現でローマ字地名とNo.数字を抽出
        place_match = re.search(r"\b([A-Z][a-z]+)\b", race_info)
        number_match = re.search(r"No\.(\d+)", race_info)

        if not place_match or not number_match:
            print(f"レース情報の抽出に失敗: {race_info}")
            return

        place_roman = place_match.group(1)
        race_number = number_match.group(1)

        place_jp = place_mapping.get(place_roman, place_roman)

        race_line = f"レース　　　{place_jp}{race_number}R"
        time_line = f"締切時刻　　{deadline_time}"
        message = f"{race_line}\n{time_line}"

        post_discord_message(message)

    except Exception as e:
        print(f"レース情報の処理に失敗: {e}")





# Discordに画像を送信する関数
def post_discord():
    headers = {
        "User-Agent": "DiscordBot (private use) Python-urllib/3.10",
    }
    send_race_info_before_image()

    # WATCH_DIR内の最新のPNGファイルを取得
    png_files = list(WATCH_DIR.glob("*.png"))
    if not png_files:
        print("PNGファイルが見つかりません")
        return

    # 最新のPNGファイルを取得（更新日時が最も新しいもの）
    latest_file = max(png_files, key=os.path.getmtime)
    print(f"送信するファイル: {latest_file}")

    # ファイルを添付するためのデータ
    with open(latest_file, "rb") as f:
        file_data = f.read()

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{latest_file.name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

    request = Request(
        WEBHOOK_URL,
        data=payload,
        headers=headers,
    )

    try:
        with urlopen(request) as res:
            status_code = res.getcode()
            response_body = res.read().decode("utf-8")
            if status_code in [200, 204]:  # ステータスコードが200または204の場合
                print("Discordに画像を送信しました")
            else:
                print(f"予期しないステータスコード: {status_code}")
                print(f"レスポンス内容: {response_body}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")



