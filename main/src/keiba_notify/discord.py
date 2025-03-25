import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from urllib.request import Request, urlopen
from pathlib import Path

# ディレクトリ設定
DATA_DIR = Path("..", "..", "data")
WATCH_DIR = DATA_DIR / "05_prediction_results"


from dotenv import load_dotenv  # dotenvをインポート

# .envファイルを読み込む
load_dotenv()

# DiscordのWebhook URLを環境変数から取得
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

if not WEBHOOK_URL:
    raise ValueError("Discord Webhook URLが設定されていません。'.env'ファイルを確認してください。")

# Discordに画像を送信する関数
def post_discord():
    headers = {
        "User-Agent": "DiscordBot (private use) Python-urllib/3.10",
    }

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