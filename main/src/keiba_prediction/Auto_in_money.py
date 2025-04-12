import asyncio
import re
from playwright.async_api import Playwright, async_playwright, expect
import pandas as pd
from io import StringIO
from collections import defaultdict
from bs4 import BeautifulSoup
import numpy as np
import ast

import sys
from pathlib import Path
# インポートするライブラリ
import csv
import datetime
from time import sleep
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.select import Select


import matplotlib
matplotlib.use("Agg")  # GUIバックエンドを無効化
import matplotlib.pyplot as plt
import seaborn as sns
DATA_DIR = Path("..", "..", "data")
SAVE_DIR = DATA_DIR / "05_prediction_results"
from dotenv import load_dotenv

load_dotenv(override=True)
# .envファイルを読み込む
load_dotenv()

import os

# DiscordのWebhook URLを環境変数から取得
INET_URL = os.getenv("INET_ID")
KANYUSYA_URL = os.getenv("KANYUSYA_NO")
PASSWORD_URL = os.getenv("PASSWORD_PAT")
PARS_URL = os.getenv("PARS_NO")

if not INET_URL:
    raise ValueError("INET_IDが設定されていません。'.env'ファイルを確認してください。")
if not KANYUSYA_URL:
    raise ValueError("KANYUSYA_NOが設定されていません。'.env'ファイルを確認してください。")
if not PASSWORD_URL:
    raise ValueError("PASSWORD_PATが設定されていません。'.env'ファイルを確認してください。")
if not PARS_URL:
    raise ValueError("PARS_NOが設定されていません。'.env'ファイルを確認してください。")

# if not INET_URL:
#     raise ValueError("INET_IDが設定されていません。'.env'ファイルを確認してください。")
# if not KANYUSYA_URL:
#     raise ValueError("KANYUSYA_NOが設定されていません。'.env'ファイルを確認してください。")
# if not PASSWORD_URL:
#     raise ValueError("PASSWORD_PATが設定されていません。'.env'ファイルを確認してください。")
# if not PARS_URL:
#     raise ValueError("PARS_NOが設定されていません。'.env'ファイルを確認してください。")




class TicketsPurchaser:
    def __init__(self):
        # グローバル変数
        # 曜日リスト
        self.dow_lst = ["月", "火", "水", "木", "金", "土", "日"]
        # レース会場のリスト

        self.place_lst = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
        # JRA IPATのurl
        self.pat_url = "https://www.ipat.jra.go.jp/index.cgi"
        # INETID
        self.inet_id = INET_URL
        # 加入者番号
        self.kanyusha_no = KANYUSYA_URL
        # PATのパスワード
        self.password_pat = PASSWORD_URL
        # P-ARS番号
        self.pars_no = PARS_URL
        # JRA IPATへの入金金額[yen]
        self.deposit_money = 20000
        # 馬券の購入枚数
        self.ticket_nm = 1
        # seleniumの待機時間[sec]
        self.wait_sec = 2
    # 自作関数
    def judge_day_of_week(self, date_nm):
        date_dt = datetime.datetime.strptime(str(date_nm), "%Y-%m-%d")
        # 曜日を数字で返す(月曜：1 〜 日曜：7)
        nm = date_dt.isoweekday()
        return self.dow_lst[nm - 1]
    def click_css_selector(self, driver, selector, nm):
        el = driver.find_elements(By.CSS_SELECTOR, selector)[nm]
        driver.execute_script("arguments[0].click();", el)
        sleep(self.wait_sec)
    def scrape_balance(self, driver):
        return int(np.round(float(driver.find_element(By.CSS_SELECTOR, ".text-lg.text-right.ng-binding").text.replace(',', '').strip('円')) / 100))
    def check_and_write_balance(self, driver, date_joined):
        balance = self.scrape_balance(driver)
        deposit_txt_path = "log/money/deposit.txt"
        balance_csv_path = "log/money/" + date_joined[:4] + ".csv"
        if balance != 0:
            with open(deposit_txt_path, 'w', encoding='utf-8', newline='') as deposit_txt:
                deposit_txt.write(str(balance))
            with open(balance_csv_path, 'a', encoding='utf-8', newline='') as balance_csv:
                writer = csv.writer(balance_csv)
                writer.writerow([datetime.datetime.now().strftime("%Y%m%d%H%M"), str(balance)])
        return balance
    def login_jra_pat(self):
        options = Options()
        # ヘッドレスモード
        # options.headless = True
        # options.add_argument("--headless=new")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(self.pat_url)
        success_flag = False
        
        try:
            # PAT購入画面に遷移・ログイン
            # INETIDを入力する
            driver.find_elements(By.CSS_SELECTOR, "input[name^='inetid']")[0].send_keys(self.inet_id)
            # sleep(3)
            
            self.click_css_selector(driver, "a[onclick^='javascript']", 0)
            sleep(self.wait_sec)
            # 加入者番号，PATのパスワード，P-RAS番号を入力する
            driver.find_elements(By.CSS_SELECTOR, "input[name^='p']")[0].send_keys(self.password_pat)
            driver.find_elements(By.CSS_SELECTOR, "input[name^='i']")[2].send_keys(self.kanyusha_no)
            driver.find_elements(By.CSS_SELECTOR, "input[name^='r']")[1].send_keys(self.pars_no)
            self.click_css_selector(driver, "a[onclick^='JavaScript']", 0)
            # お知らせがある場合はOKを押す
            if "announce" in driver.current_url:
                self.click_css_selector(driver, "button[href^='#!/']", 0)
            success_flag = True
        except:
            print("Login Failure")
            driver.close()
            driver.quit()
            success_flag = False
        return driver, success_flag
    def auto_in_money(self):
        driver, success_flag = self.login_jra_pat()
        if success_flag == True:
            # 入出金ページに遷移する(新しいタブに遷移する)
            self.click_css_selector(driver, "button[ng-click^='vm.clickPayment()']", 0)
            driver.switch_to.window(driver.window_handles[1])
            # 入金指示を行う
            self.click_css_selector(driver, "a[onclick^='javascript'", 1)
            nyukin_amount_element = driver.find_elements(By.CSS_SELECTOR, "input[name^='NYUKIN']")[0]
            nyukin_amount_element.clear()
            nyukin_amount_element.send_keys(self.deposit_money)
            self.click_css_selector(driver, "a[onclick^='javascript'", 1)
            driver.find_elements(By.CSS_SELECTOR, "input[name^='PASS_WORD']")[0].send_keys(self.password_pat)
            self.click_css_selector(driver, "a[onclick^='javascript'", 1)
            # 確認事項を承諾する
            Alert(driver).accept()
            sleep(self.wait_sec)
            driver.close()
            driver.quit()
        else:
            print("Deposit Failure")
if __name__ == "__main__":
    purchaser = TicketsPurchaser()
    purchaser.auto_in_money()