# インポートするライブラリ
import csv
import os
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
class TicketsPurchaser:
    def __init__(self):
        # グローバル変数
        # 曜日リスト
        self.dow_lst = ["月", "火", "水", "木", "金", "土", "日"]
        # レース会場のリスト
        self.place_lst = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
        # JRA IPATのurl
        self.pat_url = "https://www.ipat.jra.go.jp/index.cgi"
        # # INETID
        # self.inet_id = ""
        # # 加入者番号
        # self.kanyusha_no = ""
        # # PATのパスワード
        # self.password_pat = ""
        # # P-ARS番号
        # self.pars_no = ""
        # JRA IPATへの入金金額[yen]
        self.deposit_money = 6000
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
        options.add_argument("--headless=new")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(self.pat_url)
        success_flag = False
        try:
            # PAT購入画面に遷移・ログイン
            # INETIDを入力する
            driver.find_elements(By.CSS_SELECTOR, "input[name^='inetid']")[0].send_keys(self.inet_id)
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
    def deposit(self):
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
    def buy_jra_pat(self, bet_list: list, date_nm):
        driver, success_flag = self.login_jra_pat()
        date_joined = date_nm.strftime("%Y%m%d")
        fieldnames = ['bet_type', 'race_id', 'horse_number']
        if success_flag == True:
            # 購入処理開始
            # 通常投票を指定する
            self.click_css_selector(driver, "button[href^='#!/bet/basic']", 0)
            # logフォルダの中に一日ごとのログファイルを作る
            log_file_path = os.path.join("log", date_joined + ".csv")
            log_file_exist_flag = False
            if os.path.exists(log_file_path):
                log_file_exist_flag = True
                with open(log_file_path, encoding='utf-8', newline='') as csvfile:
                    reader = csv.DictReader(csvfile, fieldnames = fieldnames)
                    loaded_log = [row for row in reader]
            bet_log = []
            for bet_dict in bet_list:
                bet_exist_flag = False
                if log_file_exist_flag == True:
                    for row in loaded_log:
                        if (bet_dict['bet_type'] == row['bet_type']) and (bet_dict['race_id'] == row['race_id']):
                            bet_exist_flag = True
                            break
                if (log_file_exist_flag == True) and (bet_exist_flag == True):
                    print(bet_dict['race_id'], "Bet already exists")
                    continue
                else:
                    try:
                        place = self.place_lst[int(bet_dict['race_id'][4:6]) - 1]
                        dow = self.judge_day_of_week(date_nm)
                        lst = driver.find_elements(By.CSS_SELECTOR, "button[ng-click^='vm.selectCourse(oCourse.courseId)']")
                        for el in lst:
                            if (place in el.text) & (dow in el.text):
                                driver.execute_script("arguments[0].click();", el)
                                sleep(self.wait_sec)
                                break
                        # レース番号を指定する
                        race_nm = int(bet_dict['race_id'][10:12])
                        lst = driver.find_elements(By.CSS_SELECTOR, "button[ng-click^='vm.selectRace(oJgRn.nRaceIndex + 1)']")
                        for el in lst:
                            if str(race_nm) in el.text:
                                driver.execute_script("arguments[0].click();", el)
                                sleep(self.wait_sec)
                                break
                        if bet_dict['bet_type'] == 'umatan':
                            #馬単セレクト
                            o_select_type = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectType']"))
                            o_select_type.select_by_visible_text('馬単')
                            #方式セレクト
                            o_select_method = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectMethod']"))
                            o_select_method.select_by_visible_text('ボックス')
                        elif bet_dict['bet_type'] == 'umaren':
                            #馬連セレクト
                            o_select_type = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectType']"))
                            o_select_type.select_by_visible_text('馬連')
                            #方式セレクト
                            o_select_method = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectMethod']"))
                            o_select_method.select_by_visible_text('ボックス')
                        elif bet_dict['bet_type'] == 'wakuren':
                            #枠連セレクト
                            o_select_type = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectType']"))
                            o_select_type.select_by_visible_text('枠連')
                            #方式セレクト
                            o_select_method = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectMethod']"))
                            o_select_method.select_by_visible_text('ボックス')
                        elif bet_dict['bet_type'] == 'wide':
                            #ワイドセレクト
                            o_select_type = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectType']"))
                            o_select_type.select_by_visible_text('ワイド')
                            #方式セレクト
                            o_select_method = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectMethod']"))
                            o_select_method.select_by_visible_text('ボックス')
                        elif bet_dict['bet_type'] == 'sanrenpuku':
                            #三連複セレクト
                            o_select_type = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectType']"))
                            o_select_type.select_by_index(6)
                            #方式セレクト
                            o_select_method = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectMethod']"))
                            o_select_method.select_by_visible_text('ボックス')
                        elif bet_dict['bet_type'] == 'sanrentan':
                            #三連単セレクト
                            o_select_type = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectType']"))
                            o_select_type.select_by_index(7)
                            #方式セレクト
                            o_select_method = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectMethod']"))
                            o_select_method.select_by_visible_text('ボックス')
                        elif bet_dict['bet_type'] == 'tansho':
                            #単勝セレクト
                            o_select_type = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectType']"))
                            o_select_type.select_by_visible_text('単勝')
                        elif bet_dict['bet_type'] == 'fukusho':
                            #複勝セレクト
                            o_select_type = Select(driver.find_element(By.CSS_SELECTOR, "select[ng-model^='vm.oSelectType']"))
                            o_select_type.select_by_visible_text('複勝')
                        for horse_number in bet_dict['horse_number']:
                            # 購入する馬番をクリック
                            self.click_css_selector(driver, "label[for^=no{}]".format(horse_number), 0)
                            sleep(self.wait_sec)
                        # 購入金額を指定する
                        set_ticket_nm_element = driver.find_element(By.CSS_SELECTOR, "input[ng-model^='vm.nUnit']")
                        set_ticket_nm_element.clear()
                        set_ticket_nm_element.send_keys(self.ticket_nm)
                        # 購入用変数をセットする
                        self.click_css_selector(driver, "button[ng-click^='vm.onSet()']", 0)
                        self.click_css_selector(driver, "button[ng-click^='vm.onShowBetList()']", 0)
                        # 購入する
                        money = driver.find_element(By.CSS_SELECTOR, "span[ng-bind^='vm.getCalcTotalAmount() | number']").text
                        driver.find_element(By.CSS_SELECTOR, "input[ng-model^='vm.cAmountTotal']").send_keys(money)
                        self.click_css_selector(driver, "button[ng-click^='vm.clickPurchase()']", 0)
                        # 購入処理を完了させる
                        self.click_css_selector(driver, "button[ng-click^='vm.dismiss()']", 1)
                        #続けて投票するをクリック
                        driver.find_element(By.CSS_SELECTOR, "button[ng-click^='vm.clickContinue()']").click()
                    except:
                        print(bet_dict['race_id'], "Bet failure")
                        continue
                    sleep(self.wait_sec)
                if bet_exist_flag == False:
                    bet_log.append(bet_dict)
                sleep(self.wait_sec)
            with open(log_file_path, 'a', encoding='utf-8', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
                writer.writerows(bet_log)
            driver.close()
            driver.quit()
        else:
            print("Purchase Failure")