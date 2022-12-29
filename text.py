from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from pyvirtualdisplay import Display
import time
import urllib.request
import os
import numpy as np
import pandas as pd
from urllib.parse import quote_plus          
from bs4 import BeautifulSoup as bs 
from xvfbwrapper import Xvfb
import time
import warnings
from urllib.request import (urlopen, urlparse, urlunparse, urlretrieve)
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains
import re
from selenium.webdriver.chrome.service import Service
import os 
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import pdb
import os
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import traceback         
from selenium.webdriver.common.proxy import Proxy, ProxyType
import csv
import requests
import ray
import json
import random
import psutil
import pickle
import warnings
import re
import pickle


def find_pages(driver):
    try:
        numbers = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'numbers')))
        numbers = numbers.find_elements(By.TAG_NAME, 'div')

        num = numbers[-1].text
        return int(num)
    except Exception:
        return 1

def selenium_scroll_down(driver):
    SCROLL_PAUSE_SEC = 3
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_SEC)
        new_height = driver.execute_script("return document.body.scrollHeight")
  
        if new_height == last_height:
            return 1
        last_height = new_height

def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()


def click(driver):
    try:
        WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
        content = driver.find_element(By.ID, "sp_message_iframe_737779")
        driver.switch_to.frame(content)
        driver.find_element(By.XPATH, '//*[@id="notice"]/div[3]/button').click()
        driver.switch_to.default_content()
        time.sleep(3)
        return True
    except Exception:
        return True


def get_driver(chrome_options, url, cookies):
    driver = None
    count = 0
    
    while (driver == None) and (count < 10):
            try:

         
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            except Exception:
                count = count + 1
                clean_up()
                if driver: driver.quit()
                continue

    connect = False
    while connect == False: 
        try:
            driver.get(url)

            driver.implicitly_wait(10)
            driver.delete_all_cookies()
            for cookie in cookies: 
                try:
                    driver.add_cookie(cookie)
                except Exception:
                    continue
                
            driver.refresh()
            connect = True
        except Exception:
            del driver
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            continue
    return driver


def screenshot(driver):
    driver.save_screenshot('/home/dhkim/Fragrance/'+str(random.randrange(1,20)) + '.png')

    
def reset_driver(driver, chrome_options, url, cookies):

    try :
        driver = get_driver(chrome_options, url, cookies)
        click(driver)
    except Exception:
        driver = get_driver(chrome_options, url, cookies)
        click(driver)
    #clean_up()
    return driver


def kill_process(name):
    try:
        for proc in psutil.process_iter():
            if proc.name() == name:
                proc.kill()
    except Exception:
        return

def clean_up():
    kill_process('chrome')
    kill_process('chromedriver') 

def get_url(driver, DB, chrome_options,xpath_data, cookies):
 
    url_list = set()
    urls_have = set(DB.loc[:,'url'])

    for i in tqdm(range(1, 2855)):
        time.sleep(random.randrange(1,3))
        try:
            driver.get('https://www.parfumo.net/Reviews?current_page=' +str(i))
        except Exception:
            driver = reset_driver(driver, chrome_options, 'https://www.parfumo.net', cookies)
            driver.get('https://www.parfumo.net/Reviews?current_page=' +str(i))

        try:
            main = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'main')))
            p_boxes = main.find_element(By.CLASS_NAME, 'p-boxes-2')
            p_boxes = p_boxes.find_elements(By.CLASS_NAME, 'p-box')

            for p_box in p_boxes:

                review = p_box.find_element(By.CLASS_NAME, 'grey-box.review-box')
                title = review.find_element(By.CLASS_NAME, 'title')
                url_data = title.find_element(By.TAG_NAME,'a')
                url = url_data.get_attribute('href')
                name = url_data.get_attribute('text')
                
                if url not in urls_have:
                    brand  = review.find_element(By.CLASS_NAME, 'lightblue').get_attribute('text')
                    brand = brand.replace("- ", "")
                    data = {'brand':brand, 'name':name,'year':None,'url':url}
                    DB = DB.append(data, ignore_index = True)
                    DB.to_csv('/home/dhkim/Fragrance/data/fragrance_data2.csv', encoding ='utf-8-sig',  index=False)
                
                url_list.add((name, url))
                
        except Exception:
            continue
    url_list = list(url_list)
    df = pd.DataFrame(url_list)
    df.to_csv('/home/dhkim/Fragrance/data/rating_url.csv', encoding ='utf-8-sig',  index=False)

    return url_list

def get_img(driver):
    try:
        image_holder= WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'p_image_holder')))
        image_url = image_holder.find_element(By.CLASS_NAME, 'p-main-img')
        image_url = image_url.get_attribute('src')
        return image_url
    except Exception:
        try:
            image_holder= driver.find_element(By.ID, 'p_image_imagery_holder')
            image_url = image_holder.find_element(By.CLASS_NAME, 'p-main-img')
            image_url = image_url.get_attribute('src')
            return image_url
        except Exception as e:
            print(e) 
            return None


def img_url(chrome_options,cookies):
    
    driver = get_driver(chrome_options, 'https://www.parfumo.net/Perfumes/Serge_Lutens/ambre-sultan-eau-de-parfum', cookies)
    click(driver)

    data = pd.read_csv("/home/dhkim/Fragrance/data/rating_table3.csv", encoding ='utf-8-sig')
    urls = data.loc[:,'url']

    dic = {}
    visited = set()
    result = []
    i = 0

    for url in tqdm(urls):
        if (url not in visited):
            try:
                visited.add(url)
                driver.get(url)
                driver.implicitly_wait(10)
                img_url = get_img(driver)
                result.append(img_url)
                dic[url] = img_url
            except Exception:
                driver = reset_driver(driver, chrome_options, url, cookies)
        else:
            result.append(dic[url])
        
        if i % 100 == 0:
            print(len(result))
        i += 1

    data['img_url'] = result
    data.to_csv("/home/dhkim/Fragrance/data/rating_table.csv", encoding ='utf-8-sig', index=False)
 


def text_crawler(chrome_options,cookies):
    
    driver = get_driver(chrome_options, 'https://www.parfumo.net/Perfumes/Serge_Lutens/ambre-sultan-eau-de-parfum', cookies)
    click(driver)

    user_list = pd.read_csv("/home/dhkim/Fragrance/data/user_url.csv", encoding ='utf-8-sig')
    
    failed = []

    #result = pd.DataFrame(columns = ['link','text'])
    
    #result = result.drop_duplicates()
    result = pd.read_csv('/home/dhkim/Fragrance/data/text.csv',encoding='utf-8-sig')

    #result['user_rating'] = result['user_rating'].astype(np.float16)

    #visited = set()
    with open('/home/dhkim/Fragrance/data/visited.pkl','rb') as f:
        visited = pickle.load(f)
    
    for i in tqdm(range(len(user_list))):
        user = user_list.loc[i,'id']
        user_url = user_list.loc[i,'url']
        if (user not in visited) & (isinstance(user, str)):
            update = []
            
            user_url = 'https://www.parfumo.net/Users/' + user
        
            review_url = 'https://www.parfumo.net/Users/' + user +'/Reviews'
            statement_url = 'https://www.parfumo.net/Users/' + user + '/Statements'
            
            update1 = review_page(driver, review_url, user)
            update.extend(update1)
            update2 = statement_page(driver, statement_url, user)
            update.extend(update2)

            print(user + ':' + str(len(update)))
            visited.add(user)
            result = write_data(result, update)
            result.to_csv('/home/dhkim/Fragrance/data/text.csv', encoding ='utf-8-sig',  index=False)

            with open('/home/dhkim/Fragrance/data/visited.pkl','wb') as f:
                pickle.dump(visited,f)


def review_page(driver, url, user):

    try:
        driver.get(url)
        driver.implicitly_wait(10)
    except Exception:
        driver = reset_driver(driver, chrome_options, url, cookies)
    
    page_num = find_pages(driver)

    result = []

    for current_page in (range(1, page_num + 1)):
        try:
            goto = url +'?current_page=' + str(current_page)
            driver.get(goto)
            time.sleep(random.randrange(1))
        except:
            driver = reset_driver(driver, chrome_options, goto, cookies)
        try:
            review_boxes = driver.find_elements(By.CLASS_NAME, 'review-box-user-reviews-page.grey-box.mb-2')
            if len(review_boxes) != 0:
                for review in review_boxes:
                    dic = {}
                    link_url = review.find_element(By.CLASS_NAME, 'image.rounded').find_element(By.CLASS_NAME, 'mt-1')
                    link_url = link_url.find_elements(By.TAG_NAME, 'a')[0].get_attribute('href')
                    text = review.find_element(By.CLASS_NAME, 'text.leading-7').text
                    dic['link'] = link_url
                    dic['text'] = text
                    result.append(dic)
            else: return []

        except Exception:
            continue
            
    return result
    

def statement_page(driver, url, user):

    try:
        driver.get(url)
        driver.implicitly_wait(10)
    except Exception:
        driver = reset_driver(driver, chrome_options, url, cookies)
    
    page_num = find_pages(driver)

    result = []

    for current_page in (range(1, page_num + 1)):
        try:
            goto = url +'?current_page=' + str(current_page)
            driver.get(goto)
            time.sleep(random.randrange(1))
        except:
            driver = reset_driver(driver, chrome_options, goto, cookies)

        try:
            statement_boxes = driver.find_elements(By.CLASS_NAME, 'statement')
            if len(statement_boxes) != 0:
                for statement in statement_boxes:
                    dic = {}
                    link_url = statement.find_element(By.CLASS_NAME, 'right.right_first').find_element(By.TAG_NAME, 'a').get_attribute('href')
                    text = statement.find_element(By.CLASS_NAME, 'statement_text_text.pt-2.pl-2.pr-2').text
                    if text.split('\n')[-1].isdigit(): text = text.split('\n')[-2]
                    else: text = text.split('\n')[-1]

                    dic['link'] = link_url
                    dic['text'] = text
                    result.append(dic)
            else: return []

        except Exception:
            continue
            
    return result


def click_more(driver):
    try:
        button = driver.find_element(By.CLASS_NAME, 'btn-more-reviews.pbtn-small.ptbn-transparent.width-100.pointer')
        button.click()
        return True
    except Exception:
        return False

#-------------------------------------------------------------------------------------------------------------------------------
def write_data(write_file, datas):
    
    for data in datas:
        write_file = write_file.append(data, ignore_index = True)
    
    return write_file

def find_last_page(driver):
    try:
        pages = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, "numbers")))
        last = int(pages.find_elements(By.TAG_NAME,'a')[-1].text)
        return last
    except: return None

def find_review_boxes(driver):
    try:
        boxes = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, "p-boxes-2")))
        return boxes.find_elements(By.CLASS_NAME, 'p-box')
    except: return None



#-------------------------------------------------------------------------------------------------------------------------------
def find_statements(driver):
    try:
        main = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, "main")))
        return main.find_elements(By.CLASS_NAME, 'statement')
    except: return None





if __name__ == '__main__':

    vdisplay = Xvfb(width=1920, height=1080)
    vdisplay.start()
    chrome_options = webdriver.ChromeOptions()
    #chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-setuid-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("--disable-extensions")
    #chrome_options.add_argument('--incognito')
    #mobile_emulation = { "deviceName" : "iPhone X" }
    #chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--allow-running-insecure-content')
    chrome_options.add_argument("--single-process")
    chrome_options.add_argument("disable-infobars")
    chrome_options.add_argument("--start-maximized")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36'
    chrome_options.add_argument(f'user-agent={user_agent}')
    os.environ['WDM_LOG_LEVEL'] = '0'
    os.environ['WDM_LOG'] = "false"
  
    with open('/home/dhkim/Fragrance/cookies.csv', 'r', encoding='utf-8-sig') as f:
        cookies = csv.DictReader(f)
        cookies = list(cookies)
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
    #get_user_list(chrome_options, cookies)
    text_crawler(chrome_options,cookies)
    #img_url(chrome_options,cookies)
  

