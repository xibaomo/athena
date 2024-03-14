import time
import numpy as np
import pandas as pd
from selenium import webdriver
import requests
import os
import random
import yfinance as yf
from datetime import datetime,timedelta
from sklearn.feature_selection import mutual_info_regression
import pdb
def save_webpage(url,sym):
    try:
        # Initialize Firefox WebDriver
        driver = webdriver.Firefox()

        # Open the webpage
        driver.get(url)

        # Get the page source
        page_source = driver.page_source

        # Create a temporary directory to save the webpage
        file_path = os.path.join("data", sym+".html")

        # Save the webpage content to a file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(page_source)

        return file_path
    except Exception as e:
        print(f"Failed to save the webpage: {e}")
        return None
    finally:
        # Close the WebDriver
        driver.quit()

def download_webpages(syms):
    sym2page = {}
    for sym in syms:
        url = "https://finance.yahoo.com/quote/" + sym + "/key-statistics"
        fn = save_webpage(url,sym)
        sym2page[sym] = fn
        sec = random.uniform(0.5,1)
        time.sleep(sec)

    return sym2page
def get_file_stem(fn):
    s = os.path.splitext(fn)[0]
    return s
def get_statistics(folder):
    files = os.listdir(folder)
    syms=[]
    PEs = []
    PBs = []
    evt2rev=[]
    evt2prf=[]
    for f in files:
        stem = get_file_stem(f)
        df = pd.read_html('data/'+f)[0]
        if df.iloc[:,1].isna().any():
            continue
        if not 'Trailing P/E' in df.iloc[:,0].values:
            continue
        if (df.iloc[:,1]=='--').any():
            continue
        try:
            syms.append(stem)
            PEs.append(float(df.iloc[2,1]))
            PBs.append(float(df.iloc[6,1]))
            evt2rev.append(float(df.iloc[7,1]))
            evt2prf.append(float(df.iloc[8,1]))
        except:
            pdb.set_trace()
        print(f)
        # pdb.set_trace()
    dff = pd.DataFrame()
    dff['SYM'] = syms
    dff['PE'] = PEs
    dff['PB'] = PBs
    dff['evt2rev'] = evt2rev
    dff['evt2prf'] = evt2prf

    return dff

if __name__ == "__main__":
    syms = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    syms.append('TSM')

    # sym2page = download_webpages(syms)
    df = get_statistics('data/')
    end_date = datetime.now()

    # Calculate the start date as 12 months before today
    start_date = end_date - timedelta(days=365)
    prices = yf.download(df['SYM'].tolist(),start_date,end_date)['Adj Close']
    rtns = prices.iloc[-1]/prices.iloc[0]-1.
    df['rtn'] = rtns.values
    df.set_index('SYM',inplace=True)
    df = df.dropna()
    X=df.values
    mutual_info_regression(X, X[:,-1])

    sorted_df = df.sort_values(by=['PE','PB'])
