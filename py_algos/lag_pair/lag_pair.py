import pandas as pd
import numpy as np
import os,sys
import itertools

DATA_FILE = "data_downloaded.csv"
def locate_target_date(date_str, df):
    print("locating target date: ", date_str)
    tar_date = pd.to_datetime(date_str)
    prev_days = 1000000
    for i in range(len(df)):
        dt = tar_date - df.index[i]
        if dt.days == 0:
            return i
        if prev_days * dt.days < 0:
            return i
        prev_days = dt.days
    return -1
def generate_all_pairs(symbols):
    pairs = list(itertools.combinations(symbols, 2))
    return pairs

def score_pair(df,cm,pair):
    cr = cm[pair[0]][pair[1]]
    mu1 = df[pair[0]].values[-1]/df[pair[0]].values[0] - 1.
    mu2 = df[pair[1]].values[-1]/df[pair[1]].values[0] - 1.
    return abs(mu1-mu2)/(1-cr),cr,mu1,mu2

def generate_score_table(pairs,df):
    cm = df.corr()
    pair2score = {}
    for pair in pairs:
        score,_,_,_ = score_pair(df,cm,pair)
        pair2score[pair] = score
    return pair2score
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} <date>".format(sys.argv[0]))
        sys.exit(1)
    data = pd.read_csv(DATA_FILE, comment='#', header=[0, 1], parse_dates=[0], index_col=0)
    data = data.dropna(axis=1)

    target_date = sys.argv[1]

    df_close = data['Close']
    pairs = generate_all_pairs(df_close.keys())
    global_tid = locate_target_date(sys.argv[1], df_close)
    df = df_close.iloc[global_tid-180:global_tid]
    score_table = generate_score_table(pairs,df)
    score_table = dict(sorted(score_table.items(), key=lambda item: item[1], reverse=True))

    cm = df.corr()
    for key, value in list(score_table.items())[:20]:
        _,cr,mu1,mu2 = score_pair(df,cm,key)
        print(key, value,cr,mu1,mu2)

