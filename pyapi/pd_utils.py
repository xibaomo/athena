import pandas as pd
import numpy as np

df = pd.DataFrame()

def addCol(header, lx):
    global df
    df[header] = np.array(lx)

def dump_csv(fn):
    global df
    print(df)
    df.to_csv(fn, index = False)

def hello():
    print("hello")
