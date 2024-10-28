import sys,os
import robin_stocks.robinhood as rh
from datetime import datetime,timedelta
from mkv_cal import *
from utils import *
username = os.getenv("BROKER_USERNAME")
password = os.getenv("BROKER_PASSWD")
rh.login(username, password)
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: {} <ticker> <days> <rtn>".format(sys.argv[0]))
        sys.exit(1)

    ticker = sys.argv[1]
    days = int(sys.argv[2])
    ub_rtn = float(sys.argv[3])
    lb_rtn = -ub_rtn
    df = download_from_robinhood(ticker)
    steps = days*6
    pu,pd = compProb1stHidBounds(ticker,df,steps,ub_rtn=ub_rtn,lb_rtn=-ub_rtn)
    print(f"with in {days} days, probability of hitting {ub_rtn*100:.2f}%: {pu:.4}, hitting {lb_rtn*100:.2f}%: {pd:.4f}")
    print(f"sum: {pu+pd:.4f}")
