import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

API_KEY = "YOUR_MARKETDATA_API_KEY"
BASE_URL = "https://api.marketdata.app/v1/stocks/candles"

def download_hourly_prices(
    symbol,
    years=1,
    interval="daily",
    chunk_days=30,
    sleep_sec=0.3
):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years)

    rows = []
    cur_start = start

    while cur_start < end:
        cur_end = min(cur_start + timedelta(days=chunk_days), end)

        url = f"{BASE_URL}/{symbol}"
        params = {
            "resolution": interval,
            "from": int(cur_start.timestamp()),
            "to": int(cur_end.timestamp()),
            "apiKey": API_KEY
        }

        r = requests.get(url, params=params)

        # 更友好的错误输出
        if r.status_code != 200:
            print("ERROR:", r.status_code, r.text)
            break

        data = r.json()

        if not data.get("s") == "ok":
            print("No data:", data)
            cur_start = cur_end
            continue

        for t, o, h, l, c, v in zip(
            data["t"],
            data["o"],
            data["h"],
            data["l"],
            data["c"],
            data["v"]
        ):
            rows.append([t, o, h, l, c, v])

        print(f"{symbol} {cur_start.date()} → {cur_end.date()}")

        cur_start = cur_end
        time.sleep(sleep_sec)

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.drop_duplicates("datetime").sort_values("datetime")
    df.set_index("datetime", inplace=True)

    return df
if __name__ == "__main__":
    df = download_hourly_prices('AAPL')











