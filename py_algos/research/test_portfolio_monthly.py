import sys
from bdb import Breakpoint
import numpy as np
import pandas as pd

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import random
from download import DATA_FILE
from dateutil.relativedelta import relativedelta

def simple_stock_selector(available_tickers: list, date: datetime, lookback_data: pd.DataFrame) -> list:
    """
    A placeholder function for your stock selection logic.
    In this example, it selects a random subset of 3 tickers.

    In a real backtest, 'lookback_data' would be used to apply rules
    (e.g., momentum, value, technical indicators) to select stocks.

    Args:
        available_tickers (list): List of all tickers available for selection.
        date (datetime): The date of selection (start of the month).
        lookback_data (pd.DataFrame): Historical data up to the selection date.

    Returns:
        list: The list of tickers selected for investment.
    """
    # Simply select a random subset of 3 stocks (or all if less than 3)
    # breakpoint()
    num_to_select = min(10, len(available_tickers))
    selected_stocks = random.sample(available_tickers, num_to_select)
    selected_stocks = ['SPY','QQQ','TSM']
    return selected_stocks

def selecctor_vp_log_slope(available_tickers: list, date: datetime, data: pd.DataFrame, lookback: int) -> list:
    # lookback = 400
    if len(data) < lookback:
        print("history shorter than lookback. skipping.")
        return []
    df_vol = data['Volume']
    df_close = data['Close']
    x = np.linspace(0, lookback, lookback)
    vls = []
    pls = []
    df = pd.DataFrame(index = df_vol.keys())
    # breakpoint()
    for ticker in df_vol.keys().tolist():
        y = df_vol[ticker].values[-lookback:]
        p = np.polyfit(x, np.log(y), 1)
        vls.append(p[0])
        y = df_close[ticker].values[-lookback:]
        p = np.polyfit(x, np.log(y), 1)
        pls.append(p[0])
    df['vls'] = vls
    df['pls'] = pls
    df['vpls'] = np.log(df['vls']) + np.log(df['pls'])
    df = df.dropna(axis=0, how='any').sort_values(by='vpls', ascending=False)
    df = df[(df['vls'] >= 0) & (df['pls'] >= 0)]

    selected = df.index[:10].tolist()
    # breakpoint()
    print(f"selected stock: {selected}")

    return selected


# --- 3. Backtesting Logic ---

class MonthlyBacktester:
    def __init__(self, initial_capital: float, selection_strategy, daily_data: pd.DataFrame):
        """Initializes the backtester."""
        self.capital = initial_capital
        self.selection_strategy = selection_strategy
        self.daily_data = daily_data
        self.portfolio_value = initial_capital
        self.holdings = {}  # {ticker: shares}
        self.trade_log = []  # To track monthly performance

    def run_backtest(self, start_date_str: str, holding_months: int, lookback: int) -> float:
        """
        Runs the monthly rebalancing backtest strategy.
        """
        if self.daily_data.empty:
            print("Cannot run backtest: Daily data is empty.")
            return self.portfolio_value

        print("\n--- Starting Monthly Backtest ---")
        print(f"Initial Capital: ${self.capital:,.2f}")

        # Ensure the date is the primary index level
        # data_indexed_by_date = self.daily_data.swaplevel(0, 1).sort_index()
        data_indexed_by_date = self.daily_data

        # Group data by year and month to find start and end dates
        monthly_groups = data_indexed_by_date.groupby(
            [data_indexed_by_date.index.get_level_values('Date').year,
             data_indexed_by_date.index.get_level_values('Date').month]
        )

        first_month = True
        # breakpoint()
        first_date = None
        last_date = None
        start_date = pd.to_datetime(start_date_str)
        for (year, month), month_data in monthly_groups:
            # Get the first and last trading day of the month
            # Skip the first month if it doesn't represent a full month of trading

            # --- 2. Monthly Buy (Rebalancing) ---

            # Get all tickers that have a price on the buy date
            # breakpoint()
            if not self.holdings:
                buy_date = month_data.index.get_level_values('Date').min()
                if buy_date < start_date:
                    continue
                if first_date is None:
                    first_date = buy_date
                buy_prices_df = month_data.loc[buy_date, 'Close'] #.set_index('Ticker')
                available_tickers = buy_prices_df.index.tolist()

                # Use the stock selection function to pick which stocks to buy
                # breakpoint()
                selected_tickers = self.selection_strategy(
                    available_tickers,
                    buy_date,
                    data_indexed_by_date.loc[data_indexed_by_date.index.get_level_values('Date') <= buy_date],
                    lookback=lookback
                )

                if not selected_tickers:
                    print(f"  > No stocks selected on {buy_date.strftime('%Y-%m-%d')}. Holding cash.")
                    continue

                # Calculate investment amount per stock
                num_stocks = len(selected_tickers)
                investment_per_stock = self.portfolio_value / num_stocks

                # Execute purchases
                for ticker in selected_tickers:
                    buy_price = buy_prices_df.loc[ticker]
                    shares_to_buy = investment_per_stock / buy_price
                    self.holdings[ticker] = shares_to_buy

                # Log the trade
                self.trade_log.append({
                    'Month': f'{year}-{month}',
                    'Action': 'Buy',
                    'Buy Date': buy_date.strftime('%Y-%m-%d'),
                    'Holdings': ', '.join([f'{t}: {s:.2f} shares' for t, s in self.holdings.items()])
                })
                print(f"  > BOUGHT {num_stocks} stocks on {buy_date.strftime('%Y-%m-%d')}. Holding cash: $0.00")

            # --- 1. Monthly Sell (Liquidation) ---
            if self.holdings:
                # breakpoint()
                sell_date = month_data.index.get_level_values('Date').max()
                if sell_date < buy_date + relativedelta(months=holding_months-1):
                    continue
                last_date = sell_date
                liquidation_value = 0
                sell_prices = month_data.loc[sell_date, 'Close']  # .set_index('Ticker')['Close']
                # breakpoint()

                for ticker, shares in self.holdings.items():
                    if ticker in sell_prices:
                        liquidation_value += shares * sell_prices.loc[ticker]
                    else:
                        # Handle case where a stock might delist or have no data on sell_date
                        liquidation_value += shares * month_data.loc[(slice(None), ticker), 'Close'].iloc[-1]
                        print(f"Warning: {ticker} data missing on {sell_date}. Using last available price.")

                # Add profits/losses to the total portfolio value
                self.portfolio_value = liquidation_value
                self.holdings = {}  # Clear holdings

                # Log the trade
                self.trade_log.append({
                    'Month': f'{year}-{month}',
                    'Action': 'Sell',
                    'Sell Date': sell_date.strftime('%Y-%m-%d'),
                    'Portfolio Value': self.portfolio_value
                })
                print(
                    f"  > SOLD on {sell_date.strftime('%Y-%m-%d')}. New Portfolio Value: ${self.portfolio_value:,.2f}")

        # --- 4. Final Valuation at the end of the simulation ---

        if self.holdings:
            final_valuation = 0
            last_date = data_indexed_by_date.index.get_level_values('Date').max()
            final_prices = data_indexed_by_date.loc[last_date, 'Close']

            for ticker, shares in self.holdings.items():
                if ticker in final_prices:
                    final_valuation += shares * final_prices.loc[ticker]
                else:
                    # Fallback to the last available price in the entire dataset
                    final_valuation += shares * data_indexed_by_date.loc[(slice(None), ticker), 'Close'].iloc[-1]

            self.portfolio_value = final_valuation
            print(f"\n--- Final Valuation on {last_date.strftime('%Y-%m-%d')} ---")
            print(f"Remaining Holdings Valued At: ${final_valuation:,.2f}")

        print("\nBacktest Complete.")
        return self.portfolio_value, first_date,last_date


# --- 4. Example Usage ---

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print(f"Usage: {sys.argv[0]}")
        sys.exit(1)
    INITIAL_CAPITAL = 10000.00


    data = pd.read_csv(DATA_FILE, comment='#', header=[0, 1], parse_dates=[0], index_col=0)
    daily_stock_data = data.dropna(axis=1)

    if not daily_stock_data.empty:
        # 3. Run Backtest
        backtester = MonthlyBacktester(
            initial_capital=INITIAL_CAPITAL,
            selection_strategy=selecctor_vp_log_slope,
            daily_data=daily_stock_data
        )

        final_asset_value, first_date,last_date = backtester.run_backtest("2022-01-01", holding_months=3,lookback=350)

        # 4. Results Summary
        return_pct = (final_asset_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        print("\n=============================================")
        print(f"   Backtest Strategy: Monthly Buy & Sell")
        print(f"   Starting Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"   Final Asset Value: ${final_asset_value:,.2f}")
        print(f"   Total Return: {return_pct:.2f}%")
        print(f"   Duration: {first_date} - {last_date}")
        print("=============================================")

        # Optional: Display the trade log
        # print("\n--- Trade Log (First 10 entries) ---")
        # log_df = pd.DataFrame(backtester.trade_log)
        # print(log_df.head(10).to_string())

    else:
        print("\nCould not run the backtest due to data download failure.")
