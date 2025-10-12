import yfinance as yf
import pandas as pd
import numpy as np
from typing import Literal, Tuple
import os,sys

# --- Configuration ---
TICKER = "AAPL"
START_DATE = "2022-01-01"
END_DATE = "2025-10-01"
INITIAL_CAPITAL = 10000.0
LOOKBACK_DAYS = 20 # New configuration: Number of days to look back for slope calculation

def calculate_log_slope(series: pd.Series) -> float:
    """
    Calculates the slope of the linear regression fit to the log of the series.
    This slope estimates the compound growth rate over the period.

    Args:
        series: A pandas Series (e.g., 'Close' or 'Volume') over the lookback window.

    Returns:
        The slope of the fitted line.
    """
    # Ensure the series is not empty and contains positive values (for log)
    # breakpoint()
    if series.empty:
        return 0.0

    # Convert series to log scale
    log_series = np.log(series.values)

    # Create x-values (0, 1, 2, ..., N-1) representing time
    x = np.arange(len(log_series))

    # Fit a linear regression (degree 1 polynomial) and return the slope (coefficient 0)
    # The result is [slope, intercept]
    slope = np.polyfit(x, log_series, 1)[0]
    return slope

def trade_signal(
    data: pd.DataFrame,
    current_index: int,
    lookback_days: int
) -> Literal[-1, 0, 1]:
    """
    Generates a trade signal based on the log slope of Price and Volume over
    the lookback period, ensuring no look-ahead bias.

    Strategy:
    - BUY (1) if both Price log slope and Volume log slope are positive.
    - SELL (-1) otherwise (or if we don't have enough history).
    - HOLD (0) during the initial warm-up period.

    Args:
        data: The complete historical DataFrame (must contain 'Close' and 'Volume').
        current_index: The integer index of the current day being evaluated.
        lookback_days: The number of days to look back for slope calculation.

    Returns:
        1: Buy signal
        0: Hold signal (during warm-up)
        -1: Sell signal
    """
    # 1. Warm-up check: Ensure we have enough data for the lookback period

    if current_index < lookback_days - 1:
        return 0  # Hold until we have enough data

    # breakpoint()
    # 2. Define the start index for the lookback window
    start_index = current_index - lookback_days + 1

    # 3. Slice the data for the lookback window (up to and including the current day)
    window_data = data.iloc[start_index : current_index + 1]

    # 4. Calculate log slopes
    price_slope = calculate_log_slope(window_data['Close'])
    volume_slope = calculate_log_slope(window_data['Volume'])

    # 5. Signal Logic
    # breakpoint()
    if price_slope > 0 and volume_slope > 0:
        # Both Price and Volume are trending up on a log scale: Strong uptrend confirmation
        return 1  # BUY
    else:
        # If the condition isn't met, exit the position or wait.
        # Since the prompt asks for -1 otherwise, we use -1.
        return -1 # SELL (Exit position)


def backtest_strategy(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
    lookback_days: int # Added parameter for flexibility
) -> Tuple[float, pd.DataFrame]:
    """
    Performs the backtest simulation for a given stock and time frame.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        start_date: Start date for historical data (YYYY-MM-DD).
        end_date: End date for historical data (YYYY-MM-DD).
        initial_capital: The starting amount of cash for the backtest.
        lookback_days: The lookback period for the trade signal.

    Returns:
        A tuple containing the final total return percentage and the
        detailed transaction DataFrame.
    """
    print(f"--- Starting Backtest for {ticker} (Lookback: {lookback_days} days) ---")

    # 1. Download historical data
    try:
        # Need 'Close' and 'Volume' for the new signal
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print("Error: No historical data downloaded. Check ticker/dates.")
            return 0.0, pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while downloading data: {e}")
        return 0.0, pd.DataFrame()

    # 2. Initialize portfolio tracking variables
    cash = initial_capital
    shares = 0.0

    # Prepare a DataFrame to log daily portfolio status and transactions
    # Now includes 'Volume'
    portfolio = data[['Close', 'Volume']].copy()
    portfolio['Signal'] = 0
    portfolio['Shares'] = np.nan
    portfolio['Cash'] = np.nan
    portfolio['Portfolio_Value'] = np.nan

    print(f"Initial Capital: ${initial_capital:,.2f}")

    # 3. Loop through the historical data day by day
    for i in range(len(portfolio)):
        date = portfolio.index[i].strftime('%Y-%m-%d')
        close_price = portfolio['Close'].iloc[i].values[0]

        # 3a. Run the trade signal function for the current day
        signal = trade_signal(portfolio, i, lookback_days)

        # Log the signal
        portfolio.loc[portfolio.index[i], 'Signal'] = signal

        # 3b. Execute trades based on the signal

        # --- BUY (Signal 1) ---
        # Buy if we have a signal and no existing shares
        if signal == 1:
            if cash > 0 and shares == 0:
                shares_to_buy = cash / close_price
                shares += shares_to_buy
                cash = 0.0
                # breakpoint()
                print(f"[{date}] BUY @ ${close_price:.2f}. Acquired {shares_to_buy:.4f} shares.")

        # --- SELL (Signal -1) ---
        # Sell if we have a signal to exit and currently hold shares
        elif signal == -1:
            # breakpoint()
            if shares > 0:
                value_to_sell = shares * close_price
                cash += value_to_sell
                shares = 0.0
                print(f"[{date}] SELL @ ${close_price:.2f}. Total cash: ${cash:,.2f}.")

        # --- HOLD (Signal 0) ---
        # elif signal == 0:
        #    Do nothing (only happens during the initial lookback_days warm-up)

        # 3c. Track daily portfolio metrics
        portfolio.loc[portfolio.index[i], 'Shares'] = shares
        portfolio.loc[portfolio.index[i], 'Cash'] = cash

        current_value = cash + (shares * close_price)
        portfolio.loc[portfolio.index[i], 'Portfolio_Value'] = current_value


    # 4. Final calculation

    # Get the final portfolio value from the last day of the backtest
    final_value = portfolio['Portfolio_Value'].iloc[-1]

    # Calculate the total return
    total_return = (final_value / initial_capital) - 1.0

    # Calculate a simple Buy-and-Hold benchmark
    first_price = data['Close'].iloc[0].values[0]
    last_price = data['Close'].iloc[-1].values[0]
    bh_return = (last_price / first_price) - 1.0

    print("\n--- Backtest Results ---")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return * 100:.2f}%")
    print(f"Buy & Hold Return: {bh_return * 100:.2f}%")

    return total_return, portfolio


# --- Main Execution ---
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ticker>")
        sys.exit(1)

    TICKER = sys.argv[1]
    # Run the backtest
    final_return, portfolio_history = backtest_strategy(
        TICKER,
        START_DATE,
        END_DATE,
        INITIAL_CAPITAL,
        LOOKBACK_DAYS # Pass the new configuration variable
    )

    # Display the first and last few rows of the detailed history
    print("\n--- Portfolio History Sample (First 5 days) ---")
    print(portfolio_history.head())
    print("\n--- Portfolio History Sample (Last 5 days) ---")
    print(portfolio_history.tail())

    # Optional: Save the history to a CSV file
    # portfolio_history.to_csv(f'{TICKER}_backtest_history.csv')
    # print(f"\nSaved detailed history to {TICKER}_backtest_history.csv")
