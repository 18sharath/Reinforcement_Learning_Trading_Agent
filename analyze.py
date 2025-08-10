# analyze.py

import pandas as pd
import numpy as np
import os

def calculate_cumulative_return(portfolio_values: pd.Series) -> float:
    """
    Calculates the cumulative return from a pandas Series of portfolio values.

    Args:
        portfolio_values (pd.Series): A pandas Series where each entry is the
                                      portfolio's total value at a point in time.
                                      The series should be chronologically ordered.

    Returns:
        float: The cumulative return as a decimal (e.g., 0.5 for 50%).
    """
    # Ensure the series is not empty
    if portfolio_values.empty:
        return 0.0

    # Get the starting value (the first entry in the series)
    # .iloc[0] is used to access the first item by its integer position.
    starting_value = portfolio_values.iloc[0]

    # Get the ending value (the last entry in the series)
    # .iloc[-1] is used to access the last item by its integer position.
    ending_value = portfolio_values.iloc[-1]
    
    # Check if starting value is zero to avoid division by zero error
    if starting_value == 0:
        return 0.0

    # Calculate the cumulative return using the formula
    cumulative_return = (ending_value / starting_value) - 1
    
    return cumulative_return


# highlight-start
def calculate_annualized_return(portfolio_values: pd.Series) -> float:
    """
    Calculates the annualized return (CAGR) from a pandas Series of portfolio values.

    Args:
        portfolio_values (pd.Series): A pandas Series of portfolio values with a
                                      DateTimeIndex.

    Returns:
        float: The annualized return as a decimal (e.g., 0.12 for 12%).
    """
    if portfolio_values.empty:
        return 0.0
        
    # Get the starting and ending values
    starting_value = portfolio_values.iloc[0]
    ending_value = portfolio_values.iloc[-1]

    # Calculate the number of years.
    # The index must be a DateTimeIndex.
    start_date = portfolio_values.index[0]
    end_date = portfolio_values.index[-1]
    
    # Calculate the number of days in the period
    num_days = (end_date - start_date).days
    
    # Handle edge case where the period is less than a day
    if num_days == 0:
        return 0.0

    # Convert the number of days to years. Using 365.25 to account for leap years.
    num_years = num_days / 365.25

    # Calculate the annualized return (CAGR)
    # First, calculate the total return factor
    total_return_factor = ending_value / starting_value

    # Then, apply the CAGR formula
    # Note: We use max(0, total_return_factor) to handle cases where the strategy loses all money,
    # preventing errors from taking a root of a negative number.
    annualized_return = (max(0, total_return_factor)) ** (1 / num_years) - 1
    
    return annualized_return
# highlight-end

def calculate_annualized_volatility(portfolio_values: pd.Series) -> float:
    """
    Calculates the annualized volatility of a portfolio.

    Args:
        portfolio_values (pd.Series): A pandas Series of portfolio values.

    Returns:
        float: The annualized volatility as a decimal.
    """
    if portfolio_values.empty or len(portfolio_values) < 2:
        return 0.0

    # Calculate daily returns from the portfolio values.
    # .pct_change() computes the percentage change from the previous element.
    # The first value will be NaN, so we drop it.
    daily_returns = portfolio_values.pct_change().dropna()
    
    # Calculate the standard deviation of the daily returns. This is the daily volatility.
    daily_volatility = daily_returns.std()
    
    # Annualize the volatility by multiplying by the square root of the number of trading days in a year (252).
    # This scaling is based on the assumption that returns follow a random walk.
    annualized_volatility = daily_volatility * np.sqrt(252)
    
    return annualized_volatility
def calculate_sharpe_ratio(portfolio_values: pd.Series, risk_free_rate: float) -> float:
    """
    Calculates the Sharpe Ratio of a portfolio.

    Args:
        portfolio_values (pd.Series): A pandas Series of portfolio values.
        risk_free_rate (float): The annualized risk-free rate as a decimal (e.g., 0.02 for 2%).

    Returns:
        float: The annualized Sharpe Ratio.
    """
    # This function beautifully demonstrates code reuse by calling our other metrics.
    annualized_return = calculate_annualized_return(portfolio_values)
    annualized_volatility = calculate_annualized_volatility(portfolio_values)
    
    # It's crucial to handle the case where volatility is zero to avoid a division-by-zero error.
    # This can happen if the portfolio value never changed.
    if annualized_volatility == 0:
        return 0.0
        
    # Calculate the excess return over the risk-free rate
    excess_return = annualized_return - risk_free_rate
    
    # Calculate the Sharpe Ratio
    sharpe_ratio = excess_return / annualized_volatility
    
    return sharpe_ratio
def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Calculates the Maximum Drawdown of a portfolio.

    Args:
        portfolio_values (pd.Series): A pandas Series of portfolio values.

    Returns:
        float: The maximum drawdown as a negative decimal (e.g., -0.25 for a 25% loss).
    """
    if portfolio_values.empty:
        return 0.0

    # Step 1: Calculate the high-water mark (the running maximum).
    # .cummax() creates a series where each element is the maximum value seen so far.
    high_water_mark = portfolio_values.cummax()
    
    # Step 2: Calculate the drawdown series.
    # This is the percentage drop from the high-water mark to the current value.
    drawdown = (portfolio_values - high_water_mark) / high_water_mark
    
    # Step 3: Find the maximum drawdown (which will be the minimum value in the series).
    # We use .min() because drawdowns are represented as negative numbers.
    max_drawdown = drawdown.min()
    
    return max_drawdown

def calculate_sortino_ratio(portfolio_values: pd.Series, risk_free_rate: float) -> float:
    """
    Calculates the Sortino Ratio of a portfolio.

    Args:
        portfolio_values (pd.Series): A pandas Series of portfolio values.
        risk_free_rate (float): The annualized risk-free rate, used as the target return.

    Returns:
        float: The annualized Sortino Ratio.
    """
    annualized_return = calculate_annualized_return(portfolio_values)
    
    # Step 1: Calculate daily returns
    daily_returns = portfolio_values.pct_change().dropna()
    
    # Step 2: Calculate the target daily return (de-annualize the risk-free rate)
    target_daily_return = risk_free_rate / 252
    
    # Step 3: Identify returns below the target
    # This creates a series where below-target returns are kept, and others are zero.
    downside_diff = target_daily_return - daily_returns
    downside_diff[downside_diff < 0] = 0
    
    # Step 4: Calculate the squared differences and get the downside deviation
    downside_variance = (downside_diff ** 2).mean()
    daily_downside_deviation = np.sqrt(downside_variance)
    
    # Step 5: Annualize the downside deviation
    annualized_downside_deviation = daily_downside_deviation * np.sqrt(252)
    
    if annualized_downside_deviation == 0:
        # Return infinity if there's no downside risk but positive excess return
        return np.inf if (annualized_return - risk_free_rate) > 0 else 0.0

    # Step 6: Calculate the Sortino Ratio
    excess_return = annualized_return - risk_free_rate
    sortino_ratio = excess_return / annualized_downside_deviation
    
    return sortino_ratio

def simulate_buy_and_hold(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """
    Simulates a Buy and Hold strategy.
    
    Args:
        df (pd.DataFrame): DataFrame with at least a 'Close' price column and DateTimeIndex.
        initial_capital (float): The starting capital for the backtest.
        
    Returns:
        pd.Series: A series of portfolio values over time.
    """
    first_day_price = df['Close'].iloc[0]
    shares_bought = initial_capital / first_day_price
    
    # Calculate portfolio value for each day
    portfolio_values = shares_bought * df['Close']
    portfolio_values.name = 'portfolio_value'
    
    return portfolio_values
def simulate_sma_crossover(
    df: pd.DataFrame, 
    initial_capital: float, 
    short_window: int, 
    long_window: int, 
    transaction_cost_pct: float
) -> pd.Series:
    """
    Simulates a Simple Moving Average (SMA) Crossover strategy.
    
    Args:
        df (pd.DataFrame): Processed data with 'Close', 'SMA_short', and 'SMA_long' columns.
        initial_capital (float): Starting capital.
        short_window (int): The short SMA window size.
        long_window (int): The long SMA window size.
        transaction_cost_pct (float): The percentage cost per trade.
        
    Returns:
        pd.Series: A series of portfolio values over time.
    """
    # Use the SMA columns from the processed data
    sma_short_col = f'SMA_{short_window}'
    sma_long_col = f'SMA_{long_window}'
    
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    
    # Create the signal: 1 for buy (short > long), 0 for sell (short < long)
    signals['position'] = np.where(df[sma_short_col] > df[sma_long_col], 1.0, 0.0)
    
    # Take the difference to generate the trading signal (entry/exit)
    signals['signal'] = signals['position'].diff()
    
    cash = initial_capital
    shares_held = 0.0
    portfolio_values = []

    for index, row in df.iterrows():
        signal = signals.loc[index, 'signal']
        current_price = row['Close']
        
        # Buy signal
        if signal == 1.0 and cash > 0:
            shares_to_buy = cash / current_price
            cost = shares_to_buy * current_price * (1 + transaction_cost_pct)
            cash -= cost
            shares_held += shares_to_buy
            
        # Sell signal
        elif signal == -1.0 and shares_held > 0:
            sale_value = shares_held * current_price * (1 - transaction_cost_pct)
            cash += sale_value
            shares_held = 0.0
            
        current_portfolio_value = cash + (shares_held * current_price)
        portfolio_values.append(current_portfolio_value)
        
    return pd.Series(portfolio_values, index=df.index, name='portfolio_value')

def main():
    """
    Main function to load backtest results and run analysis.
    """
    TICKER = 'AAPL'
    RISK_FREE_RATE = 0.02
    INITIAL_CAPITAL = 100000.0
    TRANSACTION_COST_PCT = 0.001
    SHORT_WINDOW = 20 # Must match the one used in data processing
    LONG_WINDOW = 50  # Must match the one used in data processing

    # --- 1. Load the Backtest Results ---
    results_path = os.path.join('results', 'backtest_results.csv')
    print(f"Loading backtest results from: {results_path}")
    
    try:
        # Load the CSV, ensuring the 'date' column is parsed as dates and set as the index.
        # This is crucial for any time-series analysis.
        results_df = pd.read_csv(results_path, index_col='date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}")
        print("Please run `evaluate.py` first to generate the backtest results.")
        return

    print("Backtest results loaded successfully.")

    # --- 2. Calculate and Display KPIs ---
    data_path = os.path.join('processed_data', f'{TICKER}_test.csv')
    print(f"Loading processed market data from: {data_path}")
    try:
        processed_df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Processed data not found. Run the data processing script from Step 2.")
        return

    test_start_date = results_df.index[0]
    test_df = processed_df.loc[test_start_date:]
    
    # Extract the 'portfolio_value' column to pass to our function
    # --- 3. Run Baseline Simulations ---
    print("Simulating baseline strategies on the test period...")
    buy_and_hold_values = simulate_buy_and_hold(test_df, INITIAL_CAPITAL)
    sma_crossover_values = simulate_sma_crossover(
        test_df, INITIAL_CAPITAL, SHORT_WINDOW, LONG_WINDOW, TRANSACTION_COST_PCT
    )

    # --- 4. Calculate KPIs for All Strategies ---
    strategies = {
        'RL Agent': results_df['portfolio_value'],
        'Buy and Hold': buy_and_hold_values,
        'SMA Crossover': sma_crossover_values
    }

    performance_data = {}

    for name, values in strategies.items():
        print(f"Calculating KPIs for {name}...")
        performance_data[name] = {
            'Cumulative Return': calculate_cumulative_return(values),
            'Annualized Return (CAGR)': calculate_annualized_return(values),
            'Annualized Volatility': calculate_annualized_volatility(values),
            'Sharpe Ratio': calculate_sharpe_ratio(values, RISK_FREE_RATE),
            'Sortino Ratio': calculate_sortino_ratio(values, RISK_FREE_RATE),
            'Maximum Drawdown': calculate_max_drawdown(values)
        }

         # --- 5. Generate and Display the Comparison Table ---
    # Convert the dictionary of results into a Pandas DataFrame for beautiful printing
    performance_df = pd.DataFrame.from_dict(performance_data, orient='index')

    # Format the DataFrame for better readability
    performance_df['Cumulative Return'] = performance_df['Cumulative Return'].map('{:.2%}'.format)
    performance_df['Annualized Return (CAGR)'] = performance_df['Annualized Return (CAGR)'].map('{:.2%}'.format)
    performance_df['Annualized Volatility'] = performance_df['Annualized Volatility'].map('{:.2%}'.format)
    performance_df['Maximum Drawdown'] = performance_df['Maximum Drawdown'].map('{:.2%}'.format)
    performance_df['Sharpe Ratio'] = performance_df['Sharpe Ratio'].map('{:.2f}'.format)
    performance_df['Sortino Ratio'] = performance_df['Sortino Ratio'].map('{:.2f}'.format)


    print("\n\n--- Strategy Performance Comparison ---")
    print(performance_df)
    # Save the performance table to a CSV file
    performance_save_path = os.path.join('results', 'performance_comparison.csv')
    performance_df.to_csv(performance_save_path)
    print(f"\nPerformance comparison table saved to: {performance_save_path}")
    

if __name__ == '__main__':
    main()