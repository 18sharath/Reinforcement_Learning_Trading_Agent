# evaluate_baselines.py

import pandas as pd
from baselines import simulate_buy_and_hold
from performance_metrics import calculate_performance_metrics

# --- Configuration ---
# You can change the ticker and file paths as needed.
TICKER = 'AAPL'
PROCESSED_DATA_PATH = f'data/{TICKER}.csv'
INITIAL_CAPITAL = 100000.0
TRANSACTION_COST_PCT = 0.001

# --- Load Data ---
# It's assumed you have your processed data from Step 2 of the project.
# We set the 'Date' column as the index for time-series operations.
try:
    test_data = pd.read_csv(PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
except FileNotFoundError:
    print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}")
    print("Please ensure you have run the data processing scripts from Step 2.")
    exit()

# --- Simulate Buy and Hold ---
print(f"--- Running 'Buy and Hold' Baseline for {TICKER} ---")
# Call the simulation function from our baselines module
final_value_bh, history_bh = simulate_buy_and_hold(
    data=test_data, 
    initial_capital=INITIAL_CAPITAL, 
    transaction_cost_pct=TRANSACTION_COST_PCT
)

# --- Calculate and Print Performance Metrics ---
# Use our new metrics calculator to get the KPIs
metrics_bh = calculate_performance_metrics(history_bh)

# Print the results in a clean, readable format
print("Performance Metrics for Buy and Hold:")
for metric, value in metrics_bh.items():
    if '%' in metric:
        print(f"  {metric}: {value:.2f}%")
    else:
        # Format currency and Sharpe Ratio differently for clarity
        if 'Value' in metric:
            print(f"  {metric}: ${value:,.2f}")
        else:
            print(f"  {metric}: {value:.2f}")






