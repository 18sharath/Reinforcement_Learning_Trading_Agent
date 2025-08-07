# evaluate_baselines.py
import os
import pandas as pd
from baselines import simulate_buy_and_hold, simulate_sma_crossover
from performance_metrics import calculate_performance_metrics

# --- Configuration ---
# You can change the ticker and file paths as needed.
TICKER = 'AAPL'

PROCESSED_DATA_PATH = os.path.join('processed_data', f"{TICKER}_processed.csv")

# PROCESSED_DATA_PATH = f'data/{TICKER}.csv'
INITIAL_CAPITAL = 100000.0
TRANSACTION_COST_PCT = 0.001

SHORT_WINDOW = 20
LONG_WINDOW = 50

# highlight-start
# Define the output path for our results
RESULTS_DIR = 'results'
BASELINE_METRICS_PATH = os.path.join(RESULTS_DIR, 'baseline_metrics.csv')
# highlight-end

# --- Load Data ---
# It's assumed you have your processed data from Step 2 of the project.
# We set the 'Date' column as the index for time-series operations.
try:
    test_data = pd.read_csv(PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
except FileNotFoundError:
    print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}")
    print("Please ensure you have run the data processing scripts from Step 2.")
    exit()

# highlight-start
# --- Data Structure to Hold All Results ---
# We will create a list to hold the metrics dictionary for each strategy.
# This list of dictionaries is a perfect format to convert into a pandas DataFrame.
all_results = []
# highlight-end



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
# highlight-start
# Add a 'Strategy' key to the dictionary before appending it to our list
metrics_bh['Strategy'] = 'Buy and Hold'
all_results.append(metrics_bh)
# highlight-end



# highlight-start
# --- Simulate SMA Crossover ---\n
print(f"--- Running 'SMA Crossover' Baseline for {TICKER} ---")
# Call the new simulation function for the SMA Crossover strategy
_, history_sma = simulate_sma_crossover(
    data=test_data,
    initial_capital=INITIAL_CAPITAL,
    transaction_cost_pct=TRANSACTION_COST_PCT,
    short_window=SHORT_WINDOW,
    long_window=LONG_WINDOW
)

# --- Calculate and Print Performance Metrics ---\n
# We reuse the exact same metrics calculator!
metrics_sma = calculate_performance_metrics(history_sma)

# highlight-start
# Add a 'Strategy' key to this dictionary as well
metrics_sma['Strategy'] = f'SMA Crossover ({SHORT_WINDOW}/{LONG_WINDOW})'
all_results.append(metrics_sma)
# highlight-end

# highlight-start
# --- Consolidate and Save Results ---

# Convert the list of dictionaries into a single pandas DataFrame
results_df = pd.DataFrame(all_results)

# Set the 'Strategy' column as the index of the DataFrame for clear, tabular display
results_df.set_index('Strategy', inplace=True)

# Ensure the results directory exists before trying to save the file
os.makedirs(RESULTS_DIR, exist_ok=True)

# Save the DataFrame to a CSV file. The index=True argument ensures that our
# strategy names are saved as the first column.
results_df.to_csv(BASELINE_METRICS_PATH, index=True)

# Print the final, consolidated DataFrame to the console
print("\n" + "="*50)
print("      CONSOLIDATED BASELINE PERFORMANCE")
print("="*50)
# We use pd.options to format the floating point numbers for a cleaner printout
with pd.option_context('display.float_format', '{:,.2f}'.format):
    print(results_df)
print("="*50)
print(f"\nBaseline results have been saved to: {BASELINE_METRICS_PATH}")
# highlight-end



# # Print the results for the SMA Crossover strategy
# print(f"Performance Metrics for SMA Crossover ({SHORT_WINDOW}/{LONG_WINDOW}):")
# for metric, value in metrics_sma.items():
#     if '%' in metric:
#         print(f"  {metric}: {value:.2f}%")
#     else:
#         if 'Value' in metric:
#             print(f"  {metric}: ${value:,.2f}")
#         else:
#             print(f"  {metric}: {value:.2f}")
# # highlight-end







