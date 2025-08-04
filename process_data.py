# Import necessary libraries
import pandas as pd
import os

# --- 1. Define Constants and Load Data ---

# Define the path to the data directory, consistent with our download script
DATA_PATH = 'data'

# Define the list of tickers, ensuring it's the same as in our download script
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ']

# Create an empty dictionary to hold our dataframes
# The keys will be the ticker symbols, and the values will be the pandas DataFrames
dataframes = {}

print("Loading data into DataFrames...")

# Loop through each ticker to load its corresponding CSV file
for ticker in TICKERS:
    # Construct the full file path for the ticker's CSV file
    # e.g., 'data/AAPL.csv'
    file_path = os.path.join(DATA_PATH, f"{ticker}.csv")
    
    # Check if the file actually exists before trying to load it
    if os.path.exists(file_path):
        # Load the CSV file into a pandas DataFrame
        # The new yfinance format has:
        # Row 1: Column names (Price, Adj Close, Close, High, Low, Open, Volume)
        # Row 2: Ticker information
        # Row 3: Empty row
        # Row 4+: Actual data
        # We need to:
        # 1. Read the header row to get column names
        # 2. Skip the ticker and empty rows
        # 3. Use the first column as the index
        df = pd.read_csv(file_path, skiprows=2, index_col=0, parse_dates=True)
        # Set the column names manually since we skipped the header row
        df.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

        
        # Store the loaded DataFrame in our dictionary with the ticker as the key
        dataframes[ticker] = df
        
        print(f"  - Successfully loaded {ticker}. Shape: {df.shape}")
    else:
        # Print a warning if a ticker's CSV file is not found
        print(f"  - Warning: Could not find data file for {ticker} at {file_path}")

print("\nData loading complete.")

# --- 2. Initial Inspection ---

# It's always a good practice to inspect the loaded data.
# Let's check the first few rows and the info of one of the DataFrames (e.g., AAPL).

if 'AAPL' in dataframes:
    print("\n--- Inspecting AAPL DataFrame ---")
    
    # .head() prints the first 5 rows of the DataFrame
    print("First 5 rows:")
    print(dataframes['AAPL'].head())
    
    print("\nDataFrame Info:")
    # .info() gives a concise summary of the DataFrame, including the index type,
    # column types, non-null values, and memory usage.
    dataframes['AAPL'].info()


# 3 one 
print("\n--- Verifying Data Integrity ---")

# We will iterate through each ticker and its corresponding DataFrame
for ticker, df in dataframes.items():
    # .isnull() returns a DataFrame of booleans (True for NaN, False otherwise)
    # .sum() then adds up all the 'True' values (since True=1, False=0) per column.
    missing_values = df.isnull().sum()
    
    # Calculate the total number of missing values in the entire DataFrame
    total_missing = missing_values.sum()
    
    if total_missing > 0:
        # If there are missing values, print a detailed report
        print(f"\nMissing values found in {ticker}:")
        # We only print the columns that have at least one missing value
        print(missing_values[missing_values > 0])
    else:
        # If the DataFrame is complete, print a confirmation message
        print(f"  - No missing values found in {ticker} DataFrame. It's clean!")

print("\nData integrity check complete.")
# highlight-end
    

# highlight-start
# --- 4. Handle Missing Data with Forward-Fill ---

print("\n--- Handling Missing Data ---")

for ticker, df in dataframes.items():
    # Count missing values before filling
    initial_missing = df.isnull().sum().sum()
    
    # Apply forward-fill to the DataFrame.
    # The `fillna` method is the primary tool for handling missing values.
    # `method='ffill'` specifies the forward-fill strategy.
    # `inplace=True` modifies the DataFrame directly, without needing to reassign it
    # (e.g., df = df.fillna(...)). This is often more memory-efficient.
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    
    # As a final check, we can also fill any remaining NaNs (e.g., at the start of the file)
    # with a backward-fill. This is generally safe if `ffill` has been run first.
    

    # Re-verify to ensure no NaNs remain
    final_missing = df.isnull().sum().sum()

    if initial_missing > 0:
        print(f"  - {ticker}: Filled {initial_missing} missing value(s).")
    
    if final_missing == 0:
        print(f"  - {ticker}: Data is now complete. No missing values remain.")
    else:
        # This should ideally not be triggered, but it's a good safeguard.
        print(f"  - WARNING: {ticker} still has {final_missing} missing values after filling.")

print("\nMissing data handling complete.")
# highlight-end

# highlight-start
# --- 5. Feature Engineering: Calculate Simple Moving Averages (SMAs) ---

print("\n--- Engineering Features: Simple Moving Averages (SMAs) ---")

# Define the window sizes for the moving averages we want to calculate.
# 20-day and 50-day are common short-term and medium-term trend indicators.
SMA_WINDOWS = [20, 50]

for ticker, df in dataframes.items():
    print(f"  - Calculating SMAs for {ticker}...")
    for window in SMA_WINDOWS:
        # Define the name for the new SMA column, e.g., 'SMA_20'.
        column_name = f"SMA_{window}"
        
        # Calculate the SMA using the .rolling() method on the 'Adj Close' price.
        # .rolling(window=window) creates a sliding window of the specified size.
        # .mean() then calculates the average of the values within that window.
        # The result is a new column in our DataFrame containing the SMA.
        df[column_name] = df['Adj Close'].rolling(window=window).mean()
        
print("\nSMA feature engineering complete.")

# --- 6. Final Inspection After Feature Engineering ---

# Let's inspect one of the DataFrames again to see our new features.
if 'AAPL' in dataframes:
    print("\n--- Inspecting AAPL DataFrame After Adding SMAs ---")
    
    # Using .tail() shows the last 5 rows, where the SMA values will be populated.
    print("Last 5 rows with new SMA features:")
    print(dataframes['AAPL'].tail())
    
    # Notice the first few rows will have NaN for the SMA columns.
    print("\nFirst 25 rows to show initial NaN values in SMA columns:")
    print(dataframes['AAPL'].head(25))
# highlight-end



# highlight-start
# --- 6. Feature Engineering: Calculate Exponential Moving Averages (EMAs) ---

print("\n--- Engineering Features: Exponential Moving Averages (EMAs) ---")

# Define the spans for the exponential moving averages.
# We use the same periods as the SMAs for consistency and comparison.
EMA_SPANS = [20, 50]

for ticker, df in dataframes.items():
    print(f"  - Calculating EMAs for {ticker}...")
    for span in EMA_SPANS:
        # Define the name for the new EMA column, e.g., 'EMA_20'.
        column_name = f"EMA_{span}"
        
        # Calculate the EMA using the .ewm() method.
        # - span: Specifies the 'N-day' period of the EMA.
        # - adjust=False: This is crucial for financial data. It ensures that the EMA
        #   calculation uses a constant smoothing factor, matching the formula used by
        #   most trading platforms and technical analysis libraries.
        df[column_name] = df['Adj Close'].ewm(span=span, adjust=False).mean()

print("\nEMA feature engineering complete.")

# --- 7. Final Inspection After Feature Engineering ---

# Let's inspect one of the DataFrames again to see all our new features.
if 'AAPL' in dataframes:
    print("\n--- Inspecting AAPL DataFrame After Adding EMAs ---")
    
    # We can use .tail() to see the latest data, where all indicators are populated.
    print("Last 5 rows with new SMA and EMA features:")
    print(dataframes['AAPL'].tail())
    
    # Let's look at the first few rows. Unlike SMA, EMA doesn't start with NaNs
    # when adjust=False, but its early values are heavily based on the first few data points.
    print("\nFirst 25 rows to show initial EMA values:")
    print(dataframes['AAPL'].head(25))
# highlight-end


# highlight-start
# --- 7. Feature Engineering: Calculate Relative Strength Index (RSI) ---

print("\n--- Engineering Features: Relative Strength Index (RSI) ---")

# Define the window for RSI calculation; 14 is the standard period.
RSI_WINDOW = 14

for ticker, df in dataframes.items():
    print(f"  - Calculating RSI for {ticker}...")
    
    # Step 1: Calculate the difference in price from the previous day (delta).
    # The .diff(1) method calculates the difference between an element and the one before it.
    delta = df['Adj Close'].diff(1)
    
    # Step 2: Separate gains (positive changes) and losses (negative changes).
    # .clip(lower=0) replaces all negative values in the Series with 0.
    gain = delta.clip(lower=0)
    
    # .clip(upper=0) replaces all positive values with 0. We then take the absolute value.
    loss = -1 * delta.clip(upper=0)
    
    # Step 3: Calculate the Wilder's Smoothing Moving Average for gains and losses.
    # This is the standard method for RSI and is equivalent to an EMA with alpha = 1 / N.
    # In pandas, this can be achieved using .ewm() with com (center of mass) = N - 1.
    # We use min_periods=RSI_WINDOW to ensure the first calculation has enough data, resulting in NaNs before that point.
    avg_gain = gain.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean()
    avg_loss = loss.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean()
    
    # Step 4: Calculate Relative Strength (RS).
    # We add a small number (epsilon) to the denominator to avoid division by zero errors.
    rs = avg_gain / (avg_loss + 1e-9)
    
    # Step 5: Calculate the final RSI.
    rsi = 100 - (100 / (1 + rs))
    
    # Add the RSI as a new column to our DataFrame.
    df['RSI'] = rsi

print("\nRSI feature engineering complete.")

# --- 8. Final Inspection After Feature Engineering ---

# Let's inspect one of the DataFrames again to see all our new features.
if 'AAPL' in dataframes:
    print("\n--- Inspecting AAPL DataFrame After Adding RSI ---")
    
    # .tail() will show the latest data where RSI is fully calculated.
    print(dataframes['AAPL'][['Adj Close', 'SMA_50', 'EMA_50', 'RSI']].tail())
    
    # .head(20) will show the initial NaN values. The first valid RSI value will appear on the 15th row (index 14).
    # This is because .diff() introduces one NaN, and our ewm min_periods introduces 13 more.
    print("\n")
    print(dataframes['AAPL'][['Adj Close', 'RSI']].head(20))
# highlight-end



# highlight-start
# --- 8. Feature Engineering: Calculate Moving Average Convergence Divergence (MACD) ---

print("\n--- Engineering Features: Moving Average Convergence Divergence (MACD) ---")

# Define the standard short, long, and signal window sizes for MACD.
MACD_SHORT_WINDOW = 12
MACD_LONG_WINDOW = 26
MACD_SIGNAL_WINDOW = 9

for ticker, df in dataframes.items():
    print(f"  - Calculating MACD for {ticker}...")
    
    # Step 1: Calculate the Short-term EMA (12-period)
    ema_short = df['Adj Close'].ewm(span=MACD_SHORT_WINDOW, adjust=False).mean()
    
    # Step 2: Calculate the Long-term EMA (26-period)
    ema_long = df['Adj Close'].ewm(span=MACD_LONG_WINDOW, adjust=False).mean()
    
    # Step 3: Calculate the MACD Line (the difference between short and long EMAs)
    # This is the core of the indicator.
    df['MACD'] = ema_short - ema_long
    
    # Step 4: Calculate the Signal Line (a 9-period EMA of the MACD line)
    # This acts as a trigger line for trading signals.
    df['MACD_Signal'] = df['MACD'].ewm(span=MACD_SIGNAL_WINDOW, adjust=False).mean()
    
    # Step 5: Calculate the MACD Histogram (the difference between the MACD and Signal lines)
    # This visualizes the convergence and divergence, indicating the strength of momentum.
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

print("\nMACD feature engineering complete.")

# --- 9. Final Inspection After All Feature Engineering ---

# Let's inspect one of the DataFrames to see all our new features together.
if 'AAPL' in dataframes:
    print("\n--- Inspecting AAPL DataFrame After Adding MACD ---")
    
    # We select a few key columns for a clean view, including our new MACD features.
    print(dataframes['AAPL'][['Adj Close', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']].tail(10))
# highlight-end




# highlight-start
# --- 9. Normalize Feature Columns ---

# First, we need to import the scaler from scikit-learn
from sklearn.preprocessing import MinMaxScaler

print("\n--- Normalizing Feature Columns ---")

# We must first handle the NaN values that were created by our rolling indicators (SMA, EMA, RSI, etc.).
# Scalers cannot handle missing data. The simplest and most robust approach is to drop any
# rows that contain at least one NaN value. This effectively removes the initial "warm-up"
# period for our indicators, leaving us with a clean, complete dataset for the model.
for ticker, df in dataframes.items():
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)
    print(f"  - {ticker}: Dropped {rows_before - rows_after} initial rows with NaN indicator values.")

# # Define the list of columns that we want to normalize.
# # This should include all the numerical features that our agent will observe.
# COLUMNS_TO_NORMALIZE = [
#     'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
#     'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
#     'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist'
# ]

# # A VERY IMPORTANT NOTE ON DATA LEAKAGE:
# # In a production-grade ML pipeline, you should fit the scaler ONLY on the training data.
# # Then, you use that SAME fitted scaler to transform the validation and test data.
# # This prevents information from the future (e.g., the max price in the test set)
# # from "leaking" into the training process and giving you an unrealistically optimistic model.
# # For simplicity in this step of the project, we are normalizing the entire dataset at once.
# # We will perform the train/val/test split in the next step. Be aware of this
# # simplification—it's a crucial concept in applied machine learning!

# for ticker, df in dataframes.items():
#     # Instantiate the MinMaxScaler. This will scale data to the default range of [0, 1].
#     scaler = MinMaxScaler()
    
#     # We apply the scaler to the selected columns.
#     # .fit_transform() is a convenient method that first 'learns' the min and max values
#     # of each column (the 'fit' part) and then applies the transformation (the 'transform' part).
#     # It returns a NumPy array, which we immediately assign back to the DataFrame's columns,
#     # overwriting the original, unscaled values.
#     df[COLUMNS_TO_NORMALIZE] = scaler.fit_transform(df[COLUMNS_TO_NORMALIZE])
    
#     print(f"  - {ticker}: Columns normalized successfully.")

# print("\nFeature normalization complete.")

# # --- 10. Final Inspection After Normalization ---

# # Let's inspect the data one last time to confirm the normalization.
# if 'AAPL' in dataframes:
#     print("\n--- Inspecting AAPL DataFrame After Normalization ---")
    
#     # The .describe() method is perfect for this. It shows summary statistics.
#     # We should now see that for our normalized columns, the 'min' is 0.0 and the 'max' is 1.0.
#     # This confirms our transformation was successful.
#     print(dataframes['AAPL'][COLUMNS_TO_NORMALIZE].describe())
# # highlight-end


# highlight-start
# --- 10. Chronological Data Splitting (Train, Validation, Test) ---

print("\n--- Splitting Data Chronologically ---")

# Define the split percentages
TRAIN_PCT = 0.70
VALIDATION_PCT = 0.15
# TEST_PCT is implicitly 1 - TRAIN_PCT - VALIDATION_PCT (0.15)

# Create dictionaries to hold the split data for each ticker
train_data = {}
validation_data = {}
test_data = {}

for ticker, df in dataframes.items():
    # Calculate the split indices
    train_end_idx = int(len(df) * TRAIN_PCT)
    validation_end_idx = int(len(df) * (TRAIN_PCT + VALIDATION_PCT))
    
    # Slice the DataFrame using iloc for integer-location based indexing
    train_data[ticker] = df.iloc[:train_end_idx].copy()
    validation_data[ticker] = df.iloc[train_end_idx:validation_end_idx].copy()
    test_data[ticker] = df.iloc[validation_end_idx:].copy()
    
    print(f"  - {ticker}:")
    print(f"    - Training set shape:   {train_data[ticker].shape}")
    print(f"    - Validation set shape: {validation_data[ticker].shape}")
    print(f"    - Test set shape:       {test_data[ticker].shape}")

# --- 11. Normalize Data (The Correct Way - Preventing Data Leakage) ---

from sklearn.preprocessing import MinMaxScaler

print("\n--- Normalizing Data Sets ---")

COLUMNS_TO_NORMALIZE = [
    'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
    'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist'
]

# Loop through each ticker to apply normalization
for ticker in TICKERS:
    # Instantiate the scaler
    scaler = MinMaxScaler()
    
    # Fit the scaler ONLY on the training data for the current ticker
    # This learns the min/max values from the training period ONLY.
    scaler.fit(train_data[ticker][COLUMNS_TO_NORMALIZE])
    
    # Transform the training, validation, and test sets using the FITTED scaler
    # This applies the learned scaling rules to all three datasets.
    train_data[ticker][COLUMNS_TO_NORMALIZE] = scaler.transform(train_data[ticker][COLUMNS_TO_NORMALIZE])
    validation_data[ticker][COLUMNS_TO_NORMALIZE] = scaler.transform(validation_data[ticker][COLUMNS_TO_NORMALIZE])
    test_data[ticker][COLUMNS_TO_NORMALIZE] = scaler.transform(test_data[ticker][COLUMNS_TO_NORMALIZE])
    
    print(f"  - {ticker}: Train, Validation, and Test sets normalized.")

# --- 12. Final Inspection ---
print("\n--- Final Inspection of Normalized Data ---")
# Let's check the test set for AAPL to confirm it's scaled correctly
# Note: The min/max won't be exactly 0 and 1, as it was scaled based on the TRAIN set's range.
# This is correct and proves we have avoided data leakage.
print(test_data['AAPL'][COLUMNS_TO_NORMALIZE].describe())
# highlight-end



