# evaluate.py

import pandas as pd
import os
import numpy as np

# We need to import our custom environment
from trading_env import TradingEnv
# We need to import the PPO model class from stable-baselines3
from stable_baselines3 import PPO

def main():
    """
    The main function to orchestrate the evaluation of the trained agent.
    """
    # --- 1. Configuration and Setup ---
    # Define paths and key parameters. These should match the training configuration
    # to ensure a fair and consistent evaluation environment.
    TICKER = 'AAPL'
    DATA_PATH = os.path.join( 'processed_data', f'{TICKER}_test.csv')
    MODEL_PATH = os.path.join('models', 'ppo_trading_agent.zip')
    
    # Environment parameters must be identical to the ones used for training
    INITIAL_CAPITAL = 100000.0
    TRANSACTION_COST_PCT = 0.001
    LOOKBACK_WINDOW = 20
    # The sharpe_reward_eta is not used during evaluation but is part of the env's signature
    SHARPE_REWARD_ETA = 0.1 

    # --- 2. Load the Trained Model ---
    # In the next task, we will load the agent's saved "brain".
    print(f"Step 2: Loading the trained PPO model from {MODEL_PATH}...")
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure you have successfully run `train.py` to create and save the model.")
        return # Exit if the model isn't there

    print("Model loaded successfully.")


    # --- 3. Load and Prepare the Test Data ---
    # It is CRITICAL to use the exact same data processing steps and splits
    # to get the test set that the model has NEVER seen before.
    print("Step 3: Loading and preparing the test data...")
    try:
        df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure you have run the data processing scripts from Step 2.")
        return

    # --- Perform the same chronological split ---
    # We must use the exact same split percentages to isolate the unseen test data.
   
    
    # The test data is the final 15% of the dataset.
    # This is the out-of-sample data the agent has never encountered.
    test_df = df
    
    print(f"Data loaded. Test period: {test_df.index.min()} to {test_df.index.max()}")
    print(f"Number of days in test set: {len(test_df)}")

    # --- 4. Instantiate the Test Environment ---
    # We create a new instance of our TradingEnv, but this time,
    # we pass it the `test_df`.
    print("Step 4: Instantiating the test environment...")
    
    test_env = TradingEnv(
        df=test_df,
        lookback_window=LOOKBACK_WINDOW,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        sharpe_reward_eta=SHARPE_REWARD_ETA # Must be passed to satisfy the constructor
    )
    # --- 5. Run the Evaluation Loop (Backtest) ---
    print("Step 5: Running the evaluation loop (backtest)...")

    # Reset the environment to get the initial observation. This is the agent's
    # very first perception of the test world. The `obs` variable now holds the
    # state for the first day of the test period. We ignore the second return
    # value (`info` dictionary) with an underscore `_`.
    obs, _ = test_env.reset()

    # The backtesting loop iterates through each day in the test dataset.
    # The length of our test_df determines how many steps the evaluation will run.
    for i in range(len(test_df)):
        # Predict: Use the loaded model to predict the best action.
        # `deterministic=True` ensures the agent doesn't explore and uses its best learned strategy.
        action, _ = model.predict(obs, deterministic=True)
        
        # Act & Observe: Pass the action to the environment's `step` method.
        # This simulates taking the action and returns the results for the next day.
        # - obs: The observation for the *next* day, which will be used in the next loop iteration.
        # - reward: The reward received (we don't use this for evaluation analysis, but it's returned).
        # - done/truncated: Flags indicating if the episode is over.
        # - info: A dictionary containing our custom logging info (portfolio value, trades, etc.).
        obs, reward, done, truncated, info = test_env.step(action)
        
        # It's good practice to check if the episode has ended and break the loop.
        # For our environment, this will happen on the last day of the test data.
        if done or truncated:
            print("Evaluation finished.")
            break


    

    # Finally, we will collect the results from the backtest
    # and calculate our key performance indicators (KPIs).
    # --- 6. Store and Analyze Results ---
    print("Step 6: Storing and analyzing results...")
    
    # Convert the environment's history (a list of dictionaries) into a Pandas DataFrame.
    # This is the most critical step for preparing the data for analysis.
    backtest_results_df = pd.DataFrame(test_env.history)
    
    # Set the 'date' column as the index, which is standard practice for time-series data.
    backtest_results_df.set_index('date', inplace=True)

    # Display the first 5 rows to verify the data
    print("\n--- Backtest Results (First 5 Days) ---")
    print(backtest_results_df.head())

    # Display the last 5 rows to see the final outcome
    print("\n--- Backtest Results (Last 5 Days) ---")
    print(backtest_results_df.tail())
    
    # Best Practice: Save the backtest results to a CSV file for persistence and sharing.
    # This allows you to analyze the results later without re-running the evaluation.
    results_save_path = os.path.join('results', 'backtest_results.csv')
    os.makedirs('results', exist_ok=True) # Ensure the 'results' directory exists
    backtest_results_df.to_csv(results_save_path)
    print(f"\nBacktest results saved to: {results_save_path}")
    
    print("\nEvaluation complete.")

if __name__ == '__main__':
    # This ensures the main function is called only when the script is executed directly.
    main()
