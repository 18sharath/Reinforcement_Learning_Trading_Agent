# train.py

import pandas as pd
import os
from trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env



# --- Best Practice: Use a main function and a __name__ == "__main__" block ---
def main():
    """
    The main function to orchestrate the training process.
    """
    # --- 1. Configuration and Setup ---
    # Define paths for data, logging, and saving the model.
    # Using os.path.join is a good practice for cross-platform compatibility.
    TICKER = 'AAPL'
    DATA_PATH = os.path.join('processed_data', f'{TICKER}_train.csv')
    TRAIN_LOG_DIR = os.path.join('logs', 'train_logs')
    MODEL_SAVE_DIR = os.path.join('models')
    TENSORBOARD_LOG_DIR = os.path.join('logs', 'tensorboard_logs')

      # Environment and Model Hyperparameters
    INITIAL_CAPITAL = 100000.0
    TRANSACTION_COST_PCT = 0.001
    LOOKBACK_WINDOW = 20 # Use a slightly larger window for more context
    SHARPE_REWARD_ETA = 0.1
    TRAINING_TIMESTEPS = 1_000_000 # Use underscore for readability (Python ignores it)


    # Ensure directories exist
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
  
    os.makedirs(TRAIN_LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# highlight-start
    # --- 2. Load and Prepare Data ---
    print("Step 2: Loading and preparing data...")
    try:
        # Load the entire processed dataset
        df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure you have run the data processing scripts from Step 2.")
        return # Exit if the data isn't there

    # --- Perform a chronological split (70% train, 15% validation, 15% test) ---
    # This is crucial to prevent lookahead bias.
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    # The training data is the first 70% of the dataset
    train_df = df.iloc[:train_size]
    
    # The validation and test sets can be defined for later use, but are not used for training.
    # val_df = df.iloc[train_size : train_size + val_size]
    # test_df = df.iloc[train_size + val_size :]

    print(f"Data loaded. Training period: {train_df.index.min()} to {train_df.index.max()}")
    
    # --- 3. Instantiate the Trading Environment ---
    # This creates an instance of our custom `TradingEnv` class,
    # passing the training data and all the necessary configuration parameters.
    # This `env` object is the "world" our agent will learn in.
    print("Step 3: Instantiating the trading environment...")
    env = TradingEnv(
        df=train_df,
        lookback_window=LOOKBACK_WINDOW,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        sharpe_reward_eta=SHARPE_REWARD_ETA
    )
# highlight-end
    
    
# highlight-start
    # --- 3a. Check the Environment's Compatibility ---
    # This is a crucial sanity check to ensure our custom environment
    # correctly follows the Gymnasium API that stable-baselines3 expects.
    # It will raise an error if any inconsistencies are found.
    print("Step 3a: Checking the environment's compatibility...")
    try:
        check_env(env)
        print("Environment check passed! Your custom environment is compliant.")
    except Exception as e:
        print(f"Environment check failed: {e}")
        # If the check fails, we stop the script because training would be impossible.
        return
# highlight-end




    # --- 4. Instantiate the RL Model (PPO) ---
    # This is the moment we create our AI agent.
    # 'MlpPolicy': Use a Multi-Layer Perceptron as the policy network. This is a standard
    #              choice for environments with vectorized, non-image data like ours.
    # env: This is our custom-built, validated trading environment. The model is now
    #      directly connected to this specific instance of the environment.
    # verbose=1: This will print out training progress information (like mean reward)
    #            to the console, which is very useful for monitoring.
    print("Step 4: Instantiating the PPO model...")
     # We add the `tensorboard_log` parameter, pointing it to our newly defined directory.
    # Now, during training, SB3 will automatically write logs to this location.
   
    model = PPO('MultiInputPolicy', env, verbose=1,tensorboard_log=TENSORBOARD_LOG_DIR)  #here i changed from MlpPolicy


    # --- 5. Train the Model ---
    # This is the command that starts the entire training process.
    # The agent will interact with the environment for the specified number of timesteps,
    # collecting experience and updating its policy network to maximize the reward.
    print(f"Step 5: Training the model for {TRAINING_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TRAINING_TIMESTEPS)

    # --- 6. Save the Trained Model ---
    # After training, we save the agent's learned policy for later evaluation.
    print("Step 6: Saving the trained model...")
    # model.save(os.path.join(MODEL_SAVE_DIR, 'ppo_trading_agent'))
    
    print("\\nTraining complete and model saved!")

if __name__ == '__main__':
    # This block ensures that the main() function is called only when
    # the script is executed directly (e.g., `python train.py`).
    # It won't run if the script is imported as a module into another file.
    main()