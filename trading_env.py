# Import the necessary libraries from the Gymnasium package
import gymnasium as gym
from gymnasium import spaces
# Import other essential libraries
import pandas as pd
import numpy as np

# A class for our custom trading environment that will inherit from gymnasium.Env
class TradingEnv(gym.Env):
    """
    A custom stock trading environment for reinforcement learning.
    This environment is compatible with the Gymnasium API.
    """
    
    # Optional: metadata for rendering, can be useful for visualization
    metadata = {'render_modes': ['human'], 'render_fps': 1}



# highlight-start
    def __init__(self, df, lookback_window=10, initial_capital=100000, transaction_cost_pct=0.001,sharpe_reward_eta=0.1):
        """
        The constructor for the TradingEnv.

        Parameters:
            df (pd.DataFrame): A DataFrame containing the preprocessed and feature-engineered stock data.
            lookback_window (int): The number of previous time steps to include in the observation.
            initial_capital (float): The initial amount of cash the agent starts with.
            transaction_cost_pct (float): The percentage cost for each transaction (buy or sell).
            # highlight-start
            risk_aversion_multiplier (float): A hyperparameter to control how much to penalize risk (volatility).
            # highlight-end
        """
        # It's essential to call the parent class's constructor to ensure proper setup.
        super().__init__()

        # --- 1. Store Essential Data and Parameters ---
        self.df = df
        self.lookback_window = lookback_window # Store the lookback window size
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
# highlight-start
#         # Store the risk aversion parameter
#         self.risk_aversion_multiplier = risk_aversion_multiplier
# # highlight-end
 # highlight-start
        # Store the DSR smoothing factor
        self.sharpe_reward_eta = sharpe_reward_eta
# highlight-end

        # --- 2. Define the dimensions of our data ---\
        # The number of steps is the number of trading days in our DataFrame.
        self.n_steps = len(self.df)
        # The number of features from our dataframe that the agent will observe.
        self.n_features = len(self.df.columns)
        
        # --- 3. Initialize the state of the environment ---\
        # The `current_step` will now need to start after the initial lookback period.
        # We declare these attributes here, but they will be properly initialized in reset()
        self.current_step = 0
        self.cash = 0
        self.shares_held = 0
        self.portfolio_value = 0
        
# # highlight-start
#         # Add attributes to track portfolio returns for volatility calculation
#         self.portfolio_return_history = []
# # highlight-end
#highlight-start
        # Add attributes for the Differential Sharpe Ratio calculation
        self.A = 0.0  # EWMA of returns
        self.B = 0.0  # EWMA of squared returns
# highlight-end


        # Add attributes to store information for rendering
        self._last_action = 2 # Start with 'Hold' action
        self._last_reward = 0.0
        
        # --- 4. Define the Observation Space using a Dictionary Space ---
        # The observation space is a dictionary with two components:
        # 1. 'market_data': A 2D array representing the historical market data (lookback_window x n_features).
        # 2. 'portfolio_status': A 1D array representing the agent's current portfolio state.
         # --- 4. Define the Observation Space using a Dictionary Space ---
        # highlight-start
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(
                low=0, 
                high=1, 
                shape=(self.lookback_window, self.n_features), 
                dtype=np.float32
            ),
            # This is the definition of our agent's self-awareness.
            # It's a Box space for a 1D vector of 3 continuous values.
            'portfolio_status': spaces.Box(
                low=0, 
                high=np.inf, # Cash and portfolio value can theoretically grow infinitely.
                shape=(3,),   # The vector will contain [cash, shares_held, portfolio_value].
                dtype=np.float32
            )
        })
        # highlight-end
        
        # --- 5. Define the Action Space ---
        self.action_space = spaces.Discrete(3)
        # We create a mapping for actions to make rendering more readable
        self._action_to_string = {0: 'Buy', 1: 'Sell', 2: 'Hold'}

    def _get_observation(self):
        """
        Constructs the observation dictionary for the current time step.
        """
        # Get the slice of the DataFrame for the lookback window.
        # It ends at the current_step (inclusive), so we take step - window + 1 to step.
        start_idx = self.current_step - self.lookback_window + 1
        end_idx = self.current_step + 1
        market_features = self.df.iloc[start_idx:end_idx].values.astype(np.float32)
        
         # highlight-start
        # Here we construct the portfolio status vector.
        # It is CRITICAL to normalize these values to ensure they are on a similar
        # scale to our normalized market features.
        portfolio_status = np.array([
            self.cash / self.initial_capital, # Normalize cash relative to the starting capital.
            self.shares_held / 1e6, # Normalize shares by a large, constant factor.
            self.portfolio_value / self.initial_capital # Normalize portfolio value relative to start.
        ], dtype=np.float32)
        # highlight-end


        # Return the observation as a dictionary
        return {
            'market_data': market_features,
            'portfolio_status': portfolio_status
        }

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        """
        super().reset(seed=seed)
        
        # Reset the timeline. The simulation must start at a point where a full
        # lookback window is available. So, we start at `lookback_window - 1`.
        self.current_step = self.lookback_window - 1
        
        # Reset the portfolio to its initial state.
        self.cash = self.initial_capital
        self.shares_held = 0
        self.portfolio_value = self.initial_capital
# # highlight-start
#         # Clear the portfolio return history at the start of a new episode
#         self.portfolio_return_history = []
# # highlight-end
# highlight-start
        # Reset the DSR statistics at the start of a new episode
        self.A = 0.0
        self.B = 0.0
# highlight-end
        # Reset render-specific info
        self._last_action = 2 # Reset to 'Hold'
        self._last_reward = 0.0
        
        # Get the very first observation.
        observation = self._get_observation()
        
        # The info dictionary is required by the Gymnasium API.
        info = {}
        
        return observation, info
        
    def step(self, action):
        """
        Executes one time step within the environment based on the agent's action.
        """
        previous_portfolio_value = self.portfolio_value
        
        # We use the 'Adj Close' of the CURRENT day for transaction price.
        # The current day is the last day in our lookback window.
        current_price = self.df.loc[self.df.index[self.current_step], 'Adj Close']
        
        if action == 0: # Buy
            cost_of_buy = current_price * (1 + self.transaction_cost_pct)
            if self.cash >= cost_of_buy:
                self.shares_held += 1
                self.cash -= cost_of_buy
        elif action == 1: # Sell
            if self.shares_held > 0:
                proceeds_from_sell = (self.shares_held * current_price) * (1 - self.transaction_cost_pct)
                self.cash += proceeds_from_sell
                self.shares_held = 0
            
                # --- Update Portfolio Value ---
    # highlight-start
        # 2. After the action, recalculate the portfolio's new value based on the
        # current market price. This new value reflects the outcome of our action.
        self.portfolio_value = self.cash + (self.shares_held * current_price)
    # highlight-end
# highlight-start
#         # --- 3. Calculate Reward (Refined with Risk Penalty) ---
#         # Calculate the raw profit-and-loss (P&L) for this step
#         pnl = self.portfolio_value - previous_portfolio_value

#         # Calculate the daily return percentage to track volatility
#         # We add a small epsilon to the denominator to avoid division by zero
#         daily_return = (self.portfolio_value / (previous_portfolio_value + 1e-9)) - 1
#         self.portfolio_return_history.append(daily_return)

#         # Calculate the volatility penalty
#         # We only start penalizing after we have a few data points
#         if len(self.portfolio_return_history) > 1:
#             # Standard deviation of returns is a common measure of volatility (risk)
#             volatility = np.std(self.portfolio_return_history)
#             volatility_penalty = volatility * self.risk_aversion_multiplier
#         else:
#             volatility_penalty = 0

#         # The final reward is the P&L minus the risk penalty
#         reward = pnl - volatility_penalty
# # highlight-end

# highlight-start
#         # --- 3. Calculate Reward (Differential Sharpe Ratio) ---
        

#         # Calculate the daily return percentage. Add a small epsilon for numerical stability.
        daily_return = (self.portfolio_value / (previous_portfolio_value + 1e-9)) - 1
        
        # Store previous values of A and B for the DSR calculation
        prev_A = self.A
        prev_B = self.B
        
        # Update the EWMA of returns (A) and squared returns (B)
        self.A = (1 - self.sharpe_reward_eta) * self.A + self.sharpe_reward_eta * daily_return
        self.B = (1 - self.sharpe_reward_eta) * self.B + self.sharpe_reward_eta * (daily_return ** 2)
        
        # Calculate the DSR. Check for the edge case where the denominator is zero.
        # This can happen at the beginning of the episode.
        denominator = (prev_B - prev_A**2)**(3/2)
        if denominator > 1e-9:
            reward = (prev_B * daily_return - prev_A * (daily_return**2)) / denominator
        else:
            reward = 0.0 # Assign a neutral reward if the calculation is unstable
# # highlight-end
        
        #  reward = self.portfolio_value - previous_portfolio_value  #here i canged 


        # Store action and reward for rendering
        self._last_action = action
        self._last_reward = reward
        
        self.current_step += 1
        
        # Check for termination: if we've reached the end of the data
        terminated = self.current_step >= self.n_steps - 1
        
        # Get next observation
        if terminated:
            # If terminated, return a zeroed-out observation that matches the space structure
            observation = {
                'market_data': np.zeros((self.lookback_window, self.n_features), dtype=np.float32),
                'portfolio_status': np.zeros(3, dtype=np.float32)
            }
        else:
            observation = self._get_observation()

        info = {'portfolio_value': self.portfolio_value}
       
        print(f"Reward: {reward}")
        return observation, reward, terminated, False, info
# highlight-end
    def render(self):
        """
        Renders the environment's current state for human consumption.
        This method prints a formatted dashboard to the console.
        """
        # Get the date for the current step
        current_date = self.df.index[self.current_step]
        
        # Format the output string
        render_string = f"--- Step: {self.current_step} | Date: {current_date.strftime('%Y-%m-%d')} ---\n"
        render_string += f"  Action Taken:      {self._action_to_string[self._last_action]}\n"
        render_string += f"  Reward Received:   {self._last_reward:,.2f}\n"
        render_string += f"-----------------------------------------\n"
        render_string += f"  Portfolio Value:   ${self.portfolio_value:,.2f}\n"
        render_string += f"  Cash Balance:      ${self.cash:,.2f}\n"
        render_string += f"  Shares Held:       {self.shares_held}\n"
        render_string += f"-----------------------------------------\n"
        
        print(render_string)
# highlight-end

    def close(self):
        """
        Performs any necessary cleanup. Called when the environment is no longer needed.
        """
        pass




