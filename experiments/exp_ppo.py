import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from physics.env import QubitReadoutEnv

# 1. Initialize the Black-Box Environment
env = QubitReadoutEnv()

# 2. Initialize the PPO Agent
# MlpPolicy creates the Actor and Critic neural networks automatically.
# We set the learning rate to 3e-4 to exactly match the research paper.
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,           # Collect 2048 pulses before updating the networks
    batch_size=64,
    gamma=0.99,             # Discount factor
    verbose=1,
    tensorboard_log="./ppo_readout_logs/"
)

# 3. Setup an Evaluation Callback
# This automatically saves the best model whenever it hits a new high score
eval_callback = EvalCallback(
    env, 
    best_model_save_path='./saved_models/',
    log_path='./ppo_readout_logs/', 
    eval_freq=2048,
    deterministic=True, 
    render_eval=False
)

print('------------------------------STARTING PPO TRAINING-------------------------------')
print("WARNING: Model-Free RL is highly sample inefficient.")
print("This will require significantly more pulses to converge than the Differentiable Physics approach.")

# 4. Train the Agent
# 500,000 timesteps = 500,000 pulses evaluated
model.learn(total_timesteps=50, callback=eval_callback)

print("Training Complete!")