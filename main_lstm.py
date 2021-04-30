from itertools import count
import os
import random
import sys
import gym
import tensorflow as tf
from gym_environments.trading_environment import TradingEnvironment
from optimizers import A2C_LSTM, load_a2c_lstm_model
from utils.generate_tickers import generate_training_test_environments
from utils.test_model import test_lstm_model

tf.get_logger().setLevel('WARNING')

# Set model name
model_name = 'trading_lstm'

# Create training environments
print('Creating training/test environments...')
num_training = 100
num_test = 20
training_envs, test_envs = generate_training_test_environments('data/ticker_list/nyse-listed.csv', num_training, num_test, seed=0)
print('Done setting up environments.')

# Create model to use across training
model_save_location = os.path.join('models', model_name)
actor_save_location = os.path.join(model_save_location, 'actor')
critic_save_location = os.path.join(model_save_location, 'critic')
if os.path.exists(model_save_location):
    print(f'Loaded existing model from {model_save_location}')
    actor, critic = load_a2c_lstm_model(actor_save_location, critic_save_location)
else:
    print(f'Creating new model for {model_name}')
    actor, critic = A2C_LSTM(training_envs[0], lr=0.0003).get_models()
best_test_profit = -float('inf')

# Create optimizers for each training environment
optimizers = [A2C_LSTM(env, actor=actor, critic=critic) for env in training_envs]

# Train until stopped manually
for i in count(1):
    # Train on each environment a single time
    for j, optimizer in enumerate(optimizers):
        sys.stdout.write(f'\rTraining iteration {i} ({j}/{len(optimizers)})...')
        sys.stdout.flush()
        optimizer.train(1)
    print()

    # Determine test performance
    print('Done training, testing...')
    test_profit = test_lstm_model(actor, test_envs)

    # Save model if it's better than any others reached
    if test_profit > best_test_profit:
        print('New best profit reached, saving model...')
        actor.save(actor_save_location)
        critic.save(critic_save_location)
        print(f'Saved model to {model_save_location}.')
