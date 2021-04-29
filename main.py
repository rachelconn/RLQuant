from itertools import count
import os
import random
import sys
import gym
import tensorflow as tf
from gym_environments.trading_environment import TradingEnvironment
from optimizers import A2C, load_a2c_model
from utils.generate_tickers import generate_training_test_environments
from utils.test_model import test_model

tf.get_logger().setLevel('WARNING')

# Set model name
model_name = 'trading'

# Create training environments
print('Creating training/test environments...')
num_training = 100
num_test = 20
training_envs, test_envs = generate_training_test_environments('data/ticker_list/nyse-listed.csv', num_training, num_test, seed=0)
print('Done setting up environments.')

# Create model to use across training
model_save_location = os.path.join('models', model_name)
most_recent_model_save_location = os.path.join('models', f'{model_name}_most_recent')
model_profit_save_location = os.path.join(model_save_location, 'best_profit.txt')
if os.path.exists(model_save_location):
    model = load_a2c_model(model_save_location)
    print(f'Using existing model from {model_save_location}')
    try:
        with open(model_profit_save_location, 'r') as f:
            best_test_profit = float(f.readline())
            print(f'Best test profit loaded: {best_test_profit}')
    except Exception:
        best_test_profit = -float('inf')
else:
    print(f'Creating new model for {model_name}')
    model = A2C(training_envs[0], lr=0.0003).get_model()
    best_test_profit = -float('inf')

# Create optimizers for each training environment
optimizers = [A2C(env, network=model) for env in training_envs]

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
    test_profit = test_model(model, test_envs)

    # Save model if it's better than any others reached
    if test_profit > best_test_profit:
        print('New best profit reached, saving model...')
        model.save(model_save_location)
        print(f'Saved model to {model_save_location}.')

        best_test_profit = test_profit
        with open(model_profit_save_location, 'w+') as f:
            f.write(str(best_test_profit))

    # Save most recent verstion of model if it wasn't an improvement in case you want to use it
    else:
        model.save(most_recent_model_save_location)
