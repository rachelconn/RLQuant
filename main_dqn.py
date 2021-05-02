from itertools import count
import os
import random
import sys
import gym
import tensorflow as tf
from data_structures.replay_buffer import ReplayBuffer
from gym_environments.trading_environment import TradingEnvironment
from optimizers import DQN, load_dqn_model, create_lstm_network
from utils.generate_tickers import generate_training_test_environments
from utils.test_model import test_lstm_model

tf.get_logger().setLevel('WARNING')

# Param config
model_name = 'lstm_dqn'
num_training = 30
testing_stocks = ['IMAX', 'PCG', 'M', 'SHLDQ', 'ANGI', 'ADMS', 'WBA','TSLA', 'AAPL', 'WMT', 'MSFT']
lr = 0.0001
epsilon = 0.99
min_epsilon = 0.01
epsilon_decay = 0.96
gamma = 0.99
layer_nodes = [64, 128, 128]
time_steps = 8
replay_buffer_size = 1_000_000
batch_size = 100
batches_per_episode = 10
update_target_every = 5

replay_buffer = ReplayBuffer(replay_buffer_size)
batches_until_target_update = update_target_every

# Create training environments
print('Creating training/test environments...')
training_envs, _ = generate_training_test_environments('data/ticker_list/nyse-listed.csv', num_training, 0, seed=0)
test_envs = [TradingEnvironment(ticker) for ticker in testing_stocks]
print('Done setting up environments.')

# Create model to use across training
model_base_save_location = os.path.join('models', model_name)
recent_model_base_save_location = os.path.join('models', f'{model_name}_recent')
model_save_location = os.path.join(model_base_save_location, 'model')
target_model_save_location = os.path.join(model_base_save_location, 'target_model')
recent_model_save_location = os.path.join(recent_model_base_save_location, 'model')
recent_target_model_save_location = os.path.join(recent_model_base_save_location, 'target_model')
if os.path.exists(model_save_location):
    print(f'Loaded existing model from {model_base_save_location}')
    model, target_model = load_dqn_model(model_save_location, target_model_save_location)
else:
    print(f'Creating new model for {model_name}')
    model = create_lstm_network(training_envs[0], lr=lr, layer_nodes=layer_nodes, time_steps=time_steps)
    target_model = create_lstm_network(training_envs[0], lr=lr, layer_nodes=layer_nodes, time_steps=time_steps)

best_test_profit = -float('inf')

# Train until stopped manually
for i in count(1):
    # Create optimizers for each training environment
    optimizers = [DQN(env, model=model, target_model=target_model, epsilon=epsilon, gamma=gamma, time_steps=time_steps, replay_buffer=replay_buffer, update_target_every=update_target_every) for env in training_envs]

    # Generate episode for each optimizer, then run a batch update
    for j, optimizer in enumerate(optimizers):

        sys.stdout.write(f'\rGenerating episodes, current epsilon {epsilon}, iteration {i} ({j}/{len(optimizers)})...')
        sys.stdout.flush()
        optimizer.add_episode_to_replay_buffer()

        for _ in range(batches_per_episode):
            optimizer.train_minibatch(100)
            batches_until_target_update -= 1
            if batches_until_target_update == 0:
                optimizer.update_target_model()
                batches_until_target_update = update_target_every

        epsilon *= epsilon_decay
        epsilon = max(epsilon, min_epsilon)
    print()

    # Determine test performance
    print('Done training, testing...')
    test_profit = test_lstm_model(model, test_envs)

    # Save model if it's better than any others reached
    if test_profit > best_test_profit:
        best_test_profit = test_profit
        print('New best profit reached, saving model...')
        model.save(model_save_location)
        target_model.save(target_model_save_location)
        print(f'Saved model to {model_base_save_location}.')
    else:
        model.save(recent_model_save_location)
        target_model.save(recent_target_model_save_location)

    training_envs, _ = generate_training_test_environments('data/ticker_list/nyse-listed.csv', num_training, 0)
