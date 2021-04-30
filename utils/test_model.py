from collections import deque
from statistics import mean
import numpy as np

def test_model(model, test_envs):
    nA = test_envs[0].action_space.n
    profit_percentages = []
    # Run single iteration on each test env
    for env in test_envs:
        s = env.reset()
        done = False
        while not done:
            a = np.random.choice(nA, p=model(np.array([s]))[0][0].numpy())
            s, _, done, _ = env.step(a)

        # Done, get final profit percentage
        profit_percentages.append(env.render())

    # Calculate average profit percentage
    mean_profit_percentage = mean(profit_percentages)
    print(f'Average profit percentage on testing set: {mean_profit_percentage}%')
    return mean_profit_percentage

def test_lstm_model(model, test_envs):
    """ Tests the actor of an LSTM network """
    num_timesteps = model.input_shape[-1]
    nA = test_envs[0].action_space.n
    profit_percentages = []

    # Run single iteration on each test env
    for env in test_envs:
        steps = deque()
        for _ in range(num_timesteps):
            steps.append((0,) * 7)

        s = env.reset()
        done = False
        while not done:
            steps.append(s)
            steps.popleft()

            a = np.random.choice(nA, p=model(np.array([steps]))[0].numpy())
            s, _, done, _ = env.step(a)

        # Done, get final profit percentage
        profit_percentages.append(env.render())

    # Calculate average profit percentage
    mean_profit_percentage = mean(profit_percentages)
    print(f'Average profit percentage on testing set: {mean_profit_percentage}%')
    return mean_profit_percentage
