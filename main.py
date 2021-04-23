import gym
from gym_environments.trading_environment import TradingEnvironment
from optimizers import A2C

# Simple demo that environment works - add your algo here to train
trading_env = TradingEnvironment('AAPL')
s = trading_env.reset()
print(f'Initial state: {s}')

for x in range(10):
    s, r, done, _ = trading_env.step(0)
    print(f'New state: s:{s},r:{r}')

for x in range(10):
    s, r, done, _ = trading_env.step(1)
    print(f'New state: s:{s},r:{r}')


# Test A2C implementation on CartPole
cartpole_env = gym.make('CartPole-v0')
cartpole_env._max_episode_steps = 2000
a2c = A2C(cartpole_env)
a2c.train(2000, render_every=200)

# Save model
model_save_location = 'models/cartpole'
model = a2c.get_model().save(model_save_location)
print(f'Saved model to {model_save_location}')
