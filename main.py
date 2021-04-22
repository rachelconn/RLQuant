from gym_environments.trading_environment import TradingEnvironment

# Simple demo that environment works - add your algo here to train
env = TradingEnvironment('AAPL')
s = env.reset()
print(f'Initial state: {s}')

done = False

"""
while not done:
    s, r, done, _ = env.step(0)
    print(f'New state: {s}')
"""

for x in range(10):
    s, r, done, _ = env.step(0)
    print(f'New state: s:{s},r:{r}')

for x in range(10):
    s, r, done, _ = env.step(1)
    print(f'New state: s:{s},r:{r}')
