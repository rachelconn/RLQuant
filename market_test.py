from collections import namedtuple
import os
from typing import List
import numpy as np

from gym_environments.trading_environment import TradingState
from optimizers import load_a2c_model
from utils.generate_tickers import generate_training_test_environments

""" Tests an already trained model over a wide range of stocks """


model_name = 'trading_high_gamma'
ticker_list_location = 'data/ticker_list/nyse-listed.csv'

model_save_location = os.path.join('models', model_name)

# Load model
try:
    model = load_a2c_model(model_save_location)
except Exception as e:
    print(f'Error loading model: {str(e)}')
    exit(1)

Decision = namedtuple('Decision', ('ticker', 'action', 'confidence'))

class MarketTester:
    def __init__(self, *_, num_stocks, initial_money, seed):
        # Create environments
        self.stock_envs, _ = generate_training_test_environments(ticker_list_location, num_stocks, 0, seed=seed)
        self.nA = self.stock_envs[0].action_space.n
        # Use dict rather than list for envs for convenient access
        self.stock_envs = {env.get_ticker(): env for env in self.stock_envs}

        # State setup
        self.initial_money = initial_money
        self.money = initial_money
        self.env_states: Dict[TradingState] = {ticker: env.reset() for ticker, env in self.stock_envs.items()}
        self.stock_owned: Dict[int] = {ticker: 0 for ticker, env in self.stock_envs.items()}
        self.elapsed_days = 0

    def _print_state(self):
        money_in_stock = 0
        for ticker, owned in self.stock_owned.items():
            money_in_stock += owned * self.env_states[ticker].price

        held_stock = {ticker: owned for ticker, owned in self.stock_owned.items() if owned > 0}

        print(f'After {self.elapsed_days} day(s): ${self.money} held (${self.money +money_in_stock} including investments)')
        # print(f'  Stock held: {held_stock}')

    def _get_decisions(self) -> List[Decision]:
        decisions: List[Decision] = []
        finished = []
        for ticker, env in self.stock_envs.items():
            # Manually adjust money and stock owned so that decisions are made off of actual state rather than internal env state
            state_dict = {
                **self.env_states[ticker]._asdict(),
                'money': self.money,
                'stock_owned': self.stock_owned[ticker],
            }
            s = TradingState(**state_dict)

            # Determine action to take and state value
            action_probs = model(np.array([s]))[0][0].numpy()
            action = np.random.choice(self.nA, p=action_probs)
            # action = np.argmax(action_probs)
            confidence = action_probs[action]

            # Update environment and decision
            s_prime, _, done, _ = env.step(action)
            if done:
                # Ticker has no more data - sell immediately
                decisions.append(Decision(ticker=ticker, action=2, confidence=1))
                print(f'{ticker} finished after {self.elapsed_days} days')
                finished.append(ticker)
            else:
                decisions.append(Decision(ticker=ticker, action=action, confidence=confidence))
                self.env_states[ticker] = s_prime

        # If ticker has no more data, don't check it anymore
        for ticker in finished:
            del self.stock_envs[ticker]

        return decisions

    def _handle_decisions(self, decisions: List[Decision]):
        # Go through each decision: neutral + shorts should be done as-is, only most confident long positions should be made
        longs: List[Decision] = []
        shorts: List[str] = []
        for decision in decisions:
            # Long
            if decision.action == 0:
                longs.append(decision)
            # Short
            elif decision.action == 2:
                shorts.append(decision.ticker)
            # No need to handle neutral position - just hold the stock

        # Sell all shorts
        for ticker in shorts:
            self.money += self.stock_owned[ticker] * self.env_states[ticker].price
            self.stock_owned[ticker] = 0

        # Buy most promising longs (one at a time to diversify portfolio)
        longs = [decision.ticker for decision in sorted(longs, key=lambda x: x.confidence)]
        while longs:
            ticker = longs.pop()
            stock_price = self.env_states[ticker].price
            # Skip if the stock can't be bought
            if stock_price > self.money:
                continue

            self.money -= stock_price
            self.stock_owned[ticker] += 1

    def run(self):
        while True:
            self.elapsed_days += 1
            decisions = self._get_decisions()

            self._handle_decisions(decisions)
            self._print_state()

num_stocks = 200
initial_money = 10_000
seed = 123123123
MarketTester(num_stocks=num_stocks, initial_money=initial_money, seed=seed).run()
