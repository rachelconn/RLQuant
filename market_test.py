import datetime
from matplotlib import pyplot
from collections import deque, namedtuple
import os
from typing import List
import numpy as np

from gym_environments.trading_environment import TradingState
from optimizers import load_a2c_model, load_a2c_lstm_model, load_dqn_model
from utils.generate_tickers import generate_training_test_environments

""" Tests an already trained model over a wide range of stocks """


model_name = 'lstm_dqn_recent'
model_is_lstm = True
model_is_dqn = True
num_stocks = 10
initial_money = 10_000
seed = 123123123
ticker_list_location = 'data/ticker_list/nyse-listed.csv'

model_save_location = os.path.join('models', model_name)

# Load model
try:
    if model_is_lstm:
        if model_is_dqn:
            model_save_location = os.path.join(model_save_location, 'model')
            model, _ = load_dqn_model(model_save_location, model_save_location)
        else:
            actor_save_location = os.path.join(model_save_location, 'actor')
            critic_save_location = os.path.join(model_save_location, 'critic')
            model, _ = load_a2c_lstm_model(actor_save_location, critic_save_location)
    else:
        model = load_a2c_model(model_save_location)
except Exception as e:
    print(f'Error loading model: {str(e)}')
    exit(1)

Decision = namedtuple('Decision', ('ticker', 'action', 'confidence'))

class MarketTester:
    def __init__(self, *_, num_stocks, initial_money, seed, model_is_lstm, model_is_dqn):
        # Create environments
        self.stock_envs, _ = generate_training_test_environments(ticker_list_location, num_stocks, 0, seed=seed)
        self.nA = self.stock_envs[0].action_space.n
        self.model_is_lstm = model_is_lstm
        self.model_is_dqn = model_is_dqn
        # Use dict rather than list for envs for convenient access
        self.stock_envs = {env.get_ticker(): env for env in self.stock_envs}

        # State setup
        self.initial_money = initial_money
        self.money = initial_money
        if model_is_lstm:
            self.num_timesteps = model.input_shape[-2]
            self.state_size = model.input_shape[-1]
            def get_steps(env):
                steps = deque((0,) * self.state_size for _ in range(self.num_timesteps - 1))
                steps.append(env.reset())
                return steps
            self.env_states: Dict[TradingState] = {ticker: get_steps(env) for ticker, env in self.stock_envs.items()}
        else:
            self.env_states: Dict[TradingState] = {ticker: env.reset() for ticker, env in self.stock_envs.items()}

        self.stock_owned: Dict[int] = {ticker: 0 for ticker, env in self.stock_envs.items()}
        self.elapsed_days = 0

    def _print_state(self):
        money_in_stock = 0
        for ticker, owned in self.stock_owned.items():
            if self.model_is_lstm:
                money_in_stock += owned * self.env_states[ticker][-1].price
            else:
                money_in_stock += owned * self.env_states[ticker].price

        portfolio_value = self.money + money_in_stock
        print(f'After {self.elapsed_days} day(s): ${self.money} held (${self.money + money_in_stock} including investments)')
        return portfolio_value

    def _get_decisions(self) -> List[Decision]:
        decisions: List[Decision] = []
        finished = []
        for ticker, env in self.stock_envs.items():
            s = self.env_states[ticker]

            # Determine action to take and state value
            prediction = model(np.array([s]))[0]
            if not self.model_is_dqn:
                if self.model_is_lstm:
                    action_probs = prediction.numpy()
                else:
                    action_probs = prediction[0].numpy()
                action = np.random.choice(self.nA, p=action_probs)
                # action = np.argmax(action_probs)
                confidence = action_probs[action]
            else:
                # prediction = prediction.numpy()
                # action = np.argmax(prediction)
                # # Rank decisions by advantage over neutral position (normalized to account differences in price)
                # confidence = (prediction[action] - prediction[1]) / self.env_states[ticker][-1].price
                action = np.random.choice(self.nA - 1)
                confidence = 1

            # Update environment and decision
            s_prime, _, done, _ = env.step(action)
            if done:
                # Ticker has no more data - sell immediately
                decisions.append(Decision(ticker=ticker, action=2, confidence=1))
                print(f'{ticker} finished after {self.elapsed_days} days')
                finished.append(ticker)
            else:
                decisions.append(Decision(ticker=ticker, action=action, confidence=confidence))
                # Manually adjust money and stock owned so that decisions are made off of actual state rather than internal env state
                s_prime = TradingState(**{
                    **s_prime._asdict(),
                    'money': self.money,
                    'stock_owned': self.stock_owned[ticker],
                })
                if self.model_is_lstm:
                    self.env_states[ticker].append(s_prime)
                    self.env_states[ticker].popleft()
                else:
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
            # print(f'Selling {self.stock_owned[ticker]}x {ticker}')
            if self.model_is_lstm:
                stock_price = self.env_states[ticker][-1].price
            else:
                stock_price = self.env_states[ticker].price
            self.money += self.stock_owned[ticker] * stock_price
            self.stock_owned[ticker] = 0

        # Buy most promising longs (one at a time to diversify portfolio)
        longs = [decision.ticker for decision in sorted(longs, key=lambda x: x.confidence)]
        while longs:
            ticker = longs.pop()
            if self.model_is_lstm:
                stock_price = self.env_states[ticker][-1].price
            else:
                stock_price = self.env_states[ticker].price
            # Skip if the stock can't be bought
            if stock_price > self.money:
                continue

            self.money -= stock_price
            self.stock_owned[ticker] += 1

    def run(self):
        current_date = datetime.datetime(2018, 1, 1)
        portfolio_value_history = [self.initial_money]
        dates = [current_date]

        while self.stock_envs:
            self.elapsed_days += 1
            current_date = current_date + datetime.timedelta(days=1)
            decisions = self._get_decisions()

            self._handle_decisions(decisions)
            portfolio_value = self._print_state()
            portfolio_value_history.append(portfolio_value)
            dates.append(current_date)

        # Plot value over time
        pyplot.title('Portfolio value over time')
        pyplot.plot_date(dates, portfolio_value_history, 'bo-', markersize=2)
        pyplot.show()


MarketTester(num_stocks=num_stocks, initial_money=initial_money, seed=seed, model_is_lstm=model_is_lstm, model_is_dqn=model_is_dqn).run()
