from collections import namedtuple
from datetime import datetime
import queue
import backtrader as bt
import backtrader.indicators as btind
import gym
from gym import spaces
import numpy as np
import pandas as pd
from utils.get_ticker_file import get_ticker_file

TickerValue = namedtuple('TickerValue', ('price', 'MA', 'EMA', 'MACD', 'RSI'))
TradingState = namedtuple('TradingState', ('money', 'stock_owned') + TickerValue._fields)

default_fromdate = datetime(2018, 1, 1)
default_todate = datetime(2021, 4, 1)

cached = {}

def check_time_valid(dataname, fromdate):
    """ Makes sure that ticker data at location dataname starts on or before fromdate """
    data = pd.read_csv(dataname, usecols=['Date'], nrows=1)
    assert datetime.strptime(data['Date'][0], '%Y-%m-%d') <= fromdate, f'Data in {dataname} starts after {fromdate}'

def create_trajectory(ticker, fromdate=default_fromdate, todate=default_todate):
    global cached
    if cached.get(ticker) is not None:
        return cached[ticker]
    """ Returns a TickerValue at the end of each day between start_date and end_date """
    ticker_values = []

    # Use backtrader to create ticker values for each day in range
    class TrackValues(bt.Strategy):
        def __init__(self):
            self.MA = btind.SMA(self.data)
            self.EMA = btind.EMA(self.data)
            self.MACD = btind.EMA(self.data, period=12) - btind.EMA(self.data, period=26)
            self.RSI = btind.RSI_EMA(safediv=True)

        def next(self):
            # Data available, put indicators into ticker values
            ticker_value = TickerValue(
                price=self.data.close[0],
                MA=self.MA[0],
                EMA=self.EMA[0],
                MACD=self.MACD[0],
                RSI=self.RSI[0],
            )
            ticker_values.append(ticker_value)

    # Create feed with ticker data and get values in the desired timeframe
    cerebro = bt.Cerebro()
    dataname = get_ticker_file(ticker)

    # Throw an error if the start date is too early
    check_time_valid(dataname, fromdate)

    ticker_data = bt.feeds.YahooFinanceCSVData(dataname=dataname, fromdate=fromdate, todate=todate)
    cerebro.adddata(ticker_data)
    cerebro.addstrategy(TrackValues)
    cerebro.run()

    cached[ticker] = ticker_values
    return ticker_values

class TradingEnvironment(gym.Env):
    """ gym environment that trades a single stock """
    def __init__(self, ticker, transaction_cost = 0, initial_money=10_000):
        self.ticker = ticker
        self.initial_money = initial_money
        self.transaction_cost = transaction_cost

        # State space: values defined below
        low = np.array([
            0,       # Money on hand
            0,       # Stock owned
            0,       # Ticker price
            0,       # MA (SimpleMovingAverage)
            0,       # EMA (ExponentialMovingAverage)
            np.NINF, # MACD (MovingAverageConvergenceDivergence)
            0,       # RSI (RelativeStrengthIndex)
        ], dtype=np.float32)
        high = np.array([
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            100,
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Actions: { long, neutral, short }
        # NOTE:
        #########
        #long position just means you think it will go up,
        #actually it does not have any specific meaning.
        #even if taking the long position, we will still need
        #to specify how much equity we will purchase
        #p.s. it is true we can allow "shorting the stock"
        #but still we need to know how much to buy...
        #########

        """self.action_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Box(np.NINF, np.inf, (1,))
        ))"""
        self.action_space = spaces.Discrete(3)
        self.reset()

    def _get_next_transition(self, action=None):
        """ Returns state given a position in the stock trajectory.
            Optionally takes an action as a param to provide reward (assuming not initial state)
        """
        ticker_value = self.trajectory[self.position_in_trajectory]
        done = self.position_in_trajectory == len(self.trajectory) - 1
        self.position_in_trajectory += 1

        # Handle action
        action_invalid = False
        if action is not None:
            # recall Actions: { long, neutral, short }
            # long
            if action == 0:
                if self.money >= ticker_value.price:
                    self.money -= ticker_value.price
                    self.stock_owned += 1
                else:
                    action_invalid = True
            # neutral
            elif action == 1:
                pass
            # short
            elif action == 2:
                if self.stock_owned > 0:
                    self.money += ticker_value.price
                    self.stock_owned -= 1
                else:
                    action_invalid = True

        unadjusted_reward = ticker_value.price * self.stock_owned + self.money
        reward = unadjusted_reward - self.last_reward
        self.last_reward = unadjusted_reward
        # Give large negative reward if action is invalid
        if action_invalid:
            reward = -100 * self.initial_money

        self.last_ticker_price = ticker_value.price

        # print(f'Asset value: {unadjusted_reward}, money: {self.money}, stock owned: {self.stock_owned}, reward: {reward} from taking action {action}')
        state = TradingState(money=self.money, stock_owned=self.stock_owned, **ticker_value._asdict())

        return state, reward, done, {}

    def reset(self):
        """ Resets the enviroment, note that environments should not reused as trends are deterministic.
            Instead create another environment using a different ticker/time frame.
        """
        # State independent from stock
        self.money = self.initial_money
        self.last_reward = self.initial_money
        self.stock_owned = 0

        # State related to stock
        self.trajectory = create_trajectory(self.ticker)
        assert len(self.trajectory) > 1, 'Cannot create trajectory with a single data point'
        self.position_in_trajectory = 0

        state, *_ = self._get_next_transition()
        return state

    def step(self, action):
        return self._get_next_transition(action)

    # Stub methods to make compatible with the gym interface
    def render(self, to_console=True):
        profit = self.stock_owned * self.last_ticker_price + self.money - self.initial_money
        profit_percentage = profit / self.initial_money * 100
        if to_console:
            print(f'Profit from {self.ticker}: {profit} ({profit_percentage}%)')
        return profit_percentage

    def close(self):
        pass

    def get_current_price(self):
        """ Returns the current price of the environment's stock. """
        return self.last_ticker_price

    def get_ticker(self):
        return self.ticker
