from collections import namedtuple
from datetime import datetime
import queue
import backtrader as bt
import backtrader.indicators as btind
import gym
from gym import spaces
import numpy as np

TickerValue = namedtuple('TickerValue', ('price', 'MA', 'EMA', 'MACD', 'RSI'))
TradingState = namedtuple('TradingState', ('money', 'stock_owned') + TickerValue._fields)

default_fromdate = datetime(2018, 1, 1)
default_todate = datetime(2021, 4, 1)

def create_trajectory(ticker, fromdate=default_fromdate, todate=default_todate):
    """ Returns a TickerValue at the end of each day between start_date and end_date """
    ticker_values = []

    # Use backtrader to create ticker values for each day in range
    class TrackValues(bt.Strategy):
        def __init__(self):
            self.MA = btind.SMA(self.data)
            self.EMA = btind.EMA(self.data)
            self.MACD = btind.EMA(self.data, period=12) - btind.EMA(self.data, period=26)
            self.RSI = btind.RSI_EMA()

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
    ticker_data = bt.feeds.YahooFinanceData(dataname=ticker, fromdate=fromdate, todate=todate)
    cerebro.adddata(ticker_data)
    cerebro.addstrategy(TrackValues)
    cerebro.run()

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

        state = TradingState(money=self.money, stock_owned=self.stock_owned, **ticker_value._asdict())

        # TODO: handle action
        # self.action_space = spaces.Discrete(3)

        # NOTE: 
        # This works but it really don't consider the long term holding

        # TODO: calculate reward
        reward = 0
        if action != None:
            # recall Actions: { long, neutral, short }
            # long: change in stock held * (closing price - opening price) + transaction cost
            if action == 0:
                self.stock_owned += 1
                reward = (state[2] - self.last_ticker_price) + self.transaction_cost
            # neutral: num held * closing price + held capital - inital capital
            # QUESTION:
            # wouldn't this cost bot to helf indefinitely? guess not..
            if action == 1:
                reward = self.stock_owned * state[2] + self.money - self.initial_money
            # short: change in stock held * (opening price - closing price) + transaction cost
            if action == 2:
                self.stock_owned -= 1
                reward = (self.last_ticker_price - state[2]) + self.transaction_cost

        self.last_ticker_price = state[2]


        return state, reward, done, {}

    def reset(self):
        """ Resets the enviroment, note that environments should not reused as trends are deterministic.
            Instead create another environment using a different ticker/time frame.
        """
        # State independent from stock
        self.money = self.initial_money
        self.stock_owned = 0

        # State related to stock
        self.trajectory = create_trajectory(self.ticker)
        assert len(self.trajectory) > 1, 'Cannot create trajectory with a single data point'
        self.position_in_trajectory = 0

        state, *_ = self._get_next_transition()
        self.last_ticker_price = state[0] #aka openign price
        return state

    def step(self, action):
        return self._get_next_transition(action)
