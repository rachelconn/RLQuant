# About
Reinforcement learning-based algorithmic trader made for TAMU's reinforcement learning course.

# Installation
Run the following to setup dependencies, preferably using a virtualenv to avoid dependency issues:
```bash
pip install -r requirements.txt
```
This project should be compatible with Python 3.6-3.9: I recommend 3.8.5 since it was used in development.

# Training and testing
Training can be done by running `python main.py`: parameters can be configured at the top of the file,
or left as-is to use the same parameters as the paper.
A pretrained model is also included in `pretrained_models/best` if you don't want to train one yourself.

Once you have a trained model, there are two ways to test it:
## Performance on single stocks
Run `python market_test.py -t {TICKER} -m {MODEL SAVE LOCATION}`,
where `{TICKER}` is the stock ticker you want to test (for example, `AAPL`, `PCG`, or `WMT`),
and `{MODEL SAVE LOCATION}` is the relative path to the model you want to use:
for the pretrained model, this will be `pretrained_models/best`.

## Performance on multiple stocks
Run `python market_test.py -m {MODEL SAVE LOCATION}` to test the model's performance on a set of 300
random stocks, or `python market_test.py -m {MODEL SAVE LOCATION} -d` to test on stocks from the Dow
Jones Industrial Average.
