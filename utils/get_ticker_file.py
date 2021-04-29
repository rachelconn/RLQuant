import os

def get_ticker_file(ticker):
    outfile = os.path.join(os.getcwd(), f'data/tickers/{ticker}.csv')
    if not os.path.exists(os.path.dirname(outfile)):
        os.mkdir(os.path.dirname(outfile))
    if not os.path.exists(outfile):
        yahoodownload_path = os.path.join(os.path.dirname(__file__), 'yahoodownload.py')
        os.system(f'python {yahoodownload_path} --ticker {ticker} --fromdate 2017-1-1 --todate 2021-4-27 --outfile {outfile}')
    return outfile
