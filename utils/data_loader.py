import yfinance as yf

def load_data(stock):
    data = yf.download(stock, period="6mo")
    return data