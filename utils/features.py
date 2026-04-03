def add_features(data):
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['Return'] = data['Close'].pct_change()
    data['Target'] = (data['Return'] > 0).astype(int)
    return data.dropna()