from textblob import TextBlob

def get_sentiment():
    news = [
        "Company profits increased",
        "Strong financial performance"
    ]
    
    score = sum(TextBlob(n).sentiment.polarity for n in news) / len(news)
    return score