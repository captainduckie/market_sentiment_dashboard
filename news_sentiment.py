# news_sentiment.py
import os
from dotenv import load_dotenv
import requests
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

API_KEY = '9ac5787362204ca0b18fe27cfd86de0e'
SEARCH_TERM = 'tech stocks'
FROM_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

def fetch_news():
    url = ('https://newsapi.org/v2/everything?q={}&from={}&sortBy=publishedAt&apiKey={}'
          .format(SEARCH_TERM, FROM_DATE, API_KEY))
    response = requests.get(url)
    data = response.json()
    articles = data.get('articles', [])
    
    df = pd.DataFrame([{
        'title': a['title'],
        'description': a['description'],
        'publishedAt': a['publishedAt'],
        'url': a['url']
    } for a in articles])
    return df

def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['description'].apply(lambda text: analyzer.polarity_scores(str(text))['compound'])
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    return df

def get_sentiment_data():
    df = fetch_news()
    df = analyze_sentiment(df)

    if os.getenv("STREAMLIT_CLOUD") != "1":
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/news_sentiment.csv", index=False)

    return df