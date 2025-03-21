# sentiment_analysis.py

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from transformers import pipeline
from dotenv import load_dotenv

# Load env vars locally
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

# Load models
analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# News source mapping
source_country_map = {
    'BBC News': 'United Kingdom',
    'CNN': 'United States',
    'The Wall Street Journal': 'United States',
    'Al Jazeera English': 'Qatar',
    'Reuters': 'United Kingdom',
    'CNBC': 'United States',
    'The Guardian': 'United Kingdom',
    'Bloomberg': 'United States',
    'Fox News': 'United States',
    'Financial Times': 'United Kingdom',
}

def fetch_news():
    from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    url = f'https://newsapi.org/v2/everything?q=market&from={from_date}&sortBy=publishedAt&apiKey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    articles = data.get('articles', [])
    df = pd.DataFrame([{
        'source': a['source']['name'],
        'title': a['title'],
        'description': a['description'],
        'publishedAt': a['publishedAt'],
        'url': a['url']
    } for a in articles])
    return df

def analyze_sentiment(df):
    df['sentiment'] = df['description'].apply(lambda text: analyzer.polarity_scores(str(text))['compound'])
    def label(score):
        if score >= 0.6:
            return "🟢 Very Positive"
        elif score >= 0.2:
            return "🟢 Positive"
        elif score <= -0.6:
            return "🔴 Very Negative"
        elif score <= -0.2:
            return "🔴 Negative"
        else:
            return "🟡 Neutral"
    df['sentiment_label'] = df['sentiment'].apply(label)
    return df

def add_country_column(df):
    df['country'] = df['source'].map(source_country_map).fillna('Unknown')
    return df

def extract_named_entities(df):
    all_entities = []
    for text in df['title'].dropna():
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "GPE", "PERSON"]:
                all_entities.append(ent.text)
    entity_freq = pd.Series(all_entities).value_counts().head(30)
    return entity_freq

def classify_emotions(df):
    df['emotion'] = df['description'].fillna("").apply(lambda text: emotion_classifier(text)[0]["label"] if text else "Unknown")
    return df

def get_sentiment_data():
    df = fetch_news()
    df = analyze_sentiment(df)
    df = add_country_column(df)
    df['retrieved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return df
