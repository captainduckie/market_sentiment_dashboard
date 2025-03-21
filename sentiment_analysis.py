# sentiment_analysis.py

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from transformers import pipeline
from dotenv import load_dotenv
import streamlit as st
import feedparser

# Load env vars locally
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

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
    'Reddit': 'Global',
    'Google News': 'Global'
}

# ... (fetching functions omitted for brevity) ...

def analyze_sentiment(df):
    df['sentiment'] = df['description'].fillna("").apply(lambda text: analyzer.polarity_scores(str(text))['compound'])
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

def extract_emotion_label(text):
    try:
        result = emotion_classifier(text)
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("label", "Unknown")
        elif isinstance(result, dict):
            return result.get("label", "Unknown")
        else:
            return "Unknown"
    except:
        return "Unknown"

def classify_emotions(df):
    df['emotion'] = df['description'].fillna("").apply(lambda text: extract_emotion_label(text))
    return df

def get_sentiment_data():
    df = fetch_news()
    provider = st.session_state.get("news_provider", "Unknown")
    if provider.startswith("Google") or provider.startswith("Reddit"):
        st.warning("⚠️ You're currently using demo sources (Google News RSS or Reddit RSS). Results may be less relevant.")
    if df.empty:
        print("⚠️ No news data returned. Check all API keys and services.")
        return pd.DataFrame(columns=[
            'source', 'title', 'description', 'publishedAt', 'url',
            'sentiment', 'sentiment_label', 'emotion', 'country', 'retrieved_at'
        ])
    df = analyze_sentiment(df)
    df = add_country_column(df)
    df['retrieved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return df
