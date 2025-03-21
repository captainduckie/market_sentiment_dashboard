# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from news_sentiment import get_sentiment_data

st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")
st.title("🧠 Real-Time Market Sentiment Dashboard")

with st.spinner("Fetching latest news and analyzing sentiment..."):
    df = get_sentiment_data()

st.subheader("📰 Latest Articles")
st.dataframe(df[['publishedAt', 'title', 'sentiment']].sort_values(by='publishedAt', ascending=False))

st.subheader("📈 Sentiment Trend Over Time")
df['published_hour'] = df['publishedAt'].dt.floor('H')
trend = df.groupby('published_hour')['sentiment'].mean().reset_index()

fig, ax = plt.subplots()
ax.plot(trend['published_hour'], trend['sentiment'], marker='o')
ax.set_xlabel("Time")
ax.set_ylabel("Average Sentiment")
ax.set_title("Sentiment Trend")
st.pyplot(fig)
