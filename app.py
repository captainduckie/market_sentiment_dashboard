# app.py (updated polished dashboard with emotion classification, NER, VIX, sector overlays)

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import yfinance as yf
from sentiment_analysis import get_sentiment_data, extract_named_entities, classify_emotions

# Page config
st.set_page_config(page_title="🧠 Market Sentiment Dashboard", layout="wide")

# Custom style
st.markdown("""
    <style>
        .big-title {
            font-size:48px !important;
            font-weight:700;
            text-align:center;
        }
        .subtext {
            font-size:20px !important;
            text-align:center;
            color: gray;
            margin-bottom: 30px;
        }
        .stDataFrame th {
            font-size: 16px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🧠 Real-Time Market Sentiment</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Live sentiment and emotion analysis + market overlays</div>', unsafe_allow_html=True)

# Fetch & process data
with st.spinner("🔄 Fetching latest news and analyzing sentiment & emotions..."):
    df = get_sentiment_data()
    df = classify_emotions(df)

# Display articles
st.subheader("📰 Latest News")
st.dataframe(df[['publishedAt', 'title', 'sentiment', 'emotion', 'sentiment_label']], use_container_width=True)

# Sentiment Trend
st.subheader("📈 Sentiment Trend")
df['published_hour'] = pd.to_datetime(df['publishedAt']).dt.floor('H')
trend = df.groupby('published_hour')['sentiment'].mean().reset_index()
fig = px.line(trend, x='published_hour', y='sentiment', markers=True, title='Average Sentiment Over Time')
fig.update_layout(title_x=0.5)
st.plotly_chart(fig, use_container_width=True)

# Emotion Distribution
st.subheader("🌿 Emotion Distribution")
emotion_counts = df['emotion'].value_counts().reset_index()
fig_emotion = px.bar(emotion_counts, x='index', y='emotion', labels={'index': 'Emotion', 'emotion': 'Count'}, title='Emotion Breakdown')
fig_emotion.update_layout(title_x=0.5)
st.plotly_chart(fig_emotion, use_container_width=True)

# Word Cloud
st.subheader("☁️ Trending Terms")
text = " ".join(df['description'].dropna())
wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(text)
fig_wc, ax_wc = plt.subplots(figsize=(15, 6))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis("off")
st.pyplot(fig_wc)

# NER
st.subheader("🔍 Named Entities in Headlines")
entity_freq = extract_named_entities(df)
fig_ner = px.bar(x=entity_freq.values, y=entity_freq.index, orientation='h', title='Top Entities')
fig_ner.update_layout(title_x=0.5, height=600)
st.plotly_chart(fig_ner, use_container_width=True)

# VIX overlay
st.subheader("📊 Sentiment vs. Market Volatility (VIX)")
vix = yf.download('^VIX', period='7d', interval='1h')
vix = vix.reset_index()
vix['Datetime'] = pd.to_datetime(vix['Datetime']).dt.floor('H')

merged = pd.merge(trend, vix[['Datetime', 'Close']], left_on='published_hour', right_on='Datetime', how='inner')
fig_overlay = px.line(merged, x='published_hour', y=['sentiment', 'Close'], labels={'value': 'Metric', 'variable': 'Type'}, title='Sentiment vs. VIX')
fig_overlay.update_layout(title_x=0.5)
st.plotly_chart(fig_overlay, use_container_width=True)

# Sector ETF overlay
st.subheader("📊 Sector Performance Comparison")
sectors = {
    'Tech (XLK)': 'XLK',
    'Energy (XLE)': 'XLE',
    'Finance (XLF)': 'XLF'
}

sector_df = pd.DataFrame()
for name, ticker in sectors.items():
    data = yf.download(ticker, period='7d', interval='1h').reset_index()
    data['Datetime'] = pd.to_datetime(data['Datetime']).dt.floor('H')
    data['Sector'] = name
    sector_df = pd.concat([sector_df, data[['Datetime', 'Close', 'Sector']]])

fig_sector = px.line(sector_df, x='Datetime', y='Close', color='Sector', title='Sector ETF Prices Over Time')
fig_sector.update_layout(title_x=0.5)
st.plotly_chart(fig_sector, use_container_width=True)
