import streamlit as st
import pandas as pd
import plotly.express as px
from sentiment_analysis import get_sentiment_data, extract_named_entities

st.set_page_config(layout="wide")
st.title("📈 Real-Time Market Sentiment Dashboard")

st.markdown("""
This dashboard shows real-time news sentiment, emotion classification, and named entities extracted from headlines related to the financial market.
""")

df = get_sentiment_data()

if df.empty:
    st.error("No data available. Please check API keys or try again later.")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    sectors = df['country'].unique().tolist()
    emotions = df['emotion'].unique().tolist()
    sources = df['source'].unique().tolist()

    selected_sectors = st.multiselect("Select Countries", sectors, default=sectors)
    selected_emotions = st.multiselect("Select Emotions", emotions, default=emotions)
    selected_sources = st.multiselect("Select Sources", sources, default=sources)

# Filter data
filtered_df = df[
    df['country'].isin(selected_sectors) &
    df['emotion'].isin(selected_emotions) &
    df['source'].isin(selected_sources)
]

st.markdown(f"**Data last pulled:** {df['retrieved_at'].iloc[0]}")
st.dataframe(filtered_df[['publishedAt', 'title', 'source', 'sentiment_label', 'emotion']], use_container_width=True)

# Emotion bar chart
emotion_counts = filtered_df['emotion'].value_counts().reset_index()
emotion_counts.columns = ['emotion', 'count']
fig_emotion = px.bar(emotion_counts, x='emotion', y='count', labels={'emotion': 'Emotion', 'count': 'Count'}, title='Emotion Breakdown')
st.plotly_chart(fig_emotion, use_container_width=True)

# Named Entities
st.subheader("🧠 Top Named Entities")
entity_freq = extract_named_entities(filtered_df)
st.write(entity_freq)
