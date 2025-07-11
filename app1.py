import streamlit as st
from newspaper import Article
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

# ---------------------------
# Load or simulate a fake news classifier
# ---------------------------
@st.cache_resource
def load_model():
    model = LogisticRegression()
    tfidf = TfidfVectorizer()
    sample_texts = [
        "The government has announced a new policy to improve the economy.",
        "Aliens have landed in Texas and taken over the White House.",
        "Scientists discover new cure for cancer using plant-based diets.",
        "COVID-19 vaccine includes microchips to track people."
    ]
    labels = [0, 1, 0, 1]  # 0 = Real, 1 = Fake
    X = tfidf.fit_transform(sample_texts)
    model.fit(X, labels)
    return model, tfidf

model, tfidf = load_model()

# ---------------------------
# News Text Extractor
# ---------------------------
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        st.error(f"Error extracting article: {e}")
        return None

# ---------------------------
# Fake News Predictor
# ---------------------------
def predict_news(text):
    features = tfidf.transform([text])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][pred]
    return pred, round(prob * 100, 2)

# ---------------------------
# Sentiment Analysis
# ---------------------------
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# ---------------------------
# Generate Word Cloud
# ---------------------------
def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Fake News Radar", layout="wide")
st.title("Fake News Radar")
st.markdown("Analyze news articles for authenticity, sentiment, and keyword distribution.")

input_method = st.radio("Choose Input Method", ["Paste Text", "Paste URL"])

user_input = ""
if input_method == "Paste Text":
    user_input = st.text_area("Paste a news article below:", height=250)
elif input_method == "Paste URL":
    url = st.text_input("Enter article URL:")
    if url:
        with st.spinner("Extracting text from URL..."):
            article_text = extract_text_from_url(url)
            if article_text:
                st.success("Article text extracted successfully.")
                user_input = article_text
                st.text_area("Extracted Article:", article_text, height=250)

if user_input:
    st.subheader("Analysis Results")
    with st.spinner("Analyzing..."):
        pred, confidence = predict_news(user_input)
        label = "Fake News" if pred == 1 else "Real News"
        sentiment = analyze_sentiment(user_input)

        st.markdown(f"**Verdict:** `{label}`")
        st.markdown(f"**Confidence:** `{confidence}%`")
        st.markdown(f"**Sentiment:** `{sentiment}`")

        with st.expander("Word Cloud"):
            generate_wordcloud(user_input)

        st.info("This is a demonstration model. For better accuracy, consider integrating a pre-trained NLP model.")
