# Fake News Radar

This is a Streamlit application designed to analyze news articles for authenticity, sentiment, and keyword relevance. It uses a demo machine learning model and simple NLP tools to provide quick insights.

## Features

- Fake news detection using TF-IDF and Logistic Regression
- Support for raw text or URL input
- Sentiment analysis with TextBlob
- Keyword visualization via word cloud
- Interactive Streamlit interface

## Project Structure

fake-news-radar/ ├── app.py # Main Streamlit script ├── requirements.txt # Python dependencies └── README.md 
# Project overview

## How It Works

1. A demo classifier is trained on sample articles using TF-IDF and Logistic Regression.
2. Users can paste raw text or enter a news article URL.
3. The app:
   - Extracts and analyzes text
   - Predicts authenticity
   - Evaluates sentiment
   - Generates a word cloud

## Installation

Make sure you have Python 3.8 or above. Then install required libraries:

```bash
pip install -r requirements.txt
