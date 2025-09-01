import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("✈️ Airline Tweet Sentiment Predictor")
tweet = st.text_area("Enter a tweet about an airline:")

if st.button("Predict"):
    if tweet:
        X_input = vectorizer.transform([tweet])
        prediction = model.predict(X_input)[0]
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.success(f"Predicted Sentiment: **{label_map[prediction]}**")
