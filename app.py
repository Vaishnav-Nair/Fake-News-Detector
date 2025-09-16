# app.py
import streamlit as st
import joblib
import os
import numpy as np

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("<h1 style='text-align:center'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.write("Paste an article or headline and the model will predict Real vs Fake (confidence shown).")

MODEL_PATH = "fake_news_detector.pkl"
VECT_PATH = "vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    st.error("Model or vectorizer file missing. Make sure fake_news_detector.pkl and vectorizer.pkl are in the app folder.")
    st.stop()

with st.spinner("Loading model and vectorizer..."):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)

st.sidebar.title("About")
st.sidebar.info(
    """
    Built with Python, scikit-learn and Streamlit.
    Author: Vaishnav Vinod Nair
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Try example headlines:")
examples = [
    "Breaking: Major earthquake hits city, authorities respond",
    "Celebrity adopts 100 stray dogs after viral post",
    "Study finds coffee reduces risk of some cancers",
]
choice = st.sidebar.selectbox("Select example", ["-- none --"] + examples)

user_input = st.text_area("Paste news article or headline here:", value=(choice if choice != "-- none --" else ""), height=220)

col1, col2 = st.columns([3,1])
with col1:
    if st.button("Predict"):
        if not user_input or user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            text_vec = vectorizer.transform([user_input])
            pred = model.predict(text_vec)[0]
            probs = model.predict_proba(text_vec)[0]  # [prob_class0, prob_class1]
            # ASSUMPTION: label mapping used during training is: 0 = Fake, 1 = Real
            if pred == 1:
                st.success(f"✅ REAL  — Confidence: {probs[1]*100:.2f}%")
            else:
                st.error(f"❌ FAKE  — Confidence: {probs[0]*100:.2f}%")
            st.write("---")
            st.write("**Model probabilities:**")
            st.write({ "Fake": f"{probs[0]*100:.2f}%", "Real": f"{probs[1]*100:.2f}%" })

with col2:
    st.write("Tips:")
    st.write("- Use complete text or a clear headline.")
    st.write("- If you get one-class predictions, re-check vectorizer/model versions and label mapping.")
    st.write("")
st.markdown("---")
st.write("If you like this app, add it to your portfolio. Contact: Vaishnav Vinod Nair")
