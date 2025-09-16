📰 Fake News Detector

A machine learning web app that predicts whether a news article is Real or Fake. Built with Python, Scikit-learn, and Streamlit.

🚀 Features

Predicts whether a news article is Fake or Real.

Built using TF-IDF Vectorization + Machine Learning models (Logistic Regression, Naive Bayes, Random Forest).

Interactive web app deployed with Streamlit.

Clean and simple UI with real-time predictions.

📂 Project Structure
Fake-News-Detector/
app.py                 
fake_news_detector.pkl 
vectorizer.pkl         
requirements.txt       
README.md

⚙️ Installation

Clone this repo and install dependencies:

git clone https://github.com/Vaishnav-Nair/Fake-News-Detector
cd Fake-News-Detector
pip install -r requirements.txt

▶️ Run Locally
streamlit run app.py

🌐 Deployment

This app is deployed using Streamlit Cloud.
👉 Click here to try it out: https://fake-news-detector-bx9svhb2t7omtueavrf2gr.streamlit.app/

📊 Models Used

Logistic Regression → Best balance of speed and accuracy.

Naive Bayes → Works well for text classification.

Final model selected: ✅ Logistic Regression

📖 Dataset

Dataset used: Fake and Real News Dataset (Kaggle)

True.csv → Real news articles

Fake.csv → Fake news articles

Kaggle Link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

✨ Future Improvements

Improve UI design with better layout.

Allow users to paste a URL and auto-fetch the news text.

Deploy with Docker / Heroku / AWS for scalability.

👨‍💻 Author

Vaishnav Vinod Nair
