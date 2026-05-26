import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Phishing Email Detection",
    page_icon="🛡️",
    layout="centered"
)

# ============================================================
# DOWNLOAD NLTK DATA
# ============================================================

nltk.download('punkt')
nltk.download('stopwords')

# ============================================================
# LOAD MODEL + VECTORIZER
# ============================================================

model = joblib.load("model.pkl")

vectorizer = joblib.load("vectorizer.pkl")

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("About")

st.sidebar.info(
    """
    This application uses Machine Learning and NLP
    techniques to detect phishing or spam messages.

    Model Used:
    • TF-IDF Vectorization
    • Logistic Regression

    Developed using:
    • Python
    • Scikit-learn
    • Streamlit
    """
)

# ============================================================
# MAIN TITLE
# ============================================================

st.title("🛡️ Phishing Email Detection System")

st.write(
    "Enter a message below to check whether it is phishing or legitimate."
)

# ============================================================
# USER INPUT
# ============================================================

email_text = st.text_area(
    "Enter Message",
    height=200
)

# ============================================================
# DETECTION
# ============================================================

if st.button("Detect"):

    if email_text.strip() == "":
        st.warning("Please enter a message.")

    else:

        # Preprocess
        cleaned_text = preprocess_text(email_text)

        # Vectorize
        vectorized_text = vectorizer.transform([cleaned_text])

        # Probability
        probability = model.predict_proba(vectorized_text)[0][1]

        # Display probability
        st.write(f"### Phishing Probability: {probability:.2f}")

        # Threshold
        if probability > 0.35:

            st.error("⚠️ Phishing Message Detected!")

        else:

            st.success("✅ Legitimate Message")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption("Cybersecurity + Machine Learning Project")