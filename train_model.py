import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# DOWNLOAD NLTK DATA
# ============================================================

nltk.download('punkt')
nltk.download('stopwords')

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "phishing.csv",
    encoding="latin-1",
    on_bad_lines="skip"
)

# ============================================================
# CHECK COLUMNS
# ============================================================

print("\nCOLUMNS:\n")

print(df.columns)

# ============================================================
# SELECT REQUIRED COLUMNS
# ============================================================

df = df[['Email Text', 'Email Type']]

df.columns = ['text', 'label']

# ============================================================
# CLEAN DATA
# ============================================================

df.dropna(inplace=True)

df['text'] = df['text'].astype(str)

df = df[
    df['text'].str.len() > 10
]

# ============================================================
# LABEL ENCODING
# ============================================================

df['label'] = df['label'].str.strip()

print(df['label'].unique())

df['label'] = df['label'].map({
    'Phishing Email': 1,
    'Safe Email': 0
})

# Remove rows with missing labels
df.dropna(inplace=True)

# ============================================================
# TEXT PREPROCESSING
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
# APPLY PREPROCESSING
# ============================================================

df['cleaned_text'] = df['text'].apply(preprocess_text)

# ============================================================
# TF-IDF VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=5000
)

X = vectorizer.fit_transform(
    df['cleaned_text']
)

y = df['label']

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nAccuracy: {accuracy:.2f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ============================================================
# SAVE MODEL + VECTORIZER
# ============================================================

joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)

print("\nMODEL TRAINED SUCCESSFULLY!")