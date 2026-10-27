import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    GRU,
    Dense,
    Dropout,
    Bidirectional
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import clean_text


DATA_PATH = "data/feedback.csv"
MODEL_DIR = "models"

MAX_WORDS = 5000
MAX_LEN = 40


os.makedirs(MODEL_DIR, exist_ok=True)


print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df["clean_text"] = df["text"].apply(clean_text)


label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label_id"] = df["label"].map(label_map)

df = df.dropna(subset=["label_id"])

print("\nDataset distribution:")
print(df["label"].value_counts())


X = df["clean_text"]
y = df["label_id"].astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================================
# TF-IDF + LOGISTIC REGRESSION
# ==========================================================

print("\n")
print("=" * 50)
print("TRAINING TF-IDF + LOGISTIC REGRESSION")
print("=" * 50)


tfidf_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


tfidf_model.fit(X_train, y_train)


tfidf_predictions = tfidf_model.predict(X_test)

tfidf_accuracy = accuracy_score(
    y_test,
    tfidf_predictions
)


print(
    "\nTF-IDF Accuracy:",
    round(tfidf_accuracy * 100, 2),
    "%"
)


print("\nTF-IDF Classification Report:")

print(
    classification_report(
        y_test,
        tfidf_predictions,
        labels=[0, 1, 2],
        target_names=[
            "negative",
            "neutral",
            "positive"
        ],
        zero_division=0
    )
)


import joblib

joblib.dump(
    tfidf_model,
    os.path.join(
        MODEL_DIR,
        "tfidf_sentiment.pkl"
    )
)


# ==========================================================
# TOKENIZATION
# ==========================================================

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)


X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)


X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)


y_train = np.array(y_train)
y_test = np.array(y_test)


vocab_size = min(
    MAX_WORDS,
    len(tokenizer.word_index) + 1
)


# ==========================================================
# LSTM MODEL
# ==========================================================

def build_lstm():

    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=128
        ),

        Bidirectional(
            LSTM(64)
        ),

        Dropout(0.5),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.3),

        Dense(
            3,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================================
# GRU MODEL
# ==========================================================

def build_gru():

    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=128
        ),

        Bidirectional(
            GRU(64)
        ),

        Dropout(0.5),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.3),

        Dense(
            3,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ==========================================================
# TRAIN LSTM
# ==========================================================

print("\n")
print("=" * 50)
print("TRAINING LSTM")
print("=" * 50)


lstm = build_lstm()


lstm.fit(
    X_train_pad,
    y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=8,
    callbacks=[early_stopping],
    verbose=1
)


lstm_loss, lstm_accuracy = lstm.evaluate(
    X_test_pad,
    y_test,
    verbose=0
)


lstm_predictions = np.argmax(
    lstm.predict(
        X_test_pad,
        verbose=0
    ),
    axis=1
)


print("\nLSTM Accuracy:")
print(
    round(lstm_accuracy * 100, 2),
    "%"
)


print("\nLSTM Classification Report:")

print(
    classification_report(
        y_test,
        lstm_predictions,
        labels=[0, 1, 2],
        target_names=[
            "negative",
            "neutral",
            "positive"
        ],
        zero_division=0
    )
)


lstm.save(
    os.path.join(
        MODEL_DIR,
        "lstm_sentiment.keras"
    )
)


# ==========================================================
# TRAIN GRU
# ==========================================================

print("\n")
print("=" * 50)
print("TRAINING GRU")
print("=" * 50)


gru = build_gru()


gru.fit(
    X_train_pad,
    y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=8,
    callbacks=[early_stopping],
    verbose=1
)


gru_loss, gru_accuracy = gru.evaluate(
    X_test_pad,
    y_test,
    verbose=0
)


gru_predictions = np.argmax(
    gru.predict(
        X_test_pad,
        verbose=0
    ),
    axis=1
)


print("\nGRU Accuracy:")
print(
    round(gru_accuracy * 100, 2),
    "%"
)


print("\nGRU Classification Report:")

print(
    classification_report(
        y_test,
        gru_predictions,
        labels=[0, 1, 2],
        target_names=[
            "negative",
            "neutral",
            "positive"
        ],
        zero_division=0
    )
)


gru.save(
    os.path.join(
        MODEL_DIR,
        "gru_sentiment.keras"
    )
)


# ==========================================================
# SAVE TOKENIZER
# ==========================================================

with open(
    os.path.join(
        MODEL_DIR,
        "tokenizer.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        tokenizer.to_json()
    )


# ==========================================================
# SAVE CONFIGURATION
# ==========================================================

results = {
    "tfidf_accuracy": float(tfidf_accuracy),
    "lstm_accuracy": float(lstm_accuracy),
    "gru_accuracy": float(gru_accuracy),
    "max_len": MAX_LEN,
    "max_words": MAX_WORDS,
    "classes": [
        "negative",
        "neutral",
        "positive"
    ]
}


with open(
    os.path.join(
        MODEL_DIR,
        "config.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ==========================================================
# FINAL COMPARISON
# ==========================================================

print("\n")
print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    f"TF-IDF + Logistic Regression : "
    f"{tfidf_accuracy * 100:.2f}%"
)

print(
    f"LSTM                        : "
    f"{lstm_accuracy * 100:.2f}%"
)

print(
    f"GRU                         : "
    f"{gru_accuracy * 100:.2f}%"
)


scores = {
    "TF-IDF": tfidf_accuracy,
    "LSTM": lstm_accuracy,
    "GRU": gru_accuracy
}


best_model = max(
    scores,
    key=scores.get
)


print("\nBest model based on test accuracy:")
print(best_model)

print("\nModels saved successfully.")

print("=" * 60)