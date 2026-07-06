# RNN Practical (Many to One)

# Data: spam.csv

import os
import re
import pickle
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Configuration

MODEL = "spam_model.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000
MAX_LEN = 50


# Clean Text Function

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Train Model

def train_model():
    print("Training dataset...")

    df = pd.read_csv("spam.csv", encoding="latin-1")

    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    print(df.head())
    print(df["label"].value_counts())

    # Convert labels into numbers
    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Clean SMS
    df["text"] = df["text"].apply(clean_text)

    # Tokenizer
    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(df["text"])

    sequences = tokenizer.texts_to_sequences(df["text"])

    X = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    y = df["label"]

    print("X Shape:", X.shape)
    print("y Shape:", y.shape)

    # Save Tokenizer
    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Build RNN Model
    model = Sequential()

    # Embedding Layer
    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=32
        )
    )

    # Simple RNN Layer
    model.add(SimpleRNN(128))

    # Hidden Layer
    model.add(Dense(32, activation="relu"))

    # Output Layer
    model.add(Dense(1, activation="sigmoid"))

    # Compile Model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Train Model
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2
    )

    # Save Model
    model.save(MODEL)

    # Evaluate Model
    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print("Test Loss:", loss)
    print("Accuracy:", accuracy)

    # Predictions
    probabilities = model.predict(
        x_test,
        verbose=0
    )

    predictions = (
        probabilities > 0.5
    ).astype(int).flatten()

    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


# Predict SMS

def predict_sms(message):
    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    message = clean_text(message)

    sequence = tokenizer.texts_to_sequences(
        [message]
    )

    sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probability = model.predict(
        sequence,
        verbose=0
    )[0][0]

    if probability > 0.5:
        return "Spam", probability

    return "Ham", 1 - probability


# Train Model If Model Does Not Exist

if not os.path.exists(MODEL):
    train_model()


# Streamlit App

st.title("Spam Detection using RNN")

st.write(
    "This is a simple spam detection application "
    "using Recurrent Neural Networks (RNN)."
)

message = st.text_area(
    "Enter your message here:"
)

st.write(
    "Click the button below to check "
    "if the message is spam or not."
)

if st.button("Predict"):
    if message.strip():
        prediction, probability = predict_sms(message)

        st.success(prediction)

        st.write(
            "Confidence Score: {:.2f}%".format(
                probability * 100
            )
        )

    else:
        st.warning("Please enter a message.")