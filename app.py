# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# -------------------------------
# Load your trained model & tokenizer
# -------------------------------
@st.cache_resource
def load_trained_model():
    model = load_model("next_word_model.h5")  # path to your saved model
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_trained_model()

# -------------------------------
# Next-word prediction function
# -------------------------------
def predict_next_word(model, tokenizer, text):
    """
    Predicts the next word for a given text using the trained LSTM model.
    """
    max_sequence_len = model.input_shape[1]  # expected input length

    # Convert text to sequence of integers
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return "No known words in input."

    # Use last max_sequence_len tokens if input is long
    token_list = token_list[-max_sequence_len:]

    # Pad sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')

    # Predict next word
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]

    # Map index back to word
    predicted_word = tokenizer.index_word.get(predicted_index, "Unknown")
    return predicted_word

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Next-Word Predictor 📝")
st.write("Type a sentence and the model will predict the next word.")

# User input
user_input = st.text_input("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        next_word = predict_next_word(model, tokenizer, user_input.lower())
        st.success(f"Predicted next word: **{next_word}**")
