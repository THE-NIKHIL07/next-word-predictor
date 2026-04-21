# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle


@st.cache_resource
def load_trained_model():
    model = load_model("next_word_model.h5")  
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_trained_model()

def predict_next_word(model, tokenizer, text):
    """
    Predicts the next word for a given text using the trained LSTM model.
    """
    max_sequence_len = model.input_shape[1]  
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return "No known words in input."

    token_list = token_list[-max_sequence_len:]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')

 
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]

   
    predicted_word = tokenizer.index_word.get(predicted_index, "Unknown")
    return predicted_word


st.title("Next-Word Predictor 📝")
st.write("Type a sentence and the model will predict the next word.")

user_input = st.text_input("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        next_word = predict_next_word(model, tokenizer, user_input.lower())
        st.success(f"Predicted next word: **{next_word}**")
