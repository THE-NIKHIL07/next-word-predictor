# Next-Word Predictor 📝

A simple **Next-Word Prediction** web app using a trained **LSTM model** in Keras/TensorFlow.  
Users can type an English sentence, and the model predicts the most likely next word.

---

## Features

- Predicts the next word for any English input.  
- Uses a **word-level LSTM model** trained on English text.  
- Simple and fast **argmax-based prediction** (no fancy sampling).  
- Interactive web interface using **Streamlit**.  

---

## Files in the Repository

- `app.py` – Streamlit app for next-word prediction.  
- `next_word_model.h5` – Trained LSTM model.  
- `tokenizer.pickle` – Keras tokenizer used for training.  
- `requirements.txt` – Python dependencies.  
- `hamlet.txt` – Sample text used for training (optional).  
- `main.ipynb` / `experiments.ipynb` – Training notebooks.  

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
