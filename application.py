import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model_path = os.path.abspath("Models/next_word_lstm.h5")  # Ensure correct file path
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

try:
    model = load_model(model_path, compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the Tokenizer
tokenizer_path = "Models/tokenizer.pickle"
if not os.path.exists(tokenizer_path):
    st.error(f"Tokenizer file not found: {tokenizer_path}")
    st.stop()

try:
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

max_sequence_len = model.input_shape[1] + 1 if model.input_shape else 39 

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Unknown Word"

# Function to generate a full sentence
def generate_sentence(seed_text, next_words=10):
    sentence = seed_text
    for _ in range(next_words):
        next_word = predict_next_word(model, tokenizer, sentence, max_sequence_len)
        if next_word == "Unknown Word":
            break
        sentence += " " + next_word
    return sentence

# Streamlit App UI
st.title("📝 Next Word Prediction with LSTM & Attention")

input_text = st.text_input("Type a few words:", "What do you")

col1, col2 = st.columns(2)

# Predict Next Word in Real-Time
if col1.button("Predict Next Word"):
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    if next_word:
        st.write(f"**Predicted Next Word:** {next_word}")
    else:
        st.write("⚠️ No prediction found. Try different input.")

# Generate Full Sentence
if col2.button("Generate Full Sentence"):
    full_sentence = generate_sentence(input_text, next_words=10)
    st.write(f"**Generated Sentence:** {full_sentence}")
