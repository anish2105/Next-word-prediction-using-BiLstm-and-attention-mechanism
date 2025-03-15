import streamlit as st
import numpy as np
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, lstm_output):
        """
        lstm_output: The output from the BiLSTM layer (batch_size, seq_length, hidden_dim)
        """
        attention_scores = tf.nn.softmax(lstm_output, axis=1)
        attention_output = tf.reduce_sum(lstm_output * attention_scores, axis=1)
        return attention_output, attention_scores 

tf.keras.utils.get_custom_objects()["AttentionLayer"] = AttentionLayer

# ----------------------------
# Load the BiLSTM Model & Tokenizer
# ----------------------------
st.title("📝 Next Word Prediction with BiLSTM & Attention")

model_path = os.path.abspath("Models/Bi_lstm.h5")  
tokenizer_path = "Models/tokenizer.pickle"

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

try:
    model = load_model(model_path, custom_objects={"AttentionLayer": AttentionLayer}, compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

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

# ----------------------------
# Next Word Prediction Function
# ----------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Unknown Word"

# ----------------------------
# Generate Full Sentence Function
# ----------------------------
def generate_sentence(seed_text, next_words=10):
    sentence = seed_text
    for _ in range(next_words):
        next_word = predict_next_word(model, tokenizer, sentence, max_sequence_len)
        if next_word == "Unknown Word":
            break
        sentence += " " + next_word
    return sentence

# ----------------------------
# Attention Visualization Function
# ----------------------------
def visualize_attention(input_text):
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    padded_sequence = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    attention_layer = None
    for layer in model.layers:
        if isinstance(layer, AttentionLayer):
            attention_layer = layer
            break

    if attention_layer is None:
        for layer in model.layers:
            if 'attention' in layer.name.lower():
                attention_layer = layer
                break

    bilstm_layer = None
    for i, layer in enumerate(model.layers):
        if 'bidirectional' in layer.name.lower() or 'lstm' in layer.name.lower():
            bilstm_layer = layer
            break

    if bilstm_layer is None or attention_layer is None:
        st.error("Could not identify required layers in the model")
        return
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=bilstm_layer.output)
 
    bilstm_output = intermediate_model.predict(padded_sequence)
    
    _, attention_scores = attention_layer(bilstm_output)
    
    attention_scores = attention_scores.numpy()[0]  
    
    if len(attention_scores.shape) > 1:
        attention_scores = np.mean(attention_scores, axis=-1)
    
    words = input_text.split()
    
    num_pad_tokens = (max_sequence_len - 1) - len(token_list)
    
    relevant_attention = attention_scores[num_pad_tokens:] if num_pad_tokens > 0 else attention_scores
    
    num_words = min(len(words), len(relevant_attention))
    words = words[-num_words:]
    relevant_attention = relevant_attention[-num_words:]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(words, relevant_attention)
    ax.set_xlabel("Words in Input Text")
    ax.set_ylabel("Attention Score")
    ax.set_title("Attention Weights for Words")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# Streamlit UI
# ----------------------------
input_text = st.text_input("Type a few words:", "What do you")

col1, col2, col3 = st.columns(3)

# Predict Next Word
if col1.button("Predict Next Word"):
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"**Predicted Next Word:** {next_word}")

# Generate Full Sentence
if col2.button("Generate Full Sentence"):
    full_sentence = generate_sentence(input_text, next_words=20)
    st.write(f"**Generated Sentence:** {full_sentence}")

# Visualize Attention
if col3.button("Show Attention Weights"):
    visualize_attention(input_text)
