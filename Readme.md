# Next Word Prediction Using LSTM and BiLSTM with Attention

## Project Overview
This project aims to develop a deep learning model that predicts the next word in a given sequence of words. Two models have been implemented:
1. **LSTM-Based Model**
2. **BiLSTM with Attention Mechanism**

Both models are trained using a dataset of jokes from Hugging Face, and a Streamlit web application is built to allow users to interactively generate text.

## Features
- **Preprocessing:** Tokenization, sequence generation, and padding.
- **Model Architectures:**
  - LSTM Model with an embedding layer, LSTM layers, and a softmax output layer.
  - BiLSTM Model with an Attention Layer to improve contextual learning.
- **Training and Evaluation:**
  - Trained using categorical cross-entropy loss and Adam optimizer.
  - Uses Early Stopping to prevent overfitting.
- **Deployment:**
  - A Streamlit-based web interface for real-time next-word prediction.
  - Model is saved and loaded efficiently for inference.

---

## Dataset
The dataset used for training is a collection of jokes from Hugging Face. The text is preprocessed by:
1. **Tokenization**: Converting words into numerical tokens.
2. **Generating Sequences**: Creating input sequences from sentences.
3. **Padding**: Ensuring uniform input lengths.
4. **Splitting Data**: Dividing into training and testing sets.

---

## Model 1: LSTM-Based Next Word Prediction

### Architecture:
- **Embedding Layer**: Converts words into dense vectors.
- **GRU Layers**: Two GRU layers for sequence processing.
- **Dropout Layer**: Reduces overfitting.
- **Dense Output Layer**: Uses softmax activation to predict the next word.

### Training:
- Loss Function: `categorical_crossentropy`
- Optimizer: `adam`
- Training for 50 epochs with early stopping.

### Deployment:
A **Streamlit web application** is developed, where users can enter text and receive the next predicted word.

---

## Model 2: BiLSTM with Attention Mechanism

### Architecture:
- **Embedding Layer**: Converts words into vectors.
- **Bidirectional LSTM Layer**: Captures context from both directions.
- **Attention Mechanism**: Focuses on important words in the sequence.
- **Dropout Layer**: Prevents overfitting.
- **Dense Output Layer**: Uses softmax activation for word prediction.

### Training:
- Loss Function: `categorical_crossentropy`
- Optimizer: `adam`
- Training for 50 epochs with early stopping.

### Deployment:
A **Streamlit-based UI** is built to:
- Predict the next word.
- Generate full sentences dynamically.
- **Visualize Attention Scores**: Shows which words contributed most to predictions.

---

## Installation & Usage

### Prerequisites
Ensure you have Python installed along with the following dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
1. **Run the Streamlit App for LSTM model:**
   ```bash
   streamlit run application.py
   ```
2. **Run the Streamlit App for Bi-LSTM model with attention mechanism:**
   ```bash
   streamlit run app.py
   ```
---

## Future Enhancements
- Implement Transformer-based models (e.g., BERT or GPT) for improved predictions.
- Train on larger, more diverse datasets.
- Fine-tune hyperparameters for better generalization.

---

## Repository Structure
```
|-- Models/
|   |-- next_word_lstm.h5
|   |-- Bi_lstm.h5
|   |-- tokenizer.pickle
|
|-- notebook/
|   |-- Next_Word_Prediction_Using_LSTM.ipynb
|
|-- application.py   # Streamlit UI for LSTM
|-- application_bilstm.py  # Streamlit UI for BiLSTM with Attention
|-- requirements.py         
|-- README.md        
```

---

## Contributors
- **Anish**

---


