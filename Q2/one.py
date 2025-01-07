# Import necessary libraries
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import pickle
from nltk.tokenize import word_tokenize
import nltk
import string
import re

# Download NLTK data files (only need to run once)
nltk.download('punkt')

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model layers and vocabulary mappings
@st.cache_resource
def load_resources():
    """
    Loads the pre-trained LSTM model layers and vocabulary mappings, initializes 
    them with saved parameters, and sets them to evaluation mode. Returns the 
    layers and vocabulary dictionaries.
    """
    checkpoint = torch.load('word_prediction_model.pt', map_location=device)
    vocab_to_int = checkpoint['vocab_to_int']
    int_to_vocab = checkpoint['int_to_vocab']
    embedding_dim = checkpoint['embedding_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_layers = checkpoint['num_layers']
    dropout = checkpoint['dropout']
    seq_length = checkpoint['seq_length']
    vocab_size = checkpoint['vocab_size']

    # Initialize layers
    embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    lstm_layer = nn.LSTM(
        input_size=embedding_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout,
    )
    fc_layer = nn.Linear(hidden_dim, vocab_size)
    dropout_layer = nn.Dropout(dropout)

    # Load saved state dictionaries
    embedding_layer.load_state_dict(checkpoint['embedding_layer_state_dict'])
    lstm_layer.load_state_dict(checkpoint['lstm_layer_state_dict'])
    fc_layer.load_state_dict(checkpoint['fc_layer_state_dict'])
    dropout_layer.load_state_dict(checkpoint['dropout_layer_state_dict'])

    # Move layers to device
    embedding_layer.to(device)
    lstm_layer.to(device)
    fc_layer.to(device)
    dropout_layer.to(device)

    # Set layers to evaluation mode
    embedding_layer.eval()
    lstm_layer.eval()
    fc_layer.eval()
    dropout_layer.eval()

    return embedding_layer, lstm_layer, fc_layer, dropout_layer, vocab_to_int, int_to_vocab, seq_length

embedding_layer, lstm_layer, fc_layer, dropout_layer, vocab_to_int, int_to_vocab, seq_length = load_resources()

# Function to preprocess input text
def preprocess_input(text):
    """
    Preprocesses the input text by converting to lowercase, removing punctuations, 
    numbers, and extra spaces, and then tokenizing into words. Returns the processed words.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    words = word_tokenize(text)
    return words

# Function to predict the next words
def predict_next_words(
    embedding_layer, lstm_layer, fc_layer, dropout_layer,
    vocab_to_int, int_to_vocab, text, seq_length, top_k=3
):
    """
    Takes a partial sentence as input, processes it into a sequence of integers, 
    feeds it into the LSTM model, and returns the top `top_k` predicted words 
    based on the final layer outputs.
    """
    tokens = preprocess_input(text)
    if not tokens:
        return []
    
    tokens = tokens[-seq_length:]
    unk_index = vocab_to_int.get('<UNK>')
    encoded = [vocab_to_int.get(word, unk_index) for word in tokens]
    encoded = [0] * (seq_length - len(encoded)) + encoded  # Pre-padding
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    # Predict the next word
    with torch.no_grad():
        embeddings = embedding_layer(input_tensor)
        lstm_out, (hn, cn) = lstm_layer(embeddings)
        lstm_out = dropout_layer(lstm_out[:, -1, :])
        outputs = fc_layer(lstm_out)
        probabilities = softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_indices = top_indices.cpu().numpy()[0]

    predicted_words = [int_to_vocab.get(idx, '<UNK>') for idx in top_indices]
    return predicted_words

# Streamlit App for Next Word Prediction
st.title("Next Word Prediction")
st.write("Type a partial sentence, and the model will predict the next words every time you press the space key.")

# Initialize session state
if 'prev_input_text' not in st.session_state:
    st.session_state.prev_input_text = ''
if 'prev_space_count' not in st.session_state:
    st.session_state.prev_space_count = 0

# Create a text input that updates as the user types
input_text = st.text_input("Enter your text here:", value='', max_chars=None, key='input_text')

# Count the number of spaces in the current and previous input
current_space_count = input_text.count(' ')
prev_space_count = st.session_state.prev_space_count

# Check if a new space has been added
if current_space_count > prev_space_count and input_text.strip():
    # Make prediction
    predicted_words = predict_next_words(
        embedding_layer, lstm_layer, fc_layer, dropout_layer,
        vocab_to_int, int_to_vocab, input_text, seq_length
    )
    if predicted_words:
        st.write(f"**Predicted Next Words:** {', '.join(predicted_words)}")
    else:
        st.write("**Predicted Next Words:** (No prediction)")
else:
    st.write("**Predicted Next Words:** (Waiting for space key...)")

# Update session state
st.session_state.prev_input_text = input_text
st.session_state.prev_space_count = current_space_count
