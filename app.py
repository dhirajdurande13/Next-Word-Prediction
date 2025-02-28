import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# LOad the model

# model=load_model('next_word_lstm.h5')
model = load_model('next_word_lstm.h5', compile=False)

# Load the tokenizer
with open('tokenizer.pkl','rb') as handle:
    try:
        tokenizer=pickle.load(handle)
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
    
    # Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


# Streamlit app
st.title("Next Word prediction with LSTM")

input_text = st.text_input("Enter text for the next word prediction:", "To be or not to be")

if st.button("Next word"):
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Prediction:{input_text,next_word}")
    