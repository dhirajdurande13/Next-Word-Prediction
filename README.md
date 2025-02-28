#Next-Word Prediction using LSTM (Hamlet Dataset)

This project develops a deep learning model to predict the next word in a given text sequence using Long Short-Term Memory (LSTM) networks. The model is trained on Shakespeare’s "Hamlet", making it a great challenge for natural language understanding.

Project Steps
Data Collection: Uses the text from Hamlet as training data.
Preprocessing: Tokenization, sequence generation, and padding for uniform input length.
Model Building:
Embedding layer
Two LSTM layers
Dense layer with softmax activation for word prediction
Training: The model learns from sequences using categorical cross-entropy loss, with early stopping to prevent overfitting.
Evaluation: The trained model is tested on example inputs to generate meaningful next-word predictions.
Usage
Run the Streamlit app to interact with the model:


streamlit run app.py
Requirements
Python 3.x
TensorFlow/Keras
Streamlit
NumPy, Pickle
Future Improvements
Train on larger datasets
Implement bidirectional LSTM for better predictions
Deploy as a web app
URL:(https://next-word-prediction-8fkgb6onxidduj5q2shsrf.streamlit.app/)
