#RNN then BERT
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load the CSV file
df = pd.read_csv("updated_file.csv")

# Define labels and convert them to one-hot encoding
labels = df['rating'].apply(lambda x: [0, 1, 0] if x == 0 else [1, 0, 0] if x == 1 else [0, 0, 1])
labels = np.array(list(labels))

# Tokenize the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['comment'])
X = tokenizer.texts_to_sequences(df['comment'])
X = pad_sequences(X, maxlen=100)  # Padding sequences to the same length

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Define the RNN model
model_rnn = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 128, input_length=100),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(3, activation='softmax')
])

model_rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the RNN model
model_rnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

model_rnn.save('sentiment_rnn_model.h5')

# Save the tokenizer to a file
import pickle
with open('rnn_tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)
    
# Make predictions using the RNN model
y_pred_rnn = model_rnn.predict(X_test)

# Convert one-hot encoded predictions to labels
y_pred_labels_rnn = [np.argmax(pred) for pred in y_pred_rnn]
y_true_labels_rnn = [np.argmax(label) for label in y_test]

# Define the BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer_bert = BertTokenizer.from_pretrained(model_name)
model_bert = BertForSequenceClassification.from_pretrained(model_name)

# Define function for sentiment analysis using BERT
def predict_sentiment_bert(text):
    inputs = tokenizer_bert(text, return_tensors="pt", padding=True, truncation=True)
    logits = model_bert(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)
    return torch.argmax(probabilities, dim=1).item()

# Use the BERT model to make predictions
y_pred_bert = [predict_sentiment_bert(text) for text in df['comment']]

# Combine predictions from RNN and BERT
combined_predictions = [y_pred_rnn[i] + y_pred_bert[i] for i in range(len(y_pred_rnn))]

# Convert combined predictions to labels
combined_labels = [np.argmax(pred) for pred in combined_predictions]

# Calculate TP, TN, FP, FN using confusion matrix for the hybrid model
conf_matrix = confusion_matrix(y_true_labels_rnn, combined_labels)

tp_hybrid, fp_hybrid, fn_hybrid, tn_hybrid = conf_matrix[0, 0], conf_matrix[0, 2], conf_matrix[2, 0], conf_matrix[2, 2]

accuracy_hybrid = (tp_hybrid + tn_hybrid) / (tp_hybrid + tn_hybrid + fp_hybrid + fn_hybrid)

print(f"True Positives (TP) for Hybrid Model: {tp_hybrid}")
print(f"True Negatives (TN) for Hybrid Model: {tn_hybrid}")
print(f"False Positives (FP) for Hybrid Model: {fp_hybrid}")
print(f"False Negatives (FN) for Hybrid Model: {fn_hybrid}")
print(f"Accuracy for Hybrid Model: {accuracy_hybrid * 100:.2f}%")

# Save the BERT model and tokenizer
model_bert.save_pretrained('bert_sentiment_model')
tokenizer_bert.save_pretrained('bert_sentiment_tokenizer')
