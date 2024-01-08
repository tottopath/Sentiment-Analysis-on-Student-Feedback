from flask import Flask, render_template, request, send_from_directory, jsonify
import tensorflow as tf
import pickle
import numpy as np
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from collections import defaultdict
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the RNN model and tokenizer
model_rnn = tf.keras.models.load_model('sentiment_rnn_model.h5')
with open('rnn_tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer_rnn = pickle.load(tokenizer_file)

# Load the BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer_bert = BertTokenizer.from_pretrained('bert_sentiment_tokenizer')
model_bert = BertForSequenceClassification.from_pretrained('bert_sentiment_model')

# Define function for sentiment analysis using BERT
def predict_sentiment_bert(text):
    inputs = tokenizer_bert(text, return_tensors="pt", padding=True, truncation=True)
    logits = model_bert(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)
    return torch.argmax(probabilities, dim=1).item()

@app.route('/')
def home():
    return render_template('comment.html')

@app.route('/index')
def index():
    return render_template('comment.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/example')
def example():
    return render_template('example.html')

def analyze_csv(text):
    user_input = text
    rnn_text_sequence = tokenizer_rnn.texts_to_sequences([user_input])
    rnn_text_sequence = pad_sequences(rnn_text_sequence, maxlen=100)
    rnn_sentiment = model_rnn.predict(rnn_text_sequence)
    rnn_sentiment_label = np.argmax(rnn_sentiment)
    
    bert_sentiment_label = predict_sentiment_bert(user_input)

    # Combine predictions from RNN and BERT
    combined_label = rnn_sentiment_label + bert_sentiment_label
    if combined_label == 1:
        combined_label = 'positive'
    elif combined_label == 2:
        combined_label = 'neutral'
    else:
        combined_label = 'negative'
    
    return combined_label

@app.route('/predict', methods=['POST'])
def analyze_sentiment():
    user_input = request.form['comment']

    # Sentiment analysis with RNN model
    rnn_text_sequence = tokenizer_rnn.texts_to_sequences([user_input])
    rnn_text_sequence = pad_sequences(rnn_text_sequence, maxlen=100)
    rnn_sentiment = model_rnn.predict(rnn_text_sequence)
    rnn_sentiment_label = np.argmax(rnn_sentiment)

    # Sentiment analysis with BERT model
    bert_sentiment_label = predict_sentiment_bert(user_input)

    # Combine predictions from RNN and BERT
    combined_label = rnn_sentiment_label + bert_sentiment_label
    if combined_label == 1:
        combined_label = 'positive'
    elif combined_label == 2:
        combined_label = 'neutral'
    else:
        combined_label = 'negative'
    
    print(combined_label)

    return render_template('comment.html', comment=user_input, sentiment=combined_label)

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    if 'csv-file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['csv-file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        df = pd.read_csv(file)
        custom_stopwords = ["courses", "class", "assignment", "teacher", "student", "school", "learning", "course", "subject", "matter", "material", "found"]

        # Convert the custom stopwords to lowercase
        custom_stopwords_lower = [stopword.lower() for stopword in custom_stopwords]

        # Create a dictionary to store comments for each sentiment
        sentiments = {-1: [], 0: [], 1: []}

        # Separate comments into sentiment groups
        for index, row in df.iterrows():
            sentiment = row["rating"]
            comment = row["comment"]
            sentiments[sentiment].append(comment)

        # Generate word clouds for each sentiment
        for sentiment in [-1, 0, 1]:
            sentiment_name = "negative" if sentiment == -1 else "neutral" if sentiment == 0 else "positive"
            
            # Combine comments into a single text for the word cloud
            combined_text = ' '.join(sentiments[sentiment])

            # Tokenize the text into words
            words = combined_text.split()

            # Remove stopwords (case-insensitive)
            filtered_words = [word for word in words if word.lower() not in custom_stopwords_lower]

            # Join the filtered words back into a single text
            filtered_text = ' '.join(filtered_words)

            # Create a WordCloud object without custom stopwords
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)

            # Plot the WordCloud for the current sentiment
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Word Cloud for {sentiment_name.capitalize()} Sentiment')
            plt.axis('off')

            # Save the WordCloud as an image in the "static" folder
            plt.savefig(f'static/{sentiment_name}_wordcloud_upload.png')

        # Combine the word cloud images into one graph
        images = [plt.imread(f'static/{sentiment_name}_wordcloud_upload.png') for sentiment_name in ["negative", "neutral", "positive"]]

        plt.figure(figsize=(15, 5))
        for i, sentiment_name in enumerate(["Negative", "Neutral", "Positive"]):
            plt.subplot(1, 3, i+1)
            plt.imshow(images[i])
            plt.title(f'{sentiment_name} Sentiment')
            plt.axis('off')

        # Save the combined graph as an image in the "static" folder
        plt.savefig('static/wordcloud_upload.png')

        rating_counts = df['rating'].value_counts()

        # Generate a bar chart
        plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots()
        plt.gca().set_facecolor('#FBFAFF')
        rating_counts.plot(kind='bar', color=['blue', 'red', 'green'])
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        ax.margins(0) 
        plt.savefig('static/bar_chart_upload.png')
        plt.close()

        # Generate a pie chart
        plt.figure(figsize=(8, 6))
        rating_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Rating Distribution')
        plt.axis('equal')
        plt.savefig('static/pie_chart_upload.png', bbox_inches='tight')
        plt.close()
        if 'comment' in df.columns:
            df['predicted_sentiment'] = df['comment'].apply(analyze_csv)
            output_filename = 'sentiment_data_with_predictions.csv'
            df.to_csv(output_filename, index=False)
            return send_from_directory('.', output_filename, as_attachment=True)

    return jsonify({"error": "Invalid CSV format or missing 'comment' column"})

@app.route('/download-csv')
def download_csv():
    return send_from_directory('static', 'sentiment_data.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
