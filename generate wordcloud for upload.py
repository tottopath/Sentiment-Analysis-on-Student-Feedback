import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from collections import defaultdict
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv("updated_file.csv")  # Update the path to your CSV file

# Define custom stopwords specific to education sentiment analysis (in lowercase)
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