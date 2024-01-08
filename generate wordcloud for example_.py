import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from collections import defaultdict
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

data = pd.read_csv("updated_file.csv")  

custom_stopwords = ["courses", "class", "assignment", "teacher", "student", "school", "learning", "course", "subject", "matter", "material", "found"]

custom_stopwords_lower = [stopword.lower() for stopword in custom_stopwords]

sentiments = {-1: [], 0: [], 1: []}

for index, row in data.iterrows():
    sentiment = row["rating"]
    comment = row["comment"]
    sentiments[sentiment].append(comment)

for sentiment in [-1, 0, 1]:
    sentiment_name = "negative" if sentiment == -1 else "neutral" if sentiment == 0 else "positive"
    
    combined_text = ' '.join(sentiments[sentiment])

    words = combined_text.split()

    filtered_words = [word for word in words if word.lower() not in custom_stopwords_lower]

    filtered_text = ' '.join(filtered_words)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment_name.capitalize()} Sentiment')
    plt.axis('off')

    plt.savefig(f'static/{sentiment_name}_wordcloud.png')

images = [plt.imread(f'static/{sentiment_name}_wordcloud.png') for sentiment_name in ["negative", "neutral", "positive"]]

plt.figure(figsize=(15, 5))
for i, sentiment_name in enumerate(["Negative", "Neutral", "Positive"]):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i])
    plt.title(f'{sentiment_name} Sentiment')
    plt.axis('off')

plt.savefig('static/wordcloud.png')
