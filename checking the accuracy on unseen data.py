import pandas as pd

# Read the CSV file
df = pd.read_csv('sentiment_data_with_predictions2.csv')

# Create a dictionary to map predicted sentiments to their corresponding ratings
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}

# Map the 'predicted_sentiment' column to ratings
df['predicted_rating'] = df['predicted_sentiment'].map(sentiment_mapping)

# Calculate the percentage of matches
matches = (df['rating'] == df['predicted_rating']).sum()
total_rows = len(df)
percentage_matched = (matches / total_rows) * 100

print(f"Percentage of matches between 'rating' and 'predicted_sentiment': {percentage_matched:.2f}%")
