# FOR UPDATING THE RATINGS
# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to convert sentiment scores to labels
def sentiment_label(score):
    if score >= 0.05:
        return 1  # Positive
    elif score <= -0.05:
        return -1  # Negative
    else:
        return 0  # Neutral

# Load the CSV file into a DataFrame
df = pd.read_csv('trial.csv')

# Perform sentiment analysis on the comments and update incorrect ratings
for index, row in df.iterrows():
    comment = row['comment']
    rating = row['rating']

    # Calculate sentiment score
    sentiment_score = sia.polarity_scores(comment)['compound']

    # Convert sentiment score to label
    predicted_rating = sentiment_label(sentiment_score)

    # Check if the predicted rating is different from the provided rating
    if predicted_rating != rating:
        print(f"Updating rating for row {index}: {rating} => {predicted_rating}")
        df.at[index, 'rating'] = predicted_rating

# Save the updated DataFrame back to the CSV file
df.to_csv('updated_file.csv', index=False)
