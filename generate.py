import pandas as pd
import random
import language_tool_python

existing_data = pd.read_csv('sentiment_data.csv')

desired_entries = 25000 - len(existing_data)

synthetic_ratings = []
synthetic_comments = []

tool = language_tool_python.LanguageTool('en-US')

def improve_coherence(sentence):
    random_index = random.randint(0, len(existing_data) - 1)
    context_sentence = existing_data.iloc[random_index]['comment']
    improved_sentence = f"{context_sentence} {sentence}"
    return improved_sentence

for _ in range(desired_entries):
    random_index = random.randint(0, len(existing_data) - 1)
    existing_comment = existing_data.iloc[random_index]['comment']

    improved_comment = tool.correct(existing_comment)

    improved_comment = improve_coherence(improved_comment)

    synthetic_rating = random.choice([-1, 0, 1])

    synthetic_ratings.append(synthetic_rating)
    synthetic_comments.append(improved_comment)

synthetic_data = pd.DataFrame({'rating': synthetic_ratings, 'comment': synthetic_comments})

combined_data = pd.concat([existing_data, synthetic_data])

combined_data.to_csv('Final.csv', index=False)

print(f"Combined data with {len(combined_data)} entries saved to 'Final.csv'.")
