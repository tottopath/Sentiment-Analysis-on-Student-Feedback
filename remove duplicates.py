import pandas as pd

# Read the CSV file into a DataFrame
input_file = "updated_file.csv"
df = pd.read_csv(input_file)

# Drop duplicates based on the "comment" column
df_no_duplicates = df.drop_duplicates(subset=["comment"])

# Save the result to a new CSV file
output_file = "sentiment_data_no_duplicates.csv"
df_no_duplicates.to_csv(output_file, index=False)

print(f"Removed {len(df) - len(df_no_duplicates)} duplicate entries.")
