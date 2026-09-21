import pandas as pd

# Load the dataset
df = pd.read_csv("data/AI_data.csv")

# Show basic information
print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 rows:")
print(df.head())