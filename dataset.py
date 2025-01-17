import pandas as pd

# Load the dataset
df = pd.read_csv("Data/augmented_dataset1.csv")

# Display basic info about the dataset
print(df.info())

# Display the first few rows to see data structure
print(df.head())

# Check for class distribution
print(df['label'].value_counts())
