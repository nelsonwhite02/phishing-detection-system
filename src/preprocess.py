import re
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to tokenize URLs
def tokenize_url(url):
    tokens = re.split(r'[./\-]', url)
    return [token for token in tokens if token]

def preprocess_data(input_file):
    df = pd.read_csv(input_file)
    df['tokens'] = df['url'].apply(tokenize_url)

    X = df['tokens']
    y = df['label']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_df = pd.DataFrame({'tokens': X_train, 'label': y_train})
    val_df = pd.DataFrame({'tokens': X_val, 'label': y_val})
    test_df = pd.DataFrame({'tokens': X_test, 'label': y_test})

    return train_df, val_df, test_df

# Run the preprocessing
if __name__ == "__main__":
    train_df, val_df, test_df = preprocess_data("data/augmented_dataset.csv")
