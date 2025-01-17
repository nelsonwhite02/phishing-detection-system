import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

# Preprocess and vectorize URLs
def preprocess_data(input_file):
    # Load the dataset
    df = pd.read_csv(input_file)
    df['tokens'] = df['url'].apply(lambda x: x.split('/'))

    # Remove the 'type' column and drop rows with missing values
    df = df.drop(columns=['type'], errors='ignore').dropna(subset=['url', 'label'])

    # Balance the dataset by undersampling the legitimate class
    df_phishing = df[df['label'] == 'phishing']
    df_legitimate = df[df['label'] == 'legitimate'].sample(n=len(df_phishing), random_state=42)
    df_balanced = pd.concat([df_phishing, df_legitimate])

    # Confirm balanced class distribution
    print("Class distribution in balanced data:", df_balanced['label'].value_counts())

    # Vectorize the tokenized URLs
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, max_features=5000)
    X_tfidf = vectorizer.fit_transform(df_balanced['tokens'])

    # Additional URL-specific features
    df_balanced['url_length'] = df_balanced['url'].apply(len)
    df_balanced['special_char_count'] = df_balanced['url'].apply(lambda x: sum([x.count(char) for char in ['@', '-', '?']]))
    df_balanced['subdomain_count'] = df_balanced['url'].apply(lambda x: x.count('.'))
    
    # New features
    df_balanced['num_letter_ratio'] = df_balanced['url'].apply(lambda x: sum(char.isdigit() for char in x) / max(1, sum(char.isalpha() for char in x)))
    df_balanced['has_ip'] = df_balanced['url'].apply(lambda x: 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', x) else 0)

    # Extract additional features as a matrix
    additional_features = df_balanced[['url_length', 'special_char_count', 'subdomain_count', 'num_letter_ratio', 'has_ip']].values

    # Combine TF-IDF features and additional features
    X = hstack([X_tfidf, additional_features])

    # Encode the labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_balanced['label'])

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Train the Random Forest model
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Test Set Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Main function to run the whole process
if __name__ == "__main__":
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data("data/augmented_dataset.csv")

    # Train Random Forest model
    model = train_random_forest(X_train, y_train)

    # Evaluate on test set
    evaluate_model(model, X_test, y_test)
