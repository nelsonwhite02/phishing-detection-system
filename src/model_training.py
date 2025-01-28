import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import BertTokenizer, TFBertModel
# import tensorflow as tf
import re
import math
import shap
from lime.lime_tabular import LimeTabularExplainer

# Analyzer function for TF-IDF
def identity_analyzer(tokens):
    return tokens

# Calculate entropy for zero-day detection
def calculate_entropy(url):
    probabilities = [float(url.count(c)) / len(url) for c in dict.fromkeys(url)]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

feature_descriptions = {
    "Subdomain Count": "The number of subdomains in the URL. Phishing URLs often use multiple subdomains to appear legitimate.",
    "Special Char Count": "The count of special characters (e.g., `@`, `-`, `?`). Phishing links often include these to confuse users.",
    "BERT Dim": "Patterns learned by BERT that may indicate suspicious language in the URL.",
    "Entropy": "The randomness of characters in the URL. High randomness can indicate obfuscation, a common phishing tactic.",
    "URL Length": "The total length of the URL. Phishing URLs are often excessively long to obscure their true intent.",
    "Num-Letter Ratio": "The ratio of numeric to alphabetic characters. An unusual ratio can indicate phishing.",
    "Has IP": "Whether the URL contains an IP address instead of a domain name. IP-based URLs are often used in phishing."
}

# Generate feature names dynamically
def generate_feature_names(num_features, tfidf_count=7000, bert_count=768, additional_features=None):
    additional_features = additional_features or ['Entropy', 'URL Length', 'Special Char Count', 'Subdomain Count', 'Num-Letter Ratio', 'Has IP']
    feature_names = []

    # Add TF-IDF features
    if num_features >= tfidf_count:
        feature_names.extend([f"TF-IDF Feature {i}" for i in range(tfidf_count)])
        num_features -= tfidf_count

    # Add BERT features
    if num_features >= bert_count:
        feature_names.extend([f"BERT Dim {i}" for i in range(bert_count)])
        num_features -= bert_count

    # Add additional features
    feature_names.extend(additional_features[:num_features])
    return feature_names

# Extract BERT features
def extract_bert_features(urls, bert_model, tokenizer, max_length=64, batch_size=100):
    bert_features = []
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i + batch_size]
        encodings = tokenizer(
            batch_urls,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="tf"
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        bert_features.append(outputs.last_hidden_state[:, 0, :].numpy())  # CLS token
    return np.vstack(bert_features)

# Preprocess and vectorize URLs
def preprocess_data(input_file, bert_model, tokenizer, max_length=64, batch_size=256):
    df = pd.read_csv(input_file)
    df['tokens'] = df['url'].apply(lambda x: x.split('/'))

    df = df.drop(columns=['type'], errors='ignore').dropna(subset=['url', 'label'])

    df_phishing = df[df['label'] == 'phishing']
    df_legitimate = df[df['label'] == 'legitimate'].sample(n=len(df_phishing), random_state=42)
    df_balanced = pd.concat([df_phishing, df_legitimate])

    print("Class distribution in balanced data:", df_balanced['label'].value_counts())

    vectorizer = TfidfVectorizer(analyzer=identity_analyzer, max_features=7000)
    X_tfidf = vectorizer.fit_transform(df_balanced['tokens'])

    df_balanced['entropy'] = df_balanced['url'].apply(calculate_entropy)
    df_balanced['url_length'] = df_balanced['url'].apply(len)
    df_balanced['special_char_count'] = df_balanced['url'].apply(lambda x: sum([x.count(char) for char in ['@', '-', '?']]))
    df_balanced['subdomain_count'] = df_balanced['url'].apply(lambda x: x.count('.'))
    df_balanced['num_letter_ratio'] = df_balanced['url'].apply(lambda x: sum(char.isdigit() for char in x) / max(1, sum(char.isalpha() for char in x)))
    df_balanced['has_ip'] = df_balanced['url'].apply(lambda x: 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', x) else 0)

    print("Starting BERT feature extraction...")
    bert_features = extract_bert_features(df_balanced['url'].tolist(), bert_model, tokenizer, max_length, batch_size)
    print("BERT feature extraction completed.")

    additional_features = df_balanced[['entropy', 'url_length', 'special_char_count', 'subdomain_count', 'num_letter_ratio', 'has_ip']].values
    X_combined = hstack([X_tfidf, bert_features, additional_features])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_balanced['label'])

    X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer

# Train ensemble model
def train_ensemble_model(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    xgb_model = XGBClassifier(n_estimators=200, max_depth=15, learning_rate=0.1, random_state=42)
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model)],
        voting='soft'
    )
    ensemble_model.fit(X_train, y_train)
    return ensemble_model

# SHAP explanations
def explain_model_shap(model, X_sample, feature_names, class_index=1):
    print(f"SHAP Explainer Input Shape: {X_sample.shape}")

    # Explain Random Forest
    try:
        rf_model = model.named_estimators_['rf']
        explainer_rf = shap.TreeExplainer(rf_model)
        shap_values_rf = explainer_rf.shap_values(X_sample)
        shap_values_rf_class = shap_values_rf[..., class_index]
        print(f"SHAP Values for Class {class_index} Shape (Random Forest): {shap_values_rf_class.shape}")
        textual_explanation = generate_text_explanation(shap_values_rf_class[0], feature_names)
        print(f"Explanation for Class {class_index} (Random Forest): {textual_explanation}")
    except Exception as e:
        print(f"Error generating Random Forest explanation: {e}")

# Generate textual explanation from SHAP values
# def generate_text_explanation(shap_values, feature_names, top_n=5):
#     top_indices = np.argsort(-np.abs(shap_values))[:top_n]
#     explanations = [f"{feature_names[i]}: {shap_values[i]:.4f}" for i in top_indices]
#     return " | ".join(explanations)
# def generate_text_explanation(shap_values, feature_names, top_k=5):
#     # Sort features by absolute SHAP value
#     feature_importance = sorted(
#         zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True
#     )[:top_k]
    
#     # Map to meaningful descriptions
#     explanations = []
#     for feature, value in feature_importance:
#         description = (
#             feature_descriptions.get(feature, feature)  # Default to the feature name
#         )
#         explanations.append(f"{description}: {value:.4f}")
    
#     return " | ".join(explanations)
def generate_text_explanation(shap_values, feature_names, top_k=5):
    # Sort features by absolute SHAP value
    feature_importance = sorted(
        zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True
    )[:top_k]
    
    # Map to meaningful descriptions
    explanations = []
    for feature, value in feature_importance:
        # Check if the feature is one of the predefined features or a BERT feature
        if "BERT Dim" in feature:
            description = f"BERT model detected linguistic patterns related to phishing ({feature})"
        else:
            description = feature_descriptions.get(feature, feature)  # Default to the feature name
        
        explanations.append(f"{description}: {value:.4f}")
    
    return " | ".join(explanations)

if __name__ == "__main__":
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = preprocess_data("data/augmented_dataset.csv", bert_model, tokenizer)

    model = train_ensemble_model(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    optimal_threshold = 0.45
    y_pred = (y_pred_proba > optimal_threshold).astype(int)

    print("\nTest Set Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    feature_names = generate_feature_names(
        num_features=X_test.shape[1],
        tfidf_count=7000,
        bert_count=768,
        additional_features=['Entropy', 'URL Length', 'Special Char Count', 'Subdomain Count', 'Num-Letter Ratio', 'Has IP']
    )

    explain_model_shap(model, X_test[:100].toarray(), feature_names)
