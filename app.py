from flask import Flask, request, jsonify
import joblib
from transformers import BertTokenizer, TFBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from seleniumwire import webdriver
import numpy as np
import re
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.model_training import generate_feature_names
from scipy.sparse import hstack
from collections import Counter
import logging
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
from flask_cors import CORS
from scipy.sparse import issparse
import shap
import math
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials






# Configure logging
logging.basicConfig(filename="api.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Initialize Flask App
app = Flask(__name__)

# Enable CORS for the app
CORS(app)

# Set up rate limiting
limiter = Limiter(get_remote_address, app=app, default_limits=["10 per minute"])

# Set TensorFlow log level to suppress unnecessary messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load the model, vectorizer, and threshold
print("Loading model, vectorizer, and threshold...")
model = joblib.load("Models/phishing_detection_model.pkl")
vectorizer = joblib.load("Models/tfidf_vectorizer.pkl")
with open("Models/optimal_threshold.txt", "r") as f:
    optimal_threshold = float(f.read())
print("Model, vectorizer, and threshold loaded successfully.")

# Load BERT tokenizer and model
print("Loading BERT model and tokenizer...")
bert_model = TFBertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
print("BERT model and tokenizer loaded successfully.")



def identity_analyzer(tokens):
    """Analyzer function to replace the lambda in TfidfVectorizer."""
    return tokens

def detect_downloads(driver):
    """Custom logic to detect downloads from network traffic."""
    downloads = []
    for request in driver.requests:
        if request.response and "application" in request.response.headers.get("Content-Type", ""):
            downloads.append(request.url)
    return downloads

# Helper function for BERT embeddings
# def extract_bert_features(urls, max_length=64, batch_size=256):
#     embeddings = []
#     for i in range(0, len(urls), batch_size):
#         batch_urls = urls[i:i + batch_size]
#         encodings = tokenizer(
#             batch_urls,
#             padding="max_length",
#             truncation=True,
#             max_length=max_length,
#             return_tensors="tf"
#         )
#         input_ids = encodings["input_ids"]
#         attention_mask = encodings["attention_mask"]
#         outputs = bert_model(input_ids, attention_mask=attention_mask)
#         embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())  # CLS token
#     return np.vstack(embeddings)

def extract_bert_features(urls, max_length=64, batch_size=256):
    embeddings = []
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
        embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())  # CLS token
    return np.vstack(embeddings)


def calculate_entropy(url):
    probabilities = [float(url.count(c)) / len(url) for c in dict.fromkeys(url)]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

# Preprocess additional features
def extract_additional_features(urls):
    additional_features = []
    for url in urls:
        url_length = len(url)
        special_char_count = sum(url.count(char) for char in ['@', '-', '?'])
        subdomain_count = url.count('.')
        num_letter_ratio = sum(c.isdigit() for c in url) / max(1, sum(c.isalpha() for c in url))
        has_ip = 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url) else 0
        entropy = calculate_entropy(url)
        additional_features.append([float(entropy), float(url_length), float(special_char_count),
                                    float(subdomain_count), float(num_letter_ratio), float(has_ip)])
    return np.array(additional_features, dtype=np.float32)


def extract_webpage_features_from_text(text_content):
    """
    Extract textual features from raw text content for NLP analysis.
    """
    try:
        # Clean the text (basic cleaning, you can expand it further)
        clean_text = ' '.join(text_content.split())

        # Extract features using NLP techniques
        features = {}

        # TF-IDF feature
        vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([clean_text])
        features['tfidf_terms'] = vectorizer.get_feature_names_out()
        features['tfidf_scores'] = tfidf_matrix.toarray()[0].tolist()

        # Sentiment Analysis
        sentiment = TextBlob(clean_text).sentiment
        features['sentiment'] = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}

        # Average word length
        words = clean_text.split()
        if words:
            features['average_word_length'] = sum(len(word) for word in words) / len(words)
        else:
            features['average_word_length'] = 0

        # Word count
        features['word_count'] = len(words)

        # Top keywords
        word_frequencies = Counter(words)
        features['top_keywords'] = dict(word_frequencies.most_common(5))

        return features

    except Exception as e:
        return {"error": str(e)}

# Generate textual explanation from SHAP values
def generate_text_explanation(shap_values, feature_names, top_k=5):
    """
    Generate a human-readable explanation of the model's prediction.

    Args:
        shap_values (np.ndarray): Array of SHAP values for the input instance.
        feature_names (list): List of feature names corresponding to the SHAP values.
        top_k (int): Number of top features to include in the explanation.

    Returns:
        str: A formatted string explaining the model's decision.
    """
    # Predefined descriptions for features
    feature_descriptions = {
        "Subdomain Count": "The number of subdomains in the URL. Phishing URLs often use multiple subdomains to appear legitimate.",
        "Special Char Count": "The count of special characters (e.g., `@`, `-`, `?`). Phishing links often include these to confuse users.",
        "URL Length": "The total number of characters in the URL. Longer URLs may indicate obfuscation attempts.",
        "Entropy": "The randomness in the URL's character distribution. Higher entropy can indicate suspicious URLs.",
        "Num-Letter Ratio": "The ratio of numbers to letters in the URL. Phishing links often have unusual patterns.",
        "Has IP": "Indicates whether the URL contains an IP address instead of a domain name, a common phishing tactic.",
    }

    # Sort features by absolute SHAP value (importance)
    feature_importance = sorted(
        zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True
    )[:top_k]

    # Generate explanations
    explanations = []
    for feature, value in feature_importance:
        # Map feature name to its description if available
        description = feature_descriptions.get(feature, feature)
        
        # If the feature is a BERT dimension, provide a generic explanation
        if "BERT Dim" in feature:
            description = f"BERT model detected linguistic patterns ({feature})"
        
        # Add feature contribution to the explanation
        explanations.append(f"{description}: {value:.4f}")

    return " | ".join(explanations)

def extract_webpage_features(url):
    """
    Extract textual features from the webpage for NLP analysis.
    """
    try:
        # Fetch the webpage content
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the text from the webpage
        page_text = soup.get_text(separator=' ', strip=True)

        # Clean the text (remove excess whitespace, non-ASCII characters, etc.)
        clean_text = ' '.join(page_text.split())
        clean_text = ''.join(char for char in clean_text if char.isprintable())

        # Initialize features dictionary
        features = {}

        # TF-IDF feature extraction
        vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([clean_text])
        features['tfidf_terms'] = vectorizer.get_feature_names_out().tolist()
        features['tfidf_scores'] = tfidf_matrix.toarray()[0].tolist()

        # Sentiment Analysis
        sentiment = TextBlob(clean_text).sentiment
        features['sentiment'] = {
            'polarity': round(sentiment.polarity, 4),
            'subjectivity': round(sentiment.subjectivity, 4)
        }

        # Length-based metrics
        features['text_length'] = len(clean_text)  # Total character count
        features['word_count'] = len(clean_text.split())  # Number of words
        features['average_word_length'] = (
            features['text_length'] / features['word_count'] if features['word_count'] > 0 else 0
        )

        # Keyword Frequency (Top 5 Words by Count)
        word_counts = {}
        for word in clean_text.split():
            word = word.lower()
            word_counts[word] = word_counts.get(word, 0) + 1
        sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        features['top_keywords'] = {word: count for word, count in sorted_keywords}

        # Count of HTML Tags (basic structural analysis)
        features['html_tag_counts'] = {tag.name: len(soup.find_all(tag.name)) for tag in soup.find_all(True)}

        return features

    except requests.exceptions.RequestException as req_err:
        return {"error": f"HTTP request error: {str(req_err)}"}
    except Exception as e:
        return {"error": f"An error occurred during feature extraction: {str(e)}"}

def analyze_url_behavior(url):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    forms = driver.find_elements(By.TAG_NAME, "form")
    form_details = []
    for index, form in enumerate(forms):
        inputs = form.find_elements(By.TAG_NAME, "input")
        input_types = [inp.get_attribute("type") for inp in inputs]
        action_url = form.get_attribute("action") or "N/A"
        is_insecure = action_url.startswith("http://")
        form_details.append({
            "form_id": f"Form {index + 1}",
            "action": action_url,
            "input_fields": input_types,
            "insecure_action": is_insecure,
        })

    downloads = detect_downloads(driver)  # Existing function for downloads
    redirections = len(driver.window_handles) - 1

    driver.quit()
    return {
        "url": url,
        "redirections": redirections,
        "forms": form_details,
        "downloads": downloads,
    }


SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def extract_links_from_emails():
    # Authenticate with Gmail API
    creds = Credentials.from_authorized_user_file('credentials.json', SCOPES)
    service = build('gmail', 'v1', credentials=creds)

    results = service.users().messages().list(userId='me', maxResults=10).execute()
    messages = results.get('messages', [])
    
    links = []
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        email_body = msg_data['snippet']
        links.extend(re.findall(r'(https?://\S+)', email_body))

    return links



# Preprocess URLs for prediction
def preprocess_urls(urls):
    # TF-IDF features
    tokens = [url.split('/') for url in urls]
    X_tfidf = vectorizer.transform(tokens)
    print(f"TF-IDF Shape: {X_tfidf.shape}")

    # BERT embeddings
    X_bert = extract_bert_features(urls)
    print(f"BERT Shape: {X_bert.shape}")

    # Additional features
    additional_features = extract_additional_features(urls)
    print(f"Additional Features Shape: {additional_features.shape}")

    # Combine features
    X_combined = hstack([X_tfidf, X_bert, additional_features])
    print(f"Combined Features Shape: {X_combined.shape}")

    # Ensure dense format
    if issparse(X_combined):
        X_combined = X_combined.toarray()

    return X_combined


@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    urls = data.get('urls', [])

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    # Preprocess the URLs
    X_input = preprocess_urls(urls)
    print(f"Combined Features Shape: {X_input.shape}")

    # Generate feature names dynamically
    feature_names = generate_feature_names(
        num_features=X_input.shape[1],
        tfidf_count=7000,
        bert_count=768,
        additional_features=['Entropy', 'URL Length', 'Special Char Count', 'Subdomain Count', 'Num-Letter Ratio', 'Has IP']
    )

    # Generate predictions
    predictions = model.predict(X_input)
    probabilities = model.predict_proba(X_input)[:, 1]

    try:
        # Explain using SHAP
        shap_explainer_rf = shap.TreeExplainer(model.named_estimators_['rf'])
        shap_values_rf = shap_explainer_rf.shap_values(X_input)
        print(f"SHAP Values Shape (Random Forest): {shap_values_rf.shape}")

        # Extract SHAP values for the phishing class
        shap_values_rf_class = shap_values_rf[..., 1]  # Focus on phishing class

        # Generate text explanations
        explanations = [
            generate_text_explanation(shap_values_rf_class[i], feature_names)
            for i in range(len(urls))
        ]
    except Exception as e:
        return jsonify({"error": f"Unexpected SHAP values structure: {str(e)}"})

    # Combine results
    response = [
        {
            "url": urls[i],
            "prediction": int(predictions[i]),
            "probability": float(probabilities[i]),
            "explanation": explanations[i]
        }
        for i in range(len(urls))
    ]

    return jsonify(response)


# @app.route('/analyze-webpage', methods=['POST'])
# def analyze_webpage():
#     data = request.json
#     url = data.get('url')

#     if not url:
#         return jsonify({"error": "No URL provided"}), 400

#     try:
#         # Extract textual features
#         features = extract_webpage_features(url)

#         # Generate embeddings for phishing detection
#         embeddings = extract_bert_features(
#             [url],  # Ensure this is a list
#             max_length=64,
#             batch_size=256
#         )

#         # Combine embeddings with textual features
#         combined_features = {
#             "text_features": features,
#             "embeddings": embeddings.tolist()  # Convert NumPy array to a list
#         }

#         return jsonify(combined_features)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/analyze-webpage', methods=['POST'])
def analyze_webpage():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Step 1: Extract text content and features from the webpage
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text(separator=' ', strip=True)

        # Clean the text
        clean_text = ' '.join(text_content.split())

        # TF-IDF vectorization
        tfidf_vectorizer = TfidfVectorizer(max_features=7000, stop_words='english')
        tfidf_vectorized = tfidf_vectorizer.fit_transform([clean_text]).toarray()

        # BERT embeddings
        bert_embedding = extract_bert_features(
            [url],
            tokenizer=tokenizer,
            bert_model=bert_model,
            max_length=64,
            batch_size=32
        )

        # Additional features
        additional_features = np.array([
            [
                calculate_entropy(url),
                len(url),
                sum(url.count(c) for c in ['@', '-', '?']),
                url.count('.'),
                sum(c.isdigit() for c in url) / max(1, sum(c.isalpha() for c in url)),
                1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url) else 0,
            ]
        ])

        # Step 2: Combine features into the same format as the model expects
        combined_features = np.hstack([tfidf_vectorized, bert_embedding, additional_features])

        # Step 3: Make predictions using the trained model
        prediction = model.predict(combined_features)[0]
        probability = model.predict_proba(combined_features)[0][1]

        # Step 4: Generate an explanation using SHAP
        shap_explainer_rf = shap.TreeExplainer(model.named_estimators_['rf'])
        shap_values_rf = shap_explainer_rf.shap_values(combined_features)
        shap_values_rf_class = shap_values_rf[..., 1]
        feature_names = generate_feature_names(
            num_features=combined_features.shape[1],
            tfidf_count=7000,
            bert_count=768,
            additional_features=['Entropy', 'URL Length', 'Special Char Count', 'Subdomain Count', 'Num-Letter Ratio', 'Has IP']
        )
        explanation = generate_text_explanation(shap_values_rf_class[0], feature_names)

        # Return the results
        return jsonify({
            "url": url,
            "prediction": int(prediction),
            "probability": float(probability),
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/scan-email', methods=['POST'])
# def scan_email():
#     data = request.json
#     links = data.get('links', [])

#     if not links:
#         return jsonify({"error": "No links provided"}), 400

#     results = []
#     for url in links:
#         try:
#             # Preprocess the URL
#             features = preprocess_urls([url])  # Pass as list for consistency

#             # Make Predictions
#             prediction = model.predict(features)[0]
#             probability = model.predict_proba(features)[0, 1]

#             # Generate Explanations
#             shap_explainer_rf = shap.TreeExplainer(model.named_estimators_['rf'])
#             shap_values_rf = shap_explainer_rf.shap_values(features)
#             shap_values_rf_class = shap_values_rf[..., 1]  # Focus on phishing class
#             feature_names = generate_feature_names(
#                 num_features=features.shape[1],
#                 tfidf_count=7000,
#                 bert_count=768,
#                 additional_features=['Entropy', 'URL Length', 'Special Char Count', 'Subdomain Count', 'Num-Letter Ratio', 'Has IP']
#             )
#             explanation = generate_text_explanation(shap_values_rf_class[0], feature_names)

#             results.append({
#                 "url": url,
#                 "prediction": int(prediction),
#                 "probability": float(probability),
#                 "explanation": explanation
#             })

#         except Exception as e:
#             print(f"Error processing URL {url}: {e}")
#             results.append({
#                 "url": url,
#                 "error": str(e)
#             })

#     return jsonify(results)

@app.route('/scan-email', methods=['POST'])
def scan_email():
    try:
        # Step 1: Authenticate and create credentials.json if not already available
        if not os.path.exists('credentials.json'):
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_id.json', SCOPES
            )  # Ensure `client_id.json` is in your working directory
            creds = flow.run_local_server(port=0)  # Start OAuth flow in a browser
            with open('credentials.json', 'w') as token:
                token.write(creds.to_json())
        
        # Step 2: Extract Links from Emails
        def extract_links_from_emails():
            creds = Credentials.from_authorized_user_file('credentials.json', SCOPES)
            service = build('gmail', 'v1', credentials=creds)

            # Retrieve messages
            results = service.users().messages().list(userId='me', maxResults=10).execute()
            messages = results.get('messages', [])
            
            links = []
            for msg in messages:
                msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
                email_body = msg_data['snippet']
                links.extend(re.findall(r'(https?://\S+)', email_body))
            
            return links

        # Extract links from Gmail
        links = extract_links_from_emails()

        # Step 3: Validate if links were extracted
        if not links:
            return jsonify({"error": "No links found in emails"}), 400

        # Step 4: Process Each Link
        results = []
        for url in links:
            try:
                # Preprocess the URL
                features = preprocess_urls([url])  # Pass as list for consistency

                # Make Predictions
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0, 1]

                # Generate Explanations
                shap_explainer_rf = shap.TreeExplainer(model.named_estimators_['rf'])
                shap_values_rf = shap_explainer_rf.shap_values(features)
                shap_values_rf_class = shap_values_rf[..., 1]  # Focus on phishing class
                feature_names = generate_feature_names(
                    num_features=features.shape[1],
                    tfidf_count=7000,
                    bert_count=768,
                    additional_features=['Entropy', 'URL Length', 'Special Char Count', 'Subdomain Count', 'Num-Letter Ratio', 'Has IP']
                )
                explanation = generate_text_explanation(shap_values_rf_class[0], feature_names)

                # Append results
                results.append({
                    "url": url,
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "explanation": explanation
                })

            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                results.append({
                    "url": url,
                    "error": str(e)
                })

        # Step 5: Return Processed Results
        return jsonify(results)

    except Exception as e:
        print(f"Error in scan-email endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/behavior', methods=['POST'])
def behavior_analysis():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    try:
        report = analyze_url_behavior(url)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
