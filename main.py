import logging
import numpy as np
import joblib
from transformers import BertTokenizer, TFBertModel
from src.model_training import preprocess_data, train_ensemble_model, explain_model_shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    try:
        # Step 1: Load BERT Model and Tokenizer
        logging.info("Loading BERT model and tokenizer...")
        bert_model = TFBertModel.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        logging.info("BERT model and tokenizer loaded successfully.")

        # Step 2: Preprocess Data
        logging.info("Preprocessing data...")
        X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = preprocess_data(
            "data/augmented_dataset1.csv", bert_model, tokenizer, max_length=64, batch_size=256
        )
        logging.info("Data preprocessing completed.")

        # Step 3: Train the Ensemble Model
        logging.info("Training the ensemble model...")
        model = train_ensemble_model(X_train, y_train)
        logging.info("Model training completed.")

        # Step 4: Evaluate the Model
        logging.info("Evaluating the model on the test set...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability for the positive class
        optimal_threshold = 0.45
        y_pred = (y_pred_proba > optimal_threshold).astype(int)

        logging.info("\nTest Set Performance:")
        logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logging.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
        logging.info(f"Recall: {recall_score(y_test, y_pred):.4f}")
        logging.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        logging.info(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

        # Step 5: Save the Model and Artifacts
        logging.info("\nSaving the trained model and artifacts...")
        joblib.dump(model, "Models/phishing_detection_model.pkl")
        joblib.dump(vectorizer, "Models/tfidf_vectorizer.pkl")
        with open("Models/optimal_threshold.txt", "w") as f:
            f.write(str(optimal_threshold))
        logging.info("Model, vectorizer, and threshold saved successfully.")

        # Step 6: Generate SHAP Explanations
        logging.info("\nGenerating SHAP explanations for the model...")
        feature_names = vectorizer.get_feature_names_out().tolist() + [
            "Entropy", "URL Length", "Special Char Count", "Subdomain Count", "Num-Letter Ratio", "Has IP"
        ]
        explain_model_shap(model, X_test[:100].toarray(), feature_names)
        logging.info("SHAP explanations generated successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
