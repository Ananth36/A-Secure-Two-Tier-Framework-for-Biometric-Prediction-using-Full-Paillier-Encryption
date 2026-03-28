# File: train_models.py

import os
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
import encryption_utils as eu
import paillier_ops as paillier

# --- Configuration ---
ENCRYPTED_DATA_DIR = "encrypted_data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load Paillier Private Key for Decryption ---
try:
    paillier_pub_key, paillier_priv_key = eu.load_paillier_keys()
    print("🔑 Paillier keys loaded for training.")
except FileNotFoundError:
    print("❌ ERROR: Paillier keys not found. Please run 'encryption_utils.py' first.")
    exit()


def save_weights(model, feature_names, filepath):
    """Saves model weights and intercept to a JSON file."""
    weights = {
        "intercept": float(model.intercept_[0]),
        "coefficients": {name: float(coef) for name, coef in zip(feature_names, model.coef_[0])}
    }
    with open(filepath, 'w') as f:
        json.dump(weights, f, indent=4)
    print(f"✅ Model weights saved to '{filepath}'")


def decrypt_records_to_df(encrypted_records, features, target_col):
    """Helper function to decrypt records into a pandas DataFrame."""
    decrypted_data = []
    for record in encrypted_records:
        decrypted_row = {}
        try:
            for feature in features:
                if feature not in record:
                    raise KeyError(f"Missing feature: {feature}")
                enc_num = paillier.deserialize_encrypted_number(paillier_pub_key, record[feature])
                decrypted_row[feature] = paillier_priv_key.decrypt(enc_num)
            decrypted_row[target_col] = record[target_col]
            decrypted_data.append(decrypted_row)
        except (KeyError, TypeError):
            continue  # Skip malformed records
    return pd.DataFrame(decrypted_data)


def train_model_from_file(filepath, features, target_col, model_save_path, model_name):
    """Generic function to train a model from an encrypted file."""
    print(f"\n--- Training {model_name} ---")
    try:
        with open(filepath, 'r') as f:
            encrypted_records = json.load(f)

        df = decrypt_records_to_df(encrypted_records, features, target_col)
        if df.empty:
            print(f"❌ ERROR: No valid data to train on for {model_name}. Please check the encryption script.")
            return

        X = df[features]
        y = df[target_col]

        print("   Training model...")
        model = LogisticRegression(max_iter=1000, solver='liblinear')
        model.fit(X, y)
        save_weights(model, features, model_save_path)
    except FileNotFoundError:
        print(f"❌ ERROR: '{os.path.basename(filepath)}' not found. Please run 'encrypt_datasets.py' first.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    print("---  Starting Model Training from Encrypted Sources ---")
    print("NOTE: Data is decrypted in this simulated trusted environment for training.")

    # --- HEART DISEASE ---
    train_model_from_file(
        os.path.join(ENCRYPTED_DATA_DIR, 'encrypted_heart_disease.json'),
        ['age', 'sex', 'trestbps', 'chol'], 'target',
        os.path.join(MODEL_DIR, "disease_risk_model.json"),
        "Disease Risk Model"
    )

    # --- UPDATED: MENTAL HEALTH MODEL ---
    train_model_from_file(
        os.path.join(ENCRYPTED_DATA_DIR, 'encrypted_mental_health.json'),
        ['age', 'work_stress', 'sleep_hours', 'family_support'],  # ✅ Updated feature names
        'mental_illness',  # ✅ Updated target
        os.path.join(MODEL_DIR, "mental_health_model.json"),
        "Mental Health Model"
    )

    # --- BREAST CANCER ---
    train_model_from_file(
        os.path.join(ENCRYPTED_DATA_DIR, 'encrypted_cancer_check.json'),
        ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'], 'diagnosis',
        os.path.join(MODEL_DIR, "cancer_check_model.json"),
        "Cancer Check Model"
    )

    # --- FRAMINGHAM GENOMIC MODEL ---
    train_model_from_file(
        os.path.join(ENCRYPTED_DATA_DIR, 'encrypted_framingham.json'),
        ['totChol', 'sysBP', 'glucose', 'age'], 'TenYearCHD',
        os.path.join(MODEL_DIR, "genomic_disease_model.json"),
        "Genomic Susceptibility Model"
    )

    print("\n✅ All models have been trained.")
