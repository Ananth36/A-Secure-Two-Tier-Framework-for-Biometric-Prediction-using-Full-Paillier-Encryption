# File: encrypt_datasets.py

import os
import pandas as pd
import json
import encryption_utils as eu
import paillier_ops 


RAW_DATA_DIR = "./datasets"  
ENCRYPTED_DATA_DIR = "encrypted_data"
os.makedirs(ENCRYPTED_DATA_DIR, exist_ok=True)

try:
    paillier_pub_key = eu.load_paillier_keys()[0]
    print(" Paillier public key loaded.")
except FileNotFoundError:
    print(" ERROR: Paillier keys not found. Please run 'encryption_utils.py' first.")
    exit()


def encrypt_and_save_dataframe(df, features_to_encrypt, target_column, output_filename):
    """Encrypts specified columns of a DataFrame and saves it as JSON."""
    print(f"   Encrypting '{output_filename}'...")

    df_cleaned = df[features_to_encrypt + [target_column]].dropna()
    print(f"   Found {len(df_cleaned)} valid records to encrypt.")

    if len(df_cleaned) == 0:
        print(f"    WARNING: No data. Skipping file generation.")
        return

    encrypted_records = []
    for _, row in df_cleaned.iterrows():
        encrypted_row = {
            feature: paillier_ops.serialize_encrypted_number(
                paillier_pub_key.encrypt(float(row[feature]))
            )
            for feature in features_to_encrypt
        }
        encrypted_row[target_column] = int(row[target_column])
        encrypted_records.append(encrypted_row)

    filepath = os.path.join(ENCRYPTED_DATA_DIR, output_filename)
    with open(filepath, 'w') as f:
        json.dump(encrypted_records, f, indent=4)
    print(f" Data saved to '{filepath}'")


def process_heart_disease():
    print("\n--- Processing Heart Disease Dataset ---")
    try:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'heart_disease_dataset.csv'))
        df['sex'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)
        df.rename(columns={'num': 'target'}, inplace=True)
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        features = ['age', 'sex', 'trestbps', 'chol']
        encrypt_and_save_dataframe(df, features, 'target', 'encrypted_heart_disease.json')
    except FileNotFoundError:
        print(" ERROR: 'heart_disease_dataset.csv' not found.")


def process_mental_health():
    print("\n--- Processing Mental Health Survey (Dummy) ---")
    try:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'mental_health_survey.csv'))

        # Convert categorical columns
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})

        # Target: mental_illness (already 0 or 1)
        if 'mental_illness' not in df.columns:
            raise KeyError("Expected 'mental_illness' column not found in dataset.")

        # Choose features to encrypt (you can add more if needed)
        features = ['age', 'work_stress', 'sleep_hours', 'family_support']

        encrypt_and_save_dataframe(df, features, 'mental_illness', 'encrypted_mental_health.json')

    except FileNotFoundError:
        print(" ERROR: 'mental_health_survey.csv' not found.")
    except KeyError as e:
        print(f" ERROR: Missing column in mental_health_survey.csv — {e}")


def process_cancer_check():
    print("\n--- Processing Breast Cancer Dataset ---")
    try:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'breast_cancer_data.csv'))
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
        encrypt_and_save_dataframe(df, features, 'diagnosis', 'encrypted_cancer_check.json')
    except FileNotFoundError:
        print(" ERROR: 'breast_cancer_data.csv' not found.")


def process_framingham():
    print("\n--- Processing Framingham Dataset (Genomic) ---")
    try:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'framingham.csv'))
        features = ['totChol', 'sysBP', 'glucose', 'age']
        encrypt_and_save_dataframe(df, features, 'TenYearCHD', 'encrypted_framingham.json')
    except FileNotFoundError:
        print(" ERROR: 'framingham.csv' not found.")


if __name__ == "__main__":
    print("---  Starting Dataset Encryption Process ---")
    process_heart_disease()
    process_mental_health()
    process_cancer_check()
    process_framingham()
    print("\n All datasets have been encrypted and stored in the 'encrypted_data/' directory.")
