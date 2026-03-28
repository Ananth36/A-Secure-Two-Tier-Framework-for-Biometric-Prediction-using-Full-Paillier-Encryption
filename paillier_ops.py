# File: paillier_ops.py

import json
from phe import paillier
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-1*x))

def perform_prediction(encrypted_data, model_weights_path):
    """
    Performs logistic regression prediction on Paillier-encrypted data.
    This simulates the secure server-side computation.
    """

    try:
        with open(model_weights_path, 'r') as f:
            weights = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Model weights not found at {model_weights_path}")

    intercept = weights['intercept']
    coefficients = weights['coefficients']

    # Homomorphically compute the dot product of encrypted data and plaintext weights
    log_odds = intercept
    for feature, coeff in coefficients.items():
        if feature in encrypted_data:
            log_odds += encrypted_data[feature] * coeff

    # The server returns the encrypted log-odds.
    # The client will decrypt this to get the final result.
    return log_odds


def serialize_encrypted_number(enc_num):
    """Converts a Paillier EncryptedNumber to a serializable dictionary."""
    return {'ciphertext': str(enc_num.ciphertext()), 'exponent': enc_num.exponent}

def deserialize_encrypted_number(pub_key, enc_dict):
    """Converts a dictionary back into a Paillier EncryptedNumber."""
    return paillier.EncryptedNumber(
        pub_key,
        int(enc_dict['ciphertext']),
        enc_dict['exponent']
    )
