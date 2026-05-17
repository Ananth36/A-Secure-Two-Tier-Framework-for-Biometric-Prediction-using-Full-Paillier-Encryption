# File: encryption_utils.py

import os
import json
import pandas as pd
import uuid
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric.utils import Prehashed
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from phe import paillier
from werkzeug.security import generate_password_hash, check_password_hash


KEYS_DIR = 'keys'
DATA_DIR = 'data'  
USER_CSV = os.path.join(DATA_DIR, 'users.csv')  
RSA_KEY_SIZE = 2048
PAILLIER_KEY_SIZE = 1024


os.makedirs(KEYS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)




def hash_password(password):
    """Hashes a password using SHA256."""
    return generate_password_hash(password, method='pbkdf2:sha256')


def init_user_csv(filepath):
    """Creates the user CSV file if it doesn't exist."""
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=["user_id", "username", "password_hash", "role"])
        df.to_csv(filepath, index=False)
        print(f"User CSV created at {filepath}")


def add_user_to_csv(filepath, username, hashed_password, role):
    """
    Adds a new user to the CSV file and returns their new user_id.
    """
    user_id = str(uuid.uuid4())

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        df = pd.DataFrame(columns=["user_id", "username", "password_hash", "role"])

    if not df[df['username'].str.lower() == username.lower()].empty:
        raise ValueError("Username already exists")

    new_user = pd.DataFrame({
        "user_id": [user_id],
        "username": [username],
        "password_hash": [hashed_password],
        "role": [role]
    })
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(filepath, index=False)
    return user_id


def get_user_from_csv(filepath, username):
    """Finds a user by username (case-insensitive) and returns their data as a dict."""
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    user = df[df['username'].str.lower() == username.lower()]
    if not user.empty:
        return user.to_dict('records')[0]
    return None




def generate_and_save_rsa_keys(name):
    """Generates an RSA key pair and saves it to the keys directory."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=RSA_KEY_SIZE,
        backend=default_backend()
    )
    public_key = private_key.public_key()

    # Save Private Key
    pem_priv = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    with open(os.path.join(KEYS_DIR, f'{name}_rsa.priv'), 'wb') as f:
        f.write(pem_priv)

    # Save Public Key
    pem_pub = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    with open(os.path.join(KEYS_DIR, f'{name}_rsa.pub'), 'wb') as f:
        f.write(pem_pub)
    print(f" Generated RSA keys for '{name}'.")


def generate_and_save_paillier_keys():
    """Generates a Paillier key pair and saves it."""
    public_key, private_key = paillier.generate_paillier_keypair(n_length=PAILLIER_KEY_SIZE)

    keys = {
        'public_key': {'n': str(public_key.n)},
        'private_key': {'p': str(private_key.p), 'q': str(private_key.q)}
    }
    with open(os.path.join(KEYS_DIR, 'paillier_keys.json'), 'w') as f:
        json.dump(keys, f, indent=4)
    print(" Generated Paillier keys.")




def load_rsa_private_key(name):
    """Loads an RSA private key from the keys directory."""
    filepath = os.path.join(KEYS_DIR, name)
    with open(filepath, 'rb') as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend()
        )
    return private_key


def load_rsa_public_key(name):
    """Loads an RSA public key from the keys directory."""
    filepath = os.path.join(KEYS_DIR, name)
    with open(filepath, 'rb') as f:
        public_key = serialization.load_pem_public_key(
            f.read(),
            backend=default_backend()
        )
    return public_key


def load_rsa_public_key_str(name):
    """Loads an RSA public key and returns it as a PEM string."""
    filepath = os.path.join(KEYS_DIR, name)
    with open(filepath, 'rb') as f:
        return f.read().decode('utf-8')


def load_paillier_keys():
    """Loads Paillier keys from the JSON file."""
    filepath = os.path.join(KEYS_DIR, 'paillier_keys.json')
    with open(filepath, 'r') as f:
        keys = json.load(f)
        pub_n = int(keys['public_key']['n'])
        priv_p = int(keys['private_key']['p'])
        priv_q = int(keys['private_key']['q'])

        public_key = paillier.PaillierPublicKey(n=pub_n)
        private_key = paillier.PaillierPrivateKey(public_key, p=priv_p, q=priv_q)
        return public_key, private_key




def rsa_encrypt(message_bytes, public_key):
    """Encrypts a message (bytes) using an RSA public key."""
    return public_key.encrypt(
        message_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )


def rsa_decrypt(ciphertext, private_key):
    """Decrypts a message (bytes) using an RSA private key."""
    return private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )


def rsa_sign(message_str, private_key):
    """Signs a string message with an RSA private key."""
    message_bytes = message_str.encode('utf-8')
    digest = hashes.Hash(hashes.SHA256(), default_backend())
    digest.update(message_bytes)
    hashed_message = digest.finalize()

    return private_key.sign(
        hashed_message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        Prehashed(hashes.SHA256())
    )


def rsa_verify(message_str, signature, public_key):
    """Verifies a signature for a string message with an RSA public key."""
    message_bytes = message_str.encode('utf-8')
    digest = hashes.Hash(hashes.SHA256(), default_backend())
    digest.update(message_bytes)
    hashed_message = digest.finalize()

    try:
        public_key.verify(
            signature,
            hashed_message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            Prehashed(hashes.SHA256())
        )
        return True
    except InvalidSignature:
        return False
    except Exception:
        return False


# ===============================================
# ==         MAIN (for initial setup)          ==
# ===============================================

if __name__ == "__main__":
    print("---  Generating and saving all cryptographic keys... ---")

    # Note: Directories are created at the top of the file

    # Generate keys for the two servers (Flask Project)
    generate_and_save_rsa_keys("app_server")
    generate_and_save_rsa_keys("main_server")
    print(" RSA keys for app and main servers generated.")

    # Generate Paillier keys for homomorphic encryption
    generate_and_save_paillier_keys()
    print(" Paillier keys for homomorphic encryption generated.")

    # Initialize the user database
    init_user_csv(USER_CSV)
    print(" User database initialized.")

    print("\n---  All keys and databases initialized successfully. ---")
    print(f"Keys are saved in the '{KEYS_DIR}' directory.")
    print(f"User CSV is at '{USER_CSV}'.")
