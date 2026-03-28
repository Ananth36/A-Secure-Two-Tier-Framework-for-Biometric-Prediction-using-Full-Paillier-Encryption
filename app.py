# File: app.py
#
# --- THIS FILE IS HEAVILY MODIFIED ---
# 1. All API handlers (e.g., /api/patient_request) have been
#    REWRITTEN to match the logic of your new 'hosp_app.py'.
# 2. Patient 'submit_request' now only logs a request (no prediction).
# 3. Patient 'get_patient_data' now only fetches messages, not results.
# 4. Doctor 'get_doctor_data' is split to fetch requests and reports.
# 5. Doctor dashboard now decrypts and shows results, mimicking 'hosp_app.py'.
# 6. Lab Assistant 'lab_upload_results' is now the *only* function
#    that triggers a prediction.
# 7. All auth, 2FA, and security from the original 'app.py' is kept.

import os
import json
import base64
import pandas as pd
import numpy as np
import cv2
import requests
import encryption_utils as eu
import paillier_ops as paillier
import math
import traceback
import binascii
from flask import Flask, request, jsonify, session, send_from_directory
from flask_session import Session
from werkzeug.security import check_password_hash

# --- NEW IMPORTS for Python 3.12 Compatibility ---
from deepface import DeepFace  # Replaces face_recognition
import shutil  # Used for temporary files

# --- App Setup ---
app = Flask(__name__, static_folder='.', static_url_path='')
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = os.urandom(24)
Session(app)

# --- Configuration ---
DATA_DIR = 'data'
FACE_DATA_DIR = os.path.join(DATA_DIR, 'face_images')
USER_CSV = os.path.join(DATA_DIR, 'users.csv')
MAIN_SERVER_URL = "http://127.0.0.1:5001"

# --- Create Directories ---
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FACE_DATA_DIR, exist_ok=True)
eu.init_user_csv(USER_CSV)

# --- Load App Server's Keys ---
try:
    APP_RSA_PRIV_KEY = eu.load_rsa_private_key('app_server_rsa.priv')
    APP_RSA_PUB_KEY_STR = eu.load_rsa_public_key_str('app_server_rsa.pub')
    MAIN_SERVER_RSA_PUB_KEY = eu.load_rsa_public_key('main_server_rsa.pub')
    PAILLIER_PUB_KEY, PAILLIER_PRIV_KEY = eu.load_paillier_keys()
    print("🔑 App Server keys loaded successfully.")
except FileNotFoundError:
    print("❌ CRITICAL: Keys not found. Please run 'encryption_utils.py' first.")
    exit()

# --- Warm up DeepFace ---
try:
    print("🚀 Initializing Face Recognition Engine (DeepFace)...")
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    DeepFace.analyze(dummy_image, actions=['age'], enforce_detection=False)
    print("✅ Face Recognition Engine ready.")
except Exception as e:
    print(f"Warning: Could not pre-initialize DeepFace: {e}")

# --- Data Schemas (MODIFIED) ---
DATA_REQUIREMENTS = {
    "Disease Risk Stratification": ['age', 'sex', 'trestbps', 'chol'],
    "Mental Health Check": ['age', 'work_stress', 'sleep_hours', 'family_support'],  # <-- MODIFIED
    "Cancer Check": ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'],
    "Genomic Disease Susceptibility": ['totChol', 'sysBP', 'glucose', 'age']
}
PROMPTS = {
    # General / Disease Risk / Genomic
    'age': "Enter age",
    'sex': "Enter sex (0=F, 1=M)",
    'trestbps': "Enter resting BP (e.g., 120)",
    'chol': "Enter cholesterol (e.g., 200)",
    'totChol': "Enter total cholesterol",
    'sysBP': "Enter systolic BP",
    'glucose': "Enter glucose",

    # Cancer Check
    'radius_mean': "Enter radius_mean",
    'texture_mean': "Enter texture_mean",
    'perimeter_mean': "Enter perimeter_mean",
    'area_mean': "Enter area_mean",

    # NEW 4-Feature Mental Health Check
    'work_stress': "Work Stress (0-10)",
    'sleep_hours': "Avg. sleep hours per night",
    'family_support': "Family Support (0-10)"

    # Removed old 'Age' and 'Gender' prompts
}


# ===============================================
# ==         HELPER FUNCTIONS                  ==
# ===============================================

def send_signed_request(endpoint, payload):
    """
    Signs a request and sends it to the main server.
    (Uses plaintext payload + signature, from previous fix)
    """
    try:
        payload['app_server_timestamp'] = pd.Timestamp.now().isoformat()
        payload_json = json.dumps(payload)
        signature = eu.rsa_sign(payload_json, APP_RSA_PRIV_KEY)

        request_data = {
            'payload_plaintext': payload_json,
            'signature': base64.b64encode(signature).decode('utf-8'),
            'app_server_pub_key_str': APP_RSA_PUB_KEY_STR
        }

        response = requests.post(f"{MAIN_SERVER_URL}{endpoint}", json=request_data)
        response.raise_for_status()

        response_data = response.json()
        if response_data['status'] == 'error':
            return response_data

        decrypted_payload_json = response_data['payload_plaintext']
        resp_signature = base64.b64decode(response_data['signature'])

        if not eu.rsa_verify(decrypted_payload_json, resp_signature, MAIN_SERVER_RSA_PUB_KEY):
            return {"status": "error", "message": "Security Error: Could not verify main server's signature."}

        # This returns the *inner* payload, e.g., {"status": "success", ...}
        return json.loads(decrypted_payload_json)

    except requests.exceptions.ConnectionError as e:
        print(f"Error connecting to main server: {e}")
        return {"status": "error", "message": f"Cannot connect to main server (port 5001). Is it running?"}
    except Exception as e:
        print(f"--- ERROR IN send_signed_request ---")
        print(traceback.format_exc())
        return {"status": "error", "message": f"An internal error occurred: {e}"}


def sigmoid(x):
    """Compute sigmoid function (Unchanged)"""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# ===============================================
# ==         MAIN WEB ROUTES (Auth)            ==
# ===============================================
# (These routes are unchanged and handle auth)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role')
    image_data_url = data.get('image')

    if not all([username, password, role, image_data_url]):
        return jsonify({"status": "error", "message": "All fields are required."}), 400

    if eu.get_user_from_csv(USER_CSV, username):
        return jsonify({"status": "error", "message": "Username already exists."}), 400

    try:
        header, encoded = image_data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        image_np = np.frombuffer(image_data, np.uint8)
        image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image_cv is None:
            return jsonify({"status": "error", "message": "Could not decode image."}), 400

        try:
            DeepFace.detectFace(image_cv, enforce_detection=True)
        except ValueError:
            return jsonify({"status": "error", "message": "No face found in the image. Please try again."}), 400

        hashed_password = eu.hash_password(password)
        user_id = eu.add_user_to_csv(USER_CSV, username, hashed_password, role)

        image_filename = f"user_{user_id}.jpg"
        image_path = os.path.join(FACE_DATA_DIR, image_filename)

        with open(image_path, 'wb') as f:
            f.write(image_data)

        return jsonify({"status": "success", "message": "Registration successful. Please log in."})

    except (binascii.Error, TypeError, ValueError) as e:
        print(f"Image decoding error: {e}")
        return jsonify({"status": "error", "message": "Invalid image data format."}), 400
    except Exception as e:
        print(f"--- REGISTRATION CRASH ---")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": f"An internal server error occurred: {e}"}), 500


@app.route('/api/login', methods=['POST'])
def login_step1():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = eu.get_user_from_csv(USER_CSV, username)
    if user and check_password_hash(user['password_hash'], password):
        session['2fa_user_id'] = user['user_id']
        session['2fa_username'] = user['username']
        session['2fa_role'] = user['role']
        return jsonify({"status": "2fa_required", "message": "Password OK. Proceed to 2FA."})

    return jsonify({"status": "error", "message": "Invalid username or password."}), 401


@app.route('/api/2fa', methods=['POST'])
def login_step2():
    if '2fa_user_id' not in session:
        return jsonify({"status": "error", "message": "Session expired. Please log in again."}), 401

    image_data_url = request.json.get('image')
    if not image_data_url:
        return jsonify({"status": "error", "message": "No image data received."}), 400

    temp_image_path = None
    try:
        user_id = session['2fa_user_id']
        stored_image_path = os.path.join(FACE_DATA_DIR, f"user_{user_id}.jpg")

        if not os.path.exists(stored_image_path):
            return jsonify({"status": "error", "message": "No face encoding found for this user."}), 500

        header, encoded = image_data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        temp_image_path = os.path.join(DATA_DIR, f"temp_2fa_{user_id}.jpg")

        with open(temp_image_path, 'wb') as f:
            f.write(image_data)

        verification_result = DeepFace.verify(
            img1_path=stored_image_path,
            img2_path=temp_image_path,
            enforce_detection=False
        )

        if verification_result['verified']:
            user_id = session.pop('2fa_user_id')
            session['user_id'] = user_id
            session['username'] = session.pop('2fa_username')
            session['role'] = session.pop('2fa_role')

            return jsonify({
                "status": "ok",
                "message": "Login successful!",
                "username": session['username'],
                "role": session['role']
            })
        else:
            return jsonify({"status": "error", "message": "Face does not match. Please try again."}), 401

    except Exception as e:
        print(f"--- 2FA CRASH ---")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": f"An internal error occurred during 2FA: {e}"}), 500
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"status": "ok", "message": "Logged out."})


@app.route('/api/check_session', methods=['GET'])
def session_check():
    if 'user_id' in session:
        return jsonify({
            "status": "ok",
            "username": session['username'],
            "role": session['role']
        })
    return jsonify({"status": "error", "message": "No active session"}), 401


@app.route('/api/get_schema', methods=['GET'])
def get_prompts():
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Not authenticated"}), 401

    return jsonify({
        "status": "ok",
        "prompts": PROMPTS,
        "requirements": DATA_REQUIREMENTS
    })


# ===============================================
# ==     API ENDPOINTS (NEW LOGIC)             ==
# ===============================================

@app.route('/api/patient_request', methods=['POST'])
def submit_request():
    """
    Patient: Submits a new request for a test.
    (NEW LOGIC: This just logs the request, no prediction)
    """
    if 'user_id' not in session or session['role'] != 'patient':
        return jsonify({"status": "error", "message": "Not authenticated"}), 401

    data = request.json
    request_type_name = data.get('request_type')
    if not request_type_name:
        return jsonify({"status": "error", "message": "Missing request type."}), 400

    try:
        payload = {
            "role": "patient",
            "action": "request_test",
            "patient_id": session['user_id'],
            "patient_username": session['username'],
            "request_type": request_type_name
        }
        # Use '/action' endpoint as this is a DB write
        response = send_signed_request("/action", payload)

        # Pass the server's response directly to the client
        return jsonify(response)

    except Exception as e:
        print(f"--- SUBMIT REQUEST ERROR ---")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": f"An error occurred: {e}"}), 500


@app.route('/api/get-patient-data', methods=['GET'])
def get_patient_data():
    """
    Patient: Gets their messages.
    (NEW LOGIC: Patients do not see results, only messages)
    """
    if 'user_id' not in session or session['role'] != 'patient':
        return jsonify({"status": "error", "message": "Not authenticated"}), 401

    payload = {
        "role": "patient",
        "action": "view_messages",
        "patient_id": session['user_id']
    }
    # Use '/query' endpoint as this is a DB read
    response = send_signed_request("/query", payload)

    if response.get('status') == 'success':
        # Send an empty results array to satisfy index.html
        return jsonify({
            "status": "ok",
            "results": [],
            "messages": response.get("data", [])
        })
    else:
        return jsonify(response)


@app.route('/api/doctor_view_requests', methods=['GET'])
def get_doctor_requests():
    """Doctor: Gets all pending patient requests."""
    if 'user_id' not in session or session['role'] != 'doctor':
        return jsonify({"status": "error", "message": "Not authenticated"}), 401

    payload = {
        "role": "doctor",
        "action": "view_patient_requests"
    }
    response = send_signed_request("/query", payload)
    return jsonify(response)


@app.route('/api/doctor_view_reports', methods=['GET'])
def get_doctor_reports():
    """
    Doctor: Gets all completed lab reports.
    (NEW LOGIC: Doctor decrypts results here)
    """
    if 'user_id' not in session or session['role'] != 'doctor':
        return jsonify({"status": "error", "message": "Not authenticated"}), 401

    payload = {
        "role": "doctor",
        "action": "view_lab_reports"
    }
    response = send_signed_request("/query", payload)

    # Decrypt the results before sending to frontend
    if response.get('status') == 'success' and 'data' in response:
        decrypted_reports = []
        for report in response['data']:
            try:
                # Decrypt the result using the app's private key
                enc_result = paillier.deserialize_encrypted_number(PAILLIER_PUB_KEY,
                                                                   json.loads(report['result_encrypted']))
                log_odds = PAILLIER_PRIV_KEY.decrypt(enc_result)
                probability = sigmoid(log_odds)

                risk_text = f"High Risk ({probability * 100:.2f}%)" if probability > 0.5 else f"Low Risk ({probability * 100:.2f}%)"

                decrypted_reports.append({
                    "patient_id": report['patient_id'],
                    "request_type": report['request_type'],
                    "lab_assistant_name": report.get('lab_assistant_name', 'N/A'),
                    "timestamp": report.get('timestamp', 'N/A'),
                    "result_text": risk_text
                })
            except Exception as e:
                print(f"Could not decrypt report: {e}")
                decrypted_reports.append({**report, "result_text": "[Error Decrypting]"})

        response['data'] = decrypted_reports

    return jsonify(response)


@app.route('/api/doctor_assign_request', methods=['POST'])
def doctor_approve():
    """Doctor: Approves a patient request, sending it to lab."""
    if 'user_id' not in session or session['role'] != 'doctor':
        return jsonify({"status": "error", "message": "Not authenticated"}), 401

    data = request.json
    request_id = data.get('request_id')
    if not request_id:
        return jsonify({"status": "error", "message": "Request ID is missing."}), 400

    payload = {
        "role": "doctor",
        "action": "request_lab_work",
        "patient_request_id": request_id,
        "doctor_name": session['username']
    }
    response = send_signed_request("/action", payload)
    return jsonify(response)


@app.route('/api/doctor-send-message', methods=['POST'])
def doctor_send_message():
    """Doctor: Sends a message to a patient."""
    if 'user_id' not in session or session['role'] != 'doctor':
        return jsonify({"status": "error", "message": "Not authenticated"}), 401

    data = request.json
    patient_id = data.get('patient_id')
    message_text = data.get('message')
    if not patient_id or not message_text:
        return jsonify({"status": "error", "message": "Patient ID and message are required."}), 400

    payload = {
        "role": "doctor",
        "action": "send_message",
        "patient_id": patient_id,
        "doctor_name": session['username'],
        "message": message_text
    }
    response = send_signed_request("/action", payload)
    return jsonify(response)


@app.route('/api/lab_view_requests', methods=['GET'])
def get_lab_data():
    """Lab Assistant: Gets all approved doctor requests."""
    if 'user_id' not in session or session['role'] != 'lab_assistant':
        return jsonify({"status": "error", "message": "Not authenticated"}), 401

    payload = {
        "role": "lab_assistant",
        "action": "view_doctor_requests"
    }
    response = send_signed_request("/query", payload)
    return jsonify(response)


@app.route('/api/lab_upload_results', methods=['POST'])
def lab_upload_result():
    """
    Lab Assistant: Uploads the final encrypted result.
    (NEW LOGIC: This is the only place prediction happens)
    """
    if 'user_id' not in session or session['role'] != 'lab_assistant':
        return jsonify({"status": "error", "message": "Not authenticated"}), 401

    data = request.json
    doctor_request_id = data.get('doctor_request_id')
    request_type_name = data.get('request_type')
    form_data = data.get('form')

    if not doctor_request_id or not request_type_name or not form_data:
        return jsonify({"status": "error", "message": "Doctor request ID, type, and form data are missing."}), 400

    if request_type_name not in DATA_REQUIREMENTS:
        return jsonify({"status": "error", "message": "Invalid request type."}), 400

    try:
        encrypted_data = {}
        required_features = DATA_REQUIREMENTS[request_type_name]
        for feature in required_features:
            val = float(form_data[feature])
            encrypted_data[feature] = paillier.serialize_encrypted_number(PAILLIER_PUB_KEY.encrypt(val))

        payload = {
            "role": "lab_assistant",
            "action": "upload_lab_results",
            "doctor_request_id": doctor_request_id,
            "request_type": request_type_name,
            "paillier_data": encrypted_data,
            "lab_assistant_name": session['username']
        }
        # This is the only ML-related call, so it uses /predict
        response = send_signed_request("/predict", payload)
        return jsonify(response)

    except ValueError:
        return jsonify({"status": "error", "message": "Invalid input. Please enter numbers only."}), 400
    except Exception as e:
        print(f"--- LAB UPLOAD ERROR ---")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": f"An error occurred: {e}"}), 500


# ===============================================
# ==         RUN THE APP                       ==
# ===============================================

if __name__ == '__main__':
    print("--- 🚀 Starting Auth & App Server on http://127.0.0.1:5000 ---")
    app.run(host='127.0.0.1', port=5000, debug=False)