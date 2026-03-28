# File: main_server.py
#
# --- THIS FILE IS HEAVILY MODIFIED ---
# 1. Keeps the Flask server structure and security wrappers.
# 2. All business logic (handle_prediction, handle_query, handle_action)
#    has been REPLACED with the logic from your new 'server.py'.
# 3. All endpoints (/predict, /query, /action) now route to a
#    single 'process_request_logic' function, just like your new 'server.py'.
# 4. CSV initialization is updated to match the new schema.

import os
import json
import pandas as pd
import time
import base64
import traceback
from flask import Flask, request, jsonify
import uuid

# Custom crypto libraries
import encryption_utils as eu
import paillier_ops as paillier

# --- Configuration ---
app = Flask(__name__)

# --- Server & Key Configuration ---
try:
    APP_SERVER_RSA_PUB_KEY = eu.load_rsa_public_key('app_server_rsa.pub')
    MAIN_SERVER_RSA_PRIV_KEY = eu.load_rsa_private_key('main_server_rsa.priv')
    PAILLIER_PUB_KEY, _ = eu.load_paillier_keys()
    print("🔑 Main Server keys (Main Private, App Public, Paillier Public) loaded.")
except FileNotFoundError as e:
    print(f"❌ ERROR: Key file not found. {e}")
    print("Please run 'encryption_utils.py' to generate all keys first.")
    exit()
except Exception as e:
    print(f"❌ An unexpected error occurred loading keys: {e}")
    exit()

# --- Data File Configuration (from new server.py) ---
DATA_DIR = 'data'
PATIENT_REQUESTS_CSV = os.path.join(DATA_DIR, 'patient_requests.csv')
DOCTOR_REQUESTS_CSV = os.path.join(DATA_DIR, 'doctor_requests.csv')
LAB_REPORTS_CSV = os.path.join(DATA_DIR, 'lab_reports.csv')
PATIENT_MESSAGES_CSV = os.path.join(DATA_DIR, 'patient_messages.csv')

MODEL_PATHS = {
    "Disease Risk Stratification": 'models/disease_risk_model.json',
    "Mental Health Check": 'models/mental_health_model.json',
    "Cancer Check": 'models/cancer_check_model.json',
    "Genomic Disease Susceptibility": 'models/genomic_disease_model.json'
}


# --- Helper Functions (Security Wrappers) ---

def init_csv(filepath, columns):
    """Initializes a CSV file with headers if it doesn't exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        pd.DataFrame(columns=columns).to_csv(filepath, index=False)
        print(f"Initialized {filepath}")


def load_request_payload(flask_request):
    """
    Handles a request from app.py:
    1. Reads plaintext payload (from previous fix).
    2. Verifies signature with app_server's public key.
    3. Returns the trusted payload dictionary.
    """
    try:
        request_data = flask_request.json
        decrypted_payload_json = request_data.get('payload_plaintext')
        signature_b64 = request_data.get('signature')

        if not decrypted_payload_json or not signature_b64:
            return None, {"status": "error", "message": "Security Error: Missing payload or signature."}

        signature = base64.b64decode(signature_b64)
        if not eu.rsa_verify(decrypted_payload_json, signature, APP_SERVER_RSA_PUB_KEY):
            return None, {"status": "error", "message": "Security Error: Invalid signature from App Server."}

        payload = json.loads(decrypted_payload_json)
        return payload, None

    except Exception as e:
        print(f"--- ERROR IN load_request_payload ---")
        print(traceback.format_exc())
        return None, {"status": "error", "message": f"Security Error: Could not process request. {e}"}


def create_signed_response(response_payload):
    """
    Handles a response to app.py:
    1. Dumps payload to JSON string.
    2. Signs string with main_server's private key.
    3. Returns wrapped dictionary with plaintext payload (from previous fix).
    """
    try:
        response_payload_json = json.dumps(response_payload)
        signature = eu.rsa_sign(response_payload_json, MAIN_SERVER_RSA_PRIV_KEY)

        return jsonify({
            "status": "ok",  # Top-level status
            "payload_plaintext": response_payload_json,
            "signature": base64.b64encode(signature).decode('utf-8')
        })

    except Exception as e:
        print(f"--- ERROR IN create_signed_response ---")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": f"Error creating signed response: {e}"})


# --- NEW UNIFIED LOGIC HANDLER (from new server.py) ---

def process_request_logic(request_data):
    """
    This function contains all business logic, ported from 'server.py'.
    It handles all roles and actions.
    """
    try:
        role = request_data.get('role')
        action = request_data.get('action')

        # PATIENT ACTIONS
        if role == 'patient':
            if action == 'request_test':
                df = pd.read_csv(PATIENT_REQUESTS_CSV)
                # Use UUID for request_id to avoid race conditions
                new_id = str(uuid.uuid4())
                new_request = pd.DataFrame([{'request_id': new_id, 'patient_id': request_data['patient_id'],
                                             'patient_username': request_data['patient_username'],
                                             'request_type': request_data['request_type'], 'timestamp': time.time(),
                                             'is_addressed': False}])
                df = pd.concat([df, new_request], ignore_index=True)
                df.to_csv(PATIENT_REQUESTS_CSV, index=False)
                return {"status": "success", "message": "Request submitted successfully."}

            elif action == 'view_messages':
                df = pd.read_csv(PATIENT_MESSAGES_CSV)
                messages = df[df['patient_id'] == request_data['patient_id']].to_dict('records')

                # Convert seconds to milliseconds for JS frontend
                for msg in messages:
                    if 'timestamp' in msg:
                        msg['timestamp'] = msg['timestamp'] * 1000

                return {"status": "success", "data": messages}

        # DOCTOR ACTIONS
        elif role == 'doctor':
            if action == 'view_patient_requests':
                df = pd.read_csv(PATIENT_REQUESTS_CSV)
                pending = df[df['is_addressed'] == False].to_dict('records')

                # Convert seconds to milliseconds for JS frontend
                for req in pending:
                    if 'timestamp' in req:
                        req['timestamp'] = req['timestamp'] * 1000

                return {"status": "success", "data": pending}

            elif action == 'request_lab_work':
                req_id = request_data['patient_request_id']
                df_patient = pd.read_csv(PATIENT_REQUESTS_CSV)

                if req_id not in df_patient['request_id'].values:
                    return {"status": "error", "message": "Patient request ID not found."}

                patient_req_data = df_patient[df_patient['request_id'] == req_id].iloc[0]
                df_patient.loc[df_patient['request_id'] == req_id, 'is_addressed'] = True
                df_patient.to_csv(PATIENT_REQUESTS_CSV, index=False)

                df_doctor = pd.read_csv(DOCTOR_REQUESTS_CSV)
                new_id = str(uuid.uuid4())  # Use UUID
                new_request = pd.DataFrame([{'doctor_request_id': new_id, 'patient_request_id': req_id,
                                             'patient_id': patient_req_data['patient_id'],
                                             'patient_username': patient_req_data['patient_username'],
                                             'request_type': patient_req_data['request_type'],
                                             'doctor_name': request_data['doctor_name'], 'is_complete': False,
                                             'timestamp': time.time()}])
                df_doctor = pd.concat([df_doctor, new_request], ignore_index=True)
                df_doctor.to_csv(DOCTOR_REQUESTS_CSV, index=False)
                return {"status": "success", "message": "Lab work requested."}

            elif action == 'view_lab_reports':
                df = pd.read_csv(LAB_REPORTS_CSV)
                data = df.to_dict('records')

                # Convert seconds to milliseconds for JS frontend
                for report in data:
                    if 'timestamp' in report:
                        report['timestamp'] = report['timestamp'] * 1000

                return {"status": "success", "data": data}

            elif action == 'send_message':
                new_message = pd.DataFrame([{'patient_id': request_data['patient_id'],
                                             'doctor_name': request_data['doctor_name'],
                                             'message': request_data['message'],
                                             'timestamp': time.time()}])
                pd.concat([pd.read_csv(PATIENT_MESSAGES_CSV), new_message]).to_csv(PATIENT_MESSAGES_CSV, index=False)
                return {"status": "success", "message": "Message sent."}

        # LAB ASSISTANT ACTIONS
        elif role == 'lab_assistant':
            if action == 'view_doctor_requests':
                df = pd.read_csv(DOCTOR_REQUESTS_CSV)
                pending = df[df['is_complete'] == False].to_dict('records')

                # Convert seconds to milliseconds for JS frontend
                for req in pending:
                    if 'timestamp' in req:
                        req['timestamp'] = req['timestamp'] * 1000

                return {"status": "success", "data": pending}

            elif action == 'upload_lab_results':
                doc_req_id = request_data['doctor_request_id']
                df_doctor = pd.read_csv(DOCTOR_REQUESTS_CSV)

                if doc_req_id not in df_doctor['doctor_request_id'].values:
                    return {"status": "error", "message": "Doctor request ID not found."}

                # This is the ML Prediction step
                deserialized_data = {k: paillier.deserialize_encrypted_number(PAILLIER_PUB_KEY, v) for k, v in
                                     request_data['paillier_data'].items()}
                encrypted_result = paillier.perform_prediction(deserialized_data,
                                                               MODEL_PATHS[request_data['request_type']])

                # Mark as complete
                df_doctor.loc[df_doctor['doctor_request_id'] == doc_req_id, 'is_complete'] = True
                df_doctor.to_csv(DOCTOR_REQUESTS_CSV, index=False)

                # Save the encrypted result
                patient_id = df_doctor[df_doctor['doctor_request_id'] == doc_req_id].iloc[0]['patient_id']
                lab_report_id = str(uuid.uuid4())
                new_report = pd.DataFrame([{"lab_report_id": lab_report_id, "doctor_request_id": doc_req_id,
                                            "patient_id": patient_id, "request_type": request_data['request_type'],
                                            "lab_assistant_name": request_data['lab_assistant_name'],
                                            "result_encrypted": json.dumps(
                                                paillier.serialize_encrypted_number(encrypted_result)),
                                            'timestamp': time.time()}])
                pd.concat([pd.read_csv(LAB_REPORTS_CSV), new_report]).to_csv(LAB_REPORTS_CSV, index=False)
                return {"status": "success", "message": "Lab results uploaded securely."}

        return {"status": "error", "message": "Invalid role or action."}

    except Exception as e:
        print(f"--- ERROR IN process_request_logic ---")
        print(traceback.format_exc())
        return {"status": "error", "message": f"An unexpected server error occurred: {e}"}


# --- Main API Endpoints ---
# All endpoints now route to the same logic function,
# which distinguishes based on the payload's 'role' and 'action'.

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    payload, error = load_request_payload(request)
    if error:
        return jsonify(error), 401

    # This endpoint is now used for 'upload_lab_results'
    response_payload = process_request_logic(payload)
    return create_signed_response(response_payload)


@app.route('/query', methods=['POST'])
def query_endpoint():
    payload, error = load_request_payload(request)
    if error:
        return jsonify(error), 401

    # Used for 'view_messages', 'view_patient_requests', 'view_lab_reports', etc.
    response_payload = process_request_logic(payload)
    return create_signed_response(response_payload)


@app.route('/action', methods=['POST'])
def action_endpoint():
    payload, error = load_request_payload(request)
    if error:
        return jsonify(error), 401

    # Used for 'request_test', 'request_lab_work', 'send_message'
    response_payload = process_request_logic(payload)
    return create_signed_response(response_payload)


# --- Main Runner ---
if __name__ == '__main__':
    # Initialize all data CSVs (from new server.py)
    init_csv(PATIENT_REQUESTS_CSV,
             ['request_id', 'patient_id', 'patient_username', 'request_type', 'timestamp', 'is_addressed'])
    init_csv(DOCTOR_REQUESTS_CSV,
             ['doctor_request_id', 'patient_request_id', 'patient_id', 'patient_username', 'request_type',
              'doctor_name', 'is_complete', 'timestamp'])
    init_csv(LAB_REPORTS_CSV, ['lab_report_id', 'doctor_request_id', 'patient_id', 'request_type', 'lab_assistant_name',
                               'result_encrypted', 'timestamp'])
    init_csv(PATIENT_MESSAGES_CSV, ['patient_id', 'doctor_name', 'message', 'timestamp'])  # <-- 'timestamp' ADDED

    print("--- 🚀 Starting Secure Main Resource Server on http://127.0.0.1:5001 ---")
    app.run(host='127.0.0.1', port=5001, debug=False)