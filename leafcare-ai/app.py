"""
LeafCare AI - Flask Application Server
Web server providing UI rendering, REST APIs for disease detection,
SQLite history tracking, and botanical knowledge queries.
"""

import os
import uuid
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

import database
import model_service

# Initialize Flask App
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'leafcare-ai-engineering-capstone-key-2026'

# File upload configs
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max request payload
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Disease Knowledge Base
DISEASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'disease_data.json')
with open(DISEASE_DATA_PATH, 'r', encoding='utf-8') as f:
    DISEASE_DATA = json.load(f)

# Ensure SQLite schema is prepared
database.init_db()


def allowed_file(filename):
    """Check if the uploaded file has an allowed image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_file(file_storage):
    """
    Validate that the file is not empty and is indeed a readable image.
    Returns (True, None) or (False, error_message).
    """
    if not file_storage or file_storage.filename == '':
        return False, "No file was selected for upload."

    if not allowed_file(file_storage.filename):
        return False, "Unsupported file format. Please upload JPG, PNG, or WEBP."

    try:
        # Verify image using Pillow
        image = Image.open(file_storage.stream)
        image.verify()
        # Reset stream position after verify()
        file_storage.stream.seek(0)
        return True, None
    except Exception as e:
        return False, f"Corrupted or invalid image file: {str(e)}"


# ==============================================================================
# UI Route
# ==============================================================================

@app.route('/')
def index():
    """Render the single-page application dashboard."""
    return render_template('index.html')


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.route('/api/model-status', methods=['GET'])
def get_model_status():
    """
    Check if the trained TensorFlow/Keras model is loaded and ready.
    """
    is_ready, details = model_service.get_status()
    return jsonify({
        "status": "ready" if is_ready else "pending",
        "details": details,
        "model_name": "MobileNetV2 PlantVillage Classifier" if is_ready else "Not Loaded"
    })


@app.route('/api/predict', methods=['POST'])
def predict_disease():
    """
    Run leaf disease prediction on an uploaded image using the trained deep learning model.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    uploaded_file = request.files['file']

    # Validate image
    is_valid, error_msg = validate_image_file(uploaded_file)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    # Ensure model is trained and available
    is_ready, _ = model_service.get_status()
    if not is_ready:
        return jsonify({
            "error": "The machine learning model has not been trained yet. Please run the model training script (train_model.py) first to generate real neural network weights."
        }), 503

    try:
        # Save uploaded file with a secure, unique identifier
        original_name = secure_filename(uploaded_file.filename)
        file_ext = original_name.rsplit('.', 1)[1].lower() if '.' in original_name else 'jpg'
        unique_name = f"{uuid.uuid4().hex}_{original_name}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        uploaded_file.save(save_path)

        # Run real model prediction
        pred_result = model_service.predict(save_path)
        class_key = pred_result['class_name']
        confidence = pred_result['confidence']

        # Cross-reference with botanical knowledge base
        disease_info = DISEASE_DATA.get(class_key, {
            "plant": class_key.split('___')[0].replace('_', ' ') if '___' in class_key else "Unknown Plant",
            "disease": class_key.split('___')[1].replace('_', ' ') if '___' in class_key else "Unknown Disease",
            "status": "Healthy" if "healthy" in class_key.lower() else "Diseased",
            "severity": "Unknown",
            "pathogen": "Under investigation",
            "description": "Comprehensive profile currently being compiled.",
            "symptoms": ["Visible foliar discolouration or lesions"],
            "organic_treatment": ["Isolate affected leaves and monitor watering"],
            "chemical_treatment": ["Consult regional agronomist for targeted treatment"],
            "prevention": ["Maintain clean cultivation practices"]
        })

        image_url = f"/static/uploads/{unique_name}"

        # Persist prediction in SQLite history
        pred_id = database.insert_prediction(
            filename=original_name,
            image_url=image_url,
            plant_name=disease_info.get('plant', 'Unknown'),
            disease_name=disease_info.get('disease', 'Unknown'),
            confidence=confidence,
            status=disease_info.get('status', 'Unknown'),
            severity=disease_info.get('severity', 'Moderate'),
            recommendations={
                "organic": disease_info.get('organic_treatment', []),
                "chemical": disease_info.get('chemical_treatment', []),
                "prevention": disease_info.get('prevention', [])
            }
        )

        response_payload = {
            "id": pred_id,
            "filename": original_name,
            "image_url": image_url,
            "plant": disease_info.get('plant'),
            "disease": disease_info.get('disease'),
            "status": disease_info.get('status'),
            "severity": disease_info.get('severity'),
            "pathogen": disease_info.get('pathogen'),
            "confidence": confidence,
            "top_probabilities": pred_result.get('top_predictions', []),
            "description": disease_info.get('description'),
            "symptoms": disease_info.get('symptoms', []),
            "organic_treatment": disease_info.get('organic_treatment', []),
            "chemical_treatment": disease_info.get('chemical_treatment', []),
            "prevention": disease_info.get('prevention', [])
        }

        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({"error": f"Inference pipeline failed: {str(e)}"}), 500


@app.route('/api/diseases', methods=['GET'])
def get_all_diseases():
    """Retrieve full repository of disease profiles."""
    return jsonify(DISEASE_DATA)


@app.route('/api/history', methods=['GET'])
def get_history():
    """Fetch stored diagnosis history records from SQLite."""
    limit = request.args.get('limit', default=100, type=int)
    records = database.get_all_predictions(limit=limit)
    return jsonify({"history": records})


@app.route('/api/history/<int:pred_id>', methods=['GET'])
def get_history_item(pred_id):
    """Fetch a single prediction record."""
    item = database.get_prediction_by_id(pred_id)
    if not item:
        return jsonify({"error": "Record not found"}), 404
    return jsonify(item)


@app.route('/api/history/<int:pred_id>', methods=['DELETE'])
def delete_history_item(pred_id):
    """Delete a prediction record from history."""
    deleted = database.delete_prediction(pred_id)
    if not deleted:
        return jsonify({"error": "Record not found"}), 404
    return jsonify({"success": True, "message": "Record deleted"})


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear all records from prediction history."""
    database.clear_all_history()
    return jsonify({"success": True, "message": "History cleared"})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Fetch aggregate diagnosis metrics for dashboard display."""
    stats = database.get_statistics()
    return jsonify(stats)


if __name__ == '__main__':
    # Initialize model service on startup
    model_service.load_model_weights()
    print("=" * 60)
    print("LeafCare AI Application Server running at http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
