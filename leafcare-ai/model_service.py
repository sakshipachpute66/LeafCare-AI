"""
LeafCare AI - Machine Learning Model Service
Handles neural network loading, image tensor preprocessing,
and inference prediction.
"""

import os
import json
import numpy as np
from PIL import Image

# Global Model References
_MODEL = None
_CLASS_NAMES = None
_IS_READY = False
_MODEL_INFO = "No model loaded"

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'data', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'leafcare_model.keras')
CLASSES_PATH = os.path.join(MODEL_DIR, 'class_indices.json')


def load_model_weights():
    """
    Attempt to load the trained TensorFlow/Keras model and class indices.
    """
    global _MODEL, _CLASS_NAMES, _IS_READY, _MODEL_INFO

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        _IS_READY = False
        _MODEL_INFO = "Model weights or class_indices.json not found in data/models/"
        return False

    try:
        # Import tensorflow only when attempting to load model
        import tensorflow as tf
        from tensorflow import keras

        # Load class index mapping
        with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
            raw_indices = json.load(f)
            # Support both {class_name: idx} and {idx: class_name}
            if isinstance(list(raw_indices.keys())[0], str) and isinstance(list(raw_indices.values())[0], int):
                # Invert so idx -> class_name
                _CLASS_NAMES = {v: k for k, v in raw_indices.items()}
            else:
                _CLASS_NAMES = {int(k): v for k, v in raw_indices.items()}

        _MODEL = keras.models.load_model(MODEL_PATH)
        _IS_READY = True
        _MODEL_INFO = f"Loaded model successfully ({len(_CLASS_NAMES)} classes)"
        print(f"[ModelService] {_MODEL_INFO}")
        return True

    except Exception as e:
        _IS_READY = False
        _MODEL_INFO = f"Failed to load model: {str(e)}"
        print(f"[ModelService Error] {_MODEL_INFO}")
        return False


def get_status():
    """Return model readiness status and descriptive message."""
    global _IS_READY, _MODEL_INFO
    if _MODEL is None:
        load_model_weights()
    return _IS_READY, _MODEL_INFO


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load an image from disk, convert to RGB, resize to target dimensions,
    and format into float32 tensor in range [0, 255] (network Rescaling layer handles 1/255).
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # Shape: (1, 224, 224, 3)
    return arr


def predict(image_path):
    """
    Run deep learning inference on the given leaf image.
    Returns:
        dict with 'class_name', 'confidence', 'top_predictions'
    """
    global _MODEL, _CLASS_NAMES, _IS_READY

    if not _IS_READY or _MODEL is None:
        if not load_model_weights():
            raise RuntimeError("Model is not loaded. Please train the model using train_model.py first.")

    input_tensor = preprocess_image(image_path)
    predictions = _MODEL.predict(input_tensor, verbose=0)[0]

    top_idx = int(np.argmax(predictions))
    top_confidence = float(predictions[top_idx]) * 100.0
    predicted_class = _CLASS_NAMES.get(top_idx, f"Class_{top_idx}")

    # Build top 3 prediction probabilities for model transparency
    sorted_indices = np.argsort(predictions)[::-1][:3]
    top_predictions = []
    for idx in sorted_indices:
        idx_int = int(idx)
        top_predictions.append({
            "class_name": _CLASS_NAMES.get(idx_int, f"Class_{idx_int}"),
            "probability": round(float(predictions[idx_int]) * 100.0, 2)
        })

    return {
        "class_name": predicted_class,
        "confidence": round(top_confidence, 2),
        "top_predictions": top_predictions
    }
