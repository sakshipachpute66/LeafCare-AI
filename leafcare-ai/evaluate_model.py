"""
LeafCare AI - Comprehensive Model Evaluation Engine
Evaluates the trained TensorFlow/Keras neural network against an isolated test split.
Computes true Test Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(__file__)
TEST_DIR = os.path.join(BASE_DIR, 'data', 'dataset', 'test')
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'leafcare_model.keras')
CLASSES_PATH = os.path.join(BASE_DIR, 'data', 'models', 'class_indices.json')
CONF_MATRIX_PATH = os.path.join(BASE_DIR, 'static', 'images', 'confusion_matrix.png')
METRICS_JSON_PATH = os.path.join(BASE_DIR, 'data', 'models', 'evaluation_metrics.json')


def evaluate():
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print("=" * 70)
    print("LeafCare AI - Neural Model Independent Test Set Evaluation")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train model first!")

    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"Class indices file not found at {CLASSES_PATH}.")

    # Load class indices
    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        class_map = json.load(f)
        # normalize to {int_idx: class_name}
        idx_to_name = {int(k): v for k, v in class_map.items()}

    class_names = [idx_to_name[i] for i in sorted(idx_to_name.keys())]
    name_to_idx = {name: i for i, name in idx_to_name.items()}

    print(f"Loading trained neural weights from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)

    # Gather test samples
    y_true = []
    y_pred = []
    y_pred_probs = []

    test_classes = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    print(f"Found {len(test_classes)} test class directories in {TEST_DIR}.")

    total_samples = 0
    for cls in sorted(test_classes):
        if cls not in name_to_idx:
            continue
        cls_idx = name_to_idx[cls]
        cls_dir = os.path.join(TEST_DIR, cls)
        img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for fname in img_files:
            img_path = os.path.join(cls_dir, fname)
            img = Image.open(img_path).convert('RGB').resize((224, 224))
            arr = np.array(img, dtype=np.float32)  # Model internal Rescaling(1./255) handles normalization
            tensor = np.expand_dims(arr, axis=0)

            preds = model.predict(tensor, verbose=0)[0]
            pred_idx = int(np.argmax(preds))

            y_true.append(cls_idx)
            y_pred.append(pred_idx)
            y_pred_probs.append(float(preds[pred_idx]))
            total_samples += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Compute Overall Test Accuracy
    overall_acc = accuracy_score(y_true, y_pred)
    avg_confidence = np.mean(y_pred_probs) * 100.0

    # 2. Detailed Classification Report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT ON TEST SET:")
    print("=" * 70)
    print(report_text)
    print("=" * 70)
    print(f"Total Test Samples Evaluated:  {total_samples}")
    print(f"Overall Test Accuracy:         {overall_acc * 100:.2f}%")
    print(f"Average Prediction Confidence: {avg_confidence:.2f}%")
    print(f"Macro Average F1-Score:        {report_dict['macro avg']['f1-score'] * 100:.2f}%")
    print(f"Weighted Average F1-Score:     {report_dict['weighted avg']['f1-score'] * 100:.2f}%")
    print("=" * 70)

    # 3. Confusion Matrix Computation & Visualization
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Test Dataset Confusion Matrix — LeafCare AI', fontsize=14, pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    short_labels = [c.replace('___', ' \n') for c in class_names]
    plt.xticks(tick_marks, short_labels, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_marks, short_labels, fontsize=9)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight='bold')

    plt.tight_layout()
    plt.ylabel('True Class Label', fontsize=11)
    plt.xlabel('Predicted Class Label', fontsize=11)
    plt.savefig(CONF_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"Saved Confusion Matrix visualization to: {CONF_MATRIX_PATH}")

    # 4. Save Metrics Summary to JSON
    summary = {
        "total_test_samples": total_samples,
        "test_accuracy": round(overall_acc * 100, 2),
        "avg_confidence": round(avg_confidence, 2),
        "macro_avg_f1": round(report_dict['macro avg']['f1-score'] * 100, 2),
        "weighted_avg_f1": round(report_dict['weighted avg']['f1-score'] * 100, 2),
        "classification_report": report_dict
    }

    with open(METRICS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation metrics JSON to: {METRICS_JSON_PATH}")

    return summary


if __name__ == '__main__':
    evaluate()
