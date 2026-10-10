# LeafCare AI — Leaf Disease Prediction System
> **An Intelligent Botanical Pathology Diagnosis & Agronomic Remediation Platform**  
> *Developed for College Final Year / Capstone Engineering Project Evaluation*

---

## 📌 1. Project Overview

**LeafCare AI** is an end-to-end artificial intelligence web application designed to detect, classify, and manage plant foliar diseases. Using deep learning transfer learning (MobileNetV2 / Deep CNN) trained on the **PlantVillage** benchmark dataset, it allows farmers, agronomists, and students to upload leaf photos, obtain instant disease classifications with authentic confidence ratings, examine organic/chemical treatments, and track longitudinal scan history via an embedded SQLite database.

### 🌟 Key Highlights
- **Real Deep Learning Pipeline**: Powered by TensorFlow and Keras trained on real crop foliage classes (Tomato, Potato, Corn, Bell Pepper, Apple). **Zero hardcoded predictions or fake confidence ratings.**
- **Modern Responsive Web Interface**: Crafted with semantic HTML5, custom responsive CSS3, and modern vanilla JavaScript following a warm **forest green, cream, and white** agricultural aesthetic.
- **SQLite Relational Persistence**: Complete CRUD history logging for scans, timestamps, confidence scores, and remedy snapshots.
- **Dual-Mode Remediation**: Provides both organic (bio-fungicide, cultural practices, neem oil) and synthetic chemical treatments for every identified pathology.
- **Comprehensive Model Evaluation**: Scripts evaluate real test accuracy, precision, recall, macro F1-score, and render graphical confusion matrices.
- **Defensive Error Handling**: Rejects corrupt images, unsupported MIME types, and oversized payloads.

---

## 🗂️ 2. Project File Structure

```
leafcare-ai/
│
├── app.py                      # Flask backend server & REST API router
├── database.py                 # SQLite database helper (CRUD for scan history)
├── model_service.py            # Model loading, tensor preprocessing & inference engine
├── disease_data.json           # Comprehensive botanical knowledge base & remedies
├── download_dataset.py         # PlantVillage dataset generator & partition builder
├── train_model.py              # TensorFlow/Keras training pipeline (epochs, augmentations)
├── evaluate_model.py           # Independent test set evaluation & confusion matrix generator
├── requirements.txt            # Python runtime dependencies
├── README.md                   # Complete documentation, setup guide & Viva prep
│
├── data/
│   ├── dataset/                # PlantVillage dataset splits
│   │   ├── train/              # Training images per class
│   │   └── test/               # Independent testing images per class
│   ├── models/                 # Saved trained model artifacts
│   │   ├── leafcare_model.keras  # Neural network weights
│   │   ├── class_indices.json    # Class label to index mapping
│   │   └── evaluation_metrics.json # True performance metrics
│   └── leafcare.db             # SQLite persistent database
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom styling with green, cream, white theme
│   ├── js/
│   │   └── main.js             # Client interactivity, drag & drop, AJAX, history UI
│   ├── images/
│   │   ├── samples/            # Pre-loaded reference leaf specimens for quick testing
│   │   ├── training_history.png # Accuracy & loss curves plot
│   │   └── confusion_matrix.png # Generated test set confusion matrix
│   └── uploads/                # User uploaded leaf images (auto-saved)
│
└── templates/
    └── index.html              # Responsive single-page application with 5 sections
```

---

## 🎨 3. Design Theme & Aesthetics

The UI implements a cohesive agricultural color scheme:
- **Primary Deep Forest Green**: `#1b4332` / `#0a1f16` (Brand header, nav, primary buttons)
- **Secondary Leaf Emerald**: `#2d6a4f` / `#52b788` (Accents, progress meters, badges)
- **Soft Cream & Almond**: `#fcfbf7` / `#f1e9d8` (Page background, warm card surfaces)
- **Crisp Pure White**: `#ffffff` (Card containers, modal popups)

---

## ⚙️ 4. Prerequisites & Installation

### Step 1: Verify Python Installation
LeafCare AI requires **Python 3.10 to 3.13**. Verify your installation in PowerShell or Command Prompt:
```bash
python --version
pip --version
```

### Step 2: Navigate to Project Directory
```bash
cd "C:\Users\HP\.gemini\antigravity\scratch\leafcare-ai"
```

### Step 3: Install Required Dependencies
Install the required packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```
*(Dependencies: Flask, Werkzeug, Pillow, numpy, tensorflow, scikit-learn, matplotlib, requests)*

---

## 🚀 5. How to Run the Project (Step-by-Step)

Follow this structured workflow for initial setup or college demonstration:

### Step A: Prepare Dataset
Generate/prepare the train and test splits for the PlantVillage classes:
```bash
python download_dataset.py
```

### Step B: Train the Deep Learning Model
Train the neural network using TensorFlow/Keras on the prepared images:
```bash
python train_model.py
```
- Trains for 8 epochs with data augmentation (horizontal flip, random zoom, rotation).
- Saves the trained model to `data/models/leafcare_model.keras`.
- Generates the training curves chart at `static/images/training_history.png`.

### Step C: Evaluate Model on Test Dataset
Evaluate the trained model on the independent test set:
```bash
python evaluate_model.py
```
- Computes true Test Accuracy, Precision, Recall, and F1-Scores.
- Generates the confusion matrix heatmap at `static/images/confusion_matrix.png`.
- Saves metrics summary to `data/models/evaluation_metrics.json`.

### Step D: Launch the Flask Web Application
Start the application server:
```bash
python app.py
```
- Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 🌐 6. Website Sections Overview

1. **Home**:
   - Project hero banner with live statistics counter.
   - Interactive 3-step workflow diagram.
   - Live backend model health badge.
2. **Detect Disease**:
   - Drag-and-drop dropzone or manual file selection.
   - Live image preview with filename and size validation.
   - One-click sample leaf buttons (for instant demonstration).
   - Real-time neural inference with exact confidence score, pathogen identity, severity rating, and tabbed treatment recommendations.
   - "Print Diagnosis" report button.
3. **Disease Information**:
   - Botanical encyclopedia covering symptoms, causes, organic control, and chemical treatment.
   - Live search filter (searches across crop, disease, or symptom).
   - Crop category filter pills (Tomato, Potato, Corn, Pepper, Apple, Healthy).
   - Modal drawer with detailed disease breakdown.
4. **Prediction History**:
   - Historical records table stored in SQLite.
   - Displays thumbnail preview, plant name, diagnosed condition, confidence percentage, and timestamp.
   - Single-record deletion and complete history reset.
5. **About Project**:
   - Academic motivation, system architecture flowchart, deep learning pipeline specs, and viva credentials.

---

## 🔌 7. REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the main web dashboard |
| `GET` | `/api/model-status` | Returns neural model readiness status and class count |
| `POST` | `/api/predict` | Receives multipart leaf image, runs model inference, stores result in SQLite |
| `GET` | `/api/diseases` | Returns complete disease knowledge base |
| `GET` | `/api/history` | Fetches historical scan logs from SQLite |
| `DELETE`| `/api/history/<id>` | Deletes specific scan record from database |
| `POST` | `/api/history/clear`| Clears all records from scan history |
| `GET` | `/api/stats` | Returns aggregated metrics (total scans, healthy vs diseased) |

---

## 🎓 8. College Viva Voce Questions & Answers

**Q1: Why is deep learning better than traditional image processing for plant disease detection?**  
*Answer:* Traditional image processing relies on handcrafted features (e.g., color histograms, Sobel edge detectors, texture filters) which fail under varying lighting, shadows, angles, and background variations. Deep Convolutional Neural Networks automatically learn hierarchical spatial features—from low-level edges and textures to complex lesion patterns and necrotic boundaries—resulting in higher generalization accuracy.

**Q2: What is the purpose of Softmax in the output layer?**  
*Answer:* Softmax transforms the raw unnormalized logits of the neural network into a valid probability distribution where each value is between 0 and 1, and the sum of all probabilities equals 1. The class with the highest probability is our predicted diagnosis, and this probability multiplied by 100 serves as the authentic confidence score.

**Q3: How do we prevent overfitting during training?**  
*Answer:* Overfitting is mitigated through four proven techniques:
1. **Data Augmentation**: Generating rotated, flipped, and zoomed variants to expose the model to diverse spatial orientations.
2. **Dropout Regularization**: Randomly deactivating neuron connections during training (rate: 0.2–0.3) to prevent co-adaptation.
3. **Early Stopping**: Halting training if validation loss fails to improve for consecutive epochs.
4. **Learning Rate Reduction**: Reducing the learning rate via `ReduceLROnPlateau` when progress stagnates.

**Q4: Why was SQLite selected instead of MongoDB or MySQL for this project?**  
*Answer:* SQLite is a serverless, zero-configuration, self-contained relational database embedded directly in Python. It avoids the overhead of managing a separate database daemon while providing ACID transactions, fast query execution, and portability across any operating system.

**Q5: What is the difference between Precision, Recall, and F1-score in agricultural diagnosis?**  
*Answer:*
- **Precision**: Of all leaves predicted to have a specific disease, what fraction actually had it? (Minimizes false alarms).
- **Recall**: Of all leaves that actually had the disease, what fraction did the model successfully catch? (Critical in agriculture to prevent uncontained disease outbreaks).
- **F1-Score**: The harmonic mean of Precision and Recall, providing a balanced metric especially on uneven class distributions.

---

## 📜 9. License & Academic Attribution
Developed as an engineering capstone project utilizing the public **PlantVillage Dataset** (Hughes & Salathé, Penn State University / EPFL). Free for academic, educational, and research use.
