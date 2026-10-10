"""
LeafCare AI - Deep Learning Model Training Pipeline
Trains a real Convolutional Neural Network (MobileNetV2 / CNN architecture)
on the PlantVillage leaf dataset using TensorFlow and Keras.
"""

import os
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Ensure directories exist
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data', 'dataset')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'models')
STATIC_IMG_DIR = os.path.join(BASE_DIR, 'static', 'images')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_IMG_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'leafcare_model.keras')
CLASSES_SAVE_PATH = os.path.join(MODEL_DIR, 'class_indices.json')
HISTORY_PLOT_PATH = os.path.join(STATIC_IMG_DIR, 'training_history.png')


def train():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models

    print("=" * 65)
    print("LeafCare AI - TensorFlow / Keras Model Training")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Training Directory: {TRAIN_DIR}")
    print("=" * 65)

    if not os.path.exists(TRAIN_DIR) or len(os.listdir(TRAIN_DIR)) == 0:
        print("[Error] Training dataset not found. Generating dataset first...")
        import download_dataset
        download_dataset.build_dataset()

    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 16
    EPOCHS = 8

    # 1. Load Data Generators / Datasets
    print("\n[Step 1/5] Loading and splitting training and validation data...")
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Discovered {num_classes} classes: {class_names}")

    # Save class indices mapping
    class_indices = {i: name for i, name in enumerate(class_names)}
    with open(CLASSES_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(class_indices, f, indent=2)
    print(f"Saved class indices mapping to {CLASSES_SAVE_PATH}")

    # Use test directory as validation during training
    val_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # 2. Build Deep CNN Architecture
    print("\n[Step 2/5] Constructing Deep Neural Network Architecture...")
    
    # Data Augmentation & Normalization layers
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    inputs = layers.Input(shape=(224, 224, 3), name="input_leaf_tensor")
    x = layers.Rescaling(1./255)(inputs)
    x = data_augmentation(x)

    # Try MobileNetV2 feature extractor or robust ConvNet
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Feature extraction freeze
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation='softmax', name="disease_prediction")(x)
        model = models.Model(inputs, outputs, name="LeafCare_MobileNetV2")
        print("Initialized MobileNetV2 Transfer Learning Backbone.")
    except Exception as e:
        print(f"Transfer learning fallback to Custom Deep CNN: {e}")
        # Convolutional Block 1
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Convolutional Block 2
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Convolutional Block 3
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Dense Classification Head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax', name="disease_prediction")(x)
        model = models.Model(inputs, outputs, name="LeafCare_Custom_CNN")

    # 3. Compile Model
    print("\n[Step 3/5] Compiling Model with Adam Optimizer...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 4. Train Model
    print(f"\n[Step 4/5] Training for {EPOCHS} Epochs on PlantVillage data...")
    start_time = time.time()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")

    # 5. Save Model Artifacts
    print("\n[Step 5/5] Saving Trained Model Weights & Generating Plots...")
    model.save(MODEL_SAVE_PATH)
    print(f"Trained model saved to: {MODEL_SAVE_PATH}")

    # Plot Training & Validation Curves
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    plt.figure(figsize=(10, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy', color='#2d6a4f', linewidth=2)
    plt.plot(val_acc, label='Validation Accuracy', color='#52b788', linestyle='--', linewidth=2)
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss', color='#dc2626', linewidth=2)
    plt.plot(val_loss, label='Validation Loss', color='#f59e0b', linestyle='--', linewidth=2)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved training curves chart to: {HISTORY_PLOT_PATH}")

    final_train_acc = acc[-1] if acc else 0
    final_val_acc = val_acc[-1] if val_acc else 0
    print("\n" + "=" * 65)
    print(f"FINAL TRAINING ACCURACY:   {final_train_acc * 100:.2f}%")
    print(f"FINAL VALIDATION ACCURACY: {final_val_acc * 100:.2f}%")
    print("=" * 65)

    return model, history


if __name__ == '__main__':
    train()
