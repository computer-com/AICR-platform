"""
Step 3B: Image Classification Model Training (Optional)
Trains a CNN to classify crisis types from images
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("Step 3B: Image Classification Model Training")
print("="*60)

# ---- Configuration ----
DATA_PATH = 'data/processed/clean_crisis_reports.csv'
IMAGE_DIR = 'data/images'  # Directory containing crisis images
MODEL_DIR = 'src/ml_models'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 20

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

print(f"\n[1/7] Checking for image data...")

# Check if we have image data
if not os.path.exists(DATA_PATH):
    print(f"ERROR: {DATA_PATH} not found!")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)

# Check if we have image paths
if 'image_path' not in df.columns:
    print("[WARNING] No 'image_path' column found in dataset.")
    print("[INFO] Creating sample synthetic images for demonstration...")
    
    # Create synthetic sample images for each crisis type
    crisis_types = df['crisis_type'].unique()
    
    for crisis_type in crisis_types:
        crisis_dir = os.path.join(IMAGE_DIR, crisis_type)
        os.makedirs(crisis_dir, exist_ok=True)
        
        # Create 10 sample images per crisis type
        for i in range(10):
            # Create a simple colored image (different color per crisis type)
            img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), 
                          color=(np.random.randint(0, 255), 
                                np.random.randint(0, 255), 
                                np.random.randint(0, 255)))
            img.save(os.path.join(crisis_dir, f'{crisis_type}_{i}.jpg'))
    
    print(f"[OK] Created synthetic images in {IMAGE_DIR}/")
    print("[INFO] In a real project, you would use actual crisis images.")

# ---- Load Data ----
print(f"\n[2/7] Preparing image data...")

# For demo purposes, we'll use ImageDataGenerator with directory structure
# Expected structure: IMAGE_DIR/crisis_type/image.jpg

# Count images per class
crisis_counts = {}
for crisis_type in os.listdir(IMAGE_DIR):
    crisis_path = os.path.join(IMAGE_DIR, crisis_type)
    if os.path.isdir(crisis_path):
        count = len([f for f in os.listdir(crisis_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        crisis_counts[crisis_type] = count

print(f"[OK] Found images:")
for crisis_type, count in crisis_counts.items():
    print(f"  {crisis_type}: {count} images")

total_images = sum(crisis_counts.values())
if total_images == 0:
    print("ERROR: No images found!")
    sys.exit(1)

# ---- Data Augmentation ----
print(f"\n[3/7] Setting up data augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Create generators
train_generator = train_datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

validation_generator = test_datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())

print(f"[OK] Number of classes: {num_classes}")
print(f"[OK] Classes: {class_names}")
print(f"[OK] Training samples: {train_generator.samples}")
print(f"[OK] Validation samples: {validation_generator.samples}")

# ---- Build Model (Transfer Learning with MobileNetV2) ----
print(f"\n[4/7] Building image classification model...")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Add custom layers
model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
], name='crisis_image_classifier')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n[OK] Model architecture:")
model.summary()

# ---- Training Callbacks ----
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(MODEL_DIR, 'image_model_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

# ---- Train Model ----
print(f"\n[5/7] Training model (epochs={EPOCHS})...")
print("-" * 60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("-" * 60)
print("[OK] Training complete!")

# ---- Evaluation ----
print(f"\n[6/7] Evaluating model...")
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
print(f"[OK] Validation Loss: {val_loss:.4f}")
print(f"[OK] Validation Accuracy: {val_accuracy:.4f}")

# Get predictions
validation_generator.reset()
y_pred = model.predict(validation_generator, verbose=0)
y_pred_labels = y_pred.argmax(axis=1)
y_true = validation_generator.classes

# Classification Report
print("\n" + "="*60)
print("Classification Report:")
print("="*60)
print(classification_report(
    y_true, y_pred_labels,
    target_names=class_names,
    digits=3
))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Confusion Matrix - Image Crisis Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'image_confusion_matrix.png'), dpi=150)
plt.close()
print(f"[OK] Confusion matrix saved")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Image Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Image Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'image_training_history.png'), dpi=150)
plt.close()
print(f"[OK] Training history saved")

# ---- Save Model ----
print(f"\n[7/7] Saving model and artifacts...")

model_path = os.path.join(MODEL_DIR, 'image_model.h5')
model.save(model_path)
print(f"[OK] Model saved to {model_path}")

# Save class indices
class_indices_path = os.path.join(MODEL_DIR, 'image_class_indices.pkl')
with open(class_indices_path, 'wb') as f:
    pickle.dump(train_generator.class_indices, f)
print(f"[OK] Class indices saved")

# Save config
config = {
    'img_height': IMG_HEIGHT,
    'img_width': IMG_WIDTH,
    'num_classes': num_classes,
    'class_names': class_names,
    'val_accuracy': float(val_accuracy),
    'val_loss': float(val_loss)
}

config_path = os.path.join(MODEL_DIR, 'image_model_config.pkl')
with open(config_path, 'wb') as f:
    pickle.dump(config, f)
print(f"[OK] Config saved")

# ---- Summary ----
print("\n" + "="*60)
print("Step 3B Complete - Image Model Training [OK]")
print("="*60)
print(f"\nModel Performance:")
print(f"  Validation Accuracy: {val_accuracy:.2%}")
print(f"  Validation Loss: {val_loss:.4f}")
print("="*60)