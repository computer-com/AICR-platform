"""
Step 3C: Multimodal Fusion Model Training
Combines text + image + geospatial features for crisis classification
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Embedding, GlobalAveragePooling1D,
                                     Input, concatenate, BatchNormalization)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle

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
print("Step 3C: Multimodal Fusion Model Training")
print("="*60)

# ---- Configuration ----
DATA_PATH = 'data/processed/clean_crisis_reports.csv'
MODEL_DIR = 'src/ml_models'
MAX_WORDS = 5000
MAX_LEN = 50
EMBEDDING_DIM = 64
EPOCHS = 20
BATCH_SIZE = 16

os.makedirs(MODEL_DIR, exist_ok=True)

# ---- Load Data ----
print(f"\n[1/7] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"[OK] Loaded {len(df)} records")

# Ensure required columns
if 'clean_text' not in df.columns:
    df['clean_text'] = df['description']

# Remove missing data
df = df[df['clean_text'].notna() & df['crisis_type'].notna()]
df = df[df['clean_text'].str.strip() != '']

# Check for geospatial data
has_geo = {'latitude', 'longitude'}.issubset(df.columns)
if has_geo:
    df = df[df['latitude'].notna() & df['longitude'].notna()]
    print(f"[OK] Found geospatial features (lat/lon)")
else:
    print(f"[INFO] No geospatial features found")

print(f"[OK] Clean dataset: {len(df)} records")

# ---- Process Text Features ----
print(f"\n[2/7] Processing text features...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['clean_text'])

X_text_seq = tokenizer.texts_to_sequences(df['clean_text'])
X_text = pad_sequences(X_text_seq, maxlen=MAX_LEN, padding='post')
print(f"[OK] Text sequences shape: {X_text.shape}")

# ---- Process Geospatial Features ----
if has_geo:
    print(f"\n[3/7] Processing geospatial features...")
    X_geo = df[['latitude', 'longitude']].values
    
    # Normalize geospatial features
    geo_scaler = StandardScaler()
    X_geo = geo_scaler.fit_transform(X_geo)
    print(f"[OK] Geospatial features shape: {X_geo.shape}")
    
    # Save scaler
    with open(os.path.join(MODEL_DIR, 'geo_scaler.pkl'), 'wb') as f:
        pickle.dump(geo_scaler, f)
else:
    X_geo = None

# ---- Encode Labels ----
print(f"\n[4/7] Encoding labels...")
le = LabelEncoder()
y = le.fit_transform(df['crisis_type'])
num_classes = len(le.classes_)
print(f"[OK] Number of classes: {num_classes}")
print(f"[OK] Classes: {list(le.classes_)}")

# ---- Train/Test Split ----
print(f"\n[5/7] Splitting data...")
if has_geo:
    X_text_train, X_text_test, X_geo_train, X_geo_test, y_train, y_test = train_test_split(
        X_text, X_geo, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[OK] Training: {len(X_text_train)} samples (text + geo)")
    print(f"[OK] Testing: {len(X_text_test)} samples (text + geo)")
else:
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    X_geo_train = None
    X_geo_test = None
    print(f"[OK] Training: {len(X_text_train)} samples (text only)")
    print(f"[OK] Testing: {len(X_text_test)} samples (text only)")

# ---- Build Multimodal Model ----
print(f"\n[6/7] Building multimodal fusion model...")

# Text input branch
text_input = Input(shape=(MAX_LEN,), name='text_input')
text_embedding = Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN)(text_input)
text_pool = GlobalAveragePooling1D()(text_embedding)
text_dense = Dense(64, activation='relu')(text_pool)
text_dropout = Dropout(0.3)(text_dense)

if has_geo:
    # Geospatial input branch
    geo_input = Input(shape=(2,), name='geo_input')
    geo_dense1 = Dense(16, activation='relu')(geo_input)
    geo_dense2 = Dense(8, activation='relu')(geo_dense1)
    
    # Fusion layer
    fusion = concatenate([text_dropout, geo_dense2])
    fusion_dense1 = Dense(64, activation='relu')(fusion)
    fusion_bn = BatchNormalization()(fusion_dense1)
    fusion_dropout = Dropout(0.5)(fusion_bn)
    fusion_dense2 = Dense(32, activation='relu')(fusion_dropout)
    fusion_dropout2 = Dropout(0.3)(fusion_dense2)
    
    # Output
    output = Dense(num_classes, activation='softmax', name='output')(fusion_dropout2)
    
    model = Model(inputs=[text_input, geo_input], outputs=output, name='multimodal_crisis_classifier')
else:
    # Text-only model
    text_dense2 = Dense(32, activation='relu')(text_dropout)
    text_dropout2 = Dropout(0.3)(text_dense2)
    output = Dense(num_classes, activation='softmax', name='output')(text_dropout2)
    
    model = Model(inputs=text_input, outputs=output, name='text_crisis_classifier')

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
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
        os.path.join(MODEL_DIR, 'multimodal_model_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ---- Train Model ----
print(f"\n[7/7] Training multimodal model...")
print("-" * 60)

if has_geo:
    history = model.fit(
        [X_text_train, X_geo_train], y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
else:
    history = model.fit(
        X_text_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

print("-" * 60)
print("[OK] Training complete!")

# ---- Evaluation ----
print(f"\nEvaluating model on test set...")
if has_geo:
    test_loss, test_accuracy = model.evaluate([X_text_test, X_geo_test], y_test, verbose=0)
    y_pred = model.predict([X_text_test, X_geo_test], verbose=0)
else:
    test_loss, test_accuracy = model.evaluate(X_text_test, y_test, verbose=0)
    y_pred = model.predict(X_text_test, verbose=0)

y_pred_labels = y_pred.argmax(axis=1)

print(f"[OK] Test Loss: {test_loss:.4f}")
print(f"[OK] Test Accuracy: {test_accuracy:.4f}")

# Classification Report
print("\n" + "="*60)
print("Classification Report:")
print("="*60)
print(classification_report(
    y_test, y_pred_labels,
    target_names=le.classes_,
    digits=3
))

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Multimodal Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Multimodal Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'multimodal_training_history.png'), dpi=150)
plt.close()
print(f"[OK] Training history saved")

# ---- Save Model and Artifacts ----
print(f"\nSaving model and artifacts...")

model_path = os.path.join(MODEL_DIR, 'multimodal_model.h5')
model.save(model_path)
print(f"[OK] Model saved to {model_path}")

# Save tokenizer (if not already saved)
tokenizer_path = os.path.join(MODEL_DIR, 'multimodal_tokenizer.pkl')
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"[OK] Tokenizer saved")

# Save label encoder (if not already saved)
le_path = os.path.join(MODEL_DIR, 'multimodal_label_encoder.pkl')
with open(le_path, 'wb') as f:
    pickle.dump(le, f)
print(f"[OK] Label encoder saved")

# Save configuration
config = {
    'max_words': MAX_WORDS,
    'max_len': MAX_LEN,
    'embedding_dim': EMBEDDING_DIM,
    'num_classes': num_classes,
    'classes': list(le.classes_),
    'has_geo': has_geo,
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss)
}

config_path = os.path.join(MODEL_DIR, 'multimodal_config.pkl')
with open(config_path, 'wb') as f:
    pickle.dump(config, f)
print(f"[OK] Config saved")

# ---- Summary ----
print("\n" + "="*60)
print("Step 3C Complete - Multimodal Fusion Training [OK]")
print("="*60)
print(f"\nModel Type: {'Text + Geospatial' if has_geo else 'Text Only'}")
print(f"Test Accuracy: {test_accuracy:.2%}")
print(f"Test Loss: {test_loss:.4f}")
print("\nGenerated files:")
print(f"  - {model_path}")
print(f"  - {tokenizer_path}")
print(f"  - {le_path}")
print(f"  - {config_path}")
if has_geo:
    print(f"  - {MODEL_DIR}/geo_scaler.pkl")
print("="*60)