"""
Step 3A: Text Classification Model Training
Trains a TensorFlow/Keras model to classify crisis types from text descriptions
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("Step 3A: Text Classification Model Training")
print("="*60)

# ---- Configuration ----
DATA_PATH = 'data/processed/clean_crisis_reports.csv'
MODEL_DIR = 'src/ml_models'
MAX_WORDS = 5000
MAX_LEN = 50
EMBEDDING_DIM = 64
EPOCHS = 20
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ---- Load Data ----
print(f"\n[1/8] Loading dataset from {DATA_PATH}...")
if not os.path.exists(DATA_PATH):
    print(f"ERROR: File not found: {DATA_PATH}")
    print("Please run the preprocessing script first (eda_nlp_prep.py)")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"[OK] Loaded {len(df)} records")
print(f"[OK] Columns: {list(df.columns)}")

# Check for required columns
if 'clean_text' not in df.columns:
    print("ERROR: 'clean_text' column not found. Using 'description' instead.")
    if 'description' not in df.columns:
        print("ERROR: Neither 'clean_text' nor 'description' found!")
        sys.exit(1)
    df['clean_text'] = df['description']

if 'crisis_type' not in df.columns:
    print("ERROR: 'crisis_type' column not found!")
    sys.exit(1)

# Remove any rows with missing data
df = df[df['clean_text'].notna() & df['crisis_type'].notna()]
df = df[df['clean_text'].str.strip() != '']
print(f"[OK] After cleaning: {len(df)} valid records")

# ---- Tokenization ----
print(f"\n[2/8] Tokenizing text (max_words={MAX_WORDS}, max_len={MAX_LEN})...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['clean_text'])

X_seq = tokenizer.texts_to_sequences(df['clean_text'])
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"[OK] Vocabulary size: {len(tokenizer.word_index)}")
print(f"[OK] Sequence shape: {X_pad.shape}")

# ---- Encode Labels ----
print(f"\n[3/8] Encoding crisis type labels...")
le = LabelEncoder()
y = le.fit_transform(df['crisis_type'])
num_classes = len(le.classes_)

print(f"[OK] Number of classes: {num_classes}")
print(f"[OK] Classes: {list(le.classes_)}")
print(f"\nClass distribution:")
for i, cls in enumerate(le.classes_):
    count = np.sum(y == i)
    print(f"  {cls}: {count} ({count/len(y)*100:.1f}%)")

# ---- Train/Test Split ----
print(f"\n[4/8] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[OK] Training samples: {len(X_train)}")
print(f"[OK] Testing samples: {len(X_test)}")

# ---- Build Model ----
print(f"\n[5/8] Building text classification model...")
model = Sequential([
    Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
], name='crisis_text_classifier')

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
        os.path.join(MODEL_DIR, 'text_model_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ---- Train Model ----
print(f"\n[6/8] Training model (epochs={EPOCHS}, batch_size={BATCH_SIZE})...")
print("-" * 60)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1
)

print("-" * 60)
print("[OK] Training complete!")

# ---- Evaluation ----
print(f"\n[7/8] Evaluating model on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"[OK] Test Loss: {test_loss:.4f}")
print(f"[OK] Test Accuracy: {test_accuracy:.4f}")

# Predictions
y_pred = model.predict(X_test, verbose=0)
y_pred_labels = y_pred.argmax(axis=1)

# Classification Report
print("\n" + "="*60)
print("Classification Report:")
print("="*60)
print(classification_report(
    y_test, y_pred_labels,
    target_names=le.classes_,
    digits=3
))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_labels)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title('Confusion Matrix - Crisis Type Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()
print(f"[OK] Confusion matrix saved to {MODEL_DIR}/confusion_matrix.png")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=150)
plt.close()
print(f"[OK] Training history saved to {MODEL_DIR}/training_history.png")

# ---- Save Model and Artifacts ----
print(f"\n[8/8] Saving model and artifacts...")

# Save model
model_path = os.path.join(MODEL_DIR, 'text_model.h5')
model.save(model_path)
print(f"[OK] Model saved to {model_path}")

# Save tokenizer
tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"[OK] Tokenizer saved to {tokenizer_path}")

# Save label encoder
le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
with open(le_path, 'wb') as f:
    pickle.dump(le, f)
print(f"[OK] Label encoder saved to {le_path}")

# Save model configuration
config = {
    'max_words': MAX_WORDS,
    'max_len': MAX_LEN,
    'embedding_dim': EMBEDDING_DIM,
    'num_classes': num_classes,
    'classes': list(le.classes_),
    'vocabulary_size': len(tokenizer.word_index),
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss)
}

config_path = os.path.join(MODEL_DIR, 'model_config.pkl')
with open(config_path, 'wb') as f:
    pickle.dump(config, f)
print(f"[OK] Model config saved to {config_path}")

# ---- Summary ----
print("\n" + "="*60)
print("Step 3A Complete - Text Model Training [OK]")
print("="*60)
print("\nGenerated files:")
print(f"  - {model_path}")
print(f"  - {tokenizer_path}")
print(f"  - {le_path}")
print(f"  - {config_path}")
print(f"  - {MODEL_DIR}/confusion_matrix.png")
print(f"  - {MODEL_DIR}/training_history.png")
print(f"\nModel Performance:")
print(f"  Test Accuracy: {test_accuracy:.2%}")
print(f"  Test Loss: {test_loss:.4f}")
print("="*60)