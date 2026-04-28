"""
Model Training Script for Cancer Detection - SIMPLIFIED VERSION
This script loads the BUSI dataset, trains the CNN model with class weights
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import pickle
import seaborn as sns

# Configuration
DATASET_PATH = r"C:\Users\sudhe\OneDrive\Desktop\main_project\cancerdetection\Dataset_BUSI_with_GT"
MODEL_DIR = "model"
IMAGE_SIZE = 32

# Create model directory
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*70)
print("CANCER DETECTION MODEL TRAINING - FIXED FOR ALL 3 CLASSES")
print("="*70)

# Step 1: Load Dataset
print("\n[1/4] Loading dataset...")

X = []
Y = []

def getLabel(name):
    """Extract label from filename"""
    name_lower = name.lower()
    if "benign" in name_lower:
        return 0
    elif "malignant" in name_lower:
        return 1
    elif "normal" in name_lower:
        return 2
    else:
        return None

# Load images
img_count = 0
for root, dirs, files in os.walk(DATASET_PATH):
    dirs[:] = [d for d in dirs if d.lower() not in ('.venv', 'venv', '__pycache__')]

    folder = os.path.basename(root).lower()
    if folder == 'benign':
        current_label = 0
    elif folder == 'malignant':
        current_label = 1
    elif folder == 'normal':
        current_label = 2
    else:
        continue

    for file in files:
        if 'mask' in file.lower():
            continue
        
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                X.append(img)
                Y.append(current_label)
                img_count += 1
                
                if img_count % 100 == 0:
                    print(f"  Loaded {img_count} images...")
            except Exception as e:
                print(f"  ⚠ Skipping invalid image: {img_path} ({str(e)})")
                continue

X = np.array(X)
Y = np.array(Y)

print(f"✓ Dataset loaded successfully!")
print(f"  - Total images: {X.shape[0]}")
print(f"  - Image shape: {X.shape[1:]}")

# Count class distribution
unique, counts = np.unique(Y, return_counts=True)
print(f"\n  Class Distribution:")
print(f"  - Benign:    {counts[0]} (ID: 0)")
print(f"  - Malignant: {counts[1]} (ID: 1)")
print(f"  - Normal:    {counts[2]} (ID: 2)")

# Step 2: Preprocess Data
print("\n[2/4] Preprocessing data...")

X = X.astype('float32') / 255.0

# Shuffle data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# Convert to categorical
Y_cat = to_categorical(Y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.2, random_state=42)

print(f"✓ Data preprocessing completed!")
print(f"  - Training samples: {X_train.shape[0]} (80%)")
print(f"  - Test samples: {X_test.shape[0]} (20%)")

# Step 3: Build and Train Model
print("\n[3/4] Building and training CNN model...")

cnn_model = Sequential()

# Block 1: 64 filters
cnn_model.add(Conv2D(64, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), 
                      activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.2))

# Block 2: 128 filters
cnn_model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

# Block 3: 256 filters
cnn_model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

# Block 4: 512 filters
cnn_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.3))

# Global Average Pooling
cnn_model.add(GlobalAveragePooling2D())

# Dense layers
cnn_model.add(Dense(units=512, activation='relu', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.3))

cnn_model.add(Dense(units=256, activation='relu', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.3))

cnn_model.add(Dense(units=128, activation='relu', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.2))

cnn_model.add(Dense(units=3, activation='softmax'))

# Compile model
optimizer = Adam(learning_rate=0.001)
cnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print(f"✓ Model built successfully!")
print(f"  - Total parameters: {cnn_model.count_params():,}")

# Calculate class weights to balance training
y_train_labels = np.argmax(y_train, axis=1)
classes = np.unique(y_train_labels)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_labels)
class_weight_dict = {i: class_weights[i] for i in range(len(classes))}

print(f"\n  Class Weights (to handle imbalance):")
class_names_map = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}
for class_idx, weight in class_weight_dict.items():
    print(f"    {class_names_map[class_idx]:12}: {weight:.4f}")

# Setup callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=20, 
    restore_best_weights=True, 
    verbose=1, 
    min_delta=0.002
)

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, 'cnn_model.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=0.00001, 
    verbose=1
)

# Train model WITHOUT data augmentation to avoid TensorFlow issues
print(f"\n  Training for 150 epochs...")
print(f"  Goal: Predict all 3 classes (Benign, Malignant, Normal)")

history = cnn_model.fit(
    X_train, y_train,
    epochs=150,
    validation_data=(X_test, y_test),
    callbacks=[model_checkpoint, early_stopping, reduce_lr],
    class_weight=class_weight_dict,
    batch_size=16,
    verbose=1
)

print(f"\n✓ Model training completed!")

# Save training history
with open(os.path.join(MODEL_DIR, 'cnn_history.pckl'), 'wb') as f:
    pickle.dump(history.history, f)

# Step 4: Evaluate Model
print("\n[4/4] Evaluating model...")

# Get predictions
y_pred_probs = cnn_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred) * 100
precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
recall = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100

print(f"✓ Model evaluation completed!")
print(f"\n  Performance Metrics:")
print(f"  - Accuracy:  {accuracy:.2f}%")
print(f"  - Precision: {precision:.2f}%")
print(f"  - Recall:    {recall:.2f}%")
print(f"  - F1-Score:  {f1:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print(f"\n  Confusion Matrix:")
print(conf_matrix)

# Plot training results
fig = plt.figure(figsize=(16, 10))

# Training & Validation Accuracy
ax1 = plt.subplot(2, 2, 1)
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(1, len(train_acc) + 1)

ax1.plot(epochs_range, train_acc, 'b-o', linewidth=2, markersize=5, label='Training Accuracy')
ax1.plot(epochs_range, val_acc, 'r-s', linewidth=2, markersize=5, label='Validation Accuracy')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Training & Validation Loss
ax2 = plt.subplot(2, 2, 2)
train_loss = history.history['loss']
val_loss = history.history['val_loss']

ax2.plot(epochs_range, train_loss, 'g-o', linewidth=2, markersize=5, label='Training Loss')
ax2.plot(epochs_range, val_loss, 'orange', marker='s', linewidth=2, markersize=5, label='Validation Loss')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Confusion Matrix
ax3 = plt.subplot(2, 2, 3)
class_names = ['Benign', 'Malignant', 'Normal']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', 
            xticklabels=class_names, yticklabels=class_names, ax=ax3, cbar_kws={'label': 'Count'})
ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Label', fontsize=11)
ax3.set_xlabel('Predicted Label', fontsize=11)

# Summary Metrics
ax4 = plt.subplot(2, 2, 4)
summary_text = f"""
TRAINING SUMMARY
{'='*40}

Total Epochs: {len(train_acc)}

FINAL TEST METRICS:
  Accuracy:  {accuracy:.2f}%
  Precision: {precision:.2f}%
  Recall:    {recall:.2f}%
  F1-Score:  {f1:.2f}%

TRAINING HISTORY:
  Best Train Acc:  {max(train_acc)*100:.2f}%
  Best Val Acc:    {max(val_acc)*100:.2f}%
  Final Train Acc: {train_acc[-1]*100:.2f}%
  Final Val Acc:   {val_acc[-1]*100:.2f}%

CLASS WEIGHTS:
  Benign:    {class_weight_dict[0]:.4f}
  Malignant: {class_weight_dict[1]:.4f}
  Normal:    {class_weight_dict[2]:.4f}
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax4.axis('off')

plt.suptitle('Cancer Detection Model - Training Analysis (All 3 Classes)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_results.png'), dpi=100, bbox_inches='tight')
print(f"\n✓ Results saved to: {os.path.join(MODEL_DIR, 'training_results.png')}")
plt.close()

print("\n" + "="*70)
print("✓ TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\nModel saved at: {os.path.join(MODEL_DIR, 'cnn_model.keras')}")
print(f"History saved at: {os.path.join(MODEL_DIR, 'cnn_history.pckl')}")
print(f"\nNow run: python app.py")
print("="*70 + "\n")
