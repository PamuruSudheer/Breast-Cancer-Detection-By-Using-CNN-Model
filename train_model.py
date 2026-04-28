"""
Model Training Script for Cancer Detection
This script loads the BUSI dataset, trains the CNN model, and saves it
"""

import os       #built-in Python module used to interact with the operating system, connection between the python program and the operating system..
import cv2      #   OpenCV library for image processing
import numpy as np  #   is used to work with numbers, arrays, and mathematical calculations.
import matplotlib.pyplot as plt # for showing the graphs and plots. 
from sklearn.model_selection import train_test_split    # to split the dataset into training and testing sets.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # for evaluating model performance.
from tensorflow.keras.utils import to_categorical # to convert class vectors to binary class matrices.
from tensorflow.keras.models import Sequential  # to build the CNN model as a linear stack of layers.
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, RepeatVector, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2 # keras : building the nn,train the dl, test model accuracy,implement cnn for image.
import pickle   # for saving training history.
import seaborn as sns #  Seaborn is used to create beautiful and easy statistical graphs.

# Configuration
DATASET_PATH = r"C:\Users\sudhe\OneDrive\Desktop\main_project\cancerdetection\Dataset_BUSI_with_GT"
MODEL_DIR = "model"
IMAGE_SIZE = 64  # Increased for better feature extraction

# Create model directory
os.makedirs(MODEL_DIR, exist_ok=True)

def log_info(msg):
    print(f"[TRAIN] {msg}")

log_info("="*70)
log_info("CANCER DETECTION MODEL TRAINING")
log_info("="*70)

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
benign_images = []
malignant_images = []
normal_images = []

for root, dirs, files in os.walk(DATASET_PATH):
    # Skip nested virtual environment and cache directories
    dirs[:] = [d for d in dirs if d.lower() not in ('.venv', 'venv', '__pycache__')]

    # Determine class by current folder name
    folder = os.path.basename(root).lower()
    if folder == 'benign':
        current_label = 0
        target_list = benign_images
    elif folder == 'malignant':
        current_label = 1
        target_list = malignant_images
    elif folder == 'normal':
        current_label = 2
        target_list = normal_images
    else:
        continue

    for file in files:
        # Skip mask images and unrelated files
        if 'mask' in file.lower():
            continue
        
        # Only load ultrasound images that match the pattern: class_name (number).png
        expected_prefix = folder + ' ('
        if not file.startswith(expected_prefix) or not file.endswith(').png'):
            continue

        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"  ⚠ Skipping unreadable image: {img_path}")
                continue

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            target_list.append((img, current_label))

# Balance the dataset by undersampling benign to match malignant count
min_count = min(len(benign_images), len(malignant_images), len(normal_images))
print(f"  Balancing dataset to {min_count} images per class...")

import random
random.shuffle(benign_images)
random.shuffle(malignant_images)
random.shuffle(normal_images)

balanced_images = benign_images[:min_count] + malignant_images[:min_count] + normal_images[:min_count]

# Separate X and Y
X = [img for img, label in balanced_images]
Y = [label for img, label in balanced_images]

X = np.array(X)
Y = np.array(Y)

print(f"✓ Dataset loaded and balanced successfully!")
print(f"  - Total images: {X.shape[0]} ({min_count} per class)")
print(f"  - Image shape: {X.shape[1:]}")

# Count class distribution
unique, counts = np.unique(Y, return_counts=True)
benign_count = counts[np.where(unique == 0)][0] if 0 in unique else 0
malignant_count = counts[np.where(unique == 1)][0] if 1 in unique else 0
normal_count = counts[np.where(unique == 2)][0] if 2 in unique else 0

print(f"  - Benign: {benign_count}")
print(f"  - Malignant: {malignant_count}")
print(f"  - Normal: {normal_count}")

# Step 2: Preprocess Data
print("\n[2/4] Preprocessing data...")

X = X.astype('float32')
X = X / 255.0

# Shuffle data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# Convert to categorical
Y = to_categorical(Y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"✓ Data preprocessing completed!")
print(f"  - Training samples: {X_train.shape[0]} (80%)")
print(f"  - Test samples: {X_test.shape[0]} (20%)")
print(f"  - Classes: 3 (Benign, Malignant, Normal)")

# Step 3: Build and Train Model
print("\n[3/4] Building and training CNN model (Enhanced for 98% accuracy)...")
print("  Architecture: Advanced Deep CNN with Residual-like features")

cnn_model = Sequential()

# Block 1: 64 filters with balanced regularization
cnn_model.add(Conv2D(64, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.2))

# Block 2: 128 filters with balanced regularization
cnn_model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

# Block 3: 256 filters with balanced regularization
cnn_model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

# Block 4: 512 filters for deeper feature extraction
cnn_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.3))

# Global Average Pooling for better feature representation
cnn_model.add(GlobalAveragePooling2D())

# Dense layers with balanced regularization
cnn_model.add(Dense(units=512, activation='relu', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.3))

cnn_model.add(Dense(units=256, activation='relu', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.3))

cnn_model.add(Dense(units=128, activation='relu', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.2))

cnn_model.add(Dense(units=y_train.shape[1], activation='softmax'))

# Compile model with balanced settings
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
cnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print(f"✓ Model built successfully!")
print(f"  - Total parameters: {cnn_model.count_params():,}")

# Advanced Data Augmentation for better generalization
train_datagen = ImageDataGenerator(
    rotation_range=35,
    width_shift_range=0.35,
    height_shift_range=0.35,
    zoom_range=0.35,
    shear_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest'
)

# Calculate class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight

y_train_labels = np.argmax(y_train, axis=1)
classes = np.unique(y_train_labels)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_labels)
class_weight_dict = {i: class_weights[i] for i in range(len(classes))}

print(f"\n  Class weights (to handle imbalance):")
class_names_map = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}
for class_idx, weight in class_weight_dict.items():
    print(f"    {class_names_map[class_idx]:12}: {weight:.4f}")

# Train model with callbacks for balanced learning (training & validation accuracy both increase)
print(f"\n  Training for 200 epochs with validation monitoring...")
print(f"  Goal: Balanced training - predict all 3 classes (Benign, Malignant, Normal)")

early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=25, 
    restore_best_weights=True, 
    verbose=1, 
    min_delta=0.002
)

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, 'cnn_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Learning rate reduction to fine-tune when accuracy plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=6, 
    min_lr=0.00001, 
    verbose=1
)

history = cnn_model.fit(
    train_datagen.flow(X_train, y_train, batch_size=8),
    epochs=200,
    validation_data=(X_test, y_test),
    callbacks=[model_checkpoint, early_stopping, reduce_lr],
    verbose=1,
    steps_per_epoch=len(X_train)//8
)

print(f"\n✓ Model training completed!")

# Save training history
with open(os.path.join(MODEL_DIR, 'cnn_history.pckl'), 'wb') as f:
    pickle.dump(history.history, f)

# Step 4: Evaluate Model
print("\n[4/4] Evaluating model...")

predictions = cnn_model.predict(X_test, verbose=0)
predict_labels = np.argmax(predictions, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_labels, predict_labels) * 100
precision = precision_score(y_test_labels, predict_labels, average='macro') * 100
recall = recall_score(y_test_labels, predict_labels, average='macro') * 100
f1 = f1_score(y_test_labels, predict_labels, average='macro') * 100

print(f"✓ Model evaluation completed!")
print(f"  - Accuracy:  {accuracy:.2f}%")
print(f"  - Precision: {precision:.2f}%")
print(f"  - Recall:    {recall:.2f}%")
print(f"  - F1-Score:  {f1:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_labels, predict_labels)
print(f"\nConfusion Matrix:")
print(conf_matrix)

# Plot results with detailed metrics
fig = plt.figure(figsize=(16, 10))

# Create grid for subplots
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, :])  # Accuracy - full width
ax2 = fig.add_subplot(gs[1, :])  # Loss - full width
ax3 = fig.add_subplot(gs[2, 0])  # Confusion Matrix
ax4 = fig.add_subplot(gs[2, 1])  # Summary metrics

# 1. Training & Validation Accuracy with values labeled
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(1, len(train_acc) + 1)

ax1.plot(epochs_range, train_acc, 'b-o', linewidth=2, markersize=6, label='Training Accuracy')
ax1.plot(epochs_range, val_acc, 'r-s', linewidth=2, markersize=6, label='Validation Accuracy')

# Add value labels on key points
ax1.text(len(train_acc), train_acc[-1], f'{train_acc[-1]*100:.2f}%', 
         fontsize=10, fontweight='bold', color='blue', ha='right')
ax1.text(len(val_acc), val_acc[-1], f'{val_acc[-1]*100:.2f}%', 
         fontsize=10, fontweight='bold', color='red', ha='right')

# Add best accuracy markers
best_train_acc = max(train_acc)
best_val_acc = max(val_acc)
best_train_epoch = train_acc.index(best_train_acc) + 1
best_val_epoch = val_acc.index(best_val_acc) + 1

ax1.axhline(y=best_train_acc, color='blue', linestyle='--', alpha=0.3)
ax1.axhline(y=best_val_acc, color='red', linestyle='--', alpha=0.3)
ax1.scatter([best_train_epoch], [best_train_acc], color='blue', s=150, marker='*', zorder=5)
ax1.scatter([best_val_epoch], [best_val_acc], color='red', s=150, marker='*', zorder=5)

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Training & Validation Accuracy', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.05])

# 2. Training & Validation Loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

ax2.plot(epochs_range, train_loss, 'g-o', linewidth=2, markersize=6, label='Training Loss')
ax2.plot(epochs_range, val_loss, 'orange', marker='s', linewidth=2, markersize=6, label='Validation Loss')

# Add value labels on key points
ax2.text(len(train_loss), train_loss[-1], f'{train_loss[-1]:.4f}', 
         fontsize=10, fontweight='bold', color='green', ha='right')
ax2.text(len(val_loss), val_loss[-1], f'{val_loss[-1]:.4f}', 
         fontsize=10, fontweight='bold', color='orange', ha='right')

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Model Training & Validation Loss', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3)

# 3. Confusion Matrix
class_names = ['Benign', 'Malignant', 'Normal']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', 
            xticklabels=class_names, yticklabels=class_names, ax=ax3, cbar_kws={'label': 'Count'})
ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Label', fontsize=11)
ax3.set_xlabel('Predicted Label', fontsize=11)

# 4. Summary Metrics Text
summary_text = f"""
TRAINING SUMMARY
{'='*40}

Total Epochs: {len(train_acc)}

FINAL METRICS:
  Training Accuracy:   {train_acc[-1]*100:.2f}%
  Validation Accuracy: {val_acc[-1]*100:.2f}%
  Training Loss:       {train_loss[-1]:.4f}
  Validation Loss:     {val_loss[-1]:.4f}

BEST METRICS:
  Best Train Acc:  {best_train_acc*100:.2f}% (Epoch {best_train_epoch})
  Best Val Acc:    {best_val_acc*100:.2f}% (Epoch {best_val_epoch})

GENERALIZATION:
  Accuracy Gap:    {(train_acc[-1]-val_acc[-1])*100:.2f}%
  
MODEL PERFORMANCE:
  Accuracy:  {accuracy:.2f}%
  Precision: {precision:.2f}%
  Recall:    {recall:.2f}%
  F1-Score:  {f1:.2f}%
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax4.axis('off')

plt.suptitle('Cancer Detection Model - Training Analysis', fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_results.png'), dpi=100, bbox_inches='tight')
print(f"\n✓ Training results saved to: {os.path.join(MODEL_DIR, 'training_results.png')}")

# Don't block on plt.show() - just close the figure
plt.close()

print("\n" + "="*70)
print("✓ TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\nModel saved at: {os.path.join(MODEL_DIR, 'cnn_model.h5')}")
print(f"\nYou can now run the Flask server with:")
print(f"  python app.py")
print("="*70 + "\n")
print(plt.show)