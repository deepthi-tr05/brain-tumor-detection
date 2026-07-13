import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# ── Keras API: resolved through tf.keras so Pylance finds the type stubs ──────
# Using tf.keras.* attribute access avoids "cannot resolve import" linter errors.
keras  = tf.keras
layers = tf.keras.layers
models = tf.keras.models
VGG16  = tf.keras.applications.VGG16
EarlyStopping     = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
ModelCheckpoint   = tf.keras.callbacks.ModelCheckpoint

# ── ImageDataGenerator: present in TF 2.15, deprecated in TF 2.16+ / Keras 3 ─
try:
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator  # type: ignore[attr-defined]
except AttributeError:
    # Keras 3 / TF 2.16+: provide a minimal shim so the rest of the file runs
    import math as _math
    class ImageDataGenerator:  # type: ignore[no-redef]
        """Minimal shim for Keras 3 compatibility."""
        def __init__(self, **kwargs): pass
        def flow(self, X, y, batch_size=32):
            class _Gen:
                def __init__(self, X, y, bs): self.X, self.y, self.bs = X, y, bs
                def __len__(self): return _math.ceil(len(self.X) / self.bs)
                def __iter__(self):
                    for i in range(0, len(self.X), self.bs):
                        yield self.X[i:i+self.bs], self.y[i:i+self.bs]
            return _Gen(X, y, batch_size)

import matplotlib
matplotlib.use('Agg')   # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 4
CLASS_NAMES = ['no_tumor', 'pituitary_tumor', 'meningioma_tumor', 'glioma_tumor']

def create_sample_data():
    """Create sample training data for demonstration"""
    print("[INFO] Creating sample training data...")
    
    for split in ['train', 'test']:
        for class_name in CLASS_NAMES:
            os.makedirs(f'data/{split}/{class_name}', exist_ok=True)
    
    # Create synthetic images for each class
    samples_per_class = {'train': 50, 'test': 15}
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        for split, count in samples_per_class.items():
            for i in range(count):
                # Create base image with noise
                img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                
                # Add patterns based on tumor type
                center = (IMG_SIZE//2, IMG_SIZE//2)
                
                if class_name == 'pituitary_tumor':
                    # Circular pattern
                    cv2.circle(img, center, 50, (200, 200, 200), -1)
                    cv2.circle(img, center, 30, (100, 100, 100), -1)
                elif class_name == 'meningioma_tumor':
                    # Rectangular pattern
                    x, y = center[0]-40, center[1]-40
                    cv2.rectangle(img, (x, y), (x+80, y+80), (200, 200, 200), -1)
                elif class_name == 'glioma_tumor':
                    # Irregular pattern
                    axes = (50, 35)
                    cv2.ellipse(img, center, axes, 0, 0, 360, (200, 200, 200), -1)
                else:
                    # No tumor - smooth pattern
                    img = cv2.GaussianBlur(img, (15, 15), 0)
                
                # Add noise
                noise = np.random.randint(0, 40, img.shape, dtype=np.uint8)
                img = cv2.add(img, noise)
                
                # Save image
                save_path = f'data/{split}/{class_name}/sample_{i}.jpg'
                cv2.imwrite(save_path, img)
    
    print(f"[OK] Created sample images in data/ folder")
    print(f"   Training: {samples_per_class['train'] * 4} images")
    print(f"   Testing: {samples_per_class['test'] * 4} images")

def load_and_preprocess_data():
    """Load images from folders and preprocess"""
    print("[INFO] Loading and preprocessing data...")
    
    X = []
    y = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        folder_path = f'data/train/{class_name}'
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        X.append(img)
                        y.append(class_idx)
    
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y)
    
    print(f"[OK] Loaded {len(X)} training images")
    print(f"   Class distribution: {np.bincount(y)}")
    
    return X, y

def create_cnn_model():
    """Create a CNN model from scratch."""
    # Use keras.Input() as the first layer — the modern Keras pattern.
    # Avoids the deprecated `input_shape` argument inside Conv2D which
    # causes a Pylance red underline and a DeprecationWarning at runtime.
    model = models.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),

        # Dense Layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

    return model

def create_vgg16_transfer_model():
    """Create a transfer learning model using VGG16 (like in the video)"""
    # Load pre-trained VGG16 without top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze base model layers
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def train_model():
    """Main training function"""
    print("=" * 60)
    print("[BRAIN] Brain Tumor Detection - CNN Model Training")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists('data/train/no_tumor') or len(os.listdir('data/train/no_tumor')) < 5:
        print("[WARN] Not enough training data found. Creating sample data...")
        # create_sample_data()
    
    # Load data
    X, y = load_and_preprocess_data()
    
    if len(X) == 0:
        print("[ERROR] No training data found!")
        return None
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Validation samples: {len(X_val)}")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create data generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    val_generator = ImageDataGenerator().flow(X_val, y_val, batch_size=BATCH_SIZE)
    
    # Create model (using VGG16 for better accuracy)
    print("\n[BUILD] Creating VGG16 Transfer Learning Model...")
    model = create_vgg16_transfer_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    os.makedirs('models', exist_ok=True)
    
    # Train model
    print("\n[START] Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"\n[OK] Validation Accuracy: {val_accuracy:.4f}")
    print(f"[OK] Validation Loss: {val_loss:.4f}")
    
    # Save model in both .h5 and .keras format for compatibility
    model.save('models/brain_tumor_cnn_model.h5')
    model.save('models/brain_tumor_cnn_model.keras')
    print("[OK] Model saved to models/brain_tumor_cnn_model.h5")
    print("[OK] Model also saved to models/brain_tumor_cnn_model.keras")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()
    print("[OK] Training history saved to models/training_history.png")

if __name__ == '__main__':
    train_model()