import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ⚙️ Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
DATA_DIR = "C:/Users/riadm/Desktop/Real Time Sign Language Detection/Data"
MODEL_PATH = "sign_language_mobilenetv2_model.h5"
MODEL_NAME = "MobileNetV2"

# 1. Image Data Generator (Augmentation + Normalization)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 2. Load Training and Validation Data
train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 3. Load MobileNetV2 base
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False  # Freeze base layers

# 4. Build Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# 5. Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Early Stopping
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 7. Train the Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# 8. Save the Best Model
model.save(MODEL_PATH)

# 9. Plot Accuracy and Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title(f'{MODEL_NAME} Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'{MODEL_NAME} Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
    
plt.tight_layout()
plt.show()

# 10. Evaluation: Accuracy, Precision, Recall, F1-Score
val_gen.reset()
Y_true = val_gen.classes
Y_pred = model.predict(val_gen)
Y_pred_classes = np.argmax(Y_pred, axis=1)
labels = list(val_gen.class_indices.keys())

# General scores
acc = accuracy_score(Y_true, Y_pred_classes)
precision = precision_score(Y_true, Y_pred_classes, average='weighted')
recall = recall_score(Y_true, Y_pred_classes, average='weighted')
f1 = f1_score(Y_true, Y_pred_classes, average='weighted')

print("\n🎯 Model Evaluation on Validation Set:")
print(f"✔️ Accuracy : {acc*100:.2f}%")
print(f"✔️ Precision: {precision*100:.2f}%")
print(f"✔️ Recall   : {recall*100:.2f}%")
print(f"✔️ F1-Score : {f1*100:.2f}%")

# Classification report
print("\n🧪 Per-Class Performance:")
print(classification_report(Y_true, Y_pred_classes, target_names=labels))