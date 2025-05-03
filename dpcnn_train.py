import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
MODEL_NAME = "DPCNN"
MODEL_PATH = "sign_language_dpcnn_model.h5"
DATA_DIR = "C:/Users/riadm/Desktop/Real Time Sign Language Detection/Data"

# Preprocessing: blurred image for background stream
def dual_input_generator(generator):
    while True:
        images, labels = next(generator)
        blurred_images = np.array([cv2.GaussianBlur(img, (15, 15), 0) for img in images])
        yield [images, blurred_images], labels

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen_base = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen_base = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

train_gen = dual_input_generator(train_gen_base)
val_gen = dual_input_generator(val_gen_base)

# Input shapes
input_shape = (IMG_SIZE, IMG_SIZE, 3)
num_classes = train_gen_base.num_classes
class_names = list(train_gen_base.class_indices.keys())

# Shared CNN block
def cnn_branch(input_tensor):
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    return x

# Build Model
input_main = Input(shape=input_shape)
input_blur = Input(shape=input_shape)

feat_main = cnn_branch(input_main)
feat_blur = cnn_branch(input_blur)
feat_subtracted = layers.Subtract()([feat_main, feat_blur])

x = layers.Dense(128, activation='relu')(feat_subtracted)
x = layers.Dropout(0.3)(x)
output = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=[input_main, input_blur], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    train_gen,
    steps_per_epoch=len(train_gen_base),
    validation_data=val_gen,
    validation_steps=len(val_gen_base),
    epochs=EPOCHS
)

# Save
model.save(MODEL_PATH)

# 📊 Plot Accuracy and Loss
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

# 🧪 Evaluation: Accuracy, Precision, Recall, F1
val_gen_base.reset()
Y_true = val_gen_base.classes
Y_pred_probs = model.predict(val_gen, steps=len(val_gen_base), verbose=1)
Y_pred = np.argmax(Y_pred_probs, axis=1)

# Metrics
acc = accuracy_score(Y_true, Y_pred)
precision = precision_score(Y_true, Y_pred, average='weighted')
recall = recall_score(Y_true, Y_pred, average='weighted')
f1 = f1_score(Y_true, Y_pred, average='weighted')

print("\n🎯 Model Evaluation on Validation Set:")
print(f"✔️ Accuracy : {acc*100:.2f}%")
print(f"✔️ Precision: {precision*100:.2f}%")
print(f"✔️ Recall   : {recall*100:.2f}%")
print(f"✔️ F1-Score : {f1*100:.2f}%")

# Detailed report
print("\n📋 Classification Report:")
print(classification_report(Y_true, Y_pred, target_names=class_names))
