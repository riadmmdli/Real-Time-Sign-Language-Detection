import cv2
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Load model
model = load_model("C:/Users/riadm/Desktop/Real Time Sign Language Detection/sign_language_mobilenetv2_model.h5")

# Set constants
offset = 20
imgSize = 300
labels = ['Hello', 'I Love You', 'No', 'Okay', 'Thank you', 'Yes']

# Webcam and detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # White image for background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Preprocess for prediction
        imgInput = cv2.resize(imgWhite, (224, 224))  # Resize for model input
        imgInput = imgInput.astype("float32") / 255.0
        imgInput = np.expand_dims(imgInput, axis=0)

        # Prediction
        prediction = model.predict(imgInput)
        index = np.argmax(prediction)
        confidence = np.max(prediction)

        # Display result
        label = f"{labels[index]} ({confidence*100:.1f}%)"
        cv2.putText(img, label, (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(img, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 2)

    # Show camera feed
    cv2.imshow("Webcam", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
