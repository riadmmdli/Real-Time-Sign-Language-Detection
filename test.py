import cv2
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Load model
model = load_model("C:/Users/riadm/Desktop/Real Time Sign Language Detection/sign_language_mobilenetv2_model.h5")

# Constants
offset = 20
imgSize = 300
labels = ['Call me', 'Dislike', 'Hello', 'I Love You', 'Like', 'No', 'Okay', 'Peace', 'Thank you', 'Yes']

# Webcam and detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)
    height, width, _ = img.shape

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Safe crop coordinates
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(width, x + w + offset)
        y2 = min(height, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            # 🟡 Show message if hand is too close to frame edge
            cv2.putText(img, "Hand too close to edge", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # White canvas
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            try:
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

                # Preprocess
                imgInput = cv2.resize(imgWhite, (224, 224)).astype("float32") / 255.0
                imgInput = np.expand_dims(imgInput, axis=0)

                # Predict
                prediction = model.predict(imgInput)
                index = np.argmax(prediction)
                confidence = np.max(prediction)

                # Show result
                label = f"{labels[index]} ({confidence*100:.1f}%)"
                cv2.putText(img, label, (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            except:
                cv2.putText(img, "Processing error", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    else:
        cv2.putText(img, "No hand detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    # Show webcam feed
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
