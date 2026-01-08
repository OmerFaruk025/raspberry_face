import cv2
import os
from picamera2 import Picamera2
import numpy as np

# =====================
# AYARLAR
# =====================
DATASET_DIR = "data/faces"
CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
THRESHOLD = 60  # LBPH için: DÜŞÜK = DAHA İYİ

# =====================
# FACE DETECTOR
# =====================
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError("❌ Haar Cascade yüklenemedi")

print("✅ Haar Cascade hazır")

# =====================
# MODEL YÜKLE
# =====================
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}

current_label = 0

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label += 1

recognizer.train(faces, np.array(labels))
print(f"✅ Model eğitildi | Kişi sayısı: {len(label_map)}")

# =====================
# KAMERA
# =====================
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
)
picam2.start()

print("📸 Kamera başladı")

# =====================
# ANA DÖNGÜ
# =====================
while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces_detected = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    status_text = "FACE: NOT FOUND"

    for (x, y, w, h) in faces_detected:
        face_img = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face_img)
        name = label_map.get(label, "Unknown")

        # LBPH mantığı:
        # confidence DÜŞÜKSE daha iyi
        if confidence < THRESHOLD:
            result = "ACCEPTED"
            color = (0, 255, 0)
        else:
            result = "REJECTED"
            name = "Unknown"
            color = (0, 0, 255)

        status_text = (
            f"FACE: DETECTED | "
            f"MATCH: {name} | "
            f"SCORE: {confidence:.2f} | "
            f"TH: {THRESHOLD} | "
            f"STATUS: {result}"
        )

        # ÇERÇEVE
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # İSİM
        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # ALT BİLGİ BANDI
    cv2.putText(
        frame,
        status_text,
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1
    )

    cv2.imshow("FACE RECOGNITION DEBUG", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
picam2.stop()
