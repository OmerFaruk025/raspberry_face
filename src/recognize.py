import cv2
import os

from face_detect import FaceDetector


# -----------------------------
# MODEL & LABEL YÜKLEME
# -----------------------------

MODEL_PATH = "lbph_model.yml"
LABEL_PATH = "labels.txt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("trainer.yml bulunamadı")

if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError("labels.txt bulunamadı")

# LBPH modeli yükle
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# id -> isim eşleşmesi
labels = {}
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        idx, name = line.strip().split(":")
        labels[int(idx)] = name


# -----------------------------
# YÜZ DEDEKTÖR
# -----------------------------

detector = FaceDetector()


# -----------------------------
# KAMERA
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı")


print("🎥 Canlı tanıma başladı | Çıkış: Q")


# -----------------------------
# ANA DÖNGÜ
# -----------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # yüz bul + crop al
    face_img, bbox = detector.detect_and_crop(frame)

    if face_img is not None:
        x, y, w, h = bbox

        # grayscale (performans + LBPH şartı)
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # boyut eşitle (train ile aynı)
        gray_face = cv2.resize(gray_face, (200, 200))

        # tahmin
        label_id, confidence = recognizer.predict(gray_face)

        # confidence küçükse daha iyi (LBPH mantığı)
        if confidence < 80:
            name = labels.get(label_id, "Unknown")
            text = f"{name} ({int(confidence)})"
            color = (0, 255, 0)
        else:
            text = "Unknown"
            color = (0, 0, 255)

        # yüz kutusu
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # isim
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# TEMİZLİK
# -----------------------------

cap.release()
cv2.destroyAllWindows()
