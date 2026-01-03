import cv2
import os
from camera import Camera # Kanka yine senin sınıf
from face_detect import FaceDetector

# -----------------------------
# YOLLAR & AYARLAR
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lbph_model.yml")
LABEL_PATH = os.path.join(BASE_DIR, "labels.txt")

# Laptop IP'ni buraya da yazıyoruz (Diğerleriyle aynı olmalı)
LAPTOP_IP = "192.168.1.47" 
stream_url = f"http://{LAPTOP_IP}:5000/video"

if not os.path.exists(MODEL_PATH):
    print("❌ lbph_model.yml bulunamadı! Önce train_lbph.py çalıştır kanka.")
    exit()

if not os.path.exists(LABEL_PATH):
    print("❌ labels.txt bulunamadı!")
    exit()

# -----------------------------
# MODEL & LABEL YÜKLEME
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

labels = {}
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        idx, name = line.strip().split(":")
        labels[int(idx)] = name

# -----------------------------
# SİSTEMİ BAŞLAT
# -----------------------------
detector = FaceDetector()
cam = Camera(source=stream_url) # Laptop kamerasını buradan yakalıyoruz

print("🎥 Canlı tanıma başladı | Çıkış için 'Q'ya bas kanka")

# -----------------------------
# ANA DÖNGÜ
# -----------------------------
while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        continue

    # Yüz bul + crop al (Senin face_detect metodun)
    face_img, bbox = detector.detect_and_crop(frame)

    if face_img is not None:
        x, y, w, h = bbox

        # Grayscale (LBPH şartı)
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Boyut eşitle (Train aşamasındaki 200x200 ile aynı olmalı)
        gray_face = cv2.resize(gray_face, (200, 200))

        # Tahmin yap
        label_id, confidence = recognizer.predict(gray_face)

        # Confidence (Eşik) Ayarı: LBPH'da sayı düştükçe doğruluk artar
        if confidence < 80:
            name = labels.get(label_id, "Unknown")
            text = f"{name} ({int(confidence)})"
            color = (0, 255, 0) # Yeşil - Tanıdı
        else:
            text = "Unknown"
            color = (0, 0, 255) # Kırmızı - Yabancı

        # Yüz kutusu ve Metin (Senin görsel tasarımın)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, 
            text, 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            color, 
            2
        )

    # Görüntüyü göster
    cv2.imshow("Pi-FaceID | Live Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Temizlik
cam.release()
cv2.destroyAllWindows()