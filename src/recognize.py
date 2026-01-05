import cv2
import os
from camera import Camera
from face_detect import FaceDetector

# -----------------------------
# YOLLAR & AYARLAR
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lbph_model.yml")
LABEL_PATH = os.path.join(BASE_DIR, "labels.txt")

# KANKA DİKKAT: Laptop IP'ni yeni haliyle güncelledim (.128)
LAPTOP_IP = "192.168.1.128" 
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
cam = Camera(source=stream_url)

print("🎥 Canlı tanıma başladı | Durdurmak için CTRL+C yap kanka")

# -----------------------------
# ANA DÖNGÜ
# -----------------------------
try:
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        # Yüz bul + crop al
        face_img, bbox = detector.detect_and_crop(frame)

        if face_img is not None:
            x, y, w, h = bbox

            # Grayscale (LBPH şartı)
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (200, 200))

            # Tahmin yap
            label_id, confidence = recognizer.predict(gray_face)

            # Sonucu terminale yaz (Ekran olmadığı için buradan takip ediyoruz)
            if confidence < 80:
                name = labels.get(label_id, "Unknown")
                print(f"✅ Tanındı: {name} | Güven: {int(confidence)}")
            else:
                print(f"👤 Bilinmeyen biri var! (Güven: {int(confidence)})")

        # SSH üzerinden hata almamak için cv2.imshow ve waitKey iptal edildi!
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

except KeyboardInterrupt:
    print("\n👋 Sistem kapatılıyor kanka...")

finally:
    # Temizlik
    cam.release()
    cv2.destroyAllWindows()