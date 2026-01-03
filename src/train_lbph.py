import cv2
import os
import numpy as np

# Dosya yollarını Pi-FaceID/src klasörüne göre ayarlıyoruz
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # src klasörü
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data/faces") # Pi-FaceID/data/faces

faces = []
face_labels = []

label_map = {}
current_label = 0

if not os.path.exists(DATA_DIR):
    print(f"❌ Hata: {DATA_DIR} klasörü bulunamadı!")
    exit()

# Kişileri sabit sırayla al
for person in sorted(os.listdir(DATA_DIR)):
    person_path = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person
    print(f"Eğitiliyor: {person} (ID: {current_label})")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        # IMREAD_GRAYSCALE zaten gri okur
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Kanka burada boyutları 200x200'e sabitliyoruz ki hata payı kalmasın
        img = cv2.resize(img, (200, 200))

        faces.append(img)
        face_labels.append(current_label)

    current_label += 1

if len(faces) == 0:
    print("❌ Hiç yüz verisi bulunamadı! Lütfen önce collect_faces.py çalıştır.")
    exit()

# LBPH Ayarları
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

print("🧠 Model eğitiliyor, lütfen bekle...")
recognizer.train(faces, np.array(face_labels))

# Model ve etiketleri src klasörüne (main'in yanına) kaydet
model_save_path = os.path.join(BASE_DIR, "lbph_model.yml")
labels_save_path = os.path.join(BASE_DIR, "labels.txt")

recognizer.save(model_save_path)

# 🔥 LABEL DOSYASINI YAZ
with open(labels_save_path, "w", encoding="utf-8") as f:
    for label, name in label_map.items():
        f.write(f"{label}:{name}\n")

print("✅ MODEL EĞİTİLDİ VE KAYDEDİLDİ")
print(f"Toplam kişi: {len(label_map)}")
print(f"Toplam foto: {len(faces)}")