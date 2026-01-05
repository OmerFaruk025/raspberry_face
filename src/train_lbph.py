import cv2
import os
import numpy as np
from pathlib import Path

# -----------------------------
# YOLLAR (Artık Proje Kökünde!)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent # src klasörü
ROOT_DIR = BASE_DIR.parent # Proje ana klasörü

DATA_PATH = ROOT_DIR / "data"
# Modelleri src içinden çıkarıp ana klasöre (ROOT_DIR) alıyoruz
MODEL_SAVE_PATH = str(ROOT_DIR / "lbph_model.yml")
LABEL_SAVE_PATH = str(ROOT_DIR / "labels.txt")

# -----------------------------
# EĞİTİMCİ HAZIRLIĞI
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(path):
    face_samples = []
    ids = []
    labels_map = {}
    current_id = 0

    if not path.exists():
        print(f"❌ HATA: {path} klasörü bulunamadı!")
        return [], [], {}

    for person_dir in path.iterdir():
        if person_dir.is_dir():
            name = person_dir.name
            if name not in labels_map:
                labels_map[name] = current_id
                current_id += 1
            
            print(f"📂 '{name}' klasörü işleniyor...")
            
            for img_path in person_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    try:
                        # Türkçe karakterli yolları okumak için byte yöntemi
                        img_array = np.fromfile(str(img_path), np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                        
                        if img is not None:
                            face_samples.append(img)
                            ids.append(labels_map[name])
                    except Exception as e:
                        print(f"⚠️ Dosya okunamadı {img_path.name}: {e}")

    return face_samples, ids, labels_map

print("🧠 Eğitim başladı, kök dizine kayıt yapılacak...")

faces, ids, labels_map = get_images_and_labels(DATA_PATH)

if len(faces) == 0:
    print("❌ HATA: Eğitilecek veri bulunamadı!")
    exit()

# Modeli eğit
recognizer.train(faces, np.array(ids))

# Modeli ve etiketleri ANA DİZİNE kaydet
recognizer.write(MODEL_SAVE_PATH)

with open(LABEL_SAVE_PATH, "w", encoding="utf-8") as f:
    for name, idx in labels_map.items():
        f.write(f"{idx}:{name}\n")

print(f"✅ Başardık kral! Dosyalar ana dizine (root) kaydedildi.")
print(f"📁 Model: {MODEL_SAVE_PATH}")
print(f"📁 Etiketler: {LABEL_SAVE_PATH}")