import cv2
import os
import numpy as np
from pathlib import Path

# -----------------------------
# YOLLAR
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent # src
ROOT_DIR = BASE_DIR.parent # Proje ana klasörü

# BURASI KRİTİK: data/faces içine bakması gerekiyor boş ise dolduracak.
DATA_PATH = ROOT_DIR / "data" / "faces"
MODEL_SAVE_PATH = str(ROOT_DIR / "lbph_model.yml")
LABEL_SAVE_PATH = str(ROOT_DIR / "labels.txt")

recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(path):
    face_samples = []
    ids = []
    labels_map = {}
    current_id = 0

    if not path.exists():
        print(f"❌ HATA: {path} yolu bulunamadı! Önce yüz kaydı yapmalısın.")
        return [], [], {}

    # data/faces altındaki klasörleri (kişileri) döner
    for person_dir in path.iterdir():
        if person_dir.is_dir():
            name = person_dir.name
            if name not in labels_map:
                labels_map[name] = current_id
                current_id += 1
            
            print(f"📂 '{name}' klasörü işleniyor...")
            
            # Kişi klasörünün içindeki resimleri bulur
            for img_path in person_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    try:
                        img_array = np.fromfile(str(img_path), np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                        
                        if img is not None:
                            face_samples.append(img)
                            ids.append(labels_map[name])
                    except Exception as e:
                        print(f"⚠️ Dosya okunamadı {img_path.name}: {e}")

    return face_samples, ids, labels_map

print(f"🧠 Eğitim başladı. Kaynak: {DATA_PATH}")

faces, ids, labels_map = get_images_and_labels(DATA_PATH)

if len(faces) == 0:
    print("❌ HATA: Eğitilecek veri bulunamadı! Klasörleri kontrol et.")
    print(f"Bakılan yol: {DATA_PATH}")
    exit()

# Eğit ve kaydet
recognizer.train(faces, np.array(ids))
recognizer.write(MODEL_SAVE_PATH)

with open(LABEL_SAVE_PATH, "w", encoding="utf-8") as f:
    for name, idx in labels_map.items():
        f.write(f"{idx}:{name}\n")

print(f"✅ Eğitim Başarılı Model '{ROOT_DIR}' içine kaydedildi.")