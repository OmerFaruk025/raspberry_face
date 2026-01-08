import cv2
import time
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# -----------------------------
# AYARLAR
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

MODEL_PATH = str(ROOT_DIR / "lbph_model.yml")
LABEL_PATH = str(ROOT_DIR / "labels.txt")

MATCH_THRESHOLD = 60        # kabul eşiği
FRAME_DELAY = 0.2           # SSH için CPU dostu bekleme

# -----------------------------
# MODEL + LABELS
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

labels = {}
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        idx, name = line.strip().split(":")
        labels[int(idx)] = name

print("✅ Model yüklendi | Kişi sayısı:", len(labels))

# -----------------------------
# FACE DETECTOR
# -----------------------------
detector = FaceDetector()
print("✅ Haar Cascade hazır")

# -----------------------------
# KAMERA
# -----------------------------
cam = Camera()
print("📸 Kamera başladı")

print("\n--- SSH RECOGNITION MODE AKTİF ---")
print("Ctrl + C ile çıkabilirsin\n")

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("⚠️ Frame alınamadı")
            time.sleep(FRAME_DELAY)
            continue

        face_img, bbox = detector.detect_and_crop(frame)

        # -----------------------------
        # YÜZ YOK
        # -----------------------------
        if face_img is None:
            print("👀 FACE: YOK")
            time.sleep(FRAME_DELAY)
            continue

        # -----------------------------
        # YÜZ VAR → TANI
        # -----------------------------
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200))

        label_id, confidence = recognizer.predict(gray)
        match_percent = round(max(0, 100 - confidence), 2)
        name = labels.get(label_id, "Bilinmeyen")

        # -----------------------------
        # KARAR
        # -----------------------------
        if match_percent >= MATCH_THRESHOLD:
            status = "✅ KABUL"
        else:
            status = "❌ RED (Eşik altı)"

        # -----------------------------
        # SSH DEBUG ÇIKTI
        # -----------------------------
        print(
            f"👤 FACE: VAR | "
            f"TAHMİN: {name} | "
            f"SKOR: %{match_percent} | "
            f"EŞİK: %{MATCH_THRESHOLD} | "
            f"DURUM: {status}"
        )

        # -----------------------------
        # GUI NOTU (MONİTOR TAKARSAN)
        # -----------------------------
        # cv2.rectangle(frame,
        #               (bbox[0], bbox[1]),
        #               (bbox[0]+bbox[2], bbox[1]+bbox[3]),
        #               (0, 255, 0), 2)
        #
        # cv2.putText(
        #     frame,
        #     f"{name} %{match_percent}",
        #     (bbox[0], bbox[1] - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     (0, 255, 0),
        #     2
        # )
        #
        # cv2.imshow("FACE RECOGNITION", frame)
        #
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

        time.sleep(FRAME_DELAY)

except KeyboardInterrupt:
    print("\n👋 Çıkış yapıldı")

finally:
    cam.release()
    cv2.destroyAllWindows()
    print("✅ Kamera kapandı")