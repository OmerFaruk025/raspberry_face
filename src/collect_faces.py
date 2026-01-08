import cv2
import time
import shutil
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

# -----------------------------
# AYARLAR
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "faces"
DATA_PATH.mkdir(parents=True, exist_ok=True)

MAX_COUNT = 50

# -----------------------------
# YARDIMCI FONKSİYONLAR
# -----------------------------
def get_registered_users():
    return [d.name for d in DATA_PATH.iterdir() if d.is_dir()]

# -----------------------------
# ANA KAYIT FONKSİYONU
# -----------------------------
def collect_data(user_name, mode="ekle"):
    user_dir = DATA_PATH / user_name

    if mode == "guncelle" and user_dir.exists():
        print(f"🔄 '{user_name}' eski verileri siliniyor...")
        shutil.rmtree(user_dir)

    user_dir.mkdir(parents=True, exist_ok=True)

    cam = Camera()               # PiCam
    detector = FaceDetector()    # Haar + filtreli

    count = 0
    print(f"📸 Kayıt başlıyor: {user_name}")
    time.sleep(1)

    try:
        while count < MAX_COUNT:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            face_img, bbox = detector.detect_and_crop(frame)

            if bbox is None:
                time.sleep(0.03)
                continue

            x, y, w, h = bbox

            # --- EK GÜVENLİK ---
            if w < 80 or h < 80:
                continue

            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (200, 200))

            count += 1
            img_path = user_dir / f"{user_name}_{count}.jpg"
            cv2.imwrite(str(img_path), gray_face)

            print(f"{count}/{MAX_COUNT} kaydedildi")

            # Aynı kareyi tekrar almamak için kısa bekleme
            time.sleep(0.25)

    finally:
        cam.release()
        print(f"✅ Kayıt tamamlandı → data/faces/{user_name}")

# -----------------------------
# MENÜ
# -----------------------------
def main_menu():
    while True:
        users = get_registered_users()

        print("\n" + "=" * 35)
        print("🛡️ Pi-FaceID | VERİ TOPLAMA")
        print("=" * 35)

        if users:
            for i, u in enumerate(users, 1):
                print(f"{i}- {u}")
            print("1-Yeni Ekle | 2-Güncelle | 3-Çıkış")
        else:
            print("⚠️ Kayıt yok.")
            print("1-Yeni Ekle | 3-Çıkış")

        secim = input("Seçim: ").strip()

        if secim == "1":
            name = input("İsim: ").strip()
            if name:
                collect_data(name, mode="ekle")

        elif secim == "2" and users:
            idx = input("No: ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(users):
                collect_data(users[int(idx) - 1], mode="guncelle")

        elif secim == "3":
            break

# -----------------------------
if __name__ == "__main__":
    main_menu()
