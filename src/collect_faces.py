import cv2
import time
import shutil
from pathlib import Path
from camera import Camera
from face_detect import FaceDetector

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
FACES_PATH = DATA_PATH / "faces"

FACES_PATH.mkdir(parents=True, exist_ok=True)


def get_registered_users():
    return [d.name for d in FACES_PATH.iterdir() if d.is_dir()]


def collect_data(user_name, mode="ekle"):
    user_dir = FACES_PATH / user_name

    if mode == "guncelle" and user_dir.exists():
        shutil.rmtree(user_dir)

    user_dir.mkdir(parents=True, exist_ok=True)

    cam = Camera()
    detector = FaceDetector()

    count = 0
    max_count = 50

    print(f"📸 Kayıt başlıyor: {user_name}")
    time.sleep(2)

    try:
        while count < max_count:
            ret, frame = cam.read()
            if not ret:
                continue

            face_img, bbox = detector.detect_and_crop(frame)

            if bbox is not None:
                count += 1
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                img_path = user_dir / f"{user_name}_{count}.jpg"
                cv2.imwrite(str(img_path), gray_face)
                print(f"{count}/{max_count} kaydedildi")
                time.sleep(0.2)

    finally:
        cam.release()
        print(f"✅ Kayıt tamamlandı → data/faces/{user_name}")


def main_menu():
    while True:
        users = get_registered_users()
        print("\n1-Yeni Ekle | 2-Güncelle | 3-Çıkış")
        secim = input("Seçim: ")

        if secim == "1":
            name = input("İsim: ").strip()
            if name:
                collect_data(name)
        elif secim == "2" and users:
            for i, u in enumerate(users, 1):
                print(f"{i}- {u}")
            idx = int(input("No: ")) - 1
            collect_data(users[idx], mode="guncelle")
        elif secim == "3":
            break


if __name__ == "__main__":
    main_menu()
