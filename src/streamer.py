import cv2
from flask import Flask, Response

app = Flask(__name__)

# Laptop kamerasını başlat (0, 1 veya 2 denenebilir)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Görüntüyü JPEG formatına çevir
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Görüntüyü stream formatına sok
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video_feed():
    # Bu adrese gelenlere görüntüyü gönder
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🚀 Streamer başlatıldı!")
    print("Kanka Pi'deki kodlara bu laptopun IP'sini (192.168.1.47) yazmayı unutma.")
    # host='0.0.0.0' sayesinde ağdaki Pi sana ulaşabilir
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)