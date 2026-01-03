import cv2
from flask import Flask, Response

app = Flask(__name__)

# Laptop kamerasını başlat
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Kameradan kare oku
        success, frame = camera.read()
        if not success:
            break
        else:
            # Görüntüyü JPEG formatına çevir (Ağda hızlı gitsin diye)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Görüntüyü stream formatına sok
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video_feed():
    # Bu adrese (http://IP:5000/video) gelenlere görüntüyü gönder
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🚀 Streamer başlatıldı!")
    print("Kanka Pi'deki kodlara bu laptopun IP'sini yazmayı unutma.")
    # host='0.0.0.0' demek, ağdaki diğer cihazlar (Pi gibi) bana ulaşabilsin demek
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True) 
    
    #http://localhost:5000/video