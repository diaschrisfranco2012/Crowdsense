from flask import Flask, Response, send_from_directory
from pi_stream import VideoCamera # Imports the pi_stream.py
import time

app = Flask(__name__)

# Start the Camera/AI thread immediately
camera_system = VideoCamera()

@app.route('/')
def index():
    return "CrowdSense AI Gateway is Running!"

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera_system),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 📸 SERVE EVIDENCE PHOTOS
@app.route('/evidence/<path:filename>')
def serve_evidence(filename):
    # This lets your dashboard fetch: http://PI_IP:5000/evidence/fall_12345.jpg
    return send_from_directory('static/falls', filename)

if __name__ == '__main__':
    # Run on 0.0.0.0 so it's accessible on Wi-Fi
    app.run(host='0.0.0.0', port=5000, threaded=True)