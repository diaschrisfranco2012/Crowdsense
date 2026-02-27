import cv2
import time
import psutil
import threading
from flask import Flask, Response
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db
import cloudinary
import cloudinary.uploader
from twilio.rest import Client

# ==========================================
# 1. CONFIGURATION
# ==========================================
FIREBASE_DB_URL = 'Firebase URL here bro' 
KEY_PATH = "firebase_key.json"

cloudinary.config( 
  cloud_name = "YOUR_CLOUD_NAME", 
  api_key = "YOUR_API_KEY", 
  api_secret = "YOUR_API_SECRET" 
)

TWILIO_SID = "Enter sid here"
TWILIO_TOKEN = "Enter Token here"
FROM_NUMBER = "twilio phone number enter here"
TO_NUMBER = "+919321536870"

CAMERA_INDEX = 0
WARNING_LIMIT = 30
CRITICAL_LIMIT = 50 
BUFFER_SIZE = 45 

# ==========================================
#  2. INITIALIZATION
# ==========================================
print("Connecting to Firebase...")
cred = credentials.Certificate(KEY_PATH)
firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
db_ref = db.reference('crowd_monitor/zone_A')

try:
    twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
except Exception as e:
    twilio_client = None

print("Loading YOLO AI...")
model = YOLO('yolo11n.pt')
cap = cv2.VideoCapture(CAMERA_INDEX)

# Flask & Threading setup for Live Stream
app = Flask(__name__)
output_frame = None
lock = threading.Lock()

def make_emergency_call(message):
    if twilio_client:
        try:
            twilio_client.calls.create(
                twiml=f'<Response><Say voice="alice">{message}</Say></Response>',
                to=TO_NUMBER,
                from_=FROM_NUMBER
            )
        except Exception as e:
            print(f" Call Failed: {e}")

# ==========================================
# 3. THE AI & FIREBASE THREAD
# ==========================================
def process_camera():
    global output_frame, lock
    
    consecutive_critical_frames = 0
    last_alert_time = 0
    last_firebase_ping = 0

    while True:
        ret, frame = cap.read()
        if not ret: continue

        # Run YOLO AI
        results = model.predict(frame, conf=0.10, classes=[0], verbose=False)
        total_persons = 0
        
        # Draw bounding boxes on the frame for the live stream
        if results[0].boxes:
            for box in results[0].boxes:
                total_persons += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                box_color = (0, 255, 0) # Green
                if total_persons > WARNING_LIMIT: box_color = (0, 165, 255) # Orange
                if total_persons > CRITICAL_LIMIT: box_color = (0, 0, 255) # Red
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Draw UI overlay on the video stream
        cv2.putText(frame, f"Live Count: {total_persons}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Buffer Logic
        if total_persons > CRITICAL_LIMIT:
            consecutive_critical_frames += 1
        else:
            consecutive_critical_frames = 0 

        # --- ALERT LOGIC ---
        if consecutive_critical_frames > BUFFER_SIZE and (time.time() - last_alert_time) > 60:
            print("CRITICAL ALERT! Uploading evidence...")
            current_timestamp_ms = int(time.time() * 1000)

            cv2.imwrite("temp_evidence.jpg", frame)
            upload_result = cloudinary.uploader.upload("temp_evidence.jpg", folder="crowdsense_alerts")
            
            new_log = {
                "timestamp": current_timestamp_ms,
                "description": "Critical Stampede Risk Detected",
                "type": "Pending",
                "people_count": total_persons,
                "image_url": upload_result['secure_url']
            }
            db_ref.child('history').push(new_log)
            db_ref.update({"status": "CRITICAL RISK", "last_alert_timestamp": current_timestamp_ms})
            make_emergency_call("Critical Alert. Stampede risk detected. Immediate action required.")

            last_alert_time = time.time()
            consecutive_critical_frames = 0

        # --- FIREBASE PING ---
        if time.time() - last_firebase_ping > 2:
            disk = psutil.disk_usage('/')
            status_text = "Normal"
            if total_persons > WARNING_LIMIT: status_text = "High Density"
            if time.time() - last_alert_time < 30: status_text = "CRITICAL RISK"

            db_ref.update({
                "live_count": total_persons,
                "pi_is_online": True,
                "pi_storage_used": round(disk.used / (1024 ** 3), 2),
                "status": status_text
            })
            last_firebase_ping = time.time()

        # Safely update the global frame for Flask to stream
        with lock:
            output_frame = frame.copy()

# ==========================================
#  4. FLASK STREAMING SERVER
# ==========================================
def generate_feed():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            # Encode frame to JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        # Yield the output frame in byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/')
def index():
    return "CrowdSense Camera is Live!"

if __name__ == '__main__':
    # Start AI processing in a background thread
    t = threading.Thread(target=process_camera)
    t.daemon = True
    t.start()
    
    # Start Flask Server on Port 5000
    print("📡 Starting Live Stream Server on Port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)