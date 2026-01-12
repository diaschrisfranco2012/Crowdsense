import cv2
import time
import numpy as np
from ultralytics import YOLO
from fluvio import Fluvio

# ==========================================
# ⚙️ CONFIGURATION SETTINGS
# ==========================================
TOPIC_NAME = "crowd-stream"  
CAMERA_INDEX = 0             # 0 = Default USB Cam. Change if using Pi Cam.

# --- CROWD THRESHOLDS 
# Conf is 0.10,
WARNING_LIMIT = 30
CRITICAL_LIMIT = 50   

# --- TIME BUFFER (False Alarm Killer) ---
FPS_LIMIT = 15            # Limit FPS to keep Pi cool
REQUIRED_SECONDS = 3.0    # Crowd must be dense for 3 seconds to trigger ALERT
BUFFER_SIZE = int(FPS_LIMIT * REQUIRED_SECONDS)

# ==========================================
#  INITIALIZATION
# ==========================================
print(f"🔌 Connecting to Fluvio Topic: {TOPIC_NAME}...")
try:
    fluvio = Fluvio.connect()
    producer = fluvio.topic_producer(TOPIC_NAME)
    print("Fluvio Connected!")
except Exception as e:
    print(f"Fluvio Error: {e}")
    print("Run 'fluvio cluster start' first!")
    exit()

print("Loading YOLOv11 Nano Model...")
model = YOLO('yolo11n.pt') 

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

# Global Variables
consecutive_critical_frames = 0
last_alert_time = 0

print("Starting Stable Stream... Press Ctrl+C to stop.")

# ==========================================
# 🔄 MAIN LOOP
# ==========================================
while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret: break

    # 1. RUN YOLO (Aggressive Mode: conf=0.10)
    results = model.predict(frame, conf=0.10, iou=0.85, imgsz=320, classes=[0], verbose=False)
    
    total_persons = 0
    overlay = frame.copy()

    # 2. PROCESS DETECTIONS
    if results[0].boxes:
        for box in results[0].boxes:
            total_persons += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # --- COLOR LOGIC ---
            box_color = (0, 255, 0) # Green
            if total_persons > WARNING_LIMIT: box_color = (0, 165, 255) # Orange
            if total_persons > CRITICAL_LIMIT: box_color = (0, 0, 255)   # Red

            # Draw Filled Box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)

    # Apply Transparency on people detection
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # 3. FALSE ALARM FILTER (Time Buffer)
    if total_persons > CRITICAL_LIMIT:
        consecutive_critical_frames += 1
    else:
        # Reset immediately if density drops
        consecutive_critical_frames = 0 

    # Determine Status
    display_status = "Normal"
    status_color = (0, 255, 0) # Green

    # Only go CRITICAL if it persisted for the full buffer duration
    if consecutive_critical_frames > BUFFER_SIZE:
        display_status = "CRITICAL RISK"
        status_color = (0, 0, 255) # Red
        
        # Debug Print (Optional)
        if time.time() - last_alert_time > 5:
            print(f"ALERT! Stampede Risk Confirmed! Count: {total_persons}")
            last_alert_time = time.time()
            
    elif total_persons > WARNING_LIMIT:
        display_status = "High Density"
        status_color = (0, 165, 255) # Orange

    # 4. DRAW DASHBOARD
    height, width, _ = frame.shape
    
    # White Bar background
    cv2.rectangle(frame, (0, 0), (width, 50), (255, 255, 255), -1)
    
    # Status Text
    cv2.putText(frame, f"Status: {display_status}", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Count Text
    cv2.putText(frame, f"Count: {total_persons}", (width - 160, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # 5. SEND TO FLUVIO
    # Compress frame to JPEG to save bandwidth
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    if ret:
        frame_bytes = buffer.tobytes()
        try:
            # Send to topic
            producer.send_record(frame_bytes, 0)
        except Exception as e:
            print(f"Fluvio Send Error: {e}")

    # 6. FPS LIMITER
    elapsed = time.time() - start_time
    wait_time = max(0, (1/FPS_LIMIT) - elapsed)
    time.sleep(wait_time)

cap.release()
