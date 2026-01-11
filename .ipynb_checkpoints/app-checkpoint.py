import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# === 1. PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Stampede Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === 2. TEAM ARETE STYLE CSS (FIXED TO STOP FLASHING) ===
st.markdown("""
    <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        /* RESET STREAMLIT DEFAULTS */
        .stApp {
            background: linear-gradient(135deg, #f8fafc, #e0f2fe);
            font-family: 'Poppins', sans-serif;
        }
        header {visibility: hidden;}
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
            padding-right: 0rem;
            max-width: 100%;
        }
        
        /* CUSTOM NAVBAR */
        .navbar {
            background-color: #0f172a;
            color: white;
            padding: 15px 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            margin-bottom: 20px;
        }
        .navbar h1 {
            color: white;
            margin: 0;
            font-size: 1.8rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* STATUS BAR CONTAINER */
        .status-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        /* STATUS CHIPS */
        .status-box {
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1.1rem;
            min-width: 200px;
            text-align: center;
            border: 1px solid transparent;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* DYNAMIC COLORS */
        .status-normal { background-color: #d4edda; color: #155724; border-color: #c3e6cb; }
        .status-warning { background-color: #fff3cd; color: #856404; border-color: #ffeeba; }
        .status-critical { 
            background-color: #f8d7da; 
            color: #721c24; 
            border-color: #f5c6cb; 
            animation: pulse-red 1.5s infinite;
        }
        
        .count-box {
            background-color: #e9ecef;
            color: #495057;
            border: 1px solid #ced4da;
        }

        /* ANIMATION */
        @keyframes pulse-red {
            0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(220, 53, 69, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
        }

        /* --- VIDEO STYLING FIX --- */
        /* Target the Streamlit Image Element directly via CSS instead of HTML wrapper */
        [data-testid="stImage"] {
            display: flex;
            justify-content: center;
        }
        [data-testid="stImage"] img {
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            max-width: 850px; /* Limits video size */
        }
        
        /* HIDE FILE UPLOADER LABEL */
        .stFileUploader label { display: none; }
        .stFileUploader { margin: auto; max-width: 500px; }

    </style>
""", unsafe_allow_html=True)

# === 3. HEADER (NAVBAR) ===
st.markdown("""
    <div class="navbar">
        <h1>🛡️ CrowdSense: Live Analysis</h1>
    </div>
""", unsafe_allow_html=True)

# === 4. PLACEHOLDERS ===
status_placeholder = st.empty()
# Added a container for the video to help with centering
video_container = st.empty()

# === 5. LOAD MODEL ===
@st.cache_resource
def load_model():
    return YOLO('yolo11n.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# === 6. FILE UPLOADER ===
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

# === 7. PROCESSING LOGIC ===
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    # Hardcoded Settings
    CONF_LEVEL = 0.10
    IOU_LEVEL = 0.90
    RISK_LIMIT = 25

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. AI Detection
        results = model.predict(frame, conf=CONF_LEVEL, iou=IOU_LEVEL, imgsz=640, classes=[0], verbose=False)
        
        total_persons = 0
        overlay = frame.copy()
        
        # 2. Draw Filled Boxes
        if results[0].boxes:
            for box in results[0].boxes:
                total_persons += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Colors
                box_color = (0, 255, 0) # Green
                if total_persons > 20: box_color = (0, 165, 255) # Orange
                if total_persons > RISK_LIMIT: box_color = (0, 0, 255)   # Red

                cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)

        # Transparency
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 3. Determine Status
        status_text = "Normal"
        css_class = "status-normal"
        
        if total_persons > RISK_LIMIT:
            status_text = "CRITICAL RISK"
            css_class = "status-critical"
        elif total_persons > 20:
            status_text = "High Density"
            css_class = "status-warning"

        # 4. UPDATE THE UI
        
        # Update Status Bar (HTML is okay here if text doesn't change too violently)
        status_placeholder.markdown(f"""
            <div class="status-container">
                <div class="status-box {css_class}">
                    Stampede Chances: {status_text}
                </div>
                <div class="status-box count-box">
                    Detected Persons: {total_persons}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- THE FIX FOR FLASHING ---
        # Don't use markdown here! Use .image() directly.
        # The CSS at the top handles the centering and shadow now.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_container.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
else:
    # Default State
    status_placeholder.markdown("""
        <div class="status-container">
            <div class="status-box status-normal" style="background-color: #e2e8f0; border-color: #cbd5e1; color: #64748b;">
                Waiting for Video...
            </div>
        </div>
    """, unsafe_allow_html=True)
