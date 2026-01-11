import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time

# === 1. APP CONFIGURATION ===
st.set_page_config(
    page_title="Stampede Risk Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === 2. SESSION STATE ===
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'video_source' not in st.session_state:
    st.session_state['video_source'] = None
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'

# === 3. DYNAMIC CSS ===
if st.session_state['theme'] == 'dark':
    bg_color = "#0f172a"
    text_color = "#f8fafc"
    card_bg = "#1e293b"
    border_color = "#334155"
    shadow_color = "rgba(0,0,0,0.3)"
    sidebar_icon_color = "#ffffff"
else:
    bg_color = "#f1f5f9"
    text_color = "#0f172a"
    card_bg = "#ffffff"
    border_color = "#e2e8f0"
    shadow_color = "rgba(0,0,0,0.05)"
    sidebar_icon_color = "#334155"

st.markdown(f"""
    <style>
        .stApp {{
            background-color: {bg_color};
            font-family: 'Poppins', sans-serif;
            color: {text_color};
        }}
        header {{visibility: hidden;}}
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 0rem;
        }}
        
        /* FIX SIDEBAR TOGGLE VISIBILITY */
        [data-testid="stSidebarCollapsedControl"] {{
            color: {sidebar_icon_color} !important;
            background-color: {card_bg};
            border-radius: 50%;
            padding: 5px;
            box-shadow: 0 2px 5px {shadow_color};
        }}
        
        /* STATUS CHIPS */
        .status-container {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 10px;
        }}
        .status-box {{
            padding: 12px 20px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1.1rem;
            min-width: 250px; /* Made wider for the bar */
            text-align: center;
            background: {card_bg};
            box-shadow: 0 4px 6px {shadow_color};
            border: 1px solid {border_color};
            color: {text_color};
        }}
        
        /* COLORS */
        .status-normal {{ background-color: #dcfce7; color: #15803d; border-color: #bbf7d0; }}
        .status-warning {{ background-color: #fef9c3; color: #a16207; border-color: #fde047; }}
        .status-critical {{ 
            background-color: #fee2e2; 
            color: #b91c1c; 
            border-color: #fecaca;
            animation: pulse 1.5s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }}
            70% {{ box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }}
        }}

        /* VIDEO SIZE FIX */
        [data-testid="stImage"] {{
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        [data-testid="stImage"] img {{
            border-radius: 12px;
            max-height: 400px;
            width: auto;
            object-fit: contain;
            box-shadow: 0 10px 30px {shadow_color};
        }}
        
        h1, h2, h3, p {{ color: {text_color} !important; }}
    </style>
""", unsafe_allow_html=True)

# === 4. HELPER FUNCTIONS ===
@st.cache_resource
def load_model():
    return YOLO('yolo11n.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# UPDATED BAR FUNCTION: Scales up to 100 people
def create_density_bar(count):
    # Let's say 100 people is "Full Capacity" for the visual bar
    max_capacity = 100 
    percentage = min(count / max_capacity, 1.0)
    
    bar_length = 10
    filled = int(bar_length * percentage)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    # Returns: "24 Persons ██░░░░░░░░"
    return f"{count} People detected &nbsp; {bar}"

def process_frame(frame):
    # Aggressive Settings
    results = model.predict(frame, conf=0.10, iou=0.90, imgsz=640, classes=[0], verbose=False)
    
    total_persons = 0
    overlay = frame.copy()
    
    if results[0].boxes:
        for box in results[0].boxes:
            total_persons += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            box_color = (0, 255, 0) # Green
            if total_persons > 20: box_color = (0, 165, 255) # Orange
            if total_persons > 25: box_color = (0, 0, 255)   # Red

            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    status_text = "Normal"
    css_class = "status-normal"
    
    if total_persons > 25:
        status_text = "CRITICAL RISK"
        css_class = "status-critical"
    elif total_persons > 20:
        status_text = "High Density"
        css_class = "status-warning"
        
    return frame, total_persons, status_text, css_class

# === 5. SIDEBAR ===
with st.sidebar:
    st.title("⚙️ Settings")
    mode = st.radio("Display Mode", ["Light", "Dark"], index=0 if st.session_state['theme'] == 'light' else 1)
    if mode == "Dark" and st.session_state['theme'] != 'dark':
        st.session_state['theme'] = 'dark'
        st.rerun()
    elif mode == "Light" and st.session_state['theme'] != 'light':
        st.session_state['theme'] = 'light'
        st.rerun()

# === 6. PAGES ===
def show_home():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.markdown(f"""
                <h1 style='text-align: center; margin-bottom: 10px;'>🛡️ Stampede Risk Analysis</h1>
                <p style='text-align: center; margin-bottom: 30px; opacity: 0.8;'>
                    Choose an input source to begin real-time crowd density detection.
                </p>
            """, unsafe_allow_html=True)
            if st.button("🔴 Start Live Webcam Feed", use_container_width=True, type="primary"):
                st.session_state['page'] = 'live'
                st.rerun()
            st.markdown("---")
            uploaded_file = st.file_uploader(" ", type=["mp4", "avi"])
            if uploaded_file:
                if st.button("Analyze Uploaded Media", use_container_width=True):
                    st.session_state['video_source'] = uploaded_file
                    st.session_state['page'] = 'analysis'
                    st.rerun()

def show_live():
    c1, c2 = st.columns([1, 15])
    with c1:
        if st.button("⬅"):
            st.session_state['page'] = 'home'
            st.rerun()
    with c2:
        st.markdown("<h4 style='margin: 5px 0 0 0;'>📡 Live Surveillance Feed</h4>", unsafe_allow_html=True)

    status_placeholder = st.empty()
    video_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    frame_count = 0
    display_status = "Initializing..."
    display_density = "0 People ░░░░░░░░░░"
    display_css = "status-normal"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            st.error("Camera error.")
            break
            
        processed_frame, count, status, css = process_frame(frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            display_status = status
            display_density = create_density_bar(count)
            display_css = css
        
        status_placeholder.markdown(f"""
            <div class="status-container">
                <div class="status-box {display_css}">Stampede Chances: {display_status}</div>
                <div class="status-box">{display_density}</div>
            </div>
        """, unsafe_allow_html=True)
        
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

def show_analysis():
    c1, c2 = st.columns([1, 15])
    with c1:
        if st.button("⬅"):
            st.session_state['page'] = 'home'
            st.session_state['video_source'] = None
            st.rerun()
    with c2:
        st.markdown("<h4 style='margin: 5px 0 0 0;'>📊 Media Analysis</h4>", unsafe_allow_html=True)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(st.session_state['video_source'].read())
    cap = cv2.VideoCapture(tfile.name)
    
    status_placeholder = st.empty()
    video_placeholder = st.empty()

    frame_count = 0
    display_status = "Initializing..."
    display_density = "0 People ░░░░░░░░░░"
    display_css = "status-normal"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        processed_frame, count, status, css = process_frame(frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            display_status = status
            display_density = create_density_bar(count)
            display_css = css
        
        status_placeholder.markdown(f"""
            <div class="status-container">
                <div class="status-box {display_css}">Stampede Chances: {display_status}</div>
                <div class="status-box">{display_density}</div>
            </div>
        """, unsafe_allow_html=True)
        
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
    cap.release()

# === 7. ROUTER ===
if st.session_state['page'] == 'home':
    show_home()
elif st.session_state['page'] == 'live':
    show_live()
elif st.session_state['page'] == 'analysis':
    show_analysis()
