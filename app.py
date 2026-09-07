import json
import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(
    page_title="Maleo ScrapNode",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load Custom Theme
if os.path.exists("custom.css"):
    with open("custom.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Load Trash/Metal Model
MODEL_PATH = "trashdet-base.pt"


@st.cache_resource
def load_trash_model():
    if not os.path.exists(MODEL_PATH):
        st.warning(
            f"'{MODEL_PATH}' not found in root folder. Defaulting to 'yolov8n.pt'."
        )
        return YOLO("yolov8n.pt")
    return YOLO(MODEL_PATH)


model = load_trash_model()

METAL_CLASSES = [
    "metal",
    "can",
    "copper",
    "aluminum",
    "steel",
    "iron",
    "scrap_metal",
    "tin",
]

# 3. Session State Setup
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Sorting"
if "recorded_session_logs" not in st.session_state:
    st.session_state.recorded_session_logs = []
if "live_logs_display" not in st.session_state:
    st.session_state.live_logs_display = []


# Helper to auto-save scan records to JSON
def auto_save_to_json():
    if st.session_state.recorded_session_logs:
        log_file = "audit_logs.json"

        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        session_entry = {
            "app_name": "Maleo ScrapNode",
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "unique_items_logged": len(st.session_state.recorded_session_logs),
            "detections": st.session_state.recorded_session_logs,
        }

        existing_data.append(session_entry)

        with open(log_file, "w") as f:
            json.dump(existing_data, f, indent=4)

        st.toast(
            f"Session saved ({len(st.session_state.recorded_session_logs)} items) to {log_file}!",
            icon="💾",
        )

        st.session_state.recorded_session_logs = []


# 4. Header & Top Navigation
st.title("🐦 Maleo ScrapNode - Material & Quality Scanner")

current_tab = st.radio(
    "Navigation",
    ["Sorting", "Database"],
    index=0 if st.session_state.current_tab == "Sorting" else 1,
    horizontal=True,
    label_visibility="collapsed",
)

st.session_state.current_tab = current_tab
st.markdown("---")

# ==========================================
# MENU 1: SORTING (MOBILE BROWSER SCANNER)
# ==========================================
if current_tab == "Sorting":
    col_video, col_side = st.columns([3, 2])

    with col_side:
        st.subheader("Controls & Live Audit")
        conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

        if st.button("💾 Save Session Data to JSON"):
            auto_save_to_json()
            st.rerun()

        st.subheader("Recent Material Scans")
        log_table_placeholder = st.empty()

        if st.session_state.live_logs_display:
            df_logs = pd.DataFrame(st.session_state.live_logs_display)
            log_table_placeholder.dataframe(
                df_logs, use_container_width=True, hide_index=True
            )
        else:
            log_table_placeholder.markdown(
                """
                <div class="placeholder-box-offline" style="height: 200px;">
                    <div>📋 FEED LOG EMPTY</div>
                    <div style="font-size: 0.85rem; margin-top: 6px; color: #64748B;">
                        Capture a snapshot to record materials
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_video:
        st.subheader("Webcam Viewfinder")

        # Browser Native Camera Widget
        camera_photo = st.camera_input("Scan Scrap Material")

        if camera_photo is not None:
            # Convert image buffer from browser into OpenCV BGR format
            bytes_data = camera_photo.getvalue()
            frame = cv2.imdecode(
                np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
            )

            # Run YOLO Prediction
            results = model.predict(
                source=frame, conf=conf_thresh, verbose=False
            )
            annotated_frame = results[0].plot()

            detected_names = [
                model.names[int(box.cls[0])] for box in results[0].boxes
            ]
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            time_display = datetime.now().strftime("%H:%M:%S")

            metal_detected = False

            if detected_names:
                for obj_name in detected_names:
                    is_obj_metal = (
                        obj_name.lower() in [m.lower() for m in METAL_CLASSES]
                        or "metal" in obj_name.lower()
                    )
                    if is_obj_metal:
                        metal_detected = True

                    # Record item entry
                    item_entry = {
                        "material": obj_name,
                        "is_metal": is_obj_metal,
                        "timestamp": timestamp_str,
                    }
                    st.session_state.recorded_session_logs.append(item_entry)

                    st.session_state.live_logs_display.insert(
                        0,
                        {
                            "Time": time_display,
                            "Material": obj_name,
                            "Type": "⚡ Metal" if is_obj_metal else "Other",
                        },
                    )
                    st.session_state.live_logs_display = (
                        st.session_state.live_logs_display[:8]
                    )

            if metal_detected:
                cv2.putText(
                    annotated_frame,
                    "METAL DETECTED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 200, 100),
                    3,
                )

            # Display Annotated Image Stream Result
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(
                rgb_frame,
                caption="Inference Result",
                use_container_width=True,
            )

            # Refresh table display
            if st.session_state.live_logs_display:
                df_logs = pd.DataFrame(st.session_state.live_logs_display)
                log_table_placeholder.dataframe(
                    df_logs, use_container_width=True, hide_index=True
                )

# ==========================================
# MENU 2: DATABASE & VISUALIZATION
# ==========================================
elif current_tab == "Database":
    st.subheader("📁 Saved Inspection Database")

    log_file = "audit_logs.json"

    if not os.path.exists(log_file):
        st.info("No recorded sessions found in 'audit_logs.json' yet.")
    else:
        with open(log_file, "r") as f:
            try:
                logs_data = json.load(f)
            except json.JSONDecodeError:
                logs_data = []

        if not logs_data:
            st.warning("The log file is empty.")
        else:
            total_sessions = len(logs_data)
            total_items = sum(
                s.get("unique_items_logged", 0) for s in logs_data
            )

            col_db1, col_db2 = st.columns(2)
            with col_db1:
                st.metric(label="Total Sessions Saved", value=total_sessions)
            with col_db2:
                st.metric(label="Total Unique Items Logged", value=total_items)

            st.markdown("---")

            st.subheader("📊 Scrap Composition")

            chart_records = []
            all_materials = set()

            for session in logs_data:
                sess_id = f"S-{session.get('session_id', 'N/A')[-6:]}"
                for det in session.get("detections", []):
                    mat = det.get("material", "Unknown")
                    all_materials.add(mat)
                    chart_records.append(
                        {"Session": sess_id, "Material": mat, "Count": 1}
                    )

            if chart_records:
                df_chart_raw = pd.DataFrame(chart_records)

                selected_materials = st.multiselect(
                    "Filter Materials to Include in Visualization:",
                    options=sorted(list(all_materials)),
                    default=sorted(list(all_materials)),
                )

                if selected_materials:
                    df_filtered = df_chart_raw[
                        df_chart_raw["Material"].isin(selected_materials)
                    ]
                    df_pivot = (
                        df_filtered.groupby(["Session", "Material"])
                        .size()
                        .unstack(fill_value=0)
                    )
                    st.bar_chart(df_pivot, stack=True)
                else:
                    st.warning("Please select at least one material.")

            st.markdown("---")

            session_options = [
                f"Session {s.get('session_id', 'N/A')} ({s.get('unique_items_logged', 0)} items)"
                for s in logs_data
            ]
            selected_session_idx = st.selectbox(
                "Select Session to View Details",
                range(len(session_options)),
                format_func=lambda i: session_options[i],
            )

            selected_session = logs_data[selected_session_idx]
            st.markdown(
                f"### Details for Session: `{selected_session.get('session_id')}`"
            )

            detections_list = selected_session.get("detections", [])
            if detections_list:
                df_session = pd.DataFrame(detections_list)
                st.dataframe(
                    df_session, use_container_width=True, hide_index=True
                )

            with st.expander("🔍 View Raw JSON File Data"):
                st.json(logs_data)