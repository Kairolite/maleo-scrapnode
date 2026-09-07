import json
import os
from datetime import datetime
import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(
    page_title="Maleo ScrapNode",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load Light Theme Styles
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
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "mirror_camera" not in st.session_state:
    st.session_state.mirror_camera = False
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Sorting"
if "recorded_session_logs" not in st.session_state:
    st.session_state.recorded_session_logs = []
if "live_logs_display" not in st.session_state:
    st.session_state.live_logs_display = []
if "logged_object_ids" not in st.session_state:
    st.session_state.logged_object_ids = set()


# Helper to save session log automatically
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
        st.session_state.logged_object_ids = set()


# 4. Header & Navigation Lock Logic
st.title("🐦 Maleo ScrapNode - Material & Quality Scanner")

nav_disabled = st.session_state.camera_active

current_tab = st.radio(
    "Navigation",
    ["Sorting", "Database"],
    index=0 if st.session_state.current_tab == "Sorting" else 1,
    horizontal=True,
    disabled=nav_disabled,
    label_visibility="collapsed",
)

if not nav_disabled:
    st.session_state.current_tab = current_tab

if st.session_state.camera_active:
    st.warning(
        "🔒 **Navigation Locked**: Stop the live camera feed before accessing the Database dashboard."
    )

st.markdown("---")

# ==========================================
# MENU 1: SORTING (LIVE SCANNER)
# ==========================================
if current_tab == "Sorting":
    col_video, col_side = st.columns([3, 2])

    with col_side:
        st.subheader("Controls & Live Audit")

        # Side-by-side Camera Action & Mirror Buttons
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if not st.session_state.camera_active:
                if st.button("▶️ Activate Camera"):
                    st.session_state.recorded_session_logs = []
                    st.session_state.live_logs_display = []
                    st.session_state.logged_object_ids = set()
                    st.session_state.camera_active = True
                    st.rerun()
            else:
                if st.button("⏹️ Stop Camera"):
                    st.session_state.camera_active = False
                    auto_save_to_json()
                    st.rerun()

        with col_btn2:
            mirror_label = (
                "🪞 Mirror: ON"
                if st.session_state.mirror_camera
                else "🪞 Mirror: OFF"
            )
            if st.button(mirror_label):
                st.session_state.mirror_camera = (
                    not st.session_state.mirror_camera
                )
                st.rerun()

        conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

        st.subheader("Live Unique Materials Stream")
        log_table_placeholder = st.empty()

        if not st.session_state.camera_active:
            log_table_placeholder.markdown(
                """
                <div class="placeholder-box-offline" style="height: 250px;">
                    <div>📋 FEED LOG INACTIVE</div>
                    <div style="font-size: 0.85rem; margin-top: 6px; color: #64748B;">
                        Start stream to track unique material object IDs
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_video:
        st.subheader("Webcam Viewfinder")
        video_placeholder = st.empty()

        if not st.session_state.camera_active:
            video_placeholder.markdown(
                """
                <div class="placeholder-box-offline">
                    <div>📷 CAMERA OFFLINE</div>
                    <div style="font-size: 0.9rem; margin-top: 8px; color: #64748B;">
                        Tap "Activate Camera" to start live feed
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # OpenCV Live Loop
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)

        try:
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera connection lost.")
                    st.session_state.camera_active = False
                    auto_save_to_json()
                    break

                # Apply camera mirroring if enabled
                if st.session_state.mirror_camera:
                    frame = cv2.flip(frame, 1)

                results = model.track(
                    source=frame, conf=conf_thresh, persist=True, verbose=False
                )
                annotated_frame = results[0].plot()

                metal_detected = False

                if (
                    results[0].boxes is not None
                    and results[0].boxes.id is not None
                ):
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    class_indices = results[0].boxes.cls.int().cpu().tolist()

                    timestamp_str = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    time_display = datetime.now().strftime("%H:%M:%S")

                    for track_id, class_idx in zip(track_ids, class_indices):
                        obj_name = model.names[class_idx]

                        is_obj_metal = (
                            obj_name.lower()
                            in [m.lower() for m in METAL_CLASSES]
                            or "metal" in obj_name.lower()
                        )
                        if is_obj_metal:
                            metal_detected = True

                        if track_id not in st.session_state.logged_object_ids:
                            st.session_state.logged_object_ids.add(track_id)

                            item_entry = {
                                "track_id": track_id,
                                "material": obj_name,
                                "is_metal": is_obj_metal,
                                "timestamp": timestamp_str,
                            }
                            st.session_state.recorded_session_logs.append(
                                item_entry
                            )

                            st.session_state.live_logs_display.insert(
                                0,
                                {
                                    "ID": f"#{track_id}",
                                    "Time": time_display,
                                    "Material": obj_name,
                                    "Type": (
                                        "⚡ Metal" if is_obj_metal else "Other"
                                    ),
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

                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(
                    rgb_frame, channels="RGB", use_container_width=True
                )

                if st.session_state.live_logs_display:
                    df_logs = pd.DataFrame(st.session_state.live_logs_display)
                    log_table_placeholder.dataframe(
                        df_logs, use_container_width=True, hide_index=True
                    )

        finally:
            cap.release()

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
            # Summary Metrics
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

            # ----------------------------------------------------
            # DATA VISUALIZATION: "Scrap Composition" Stacked Bar Chart
            # ----------------------------------------------------
            st.subheader("📊 Scrap Composition")

            # Prepare Data Frame for Charting
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

                # Filter Controls
                selected_materials = st.multiselect(
                    "Filter Materials to Include in Visualization:",
                    options=sorted(list(all_materials)),
                    default=sorted(list(all_materials)),
                )

                if selected_materials:
                    # Filter data based on selection
                    df_filtered = df_chart_raw[
                        df_chart_raw["Material"].isin(selected_materials)
                    ]

                    # Group and pivot data into stacked matrix form: Index = Session, Columns = Materials
                    df_pivot = (
                        df_filtered.groupby(["Session", "Material"])
                        .size()
                        .unstack(fill_value=0)
                    )

                    # Display Stacked Bar Chart
                    st.bar_chart(df_pivot, stack=True)
                else:
                    st.warning("Please select at least one material to display.")
            else:
                st.info("No material detection details available to display.")

            st.markdown("---")

            # Inspect Individual Session Details
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
            else:
                st.write("No items detected in this session.")

            with st.expander("🔍 View Raw JSON File Data"):
                st.json(logs_data)