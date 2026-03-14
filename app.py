from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import zipfile
import io
import time

# PAGE CONFIG
st.set_page_config(
    page_title="iRATco TrackR",
    page_icon="logo.png",
    layout="wide"
)

# HEADER
col1, col2 = st.columns([8, 2])
with col1:
    st.title("iRATco TrackR")
    st.markdown("<span style='font-size:16px;color:gray;'>**version 1.2.1**</span>", unsafe_allow_html=True)
with col2:
    st.image("logo_iratco.png", width=250)

uploaded_video = st.file_uploader("Upload your video")

if uploaded_video:
    if st.button("Reset ROI"):
        for key in ["roi", "roi_points", "pixel_to_mm", "real_roi_width_mm"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# reset ROI if new video uploaded
if uploaded_video is not None:
    if "last_video" not in st.session_state:
        st.session_state.last_video = uploaded_video.name

    if uploaded_video.name != st.session_state.last_video:
        st.session_state.last_video = uploaded_video.name

        for key in ["roi", "roi_points", "pixel_to_mm", "real_roi_width_mm"]:
            if key in st.session_state:
                del st.session_state[key]

##### selector
# ROI selection
roi = None

if uploaded_video:
    # simpan video ke temp file sekali saja
    video_bytes = uploaded_video.getvalue()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_bytes)
    st.session_state.video_path = tfile.name

    cap = cv2.VideoCapture(st.session_state.video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        display_width = 300
        scale = display_width / frame.shape[1]
        display_height = int(frame.shape[0] * scale)

        display_frame = cv2.resize(frame, (display_width, display_height))

        st.subheader("Select ROI (click TOP LEFT then BOTTOM RIGHT)")

        if "roi_points" not in st.session_state:
            st.session_state.roi_points = []

        roi_display = display_frame.copy()

        if "roi" in st.session_state:
            x, y, w, h = st.session_state.roi
            x_disp = int(x * scale)
            y_disp = int(y * scale)
            w_disp = int(w * scale)
            h_disp = int(h * scale)

            cv2.rectangle(
                roi_display,
                (x_disp, y_disp),
                (x_disp + w_disp, y_disp + h_disp),
                (0, 255, 0),
                3
            )

        point = streamlit_image_coordinates(roi_display)

        if point is not None and len(st.session_state.roi_points) < 2:
            real_x = int(point["x"] / scale)
            real_y = int(point["y"] / scale)
            st.session_state.roi_points.append((real_x, real_y))

        if len(st.session_state.roi_points) == 1 and "roi" not in st.session_state:
            st.info("Click BOTTOM RIGHT corner")

        if len(st.session_state.roi_points) == 2 and "roi" not in st.session_state:
            (x1, y1), (x2, y2) = st.session_state.roi_points

            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            st.session_state.roi = (x, y, w, h)
            st.rerun()

        # =========================
        # Calibration input
        # =========================
        if "roi" in st.session_state:
            x, y, w, h = st.session_state.roi

            st.subheader("Spatial Calibration")

            cal_col1, cal_col2 = st.columns(2)

            with cal_col1:
                st.write(f"ROI width (pixels): {w}")

            with cal_col2:
                real_roi_width_mm = st.number_input(
                    "Real length of ROI width (mm)",
                    min_value=0.0,
                    value=float(st.session_state.get("real_roi_width_mm", 500.0)),
                    step=10.0
                )

            st.session_state.real_roi_width_mm = real_roi_width_mm

            if w > 0 and real_roi_width_mm > 0:
                st.session_state.pixel_to_mm = real_roi_width_mm / w
                st.write(f"Scale: {st.session_state.pixel_to_mm:.4f} mm/pixel")

######
analysis_speed = st.selectbox(
    "Analysis Speed",
    ["1X", "2X", "4X", "8X", "20X"]
)

speed_map = {
    "1X": 1,
    "2X": 2,
    "4X": 4,
    "8X": 8,
    "20X": 20
}

skip = speed_map[analysis_speed]

# object contrast selection
contrast_mode = st.radio(
    "Object Type",
    ["Bright object", "Dark object"],
    horizontal=True
)

# SESSION STATE
if "running" not in st.session_state:
    st.session_state.running = False

# CONTROL BUTTONS
c1, c2 = st.columns(2)

with c1:
    if st.button("▶ Run Analysis"):
        if "roi" not in st.session_state:
            st.warning("Please select ROI first.")
        elif "pixel_to_mm" not in st.session_state:
            st.warning("Please enter real ROI width first.")
        else:
            st.session_state.running = True
            st.session_state.paused = False

with c2:
    if st.button("⏹ Stop Analysis"):
        st.session_state.running = False
        st.session_state.paused = False

def negative_mouse_view(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # invert background
    inv = cv2.bitwise_not(gray)

    # threshold sesuai tipe objek
    if contrast_mode == "Bright object":
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # ubah background jadi BGR
    neg_frame = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    # warnai objek merah
    neg_frame[mask > 0] = [0, 0, 255]

    return neg_frame

def detect_mouse(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if contrast_mode == "Bright object":
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    coords = np.column_stack(np.where(mask > 0))

    if len(coords) == 0:
        return None, None

    y, x = coords.mean(axis=0)

    return int(x), int(y)

if uploaded_video and st.session_state.running:
    cap = cv2.VideoCapture(st.session_state.video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if "roi" in st.session_state:
        width = st.session_state.roi[2]
        height = st.session_state.roi[3]

    fps = cap.get(cv2.CAP_PROP_FPS)

    X = []
    Y = []

    video_col1, video_col2 = st.columns(2)

    with video_col1:
        st.markdown("**Raw Video**")
        raw_video = st.empty()

    with video_col2:
        st.markdown("**Tracking View**")
        neg_video = st.empty()

    progress = st.progress(0)

    st.markdown(
        "<h2 style='text-align:center;'>Movement Analysis</h2>",
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)

    traj_plot = col1.empty()
    dist_plot = col2.empty()
    vel_plot = col3.empty()

    st.markdown(
        "<h2 style='text-align:center;'>Spatial Behaviour</h2>",
        unsafe_allow_html=True
    )

    spatial_col1, spatial_col2 = st.columns(2)

    heat_plot = spatial_col1.empty()
    zone_plot = spatial_col2.empty()

    st.markdown(
        "<h2 style='text-align:center;'>Directional Analysis</h2>",
        unsafe_allow_html=True
    )

    dir_col1, dir_col2 = st.columns(2)

    bearing_plot = dir_col1.empty()
    turn_plot = dir_col2.empty()

    frame_id = 0
    saved_plots = []

    st.markdown(
        "<h2 style='text-align:center;'>Behavior Metrics</h2>",
        unsafe_allow_html=True
    )
    metrics_table = st.empty()

    st.markdown("""
    <style>
    .metrics-table {
        font-size:22px;
        text-align:center;
        margin:auto;
        margin-top:5px;
        margin-bottom:5px;
        border-collapse:collapse;
    }

    .metrics-table th {
        font-size:24px;
        font-weight:bold;
        text-align:center;
        padding:8px;
    }

    .metrics-table td {
        font-size:22px;
        padding:8px;
    }
    </style>
    """, unsafe_allow_html=True)

    while True:
        if not st.session_state.running:
            break

        ret, frame = cap.read()

        if not ret or frame is None:
            break

        if "roi" not in st.session_state:
            st.error("Please select ROI first.")
            st.stop()

        x, y, w, h = st.session_state.roi
        fh, fw = frame.shape[:2]

        x = max(0, min(int(x), fw - 1))
        y = max(0, min(int(y), fh - 1))
        w = max(1, min(int(w), fw - x))
        h = max(1, min(int(h), fh - y))

        if w <= 0 or h <= 0:
            st.error("ROI is invalid.")
            st.stop()

        frame = frame[y:y+h, x:x+w]

        if frame_id % skip != 0:
            frame_id += 1
            continue

        x, y = detect_mouse(frame)

        if x is not None and y is not None:
            X.append(x)
            Y.append(y)
        else:
            X.append(np.nan)
            Y.append(np.nan)

        neg_frame = negative_mouse_view(frame)

        if x is not None:
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            cv2.circle(neg_frame, (x, y), 6, (255, 0, 0), -1)

        raw_video.image(frame, channels="BGR")
        neg_video.image(neg_frame, channels="BGR")

        track = pd.DataFrame({"X": X, "Y": Y})

        track["Y"] = height - track["Y"]

        track["Xs"] = track["X"].rolling(9, center=True).mean()
        track["Ys"] = track["Y"].rolling(9, center=True).mean()

        alpha = 0.2
        track["Xs"] = track["X"].ewm(alpha=alpha).mean()
        track["Ys"] = track["Y"].ewm(alpha=alpha).mean()

        track["Xs"].fillna(track["X"], inplace=True)
        track["Ys"].fillna(track["Y"], inplace=True)

        if len(track) > 2:
            track["dx"] = track.Xs.diff()
            track["dy"] = track.Ys.diff()

            track["step_distance"] = np.sqrt(track.dx**2 + track.dy**2)

            # convert pixel to mm
            if "pixel_to_mm" in st.session_state:
                track["step_distance"] = track["step_distance"] * st.session_state.pixel_to_mm

            movement_threshold = 0.3
            track.loc[track["step_distance"] < movement_threshold, "Xs"] = np.nan
            track.loc[track["step_distance"] < movement_threshold, "Ys"] = np.nan

            track["Xs"].fillna(method="ffill", inplace=True)
            track["Ys"].fillna(method="ffill", inplace=True)

            dt = skip / fps
            track["velocity"] = track["step_distance"] / dt
            track["cumulative_distance"] = track.step_distance.fillna(0).cumsum()

            track["bearing"] = np.arctan2(track.dy, track.dx)
            track["bearing_deg"] = np.degrees(track["bearing"])

            track["turn_angle"] = track["bearing_deg"].diff()
            track["turn_angle"] = (track["turn_angle"] + 180) % 360 - 180

            mean_velocity = track["velocity"].mean()

            # freezing
            freezing_threshold = 0.5
            track["freezing"] = track["velocity"] < freezing_threshold
            freezing_time = track["freezing"].sum() * dt

            # zone analysis
            cx = width / 2
            cy = height / 2
            center_radius = min(width, height) * 0.25

            dist_center = np.sqrt((track.Xs - cx)**2 + (track.Ys - cy)**2)
            track["zone"] = np.where(dist_center < center_radius, "center", "wall")

            center_time = (track.zone == "center").sum() * dt
            wall_time = (track.zone == "wall").sum() * dt

            anxiety_index = wall_time / (center_time + wall_time) if (center_time + wall_time) > 0 else np.nan

            # exploration
            grid_size = 5
            xbins = np.linspace(track.Xs.min(), track.Xs.max(), grid_size)
            ybins = np.linspace(track.Ys.min(), track.Ys.max(), grid_size)

            grid_counts, _, _ = np.histogram2d(track.Xs, track.Ys, bins=[xbins, ybins])

            visited_cells = np.sum(grid_counts > 0)
            total_cells = (grid_size - 1) * (grid_size - 1)
            exploration_index = visited_cells / total_cells if total_cells > 0 else np.nan

            total_distance = track["cumulative_distance"].iloc[-1]
            total_time = len(track) * dt

            if frame_id % 20 == 0:
                fig1, ax1 = plt.subplots()
                ax1.plot(track.Xs, track.Ys, color="red")
                ax1.set_aspect("equal")
                ax1.set_title("Trajectory")
                traj_plot.pyplot(fig1)
                plt.close(fig1)

                fig2, ax2 = plt.subplots()
                ax2.plot(track["cumulative_distance"])
                ax2.set_title("Cumulative Distance (mm)")
                dist_plot.pyplot(fig2)
                plt.close(fig2)

                fig3, ax3 = plt.subplots()
                ax3.plot(track["velocity"])
                ax3.set_title("Velocity (mm/s)")
                vel_plot.pyplot(fig3)
                plt.close(fig3)

                # dwell time heatmap
                if len(track) > 20:
                    heat_data = track[["Xs", "Ys"]].dropna()

                    if len(heat_data) > 0:
                    # setiap titik menyumbang waktu dt
                        dwell_weights = np.full(len(heat_data), dt)

                    # resolusi heatmap
                        n_bins = 60
                        x_edges = np.linspace(0, width, n_bins + 1)
                        y_edges = np.linspace(0, height, n_bins + 1)

                        heatmap, _, _ = np.histogram2d(
                            heat_data["Xs"],
                            heat_data["Ys"],
                            bins=[x_edges, y_edges],
                            weights=dwell_weights
                        )

                        # smooth supaya area dwell menyebar dan tidak cuma titik
                        heatmap_smooth = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=2.5, sigmaY=2.5)

                        # custom colormap: putih -> hijau -> kuning -> merah
                        from matplotlib.colors import LinearSegmentedColormap
                        dwell_cmap = LinearSegmentedColormap.from_list(
                            "dwell_cmap",
                            ["#ffffff", "#00ff00", "#ffff00", "#ff0000"]
                        )

                        fig4, ax4 = plt.subplots(figsize=(6, 5))

                        im = ax4.imshow(
                            heatmap_smooth.T,
                            origin="lower",
                            extent=[0, width, 0, height],
                            aspect="equal",
                            cmap=dwell_cmap,
                            interpolation="bilinear"
                        )

                        # overlay trajectory
                        ax4.plot(track["Xs"], track["Ys"], color="black", alpha=0.35, linewidth=1)

                        ax4.set_title("Movement Trajectory")
                        ax4.set_xlabel("X")
                        ax4.set_ylabel("Y")

                        cbar = fig4.colorbar(im, ax=ax4, shrink=0.8)
                        cbar.set_label("Dwell Time Level")

                        heat_plot.pyplot(fig4)
                        plt.close(fig4)

                # absolute bearing
                bins = np.linspace(-180, 180, 24)

                fig5 = plt.figure(figsize=(4, 4))
                hist, _ = np.histogram(track["bearing_deg"].dropna(), bins=bins)
                theta = np.deg2rad((bins[:-1] + bins[1:]) / 2)

                ax5 = fig5.add_subplot(111, polar=True)
                ax5.bar(theta, hist, width=np.deg2rad(15))
                ax5.set_title("Absolute Bearing")
                bearing_plot.pyplot(fig5)
                plt.close(fig5)

                # turn direction
                fig6 = plt.figure(figsize=(4, 4))
                hist, _ = np.histogram(track["turn_angle"].dropna(), bins=bins)
                theta = np.deg2rad((bins[:-1] + bins[1:]) / 2)

                ax6 = fig6.add_subplot(111, polar=True)
                ax6.bar(theta, hist, width=np.deg2rad(15))
                ax6.set_title("Turn Direction")
                turn_plot.pyplot(fig6)
                plt.close(fig6)

                # zone occupancy
                fig7, ax7 = plt.subplots()
                zone_counts = track.zone.value_counts()
                ax7.bar(zone_counts.index, zone_counts.values)
                ax7.set_title("Zone Occupancy")
                zone_plot.pyplot(fig7)
                plt.close(fig7)

                metrics_df = pd.DataFrame([{
                    "Mean velocity (mm/s)": round(mean_velocity, 2) if pd.notna(mean_velocity) else np.nan,
                    "Anxiety index": round(anxiety_index, 2) if pd.notna(anxiety_index) else np.nan,
                    "Freezing time (s)": round(freezing_time, 2) if pd.notna(freezing_time) else np.nan,
                    "Exploration index": round(exploration_index, 2) if pd.notna(exploration_index) else np.nan,
                    "Total Distance (mm)": round(total_distance, 2) if pd.notna(total_distance) else np.nan,
                    "Total Time (s)": round(total_time, 2) if pd.notna(total_time) else np.nan
                }])

                metrics_table.markdown(
                    metrics_df.to_html(classes="metrics-table", index=False),
                    unsafe_allow_html=True
                )

        frame_id += 1
        progress.progress(frame_id / total_frames)

    cap.release()
    st.success("Analysis complete")

    if "track" in locals() and not track.empty:
        csv = track.to_csv(index=False)

        st.download_button(
            label="Download Tracking Data (CSV)",
            data=csv,
            file_name="tracking_data.csv",
            mime="text/csv"
        )

st.markdown("---")

st.markdown("""
© 2026 Mawar Subangkit  
**Mouse Behavioral Tracking Software**  

If you use this software, please cite:

**Subangkit**, MAWAR (2026)  
**iRATco TrackR: Open-field Behavioral Tracking Software**  
Available at: https://iratcotrackr.streamlit.app/
""")
