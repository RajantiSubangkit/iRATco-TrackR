import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import zipfile
import io

# PAGE CONFIG (HARUS PALING ATAS)
st.set_page_config(
    page_title="iRATco TrackR",
    page_icon="logo.png",
    layout="wide"
)

# HEADER
col1, col2 = st.columns([8,2])
with col1:
    st.title("iRATco TrackR")
with col2:
    st.image("logo_iratco.png", width=250)

uploaded_video = st.file_uploader("Upload your video")

analysis_speed = st.selectbox(
    "Analysis Speed",
    ["1X","2X","4X","8X","20X"]
)

speed_map={
    "1X":1,
    "2X":2,
    "4X":4,
    "8X":8,
    "20X":20
}

skip=speed_map[analysis_speed]

# SESSION STATE
if "running" not in st.session_state:
    st.session_state.running=False

if "paused" not in st.session_state:
    st.session_state.paused=False

# CONTROL BUTTONS
c1,c2,c3=st.columns(3)

with c1:
    if st.button("▶ Start"):
        st.session_state.running=True
        st.session_state.paused=False

with c2:
    if st.button("⏸ Pause"):
        st.session_state.paused=True

with c3:
    if st.button("⏹ Stop"):
        st.session_state.running=False
        st.session_state.paused=False


def detect_mouse(frame):

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    _,mask=cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

    coords=np.column_stack(np.where(mask>0))

    if len(coords)==0:
        return None,None

    y,x=coords.mean(axis=0)

    return int(x),int(y)


if uploaded_video and st.session_state.running:

    tfile=tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap=cv2.VideoCapture(tfile.name)

    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps=cap.get(cv2.CAP_PROP_FPS)

    X=[]
    Y=[]

    frame_window=st.empty()

    progress=st.progress(0)

    col1,col2,col3=st.columns(3)

    traj_plot=col1.empty()
    dist_plot=col2.empty()
    vel_plot=col3.empty()

    st.subheader("Dwell Time Heatmap")
    heat_plot=st.empty()

    st.subheader("Directional Analysis")
    dir_col1,dir_col2=st.columns(2)

    bearing_plot=dir_col1.empty()
    turn_plot=dir_col2.empty()

    st.subheader("Zone Occupancy")
    zone_plot=st.empty()

    metric_col1,metric_col2,metric_col3,metric_col4,metric_col5,metric_col6=st.columns(6)

    mean_vel_display=metric_col1.empty()
    anxiety_display=metric_col2.empty()
    freezing_display=metric_col3.empty()
    exploration_display=metric_col4.empty()
    distance_display=metric_col5.empty()
    time_display=metric_col6.empty()

    frame_id=0
    saved_plots=[]

    while True:

        if not st.session_state.running:
            break

        if st.session_state.paused:
            st.warning("Paused")
            continue

        ret,frame=cap.read()

        if not ret:
            break

        if frame_id % skip !=0:
            frame_id+=1
            continue

        x,y=detect_mouse(frame)

        X.append(x)
        Y.append(y)

        if x is not None:
            cv2.circle(frame,(x,y),6,(0,0,255),-1)

        frame_window.image(frame,channels="BGR")

        track=pd.DataFrame({"X":X,"Y":Y})

        track["Y"]=height-track["Y"]

        track["Xs"]=track["X"].rolling(5,center=True).mean()
        track["Ys"]=track["Y"].rolling(5,center=True).mean()

        track["Xs"].fillna(track["X"],inplace=True)
        track["Ys"].fillna(track["Y"],inplace=True)

        if len(track)>2:

            track["dx"]=track.Xs.diff()
            track["dy"]=track.Ys.diff()

            track["step_distance"]=np.sqrt(track.dx**2+track.dy**2)

            track["velocity"]=track["step_distance"]

            track["cumulative_distance"]=track.step_distance.fillna(0).cumsum()

            mean_velocity=track["velocity"].mean()

            total_distance=track["cumulative_distance"].iloc[-1]

            total_time=len(track)/fps

            if frame_id % 10==0:

                fig1,ax1=plt.subplots()
                ax1.plot(track.Xs,track.Ys,color="red")
                ax1.set_xlim(0,width)
                ax1.set_ylim(0,height)
                ax1.set_aspect("equal")
                ax1.set_title("Movement Trajectory")
                traj_plot.pyplot(fig1)

                buf=io.BytesIO()
                fig1.savefig(buf,format="png")
                saved_plots.append(("trajectory.png",buf.getvalue()))
                plt.close(fig1)

                fig2,ax2=plt.subplots()
                ax2.plot(track["cumulative_distance"])
                ax2.set_title("Cumulative Distance")
                dist_plot.pyplot(fig2)

                buf=io.BytesIO()
                fig2.savefig(buf,format="png")
                saved_plots.append(("distance.png",buf.getvalue()))
                plt.close(fig2)

                fig3,ax3=plt.subplots()
                ax3.plot(track["velocity"])
                ax3.set_title("Velocity")
                vel_plot.pyplot(fig3)

                buf=io.BytesIO()
                fig3.savefig(buf,format="png")
                saved_plots.append(("velocity.png",buf.getvalue()))
                plt.close(fig3)

                mean_vel_display.metric("Mean velocity",f"{mean_velocity:.2f}")
                distance_display.metric("Total Distance",f"{total_distance:.2f}")
                time_display.metric("Total Time (s)",f"{total_time:.2f}")

        frame_id+=1
        progress.progress(frame_id/total_frames)

    cap.release()

    st.success("Analysis complete")

    zip_buffer=io.BytesIO()

    with zipfile.ZipFile(zip_buffer,"w") as zip_file:
        for filename,data in saved_plots:
            zip_file.writestr(filename,data)

    st.download_button(
        label="Download all plots",
        data=zip_buffer.getvalue(),
        file_name="iratco_plots.zip",
        mime="application/zip"
    )

    csv=track.to_csv(index=False)

    st.download_button(
        "Download tracking data",
        csv,
        "tracking.csv"
    )


st.markdown("---")

st.markdown("""
© 2026 Mawar Subangkit  
Mouse Behavioral Tracking Software

Subangkit, M. (2026)  
iRATco TrackR: Open-field Behavioral Tracking Software

Available at: https://iratcotrackr.streamlit.app/
""")
