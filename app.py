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

# CONTROL BUTTONS
c1,c2=st.columns(2)

with c1:
    if st.button("▶ Run Analysis"):
        st.session_state.running=True
        st.session_state.paused=False

with c2:
    if st.button("⏹ Stop Analysis"):
        st.session_state.running=False
        st.session_state.paused=False

def negative_mouse_view(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # invert background
    inv = cv2.bitwise_not(gray)

    # threshold tikus
    _, mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

    # ubah background jadi BGR
    neg_frame = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    # warnai tikus merah
    neg_frame[mask>0] = [0,0,255]

    return neg_frame
    
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

    video_col1, video_col2 = st.columns(2)

    with video_col1:
        st.markdown("**Raw Video**")
        raw_video = st.empty()

    with video_col2:
        st.markdown("**Tracking View**")
        neg_video = st.empty()
    
    progress=st.progress(0)

    col1,col2,col3=st.columns(3)

    traj_plot=col1.empty()
    dist_plot=col2.empty()
    vel_plot=col3.empty()
###
    st.subheader("Spatial Behaviour")

    spatial_col1, spatial_col2 = st.columns(2)

    heat_plot = spatial_col1.empty()
    zone_plot = spatial_col2.empty()

    st.subheader("Directional Analysis")

    dir_col1,dir_col2 = st.columns(2)

    bearing_plot = dir_col1.empty()
    turn_plot = dir_col2.empty()
### line ####
    st.subheader("Behavior Metrics")
    metrics_table = st.empty()
###
    frame_id=0
    saved_plots=[]
    st.markdown(
    "<h2 style='text-align:center;'>Behavior Metrics</h2>",
    unsafe_allow_html=True
    )

    metrics_table = st.empty()
    while True:

        if not st.session_state.running:
            break

        ret,frame=cap.read()

        if not ret:
            break

        if frame_id % skip !=0:
            frame_id+=1
            continue

        x,y=detect_mouse(frame)

        X.append(x)
        Y.append(y)

        neg_frame = negative_mouse_view(frame)

        if x is not None:
            cv2.circle(frame,(x,y),6,(0,0,255),-1)
            cv2.circle(neg_frame,(x,y),6,(255,0,0),-1)

        raw_video.image(frame, channels="BGR")
        neg_video.image(neg_frame, channels="BGR")

        # buat negative frame
        neg_frame = negative_mouse_view(frame)

        # tampilkan raw video
        raw_video.image(frame, channels="BGR")

        # tampilkan negative video
        neg_video.image(neg_frame, channels="BGR")

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
            movement_threshold =  1  # pixel

            track.loc[track["step_distance"] < movement_threshold, "Xs"] = np.nan
            track.loc[track["step_distance"] < movement_threshold, "Ys"] = np.nan

            track["Xs"].fillna(method="ffill", inplace=True)
            track["Ys"].fillna(method="ffill", inplace=True)

            track["velocity"]=track["step_distance"]

            track["cumulative_distance"]=track.step_distance.fillna(0).cumsum()

            track["bearing"]=np.arctan2(track.dy,track.dx)
            track["bearing_deg"]=np.degrees(track["bearing"])

            track["turn_angle"]=track["bearing_deg"].diff()
            track["turn_angle"]=(track["turn_angle"]+180)%360-180

            mean_velocity=track["velocity"].mean()

            # freezing
            freezing_threshold=0.5
            track["freezing"]=track["velocity"]<freezing_threshold
            freezing_time=track["freezing"].sum()/fps

            # zone analysis
            cx=width/2
            cy=height/2
            center_radius=min(width,height)*0.25

            dist_center=np.sqrt((track.Xs-cx)**2+(track.Ys-cy)**2)

            track["zone"]=np.where(dist_center<center_radius,"center","wall")

            center_time=(track.zone=="center").sum()/fps
            wall_time=(track.zone=="wall").sum()/fps

            anxiety_index=wall_time/(center_time+wall_time)

            # exploration
            grid_size=5
            xbins=np.linspace(track.Xs.min(),track.Xs.max(),grid_size)
            ybins=np.linspace(track.Ys.min(),track.Ys.max(),grid_size)

            grid_counts,_,_=np.histogram2d(track.Xs,track.Ys,bins=[xbins,ybins])

            visited_cells=np.sum(grid_counts>0)
            total_cells=(grid_size-1)*(grid_size-1)

            exploration_index=visited_cells/total_cells

            total_distance=track["cumulative_distance"].iloc[-1]
            total_time=len(track)/fps

            if frame_id % 10==0:

                fig1,ax1=plt.subplots()
                ax1.plot(track.Xs,track.Ys,color="red")
                ax1.set_aspect("equal")
                ax1.set_title("Movement Trajectory")
                traj_plot.pyplot(fig1)
                plt.close(fig1)

                fig2,ax2=plt.subplots()
                ax2.plot(track["cumulative_distance"])
                ax2.set_title("Cumulative Distance")
                dist_plot.pyplot(fig2)
                plt.close(fig2)

                fig3,ax3=plt.subplots()
                ax3.plot(track["velocity"])
                ax3.set_title("Velocity")
                vel_plot.pyplot(fig3)
                plt.close(fig3)

                # heatmap
                if len(track)>20:
                    fig4,ax4=plt.subplots()
                    sns.kdeplot(x=track.Xs,y=track.Ys,fill=True,cmap="RdYlGn_r",ax=ax4)
                    ax4.set_aspect("equal")
                    ax4.set_title("Dwell Time Heatmap")
                    heat_plot.pyplot(fig4)
                    plt.close(fig4)

                # absolute bearing
                bins=np.linspace(-180,180,24)

                fig5=plt.figure(figsize=(4,4))
                hist,_=np.histogram(track["bearing_deg"].dropna(),bins=bins)
                theta=np.deg2rad((bins[:-1]+bins[1:])/2)

                ax5=fig5.add_subplot(111,polar=True)
                ax5.bar(theta,hist,width=np.deg2rad(15))
                ax5.set_title("Absolute Bearing")

                bearing_plot.pyplot(fig5)
                plt.close(fig5)

                # turn direction
                fig6=plt.figure(figsize=(4,4))
                hist,_=np.histogram(track["turn_angle"].dropna(),bins=bins)
                theta=np.deg2rad((bins[:-1]+bins[1:])/2)

                ax6=fig6.add_subplot(111,polar=True)
                ax6.bar(theta,hist,width=np.deg2rad(15))
                ax6.set_title("Turn Direction")

                turn_plot.pyplot(fig6)
                plt.close(fig6)

                # zone occupancy
                fig7,ax7=plt.subplots()
                zone_counts=track.zone.value_counts()
                ax7.bar(zone_counts.index,zone_counts.values)
                ax7.set_title("Zone Occupancy")
                zone_plot.pyplot(fig7)
                plt.close(fig7)

                metrics_df = pd.DataFrame([{
                    "Mean velocity": round(mean_velocity,2),
                    "Anxiety index": round(anxiety_index,2),
                    "Freezing time (s)": round(freezing_time,2),
                    "Exploration index": round(exploration_index,2),
                    "Total Distance": round(total_distance,2),
                    "Total Time (s)": round(total_time,2)
                }])

                st.markdown("""
                <style>
                .metrics-table {
                font-size:22px;
                text-align:center;
                margin:auto;
                }

                .metrics-table th {
                font-size:24px;
                font-weight:bold;
                text-align:center;
                }

                .metrics-table td {
                font-size:22px;
                }
                </style>
                """, unsafe_allow_html=True)

                metrics_table.markdown(
                metrics_df.to_html(classes="metrics-table", index=False),
                unsafe_allow_html=True
                )


        frame_id+=1
        progress.progress(frame_id/total_frames)

    cap.release()
    st.success("Analysis complete")

    csv = track.to_csv(index=False)

    st.download_button(
        label="Download tracking data (CSV)",
        data=csv,
        file_name="tracking_data.csv",
        mime="text/csv"
    )

st.markdown("---")

st.markdown("""
© 2026 Mawar Subangkit  
Mouse Behavioral Tracking Software  

**Subangkit**, MAWAR (2026)  
**iRATco TrackR: Open-field Behavioral Tracking Software**  

Available at: https://iratcotrackr.streamlit.app/
""")
