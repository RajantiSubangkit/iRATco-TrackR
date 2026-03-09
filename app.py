import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

st.set_page_config(layout="wide")

st.title("iRATco TrackR")

uploaded_video = st.file_uploader("Upload mouse video")

analysis_speed = st.selectbox(
    "Analysis speed",
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


# -------------------------------------------------
# Mouse detection
# -------------------------------------------------

def detect_mouse(frame):

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    _,mask=cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

    coords=np.column_stack(np.where(mask>0))

    if len(coords)==0:
        return None,None

    y,x=coords.mean(axis=0)

    return int(x),int(y)


# -------------------------------------------------
# Run analysis
# -------------------------------------------------

if uploaded_video:

    if st.button("Run analysis"):

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

        heat_plot=st.empty()

        dir_col1,dir_col2=st.columns(2)

        bearing_plot=dir_col1.empty()
        turn_plot=dir_col2.empty()

        zone_plot=st.empty()

        metric_col1,metric_col2,metric_col3=st.columns(3)

        mean_vel_display=metric_col1.empty()
        dist60_display=metric_col2.empty()
        anxiety_display=metric_col3.empty()

        frame_id=0

        while True:

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

                track["step_distance"]=np.sqrt(track.dx**2 + track.dy**2)

                track["velocity"]=track["step_distance"]

                track["cumulative_distance"]=track.step_distance.fillna(0).cumsum()

                track["bearing"]=np.arctan2(track.dy,track.dx)

                track["bearing_deg"]=np.degrees(track["bearing"])

                track["turn_angle"]=track["bearing_deg"].diff()

                track["turn_angle"]=(track["turn_angle"]+180)%360-180


                mean_velocity=track["velocity"].mean()

                frames_60=int(fps*60)

                if len(track)>frames_60:
                    distance_60s=track["cumulative_distance"].iloc[frames_60]
                else:
                    distance_60s=track["cumulative_distance"].iloc[-1]


                # ----------------------------------
                # ZONE ANALYSIS
                # ----------------------------------

                cx=width/2
                cy=height/2

                center_radius=min(width,height)*0.25

                dist_center=np.sqrt((track.Xs-cx)**2 + (track.Ys-cy)**2)

                track["zone"]=np.where(dist_center<center_radius,"center","wall")

                center_time=(track.zone=="center").sum()/fps
                wall_time=(track.zone=="wall").sum()/fps

                anxiety_index=wall_time/(center_time+wall_time)


                # ----------------------------------
                # PLOTS
                # ----------------------------------

                if frame_id % 10==0:

                    fig1,ax1=plt.subplots()
                    ax1.plot(track.Xs,track.Ys,color="red")
                    ax1.set_aspect("equal")
                    ax1.set_title("Trajectory")
                    traj_plot.pyplot(fig1)
                    plt.close(fig1)


                    fig2,ax2=plt.subplots()
                    ax2.plot(track["cumulative_distance"])
                    ax2.set_title("Cumulative distance")
                    dist_plot.pyplot(fig2)
                    plt.close(fig2)


                    fig3,ax3=plt.subplots()

                    ax3.plot(track["velocity"],color="purple")

                    ax3.axhline(mean_velocity,color="black",linestyle="--")

                    ax3.set_title("Velocity")

                    vel_plot.pyplot(fig3)
                    plt.close(fig3)


                    # Heatmap
                    if len(track)>20:

                        fig4,ax4=plt.subplots()

                        sns.kdeplot(
                            x=track.Xs,
                            y=track.Ys,
                            fill=True,
                            cmap="RdYlGn_r",
                            ax=ax4
                        )

                        ax4.set_aspect("equal")

                        heat_plot.pyplot(fig4)

                        plt.close(fig4)


                    # -----------------------
                    # Directional Analysis
                    # -----------------------

                    bins=np.linspace(-180,180,24)

                    fig5=plt.figure(figsize=(4,4))

                    hist,_=np.histogram(track["bearing_deg"].dropna(),bins=bins)

                    theta=np.deg2rad((bins[:-1]+bins[1:])/2)

                    ax5=fig5.add_subplot(111,polar=True)

                    ax5.bar(theta,hist,width=np.deg2rad(15),color="steelblue")

                    ax5.set_title("Absolute bearing")

                    bearing_plot.pyplot(fig5)

                    plt.close(fig5)


                    fig6=plt.figure(figsize=(4,4))

                    hist,_=np.histogram(track["turn_angle"].dropna(),bins=bins)

                    theta=np.deg2rad((bins[:-1]+bins[1:])/2)

                    ax6=fig6.add_subplot(111,polar=True)

                    ax6.bar(theta,hist,width=np.deg2rad(15),color="tomato")

                    ax6.set_title("Turn direction")

                    turn_plot.pyplot(fig6)

                    plt.close(fig6)


                    # Zone occupancy
                    fig7,ax7=plt.subplots()

                    zone_counts=track.zone.value_counts()

                    ax7.bar(zone_counts.index,zone_counts.values)

                    ax7.set_title("Zone occupancy")

                    zone_plot.pyplot(fig7)

                    plt.close(fig7)


                    mean_vel_display.metric("Mean velocity",f"{mean_velocity:.2f}")
                    dist60_display.metric("Distance first 60s",f"{distance_60s:.2f}")
                    anxiety_display.metric("Anxiety index",f"{anxiety_index:.2f}")


            frame_id+=1

            progress.progress(frame_id/total_frames)

        cap.release()

        st.success("Analysis complete")

        csv=track.to_csv(index=False)

        st.download_button(
            "Download tracking data",
            csv,
            "tracking.csv"
        )


st.markdown("---")

st.markdown("""
© 2026 Mawar Subangkit  
Mouse Tracking Analysis Software  

If you use this software, please cite:

**Subangkit**, MAWAR (2026).  
**IRATCO TrackR: Open-field Behavioral Tracking Software.**  
Available at: https://iratcotrackr.streamlit.app/
""")
