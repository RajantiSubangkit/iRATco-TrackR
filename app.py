import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import zipfile
import io

st.set_page_config(layout="wide")

st.title("iRATco TrackR")

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


def detect_mouse(frame):

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    _,mask=cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

    coords=np.column_stack(np.where(mask>0))

    if len(coords)==0:
        return None,None

    y,x=coords.mean(axis=0)

    return int(x),int(y)


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

        metric_col1,metric_col2,metric_col3,metric_col4=st.columns(4)

        mean_vel_display=metric_col1.empty()
        anxiety_display=metric_col2.empty()
        freezing_display=metric_col3.empty()
        exploration_display=metric_col4.empty()

        frame_id=0
        saved_plots = []

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

                track["step_distance"]=np.sqrt(track.dx**2+track.dy**2)

                track["velocity"]=track["step_distance"]

                track["cumulative_distance"]=track.step_distance.fillna(0).cumsum()

                track["bearing"]=np.arctan2(track.dy,track.dx)

                track["bearing_deg"]=np.degrees(track["bearing"])

                track["turn_angle"]=track["bearing_deg"].diff()

                track["turn_angle"]=(track["turn_angle"]+180)%360-180

                mean_velocity=track["velocity"].mean()

                # freezing detection
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

                # grid exploration
                grid_size=5
                xbins=np.linspace(track.Xs.min(),track.Xs.max(),grid_size)
                ybins=np.linspace(track.Ys.min(),track.Ys.max(),grid_size)

                grid_counts,_,_=np.histogram2d(track.Xs,track.Ys,bins=[xbins,ybins])

                visited_cells=np.sum(grid_counts>0)
                total_cells=(grid_size-1)*(grid_size-1)

                exploration_index=visited_cells/total_cells

                if frame_id % 10==0:

                    fig1,ax1=plt.subplots()
                    ax1.plot(track.Xs,track.Ys,color="red")
                    ax1.set_aspect("equal")
                    ax1.set_title("Trajectory")
                    traj_plot.pyplot(fig1)
                    buf = io.BytesIO()
                    fig1.savefig(buf, format="png")
                    saved_plots.append(("trajectory.png", buf.getvalue()))
                    plt.close(fig1)

                    fig2,ax2=plt.subplots()
                    ax2.plot(track["cumulative_distance"])
                    ax2.set_title("Cumulative distance")
                    dist_plot.pyplot(fig2)
                    plt.close(fig2)

                    fig3,ax3=plt.subplots()
                    ax3.plot(track["velocity"])
                    ax3.set_title("Velocity")
                    vel_plot.pyplot(fig3)
                    plt.close(fig3)

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
                        buf = io.BytesIO()
                        fig4.savefig(buf, format="png")
                        saved_plots.append(("heatmap.png", buf.getvalue()))
                        plt.close(fig4)

                    # directional analysis

                    bins=np.linspace(-180,180,24)

                    fig5=plt.figure(figsize=(4,4))

                    hist,_=np.histogram(track["bearing_deg"].dropna(),bins=bins)

                    theta=np.deg2rad((bins[:-1]+bins[1:])/2)

                    ax5=fig5.add_subplot(111,polar=True)

                    ax5.bar(theta,hist,width=np.deg2rad(15))

                    bearing_plot.pyplot(fig5)

                    plt.close(fig5)

                    fig6=plt.figure(figsize=(4,4))

                    hist,_=np.histogram(track["turn_angle"].dropna(),bins=bins)

                    theta=np.deg2rad((bins[:-1]+bins[1:])/2)

                    ax6=fig6.add_subplot(111,polar=True)

                    ax6.bar(theta,hist,width=np.deg2rad(15))

                    turn_plot.pyplot(fig6)

                    plt.close(fig6)

                    fig7,ax7=plt.subplots()

                    zone_counts=track.zone.value_counts()

                    ax7.bar(zone_counts.index,zone_counts.values)

                    zone_plot.pyplot(fig7)

                    plt.close(fig7)

                    mean_vel_display.metric("Mean velocity",f"{mean_velocity:.2f}")
                    anxiety_display.metric("Anxiety index",f"{anxiety_index:.2f}")
                    freezing_display.metric("Freezing time (s)",f"{freezing_time:.2f}")
                    exploration_display.metric("Exploration index",f"{exploration_index:.2f}")

            frame_id+=1

            progress.progress(frame_id/total_frames)

        cap.release()

        st.success("Analysis complete")
        # ---------------------------------
        # Download all plots
        # ---------------------------------

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zip_file:

            for filename, data in saved_plots:
            zip_file.writestr(filename, data)

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
**Mouse Behavioral Tracking Software**

**Subangkit**, MAWAR (2026)  
**IRATCO TrackR: Open-field Behavioral Tracking Software**

Available at: https://iratcotrackr.streamlit.app/""")
