import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import zipfile
import os

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

        os.makedirs("outputs",exist_ok=True)

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

            frame_id+=1
            progress.progress(frame_id/total_frames)

        cap.release()


        track=pd.DataFrame({"X":X,"Y":Y})

        track["Y"]=height-track["Y"]

        track["Xs"]=track["X"].rolling(5,center=True).mean()
        track["Ys"]=track["Y"].rolling(5,center=True).mean()

        track["Xs"].fillna(track["X"],inplace=True)
        track["Ys"].fillna(track["Y"],inplace=True)

        track["dx"]=track.Xs.diff()
        track["dy"]=track.Ys.diff()

        track["step_distance"]=np.sqrt(track.dx**2 + track.dy**2)

        track["velocity"]=track["step_distance"]

        track["cumulative_distance"]=track.step_distance.fillna(0).cumsum()

        track["bearing"]=np.arctan2(track.dy,track.dx)

        track["bearing_deg"]=np.degrees(track["bearing"])

        track["turn_angle"]=track["bearing_deg"].diff()

        track["turn_angle"]=(track["turn_angle"]+180)%360-180


        # ------------------------------------------------
        # Freezing detection
        # ------------------------------------------------

        freezing_threshold=0.5

        track["freezing"]=track["velocity"]<freezing_threshold

        freezing_time=track["freezing"].sum()/fps


        # ------------------------------------------------
        # Zone analysis
        # ------------------------------------------------

        cx=width/2
        cy=height/2

        center_radius=min(width,height)*0.25

        dist_center=np.sqrt((track.Xs-cx)**2 + (track.Ys-cy)**2)

        track["zone"]=np.where(dist_center<center_radius,"center","wall")

        center_time=(track.zone=="center").sum()/fps
        wall_time=(track.zone=="wall").sum()/fps

        anxiety_index=wall_time/(center_time+wall_time)


        # ------------------------------------------------
        # Grid exploration index
        # ------------------------------------------------

        grid_size=5

        xbins=np.linspace(track.Xs.min(),track.Xs.max(),grid_size)
        ybins=np.linspace(track.Ys.min(),track.Ys.max(),grid_size)

        grid_counts,_,_=np.histogram2d(track.Xs,track.Ys,bins=[xbins,ybins])

        visited_cells=np.sum(grid_counts>0)

        total_cells=(grid_size-1)*(grid_size-1)

        exploration_index=visited_cells/total_cells


        # ------------------------------------------------
        # Behavior classification
        # ------------------------------------------------

        conditions=[
            track["velocity"]<0.5,
            track["velocity"]>5
        ]

        choices=["freezing","running"]

        track["behavior"]=np.select(conditions,choices,default="walking")


        # ------------------------------------------------
        # PLOTS
        # ------------------------------------------------

        figures=[]


        # trajectory

        fig1,ax1=plt.subplots()
        ax1.plot(track.Xs,track.Ys,color="red")
        ax1.set_title("Trajectory")
        ax1.set_aspect("equal")
        fig1.savefig("outputs/trajectory.png")
        figures.append("outputs/trajectory.png")


        # cumulative distance

        fig2,ax2=plt.subplots()
        ax2.plot(track["cumulative_distance"])
        ax2.set_title("Cumulative distance")
        fig2.savefig("outputs/cumulative_distance.png")
        figures.append("outputs/cumulative_distance.png")


        # velocity

        fig3,ax3=plt.subplots()
        ax3.plot(track["velocity"])
        ax3.set_title("Velocity")
        fig3.savefig("outputs/velocity.png")
        figures.append("outputs/velocity.png")


        # heatmap

        fig4,ax4=plt.subplots()
        sns.kdeplot(x=track.Xs,y=track.Ys,fill=True,cmap="RdYlGn_r",ax=ax4)
        ax4.set_aspect("equal")
        ax4.set_title("Exploration heatmap")
        fig4.savefig("outputs/heatmap.png")
        figures.append("outputs/heatmap.png")


        # speed heatmap

        fig5,ax5=plt.subplots()

        sc=ax5.scatter(track.Xs,track.Ys,c=track.velocity,cmap="viridis")

        plt.colorbar(sc)

        ax5.set_title("Speed heatmap")

        fig5.savefig("outputs/speed_heatmap.png")

        figures.append("outputs/speed_heatmap.png")


        # directional analysis

        bins=np.linspace(-180,180,24)

        fig6=plt.figure()

        hist,_=np.histogram(track["bearing_deg"].dropna(),bins=bins)

        theta=np.deg2rad((bins[:-1]+bins[1:])/2)

        ax6=fig6.add_subplot(111,polar=True)

        ax6.bar(theta,hist,width=np.deg2rad(15))

        ax6.set_title("Absolute bearing")

        fig6.savefig("outputs/bearing.png")

        figures.append("outputs/bearing.png")


        fig7=plt.figure()

        hist,_=np.histogram(track["turn_angle"].dropna(),bins=bins)

        theta=np.deg2rad((bins[:-1]+bins[1:])/2)

        ax7=fig7.add_subplot(111,polar=True)

        ax7.bar(theta,hist,width=np.deg2rad(15))

        ax7.set_title("Turn direction")

        fig7.savefig("outputs/turn_angle.png")

        figures.append("outputs/turn_angle.png")


        # zone occupancy

        fig8,ax8=plt.subplots()

        zone_counts=track.zone.value_counts()

        ax8.bar(zone_counts.index,zone_counts.values)

        ax8.set_title("Zone occupancy")

        fig8.savefig("outputs/zones.png")

        figures.append("outputs/zones.png")


        # grid exploration

        fig9,ax9=plt.subplots()

        ax9.imshow(grid_counts,cmap="hot")

        ax9.set_title("Grid exploration")

        fig9.savefig("outputs/grid_exploration.png")

        figures.append("outputs/grid_exploration.png")


        st.success("Analysis complete")


        # ------------------------------------------------
        # Metrics display
        # ------------------------------------------------

        st.metric("Freezing time (s)",round(freezing_time,2))

        st.metric("Anxiety index",round(anxiety_index,2))

        st.metric("Exploration index",round(exploration_index,2))


        # ------------------------------------------------
        # Save results
        # ------------------------------------------------

        track.to_csv("outputs/tracking.csv",index=False)


        zip_path="outputs/results.zip"

        with zipfile.ZipFile(zip_path,"w") as zipf:

            zipf.write("outputs/tracking.csv")

            for fig in figures:
                zipf.write(fig)


        with open(zip_path,"rb") as f:

            st.download_button(
                "Download all results",
                f,
                file_name="iratco_results.zip"
            )


st.markdown("---")

st.markdown("""
© 2026 Mawar Subangkit  

Mouse Behavioral Tracking Software

**Subangkit**, MAWAR (2026)  
**IRATCO TrackR: Open-field Behavioral Tracking Software**

Available at: https://iratcotrackr.streamlit.app/""")
