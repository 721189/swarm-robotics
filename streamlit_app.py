"""Interactive Streamlit demo for the swarm self-organization behavior in
swarm_visualisation.py. Same Robot logic (lonely / comfortable / crowded),
just parametrized with sliders instead of hardcoded values, and rendered
frame-by-frame instead of via matplotlib.animation (which doesn't run in a
browser). Deploy as-is to Streamlit Community Cloud
"""

import time
import random

import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Swarm Robotics — Interactive Demo", layout="wide")


class Robot:
    def __init__(self, robot_id, x, y):
        self.robot_id = robot_id
        self.x = x
        self.y = y
        self.state = "comfortable"

    def decide_move(self, all_robots, radius, crowded_threshold):
        nearby = []
        for other in all_robots:
            if other.robot_id != self.robot_id:
                distance = ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
                if distance <= radius:
                    nearby.append(other)

        if len(nearby) == 0:
            self.x += 1 if self.x < 0 else -1
            self.y += 1 if self.y < 0 else -1
            self.state = "lonely"
        elif len(nearby) >= crowded_threshold:
            self.x += random.choice([-1, 1])
            self.y += random.choice([-1, 1])
            self.state = "crowded"
        else:
            self.state = "comfortable"


COLOR_MAP = {"lonely": "#f4d35e", "crowded": "#ef476f", "comfortable": "#06d6a0"}


def run_simulation(num_robots, radius, crowded_threshold, world_bound, num_frames, seed):
    random.seed(seed)
    swarm = [
        Robot(i, random.randint(-world_bound, world_bound), random.randint(-world_bound, world_bound))
        for i in range(num_robots)
    ]

    frames = []
    for _ in range(num_frames):
        for robot in swarm:
            robot.decide_move(swarm, radius, crowded_threshold)
        frames.append(
            {
                "xs": [r.x for r in swarm],
                "ys": [r.y for r in swarm],
                "colors": [COLOR_MAP[r.state] for r in swarm],
                "lonely": sum(1 for r in swarm if r.state == "lonely"),
                "crowded": sum(1 for r in swarm if r.state == "crowded"),
                "comfy": sum(1 for r in swarm if r.state == "comfortable"),
            }
        )
    return frames


def render_frame(frame_data, frame_idx, world_bound):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-world_bound - 5, world_bound + 5)
    ax.set_ylim(-world_bound - 5, world_bound + 5)
    ax.set_facecolor("#0b0e14")
    fig.patch.set_facecolor("#0b0e14")
    ax.scatter(
        frame_data["xs"], frame_data["ys"],
        c=frame_data["colors"], s=90, edgecolors="white", linewidths=0.4,
    )
    ax.set_title(
        f"Frame {frame_idx} | Lonely: {frame_data['lonely']}  "
        f"Crowded: {frame_data['crowded']}  Comfortable: {frame_data['comfy']}",
        color="white", fontsize=10,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    return fig


st.title("Swarm Robotics — Interactive Demo")
st.caption(
    "Decentralized self-organization: each robot only knows its own local "
    "neighbor count (no central controller). Adjust parameters and re-run."
)

with st.sidebar:
    st.header("Parameters")
    num_robots = st.slider("Number of robots", 5, 150, 30)
    radius = st.slider("Sensing radius", 1, 25, 5)
    crowded_threshold = st.slider("Crowded threshold (neighbors)", 2, 10, 3)
    world_bound = st.slider("World bound (± units)", 10, 100, 50)
    num_frames = st.slider("Frames to simulate", 20, 400, 150)
    seed = st.number_input("Random seed", value=42, step=1)
    run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)

if "frames" not in st.session_state:
    st.session_state.frames = None
    st.session_state.world_bound = world_bound

if run_clicked:
    with st.spinner("Simulating..."):
        st.session_state.frames = run_simulation(
            num_robots, radius, crowded_threshold, world_bound, num_frames, seed
        )
        st.session_state.world_bound = world_bound

if st.session_state.frames:
    frames = st.session_state.frames
    col1, col2 = st.columns([3, 1])

    with col2:
        autoplay = st.checkbox("Autoplay", value=False)
        speed = st.slider("Playback speed (fps)", 1, 30, 10)

    with col1:
        placeholder = st.empty()

    if autoplay:
        for i in range(len(frames)):
            fig = render_frame(frames[i], i, st.session_state.world_bound)
            placeholder.pyplot(fig)
            plt.close(fig)
            time.sleep(1.0 / speed)
    else:
        frame_idx = st.slider("Frame", 0, len(frames) - 1, 0)
        fig = render_frame(frames[frame_idx], frame_idx, st.session_state.world_bound)
        placeholder.pyplot(fig)
        plt.close(fig)
else:
    st.info("Set your parameters in the sidebar and click **Run Simulation** to begin.")
