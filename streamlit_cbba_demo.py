import streamlit as st
import matplotlib.pyplot as plt
import random
import math
from typing import Dict, Any, List

from src.core.simulation_engine import SwarmSimulation
from src.config.config import SwarmConfig, SimParamsConfig, AgentConfig, ObjectiveConfig, ThreatConfig
from src.communication.tdma_scheduler import TDMAScheduler

st.set_page_config(page_title="CBBA Swarm Task Allocation Demo", layout="wide")

st.title("CBBA Swarm Task Allocation Demo")
st.caption("Decentralized auction-based consensus and task allocation under threat fields (no central controller).")

# Sidebar parameters
with st.sidebar:
    st.header("Simulation Settings")
    num_drones = st.slider("Number of combat drones", 5, 50, 20)
    num_objectives = st.slider("Number of objectives/targets", 2, 10, 5)
    num_threats = st.slider("Number of SAM threats", 1, 5, 2)
    threat_radius = st.slider("Threat radius", 5, 30, 15)
    frames = st.slider("Simulation frames", 50, 500, 200)
    seed = st.number_input("Random seed", value=42, step=1)
    
    st.markdown("---")
    st.header("Communication Settings")
    use_tdma = st.checkbox("🚧 Enable Realistic TDMA Radio (Slower Consensus)", value=False)
    slot_duration_ms = st.slider("TDMA Slot Duration (ms)", 10, 200, 50, disabled=not use_tdma)
    
    st.markdown("---")
    st.markdown("**CBBA Agent Defaults:**")
    st.markdown("- Speed: `1.5`\n- Max Force: `0.2`\n- Comm Range: `25.0`\n- Separation Weight: `1.5`\n- SAM Repulsion Weight: `4.5`")


def build_cbba_config(num_drones: int, num_objectives: int, num_threats: int, threat_radius: float, seed_val: int) -> SwarmConfig:
    sim = SimParamsConfig(
        seed=seed_val,
        bounds=(0.0, 100.0),
        dt=0.1,
        max_frames=frames,
        algorithm="cbba"
    )
    agents = [
        AgentConfig(
            name="combat_drone",
            count=num_drones,
            params={
                "speed": 1.5,
                "max_force": 0.2,
                "sense_radius": 25.0,
                "battery": 9999.0,
                "drain": 0.0,
                "attr_weight": 1.0,
                "sam_repulsion_weight": 4.5,
                "boids_separation_weight": 1.5
            }
        )
    ]
    
    # Generate random but reproducible objectives and threats based on the seed
    rng = random.Random(seed_val)
    
    positions = []
    for _ in range(num_objectives):
        # Keep objectives away from edges for better rendering
        positions.append((rng.uniform(15.0, 85.0), rng.uniform(15.0, 85.0)))
    
    objs = ObjectiveConfig(count=num_objectives, positions=positions)
    
    threats = []
    for i in range(num_threats):
        tx = rng.uniform(20.0, 80.0)
        ty = rng.uniform(20.0, 80.0)
        threats.append(ThreatConfig(id=i, x=tx, y=ty, radius=threat_radius, strength=60.0))
        
    return SwarmConfig(simulation=sim, agents=agents, objectives=objs, threats=threats)

def render_cbba_frame(sim: SwarmSimulation):
    bounds = sim.config.simulation.bounds
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    
    # 1. Draw SAM threats
    for t in sim.threats:
        circle = plt.Circle((t.x, t.y), t.radius, color='red', alpha=0.12, label='SAM Threat' if t.threat_id == 0 else "")
        ax.add_patch(circle)
        ax.plot(t.x, t.y, 'r*', markersize=10)
        # Red dashed boundary
        edge = plt.Circle((t.x, t.y), t.radius, color='red', fill=False, linestyle='--', linewidth=0.8, alpha=0.3)
        ax.add_patch(edge)
        
    # 2. Draw Objectives
    targets_x = [o.x for o in sim.objectives]
    targets_y = [o.y for o in sim.objectives]
    ax.scatter(targets_x, targets_y, color='#00ff66', marker='X', s=120, label='Objective', zorder=3)
    for o in sim.objectives:
        ax.text(o.x + 1.5, o.y + 1.5, f"T{o.obj_id}", color='#00ff66', fontsize=9, fontweight='bold')
        
    # 3. Draw Drones color-coded by state
    drones_x = [a.x for a in sim.agents if a.alive]
    drones_y = [a.y for a in sim.agents if a.alive]
    
    # Orange (#ff9f1c) if assigned, Blue (#00b4d8) if searching/idle
    colors = ['#ff9f1c' if a.assigned_task_id is not None else '#00b4d8' for a in sim.agents if a.alive]
    ax.scatter(drones_x, drones_y, c=colors, marker='v', s=80, label='Drone Agent', zorder=5)
    
    # Draw communication range circles (subtle)
    for a in sim.agents:
        if a.alive:
            circle = plt.Circle((a.x, a.y), a.sense_radius, fill=False, linestyle="--", linewidth=0.5, alpha=0.1, color="#00b4d8")
            ax.add_patch(circle)
            
    # 4. Draw Assignment arrows (drone -> assigned objective)
    for a in sim.agents:
        if a.alive and a.assigned_task_id is not None:
            target = next((o for o in sim.objectives if o.obj_id == a.assigned_task_id), None)
            if target:
                ax.annotate(
                    "",
                    xy=(target.x, target.y),
                    xytext=(a.x, a.y),
                    arrowprops=dict(arrowstyle="->", color="#ff9f1c", lw=1.2, ls=":", alpha=0.6)
                )
                
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(color='#21262d', linestyle=':', linewidth=0.5)
    ax.legend(loc='upper right', facecolor='#161b22', labelcolor='white', fontsize=9)
    
    return fig

# Simulation Execution & Results
if st.button("Run CBBA Simulation", type="primary", use_container_width=True):
    config = build_cbba_config(num_drones, num_objectives, num_threats, threat_radius, seed)
    sim = SwarmSimulation(config)

    # Run simulation steps while tracking stats frame-by-frame
    first_assignment_frame = {}
    completed_objectives = set()
    consensus_time = None

    with st.spinner("Running CBBA Auction Simulation..."):
        if use_tdma:
            tdma_scheduler = TDMAScheduler(num_drones=num_drones, slot_duration_ms=slot_duration_ms)
            
            for frame_idx in range(frames):
                # Step A: Determine who is the master of this millisecond
                current_speaker_id = tdma_scheduler.get_current_speaker()
                broadcast_packet = None

                # Step B: The "Transmission" Event (Only 1 drone broadcasts)
                if current_speaker_id is not None and current_speaker_id < len(sim.agents):
                    speaker_agent = sim.agents[current_speaker_id]
                    if speaker_agent.alive:
                        broadcast_packet = speaker_agent.prepare_broadcast(tdma_scheduler.current_time)

                # Step C: The "Reception" Event (All other drones listen)
                if broadcast_packet is not None:
                    for agent in sim.agents:
                        if agent.agent_id != current_speaker_id and agent.alive:
                            agent.receive_broadcast(current_speaker_id, broadcast_packet)

                # Step D: Agent Decision Making (Using ONLY the mailbox)
                for agent in sim.agents:
                    if not agent.alive:
                        continue
                    perceived_swarm_state = agent.get_perceived_world(tdma_scheduler.current_time)
                    agent.step(dt=0.1, perceived_swarm=perceived_swarm_state, simulation=sim)

                # Step E: Advance the clock
                tdma_scheduler.update(dt=0.1)

                # Track when each drone is first assigned a task
                for agent in sim.agents:
                    if agent.assigned_task_id is not None and agent.agent_id not in first_assignment_frame:
                        first_assignment_frame[agent.agent_id] = frame_idx
                        
                # Check for objectives reached (within 3.0 distance of any drone)
                for agent in sim.agents:
                    for obj in sim.objectives:
                        dist = math.hypot(agent.x - obj.x, agent.y - obj.y)
                        if dist <= 3.0:
                            completed_objectives.add(obj.obj_id)

                # Track consensus time
                assigned_count = sum(1 for a in sim.agents if a.assigned_task_id is not None)
                if assigned_count == num_drones and consensus_time is None:
                    consensus_time = tdma_scheduler.current_time

                # Log metrics manually
                sim.metrics_logger.log_frame(sim.frame, sim.agents, sim.objectives)
                sim.frame += 1
        else:
            # God Mode / Magical Telepathy Loop
            for frame_idx in range(frames):
                sim.step(0.1)
                
                # Track when each drone is first assigned a task
                for agent in sim.agents:
                    if agent.assigned_task_id is not None and agent.agent_id not in first_assignment_frame:
                        first_assignment_frame[agent.agent_id] = frame_idx
                        
                # Check for objectives reached (within 3.0 distance of any drone)
                for agent in sim.agents:
                    for obj in sim.objectives:
                        dist = math.hypot(agent.x - obj.x, agent.y - obj.y)
                        if dist <= 3.0:
                            completed_objectives.add(obj.obj_id)

                # Track consensus time
                assigned_count = sum(1 for a in sim.agents if a.assigned_task_id is not None)
                if assigned_count == num_drones and consensus_time is None:
                    consensus_time = frame_idx * 0.1

    def count_completed(sim_instance):
        return len(completed_objectives)

    def compute_assignment_time(sim_instance):
        if not first_assignment_frame:
            return 0.0
        avg_frames = sum(first_assignment_frame.values()) / len(first_assignment_frame)
        return avg_frames * 0.1

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = render_cbba_frame(sim)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Sim Metrics")
        completed = count_completed(sim)
        avg_time = compute_assignment_time(sim)
        
        st.metric("Tasks Completed", f"{completed} / {num_objectives}")
        st.metric("Avg Task Assignment Time", f"{avg_time:.2f}s")
        
        assigned_drones = sum(1 for a in sim.agents if a.assigned_task_id is not None)
        st.metric("Drones Assigned", f"{assigned_drones} / {num_drones}")
        
        if consensus_time is not None:
            st.metric("Consensus Convergence Time", f"{consensus_time:.2f}s")
        else:
            st.metric("Consensus Convergence Time", "N/A (Did not converge)")
        
        # State counts
        st.markdown("### Fleet Status")
        st.markdown(f"- 🔵 **Idle Drones:** {num_drones - assigned_drones}")
        st.markdown(f"- 🟡 **Assigned Drones:** {assigned_drones}")
        st.markdown(f"- 📡 **Radio Protocol:** {'TDMA Slot' if use_tdma else 'Magical Telepathy'}")
else:
    st.info("Click **Run CBBA Simulation** to execute decentralized task allocation.")


