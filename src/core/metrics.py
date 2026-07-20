from typing import List, Dict, Any, Tuple
import csv
import os

class MetricsLogger:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def reset(self):
        self.history = []

    def log_frame(self, frame_id: int, agents: list, objectives: list):
        active_agents = [a for a in agents if a.alive]
        states: Dict[str, int] = {}
        for a in agents:
            states[a.state] = states.get(a.state, 0) + 1

        frame_log = {
            "frame": frame_id,
            "active_count": len(active_agents),
            "states": states,
            "agent_positions": [(a.agent_id, a.x, a.y, a.state, a.battery) for a in agents],
            "objective_states": [(o.obj_id, o.x, o.y, getattr(o, 'paint_strength', 0.0)) for o in objectives]
        }
        self.history.append(frame_log)

    def get_summary(self) -> Dict[str, Any]:
        if not self.history:
            return {}

        total_frames = len(self.history)
        final_frame = self.history[-1]
        
        # Calculate when stability occurred (e.g. all agents comfortable, or states stopped changing)
        stability_frame = -1
        for i in range(total_frames):
            frame_states = self.history[i]["states"]
            # For Reynolds, comfortable state is the target
            if frame_states.get("comfortable", 0) == len(self.history[i]["agent_positions"]):
                stability_frame = i
                break

        # Calculate consensus frame for CBBA (e.g. all objectives have a winner assigned or consensus is reached)
        # For simplicity, if we check CBBA bids, consensus is when winner IDs stop changing.
        # But we can calculate a general metric: active count, average battery, etc.
        return {
            "total_frames": total_frames,
            "final_active_count": final_frame["active_count"],
            "final_states": final_frame["states"],
            "stability_frame": stability_frame
        }

    def save_to_csv(self, filepath: str):
        if not self.history:
            return

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Headers
            writer.writerow(["frame", "active_count", "lonely", "crowded", "comfortable", "depleted"])
            for h in self.history:
                st = h["states"]
                writer.writerow([
                    h["frame"],
                    h["active_count"],
                    st.get("lonely", 0),
                    st.get("crowded", 0),
                    st.get("comfortable", 0),
                    st.get("depleted", 0)
                ])
