import threading
import time
import queue
import math
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

from src.core.simulation_engine import SwarmSimulation

class AsyncRenderer:
    def __init__(self, sim: SwarmSimulation, interval: int = 100):
        self.sim = sim
        self.interval = interval
        self.state_queue = queue.Queue(maxsize=20)
        self.running = False
        self.sim_thread: Optional[threading.Thread] = None

    def _sim_loop(self):
        dt = self.sim.config.simulation.dt
        max_frames = self.sim.config.simulation.max_frames

        while self.running and self.sim.frame < max_frames:
            # Step the simulation
            self.sim.step(dt)

            # Package state for thread-safe rendering
            state = {
                "frame": self.sim.frame,
                "agents": [a.to_dict() for a in self.sim.agents],
                "objectives": [(o.obj_id, o.x, o.y, o.paint_strength, o.painted) for o in self.sim.objectives],
                "threats": [(t.threat_id, t.x, t.y, t.radius, t.strength) for t in self.sim.threats]
            }

            try:
                # Put in queue, overwrite if full to always render newest
                self.state_queue.put(state, timeout=1.0)
            except queue.Full:
                try:
                    self.state_queue.get_nowait()
                    self.state_queue.put_nowait(state)
                except (queue.Empty, queue.Full):
                    pass

            # Pace the simulation
            time.sleep(self.interval / 1000.0)

    def start(self, save_path: Optional[str] = None):
        if save_path:
            # Headless or offline frame capture to save path (GIF/MP4)
            self._render_to_file(save_path)
        else:
            # Interactive visualization (Decoupled Async Simulation)
            self.running = True
            self.sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
            self.sim_thread.start()
            self._run_gui()

    def _run_gui(self):
        bounds = self.sim.config.simulation.bounds
        algo = self.sim.config.simulation.algorithm.lower()

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_facecolor('#0d1117')
        fig.patch.set_facecolor('#0d1117')

        # Static elements for CBBA
        if algo == "cbba":
            for t in self.sim.threats:
                circle = plt.Circle((t.x, t.y), t.radius, color='red', alpha=0.15)
                ax.add_patch(circle)
                ax.plot(t.x, t.y, 'r*', markersize=10)

        # Matplotlib artists based on algorithm
        artists = {}
        if algo == "reynolds":
            # Grid setup for Voronoi
            grid_res = 120
            x_grid = np.linspace(bounds[0], bounds[1], grid_res)
            y_grid = np.linspace(bounds[0], bounds[1], grid_res)
            xx, yy = np.meshgrid(x_grid, y_grid)
            extent = (bounds[0], bounds[1], bounds[0], bounds[1])
            voronoi_colors = plt.cm.tab20(np.linspace(0, 1, len(self.sim.agents)))
            
            artists["voronoi_im"] = ax.imshow(
                np.zeros((grid_res, grid_res, 3)),
                extent=extent,
                origin='lower',
                alpha=0.3,
                zorder=1
            )
            artists["scatter"] = ax.scatter([], [], c=[], s=50, zorder=5)

        elif algo == "stigmergy":
            artists["scouts"] = ax.scatter([], [], color='#ff7b00', marker='o', s=80, label='Scout', zorder=5)
            artists["workers"] = ax.scatter([], [], color='#58a6ff', marker='o', s=60, label='Worker', zorder=5)
            artists["unpainted"] = ax.scatter([], [], color='#6e7681', marker='s', s=100, label='Unpainted Target', zorder=3)
            artists["painted"] = ax.scatter([], [], color='#ffd700', marker='s', s=110, label='Painted Target', zorder=3)
            # Rings
            artists["sense_rings"] = []
            for agent in self.sim.agents:
                circle = plt.Circle(
                    (agent.x, agent.y),
                    agent.sense_radius,
                    fill=False,
                    linestyle="--",
                    linewidth=0.6,
                    alpha=0.25,
                    color="#ff7b00" if getattr(agent, "species", "") == "scout" else "#58a6ff"
                )
                ax.add_patch(circle)
                artists["sense_rings"].append((agent.agent_id, circle))

        elif algo == "cbba":
            artists["drones"] = ax.scatter([], [], marker='v', s=60, label='Drone Node', zorder=5)
            artists["targets"] = ax.scatter([], [], color='green', marker='X', s=120, label='Target Objective', zorder=3)

        if algo != "reynolds":
            ax.legend(loc='upper right', facecolor='#161b22', labelcolor='white', fontsize=9)

        title = ax.set_title("", color='white', fontsize=11)

        def update_plot(frame_count):
            # Pull state from queue
            state = None
            try:
                state = self.state_queue.get_nowait()
                while not self.state_queue.empty():
                    state = self.state_queue.get_nowait()
            except queue.Empty:
                pass

            if state is None:
                return []

            frame = state["frame"]
            agents = state["agents"]
            objectives = state["objectives"]

            if algo == "reynolds":
                # Compute Voronoi coloring
                pts = np.array([[a["x"], a["y"]] for a in agents if a["alive"]])
                if len(pts) > 0:
                    min_dist2 = np.full(xx.shape, np.inf)
                    closest_idx = np.zeros(xx.shape, dtype=int)
                    for i, (px, py) in enumerate(pts):
                        d2 = (xx - px) ** 2 + (yy - py) ** 2
                        mask = d2 < min_dist2
                        closest_idx[mask] = i
                        min_dist2[mask] = d2[mask]
                    
                    color_grid = voronoi_colors[closest_idx][:, :, :3]
                    artists["voronoi_im"].set_data(color_grid)
                
                # Plot agents with state coloring
                x_pts = [a["x"] for a in agents if a["alive"]]
                y_pts = [a["y"] for a in agents if a["alive"]]
                colors = []
                for a in agents:
                    if a["alive"]:
                        if a["state"] == "lonely":
                            colors.append("yellow")
                        elif a["state"] == "crowded":
                            colors.append("red")
                        else:
                            colors.append("cyan")
                
                artists["scatter"].set_offsets(np.column_stack([x_pts, y_pts]) if x_pts else np.zeros((0, 2)))
                artists["scatter"].set_color(colors)

                lonely = sum(1 for a in agents if a["state"] == "lonely" and a["alive"])
                crowded = sum(1 for a in agents if a["state"] == "crowded" and a["alive"])
                comfy = sum(1 for a in agents if a["state"] == "comfortable" and a["alive"])
                title.set_text(f'Voronoi Swarm | Frame {frame} | Lonely: {lonely}  Crowded: {crowded}  Comfortable: {comfy}')

            elif algo == "stigmergy":
                scouts_x = [a["x"] for a in agents if a["alive"] and a["species"] == "scout"]
                scouts_y = [a["y"] for a in agents if a["alive"] and a["species"] == "scout"]
                workers_x = [a["x"] for a in agents if a["alive"] and a["species"] == "worker"]
                workers_y = [a["y"] for a in agents if a["alive"] and a["species"] == "worker"]

                artists["scouts"].set_offsets(np.column_stack([scouts_x, scouts_y]) if scouts_x else np.zeros((0, 2)))
                artists["workers"].set_offsets(np.column_stack([workers_x, workers_y]) if workers_x else np.zeros((0, 2)))

                # Objectives
                unpainted_x = [o[1] for o in objectives if not o[4]]
                unpainted_y = [o[2] for o in objectives if not o[4]]
                painted_x = [o[1] for o in objectives if o[4]]
                painted_y = [o[2] for o in objectives if o[4]]

                artists["unpainted"].set_offsets(np.column_stack([unpainted_x, unpainted_y]) if unpainted_x else np.zeros((0, 2)))
                artists["painted"].set_offsets(np.column_stack([painted_x, painted_y]) if painted_x else np.zeros((0, 2)))

                # Dynamic color objectives
                colors = []
                for o in objectives:
                    if o[4]:
                        colors.append((1.0, 1.0 - o[3], 1.0 - o[3]))
                if colors:
                    artists["painted"].set_color(colors)

                # Move sense rings
                for agent_id, circle in artists["sense_rings"]:
                    agent = next((a for a in agents if a["agent_id"] == agent_id), None)
                    if agent and agent["alive"]:
                        circle.center = (agent["x"], agent["y"])
                        circle.set_visible(True)
                    else:
                        circle.set_visible(False)

                painted_count = sum(1 for o in objectives if o[4])
                title.set_text(f"Stigmergy Swarm | Frame {frame} | Painted: {painted_count}/{len(objectives)}")

            elif algo == "cbba":
                drones_x = [a["x"] for a in agents if a["alive"]]
                drones_y = [a["y"] for a in agents if a["alive"]]
                artists["drones"].set_offsets(np.column_stack([drones_x, drones_y]) if drones_x else np.zeros((0, 2)))
                
                # Orange if assigned, blue if searching/outbid
                colors = ['orange' if a["state"] == 'comfortable' else 'blue' for a in agents if a["alive"]]
                artists["drones"].set_color(colors)

                targets_x = [o[1] for o in objectives]
                targets_y = [o[2] for o in objectives]
                artists["targets"].set_offsets(np.column_stack([targets_x, targets_y]))

                assigned = sum(1 for a in agents if a["state"] == "comfortable" and a["alive"])
                title.set_text(f"Decentralized CBBA Auction | Frame {frame} | Drones Assigned: {assigned}/{len(agents)}")

            return list(artists.values()) + [title]

        # FuncAnimation runs at requested interval
        ani = animation.FuncAnimation(fig, update_plot, frames=300, interval=self.interval, blit=False)
        plt.show()
        self.running = False

    def _render_to_file(self, save_path: str):
        bounds = self.sim.config.simulation.bounds
        algo = self.sim.config.simulation.algorithm.lower()
        max_frames = self.sim.config.simulation.max_frames
        dt = self.sim.config.simulation.dt

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_facecolor('#0d1117')
        fig.patch.set_facecolor('#0d1117')

        # Static threats
        if algo == "cbba":
            for t in self.sim.threats:
                circle = plt.Circle((t.x, t.y), t.radius, color='red', alpha=0.15)
                ax.add_patch(circle)
                ax.plot(t.x, t.y, 'r*', markersize=10)

        # Set up grid for Voronoi in Reynolds
        if algo == "reynolds":
            grid_res = 120
            x_grid = np.linspace(bounds[0], bounds[1], grid_res)
            y_grid = np.linspace(bounds[0], bounds[1], grid_res)
            xx, yy = np.meshgrid(x_grid, y_grid)
            extent = (bounds[0], bounds[1], bounds[0], bounds[1])
            voronoi_colors = plt.cm.tab20(np.linspace(0, 1, len(self.sim.agents)))

        images = []

        def fig_to_image(fig):
            canvas = fig.canvas
            canvas.draw()
            w, h = canvas.get_width_height()
            if hasattr(canvas, "tostring_rgb"):
                buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape((h, w, 3))
            else:
                buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
                buf = buf.reshape((h, w, 4))
                buf = buf[:, :, 1:4] # drop alpha
            return Image.fromarray(buf)

        print(f"Rendering {max_frames} frames to {save_path} offline...")

        for frame in range(max_frames):
            self.sim.step(dt)
            ax.cla()
            ax.set_xlim(bounds)
            ax.set_ylim(bounds)
            ax.set_facecolor('#0d1117')

            # Render frame static threats
            if algo == "cbba":
                for t in self.sim.threats:
                    circle = plt.Circle((t.x, t.y), t.radius, color='red', alpha=0.15)
                    ax.add_patch(circle)
                    ax.plot(t.x, t.y, 'r*', markersize=10)

            agents = self.sim.agents
            objectives = self.sim.objectives

            if algo == "reynolds":
                # Compute Voronoi coloring
                pts = np.array([[a.x, a.y] for a in agents if a.alive])
                if len(pts) > 0:
                    min_dist2 = np.full(xx.shape, np.inf)
                    closest_idx = np.zeros(xx.shape, dtype=int)
                    for i, (px, py) in enumerate(pts):
                        d2 = (xx - px) ** 2 + (yy - py) ** 2
                        mask = d2 < min_dist2
                        closest_idx[mask] = i
                        min_dist2[mask] = d2[mask]
                    
                    color_grid = voronoi_colors[closest_idx][:, :, :3]
                    ax.imshow(color_grid, extent=extent, origin='lower', alpha=0.3, zorder=1)

                for a in agents:
                    if a.alive:
                        color = 'yellow' if a.state == 'lonely' else \
                                'red'    if a.state == 'crowded' else 'cyan'
                        ax.plot(a.x, a.y, 'o', color=color, markersize=8, zorder=5)

                lonely = sum(1 for a in agents if a.state == "lonely" and a.alive)
                crowded = sum(1 for a in agents if a.state == "crowded" and a.alive)
                comfy = sum(1 for a in agents if a.state == "comfortable" and a.alive)
                ax.set_title(f'Voronoi Swarm | Frame {frame} | Lonely: {lonely}  Crowded: {crowded}  Comfortable: {comfy}', color='white')

            elif algo == "stigmergy":
                scouts = [a for a in agents if a.alive and getattr(a, "species", "") == "scout"]
                workers = [a for a in agents if a.alive and getattr(a, "species", "") == "worker"]

                ax.scatter([s.x for s in scouts], [s.y for s in scouts], color='#ff7b00', marker='o', s=80, label='Scout', zorder=5)
                ax.scatter([w.x for w in workers], [w.y for w in workers], color='#58a6ff', marker='o', s=60, label='Worker', zorder=5)

                unpainted_x = [o.x for o in objectives if not o.painted]
                unpainted_y = [o.y for o in objectives if not o.painted]
                painted_x = [o.x for o in objectives if o.painted]
                painted_y = [o.y for o in objectives if o.painted]

                ax.scatter(unpainted_x, unpainted_y, color='#6e7681', marker='s', s=100, label='Unpainted Target', zorder=3)
                for o in objectives:
                    if o.painted:
                        ax.scatter([o.x], [o.y], color=(1.0, 1.0 - o.paint_strength, 1.0 - o.paint_strength), marker='s', s=110, zorder=3)

                for agent in agents:
                    if agent.alive:
                        circle = plt.Circle(
                            (agent.x, agent.y),
                            agent.sense_radius,
                            fill=False,
                            linestyle="--",
                            linewidth=0.6,
                            alpha=0.25,
                            color="#ff7b00" if getattr(agent, "species", "") == "scout" else "#58a6ff"
                        )
                        ax.add_patch(circle)

                painted_count = sum(1 for o in objectives if o.painted)
                ax.set_title(f"Stigmergy Swarm | Frame {frame} | Painted: {painted_count}/{len(objectives)}", color='white')

            elif algo == "cbba":
                drones_x = [a.x for a in agents if a.alive]
                drones_y = [a.y for a in agents if a.alive]
                colors = ['orange' if a.assigned_task_id is not None else 'blue' for a in agents if a.alive]
                ax.scatter(drones_x, drones_y, c=colors, marker='v', s=60, label='Drone Node', zorder=5)

                targets_x = [o.x for o in objectives]
                targets_y = [o.y for o in objectives]
                ax.scatter(targets_x, targets_y, color='green', marker='X', s=120, label='Target Objective', zorder=3)

                assigned = sum(1 for a in agents if a.assigned_task_id is not None and a.alive)
                ax.set_title(f"Decentralized CBBA Auction | Frame {frame} | Drones Assigned: {assigned}/{len(agents)}", color='white')

            images.append(fig_to_image(fig))

        # Save GIF
        images[0].save(
            save_path,
            save_all=True,
            append_images=images[1:],
            duration=self.interval,
            loop=0
        )
        plt.close(fig)
        print(f"Successfully saved animation to {save_path}")
