"""Voronoi swarm demo (clean version).

Run:
    python voronoi_demo.py --out voronoi.gif

This script saves an animation file so you can view the result even when a GUI display
is not available.
"""

import argparse
import random
from typing import Optional, Tuple

from PIL import Image
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# Reproducible behavior across runs
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


class Robot:
    def __init__(self, robot_id: int, x: float, y: float):
        self.robot_id = robot_id
        self.x = x
        self.y = y
        self.state = "comfortable"

    def decide_move(self, all_robots, bounds=(-30, 30)):
        nearby = []
        for other in all_robots:
            if other.robot_id == self.robot_id:
                continue
            distance = ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
            if distance <= 8:
                nearby.append(other)

        dx = dy = 0.0
        if len(nearby) == 0:
            dx = -self.x
            dy = -self.y
            self.state = "lonely"

        elif len(nearby) >= 4:
            avg_x = sum(r.x for r in nearby) / len(nearby)
            avg_y = sum(r.y for r in nearby) / len(nearby)
            dx = self.x - avg_x
            dy = self.y - avg_y
            self.state = "crowded"

        else:
            self.state = "comfortable"

        length = (dx**2 + dy**2) ** 0.5
        if length > 0:
            dx /= length
            dy /= length

        self.x = clamp(self.x + dx, bounds[0] + 1, bounds[1] - 1)
        self.y = clamp(self.y + dy, bounds[0] + 1, bounds[1] - 1)


def draw_voronoi(ax, swarm, xx, yy, colors, extent):
    # Compute a Voronoi-style coloring by finding the nearest robot for each grid point.
    pts = np.array([[r.x, r.y] for r in swarm])

    min_dist2 = np.full(xx.shape, np.inf)
    closest_idx = np.zeros(xx.shape, dtype=int)

    for i, (px, py) in enumerate(pts):
        d2 = (xx - px) ** 2 + (yy - py) ** 2
        mask = d2 < min_dist2
        closest_idx[mask] = i
        min_dist2[mask] = d2[mask]

    color_grid = colors[closest_idx][:, :, :3]

    ax.imshow(
        color_grid,
        extent=extent,
        origin='lower',
        alpha=0.4,
    )


def run_swarm(
    num_robots: int = 30,
    bounds: Tuple[float, float] = (-30, 30),
    frames: int = 200,
    interval: int = 150,
    grid_size: int = 60,
    resolution: int = 150,
    save_path: Optional[str] = None,
):
    swarm = [
        Robot(
            i,
            x=random.uniform(bounds[0] + 1, bounds[1] - 1),
            y=random.uniform(bounds[0] + 1, bounds[1] - 1),
        )
        for i in range(num_robots)
    ]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    half = grid_size / 2
    x_grid = np.linspace(-half, half, resolution)
    y_grid = np.linspace(-half, half, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    extent = (-half, half, -half, half)

    colors = plt.cm.tab20(np.linspace(0, 1, len(swarm)))

    def update(frame):
        ax.cla()
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_facecolor('black')

        draw_voronoi(ax, swarm, xx, yy, colors, extent)

        for robot in swarm:
            robot.decide_move(swarm, bounds=bounds)

        for robot in swarm:
            color = 'yellow' if robot.state == 'lonely' else \
                    'red'    if robot.state == 'crowded' else 'white'
            ax.plot(robot.x, robot.y, 'o', color=color, markersize=8, zorder=5)

        lonely = sum(1 for r in swarm if r.state == "lonely")
        crowded = sum(1 for r in swarm if r.state == "crowded")
        comfy = sum(1 for r in swarm if r.state == "comfortable")

        ax.set_title(
            f'Voronoi Swarm | Frame {frame} | '
            f'Lonely: {lonely}  Crowded: {crowded}  Comfortable: {comfy}',
            color='white',
            fontsize=10,
        )

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)

    if save_path:
        # GIF saving via Pillow is more reliable than matplotlib's writer on some systems.
        if save_path.lower().endswith('.gif'):
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
                    buf = buf[:, :, 1:4]

                return Image.fromarray(buf)

            images = []
            for frame in range(frames):
                update(frame)
                images.append(fig_to_image(fig))

            images[0].save(
                save_path,
                save_all=True,
                append_images=images[1:],
                duration=interval,
                loop=0,
            )
            plt.close(fig)
            return

        writer = None
        if save_path.lower().endswith(('.mp4', '.mov')):
            writer = 'ffmpeg'

        ani.save(save_path, fps=1000 / interval, dpi=150, writer=writer)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Voronoi swarm animation")
    parser.add_argument("--robots", type=int, default=30, help="Number of robots")
    parser.add_argument("--frames", type=int, default=200, help="Animation frames")
    parser.add_argument("--interval", type=int, default=150, help="Milliseconds between frames")
    parser.add_argument("--grid-size", type=int, default=60, help="Voronoi grid size")
    parser.add_argument("--resolution", type=int, default=150, help="Voronoi grid resolution (lower is faster)")
    parser.add_argument("--out", type=str, default=None, help="Path to save animation (mp4/gif)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_swarm(
        num_robots=args.robots,
        frames=args.frames,
        interval=args.interval,
        grid_size=args.grid_size,
        resolution=args.resolution,
        save_path=args.out,
    )
