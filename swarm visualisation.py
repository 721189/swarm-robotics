import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

class Robot:
    def __init__(self, robot_id, x, y):
        self.robot_id = robot_id
        self.x = x
        self.y = y
        self.state = "comfortable"  # NEW — tracks robot's current state

    def decide_move(self, all_robots):
        nearby = []
        for other in all_robots:
            if other.robot_id != self.robot_id:
                distance = ((self.x - other.x)**2 + (self.y - other.y)**2) ** 0.5
                if distance <= 5:
                    nearby.append(other)

        if len(nearby) == 0:
            self.x += 1 if self.x < 0 else -1
            self.y += 1 if self.y < 0 else -1
            self.state = "lonely"        # YELLOW

        elif len(nearby) >= 3:
            self.x += random.choice([-1, 1])
            self.y += random.choice([-1, 1])
            self.state = "crowded"       # RED

        else:
            self.state = "comfortable"   # BLUE

# Build swarm
swarm = [Robot(i, x=random.randint(-20, 20), y=random.randint(-20, 20)) for i in range(30)]

# Color map
def get_color(state):
    if state == "lonely":     return "yellow"
    if state == "crowded":    return "red"
    return "cyan"

# Set up visual
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# One dot per robot
dots = [ax.plot([], [], 'o', markersize=7)[0] for _ in range(len(swarm))]

# Legend
ax.plot([], [], 'o', color='cyan',   label='Comfortable')
ax.plot([], [], 'o', color='red',    label='Crowded → spreading')
ax.plot([], [], 'o', color='yellow', label='Lonely → moving to center')
ax.legend(loc='upper right', facecolor='black', labelcolor='white')

title = ax.set_title('', color='white', fontsize=11)

def update(frame):
    for robot in swarm:
        robot.decide_move(swarm)

    for i, robot in enumerate(swarm):
        dots[i].set_data([robot.x], [robot.y])
        dots[i].set_color(get_color(robot.state))

    # Count states
    lonely    = sum(1 for r in swarm if r.state == "lonely")
    crowded   = sum(1 for r in swarm if r.state == "crowded")
    comfy     = sum(1 for r in swarm if r.state == "comfortable")

    title.set_text(f'Frame {frame} | Lonely: {lonely}  Crowded: {crowded}  Comfortable: {comfy}')
    return dots + [title]

ani = animation.FuncAnimation(fig, update, frames=300, interval=100, blit=True)
plt.show()