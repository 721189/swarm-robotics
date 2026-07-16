import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- SYSTEM TOPOLOGY & CONFIGURATION ---
NUM_SCOUTS = 5
NUM_WORKERS = 20
NUM_OBJECTIVES = 5
SPACE_BOUNDS = 100.0

DECAY_RATE = 0.005           # Evaporation coefficient per frame
MIN_PHEROMONE_THRESHOLD = 0.1 # Minimum scalar intensity required for Worker detection

# --- VECTOR UTILITIES ---
def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def normalize(dx, dy):
    mag = math.hypot(dx, dy)
    if mag == 0:
        return 0, 0
    return dx / mag, dy / mag

# --- ENVIRONMENT MARKERS ---
class Objective:
    def __init__(self, obj_id, x, y):
        self.obj_id = obj_id
        self.x = x
        self.y = y
        self.paint_strength = 0.0

    def apply_paint(self, amount=1.0):
        # Mediated stigmergy: Paint stacks but cannot exceed maximum capacity of 1.0
        self.paint_strength = min(1.0, self.paint_strength + amount)

    def decay(self):
        # Environmental evaporation applied every frame
        if self.paint_strength > 0:
            self.paint_strength -= DECAY_RATE
            if self.paint_strength <= 0:
                self.paint_strength = 0.0

# --- BASE KINEMATIC CLASS ---
class Agent:
    def __init__(self, agent_id, x, y, speed, sense_radius, battery, drain):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.speed = speed
        self.sense_radius = sense_radius
        self.battery = battery
        self.drain = drain
        self.active = True
        
        # Initialize random trajectory
        angle = random.uniform(0, 2 * math.pi)
        self.dx = math.cos(angle)
        self.dy = math.sin(angle)

    def move(self):
        if not self.active:
            return
        
        # Thermodynamic drain
        self.battery -= self.drain
        if self.battery <= 0:
            self.battery = 0
            self.active = False
            return

        # Kinematic update
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed

        # Bounding box collision
        if self.x < 0 or self.x > SPACE_BOUNDS:
            self.dx *= -1
            self.x = max(0.0, min(self.x, SPACE_BOUNDS))
        if self.y < 0 or self.y > SPACE_BOUNDS:
            self.dy *= -1
            self.y = max(0.0, min(self.y, SPACE_BOUNDS))

# --- ASYMMETRIC CLASSES ---
class Scout(Agent):
    def __init__(self, agent_id, x, y):
        # High Velocity, High Perception, High Drain
        super().__init__(agent_id, x, y, speed=2.0, sense_radius=18.0, battery=80.0, drain=0.22)

    def step(self, objectives):
        if not self.active:
            return

        closest_obj = None
        min_dist = float('inf')
        
        # Information Gatherer: Scans raw objectives
        for obj in objectives:
            dist = distance(self.x, self.y, obj.x, obj.y)
            if dist <= self.sense_radius and dist < min_dist:
                closest_obj = obj
                min_dist = dist
                
        if closest_obj:
            if min_dist < 2.0:
                closest_obj.apply_paint(1.0)
                # Scatter after painting to explore new sectors
                angle = random.uniform(0, 2 * math.pi)
                self.dx = math.cos(angle)
                self.dy = math.sin(angle)
            else:
                self.dx, self.dy = normalize(closest_obj.x - self.x, closest_obj.y - self.y)
        
        self.move()

class Worker(Agent):
    def __init__(self, agent_id, x, y):
        # Low Velocity, Low Perception, Low Drain
        super().__init__(agent_id, x, y, speed=1.0, sense_radius=7.0, battery=220.0, drain=0.04)

    def step(self, objectives):
        if not self.active:
            return

        # Cognitive Filter: Only perceives targets with active chemical signatures
        valid_targets = [
            obj for obj in objectives 
            if distance(self.x, self.y, obj.x, obj.y) <= self.sense_radius 
            and obj.paint_strength >= MIN_PHEROMONE_THRESHOLD
        ]
        
        if valid_targets:
            # Exploitation: Prioritize the highest intensity pheromone
            target = max(valid_targets, key=lambda o: o.paint_strength)
            dist = distance(self.x, self.y, target.x, target.y)
            
            if dist > 1.0:
                self.dx, self.dy = normalize(target.x - self.x, target.y - self.y)
            else:
                # Idle state: Preserves energy upon reaching objective
                self.dx, self.dy = 0, 0
                self.battery -= (self.drain / 4) 
                return 
        else:
            # Brownian motion / Foraging drift
            if random.random() < 0.05:
                angle = random.uniform(0, 2 * math.pi)
                self.dx = math.cos(angle)
                self.dy = math.sin(angle)

        self.move()

# --- SIMULATION ENGINE ---
def run_simulation():
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Initialize Entities
    objectives = [Objective(i, random.uniform(10, SPACE_BOUNDS-10), random.uniform(10, SPACE_BOUNDS-10)) for i in range(NUM_OBJECTIVES)]
    scouts = [Scout(i, random.uniform(0, SPACE_BOUNDS), random.uniform(0, SPACE_BOUNDS)) for i in range(NUM_SCOUTS)]
    workers = [Worker(i, random.uniform(0, SPACE_BOUNDS), random.uniform(0, SPACE_BOUNDS)) for i in range(NUM_WORKERS)]
    agents = scouts + workers

    # Render plots
    scout_scatter = ax.scatter([], [], c='blue', marker='^', label='Scout')
    worker_scatter = ax.scatter([], [], c='green', marker='o', label='Worker')
    obj_scatter = ax.scatter([o.x for o in objectives], [o.y for o in objectives], c='gray', marker='x', s=100, label='Objective (Hidden)')

    ax.set_xlim(0, SPACE_BOUNDS)
    ax.set_ylim(0, SPACE_BOUNDS)
    ax.legend(loc='upper right')
    ax.set_title("Mediated Stigmergy with Pheromone Decay")

    def animate(frame):
        # 1. Update Environment
        for obj in objectives:
            obj.decay()
            
        # 2. Update Agents
        for agent in agents:
            agent.step(objectives)
            
        # 3. Render State
        active_scouts = [s for s in scouts if s.active]
        active_workers = [w for w in workers if w.active]
        
        if active_scouts:
            scout_scatter.set_offsets([[s.x, s.y] for s in active_scouts])
        else:
            scout_scatter.set_offsets(math.nan) 
            
        if active_workers:
            worker_scatter.set_offsets([[w.x, w.y] for w in active_workers])
            
        # Dynamically color objectives based on pheromone intensity
        obj_colors = []
        for obj in objectives:
            if obj.paint_strength >= MIN_PHEROMONE_THRESHOLD:
                # Transitions from light red to solid red based on intensity
                obj_colors.append((1.0, 1.0 - obj.paint_strength, 1.0 - obj.paint_strength))
            else:
                obj_colors.append('gray')
                
        obj_scatter.set_color(obj_colors)
        
        return scout_scatter, worker_scatter, obj_scatter

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    run_simulation()