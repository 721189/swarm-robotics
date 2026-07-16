import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- SIMULATION CONFIGURATION ---
SPACE_SIZE = 100.0
NUM_AGENTS = 12
NUM_OBJECTIVES = 5
COMMUNICATION_RANGE = 25.0  # Max distance for decentralized ledger gossip

# --- WEIGHTED FORCE CONFIGURATION ---
ATTRACTIVE_WEIGHT = 1.0     # Pull toward assigned target
SAM_REPULSION_WEIGHT = 4.5  # Push away from radar threats (Must be high for safety)
BOIDS_SEPARATION_WEIGHT = 1.5 # Avoid colliding with other drones

# --- UTILITY MATH FUNCTIONS ---
def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def normalize(dx, dy):
    mag = math.hypot(dx, dy)
    if mag == 0:
        return 0.0, 0.0
    return dx / mag, dy / mag

# --- DEFENSE LAYER CLASSES ---
class SAMThreat:
    """Represents a Surface-to-Air Missile / Radar engagement bubble."""
    def __init__(self, threat_id, x, y, radius, strength=50.0):
        self.threat_id = threat_id
        self.x = x
        self.y = y
        self.radius = radius
        self.strength = strength

class Objective:
    """Target location that requires execution by exactly one drone."""
    def __init__(self, obj_id, x, y):
        self.obj_id = obj_id
        self.x = x
        self.y = y

# --- DECENRALIZED AGENT CLASS ---
class CombatDrone:
    def __init__(self, drone_id, x, y):
        self.drone_id = drone_id
        self.x = x
        self.y = y
        self.speed = 1.5
        self.max_force = 0.2
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.vx, self.vy = normalize(self.vx, self.vy)
        
        # --- LOCAL AUCTION LEDGERS (CBBA Style) ---
        self.assigned_task_id = None
        self.local_winners = {i: -1 for i in range(NUM_OBJECTIVES)}  # Maps Obj ID -> Winning Drone ID
        self.local_bids = {i: 0.0 for i in range(NUM_OBJECTIVES)}     # Maps Obj ID -> Highest Known Bid

    def calculate_local_bid(self, objective):
        """Calculates value of a target based strictly on proximity."""
        dist = distance(self.x, self.y, objective.x, objective.y)
        # Bid drops linearly with distance. If too far, bid is zero.
        return max(0.1, 100.0 - dist)

    def run_local_auction(self, objectives):
        """Phase 1: Bundle/Bidding Phase.

        Evaluates if drone can outbid current known owners.
        """
        best_target_id = None
        highest_net_gain = -1.0

        for obj in objectives:
            my_bid = self.calculate_local_bid(obj)
            # Check if my calculated bid beats my current knowledge of the winning bid
            if my_bid > self.local_bids[obj.obj_id]:
                gain = my_bid - self.local_bids[obj.obj_id]
                if gain > highest_net_gain:
                    highest_net_gain = gain
                    best_target_id = obj.obj_id

        # If I can successfully outbid a target, claim ownership locally
        if best_target_id is not None:
            # If I was previously owning another task, reset its bid in my table
            if self.assigned_task_id is not None and self.assigned_task_id != best_target_id:
                self.local_bids[self.assigned_task_id] = 0.0
                self.local_winners[self.assigned_task_id] = -1
            
            self.assigned_task_id = best_target_id
            self.local_bids[best_target_id] = self.calculate_local_bid(objectives[best_target_id])
            self.local_winners[best_target_id] = self.drone_id

    def sync_ledgers(self, neighbors):
        """Phase 2: Consensus Phase.

        Gossip network exchanges tracking tables to resolve conflicts.
        """
        for neighbor in neighbors:
            for obj_id in range(NUM_OBJECTIVES):
                # If neighbor knows of a higher bid for this objective, update local ledger
                if neighbor.local_bids[obj_id] > self.local_bids[obj_id]:
                    self.local_bids[obj_id] = neighbor.local_bids[obj_id]
                    self.local_winners[obj_id] = neighbor.local_winners[obj_id]
                    
                    # CATASTROPHIC OUTBID GATING: If I just discovered I was outbid on my active task, drop it
                    if self.assigned_task_id == obj_id and self.local_winners[obj_id] != self.drone_id:
                        self.assigned_task_id = None

    def compute_steering_forces(self, objectives, threats, all_drones):
        """Computes Artificial Potential Fields + Flocking Separation."""
        fx, fy = 0.0, 0.0

        # 1. ATTRACTIVE POTENTIAL FIELD (Task Assignment Target)
        if self.assigned_task_id is not None:
            target = objectives[self.assigned_task_id]
            tax, tay = normalize(target.x - self.x, target.y - self.y)
            fx += tax * ATTRACTIVE_WEIGHT
            fy += tay * ATTRACTIVE_WEIGHT
        else:
            # Idle Brownian drift if outbid / unassigned
            fx += self.vx * 0.2
            fy += self.vy * 0.2

        # 2. REPULSIVE POTENTIAL FIELD (SAM Threat Barrier Method)
        for threat in threats:
            dist_to_threat = distance(self.x, self.y, threat.x, threat.y)
            if dist_to_threat < threat.radius:
                dist_to_threat = max(dist_to_threat, 1.0) # Prevent division by zero
                # Inverse-square barrier calculation: spikes massively near threat center
                repulsion_mag = threat.strength * ((1.0 / dist_to_threat) - (1.0 / threat.radius))**2
                trx, try_ = normalize(self.x - threat.x, self.y - threat.y)
                fx += trx * repulsion_mag * SAM_REPULSION_WEIGHT
                fy += try_ * repulsion_mag * SAM_REPULSION_WEIGHT

        # 3. REYNOLDS BOIDS SEPARATION (Prevent drone-on-drone collision)
        sep_x, sep_y = 0.0, 0.0
        for other in all_drones:
            if other.drone_id != self.drone_id:
                d = distance(self.x, self.y, other.x, other.y)
                if d < 4.0: # Close proximity boundary
                    d = max(d, 0.5)
                    dx, dy = normalize(self.x - other.x, self.y - other.y)
                    sep_x += dx / d
                    sep_y += dy / d
        
        fx += sep_x * BOIDS_SEPARATION_WEIGHT
        fy += sep_y * BOIDS_SEPARATION_WEIGHT

        # Application of kinematic limits
        fx, fy = normalize(fx, fy)
        self.vx += fx * self.max_force
        self.vy += fy * self.max_force
        self.vx, self.vy = normalize(self.vx, self.vy)

    def execute_kinematics(self):
        """Updates physics space locations and applies boundaries."""
        self.x += self.vx * self.speed
        self.y += self.vy * self.speed

        # Hard physical boundary deflection
        if self.x < 0 or self.x > SPACE_SIZE:
            self.vx *= -1
            self.x = max(0.0, min(self.x, SPACE_SIZE))
        if self.y < 0 or self.y > SPACE_SIZE:
            self.vy *= -1
            self.y = max(0.0, min(self.y, SPACE_SIZE))

# --- MAIN RUNNER AND ANIMATION GRID ---
def run_defense_simulation():
    # Setup Entities
    threats = [
        SAMThreat(0, 30.0, 40.0, radius=15.0, strength=60.0),
        SAMThreat(1, 70.0, 65.0, radius=18.0, strength=70.0),
        SAMThreat(2, 50.0, 20.0, radius=12.0, strength=50.0)
    ]
    
    objectives = [
        Objective(0, 15.0, 85.0),
        Objective(1, 85.0, 85.0),
        Objective(2, 50.0, 50.0), # Trapped directly between radar zones!
        Objective(3, 85.0, 15.0),
        Objective(4, 15.0, 15.0)
    ]
    
    drones = [CombatDrone(i, random.uniform(5, 25), random.uniform(5, 95)) for i in range(NUM_AGENTS)]

    # Matplotlib Plot Layout Init
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(0, SPACE_SIZE)
    ax.set_ylim(0, SPACE_SIZE)
    
    # Render static structures
    for t in threats:
        circle = plt.Circle((t.x, t.y), t.radius, color='red', alpha=0.15, label='SAM Engagement Zone' if t.threat_id == 0 else "")
        ax.add_patch(circle)
        ax.plot(t.x, t.y, 'r*', markersize=10)

    obj_scatter = ax.scatter([o.x for o in objectives], [o.y for o in objectives], c='green', marker='X', s=120, label='Target Objective')
    drone_scatter = ax.scatter([], [], c='blue', marker='v', s=60, label='Autonomous Drone Node')
    
    ax.legend(loc='upper right')
    ax.set_title("Decentralized Auction (CBBA) & SAM Artificial Potential Fields")

    def update(frame):
        # 1. Distributed Consensus Network Check
        for d in drones:
            # Find neighbors within wireless radio range
            neighbors = [other for other in drones if other.drone_id != d.drone_id and distance(d.x, d.y, other.x, other.y) <= COMMUNICATION_RANGE]
            d.sync_ledgers(neighbors)
        
        # 2. Local Re-Bidding Calculation
        for d in drones:
            d.run_local_auction(objectives)

        # 3. Dynamic Steering Execution
        for d in drones:
            d.compute_steering_forces(objectives, threats, drones)
            d.execute_kinematics()

        # 4. Refresh UI State Elements
        drone_scatter.set_offsets([[d.x, d.y] for d in drones])
        
        # Color drones dynamically: Gold if executing a task, Blue if searching/outbid
        colors = ['orange' if d.assigned_task_id is not None else 'blue' for d in drones]
        drone_scatter.set_color(colors)

        return drone_scatter, obj_scatter

    ani = animation.FuncAnimation(fig, update, frames=300, interval=60, blit=True)
    plt.show()

if __name__ == "__main__":
    run_defense_simulation()