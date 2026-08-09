import math
from typing import Optional

class TDMAScheduler:
    """
    The heart of the radio discipline.
    Determines WHO gets to speak at exactly this millisecond.
    Supports stateful Dynamic TDMA (D-TDMA) slot allocation.
    """
    def __init__(self, num_drones: int, slot_duration_ms: float = 50.0):
        # Convert to seconds for simulation math
        self.slot_duration = slot_duration_ms / 1000.0 
        self.num_drones = num_drones
        self.current_time = 0.0

        # Stateful D-TDMA scheduling
        self.current_speaker = 0
        self.time_left_in_slot = self.slot_duration
        self.double_slot_requests = {}  # Maps drone_id -> bool

    def update(self, dt: float):
        """Advance the global radio clock and update current slot/speaker state."""
        self.current_time += dt
        self.time_left_in_slot -= dt

        if self.time_left_in_slot <= 0:
            # Switch to the next drone
            self.current_speaker = (self.current_speaker + 1) % self.num_drones
            
            # Grant double slot if requested
            if self.double_slot_requests.get(self.current_speaker, False):
                self.time_left_in_slot = 2.0 * self.slot_duration
                # Clear request after granting
                self.double_slot_requests[self.current_speaker] = False
            else:
                self.time_left_in_slot = self.slot_duration

    def request_double_slot(self, drone_id: int):
        """Register a demand for a double transmission slot (D-TDMA)."""
        self.double_slot_requests[drone_id] = True

    def get_current_speaker(self) -> Optional[int]:
        """Returns the Drone ID that has the right to transmit right now."""
        if self.num_drones == 0:
            return None
        return self.current_speaker

    def get_time_until_next_slot(self, drone_id: int) -> float:
        """How many seconds until THIS specific drone gets to speak?"""
        current_speaker = self.get_current_speaker()
        if current_speaker is None:
            return self.slot_duration
            
        if current_speaker <= drone_id:
            slots_to_wait = drone_id - current_speaker
        else:
            slots_to_wait = (self.num_drones - current_speaker) + drone_id
            
        return slots_to_wait * self.slot_duration

