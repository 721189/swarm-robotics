import math
from typing import Optional

class TDMAScheduler:
    """
    The heart of the radio discipline.
    Determines WHO gets to speak at exactly this millisecond.
    """
    def __init__(self, num_drones: int, slot_duration_ms: float = 50.0):
        # Convert to seconds for simulation math
        self.slot_duration = slot_duration_ms / 1000.0 
        self.num_drones = num_drones
        self.frame_duration = self.slot_duration * num_drones
        self.current_time = 0.0

    def update(self, dt: float):
        """Advance the global radio clock."""
        self.current_time += dt

    def get_current_speaker(self) -> Optional[int]:
        """
        Returns the Drone ID that has the right to transmit right now.
        Returns None if no drone is speaking (simulates idle radio time).
        """
        if self.frame_duration == 0 or self.num_drones == 0:
            return None
        
        # Determine which slot we are in based on the time modulo the frame length
        slot_index = int((self.current_time % self.frame_duration) // self.slot_duration)
        
        # Ensure we don't exceed the number of drones
        if slot_index < self.num_drones:
            return slot_index
        return None

    def get_time_until_next_slot(self, drone_id: int) -> float:
        """How many seconds until THIS specific drone gets to speak?"""
        current_speaker = self.get_current_speaker()
        if current_speaker is None:
            return self.slot_duration # Fallback
            
        if current_speaker <= drone_id:
            slots_to_wait = drone_id - current_speaker
        else:
            # We need to wait until the next frame cycle
            slots_to_wait = (self.num_drones - current_speaker) + drone_id
            
        return slots_to_wait * self.slot_duration
