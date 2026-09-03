from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SlotEvent:
    speaker_id: int
    start_time: float
    duration: float


class TDMAScheduler:
    """
    Event-driven TDMA scheduler.

    The physics timestep and radio slot duration are independent.
    Multiple radio slots may therefore occur inside one physics step.
    """

    def __init__(self, num_drones: int, slot_duration_ms: float = 50.0):
        if num_drones < 1:
            raise ValueError("num_drones must be >= 1")
        if slot_duration_ms <= 0:
            raise ValueError("slot_duration_ms must be > 0")

        self.num_drones = int(num_drones)
        self.slot_duration = float(slot_duration_ms) / 1000.0

        self.current_time = 0.0
        self.current_speaker = 0
        self.time_left_in_slot = self.slot_duration

        self.double_slot_requests = {
            drone_id: False for drone_id in range(self.num_drones)
        }

    def request_double_slot(self, drone_id: int) -> None:
        if 0 <= drone_id < self.num_drones:
            self.double_slot_requests[drone_id] = True

    def _duration_for(self, drone_id: int) -> float:
        if self.double_slot_requests.get(drone_id, False):
            return 2.0 * self.slot_duration
        return self.slot_duration

    def _start_next_slot(self) -> None:
        self.current_speaker = (
            self.current_speaker + 1
        ) % self.num_drones

        self.time_left_in_slot = self._duration_for(
            self.current_speaker
        )

        # A request is one-shot.
        self.double_slot_requests[self.current_speaker] = False

    def advance(self, dt: float) -> List[SlotEvent]:
        """
        Advance the radio clock by dt seconds.

        Returns every slot that overlaps the interval
        [old_time, old_time + dt).
        """
        if dt < 0:
            raise ValueError("dt must be >= 0")

        events: List[SlotEvent] = []
        remaining = float(dt)

        eps = 1e-12

        while remaining > eps:
            slot_start = self.current_time

            # Record the slot currently occupying the radio.
            events.append(
                SlotEvent(
                    speaker_id=self.current_speaker,
                    start_time=slot_start,
                    duration=self.time_left_in_slot,
                )
            )

            consume = min(self.time_left_in_slot, remaining)

            self.current_time += consume
            self.time_left_in_slot -= consume
            remaining -= consume

            if self.time_left_in_slot <= eps:
                self._start_next_slot()

        return events

    def update(self, dt: float) -> List[SlotEvent]:
        """
        Backwards-compatible alias for advance().
        """
        return self.advance(dt)

    def get_current_speaker(self) -> Optional[int]:
        return self.current_speaker

    def get_time_until_next_slot(self, drone_id: int) -> float:
        if not 0 <= drone_id < self.num_drones:
            raise ValueError("invalid drone_id")

        if drone_id == self.current_speaker:
            return 0.0

        wait = self.time_left_in_slot
        speaker = (self.current_speaker + 1) % self.num_drones

        while speaker != drone_id:
            wait += self._duration_for(speaker)
            speaker = (speaker + 1) % self.num_drones

        return wait

