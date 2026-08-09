import unittest
from src.communication.tdma_scheduler import TDMAScheduler
from src.algorithms.cbba import CombatDroneAgent

class TestTDMAScheduler(unittest.TestCase):
    def test_tdma_slot_sequence(self):
        scheduler = TDMAScheduler(num_drones=3, slot_duration_ms=50.0)
        # dt is in seconds, slot duration is 0.05 seconds
        self.assertEqual(scheduler.get_current_speaker(), 0)
        
        # Advance time by 0.02s
        scheduler.update(0.02)
        self.assertEqual(scheduler.get_current_speaker(), 0)
        
        # Advance time to 0.06s (slot 1)
        scheduler.update(0.04)
        self.assertEqual(scheduler.get_current_speaker(), 1)
        
        # Advance time to 0.11s (slot 2)
        scheduler.update(0.05)
        self.assertEqual(scheduler.get_current_speaker(), 2)
        
        # Advance time to 0.16s (wraps back to slot 0)
        scheduler.update(0.05)
        self.assertEqual(scheduler.get_current_speaker(), 0)

    def test_drone_telemetry_mailbox(self):
        agent = CombatDroneAgent(agent_id=1, x=10.0, y=20.0, params={})
        
        # Initially mailbox is empty
        self.assertEqual(len(agent.message_mailbox), 0)
        
        # Prepare broadcast
        telemetry = agent.prepare_broadcast(current_time=1.0)
        self.assertEqual(telemetry["pos"], (10.0, 20.0))
        self.assertEqual(telemetry["timestamp"], 1.0)
        
        # Receive broadcast from agent 2
        agent_2_data = {"pos": (30.0, 40.0), "task_id": 3, "battery": 90.0, "timestamp": 1.0}
        agent.receive_broadcast(sender_id=2, data=agent_2_data)
        self.assertEqual(len(agent.message_mailbox), 1)
        self.assertIn(2, agent.message_mailbox)
        
        # Receive broadcast from self should be ignored
        agent.receive_broadcast(sender_id=1, data=agent_2_data)
        self.assertEqual(len(agent.message_mailbox), 1)

    def test_perceived_world_timeout(self):
        agent = CombatDroneAgent(agent_id=0, x=0.0, y=0.0, params={})
        
        # Receive stale message and fresh message
        agent.receive_broadcast(sender_id=1, data={"pos": (1.0, 1.0), "timestamp": 0.5})
        agent.receive_broadcast(sender_id=2, data={"pos": (2.0, 2.0), "timestamp": 2.2})
        
        # Get perceived world at current_time = 2.5 (max_age = 2.0)
        # Message 1 timestamp = 0.5 (age 2.0s, boundary - valid)
        # Message 2 timestamp = 2.2 (age 0.3s, valid)
        perceived = agent.get_perceived_world(current_time=2.5, max_age=2.0)
        self.assertEqual(len(perceived), 2)
        
        # Get perceived world at current_time = 3.0 (max_age = 2.0)
        # Message 1 timestamp = 0.5 (age 2.5s, expired)
        # Message 2 timestamp = 2.2 (age 0.8s, valid)
        perceived_later = agent.get_perceived_world(current_time=3.0, max_age=2.0)
        self.assertEqual(len(perceived_later), 1)
        self.assertIn(2, perceived_later)
        self.assertNotIn(1, perceived_later)

if __name__ == "__main__":
    unittest.main()
