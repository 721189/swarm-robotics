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

    def test_dynamic_slot_allocation(self):
        scheduler = TDMAScheduler(num_drones=3, slot_duration_ms=50.0)
        self.assertEqual(scheduler.get_current_speaker(), 0)
        
        # Request double slot for speaker 1
        scheduler.request_double_slot(drone_id=1)
        
        # Advance through drone 0's slot (0.05s)
        scheduler.update(0.05)
        self.assertEqual(scheduler.get_current_speaker(), 1)
        # Drone 1 should get 2 * 0.05 = 0.10s
        self.assertAlmostEqual(scheduler.time_left_in_slot, 0.10)
        
        # Advance by 0.05s, drone 1 should still be speaking
        scheduler.update(0.05)
        self.assertEqual(scheduler.get_current_speaker(), 1)
        
        # Advance by 0.05s, scheduler should transition to drone 2
        scheduler.update(0.05)
        self.assertEqual(scheduler.get_current_speaker(), 2)
        # Drone 2 gets normal slot duration (0.05s)
        self.assertAlmostEqual(scheduler.time_left_in_slot, 0.05)

    def test_ghost_drone_fault_tolerance(self):
        class MockObjective:
            def __init__(self, obj_id, x=0.0, y=0.0):
                self.obj_id = obj_id
                self.x = x
                self.y = y

        agent = CombatDroneAgent(agent_id=0, x=0.0, y=0.0, params={})
        
        # Let's say agent 1 is winning objective 10
        agent.local_winners[10] = 1
        agent.local_bids[10] = 99.0
        
        # Let's say agent 0 is winning objective 11
        agent.local_winners[11] = 0
        agent.local_bids[11] = 90.0
        agent.assigned_task_id = 11

        objectives = [MockObjective(10, 5.0, 5.0), MockObjective(11, 10.0, 10.0)]



        # Scenario A: Agent 1 is active (in perceived swarm)
        perceived = {
            1: {"pos": (5.0, 5.0), "timestamp": 1.0}
        }
        agent.step(dt=0.1, perceived_swarm=perceived)
        
        # Run auction; since agent 1 is active, its win should remain
        agent.run_local_auction(objectives)
        self.assertEqual(agent.local_winners[10], 1)
        self.assertEqual(agent.local_winners[11], 0)

        # Scenario B: Agent 1 goes offline (missing from perceived swarm)
        # Perceived swarm contains no info on agent 1 (timed out / shot down)
        empty_perceived = {}
        # Update perceived swarm on agent
        agent.step(dt=0.1, perceived_swarm=empty_perceived)
        
        # Run auction; agent 1's ownership should be cleared since it timed out,
        # and objective 10 should be reallocated to agent 0 (the remaining active drone).
        agent.run_local_auction(objectives)
        self.assertEqual(agent.local_winners[10], 0)  # Reallocated to agent 0
        self.assertEqual(agent.local_winners[11], -1)  # Released by agent 0

if __name__ == "__main__":
    unittest.main()
