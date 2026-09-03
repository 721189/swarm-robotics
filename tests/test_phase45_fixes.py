"""Regression tests for the Phase-4.5 correctness fixes.

Covers:
1. TDMA scheduler consumes ALL elapsed time (multi-slot dt) -- Fatal #6.
2. Exact slot-boundary alignment so the simulated frame equals N*tau.
3. D-TDMA double-slot granting still works under the while-loop.
4. VERIFY_HOLD semantics (verified vs first-passage consensus) -- Fatal #1/#3.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.communication.tdma_scheduler import TDMAScheduler


class TestSchedulerMultiSlotConsumption(unittest.TestCase):
    def test_dt_spanning_multiple_slots_advances_multiple_speakers(self):
        """dt=100ms with tau=25ms must consume exactly 4 slots per update."""
        s = TDMAScheduler(num_drones=10, slot_duration_ms=25.0)
        # Net rotation over 5 updates: 5 x 4 slots = 20 -> speaker returns to 0.
        for _ in range(5):
            s.update(dt=0.1)
        self.assertEqual(s.current_speaker, 0)
        # Observe EVERY rotation by stepping at quarter-slot granularity:
        # 20 slots must visit all 10 drones exactly twice.
        s2 = TDMAScheduler(num_drones=10, slot_duration_ms=25.0)
        quarter = s2.slot_duration / 4.0
        seen = []
        prev = s2.get_current_speaker()
        for _ in range(20 * 4):              # 20 slots' worth of quarter-steps
            s2.update(dt=quarter)
            cur = s2.get_current_speaker()
            if cur != prev:
                seen.append(cur)
                prev = cur
        self.assertEqual(len(seen) + 1, 20 + 1)   # 20 rotations incl. initial
        self.assertEqual(set(seen + [0]), set(range(10)))

    def test_exact_slot_boundary_no_drift(self):
        """Slot boundaries stay exact: no cumulative timing drift."""
        s = TDMAScheduler(num_drones=8, slot_duration_ms=50.0)  # dt=100ms=2 slots
        t_expected = 0.0
        for i in range(100):
            s.update(dt=0.1)
            t_expected += 0.1
            self.assertAlmostEqual(s.current_time, t_expected, places=9)
            # time_left_in_slot must always be in (0, tau]
            self.assertGreater(s.time_left_in_slot, -1e-9)
            self.assertLessEqual(s.time_left_in_slot, s.slot_duration + 1e-9)

    def test_frame_period_matches_analytical_N_tau(self):
        """Each drone receives EXACTLY ONE slot transition per N*tau frame
        (N=6, tau=50ms, sub-slot stepping dt=10ms)."""
        n, tau, dt = 6, 0.05, 0.01          # dt < tau: sub-slot stepping
        s = TDMAScheduler(num_drones=n, slot_duration_ms=tau * 1000.0)
        transitions = {i: 0 for i in range(n)}
        prev = s.get_current_speaker()
        steps_per_frame = int(round(n * tau / dt))          # 30
        for _ in range(steps_per_frame * 7):                 # 7 frames
            s.update(dt=dt)
            cur = s.get_current_speaker()
            if cur != prev:
                transitions[cur] += 1
                prev = cur
        for i in range(n):
            self.assertEqual(transitions[i], 7,
                             'drone {0} got {1} slot transitions in 7 frames'
                             .format(i, transitions[i]))

    def test_double_slot_still_granted_and_cleared(self):
        s = TDMAScheduler(num_drones=4, slot_duration_ms=250.0)
        s.request_double_slot(1)
        s.update(dt=0.25)      # -> speaker 1, double slot granted
        self.assertTrue(s.double_slot_requests.get(1) is False
                        or not s.double_slot_requests.get(1))
        s.update(dt=0.5)       # consume the full double slot in one tick
        self.assertEqual(s.current_speaker, 2)


class TestConsensusSemanticsConstants(unittest.TestCase):
    def test_verify_hold_equals_one_validity_horizon(self):
        import benchmark.runners.headless_runner as hr
        self.assertEqual(hr.VERIFY_HOLD,
                         int(round(hr.TELEMETRY_MAX_AGE / hr.DT)))
        self.assertEqual(hr.VERIFY_HOLD, 20)

    def test_result_schema_contains_reliability_decomposition(self):
        cols = set(hr_cols())
        for c in ['consensus_verified', 'consensus_regressions',
                  'tx_opportunities', 'p_tx_success', 'p_rx_given_tx', 'p_e2e']:
            self.assertIn(c, cols, 'missing column: {0}'.format(c))


def hr_cols():
    import benchmark.runners.headless_runner as hr
    return hr.RESULTS_HEADERS


if __name__ == '__main__':
    unittest.main(verbosity=2)
