import unittest
from src.reliability.reliability_monitor import ReliabilityMonitor

class TestReliabilityMonitor(unittest.TestCase):

    def setUp(self):
        self.monitor = ReliabilityMonitor()

    def test_crash_isolation(self):
        result = self.monitor.crash_isolation()
        self.assertTrue(result, "Crash isolation should be successful.")

    def test_rollback_mechanism(self):
        initial_state = self.monitor.get_current_state()
        self.monitor.make_changes()
        self.monitor.rollback()
        self.assertEqual(self.monitor.get_current_state(), initial_state, "Rollback mechanism should restore the initial state.")

    def test_telemetry_logging(self):
        self.monitor.enable_telemetry()
        self.monitor.log_event("Test event")
        logs = self.monitor.get_logs()
        self.assertIn("Test event", logs, "Telemetry logs should contain the logged event.")

if __name__ == '__main__':
    unittest.main()