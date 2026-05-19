import json
import os
import tempfile
import unittest

from agentic_explorer.ui import state_emitter


class StateEmitterTests(unittest.TestCase):
    def setUp(self):
        self._original_cwd = os.getcwd()
        self._temp_dir = tempfile.mkdtemp()
        os.chdir(self._temp_dir)
        # Reset module state between tests
        state_emitter._enabled = False
        state_emitter._state = state_emitter.VisualState()

    def tearDown(self):
        os.chdir(self._original_cwd)
        state_emitter.cleanup()

    def test_is_enabled_returns_false_by_default(self):
        self.assertFalse(state_emitter.is_enabled())

    def test_enable_sets_enabled_flag(self):
        state_emitter.enable()
        self.assertTrue(state_emitter.is_enabled())

    def test_update_is_noop_when_disabled(self):
        state_emitter.update(mission_id="test", step_count=5)
        self.assertEqual(state_emitter._state.mission_id, "")
        self.assertEqual(state_emitter._state.step_count, 0)

    def test_update_merges_fields_when_enabled(self):
        state_emitter.enable()
        state_emitter.update(
            mission_id="test_mission",
            step_count=10,
            bugs_count=3,
            app_url="https://app.example",
        )

        self.assertEqual(state_emitter._state.mission_id, "test_mission")
        self.assertEqual(state_emitter._state.step_count, 10)
        self.assertEqual(state_emitter._state.bugs_count, 3)
        self.assertEqual(state_emitter._state.app_url, "https://app.example")

    def test_update_ignores_unknown_fields(self):
        state_emitter.enable()
        state_emitter.update(mission_id="valid", unknown_field="ignored")
        self.assertEqual(state_emitter._state.mission_id, "valid")

    def test_emit_is_noop_when_disabled(self):
        state_emitter.update(mission_id="test")
        state_emitter.emit()
        self.assertFalse(os.path.exists(state_emitter.STATE_FILE))

    def test_emit_writes_atomic_json_when_enabled(self):
        state_emitter.enable()
        state_emitter.update(
            mission_id="atomic_test",
            step_count=7,
            active_node="Supervisor",
            explored_paths=["/home", "/search"],
        )
        state_emitter.emit()

        self.assertTrue(os.path.exists(state_emitter.STATE_FILE))
        with open(state_emitter.STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["mission_id"], "atomic_test")
        self.assertEqual(data["step_count"], 7)
        self.assertEqual(data["active_node"], "Supervisor")
        self.assertEqual(data["explored_paths"], ["/home", "/search"])
        self.assertIn("timestamp", data)
        self.assertGreater(data["timestamp"], 0)

    def test_emit_overwrites_previous_state(self):
        state_emitter.enable()
        state_emitter.update(step_count=1)
        state_emitter.emit()

        state_emitter.update(step_count=5)
        state_emitter.emit()

        with open(state_emitter.STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["step_count"], 5)

    def test_cleanup_removes_temp_files(self):
        state_emitter.enable()
        state_emitter.emit()

        # Create temp file to simulate screenshot
        with open(state_emitter.SCREENSHOT_FILE, "wb") as f:
            f.write(b"fake-screenshot")

        # Verify files exist
        self.assertTrue(os.path.exists(state_emitter.STATE_FILE))
        self.assertTrue(os.path.exists(state_emitter.SCREENSHOT_FILE))

        # Cleanup
        state_emitter.cleanup()

        # Verify removal
        self.assertFalse(os.path.exists(state_emitter.STATE_FILE))
        self.assertFalse(os.path.exists(state_emitter.SCREENSHOT_FILE))

    def test_cleanup_does_not_error_when_files_missing(self):
        state_emitter.cleanup()  # Should not raise

    def test_mark_completed_sets_completion_flags(self):
        state_emitter.enable()
        state_emitter.update(mission_id="test", step_count=10, bugs_count=2)
        state_emitter.mark_completed(total_missions=5)

        self.assertTrue(state_emitter._state.completed)
        self.assertEqual(state_emitter._state.total_missions, 5)
        self.assertEqual(state_emitter._state.active_node, "completed")

        # Verify it was emitted to disk
        self.assertTrue(os.path.exists(state_emitter.STATE_FILE))
        with open(state_emitter.STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(data["completed"])
        self.assertEqual(data["total_missions"], 5)
        self.assertEqual(data["active_node"], "completed")

    def test_append_thought_accumulates_entries(self):
        state_emitter.enable()
        state_emitter.append_thought("explorer_agent", "Found login page")
        state_emitter.append_thought("validator_agent", "Checking inputs")

        stream = state_emitter._state.thought_stream
        self.assertEqual(len(stream), 2)
        self.assertEqual(stream[0]["node"], "explorer_agent")
        self.assertEqual(stream[0]["text"], "Found login page")
        self.assertEqual(stream[1]["node"], "validator_agent")
        self.assertIn("ts", stream[0])

    def test_append_thought_also_sets_last_thought(self):
        state_emitter.enable()
        state_emitter.append_thought("explorer_agent", "hello")
        self.assertEqual(state_emitter._state.last_thought, "hello")

    def test_append_thought_is_noop_when_disabled(self):
        state_emitter.append_thought("agent", "should not appear")
        self.assertEqual(len(state_emitter._state.thought_stream), 0)

    def test_append_thought_caps_at_max(self):
        state_emitter.enable()
        for i in range(250):
            state_emitter.append_thought("agent", f"thought {i}")
        self.assertEqual(len(state_emitter._state.thought_stream), state_emitter.MAX_THOUGHT_STREAM)
        self.assertEqual(state_emitter._state.thought_stream[0]["text"], "thought 50")

    def test_append_thought_skips_blank_text(self):
        state_emitter.enable()
        state_emitter.append_thought("agent", "   ")
        self.assertEqual(len(state_emitter._state.thought_stream), 0)

    def test_mark_completed_is_noop_when_disabled(self):
        state_emitter.mark_completed(total_missions=3)
        self.assertFalse(state_emitter._state.completed)
        self.assertFalse(os.path.exists(state_emitter.STATE_FILE))

    def test_update_sets_mission_description(self):
        state_emitter.enable()
        state_emitter.update(mission_description="Explore the login flow")
        self.assertEqual(state_emitter._state.mission_description, "Explore the login flow")

    def test_start_mission_appends_running_entry(self):
        state_emitter.enable()
        state_emitter.start_mission("mission_01", "Test the home page", "STANDARD")

        history = state_emitter._state.mission_history
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["id"], "mission_01")
        self.assertEqual(history[0]["description"], "Test the home page")
        self.assertEqual(history[0]["type"], "STANDARD")
        self.assertEqual(history[0]["status"], "running")
        self.assertIsNotNone(history[0]["start_time"])
        self.assertIsNone(history[0]["end_time"])

    def test_start_mission_is_noop_when_disabled(self):
        state_emitter.start_mission("m1", "desc", "STANDARD")
        self.assertEqual(len(state_emitter._state.mission_history), 0)

    def test_end_mission_marks_running_as_completed(self):
        state_emitter.enable()
        state_emitter.start_mission("mission_01", "Test login", "STANDARD")
        state_emitter.update(step_count=15, bugs_count=2)
        state_emitter.end_mission("mission_01")

        entry = state_emitter._state.mission_history[0]
        self.assertEqual(entry["status"], "completed")
        self.assertIsNotNone(entry["end_time"])
        self.assertEqual(entry["bugs_count"], 2)
        self.assertEqual(entry["steps"], 15)

    def test_end_mission_is_noop_when_disabled(self):
        state_emitter.enable()
        state_emitter.start_mission("m1", "desc", "STANDARD")
        state_emitter._enabled = False
        state_emitter.end_mission("m1")
        self.assertEqual(state_emitter._state.mission_history[0]["status"], "running")

    def test_end_mission_ignores_unknown_id(self):
        state_emitter.enable()
        state_emitter.start_mission("m1", "desc", "STANDARD")
        state_emitter.end_mission("nonexistent")
        self.assertEqual(state_emitter._state.mission_history[0]["status"], "running")

    def test_multiple_missions_tracked_independently(self):
        state_emitter.enable()
        state_emitter.start_mission("m1", "First mission", "STANDARD")
        state_emitter.update(step_count=5, bugs_count=1)
        state_emitter.end_mission("m1")

        state_emitter.start_mission("m2", "Second mission", "ADVANCED")

        history = state_emitter._state.mission_history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["status"], "completed")
        self.assertEqual(history[0]["bugs_count"], 1)
        self.assertEqual(history[1]["status"], "running")
        self.assertEqual(history[1]["id"], "m2")

    def test_mission_history_emitted_to_json(self):
        state_emitter.enable()
        state_emitter.start_mission("m1", "Test desc", "STANDARD")
        state_emitter.emit()

        with open(state_emitter.STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data["mission_history"]), 1)
        self.assertEqual(data["mission_history"][0]["id"], "m1")
        self.assertEqual(data["mission_history"][0]["description"], "Test desc")
        self.assertEqual(data["mission_description"], "")


if __name__ == "__main__":
    unittest.main()
