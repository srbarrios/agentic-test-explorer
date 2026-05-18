import json
import os
import sys
import tempfile
import unittest

# Dashboard tests require streamlit optional dependency
try:
    from agentic_explorer.ui.dashboard import _load_state, STATE_FILE
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    STATE_FILE = ".agent_state.json"

    def _load_state():
        return {}


@unittest.skipIf(not STREAMLIT_AVAILABLE, "Streamlit not installed (optional dependency)")
class DashboardTests(unittest.TestCase):
    def setUp(self):
        self._original_cwd = os.getcwd()
        self._temp_dir = tempfile.mkdtemp()
        os.chdir(self._temp_dir)

    def tearDown(self):
        os.chdir(self._original_cwd)

    def test_load_state_returns_empty_dict_when_file_missing(self):
        state = _load_state()
        self.assertEqual(state, {})

    def test_load_state_returns_empty_dict_when_json_corrupt(self):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            f.write("not valid json {")

        state = _load_state()
        self.assertEqual(state, {})

    def test_load_state_returns_valid_dict_from_wellformed_json(self):
        data = {
            "mission_id": "test_123",
            "step_count": 5,
            "active_node": "Supervisor",
            "bugs_count": 2,
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)

        state = _load_state()
        self.assertEqual(state["mission_id"], "test_123")
        self.assertEqual(state["step_count"], 5)
        self.assertEqual(state["active_node"], "Supervisor")
        self.assertEqual(state["bugs_count"], 2)

    def test_load_state_handles_empty_json_file(self):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            f.write("")

        state = _load_state()
        self.assertEqual(state, {})

    def test_load_state_handles_completed_state(self):
        data = {
            "mission_id": "final_mission",
            "step_count": 50,
            "bugs_count": 3,
            "completed": True,
            "total_missions": 10,
            "active_node": "completed",
            "app_url": "https://app.example",
            "provider": "claude",
            "model_name": "claude-sonnet-4.5",
            "bugs_found": ["Bug 1", "Bug 2", "Bug 3"],
            "explored_paths": ["/home", "/search", "/profile"]
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)

        state = _load_state()
        self.assertTrue(state.get("completed", False))
        self.assertEqual(state.get("total_missions"), 10)
        self.assertEqual(state.get("active_node"), "completed")
        self.assertEqual(state.get("bugs_count"), 3)
        self.assertEqual(len(state.get("bugs_found", [])), 3)


if __name__ == "__main__":
    unittest.main()
