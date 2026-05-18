import unittest

from agentic_explorer.ui.swarm_diagram import generate_swarm_diagram


class SwarmDiagramTests(unittest.TestCase):
    def test_generates_mermaid_statediagram_v2(self):
        diagram = generate_swarm_diagram("Supervisor", "standard")
        self.assertTrue(diagram.startswith("stateDiagram-v2"))

    def test_standard_graph_includes_three_agents(self):
        diagram = generate_swarm_diagram("", "standard")
        self.assertIn("new_user_agent", diagram)
        self.assertIn("power_user_agent", diagram)
        self.assertIn("adversarial_user_agent", diagram)
        self.assertNotIn("accessibility_user_agent", diagram)
        self.assertNotIn("explorer_agent", diagram)

    def test_advanced_graph_includes_five_agents(self):
        diagram = generate_swarm_diagram("", "advanced")
        self.assertIn("accessibility_user_agent", diagram)
        self.assertIn("data_heavy_user_agent", diagram)
        self.assertIn("impatient_user_agent", diagram)
        self.assertIn("returning_user_agent", diagram)
        self.assertIn("explorer_agent", diagram)
        self.assertNotIn("new_user_agent", diagram)

    def test_includes_supervisor_and_summarizer(self):
        for graph_type in ("standard", "advanced"):
            with self.subTest(graph_type=graph_type):
                diagram = generate_swarm_diagram("", graph_type)
                self.assertIn("Supervisor", diagram)
                self.assertIn("Summarizer", diagram)

    def test_includes_finish_node(self):
        diagram = generate_swarm_diagram("", "standard")
        self.assertIn("FINISH", diagram)
        self.assertIn("Supervisor --> FINISH", diagram)

    def test_highlights_active_node_with_classdef(self):
        for active in ("Supervisor", "new_user_agent", "Summarizer"):
            with self.subTest(active=active):
                diagram = generate_swarm_diagram(active, "standard")
                self.assertIn("classDef active", diagram)
                self.assertIn(f"class {active} active", diagram)

    def test_no_highlight_when_active_node_empty(self):
        diagram = generate_swarm_diagram("", "standard")
        self.assertIn("classDef active", diagram)
        self.assertNotIn("class  active", diagram)

    def test_no_highlight_when_active_node_unknown(self):
        diagram = generate_swarm_diagram("UnknownNode", "standard")
        self.assertIn("classDef active", diagram)
        self.assertNotIn("class UnknownNode active", diagram)

    def test_graph_type_case_insensitive(self):
        upper = generate_swarm_diagram("", "ADVANCED")
        lower = generate_swarm_diagram("", "advanced")
        self.assertIn("explorer_agent", upper)
        self.assertIn("explorer_agent", lower)


if __name__ == "__main__":
    unittest.main()
