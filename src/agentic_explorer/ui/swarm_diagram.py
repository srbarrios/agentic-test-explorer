"""Dynamic Mermaid diagram generator for the LangGraph swarm topology."""

from __future__ import annotations

STANDARD_AGENTS = ("new_user_agent", "power_user_agent", "adversarial_user_agent")
ADVANCED_AGENTS = (
    "accessibility_user_agent",
    "data_heavy_user_agent",
    "impatient_user_agent",
    "returning_user_agent",
    "explorer_agent",
)

_LABELS = {
    "Supervisor": "Supervisor",
    "Summarizer": "Summarizer",
    "new_user_agent": "New User",
    "power_user_agent": "Power User",
    "adversarial_user_agent": "Adversarial",
    "accessibility_user_agent": "Accessibility",
    "data_heavy_user_agent": "Data Heavy",
    "impatient_user_agent": "Impatient",
    "returning_user_agent": "Returning",
    "explorer_agent": "Explorer",
    "FINISH": "FINISH",
}


def generate_swarm_diagram(active_node: str, graph_type: str = "standard") -> str:
    """Return a Mermaid stateDiagram-v2 string with *active_node* highlighted."""
    agents = ADVANCED_AGENTS if graph_type.lower() == "advanced" else STANDARD_AGENTS

    lines = ["stateDiagram-v2"]

    # Aliases for readable labels
    for node_id in ("Supervisor", "Summarizer", *agents, "FINISH"):
        label = _LABELS.get(node_id, node_id)
        if label != node_id:
            lines.append(f"    {node_id} : {label}")

    lines.append("")

    # Edges
    lines.append("    [*] --> Supervisor")
    for agent in agents:
        lines.append(f"    Supervisor --> {agent}")
    lines.append("    Supervisor --> FINISH")
    lines.append("")
    for agent in agents:
        lines.append(f"    {agent} --> Summarizer")
    lines.append("    Summarizer --> Supervisor")
    lines.append("    FINISH --> [*]")

    # Highlight active node
    lines.append("")
    lines.append("    classDef active fill:#10b981,stroke:#047857,stroke-width:4px,color:#000")
    if active_node and active_node in ("Supervisor", "Summarizer", *agents, "FINISH"):
        lines.append(f"    class {active_node} active")

    return "\n".join(lines)
