"""The shipped ``missions/*.yaml`` files must declare the graph they run on.

Each file is named after a persona; every mission in it must say ``graph:``
explicitly and agree with that persona's tier and with the legacy keyword
routing, so adding the field changes no behaviour.
"""

from pathlib import Path

import pytest
import yaml

from agentic_explorer.orchestration.personas import PERSONAS, tier_for_thread_id

MISSIONS_DIR = Path(__file__).parent.parent / "missions"
FILES = sorted(MISSIONS_DIR.glob("*_agent.yaml"))


def test_one_mission_file_per_persona():
    assert {f.stem for f in FILES} == set(PERSONAS)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_every_mission_declares_the_graph_of_its_persona(path):
    missions = yaml.safe_load(path.read_text(encoding="utf-8"))["missions"]
    expected = PERSONAS[path.stem].tier

    assert missions
    for mission in missions:
        assert mission.get("graph") == expected, mission["thread_id"]
        assert tier_for_thread_id(mission["thread_id"]) == expected, mission["thread_id"]
