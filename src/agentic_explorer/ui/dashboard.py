"""Streamlit dashboard for Agentic Explorer Visual Mode.

Runs as a separate process, polling ``.agent_state.json`` and
``.latest_vision.jpg`` written by the main swarm process.  Never sends
signals back — purely read-only.

Launch: ``streamlit run src/agentic_explorer/ui/dashboard.py``
(Normally spawned automatically by ``--visual`` in main.py.)
"""

from __future__ import annotations

import json
import os
import time

import streamlit as st
import streamlit.components.v1 as components

from agentic_explorer.ui.swarm_diagram import generate_swarm_diagram

STATE_FILE = ".agent_state.json"
SCREENSHOT_FILE = ".latest_vision.jpg"
POLL_INTERVAL = 1.0


st.set_page_config(
    layout="wide",
    page_title="Agentic Explorer - Visual Mode",
    page_icon="🔬",
    initial_sidebar_state="auto",
)

# Custom CSS: compact layout to maximise content area
st.markdown("""
<style>
    /* Shrink main container padding */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }

    /* Reduce vertical gap between every Streamlit element */
    [data-testid="stVerticalBlock"] > div {
        gap: 0.25rem !important;
    }

    /* Tighten column gaps */
    [data-testid="stHorizontalBlock"] {
        gap: 0.3rem !important;
    }

    /* Sidebar — expanded */
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 190px;
        max-width: 190px;
        padding-top: 0.5rem;
    }
    /* Sidebar — collapsed: shrink to zero so panels reclaim the space */
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0px !important;
        max-width: 0px !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    [data-testid="stSidebar"] p {
        font-size: 0.78rem;
        margin-bottom: 0.15rem;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.1rem;
    }
    [data-testid="stSidebar"] hr {
        margin-top: 0.3rem;
        margin-bottom: 0.3rem;
    }

    /* Section headings — ensure enough line-height so text is not clipped */
    h3 {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0 !important;
        margin-bottom: 0.2rem;
        line-height: 1.4 !important;
        overflow: visible !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.72rem;
        padding: 0.2rem 0.4rem;
    }

    /* Reduce image caption spacing */
    [data-testid="stImage"] {
        margin-bottom: 0 !important;
    }

    /* Reduce top header bar height */
    header[data-testid="stHeader"] {
        height: 2rem !important;
    }

    /* Compact metrics */
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
    }
</style>
""", unsafe_allow_html=True)


def _load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _main() -> None:
    state = _load_state()

    if not state:
        st.title("🔬 Agentic Explorer — Visual Mode")
        st.info("Waiting for the agent swarm to start… (polling every 1 s)")
        time.sleep(POLL_INTERVAL)
        st.rerun()
        return

    # Check if missions have completed
    if state.get("completed", False):
        st.title("🎉 Mission Complete!")
        st.success("All test missions have been successfully completed.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Missions", state.get("total_missions", 0))
        col2.metric("Final Step Count", state.get("step_count", 0))
        col3.metric("Bugs Found", state.get("bugs_count", 0))
        col4.metric("Paths Explored", len(state.get("explored_paths", [])))

        st.divider()
        st.markdown("### 📊 Session Summary")
        st.markdown(f"**App URL:** `{state.get('app_url', 'N/A')}`")
        st.markdown(f"**Provider:** `{state.get('provider', 'N/A')}` · `{state.get('model_name', '')}`")

        if state.get("bugs_found"):
            st.divider()
            st.markdown("### 🐛 Bugs Discovered")
            for i, bug in enumerate(state.get("bugs_found", []), 1):
                with st.expander(f"Bug #{i}"):
                    st.markdown(bug)

        st.divider()
        st.info("You can now close this window. Check the `report_*` directories for detailed test reports.")
        return

    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Mission Control")
        st.markdown(f"**App URL:**  \n{state.get('app_url', 'N/A')}")
        st.markdown(f"**Mission:**  \n`{state.get('mission_id', 'N/A')}`")
        st.markdown(f"**Graph:** `{state.get('mission_type', 'N/A')}`")
        st.markdown(
            f"**Provider:** `{state.get('provider', 'N/A')}` · `{state.get('model_name', '')}`"
        )
        st.divider()

        col1, col2, col3 = st.columns(3)
        col1.metric("Step", state.get("step_count", 0))
        col2.metric("Bugs", state.get("bugs_count", 0))
        col3.metric("Paths", len(state.get("explored_paths", [])))

        st.divider()
        active = state.get("active_node", "")
        st.markdown(f"**Active Node:**  \n`{active or 'N/A'}`")

        age = time.time() - state.get("timestamp", 0)
        if age > 30:
            st.warning(f"Stale ({age:.0f}s)")
        else:
            st.caption(f"Updated: {age:.0f}s ago")

    # ── Main layout ─────────────────────────────────────────
    # Three-column layout: Screenshot | Mermaid Diagram | Info Tabs
    col_screenshot, col_diagram, col_info = st.columns([1.4, 0.8, 1.1], gap="small")

    # Column 1: Browser Screenshot
    with col_screenshot:
        st.markdown("### Live Browser Vision")
        if os.path.exists(SCREENSHOT_FILE):
            mtime = os.path.getmtime(SCREENSHOT_FILE)
            st.image(
                SCREENSHOT_FILE,
                use_container_width=True,
                caption=f"Viewport · updated {time.time() - mtime:.0f} s ago",
            )
        else:
            st.info("No screenshot captured yet.")

    # Column 2: Swarm Diagram
    with col_diagram:
        st.markdown("### Swarm State")
        mermaid_code = generate_swarm_diagram(
            state.get("active_node", ""),
            state.get("graph_type", "standard"),
        )
        # Render mermaid diagram using HTML component
        mermaid_html = f"""
        <div style="width: 100%; height: 480px; overflow: auto; margin: 0; padding: 0;">
            <pre class="mermaid">
{mermaid_code}
            </pre>
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
        </script>
        """
        components.html(mermaid_html, height=480, scrolling=True)

    # Column 3: Information Tabs
    with col_info:
        thought_tab, tape_tab, bugs_tab, paths_tab = st.tabs(
            ["Thought Stream", "Action Tape", "Bugs", "Paths"],
        )

        with thought_tab:
            stream = state.get("thought_stream", [])
            if stream:
                for entry in stream:
                    node = entry.get("node", "?")
                    text = entry.get("text", "")
                    st.markdown(f"**{node}** {text}")
            else:
                st.info("No thoughts yet.")

        with tape_tab:
            tape = state.get("action_tape_recent", [])
            if tape:
                for entry in reversed(tape[-20:]):
                    ok = "✅" if entry.get("ok") else "❌"
                    action = entry.get("action", "?")
                    dur = entry.get("duration_ms", 0)
                    url = entry.get("page_url", "")
                    st.text(f"{ok} {action} ({dur} ms) → {url}")
            else:
                st.info("No actions recorded yet.")

        with bugs_tab:
            bugs = state.get("bugs_found", [])
            if bugs:
                st.markdown(f"**Total Bugs Found:** {len(bugs)}")
                st.divider()
                for i, bug in enumerate(bugs, 1):
                    with st.expander(f"Bug #{i}", expanded=(i == len(bugs))):
                        st.markdown(bug)
            else:
                st.info("No bugs found yet.")

        with paths_tab:
            paths = state.get("explored_paths", [])
            if paths:
                for p in paths:
                    st.text(f"  · {p}")
            else:
                st.info("No paths explored yet.")

    time.sleep(POLL_INTERVAL)
    st.rerun()


_main()
