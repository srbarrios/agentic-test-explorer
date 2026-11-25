"""Per-node timeouts, retries and failure handling for the swarm (LangGraph 1.2).

Three layers cooperate, from the inside out:

1. A node that shows no activity for ``node_idle_timeout_s`` is cancelled. Any LLM
   or tool call starting or finishing inside the node renews the timer, so long
   healthy turns are safe and only a genuinely hung browser or request is killed.
2. A node that timed out is retried ``node_retries`` times.
3. A node that still fails becomes a ``[NODE FAILED]`` message and control goes
   back to the supervisor, which can pick another approach. Transient API errors
   (429/503/529...) are NOT handled here: they are re-raised so the mission-level
   retry can wait for the quota and resume from the checkpoint.

Note: no ``from __future__ import annotations`` on purpose. LangGraph injects the
``NodeError`` into the handler by reading its type annotations.
"""

from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from langgraph.errors import NodeError, NodeTimeoutError
from langgraph.types import Command, RetryPolicy, TimeoutPolicy

from agentic_explorer.config import RuntimeConfig
from agentic_explorer.orchestration.graph_base import AgentState
from agentic_explorer.runtime.retry import is_transient_error
from agentic_explorer.utils import console
from agentic_explorer.utils.messages import clip_text

# After a failure, go through the summarizer so the history keeps being compacted.
RECOVERY_NODE = "Summarizer"
_ERROR_CLIP_CHARS = 300


def _timeout_policy(runtime: RuntimeConfig) -> Optional[TimeoutPolicy]:
    idle = runtime.node_idle_timeout_s or None
    hard = runtime.node_run_timeout_s or None
    if idle is None and hard is None:
        return None
    return TimeoutPolicy(idle_timeout=idle, run_timeout=hard)


def _is_node_timeout(exc: Exception) -> bool:
    return isinstance(exc, NodeTimeoutError)


def _retry_policy(runtime: RuntimeConfig, timeout: Optional[TimeoutPolicy]) -> Optional[RetryPolicy]:
    """Retry only hung nodes; API errors are left to the mission-level retry."""
    if timeout is None or runtime.node_retries <= 0:
        return None
    return RetryPolicy(
        max_attempts=runtime.node_retries + 1,
        initial_interval=1.0,
        backoff_factor=2.0,
        retry_on=_is_node_timeout,
    )


def agent_error_handler(state: AgentState, error: NodeError) -> Command:
    """Turn a failed agent node into a message for the supervisor.

    Transient API errors are re-raised untouched so ``with_retry`` can resume the
    mission from its checkpoint after waiting.
    """
    exc = error.error
    if not isinstance(exc, NodeTimeoutError) and is_transient_error(exc):
        raise exc

    detail = f"{type(exc).__name__}: {exc}"
    console.warn(f"{error.node} failed ({clip_text(detail, 120)}); handing control back to the supervisor")
    note = HumanMessage(
        content=f"[NODE FAILED] {error.node}: {clip_text(detail, _ERROR_CLIP_CHARS)}. "
                "Continue with a different approach or agent."
    )
    return Command(update={"messages": [note]}, goto=RECOVERY_NODE)


def supervisor_node_options(runtime: RuntimeConfig) -> Dict[str, Any]:
    """``add_node`` options for the supervisor: it can hang on its LLM call too."""
    timeout = _timeout_policy(runtime)
    return {"timeout": timeout, "retry_policy": _retry_policy(runtime, timeout)}


def agent_node_options(runtime: RuntimeConfig) -> Dict[str, Any]:
    """``add_node`` options for an agent node: timeout, retry and failure handling."""
    return {**supervisor_node_options(runtime), "error_handler": agent_error_handler}
