"""
Email Triage Environment — A real-world OpenEnv environment for AI agent training.

Simulates email triage: the agent must classify, prioritize, route, and respond
to incoming support emails. Includes 3 tasks of increasing difficulty with
deterministic grading.

Example:
    >>> from email_triage_env import EmailTriageEnv
    >>>
    >>> with EmailTriageEnv(base_url="http://localhost:8000").sync() as env:
    ...     obs = env.reset(task_id="email_classify")
    ...     tools = env.list_tools()
    ...     result = env.call_tool("classify_email",
    ...         category="technical", priority="high")
"""

# Re-export MCP types for convenience
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import EmailTriageEnv
from .models import (
    ActionType,
    Department,
    EmailCategory,
    EmailPriority,
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
)

__all__ = [
    "EmailTriageEnv",
    "CallToolAction",
    "ListToolsAction",
    "EmailTriageAction",
    "EmailTriageObservation",
    "EmailTriageState",
    "EmailCategory",
    "EmailPriority",
    "Department",
    "ActionType",
]
