"""
Pydantic Models for the Email Triage Environment.

Defines typed Action, Observation, and State models used across
the environment server, client, and grader.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Enums
# =============================================================================

class EmailCategory(str, Enum):
    """Valid email categories."""
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"
    SPAM = "spam"


class EmailPriority(str, Enum):
    """Valid email priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Department(str, Enum):
    """Valid routing departments."""
    ENGINEERING = "engineering"
    BILLING = "billing"
    ACCOUNT_MGMT = "account_mgmt"
    PRODUCT = "product"
    SECURITY = "security"
    SPAM_FILTER = "spam_filter"


class ActionType(str, Enum):
    """Types of actions the agent can take."""
    CLASSIFY = "classify"
    ROUTE = "route"
    RESPOND = "respond"
    GET_DETAILS = "get_details"


# =============================================================================
# Thread Message (for email threads)
# =============================================================================

class ThreadMessage(BaseModel):
    """A single message in an email thread."""
    model_config = ConfigDict(extra="forbid")

    sender: str = Field(description="Email sender address")
    date: str = Field(description="ISO 8601 timestamp")
    snippet: str = Field(description="Brief content of the message")


# =============================================================================
# Action Model
# =============================================================================

class EmailTriageAction(BaseModel):
    """Action the agent sends to the environment.

    The agent must specify an action_type and the relevant fields for that action:
    - classify: requires category and priority
    - route: requires department
    - respond: requires response_text
    - get_details: no additional fields required
    """
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    action_type: ActionType = Field(
        description="Type of action: classify, route, respond, or get_details"
    )
    category: Optional[EmailCategory] = Field(
        default=None,
        description="Email category (required for 'classify' action)"
    )
    priority: Optional[EmailPriority] = Field(
        default=None,
        description="Email priority (required for 'classify' action)"
    )
    department: Optional[Department] = Field(
        default=None,
        description="Target department (required for 'route' action)"
    )
    response_text: Optional[str] = Field(
        default=None,
        description="Draft response text (required for 'respond' action)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# =============================================================================
# Observation Model
# =============================================================================

class EmailTriageObservation(BaseModel):
    """Observation returned by the environment after each step/reset.

    Contains the email content, current classification state, and feedback.
    """
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Email content
    email_id: str = Field(description="Unique email identifier")
    email_subject: str = Field(description="Email subject line")
    email_body: str = Field(description="Full email body text")
    sender: str = Field(description="Sender email address")
    timestamp: str = Field(description="Email timestamp (ISO 8601)")
    thread_history: List[ThreadMessage] = Field(
        default_factory=list,
        description="Previous messages in the thread"
    )

    # Current state of triage (what has been done so far)
    current_classification: Optional[str] = Field(
        default=None,
        description="Current assigned category (if classified)"
    )
    current_priority: Optional[str] = Field(
        default=None,
        description="Current assigned priority (if set)"
    )
    current_department: Optional[str] = Field(
        default=None,
        description="Currently routed department (if routed)"
    )
    current_response: Optional[str] = Field(
        default=None,
        description="Current draft response (if drafted)"
    )

    # Feedback
    feedback: str = Field(
        default="",
        description="Feedback message from the environment"
    )

    # Task info
    task_id: str = Field(description="Current task identifier")
    task_description: str = Field(
        default="",
        description="Description of what the agent should do"
    )
    required_actions: List[str] = Field(
        default_factory=list,
        description="Actions the agent still needs to take"
    )
    steps_remaining: int = Field(
        default=0,
        description="Number of steps remaining"
    )

    # Episode status
    done: bool = Field(
        default=False,
        description="Whether the episode has terminated"
    )
    reward: float = Field(
        default=0.0,
        description="Reward signal from the last action"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# =============================================================================
# State Model
# =============================================================================

class ScoreBreakdown(BaseModel):
    """Detailed breakdown of the agent's score."""
    model_config = ConfigDict(extra="allow")

    category_score: float = Field(default=0.0, description="Score for category classification")
    priority_score: float = Field(default=0.0, description="Score for priority assignment")
    department_score: float = Field(default=0.0, description="Score for department routing")
    response_score: float = Field(default=0.0, description="Score for response quality")
    efficiency_score: float = Field(default=0.0, description="Score for step efficiency")
    total_score: float = Field(default=0.0, description="Weighted total score (0.0-1.0)")


class EmailTriageState(BaseModel):
    """Internal environment state.

    Tracks the episode progress, scoring, and metadata.
    """
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
    )

    episode_id: Optional[str] = Field(
        default=None,
        description="Unique episode identifier"
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps taken"
    )
    task_id: str = Field(
        default="email_classify",
        description="Current task identifier"
    )
    current_email_id: Optional[str] = Field(
        default=None,
        description="ID of the current email being triaged"
    )
    actions_taken: List[str] = Field(
        default_factory=list,
        description="List of action types taken so far"
    )
    max_steps: int = Field(
        default=1,
        description="Maximum steps allowed for this task"
    )
    score_breakdown: ScoreBreakdown = Field(
        default_factory=ScoreBreakdown,
        description="Detailed score breakdown"
    )
    done: bool = Field(
        default=False,
        description="Whether the episode is complete"
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Total reward accumulated"
    )
