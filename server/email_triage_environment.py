"""
Email Triage Environment Implementation.

An MCP environment that simulates email triage — a real-world task where an AI
agent must categorize, prioritize, route, and respond to incoming support emails.

Exposes tools via MCP:
  - classify_email(category, priority): Classify the email
  - route_email(department): Route to the correct department
  - draft_response(response_text): Draft a response to the email
  - get_email_details(): Get additional details about the current email

Supports three tasks of increasing difficulty:
  - email_classify (easy): Classify category + priority
  - email_triage (medium): Classify + route
  - email_resolve (hard): Classify + route + draft response
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP

# Import local modules — handle both in-repo and standalone
try:
    from models import (
        ActionType,
        Department,
        EmailCategory,
        EmailPriority,
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriageState,
        ScoreBreakdown,
        ThreadMessage,
    )
    from server.grader import compute_task_score, clamp_score
except ImportError:
    from email_triage_env.models import (
        ActionType,
        Department,
        EmailCategory,
        EmailPriority,
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriageState,
        ScoreBreakdown,
        ThreadMessage,
    )
    from email_triage_env.server.grader import compute_task_score, clamp_score


def _load_json(filename: str) -> Any:
    """Load a JSON file from the data/ directory."""
    # Try several possible locations
    candidates = [
        Path(__file__).parent.parent / "data" / filename,
        Path("data") / filename,
        Path(__file__).parent / ".." / "data" / filename,
    ]
    for path in candidates:
        resolved = path.resolve()
        if resolved.exists():
            with open(resolved, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Could not find {filename} in any of: {[str(p) for p in candidates]}"
    )


class EmailTriageEnvironment(MCPEnvironment):
    """
    Email Triage Environment.

    Simulates real-world email triage where an AI agent must classify,
    prioritize, route, and respond to incoming support emails.

    MCP Tools:
      - classify_email(category, priority) → Classify the email
      - route_email(department) → Route to the correct department
      - draft_response(response_text) → Draft a response
      - get_email_details() → Get more details about the email
    """

    def __init__(self):
        """Initialize the environment with MCP tools and load data."""
        # Load datasets
        self._emails = _load_json("emails.json")
        self._tasks = {t["task_id"]: t for t in _load_json("tasks.json")}

        # Build email lookup
        self._email_lookup = {e["email_id"]: e for e in self._emails}

        # Episode state
        self._current_email = None
        self._current_task = None
        self._agent_category = None
        self._agent_priority = None
        self._agent_department = None
        self._agent_response = None
        self._last_reward = clamp_score(0.0)
        self._last_feedback = ""

        # Initialize internal state
        self._state = EmailTriageState(
            episode_id=str(uuid4()),
            step_count=0,
        )

        # Create MCP server with tools
        mcp = FastMCP("email_triage_env")

        @mcp.tool
        def classify_email(category: str, priority: str) -> dict:
            """
            Classify the email into a category and assign a priority level.

            Args:
                category: One of: billing, technical, account, feature_request, spam
                priority: One of: critical, high, medium, low

            Returns:
                Dictionary with classification result and feedback
            """
            return self._handle_classify(category, priority)

        @mcp.tool
        def route_email(department: str) -> dict:
            """
            Route the email to the appropriate department.

            Args:
                department: One of: engineering, billing, account_mgmt, product, security, spam_filter

            Returns:
                Dictionary with routing result and feedback
            """
            return self._handle_route(department)

        @mcp.tool
        def draft_response(response_text: str) -> dict:
            """
            Draft a response to the email.

            Args:
                response_text: The response text to send to the email sender.
                              Should be professional and address the sender's concerns.

            Returns:
                Dictionary with response submission result and feedback
            """
            return self._handle_respond(response_text)

        @mcp.tool
        def get_email_details() -> dict:
            """
            Get additional details about the current email, including thread history.

            Returns:
                Dictionary with full email details and thread context
            """
            return self._handle_get_details()

        # Initialize base MCPEnvironment with the MCP server
        super().__init__(mcp)

    # =========================================================================
    # Core API: reset / step / state
    # =========================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        email_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            episode_id: Custom episode ID
            task_id: Task to run (email_classify, email_triage, email_resolve)
            email_id: Specific email ID to use (random if None)

        Returns:
            Initial Observation with the email to triage
        """
        # Handle task_id from kwargs (WebSocket passes data as kwargs)
        if task_id is None:
            task_id = kwargs.get("task_id", "email_classify")
        if email_id is None:
            email_id = kwargs.get("email_id", None)

        # Set seed
        if seed is not None:
            random.seed(seed)

        # Load task config
        if task_id not in self._tasks:
            task_id = "email_classify"
        self._current_task = self._tasks[task_id]

        # Select email
        if email_id and email_id in self._email_lookup:
            self._current_email = self._email_lookup[email_id]
        else:
            # Filter emails based on task difficulty requirements
            task_filter = self._current_task.get("email_filter", {})
            allowed_difficulties = task_filter.get("difficulties", ["easy", "medium", "hard"])
            eligible = [
                e for e in self._emails
                if e.get("difficulty", "easy") in allowed_difficulties
                and task_id in e.get("task_ids", [])
            ]
            if not eligible:
                eligible = self._emails
            self._current_email = random.choice(eligible)

        # Reset agent state
        self._agent_category = None
        self._agent_priority = None
        self._agent_department = None
        self._agent_response = None
        self._last_reward = clamp_score(0.0)
        self._last_feedback = ""

        # Reset episode state
        self._state = EmailTriageState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            current_email_id=self._current_email["email_id"],
            actions_taken=[],
            max_steps=self._current_task.get("max_steps", 1),
            score_breakdown=ScoreBreakdown(),
            done=False,
            cumulative_reward=clamp_score(0.0),
        )

        # Build initial observation
        return self._build_observation(
            feedback=f"New episode started. Task: {self._current_task['name']}. "
                     f"{self._current_task['description']}",
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions by parsing them as EmailTriageAction.

        For direct (non-MCP) action submission, parse the action data
        and route to the appropriate handler.
        """
        # Try to extract action data from the generic Action
        action_data = action.metadata if hasattr(action, 'metadata') else {}

        action_type = action_data.get("action_type", "")

        if action_type == "classify":
            self._handle_classify(
                action_data.get("category", ""),
                action_data.get("priority", ""),
            )
        elif action_type == "route":
            self._handle_route(action_data.get("department", ""))
        elif action_type == "respond":
            self._handle_respond(action_data.get("response_text", ""))
        elif action_type == "get_details":
            self._handle_get_details()
        else:
            self._last_feedback = (
                f"Unknown action type: '{action_type}'. "
                "Use classify_email, route_email, draft_response, or get_email_details."
            )
            self._last_reward = clamp_score(-0.10)

        return self._build_observation(feedback=self._last_feedback)

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step in the environment."""
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step for WebSocket handler."""
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """Get the current environment state."""
        # Convert our typed state to the base State
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            task_id=self._state.task_id,
            current_email_id=self._state.current_email_id,
            actions_taken=self._state.actions_taken,
            max_steps=self._state.max_steps,
            done=self._state.done,
            cumulative_reward=self._state.cumulative_reward,
            score_breakdown=self._state.score_breakdown.model_dump(),
        )

    # =========================================================================
    # Action Handlers
    # =========================================================================

    def _handle_classify(self, category: str, priority: str) -> dict:
        """Handle email classification action."""
        if self._state.done:
            self._last_feedback = "Episode is already completed."
            self._last_reward = clamp_score(0.0)
            return {"status": "error", "message": self._last_feedback}

        # Validate inputs
        valid_categories = [c.value for c in EmailCategory]
        valid_priorities = [p.value for p in EmailPriority]

        category_lower = category.lower().strip()
        priority_lower = priority.lower().strip()

        if category_lower not in valid_categories:
            self._last_feedback = (
                f"Invalid category '{category}'. "
                f"Valid categories: {valid_categories}"
            )
            self._last_reward = clamp_score(-0.10)
            return {"status": "error", "message": self._last_feedback}

        if priority_lower not in valid_priorities:
            self._last_feedback = (
                f"Invalid priority '{priority}'. "
                f"Valid priorities: {valid_priorities}"
            )
            self._last_reward = clamp_score(-0.10)
            return {"status": "error", "message": self._last_feedback}

        # Store classification
        self._agent_category = category_lower
        self._agent_priority = priority_lower
        self._state.actions_taken.append("classify")

        # Compute partial reward
        gt = self._current_email["ground_truth"]
        from server.grader import grade_category, grade_priority
        cat_score = grade_category(self._agent_category, gt["category"])
        pri_score = grade_priority(self._agent_priority, gt["priority"])

        # Scale reward based on task weights
        scoring = self._current_task.get("scoring", {})
        cat_reward = scoring.get("category_weight", 0.3) * cat_score
        pri_reward = scoring.get("priority_weight", 0.2) * pri_score
        self._state.cumulative_reward = clamp_score(self._state.cumulative_reward + self._last_reward)
        self._state.score_breakdown.category_score = cat_score
        self._state.score_breakdown.priority_score = pri_score

        self._last_feedback = (
            f"Email classified as '{category_lower}' with priority '{priority_lower}'."
        )

        # Check if task is complete
        self._check_completion()

        return {
            "status": "success",
            "category": category_lower,
            "priority": priority_lower,
            "reward": self._last_reward,
            "message": self._last_feedback,
        }

    def _handle_route(self, department: str) -> dict:
        """Handle email routing action."""
        if self._state.done:
            self._last_feedback = "Episode is already completed."
            self._last_reward = clamp_score(0.0)
            return {"status": "error", "message": self._last_feedback}

        valid_departments = [d.value for d in Department]
        department_lower = department.lower().strip()

        if department_lower not in valid_departments:
            self._last_feedback = (
                f"Invalid department '{department}'. "
                f"Valid departments: {valid_departments}"
            )
            self._last_reward = clamp_score(-0.10)
            return {"status": "error", "message": self._last_feedback}

        # Store routing
        self._agent_department = department_lower
        self._state.actions_taken.append("route")

        # Compute partial reward
        gt = self._current_email["ground_truth"]
        from server.grader import grade_department
        dept_score = grade_department(self._agent_department, gt["department"])

        scoring = self._current_task.get("scoring", {})
        dept_reward = scoring.get("department_weight", 0.2) * dept_score
        self._state.cumulative_reward = clamp_score(self._state.cumulative_reward + self._last_reward)
        self._state.score_breakdown.department_score = dept_score

        self._last_feedback = f"Email routed to '{department_lower}' department."

        self._check_completion()

        return {
            "status": "success",
            "department": department_lower,
            "reward": self._last_reward,
            "message": self._last_feedback,
        }

    def _handle_respond(self, response_text: str) -> dict:
        """Handle response drafting action."""
        if self._state.done:
            self._last_feedback = "Episode is already completed."
            self._last_reward = clamp_score(0.0)
            return {"status": "error", "message": self._last_feedback}

        # Store response
        self._agent_response = response_text
        self._state.actions_taken.append("respond")

        # Compute partial reward
        gt = self._current_email["ground_truth"]
        is_spam = gt.get("category", "").lower() == "spam"
        from server.grader import grade_response
        resp_score = grade_response(
            response_text,
            gt.get("expected_response_keywords", []),
            is_spam=is_spam,
        )

        scoring = self._current_task.get("scoring", {})
        resp_reward = scoring.get("response_weight", 0.25) * resp_score
        self._state.cumulative_reward = clamp_score(self._state.cumulative_reward + self._last_reward)
        self._state.score_breakdown.response_score = resp_score

        if is_spam:
            self._last_feedback = "Response noted. (Note: This appears to be spam — no response may be preferable.)"
        else:
            self._last_feedback = "Response drafted successfully."

        self._check_completion()

        return {
            "status": "success",
            "response_length": len(response_text),
            "reward": self._last_reward,
            "message": self._last_feedback,
        }

    def _handle_get_details(self) -> dict:
        """Handle get details action — provides additional email context."""
        if self._current_email is None:
            return {"status": "error", "message": "No email loaded. Call reset first."}

        email = self._current_email
        thread = email.get("thread_history", [])

        details = {
            "status": "success",
            "email_id": email["email_id"],
            "subject": email["subject"],
            "sender": email["sender"],
            "timestamp": email["timestamp"],
            "body": email["body"],
            "has_thread": len(thread) > 0,
            "thread_count": len(thread),
            "thread_messages": thread,
        }

        self._last_feedback = "Email details retrieved."
        self._last_reward = clamp_score(0.0)  # Informational, no reward

        return details

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _check_completion(self):
        """Check if the current task is complete."""
        required = set(self._current_task.get("required_actions", []))
        taken = set(self._state.actions_taken)

        if required.issubset(taken) or self._state.step_count >= self._state.max_steps:
            self._finalize_episode()

    def _finalize_episode(self):
        """Finalize the episode and compute final scores."""
        self._state.done = True

        # Compute efficiency score
        from server.grader import grade_efficiency
        required_actions = self._current_task.get("required_actions", [])
        min_steps = len(required_actions)
        eff_score = grade_efficiency(
            self._state.step_count,
            self._state.max_steps,
            min_steps,
        )
        self._state.score_breakdown.efficiency_score = eff_score

        # Add efficiency reward
        scoring = self._current_task.get("scoring", {})
        eff_reward = scoring.get("efficiency_weight", 0.0) * eff_score
        self._state.cumulative_reward += eff_reward

        # Compute final total score from breakdown
        gt = self._current_email["ground_truth"]
        final_score = compute_task_score(
            task_id=self._state.task_id,
            task_config=self._current_task,
            ground_truth=gt,
            agent_category=self._agent_category,
            agent_priority=self._agent_priority,
            agent_department=self._agent_department,
            agent_response=self._agent_response,
            steps_taken=self._state.step_count,
        )
        self._state.score_breakdown = final_score
        self._state.cumulative_reward = clamp_score(final_score.total_score)

        self._last_feedback = (
            f"Episode complete! Final score: {final_score.total_score:g}/1. "
            f"Category: {final_score.category_score:g}, "
            f"Priority: {final_score.priority_score:g}, "
            f"Department: {final_score.department_score:g}, "
            f"Response: {final_score.response_score:g}, "
            f"Efficiency: {final_score.efficiency_score:g}"
        )

    def _build_observation(self, feedback: str = "") -> Observation:
        """Build an Observation from the current state."""
        if self._current_email is None:
            return Observation(
                done=False,
                reward=clamp_score(0.0),
                metadata={"error": "No email loaded. Call reset first."},
            )

        email = self._current_email
        task = self._current_task or self._tasks["email_classify"]
        required = task.get("required_actions", [])
        taken = self._state.actions_taken
        remaining = [a for a in required if a not in taken]

        thread_history = [
            {"sender": t.get("from", t.get("sender", "")), "date": t["date"], "snippet": t["snippet"]}
            for t in email.get("thread_history", [])
        ]

        obs_data = {
            "email_id": email["email_id"],
            "email_subject": email["subject"],
            "email_body": email["body"],
            "sender": email["sender"],
            "timestamp": email["timestamp"],
            "thread_history": thread_history,
            "current_classification": self._agent_category,
            "current_priority": self._agent_priority,
            "current_department": self._agent_department,
            "current_response": self._agent_response,
            "feedback": feedback or self._last_feedback,
            "task_id": self._state.task_id,
            "task_description": task.get("description", ""),
            "required_actions": remaining,
            "steps_remaining": max(0, self._state.max_steps - self._state.step_count),
        }

        return Observation(
            done=self._state.done,
            reward=clamp_score(self._last_reward if not self._state.done else self._state.cumulative_reward),
            metadata=obs_data,
        )
