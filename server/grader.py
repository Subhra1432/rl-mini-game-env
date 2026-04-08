"""
Grading and scoring utilities for the Email Triage Environment.

Implements deterministic grading logic for all three tasks:
- email_classify (easy): category + priority accuracy
- email_triage (medium): category + priority + department routing
- email_resolve (hard): full triage + response quality scoring

All scores are clamped to the open interval (0, 1) — the platform
rejects exact 0.0 and 1.0 values.
"""

import re
from typing import Any, Dict, List, Optional, Union

from models import (
    ActionType,
    Department,
    EmailCategory,
    EmailPriority,
    ScoreBreakdown,
)

# Small epsilon to keep scores strictly inside (0, 1)
_EPS = 0.01


def clamp_score(score: float) -> Union[int, float]:
    """Clamp a score to the open interval (0, 1).

    The OpenEnv platform requires every task score to be strictly
    between 0 and 1 (not 0.0 and not 1.0).  This helper enforces
    that constraint.

    Returns int when the result is a whole number (no .0),
    otherwise returns float.
    """
    if score <= 0.0:
        clamped = _EPS          # 0.0 -> 0.01
    elif score >= 1.0:
        clamped = 1.0 - _EPS    # 1.0 -> 0.99
    else:
        clamped = score
    # Return int if whole number, else float
    return int(clamped) if clamped == int(clamped) else clamped


def format_score(score: float) -> Union[int, float]:
    """Return an int when the score is a whole number, else a float.

    Examples: 1.0 -> 1, 0.0 -> 0, 0.5 -> 0.5, 0.95 -> 0.95
    """
    return int(score) if score == int(score) else score


def grade_category(predicted: Optional[str], ground_truth: str) -> float:
    """Grade category classification. Returns ~1.0 for exact match, ~0.0 otherwise."""
    if predicted is None:
        return clamp_score(0.0)
    return clamp_score(1.0 if predicted.lower() == ground_truth.lower() else 0.0)


def grade_priority(predicted: Optional[str], ground_truth: str) -> float:
    """Grade priority assignment.

    Returns:
        1.0 for exact match
        0.5 for one level off (e.g., high vs critical)
        0.0 for two or more levels off
    """
    if predicted is None:
        return clamp_score(0.0)

    priority_order = ["low", "medium", "high", "critical"]

    try:
        pred_idx = priority_order.index(predicted.lower())
        truth_idx = priority_order.index(ground_truth.lower())
    except ValueError:
        return clamp_score(0.0)

    diff = abs(pred_idx - truth_idx)
    if diff == 0:
        return clamp_score(1.0)
    elif diff == 1:
        return clamp_score(0.5)
    else:
        return clamp_score(0.0)


def grade_department(predicted: Optional[str], ground_truth: str) -> float:
    """Grade department routing. Returns ~1.0 for exact match, ~0.0 otherwise."""
    if predicted is None:
        return clamp_score(0.0)
    return clamp_score(1.0 if predicted.lower() == ground_truth.lower() else 0.0)


def grade_response(
    response_text: Optional[str],
    expected_keywords: List[str],
    is_spam: bool = False,
) -> float:
    """Grade response quality based on keyword coverage and basic quality checks.

    Scoring:
        - Keyword coverage: up to 0.6 (fraction of expected keywords found)
        - Professional tone: 0.15 (no ALL CAPS, no excessive punctuation)
        - Appropriate length: 0.15 (between 50-500 chars for non-spam)
        - Non-empty for non-spam: 0.10

    For spam emails, no response is expected. Returning empty gets 1.0.
    """
    if is_spam:
        # For spam, the best action is to NOT respond
        if response_text is None or response_text.strip() == "":
            return clamp_score(1.0)
        else:
            return clamp_score(0.2)  # Partial credit for identifying but still responding

    if response_text is None or response_text.strip() == "":
        return clamp_score(0.0)

    score = 0.0
    text_lower = response_text.lower()

    # ----- Keyword coverage (0.6) -----
    if expected_keywords:
        matches = sum(
            1 for kw in expected_keywords
            if kw.lower() in text_lower
        )
        keyword_score = matches / len(expected_keywords)
        score += 0.6 * keyword_score

    # ----- Professional tone (0.15) -----
    # Penalize excessive caps or punctuation
    caps_ratio = sum(1 for c in response_text if c.isupper()) / max(len(response_text), 1)
    excessive_punct = len(re.findall(r'[!?]{2,}', response_text))
    if caps_ratio < 0.3 and excessive_punct == 0:
        score += 0.15
    elif caps_ratio < 0.5:
        score += 0.07

    # ----- Appropriate length (0.15) -----
    text_len = len(response_text.strip())
    if 50 <= text_len <= 500:
        score += 0.15
    elif 20 <= text_len < 50 or 500 < text_len <= 800:
        score += 0.07

    # ----- Non-empty (0.10) -----
    score += 0.10

    return clamp_score(min(score, 1.0))


def grade_efficiency(steps_taken: int, max_steps: int, min_steps: int) -> float:
    """Grade step efficiency.

    Returns 1.0 if completed in minimum steps, decreases linearly.
    """
    if steps_taken <= min_steps:
        return clamp_score(1.0)
    elif steps_taken >= max_steps:
        return clamp_score(0.3)  # Minimal credit for completing at all
    else:
        # Linear interpolation
        excess = steps_taken - min_steps
        max_excess = max_steps - min_steps
        return clamp_score(1.0 - (0.7 * excess / max(max_excess, 1)))


def compute_task_score(
    task_id: str,
    task_config: Dict[str, Any],
    ground_truth: Dict[str, Any],
    agent_category: Optional[str],
    agent_priority: Optional[str],
    agent_department: Optional[str],
    agent_response: Optional[str],
    steps_taken: int,
) -> ScoreBreakdown:
    """Compute the full score breakdown for a task.

    Args:
        task_id: Task identifier
        task_config: Task configuration with scoring weights
        ground_truth: Ground truth labels for the email
        agent_category: Agent's category prediction
        agent_priority: Agent's priority prediction
        agent_department: Agent's department routing
        agent_response: Agent's draft response
        steps_taken: Number of steps the agent took

    Returns:
        ScoreBreakdown with per-component and total scores
    """
    scoring = task_config.get("scoring", {})
    is_spam = ground_truth.get("category", "").lower() == "spam"

    # Compute individual component scores
    cat_score = grade_category(agent_category, ground_truth.get("category", ""))
    pri_score = grade_priority(agent_priority, ground_truth.get("priority", ""))
    dept_score = grade_department(agent_department, ground_truth.get("department", ""))
    resp_score = grade_response(
        agent_response,
        ground_truth.get("expected_response_keywords", []),
        is_spam=is_spam,
    )

    # Determine minimum steps for efficiency calc
    required_actions = task_config.get("required_actions", [])
    min_steps = len(required_actions)
    max_steps = task_config.get("max_steps", min_steps)
    eff_score = grade_efficiency(steps_taken, max_steps, min_steps)

    # Compute weighted total
    total = 0.0
    total += scoring.get("category_weight", 0.0) * cat_score
    total += scoring.get("priority_weight", 0.0) * pri_score
    total += scoring.get("department_weight", 0.0) * dept_score
    total += scoring.get("response_weight", 0.0) * resp_score
    total += scoring.get("efficiency_weight", 0.0) * eff_score

    # Clamp every component + total to strict (0, 1)
    # format_score ensures int output for whole numbers (no .0)
    return ScoreBreakdown(
        category_score=format_score(round(clamp_score(cat_score), 4)),
        priority_score=format_score(round(clamp_score(pri_score), 4)),
        department_score=format_score(round(clamp_score(dept_score), 4)),
        response_score=format_score(round(clamp_score(resp_score), 4)),
        efficiency_score=format_score(round(clamp_score(eff_score), 4)),
        total_score=format_score(round(clamp_score(min(total, 1.0)), 4)),
    )
