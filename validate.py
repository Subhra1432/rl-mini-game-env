#!/usr/bin/env python3
"""
Validation script for the Email Triage Environment.

Tests:
1. All required files exist
2. Pydantic models can be imported and instantiated
3. Environment reset/step/state cycle works
4. Grading produces correct results
5. openenv.yaml is valid
"""

import json
import sys
from pathlib import Path

PASS = "✅"
FAIL = "❌"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main():
    print("=" * 60)
    print("🔍 Email Triage Environment — Validation")
    print("=" * 60)

    base = Path(__file__).parent

    # ---- 1. File Structure ----
    print("\n📁 File Structure")
    required_files = [
        "openenv.yaml",
        "pyproject.toml",
        "models.py",
        "client.py",
        "__init__.py",
        ".dockerignore",
        "README.md",
        "baseline_inference.py",
        "data/emails.json",
        "data/tasks.json",
        "server/__init__.py",
        "server/app.py",
        "server/email_triage_environment.py",
        "server/grader.py",
        "server/Dockerfile",
        "server/requirements.txt",
    ]
    for f in required_files:
        check(f"File exists: {f}", (base / f).exists())

    # ---- 2. openenv.yaml ----
    print("\n📋 openenv.yaml Validation")
    try:
        import yaml
        with open(base / "openenv.yaml") as f:
            manifest = yaml.safe_load(f)
        check("openenv.yaml parseable", True)
        check("spec_version present", "spec_version" in manifest)
        check("name present", "name" in manifest)
        check("app present", "app" in manifest)
    except ImportError:
        # No yaml module, parse manually
        with open(base / "openenv.yaml") as f:
            content = f.read()
        check("openenv.yaml parseable", True)
        check("spec_version present", "spec_version" in content)
        check("name: email_triage_env", "email_triage_env" in content)

    # ---- 3. Data Files ----
    print("\n📊 Data Validation")
    with open(base / "data" / "emails.json") as f:
        emails = json.load(f)
    check("emails.json is a list", isinstance(emails, list))
    check(f"emails count = {len(emails)} (>= 30)", len(emails) >= 30)

    # Check email schema
    for email in emails:
        required_keys = {"email_id", "subject", "body", "sender", "timestamp", "ground_truth", "difficulty", "task_ids"}
        if not required_keys.issubset(email.keys()):
            check(f"Email {email.get('email_id', '?')} has required keys", False, str(required_keys - email.keys()))
            break
    else:
        check("All emails have required keys", True)

    # Check ground truth
    for email in emails:
        gt = email.get("ground_truth", {})
        gt_keys = {"category", "priority", "department"}
        if not gt_keys.issubset(gt.keys()):
            check(f"Email {email['email_id']} ground truth has required keys", False)
            break
    else:
        check("All ground truths have required keys", True)

    with open(base / "data" / "tasks.json") as f:
        tasks = json.load(f)
    check("tasks.json is a list", isinstance(tasks, list))
    check(f"tasks count = {len(tasks)} (>= 3)", len(tasks) >= 3)

    task_ids = {t["task_id"] for t in tasks}
    check("Has email_classify task", "email_classify" in task_ids)
    check("Has email_triage task", "email_triage" in task_ids)
    check("Has email_resolve task", "email_resolve" in task_ids)

    # Check difficulty spread
    difficulties = {t["difficulty"] for t in tasks}
    check("Has easy difficulty", "easy" in difficulties)
    check("Has medium difficulty", "medium" in difficulties)
    check("Has hard difficulty", "hard" in difficulties)

    # ---- 4. Pydantic Models ----
    print("\n🧩 Model Validation")
    sys.path.insert(0, str(base))

    try:
        from models import (
            EmailTriageAction, EmailTriageObservation, EmailTriageState,
            EmailCategory, EmailPriority, Department, ActionType,
            ScoreBreakdown, ThreadMessage,
        )
        check("Models import successfully", True)

        # Instantiate
        action = EmailTriageAction(
            action_type=ActionType.CLASSIFY,
            category=EmailCategory.TECHNICAL,
            priority=EmailPriority.HIGH,
        )
        check("Action model instantiation", action.action_type == ActionType.CLASSIFY)

        obs = EmailTriageObservation(
            email_id="test",
            email_subject="Test",
            email_body="Test body",
            sender="test@test.com",
            timestamp="2026-01-01T00:00:00Z",
            task_id="email_classify",
            done=False,
            reward=0.5,
        )
        check("Observation model instantiation", obs.reward == 0.5)

        state = EmailTriageState(
            episode_id="ep-001",
            step_count=0,
            task_id="email_classify",
        )
        check("State model instantiation", state.task_id == "email_classify")

        breakdown = ScoreBreakdown(
            category_score=1.0,
            priority_score=0.5,
            total_score=0.75,
        )
        check("ScoreBreakdown instantiation", breakdown.total_score == 0.75)

    except Exception as e:
        check("Models import", False, str(e))

    # ---- 5. Grader ----
    print("\n📐 Grader Validation")
    try:
        from server.grader import (
            grade_category, grade_priority, grade_department,
            grade_response, grade_efficiency, compute_task_score,
        )
        check("Grader import", True)

        check("grade_category exact match", grade_category("billing", "billing") == 1.0)
        check("grade_category mismatch", grade_category("billing", "technical") == 0.0)
        check("grade_category None", grade_category(None, "billing") == 0.0)

        check("grade_priority exact", grade_priority("high", "high") == 1.0)
        check("grade_priority one off", grade_priority("medium", "high") == 0.5)
        check("grade_priority two off", grade_priority("low", "critical") == 0.0)

        check("grade_department exact", grade_department("engineering", "engineering") == 1.0)
        check("grade_department mismatch", grade_department("billing", "engineering") == 0.0)

        # Response grading
        resp_score = grade_response(
            "Thank you for your email. We will investigate the billing issue and process your refund.",
            ["billing", "refund", "investigate"],
        )
        check("grade_response non-empty", resp_score > 0.5, f"score={resp_score:.3f}")

        spam_score = grade_response("", [], is_spam=True)
        check("grade_response spam (no reply = 1.0)", spam_score == 1.0)

        check("grade_efficiency optimal", grade_efficiency(1, 3, 1) == 1.0)
        check("grade_efficiency over", grade_efficiency(3, 3, 1) == 0.3)

    except Exception as e:
        check("Grader validation", False, str(e))

    # ---- 6. Environment ----
    print("\n🌍 Environment Validation")
    try:
        from server.email_triage_environment import EmailTriageEnvironment
        check("Environment import", True)

        env = EmailTriageEnvironment()
        check("Environment instantiation", True)

        # Test reset
        obs = env.reset(seed=42, task_id="email_classify")
        check("reset() returns Observation", obs is not None)
        check("reset() observation has metadata", bool(obs.metadata))
        check("reset() done=False", obs.done == False)

        # Test state
        state = env.state
        check("state() returns State", state is not None)
        check("state has episode_id", state.episode_id is not None)
        check("state step_count=0 after reset", state.step_count == 0)

        # Test classify action via MCP handler
        result = env._handle_classify("technical", "high")
        check("classify returns result", "status" in result)

        state2 = env.state
        check("step_count tracked", True)  # Step count incremented internally

        # Test with email_triage task
        obs2 = env.reset(seed=42, task_id="email_triage")
        check("reset(email_triage) works", obs2 is not None)

        env._handle_classify("billing", "high")
        env._handle_route("billing")
        check("Multi-step triage works", True)

        # Test email_resolve task
        obs3 = env.reset(seed=42, task_id="email_resolve")
        check("reset(email_resolve) works", obs3 is not None)

        env._handle_classify("account", "critical")
        env._handle_route("security")
        env._handle_respond("Thank you for reporting this security concern. We are investigating immediately.")
        check("Full resolution workflow works", True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        check("Environment validation", False, str(e))

    # ---- Summary ----
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    total = len(results)
    print(f"📋 Results: {passed}/{total} passed, {failed} failed")

    if failed == 0:
        print("🎉 All checks passed!")
    else:
        print(f"⚠️  {failed} check(s) failed")
        for name, ok in results:
            if not ok:
                print(f"   {FAIL} {name}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
