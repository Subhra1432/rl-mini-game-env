"""
Full Demo: Email Triage Environment
Runs multiple emails across all 3 task levels with a simple rule-based agent.
"""
import sys
import time
sys.path.insert(0, ".")
from client import EmailTriageEnv

BASE_URL = "http://127.0.0.1:8000"

# ─── Simple Rule-Based Agent ────────────────────────────────────────────────
def classify_email_heuristic(email: dict) -> tuple[str, str]:
    """Simple keyword-based classifier (no LLM needed)."""
    subject = (email.get("subject", "") + " " + email.get("body", "")).lower()

    # Category
    if any(w in subject for w in ["invoice", "billing", "charge", "refund", "payment", "price", "discount", "subscription", "student discount"]):
        category = "billing"
    elif any(w in subject for w in ["bug", "error", "crash", "broken", "fix", "issue", "not working", "failed", "ssl", "api", "integration", "code", "technical", "install"]):
        category = "technical"
    elif any(w in subject for w in ["account", "login", "password", "access", "locked", "username", "2fa", "two-factor", "enable", "disable"]):
        category = "account"
    elif any(w in subject for w in ["feature", "request", "suggestion", "idea", "would be great", "could you add", "wish", "enhancement"]):
        category = "feature_request"
    elif any(w in subject for w in ["investment", "guaranteed", "lottery", "winner", "million", "crypto", "exclusive", "urgent", "click here", "nigerian"]):
        category = "spam"
    else:
        category = "technical"

    # Priority
    if any(w in subject for w in ["urgent", "critical", "asap", "immediately", "emergency", "down", "outage", "security breach"]):
        priority = "critical"
    elif any(w in subject for w in ["important", "broken", "not working", "error", "bug", "crash", "high"]):
        priority = "high"
    elif any(w in subject for w in ["spam", "lottery", "guaranteed", "investment"]):
        priority = "low"
    else:
        priority = "medium"

    return category, priority


def get_department(category: str) -> str:
    """Map category to department."""
    mapping = {
        "billing": "billing",
        "technical": "engineering",
        "account": "account_mgmt",
        "feature_request": "product",
        "spam": "spam_filter",
    }
    return mapping.get(category, "engineering")


def draft_response(email: dict, category: str) -> str:
    """Generate a simple response based on category."""
    sender_name = email.get("sender", "there").split("@")[0].split(".")[0].capitalize()
    subject = email.get("subject", "your inquiry")
    responses = {
        "billing": f"Hi {sender_name},\n\nThank you for reaching out about billing. Our billing team will review your account and get back to you within 1 business day.\n\nBest regards,\nSupport Team",
        "technical": f"Hi {sender_name},\n\nThank you for reporting this technical issue. Our engineering team has been notified and will investigate. Please share any error logs you have.\n\nBest regards,\nSupport Team",
        "account": f"Hi {sender_name},\n\nThank you for contacting us about your account. Please verify your identity and we'll assist you right away.\n\nBest regards,\nSupport Team",
        "feature_request": f"Hi {sender_name},\n\nThank you for the great suggestion! We've logged your feature request and will consider it for future releases.\n\nBest regards,\nProduct Team",
        "spam": "",
    }
    return responses.get(category, f"Hi {sender_name},\n\nThank you for contacting support. We'll get back to you shortly.\n\nBest regards,\nSupport Team")


def print_separator(char="─", width=60):
    print(char * width)


def run_task(task_id: str, task_label: str, task_desc: str, episode_num: int):
    """Run a single episode for a given task."""
    print(f"\n{'═'*60}")
    print(f"  📋 Episode {episode_num} | Task: {task_label}")
    print(f"  {task_desc}")
    print(f"{'═'*60}")

    with EmailTriageEnv(base_url=BASE_URL).sync() as env:
        # Reset
        env.reset(task_id=task_id)

        # Get email
        email = env.call_tool("get_email_details")
        print(f"\n📩 From:    {email['sender']}")
        print(f"   Subject: {email['subject']}")
        body_preview = email['body'][:150].replace('\n', ' ')
        print(f"   Body:    {body_preview}...")

        # Agent decides
        category, priority = classify_email_heuristic(email)
        department = get_department(category)
        response = draft_response(email, category)

        print(f"\n🤖 Agent Decision:")
        print(f"   Category  : {category}")
        print(f"   Priority  : {priority}")

        # ── Task 1: Classify only ────────────────────────────
        if task_id == "email_classify":
            result = env.call_tool("classify_email", category=category, priority=priority)
            score = result.get("reward", 0.0)
            final_msg = result.get("message", "")

        # ── Task 2: Classify + Route ─────────────────────────
        elif task_id == "email_triage":
            print(f"   Department: {department}")
            env.call_tool("classify_email", category=category, priority=priority)
            result = env.call_tool("route_email", department=department)
            score = result.get("reward", 0.0)
            final_msg = result.get("message", "")

        # ── Task 3: Classify + Route + Respond ───────────────
        elif task_id == "email_resolve":
            print(f"   Department: {department}")
            if category != "spam":
                print(f"   Response  : {response[:80]}...")
            else:
                print(f"   Response  : [No response — marked as spam]")
            env.call_tool("classify_email", category=category, priority=priority)
            env.call_tool("route_email", department=department)
            result = env.call_tool("draft_response", response_text=response)
            score = result.get("reward", 0.0)
            final_msg = result.get("message", "")

        # Extract final score from message
        final_score = 0.0
        if "Final score:" in final_msg:
            try:
                final_score = float(final_msg.split("Final score:")[1].split("/")[0].strip())
            except Exception:
                final_score = score

        # Score bar
        filled = int(final_score * 20)
        bar = "█" * filled + "░" * (20 - filled)
        pct = int(final_score * 100)

        print(f"\n📊 Result:")
        print(f"   [{bar}] {pct}%")
        print(f"   {final_msg}")

        return final_score


def main():
    print("\n" + "🔬 EMAIL TRIAGE ENVIRONMENT — FULL DEMO ".center(60, "═"))
    print(f"   Server: {BASE_URL}")
    print(f"   Running 9 episodes across all 3 task levels")
    print("═" * 60)

    tasks = [
        ("email_classify", "EASY  — Classify", "Just pick category + priority"),
        ("email_triage",   "MEDIUM — Triage",  "Classify + route to department"),
        ("email_resolve",  "HARD  — Resolve",  "Classify + route + write response"),
    ]

    all_scores = []
    episode = 1

    for task_id, label, desc in tasks:
        task_scores = []
        print(f"\n\n{'▶'*3} Running 3 episodes of [{label}]...")
        for _ in range(3):
            score = run_task(task_id, label, desc, episode)
            task_scores.append(score)
            all_scores.append(score)
            episode += 1
            time.sleep(0.3)  # small pause between episodes

        avg = sum(task_scores) / len(task_scores)
        print(f"\n  ✅ {label} average: {avg:.0%}")

    # Final summary
    print(f"\n\n{'═'*60}")
    print("  🏆 FINAL SUMMARY")
    print(f"{'═'*60}")
    total_avg = sum(all_scores) / len(all_scores)
    print(f"  Episodes run    : {len(all_scores)}")
    print(f"  Individual scores: {[f'{s:.0%}' for s in all_scores]}")
    filled = int(total_avg * 20)
    bar = "█" * filled + "░" * (20 - filled)
    print(f"  Overall average : [{bar}] {total_avg:.0%}")

    if total_avg >= 0.7:
        print("\n  🎉 Great performance! The agent is working well.")
    elif total_avg >= 0.4:
        print("\n  📈 Decent performance. An LLM agent would score much higher.")
    else:
        print("\n  🔧 Rule-based agent struggles. This is where LLMs shine!")

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
