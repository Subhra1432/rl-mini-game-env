"""
Test script for the Email Triage Environment running in Docker.

Connects to the container, resets the environment, discovers tools,
uses get_email_details to read the email, then classifies it.
"""
import sys
sys.path.insert(0, ".")
from client import EmailTriageEnv


def test_running_container():
    print("🚀 Connecting to your local Docker container on port 8000...")

    with EmailTriageEnv(base_url="http://127.0.0.1:8000").sync() as env:

        print("\n📧 Resetting the environment for a new email task...")
        obs = env.reset(task_id="email_classify")
        print(f"   Reset complete. Done={obs.done}, Reward={obs.reward}")

        # ── Discover available tools ──────────────────────────
        print("\n🛠️  Available MCP tools:")
        tools = env.list_tools()
        for t in tools:
            print(f"   🔹 {t.name}: {t.description[:80]}...")

        # ── Fetch email details via the dedicated tool ────────
        print("\n📨 Fetching email details via get_email_details tool...")
        details = env.call_tool("get_email_details")
        print("\n--- EMAIL RECEIVED ---")
        print(f"   ID:      {details['email_id']}")
        print(f"   From:    {details['sender']}")
        print(f"   Subject: {details['subject']}")
        print(f"   Body:    {details['body'][:120]}...")
        if details.get("thread_count", 0) > 0:
            print(f"   Thread:  {details['thread_count']} previous messages")
        print("----------------------")

        # ── Agent action: classify the email ──────────────────
        print("\n🧠 Simulating AI Agent action (classifying the email)...")
        result = env.call_tool("classify_email", category="technical", priority="high")
        print(f"   ✅ Classification result:")
        print(f"      Category: {result['category']}")
        print(f"      Priority: {result['priority']}")
        print(f"      Reward:   {result['reward']}")
        print(f"      Message:  {result['message']}")

        print("\n🎉 Test completed successfully!")


if __name__ == "__main__":
    test_running_container()
