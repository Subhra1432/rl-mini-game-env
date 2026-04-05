import sys
import json
sys.path.insert(0, ".")
from client import EmailTriageEnv

def test_running_container():
    print("🚀 Connecting to your local uvicorn server on port 8001...")
    
    with EmailTriageEnv(base_url="http://127.0.0.1:8001").sync() as env:
        print("\n📧 Asking the environment for a new email task...")
        obs = env.reset(task_id="email_classify")
        print("\n--- NEW EMAIL RECEIVED ---")
        print(f"From: {obs.observation.metadata['sender']}")
        print(f"Subject: {obs.observation.metadata['email_subject']}")
        print(f"Body snippet: {obs.observation.metadata['email_body'][:100]}...\n--------------------------")
        print(f"\nTask Assigned: {obs.observation.metadata['task_id']}")
        print("\n🛠️ Querying available MCP tools from the container...")
        tools = env.list_tools()
        for t in tools:
            print(f" 🔹 {t.name}: {getattr(t, 'description', 'No description')}")
        print("\n🧠 Simulating AI Agent action (classifying the email)...")
        state = env.call_tool("classify_email", category="technical", priority="high")
        print(f"✅ Success! Observation returned from the server:")
        print(f"   Reward: {state.reward}")
        print(f"   Done: {state.done}")
        
if __name__ == "__main__":
    test_running_container()
