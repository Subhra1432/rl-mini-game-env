#!/usr/bin/env python3
"""
OpenEnv Baseline Inference Script
Complies with OpenEnv strictly structured stdout logging.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from server.email_triage_environment import EmailTriageEnvironment
from server.grader import compute_task_score

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

BENCHMARK = "email_triage_env"
TASKS = ["email_classify", "email_triage", "email_resolve"]
SEED = 42
EMAILS_PER_TASK = 10
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are an expert email triage agent. You work at a SaaS company's support department.

Your job is to analyze incoming support emails and take appropriate actions using the available tools.

## Available Tools
- classify_email(category, priority): Classify the email
  - Categories: billing, technical, account, feature_request, spam
  - Priorities: critical, high, medium, low
- route_email(department): Route to the correct department
  - Departments: engineering, billing, account_mgmt, product, security, spam_filter
- draft_response(response_text): Draft a professional response
- get_email_details(): Get additional context about the email

## Guidelines
1. Read the email carefully, considering subject, body, sender, and any thread history.
2. For classification: Consider urgency indicators, affected user count, and potential impact.
3. For routing: Match the email category and specific needs to the right team.
4. For responses: Be professional, empathetic, address the sender's concerns, and provide next steps.
5. For spam: Classify as 'spam' with 'low' priority, route to 'spam_filter', do NOT draft a response.
6. Act efficiently — use the minimum number of steps needed.
"""

def get_task_prompt(task_id: str, email_data: Dict[str, Any]) -> str:
    thread_text = ""
    thread = email_data.get("thread_history", [])
    if thread:
        thread_text = "\n\n## Thread History (oldest first)\n"
        for msg in thread:
            sender = msg.get("from", msg.get("sender", "unknown"))
            thread_text += f"- [{msg['date']}] {sender}: {msg['snippet']}\n"

    base = f"## Email to Triage\n\n**From:** {email_data['sender']}\n**Subject:** {email_data['subject']}\n**Date:** {email_data['timestamp']}\n\n**Body:**\n{email_data['body']}{thread_text}\n\n---\n"

    if task_id == "email_classify":
        base += "## Your Task (Easy)\nClassify this email by calling `classify_email` with the correct category and priority.\nYou have 1 step. Make a single classify_email call."
    elif task_id == "email_triage":
        base += "## Your Task (Medium)\n1. Classify this email (category + priority) using `classify_email`\n2. Route it to the correct department using `route_email`\nYou have up to 3 steps. Complete both actions."
    elif task_id == "email_resolve":
        base += "## Your Task (Hard)\n1. Classify this email (category + priority) using `classify_email`\n2. Route it to the correct department using `route_email`\n3. Draft a professional response using `draft_response` (unless it's spam)\nYou have up to 5 steps. Complete all required actions."
    return base

def build_tools_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "classify_email",
                "description": "Classify the email into a category and assign a priority level",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": ["billing", "technical", "account", "feature_request", "spam"]},
                        "priority": {"type": "string", "enum": ["critical", "high", "medium", "low"]}
                    },
                    "required": ["category", "priority"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "route_email",
                "description": "Route the email to the appropriate department",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "department": {"type": "string", "enum": ["engineering", "billing", "account_mgmt", "product", "security", "spam_filter"]}
                    },
                    "required": ["department"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "draft_response",
                "description": "Draft a professional response to the email sender",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response_text": {"type": "string"}
                    },
                    "required": ["response_text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_email_details",
                "description": "Get additional details about the current email",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def main():
    if not HF_TOKEN:
        print("Error: HF_TOKEN or OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print(f"Starting inference.py with Model: {MODEL_NAME}", file=sys.stderr)
    
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = EmailTriageEnvironment()

    data_dir = Path(__file__).parent / "data"
    with open(data_dir / "emails.json", "r") as f:
        all_emails = json.load(f)
    with open(data_dir / "tasks.json", "r") as f:
        all_tasks = {t["task_id"]: t for t in json.load(f)}

    tools = build_tools_schema()

    for task_id in TASKS:
        task_config = all_tasks[task_id]
        task_filter = task_config.get("email_filter", {})
        allowed_difficulties = task_filter.get("difficulties", ["easy", "medium", "hard"])

        eligible = [
            e for e in all_emails
            if e.get("difficulty", "easy") in allowed_difficulties
            and task_id in e.get("task_ids", [])
        ]

        rng = random.Random(SEED)
        selected = rng.sample(eligible, min(EMAILS_PER_TASK, len(eligible)))

        for i, email in enumerate(selected):
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            
            env.reset(seed=SEED + i, task_id=task_id, email_id=email["email_id"])
            user_prompt = get_task_prompt(task_id, email)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            if task_id == "email_classify":
                available_tools = [t for t in tools if t["function"]["name"] == "classify_email"]
            elif task_id == "email_triage":
                available_tools = [t for t in tools if t["function"]["name"] in ("classify_email", "route_email")]
            else:
                available_tools = tools

            steps = 0
            max_steps = task_config["max_steps"]
            agent_category = None
            agent_priority = None
            agent_department = None
            agent_response = None
            
            rewards = []

            while steps < max_steps:
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        tools=available_tools,
                        tool_choice="auto",
                        temperature=0.0,
                        seed=SEED,
                    )
                except Exception as e:
                    print(f"API error: {e}", file=sys.stderr)
                    steps += 1
                    rewards.append(0.0)
                    log_step(step=steps, action="api_error", reward=0.0, done=True, error=str(e).replace("\\n", " "))
                    break

                choice = response.choices[0]

                if choice.finish_reason == "stop" or not choice.message.tool_calls:
                    if steps == 0:
                        steps += 1
                        rewards.append(0.0)
                        action_str = choice.message.content or "none"
                        action_str = action_str.replace('\\n', ' ').replace('"', "'")
                        log_step(step=steps, action=f'"{action_str}"', reward=0.0, done=True, error=None)
                    break

                for tool_call in choice.message.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}
                    
                    steps += 1
                    error = None

                    if fn_name == "classify_email":
                        agent_category = fn_args.get("category")
                        agent_priority = fn_args.get("priority")
                        result = env._handle_classify(agent_category, agent_priority)
                    elif fn_name == "route_email":
                        agent_department = fn_args.get("department")
                        result = env._handle_route(agent_department)
                    elif fn_name == "draft_response":
                        agent_response = fn_args.get("response_text", "")
                        result = env._handle_respond(agent_response)
                    elif fn_name == "get_email_details":
                        result = env._handle_get_details()
                    else:
                        result = {"status": "error", "message": f"Unknown tool: {fn_name}"}
                        error = result["message"]

                    args_sorted = sorted([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k,v in fn_args.items() if k != 'response_text'])
                    args_str = ", ".join(args_sorted)
                    if fn_name == "draft_response":
                        args_str = "..."
                    action_str = f"{fn_name}({args_str})"
                    
                    messages.append(choice.message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    })
                    
                    reward_val = 0.0
                    done = (steps >= max_steps)
                    rewards.append(reward_val)
                    
                    log_step(step=steps, action=action_str, reward=reward_val, done=done, error=error)

                if steps >= max_steps:
                    break

            gt = email["ground_truth"]
            score_obj = compute_task_score(
                task_id=task_id,
                task_config=task_config,
                ground_truth=gt,
                agent_category=agent_category,
                agent_priority=agent_priority,
                agent_department=agent_department,
                agent_response=agent_response,
                steps_taken=max(steps, 1)
            )
            
            final_score = score_obj.total_score
            success = final_score >= SUCCESS_SCORE_THRESHOLD
            
            if rewards:
                rewards[-1] = float(final_score)
            
            log_end(success=success, steps=max(steps, 1), score=final_score, rewards=rewards)

if __name__ == "__main__":
    main()
