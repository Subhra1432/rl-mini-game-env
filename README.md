---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# 📧 Email Triage Environment

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-yellow)](LICENSE)

A **real-world OpenEnv environment** that simulates email triage — a task humans do every day at scale. An AI agent must classify, prioritize, route, and respond to incoming support emails, scored by deterministic graders.

Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework with full `step()` / `reset()` / `state()` API compliance.

---

## 🎯 What is This?

This environment simulates a **support team inbox** where an AI agent receives emails and must:

1. **Classify** — Assign the correct category (billing, technical, account, feature\_request, spam)
2. **Prioritize** — Set appropriate urgency (critical, high, medium, low)
3. **Route** — Send to the right department (engineering, billing, account\_mgmt, product, security, spam\_filter)
4. **Respond** — Draft a professional, contextual reply

The environment includes **35 realistic support emails** covering billing disputes, security incidents, feature requests, multi-issue threads, escalation scenarios, GDPR requests, accessibility complaints, and spam.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              AI Agent (Client)                   │
│  ┌──────────────────────────────────────────┐    │
│  │  EmailTriageEnv (MCPToolClient)          │    │
│  │  - list_tools() / call_tool()            │    │
│  │  - reset(task_id=...) / step() / state() │    │
│  └──────────┬───────────────────────────────┘    │
└─────────────┼────────────────────────────────────┘
              │ WebSocket / HTTP
┌─────────────▼────────────────────────────────────┐
│           Docker Container (Server)               │
│  ┌──────────────────────────────────────────┐    │
│  │  FastAPI + EmailTriageEnvironment         │    │
│  │  MCP Tools:                               │    │
│  │  - classify_email(category, priority)     │    │
│  │  - route_email(department)                │    │
│  │  - draft_response(response_text)          │    │
│  │  - get_email_details()                    │    │
│  │                                           │    │
│  │  Grader → Deterministic scoring (0-1)     │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

---

## 📋 Tasks

### Task 1: Email Classification (Easy)
| Property | Value |
|----------|-------|
| **Task ID** | `email_classify` |
| **Max Steps** | 1 |
| **Required Actions** | `classify` |
| **Description** | Single-step: assign category + priority |

**Scoring Weights:** Category (60%) + Priority (40%)

### Task 2: Email Triage (Medium)
| Property | Value |
|----------|-------|
| **Task ID** | `email_triage` |
| **Max Steps** | 3 |
| **Required Actions** | `classify`, `route` |
| **Description** | Multi-step: classify, prioritize, and route |

**Scoring Weights:** Category (30%) + Priority (20%) + Department (35%) + Efficiency (15%)

### Task 3: Email Resolution (Hard)
| Property | Value |
|----------|-------|
| **Task ID** | `email_resolve` |
| **Max Steps** | 5 |
| **Required Actions** | `classify`, `route`, `respond` |
| **Description** | Full resolution with ambiguous emails and threaded conversations |

**Scoring Weights:** Category (20%) + Priority (10%) + Department (20%) + Response (35%) + Efficiency (15%)

---

## 🔧 Action Space

All actions are exposed as **MCP tools** (compatible with function-calling LLMs):

| Tool | Parameters | Description |
|------|-----------|-------------|
| `classify_email` | `category: str, priority: str` | Classify category and priority |
| `route_email` | `department: str` | Route to department |
| `draft_response` | `response_text: str` | Draft a response |
| `get_email_details` | *(none)* | Get additional context |

### Valid Values

**Categories:** `billing`, `technical`, `account`, `feature_request`, `spam`

**Priorities:** `critical`, `high`, `medium`, `low`

**Departments:** `engineering`, `billing`, `account_mgmt`, `product`, `security`, `spam_filter`

---

## 👁️ Observation Space

After each `reset()` or `step()`, the environment returns an `Observation` with:

| Field | Type | Description |
|-------|------|-------------|
| `email_id` | `str` | Unique email identifier |
| `email_subject` | `str` | Email subject line |
| `email_body` | `str` | Full email body |
| `sender` | `str` | Sender address |
| `timestamp` | `str` | ISO 8601 timestamp |
| `thread_history` | `list` | Previous messages in thread |
| `current_classification` | `str?` | Agent's current category |
| `current_priority` | `str?` | Agent's current priority |
| `current_department` | `str?` | Agent's routed department |
| `current_response` | `str?` | Agent's draft response |
| `task_id` | `str` | Current task |
| `required_actions` | `list` | Remaining required actions |
| `steps_remaining` | `int` | Steps left |
| `done` | `bool` | Episode terminated? |
| `reward` | `float` | Reward signal (0.0–1.0) |

---

## 📊 Reward Function

Rewards provide **partial progress signal** over the trajectory:

- **Each action** receives an immediate partial reward based on correctness
- **Category accuracy:** 1.0 (exact match) or 0.0
- **Priority accuracy:** 1.0 (exact), 0.5 (one level off), 0.0 (two+ levels off)
- **Department accuracy:** 1.0 (exact match) or 0.0
- **Response quality:** 0.0–1.0 based on keyword coverage (0.6), tone (0.15), length (0.15), non-empty (0.10)
- **Efficiency bonus:** 1.0 (minimum steps) → 0.3 (maximum steps)
- **Penalties:** -0.10 for invalid actions

**Spam handling:** For spam emails, NOT responding earns 1.0 response score.

The final episode score is the weighted sum per task configuration (always 0.0–1.0).

---

## 🚀 Setup

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Install

```bash
# Clone and install
cd email_triage_env
pip install -e .

# With baseline inference support
pip install -e ".[baseline]"

# With dev dependencies
pip install -e ".[dev]"
```

### Run Locally

```bash
# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or run directly
python -m server.app
```

### Use the Client

```python
from email_triage_env import EmailTriageEnv

# Connect to server
with EmailTriageEnv(base_url="http://localhost:8000").sync() as env:
    # Reset with a specific task
    obs = env.reset(task_id="email_classify")

    # Discover tools
    tools = env.list_tools()
    print([t.name for t in tools])

    # Classify the email
    result = env.call_tool("classify_email",
        category="technical", priority="high")
    print(result)
```

### Run Baseline Inference

```bash
# Run using the configured environment variables
API_BASE_URL="https://api.openai.com/v1" MODEL_NAME="gpt-4o-mini" HF_TOKEN="sk-..." python inference.py
```

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t email-triage-env -f server/Dockerfile .

# Run locally
docker run -p 8000:7860 email-triage-env

# Run with custom port
docker run -p 8000:8000 -e PORT=8000 email-triage-env
```

### Deploy to Hugging Face Spaces

```bash
# Install OpenEnv CLI
pip install openenv-core

# Push to HF Spaces
openenv push --repo-id your-username/email-triage-env
```

---

## 📁 Project Structure

```
email_triage_env/
├── __init__.py                    # Package exports
├── models.py                      # Pydantic Action/Observation/State models
├── client.py                      # MCPToolClient subclass
├── openenv.yaml                   # OpenEnv manifest
├── pyproject.toml                 # Package config + dependencies
├── README.md                      # This file
├── .dockerignore                  # Docker exclusions
├── inference.py                   # Standardized OpenEnv inference script
├── data/
│   ├── emails.json                # 35 realistic emails with ground truth
│   └── tasks.json                 # 3 task definitions with scoring rubrics
├── server/
│   ├── __init__.py
│   ├── app.py                     # FastAPI entry point
│   ├── email_triage_environment.py # Core environment logic
│   ├── grader.py                  # Grading/scoring utilities
│   ├── Dockerfile                 # Container image
│   └── requirements.txt           # Server dependencies
└── outputs/
    ├── logs/
    └── evals/
```

---

## 🧪 Validation

```bash
# Validate OpenEnv compliance (requires openenv-core CLI)
openenv validate

# Run tests
PYTHONPATH=. pytest tests/ -v
```

---

## 📄 License

BSD 3-Clause License

---

## 📈 Baseline Scores

The baseline `gpt-4o-mini` evaluation results across the tasks (via standard inference script):

| Task | Difficulty | Average Score (0-1) |
|------|------------|---------------------|
| `email_classify` | Easy | **0.87** |
| `email_triage` | Medium | **0.78** |
| `email_resolve` | Hard | **0.65** |
