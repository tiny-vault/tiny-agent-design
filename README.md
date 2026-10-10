<div align="center">

# 🤖 TinyAgent

### A minimal, modular, educational agent framework, built step by step.

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white) ![Status](https://img.shields.io/badge/status-skeleton-orange) ![Docs](https://img.shields.io/badge/docs-Sphinx%20%2B%20Furo-8CA1AF?logo=sphinx&logoColor=white) ![Dependencies](https://img.shields.io/badge/runtime%20deps-0-brightgreen)

</div>

---

## ✨ What is this?

An LLM only turns text into text. **The harness**, the code around the model, is what makes it an *agent*. TinyAgent is that harness, built small enough to read in a minute and structured so each capability plugs in as its own step.

```
task ─▶ [memory] ─▶ LLM decides ─▶ action ─▶ tool ─▶ observation ─┐
  ▲                                                               │
  └───────────────────────────── loop ◀───────────────────────────┘
```

> **Status:** this is a skeleton. The control flow and docs are in place; the LLM, memory, tools and planner are placeholders for the steps below. No model is called yet.

## 🧩 Architecture

| Component | Role | Step |
|---|---|---|
| 🧠 `llm` | The brain: reasons and picks the next step | 1 |
| 📚 `memory` | The context: past messages and observations | 2 |
| 🛠️ `tools` | The hands: functions that act on the world | 3 |
| 🗺️ `planner` | The strategy: splits a goal into sub-steps | 4 |

| Method | Phase | Job |
|---|---|---|
| `run(task)` | Control loop | Public entry point; will loop until done |
| `_step(task)` | Think | Build the prompt, ask the LLM, return answer or action |
| `_execute_action(action)` | Act | Dispatch to a tool, return an observation |

## 🚀 Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m tiny_agent.main
```

```text
Received: Hello, agent!
```

Or from code:

```python
from tiny_agent.shared.tiny_agent import TinyAgent

agent = TinyAgent()
print(agent.run("Hello, agent!"))
```

## 📖 Documentation

Full docs (architecture, API reference) are built with Sphinx and the Furo theme:

```powershell
pip install -r docs/requirements.txt
sphinx-build -b html -n -W --keep-going docs docs/_build/html
```

Then open `docs/_build/html/index.html`.

## 🗂️ Project layout

```
tiny-agent-design/
├── tiny_agent/
│   ├── main.py      # entry point: creates the agent and runs a task
│   └── shared/      # the TinyAgent harness, LLM client, model check
└── docs/            # Sphinx documentation
```

## 🛣️ Roadmap

- [x] Agent skeleton and control-flow stubs
- [x] Sphinx documentation
- [ ] Step 1: LLM integration
- [ ] Step 2: Memory
- [ ] Step 3: Tools
- [ ] Step 4: Planning

---

<div align="center">

Built to learn how agents work, one small piece at a time.

Based on *An Illustrated Guide to AI Agents* by Maarten Grootendorst and Jay Alammar (O'Reilly). Companion code: [HandsOnLLM/An-Illustrated-Guide-To-AI-Agents](https://github.com/HandsOnLLM/An-Illustrated-Guide-To-AI-Agents). That repository has no license, so all rights remain with the authors. This is a personal learning project, not affiliated with or endorsed by them. If you find the book useful, please cite it.

</div>
