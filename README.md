<div align="center">

# 🤖 TinyAgent

### A minimal, modular, educational agent framework, built step by step.

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white) ![Status](https://img.shields.io/badge/status-LLM%20client%20%2B%20chat-yellow) ![Docs](https://img.shields.io/badge/docs-Sphinx%20%2B%20Furo-8CA1AF?logo=sphinx&logoColor=white) ![Dependencies](https://img.shields.io/badge/runtime%20deps-0-brightgreen)

</div>

---

## ✨ What is this?

An LLM only turns text into text. **The harness**, the code around the model, is what makes it an *agent*. TinyAgent is that harness, built small enough to read in a minute and structured so each capability plugs in as its own step.

```
task ─▶ [memory] ─▶ LLM decides ─▶ action ─▶ tool ─▶ observation ─┐
  ▲                                                               │
  └───────────────────────────── loop ◀───────────────────────────┘
```

> **Status:** `TinyAgent` calls a model once per task and keeps the conversation in `Memory`, so a second `run` sees the first. The multi-step loop, tools and the planner are still placeholders. Around it there is a working **LLM client**, a **model check**, a **terminal chat** (`tiny-agent`) and a **Docker chat app** that serves local models, so you can already talk to a real model.

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
| `_step()` | Think | Send the memory to the LLM, store its reply, return answer or action |
| `_execute_action(action)` | Act | Dispatch to a tool, return an observation |

## 🚀 Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m tiny_agent.main
```

The demo tells the agent the authors' names and then asks for them in a second run on the same agent, so memory is what answers; it prints the second reply. It calls a model, so a model server must be running first: the chat app (`node execute.mjs up`) or just the llm client's model server (`node execute.mjs up llm-client`, then `node execute.mjs llm-client load qwen3.5-4b`). See [INSTRUCTIONS.md](INSTRUCTIONS.md) for the settings it reads.

To talk to a model, start the Docker stack and use the terminal chat below.

### Start the stack

```powershell
node execute.mjs up
```

This starts the app (<http://localhost:3200>), the model server (llama.cpp), and a log search (<http://localhost:5080>, already logged in). llama.cpp downloads each model from Hugging Face the first time it is loaded, which takes minutes; after that, switching models takes 35 to 70 seconds. See [INSTRUCTIONS.md](INSTRUCTIONS.md) for everything else (`down`, `prune`, `test`, `download`).

The Docker side is four independent parts in `deploy/` (the chat app, the llm client with its own model server, the tiny agent, the docs); each can be brought up on its own, see [deploy/README.md](deploy/README.md). The chat app and the llm client's model server share port 3200 and the GPU, so run one of them at a time.

### Chat from the terminal

```powershell
python -m pip install -e .
tiny-agent
```

This gives you a `tiny-agent` command: an interactive chat with the model, or `tiny-agent "your question"` for a single answer. It uses the chat app stack and whichever model is loaded, and writes `experimentation/chat-app/logs/cli.log`. Without installing, `python -m tiny_agent.cli` does the same. See `INSTRUCTIONS.md` for the options.

### Settings: `tiny_agent/.env`

The Python code reads its settings from `tiny_agent/.env` (git-ignored; `tiny_agent/.env.example` is the template), so you do not have to repeat them on every command. A value set in your shell or given as an option always wins over the file. An empty or missing file is fine.

| Setting | What it does |
|---|---|
| `TINYAGENT_MODEL` | The model to chat with (default: whichever is loaded) |
| `TINYAGENT_LLM_URL` | The server address (default: the Docker gateway) |
| `TINYAGENT_SYSTEM`, `TINYAGENT_THINK` | A system message; let the model think |
| `TINYAGENT_TRIM_MEMORY` | `1` gives the `main.py` demo a `TrimmingMemory` that keeps only the last two turns (default: keep everything) |
| `TINYAGENT_SUMMARY_MEMORY` | `1` gives the `main.py` demo a `SummarizationMemory` that keeps a model-written running summary instead of the messages (one extra model call per task; wins over `TINYAGENT_TRIM_MEMORY`) |
| `TINYAGENT_RAG_MEMORY`, `TINYAGENT_EMBEDDING_MODEL` | `1` gives the `main.py` demo a `RAGMemory` that puts the three documents most similar to each question in front of it, found with an embedding model (default `embeddinggemma-300m`, on the same server). The server needs both models loaded at once (`LLAMA_MODELS_MAX=2`). Wins over the two flags above |
| `TINYAGENT_API_KEY_FILE` or `TINYAGENT_API_KEY` | A key for a hosted server (the Docker stack needs none) |
| `TINYAGENT_LOG_DIR`, `TINYAGENT_LOG_LEVEL` | Where and how much `cli.log` and `main.log` log |

Or from code:

```python
from tiny_agent.agent.memory import Memory
from tiny_agent.agent.tiny_agent import TinyAgent
from tiny_agent.llm_client.llm import LLM

llm = LLM(model="gemma4-e4b")              # the LLM client: talks to a real model
print(llm.generate([{"role": "user", "content": "Hi!"}]).content)

agent = TinyAgent(llm, Memory())           # the harness: one model call per task, remembered in memory
print(agent.run("Hello, agent!"))
```

## 📖 Documentation

Full docs (architecture, API reference, the chat app backend, the book scripts) are built with Sphinx and the Furo theme. `node execute.mjs docs` builds them in Docker and serves them at <http://localhost:3300>; to build them yourself:

```powershell
pip install -r docs/requirements.txt
python -m sphinx -b html -n -W --keep-going docs docs/_build/html
```

Then open `docs/_build/html/index.html`.

## 📊 Results so far

Measured on the development laptop (RTX 3060 Laptop, 6 GB), with the two local models.

| What | Result |
|---|---|
| Switching to `qwen3.5-4b` (2.7 GB), model on disk | 35 s |
| Switching to `gemma4-e4b` (5.2 GB), model on disk | 61 to 69 s |
| First load, including the Hugging Face download | 6 min (`qwen3.5-4b`), 15 to 20 min (`gemma4-e4b`) |
| Answer starts, model already loaded | about 1 s |
| Tests | backend 60, llm_client 123, tiny_agent 107, browser (stubbed API) 35, live (real models and logs) in INSTRUCTIONS.md |

Only one model fits in GPU memory at a time, so every switch is a full reload. The chat app is described in `docs/guides/chat-app.md`.

## 🗂️ Project layout

```
tiny-agent-design/
├── tiny_agent/          # the TinyAgent package, self-contained:
│   ├── main.py          #   entry point: creates the agent and runs a task
│   ├── cli.py           #   the tiny-agent terminal chat
│   ├── .env.example     #   settings template (copy to .env in this folder)
│   ├── agent/           #   the harness: TinyAgent, trajectory, step
│   ├── memory/          #   Memory and its variants (trimming, summarization, RAG)
│   ├── llm_client/      #   LLM client, model check, .env loader (own README and tests)
│   └── tests/           #   pytest suite
├── deploy/              # Docker, four independent parts, each with its own compose file and README:
│   ├── chat-app/        #   llama.cpp, backend, frontend, gateway, logs (OpenObserve, Vector)
│   ├── llm-client/      #   the LLM client, run on demand
│   ├── tiny-agent/      #   the TinyAgent terminal chat, run on demand
│   └── docs/            #   the Sphinx docs
├── experimentation/
│   ├── chat-app/        # the app's source and state: frontend, backend, models/, logs/, secrets/
│   ├── illustrated_agents/  # runnable scripts from the book, one file per concept (zero_shot, best_of_n/orm, ...)
│   ├── trigger_ias.py   # popup to pick and run those scripts
│   └── tests/           # tests for trigger_ias.py and the scripts
├── execute.mjs          # start, stop, test and clean up: one entry point for the four Docker parts
├── pyproject.toml       # packaging: the tiny-agent command
└── docs/                # Sphinx documentation
```

## 🛣️ Roadmap

- [x] Agent skeleton and control-flow stubs
- [x] Sphinx documentation
- [x] LLM client, model check and terminal chat (works against local models)
- [x] Docker chat app: llama.cpp with Hugging Face models, logs, secrets, tests
- [x] Step 1: LLM integration into `TinyAgent` (one model call per task)
- [x] Step 2: Memory (the conversation is kept and resent on every call)
- [x] Deploy as four independent Docker parts (`deploy/`)
- [ ] Step 3: Tools
- [ ] Step 4: Planning (and the multi-step loop)

---

<div align="center">

Built to learn how agents work, one small piece at a time.

Based on *An Illustrated Guide to AI Agents* by Maarten Grootendorst and Jay Alammar (O'Reilly). Companion code: [HandsOnLLM/An-Illustrated-Guide-To-AI-Agents](https://github.com/HandsOnLLM/An-Illustrated-Guide-To-AI-Agents). That repository has no license, so all rights remain with the authors. This is a personal learning project, not affiliated with or endorsed by them. If you find the book useful, please cite it.

The chat UI's look is adapted from [Utopia](https://github.com/deeplethe/utopia) by DeepLethe (Apache-2.0).

</div>
