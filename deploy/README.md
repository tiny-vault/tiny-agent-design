# deploy

Everything that runs in Docker, as four independent parts. Each folder has its own `docker-compose.yaml`, Dockerfiles, `.env.example` and README, its own compose project name, and can be built, started and stopped without any of the others.

| Folder | What it deploys | Start from the repo root | Reach it at |
|---|---|---|---|
| [`chat-app/`](chat-app/README.md) | llama.cpp, backend, frontend, gateway, logs | `node execute.mjs up` | <http://localhost:3200> (app), <http://localhost:5080> (logs) |
| [`llm-client/`](llm-client/README.md) | the LLM client (`tiny_agent/llm_client`), run on demand | `node execute.mjs llm-client` | prints in the terminal |
| [`tiny-agent/`](tiny-agent/README.md) | the TinyAgent terminal chat (`tiny_agent/`), run on demand | `node execute.mjs tiny-agent` | the terminal |
| [`docs/`](docs/README.md) | the Sphinx docs | `node execute.mjs up docs` | <http://localhost:3300> |

`node execute.mjs` is the one entry point: `up [chat-app|llm-client|tiny-agent|docs|all]`, `down [target]`, and the commands in `node execute.mjs help`. To use one part without it, `cd` into its folder and follow its README.

The llm client and the tiny agent need a model server. By default that is the chat app's gateway on this machine, but `TINYAGENT_LLM_URL` points them at any OpenAI-compatible server, so they do not need the chat app at all.

Why four projects and not one compose file: starting or stopping one part (`up --remove-orphans`, `down`) can never remove another part's containers, and each part builds only what it needs.
