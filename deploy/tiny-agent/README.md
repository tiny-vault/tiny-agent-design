# deploy/tiny-agent

The TinyAgent terminal chat (`code/tiny_agent`, with the `code/llm_client` it uses) in a container, deployed on its own. No chat app and no other part of the repo's Docker setup is needed to build or run it.

```
cd deploy/tiny-agent
docker compose run --rm --build tiny-agent                                      # the interactive chat
docker compose run --rm --build tiny-agent python -m tiny_agent "Why is the sky blue?"   # one question
docker compose run --rm --build tiny-agent python code/main.py                  # the two-turn memory demo
```

From the repo root the same thing is `node execute.mjs tiny-agent ["question"]`, and `node execute.mjs up tiny-agent` only builds the image.

The agent needs a model server to talk to. By default that is whatever answers on this machine's port 3200 (`http://host.docker.internal:3200/v1`): the model server of `deploy/llm-client` (`node execute.mjs up llm-client`, then `node execute.mjs llm-client load qwen3.5-4b`), or the chat app's gateway. Either way it chats with the model that server has loaded. Set `TINYAGENT_LLM_URL` (in the environment or in `.env`, see `.env.example`) to use another one, for example Ollama. If nothing answers, it says so and exits. Its logs (`cli.log`, `main.log`) are written to `experimentation/chat-app/logs/`, where the chat app's log pipeline picks them up if it is running.

| File | What it is |
|---|---|
| `docker-compose.yaml` | The service. Project name `tiny-agent`. |
| `Dockerfile` | Copies `code/` (standard library only, nothing to install). |
| `Dockerfile.dockerignore` | Keeps the repo-root build context down to `code/`, without tests. |
| `.env.example` | The optional settings. |
