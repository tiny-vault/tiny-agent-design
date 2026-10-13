# deploy/docs

The Sphinx documentation, built inside a Docker image and served by nginx on <http://localhost:3300> (this machine only). Deployed on its own: it needs no chat app, no model and no secrets.

```
cd deploy/docs
docker compose up -d --build     # build the docs and serve them
docker compose down
```

From the repo root the same thing is `node execute.mjs up docs` and `node execute.mjs down docs` (or just `node execute.mjs docs`). Change the port with `DOCS_PORT` (environment or `.env`, see `.env.example`).

The build runs the repo's docs check (`sphinx-build -n -W`), so a documentation warning fails the build and the message says where. The docs import the code they describe, which is why `docs/Dockerfile.dockerignore` lets `tiny_agent/`, `docs/`, `experimentation/illustrated_agents`, `experimentation/trigger_ias.py` and the two chat-app backend modules into the build. The book's scripts read a llama.cpp key file when imported; the build writes a placeholder one and removes it again.

Without Docker, `node execute.mjs docs --local` builds with the Sphinx installed on this machine and serves the result.

| File | What it is |
|---|---|
| `docker-compose.yaml` | The service. Project name `tiny-agent-docs`. |
| `Dockerfile` | Two stages: build the docs, then copy the HTML into nginx. |
| `Dockerfile.dockerignore` | Keeps the repo-root build context down to what the docs import. |
| `.env.example` | The optional port setting. |
