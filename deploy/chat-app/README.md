# deploy/chat-app

The chat app in Docker: llama.cpp (router mode, CUDA), the FastAPI backend, the React frontend, the nginx gateway, and the log pipeline (OpenObserve + Vector). Deployed on its own: no other folder in `deploy/` is needed.

```
node deploy/chat-app/init-secrets.mjs     # once: creates the secret files compose refers to
cd deploy/chat-app
docker compose up -d --build              # app on http://localhost:3200, logs on http://localhost:5080
docker compose down
```

From the repo root, `node execute.mjs up` does the same (it creates the secrets itself) and then waits until the stack answers; `node execute.mjs down` stops it. The other `execute.mjs` commands (`status`, `health`, `logs`, `load`, `download`, `prune`, `test`) work on this part.

The source it builds and the state it keeps stay in `experimentation/chat-app/` (frontend, backend, `models/`, `logs/`, `secrets/`); only the Docker files live here. It needs an NVIDIA GPU that Docker can see (`deploy.resources.reservations.devices` in the compose file).

| File | What it is |
|---|---|
| `docker-compose.yaml` | The services. Project name `chat-app`. |
| `backend.Dockerfile`, `frontend.Dockerfile` | The backend and the Vite build served by nginx. |
| `gateway.nginx.conf`, `frontend.nginx.conf` | The one published door (`/`, `/api`, `/v1`), and the static file server behind it. |
| `observability/` | OpenObserve image wrapper, the auto-login proxy and the Vector config. |
| `init-secrets.mjs` | Creates `llama_api_key`, `openobserve_user` and `openobserve_password` if they are missing. |
| `.env.example` | The optional settings: `GATEWAY_PORT`, `OPENOBSERVE_PORT`, `LLAMA_MODELS_MAX`. |
