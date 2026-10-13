# The Sphinx docs, built inside the image and served by nginx. Used by the docs service in
# docker-compose.yaml. The build context is the repo root; Dockerfile.dockerignore keeps it to
# the folders the docs import.
FROM python:3.12-slim AS build
WORKDIR /src
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Sphinx and its extensions, plus what the docs import: the book's package (the scripts import
# illustrated_agents) and fastapi/httpx for the chat-app backend page (both are in docs/requirements.txt).
COPY docs/requirements.txt docs/requirements.txt
RUN pip install -r docs/requirements.txt illustrated-agents==0.1.1

# The code the docs document. docs/conf.py finds these folders relative to itself.
COPY code code
COPY experimentation/illustrated_agents experimentation/illustrated_agents
COPY experimentation/trigger_ias.py experimentation/trigger_ias.py
COPY experimentation/chat-app/backend/server.py experimentation/chat-app/backend/llama_client.py experimentation/chat-app/backend/
COPY docs docs

# The book's scripts read a llama.cpp key file when they are imported (they only use the key when they
# run, and the docs never run them). A placeholder file is enough, so this folder needs none of the chat
# app's real secrets.
# -n -W: the same check as the repo's docs command; a warning fails the build.
RUN mkdir -p experimentation/chat-app/secrets \
    && echo "docs-build-placeholder" > experimentation/chat-app/secrets/llama_api_key \
    && python -m sphinx -b html -n -W --keep-going docs docs/_build/html \
    && rm -rf experimentation/chat-app/secrets

FROM nginx:alpine
COPY --from=build /src/docs/_build/html /usr/share/nginx/html
EXPOSE 80
