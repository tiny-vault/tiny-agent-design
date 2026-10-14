# The LLM client alone: tiny_agent/components/client (the OpenAI-compatible chat client and the loaded-model check), tiny_agent/config (the .env loader) and the Response type (inside the client folder),
# and ask.py, a small script that sends one question to a model with it. It needs only
# the libraries pip-installed below, plus nothing else.
# The build context is the repo root; Dockerfile.dockerignore keeps it to these (plus load.py, which waits for a model to load).
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
RUN pip install --no-cache-dir pydantic-settings httpx click
COPY tiny_agent/components/client tiny_agent/components/client
COPY tiny_agent/config tiny_agent/config
COPY deploy/llm-client/ask.py deploy/llm-client/load.py ./
# Prints which models the server has loaded. Override the command to ask a model: python ask.py "question"
CMD ["python", "-m", "tiny_agent.components.client.check"]
