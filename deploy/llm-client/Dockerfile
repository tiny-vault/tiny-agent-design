# The LLM client alone: code/llm_client (OpenAI-compatible chat client plus the loaded-model check),
# and ask.py, a small script that sends one question to a model with it. The client is self-contained
# and standard-library only, so nothing is installed and nothing else from the repo is needed.
# The build context is the repo root; Dockerfile.dockerignore keeps it to these (plus load.py, which waits for a model to load).
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/code
COPY code/llm_client code/llm_client
COPY deploy/llm-client/ask.py deploy/llm-client/load.py ./
# Prints which models the server has loaded. Override the command to ask a model: python ask.py "question"
CMD ["python", "-m", "llm_client.check"]
