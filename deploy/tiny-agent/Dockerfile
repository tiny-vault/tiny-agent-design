# The TinyAgent harness and its terminal chat. tiny_agent/ holds the harness and the model client (client/) it
# uses); its main.py and cli.py are the two scripts. Its libraries (pydantic-settings, httpx, click) are
# installed. The build context is the repo root; Dockerfile.dockerignore keeps it to tiny_agent/.
FROM python:3.12-slim
WORKDIR /app
# /app goes on the import path instead of installing the package; only its libraries are installed.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
RUN pip install --no-cache-dir pydantic-settings httpx click
COPY tiny_agent tiny_agent
# The interactive chat. The memory demo is `python -m tiny_agent.main`.
CMD ["python", "-m", "tiny_agent"]
