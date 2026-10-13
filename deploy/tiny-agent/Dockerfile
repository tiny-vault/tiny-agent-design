# The TinyAgent harness and its terminal chat. code/ holds tiny_agent, llm_client (the model client it
# uses) and the two scripts beside them, cli.py and main.py. Standard library only, so nothing is
# installed. The build context is the repo root; Dockerfile.dockerignore keeps it to code/.
FROM python:3.12-slim
WORKDIR /app
# code/ goes on the import path instead of being installed.
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/code
COPY code code
# The interactive chat. The memory demo is `python code/main.py`.
CMD ["python", "-m", "tiny_agent"]
