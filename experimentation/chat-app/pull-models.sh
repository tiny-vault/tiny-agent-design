#!/bin/sh
# Runs inside the ollama image as a one-shot job. Starts a temporary Ollama
# server, pulls every model in $PULL_MODELS that is not already in the shared
# models folder, then exits so llama-server can start.
set -eu

ollama serve >/tmp/ollama.log 2>&1 &
server=$!

# Wait for the API to answer.
i=0
until ollama list >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -gt 60 ]; then
    echo "Ollama did not start:" >&2
    cat /tmp/ollama.log >&2
    exit 1
  fi
  sleep 1
done

for model in $PULL_MODELS; do
  if ollama show "$model" >/dev/null 2>&1; then
    echo "found    $model"
  else
    echo "missing  $model, downloading"
    ollama pull "$model"
  fi
done

kill "$server" 2>/dev/null || true
echo "All models are present."
