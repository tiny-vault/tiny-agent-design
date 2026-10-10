# llama.cpp models

Everything that decides *which models llama.cpp serves and how* lives here. The Docker setup in `../deploy` only mounts these files.

| File | What it does |
|---|---|
| `models.ini` | One section per model: the name clients use, the GGUF file, and per-model settings (`ctx-size`, `n-gpu-layers`, `jinja`). The `[*]` section applies to all. |
| `pull-models.sh` | Runs once at startup inside the `pull-models` container. Downloads any model in `PULL_MODELS` that Ollama does not have yet, then exits. |

llama.cpp reads the GGUF files Ollama already downloaded (blobs named `sha256-<digest>`), so nothing is copied.

## Add a model

1. Add its Ollama name to `PULL_MODELS` (in `../deploy/.env`, or the default in `../deploy/docker-compose.yaml`), for example `llama3.2:3b`.
2. Start the stack once so it is downloaded: `node ../execute.mjs up`.
3. Find its digest in `<ollama models>/manifests/registry.ollama.ai/library/<name>/<tag>`. It is the layer whose `mediaType` ends in `image.model`.
4. Add a section to `models.ini`:

   ```ini
   [llama3-2-3b]
   model = /models/sha256-<digest>
   ```

5. Restart: `node ../execute.mjs restart llama`.

The section name (`llama3-2-3b`) is what the app and API call the model.

## Things to know

- If the registry publishes a new build of a tag, the digest changes and the path in `models.ini` must be updated.
- One model is kept loaded at a time (`LLAMA_MODELS_MAX=1`); loading another unloads the first. The app loads models through the backend and waits until they report `loaded`.
- The dev laptop has 6 GB of VRAM. If a model plus its context does not fit, lower `n-gpu-layers`; the remaining layers run on the CPU.
