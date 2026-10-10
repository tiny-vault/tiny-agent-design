# llama.cpp models

Everything that decides *which models llama.cpp serves and how* lives here. The Docker setup (`../../docker-compose.yaml`) only mounts these files.

| Path | What it is |
|---|---|
| `models.ini` | One section per model: the name clients use, where llama.cpp downloads it from (Hugging Face), and per-model settings (`ctx-size`, `n-gpu-layers`, `jinja`). The `[*]` section applies to all. |
| `cache/` | The downloaded models. Created by `node execute.mjs up`, filled by llama.cpp, git-ignored, and never removed by `prune`. |

There is no separate download step and no Ollama: the first time a model is loaded, llama.cpp downloads it from Hugging Face into `cache/`. Later loads and restarts use the cached file.

| Model name | Hugging Face source | Size |
|---|---|---|
| `gemma4-e4b` | `google/gemma-4-E4B-it-qat-q4_0-gguf`, file `gemma-4-E4B_q4_0-it.gguf` (Google's quantisation-aware 4-bit build) | about 5 GB |
| `qwen3.5-4b` | `unsloth/Qwen3.5-4B-GGUF`, quantisation `Q4_K_M` | about 2.7 GB |

## How long it takes

Measured on the development laptop (6 GB GPU).

| What | Time |
|---|---|
| First load of `gemma4-e4b` (includes the download) | 15 to 20 minutes |
| First load of `qwen3.5-4b` (includes the download) | 6 minutes |
| Switching to `qwen3.5-4b`, from the cache | 35 s |
| Switching to `gemma4-e4b`, from the cache | 61 to 69 s |

Only one model fits in GPU memory, so each switch is a full reload.

## Download the models now

```powershell
node execute.mjs download            # every model in models.ini, one after another
node execute.mjs download gemma4-e4b # just one
```

This loads each model once, which makes llama.cpp download it, and shows the progress. Otherwise the download simply happens the first time you pick the model in the app or run `node execute.mjs load <model>`; the app shows "Loading..." until it finishes. The backend allows up to 30 minutes for a load (`LLAMA_LOAD_TIMEOUT`, in seconds).

## Add a model

1. Find a GGUF repo on Hugging Face, for example `unsloth/Llama-3.2-3B-Instruct-GGUF`. Check it is not gated, or that you have a token (see below).
2. Add a section to `models.ini`:

   ```ini
   [llama3-2-3b]
   hf-repo = unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
   ```

   The part after the colon picks the quantisation. If the repo has several unrelated files (for example a vision projector next to the model), name the file instead with `hf-file = <file>.gguf`, as `gemma4-e4b` does.
3. Restart llama.cpp so it reads the file: `node execute.mjs restart llama`.
4. Load it: `node execute.mjs load llama3-2-3b`.

The section name (`llama3-2-3b`) is what the app and the API call the model.

## Things to know

- **Gated models** (some Hugging Face repos need you to accept a license) need an access token. The two models above are open, so none is needed today.
- **Offline use:** after the first download everything is cached, so the stack works without internet. To stop llama.cpp ever reaching out, set `LLAMA_ARG_OFFLINE: "1"` on the `llama` service.
- One model is kept loaded at a time (`LLAMA_MODELS_MAX=1`); loading another unloads the first. The app loads models through the backend and waits until they report `loaded`.
- The dev laptop has 6 GB of VRAM. If a model plus its context does not fit, lower `n-gpu-layers`; the remaining layers run on the CPU.
- Deleting `cache/` (or a model inside it) is safe: the model is downloaded again next time it is loaded.
