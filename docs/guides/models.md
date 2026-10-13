# Models

TinyAgent is built step by step against local models. They are chosen as a pair so the difference between a plain language model and an agentic one can be studied directly.

| Role | Model | Native reasoning | Native tool calling |
|---|---|---|---|
| Baseline | Gemma 3 4B | No | No |
| Agentic | Gemma 4 E4B (QAT) | Yes | Yes |

The baseline cannot reason or call tools natively, so any such behaviour must come from the harness (prompting, parsing, the loop). The agentic model is trained for reasoning and tool calling, so the harness can lean on it.

## Where the models come from

There are two ways to run them, and the project works with either.

| | Docker stack (the default) | Ollama (optional) |
|---|---|---|
| Server | llama.cpp in router mode | Ollama |
| Gets the model from | Hugging Face, downloaded by llama.cpp the first time the model is loaded | the Ollama registry, with `ollama pull` |
| Address | `http://localhost:3200/v1` | `http://localhost:11434/v1` |
| Model names | the names in `models.ini`: `gemma4-e4b`, `qwen3.5-4b` | Ollama tags such as `gemma4:e4b-it-qat` |
| Needed by the chat app | yes | no, it was removed |

The Docker stack serves these models (`LLAMA_MODELS_MAX=1`, so loading `embeddinggemma-300m` unloads the chat model):

| Name | Hugging Face source | On disk |
|---|---|---|
| `gemma4-e4b` | `google/gemma-4-E4B-it-qat-q4_0-gguf` (Google's quantisation-aware 4-bit build, the same QAT build as Ollama's `gemma4:e4b-it-qat`) | 5.9 GB |
| `qwen3.5-4b` | `unsloth/Qwen3.5-4B-GGUF`, `Q4_K_M` | 3.4 GB |
| `embeddinggemma-300m` | `ggml-org/embeddinggemma-300M-GGUF` (an embedding model: use `/v1/embeddings`, not chat) | 0.3 GB |

The sizes include a small vision file in each repository. Downloads go into `experimentation/chat-app/models/cache`. The llm client's own model server (`node execute.mjs up llm-client`) serves the same two models from a copy of `models.ini` and can reuse the same download folder. See {doc}`chat-app` for the measured download and switching times.

## Using Ollama instead

The `LLM` client works with an Ollama server by passing its address. This is how to get the Gemma 3 baseline, which the Docker stack does not serve:

```powershell
ollama pull gemma3:4b
ollama pull gemma4:e4b-it-qat
ollama list
```

```python
from tiny_agent.llm_client.llm import LLM

llm = LLM(model="gemma3:4b", base_url="http://localhost:11434/v1")
```

## Why these sizes

The development machine has an RTX 3060 Laptop GPU with 6 GB of VRAM.

* Gemma 3 4B fits entirely in VRAM.
* Gemma 4 E4B (QAT) is almost the size of the VRAM: measured with llama.cpp, 5.3 of 6 GB are used with it loaded, so the second model cannot be loaded at the same time. QAT (quantization-aware training) is the lowest-loss 4-bit build.
* Both models are 4B-class, so differences in behaviour come from training, not size.

## Alternatives

| Goal | Without | With | Notes |
|---|---|---|---|
| Both fully on GPU | `gemma3:4b` | `gemma4:e2b-it-qat` (4.3 GB) | Faster, weaker agentic model |
| Larger pair | `gemma3:12b` (8.2 GB) | `gemma4:e4b` (6.6 GB) | Neither fits in 6 GB, so both spill to CPU |
| Effect of reasoning only | `qwen3:4b-instruct-2507-q4_K_M` (2.5 GB) | `qwen3:4b-thinking-2507-q4_K_M` (2.5 GB) | Both support tools; only reasoning differs |

The alternatives are Ollama tags and sizes (the default Q4 builds on the Ollama library when this was written). To use one in the Docker stack, find its GGUF repository on Hugging Face and add a section with `hf-repo` to `models.ini`.
