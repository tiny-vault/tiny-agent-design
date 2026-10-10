# Models

TinyAgent is built step by step against two local models served by [Ollama](https://ollama.com). They are chosen as a pair so the difference between a plain language model and an agentic one can be studied directly.

| Role | Model | Ollama tag | Size | Native reasoning | Native tool calling |
|---|---|---|---|---|---|
| Baseline | Gemma 3 4B | `gemma3:4b` | about 3.3 GB | No | No |
| Agentic | Gemma 4 E4B (QAT) | `gemma4:e4b-it-qat` | 6.1 GB | Yes | Yes |

The baseline cannot reason or call tools natively, so any such behaviour must come from the harness (prompting, parsing, the loop). The agentic model is trained for reasoning and tool calling, so the harness can lean on it.

## Pull the models

```powershell
ollama pull gemma3:4b
ollama pull gemma4:e4b-it-qat
ollama list
```

## Why these sizes

The development machine has an RTX 3060 Laptop GPU with 6 GB of VRAM.

* `gemma3:4b` fits entirely in VRAM.
* `gemma4:e4b-it-qat` is slightly larger than VRAM, so a small part runs on the CPU. QAT (quantization-aware training) is the lowest-loss 4-bit build.
* Both models are 4B-class, so differences in behaviour come from training, not size. Use `ollama ps` to see the GPU/CPU split of a loaded model.

## Alternatives

| Goal | Without | With | Notes |
|---|---|---|---|
| Both fully on GPU | `gemma3:4b` | `gemma4:e2b-it-qat` (4.3 GB) | Faster, weaker agentic model |
| Larger pair | `gemma3:12b` (8.2 GB) | `gemma4:e4b` (6.6 GB) | Neither fits in 6 GB, so both spill to CPU |
| Effect of reasoning only | `qwen3:4b-instruct-2507-q4_K_M` (2.5 GB) | `qwen3:4b-thinking-2507-q4_K_M` (2.5 GB) | Both support tools; only reasoning differs |

Sizes are the default Q4 builds listed on the Ollama library at the time of writing.
