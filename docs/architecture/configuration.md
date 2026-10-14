# Configuration and logs

Everything the Python code can be configured with lives in one class, {py:class}`tiny_agent.config.settings.Settings`. Code asks `Settings.from_env()`, never `os.environ`, so a setting has one name, one default and one place to read it.

## Where a value comes from

A setting's variable is `TINYAGENT_` plus the field name in capitals: `llm_url` is `TINYAGENT_LLM_URL`. `Settings.from_env()` reads, in this order:

1. the real environment (always wins),
2. the `.env` file: `tiny_agent/.env`, or the file named by `TINYAGENT_ENV_FILE`,
3. the default.

Copy `tiny_agent/.env.example` to `.env` (it is git-ignored) and fill in only what you need. With an empty `.env` every default applies.

Values are parsed for you: `1`, `true`, `yes` and `on` are true, an empty variable counts as not set, and a wrong type is an error that names the setting. To configure something in code, for example in a test, pass a `Settings(...)` to the function instead.

## The settings

| Setting | What it does | Default |
|---|---|---|
| `TINYAGENT_MODEL` | The chat model to use. | whichever is loaded |
| `TINYAGENT_LLM_URL` | The server's API root, ending in `/v1`. | the Docker gateway, `http://127.0.0.1:3200/v1` |
| `TINYAGENT_API_KEY_FILE`, `TINYAGENT_API_KEY` | The key, for a server that needs one. Prefer the file form so the key stays out of `.env`. | none (the Docker stack needs none) |
| `TINYAGENT_THINK` | Let the model reason before it answers (slower). | off |
| `TINYAGENT_SYSTEM` | A system message that sets the model's role. | none |
| `TINYAGENT_TRIM_MEMORY`, `TINYAGENT_SUMMARY_MEMORY`, `TINYAGENT_RAG_MEMORY`, `TINYAGENT_BANK_MEMORY` | Which memory the demo uses. If several are on: bank, then rag, then summary, then trim. See {doc}`memory`. | the plain `Memory` |
| `TINYAGENT_EMBEDDING_MODEL`, `TINYAGENT_EMBEDDING_URL` | The embedding model for the `rag` and `bank` memories, and where it is served. | `embeddinggemma-300m` at `http://127.0.0.1:3201/v1` |
| `TINYAGENT_TOOLS`, `TINYAGENT_NATIVE_TOOLS` | Give the agent the built-in tools, prompted or learned; native wins. See {doc}`tools`. | no tools |
| `TINYAGENT_LOG_DIR`, `TINYAGENT_LOG_LEVEL` | Where the log files go, and `DEBUG`, `INFO`, `WARNING` or `ERROR`. | the chat app's `logs/` folder, `INFO` |
| `TINYAGENT_ENV_FILE` | Read this file instead of `tiny_agent/.env`. | |

```{note}
The `LLM` client itself reads nothing from the environment: it takes its settings as arguments. `main.py`, `cli.py` and the model check read `Settings` and pass the values in.
```

## The API key

`Settings.resolve_api_key()` returns the key from `api_key_file` (its first line that is not blank or a comment) or else from `api_key`, or `None`. The Docker stack's backend holds its own key, so locally you usually need neither. A key is never printed or logged.

## Logs

The two terminal tools write log files: `cli.log` for `tiny-agent` and `main.log` for the demo, under `experimentation/chat-app/logs/` (or `TINYAGENT_LOG_DIR`). They rotate at 1 MB and keep three old copies.

The library code stays silent: the model client, memories and tools never log. The tools log what those modules return or raise, naming the settings used and never secret values. A problem with logging, such as a read-only disk, never stops a tool. The containers' logs and these files all end up in OpenObserve, at <http://localhost:5080>.
