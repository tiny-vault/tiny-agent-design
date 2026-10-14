# Quickstart

## The harness

Start the stack first (`node execute.mjs up`), because the agent calls a model:

```python
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.memory.memory import Memory
from tiny_agent.agent import TinyAgent

agent = TinyAgent(LLM(model="gemma4-e4b"), Memory())
print(agent.run("Hello, agent!"))
```

This prints the model's reply, which varies between runs. `run` stores the task in `memory`, and `_step` sends the whole memory to the model and stores the reply, so a second `run` on the same agent sees the first exchange. Tools, planning and the multi-step loop are still placeholders.

See {py:class}`tiny_agent.agent.TinyAgent` for the full API.

## Talking to a model

Start a model server once, then use the LLM client from code. Either start the whole chat app (it serves the models and the web UI), or only the llm client's own model server; both listen on port 3200, so run one of them at a time:

```powershell
node execute.mjs up                                  # the chat app
node execute.mjs up llm-client                       # or: only a model server
node execute.mjs llm-client load qwen3.5-4b          # and load a model into it
```

```python
from tiny_agent.components.client.llm import LLM

llm = LLM(model="gemma4-e4b")   # the Docker gateway is the default address
reply = llm.generate([{"role": "user", "content": "Hi! How's life?"}])
print(reply.content)
print(reply.metadata)           # model, prompt_tokens, completion_tokens
```

If the model is not the one that is loaded, `generate` stops with a message that says how to load it, or, when the name is not offered at all, uses the loaded model and warns. See {py:class}`tiny_agent.components.client.llm.LLM`.

## The terminal chat

```powershell
python -m pip install -e .
tiny-agent                          # interactive chat
tiny-agent "Why is the sky blue?"   # one question, then exit
```

Inside the chat, `/help` lists the commands (`/models`, `/model NAME`, `/think on|off`, `/clear`, `/exit`). The first time a model is used, llama.cpp downloads it, which takes minutes; later it is a switch of 35 to 70 seconds. Details are in {doc}`../guides/chat-app`.
