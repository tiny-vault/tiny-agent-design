# Quickstart

## The harness

```python
from tiny_agent.shared.tiny_agent import TinyAgent

agent = TinyAgent()
print(agent.run("Hello, agent!"))
```

This prints `Received: Hello, agent!`. No model is called: `run` passes the task to `_step`, which currently echoes it back.

See {py:class}`tiny_agent.shared.tiny_agent.TinyAgent` for the full API.

## Talking to a model

Start the chat app stack once (it serves the models), then use the LLM client from code:

```powershell
node execute.mjs up
```

```python
from llm_client.llm import LLM

llm = LLM(model="gemma4-e4b")   # the Docker gateway is the default address
reply = llm.generate([{"role": "user", "content": "Hi! How's life?"}])
print(reply.content)
print(reply.metadata)           # model, prompt_tokens, completion_tokens
```

If the model is not the one that is loaded, `generate` stops with a message that says how to load it, or, when the name is not offered at all, uses the loaded model and warns. See {py:class}`llm_client.llm.LLM`.

## The terminal chat

```powershell
python -m pip install -e .
tiny-agent                          # interactive chat
tiny-agent "Why is the sky blue?"   # one question, then exit
```

Inside the chat, `/help` lists the commands (`/models`, `/model NAME`, `/think on|off`, `/clear`, `/exit`). The first time a model is used, llama.cpp downloads it, which takes minutes; later it is a switch of 35 to 70 seconds. Details are in {doc}`../guides/chat-app`.
