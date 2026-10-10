# Quickstart

```python
from tiny_agent import TinyAgent

agent = TinyAgent()
print(agent.run("Hello, agent!"))
```

This prints `Received: Hello, agent!`. No model is called yet: `run` passes
the task to `_step`, which currently echoes it back.

See {py:class}`tiny_agent.TinyAgent` for the full API.
