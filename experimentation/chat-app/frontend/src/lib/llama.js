import { API_BASE, NOTIFY_MS } from "./constants.js";

/* The message inside the backend's {"error": {"message"}} body, else the raw text. */
async function failure(res) {
  const text = await res.text();
  try {
    return JSON.parse(text).error?.message || text;
  } catch {
    return text || `HTTP ${res.status}`;
  }
}

/* The models the router offers: [{ id, status }], status loaded | loading | unloaded. */
export async function listModels() {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) throw new Error(await failure(res));
  return (await res.json()).models;
}

/* Ask the router to load a model. Resolves only once it is loaded; rejects with
   the backend's message if the load fails or times out. */
export async function loadModel(model) {
  const res = await fetch(`${API_BASE}/api/models/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  if (!res.ok) throw new Error(await failure(res));
}

/* One streamed chat call. `onUpdate({ content, thinking })` fires at most every
   NOTIFY_MS. Returns { content, calls, choice }, where choice is the streamed
   pieces joined back into the shape of a non-streaming choices[0]. */
export async function streamChat({ model, messages, tools, signal, onUpdate }) {
  const body = { model, stream: true, messages };
  if (tools) body.tools = tools;
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const err = new Error(await failure(res));
    err.status = res.status;
    err.body = err.message;
    throw err;
  }

  let content = "";
  let thinking = "";
  let finish = null;
  let timer = null;
  const calls = [];
  const paint = () => {
    timer = null;
    onUpdate({ content, thinking });
  };

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (!data || data === "[DONE]") continue;
        const chunk = JSON.parse(data).choices?.[0] ?? {};
        if (chunk.finish_reason) finish = chunk.finish_reason;
        const delta = chunk.delta ?? {};
        thinking += delta.reasoning_content || delta.reasoning || "";
        if (delta.content) content += delta.content;
        for (const tc of delta.tool_calls ?? []) {
          const i = tc.index ?? 0;
          calls[i] ??= { id: "", name: "", args: "" };
          calls[i].id ||= tc.id || "";
          calls[i].name ||= tc.function?.name || "";
          const a = tc.function?.arguments;
          if (a !== undefined) calls[i].args += typeof a === "string" ? a : JSON.stringify(a);
        }
        if (!timer) timer = setTimeout(paint, NOTIFY_MS);
      }
    }
  } finally {
    clearTimeout(timer);
    paint();
  }

  calls.forEach((c, i) => {
    if (c) c.id ||= `call_${Date.now()}_${i}`;
  });
  const done = calls.filter(Boolean);
  const message = { role: "assistant", content };
  if (thinking) message.reasoning = thinking;
  if (done.length) {
    message.tool_calls = done.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: c.args } }));
  }
  return { content, calls: done, choice: { index: 0, message, finish_reason: finish } };
}
