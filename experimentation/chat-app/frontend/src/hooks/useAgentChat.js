import { useCallback, useRef, useState } from "react";
import { MAX_ROUNDS, SYSTEM_PROMPT, TOOLS } from "../lib/constants.js";
import { streamChat } from "../lib/ollama.js";
import { runTool } from "../lib/tools.js";

const newTurn = (id, question) => ({
  id, question, content: "", thinking: "", steps: [], sources: [],
  note: "", error: "", stopped: false, working: true, choice: "",
});

const freshConvo = () => [{ role: "system", content: SYSTEM_PROMPT }];

/* The agent loop: stream a reply, run any tool calls, feed the results back,
   repeat until the model answers without calling a tool (at most MAX_ROUNDS).
   `turns` is what the UI renders; `convo` is the API history and lives in a ref
   because it is never drawn. */
export function useAgentChat({ model, kb, onDone }) {
  const [turns, setTurns] = useState([]);
  const [busy, setBusy] = useState(false);
  const convo = useRef(freshConvo());
  const controller = useRef(null);
  const noTools = useRef(new Set()); // models that rejected tool calling
  const nextId = useRef(0);

  const send = useCallback(
    async (text) => {
      if (!text || !model) return;
      const id = ++nextId.current;
      const patch = (p) => setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...p } : t)));
      const history = convo.current;
      const base = history.length;
      const steps = [];
      const sources = [];
      let content = "";

      setTurns((ts) => [...ts, newTurn(id, text)]);
      history.push({ role: "user", content: text });
      setBusy(true);
      controller.current = new AbortController();
      const signal = controller.current.signal;
      const onUpdate = (u) => { content = u.content; patch(u); };
      const call = (useTools) =>
        streamChat({ model, messages: history, tools: useTools ? TOOLS : null, signal, onUpdate });

      try {
        for (let round = 0; round < MAX_ROUNDS; round++) {
          content = "";
          patch({ content: "", thinking: "" });
          const useTools = !noTools.current.has(model);
          let reply;
          try {
            reply = await call(useTools);
          } catch (e) {
            // Models without native tool calling are rejected with a 400 that mentions tools.
            if (useTools && e.status === 400 && /tool/i.test(e.body || "")) {
              noTools.current.add(model);
              patch({ note: `${model} has no native tool calling, so it answers without tools.` });
              reply = await call(false);
            } else throw e;
          }
          patch({ choice: JSON.stringify(reply.choice, null, 2) }); // like print(result["choices"][0])

          if (!reply.calls.length) {
            history.push({ role: "assistant", content: reply.content });
            break;
          }
          history.push({
            role: "assistant",
            content: reply.content,
            tool_calls: reply.calls.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: c.args } })),
          });

          for (const c of reply.calls) {
            let args;
            try { args = c.args ? JSON.parse(c.args) : {}; } catch { args = null; }
            const step = { tool: c.name, label: c.name, detail: "", status: "ok", raw: "" };
            let result;
            if (args === null) {
              step.status = "invalid";
              step.detail = "Could not read arguments";
              result = JSON.stringify({ error: "Arguments were not valid JSON" });
            } else {
              step.label = c.name + (args.query ? ` “${args.query}”` : "");
              const out = runTool(c.name, args, kb, sources);
              result = out.result;
              step.detail = out.detail;
              step.status = out.status;
            }
            step.raw = `arguments: ${c.args || "{}"}\n\nresult: ${result}`;
            steps.push(step);
            history.push({ role: "tool", tool_call_id: c.id, content: result });
            patch({ steps: [...steps], sources: [...sources] });
          }
          if (round === MAX_ROUNDS - 1) patch({ error: "Stopped after too many tool rounds." });
        }
      } catch (e) {
        // Drop the half-finished exchange from the API history, keep what was said.
        history.length = base;
        history.push({ role: "user", content: text });
        if (content) history.push({ role: "assistant", content });
        if (e.name === "AbortError") patch({ stopped: true });
        else patch({ error: e.message });
      } finally {
        patch({ working: false });
        setBusy(false);
        controller.current = null;
        onDone?.();
      }
    },
    [model, kb, onDone],
  );

  const stop = useCallback(() => controller.current?.abort(), []);
  const reset = useCallback(() => {
    setTurns([]);
    convo.current = freshConvo();
  }, []);

  return { turns, busy, send, stop, reset };
}
