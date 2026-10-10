import { SUGGESTIONS } from "../lib/constants.js";
import { Btn } from "./ui.jsx";

export default function EmptyState({ model, onAsk }) {
  return (
    <div className="m-auto flex flex-col items-center gap-3 text-center">
      <h1 className="m-0 font-brand text-display font-semibold tracking-tight">Ask the agent</h1>
      {model ? (
        <>
          <p className="m-0 text-small text-ink-2">
            Answers come from <b>{model}</b> and a small TinyAgent knowledge base.
          </p>
          <div className="mt-3 flex flex-wrap justify-center gap-2">
            {SUGGESTIONS.map((q) => (
              <Btn key={q} onClick={() => onAsk(q)}>{q}</Btn>
            ))}
          </div>
        </>
      ) : (
        <p className="m-0 text-small text-ink-2">
          No models installed. Run <code>ollama pull &lt;model&gt;</code>, then reload.
        </p>
      )}
    </div>
  );
}
