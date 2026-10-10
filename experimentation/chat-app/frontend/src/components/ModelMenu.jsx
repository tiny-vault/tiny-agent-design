import { useEffect, useRef, useState } from "react";
import { Btn, Dot, Icon } from "./ui.jsx";

/* A themed dropdown of installed models, marking the ones loaded in memory. */
export default function ModelMenu({ models, loaded, model, loading, disabled, onOpen, onPick, onUnload }) {
  const [open, setOpen] = useState(false);
  const root = useRef(null);

  useEffect(() => {
    if (!open) return;
    const away = (e) => !root.current?.contains(e.target) && setOpen(false);
    const esc = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  const dotFor = (m) => <Dot tone={loaded.has(m) ? "ok" : undefined} running={loading === m} />;

  return (
    <div ref={root} className="relative">
      <Btn
        className="max-w-[220px]"
        disabled={disabled || !!loading}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => {
          if (!open) onOpen();
          setOpen(!open);
        }}
      >
        {model && <Dot tone={loaded.has(model) ? "ok" : undefined} running={!!loading} />}
        <span className="truncate">{loading ? `Loading ${loading}…` : model || "No model"}</span>
        <Icon name="chevron" />
      </Btn>

      {open && (
        <div role="listbox"
          className="absolute right-0 top-[calc(100%+4px)] z-10 min-w-[220px] rounded-overlay border border-line
            bg-glass-strong p-1 backdrop-blur-xl backdrop-saturate-[.3]">
          {models.map((m) => (
            <div key={m} className="flex items-center">
              <button
                type="button"
                role="option"
                aria-selected={m === model}
                onClick={() => { setOpen(false); onPick(m); }}
                className="flex w-full cursor-pointer items-center gap-3 rounded-cell bg-transparent px-3 py-2 text-left
                  transition-colors duration-[260ms] hover:bg-surface-2 hover:duration-[120ms]"
              >
                {dotFor(m)}
                <span className={`flex-1 break-words ${m === model ? "font-semibold" : ""}`}>{m}</span>
                <span className="text-fine text-ink-2">{loading === m ? "Loading…" : loaded.has(m) ? "Loaded" : ""}</span>
              </button>
              {loaded.has(m) && loading !== m && (
                <button
                  type="button"
                  onClick={() => onUnload(m)}
                  aria-label="Unload model"
                  title="Unload from memory"
                  className="flex size-7 flex-none cursor-pointer items-center justify-center rounded-cell bg-transparent
                    p-0 text-ink-2 transition-colors duration-[260ms] hover:bg-surface-2 hover:duration-[120ms]"
                >
                  <Icon name="close" size={14} />
                </button>
              )}
            </div>
          ))}
          {!models.length && (
            <p className="m-0 px-3 py-2 text-ink-2">
              No models installed. Run <code>ollama pull &lt;model&gt;</code>.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
