import { useEffect, useRef, useState } from "react";
import { Btn } from "./ui.jsx";

export default function Composer({ busy, disabled, onSend, onStop }) {
  const [text, setText] = useState("");
  const box = useRef(null);

  // Grow with the content, up to the CSS max-height.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = el.scrollHeight + "px";
  }, [text]);

  const submit = () => {
    const value = text.trim();
    if (!value || busy || disabled) return;
    setText("");
    onSend(value);
  };

  return (
    <form className="pt-3 pb-6" onSubmit={(e) => { e.preventDefault(); submit(); }}>
      <div className="flex items-end gap-2 rounded-panel border border-line-strong bg-raise p-2 focus-within:border-ring">
        <textarea
          ref={box}
          rows={1}
          value={text}
          placeholder="Message the agent"
          aria-label="Message"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          className="max-h-[8em] flex-1 resize-none border-0 bg-transparent px-3 py-1 text-title outline-0
            placeholder:text-ink-2"
        />
        {busy ? (
          <Btn variant="primary" onClick={onStop}>Stop</Btn>
        ) : (
          <Btn variant="primary" type="submit" disabled={!text.trim() || disabled}>Send</Btn>
        )}
      </div>
    </form>
  );
}
