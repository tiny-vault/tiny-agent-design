import { useEffect, useLayoutEffect, useRef } from "react";
import { followsBottom } from "../lib/citations.js";
import EmptyState from "./EmptyState.jsx";
import Turn from "./Turn.jsx";

/* The scrolling conversation. It follows a streaming answer only while the
   reader is within 48px of the bottom, so scrolling up to read is not
   interrupted. Sending a new message always jumps back to the bottom. */
export default function ChatLog({ turns, model, ready, onAsk }) {
  const log = useRef(null);
  const follow = useRef(true);

  useEffect(() => {
    follow.current = true;
  }, [turns.length]);

  useLayoutEffect(() => {
    if (follow.current && log.current) log.current.scrollTop = log.current.scrollHeight;
  }, [turns]);

  return (
    <div
      ref={log}
      onScroll={() => { follow.current = followsBottom(log.current); }}
      className="flex flex-1 flex-col gap-6 overflow-y-auto pt-4 pb-8"
    >
      {turns.length === 0 && <EmptyState model={model} ready={ready} onAsk={onAsk} />}
      {turns.map((t) => (
        <Turn key={t.id} turn={t} />
      ))}
    </div>
  );
}
