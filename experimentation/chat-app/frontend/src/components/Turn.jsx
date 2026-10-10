import { useRef, useState } from "react";
import { answeredWithoutSources, citedSources, splitCitations } from "../lib/citations.js";
import { Caption, Dot, Panel, Row, stepTone } from "./ui.jsx";

const Secondary = ({ children }) => <p className="m-0 text-small text-ink-2">{children}</p>;

/* The answer text with [n] marks turned into small buttons. Marks whose source
   was never retrieved stay as plain text. */
function Answer({ turn, onCite }) {
  return (
    <div className="break-words whitespace-pre-wrap text-title">
      {splitCitations(turn.content).map((p, i) =>
        p.text !== undefined ? (
          <span key={i}>{p.text}</span>
        ) : (
          p.cite.map((n, j) =>
            turn.sources.some((s) => s.n === n) ? (
              <button key={`${i}-${j}`} type="button" onClick={() => onCite(n)}
                className="mx-px inline-flex h-[18px] min-w-[18px] cursor-pointer items-center justify-center rounded-cell
                  border-0 bg-surface-3 px-1 align-[2px] text-fine font-medium tabular-nums
                  transition-colors duration-[260ms] hover:bg-line-strong hover:duration-[120ms]">
                {n}
              </button>
            ) : (
              <span key={`${i}-${j}`}>[{n}]</span>
            ),
          )
        ),
      )}
    </div>
  );
}

export default function Turn({ turn }) {
  const [openSource, setOpenSource] = useState(0);
  const root = useRef(null);
  const sources = citedSources(turn);

  const showSource = (n) => {
    setOpenSource(n);
    requestAnimationFrame(() =>
      root.current?.querySelector(`[data-src="${n}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" }),
    );
  };

  const thinking = turn.working && !turn.content;

  return (
    <div ref={root} className="flex flex-col gap-3">
      <div className="max-w-[85%] self-end break-words whitespace-pre-wrap rounded-panel bg-bubble px-4 py-2 text-title">
        {turn.question}
      </div>

      {turn.steps.length > 0 && (
        <Panel>
          <Caption>{turn.steps.length === 1 ? "1 step" : `${turn.steps.length} steps`}</Caption>
          {turn.steps.map((s, i) => (
            <Row key={i} dot={<Dot tone={stepTone(s.status)} />} label={s.label} detail={s.detail}>
              <pre className="m-0 font-mono text-fine whitespace-pre-wrap">{s.raw}</pre>
            </Row>
          ))}
        </Panel>
      )}

      {turn.thinking && (
        <Panel>
          <Row
            dot={<Dot tone="ok" running={thinking} />}
            label={thinking ? "Thinking…" : "Thought"}
            detail={`${turn.thinking.length} chars`}
            detailClass="tabular-nums"
          >
            <div className="whitespace-pre-wrap text-ink-2">{turn.thinking}</div>
          </Row>
        </Panel>
      )}

      {turn.content && <Answer turn={turn} onCite={showSource} />}

      {turn.choice && (
        <Panel>
          <Row dot={<Dot tone="ok" />} label="choices[0]" detail="raw response">
            <pre className="m-0 font-mono text-fine whitespace-pre-wrap">{turn.choice}</pre>
          </Row>
        </Panel>
      )}

      {turn.working && !turn.content && !turn.thinking && !turn.steps.length && <Secondary>Working…</Secondary>}
      {turn.note && <Secondary>{turn.note}</Secondary>}
      {turn.error && <p className="m-0 text-small text-danger">{turn.error}</p>}
      {turn.stopped && <Secondary>Stopped.</Secondary>}

      {sources.length > 0 && (
        <Panel>
          <Caption>Sources</Caption>
          {sources.map((src) => (
            <Row
              key={src.n}
              data-src={src.n}
              highlight={openSource === src.n}
              open={openSource === src.n}
              onToggle={() => setOpenSource(openSource === src.n ? 0 : src.n)}
              dot={
                <span className="inline-flex rounded-cell border border-line-strong px-2 text-fine text-ink-2 tabular-nums">
                  {src.n}
                </span>
              }
              label={src.title}
            >
              {src.text}
            </Row>
          ))}
        </Panel>
      )}
      {answeredWithoutSources(turn) && <Secondary>Answered without citing any source.</Secondary>}
    </div>
  );
}
