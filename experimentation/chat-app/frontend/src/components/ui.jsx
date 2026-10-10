import { useState } from "react";

const ICONS = {
  chevron: <path d="m6 9 6 6 6-6" />,
  close: <path d="M18 6 6 18M6 6l12 12" />,
  plus: <path d="M12 5v14M5 12h14" />,
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </>
  ),
};

export function Icon({ name, size = 16 }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {ICONS[name]}
    </svg>
  );
}

const cx = (...parts) => parts.filter(Boolean).join(" ");

/* Controls own their states: hover in 120ms, out 260ms; disabled dims. */
export function Btn({ variant, icon, className, ...props }) {
  return (
    <button
      type="button"
      {...props}
      className={cx(
        "inline-flex h-8 cursor-pointer items-center justify-center gap-2 rounded-control border font-medium",
        "transition-colors duration-[260ms] hover:duration-[120ms] disabled:pointer-events-none disabled:opacity-50",
        variant === "primary"
          ? "border-transparent bg-primary font-semibold text-on-primary hover:opacity-90"
          : "border-line-strong bg-transparent hover:bg-surface-2",
        icon ? "w-8 px-0" : "px-3",
        className,
      )}
    />
  );
}

/* A status is a coloured dot and plain words. */
export function Dot({ tone, running }) {
  const color = { ok: "bg-ok", warn: "bg-warn", danger: "bg-danger" }[tone] || "bg-ink-2";
  return <span className={cx("size-2 flex-none rounded-full", color, running && "animate-pulse-dot")} />;
}

export const stepTone = (status) =>
  status === "ok" ? "ok" : status === "failed" ? "danger" : "warn";

/* One bordered panel; its children are Rows or a Caption. */
export function Panel({ children }) {
  return <div className="overflow-hidden rounded-panel border border-line bg-raise">{children}</div>;
}

export function Caption({ children }) {
  return <div className="border-b border-line px-4 py-2 text-fine text-ink-2">{children}</div>;
}

/* A row with a header button that opens a body. Controlled when `open` is given. */
export function Row({ dot, label, detail, detailClass, open, onToggle, highlight, children, ...rest }) {
  const [inner, setInner] = useState(false);
  const isOpen = open ?? inner;
  const toggle = onToggle ?? (() => setInner((o) => !o));
  return (
    <div className={cx("border-t border-line first:border-t-0", highlight && "bg-surface-2")} {...rest}>
      <button
        type="button"
        onClick={toggle}
        aria-expanded={isOpen}
        className="flex w-full cursor-pointer items-center gap-3 bg-transparent px-4 py-2 text-left text-small
          transition-colors duration-[260ms] hover:bg-surface-2 hover:duration-[120ms]"
      >
        {dot}
        <span className="font-medium">{label}</span>
        {detail !== undefined && (
          <span className={cx("ml-auto truncate text-right text-ink-2", detailClass)}>{detail}</span>
        )}
      </button>
      {isOpen && (
        <div className="mx-4 mb-3 break-words rounded-cell bg-well p-3 text-small">{children}</div>
      )}
    </div>
  );
}
