import { FOLLOW_SLACK_PX } from "./constants.js";

/* True when the log is at (or within 48px of) the bottom. Content shorter than
   the viewport counts as at the bottom: there is nowhere to scroll away to. */
export function followsBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= FOLLOW_SLACK_PX;
}

/* Split text into plain text and citation groups: [1], [1][2], [1, 2].
   A marker still streaming in, like "[1", does not match until its bracket
   closes, so a half-drawn mark never flickers into place. A fresh regex per
   call, because a shared /g regex keeps lastIndex between calls. */
export function splitCitations(text) {
  const re = /\[(\d+(?:\s*[,，]\s*\d+)*)\]/g;
  const out = [];
  let last = 0;
  for (let m = re.exec(text); m; m = re.exec(text)) {
    const ns = m[1]
      .split(/[,，]/)
      .map((s) => Number(s.trim()))
      .filter((n) => Number.isInteger(n) && n > 0);
    if (!ns.length) continue;
    if (m.index > last) out.push({ text: text.slice(last, m.index) });
    out.push({ cite: ns });
    last = m.index + m[0].length;
  }
  if (!out.length) return [{ text }];
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}

/* Only the sources the answer actually cites, numbered as retrieved. */
export function citedSources(turn) {
  const cited = new Set();
  for (const p of splitCitations(turn.content)) p.cite?.forEach((n) => cited.add(n));
  return turn.sources.filter((s) => cited.has(s.n));
}

/* Flag a finished answer that used the search tool but cited nothing. */
export function answeredWithoutSources(turn) {
  if (turn.working || !turn.content || turn.error || turn.stopped) return false;
  return turn.steps.some((s) => s.tool === "search_docs") && citedSources(turn).length === 0;
}
