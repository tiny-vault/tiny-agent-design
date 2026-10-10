/* Runs one tool call. `sources` is the turn's source list and is appended to
   when search_docs finds passages. Returns { result, detail, status } where
   result is the JSON string handed back to the model. */
export function runTool(name, args, kb, sources) {
  if (name === "get_current_time") {
    const now = new Date();
    return {
      result: JSON.stringify({ local: now.toString(), iso: now.toISOString() }),
      detail: now.toLocaleTimeString(),
      status: "ok",
    };
  }

  if (name === "search_docs") {
    const words = String(args.query || "").toLowerCase().match(/[a-z0-9_]{3,}/g) || [];
    const scored = kb
      .map((p) => {
        const title = p.title.toLowerCase();
        const text = p.text.toLowerCase();
        const score = words.reduce((n, w) => n + (title.includes(w) ? 3 : 0) + (text.split(w).length - 1), 0);
        return { p, score };
      })
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);

    const out = scored.map(({ p }) => {
      let src = sources.find((s) => s.id === p.id);
      if (!src) {
        src = { n: sources.length + 1, id: p.id, title: p.title, text: p.text };
        sources.push(src);
      }
      return { n: src.n, title: p.title, text: p.text };
    });

    return {
      result: JSON.stringify(out.length ? out : { result: "No matching passages." }),
      detail: out.length === 0 ? "No sources found" : out.length === 1 ? "1 source" : `${out.length} sources`,
      status: out.length ? "ok" : "not_found",
    };
  }

  return { result: JSON.stringify({ error: `Unknown tool: ${name}` }), detail: "Unknown tool", status: "not_found" };
}
