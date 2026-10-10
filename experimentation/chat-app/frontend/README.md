# Agent chat frontend

A single-page agent chat built with [React](https://react.dev),
[Vite](https://vite.dev) and [Tailwind CSS](https://tailwindcss.com) (v4) that
talks to a local [Ollama](https://ollama.com) model. It is a small, visible
version of the agent loop: the model can call tools, and every call shows up as
a step you can open.

## Run

```powershell
cd experimentation/chat-app/frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Ollama must be running on port 11434. The dev
server runs on `localhost`, which Ollama's default CORS rules accept.

If you serve it from another host or port that Ollama rejects, set
`OLLAMA_ORIGINS` (for example `OLLAMA_ORIGINS=http://myhost:5173`) and
restart Ollama.

`npm run build` writes a static site to `dist/`; `npm run preview` serves it.

## Layout

```
frontend/
├── index.html            Vite entry
├── public/kb.json        Knowledge base searched by search_docs
├── src/
│   ├── main.jsx, App.jsx
│   ├── index.css         Tailwind import + design tokens (both themes)
│   ├── components/       Header, ModelMenu, ChatLog, Turn, Composer, ui.jsx
│   ├── hooks/            useAgentChat (the loop), useModels, useTheme, useKnowledgeBase
│   └── lib/              ollama.js (API + streaming), tools.js, citations.js, constants.js
└── legacy/alpine.html    The original Alpine.js single-file version
```

## What it does

* **Model picker** lists your installed Ollama models.
* **Tools.** `search_docs` searches `kb.json`, a small TinyAgent knowledge base,
  and `get_current_time` returns the local time. The loop runs up to five tool
  rounds per question.
* **Step trace.** Each tool call is a row with a status dot, the query, and a
  result summary. Click a row to see the raw arguments and result.
* **Citations.** The model cites passages as `[1]` or `[1, 2]`. Marks become
  buttons, and the Sources list under the answer shows only the passages the
  answer actually cites. Click a mark to open its passage.
* **Reasoning.** If the model sends its thinking separately (Gemma 4 does), it
  is shown in a dimmer block above the answer.
* **Models without tool calling.** Ollama rejects tool requests for such models
  (Gemma 3 does this). The page notices, shows a note, and answers without
  tools. This makes the difference between the two models easy to see.
* Streaming answer, Stop button, light and dark themes, new conversation.

## Design

The look and several chat behaviours are ported from
[Utopia](https://github.com/deeplethe/utopia) by DeepLethe (Apache-2.0),
following its `web/DESIGN.md`. The code is rewritten for React and Tailwind, not copied. The tokens live in
  `src/index.css` and are exposed to Tailwind through `@theme`.

* Neutral glass in two themes, no hue in the chrome. Colour only for status,
  shown as a dot plus plain words.
* Five named type sizes (12, 13, 14, 16, 24 px); chat reads at 16 px.
* Spacing from a 4, 8, 12, 16, 24, 32 px scale, and four radii derived from one
  `--radius`.
* Two text levels (ink and ink-2); lists are one panel with rows.
* The page only follows a streaming answer while you are within 48 px of the
  bottom, so scrolling up to read is not interrupted.
* Streamed tokens are painted in batches every 33 ms.
* Fonts are Geist and Manrope from Google Fonts, so the page needs internet
  the first time.

## Notes

* The conversation lives in the page only. Reloading clears it.
* The theme and the selected model are remembered in `localStorage`.
