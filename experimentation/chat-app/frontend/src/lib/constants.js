// Ollama accepts localhost origins by default, so serve this app over
// http://localhost (the Vite dev server does); file:// will not work.
export const OLLAMA = "http://localhost:11434";
export const MAX_ROUNDS = 5;
export const FOLLOW_SLACK_PX = 48; // how near the bottom still counts as "reading along"
export const NOTIFY_MS = 33; // batch streamed tokens: ~30 repaints a second

export const SYSTEM_PROMPT =
  "You are the TinyAgent assistant. For any question about TinyAgent, its design, models, setup or credits, " +
  "call search_docs first and answer only from what it returns. Cite the passages you use with their number " +
  "in square brackets, like [1] or [1, 2]. If the passages do not answer the question, say so. " +
  "Use get_current_time for questions about the current time or date. Be concise.";

export const TOOLS = [
  {
    type: "function",
    function: {
      name: "search_docs",
      description: "Search the TinyAgent knowledge base. Returns numbered passages to cite.",
      parameters: {
        type: "object",
        properties: { query: { type: "string", description: "Keywords to search for" } },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_current_time",
      description: "Get the current local date and time.",
      parameters: { type: "object", properties: {} },
    },
  },
];

export const SUGGESTIONS = ["What is TinyAgent?", "How does the agent loop work?", "What time is it?"];
