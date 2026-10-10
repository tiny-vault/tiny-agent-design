import { expect } from "@playwright/test";

/* ---- scripted model replies ------------------------------------------------
   Each helper returns what route.fulfill() needs, shaped like llama.cpp's
   streamed (server-sent events) chat reply. */

const event = (delta, finish = null) =>
  `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;

/* A streamed reply. Each argument is one delta object, e.g. { content: "Hi" }. */
export function stream(...deltas) {
  return {
    status: 200,
    contentType: "text/event-stream",
    body: deltas.map((d) => event(d)).join("") + event({}, "stop") + "data: [DONE]\n\n",
  };
}

/* A plain text answer. */
export const answer = (text) => stream({ content: text });

/* The model asking for tools: calls = [{ name, args }]; args is an object or a raw string. */
export const toolCalls = (...calls) =>
  stream({
    tool_calls: calls.map((c, index) => ({
      index,
      id: `call_${index}`,
      function: { name: c.name, arguments: typeof c.args === "string" ? c.args : JSON.stringify(c.args ?? {}) },
    })),
  });

/* An error reply in the backend's shape. */
export const failure = (status, message, code = "error") => ({
  status,
  contentType: "application/json",
  body: JSON.stringify({ error: { message, code } }),
});

/* ---- stubbed backend --------------------------------------------------------
   `state` is mutable so a test can change what the "router" reports while the
   page is open.

   state.statuses     { model: "loaded" | "loading" | "unloaded" }
   state.gate         a promise that /api/models/load waits on (holds a load open)
   state.loadError    message to fail the next load with (HTTP 500), or null
   state.chats        the bodies of the chat requests the page sent
   state.loads        the models the page asked to load
   state.chatHandler  (body, n) => reply, to script the model; n counts requests
*/
export function newState(statuses = { gemma: "loaded", qwen: "unloaded" }) {
  return { statuses: { ...statuses }, gate: null, loadError: null, chats: [], loads: [], chatHandler: null };
}

/* Hold model loads open until the returned release() is called. */
export function holdLoads(state) {
  let release;
  state.gate = new Promise((resolve) => (release = resolve));
  return () => {
    state.gate = null;
    release();
  };
}

export async function mockApi(page, state) {
  await page.route("**/api/models", (route) =>
    route.fulfill({ json: { models: Object.entries(state.statuses).map(([id, status]) => ({ id, status })) } }),
  );

  await page.route("**/api/models/load", async (route) => {
    const { model } = route.request().postDataJSON();
    state.loads.push(model);
    // Like the real router: starting to load one model unloads the others,
    // even if that load then fails.
    for (const id of Object.keys(state.statuses)) if (id !== model) state.statuses[id] = "unloaded";
    if (state.gate) await state.gate;
    if (state.loadError) {
      return route.fulfill(failure(500, state.loadError, "load_failed"));
    }
    state.statuses[model] = "loaded";
    return route.fulfill({ json: { model, status: "loaded" } });
  });

  await page.route("**/v1/chat/completions", async (route) => {
    const body = route.request().postDataJSON();
    state.chats.push(body);
    const reply = state.chatHandler ? await state.chatHandler(body, state.chats.length) : answer("Hello from the stub.");
    try {
      if (reply === "abort") await route.abort();
      else await route.fulfill(reply);
    } catch {
      /* the page aborted the request (Stop button) */
    }
  });
}

/* ---- page helpers ---------------------------------------------------------- */

export const composer = (page) => page.getByLabel("Message");
export const sendButton = (page) => page.getByTestId("send-button");
export const modelButton = (page) => page.getByTestId("model-button");
export const veil = (page) => page.getByTestId("loading-veil");

/* Open the dropdown and choose a model. */
export async function chooseModel(page, name) {
  await modelButton(page).click();
  await page.getByRole("option", { name: new RegExp(name) }).click();
}

export async function expectChatEnabled(page) {
  await expect(composer(page)).toBeEnabled();
  await composer(page).fill("hello");
  await expect(sendButton(page)).toBeEnabled();
  await composer(page).fill("");
}

export async function expectChatDisabled(page) {
  await expect(composer(page)).toBeDisabled();
  await expect(sendButton(page)).toBeDisabled();
}

export async function send(page, text) {
  await composer(page).fill(text);
  await sendButton(page).click();
}

/* Open the app with a ready model, for tests about the chat itself. */
export async function openChat(page, state = newState()) {
  await mockApi(page, state);
  await page.goto("/");
  await expect(composer(page)).toBeEnabled();
  return state;
}
