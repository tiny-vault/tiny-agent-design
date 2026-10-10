// Edge cases in the chat itself (input, streaming, tools, citations, errors),
// with a scripted model. A ready model is assumed; see gating.spec.js for switching.

import { expect, test } from "@playwright/test";
import {
  answer, composer, failure, newState, openChat, send, sendButton, stream, toolCalls,
} from "./helpers.js";

const delay = (ms, reply) => new Promise((resolve) => setTimeout(() => resolve(reply), ms));
const roles = (body) => body.messages.map((m) => m.role);
const stopButton = (page) => page.getByRole("button", { name: "Stop" });

// ---- input ----------------------------------------------------------------

test("a blank message cannot be sent", async ({ page }) => {
  const state = await openChat(page);
  await composer(page).fill("   \n  ");
  await expect(sendButton(page)).toBeDisabled();
  await composer(page).press("Enter");
  await page.waitForTimeout(300);
  expect(state.chats).toHaveLength(0);
});

test("Enter sends and Shift+Enter adds a line", async ({ page }) => {
  const state = await openChat(page);
  await composer(page).fill("line one");
  await composer(page).press("Shift+Enter");
  await composer(page).pressSequentially("line two");
  expect(await composer(page).inputValue()).toBe("line one\nline two");
  await composer(page).press("Enter");
  await expect(page.getByText("Hello from the stub.")).toBeVisible();
  expect(state.chats[0].messages.at(-1).content).toBe("line one\nline two");
  await expect(composer(page)).toHaveValue("");
});

test("long messages and emoji arrive intact", async ({ page }) => {
  const state = await openChat(page);
  const text = `héllo wörld 🌍 日本語 ${"x".repeat(12_000)}`;
  await send(page, text);
  await expect(page.getByText("Hello from the stub.")).toBeVisible();
  expect(state.chats[0].messages.at(-1).content).toBe(text);
});

test("HTML in a question or an answer is shown as text, never run", async ({ page }) => {
  const state = newState();
  state.chatHandler = () => answer('<script>window.__xss = 2</script><b>bold</b> <img src=x onerror="window.__xss = 3">');
  await openChat(page, state);
  await send(page, '<img src=x onerror="window.__xss = 1">');
  await expect(page.getByText("<b>bold</b>", { exact: false })).toBeVisible();
  await expect(page.getByText('<img src=x onerror="window.__xss = 1">')).toBeVisible();
  expect(await page.evaluate(() => window.__xss)).toBeUndefined();
  await expect(page.locator("main b, main img[src='x']")).toHaveCount(0);
});

test("a suggestion sends its question", async ({ page }) => {
  const state = await openChat(page);
  await page.getByRole("button", { name: "What is TinyAgent?" }).click();
  await expect(page.getByText("Hello from the stub.")).toBeVisible();
  expect(state.chats[0].messages.at(-1).content).toBe("What is TinyAgent?");
});

// ---- conversation history ---------------------------------------------------

test("later questions carry the earlier ones", async ({ page }) => {
  const state = newState();
  state.chatHandler = (_, n) => answer(`answer ${n}`);
  await openChat(page, state);
  await send(page, "first");
  await expect(page.getByText("answer 1")).toBeVisible();
  await send(page, "second");
  await expect(page.getByText("answer 2")).toBeVisible();
  expect(state.chats[1].messages.map((m) => [m.role, m.content])).toEqual([
    ["system", expect.any(String)], ["user", "first"], ["assistant", "answer 1"], ["user", "second"],
  ]);
});

test("a new conversation forgets the old one", async ({ page }) => {
  const state = await openChat(page);
  await send(page, "first");
  await expect(page.getByText("Hello from the stub.")).toBeVisible();
  await page.getByRole("button", { name: "New conversation" }).click();
  await expect(page.getByRole("heading", { name: "Ask the agent" })).toBeVisible();
  await send(page, "fresh start");
  await expect(page.getByText("Hello from the stub.")).toBeVisible();
  expect(roles(state.chats[1])).toEqual(["system", "user"]);
});

// ---- errors and stopping ------------------------------------------------------

test("a server error is shown and the next message still works", async ({ page }) => {
  const state = newState();
  state.chatHandler = (_, n) => (n === 1 ? failure(500, "boom: the model crashed") : answer("recovered"));
  await openChat(page, state);
  await send(page, "first");
  await expect(page.getByText("boom: the model crashed")).toBeVisible();
  await expect(composer(page)).toBeEnabled();
  await send(page, "second");
  await expect(page.getByText("recovered")).toBeVisible();
});

test("a 'model not loaded' refusal from the backend is shown", async ({ page }) => {
  const state = newState();
  state.chatHandler = () => failure(409, "gemma is unloaded. Load it first (POST /api/models/load).", "model_not_loaded");
  await openChat(page, state);
  await send(page, "hi");
  await expect(page.getByText("gemma is unloaded. Load it first")).toBeVisible();
  await expect(composer(page)).toBeEnabled();
});

test("a dropped connection is reported and does not wedge the chat", async ({ page }) => {
  const state = newState();
  state.chatHandler = (_, n) => (n === 1 ? "abort" : answer("back online"));
  await openChat(page, state);
  await send(page, "first");
  await expect(page.locator("p.text-danger").first()).toBeVisible();
  await expect(sendButton(page)).toBeVisible(); // not stuck on "Stop"
  await send(page, "second");
  await expect(page.getByText("back online")).toBeVisible();
});

test("Stop cancels the answer and the chat can go on", async ({ page }) => {
  const state = newState();
  state.chatHandler = (_, n) => (n === 1 ? delay(8_000, answer("too late")) : answer("second answer"));
  await openChat(page, state);
  await send(page, "slow question");
  await stopButton(page).click();
  await expect(page.getByText("Stopped.")).toBeVisible();
  await expect(page.getByText("too late")).toHaveCount(0);
  await expect(sendButton(page)).toBeVisible();

  await send(page, "next question");
  await expect(page.getByText("second answer")).toBeVisible();
  // The stopped question has no answer, so it must not leave two user turns in a row
  // (chat templates reject that).
  expect(roles(state.chats[1])).toEqual(["system", "user"]);
  expect(state.chats[1].messages.at(-1).content).toBe("next question");
});

test("a second Enter while the model is answering is ignored", async ({ page }) => {
  const state = newState();
  state.chatHandler = () => delay(1_500, answer("done"));
  await openChat(page, state);
  await send(page, "one");
  await expect(stopButton(page)).toBeVisible();
  await composer(page).fill("two");
  await composer(page).press("Enter");
  await expect(page.getByText("done")).toBeVisible();
  expect(state.chats).toHaveLength(1);
});

test("an empty reply does not break the page", async ({ page }) => {
  const state = newState();
  state.chatHandler = () => stream({ content: "" });
  await openChat(page, state);
  await send(page, "say nothing");
  await expect(sendButton(page)).toBeVisible();
  await expect(composer(page)).toBeEnabled();
  await expect(page.getByText("Working…")).toHaveCount(0);
});

// ---- reasoning --------------------------------------------------------------

test("streamed reasoning is shown apart from the answer", async ({ page }) => {
  const state = newState();
  state.chatHandler = () => stream({ reasoning_content: "Let me think. " }, { reasoning_content: "Done." }, { content: "The answer." });
  await openChat(page, state);
  await send(page, "think");
  await expect(page.getByText("The answer.")).toBeVisible();
  await expect(page.getByRole("button", { name: /Thought/ })).toContainText("19 chars");
  await page.getByRole("button", { name: /Thought/ }).click();
  await expect(page.getByText("Let me think. Done.")).toBeVisible();
});

// ---- tools and citations ------------------------------------------------------

test("a tool call is traced, its source is cited, and only cited sources are listed", async ({ page }) => {
  const state = newState();
  state.chatHandler = (_, n) =>
    n === 1 ? toolCalls({ name: "search_docs", args: { query: "TinyAgent" } }) : answer("TinyAgent is a harness [1].");
  await openChat(page, state);
  await send(page, "What is TinyAgent?");

  await expect(page.getByText("TinyAgent is a harness")).toBeVisible();
  await expect(page.getByText("1 step", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /search_docs/ })).toContainText("sources");
  // The tool result went back to the model.
  expect(state.chats[1].messages.at(-1).role).toBe("tool");
  expect(roles(state.chats[1])).toEqual(["system", "user", "assistant", "tool"]);

  // [1] became a button; only source 1 is listed even though the search found three.
  await expect(page.getByRole("button", { name: "1", exact: true })).toBeVisible();
  await expect(page.getByText("Sources", { exact: true })).toBeVisible();
  await expect(page.locator("[data-src]")).toHaveCount(1);
  await page.getByRole("button", { name: "1", exact: true }).click();
  await expect(page.locator("[data-src='1']")).toContainText("TinyAgent");
});

test("a citation to a source that was never retrieved stays plain text", async ({ page }) => {
  const state = newState();
  state.chatHandler = (_, n) =>
    n === 1 ? toolCalls({ name: "search_docs", args: { query: "TinyAgent" } }) : answer("Real [1] and made up [9].");
  await openChat(page, state);
  await send(page, "cite");
  await expect(page.getByText(/made up \[9\]/)).toBeVisible();
  await expect(page.getByRole("button", { name: "9", exact: true })).toHaveCount(0);
  await expect(page.locator("[data-src]")).toHaveCount(1);
});

test("an answer that used search but cited nothing is flagged", async ({ page }) => {
  const state = newState();
  state.chatHandler = (_, n) =>
    n === 1 ? toolCalls({ name: "search_docs", args: { query: "TinyAgent" } }) : answer("It is a thing.");
  await openChat(page, state);
  await send(page, "no cites");
  await expect(page.getByText("Answered without citing any source.")).toBeVisible();
});

test("a search with no matches says so and the model can still answer", async ({ page }) => {
  const state = newState();
  state.chatHandler = (_, n) =>
    n === 1 ? toolCalls({ name: "search_docs", args: { query: "zzzqqq" } }) : answer("I could not find that.");
  await openChat(page, state);
  await send(page, "nonsense");
  await expect(page.getByText("I could not find that.")).toBeVisible();
  await expect(page.getByRole("button", { name: /search_docs/ })).toContainText("No sources found");
});

test("tool arguments that are not valid JSON are reported, not fatal", async ({ page }) => {
  const state = newState();
  state.chatHandler = (_, n) => (n === 1 ? toolCalls({ name: "search_docs", args: "{not json" }) : answer("carried on"));
  await openChat(page, state);
  await send(page, "bad args");
  await expect(page.getByText("carried on")).toBeVisible();
  await expect(page.getByText("Could not read arguments")).toBeVisible();
});

test("an unknown tool is reported and the loop continues", async ({ page }) => {
  const state = newState();
  state.chatHandler = (_, n) => (n === 1 ? toolCalls({ name: "delete_everything", args: {} }) : answer("ok then"));
  await openChat(page, state);
  await send(page, "bad tool");
  await expect(page.getByText("ok then")).toBeVisible();
  await expect(page.getByText("Unknown tool")).toBeVisible();
});

test("the clock tool runs in the page", async ({ page }) => {
  const state = newState();
  state.chatHandler = (_, n) => (n === 1 ? toolCalls({ name: "get_current_time" }) : answer("It is time."));
  await openChat(page, state);
  await send(page, "what time is it");
  await expect(page.getByText("It is time.")).toBeVisible();
  const toolReply = JSON.parse(state.chats[1].messages.at(-1).content);
  expect(new Date(toolReply.iso).getTime()).toBeGreaterThan(Date.now() - 60_000);
});

test("two tool calls in one reply are both run", async ({ page }) => {
  const state = newState();
  state.chatHandler = (_, n) =>
    n === 1
      ? toolCalls({ name: "get_current_time" }, { name: "search_docs", args: { query: "TinyAgent" } })
      : answer("both done [1]");
  await openChat(page, state);
  await send(page, "two tools");
  await expect(page.getByText("both done")).toBeVisible();
  await expect(page.getByText("2 steps", { exact: true })).toBeVisible();
  expect(roles(state.chats[1])).toEqual(["system", "user", "assistant", "tool", "tool"]);
});

test("a model that never stops calling tools is cut off after five rounds", async ({ page }) => {
  const state = newState();
  state.chatHandler = () => toolCalls({ name: "get_current_time" });
  await openChat(page, state);
  await send(page, "loop");
  await expect(page.getByText("Stopped after too many tool rounds.")).toBeVisible();
  expect(state.chats).toHaveLength(5);
  await expect(composer(page)).toBeEnabled();
});

test("a model without tool support answers without tools, and is remembered", async ({ page }) => {
  const state = newState();
  state.chatHandler = (body) =>
    body.tools ? failure(400, "registry.ollama.ai/library/gemma3 does not support tools") : answer("plain answer");
  await openChat(page, state);
  await send(page, "first");
  await expect(page.getByText("plain answer")).toBeVisible();
  await expect(page.getByText(/has no native tool calling/)).toBeVisible();
  expect(state.chats.map((c) => Boolean(c.tools))).toEqual([true, false]);

  await send(page, "second");
  await expect(page.getByText("plain answer")).toHaveCount(2);
  expect(state.chats.map((c) => Boolean(c.tools))).toEqual([true, false, false]); // did not ask again
});
