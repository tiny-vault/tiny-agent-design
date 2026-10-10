// Frontend behaviour around model switching, with the model API stubbed.
// Chat must stay off until the router reports the selected model "loaded".

import { expect, test } from "@playwright/test";
import {
  chooseModel, composer, expectChatDisabled, expectChatEnabled, holdLoads,
  modelButton, mockApi, newState, sendButton, veil,
} from "./helpers.js";

test("chat is available at once when the selected model is already loaded", async ({ page }) => {
  const state = newState({ gemma: "loaded", qwen: "unloaded" });
  await mockApi(page, state);
  await page.goto("/");
  await expect(modelButton(page)).toContainText("gemma");
  await expectChatEnabled(page);
  await expect(page.getByTestId("load-banner")).toHaveCount(0);
  expect(state.loads).toEqual([]); // nothing needed loading
});

test("an unloaded starting model is loaded automatically, and chat is off until then", async ({ page }) => {
  const state = newState({ gemma: "unloaded", qwen: "unloaded" });
  const release = holdLoads(state);
  await mockApi(page, state);
  await page.goto("/");

  await expect(veil(page)).toBeVisible();
  await expect(veil(page)).toContainText("gemma");
  await expectChatDisabled(page);
  await expect(page.getByRole("button", { name: "Send" })).toBeDisabled();

  release();
  await expect(veil(page)).toBeHidden();
  await expectChatEnabled(page);
  expect(state.loads).toEqual(["gemma"]);
});

test("switching models freezes the page until the router says loaded", async ({ page }) => {
  const state = newState({ gemma: "loaded", qwen: "unloaded" });
  await mockApi(page, state);
  await page.goto("/");
  await expectChatEnabled(page);

  const release = holdLoads(state);
  await chooseModel(page, "qwen");

  await expect(veil(page)).toBeVisible();
  await expect(veil(page)).toContainText("qwen");
  await expect(modelButton(page)).toBeDisabled();
  await expect(modelButton(page)).toContainText("Loading qwen");
  await expectChatDisabled(page);
  await expect(composer(page)).toHaveAttribute("placeholder", /Loading qwen/);

  release();
  await expect(veil(page)).toBeHidden();
  await expect(modelButton(page)).toContainText("qwen");
  await expectChatEnabled(page);
  expect(state.loads).toEqual(["qwen"]);
});

test("after switching, messages go to the new model", async ({ page }) => {
  const state = newState({ gemma: "loaded", qwen: "unloaded" });
  await mockApi(page, state);
  await page.goto("/");
  await chooseModel(page, "qwen");
  await expect(veil(page)).toBeHidden();

  await composer(page).fill("Which model are you?");
  await sendButton(page).click();
  await expect(page.getByText("Hello from the stub.")).toBeVisible();
  expect(state.chats.map((c) => c.model)).toEqual(["qwen"]);
});

test("the dropdown shows which model is loaded", async ({ page }) => {
  await mockApi(page, newState({ gemma: "loaded", qwen: "unloaded" }));
  await page.goto("/");
  await modelButton(page).click();
  await expect(page.getByRole("option", { name: /gemma/ })).toContainText("Loaded");
  await expect(page.getByRole("option", { name: /qwen/ })).not.toContainText("Loaded");
});

test("switching back and forth keeps exactly one model usable", async ({ page }) => {
  const state = newState({ gemma: "loaded", qwen: "unloaded" });
  await mockApi(page, state);
  await page.goto("/");

  for (const next of ["qwen", "gemma", "qwen"]) {
    await chooseModel(page, next);
    await expect(veil(page)).toBeHidden();
    await expect(modelButton(page)).toContainText(next);
    await expectChatEnabled(page);
  }
  expect(state.loads).toEqual(["qwen", "gemma", "qwen"]);
});

test("choosing the model that is already loaded does nothing", async ({ page }) => {
  const state = newState({ gemma: "loaded", qwen: "unloaded" });
  await mockApi(page, state);
  await page.goto("/");
  await chooseModel(page, "gemma");
  await expectChatEnabled(page);
  expect(state.loads).toEqual([]);
});

test("a failed load shows the reason, keeps chat off, and can be retried", async ({ page }) => {
  const state = newState({ gemma: "loaded", qwen: "unloaded" });
  await mockApi(page, state);
  await page.goto("/");

  state.loadError = "qwen failed to load. Check the llama container logs.";
  await chooseModel(page, "qwen");

  await expect(page.getByTestId("error")).toContainText("qwen failed to load");
  await expect(veil(page)).toBeHidden();
  // The router unloaded gemma when it started loading qwen, so nothing is usable.
  await expect(page.getByTestId("load-banner")).toContainText("gemma");
  await expectChatDisabled(page);

  state.loadError = null;
  await page.getByTestId("load-banner").getByRole("button", { name: "Load" }).click();
  await expect(veil(page)).toBeHidden();
  await expect(page.getByTestId("load-banner")).toHaveCount(0);
  await expectChatEnabled(page);
});

test("a model unloaded outside the page switches chat off", async ({ page }) => {
  const state = newState({ gemma: "loaded", qwen: "unloaded" });
  await mockApi(page, state);
  await page.goto("/");
  await expectChatEnabled(page);

  state.statuses.gemma = "unloaded"; // someone unloaded it; the page polls every 5s
  await expect(page.getByTestId("load-banner")).toBeVisible({ timeout: 12_000 });
  await expectChatDisabled(page);
});

test("suggestion buttons are disabled until a model is ready", async ({ page }) => {
  const state = newState({ gemma: "unloaded", qwen: "unloaded" });
  const release = holdLoads(state);
  await mockApi(page, state);
  await page.goto("/");

  await expect(page.getByRole("button", { name: "What is TinyAgent?" })).toBeDisabled();
  release();
  await expect(page.getByRole("button", { name: "What is TinyAgent?" })).toBeEnabled();
});

test("an unreachable backend is reported and chat stays off", async ({ page }) => {
  await page.route("**/api/models", (route) => route.abort());
  await page.goto("/");
  await expect(page.getByTestId("error")).toContainText("Cannot reach the backend");
  await expectChatDisabled(page);
});
