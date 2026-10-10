// End-to-end against the REAL stack: real router, real models. Slow, because
// every switch loads a model onto the GPU. Start the stack first:
//   node ../execute.mjs up

import { expect, test } from "@playwright/test";
import { chooseModel, composer, expectChatDisabled, expectChatEnabled, modelButton, sendButton, veil } from "./helpers.js";

test.describe.configure({ mode: "serial" });

const loadedModels = async (request) => {
  const { models } = await (await request.get("/api/models")).json();
  return models.filter((m) => m.status === "loaded").map((m) => m.id);
};

test("the real router: switch models from the dropdown and chat with each", async ({ page, request }) => {
  const { models } = await (await request.get("/api/models")).json();
  const ids = models.map((m) => m.id);
  expect(ids.length, "needs at least two models in models.ini").toBeGreaterThanOrEqual(2);

  await page.goto("/");
  // The starting model is loaded automatically; chat opens only once it is.
  await expect(composer(page)).toBeEnabled({ timeout: 240_000 });
  expect(await loadedModels(request)).toHaveLength(1);

  let current = (await modelButton(page).innerText()).trim();
  expect(ids).toContain(current);

  for (let round = 0; round < 2; round++) {
    const target = ids.find((id) => id !== current);

    await chooseModel(page, target);
    // While the router loads, the page is frozen and chat is off.
    await expect(veil(page)).toBeVisible();
    await expectChatDisabled(page);
    await expect(veil(page)).toBeHidden({ timeout: 300_000 });

    // Afterwards the router has exactly the chosen model loaded.
    await expect(modelButton(page)).toContainText(target);
    await expectChatEnabled(page);
    expect(await loadedModels(request)).toEqual([target]);

    // A real chat request goes to that model and is accepted by the guard.
    const reply = page.waitForResponse((r) => r.url().endsWith("/v1/chat/completions"), { timeout: 120_000 });
    await composer(page).fill(`Round ${round}: reply with the word OK.`);
    await sendButton(page).click();
    const response = await reply;
    expect(response.status()).toBe(200);
    expect(response.request().postDataJSON().model).toBe(target);
    await expect(page.getByText(`Round ${round}: reply with the word OK.`)).toBeVisible();
    // Wait for the turn to finish so the next switch does not interrupt it.
    await expect(page.getByRole("button", { name: "Stop" })).toBeHidden({ timeout: 240_000 });

    current = target;
  }
});

test("the real backend refuses to chat with a model that is not loaded", async ({ request }) => {
  const { models } = await (await request.get("/api/models")).json();
  const notLoaded = models.find((m) => m.status !== "loaded");
  test.skip(!notLoaded, "every model is loaded");

  const response = await request.post("/v1/chat/completions", {
    data: { model: notLoaded.id, messages: [{ role: "user", content: "hi" }] },
  });
  expect(response.status()).toBe(409);
  expect((await response.json()).error.code).toBe("model_not_loaded");
});
