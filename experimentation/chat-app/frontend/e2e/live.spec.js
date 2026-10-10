// End-to-end against the REAL stack: real router, real models. Slow, because
// every switch loads a model onto the GPU. Start the stack first:
//   node ../../../execute.mjs up

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
    // Do not wait for the whole answer: these are reasoning models and a full reply takes
    // minutes. The router accepted the request, which is what this test is about. Stop it
    // (which also checks that stopping does not block the next switch).
    await page.getByRole("button", { name: "Stop" }).click();
    // Each stopped turn shows "Stopped.", so by round 1 there are two: check the newest.
    await expect(page.getByText("Stopped.").last()).toBeVisible();

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

// ---- OpenObserve (the log search UI) opens already logged in ---------------------------------

const OBSERVE = process.env.OBSERVE_URL || "http://localhost:5080";
const onLoginPage = (page) => page.getByRole("button", { name: /^login$/i }).count();

test("OpenObserve opens already logged in, with no login screen", async ({ browser }) => {
  const context = await browser.newContext(); // a brand-new browser: no cookies, no storage
  const page = await context.newPage();

  await page.goto(OBSERVE + "/", { waitUntil: "networkidle" });
  await page.waitForURL(/\/web\/(?!login)/, { timeout: 30_000 });
  expect(await onLoginPage(page)).toBe(0);
  await expect(page.getByText(/user email/i)).toHaveCount(0);

  // The browser really holds a login: the cookie is set and the API answers as the admin.
  expect((await context.cookies()).some((c) => c.name === "auth_tokens")).toBe(true);
  const me = await page.evaluate(async () => (await fetch("/api/default/streams?type=logs")).status);
  expect(me).toBe(200);
  await context.close();
});

test("OpenObserve: opening the login page, or losing the stored login, signs in again by itself", async ({ browser }) => {
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto(OBSERVE + "/web/login", { waitUntil: "networkidle" }); // the login URL itself
  await page.waitForURL(/\/web\/(?!login)/, { timeout: 30_000 });
  expect(await onLoginPage(page)).toBe(0);

  await page.evaluate(() => localStorage.clear()); // as if the browser storage was wiped
  await page.goto(OBSERVE + "/web/", { waitUntil: "networkidle" });
  await page.waitForURL(/\/web\/(?!login)/, { timeout: 30_000 });
  expect(await onLoginPage(page)).toBe(0);
  await context.close();
});

test("OpenObserve: the API is not open to someone with no login", async ({ request }) => {
  // The auto-login is a browser cookie. A bare API call with no cookie and no credentials is refused.
  const response = await request.get(OBSERVE + "/api/default/streams?type=logs");
  expect(response.status()).toBe(401);
});

test("OpenObserve receives the container logs and the tiny-agent logs", async ({ request }) => {
  const fs = await import("node:fs");
  const secrets = "../../../experimentation/secrets/";
  const user = fs.readFileSync(secrets + "openobserve_user", "utf8").trim();
  const password = fs.readFileSync(secrets + "openobserve_password", "utf8").trim();
  const auth = { Authorization: "Basic " + Buffer.from(`${user}:${password}`).toString("base64") };

  const now = Date.now() * 1000; // microseconds
  const count = async (stream) => {
    const res = await request.post(`${OBSERVE}/api/default/_search?type=logs`, {
      headers: auth,
      data: { query: { sql: `SELECT * FROM "${stream}"`, start_time: now - 7 * 24 * 3600 * 1e6, end_time: now, from: 0, size: 1 } },
    });
    return res.ok() ? (await res.json()).total : 0;
  };
  expect(await count("docker")).toBeGreaterThan(0);
  expect(await count("tiny_agent")).toBeGreaterThan(0);
});
