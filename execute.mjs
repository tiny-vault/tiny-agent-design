#!/usr/bin/env node
// Control the chat app's Docker deployment.  Usage: node execute.mjs <command> [args]
// No dependencies; needs Node 18+ and Docker with the compose plugin.

import { spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { request as httpRequest } from "node:http";
import { dirname, join } from "node:path";
import { createInterface } from "node:readline/promises";
import { fileURLToPath } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));
// ROOT is the repo root. The compose file is in experimentation/, next to .env and logs/.
const DEPLOY = join(ROOT, "experimentation");
const CHAT_APP = join(DEPLOY, "chat-app");
const COMPOSE_FILE = join(DEPLOY, "docker-compose.yaml");
const PORT = process.env.GATEWAY_PORT || "3200";
const URL_BASE = `http://localhost:${PORT}`;
// OpenObserve (log search) and the folder that holds its data and the tiny-agent/main.py logs.
const OBSERVE_URL = `http://localhost:${process.env.OPENOBSERVE_PORT || "5080"}`;
const LOG_DIR = join(CHAT_APP, "logs");
// Models llama.cpp downloaded from Hugging Face (git-ignored). prune never touches this folder.
const MODELS_DIR = join(CHAT_APP, "models", "cache");
// The OpenObserve login is a Docker secret: two files that are not in git (secrets/ is ignored).
const SECRETS_DIR = join(CHAT_APP, "secrets");
const SECRET_USER = join(SECRETS_DIR, "openobserve_user");
const SECRET_PASSWORD = join(SECRETS_DIR, "openobserve_password");
const SECRET_LLAMA_KEY = join(SECRETS_DIR, "llama_api_key");

const HELP = `
TinyAgent chat deployment

  node execute.mjs <command> [args]

Lifecycle
  up [--no-build]     Start the COMPLETE stack: app, backend, llama.cpp and the log pipeline
                      (OpenObserve + Vector). Models download on first load; see "download"
  down                Stop and remove the containers (models and images are kept)
  restart [service]   Restart one service or all of them
  rebuild [service]   Rebuild images and recreate containers
  clean               Stop, remove containers, volumes and locally built images
  prune [--yes] [--dry-run] [--keep-logs] [--keep-secrets]
                      Remove the WHOLE setup except the downloaded models and the images:
                      containers, networks, every volume of the stack, generated secrets, logs
                      (incl. OpenObserve data), build output (node_modules, dist, caches).
                      Images are kept, so the next "up" needs no big downloads. Asks first
                      unless --yes.

Inspect
  status              Container state plus the router's model status
  logs [service]      Follow a container's logs (llama, backend, frontend,
                      gateway, openobserve, vector)
  observe [--password]  Where to search ALL logs: the OpenObserve URL (opens already logged in),
                      what is collected, and the login (a Docker secret) with --password
  health              Check the gateway, backend, llama.cpp and OpenObserve
  config              Print the resolved compose configuration

Models (through the gateway)
  models              List models and whether each is loaded
  download [model...] Fetch models from Hugging Face ahead of time (all in models.ini by
                      default): loads each once and shows the download progress
  load <model>        Load a model through the router and wait until it is loaded

Tests  (extra arguments after "--" go to the test runner)
  test                Everything that can run: backend + agent + llm + mocked
                      (+ live and the quick ias tests if the stack is up)
  test backend        Backend pytest suite (fake llama router; no Docker needed)
  test agent          tiny_agent pytest suite (fake model server; no Docker needed)
  test llm            llm_client pytest suite (fake model server; no Docker needed)
  test mocked         Playwright, API stubbed in the browser (current source; no Docker needed)
  test live           Playwright against the REAL stack: switches real models (slow; needs "up")
  test e2e            mocked + live
  test ias            Run every experimentation/illustrated_agents script for real against
                      the stack (qwen3.5-4b loaded); slow ones take many minutes. Quick
                      subset: test ias -- -m "not slow"
  test list           List every test without running it

  Examples
    node execute.mjs test backend -- -k load        backend tests with "load" in the name
    node execute.mjs test mocked -- --grep Stop     mocked tests whose title has "Stop"
    node execute.mjs test mocked -- --headed        watch the browser

  open                Print the app URL
  help                Show this text

Settings come from experimentation/.env or the environment: GATEWAY_PORT,
OPENOBSERVE_PORT, LLAMA_MODELS_MAX (see experimentation/.env.example).
`;

function run(cmd, args, options = {}) {
  const result = spawnSync(cmd, args, { stdio: "inherit", shell: false, ...options });
  if (result.error) {
    if (result.error.code === "ENOENT") fail(`'${cmd}' was not found. Is it installed and on your PATH?`);
    fail(result.error.message);
  }
  return result.status ?? 1;
}

/* Create the secrets if they do not exist yet: the OpenObserve login and the llama.cpp API key. The user defaults to
   admin@example.com (or $OPENOBSERVE_USER); the password is random and never printed
   here: read it with `node execute.mjs observe --password`. Existing files are kept, except an
   OpenObserve password that breaks its policy (it refuses to start with one). */

/* OpenObserve refuses to start unless the root password is 8-128 characters with a lowercase
   letter, an uppercase letter, a digit and a special character. */
const passwordIsStrong = (p) => p.length >= 8 && p.length <= 128 && /[a-z]/.test(p) && /[A-Z]/.test(p) && /\d/.test(p) && /[^A-Za-z0-9]/.test(p);

/* Random, and always meets that policy: alphanumeric random part plus one of each class. */
const generatePassword = () => randomBytes(18).toString("base64url").replace(/[-_]/g, "x") + "Aa9!";

function ensureSecrets() {
  mkdirSync(SECRETS_DIR, { recursive: true });
  if (!existsSync(SECRET_USER)) writeFileSync(SECRET_USER, process.env.OPENOBSERVE_USER || "admin@example.com", { mode: 0o600 });
  if (!existsSync(SECRET_PASSWORD) || !passwordIsStrong(readFileSync(SECRET_PASSWORD, "utf8").trim())) {
    writeFileSync(SECRET_PASSWORD, generatePassword(), { mode: 0o600 });
    console.log(`Created the OpenObserve login in ${SECRETS_DIR} (not in git).`);
  }
  if (!existsSync(SECRET_LLAMA_KEY)) {
    writeFileSync(SECRET_LLAMA_KEY, randomBytes(24).toString("hex"), { mode: 0o600 });
    console.log(`Created the llama.cpp API key in ${SECRETS_DIR} (not in git).`);
  }
}

const readSecret = (file) => readFileSync(file, "utf8").trim();

const compose = (...args) => {
  ensureSecrets(); // the compose file refers to these files, so they must exist for every command
  return run("docker", ["compose", "-f", COMPOSE_FILE, "--project-directory", DEPLOY, ...args]);
};

function fail(message, code = 1) {
  console.error(`\nError: ${message}`);
  process.exit(code);
}

/* A call to the gateway. Uses node:http rather than fetch: fetch gives up after 5 minutes of
   waiting for the response headers whatever timeout is set, and the first load of a model
   (it downloads several GB first) takes longer than that. `timeoutMs` is how long the
   connection may stay completely silent. */
function api(path, options = {}, timeoutMs = 10_000) {
  return new Promise((resolve, reject) => {
    const req = httpRequest(
      new URL(URL_BASE + path),
      { method: options.method || "GET", headers: options.headers || {}, timeout: timeoutMs },
      (res) => {
        let text = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => (text += chunk));
        res.on("end", () => {
          let body;
          try {
            body = JSON.parse(text);
          } catch {
            body = text;
          }
          resolve({ ok: res.statusCode >= 200 && res.statusCode < 300, status: res.statusCode, body });
        });
      },
    );
    req.on("timeout", () => req.destroy(new Error(`no answer for ${Math.round(timeoutMs / 1000)}s`)));
    req.on("error", reject);
    if (options.body) req.write(options.body);
    req.end();
  });
}

async function printModels() {
  try {
    const { ok, status, body } = await api("/api/models");
    if (!ok) return console.log(`Models: unavailable (HTTP ${status})`);
    console.log("\nModels");
    for (const m of body.models) console.log(`  ${m.status === "loaded" ? "●" : "○"} ${m.id.padEnd(16)} ${m.status}`);
  } catch (e) {
    console.log(`\nModels: gateway not reachable at ${URL_BASE} (${e.message})`);
  }
}

async function health() {
  const checks = [
    ["gateway  /          ", "/"],
    ["backend  /health    ", "/health"],
    ["llama    /api/models", "/api/models"],
  ];
  let bad = 0;
  try {
    const res = await fetch(`${OBSERVE_URL}/healthz`, { signal: AbortSignal.timeout(5000) });
    console.log(`${res.ok ? "OK  " : "FAIL"} openobserve /healthz HTTP ${res.status}`);
    if (!res.ok) bad++;
  } catch (e) {
    console.log(`FAIL openobserve /healthz ${e.message}`);
    bad++;
  }
  for (const [label, path] of checks) {
    try {
      const { ok, status, body } = await api(path);
      const extra = path === "/health" && body?.llama ? ` (llama ${body.llama})` : "";
      console.log(`${ok ? "OK  " : "FAIL"} ${label} HTTP ${status}${extra}`);
      if (!ok || body?.llama === "down") bad++;
    } catch (e) {
      console.log(`FAIL ${label} ${e.message}`);
      bad++;
    }
  }
  return bad === 0;
}

/* Wait until OpenObserve answers, so the first logs are not lost to a cold start. */
async function waitForObserve(seconds = 60) {
  process.stdout.write("Waiting for OpenObserve");
  const deadline = Date.now() + seconds * 1000;
  while (Date.now() < deadline) {
    try {
      if ((await fetch(`${OBSERVE_URL}/healthz`, { signal: AbortSignal.timeout(3000) })).ok) return console.log(" ready.");
    } catch {
      /* not up yet */
    }
    process.stdout.write(".");
    await new Promise((r) => setTimeout(r, 2000));
  }
  console.log();
  console.log(`Warning: OpenObserve did not answer within ${seconds}s. The app works; check: node execute.mjs logs openobserve`);
}

function printUrls() {
  console.log("\nApp and API:  " + URL_BASE);
  console.log("Logs (OpenObserve):  " + OBSERVE_URL);
}

async function waitUntilHealthy(seconds = 180) {
  process.stdout.write("Waiting for the stack to answer");
  const deadline = Date.now() + seconds * 1000;
  while (Date.now() < deadline) {
    try {
      const { ok, body } = await api("/health", {}, 3000);
      if (ok && body?.llama === "up") return console.log(" ready.");
    } catch {
      /* not up yet */
    }
    process.stdout.write(".");
    await new Promise((r) => setTimeout(r, 2000));
  }
  console.log();
  fail(`The stack did not become healthy within ${seconds}s. Try: node execute.mjs logs`);
}

/* Total size in bytes of every file under a folder (0 if it does not exist). In-progress
   Hugging Face downloads count, so this grows while a model downloads. */
function folderSize(dir) {
  if (!existsSync(dir)) return 0;
  let total = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.isSymbolicLink()) continue; // the Hugging Face cache links snapshots to blobs; Windows cannot read links made in the container
    const full = join(dir, entry.name);
    try {
      total += entry.isDirectory() ? folderSize(full) : statSync(full).size;
    } catch {
      /* a file that vanished or cannot be read: skip it, this is only a progress counter */
    }
  }
  return total;
}

/* Make llama.cpp download models ahead of time. llama.cpp fetches a model from Hugging Face the
   first time it is loaded, so this loads each one through the backend and reports progress. */
async function download(names) {
  const { ok, body } = await api("/api/models");
  if (!ok) fail(`The stack is not answering at ${URL_BASE}. Start it with: node execute.mjs up`);
  const available = body.models.map((m) => m.id);
  const unknown = names.filter((n) => !available.includes(n));
  if (unknown.length) fail(`Unknown model(s): ${unknown.join(", ")}. Available: ${available.join(", ")}`);

  for (const model of names.length ? names : available) {
    console.log(`\n${model}: loading (it is downloaded first if it is not cached yet)...`);
    const startSize = folderSize(MODELS_DIR);
    const started = Date.now();
    const ticker = setInterval(() => {
      const mb = Math.round((folderSize(MODELS_DIR) - startSize) / 1e6);
      const seconds = Math.round((Date.now() - started) / 1000);
      process.stdout.write(`\r  ${mb} MB downloaded so far, ${seconds}s elapsed   `);
    }, 3000);
    let result;
    try {
      result = await api(
        "/api/models/load",
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model }) },
        2_000_000,
      );
    } finally {
      clearInterval(ticker);
    }
    process.stdout.write("\r" + " ".repeat(60) + "\r");
    if (!result.ok) fail(`${model}: ${result.body?.error?.message || `HTTP ${result.status}`}`);
    const mb = Math.round((folderSize(MODELS_DIR) - startSize) / 1e6);
    console.log(`${model}: loaded in ${Math.round((Date.now() - started) / 1000)}s${mb > 0 ? ` (${mb} MB downloaded)` : " (already cached)"}`);
  }
  await printModels();
}

/* Every Python bytecode cache folder in the repo (not inside .venv or node_modules). */
function findPycache(dir, out = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (!entry.isDirectory() || [".git", ".venv", "node_modules"].includes(entry.name)) continue;
    const full = join(dir, entry.name);
    if (entry.name === "__pycache__") out.push(full);
    else findPycache(full, out);
  }
  return out;
}

/* The names of the Docker volumes that belong to this stack (labelled by compose). */
function relatedVolumes() {
  const result = spawnSync("docker", ["volume", "ls", "-q", "--filter", "label=com.docker.compose.project=chat-app"], { encoding: "utf8" });
  return result.status === 0 ? result.stdout.split(/\r?\n/).filter(Boolean) : [];
}

/* Remove the complete setup, but never the downloaded models and never the images.
   Docker: containers, networks and every volume of the stack (the vector-data volume and any
   anonymous volume a container left behind). Images are kept on purpose: llama.cpp alone is
   several GB. Never `docker system prune`: other projects on this machine are not ours to touch.
   Files: the generated secrets, the logs folder (OpenObserve data and the tiny-agent/main.py
   logs), and build output. Kept: source, .env, .venv, git, and the models folder. */
async function prune(args) {
  const keepLogs = args.includes("--keep-logs");
  const keepSecrets = args.includes("--keep-secrets");
  const dryRun = args.includes("--dry-run");
  const frontend = join(CHAT_APP, "frontend");
  const modelsDir = MODELS_DIR; // the Hugging Face downloads: never removed here

  const paths = [
    ...(keepLogs ? [] : [LOG_DIR]),
    ...(keepSecrets ? [] : [SECRETS_DIR]),
    join(frontend, "node_modules"), join(frontend, "dist"), join(frontend, "test-results"),
    join(frontend, "playwright-report"), join(ROOT, "docs", "_build"), join(ROOT, "docs", "api", "generated"),
    join(ROOT, ".pytest_cache"), join(CHAT_APP, "backend", ".pytest_cache"), join(ROOT, "tiny_agent.egg-info"), join(ROOT, "code", "tiny_agent.egg-info"),
    ...findPycache(ROOT),
  ].filter((p) => existsSync(p));

  console.log("This will remove:");
  const volumes = relatedVolumes();
  console.log("  Docker (project chat-app only): containers and networks");
  console.log(`  Docker volumes: ${volumes.length ? volumes.join(", ") : "(none right now)"}`);
  for (const p of paths) console.log("  " + p);
  console.log(`\nThis will NOT touch: your downloaded models (${modelsDir}), Docker images, source code, .env, .venv, git.`);
  if (keepLogs) console.log("  (keeping logs, --keep-logs)");
  if (keepSecrets) console.log("  (keeping secrets, --keep-secrets)");

  if (dryRun) return console.log("\nDry run: nothing was removed.");
  if (!args.includes("--yes")) {
    if (!process.stdin.isTTY) fail("Refusing to prune without confirmation. Re-run with --yes (or --dry-run to see the plan).");
    const rl = createInterface({ input: process.stdin, output: process.stdout });
    const answer = (await rl.question("\nType 'yes' to continue: ")).trim().toLowerCase();
    rl.close();
    if (answer !== "yes") return console.log("Cancelled. Nothing was removed.");
  }

  // Docker first: the containers hold the secrets and the OpenObserve data folder open.
  console.log("\nRemoving the Docker setup...");
  const code = compose("down", "--remove-orphans", "--volumes"); // no --rmi: images stay
  if (code) console.log("Warning: docker compose down reported a problem (is Docker running?). Continuing with the files.");
  // `down --volumes` removes the named volumes in the compose file. Sweep up any other volume
  // compose labelled for this project (for example an anonymous volume from an old container).
  for (const volume of relatedVolumes()) {
    const removed = spawnSync("docker", ["volume", "rm", "-f", volume], { encoding: "utf8" });
    console.log(removed.status === 0 ? `removed volume ${volume}` : `could not remove volume ${volume}: ${removed.stderr.trim()}`);
  }

  for (const p of paths) {
    try {
      rmSync(p, { recursive: true, force: true });
      console.log("removed " + p);
    } catch (e) {
      console.log(`could not remove ${p}: ${e.message}`);
    }
  }
  console.log(`\nDone. The models in ${modelsDir} were left alone.`);
  console.log("Start again with: node execute.mjs up   (the models are not downloaded again)");
}

const [command = "help", ...args] = process.argv.slice(2);

switch (command) {
  case "up": {
    // OpenObserve's data folder must exist and be yours before Docker mounts it.
    mkdirSync(join(LOG_DIR, "openobserve"), { recursive: true });
    mkdirSync(MODELS_DIR, { recursive: true });
    const code = compose("up", "-d", ...(args.includes("--no-build") ? [] : ["--build"]), "--remove-orphans");
    if (code) process.exit(code);
    await waitUntilHealthy();
    await waitForObserve();
    await printModels();
    printUrls();
    break;
  }
  case "observe":
    ensureSecrets();
    printUrls();
    console.log("\nOpens already logged in: no login screen (this machine only; the port is bound to 127.0.0.1).");
    console.log(`If you ever need the login: ${readSecret(SECRET_USER)}`);
    console.log(args.includes("--password") ? `Password: ${readSecret(SECRET_PASSWORD)}` : `Password: stored in ${SECRET_PASSWORD}  (show it with: node execute.mjs observe --password)`);
    console.log("Streams (Logs > pick a stream):");
    console.log("  docker      the chat-app containers: gateway, frontend, backend, llama");
    console.log("  tiny_agent  tiny-agent (service tiny-agent-cli) and main.py (service tiny-agent-main)");
    console.log(`Files: ${join(LOG_DIR, "cli.log")}, ${join(LOG_DIR, "main.log")}; OpenObserve data: ${join(LOG_DIR, "openobserve")}`);
    break;
  case "down":
    process.exit(compose("down", "--remove-orphans"));
  case "restart":
    process.exit(compose("restart", ...args));
  case "rebuild": {
    const code = compose("up", "-d", "--build", "--force-recreate", ...args);
    if (code) process.exit(code);
    await waitUntilHealthy();
    break;
  }
  case "clean":
    process.exit(compose("down", "--remove-orphans", "--volumes", "--rmi", "local"));
  case "prune":
    await prune(args);
    break;
  case "status": {
    compose("ps", "--all");
    await printModels();
    break;
  }
  case "logs":
    process.exit(compose("logs", "-f", "--tail", "100", ...args));
  case "health":
    // Set the code and let Node exit on its own: process.exit() right after fetch
    // timeouts can trip a libuv assertion on Windows.
    process.exitCode = (await health()) ? 0 : 1;
    break;
  case "config":
    process.exit(compose("config"));
  case "models":
    await printModels();
    break;
  case "download":
    await download(args);
    break;
  case "load": {
    const model = args[0];
    if (!model) fail("Usage: node execute.mjs load <model>   (see: node execute.mjs models)");
    console.log(`Loading ${model} (the first time it is downloaded from Hugging Face: several GB, minutes)...`);
    const { ok, status, body } = await api(
      "/api/models/load",
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model }) },
      2_000_000, // the first load downloads the model from Hugging Face
    );
    if (!ok) fail(`${body?.error?.message || `HTTP ${status}`}`);
    console.log(`${model} is loaded.`);
    await printModels();
    break;
  }
  case "test": {
    const split = args.indexOf("--");
    const own = split < 0 ? args : args.slice(0, split);
    const extra = split < 0 ? [] : args.slice(split + 1);
    const which = own[0] || "all";
    const known = ["all", "backend", "agent", "llm", "mocked", "live", "e2e", "ias", "list"];
    if (!known.includes(which)) fail(`Unknown test target '${which}'. Use one of: ${known.join(", ")}`);

    const agent = (more = []) => run("python", ["-m", "pytest", "-q", "code/tiny_agent/tests", ...more], { cwd: ROOT });
    const llm = (more = []) => run("python", ["-m", "pytest", "-q", "code/llm_client/tests", ...more], { cwd: ROOT });
    const backend = (more = []) => run("python", ["-m", "pytest", "-q", ...more], { cwd: join(CHAT_APP, "backend") });
    // -rs lists why model tests were skipped (stack down, model not loaded).
    const ias = (more = []) =>
      run("python", ["-m", "pytest", "-q", "-rs", "experimentation/illustrated_agents/tests", ...more], { cwd: ROOT });
    const playwright = (project, more = []) =>
      run("npx", ["playwright", "test", ...(project ? ["--project", project] : []), ...more], {
        cwd: join(CHAT_APP, "frontend"),
        shell: process.platform === "win32", // npx is a .cmd file on Windows
      });

    if (which === "list") {
      backend(["--collect-only", "-q"]);
      agent(["--collect-only", "-q"]);
      llm(["--collect-only", "-q"]);
      ias(["--collect-only", "-q"]);
      process.exitCode = playwright(null, ["--list"]);
      break;
    }

    const results = [];
    const stackIsUp = async () => {
      try {
        return (await api("/health", {}, 3000)).body?.llama === "up";
      } catch {
        return false;
      }
    };

    if (which === "backend" || which === "all") results.push(["backend", backend(extra)]);
    if (which === "agent" || which === "all") results.push(["agent", agent(extra)]);
    if (which === "llm" || which === "all") results.push(["llm", llm(extra)]);
    if (which === "mocked" || which === "e2e" || which === "all") results.push(["mocked", playwright("mocked", extra)]);
    if (which === "live" || which === "e2e" || which === "all") {
      if (await stackIsUp()) results.push(["live", playwright("live", extra)]);
      else if (which === "all") console.log("\nSkipping live tests: the stack is not running (node execute.mjs up).");
      else fail(`The live tests need the running stack at ${URL_BASE}. Start it with: node execute.mjs up`);
    }
    // The scripts call the real model, so "all" runs only the quick ones; "test ias" runs every one.
    if (which === "ias") results.push(["ias", ias(extra)]);
    else if (which === "all" && (await stackIsUp())) results.push(["ias (quick)", ias(["-m", "not slow", ...extra])]);

    console.log("\nTest summary");
    for (const [name, code] of results) console.log(`  ${code === 0 ? "PASS" : "FAIL"}  ${name}`);
    process.exitCode = results.some(([, code]) => code !== 0) ? 1 : 0;
    break;
  }
  case "open":
    console.log(URL_BASE);
    break;
  case "help":
  case "--help":
  case "-h":
    console.log(HELP);
    break;
  default:
    console.error(`Unknown command: ${command}`);
    console.log(HELP);
    process.exit(2);
}
