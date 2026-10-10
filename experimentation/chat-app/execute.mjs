#!/usr/bin/env node
// Control the chat app's Docker deployment.  Usage: node execute.mjs <command> [args]
// No dependencies; needs Node 18+ and Docker with the compose plugin.

import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));
const DEPLOY = join(ROOT, "deploy");
const COMPOSE_FILE = join(DEPLOY, "docker-compose.yaml");
const PORT = process.env.GATEWAY_PORT || "3200";
const URL_BASE = `http://localhost:${PORT}`;

const HELP = `
TinyAgent chat deployment

  node execute.mjs <command> [args]

Lifecycle
  up [--no-build]     Build and start everything (downloads missing models first)
  down                Stop and remove the containers (models and images are kept)
  restart [service]   Restart one service or all of them
  rebuild [service]   Rebuild images and recreate containers
  clean               Stop, remove containers, volumes and locally built images

Inspect
  status              Container state plus the router's model status
  logs [service]      Follow logs (services: pull-models, llama, backend, frontend, gateway)
  health              Check the gateway, backend and llama.cpp
  config              Print the resolved compose configuration

Models (through the gateway)
  models              List models and whether each is loaded
  load <model>        Load a model through the router and wait until it is loaded

Tests  (extra arguments after "--" go to the test runner)
  test                Everything that can run: backend + mocked (+ live if the stack is up)
  test backend        Backend pytest suite (fake llama router; no Docker needed)
  test mocked         Playwright, API stubbed in the browser (current source; no Docker needed)
  test live           Playwright against the REAL stack: switches real models (slow; needs "up")
  test e2e            mocked + live
  test list           List every test without running it

  Examples
    node execute.mjs test backend -- -k load        backend tests with "load" in the name
    node execute.mjs test mocked -- --grep Stop     mocked tests whose title has "Stop"
    node execute.mjs test mocked -- --headed        watch the browser

  open                Print the app URL
  help                Show this text

Settings come from deploy/.env or the environment: GATEWAY_PORT, PULL_MODELS,
LLAMA_MODELS_MAX, OLLAMA_MODELS_DIR (see deploy/.env.example).
`;

function run(cmd, args, options = {}) {
  const result = spawnSync(cmd, args, { stdio: "inherit", shell: false, ...options });
  if (result.error) {
    if (result.error.code === "ENOENT") fail(`'${cmd}' was not found. Is it installed and on your PATH?`);
    fail(result.error.message);
  }
  return result.status ?? 1;
}

const compose = (...args) =>
  run("docker", ["compose", "-f", COMPOSE_FILE, "--project-directory", DEPLOY, ...args]);

function fail(message, code = 1) {
  console.error(`\nError: ${message}`);
  process.exit(code);
}

async function api(path, options = {}, timeoutMs = 10_000) {
  const res = await fetch(URL_BASE + path, { ...options, signal: AbortSignal.timeout(timeoutMs) });
  const text = await res.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  return { ok: res.ok, status: res.status, body };
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

const [command = "help", ...args] = process.argv.slice(2);

switch (command) {
  case "up": {
    const code = compose("up", "-d", ...(args.includes("--no-build") ? [] : ["--build"]), "--remove-orphans");
    if (code) process.exit(code);
    await waitUntilHealthy();
    await printModels();
    console.log(`\nApp: ${URL_BASE}`);
    break;
  }
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
  case "load": {
    const model = args[0];
    if (!model) fail("Usage: node execute.mjs load <model>   (see: node execute.mjs models)");
    console.log(`Loading ${model} (this can take a minute)...`);
    const { ok, status, body } = await api(
      "/api/models/load",
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model }) },
      330_000,
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
    const known = ["all", "backend", "mocked", "live", "e2e", "list"];
    if (!known.includes(which)) fail(`Unknown test target '${which}'. Use one of: ${known.join(", ")}`);

    const backend = (more = []) => run("python", ["-m", "pytest", "-q", ...more], { cwd: join(ROOT, "backend") });
    const playwright = (project, more = []) =>
      run("npx", ["playwright", "test", ...(project ? ["--project", project] : []), ...more], {
        cwd: join(ROOT, "frontend"),
        shell: process.platform === "win32", // npx is a .cmd file on Windows
      });

    if (which === "list") {
      backend(["--collect-only", "-q"]);
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
    if (which === "mocked" || which === "e2e" || which === "all") results.push(["mocked", playwright("mocked", extra)]);
    if (which === "live" || which === "e2e" || which === "all") {
      if (await stackIsUp()) results.push(["live", playwright("live", extra)]);
      else if (which === "all") console.log("\nSkipping live tests: the stack is not running (node execute.mjs up).");
      else fail(`The live tests need the running stack at ${URL_BASE}. Start it with: node execute.mjs up`);
    }

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
