#!/usr/bin/env node
// Create the Docker secret files the chat app's compose file refers to, if they do not exist yet.
// Run it once before a hand-run `docker compose up` in this folder:   node deploy/chat-app/init-secrets.mjs
// `node execute.mjs up` calls the same function for you. Needs only Node 18+, no packages.
//
// The files go in experimentation/chat-app/secrets/ (git-ignored) and are mounted into the containers at
// /run/secrets/<name>. A secret never goes in the compose file, an environment variable or an image.
//   openobserve_user      the OpenObserve login name ($OPENOBSERVE_USER, default admin@example.com)
//   openobserve_password  a random password that meets OpenObserve's policy; never printed here
//   llama_api_key         a random key shared by llama-server (which requires it) and the backend (which sends it)
// Existing files are kept, except an OpenObserve password that breaks its policy (it refuses to start with one).

import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/* The secrets folder: <repo>/experimentation/chat-app/secrets (this file is <repo>/deploy/chat-app/). */
export const SECRETS_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "experimentation", "chat-app", "secrets");
export const SECRET_USER = join(SECRETS_DIR, "openobserve_user");
export const SECRET_PASSWORD = join(SECRETS_DIR, "openobserve_password");
export const SECRET_LLAMA_KEY = join(SECRETS_DIR, "llama_api_key");

/* OpenObserve refuses to start unless the root password is 8-128 characters with a lowercase
   letter, an uppercase letter, a digit and a special character. */
const passwordIsStrong = (p) => p.length >= 8 && p.length <= 128 && /[a-z]/.test(p) && /[A-Z]/.test(p) && /\d/.test(p) && /[^A-Za-z0-9]/.test(p);

/* Random, and always meets that policy: alphanumeric random part plus one of each class. */
const generatePassword = () => randomBytes(18).toString("base64url").replace(/[-_]/g, "x") + "Aa9!";

export function ensureSecrets() {
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

// Run as a script (not imported by execute.mjs): create the files and say where they are.
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  ensureSecrets();
  console.log(`Secrets are in ${SECRETS_DIR}. Now: docker compose up -d --build   (from deploy/chat-app)`);
}
