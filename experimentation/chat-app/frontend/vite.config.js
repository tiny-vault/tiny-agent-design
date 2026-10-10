import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In dev, API calls go to the gateway (docker compose publishes it on :3200),
// the same way they do in production. Override with VITE_API_TARGET.
const target = process.env.VITE_API_TARGET || "http://localhost:3200";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { "/api": target, "/v1": target, "/health": target },
  },
});
