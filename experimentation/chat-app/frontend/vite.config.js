import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Ollama accepts localhost origins, so the dev server (http://localhost:5173)
// can call it directly. No proxy is needed.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173 },
});
