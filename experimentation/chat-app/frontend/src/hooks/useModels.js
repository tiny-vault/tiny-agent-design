import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { listModels, loadModel } from "../lib/llama.js";

const POLL_MS = 5000;

function savedModel() {
  try {
    return localStorage.getItem("model");
  } catch {
    return null;
  }
}

/* The router's models and the selected one. Chatting is allowed only when
   `ready`: a model is selected, the router reports it "loaded", and no switch
   is in progress. Picking a model asks the backend to load it and waits. */
export function useModels() {
  const [models, setModels] = useState([]); // [{ id, status }]
  const [model, setModel] = useState("");
  const [switching, setSwitching] = useState(""); // the model being loaded
  const [error, setError] = useState("");
  const first = useRef(true);
  const autoLoad = useRef(true); // load the starting model once if it is not loaded

  const loaded = useMemo(() => new Set(models.filter((m) => m.status === "loaded").map((m) => m.id)), [models]);
  const statusOf = useCallback((id) => models.find((m) => m.id === id)?.status, [models]);

  const refresh = useCallback(async () => {
    try {
      const list = await listModels();
      setModels(list);
      setError((e) => (e.startsWith("Cannot reach") ? "" : e));
      setModel((current) => {
        const ids = list.map((m) => m.id);
        if (first.current) {
          first.current = false;
          const saved = savedModel();
          return ids.includes(saved) ? saved : list.find((m) => m.status === "loaded")?.id || ids[0] || "";
        }
        return current || list.find((m) => m.status === "loaded")?.id || ids[0] || "";
      });
    } catch (e) {
      setError(`Cannot reach the backend: ${e.message}`);
    }
  }, []);

  const switchTo = useCallback(
    async (m) => {
      setSwitching(m);
      setError("");
      try {
        await loadModel(m);
        setModel(m);
        try {
          localStorage.setItem("model", m);
        } catch {
          /* remembering the choice is optional */
        }
      } catch (e) {
        setError(`Could not load ${m}: ${e.message}`);
      } finally {
        setSwitching("");
        await refresh();
      }
    },
    [refresh],
  );

  const pick = useCallback(
    (m) => {
      if (switching || (m === model && loaded.has(m))) return;
      switchTo(m);
    },
    [switching, model, loaded, switchTo],
  );

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Pick up changes made outside this page (a model unloaded, another tab).
  useEffect(() => {
    if (switching) return;
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh, switching]);

  // The first selection is loaded automatically so the page is usable at once.
  useEffect(() => {
    if (!autoLoad.current || !model || !statusOf(model) || switching) return;
    autoLoad.current = false;
    if (statusOf(model) !== "loaded") switchTo(model);
  }, [model, statusOf, switching, switchTo]);

  const ready = !!model && !switching && loaded.has(model);
  return { models, loaded, model, switching, ready, error, setError, refresh, pick, reload: () => switchTo(model), statusOf };
}
