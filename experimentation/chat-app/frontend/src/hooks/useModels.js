import { useCallback, useEffect, useRef, useState } from "react";
import { listModels, setLoaded } from "../lib/ollama.js";
import { OLLAMA } from "../lib/constants.js";

function savedModel() {
  try {
    return localStorage.getItem("model");
  } catch {
    return null;
  }
}

/* Installed models, which are loaded in memory, the selected one, and the
   load/unload actions. `loading` is the name of the model being loaded. */
export function useModels() {
  const [models, setModels] = useState([]);
  const [loaded, setLoadedSet] = useState(() => new Set());
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");
  const first = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const { models: all, loaded: inMemory } = await listModels();
      setModels(all);
      setLoadedSet(inMemory);
      setError("");
      setModel((current) => {
        if (first.current) {
          first.current = false;
          const saved = savedModel();
          return all.includes(saved) ? saved : [...inMemory][0] || all[0] || "";
        }
        return current || [...inMemory][0] || all[0] || "";
      });
    } catch (e) {
      setError(`Cannot reach Ollama at ${OLLAMA}: ${e.message}`);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const load = useCallback(
    async (m) => {
      setLoading(m);
      setError("");
      try {
        await setLoaded(m, "5m");
      } catch (e) {
        setError(`Could not load ${m}: ${e.message}`);
      } finally {
        setLoading("");
        await refresh();
      }
    },
    [refresh],
  );

  const unload = useCallback(
    async (m) => {
      setError("");
      try {
        await setLoaded(m, 0);
      } catch (e) {
        setError(`Could not unload ${m}: ${e.message}`);
      }
      await refresh();
    },
    [refresh],
  );

  const pick = useCallback(
    (m) => {
      setModel(m);
      try {
        localStorage.setItem("model", m);
      } catch {
        /* remembering the choice is optional */
      }
      if (!loaded.has(m) && !loading) load(m);
    },
    [loaded, loading, load],
  );

  return { models, loaded, model, loading, error, setError, refresh, pick, unload };
}
