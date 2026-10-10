import { useEffect, useState } from "react";

/* The small TinyAgent knowledge base that search_docs reads (public/kb.json). */
export function useKnowledgeBase() {
  const [kb, setKb] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    fetch("/kb.json")
      .then((r) => r.json())
      .then((data) => alive && setKb(data))
      .catch(() => alive && setError("Could not load kb.json"));
    return () => {
      alive = false;
    };
  }, []);

  return { kb, error };
}
