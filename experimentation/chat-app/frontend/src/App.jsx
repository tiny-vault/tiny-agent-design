import ChatLog from "./components/ChatLog.jsx";
import Composer from "./components/Composer.jsx";
import Header from "./components/Header.jsx";
import { Dot } from "./components/ui.jsx";
import { useAgentChat } from "./hooks/useAgentChat.js";
import { useKnowledgeBase } from "./hooks/useKnowledgeBase.js";
import { useModels } from "./hooks/useModels.js";
import { useTheme } from "./hooks/useTheme.js";

/* TinyAgent Chat: an agent chat for a local Ollama model. The model can call
   tools, and every call shows up as a step you can open. */
export default function App() {
  const { toggle } = useTheme();
  const models = useModels();
  const { kb, error: kbError } = useKnowledgeBase();
  const chat = useAgentChat({ model: models.model, kb, onDone: models.refresh });

  const error = models.error || kbError;
  const frozen = !!models.loading; // while a model loads, the chat pane is frozen under a veil

  return (
    <div className="mx-auto flex h-dvh max-w-3xl flex-col px-4 sm:px-6">
      <Header
        models={models}
        busy={chat.busy}
        canReset={chat.turns.length > 0}
        onReset={() => { chat.reset(); models.setError(""); }}
        onToggleTheme={toggle}
      />

      <main className="relative flex min-h-0 flex-1 flex-col" inert={frozen} aria-busy={frozen}>
        <ChatLog turns={chat.turns} model={models.model} onAsk={chat.send} />
        {error && <p className="m-0 pb-2 text-small text-danger">{error}</p>}
        <Composer
          busy={chat.busy}
          disabled={!models.model || frozen}
          onSend={chat.send}
          onStop={chat.stop}
        />
      </main>

      {frozen && (
        <div role="status"
          className="fixed inset-0 z-20 flex cursor-progress items-center justify-center gap-3 bg-ground/70 text-ink-2
            backdrop-blur-sm backdrop-saturate-[.3]">
          <Dot running />
          <span>Loading <b className="font-medium text-ink">{models.loading}</b>…</span>
        </div>
      )}
    </div>
  );
}
