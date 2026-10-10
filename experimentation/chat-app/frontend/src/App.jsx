import ChatLog from "./components/ChatLog.jsx";
import Composer from "./components/Composer.jsx";
import Header from "./components/Header.jsx";
import { Btn, Dot } from "./components/ui.jsx";
import { useAgentChat } from "./hooks/useAgentChat.js";
import { useKnowledgeBase } from "./hooks/useKnowledgeBase.js";
import { useModels } from "./hooks/useModels.js";
import { useTheme } from "./hooks/useTheme.js";

/* TinyAgent Chat: an agent chat for local models served by llama.cpp. The model can call
   tools, and every call shows up as a step you can open. */
export default function App() {
  const { toggle } = useTheme();
  const models = useModels();
  const { kb, error: kbError } = useKnowledgeBase();
  const chat = useAgentChat({ model: models.model, ready: models.ready, kb, onDone: models.refresh });

  const error = models.error || kbError;
  const idle = !models.ready && !models.switching && !!models.model; // selected but not loaded

  return (
    <div className="mx-auto flex h-dvh max-w-3xl flex-col px-4 sm:px-6">
      <Header
        models={models}
        busy={chat.busy}
        canReset={chat.turns.length > 0}
        onReset={() => { chat.reset(); models.setError(""); }}
        onToggleTheme={toggle}
      />

      <main className="relative flex min-h-0 flex-1 flex-col" inert={!!models.switching} aria-busy={!!models.switching}>
        <ChatLog turns={chat.turns} model={models.model} ready={models.ready} onAsk={chat.send} />
        {error && <p role="alert" data-testid="error" className="m-0 pb-2 text-small text-danger">{error}</p>}
        {idle && (
          <div data-testid="load-banner" className="mb-2 flex items-center justify-between gap-3 rounded-panel border border-line bg-raise px-4 py-2 text-small">
            <span><b className="font-medium">{models.model}</b> is not loaded, so chat is off.</span>
            <Btn onClick={models.reload}>Load</Btn>
          </div>
        )}
        <Composer
          busy={chat.busy}
          disabled={!models.ready}
          placeholder={models.switching ? `Loading ${models.switching}…` : models.ready ? "Message the agent" : "Load a model to chat"}
          onSend={chat.send}
          onStop={chat.stop}
        />
      </main>

      {models.switching && (
        <div role="status" data-testid="loading-veil"
          className="fixed inset-0 z-20 flex cursor-progress items-center justify-center gap-3 bg-ground/70 text-ink-2
            backdrop-blur-sm backdrop-saturate-[.3]">
          <Dot running />
          <span>Loading <b className="font-medium text-ink">{models.switching}</b>…</span>
        </div>
      )}
    </div>
  );
}
