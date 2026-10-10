import ModelMenu from "./ModelMenu.jsx";
import { Btn, Icon } from "./ui.jsx";

export default function Header({ models, busy, canReset, onReset, onToggleTheme }) {
  const locked = busy || !!models.switching;
  return (
    <header className="flex items-center justify-between gap-3 py-4">
      <span className="font-brand text-title font-semibold tracking-tight">TinyAgent</span>
      <div className="flex items-center gap-2">
        <ModelMenu
          models={models.models}
          model={models.model}
          switching={models.switching}
          disabled={busy}
          onOpen={models.refresh}
          onPick={models.pick}
        />
        <Btn icon onClick={onReset} disabled={locked || !canReset} aria-label="New conversation" title="New conversation">
          <Icon name="plus" />
        </Btn>
        <Btn icon onClick={onToggleTheme} aria-label="Toggle theme" title="Toggle theme">
          <Icon name="sun" />
        </Btn>
      </div>
    </header>
  );
}
