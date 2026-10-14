"""Settings: every ``TINYAGENT_*`` value in one place, read by ``pydantic-settings``.

A setting is a field of :class:`Settings`; its environment variable is ``TINYAGENT_`` plus the field name in capitals
(``llm_url`` is ``TINYAGENT_LLM_URL``). The values come from the real environment first, then from the ``.env`` file
(``tiny_agent/.env``, or the file named by ``TINYAGENT_ENV_FILE``), then from the defaults below, and pydantic does
the parsing: ``1``, ``true``, ``yes`` and ``on`` are true, an empty variable counts as not set, a wrong type is
an error that names the setting.

Everything else asks ``Settings.from_env()`` (never ``os.environ``). Pass a ``Settings(...)`` to a function to
configure it in code instead, for example in a test.

Explicit constructor values win, but ``Settings()`` itself still reads the
environment. ``from_env`` also chooses the dotenv file; it does not copy its
values into ``os.environ``. Instances are frozen rather than mutated in place.

``memory`` resolves bank, rag, summary and trim flags in that order.
``tools_kind`` gives native calling priority over prompt calling. These choose
components rather than run them. ``resolve_api_key`` prefers the first
nonblank, non-comment line in a key file to the secret variable. ``SecretStr``
masks representations but does not make logging the resolved string safe.
Missing or empty key files raise; this module makes no model requests.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and costs about 2 s per request
DEFAULT_LLM_URL = "http://127.0.0.1:3200/v1"  # the Docker gateway
DEFAULT_EMBEDDING_URL = "http://127.0.0.1:3201/v1"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma-300m"

MemoryKind = Literal["plain", "trim", "summary", "rag", "bank"]


def default_env_file() -> Path:
    """The ``.env`` file to read: ``$TINYAGENT_ENV_FILE`` if set, else ``tiny_agent/.env``."""
    configured = os.environ.get("TINYAGENT_ENV_FILE")
    return Path(configured) if configured else Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    """What the agent, the CLI, the demo, the model check and the logger can be configured with.

    Attributes:
        model: The chat model name; ``None`` means "whichever is loaded".
        llm_url: The model server's API root, ending in ``/v1``.
        api_key: The API key itself; prefer ``api_key_file``. See :meth:`resolve_api_key`.
        api_key_file: A file whose first line that is not blank or a comment is the API key.
        think: Whether the model may reason before answering.
        system: A system message that sets the model's role.
        trim_memory, summary_memory, rag_memory, bank_memory: Switch on a memory variant in the demo; see :attr:`memory`.
        tools: Bind the built-in tools to the agent, told to the model in its prompt (see :attr:`tools_kind`).
        native_tools: Bind them for a model that learned tool calling; wins over ``tools``.
        embedding_model: The embedding model for the ``rag`` and ``bank`` memories.
        embedding_url: Where it is served.
        a2a: Let the orchestrator reach its specialists over the A2A protocol instead of in-process.
        log_dir: The folder for the log files; ``None`` means the default folder.
        log_level: ``DEBUG``, ``INFO``, ``WARNING`` or ``ERROR``.
    """

    model_config = SettingsConfigDict(
        env_prefix="TINYAGENT_",
        env_ignore_empty=True,  # TINYAGENT_X= (empty) counts as not set
        extra="ignore",  # the .env file may hold other names
        protected_namespaces=(),  # a field called "model" is fine
        frozen=True,
    )

    model: str | None = None
    llm_url: str = DEFAULT_LLM_URL
    api_key: SecretStr | None = None
    api_key_file: Path | None = None
    think: bool = False
    system: str | None = None
    trim_memory: bool = False
    summary_memory: bool = False
    rag_memory: bool = False
    bank_memory: bool = False
    tools: bool = False
    native_tools: bool = False
    a2a: bool = False
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_url: str = DEFAULT_EMBEDDING_URL
    log_dir: Path | None = None
    log_level: str = "INFO"

    @field_validator("*", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        """Ignore spaces around a value, so ' On ' is on."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        """Normalize spelling; the logger handles unknown level names."""
        return value.upper()

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        """Read the settings: the real environment, then the ``.env`` file, then the defaults.

        Args:
            env_file: The file to read; ``None`` means :func:`default_env_file`. A missing file is fine.

        Raises:
            pydantic.ValidationError: A value has the wrong type; the message names the setting.
        """
        return cls(_env_file=env_file or default_env_file())  # pyright: ignore[reportCallIssue]

    @property
    def memory(self) -> MemoryKind:
        """Which memory the demo uses. When several flags are on the strongest wins: bank, rag, summary, trim."""
        for kind in ("bank", "rag", "summary", "trim"):
            if getattr(self, f"{kind}_memory"):
                return kind  # pyright: ignore[reportReturnType]
        return "plain"

    @property
    def tools_kind(self) -> Literal["native", "prompt"] | None:
        """Which tools the demo binds: ``"native"`` (``native_tools``), ``"prompt"`` (``tools``) or ``None``."""
        return "native" if self.native_tools else "prompt" if self.tools else None

    def resolve_api_key(self) -> str | None:
        """The API key for a server that needs one, or ``None``.

        Read from ``api_key_file`` (its first line that is not blank or a comment), or else from ``api_key``.
        The file form keeps the key out of ``.env``. The Docker stack needs no key: its backend holds it.

        Raises:
            OSError: The key file cannot be read.
            ValueError: The key file has no key in it.
        """
        if self.api_key_file:
            with open(self.api_key_file, encoding="utf-8-sig") as handle:
                for line in handle:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        return line
            raise ValueError(f"{self.api_key_file} is empty")
        return self.api_key.get_secret_value() if self.api_key else None
