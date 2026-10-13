"""Tests for ``RAGMemory`` (documents retrieved by meaning and put in front of each question) and its flag.

All against real models, deployed in Docker (``node execute.mjs up llm-client``): EmbeddingGemma in the
``llama-embed`` container at ``TINYAGENT_EMBEDDING_URL`` (default ``http://127.0.0.1:3201/v1``) and whichever
chat model is loaded on the model server at ``TINYAGENT_LLM_URL`` (default ``http://127.0.0.1:3200/v1``; load one
with ``node execute.mjs llm-client load qwen3.5-4b``). There are no fakes: when a server is not up, the tests
that need it are skipped and say which command starts it. Chat replies are not deterministic, so those tests
check for a key word, not for exact text.
"""

import os

import pytest

from tiny_agent import main as demo
from tiny_agent.agent.memory import Memory
from tiny_agent.agent.tiny_agent import TinyAgent
from tiny_agent.llm_client import check
from tiny_agent.llm_client.embedding import EmbeddingModel
from tiny_agent.llm_client.llm import LLM
from tiny_agent.memory.rag import RAGMemory
from tiny_agent.memory.summarization import SummarizationMemory
from tiny_agent.memory.trimming import TrimmingMemory

EMBEDDING_URL = os.environ.get("TINYAGENT_EMBEDDING_URL", "http://127.0.0.1:3201/v1")
LLM_URL = os.environ.get("TINYAGENT_LLM_URL", check.DEFAULT_URL)
DOCS = demo.DOCUMENTS


@pytest.fixture(scope="module")
def embedder():
    """The real EmbeddingGemma client, or a skip when its container is not running."""
    model = EmbeddingModel("embeddinggemma-300m", base_url=EMBEDDING_URL)
    try:
        model.embed("ping")
    except RuntimeError as error:
        pytest.skip(f"no embedding model at {EMBEDDING_URL} ({error}); start it with: node execute.mjs up llm-client")
    return model


@pytest.fixture(scope="module")
def chat_model():
    """The name of a chat model that is loaded on the real server, or a skip when there is none."""
    loaded = check.loaded_models(LLM_URL)
    if not loaded:
        pytest.skip(f"no chat model loaded at {LLM_URL}; run: node execute.mjs llm-client load qwen3.5-4b")
    return loaded[0]


@pytest.fixture
def llm(chat_model):
    """A real chat model."""
    return LLM(model=chat_model, base_url=LLM_URL)


@pytest.fixture(scope="module")
def memory(embedder):
    """A ``RAGMemory`` over the ten demo documents, for tests that only search. Module-scoped because embedding
    the documents takes a moment; tests that add messages build their own."""
    return RAGMemory(embedder, DOCS)


def texts(mem: Memory) -> list[str]:
    """The message texts in order."""
    return [message["content"] for message in mem.get_messages()]


# ---- the memory on its own ----------------------------------------------------------

def test_it_is_a_memory_and_starts_empty(embedder):
    fresh = RAGMemory(embedder, DOCS[:2])
    assert isinstance(fresh, Memory) and fresh.get_messages() == []


def test_every_document_is_embedded_at_construction(memory):
    assert memory.documents == DOCS
    assert len(memory.embeddings) == len(DOCS)
    assert all(len(vector) == 768 for vector in memory.embeddings)


def test_no_documents_means_nothing_to_retrieve(embedder):
    empty = RAGMemory(embedder, [])
    assert empty.embeddings == [] and empty.search("anything") == []


def test_search_finds_sarahs_favorite_animal(memory):
    results = memory.search("What is Sarah's favorite animal?")
    assert len(results) == 3
    assert results[0] == "Sarah favorite animal is flamingos."


def test_search_tells_the_two_people_apart(memory):
    assert memory.search("What instrument does Ilse play?")[0] == "Ilse plays the cello in a local string quartet."
    assert memory.search("What is Ilse's favorite animal?")[0] == "Ilse's favorite animal is dolphins."
    assert memory.search("Where does Sarah live?")[0] == "Sarah lives in Lisbon, Portugal."


def test_search_with_fewer_than_three_documents_returns_them_all(embedder):
    small = RAGMemory(embedder, ["Sarah favorite animal is flamingos.", "Ilse plays the cello."])
    assert sorted(small.search("flamingos")) == sorted(small.documents)


def test_cosine_of_the_same_direction_is_one_and_of_perpendicular_vectors_zero(memory):
    assert memory._cosine([2.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)
    assert memory._cosine([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
    assert memory._cosine([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


def test_a_user_message_is_stored_with_its_context_in_front(embedder):
    mem = RAGMemory(embedder, DOCS)
    mem.add("user", "What is Sarah's favorite animal?")
    [stored] = mem.get_messages()
    assert stored["role"] == "user"
    assert stored["content"].startswith("Context:\n")
    assert "Sarah favorite animal is flamingos." in stored["content"]
    assert stored["content"].endswith("\n\nQuestion: What is Sarah's favorite animal?")


def test_other_roles_are_stored_unchanged(embedder):
    mem = RAGMemory(embedder, DOCS[:2])
    mem.add("system", "Be brief.")
    mem.add("assistant", "a1")
    mem.add("tool", "result")
    assert texts(mem) == ["Be brief.", "a1", "result"]


def test_each_user_message_gets_its_own_context_and_all_are_kept(embedder):
    mem = RAGMemory(embedder, DOCS)
    mem.add("user", "What instrument does Ilse play?")
    mem.add("assistant", "a1")
    mem.add("user", "Where does Sarah live?")
    first, _, second = texts(mem)
    assert "cello" in first.split("Question:")[0]
    assert "Lisbon" in second.split("Question:")[0]


def test_a_tool_call_is_stored_like_in_memory(embedder):
    call = {"id": "1", "function": {"name": "f", "arguments": "{}"}}
    mem = RAGMemory(embedder, DOCS[:1])
    mem.add("assistant", "", tool_call=call)
    assert mem.get_messages() == [{"role": "assistant", "content": "", "tool_calls": [call]}]


def test_get_messages_returns_the_history_itself(embedder):
    mem = RAGMemory(embedder, DOCS[:1])
    mem.add("user", "hi")
    assert mem.get_messages() is mem.messages


def test_an_unreachable_embedding_server_raises_and_stores_nothing(embedder, dead_url):
    mem = RAGMemory(embedder, DOCS[:1])
    mem.embedding_model = EmbeddingModel("embeddinggemma-300m", base_url=dead_url)
    with pytest.raises(RuntimeError, match="Cannot reach the embedding model"):
        mem.add("user", "hi")
    assert mem.get_messages() == []


# ---- the memory inside a TinyAgent ----------------------------------------------------------

def test_the_model_answers_from_the_retrieved_document(embedder, llm):
    agent = TinyAgent(llm, RAGMemory(embedder, DOCS))
    assert "flamingo" in agent.run("What is Sarah's favorite animal?").lower()


def test_the_model_tells_the_two_people_apart(embedder, llm):
    agent = TinyAgent(llm, RAGMemory(embedder, DOCS))
    assert "dolphin" in agent.run("What is Ilse's favorite animal?").lower()


def test_a_second_run_keeps_the_first_question_with_its_context(embedder, llm):
    mem = RAGMemory(embedder, DOCS)
    agent = TinyAgent(llm, mem)
    agent.run("What is Sarah's favorite animal?")
    agent.run("What instrument does Ilse play?")
    messages = mem.get_messages()
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert "flamingos" in messages[0]["content"] and "cello" in messages[2]["content"]


# ---- the flag ----------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_these_values_turn_retrieval_on(monkeypatch, embedder, llm, value):
    monkeypatch.setenv("TINYAGENT_RAG_MEMORY", value)
    monkeypatch.setenv("TINYAGENT_EMBEDDING_URL", EMBEDDING_URL)
    mem = demo.memory_from_env(llm)
    assert type(mem) is RAGMemory and mem.documents == demo.DOCUMENTS


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no", "maybe"])
def test_other_values_leave_retrieval_off(monkeypatch, value):
    monkeypatch.setenv("TINYAGENT_RAG_MEMORY", value)
    assert type(demo.memory_from_env(LLM(model="m"))) is Memory  # nothing is embedded, so no server is needed


def test_the_embedding_model_defaults_to_embeddinggemma_with_the_llms_key(monkeypatch, embedder, llm):
    monkeypatch.setenv("TINYAGENT_RAG_MEMORY", "1")
    monkeypatch.setenv("TINYAGENT_EMBEDDING_URL", EMBEDDING_URL)
    mem = demo.memory_from_env(llm)
    assert isinstance(mem, RAGMemory)
    assert mem.embedding_model.model == "embeddinggemma-300m"
    assert mem.embedding_model.base_url == EMBEDDING_URL and mem.embedding_model.api_key == llm.api_key


def test_the_embedding_server_defaults_to_the_llama_embed_container(monkeypatch):
    """Only the address is checked, so the memory is replaced by its embedding model (nothing is embedded)."""
    monkeypatch.setenv("TINYAGENT_RAG_MEMORY", "1")
    monkeypatch.delenv("TINYAGENT_EMBEDDING_URL", raising=False)
    monkeypatch.setattr(demo, "RAGMemory", lambda embedding_model, documents: embedding_model)
    assert demo.memory_from_env(LLM(model="m")).base_url == "http://127.0.0.1:3201/v1"  # pyright: ignore[reportAttributeAccessIssue]


def test_the_embedding_model_can_be_chosen(monkeypatch):
    monkeypatch.setenv("TINYAGENT_RAG_MEMORY", "1")
    monkeypatch.setenv("TINYAGENT_EMBEDDING_MODEL", "other-embedder")
    monkeypatch.setattr(demo, "RAGMemory", lambda embedding_model, documents: embedding_model)
    assert demo.memory_from_env(LLM(model="m")).model == "other-embedder"  # pyright: ignore[reportAttributeAccessIssue]


def test_retrieval_needs_an_llm(monkeypatch):
    monkeypatch.setenv("TINYAGENT_RAG_MEMORY", "1")
    with pytest.raises(ValueError, match="llm"):
        demo.memory_from_env()


def test_the_rag_flag_wins_over_the_summary_and_trim_flags(monkeypatch, embedder, llm):
    monkeypatch.setenv("TINYAGENT_EMBEDDING_URL", EMBEDDING_URL)
    for name in ("TINYAGENT_RAG_MEMORY", "TINYAGENT_SUMMARY_MEMORY", "TINYAGENT_TRIM_MEMORY"):
        monkeypatch.setenv(name, "1")
    assert type(demo.memory_from_env(llm)) is RAGMemory


def test_the_other_flags_still_work_when_the_rag_flag_is_off(monkeypatch, llm):
    monkeypatch.setenv("TINYAGENT_SUMMARY_MEMORY", "1")
    assert type(demo.memory_from_env(llm)) is SummarizationMemory
    monkeypatch.delenv("TINYAGENT_SUMMARY_MEMORY")
    monkeypatch.setenv("TINYAGENT_TRIM_MEMORY", "1")
    assert type(demo.memory_from_env(llm)) is TrimmingMemory


def test_the_demo_runs_with_retrieval_on(monkeypatch, embedder, chat_model, capsys):
    """The two demo tasks on the real models: the greeting is in the history, so the names come back."""
    monkeypatch.setenv("TINYAGENT_RAG_MEMORY", "1")
    monkeypatch.setenv("TINYAGENT_EMBEDDING_URL", EMBEDDING_URL)
    monkeypatch.setenv("TINYAGENT_LLM_URL", LLM_URL)
    monkeypatch.setenv("TINYAGENT_MODEL", chat_model)
    demo.main()
    answer = capsys.readouterr().out
    assert "Maarten" in answer and "Jay" in answer
