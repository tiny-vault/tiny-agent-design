"""Tests for the memory variants: trimming, summarization, retrieval (RAG) and the memory bank.

The trimming, summarization and bank tests use stand-in models; the retrieval tests use the real models in Docker
(``node execute.mjs up llm-client``) and are skipped when those are down.
"""

from typing import Any

import pytest

from tiny_agent import main as demo
from tiny_agent.agent import TinyAgent
from tiny_agent.components.client.embedding import EmbeddingModel
from tiny_agent.components.client.llm import LLM
from tiny_agent.components.client.response import Response
from tiny_agent.components.memory.memory import BaseMemory, Memory
from tiny_agent.components.memory.memory_bank import (
    FORGET_BELOW,
    SECONDS_PER_DAY,
    TOP_K,
    MemoryBank,
)
from tiny_agent.components.memory.rag import RAGMemory
from tiny_agent.components.memory.similarity import cosine
from tiny_agent.components.memory.summarization import SummarizationMemory
from tiny_agent.components.memory.trimming import TrimmingMemory
from tiny_agent.tests.stubs import EMBEDDING_URL, LLM_URL, RecordingLLM, texts

# ---- what every memory does the same way (one test, run for each memory) -------------------------------

CALL = {"id": "1", "function": {"name": "f", "arguments": "{}"}}


@pytest.fixture(params=["trim", "summary", "rag", "bank"])
def any_memory(request):
    """Each memory variant in turn; the retrieval one needs the real embedding model and is skipped without it."""
    if request.param == "trim":
        return TrimmingMemory()
    if request.param == "summary":
        return SummarizationMemory(RecordingLLM())
    if request.param == "rag":
        return RAGMemory(request.getfixturevalue("embedder"), DOCS[:2])
    return MemoryBank(FakeLLM(), FakeEmbedder())  # pyright: ignore[reportArgumentType]


def test_every_memory_is_a_memory_and_starts_empty(any_memory):
    assert isinstance(any_memory, BaseMemory)
    assert any_memory.get_messages() == []


def test_get_messages_returns_the_history_itself(any_memory):
    any_memory.add("user", "hi")
    assert any_memory.get_messages() is any_memory.messages


def test_a_tool_call_is_stored_with_its_message(any_memory):
    any_memory.add("tool", "result", tool_call=CALL)
    assert any_memory.get_messages() == [{"role": "tool", "content": "result", "tool_calls": [CALL]}]


def test_other_roles_are_stored_unchanged(any_memory):
    any_memory.add("tool", "result")
    assert any_memory.get_messages() == [{"role": "tool", "content": "result"}]


# ====================================================================================================
# test_trimming_memory
# A sliding message window and its boundary effects.
#
# A fixed message window reduces retained history without summarizing it.
# Recording requests lets tests observe exactly which older messages disappear
# and whether system instructions remain first.
#
# What this file demonstrates:
# Tests for ``TrimmingMemory`` (the sliding window over the conversation) and the flag that turns it on.
#
# Three groups: the memory on its own (what is kept, in which order), the memory inside a ``TinyAgent`` (what
# the model is actually sent over several runs), and ``main.memory_from_env`` (the ``TINYAGENT_TRIM_MEMORY``
# flag). The agent tests use a stand-in model that records every call, so no server is needed.
#
# Expected observations and limits:
# The window counts four non-system messages, not tokens or complete turns at
# every intermediate point. It can start with an assistant message. Assertions
# verify the current rule, not acceptance by every server's chat template.
# ====================================================================================================

def add_turns(memory: BaseMemory, count: int) -> None:
    """Add ``count`` user/assistant pairs: ``u1``, ``a1``, ``u2``, ``a2``, ..."""
    for i in range(1, count + 1):
        memory.add("user", f"u{i}")
        memory.add("assistant", f"a{i}")


# ---- the memory on its own ----------------------------------------------------------

def test_two_turns_fit_in_the_window_and_nothing_is_dropped():
    memory = TrimmingMemory()
    add_turns(memory, 2)
    assert texts(memory) == ["u1", "a1", "u2", "a2"]


def test_the_third_turn_pushes_the_first_out():
    memory = TrimmingMemory()
    add_turns(memory, 3)
    assert texts(memory) == ["u2", "a2", "u3", "a3"]


def test_the_window_can_start_with_an_assistant_message_right_after_a_user_message():
    """Pins the current behaviour: the trim counts messages, so adding u3 cuts u1 but keeps a1."""
    memory = TrimmingMemory()
    add_turns(memory, 2)
    memory.add("user", "u3")
    assert texts(memory) == ["a1", "u2", "a2", "u3"]


def test_the_history_never_grows_past_four_messages():
    memory = TrimmingMemory()
    for i in range(1, 51):
        memory.add("user", f"u{i}")
        memory.add("assistant", f"a{i}")
        assert len(memory.get_messages()) <= 4
    assert texts(memory) == ["u49", "a49", "u50", "a50"]


def test_plain_memory_keeps_everything_for_contrast():
    memory = Memory()
    add_turns(memory, 3)
    assert len(memory.get_messages()) == 6


def test_the_system_message_is_always_kept_and_stays_first():
    memory = TrimmingMemory()
    memory.add("system", "Be brief.")
    add_turns(memory, 4)
    assert texts(memory) == ["Be brief.", "u3", "a3", "u4", "a4"]


def test_a_system_message_added_late_still_comes_first():
    memory = TrimmingMemory()
    add_turns(memory, 1)
    memory.add("system", "Be brief.")
    assert texts(memory) == ["Be brief.", "u1", "a1"]


def test_every_system_message_is_kept():
    memory = TrimmingMemory()
    memory.add("system", "s1")
    memory.add("system", "s2")
    add_turns(memory, 3)
    assert texts(memory) == ["s1", "s2", "u2", "a2", "u3", "a3"]


def test_a_tool_call_is_stored_like_in_memory_and_a_tool_message_counts_toward_the_window():
    call = {"id": "1", "function": {"name": "f", "arguments": "{}"}}
    memory = TrimmingMemory()
    memory.add("user", "u1")
    memory.add("assistant", "", tool_call=call)
    memory.add("tool", "result")
    memory.add("assistant", "a1")
    assert memory.get_messages()[1] == {"role": "assistant", "content": "", "tool_calls": [call]}
    memory.add("user", "u2")
    # Five non-system messages: the oldest (u1) is out, the tool message is one of the four kept.
    assert [m["role"] for m in memory.get_messages()] == ["assistant", "tool", "assistant", "user"]


def test_get_messages_returns_the_trimmed_history_itself():
    memory = TrimmingMemory()
    add_turns(memory, 3)
    assert memory.get_messages() is memory.messages


# ---- the memory inside a TinyAgent ----------------------------------------------------------

def test_the_model_is_sent_only_the_window_and_forgets_the_first_task():
    llm = RecordingLLM(prefix="a")
    agent = TinyAgent(llm, TrimmingMemory())
    for task in ("u1", "u2", "u3", "u4"):
        agent.run(task)
    # How many messages each call carried: the window stops growing at four.
    assert [len(call) for call in llm.calls] == [1, 3, 4, 4]
    # By the fourth call the first task is gone from what the model sees.
    assert "u1" not in [message["content"] for message in llm.calls[3]]
    assert [message["content"] for message in llm.calls[3]] == ["a2", "u3", "a3", "u4"]


def test_with_plain_memory_the_same_runs_keep_the_first_task():
    llm = RecordingLLM(prefix="a")
    agent = TinyAgent(llm, Memory())
    for task in ("u1", "u2", "u3", "u4"):
        agent.run(task)
    assert [len(call) for call in llm.calls] == [1, 3, 5, 7]
    assert llm.calls[3][0]["content"] == "u1"


# ---- the flag ----------------------------------------------------------------------------

def test_the_flag_is_off_by_default():
    assert type(demo.memory_from_env()) is Memory


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_these_values_turn_trimming_on(monkeypatch, value):
    monkeypatch.setenv("TINYAGENT_TRIM_MEMORY", value)
    assert type(demo.memory_from_env()) is TrimmingMemory


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_other_values_leave_trimming_off(monkeypatch, value):
    monkeypatch.setenv("TINYAGENT_TRIM_MEMORY", value)
    assert type(demo.memory_from_env()) is Memory


def test_the_demo_gives_the_same_answer_with_the_flag_on(demo_server, monkeypatch, capsys):
    """The demo's two tasks fit in the window, so trimming changes nothing it sends or prints."""
    monkeypatch.setenv("TINYAGENT_TRIM_MEMORY", "1")
    demo.main()
    assert capsys.readouterr().out.strip() == "Your names are Maarten and Jay."
    assert [m["content"] for m in demo_server.requests[1]["body"]["messages"]][0] == demo.GREETING


# ====================================================================================================
# test_summarization_memory
# Running summaries as lossy conversation compression.
#
# Replacing messages with a summary keeps the message count small but can lose
# facts. A recording model makes the summary content and extra calls exact,
# allowing the tests to isolate replacement mechanics.
#
# What this file demonstrates:
# Tests for ``SummarizationMemory`` (a running summary written by the model) and the flag that turns it on.
#
# Three groups: the memory on its own (when it summarises, what it asks the model, what it keeps), the memory
# inside a ``TinyAgent`` (what the model is actually sent over several runs), and ``main.memory_from_env`` (the
# ``TINYAGENT_SUMMARY_MEMORY`` flag). A stand-in model records every call and answers ``"s<n>"``, so no server is
# needed except in the last test.
#
# Expected observations and limits:
# The assertions show one system summary after a successful reply, the next
# question beside it, and one extra model call per completed turn. Scripted
# summaries do not demonstrate faithful compression or a bounded token count.
# The final demo test also uses the shared fake server.
# ====================================================================================================

# ---- the memory on its own ----------------------------------------------------------

def test_it_starts_empty_and_makes_no_model_call():
    llm = RecordingLLM()
    memory = SummarizationMemory(llm)
    assert memory.get_messages() == []
    assert llm.calls == []


def test_user_and_tool_messages_are_stored_without_a_model_call():
    llm = RecordingLLM()
    memory = SummarizationMemory(llm)
    memory.add("user", "u1")
    memory.add("tool", "result")
    assert texts(memory) == ["u1", "result"]
    assert llm.calls == []


def test_an_assistant_reply_replaces_the_history_with_one_system_summary():
    llm = RecordingLLM()
    memory = SummarizationMemory(llm)
    memory.add("user", "u1")
    memory.add("assistant", "a1")
    assert memory.get_messages() == [{"role": "system", "content": "s1"}]
    assert len(llm.calls) == 1


def test_the_summary_prompt_holds_the_old_summary_and_the_new_turn():
    llm = RecordingLLM()
    memory = SummarizationMemory(llm)
    memory.add("user", "u1")
    memory.add("assistant", "a1")
    memory.add("user", "u2")
    memory.add("assistant", "a2")
    first, second = (call[0]["content"] for call in llm.calls)
    assert [len(call) for call in llm.calls] == [1, 1]
    assert "Summary: \n" in first and "user: u1\nassistant: a1\n" in first
    # The second prompt carries the first summary ("s1") and only the turn added since.
    assert "Summary: s1\n" in second and "user: u2\nassistant: a2\n" in second
    assert "u1" not in second
    assert texts(memory) == ["s2"]


def test_a_user_message_after_a_summary_is_kept_next_to_it():
    memory = SummarizationMemory(RecordingLLM())
    memory.add("user", "u1")
    memory.add("assistant", "a1")
    memory.add("user", "u2")
    assert texts(memory) == ["s1", "u2"]


def test_the_history_never_grows_past_one_message_after_a_reply():
    memory = SummarizationMemory(RecordingLLM())
    for i in range(1, 21):
        memory.add("user", f"u{i}")
        memory.add("assistant", f"a{i}")
        assert len(memory.get_messages()) == 1
    assert texts(memory) == ["s20"]


def test_a_reply_with_no_text_keeps_the_old_summary():
    llm = RecordingLLM()
    memory = SummarizationMemory(llm)
    memory.add("user", "u1")
    memory.add("assistant", "a1")
    llm.fixed = ""
    memory.add("user", "u2")
    memory.add("assistant", "a2")
    assert texts(memory) == ["s1"]


def test_an_existing_system_message_is_taken_for_the_summary_and_overwritten():
    """Pins the current behaviour: the summary lives in the system message, so a system prompt does not survive."""
    memory = SummarizationMemory(RecordingLLM())
    memory.add("system", "Be brief.")
    memory.add("user", "u1")
    memory.add("assistant", "a1")
    assert texts(memory) == ["s1"]


def test_a_failing_model_raises_and_leaves_the_reply_unsummarised():
    class Broken(RecordingLLM):
        def generate(self, messages, tools=None):
            raise RuntimeError("server down")

    memory = SummarizationMemory(Broken())
    memory.add("user", "u1")
    with pytest.raises(RuntimeError, match="server down"):
        memory.add("assistant", "a1")
    assert texts(memory) == ["u1", "a1"]


# ---- the memory inside a TinyAgent ----------------------------------------------------------

def test_the_model_is_sent_the_summary_and_the_new_question_each_run():
    llm = RecordingLLM()
    agent = TinyAgent(llm, SummarizationMemory(llm))
    for task in ("u1", "u2", "u3"):
        agent.run(task)
    # Calls alternate: the agent's answer call, then the memory's summary call (one message, the prompt).
    answer_calls = llm.calls[0::2]
    summary_calls = llm.calls[1::2]
    assert [[m["content"] for m in call] for call in answer_calls] == [
        ["u1"],
        ["s2", "u2"],  # s2 is the 2nd model call overall: the summary of the first turn
        ["s4", "u3"],
    ]
    assert all(len(call) == 1 and call[0]["role"] == "user" for call in summary_calls)
    # Three answers and three summaries.
    assert len(llm.calls) == 6


def test_the_first_task_is_not_resent_after_it_is_summarised():
    llm = RecordingLLM()
    agent = TinyAgent(llm, SummarizationMemory(llm))
    for task in ("u1", "u2", "u3"):
        agent.run(task)
    assert "u1" not in [m["content"] for m in llm.calls[4]]


# ---- the flag ----------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_these_values_turn_summarizing_on(monkeypatch, value):
    monkeypatch.setenv("TINYAGENT_SUMMARY_MEMORY", value)
    assert type(demo.memory_from_env(RecordingLLM())) is SummarizationMemory


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_other_values_leave_summarizing_off(monkeypatch, value):
    monkeypatch.setenv("TINYAGENT_SUMMARY_MEMORY", value)
    assert type(demo.memory_from_env(RecordingLLM())) is Memory


def test_the_memory_gets_the_llm_it_was_given(monkeypatch):
    monkeypatch.setenv("TINYAGENT_SUMMARY_MEMORY", "1")
    llm = RecordingLLM()
    memory = demo.memory_from_env(llm)
    assert isinstance(memory, SummarizationMemory) and memory.llm is llm


def test_summarizing_needs_an_llm(monkeypatch):
    monkeypatch.setenv("TINYAGENT_SUMMARY_MEMORY", "1")
    with pytest.raises(ValueError, match="llm"):
        demo.memory_from_env()


def test_the_summary_flag_wins_over_the_trim_flag(monkeypatch):
    monkeypatch.setenv("TINYAGENT_SUMMARY_MEMORY", "1")
    monkeypatch.setenv("TINYAGENT_TRIM_MEMORY", "1")
    assert type(demo.memory_from_env(RecordingLLM())) is SummarizationMemory


def test_the_trim_flag_still_works_when_the_summary_flag_is_off(monkeypatch):
    monkeypatch.setenv("TINYAGENT_TRIM_MEMORY", "1")
    assert type(demo.memory_from_env(RecordingLLM())) is TrimmingMemory


def test_the_demo_sends_the_summary_instead_of_the_history(demo_server, monkeypatch, capsys):
    """With the flag on, the second task carries the summary, not the greeting: four requests in all."""
    monkeypatch.setenv("TINYAGENT_SUMMARY_MEMORY", "1")
    demo.main()
    assert capsys.readouterr().out.strip() == "Your names are Maarten and Jay."
    assert len(demo_server.requests) == 4  # greeting, summary, question, summary
    second_task = demo_server.requests[2]["body"]["messages"]
    assert [m["role"] for m in second_task] == ["system", "user"]
    assert second_task[1]["content"] == demo.QUESTION
    assert demo.GREETING not in second_task[0]["content"]


# ====================================================================================================
# test_rag_memory
# Retrieval-augmented memory and real-model integration checks.
#
# Documents are embedded once, then ranked against each question. Tests inspect
# both retrieved document order and the augmented message so retrieval can be
# distinguished from the model's subsequent answer.
#
# What this file demonstrates:
# Tests for ``RAGMemory`` (documents retrieved by meaning and put in front of each question) and its flag.
#
# All against real models, deployed in Docker (``node execute.mjs up llm-client``): EmbeddingGemma in the
# ``llama-embed`` container at ``TINYAGENT_EMBEDDING_URL`` (default ``http://127.0.0.1:3201/v1``) and whichever
# chat model is loaded on the model server at ``TINYAGENT_LLM_URL`` (default ``http://127.0.0.1:3200/v1``; load one
# with ``node execute.mjs llm-client load qwen3.5-4b``). There are no fakes: when a server is not up, the tests
# that need it are skipped and say which command starts it. Chat replies are not deterministic, so those tests
# check for a key word, not for exact text.
#
# Expected observations and limits:
# Some checks need only arithmetic or settings, while retrieval and answer
# checks require the deployed models. A skip leaves those behaviors unverified.
# Exact document ranking depends on the configured embedding model; keyword
# answers do not prove factual reliability on other questions.
# ====================================================================================================

DOCS = demo.DOCUMENTS


@pytest.fixture
def llm(chat_model):
    """A real chat model."""
    return LLM(model=chat_model, base_url=LLM_URL)


@pytest.fixture(scope="module")
def memory(embedder):
    """A ``RAGMemory`` over the ten demo documents, for tests that only search. Module-scoped because embedding
    the documents takes a moment; tests that add messages build their own."""
    return RAGMemory(embedder, DOCS)


# ---- the memory on its own ----------------------------------------------------------

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


def test_cosine_of_the_same_direction_is_one_and_of_perpendicular_vectors_zero():
    assert cosine([2.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
    assert cosine([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


def test_a_user_message_is_stored_with_its_context_in_front(embedder):
    mem = RAGMemory(embedder, DOCS)
    mem.add("user", "What is Sarah's favorite animal?")
    [stored] = mem.get_messages()
    assert stored["role"] == "user"
    assert stored["content"].startswith("Context:\n")
    assert "Sarah favorite animal is flamingos." in stored["content"]
    assert stored["content"].endswith("\n\nQuestion: What is Sarah's favorite animal?")


def test_each_user_message_gets_its_own_context_and_all_are_kept(embedder):
    mem = RAGMemory(embedder, DOCS)
    mem.add("user", "What instrument does Ilse play?")
    mem.add("assistant", "a1")
    mem.add("user", "Where does Sarah live?")
    first, _, second = texts(mem)
    assert "cello" in first.split("Question:")[0]
    assert "Lisbon" in second.split("Question:")[0]


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


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
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
    """The two demo tasks on the real models: the greeting is in the history, so the names come back.

    A small model sometimes overlooks the greeting among the retrieved documents, so up to three attempts are made.
    """
    monkeypatch.setenv("TINYAGENT_RAG_MEMORY", "1")
    monkeypatch.setenv("TINYAGENT_EMBEDDING_URL", EMBEDDING_URL)
    monkeypatch.setenv("TINYAGENT_LLM_URL", LLM_URL)
    monkeypatch.setenv("TINYAGENT_MODEL", chat_model)
    answer = ""
    for _ in range(3):
        demo.main()
        answer = capsys.readouterr().out
        if "Maarten" in answer and "Jay" in answer:
            break
    assert "Maarten" in answer and "Jay" in answer


# ====================================================================================================
# test_memory_bank
# Testing memory retrieval, decay and reinforcement with a movable clock.
#
# Time-dependent memory is hard to test by waiting. A controlled clock and
# keyword vectors make decay and ranking exact; scripted summaries separate
# memory mechanics from model quality.
#
# What this file demonstrates:
# Tests for ``MemoryBank`` (event summaries, a user portrait and a forgetting curve).
#
# Stand-ins replace both models so the tests are fast and exact: ``FakeLLM`` answers ``"summary<n>"`` or
# ``"portrait<n>"`` depending on the prompt and records every call, ``FakeEmbedder`` maps a text to a vector by the
# key words it contains, and ``Clock`` is a time the tests move by hand.
#
# Expected observations and limits:
# The assertions track raw history, summary strength, recall times and model
# calls. Fake portrait text does not prove valid psychological inference.
# The bank is in-memory, and forgotten entries remain stored.
# ====================================================================================================

DAY = SECONDS_PER_DAY


class FakeLLM(LLM):
    """Answers ``summary<n>`` to a summary prompt and ``portrait<n>`` to a portrait prompt; records the prompts."""

    def __init__(self):
        super().__init__(model="fake")
        self.prompts: list[str] = []

    def generate(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> Response:
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        kind = "portrait" if prompt.startswith("Update the user's portrait") else "summary"
        return Response(content=f"{kind}{len(self.prompts)}")


class FakeEmbedder:
    """Vector of key-word counts: ``cat``, ``dog``, ``fish``. Counts its requests."""

    WORDS = ("cat", "dog", "fish")

    def __init__(self):
        self.requests: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.requests.append(text)
        return [float(text.lower().count(word)) + 0.01 for word in self.WORDS]


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def make() -> tuple[MemoryBank, FakeLLM, FakeEmbedder, Clock]:
    llm, embedder, clock = FakeLLM(), FakeEmbedder(), Clock()
    return MemoryBank(llm, embedder, clock), llm, embedder, clock  # pyright: ignore[reportArgumentType]


def bank_with(texts: list[str]) -> tuple[MemoryBank, Clock]:
    """A bank that already holds one event summary per text (put there directly, no model calls)."""
    memory, _, embedder, clock = make()
    memory.event_summaries = [
        {"text": t, "embedding": embedder.embed(t), "strength": 1, "last_recalled": clock.now} for t in texts
    ]
    return memory, clock


# ---- storage ----------------------------------------------------------------------------

def test_every_message_is_kept_raw_with_its_time():
    memory, _, _, clock = make()
    memory.add("user", "I have a cat")
    clock.now += 5
    memory.add("assistant", "Nice")
    assert memory.conversation_history == [
        {"role": "user", "content": "I have a cat", "time": 1_000_000.0},
        {"role": "assistant", "content": "Nice", "time": 1_000_005.0},
    ]


def test_an_assistant_reply_stores_an_event_summary_and_a_portrait():
    memory, llm, _, _ = make()
    memory.add("user", "I have a cat")
    memory.add("assistant", "Nice")
    assert [s["text"] for s in memory.event_summaries] == ["summary1"]
    assert memory.event_summaries[0]["strength"] == 1
    assert memory.user_portrait == "portrait2"
    assert len(llm.prompts) == 2
    assert "user: I have a cat\nassistant: Nice\n" in llm.prompts[0]


def test_a_second_turn_summarises_only_the_new_turn_and_extends_the_portrait():
    memory, llm, _, _ = make()
    for i in (1, 2):
        memory.add("user", f"u{i}")
        memory.add("assistant", f"a{i}")
    assert "u1" not in llm.prompts[2] and "user: u2\nassistant: a2\n" in llm.prompts[2]
    assert "Portrait: portrait2\n" in llm.prompts[3]
    assert [s["text"] for s in memory.event_summaries] == ["summary1", "summary3"]


def test_user_and_tool_messages_alone_make_no_model_call():
    memory, llm, _, _ = make()
    memory.add("user", "u1")
    memory.add("tool", "result")
    assert llm.prompts == []


# ---- retrieval ---------------------------------------------------------------------------

def test_search_returns_the_most_similar_summaries_first_and_at_most_top_k():
    memory, _ = bank_with(["cat cat", "dog dog", "fish fish", "cat dog", "cat"])
    found = memory.search("my cat")
    assert len(found) == TOP_K and found[0] in ("cat cat", "cat")


def test_search_with_nothing_stored_makes_no_embedding_request():
    memory, _, embedder, _ = make()
    assert memory.search("cat") == []
    assert embedder.requests == []


def test_a_user_message_gets_the_portrait_and_memories_in_front_of_it():
    memory, _ = bank_with(["dog walk"])
    memory.user_portrait = "calm"
    memory.add("user", "my dog?")
    assert memory.get_messages()[0]["content"] == "Portrait:\ncalm\n\nMemories:\ndog walk\n\nQuestion: my dog?"
    # The raw history keeps what was said, not the augmented text.
    assert memory.conversation_history[0]["content"] == "my dog?"


# ---- forgetting and reinforcement ----------------------------------------------------------

def test_recalling_a_summary_adds_one_to_its_strength_and_restarts_its_clock():
    memory, clock = bank_with(["dog walk"])
    clock.now += 2 * DAY
    memory.search("dog")
    item = memory.event_summaries[0]
    assert item["strength"] == 2 and item["last_recalled"] == clock.now


def test_a_summary_not_recalled_for_long_enough_is_forgotten_but_kept():
    memory, clock = bank_with(["dog walk"])
    clock.now += 3 * DAY  # exp(-3) is about 0.05, below FORGET_BELOW
    assert memory.search("dog") == []
    assert len(memory.event_summaries) == 1 and FORGET_BELOW > 0.05


def test_a_stronger_summary_outlasts_a_weaker_one():
    memory, clock = bank_with(["dog walk", "dog park"])
    memory.event_summaries[1]["strength"] = 5
    clock.now += 3 * DAY  # exp(-3/5) is about 0.55
    assert memory.search("dog") == ["dog park"]


def test_an_unrecalled_summary_stays_until_its_retention_drops_below_the_limit():
    memory, clock = bank_with(["dog walk"])
    clock.now += 2 * DAY  # exp(-2) is about 0.14, still above
    assert memory.search("dog") == ["dog walk"]


# ---- inside a TinyAgent --------------------------------------------------------------------

def test_the_agent_gets_an_answer_call_then_two_memory_calls_per_task():
    memory, _, _, _ = make()
    answers = []

    class Answering(FakeLLM):
        def generate(self, messages, tools=None):
            if messages[0]["content"].startswith(("Summarize", "Update the user's portrait")):
                return super().generate(messages)
            answers.append(list(messages))
            return Response(content="ok")

    answering = Answering()
    memory.llm = answering
    agent = TinyAgent(answering, memory)
    assert agent.run("I have a cat") == "ok"
    assert agent.run("what pet do I have?") == "ok"
    # The second question reached the model with the first turn's summary in front of it.
    assert "Memories:\nsummary1" in answers[1][-1]["content"]


# ---- the flag ----------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_these_values_turn_the_bank_on(monkeypatch, value):
    monkeypatch.setenv("TINYAGENT_BANK_MEMORY", value)
    assert type(demo.memory_from_env(FakeLLM())) is MemoryBank


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_other_values_leave_the_bank_off(monkeypatch, value):
    monkeypatch.setenv("TINYAGENT_BANK_MEMORY", value)
    assert type(demo.memory_from_env(FakeLLM())) is Memory


def test_the_bank_gets_the_llm_it_was_given(monkeypatch):
    monkeypatch.setenv("TINYAGENT_BANK_MEMORY", "1")
    llm = FakeLLM()
    memory = demo.memory_from_env(llm)
    assert isinstance(memory, MemoryBank) and memory.llm is llm


def test_the_bank_needs_an_llm(monkeypatch):
    monkeypatch.setenv("TINYAGENT_BANK_MEMORY", "1")
    with pytest.raises(ValueError, match="llm"):
        demo.memory_from_env()


def test_the_bank_flag_wins_over_the_other_flags(monkeypatch):
    for name in ("TINYAGENT_RAG_MEMORY", "TINYAGENT_SUMMARY_MEMORY", "TINYAGENT_TRIM_MEMORY", "TINYAGENT_BANK_MEMORY"):
        monkeypatch.setenv(name, "1")
    assert type(demo.memory_from_env(FakeLLM())) is MemoryBank


def test_a_new_bank_is_empty():
    memory, llm, embedder, _ = make()
    assert memory.conversation_history == [] and memory.event_summaries == [] and memory.user_portrait == ""
    assert llm.prompts == [] and embedder.requests == []

