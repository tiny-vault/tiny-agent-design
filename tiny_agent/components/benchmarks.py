"""Benchmarks: ready-made ``Benchmark`` objects to run an agent against.

``mmlu_pro`` holds three questions from MMLU-Pro, a multiple-choice benchmark
with ten options (A to J) per question across many subjects. Each task lists
the options and ends with "Answer with only the letter.", and each example's
``expected`` is the correct letter, which ``exact_match_scorer`` compares with
the first standalone letter of the agent's reply.

``ifeval`` holds three prompts from IFEval, which tests whether a model follows
a verifiable instruction rather than whether it knows a fact. Each example's
``check`` is a small function that tests the reply (fewer than 10 sentences,
at most 150 words, no commas), and ``programmatic_scorer`` calls it. The checks
are rough: "fewer than 10 sentences" counts the characters ``.``, ``!`` and
``?``, so an ellipsis counts three times and a number such as "3.5" once.

Three questions show how a benchmark works; they are far too few to measure a
model (one lucky guess moves the pass rate by a third). Nothing here calls a
model: pass a benchmark to ``Evaluator.run``. Compare ``evaluator.py`` for the
classes and the scorer.
"""

from tiny_agent.components.evaluator import (
    Benchmark,
    exact_match_scorer,
    programmatic_scorer,
)

# Three examples from MMLU Pro
mmlu_pro = Benchmark(
    name="MMLU Pro",
    examples=[
        {
            "task": """Which of the following is the body cavity that contains
the pituitary gland?
A) Ventral B) Dorsal C) Buccal D) Thoracic E) Pericardial F) Abdominal
G) Spinal H) Pelvic I) Pleural J) Cranial
Answer with only the letter.""",
            "expected": "J",
        },
        {
            "task": """What is the approximate mean cranial capacity of
Homo erectus?
A) 1200 cc B) under 650 cc C) 1700 cc D) 1350 cc E) just under 1000 cc
F) 1500 cc G) under 500 cc H) about 800 cc I) just over 1100 cc J) about 900 cc
Answer with only the letter.""",
            "expected": "E",
        },
        {
            "task": """
According to Moore’s "ideal utilitarianism," the right action is the one that
brings about the greatest amount of:
A) wealth. B) virtue. C) fairness. D) pleasure. E) peace. F) justice.
G) happiness. H) power. I) good. J) knowledge.
Answer with only the letter.""",
            "expected": "I",
        },
    ],
    scorer=exact_match_scorer,
)

# Three examples from IFeval
ifeval = Benchmark(
    name="IFeval",
    examples=[
        {
            "task": "Write me a funny song with less than 10 sentences for a "
            "proposal to build a new playground at my local elementary school.",
            "check": lambda text: sum(1 for c in text if c in ".!?") < 10,
        },
        {
            "task": "Write an ad copy for a new product, a digital photo frame "
            "that connects to your social media accounts and displays your photos. "
            "Respond with at most 150 words.",
            "check": lambda text: len(text.split()) <= 150,
        },
        {
            "task": "I am planning a trip to Japan, and I would like thee to "
            "write an itinerary for my journey in a Shakespearean style. "
            "You are not allowed to use any commas in your response.",
            "check": lambda text: "," not in text,
        },
    ],
    scorer=programmatic_scorer,
)
