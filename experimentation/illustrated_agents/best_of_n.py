"""Best-of-N four ways: generate 10 ``roman_to_int`` candidates and pick one with unit tests, an ORM, a PRM or a weighted PRM vote.

Every method samples ``n`` answers to the book's coding ``query`` from ``llm`` (``qwen3.5-4b`` via the gateway, ``temperature=1`` so candidates differ) and differs only in how it picks. ``best_of_n_unit_tests`` keeps the candidate passing most ``test_cases`` (``verify``, a fraction 0.0 to 1.0; a crash or Markdown-fenced reply scores 0.0). ``best_of_n_orm`` asks an outcome reward model to rate each finished answer 0 to 10 (``orm_score`` with ``ORM_PROMPT``). ``best_of_n_prm`` asks a process reward model to rate every non-empty code line (``split_steps``, ``prm_scores`` with ``PRM_PROMPT``) and keeps the candidate with the best weakest step (``min``). ``best_of_n_weighted_prm`` groups candidates that return the same outputs on the test inputs (``behaviour``), sums their PRM rewards and keeps the heaviest group, so an answer many good samples agree on beats one lucky outlier. With no trained reward model in the stack, ``judge`` (the same model at ``temperature=0``) stands in for both; the reward methods still run ``verify`` afterwards to show whether their pick really is the best. ``python best_of_n.py orm`` runs one method (``unit_tests``, ``orm``, ``prm``, ``weighted_prm``); no argument runs all four. Compare ``self_consistency``, which votes without a verifier.

Warning:
    ``verify`` and ``behaviour`` run model-written code with ``exec`` on this machine; a real system would sandbox it.
"""

import re
import sys
from collections import defaultdict

# qwen3.5-4b on the Docker stack (key from the Docker secret); temperature=1 makes candidates differ
from common import ORM_PROMPT, PRM_PROMPT, llm, make_llm, roman_to_int_query as query

# The reward model: same model, greedy, so one candidate always gets the same score.
judge = make_llm(temperature=0)

# Ground truth for verify; the reward models never see it, and behaviour uses only the inputs
test_cases = [
    ("III", 3),
    ("IV", 4),
    ("IX", 9),
    ("LVIII", 58),
    ("MCMXCIV", 1994),
    ("MMMCMXCIX", 3999),
    ("", 0),
    ("IIII", 0),       # invalid: four in a row
    ("VV", 0),         # invalid: V can't repeat
    ("IC", 0),         # invalid: I can only subtract from V or X
    ("ABC", 0),        # invalid: non-Roman characters
    ("MMMM", 0),       # invalid: exceeds 3999
]


def verify(answer):
    try:
        # Extract the function
        namespace = {}
        exec(answer, namespace)
        fn = namespace["roman_to_int"]

        # Evaluate the function on test cases
        # (the book's listing wraps `len(test_cases)` across two lines; as printed, `/ len` divides by the
        # builtin, raises TypeError and every candidate scores 0.0, so it is joined here)
        return sum(fn(inp) == expected for inp, expected in test_cases) / len(test_cases)
    except Exception:
        # Invalid Python, a missing function or a crash on any input scores 0
        return 0.0


def orm_score(answer):
    # One judge call per candidate: the whole answer in, a single score out (unparseable scores 0)
    reply = judge.generate([{"role": "user", "content": ORM_PROMPT.format(task=query, solution=answer)}]).content or ""
    match = re.search(r"\d+(?:\.\d+)?", reply)
    return min(float(match.group()), 10.0) / 10 if match else 0.0


def split_steps(answer):
    # A step is one non-empty line of the candidate's code
    return [line for line in answer.splitlines() if line.strip()]


def prm_scores(steps):
    # One judge call per candidate: all numbered steps in, one score per step out (a missing step scores 0)
    numbered = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))
    reply = judge.generate([{"role": "user", "content": PRM_PROMPT.format(task=query, steps=numbered)}]).content or ""
    found = {int(n): min(float(s), 1.0) for n, s in re.findall(r"^\s*(\d+)\s*[:.)-]\s*(\d+(?:\.\d+)?)", reply, re.M)}
    return [found.get(i, 0.0) for i in range(1, len(steps) + 1)]


def behaviour(answer):
    # The candidate's outputs on the inputs; two candidates with the same outputs count as the same answer
    try:
        namespace = {}
        exec(answer, namespace)
        fn = namespace["roman_to_int"]
    except Exception:
        return "does not run"
    outputs = []
    for inp, _ in test_cases:
        try:
            outputs.append(repr(fn(inp)))
        except Exception:
            outputs.append("error")
    return ", ".join(outputs)


def sample():
    return llm.generate([{"role": "user", "content": query}]).content


def best_of_n_unit_tests(n=10):
    # The verifier is the unit tests themselves: the most tests passed wins
    responses = []
    for i in range(n):
        answer = sample()
        score = verify(answer)
        responses.append((answer, score))
        print(f"candidate {i + 1}: score {score:.2f}")

    best_answer, best_score = max(responses, key=lambda x: x[1])
    print([response[1] for response in responses])
    print(f"\nbest score: {best_score:.2f}\n{best_answer}")
    return best_answer


def best_of_n_orm(n=10):
    # The ORM scores each finished answer as a whole; the top score wins
    candidates = []
    for i in range(n):
        answer = sample()
        reward, tests = orm_score(answer), verify(answer)
        candidates.append((answer, reward, tests))
        print(f"candidate {i + 1}: ORM {reward:.2f}  unit tests {tests:.2f}")

    # The unit tests show whether the ORM's pick was the right one
    best_answer, best_reward, best_tests = max(candidates, key=lambda c: c[1])
    print(f"\nORM pick: ORM {best_reward:.2f}, unit tests {best_tests:.2f} (best possible {max(c[2] for c in candidates):.2f})\n{best_answer}")
    return best_answer


def best_of_n_prm(n=10):
    # The PRM scores every step; a solution is only as good as its weakest step (min; the product is another choice)
    candidates = []
    for i in range(n):
        answer = sample()
        steps = split_steps(answer)
        scores = prm_scores(steps) or [0.0]
        reward, tests = min(scores), verify(answer)
        weakest = scores.index(reward)
        candidates.append((answer, reward, tests, list(zip(steps, scores))))
        print(f"candidate {i + 1}: PRM {reward:.2f} (weakest step {weakest + 1}/{len(steps)}: {steps[weakest].strip() if steps else '-'})  unit tests {tests:.2f}")

    best_answer, best_reward, best_tests, best_steps = max(candidates, key=lambda c: c[1])
    print(f"\nPRM pick: PRM {best_reward:.2f}, unit tests {best_tests:.2f} (best possible {max(c[2] for c in candidates):.2f})\n{best_answer}")
    print("\nStep scores of the pick:")
    for step, score in best_steps:
        print(f"  {score:.2f}  {step}")
    return best_answer


def best_of_n_weighted_prm(n=10):
    # Score each candidate with the PRM (weakest step) and group candidates by behaviour
    candidates = []
    for i in range(n):
        answer = sample()
        reward, tests = min(prm_scores(split_steps(answer)) or [0.0]), verify(answer)
        candidates.append((answer, reward, tests, behaviour(answer)))
        print(f"candidate {i + 1}: PRM {reward:.2f}  unit tests {tests:.2f}")

    # Weighted vote: each answer (behaviour) collects the PRM rewards of every candidate that gives it
    weights, members = defaultdict(float), defaultdict(list)
    for candidate in candidates:
        weights[candidate[3]] += candidate[1]
        members[candidate[3]].append(candidate)

    print("\nAnswers by total PRM weight (outputs on the test inputs):")
    for key in sorted(weights, key=weights.get, reverse=True):
        print(f"  weight {weights[key]:.2f} from {len(members[key])} candidate(s), unit tests {members[key][0][2]:.2f}: {key}")

    # The heaviest answer wins; its best-scored candidate is the code we return
    best_key = max(weights, key=weights.get)
    best_answer, best_reward, best_tests, _ = max(members[best_key], key=lambda c: c[1])
    plain_pick = max(candidates, key=lambda c: c[1])
    print(f"\nWeighted pick: weight {weights[best_key]:.2f}, unit tests {best_tests:.2f}"
          f" (plain Best-of-N pick: unit tests {plain_pick[2]:.2f}; best possible {max(c[2] for c in candidates):.2f})\n{best_answer}")
    return best_answer


methods = {
    "unit_tests": best_of_n_unit_tests,
    "orm": best_of_n_orm,
    "prm": best_of_n_prm,
    "weighted_prm": best_of_n_weighted_prm,
}

if __name__ == "__main__":
    # One method by name, or all four (needs the Docker stack with qwen3.5-4b loaded).
    # Via trigger_ias.py, argv[1] is the script name, which matches no method, so all four run.
    chosen = [sys.argv[1]] if len(sys.argv) > 1 and sys.argv[1] in methods else list(methods)
    for name in chosen:
        print(f"\n===== best_of_n_{name} =====")
        methods[name]()
