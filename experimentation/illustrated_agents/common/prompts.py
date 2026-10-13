"""Every prompt the scripts send, in one place so a script can reuse or compare another's."""

# ---- prompting chapter: the penguin question and its flamingo examples ----

# The question; the answer is 9 - 2 + 4 = 11
penguin_query = "I saw 9 penguins. 2 slid in the water and disappeared from sight while 4 waddled up from the shore. How many can I see?"

# A solved example of the same shape, for one-shot prompting (answer 3)
flamingo_example = "I saw 6 flamingos. 2 flew away. 1 hid behind a tree. How many can I see?"

# Its bare answer: the model copies the format and answers with just a number
flamingo_answer = "3"

# Its worked answer: the model copies the reasoning (one-shot chain-of-thought)
flamingo_cot_answer = "You saw 6 flamingos 2 flew away. 6 - 2 = 4. Then, 1 flamingo hid. You see now 4 - 1 = 3."

# Zero-shot chain-of-thought trigger (Kojima et al., 2022), appended to a question
step_by_step = "Let's think step by step."

# ---- self-consistency: a puzzle with one right answer (seat 5) ----

seating_query = """Six friends (Maarten, Ilse, Sarah, Jor, Irene, and Chris)
    are sitting in a row.

- Sarah is in seat 3.
- Chris is sitting at one of the ends of the row.
- Jor is sitting immediately to the right of Chris.
- Ilse is sitting exactly in the middle of Sarah and Maarten.
- Irene is not sitting next to Sarah.
- Maarten is sitting somewhere to the left of Irene.

In which seat is Maarten sitting?
Let's think step by step and give back your answer only after 'Answer:'.
"""

# ---- Best-of-N: a coding task with a checkable answer, and the reward-model judges ----

roman_to_int_query = """Write a Python function `roman_to_int(s)` that converts a Roman
    numeral
string to an integer. The function should:
- Handle standard Roman numerals from 1 to 3999.
- Correctly handle subtractive notation (e.g., IV=4, IX=9, XL=40, XC=90, CD=400,
CM=900).
- Return 0 for invalid inputs (empty string, non-Roman characters, or invalid
patterns
  like "IIII" or "VV").

Give only the function definition, no explanation or markdown formatting."""

# Outcome reward model: sees only the finished solution, returns one score ({task}, {solution})
ORM_PROMPT = """You are an outcome reward model. Judge only whether the final solution fully solves the task.

Task:
{task}

Solution:
{solution}

Rate how likely the solution is completely correct on a scale from 0 to 10. Reply with only the number."""

# Process reward model: scores every numbered step ({task}, {steps})
PRM_PROMPT = """You are a process reward model. Judge each step of a partial solution, in order.

Task:
{task}

Solution, one numbered step per line:
{steps}

For every step, rate from 0 to 1 how likely it is correct and moves toward a correct solution, given the steps before it. Reply with exactly one line per step in the form `<step number>: <score>` and nothing else."""

# ---- agents: the smallest task for TinyAgent ----

agent_query = "What is 2 + 2?"
