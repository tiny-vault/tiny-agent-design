"""Unified diff of ``TinyAgent`` from chapter 1 to chapter 2.

``tinyagents_diff`` renders HTML only in Jupyter, so the script prints ``difflib.unified_diff`` of the two classes' source. No model needed.
"""

from illustrated_agents.chapters.ch2 import tinyagents_diff
tinyagents_diff

# Runnable part: DiffViewer only renders HTML inside Jupyter, so print the
# same unified diff as plain text when this file is run as a script.
if __name__ == "__main__":
    from common import print_diff

    print_diff(tinyagents_diff)
