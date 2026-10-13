"""Repo-wide pytest setup: do not write bytecode.

Python caches compiled modules in ``__pycache__`` folders next to the source, which clutters the repo.
pytest loads this file before any test module, so setting the flag here keeps both the imported code and
pytest's own rewritten test modules from leaving ``__pycache__`` behind (``python -B`` and the
``PYTHONDONTWRITEBYTECODE`` variable do the same for other runs; ``node execute.mjs`` sets the variable).
It applies to pytest runs that start at the repo root, such as ``python -m pytest tiny_agent/tests``.
"""

import sys

sys.dont_write_bytecode = True
