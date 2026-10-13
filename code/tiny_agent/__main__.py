"""``python -m tiny_agent`` starts the terminal chat (same as the ``tiny-agent`` command).

Python runs this file when a package is started with ``-m``. It only forwards to
``main()`` in ``code/cli.py`` and uses its return value as the exit code
(``0`` success, ``1`` a one-shot question failed, ``2`` no usable model).
"""

import sys

from cli import main  # code/cli.py

sys.exit(main())
