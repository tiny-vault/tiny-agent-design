TinyAgent
=========

A minimal, modular, and educational agent framework. ``TinyAgent`` is the
*harness*: the code around an LLM that turns it into an agent.

.. note::

   The agent loop (``TinyAgent``) is still a skeleton: memory, tools and the
   planner are placeholders, and the loop does not call a model yet. Around it
   there is a working LLM client, a model check, a terminal chat
   (``tiny-agent``) and a Docker chat app that serves local models. See
   :doc:`architecture/overview` and :doc:`guides/chat-app`.

.. note::

   Based on *An Illustrated Guide to AI Agents* by Maarten Grootendorst and
   Jay Alammar (O'Reilly). The `companion repository
   <https://github.com/HandsOnLLM/An-Illustrated-Guide-To-AI-Agents>`_ has no
   license, so all rights remain with the authors. This is an independent
   learning project, not affiliated with or endorsed by them.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting-started/installation
   getting-started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/overview

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   api/illustrated-agents-scripts
   api/chat-app-backend
   guides/models
   guides/chat-app
   guides/development
