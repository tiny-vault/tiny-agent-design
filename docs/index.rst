TinyAgent
=========

.. raw:: html

   <div class="gb-hero">
     <p>A minimal, educational agent harness. <code>TinyAgent</code> is the code around a language model that turns it into an agent: memory, tools and a record of what it did.</p>
   </div>

   <div class="gb-cards">
     <a class="gb-card" href="getting-started/installation.html"><strong>Install</strong><span>Create the environment and run the demo in a few minutes.</span></a>
     <a class="gb-card" href="getting-started/quickstart.html"><strong>Quickstart</strong><span>Talk to a model, run the harness, open the terminal chat.</span></a>
     <a class="gb-card" href="architecture/overview.html"><strong>How it fits together</strong><span>The harness, its parts and the data flow, with diagrams.</span></a>
     <a class="gb-card" href="architecture/agent.html"><strong>The agent harness</strong><span>What an agent is, the five slots and what <code>run</code> does today.</span></a>
     <a class="gb-card" href="architecture/memory.html"><strong>Memory</strong><span>Five ways to keep a conversation: all of it, the last turns, a summary, documents, a memory bank.</span></a>
     <a class="gb-card" href="architecture/tools.html"><strong>Tools</strong><span>Prompted versus learned tool calling, and how a call becomes an observation.</span></a>
     <a class="gb-card" href="architecture/model-client.html"><strong>Model client</strong><span>LLM, Response and the loaded-model check.</span></a>
     <a class="gb-card" href="architecture/trajectory.html"><strong>Trajectory</strong><span>The logbook of runs and steps, and how it differs from memory.</span></a>
     <a class="gb-card" href="architecture/configuration.html"><strong>Configuration</strong><span>Settings, the .env file and the log files.</span></a>
     <a class="gb-card" href="api/index.html"><strong>API reference</strong><span>Every module, generated from the docstrings.</span></a>
   </div>

.. note::

   ``TinyAgent`` makes one model call per task. It remembers the conversation in ``memory``, can parse one tool call
   from the reply and run it, and records every step in ``trajectory``. The multi-step loop and the planner are
   not built yet. Around it there is a working model client, a model check, a terminal chat (``tiny-agent``) and a
   Docker chat app that serves local models. See :doc:`architecture/overview`.

.. note::

   Based on *An Illustrated Guide to AI Agents* by Maarten Grootendorst and
   Jay Alammar (O'Reilly). The `companion repository
   <https://github.com/HandsOnLLM/An-Illustrated-Guide-To-AI-Agents>`_ has no
   license, so all rights remain with the authors. This is an independent
   learning project, not affiliated with or endorsed by them.

.. toctree::
   :hidden:
   :caption: Getting started

   getting-started/installation
   getting-started/quickstart

.. toctree::
   :hidden:
   :caption: Concepts

   architecture/overview
   architecture/agent
   architecture/memory
   architecture/tools
   architecture/model-client
   architecture/trajectory
   architecture/configuration

.. toctree::
   :hidden:
   :caption: Guides

   guides/models
   guides/chat-app
   guides/development

.. toctree::
   :hidden:
   :caption: Reference

   api/index
   api/illustrated-agents-scripts
   api/chat-app-backend
