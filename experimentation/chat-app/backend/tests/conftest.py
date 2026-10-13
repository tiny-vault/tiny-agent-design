"""pytest fixtures for the backend tests: a fake llama-server and clients that talk to it.

The backend is tested without Docker or a GPU. ``FakeRouter`` (``fake_router.py``) plays
llama-server in router mode on a local port, and each ``client`` is the real FastAPI app
pointed at it, so a test exercises the real HTTP and error paths. If the real llama.cpp
behaves differently from the fake, fix the fake (see the repo's ``AGENTS.md``).
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# The backend modules (server.py, llama_client.py) and fake_router.py are not installed packages,
# so put their folders on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_router import FakeRouter  # noqa: E402

from server import create_app  # noqa: E402

# Short waits so load, failure and timeout cases finish in a fraction of a second instead of minutes.
FAST = dict(poll_interval=0.02, start_grace=1, load_timeout=3)


@pytest.fixture(scope="session")
def router_server():
    """One fake llama-server for the whole test run (starting a server per test would be slow)."""
    router = FakeRouter()
    yield router
    router.close()


@pytest.fixture
def router(router_server):
    """The shared fake router, put back to its starting state before each test."""
    router_server.reset()
    return router_server


@pytest.fixture
def client(router):
    """A test client for the real backend app, wired to the fake router."""
    with TestClient(create_app(router.url, **FAST)) as c:
        yield c


@pytest.fixture
def offline_client():
    """A backend whose llama-server is not running."""
    with TestClient(create_app("http://127.0.0.1:9", **FAST)) as c:
        yield c
