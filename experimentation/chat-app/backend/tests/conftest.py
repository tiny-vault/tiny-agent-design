import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_router import FakeRouter  # noqa: E402
from server import create_app  # noqa: E402

FAST = dict(poll_interval=0.02, start_grace=1, load_timeout=3)


@pytest.fixture(scope="session")
def router_server():
    router = FakeRouter()
    yield router
    router.close()


@pytest.fixture
def router(router_server):
    router_server.reset()
    return router_server


@pytest.fixture
def client(router):
    with TestClient(create_app(router.url, **FAST)) as c:
        yield c


@pytest.fixture
def offline_client():
    """A backend whose llama-server is not running."""
    with TestClient(create_app("http://127.0.0.1:9", **FAST)) as c:
        yield c
