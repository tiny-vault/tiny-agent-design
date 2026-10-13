"""pytest setup for the illustrated_agents script tests: registers the ``slow`` marker."""


def pytest_configure(config):
    # Scripts that sample many answers take minutes; "test ias -- -m 'not slow'" skips them
    config.addinivalue_line("markers", "slow: runs a script that makes dozens of model calls")
