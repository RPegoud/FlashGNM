import os

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="also run tests marked slow")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long statistical tests, run with --slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="slow: pass --slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def triton_runnable(device):
    """Triton needs a GPU, or TRITON_INTERPRET=1 (set before triton is imported) on CPU."""
    return device.type == "cuda" or os.environ.get("TRITON_INTERPRET") == "1"
