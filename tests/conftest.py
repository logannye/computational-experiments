"""Shared pytest configuration: run PyTorch single-threaded for the whole session."""

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
