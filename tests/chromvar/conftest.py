"""Simulated science contracts; genuine containment is independently tested."""
import pytest

@pytest.fixture(autouse=True)
def simulated_process_boundary(monkeypatch):
    from agent.tools import _native_process
    monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
