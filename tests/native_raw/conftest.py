"""Pure adapter rejection tests avoid requiring an operator service deployment."""
import os
import pytest

@pytest.fixture(autouse=True)
def process_boundary_for_contract_tests(monkeypatch):
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION') != '1':
        from agent.tools import _native_process
        monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
