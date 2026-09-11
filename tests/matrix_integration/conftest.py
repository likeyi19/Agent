import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import pytest
from cell_by_ccre.conftest import runtime, fixture_factory


@pytest.fixture(autouse=True)
def matrix_runtime(monkeypatch):
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS','/usr/bin/bedtools')


@pytest.fixture
def matrix_case(fixture_factory):
    def make(**kwargs):
        args=fixture_factory(**kwargs)
        args.pop('bedtools_path')
        for suffix in ('path','sha256'):
            args['selected_cells_manifest_'+suffix]=args.pop('selection_manifest_'+suffix)
        return args
    return make
