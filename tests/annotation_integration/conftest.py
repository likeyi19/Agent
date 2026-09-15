import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1]/'authority_dag'))
from external_fragments.conftest import source_factory
from test_propagation import chain as original_chain, completed


@pytest.fixture
def chain(source_factory,tmp_path,monkeypatch):
    from agent.tools.data import scatac_qc_reference as qr
    original = qr.build_scatac_qc_reference_bundle
    def resource(**kwargs):
        # Explicit flank evidence for both observed barcodes; zero minimum
        # enrichment still selects both candidate cells through the real owner.
        Path(kwargs['annotation_path']).write_text(
            'chr2\t2000\t2100\t+\tg1\tt1\tprotein_coding\n'
            'chr1\t2000\t2100\t+\tg2\tt2\tprotein_coding\n')
        return original(**kwargs)
    monkeypatch.setattr(qr,'build_scatac_qc_reference_bundle',resource)
    return original_chain.__wrapped__(SimpleNamespace(param='external_fragment_adoption'),
                                    None,source_factory,tmp_path,monkeypatch)
