import pytest
from barcode_qc.conftest import runtime, fixture_factory, qc_case
from agent.tools.data.scatac_barcode_qc import compute_scATAC_qc


@pytest.fixture
def selection_case(qc_case,tmp_path):
    result=compute_scATAC_qc(**qc_case[0])
    return dict(barcode_qc_manifest_path=result['manifest_path'],barcode_qc_manifest_sha256=result['manifest_sha256'],
        min_qc_fragment_records=0,min_tss_enrichment='0',output_dir=str(tmp_path/'selection')),qc_case
