"""Raw retirement keeps shared processed-data and accepted-fragment reporting."""
from agent.report import analysis_report as report


def step(tool,values):
    return report._StepFacts('source',tool,tuple(report._Fact(str(i),'source',tool,key,value)
        for i,(key,value) in enumerate(values.items())))


def test_standalone_raw_projection_and_section_are_absent():
    assert 'inspect_raw_scATAC' not in report._REPORT_FIELDS
    assert not hasattr(report,'_render_raw_intake')


def test_processed_h5ad_report_remains_available():
    data=step('inspect_scATAC',dict(input_path='/selected.h5ad',n_cells=3,n_features=5))
    rendered='\n'.join(report._render_step_block(data,1))
    assert 'Cells: ` 3 `' in rendered and 'Features: ` 5 `' in rendered


def test_shared_historical_fragments_and_readiness_labels_remain():
    data=step('prepare_scATAC_fragments',dict(input_kind='fastq',species='human',assembly='hg19',
        n_libraries=1,n_fragment_records=5,total_support=8,library_summary=[],
        n_libraries_omitted_from_summary=0,backend_policy='pinned',support_definition='Recorded support'))
    rendered='\n'.join(report._render_step_block(data,1))
    assert 'Raw input kind: ` "fastq" `' in rendered and 'Canonical fragment records: ` 5 `' in rendered
    qc=step('compute_scATAC_qc',dict(readiness='READY'))
    assert 'Preprocessing readiness: ` "READY" `' in '\n'.join(report._render_step_block(qc,1))
