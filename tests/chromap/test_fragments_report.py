"""Figureless deterministic prose must preserve fragment scientific semantics."""
import hashlib
import json
from pathlib import Path
import pytest
from agent.report import (build_analysis_evidence, build_analysis_report,
    get_supported_visualization_kinds, verify_analysis_report)
from agent.report import analysis_report as r
from test_fragments_contracts import bound, fake_runtime
from test_fragments_orchestration import configured
from test_fragments_evidence import fragment_run


@pytest.fixture
def fragment_report(fragment_run, tmp_path):
    run, registry, control, *_ = fragment_run
    evidence = build_analysis_evidence(run, tmp_path / 'evidence', registry=registry)
    report = build_analysis_report(run, evidence, tmp_path / 'report', registry=registry)
    return run, registry, control, evidence, report


def test_fragment_report_precise_figureless_deterministic(fragment_report, tmp_path):
    run, registry, control, evidence, report = fragment_report
    text = Path(report['report_path']).read_text()
    assert '## Raw scATAC Preprocessing' in text and 'independent verification passed' in text
    assert 'Total exact support: ` 300 `' in text
    assert 'distinct barcodes represented in canonical fragments' in text
    assert 'including the retained representative' in text
    assert 'Observed accepted fragment barcodes are not called cells or QC-passed cells.' in text
    assert 'No cell calling, QC, or cCRE matrix construction was performed.' in text
    assert 'TSS enrichment' not in text and 'FRiP' not in text and 'Number of cells' not in text
    assert run.steps[0].result['manifest_path'] in text
    assert run.steps[0].result['manifest_sha256'] in text
    assert get_supported_visualization_kinds(run, evidence, registry=registry) == ()
    assert '## Figures' not in text and '![' not in text
    second = build_analysis_report(run, evidence, tmp_path / 'repeat', registry=registry)
    assert Path(report['report_path']).read_bytes() == Path(second['report_path']).read_bytes()
    assert Path(report['manifest_path']).read_bytes() == Path(second['manifest_path']).read_bytes()
    assert verify_analysis_report(run, evidence, report, registry=registry).passed
    assert control['calls'].count('chromap') == 1


@pytest.mark.parametrize('mutation', ['support', 'cells', 'qc', 'figure'])
def test_fragment_report_rejects_rehashed_scientific_forgery(fragment_report, mutation):
    run, registry, _, evidence, report = fragment_report
    path = Path(report['report_path']); before = path.read_text()
    after = {'support': before.replace('` 300 `', '` 255 `'),
        'cells': before.replace('distinct barcodes represented in canonical fragments', 'called cells'),
        'qc': before + '\nBiological library passed QC.\n',
        'figure': before + '\n![QC](figures/invented.png)\n'}[mutation]
    assert after != before
    path.write_text(after)
    manifest = Path(report['manifest_path'])
    payload = json.loads(manifest.read_bytes())
    payload = json.loads(json.dumps(payload).replace(hashlib.sha256(before.encode()).hexdigest(),
        hashlib.sha256(after.encode()).hexdigest()))
    manifest.write_bytes(r._canonical_json_bytes(payload))
    assert not verify_analysis_report(run, evidence, manifest, registry=registry).passed
