"""No-number HTTP acceptance with real tiny science and synthetic qualification."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
from test_selection_clarification import web, payload, post
from test_scientific_parameter_continuation import (
    runtime, fixture_factory, selection_harness, OBJECTIVE, REQUIRED,
)
from agent.tools.data.scatac_barcode_qc import compute_scATAC_qc
from rich_helpers import ScientificWork


def test_no_number_http_task_selects_with_defaults_and_exposes_exact_evidence(tmp_path, fixture_factory):
    # Distinct canonical records, not inflated read-support. One defined-TSS
    # barcode passes; another has no TSS background and remains rejected.
    rows = [('chr2', 1000, 1001, 'A', 1), ('chr2', 2000, 2100, 'B', 1)]
    rows += [('chr2', start, end, 'A', 1)
             for start in range(2950, 3050) for end in range(3051, 3061)]
    args, *_ = fixture_factory(rows=rows)
    qc = compute_scATAC_qc(**args)
    inputs = dict(barcode_qc_manifest_path=qc['manifest_path'],
                  barcode_qc_manifest_sha256=qc['manifest_sha256'])
    harness = selection_harness(tmp_path / 'application')
    work = ScientificWork()
    try:
        with TestClient(web(harness, inputs)) as client:
            client.post('/api/v1/sessions', json={'session_id': 'session'})
            view = post(client, payload(selected=True))
            assert view['status'] == 'succeeded', view
            run = harness.service._application.run_store.load(view['run_id'])
            assert run.request.prompt == OBJECTIVE and not any(c.isdigit() for c in OBJECTIVE)
            step, = run.steps
            assert not set(REQUIRED).intersection(step.resolved_arguments)
            assert step.result['n_selected'] == 1 and step.result['n_rejected'] == 1
            manifest = json.loads(Path(step.result['manifest_path']).read_bytes())
            assert manifest['thresholds']['min_qc_fragment_records'] == 1000
            assert manifest['thresholds']['min_tss_enrichment'] == dict(numerator=4, denominator=1)
            assert manifest['reason_counts']['TSS_ENRICHMENT_UNDEFINED'] == 1
            assert step.verification.passed and step.verification.artifact_authority['schema_version'] == 2
            route = '/api/v1/sessions/session/revisions/' + view['revision_id']
            evidence = client.get(route + '/evidence', params={'output_name': 'selection'}).json()
            facts = {f['field']: f['value'] for f in evidence['facts']}
            assert facts['effective_thresholds'] == manifest['thresholds']
            assert all(facts['selection_threshold_origins'][k] == 'existing_owner_default' for k in REQUIRED)
            report, = [a for a in client.get(route + '/artifacts').json()['artifacts']
                       if a['artifact_type'] == 'analysis_report']
            delivered = client.get(route + '/artifacts/' + report['handle'], params={'download': True})
            assert delivered.status_code == 200
            assert '1000' in delivered.text and 'min_tss_enrichment' in delivered.text
            assert 'can be adjusted' in delivered.text and 'pragmatic starting values' in delivered.text
            assert str(tmp_path) not in delivered.text
            original = next((tmp_path / 'application').rglob('analysis_report.md')).read_text()
            assert 'pragmatic starting values' in original and 'can be adjusted' in original
            counts = work.snapshot()
            assert counts['production.selection'] == counts['owner.selection'] == 1
            assert post(client, payload(selected=True)) == view
            assert work.snapshot() == counts
    finally:
        work.close()
