"""Fresh fragments evidence, bounded facts and exact artifact protection."""
from dataclasses import replace
import json
from pathlib import Path
import pytest

from agent.orchestration import AgentRequest, AgentRuntime, ToolRegistry, build_default_tool_registry
from agent.report import AnalysisEvidenceError, build_analysis_evidence, verify_analysis_evidence
from agent.report import evidence as e
from fragments_helpers import BC
from test_fragments_contracts import bound, fake_runtime
from test_fragments_orchestration import configured, FixedPlanner, plan_for


@pytest.fixture
def fragment_run(configured):
    args, control, group, white, _ = configured
    registry = build_default_tool_registry()
    run = AgentRuntime(registry=registry, planner=FixedPlanner(plan_for(args))).run(
        AgentRequest('fragments-request', 'Prepare canonical fragments.', {}))
    assert run.status.value == 'SUCCEEDED', run.errors
    return run, registry, control, group, white


def mutate_fragment(run, kind, white):
    result = run.steps[-1].result
    path = Path(result['manifest_path'])
    manifest = json.loads(path.read_bytes())
    target = Path(white.resource.path) if kind == 'whitelist' else (
        path if kind == 'manifest' else path.parent / manifest['libraries'][0][kind]['path'])
    with target.open('ab') as stream:
        stream.write(b'changed')


def test_fragments_evidence_facts_artifacts_and_determinism(fragment_run, tmp_path):
    run, registry, control, *_ = fragment_run
    first = build_analysis_evidence(run, tmp_path / 'evidence', registry=registry)
    payload = json.loads(Path(first['evidence_path']).read_bytes())
    facts = payload['steps'][0]['facts']
    assert facts['input_kind'] == 'FASTQ' and facts['species'] == 'human' and facts['assembly'] == 'hg38'
    assert facts['n_fragment_records'] == 1 and facts['total_support'] == 300
    assert facts['backend_policy'] == 'chromap-atac-agent-support-v1'
    assert facts['library_summary'] == [dict(namespace='library_0', n_fragment_records=1,
        total_support=300, n_distinct_fragment_barcodes=1, max_support=300)]
    assert facts['n_libraries_omitted_from_summary'] == 0
    assert payload['steps'][0]['verification']['freshly_verified']
    artifacts = payload['artifacts']
    assert [a['artifact_kind'] for a in artifacts] == [
        'scatac_fragments_manifest_json', 'scatac_fragments_bgzf', 'scatac_fragments_tabix',
        'fastq_fragment_profile', 'fastq_fragment_production_record']
    import hashlib
    for artifact in artifacts:
        assert artifact['integrity']['authoritative_digest']['value'] == hashlib.sha256(
            Path(artifact['artifact_path']).read_bytes()).hexdigest()
    assert 'full_artifact_sha256' in artifacts[1]['integrity']['verification_basis']
    assert 'tabix_contig_inventory_full_contig_and_narrow_overlap_queries' in artifacts[2]['integrity']['verification_basis']
    assert BC not in json.dumps(payload)
    assert not {'n_cells', 'reads', 'barcodes', 'scan', 'qc', 'frip'} & facts.keys()
    assert verify_analysis_evidence(run, first, registry=registry).passed
    second = build_analysis_evidence(run, tmp_path / 'repeat', registry=registry)
    assert Path(first['evidence_path']).read_bytes() == Path(second['evidence_path']).read_bytes()
    assert control['calls'].count('chromap') == 1


@pytest.mark.parametrize('kind', ['bgzf', 'tabix', 'whitelist', 'manifest'])
def test_fragments_evidence_rejects_fresh_corruption(fragment_run, tmp_path, kind):
    run, registry, _, _, white = fragment_run
    evidence = build_analysis_evidence(run, tmp_path / 'before', registry=registry)
    mutate_fragment(run, kind, white)
    with pytest.raises(AnalysisEvidenceError, match='Fresh source-step verification'):
        build_analysis_evidence(run, tmp_path / 'after', registry=registry)
    assert not verify_analysis_evidence(run, evidence, registry=registry).passed
    assert not (tmp_path / 'after' / 'analysis_evidence.json').exists()


@pytest.mark.parametrize('change', ['policy', 'schema', 'fact'])
def test_fragments_projection_closed_and_authoritative(fragment_run, tmp_path, change):
    run, registry, *_ = fragment_run
    if change == 'fact':
        evidence = build_analysis_evidence(run, tmp_path / 'evidence', registry=registry)
        path = Path(evidence['evidence_path'])
        payload = json.loads(path.read_bytes())
        payload['steps'][0]['facts']['total_support'] = 255
        path.write_bytes(e._canonical_json_bytes(payload))
        assert not verify_analysis_evidence(run, path, registry=registry).passed
    else:
        spec = registry.get('prepare_scATAC_fragments')
        spec = replace(spec, recovery_policy_version='changed') if change == 'policy' else replace(spec,
            result_contract=replace(spec.result_contract, required_fields={**spec.result_contract.required_fields, 'new': (str,)}))
        changed = ToolRegistry(tuple(spec if n == spec.name else registry.get(n) for n in registry.names()))
        with pytest.raises(AnalysisEvidenceError) as caught:
            build_analysis_evidence(run, tmp_path / 'evidence', registry=changed)
        assert caught.value.code == 'EVIDENCE_TOOL_SCHEMA_INCOMPATIBLE'


def test_fragments_projection_rechecks_after_step_verification(fragment_run, tmp_path, monkeypatch):
    run, registry, _, _, white = fragment_run
    original = e.verify_step
    def changed(*args, **kwargs):
        verified = original(*args, **kwargs)
        assert verified.passed
        mutate_fragment(run, 'bgzf', white)
        return verified
    monkeypatch.setattr(e, 'verify_step', changed)
    with pytest.raises(AnalysisEvidenceError) as caught:
        build_analysis_evidence(run, tmp_path / 'evidence', registry=registry)
    assert caught.value.code == 'EVIDENCE_SOURCE_RESULT_INVALID'


def test_library_summary_bound_is_explicit(monkeypatch):
    from agent.report.fragments import project_fragments, LIBRARY_SUMMARY_LIMIT
    from agent.tools.data import scatac_fragments
    entries = [dict(namespace=f'library_{i:03}', n_fragment_records=1, sum_support=300,
        max_support=300, n_distinct_barcodes=1,
        bgzf={'path': f'library_{i}.gz', 'sha256': '1'*64},
        tabix={'path': f'library_{i}.tbi', 'sha256': '2'*64}) for i in range(52)]
    manifest = {'libraries': entries, 'backend': {'backend_policy': 'chromap-atac-agent-support-v1',
        'upstream_version': '0.3.2', 'upstream_commit': 'pinned'}}
    from agent.tools.data import fastq_fragment_manifest as producer
    for entry in entries:
        entry['provenance'] = {'profile': {'resource': {'path':'/synthetic/profile.json','sha256':'3'*64}},
            'producer_record': {'path':'/synthetic/production.json','sha256':'4'*64}}
    monkeypatch.setattr(producer, 'record_for_manifest', lambda *a: manifest)
    # Presentation-bound test only; scientific verification is tested above.
    monkeypatch.setattr(scatac_fragments, 'verify_public_result', lambda *a: manifest)
    facts, artifacts = project_fragments({}, {'manifest_path': '/synthetic/manifest.json'}, 'fragments')
    assert len(facts['library_summary']) == LIBRARY_SUMMARY_LIMIT
    assert facts['n_libraries_omitted_from_summary'] == 2 and len(artifacts) == 106


def test_independent_namespaces_remain_separate_in_evidence_and_report(tiny, reads, fake_runtime, monkeypatch):
    from fragments_helpers import inputs_for
    from test_fragments_orchestration import public_args
    from agent.report import build_analysis_report
    one, white = reads([(100, 200, BC, 300)], name='one')
    two, _ = reads([(100, 200, BC, 300)], name='two')
    inputs = inputs_for(tiny, [one, two], white)
    indexes = tiny['root'] / 'indexes'; indexes.mkdir()
    Path(inputs.index_path).parent.rename(indexes / 'accepted')
    runtime, control = fake_runtime
    monkeypatch.setenv('AGENT_CHROMAP_BIN', runtime.chromap)
    monkeypatch.setenv('AGENT_CHROMAP_INDEX_ROOT', str(indexes))
    registry = build_default_tool_registry()
    args = public_args(inputs, tiny['root'] / 'output')
    run = AgentRuntime(planner=FixedPlanner(plan_for(args))).run(
        AgentRequest('fragments-request', 'Prepare independent libraries.', {}))
    assert run.status.value == 'SUCCEEDED', run.errors
    evidence = build_analysis_evidence(run, tiny['root'] / 'evidence', registry=registry)
    payload = json.loads(Path(evidence['evidence_path']).read_bytes())
    facts = payload['steps'][0]['facts']
    assert facts['n_libraries'] == 2 and facts['total_support'] == 600
    assert [row['namespace'] for row in facts['library_summary']] == ['library_0', 'library_1']
    assert [row['n_distinct_fragment_barcodes'] for row in facts['library_summary']] == [1, 1]
    assert len(payload['artifacts']) == 7
    report = build_analysis_report(run, evidence, tiny['root'] / 'report', registry=registry)
    text = Path(report['report_path']).read_text()
    assert 'library_0' in text and 'library_1' in text and BC not in text
    assert control['calls'].count('chromap') == 2  # One invocation per independent library.
