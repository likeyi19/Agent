"""Saved v1 identities stay readable without an executable v1 Registry route."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.orchestration import build_default_tool_registry
from agent.orchestration.durable_tool_recovery import execution_identity
from agent.orchestration.verification_authority import _accepted, _load_context
from agent.schemas import (AgentPlan, PlanStep, StepExecutionResult, StepStatus,
                           VerificationResult, RunLifecycleStatus)
from agent.schemas.verification_authority import AuthorityError
from agent.tools._file_identity import file_closure
from agent.tools.analysis import macs3
from agent.tools.data.authority_context import VerificationContext


@pytest.fixture
def historical(tmp_path):
    # A simulated saved native completion, never a scientific execution.
    root = tmp_path / 'runs' / 'history' / 'scientific' / 'peaks'
    (root / 'population').mkdir(parents=True)
    names = (*macs3.FILES, 'population/manifest.json', 'population/cells.tsv.gz', 'reference.json')
    for name in names:
        (root / name).write_bytes(b'simulated saved native payload\n')
    source = tmp_path / 'original-fragments'
    source.write_bytes(b'original source identity\n')
    args = dict(output_dir=str(root), fragments_manifest_path=str(source),
                fragments_manifest_sha256=file_closure([source])[0]['sha256'],
                effective_genome_size=1000, genome_size_basis='Simulated saved estimate')
    plan = AgentPlan('history', 'history', 'Saved v1 request',
        (PlanStep('peaks', macs3.TOOL, args),))
    policy = SimpleNamespace(tools=(SimpleNamespace(tool_name=macs3.TOOL,
        classifier_version=macs3.LEGACY_POLICY),))
    execution = execution_identity('history:run', plan, plan.steps[0],
        SimpleNamespace(name=macs3.TOOL, recovery_policy_version=macs3.LEGACY_POLICY))
    value = dict(contract=macs3.LEGACY_CONTRACT, profile=macs3.LEGACY_PROFILE,
        arguments=args, execution_identity=execution, native_completion='succeeded',
        result=dict(status='succeeded', contract_version=macs3.LEGACY_CONTRACT,
                    n_peaks=1, n_barcodes=1, total_source_support=12),
        reference=dict(manifest_path=str(root/'reference.json'), manifest_sha256='a'*64),
        cells=dict(manifest_path=str(root/'population/manifest.json'), manifest_sha256='b'*64),
        files=file_closure(root/name for name in names), sources=file_closure([source]))
    path = root / 'manifest.json'
    sha = macs3.write(path, value)
    result = macs3.result(value, path, sha)
    context = VerificationContext()
    context.register_execution(macs3.KIND, root, execution)
    authority = macs3.publication_record(context, macs3.TOOL, args, result, execution, result)
    step = StepExecutionResult('peaks', macs3.TOOL, StepStatus.SUCCEEDED, 1, args,
        result, VerificationResult(True, 'step', 'peaks', artifact_authority=authority.record))
    state = SimpleNamespace(run_id='history:run', plan=plan, steps=(step,),
        recovery_policy_snapshot=policy, lifecycle_status=RunLifecycleStatus.SUCCEEDED)
    return SimpleNamespace(path=path, sha=sha, value=value, result=result, args=args,
        authority=authority, state=state, execution=execution, source=source)


def test_historical_contract_profile_execution_and_recovery_remain_exact(historical, monkeypatch):
    h = historical
    monkeypatch.setattr(macs3, 'native', lambda *a: pytest.fail('Historical read executed MACS3'))
    monkeypatch.setattr(macs3, 'prepare_fragments', lambda *a: pytest.fail('Historical read exported fragments'))
    assert h.authority.record['artifact_contract'] == macs3.LEGACY_CONTRACT
    assert h.authority.record['science_profile'] == macs3.LEGACY_PROFILE
    store = SimpleNamespace(load=lambda run_id: h.state)
    assert _accepted(store, 'history:run', 'peaks')[2] == h.execution
    context = _load_context(store, 'history:run', 'historical_verified_sources.v1')
    context.validate_anchors()
    assert macs3.recover(h.args, h.execution) == h.result
    assert macs3.verify_publication(h.path, expected_sha256=h.sha) == h.result
    spec = build_default_tool_registry().get(macs3.TOOL)
    assert spec.scientific_delivery.publication(h.args, h.result, h.authority.record) == (h.value, h.sha)
    assert spec.recovery_policy_version == macs3.POLICY
    from agent.orchestration.registry import ToolArgumentError
    with pytest.raises(ToolArgumentError):
        build_default_tool_registry().validate_arguments(macs3.TOOL, h.args)
    assert not hasattr(macs3, 'selected_population')


def test_historical_native_delivery_keeps_original_contract(historical, tmp_path):
    from agent.application.native_delivery import accepted_artifacts, for_artifact
    h = historical
    registry = build_default_tool_registry()
    app = SimpleNamespace(_application=SimpleNamespace(
        run_store=SimpleNamespace(load=lambda _:h.state), registry=registry,
        _workspace=SimpleNamespace(runs=tmp_path/'runs', run_digest=lambda _:'history')))
    artifacts = accepted_artifacts(app, 'session', SimpleNamespace(run_id='history:run',revision_id='revision'),
        SimpleNamespace(turn_id='turn'), None, step_id='peaks')
    assert artifacts and all(a.contract == macs3.LEGACY_CONTRACT for a in artifacts)
    assert {a.role for a in artifacts} == set(macs3.FILES)
    assert all(for_artifact(a) is not None for a in artifacts)


@pytest.mark.parametrize('tool,contract,manifest_contract,role,accepted', [
    (macs3.TOOL, macs3.CONTRACT, None, 'features.bed', True),
    (macs3.TOOL, macs3.CONTRACT, macs3.CONTRACT, 'features.bed', True),
    (macs3.TOOL, macs3.LEGACY_CONTRACT, macs3.LEGACY_CONTRACT, 'features.bed', True),
    (macs3.TOOL, macs3.LEGACY_CONTRACT, None, 'features.bed', False),
    (macs3.TOOL, macs3.LEGACY_CONTRACT, macs3.CONTRACT, 'features.bed', False),
    (macs3.TOOL, 'macs3-scatac-peaks.v999', 'macs3-scatac-peaks.v999', 'features.bed', False),
    ('run_chromvar', macs3.LEGACY_CONTRACT, macs3.LEGACY_CONTRACT, 'result.tsv', False),
    ('run_chromvar', 'unreviewed.v1', 'unreviewed.v1', 'result.tsv', False),
    ('unknown_tool', macs3.LEGACY_CONTRACT, macs3.LEGACY_CONTRACT, 'features.bed', False),
    (macs3.TOOL, macs3.CONTRACT, macs3.CONTRACT, 'unreviewed.tsv', False),
    (macs3.TOOL, macs3.LEGACY_CONTRACT, macs3.LEGACY_CONTRACT, 'unreviewed.tsv', False),
])
def test_delivery_selection_requires_current_or_specific_historical_contract(
        tool, contract, manifest_contract, role, accepted):
    from agent.application.native_delivery import for_artifact
    manifest = dict(tool_name=tool)
    if manifest_contract is not None:
        manifest['contract'] = manifest_contract
    artifact = SimpleNamespace(contract=contract, manifest=manifest, role=role)
    assert (for_artifact(artifact) is not None) is accepted


@pytest.mark.parametrize('damage', ['file', 'source', 'profile', 'policy', 'execution'])
def test_historical_reads_reject_changed_identity(historical, damage):
    h = historical
    if damage in {'file', 'source'}:
        path = h.path.parent/'features.bed' if damage == 'file' else h.source
        path.write_bytes(b'changed')
        with pytest.raises(AuthorityError):
            macs3.verify_publication(h.path, expected_sha256=h.sha)
    elif damage == 'profile':
        import json
        value = dict(h.value, profile=macs3.PROFILE)
        h.path.write_text(json.dumps(value))
        sha = file_closure([h.path])[0]['sha256']
        with pytest.raises(AuthorityError):
            macs3.load(h.path, sha)
    else:
        if damage == 'policy':
            h.state.recovery_policy_snapshot = SimpleNamespace(tools=())
        else:
            authority = h.authority.to_dict()
            authority['execution_identity'] = '0'*64
            h.state.steps = (replace(h.state.steps[0], verification=VerificationResult(True, 'step', 'peaks',
                artifact_authority=authority)),)
        with pytest.raises(AuthorityError):
            _accepted(SimpleNamespace(load=lambda _:h.state), 'history:run', 'peaks')
