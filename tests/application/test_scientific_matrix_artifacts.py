"""Accepted metadata resolution fixtures; no biological acceptance or science.

Five closed owner manifests and schema-2 records exercise the Application's
delivery selector. Adversarial transport/privacy tests use separate H5AD bytes.
"""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from agent.application import InteractiveBoundaryError
from agent.application.session_state import Interaction, digest
from agent.application.interactive_schemas import PresentedResponse
from agent.orchestration import AgentPlan, PlanStep
from agent.orchestration.durable_tool_recovery import execution_identity
from agent.schemas.verification_authority import VerifiedArtifactAuthority
from agent.tools.data import (scatac_matrix_contract as canonical,
    external_matrix_contract as external, regulatory_matrix_contract as neutral_external,
    fragment_feature_matrix_contract as neutral, selected_feature_matrix_contract as selected)
from test_dialogue_evidence import app, accepted, files, forbid_work, passed
from test_interactive_rich import read_only_boundary


CASES = (
    ('build_scATAC_cell_by_ccre', canonical),
    ('adopt_scATAC_cell_by_ccre', external),
    ('build_scATAC_cell_by_features', neutral),
    ('build_scATAC_cell_by_features', selected),
    ('adopt_scATAC_cell_by_features', neutral_external),
)


def matrix_manifest(contract, payload):
    """A valid tiny owner metadata contract, without scientific reconstruction."""
    is_external = contract in (external, neutral_external)
    is_neutral = contract is not canonical and contract is not external
    pointer = lambda name, version: dict(manifest_path='/declared/' + name + '.json',
        manifest_sha256='a' * 64, identity_sha256='b' * 64, contract_version=version)
    summary = canonical.logical_matrix_identity(2, 3, [([1], [2]), ([], [])])
    value = dict(artifact_type=contract.ARTIFACT, schema_version=1, contract_version=contract.CONTRACT,
        profile=contract.PROFILE if is_external else asdict(contract.PROFILE),
        profile_sha256=contract.PROFILE_SHA256,
        species=dict(scientific_name='Macaca fascicularis', taxonomy_id=9541) if is_neutral else 'human',
        assembly='macFas5' if is_neutral else 'hg38', shape=[2, 3], **summary,
        ordered_feature_sha256='d' * 64,
        matrix=dict(path='matrix.h5ad', sha256=hashlib.sha256(payload).hexdigest(),
                    size_bytes=len(payload), format='csr', dtype='int64'), readiness='matrix_available')
    if is_external:
        value.update(operation='external_adoption', source=dict(path='/declared/source.h5ad',
            sha256='e' * 64, size_bytes=100), reference=pointer('reference', contract.REFERENCE_CONTRACT),
            matrix_semantics='fragment_counts', ordered_cells_sha256='c' * 64,
            source_logical_matrix_sha256=summary['logical_matrix_sha256'])
        value['identity_sha256'] = contract.identity(value)
    else:
        cell_key = 'cells' if is_neutral else 'selection'
        cell_contract = contract.CELL_CONTRACT if is_neutral else 'scatac-cell-selection.v1'
        value.update(upstream=dict(fragments=pointer('fragments', 'scatac-fragments.v2'),
            reference=pointer('reference', 'regulatory-feature-reference.v1' if is_neutral
                              else 'scatac-reference-bundle.v1'),
            **{cell_key: pointer(cell_key, cell_contract)}), ordered_selected_sha256='c' * 64,
            backend=dict(profile_id='bedtools-ccre-record-incidence.v1', runtime_sha256='f' * 64),
            diagnostic=contract.overlap_diagnostic(4, 2))
        value['identity_sha256'] = contract.manifest_identity(value)
    return canonical.load_manifest_bytes(canonical.canonical(value))


def capture_execute_display(application, name):
    """Attach an exact successful execute display to a test-only accepted turn."""
    state = application.sessions.load('session')
    turn = state.turn(name)
    relations = {} if turn.base_revision_id is None else {'current': turn.base_revision_id}
    base = next((r for r in state.revisions if r.revision_id == turn.base_revision_id), None)
    if base is not None and base.parent_revision_id is not None:
        relations['parent'] = base.parent_revision_id
    previous = next((n.from_revision_id for n in reversed(state.navigation[:turn.base_generation])
                     if n.to_revision_id != n.from_revision_id), None)
    if previous is not None:
        relations['previous_active'] = previous
    snapshot = dict(relations=relations, bases={r.revision_id: dict(revision_id=r.revision_id,
        outputs=[asdict(o) for o in r.outputs]) for r in state.revisions if r.revision_id in relations.values()})
    interaction = Interaction(name, 'Run the declared analysis.', turn.base_revision_id,
        turn.base_generation, snapshot, status='submitted',
        admitted=dict(kind='execute', request_id=turn.request_id),
        presentation=PresentedResponse('execute', turn.status, 'Analysis completed.').to_dict())
    application.sessions._store._write(replace(state, interactions=state.interactions + (interaction,)))


def matrix_accepted(application, monkeypatch, *, name='initial', tool='build_scATAC_cell_by_ccre',
                    contract=canonical, payload=b'synthetic accepted matrix bytes',
                    output_key='manifest_path', result_changes=None, authority_changes=None,
                    accepted_authority=True):
    """Create ordinary accepted Session records with explicitly scripted authority.

    The fixture does not run a matrix owner. Its bytes are transport test inputs,
    and its metadata must not be presented as real scientific qualification.
    """
    import test_dialogue_evidence as fixtures
    root = application._workspace.run_paths(name + ':run').scientific
    directory = root / 'cell-by-ccre-fixture' / 'artifact'
    directory.mkdir(parents=True)
    matrix = directory / 'matrix.h5ad'
    matrix.write_bytes(payload)
    manifest = matrix_manifest(contract, payload)
    raw = canonical.canonical(manifest)
    path = directory / 'manifest.json'
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    if tool == 'build_scATAC_cell_by_ccre':
        from agent.tools.data.scatac_matrix import _summary
        result = _summary(manifest, path, sha)
    elif tool == 'build_scATAC_cell_by_features':
        from agent.tools.data.fragment_feature_matrix import _summary
        result = _summary(manifest, path, sha)
    else:
        from agent.tools.data.scatac_matrix_adoption import _summary
        result = _summary(manifest, path, sha, contract=contract)
    arguments = dict(output_dir=str(root))
    plan = AgentPlan(name + ':plan', name, 'fixture', (PlanStep('step0', tool, arguments),))
    identity = execution_identity(name + ':run', plan, plan.steps[0], application.registry.get(tool))
    proof = {k: manifest[k] for k in ('identity_sha256', 'logical_matrix_sha256', 'nnz',
                                   'total_count', 'zero_row_count')}
    if 'diagnostic' in manifest:
        proof['diagnostic'] = manifest['diagnostic']
    record = dict(schema_version=2, artifact_type=contract.ARTIFACT, artifact_contract=contract.CONTRACT,
        publication_path=str(path), manifest_sha256=sha,
        files=sorted([dict(path=str(path), sha256=sha, size_bytes=len(raw)),
                      dict(path=str(matrix), sha256=manifest['matrix']['sha256'], size_bytes=len(payload))],
                     key=lambda f: f['path']),
        execution_identity=identity, arguments_sha256='1' * 64, receipt_sha256='2' * 64,
        upstream={'fixture': '3' * 64}, resources=dict(scientific_proof_sha256='4' * 64,
            manifest_identity=manifest, result_metadata=proof), science_profile=contract.PROFILE_SHA256,
        producer_qualification={}, verifier=dict(id='fixture', compatibility_version='1'),
        scope='scientific_correctness.v1', completion='succeeded',
        integrity_scope='artifact_integrity_lineage_historical_sources.v1',
        source_policy='historical_verified_sources.v1', historical_sources=[])
    record.update(authority_changes or {})
    authority = VerifiedArtifactAuthority(record)
    def fixture_verification(kind, identifier):
        original = passed(kind, identifier)
        return (replace(original, artifact_authority=authority.to_dict())
                if kind == 'step' and accepted_authority else original)
    with monkeypatch.context() as scoped:
        scoped.setattr(fixtures, 'passed', fixture_verification)
        revision, _, _ = accepted(application, tool, name=name,
            overrides=result | (result_changes or {}), arguments=arguments)
    if output_key != 'manifest_path':
        # Test-only envelope rewrite changes the explicit selected semantic port;
        # persisted run/step anchors and accepted bytes are preserved.
        state = application.sessions.load('session')
        old = next(r for r in state.revisions if r.revision_id == revision)
        output = replace(old.outputs[0], output_key=output_key)
        updated_revision = replace(old, outputs=(output,))
        turns = tuple(replace(t, selections=(replace(t.selections[0], output_key=output_key),))
                      if t.turn_id == name else t for t in state.turns)
        updated = replace(state, turns=turns,
            revisions=tuple(updated_revision if r.revision_id == revision else r for r in state.revisions))
        application.sessions._store._write(updated)
    capture_execute_display(application, name)
    return revision, matrix, manifest


@pytest.mark.parametrize('tool,contract', CASES, ids=[c.CONTRACT for _, c in CASES])
def test_all_five_exact_contracts_inventory_without_payload_reads(app, monkeypatch, tool, contract):
    revision, matrix, manifest = matrix_accepted(app, monkeypatch, tool=tool, contract=contract)
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    before = files(app)
    original = Path.open
    original_open = os.open
    def bounded_metadata(path, *args, **kwargs):
        if path == matrix:
            pytest.fail('Inventory opened matrix payload bytes.')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', bounded_metadata)
    def bounded_open(path, *args, **kwargs):
        if Path(path) == matrix or str(path) == 'matrix.h5ad':
            pytest.fail('Inventory opened matrix payload descriptor.')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(os, 'open', bounded_open)
    handles = facade.scientific_artifact_handles('session', revision)
    assert len(handles) == 1
    item = handles[0]
    assert item.sha256 == manifest['matrix']['sha256'] and item.sha256 != digest(manifest)
    assert item.size_bytes == matrix.stat().st_size and item.filename == 'matrix.h5ad'
    assert item.label == 'Original accepted matrix' and item.artifact_type == 'scientific_matrix'
    assert facade.scientific_artifacts_for_turn('session', 'initial') == handles
    assert facade._scientific_artifact('session', revision, item.handle).path == matrix
    assert str(app.workspace_root) not in json.dumps(item.to_dict())
    monkeypatch.setattr(Path, 'open', original)
    assert files(app) == before


@pytest.mark.parametrize('output_key', ['manifest_path', 'manifest_sha256', 'matrix_path'])
def test_matrix_and_dataset_semantic_ports(app, monkeypatch, output_key):
    revision, matrix, _ = matrix_accepted(app, monkeypatch, output_key=output_key)
    facade = read_only_boundary(app)
    reference, = facade.scientific_artifact_handles('session', revision)
    assert facade._scientific_artifact('session', revision, reference.handle).path == matrix


def test_historical_turns_reopen_and_navigation_preserve_exact_handles(app, monkeypatch):
    old, _, _ = matrix_accepted(app, monkeypatch)
    newer, _, _ = matrix_accepted(app, monkeypatch, name='later', payload=b'second accepted payload')
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    historical, = facade.scientific_artifacts_for_turn('session', 'initial')
    recent, = facade.scientific_artifacts_for_turn('session', 'later')
    assert historical.handle != recent.handle and historical.sha256 != recent.sha256
    facade.activate_revision('session', 'navigate', old, expected_generation=2)
    assert facade.scientific_artifacts_for_turn('session', 'navigate') == ()
    restarted = read_only_boundary(app)
    assert restarted.scientific_artifacts_for_turn('session', 'initial') == (historical,)
    assert restarted.scientific_artifacts_for_turn('session', 'later') == (recent,)
    with pytest.raises(InteractiveBoundaryError):
        restarted._scientific_artifact('session', newer, historical.handle)


def test_retained_matrix_uses_actual_originating_run_and_is_not_a_new_chat_file(app, monkeypatch):
    first, matrix, _ = matrix_accepted(app, monkeypatch)
    later, _, _ = accepted(app, 'inspect_scATAC', name='later')
    state = app.sessions.load('session')
    source = next(r for r in state.revisions if r.revision_id == first).outputs[0]
    retained = source
    current = next(r for r in state.revisions if r.revision_id == later)
    updated = replace(state, revisions=tuple(replace(r,
        outputs=(replace(r.outputs[0], name='inspection'), retained))
        if r.revision_id == later else r for r in state.revisions),
        turns=tuple(replace(t, retained_outputs=(retained,),
            selections=(replace(t.selections[0], name='inspection'),))
            if t.turn_id == 'later' else t for t in state.turns))
    app.sessions._store._write(updated)
    capture_execute_display(app, 'later')
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    retained_handle, = facade.scientific_artifact_handles('session', current.revision_id)
    resolved = facade._scientific_artifact('session', later, retained_handle.handle)
    assert resolved.path == matrix and resolved.managed_root == matrix.parents[2]
    displayed = facade.turn('session', 'later')
    assert displayed.status == 'succeeded' and displayed.response.kind == 'execute'
    assert displayed.response.status == 'activated'
    assert facade.scientific_artifacts_for_turn('session', 'later') == ()


@pytest.mark.parametrize('case', ['forged', 'result', 'authority', 'manifest', 'failed', 'unaccepted', 'wrong_port'])
def test_unaccepted_or_conflicting_identities_fail_closed(app, monkeypatch, case):
    kwargs = {}
    if case == 'result': kwargs['result_changes'] = {'matrix_sha256': '0' * 64}
    if case == 'authority': kwargs['authority_changes'] = {'artifact_contract': external.CONTRACT}
    if case == 'wrong_port': kwargs['output_key'] = 'nnz'
    if case == 'unaccepted': kwargs['accepted_authority'] = False
    revision, matrix, _ = matrix_accepted(app, monkeypatch, **kwargs)
    facade = read_only_boundary(app)
    if case == 'forged':
        with pytest.raises(InteractiveBoundaryError): facade._scientific_artifact('session', revision, '0' * 64)
        return
    if case == 'manifest': (matrix.parent / 'manifest.json').write_text('{}')
    if case == 'failed':
        import agent.orchestration.verification_authority as owner
        def invalid(*args):
            raise ValueError('Unaccepted failed step.')
        monkeypatch.setattr(owner, '_accepted', invalid)
    with pytest.raises(InteractiveBoundaryError): facade.scientific_artifact_handles('session', revision)
    assert facade.revision('session', revision).scientific_artifacts == ()


@pytest.mark.parametrize('case', ['absent_display', 'answer', 'failed', 'wrong_request'])
def test_chat_inventory_requires_an_exact_successful_execute_display(app, monkeypatch, case):
    revision, _, _ = matrix_accepted(app, monkeypatch)
    state = app.sessions.load('session')
    interaction = state.interactions[0]
    if case == 'absent_display': interaction = replace(interaction, presentation=None)
    if case == 'answer':
        interaction = replace(interaction, presentation=PresentedResponse('answer', 'answered', 'Answer.').to_dict())
    if case == 'failed':
        interaction = replace(interaction, presentation=PresentedResponse('execute', 'failed', 'Failed.').to_dict())
    if case == 'wrong_request':
        interaction = replace(interaction, admitted=dict(kind='execute', request_id='other'))
    app.sessions._store._write(replace(state, interactions=(interaction,)))
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    assert len(facade.scientific_artifact_handles('session', revision)) == 1
    assert facade.scientific_artifacts_for_turn('session', 'initial') == ()


@pytest.mark.parametrize('case', ['payload_closure', 'manifest_closure', 'manifest_identity', 'proof', 'scope'])
def test_authority_payload_metadata_must_agree(app, monkeypatch, case):
    revision, _, _ = matrix_accepted(app, monkeypatch)
    facade = read_only_boundary(app)
    from agent.orchestration import verification_authority as owner
    original = owner._accepted
    def changed(*args):
        step, authority, identity, anchor = original(*args)
        record = authority.to_dict()
        if case in {'payload_closure', 'manifest_closure'}:
            suffix = 'matrix.h5ad' if case == 'payload_closure' else 'manifest.json'
            next(f for f in record['files'] if f['path'].endswith(suffix))['sha256'] = '0' * 64
        if case == 'manifest_identity': record['resources']['manifest_identity']['assembly'] = 'changed'
        if case == 'proof': record['resources']['result_metadata']['nnz'] = 99
        if case == 'scope': record['scope'] = 'artifact_integrity_lineage_historical_sources.v1'
        return step, VerifiedArtifactAuthority(record), identity, anchor
    monkeypatch.setattr(owner, '_accepted', changed)
    with pytest.raises(InteractiveBoundaryError): facade.scientific_artifact_handles('session', revision)


def test_inventory_does_not_prevalidate_current_payload_bytes(app, monkeypatch):
    revision, matrix, _ = matrix_accepted(app, monkeypatch)
    facade = read_only_boundary(app)
    accepted_handle, = facade.scientific_artifact_handles('session', revision)
    matrix.unlink()
    assert facade.scientific_artifact_handles('session', revision) == (accepted_handle,)


@pytest.mark.parametrize('case', ['symlink', 'directory', 'fifo', 'ancestor_replacement'])
def test_manifest_nonregular_and_symlink_locations_fail_safely(app, monkeypatch, case):
    revision, matrix, _ = matrix_accepted(app, monkeypatch)
    facade = read_only_boundary(app)
    manifest = matrix.parent / 'manifest.json'
    if case == 'ancestor_replacement':
        from agent.application import matrix_delivery
        original = matrix_delivery._open_matrix_source
        def replaced(path, root):
            parent = path.parent
            moved = parent.with_name('saved-artifact')
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)
            return original(path, root)
        monkeypatch.setattr(matrix_delivery, '_open_matrix_source', replaced)
    else:
        original = manifest.read_bytes()
        manifest.unlink()
        if case == 'symlink':
            target = matrix.parents[2] / 'other-manifest.json'
            target.write_bytes(original)
            manifest.symlink_to(target)
        if case == 'directory': manifest.mkdir()
        if case == 'fifo': os.mkfifo(manifest)
    with pytest.raises(InteractiveBoundaryError): facade.scientific_artifact_handles('session', revision)


@pytest.mark.parametrize('tool', ['inspect_scATAC', 'epizoo_embed_cells', 'compute_cell_umap'])
def test_unrelated_analysis_never_invents_matrix_download(app, monkeypatch, tool):
    revision, _, _ = accepted(app, tool)
    capture_execute_display(app, 'initial')
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    displayed = facade.turn('session', 'initial')
    assert displayed.status == 'succeeded' and displayed.response.kind == 'execute'
    assert displayed.response.status == 'activated'
    assert facade.scientific_artifact_handles('session', revision) == ()
    assert facade.scientific_artifacts_for_turn('session', 'initial') == ()
    assert facade.scientific_artifacts_for_turn('session', 'missing') == ()
