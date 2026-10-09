"""Closed accepted metadata fixtures exercise delivery without owner execution."""
from dataclasses import replace
from fractions import Fraction
import gzip
import hashlib
import io
import json

import pytest

from agent.application import InteractiveBoundaryError
from agent.orchestration import AgentPlan, PlanStep
from agent.orchestration.durable_tool_recovery import execution_identity
from agent.schemas.verification_authority import VerifiedArtifactAuthority
from agent.tools.data import _barcode_qc_contract as qc, _cell_selection_contract as selection
from agent.tools.data import scatac_barcode_qc, scatac_cell_selection
from agent.tools.data.scatac_selection_profile import encode_cell_id
from test_dialogue_evidence import app, accepted, files, forbid_work, passed
from test_interactive_rich import read_only_boundary
from test_scientific_matrix_artifacts import capture_execute_display


def qc_manifest(arguments, payloads):
    counts = (3, 3, 1, 1, 0, 3, 0, 0)
    summary = dict(zip(qc.COUNTS, counts), depth_min=1, depth_max=2,
        depth_mean=qc.fraction_json(Fraction(3, 2)), tss_defined=1, tss_undefined=1,
        tss_min=qc.fraction_json(Fraction(200, 101)), tss_max=qc.fraction_json(Fraction(200, 101)),
        nucleosome_defined=2, nucleosome_undefined=0,
        nucleosome_min=qc.fraction_json(Fraction(0)), nucleosome_max=qc.fraction_json(Fraction(0)),
        histogram_records=3, histogram_overflow=0)
    value = dict(artifact_type=qc.ARTIFACT, schema_version=1, contract_version=qc.CONTRACT,
        policy=qc.POLICY, arguments=qc.validate_arguments(arguments), reference_identity_sha256='a' * 64,
        qc_resource_identity_sha256='b' * 64, science_profile_sha256=qc.PROFILE_SHA256,
        tss_method=qc.PROFILE.tss_method, producer_authority=dict(kind='external_fragment_adoption',
            profile_ids=['fixture'], verification_basis='fixture', history='fixture'),
        resource_qualification=dict(mode='synthetic_only', catalog_sha256=None, basis='fixture'),
        backend_identity=dict(profile='bedtools-qc-point-incidence.v1', executable_sha256='c' * 64,
            runtime_sha256='d' * 64, compression='gzip-mtime0-level6.v1'),
        ordered_barcode_sha256=hashlib.sha256(qc.ORDER_DOMAIN + b'library\tA\nlibrary\tB\n').hexdigest(),
        row_count=2, summary=summary,
        **{role: dict(path=filename, sha256=hashlib.sha256(payloads[filename]).hexdigest(),
            size_bytes=len(payloads[filename])) for role, filename in
            (('table', 'barcodes.tsv.gz'), ('histogram', 'lengths.tsv.gz'))})
    value['identity_sha256'] = qc.manifest_identity(value)
    return qc.validate_manifest(value)


def table_metadata(kind, arguments):
    first, second = encode_cell_id('library', 'A'), encode_cell_id('library', 'B')
    plain = {
        'barcodes.tsv.gz': qc.HEADER + qc.row_bytes('library', 'A', (2, 2, 1, 1, 0, 2, 0, 0))
            + qc.row_bytes('library', 'B', (1, 1, 0, 0, 0, 1, 0, 0)),
        'lengths.tsv.gz': b''.join(f'{i}\t{3 if i == 100 else 0}\n'.encode() for i in range(1, 1002)),
        'decisions.tsv.gz': selection.DECISION_HEADER
            + f'library\tA\t{first}\tnot_assessed\ttrue\ttrue\tNONE\n'.encode()
            + f'library\tB\t{second}\tnot_assessed\tfalse\tfalse\tTSS_ENRICHMENT_UNDEFINED\n'.encode(),
        'selected.tsv.gz': selection.SELECTED_HEADER + f'0\tlibrary\tA\t{first}\n'.encode(),
    }
    compressed = {}
    for name, value in plain.items():
        stream = io.BytesIO()
        with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0, compresslevel=6) as output:
            output.write(value)
        compressed[name] = stream.getvalue()
    qc_arguments = dict(fragments_manifest_path='/declared/fragments.json', fragments_manifest_sha256='1' * 64,
        qc_reference_manifest_path='/declared/qc-reference.json', qc_reference_manifest_sha256='2' * 64,
        output_dir=arguments['output_dir'])
    q = qc_manifest(arguments if kind == 'qc' else qc_arguments, compressed)
    if kind == 'qc':
        return q, {name: compressed[name] for name in ('barcodes.tsv.gz', 'lengths.tsv.gz')}
    value = dict(artifact_type=selection.ARTIFACT, schema_version=1, contract_version=selection.CONTRACT,
        policy=selection.POLICY, arguments=selection.arguments(arguments), qc_identity_sha256=q['identity_sha256'],
        qc_lineage=selection.lineage(q), selection_profile=selection.PROFILE,
        selection_profile_sha256=selection.PROFILE_SHA256, thresholds=selection.thresholds(arguments),
        row_count=2, selected_count=1, rejected_count=1, cell_call_not_assessed_count=2,
        reason_counts={reason: int(reason == 'TSS_ENRICHMENT_UNDEFINED') for reason in selection.REASONS},
        readiness='selected_candidates_available',
        ordered_selected_sha256=hashlib.sha256(selection.ORDER_DOMAIN + b'library\tA\n').hexdigest(),
        **{role: dict(path=filename, sha256=hashlib.sha256(compressed[filename]).hexdigest(),
            size_bytes=len(compressed[filename])) for role, filename in
            (('decisions', 'decisions.tsv.gz'), ('selected', 'selected.tsv.gz'))})
    value['identity_sha256'] = selection.identity(value)
    return selection.validate(value), {name: compressed[name] for name in ('decisions.tsv.gz', 'selected.tsv.gz')}


def table_accepted(application, monkeypatch, *, kind='qc', name='initial', result_changes=None,
                   authority_changes=None, manifest_change=None, authority_change=None, copies=1):
    """Metadata is scripted accepted; no QC or selection owner is executed."""
    import test_dialogue_evidence as fixtures
    tool = 'compute_scATAC_qc' if kind == 'qc' else 'select_scATAC_cells'
    owner = scatac_barcode_qc if kind == 'qc' else scatac_cell_selection
    root = application._workspace.run_paths(name + ':run').scientific
    if kind == 'qc':
        arguments = dict(fragments_manifest_path='/declared/fragments.json', fragments_manifest_sha256='1' * 64,
            qc_reference_manifest_path='/declared/qc-reference.json', qc_reference_manifest_sha256='2' * 64,
            output_dir=str(root))
    else:
        arguments = selection.arguments(dict(barcode_qc_manifest_path='/declared/qc.json',
            barcode_qc_manifest_sha256='3' * 64, min_qc_fragment_records=0, min_tss_enrichment='0', output_dir=str(root)))
    plan = AgentPlan(name + ':plan', name, 'fixture',
        tuple(PlanStep(f'step{i}', tool, arguments) for i in range(copies)))
    publications = {}
    for planned in plan.steps:
        identity = execution_identity(name + ':run', plan, planned, application.registry.get(tool))
        manifest, payloads = table_metadata(kind, arguments)
        token = (owner._publication_token(arguments, identity, manifest['qc_resource_identity_sha256'],
            manifest['resource_qualification'], manifest['backend_identity']) if kind == 'qc'
            else owner._publication(arguments, identity)[2])
        directory = root / (('barcode-qc-' if kind == 'qc' else 'cell-selection-') + token)
        directory.mkdir(parents=True)
        for filename, payload in payloads.items():
            (directory / filename).write_bytes(payload)
        if manifest_change is not None:
            manifest_change(manifest)
            manifest['identity_sha256'] = qc.manifest_identity(manifest) if kind == 'qc' else selection.identity(manifest)
        raw = qc.canonical(manifest)
        path = directory / 'manifest.json'
        path.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        result = owner._summary(manifest, path, sha)
        record = dict(schema_version=2, artifact_type=manifest['artifact_type'], artifact_contract=manifest['contract_version'],
            publication_path=str(path), manifest_sha256=sha,
            files=sorted([dict(path=str(path), sha256=sha, size_bytes=len(raw)),
                *(dict(path=str(directory / filename), sha256=hashlib.sha256(value).hexdigest(), size_bytes=len(value))
                    for filename, value in payloads.items())], key=lambda f: f['path']),
            execution_identity=identity, arguments_sha256=qc.digest(arguments), receipt_sha256='2' * 64,
            upstream={'fixture': '3' * 64}, resources=dict(scientific_proof_sha256='4' * 64,
                manifest_identity=manifest, result_metadata={}),
            science_profile=manifest['science_profile_sha256' if kind == 'qc' else 'selection_profile_sha256'],
            producer_qualification={}, verifier=dict(id='fixture', compatibility_version='1'),
            scope='scientific_correctness.v1', completion='succeeded',
            integrity_scope='artifact_integrity_lineage_historical_sources.v1',
            source_policy='historical_verified_sources.v1', historical_sources=[])
        record.update(authority_changes or {})
        if authority_change is not None:
            authority_change(record)
        publications[planned.step_id] = (result | (result_changes or {}), VerifiedArtifactAuthority(record), directory, manifest)
    first_result, _, first_directory, first_manifest = publications['step0']
    execution_result = fixtures.StepExecutionResult
    def scripted_step(*args, **kwargs):
        kwargs['result'] = publications[args[0]][0]
        return execution_result(*args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(fixtures, 'passed', lambda target, identifier: replace(passed(target, identifier),
            artifact_authority=publications[identifier][1].to_dict()) if target == 'step' else passed(target, identifier))
        scoped.setattr(fixtures, 'StepExecutionResult', scripted_step)
        revision, _, _ = accepted(application, tool, name=name, overrides=first_result,
            arguments=arguments, copies=copies)
    capture_execute_display(application, name)
    return revision, first_directory, first_manifest


@pytest.mark.parametrize('kind,roles', [('qc', ('table', 'histogram')), ('selection', ('decisions', 'selected'))])
def test_exact_table_roles_inventory_does_not_read_payloads(app, monkeypatch, kind, roles):
    revision, directory, manifest = table_accepted(app, monkeypatch, kind=kind)
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    before = files(app)
    import os
    original = os.open
    def metadata_only(path, *args, **kwargs):
        assert str(path) not in [manifest[role]['path'] for role in roles]
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, 'open', metadata_only)
    handles = facade.scientific_artifact_handles('session', revision)
    assert len(handles) == 2
    assert facade.scientific_artifacts_for_turn('session', 'initial') == handles
    for reference, role in zip(handles, roles):
        assert reference.sha256 == manifest[role]['sha256']
        assert reference.sha256 != hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest()
        assert reference.size_bytes == manifest[role]['size_bytes']
        locator = facade._scientific_artifact('session', revision, reference.handle)
        assert locator.path == directory / manifest[role]['path'] and locator.role == role
        assert locator.managed_root == directory.parent
        assert str(app.workspace_root) not in json.dumps(reference.to_dict())
    assert files(app) == before


@pytest.mark.parametrize('kind', ['qc', 'selection'])
def test_snapshot_original_bytes_and_zero_science(app, monkeypatch, kind):
    revision, directory, manifest = table_accepted(app, monkeypatch, kind=kind)
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    before = files(app)
    for reference in facade.scientific_artifact_handles('session', revision):
        snapshot = facade.prepare_scientific_artifact('session', revision, reference.handle)
        assert b''.join(snapshot.iter_chunks()) == (directory / reference.filename).read_bytes()
        assert snapshot.filename == reference.filename and snapshot.content_type == 'application/gzip'
        snapshot.close()
    assert files(app) == before


@pytest.mark.parametrize('kind', ['qc', 'selection'])
@pytest.mark.parametrize('changed', ['altered', 'missing', 'symlink'])
def test_download_rejects_changed_original_payload(app, monkeypatch, kind, changed):
    from agent.application.matrix_delivery import MatrixDeliveryError
    revision, directory, _ = table_accepted(app, monkeypatch, kind=kind)
    facade = read_only_boundary(app)
    reference = facade.scientific_artifact_handles('session', revision)[0]
    payload = directory / reference.filename
    if changed == 'altered': payload.write_bytes(b'changed')
    elif changed == 'missing': payload.unlink()
    else:
        other = directory / 'other.gz'; other.write_bytes(payload.read_bytes())
        payload.unlink(); payload.symlink_to(other)
    # Metadata eligibility remains exact; transport checks the actual snapshot.
    assert reference in facade.scientific_artifact_handles('session', revision)
    with pytest.raises(MatrixDeliveryError):
        facade.prepare_scientific_artifact('session', revision, reference.handle)
    assert facade._matrix_downloads._active == 0


@pytest.mark.parametrize('kind', ['qc', 'selection'])
@pytest.mark.parametrize('case', ['result', 'manifest', 'closure', 'arguments', 'authority_identity', 'metadata'])
def test_conflicting_accepted_bindings_fail_closed(app, monkeypatch, kind, case):
    changes = {}
    if case == 'result': changes['result_changes'] = {'n_observed_barcodes': 3}
    if case == 'arguments': changes['authority_changes'] = {'arguments_sha256': 'f' * 64}
    if case == 'authority_identity': changes['authority_changes'] = {'manifest_sha256': 'f' * 64}
    if case == 'closure': changes['authority_change'] = lambda r: r.update(
        files=[f for f in r['files'] if not f['path'].endswith('.tsv.gz')])
    if case == 'metadata': changes['authority_change'] = lambda r: r['resources'].update(
        result_metadata={'unexpected': True})
    revision, directory, _ = table_accepted(app, monkeypatch, kind=kind, **changes)
    facade = read_only_boundary(app)
    if case == 'manifest': (directory / 'manifest.json').write_bytes(b'{}')
    assert facade.scientific_artifact_handles('session', revision) == ()
    with pytest.raises(InteractiveBoundaryError): facade._scientific_artifact('session', revision, '0' * 64)
    assert facade.revision('session', revision).scientific_artifacts == ()


@pytest.mark.parametrize('kind,role', [('qc', 'table'), ('qc', 'histogram'), ('selection', 'decisions'), ('selection', 'selected')])
def test_unreviewed_manifest_member_is_not_downloadable(app, monkeypatch, kind, role):
    revision, _, _ = table_accepted(app, monkeypatch, kind=kind,
        manifest_change=lambda m: m[role].update(path='private-receipt.json'))
    assert read_only_boundary(app).scientific_artifact_handles('session', revision) == ()


def test_historical_tables_reopen_navigation_and_new_results_are_exact(app, monkeypatch):
    first, _, _ = table_accepted(app, monkeypatch)
    second, _, _ = table_accepted(app, monkeypatch, kind='selection', name='later')
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    old = facade.scientific_artifacts_for_turn('session', 'initial')
    recent = facade.scientific_artifacts_for_turn('session', 'later')
    assert {r.handle for r in old}.isdisjoint(r.handle for r in recent)
    facade.activate_revision('session', 'navigation', first, expected_generation=2)
    restarted = read_only_boundary(app)
    assert restarted.scientific_artifacts_for_turn('session', 'initial') == old
    assert restarted.scientific_artifacts_for_turn('session', 'later') == recent
    assert restarted.scientific_artifacts_for_turn('session', 'navigation') == ()
    with pytest.raises(InteractiveBoundaryError): restarted._scientific_artifact('session', second, old[0].handle)


@pytest.mark.parametrize('kind', ['qc', 'selection'])
def test_current_request_summary_selection_preserves_accepted_step_tables(app, monkeypatch, kind):
    revision, _, _ = table_accepted(app, monkeypatch, kind=kind)
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    expected = facade.scientific_artifacts_for_turn('session', 'initial')
    state = app.sessions.load('session')
    original = next(r for r in state.revisions if r.revision_id == revision)
    updated = replace(original, outputs=(replace(original.outputs[0], output_key='n_observed_barcodes'),))
    turns = tuple(replace(t, selections=(replace(t.selections[0], output_key='n_observed_barcodes'),))
        if t.turn_id == 'initial' else t for t in state.turns)
    app.sessions._store._write(replace(state, turns=turns,
        revisions=tuple(updated if r.revision_id == revision else r for r in state.revisions)))
    before = files(app)
    assert len(expected) == 2
    assert facade.scientific_artifact_handles('session', revision) == expected
    assert facade.scientific_artifacts_for_turn('session', 'initial') == expected
    assert files(app) == before


@pytest.mark.parametrize('kind', ['qc', 'selection'])
def test_changed_manifest_removes_old_handle_and_preserves_scientific_state(app, monkeypatch, kind):
    revision, directory, _ = table_accepted(app, monkeypatch, kind=kind)
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    reference = facade.scientific_artifact_handles('session', revision)[0]
    state = app.sessions.load('session')
    path = directory / 'manifest.json'
    path.write_bytes(b'{}')
    assert facade.scientific_artifact_handles('session', revision) == ()
    assert facade.scientific_artifacts_for_turn('session', 'initial') == ()
    with pytest.raises(InteractiveBoundaryError):
        facade.prepare_scientific_artifact('session', revision, reference.handle)
    assert app.sessions.load('session') == state


def test_changed_whole_accepted_run_during_table_inventory_is_rejected(app, monkeypatch):
    revision, _, _ = table_accepted(app, monkeypatch)
    facade = read_only_boundary(app)
    original = facade._application.run_store.load
    calls = 0
    def changed(run_id):
        nonlocal calls
        calls += 1
        state = original(run_id)
        if calls < 5:
            return state
        step = state.steps[0]
        result = dict(step.result) | {'n_observed_barcodes': 3}
        return replace(state, steps=(replace(step, result=result),))
    monkeypatch.setattr(facade._application.run_store, 'load', changed)
    with pytest.raises(InteractiveBoundaryError):
        facade.scientific_artifact_handles('session', revision)


@pytest.mark.parametrize('kind', ['qc', 'selection'])
def test_retained_tables_use_original_run_and_are_not_new_chat_files(app, monkeypatch, kind):
    first, directory, _ = table_accepted(app, monkeypatch, kind=kind)
    later, _, _ = accepted(app, 'inspect_scATAC', name='later')
    state = app.sessions.load('session')
    source = next(r for r in state.revisions if r.revision_id == first).outputs[0]
    app.sessions._store._write(replace(state,
        revisions=tuple(replace(r, outputs=(replace(r.outputs[0], name='inspection'), source))
            if r.revision_id == later else r for r in state.revisions),
        turns=tuple(replace(t, retained_outputs=(source,),
            selections=(replace(t.selections[0], name='inspection'),)) if t.turn_id == 'later' else t
            for t in state.turns)))
    capture_execute_display(app, 'later')
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    handles = facade.scientific_artifact_handles('session', later)
    assert len(handles) == 2
    for reference in handles:
        locator = facade._scientific_artifact('session', later, reference.handle)
        assert locator.path == directory / reference.filename and locator.managed_root == directory.parent
    assert facade.scientific_artifacts_for_turn('session', 'later') == ()


@pytest.mark.parametrize('kind', ['qc', 'selection'])
def test_retained_historical_summary_does_not_offer_unbound_table_files(app, monkeypatch, kind):
    first, _, _ = table_accepted(app, monkeypatch, kind=kind)
    original_state = app.sessions.load('session')
    parent = next(r for r in original_state.revisions if r.revision_id == first)
    retained = replace(parent.outputs[0], output_key='n_observed_barcodes')
    app.sessions._store._write(replace(original_state,
        revisions=tuple(replace(r, outputs=(retained,)) if r.revision_id == first else r
            for r in original_state.revisions),
        turns=tuple(replace(t, selections=(replace(t.selections[0], output_key='n_observed_barcodes'),))
            if t.turn_id == 'initial' else t for t in original_state.turns)))
    later, _, _ = accepted(app, 'inspect_scATAC', name='later')
    state = app.sessions.load('session')
    app.sessions._store._write(replace(state,
        revisions=tuple(replace(r, outputs=(replace(r.outputs[0], name='inspection'), retained))
            if r.revision_id == later else r for r in state.revisions),
        turns=tuple(replace(t, retained_outputs=(retained,),
            selections=(replace(t.selections[0], name='inspection'),)) if t.turn_id == 'later' else t
            for t in state.turns)))
    capture_execute_display(app, 'later')
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    before = files(app)
    assert facade.scientific_artifact_handles('session', later) == ()
    assert facade.scientific_artifacts_for_turn('session', 'later') == ()
    assert files(app) == before


def add_same_step_cooutput(application, revision_id):
    """Two explicit semantic members identify one accepted owner publication."""
    state = application.sessions.load('session')
    revision = next(r for r in state.revisions if r.revision_id == revision_id)
    output = replace(revision.outputs[0], name='digest', output_key='manifest_sha256')
    turn = state.turn(revision.turn_id)
    selection_ref = replace(turn.selections[0], name='digest', output_key='manifest_sha256')
    application.sessions._store._write(replace(state,
        revisions=tuple(replace(r, outputs=r.outputs + (output,)) if r.revision_id == revision_id else r
            for r in state.revisions),
        turns=tuple(replace(t, selections=t.selections + (selection_ref,)) if t.turn_id == turn.turn_id else t
            for t in state.turns)))


@pytest.mark.parametrize('kind', ['qc', 'selection'])
def test_cooutputs_of_one_accepted_step_have_two_stable_role_handles(app, monkeypatch, kind):
    revision_id, _, _ = table_accepted(app, monkeypatch, kind=kind)
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    expected = facade.scientific_artifacts_for_turn('session', 'initial')
    add_same_step_cooutput(app, revision_id)
    assert len(expected) == 2
    assert facade.scientific_artifacts_for_turn('session', 'initial') == expected
    assert facade.scientific_artifact_handles('session', revision_id) == expected
    state = app.sessions.load('session')
    app.sessions._store._write(replace(state,
        revisions=tuple(replace(r, outputs=tuple(reversed(r.outputs))) for r in state.revisions),
        turns=tuple(replace(t, selections=tuple(reversed(t.selections))) for t in state.turns)))
    restarted = read_only_boundary(app)
    assert restarted.scientific_artifacts_for_turn('session', 'initial') == expected
    for reference in expected:
        assert restarted._scientific_artifact('session', revision_id, reference.handle).reference == reference


@pytest.mark.parametrize('kind', ['qc', 'selection'])
def test_independent_accepted_steps_keep_distinct_handles_for_identical_table_bytes(app, monkeypatch, kind):
    revision, _, _ = table_accepted(app, monkeypatch, kind=kind, copies=2)
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    references = facade.scientific_artifacts_for_turn('session', 'initial')
    assert len(references) == 4 and len({r.handle for r in references}) == 4
    assert references[0].filename == references[2].filename and references[0].sha256 == references[2].sha256
    assert references[1].filename == references[3].filename and references[1].sha256 == references[3].sha256
    paths = {facade._scientific_artifact('session', revision, r.handle).path for r in references}
    assert len(paths) == 4


def test_conflicting_records_under_one_opaque_role_handle_fail_closed(app, monkeypatch):
    revision, directory, _ = table_accepted(app, monkeypatch)
    add_same_step_cooutput(app, revision)
    facade = read_only_boundary(app)
    original = facade._accepted_tables
    calls = 0
    def conflict(*args, **kwargs):
        nonlocal calls
        calls += 1
        values = original(*args, **kwargs)
        if calls > 1:
            return (replace(values[0], path=directory / 'other-table.gz'), values[1])
        return values
    monkeypatch.setattr(facade, '_accepted_tables', conflict)
    with pytest.raises(InteractiveBoundaryError):
        facade.scientific_artifact_handles('session', revision)
