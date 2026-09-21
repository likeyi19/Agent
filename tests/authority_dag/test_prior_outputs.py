"""M15.3: exact historical publication binding with real tiny owner artifacts."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import os

import pytest

from agent.application import ResearchAgentApplication, OutputSelection
from agent.orchestration import AgentRequest, AgentPlan, PlanStep, LLMPlanner, build_default_tool_registry
from agent.schemas.prior_output import PriorOutputRef
from agent.schemas.run_state import PersistedRunState, fingerprint_plan
from agent.schemas.verification_authority import AuthorityError
from agent.orchestration.prior_outputs import (bind_output, validate_binding, prior_output_store,
                                              validate_step_references, validate_active_outputs)
from agent.orchestration.active_context import planning_context
from test_application import application_chain, owner_calls
from test_propagation import chain


ALL_OUTPUTS = tuple(OutputSelection(name, step, 'manifest_path') for name, step in (
    ('fragments', 'fragments'), ('qc', 'qc'), ('selection', 'selection'), ('matrix', 'matrix')))


@pytest.fixture
def source(application_chain):
    kind, args, app = application_chain
    app.sessions.create('analysis')
    state = app.sessions.run('analysis', 'initial', AgentRequest('authority-request', 'Tiny DAG.', {}),
                             ALL_OUTPUTS, expected_generation=0)
    assert state.turn('initial').status == 'activated'
    return kind, args, app


class ContextModel:
    model_id = 'scripted-context'
    def __init__(self, matrix=False, handle_override=None):
        self.calls = []
        self.matrix = matrix
        self.handle_override = handle_override

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append((value, response_schema, prompt))
        if 'selection_schema_version' in value:
            return json.dumps({'selection_schema_version': 1, 'decision': {
                'kind': 'select', 'capability_ids': ['raw_preprocessing']}})
        items = value['active_outputs']
        qc = next(i['handle'] for i in items if i['producer_tool'] == 'compute_scATAC_qc')
        if self.handle_override: qc = self.handle_override
        steps = [{'step_id': 'new_selection', 'tool': 'select_scATAC_cells',
                  'sources': [{'target': 'barcode_qc', 'source': {'kind': 'context', 'handle': qc}}],
                  'control_dependencies': []}]
        if self.matrix:
            fragments = next(i['handle'] for i in items if i['source_port'] == 'fragments')
            steps.append({'step_id': 'new_matrix', 'tool': 'build_scATAC_cell_by_ccre',
                'sources': [{'target': 'fragments', 'source': {'kind': 'context', 'handle': fragments}},
                            {'target': 'selected_cells', 'source': {'kind': 'step', 'step': 'new_selection'}}],
                'control_dependencies': []})
        return json.dumps({'schema_version': 4, 'decision': {'kind': 'plan', 'steps': steps}})


def continuation(source, matrix=False, *, retain=('fragments', 'qc'), name='next'):
    kind, args, old = source
    model = ContextModel(matrix)
    app = ResearchAgentApplication(old.workspace_root, planner=LLMPlanner(model))
    inputs = {'min_qc_fragment_records': 1, 'min_tss_enrichment': '0'}
    if matrix:
        inputs.update(reference_manifest_path=args['reference_bundle_path'],
                      reference_manifest_sha256=args['reference_bundle_sha256'])
    req = AgentRequest(name, 'Using the current QC result, select with the explicit supplied thresholds.' +
                       (' Also build the matrix.' if matrix else ''), inputs)
    outputs = (OutputSelection('selection', 'new_selection', 'manifest_path'),)
    if matrix: outputs += (OutputSelection('matrix', 'new_matrix', 'manifest_path'),)
    return app, model, req, outputs, retain


def test_selection_only_reuses_qc_and_publishes_explicit_state(source):
    app, model, req, outputs, retain = continuation(source)
    with all_work() as counts:
        state = app.sessions.run('analysis', 'next', req, outputs, expected_generation=1,
                                 use_active_context=True, retain=retain)
    run = app.run_store.load(req.request_id + ':run')
    assert state.turn('next').status == 'activated', run.to_dict()
    assert {k:v for k,v in counts.items() if k.startswith(('production.', 'owner.'))} == {
        'production.selection': 1, 'owner.selection': 1}
    assert counts['integrity'] > 0 and counts['presentation'] == 1
    revision = state.revisions[-1]
    assert {o.name for o in revision.outputs} == {'fragments', 'qc', 'selection'}
    assert len({o.run_id for o in revision.outputs}) == 2
    assert len(model.calls) == 2
    assert len(model.calls[0][2].encode()) < 5000
    for payload, schema, prompt in model.calls:
        assert str(app.workspace_root) not in prompt
        assert 'authority-request:run' not in prompt
        assert 'authority_sha256' not in prompt
    refs = [a for a in run.plan.steps[0].arguments.values() if isinstance(a, PriorOutputRef)]
    assert len(refs) == 2 and refs[0].binding == refs[1].binding
    assert PersistedRunState.from_dict(run.to_dict()) == run
    # Otherwise identical executable plans differ for two genuinely accepted
    # publications, not just fabricated source identifiers.
    old_selection = bind_output(app.run_store, app.registry, 'authority-request:run', 'selection', 'selected_cells')
    new_selection = bind_output(app.run_store, app.registry, run.run_id, 'new_selection', 'selected_cells')
    def consumer(binding):
        arguments = dict(app.run_store.load('authority-request:run').steps[-1].resolved_arguments)
        arguments.update({
            'selected_cells_manifest_path': PriorOutputRef(binding, 'manifest_path'),
            'selected_cells_manifest_sha256': PriorOutputRef(binding, 'manifest_sha256')})
        app.registry.validate_arguments('build_scATAC_cell_by_ccre', arguments)
        return AgentPlan('p', 'r', 'explicit', (PlanStep('m', 'build_scATAC_cell_by_ccre', arguments),))
    assert fingerprint_plan(consumer(old_selection)) != fingerprint_plan(consumer(new_selection))
    # Presentation retains exact source references without promoting session metadata.
    evidence = json.loads((app._workspace.run_paths(run.run_id).evidence / 'analysis_evidence.json').read_bytes())
    assert len(evidence['provenance']['prior_outputs']) == 2
    before = run.to_dict()
    with owner_calls() as counts:
        result = ResearchAgentApplication(app.workspace_root).resume(run.run_id)
    assert result.status.value == 'SUCCEEDED' and not counts
    assert app.run_store.load(run.run_id).to_dict() == before


def test_explicit_matrix_rebuild_and_no_stale_matrix(source):
    app, model, req, outputs, retain = continuation(source, matrix=True)
    with owner_calls() as counts:
        state = app.sessions.run('analysis', 'next', req, outputs, expected_generation=1,
                                 use_active_context=True, retain=retain)
    run = app.run_store.load(req.request_id + ':run')
    assert state.turn('next').status == 'activated', run.to_dict()
    assert counts == {'selection': 1, 'matrix': 1}
    revision = state.revisions[-1]
    assert {o.name for o in revision.outputs} == {'fragments', 'qc', 'selection', 'matrix'}
    validate_active_outputs(revision.outputs, app.run_store, app.registry)
    old = state.revisions[0]
    old_matrix = next(o for o in old.outputs if o.name == 'matrix')
    with pytest.raises(AuthorityError):
        validate_active_outputs(tuple(o for o in revision.outputs if o.name != 'matrix') + (old_matrix,),
                                app.run_store, app.registry)


def test_unknown_context_fails_before_science(source):
    _, _, old = source
    app = ResearchAgentApplication(old.workspace_root, planner=LLMPlanner(ContextModel(handle_override='ctx.999')))
    with owner_calls() as counts:
        state = app.sessions.run('analysis', 'bad', AgentRequest('bad', 'Select from current QC.',
            {'min_qc_fragment_records': 1, 'min_tss_enrichment': '0'}),
            (OutputSelection('selection', 'new_selection', 'manifest_path'),),
            expected_generation=1, use_active_context=True)
    assert state.turn('bad').status == 'failed'
    assert state.generation == 1 and not counts


def test_fingerprint_and_member_atomicity(source):
    _, _, app = source
    registry = app.registry
    qc = bind_output(app.run_store, registry, 'authority-request:run', 'qc', 'barcode_qc')
    args = dict(barcode_qc_manifest_path=PriorOutputRef(qc, 'manifest_path'),
                barcode_qc_manifest_sha256=PriorOutputRef(qc, 'manifest_sha256'),
                output_dir='/tmp/m153-unused', min_qc_fragment_records=1, min_tss_enrichment='0')
    plan = AgentPlan('plan', 'request', 'explicit', (PlanStep('s', 'select_scATAC_cells', args),))
    other = replace(qc, run_id='different:run')
    changed = replace(plan, steps=(replace(plan.steps[0], arguments=args | {
        'barcode_qc_manifest_path': PriorOutputRef(other, 'manifest_path')}),))
    assert fingerprint_plan(plan) != fingerprint_plan(changed)
    with prior_output_store(app.run_store):
        validate_step_references(plan.steps[0], registry)
        with pytest.raises(AuthorityError): validate_step_references(changed.steps[0], registry)
        wrong = replace(plan.steps[0], arguments=args | {'barcode_qc_manifest_path': PriorOutputRef(qc, 'manifest_sha256')})
        with pytest.raises(AuthorityError): validate_step_references(wrong, registry)


@pytest.mark.parametrize('field,value', [('run_id','missing:run'), ('step_id','missing'),
    ('source_port','missing'), ('accepted_step_sha256','0'*64),
    ('manifest_sha256','0'*64), ('authority_sha256','0'*64)])
def test_changed_source_identity_fails(source, field, value):
    _, _, app = source
    binding = bind_output(app.run_store, app.registry, 'authority-request:run', 'qc', 'barcode_qc')
    with pytest.raises((AuthorityError, RuntimeError, ValueError)):
        validate_binding(replace(binding, **{field: value}), store=app.run_store, integrity=True)


@pytest.mark.parametrize('delete', [False, True])
def test_corrupted_or_deleted_publication_fails(source, delete):
    _, _, app = source
    binding = bind_output(app.run_store, app.registry, 'authority-request:run', 'qc', 'barcode_qc')
    run = app.run_store.load(binding.run_id)
    step = next(s for s in run.steps if s.step_id == 'qc')
    path = Path(step.result['manifest_path'])
    if delete: path.unlink()
    else: path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises((ValueError, OSError)):
        validate_binding(binding, store=app.run_store, integrity=True)


from collections import Counter
from contextlib import contextmanager


@contextmanager
def all_work():
    """Count actual bodies, not tool wrappers or cached verifier entry points."""
    from agent.tools.data import (
        _barcode_qc_production as q, _cell_selection_production as s,
        _cell_by_ccre_production as m, bam_fragments as b, external_fragments as e,
        barcode_qc_verifier as qv, cell_selection_verifier as sv,
        cell_by_ccre_verifier as mv, bam_fragments_verifier as bv,
        external_fragments_verifier as ev, scatac_fragments_v2_verifier as fv,
        fastq_verification_authority as integrity,
    )
    functions = {
        'production.qc': q.produce, 'production.selection': s.produce,
        'production.matrix': m.construct_counts, 'production.bam': b.prepare_in_stage,
        'production.external': e.prepare_in_stage,
        'owner.qc': qv.verify_barcode_qc.__wrapped__,
        'owner.selection': sv.verify_cell_selection.__wrapped__,
        'owner.matrix': mv.verify_cell_by_ccre.__wrapped__,
        'owner.bam': bv.verify_bam_fragments.__wrapped__,
        'owner.external': ev.verify_external_fragments.__wrapped__,
        'owner.generic': fv.verify_fragments_v2.__wrapped__,
        'integrity': integrity.file_closure,
        'presentation': ResearchAgentApplication._complete,
    }
    codes = {f.__code__: name for name, f in functions.items()}
    counts = Counter()
    previous = sys.getprofile()
    def observe(frame, event, arg):
        if event == 'call' and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1
    sys.setprofile(observe)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


def test_separate_accounting_and_zero_work_navigation(source):
    app, model, req, outputs, retain = continuation(source, matrix=True)
    with all_work() as calls:
        state = app.sessions.run('analysis', 'next', req, outputs, expected_generation=1,
                                 use_active_context=True, retain=retain)
    assert state.turn('next').status == 'activated'
    assert {k:v for k,v in calls.items() if k.startswith(('owner.', 'production.'))} == {
        'production.selection': 1, 'owner.selection': 1,
        'production.matrix': 1, 'owner.matrix': 1}
    assert calls['integrity'] > 0 and calls['presentation'] == 1
    planner_calls = len(model.calls)
    with all_work() as navigation:
        app.sessions.switch('analysis', 'back', state.revisions[0].revision_id, expected_generation=2)
        app.sessions.switch('analysis', 'forward', state.revisions[1].revision_id, expected_generation=3)
    assert not navigation and len(model.calls) == planner_calls


def test_frozen_context_retry_and_stale_activation(source):
    app, model, req, outputs, retain = continuation(source)
    original = model.complete
    base = app.sessions.load('analysis').active_revision_id
    alternate_app, _, alternate_req, alternate_outputs, alternate_retain = continuation(source, name='alternate')
    alternate = alternate_app.sessions.run('analysis', 'alternate', alternate_req, alternate_outputs,
        expected_generation=1, use_active_context=True, retain=alternate_retain).active_revision_id
    app.sessions.switch('analysis', 'capture-original', base, expected_generation=2)
    attempted = []
    def race(*, prompt, response_schema):
        payload = json.loads(prompt)
        if 'active_outputs' in payload:
            attempted.append(payload['active_outputs'])
            if len(attempted) == 1:
                app.sessions.switch('analysis', 'racing-navigation', alternate, expected_generation=3)
                model.handle_override = 'ctx.999'
            else:
                model.handle_override = None
        return original(prompt=prompt, response_schema=response_schema)
    model.complete = race
    with owner_calls() as counts:
        state = app.sessions.run('analysis', 'next', req, outputs, expected_generation=3,
                                 use_active_context=True, retain=retain)
    assert state.turn('next').status == 'stale'
    assert state.generation == 4 and state.active_revision_id == alternate
    assert state.turn('next').base_revision_id == base and len(attempted[0]) == 4
    assert len(attempted) >= 2 and all(a == attempted[0] for a in attempted)
    assert counts == {'selection': 1}
    run = app.run_store.load('next:run')
    assert run.plan.steps[0].arguments['barcode_qc_manifest_path'].binding.run_id == 'authority-request:run'
    # A session with no such base cannot enumerate or retain these outputs.
    from agent.application.session_state import SessionError
    app.sessions.create('other')
    with pytest.raises(SessionError):
        app.sessions.start_turn('other', 'foreign', replace(req, request_id='foreign'), outputs,
                                expected_generation=0, retain=('qc',))
    app.sessions.start_turn('other', 'empty', replace(req, request_id='empty'), outputs, expected_generation=0)
    with pytest.raises(SessionError): app.sessions.active_context('other', 'empty')


def test_stale_descendant_prevents_activation(source):
    app, model, req, outputs, retain = continuation(source, retain=('fragments', 'qc', 'matrix'))
    with pytest.raises(AuthorityError):
        app.sessions.run('analysis', 'next', req, outputs, expected_generation=1,
                          use_active_context=True, retain=retain)
    state = app.sessions.recover('analysis', 'next')
    assert state.generation == 1 and len(state.revisions) == 1
    assert state.turn('next').status == 'run_succeeded'
    assert app.run_store.load('next:run').lifecycle_status.value == 'SUCCEEDED'


def test_reject_unsuccessful_unaccepted_and_missing_authority(source):
    from types import SimpleNamespace
    from agent.schemas import StepStatus, RunLifecycleStatus
    _, _, app = source
    state = app.run_store.load('authority-request:run')
    cases = [SimpleNamespace(**(vars(state) | {'lifecycle_status': RunLifecycleStatus.FAILED}))]
    for field in ('status', 'verification'):
        step = state.steps[1]
        changed = SimpleNamespace(**(vars(step) | (
            {'status': StepStatus.FAILED} if field == 'status' else {
                'verification': replace(step.verification, artifact_authority=None)})))
        cases.append(SimpleNamespace(**(vars(state) | {'steps': (state.steps[0], changed, *state.steps[2:])})))
    for invalid in cases:
        store = SimpleNamespace(load=lambda _, value=invalid: value)
        with pytest.raises(AuthorityError): bind_output(store, app.registry, state.run_id, 'qc', 'barcode_qc')


@pytest.mark.parametrize('boundary', ['accepted_selection', 'published_selection'])
def test_fresh_process_pending_matrix_reuses_all_accepted_sources(source, monkeypatch, tmp_path, boundary):
    app, model, req, outputs, retain = continuation(source, matrix=True)
    class ProcessExit(BaseException): pass
    original = app.run_store.update
    def interrupt(state, *, expected_revision):
        saved = original(state, expected_revision=expected_revision)
        if saved.run_id == 'next:run' and any(s.step_id == 'new_selection' and s.status.value == 'SUCCEEDED'
                                             for s in saved.steps):
            raise ProcessExit()
        return saved
    if boundary == 'accepted_selection':
        monkeypatch.setattr(app.run_store, 'update', interrupt)
    else:
        from agent.orchestration import executor
        verify = executor.verify_step
        def interrupt_verification(step, *args, **kwargs):
            if step.step_id == 'new_selection':
                raise ProcessExit()
            return verify(step, *args, **kwargs)
        monkeypatch.setattr(executor, 'verify_step', interrupt_verification)
    with pytest.raises(ProcessExit):
        app.sessions.run('analysis', 'next', req, outputs, expected_generation=1,
                          use_active_context=True, retain=retain)
    before = app.run_store.load('next:run')
    assert before.lifecycle_status.value == 'RUNNING'
    source_bytes = app.run_store.state_path('authority-request:run').read_bytes()
    result_path = tmp_path / 'resume-accounting.json'
    script = '''
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests/authority_dag'))
from test_prior_outputs import all_work
from agent.application import ResearchAgentApplication
app = ResearchAgentApplication(sys.argv[1])
with all_work() as calls:
    result = app.resume('next:run')
assert result.status.value == 'SUCCEEDED', result
Path(sys.argv[2]).write_text(json.dumps(dict(calls)))
'''
    child = subprocess.run([sys.executable, '-c', script, str(app.workspace_root), str(result_path)],
                            capture_output=True, text=True, env=os.environ.copy(), timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    calls = json.loads(result_path.read_text())
    expected = {'production.matrix': 1, 'owner.matrix': 1}
    if boundary == 'published_selection':
        expected['owner.selection'] = 1  # Published output was not yet accepted.
    assert {k:v for k,v in calls.items() if k.startswith(('owner.', 'production.'))} == expected
    assert calls['integrity'] > 0 and calls['presentation'] == 1
    after = app.run_store.load('next:run')
    assert after.plan == before.plan and after.plan_fingerprint == before.plan_fingerprint
    assert app.run_store.state_path('authority-request:run').read_bytes() == source_bytes
    state = app.sessions.complete_presentation('analysis', 'next')
    assert state.turn('next').status == 'activated'


def test_publication_changed_after_context_capture_fails_before_execution(source):
    app, model, req, outputs, retain = continuation(source)
    original = model.complete
    def corrupt(*, prompt, response_schema):
        response = original(prompt=prompt, response_schema=response_schema)
        if 'active_outputs' in json.loads(prompt):
            source_run = app.run_store.load('authority-request:run')
            path = Path(source_run.steps[1].result['manifest_path'])
            path.write_bytes(path.read_bytes() + b' ')
        return response
    model.complete = corrupt
    with all_work() as calls:
        state = app.sessions.run('analysis', 'next', req, outputs, expected_generation=1,
                                 use_active_context=True, retain=retain)
    assert state.turn('next').status == 'failed' and state.generation == 1
    assert not {k:v for k,v in calls.items() if k.startswith(('production.', 'owner.'))}
