"""Exact registered FASTQ selections use ordinary interpreted intake Sessions."""
from dataclasses import replace
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agent.application import InteractiveAgentApplication
from agent.application.local_resources import LocalResourceAdmission, RegisteredInputCollection, ResourceAdmissionError
from agent.application.sessions import AnalysisSessions
from agent.orchestration import PlanningModelProfile, ToolRegistry, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.tools.data import _raw_fastq, raw_scatac_manifest


PROFILE = PlanningModelProfile('fastq-scripted', 'scripted', 'scripted/registered-fastq')
INSPECT = 'inspect_raw_scATAC'
UTTERANCE = 'Inspect the exact supplied FASTQ collection using the declared scientific inputs.'
DECLARATIONS = dict(species='human', raw_assay='TENX_ATAC')


class Model:
    """Script semantic decisions while executing the actual registered intake tool."""
    model_id = PROFILE.model_id

    def __init__(self, callback=None):
        self.callback = callback
        self.calls = []

    def complete(self, *, prompt, response_schema):
        data = json.loads(prompt)
        self.calls.append(data)
        if 'turn_schema_version' in data:
            return json.dumps(dict(turn_schema_version=1,
                decision=dict(kind='execute_plan', target=INSPECT)))
        if 'selection_schema_version' in response_schema.get('properties', {}):
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=['raw_preprocessing'])))
        if 'output_selection_schema_version' in data:
            if self.callback is not None:
                self.callback()
            return json.dumps(dict(outputs=[dict(name='intake', step_id='inspect', output_key='manifest_path')]))
        step = dict(step_id='inspect', tool=INSPECT, control_dependencies=[], sources=[
            dict(target='raw_input', source=dict(kind='input', input='raw_input_paths'))])
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[step])))


@pytest.fixture
def fastqs(tmp_path):
    def make(*, name='reads', roles=('R1', 'R2', 'R3'), lane='001', chunk='001',
             count=2, suffix='.fastq', payload=None):
        root = tmp_path / name
        root.mkdir(exist_ok=True)
        content = payload if payload is not None else b''.join(
            f'@read{i:04d}\nACGT\n+\nIIII\n'.encode() for i in range(count))
        paths = tuple(root / f'Sample_S1_L{lane}_{role}_{chunk}{suffix}' for role in roles)
        for path in paths:
            path.write_bytes(gzip.compress(content, mtime=0) if suffix.endswith('.gz') else content)
        return paths
    return make


def application(tmp_path, *, registry=None, callback=None):
    models = []
    def factory(profile):
        assert profile == PROFILE
        model = Model(callback)
        models.append(model)
        return model
    app = InteractiveAgentApplication(tmp_path / 'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, approved_source_roots=(tmp_path,), registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    return app, models


def instrument(monkeypatch):
    registry = build_default_tool_registry()
    original = registry.get(INSPECT).function
    calls = dict(intake=0)
    def inspect(**arguments):
        calls['intake'] += 1
        return original(**arguments)
    return ToolRegistry(tuple(replace(registry.get(name), function=inspect) if name == INSPECT
                              else registry.get(name) for name in registry.names())), calls


def register(app, paths, *, key='fastq'):
    return tuple(app.resources.register(f'{key}-{i}', path, input_type='fastq',
        label=f'Supplied FASTQ {i + 1}', attribution='Explicit operator-approved synthetic FASTQ.')
        for i, path in enumerate(paths))


def bind(app, records, declarations=None):
    members = tuple(dict(resource_id=record.resource_id, record_sha256=record.record_sha256)
                    for record in records)
    return app.resources.resolve_collection(members, tool_name=INSPECT,
        scientific_inputs=DECLARATIONS if declarations is None else declarations)


def submit(app, binding, *, turn='analysis', generation=0):
    return app.submit_turn('session', turn, UTTERANCE, expected_generation=generation,
                           registered_input=binding)


def step_for(app, view):
    step, = app._application.run_store.load(view.run_id).steps
    return step


def manifest_for(app, view):
    result = step_for(app, view).result
    _, manifest, _ = raw_scatac_manifest.load_raw_intake_manifest(result['manifest_path'],
        expected_sha256=result['manifest_sha256'])
    return manifest


def assert_attribution(app, view, binding):
    state = app._application.sessions.load('session')
    interaction = next(item for item in state.interactions if item.turn_id == view.turn_id)
    assert _serialize(interaction.submission['registered_input']) == binding.attribution()
    assert _serialize(interaction.submission['execution_inputs']) == _serialize(binding.execution_inputs)
    assert str(app._application.workspace_root.parent) not in json.dumps(view.to_dict())


def test_registered_fastq_exact_interpreted_inspection_and_permutation_retry(fastqs, tmp_path, monkeypatch):
    paths = fastqs(roles=('R1', 'R2', 'R3', 'I1'))
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, registry=registry)
    records = register(app, paths)
    binding = bind(app, records)
    reordered = bind(app, tuple(reversed(records)))
    assert isinstance(binding, RegisteredInputCollection) and binding == reordered
    assert not models and calls == {'intake': 0}
    assert not list(app._application.sessions._store.root.glob('*.json'))
    # An adjacent valid read set is outside this exact admitted collection.
    neighboring = fastqs(lane='002')
    app.create_session('session')
    view = submit(app, binding)
    assert view.status == 'succeeded' and view.revision_id, view
    step = step_for(app, view)
    assert step.result['readiness'] == 'READY' and step.result['n_files'] == 4
    assert step.result['n_groups'] == 1 and step.verification.passed
    assert step.verification.artifact_authority is None
    manifest = manifest_for(app, view)
    assert {file.path for file in manifest.files} == {str(path) for path in paths}
    assert not {str(path) for path in neighboring}.intersection(file.path for file in manifest.files)
    assert all(file.selection is raw_scatac_manifest.SelectionBasis.EXPLICIT
               and file.selection_root is None for file in manifest.files)
    assert_attribution(app, view, binding)
    assert app.evidence('session', view.revision_id, 'intake').status == 'available'
    assert submit(app, reordered) == view
    assert app.recover_turn('session', 'analysis', complete_presentation=True) == view
    assert calls == {'intake': 1} and len(models) == 1 and len(models[0].calls) == 4


def test_registered_fastq_first_acceptance_requires_actual_exact_inspection_membership(
        fastqs, tmp_path, monkeypatch):
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, registry=registry)
    binding = bind(app, register(app, fastqs()))
    app.create_session('session')
    view = submit(app, binding)
    assert view.status == 'succeeded', view
    state = app._application.sessions.load('session')
    submission = state.interactions[0].submission
    step = step_for(app, view)
    paths = list(binding.execution_inputs['raw_input_paths'])
    for actual in (paths, paths[::-1]):
        reordered = replace(step, resolved_arguments=dict(step.resolved_arguments, raw_input_paths=actual))
        assert app.resources.validate_result(submission, (reordered,)) is None
    unrelated = str(fastqs(name='unrelated')[0])
    for actual in (paths[:-1], [*paths, unrelated], [*paths[:-1], paths[0]],
                   [unrelated, *paths[1:]], paths[0]):
        forged = replace(step, resolved_arguments=dict(step.resolved_arguments, raw_input_paths=actual))
        with pytest.raises(ResourceAdmissionError) as error:
            app.resources.validate_result(submission, (forged,))
        assert error.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'
    assert app._application.sessions.load('session') == state
    assert calls == {'intake': 1} and len(models) == 1


@pytest.mark.parametrize('suffix', ['.fastq', '.fq', '.fastq.gz', '.fq.gz'])
def test_registered_fastq_preserves_owner_lane_chunk_and_optional_index_groups(
        fastqs, tmp_path, monkeypatch, suffix):
    paths = (fastqs(roles=('R1', 'R2', 'R3', 'I1'), suffix=suffix)
             + fastqs(lane='002', suffix=suffix)
             + fastqs(lane='002', chunk='002', roles=('R1', 'R2', 'I2'), suffix=suffix))
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, registry=registry)
    binding = bind(app, register(app, paths))
    app.create_session('session')
    view = submit(app, binding)
    assert view.status == 'succeeded', view
    step = step_for(app, view)
    assert (step.result['readiness'], step.result['n_files'], step.result['n_groups']) == ('READY', 10, 3)
    manifest = manifest_for(app, view)
    assert {(group.lane, group.chunk) for group in manifest.groups} == {('001', '001'), ('002', '001'), ('002', '002')}
    assert {file.path for file in manifest.files} == {str(path) for path in paths}
    meanings = [json.loads(evidence.value) for evidence in manifest.evidence
                if evidence.value.startswith('{') and json.loads(evidence.value).get('fact') == 'read-meaning']
    assert any(fact['role'] == 'I1' for fact in meanings)
    assert_attribution(app, view, binding)
    assert calls == {'intake': 1} and len(models) == 1


@pytest.mark.parametrize('case,readiness,code', [
    ('missing_role', 'NEEDS_USER_INPUT', 'FASTQ_REQUIRED_ROLE_MISSING'),
    ('missing_assay', 'NEEDS_USER_INPUT', 'FASTQ_ASSAY_REQUIRED'),
    ('invalid', 'INVALID', 'FASTQ_CONTENT_HEADER'),
    ('layout', 'UNSUPPORTED', 'FASTQ_ROLE_SET_UNSUPPORTED'),
    ('species', 'UNSUPPORTED', None),
])
def test_registered_fastq_success_preserves_scientific_nonready_findings(
        fastqs, tmp_path, monkeypatch, case, readiness, code):
    paths = fastqs(roles=('R1', 'R2') if case == 'missing_role' else
        ('R1', 'R2', 'R3', 'I2') if case == 'layout' else ('R1', 'R2', 'R3'),
        payload=b'broken\nACGT\n+\nIIII\n' if case == 'invalid' else None)
    declarations = dict(DECLARATIONS)
    if case == 'missing_role':
        declarations['fastq_layout'] = _raw_fastq.FastqLayout.A.value
    elif case == 'missing_assay':
        declarations.pop('raw_assay')
    elif case == 'species':
        declarations['species'] = 'macaca_fascicularis'
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, registry=registry)
    records = register(app, paths)
    binding = bind(app, records, declarations)
    assert not models and calls == {'intake': 0}
    app.create_session('session')
    view = submit(app, binding)
    assert view.status == 'succeeded' and view.revision_id, view
    assert step_for(app, view).result['readiness'] == readiness
    manifest = manifest_for(app, view)
    if code is not None:
        assert code in {issue.code for issue in manifest.issues}
    if case == 'missing_role':
        facts = [json.loads(evidence.value) for evidence in manifest.evidence
                 if evidence.value.startswith('{') and json.loads(evidence.value).get('fact') == 'missing-role']
        assert facts and {fact['role'] for fact in facts} == {'R3'}
        assert all(fact['basis'] == _raw_fastq.MissingRoleBasis.DECLARED.value for fact in facts)
    assert_attribution(app, view, binding)
    assert calls == {'intake': 1} and len(models) == 1


def test_registered_fastq_existing_session_navigation_branch_and_source_attribution(fastqs, tmp_path, monkeypatch):
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, registry=registry)
    one_binding = bind(app, register(app, fastqs(), key='one'))
    two_binding = bind(app, register(app, fastqs(name='other'), key='two'))
    app.create_session('session')
    one = submit(app, one_binding, turn='one')
    two = submit(app, two_binding, turn='two', generation=1)
    assert one.status == two.status == 'succeeded', (one, two)
    original = app._application.sessions.load('session').interactions[0].submission
    app.activate_revision('session', 'back', one.revision_id, expected_generation=2)
    assert calls == {'intake': 2} and len(models) == 2
    branch = submit(app, two_binding, turn='branch', generation=3)
    assert branch.status == 'succeeded', branch
    state = app._application.sessions.load('session')
    assert state.revisions[-1].parent_revision_id == one.revision_id
    assert state.interactions[0].submission == original
    assert [_serialize(item.submission['registered_input']) for item in state.interactions] == [
        one_binding.attribution(), two_binding.attribution(), two_binding.attribution()]
    assert_attribution(app, branch, two_binding)
    assert calls == {'intake': 3} and len(models) == 3


def change_beyond_intake_prefix(source):
    """Keep size/mtime and all bounded intake observations while replacing bytes."""
    before = source.stat()
    lines = source.read_bytes().splitlines(keepends=True)
    lines[280 * 4 + 1] = b'TGCA\n'
    source.write_bytes(b''.join(lines))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source.stat().st_size == before.st_size


@pytest.mark.parametrize('during_provider', [False, True])
def test_registered_fastq_full_hash_blocks_late_source_drift_before_new_consumption(
        fastqs, tmp_path, monkeypatch, during_provider):
    paths = fastqs(count=300)
    source = paths[-1]
    before = _raw_fastq.observe_fastq_sources((raw_scatac_manifest.FileRecord(str(source),
        raw_scatac_manifest.InputKind.FASTQ, source.stat().st_size, source.stat().st_mtime_ns),))
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, registry=registry,
        callback=lambda: change_beyond_intake_prefix(source) if during_provider else None)
    binding = bind(app, register(app, paths))
    app.create_session('session')
    if during_provider:
        view = submit(app, binding)
        assert view.status == 'failed' and view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID', view
        assert len(models) == 1
    else:
        change_beyond_intake_prefix(source)
        with pytest.raises(ResourceAdmissionError) as error:
            submit(app, binding)
        assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID' and not models
    after = _raw_fastq.observe_fastq_sources((raw_scatac_manifest.FileRecord(str(source),
        raw_scatac_manifest.InputKind.FASTQ, source.stat().st_size, source.stat().st_mtime_ns),))
    assert before == after
    state = app._application.sessions.load('session')
    assert state.generation == 0 and not state.revisions and calls == {'intake': 0}


def test_registered_fastq_drift_before_first_acceptance_is_atomic(fastqs, tmp_path, monkeypatch):
    paths = fastqs(count=300)
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, registry=registry)
    binding = bind(app, register(app, paths))
    app.create_session('session')
    original = AnalysisSessions._record_result
    def changed(*args, **kwargs):
        change_beyond_intake_prefix(paths[-1])
        return original(*args, **kwargs)
    monkeypatch.setattr(AnalysisSessions, '_record_result', changed)
    view = submit(app, binding)
    assert view.status == 'failed' and view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID', view
    assert step_for(app, view).verification.passed
    state = app._application.sessions.load('session')
    assert state.generation == 0 and not state.revisions
    assert state.turn('analysis').status == state.interactions[0].status == 'failed'
    assert submit(app, binding) == view
    assert calls == {'intake': 1} and len(models) == 1


class Interrupted(BaseException):
    pass


def unfinished_inspection(fastqs, tmp_path, monkeypatch):
    paths = fastqs(count=300)
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, registry=registry)
    binding = bind(app, register(app, paths))
    app.create_session('session')
    with monkeypatch.context() as stopped:
        def interrupt(*args, **kwargs):
            raise Interrupted()
        stopped.setattr(AnalysisSessions, '_record_result', interrupt)
        with pytest.raises(Interrupted):
            submit(app, binding)
    state = app._application.sessions.load('session')
    assert not state.revisions and state.turn('analysis').status == 'linked'
    assert app._application.run_store.load(state.turn('analysis').run_id).lifecycle_status.value == 'SUCCEEDED'
    return app, models, calls, binding, paths[-1]


@pytest.mark.parametrize('recovery', [False, True])
@pytest.mark.parametrize('phase', ['record_validation', 'completion_files'])
def test_registered_fastq_rechecks_drift_during_later_first_acceptance_validation(
        fastqs, tmp_path, monkeypatch, recovery, phase):
    if recovery:
        app, models, calls, binding, source = unfinished_inspection(fastqs, tmp_path, monkeypatch)
    else:
        paths = fastqs(count=300)
        source = paths[-1]
        registry, calls = instrument(monkeypatch)
        app, models = application(tmp_path, registry=registry)
        binding = bind(app, register(app, paths))
        app.create_session('session')
    mutations = []
    if phase == 'record_validation':
        original_record_result = AnalysisSessions._record_result
        original_records = LocalResourceAdmission._collection_records
        acceptance = dict(active=False, record_reads=0)
        def recording(*args, **kwargs):
            acceptance['active'] = True
            try:
                return original_record_result(*args, **kwargs)
            finally:
                acceptance['active'] = False
        def records_changed_after_later_read(*args, **kwargs):
            records = original_records(*args, **kwargs)
            if acceptance['active']:
                acceptance['record_reads'] += 1
                if acceptance['record_reads'] == 2:
                    change_beyond_intake_prefix(source)
                    mutations.append(phase)
            return records
        monkeypatch.setattr(AnalysisSessions, '_record_result', recording)
        monkeypatch.setattr(LocalResourceAdmission, '_collection_records', records_changed_after_later_read)
    else:
        original_files = AnalysisSessions._completion_files
        def changed_after_completion_files(*args, **kwargs):
            files = original_files(*args, **kwargs)
            change_beyond_intake_prefix(source)
            mutations.append(phase)
            return files
        monkeypatch.setattr(AnalysisSessions, '_completion_files', changed_after_completion_files)
    view = (app.recover_turn('session', 'analysis', complete_presentation=True)
            if recovery else submit(app, binding))
    assert mutations == [phase]
    assert view.status == 'failed' and view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID', view
    assert step_for(app, view).verification.passed
    state = app._application.sessions.load('session')
    assert state.generation == 0 and not state.revisions
    assert state.turn('analysis').status == state.interactions[0].status == 'failed'
    assert_attribution(app, view, binding)
    assert submit(app, binding) == view
    assert app.recover_turn('session', 'analysis', complete_presentation=True) == view
    assert mutations == [phase]
    assert calls == {'intake': 1} and len(models) == 1


@pytest.mark.parametrize('mutation', ['changed', 'deleted', 'composition'])
def test_registered_fastq_unfinished_recovery_enforces_every_full_source(
        fastqs, tmp_path, monkeypatch, mutation):
    app, models, calls, binding, source = unfinished_inspection(fastqs, tmp_path, monkeypatch)
    if mutation == 'changed':
        change_beyond_intake_prefix(source)
    elif mutation == 'deleted':
        source.unlink()
    else:
        original = app._application._complete
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            change_beyond_intake_prefix(source)
            return result
        monkeypatch.setattr(app._application, '_complete', changed)
    view = app.recover_turn('session', 'analysis', complete_presentation=True)
    assert view.status == 'failed' and view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID', view
    state = app._application.sessions.load('session')
    assert state.generation == 0 and not state.revisions
    assert_attribution(app, view, binding)
    assert app.recover_turn('session', 'analysis', complete_presentation=True) == view
    assert submit(app, binding) == view
    assert calls == {'intake': 1} and len(models) == 1


def test_registered_fastq_failed_integrity_persists_across_interrupted_presentation(
        fastqs, tmp_path, monkeypatch):
    app, models, calls, binding, source = unfinished_inspection(fastqs, tmp_path, monkeypatch)
    change_beyond_intake_prefix(source)
    original = AnalysisSessions._record_result
    def interrupted_failure(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ResourceAdmissionError:
            raise Interrupted()
    with monkeypatch.context() as stopped:
        stopped.setattr(AnalysisSessions, '_record_result', interrupted_failure)
        with pytest.raises(Interrupted):
            app.recover_turn('session', 'analysis', complete_presentation=True)
    reopened, reopened_models = application(tmp_path)
    view = reopened.turn('session', 'analysis')
    assert view.status == 'failed' and view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID', view
    state = reopened._application.sessions.load('session')
    assert state.generation == 0 and not state.revisions
    assert reopened.recover_turn('session', 'analysis', complete_presentation=True) == view
    assert submit(reopened, binding) == view
    assert reopened._application.sessions.load('session') == state
    assert calls == {'intake': 1} and len(models) == 1 and not reopened_models


def test_registered_fastq_unfinished_recovery_preserves_immutable_failure_display(
        fastqs, tmp_path, monkeypatch):
    from agent.application import service
    paths = fastqs(count=300)
    app, models = application(tmp_path)
    binding = bind(app, register(app, paths))
    app.create_session('session')
    original = service.build_analysis_report
    def failed(*args, **kwargs):
        raise RuntimeError('Synthetic report publication failure')
    monkeypatch.setattr(service, 'build_analysis_report', failed)
    view = submit(app, binding)
    assert view.status == 'finalizing' and view.response.error.code == 'APP_REPORT_FAILED', view
    shown = app._application.sessions.load('session').interactions[0].presentation
    monkeypatch.setattr(service, 'build_analysis_report', original)
    change_beyond_intake_prefix(paths[-1])
    with pytest.raises(ResourceAdmissionError) as error:
        app.recover_turn('session', 'analysis', complete_presentation=True)
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    state = app._application.sessions.load('session')
    assert not state.revisions and state.turn('analysis').status == 'failed'
    assert state.interactions[0].presentation == shown
    assert app.turn('session', 'analysis').response == view.response
    assert len(models) == 1


@pytest.mark.parametrize('lifetime', ['unchanged', 'changed'])
def test_registered_fastq_fresh_process_unfinished_completion(fastqs, tmp_path, monkeypatch, lifetime):
    app, models, calls, binding, source = unfinished_inspection(fastqs, tmp_path, monkeypatch)
    if lifetime == 'changed':
        change_beyond_intake_prefix(source)
    script = '''
import json, sys
from dataclasses import replace
from agent.application import InteractiveAgentApplication
from agent.application.local_resources import RegisteredInputCollection
from agent.orchestration import PlanningModelProfile, ToolRegistry, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
def forbidden(*args, **kwargs): raise AssertionError('Unfinished recovery repeated providers or intake production')
default = build_default_tool_registry()
registry = ToolRegistry(tuple(replace(default.get(name), function=forbidden) if name == 'inspect_raw_scATAC'
                             else default.get(name) for name in default.names()))
profile = PlanningModelProfile('fastq-scripted', 'scripted', 'scripted/registered-fastq')
app = InteractiveAgentApplication(sys.argv[1], model_profiles=(profile,), default_profile_id=profile.profile_id,
    registry=registry, planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
initial = app._application.sessions.load('session')
assert initial.turn('analysis').status == 'linked' and not initial.revisions
submission = _serialize(initial.interactions[0].submission)
binding = RegisteredInputCollection(**submission['registered_input'], execution_inputs=submission['execution_inputs'])
view = app.recover_turn('session', 'analysis', complete_presentation=True)
state = app._application.sessions.load('session')
if sys.argv[2] == 'unchanged':
    assert view.status == 'succeeded' and view.revision_id
    assert state.generation == 1 and len(state.revisions) == 1
else:
    assert view.status == 'failed' and view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    assert state.generation == 0 and not state.revisions
assert state.interactions[0].submission == initial.interactions[0].submission
assert app.recover_turn('session', 'analysis', complete_presentation=True) == view
assert app.submit_turn('session', 'analysis', sys.argv[3], expected_generation=0, registered_input=binding) == view
assert app._application.sessions.load('session') == state
print(json.dumps(view.to_dict()))
'''
    child = subprocess.run([sys.executable, '-B', '-c', script, str(app._application.workspace_root), lifetime, UTTERANCE],
        env=os.environ.copy(), capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == app.turn('session', 'analysis').to_dict()
    assert_attribution(app, app.turn('session', 'analysis'), binding)
    assert calls == {'intake': 1} and len(models) == 1


@pytest.mark.parametrize('lifetime', ['changed', 'deleted'])
def test_registered_fastq_fresh_process_completed_history_survives_source_lifetime(
        fastqs, tmp_path, monkeypatch, lifetime):
    paths = fastqs()
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, registry=registry)
    records = register(app, paths)
    binding = bind(app, records)
    app.create_session('session')
    view = submit(app, binding)
    assert view.status == 'succeeded', view
    if lifetime == 'changed':
        paths[-1].write_bytes(b'Replaced after accepted historical result.\n')
    else:
        paths[-1].unlink()
    script = '''
import json, sys
from agent.application import InteractiveAgentApplication
from agent.application.local_resources import RegisteredInputCollection, ResourceAdmissionError
from agent.orchestration import PlanningModelProfile, raw_scatac_verifier
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.tools.data import raw_scatac, _raw_fastq
def forbidden(*args, **kwargs): raise AssertionError('Completed history replayed providers or science')
raw_scatac._reconstruct = raw_scatac_verifier._reconstruct = _raw_fastq.inspect_fastq_inputs = forbidden
profile = PlanningModelProfile('fastq-scripted', 'scripted', 'scripted/registered-fastq')
app = InteractiveAgentApplication(sys.argv[1], model_profiles=(profile,), default_profile_id=profile.profile_id,
    planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
initial = app._application.sessions.load('session')
submission = _serialize(initial.interactions[0].submission)
binding = RegisteredInputCollection(**submission['registered_input'], execution_inputs=submission['execution_inputs'])
view = app.recover_turn('session', 'analysis', complete_presentation=True)
assert view.status == 'succeeded' and app.reopen_session('session').active_revision_id == view.revision_id
assert app.revision('session', view.revision_id).revision_id == view.revision_id
assert app.submit_turn('session', 'analysis', sys.argv[2], expected_generation=0, registered_input=binding) == view
assert app._application.sessions.load('session') == initial
try:
    app.submit_turn('session', 'fresh', sys.argv[2], expected_generation=1, registered_input=binding)
except ResourceAdmissionError as error:
    assert error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
else:
    raise AssertionError('Consumed a historical collection as fresh input')
assert app._application.sessions.load('session') == initial
print(json.dumps(view.to_dict()))
'''
    child = subprocess.run([sys.executable, '-B', '-c', script, str(app._application.workspace_root), UTTERANCE],
        env=os.environ.copy(), capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == view.to_dict()
    assert calls == {'intake': 1} and len(models) == 1
    assert tuple(app.resources.load(record.resource_id) for record in records) == records


def test_operator_fastq_input_set_keeps_existing_unregistered_path(fastqs, tmp_path, monkeypatch):
    from agent.web.config import ScientificInputSet
    paths = fastqs()
    configured = ScientificInputSet('fastqs', 'Operator FASTQs',
        dict(raw_input_paths=[str(path) for path in paths], **DECLARATIONS))
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, registry=registry)
    app.create_session('session')
    view = app.submit_turn('session', 'operator', UTTERANCE, expected_generation=0,
                          execution_inputs=configured.inputs())
    assert view.status == 'succeeded', view
    state = app._application.sessions.load('session')
    assert 'registered_input' not in state.interactions[0].submission
    assert calls == {'intake': 1} and len(models) == 1
    assert not (app._application.workspace_root / 'local_resources').exists()
