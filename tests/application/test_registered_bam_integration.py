"""Registered BAMs reuse bounded intake and qualified production in Sessions."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agent.application import InteractiveAgentApplication
from agent.application.local_resources import ResourceAdmissionError
from agent.application.sessions import AnalysisSessions
from agent.orchestration import PlanningModelProfile, ToolRegistry, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.report import ANALYSIS_EVIDENCE_FILENAME
from agent.schemas.orchestration import _serialize
from agent.tools.data import bam_fragments, bam_fragments_verifier, bam_fragment_manifest
from agent.tools.data import raw_scatac, raw_scatac_manifest, scatac_fragments_v2

sys.path.insert(0, str(Path(__file__).parents[1]))
from bam_fragments.conftest import bam_factory, pair


PROFILE = PlanningModelProfile('bam-scripted', 'scripted', 'scripted/registered-bam')
INSPECT = 'inspect_raw_scATAC'
PREPARE = 'prepare_scATAC_bam_fragments'
UTTERANCE = 'Execute the declared operation on this registered BAM using the supplied scientific inputs.'
DECLARATIONS = dict(species='human', raw_assay='SCATAC', source_genome_assembly='hg38')


class Model:
    """Only semantic decisions are scripted; scientific owners execute normally."""
    model_id = PROFILE.model_id

    def __init__(self, target, callback=None):
        self.target, self.callback = target, callback
        self.calls = []

    def complete(self, *, prompt, response_schema):
        data = json.loads(prompt)
        self.calls.append(data)
        if 'turn_schema_version' in data:
            return json.dumps(dict(turn_schema_version=1,
                decision=dict(kind='execute_plan', target=self.target)))
        if 'selection_schema_version' in response_schema.get('properties', {}):
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=['raw_preprocessing'])))
        step_id = 'inspect' if self.target == INSPECT else 'prepare'
        if 'output_selection_schema_version' in data:
            if self.callback is not None:
                self.callback()
            return json.dumps(dict(outputs=[dict(name='intake' if self.target == INSPECT else 'fragments',
                step_id=step_id, output_key='manifest_path')]))
        ports = ([('raw_input', 'raw_input_paths')] if self.target == INSPECT else
                 [('source', 'source_path'), ('intake', 'intake_manifest_path'),
                  ('library_context', 'library_context_path'), ('reference', 'reference_bundle_path'),
                  ('source_profile', 'source_profile')])
        step = dict(step_id=step_id, tool=self.target, control_dependencies=[],
            sources=[dict(target=port, source=dict(kind='input', input=name)) for port, name in ports])
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[step])))


def application(tmp_path, *, targets=(PREPARE,), registry=None, callback=None):
    models = []
    def factory(profile):
        assert profile == PROFILE
        model = Model(targets[min(len(models), len(targets) - 1)], callback)
        models.append(model)
        return model
    app = InteractiveAgentApplication(tmp_path / 'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, approved_source_roots=(tmp_path,), registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    return app, models


def instrument(monkeypatch):
    """Count actual tool production and independent BAM reconstruction bodies."""
    calls = dict(intake=0, production=0, verification=0)
    registry = build_default_tool_registry()
    original_inspect = registry.get(INSPECT).function
    original_prepare, original_reconstruct = bam_fragments.prepare_in_stage, bam_fragments_verifier.reconstruct
    def inspect(**arguments):
        calls['intake'] += 1
        return original_inspect(**arguments)
    def prepare(*args, **kwargs):
        calls['production'] += 1
        return original_prepare(*args, **kwargs)
    def reconstruct(*args, **kwargs):
        calls['verification'] += 1
        return original_reconstruct(*args, **kwargs)
    monkeypatch.setattr(bam_fragments, 'prepare_in_stage', prepare)
    monkeypatch.setattr(bam_fragments_verifier, 'reconstruct', reconstruct)
    return ToolRegistry(tuple(replace(registry.get(name), function=inspect) if name == INSPECT
                              else registry.get(name) for name in registry.names())), calls


def register(app, arguments, key='bam-source'):
    return app.resources.register(key, arguments['source_path'], input_type='bam', label='Supplied BAM',
                                  attribution='Explicit operator-approved synthetic paired ATAC BAM.')


def bind(app, resource, arguments, *, target=PREPARE):
    declarations = ({key: value for key, value in arguments.items()
                     if key not in {'source_path', 'source_sha256', 'output_dir'}}
                    if target == PREPARE else arguments)
    return app.resources.resolve(resource.resource_id, tool_name=target, scientific_inputs=declarations)


def submit(app, binding, turn='analysis', generation=0):
    return app.submit_turn('session', turn, UTTERANCE, expected_generation=generation, registered_input=binding)


def step_for(app, view):
    step, = app._application.run_store.load(view.run_id).steps
    return step


def assert_attribution(app, view, binding):
    state = app._application.sessions.load('session')
    interaction = next(item for item in state.interactions if item.turn_id == view.turn_id)
    assert _serialize(interaction.submission['registered_input']) == binding.attribution()
    assert _serialize(interaction.submission['execution_inputs']) == _serialize(binding.execution_inputs)
    assert str(app._application.workspace_root.parent) not in json.dumps(view.to_dict())


def test_registered_bam_inspection_then_explicit_qualified_production(bam_factory, tmp_path, monkeypatch):
    arguments = bam_factory()
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, targets=(INSPECT, PREPARE), registry=registry)
    resource = register(app, arguments)
    inspection_binding = bind(app, resource, DECLARATIONS, target=INSPECT)
    assert _serialize(inspection_binding.execution_inputs) == dict(raw_input_paths=[resource.source_path], **DECLARATIONS)
    assert not models and calls == {'intake': 0, 'production': 0, 'verification': 0}
    assert not list(app._application.sessions._store.root.glob('*.json'))
    app.create_session('session')
    inspected = submit(app, inspection_binding, turn='inspect')
    assert inspected.status == 'succeeded', inspected
    intake = step_for(app, inspected)
    assert intake.verification.passed and intake.verification.artifact_authority is None
    assert intake.result['readiness'] == 'READY' and intake.result['n_files'] == intake.result['n_groups'] == 1
    _, manifest, _ = raw_scatac_manifest.load_raw_intake_manifest(intake.result['manifest_path'],
        expected_sha256=intake.result['manifest_sha256'])
    assert [file.path for file in manifest.files] == [resource.source_path]
    assert intake.result['manifest_sha256'] == arguments['intake_manifest_sha256']
    assert_attribution(app, inspected, inspection_binding)
    # The context is explicitly prepared by its existing owner; identical intake
    # bytes at this actual application locator retain the existing context binding.
    arguments.update(intake_manifest_path=intake.result['manifest_path'],
                     intake_manifest_sha256=intake.result['manifest_sha256'])
    producer_binding = bind(app, resource, arguments)
    prepared = submit(app, producer_binding, turn='prepare', generation=1)
    assert prepared.status == 'succeeded', prepared
    step = step_for(app, prepared)
    assert step.resolved_arguments['source_path'] == resource.source_path
    assert step.resolved_arguments['source_sha256'] == resource.source_sha256
    authority = step.verification.artifact_authority
    assert authority['schema_version'] == 2 and authority['verifier']['id'] == 'agent.bam-fragments-independent'
    assert authority['producer_qualification']['scope'] == 'bam_read_pair_transformation.v1'
    assert authority['source_policy'] == 'historical_verified_sources.v1'
    assert {'path': resource.source_path, 'sha256': resource.source_sha256,
            'size_bytes': resource.size_bytes} in authority['historical_sources']
    assert (step.result['n_fragment_records'], step.result['eligible_pairs'], step.result['total_support']) == (1, 1, 1)
    fragments = scatac_fragments_v2.load_fragments_manifest_v2(step.result['manifest_path'],
        expected_sha256=step.result['manifest_sha256'])
    assert fragments['libraries'][0]['strand'] == {'mode': 'absent', 'definition': None}
    pointer = fragments['libraries'][0]['provenance']['producer_record']
    record = bam_fragment_manifest.load_record(pointer['path'], pointer['sha256'])
    assert record['source']['sha256'] == resource.source_sha256
    assert record['source_history'] == bam_fragment_manifest.HISTORY
    evidence = app._application._workspace.run_paths(prepared.run_id).evidence / ANALYSIS_EVIDENCE_FILENAME
    facts = json.loads(evidence.read_bytes())['steps'][0]['facts']
    assert facts['route'] == 'bam_fragment_production'
    assert facts['bam_transformation_verification'] == 'independently_recomputed'
    assert app.evidence('session', prepared.revision_id, 'fragments').status == 'available'
    assert_attribution(app, prepared, producer_binding)
    assert submit(app, producer_binding, turn='prepare', generation=1) == prepared
    assert calls == {'intake': 1, 'production': 1, 'verification': 1}
    assert len(models) == 2 and all(len(model.calls) == 4 for model in models)
    script = '''
import json, sys
from agent.application import InteractiveAgentApplication
from agent.application.local_resources import RegisteredInput
from agent.orchestration import PlanningModelProfile, raw_scatac_verifier
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.tools.data import raw_scatac, _raw_bam, bam_fragments, bam_fragments_verifier
def forbidden(*args, **kwargs): raise AssertionError('Accepted lifecycle replayed providers or science')
raw_scatac._reconstruct = raw_scatac_verifier._reconstruct = _raw_bam.inspect_bam_inputs = forbidden
bam_fragments.prepare_in_stage = bam_fragments_verifier.reconstruct = forbidden
profile = PlanningModelProfile('bam-scripted', 'scripted', 'scripted/registered-bam')
app = InteractiveAgentApplication(sys.argv[1], model_profiles=(profile,), default_profile_id=profile.profile_id,
    planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
initial = app._application.sessions.load('session')
assert len(initial.revisions) == 2
submission = _serialize(initial.interactions[1].submission)
binding = RegisteredInput(**submission['registered_input'], execution_inputs=submission['execution_inputs'])
view = app.recover_turn('session', 'prepare', complete_presentation=True)
assert view.status == 'succeeded' and app.reopen_session('session').active_revision_id == view.revision_id
assert app.revision('session', view.revision_id).revision_id == view.revision_id
assert app.submit_turn('session', 'prepare', sys.argv[2], expected_generation=1, registered_input=binding) == view
assert app._application.sessions.load('session') == initial
print(json.dumps(view.to_dict()))
'''
    child = subprocess.run([sys.executable, '-B', '-c', script,
        str(app._application.workspace_root), UTTERANCE],
        env=os.environ.copy(), capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == prepared.to_dict()
    assert calls == {'intake': 1, 'production': 1, 'verification': 1}


@pytest.mark.parametrize('case,readiness', [('missing', 'NEEDS_USER_INPUT'), ('invalid', 'INVALID'), ('species', 'UNSUPPORTED')])
def test_registered_bam_successful_inspection_keeps_nonready_findings(bam_factory, tmp_path, monkeypatch, case, readiness):
    arguments = bam_factory()
    declarations = {} if case == 'missing' else dict(DECLARATIONS)
    if case == 'invalid':
        Path(arguments['source_path']).write_bytes(b'Not a readable BAM source.\n')
    elif case == 'species':
        declarations['species'] = 'rat'
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, targets=(INSPECT,), registry=registry)
    resource = register(app, arguments)
    binding = bind(app, resource, declarations, target=INSPECT)
    assert not models and calls['intake'] == 0
    app.create_session('session')
    view = submit(app, binding)
    assert view.status == 'succeeded' and view.revision_id, view
    step = step_for(app, view)
    assert step.result['readiness'] == readiness and step.verification.passed
    assert step.verification.artifact_authority is None
    assert app.evidence('session', view.revision_id, 'intake').status == 'available'
    assert_attribution(app, view, binding)
    assert calls == {'intake': 1, 'production': 0, 'verification': 0}


def test_registered_bam_existing_session_navigation_and_branch(bam_factory, tmp_path, monkeypatch):
    first, second = bam_factory(namespace='first'), bam_factory(namespace='second')
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, registry=registry)
    one_binding = bind(app, register(app, first), first)
    two_binding = bind(app, register(app, second, 'second-source'), second)
    app.create_session('session')
    one = submit(app, one_binding, turn='one')
    two = submit(app, two_binding, turn='two', generation=1)
    assert one.status == two.status == 'succeeded', (one, two)
    original = app._application.sessions.load('session').interactions[0].submission
    app.activate_revision('session', 'back', one.revision_id, expected_generation=2)
    assert calls == {'intake': 0, 'production': 2, 'verification': 2}
    branch = submit(app, two_binding, turn='branch', generation=3)
    assert branch.status == 'succeeded', branch
    state = app._application.sessions.load('session')
    assert state.revisions[-1].parent_revision_id == one.revision_id
    assert state.interactions[0].submission == original
    assert_attribution(app, branch, two_binding)
    assert calls == {'intake': 0, 'production': 3, 'verification': 3} and len(models) == 3


@pytest.mark.parametrize('mismatch', ['unrelated', 'membership', 'size', 'digest'])
def test_registered_bam_intake_association_fails_before_interaction(bam_factory, tmp_path, monkeypatch, mismatch):
    arguments = bam_factory()
    if mismatch in {'unrelated', 'membership'}:
        other = bam_factory()
        intake = (other if mismatch == 'unrelated' else raw_scatac.inspect_raw_scATAC(
            [arguments['source_path'], other['source_path']], tmp_path / 'multiple-intake', **DECLARATIONS))
        arguments.update(intake_manifest_path=intake['manifest_path'] if mismatch == 'membership' else intake['intake_manifest_path'],
                         intake_manifest_sha256=intake['manifest_sha256'] if mismatch == 'membership' else intake['intake_manifest_sha256'])
    elif mismatch == 'size':
        with Path(arguments['source_path']).open('ab') as stream:
            stream.write(b'changed before registration')
    else:
        arguments['intake_manifest_sha256'] = '0' * 64
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, registry=registry)
    resource = register(app, arguments)
    with pytest.raises(ResourceAdmissionError) as error:
        bind(app, resource, arguments)
    assert error.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'
    assert not models and calls == {'intake': 0, 'production': 0, 'verification': 0}
    assert not list(app._application.sessions._store.root.glob('*.json'))


@pytest.mark.parametrize('failure,code', [
    ('reference', 'REFERENCE_DIGEST_MISMATCH'), ('context', 'BAM_FRAGMENTS_CONTEXT_MISMATCH'),
    ('correction', 'BAM_FRAGMENTS_BARCODE_POLICY_UNSUPPORTED'), ('cb', 'BAM_FRAGMENTS_BARCODE_MISMATCH'),
    ('stale', 'BAM_FRAGMENTS_INTAKE_MISMATCH'), ('sidecar', 'BAM_FRAGMENTS_INTAKE_MISMATCH'),
    ('verification', 'BAM_FRAGMENTS_CONSERVATION_MISMATCH'),
])
def test_registered_bam_scientific_failures_remain_owned(bam_factory, tmp_path, monkeypatch, failure, code):
    arguments = bam_factory(pair(b={'cb': 'different-1'}) if failure == 'cb' else None, corrected=failure != 'correction')
    if failure == 'reference':
        arguments['reference_bundle_sha256'] = '0' * 64
    elif failure == 'context':
        other = bam_factory()
        arguments.update({key: other[key] for key in ('library_context_path', 'library_context_sha256')})
    elif failure == 'stale':
        path = Path(arguments['source_path'])
        changed_time = path.stat().st_mtime_ns + 1_000_000_000
        os.utime(path, ns=(changed_time, changed_time))
    elif failure == 'sidecar':
        Path(arguments['source_path'] + '.bai').write_bytes(b'new advisory sidecar')
    elif failure == 'verification':
        original = bam_fragments.aggregate
        def incorrect(*args, **kwargs):
            path, summary = original(*args, **kwargs)
            data = path.read_bytes().replace(b'chr2\t14\t75', b'chr2\t15\t75', 1)
            path.write_bytes(data)
            summary['canonical_record_stream_sha256'] = hashlib.sha256(data).hexdigest()
            return path, summary
        monkeypatch.setattr(bam_fragments, 'aggregate', incorrect)
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, registry=registry)
    resource = register(app, arguments)
    binding = bind(app, resource, arguments)
    assert not models and not calls['production']
    app.create_session('session')
    view = submit(app, binding)
    assert view.status == 'failed' and view.revision_id is None, view
    step = step_for(app, view)
    assert view.error.code == step.error.code == code
    assert step.verification is None or step.verification.artifact_authority is None
    assert app._application.sessions.load('session').generation == 0
    assert app.resources.load(resource.resource_id).public()['status'] == 'registered'
    assert submit(app, binding) == view and len(models) == 1
    assert calls == {'intake': 0, 'production': 1, 'verification': int(failure == 'verification')}


def test_registered_bam_unsupported_profile_keeps_existing_contract_failure(bam_factory, tmp_path, monkeypatch):
    arguments = bam_factory()
    arguments['source_profile'] = 'infer-from-CB-tags'
    registry, calls = instrument(monkeypatch)
    app, _ = application(tmp_path, registry=registry)
    binding = bind(app, register(app, arguments), arguments)
    app.create_session('session')
    view = submit(app, binding)
    assert view.status == 'failed' and view.revision_id is None, view
    run = app._application.run_store.load(view.run_id)
    assert view.error.code == run.errors[0].code
    assert not view.error.code.startswith('LOCAL_RESOURCE_') and view.error.code != 'INTERACTIVE_APPLICATION_FAILED'
    assert calls == {'intake': 0, 'production': 0, 'verification': 0}


@pytest.mark.parametrize('target,during_provider', [(INSPECT, False), (INSPECT, True), (PREPARE, False), (PREPARE, True)])
def test_registered_bam_drift_blocks_new_consumption(bam_factory, tmp_path, monkeypatch, target, during_provider):
    arguments = bam_factory()
    source = Path(arguments['source_path'])
    def changed():
        source.write_bytes(b'Changed exact registered BAM bytes.\n')
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, targets=(target,), registry=registry, callback=changed if during_provider else None)
    resource = register(app, arguments)
    binding = bind(app, resource, arguments if target == PREPARE else DECLARATIONS, target=target)
    app.create_session('session')
    if during_provider:
        view = submit(app, binding)
        assert view.status == 'failed' and view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID', view
        assert len(models) == 1
    else:
        changed()
        with pytest.raises(ResourceAdmissionError) as error:
            submit(app, binding)
        assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID' and not models
    assert calls == {'intake': 0, 'production': 0, 'verification': 0}
    assert not app._application.sessions.load('session').revisions


class Interrupted(BaseException):
    pass


def unfinished_inspection(bam_factory, tmp_path, monkeypatch):
    arguments = bam_factory()
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, targets=(INSPECT,), registry=registry)
    resource = register(app, arguments)
    binding = bind(app, resource, DECLARATIONS, target=INSPECT)
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
    return app, models, calls, binding, Path(arguments['source_path'])


def test_registered_bam_raw_drift_before_initial_acceptance_is_atomic(bam_factory, tmp_path, monkeypatch):
    arguments = bam_factory()
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, targets=(INSPECT,), registry=registry)
    binding = bind(app, register(app, arguments), DECLARATIONS, target=INSPECT)
    app.create_session('session')
    original = AnalysisSessions._record_result
    def changed(*args, **kwargs):
        Path(arguments['source_path']).write_bytes(b'Changed after verified raw application completion.\n')
        return original(*args, **kwargs)
    monkeypatch.setattr(AnalysisSessions, '_record_result', changed)
    view = submit(app, binding)
    assert view.status == 'failed' and view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID', view
    assert step_for(app, view).verification.passed
    state = app._application.sessions.load('session')
    assert state.generation == 0 and not state.revisions
    assert state.turn('analysis').status == state.interactions[0].status == 'failed'
    assert submit(app, binding) == view
    assert calls == {'intake': 1, 'production': 0, 'verification': 0} and len(models) == 1


@pytest.mark.parametrize('mutation', ['changed', 'deleted', 'composition'])
def test_registered_bam_unfinished_raw_recovery_checks_full_source(bam_factory, tmp_path, monkeypatch, mutation):
    app, models, calls, binding, source = unfinished_inspection(bam_factory, tmp_path, monkeypatch)
    if mutation == 'changed':
        source.write_bytes(b'Changed before registered intake acceptance.\n')
    elif mutation == 'deleted':
        source.unlink()
    else:
        original = app._application._complete
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            source.write_bytes(b'Changed during successful recovery composition.\n')
            return result
        monkeypatch.setattr(app._application, '_complete', changed)
    view = app.recover_turn('session', 'analysis', complete_presentation=True)
    assert view.status == 'failed' and view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID', view
    state = app._application.sessions.load('session')
    assert state.generation == 0 and not state.revisions
    assert app.recover_turn('session', 'analysis', complete_presentation=True) == view
    assert submit(app, binding) == view
    assert calls == {'intake': 1, 'production': 0, 'verification': 0} and len(models) == 1


def test_registered_bam_raw_integrity_failure_survives_presentation_interruption(bam_factory, tmp_path, monkeypatch):
    app, models, calls, binding, source = unfinished_inspection(bam_factory, tmp_path, monkeypatch)
    source.write_bytes(b'Changed before atomic integrity failure.\n')
    original = AnalysisSessions._record_result
    def interrupt_after_failure(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ResourceAdmissionError:
            raise Interrupted()
    with monkeypatch.context() as stopped:
        stopped.setattr(AnalysisSessions, '_record_result', interrupt_after_failure)
        with pytest.raises(Interrupted):
            app.recover_turn('session', 'analysis', complete_presentation=True)
    reopened, reopened_models = application(tmp_path, targets=(INSPECT,))
    view = reopened.turn('session', 'analysis')
    assert view.status == 'failed' and view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID', view
    state = reopened._application.sessions.load('session')
    assert state.generation == 0 and not state.revisions
    assert reopened.recover_turn('session', 'analysis', complete_presentation=True) == view
    assert submit(reopened, binding) == view
    assert calls['intake'] == 1 and len(models) == 1 and not reopened_models
    assert reopened._application.sessions.load('session') == state


def test_registered_bam_raw_recovery_preserves_prior_immutable_failure_display(bam_factory, tmp_path, monkeypatch):
    from agent.application import service
    arguments = bam_factory()
    app, models = application(tmp_path, targets=(INSPECT,))
    binding = bind(app, register(app, arguments), DECLARATIONS, target=INSPECT)
    app.create_session('session')
    original = service.build_analysis_report
    def failed(*args, **kwargs):
        raise RuntimeError('Synthetic report publication failure')
    monkeypatch.setattr(service, 'build_analysis_report', failed)
    view = submit(app, binding)
    assert view.status == 'finalizing' and view.response.error.code == 'APP_REPORT_FAILED', view
    shown = app._application.sessions.load('session').interactions[0].presentation
    monkeypatch.setattr(service, 'build_analysis_report', original)
    Path(arguments['source_path']).write_bytes(b'Changed after immutable failure presentation.\n')
    with pytest.raises(ResourceAdmissionError) as error:
        app.recover_turn('session', 'analysis', complete_presentation=True)
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    state = app._application.sessions.load('session')
    assert not state.revisions and state.turn('analysis').status == 'failed'
    assert state.interactions[0].presentation == shown and app.turn('session', 'analysis').response == view.response
    assert len(models) == 1


@pytest.mark.parametrize('lifetime', ['unchanged', 'changed'])
def test_registered_bam_fresh_process_unfinished_raw_completion(bam_factory, tmp_path, monkeypatch, lifetime):
    app, models, calls, _, source = unfinished_inspection(bam_factory, tmp_path, monkeypatch)
    if lifetime == 'changed':
        source.write_bytes(b'Changed before fresh-process raw acceptance.\n')
    script = '''
import json, sys
from dataclasses import replace
from agent.application import InteractiveAgentApplication
from agent.application.local_resources import RegisteredInput
from agent.orchestration import PlanningModelProfile, ToolRegistry, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.tools.data import raw_scatac, bam_fragments, bam_fragments_verifier
def forbidden(*args, **kwargs): raise AssertionError('Unfinished recovery replayed providers or production')
default = build_default_tool_registry()
registry = ToolRegistry(tuple(replace(default.get(name), function=forbidden) if name == 'inspect_raw_scATAC'
                             else default.get(name) for name in default.names()))
bam_fragments.prepare_in_stage = bam_fragments_verifier.reconstruct = forbidden
profile = PlanningModelProfile('bam-scripted', 'scripted', 'scripted/registered-bam')
app = InteractiveAgentApplication(sys.argv[1], model_profiles=(profile,), default_profile_id=profile.profile_id, registry=registry,
    planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
initial = app._application.sessions.load('session')
assert initial.turn('analysis').status == 'linked' and not initial.revisions
submission = _serialize(initial.interactions[0].submission)
binding = RegisteredInput(**submission['registered_input'], execution_inputs=submission['execution_inputs'])
# Existing evidence completion may repeat bounded owner reinspection; it never
# reruns the registered inspection tool or constructs another provider.
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
    child = subprocess.run([sys.executable, '-B', '-c', script,
        str(app._application.workspace_root), lifetime, UTTERANCE],
        env=os.environ.copy(), capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == app.turn('session', 'analysis').to_dict()
    assert calls == {'intake': 1, 'production': 0, 'verification': 0} and len(models) == 1


@pytest.mark.parametrize('target,lifetime', [(INSPECT, 'changed'), (INSPECT, 'deleted'), (PREPARE, 'changed'), (PREPARE, 'deleted')])
def test_registered_bam_fresh_process_completed_history_never_replays_science(bam_factory, tmp_path, monkeypatch, target, lifetime):
    arguments = bam_factory()
    registry, calls = instrument(monkeypatch)
    app, models = application(tmp_path, targets=(target,), registry=registry)
    resource = register(app, arguments)
    binding = bind(app, resource, arguments if target == PREPARE else DECLARATIONS, target=target)
    app.create_session('session')
    view = submit(app, binding)
    assert view.status == 'succeeded', view
    if lifetime == 'changed':
        Path(resource.source_path).write_bytes(b'Historical source replaced after accepted result.\n')
    else:
        Path(resource.source_path).unlink()
    script = '''
import json, sys
from agent.application import InteractiveAgentApplication
from agent.application.local_resources import RegisteredInput, ResourceAdmissionError
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.tools.data import raw_scatac, bam_fragments, bam_fragments_verifier, _raw_bam
from agent.orchestration import raw_scatac_verifier
def forbidden(*args, **kwargs): raise AssertionError('Completed history replayed a provider or science')
raw_scatac._reconstruct = raw_scatac_verifier._reconstruct = _raw_bam.inspect_bam_inputs = forbidden
bam_fragments.prepare_in_stage = bam_fragments_verifier.reconstruct = forbidden
profile = PlanningModelProfile('bam-scripted', 'scripted', 'scripted/registered-bam')
app = InteractiveAgentApplication(sys.argv[1], model_profiles=(profile,), default_profile_id=profile.profile_id,
    planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
state = app._application.sessions.load('session')
submission = _serialize(state.interactions[0].submission)
binding = RegisteredInput(**submission['registered_input'], execution_inputs=submission['execution_inputs'])
view = app.recover_turn('session', 'analysis', complete_presentation=True)
assert view.status == 'succeeded' and app.reopen_session('session').active_revision_id == view.revision_id
assert app.revision('session', view.revision_id).revision_id == view.revision_id
assert app.submit_turn('session', 'analysis', sys.argv[2], expected_generation=0, registered_input=binding) == view
assert app._application.sessions.load('session') == state
try:
    app.submit_turn('session', 'fresh', sys.argv[2], expected_generation=1, registered_input=binding)
except ResourceAdmissionError as error:
    assert error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
else:
    raise AssertionError('Historical source consumed as fresh input')
assert app._application.sessions.load('session') == state
print(json.dumps(view.to_dict()))
'''
    child = subprocess.run([sys.executable, '-B', '-c', script, str(app._application.workspace_root), UTTERANCE],
        env=os.environ.copy(), capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == view.to_dict()
    assert calls == {'intake': int(target == INSPECT), 'production': int(target == PREPARE), 'verification': int(target == PREPARE)}
    assert len(models) == 1 and app.resources.load(resource.resource_id) == resource
