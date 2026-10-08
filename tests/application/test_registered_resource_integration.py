"""UA1 registered sources use normal interpreted science and durable Sessions."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.application.local_resources import ResourceAdmissionError
from agent.orchestration import PlanningModelProfile, ToolRegistry, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.tools.data import scatac_reference
from agent.tools.data import _external_matrix_io as production
from agent.tools.data import cell_by_ccre_verifier as verification


PROFILE = PlanningModelProfile('local-scripted', 'scripted', 'scripted/local-resource')
ADOPT = 'adopt_scATAC_cell_by_ccre'
UTTERANCE = 'Adopt the selected canonical matrix with the supplied declarations.'


@pytest.fixture
def local_source(tmp_path):
    """A full exact synthetic vocabulary and sparse source, without model science."""
    sources = tmp_path / 'approved'
    sources.mkdir()
    fasta = sources / 'reference.fa'
    fasta.write_text('>chr1\n' + 'A' * 100 + '\n')
    fai = sources / 'reference.fa.fai'
    fai.write_text('chr1\t100\t6\t100\t101\n')
    bed = sources / 'features.bed'
    bed.write_text('chr1\t0\t10\nchr1\t10\t20\nchr1\t30\t40\n')
    reference = scatac_reference.build_scatac_reference_bundle(
        species='human', target_assembly='hg38', fasta_path=fasta,
        fai_path=fai, ccre_bed_path=bed,
    )
    pointer = scatac_reference.publish_scatac_reference_bundle(
        reference, sources / 'reference.json',
    )
    source = sources / 'supplied.h5ad'
    ad.AnnData(
        csr_matrix(np.asarray([[1, 2, 0], [2, 1, 0], [0, 0, 0]], dtype=np.int64)),
        obs=pd.DataFrame(index=['cell-z', 'cell-a', 'cell-empty']),
        var=pd.DataFrame(index=['chr1:0-10', 'chr1:10-20', 'chr1:30-40']),
    ).write_h5ad(source)
    declarations = dict(
        reference_manifest_path=pointer['manifest_path'],
        reference_manifest_sha256=pointer['manifest_sha256'],
        species='human', assembly='hg38', matrix_semantics='fragment_counts',
    )
    return source, declarations


class ScriptedModel:
    """Semantic choices are scripted; all deterministic and scientific owners run."""

    def __init__(self, target=ADOPT, *, output_callback=None):
        self.model_id = PROFILE.model_id
        self.target, self.output_callback = target, output_callback
        self.calls = []

    def complete(self, *, prompt, response_schema):
        data = json.loads(prompt)
        self.calls.append(data)
        if 'turn_schema_version' in data:
            return json.dumps(dict(turn_schema_version=1,
                decision=dict(kind='execute_plan', target=self.target)))
        if 'selection_schema_version' in response_schema.get('properties', {}):
            capability = 'exact_matrix_adoption' if self.target == ADOPT else 'processed_inspection'
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=[capability])))
        if 'output_selection_schema_version' in data:
            if self.output_callback is not None:
                self.output_callback()
            outputs = ([dict(name='matrix', step_id='adopt', output_key='manifest_path'),
                        dict(name='matrix_h5ad', step_id='adopt', output_key='matrix_path')]
                       if self.target == ADOPT else
                       [dict(name='inspection', step_id='inspect', output_key='n_cells')])
            return json.dumps(dict(outputs=outputs))
        if self.target == ADOPT:
            sources = [dict(target=target, source=dict(kind='input', input=key))
                       for target, key in (
                           ('source', 'source_path'),
                           ('reference', 'reference_manifest_path'),
                           ('species', 'species'), ('assembly', 'assembly'),
                           ('matrix_semantics', 'matrix_semantics'),
                       )]
            step = dict(step_id='adopt', tool=ADOPT, sources=sources, control_dependencies=[])
        else:
            step = dict(step_id='inspect', tool='inspect_scATAC', sources=[], control_dependencies=[])
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[step])))


def service_for(tmp_path, *, target=ADOPT, output_callback=None, registry=None):
    models = []

    def factory(profile):
        assert profile == PROFILE
        model = ScriptedModel(target, output_callback=output_callback)
        models.append(model)
        return model

    service = InteractiveAgentApplication(
        tmp_path / 'workspace', model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}),
        approved_source_roots=(tmp_path / 'approved',), registry=registry,
    )
    return service, models


def register(service, source, *, key='local-source'):
    return service.resources.register(
        key, source, input_type='h5ad', label='Supplied canonical matrix',
        attribution='Operator-approved synthetic source supplied for this test.',
    )


def bind(service, resource, declarations):
    return service.resources.resolve(resource.resource_id, tool_name=ADOPT,
                                     scientific_inputs=declarations)


def submit(service, binding, *, turn='adopt', generation=0, session='analysis'):
    return service.submit_turn(session, turn, UTTERANCE, expected_generation=generation,
                               registered_input=binding)


def count_science(monkeypatch):
    calls = dict(production=0, verification=0)
    original_build, original_verify = production.build, verification._verify_external

    def build(*args, **kwargs):
        calls['production'] += 1
        return original_build(*args, **kwargs)

    def verify(*args, **kwargs):
        calls['verification'] += 1
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(production, 'build', build)
    monkeypatch.setattr(verification, '_verify_external', verify)
    return calls


def assert_failed_science(service, view, code):
    assert view.status == 'failed' and view.revision_id is None, view
    assert view.error is not None and view.error.code == code
    state = service._application.sessions.load('analysis')
    assert state.generation == 0 and not state.revisions
    assert state.turn(view.turn_id).status == 'failed'
    run = service._application.run_store.load(view.run_id)
    assert run.errors and run.errors[0].code == code
    assert run.steps[0].error.code == code
    assert run.steps[0].verification is None or run.steps[0].verification.artifact_authority is None
    assert str(service._application.workspace_root.parent) not in json.dumps(view.to_dict())


def test_interpreted_adoption_revision_and_fresh_process_retry(local_source, tmp_path, monkeypatch):
    source, declarations = local_source
    service, models = service_for(tmp_path)
    calls = count_science(monkeypatch)
    resource = register(service, source)
    assert calls == {'production': 0, 'verification': 0} and not models
    assert not list(service._application.sessions._store.root.glob('*.json'))
    binding = bind(service, resource, declarations)
    assert binding.execution_inputs['source_sha256'] == resource.source_sha256
    assert dict(binding.execution_inputs) == dict(
        source_path=str(source), source_sha256=resource.source_sha256, **declarations,
    )
    service.create_session('analysis')
    fingerprint = service.validate_submission('analysis', 'adopt', UTTERANCE,
        expected_generation=0, registered_input=binding)
    assert len(fingerprint) == 64 and not models
    assert not service._application.sessions.load('analysis').interactions
    view = submit(service, binding)
    assert view.status == 'succeeded' and view.response.status == 'activated', view
    assert calls == {'production': 1, 'verification': 1}
    assert len(models) == 1 and len(models[0].calls) == 4
    state = service._application.sessions.load('analysis')
    revision, = state.revisions
    assert revision.revision_id == view.revision_id and revision.parent_revision_id is None
    assert [(o.name, o.step_id, o.output_key) for o in revision.outputs] == [
        ('matrix', 'adopt', 'manifest_path'), ('matrix_h5ad', 'adopt', 'matrix_path'),
    ]
    submission = state.interactions[0].submission
    assert _serialize(submission['registered_input']) == binding.attribution()
    assert dict(submission['execution_inputs']) == dict(binding.execution_inputs)
    run = service._application.run_store.load(view.run_id)
    assert run.request.inputs['source_sha256'] == resource.source_sha256
    assert all(run.request.inputs[key] == value for key, value in binding.execution_inputs.items())
    step, = run.steps
    authority = step.verification.artifact_authority
    assert authority['schema_version'] == 2
    assert authority['verifier']['id'] == 'agent.cell-by-ccre-independent'
    assert step.result['source_sha256'] == resource.source_sha256
    assert step.result['logical_matrix_sha256'] == step.result['source_logical_matrix_sha256']
    original, adopted = ad.read_h5ad(source), ad.read_h5ad(step.result['matrix_path'])
    assert list(adopted.obs_names) == list(original.obs_names)
    assert list(adopted.var_names) == list(original.var_names)
    assert isinstance(adopted.X, csr_matrix) and adopted.X.dtype == np.int64
    for field in ('indptr', 'indices', 'data'):
        np.testing.assert_array_equal(getattr(adopted.X, field), getattr(original.X, field))
    assert (step.result['nnz'], step.result['total_count'], step.result['zero_row_count']) == (4, 6, 1)
    assert service.evidence('analysis', view.revision_id, 'matrix').status == 'available'
    assert submit(service, binding) == view
    assert calls == {'production': 1, 'verification': 1} and len(models) == 1
    script = '''
import json, sys
from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
from agent.tools.data import _external_matrix_io, cell_by_ccre_verifier
def forbidden(*args, **kwargs): raise AssertionError('Completed history repeated model or science')
_external_matrix_io.build = cell_by_ccre_verifier._verify_external = forbidden
profile = PlanningModelProfile('local-scripted', 'scripted', 'scripted/local-resource')
app = InteractiveAgentApplication(sys.argv[1], model_profiles=(profile,), default_profile_id=profile.profile_id,
    planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
record = app.resources.load(sys.argv[2])
binding = app.resources.resolve(record.resource_id, tool_name='adopt_scATAC_cell_by_ccre',
    scientific_inputs=json.loads(sys.argv[3]))
state = app.reopen_session('analysis')
retry = app.submit_turn('analysis', 'adopt', sys.argv[4], expected_generation=0, registered_input=binding)
assert state.active_revision_id == retry.revision_id
persisted = app._application.sessions.load('analysis')
assert persisted.interactions[0].submission['registered_input']['resource_id'] == record.resource_id
print(json.dumps(retry.to_dict()))
'''
    child = subprocess.run([sys.executable, '-B', '-c', script,
        str(service._application.workspace_root), resource.resource_id,
        json.dumps(declarations), UTTERANCE], env=os.environ.copy(), capture_output=True,
        text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == view.to_dict()
    assert calls == {'production': 1, 'verification': 1}


def test_existing_session_new_resource_navigation_and_branch(local_source, tmp_path, monkeypatch):
    source, declarations = local_source
    service, models = service_for(tmp_path)
    calls = count_science(monkeypatch)
    first = register(service, source)
    first_binding = bind(service, first, declarations)
    service.create_session('analysis')
    one = submit(service, first_binding, turn='one')
    assert one.status == 'succeeded', one
    first_submission = service._application.sessions.load('analysis').interactions[0].submission
    second_source = source.with_name('second.h5ad')
    matrix = ad.read_h5ad(source)
    matrix.X.data[0] = 3
    matrix.write_h5ad(second_source)
    second = register(service, second_source, key='second-source')
    second_binding = bind(service, second, declarations)
    assert second.resource_id != first.resource_id and second.source_sha256 != first.source_sha256
    two = submit(service, second_binding, turn='two', generation=1)
    assert two.status == 'succeeded', two
    service.activate_revision('analysis', 'back', one.revision_id, expected_generation=2)
    before_branch = service._application.sessions.load('analysis')
    assert before_branch.active_revision_id == one.revision_id
    branch = submit(service, second_binding, turn='branch', generation=3)
    assert branch.status == 'succeeded', branch
    state = service._application.sessions.load('analysis')
    assert state.revisions[-1].parent_revision_id == one.revision_id
    assert state.interactions[0].submission == first_submission
    assert state.interactions[1].submission['registered_input']['resource_id'] == second.resource_id
    assert state.interactions[2].submission['registered_input']['resource_id'] == second.resource_id
    assert [service._application.run_store.load(r.run_id).request.inputs['source_sha256']
            for r in state.revisions] == [first.source_sha256, second.source_sha256, second.source_sha256]
    assert calls == {'production': 3, 'verification': 3} and len(models) == 3


def test_registered_inspection_is_separate_from_matrix_qualification(local_source, tmp_path, monkeypatch):
    source, _ = local_source
    # Generic inspection can read this source even though canonical adoption rejects its order.
    with h5py.File(source, 'r+') as file:
        file['var/_index'][:] = ['chr1:10-20', 'chr1:0-10', 'chr1:30-40']
    service, models = service_for(tmp_path, target='inspect_scATAC')
    calls = count_science(monkeypatch)
    resource = register(service, source)
    binding = service.resources.resolve(resource.resource_id, tool_name='inspect_scATAC')
    service.create_session('analysis')
    view = service.submit_turn('analysis', 'inspect', 'Inspect the supplied H5AD.',
                               expected_generation=0, registered_input=binding)
    assert view.status == 'succeeded' and view.revision_id, view
    run = service._application.run_store.load(view.run_id)
    step, = run.steps
    assert step.tool_name == 'inspect_scATAC' and step.result['n_cells'] == 3
    assert step.verification.artifact_authority is None
    assert calls == {'production': 0, 'verification': 0} and len(models) == 1


def test_inspection_source_drift_cannot_activate_or_recover_revision(local_source, tmp_path):
    source, _ = local_source
    registry = build_default_tool_registry()
    spec = registry.get('inspect_scATAC')
    calls = []

    def changed_after_inspection(**arguments):
        result = spec.function(**arguments)
        calls.append('inspect_scATAC')
        with h5py.File(source, 'r+') as file:
            file['X/data'][0] = 3
        return result

    registry = ToolRegistry(tuple(
        replace(registry.get(name), function=changed_after_inspection)
        if name == 'inspect_scATAC' else registry.get(name)
        for name in registry.names()
    ))
    service, models = service_for(tmp_path, target='inspect_scATAC', registry=registry)
    resource = register(service, source)
    binding = service.resources.resolve(resource.resource_id, tool_name='inspect_scATAC')
    service.create_session('analysis')
    utterance = 'Inspect this registered matrix.'
    view = service.submit_turn('analysis', 'inspect', utterance,
        expected_generation=0, registered_input=binding)
    assert view.status == 'failed' and view.revision_id is None, view
    assert view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    run = service._application.run_store.load(view.run_id)
    assert run.lifecycle_status.value == 'SUCCEEDED'
    state = service._application.sessions.load('analysis')
    assert state.generation == 0 and not state.revisions and state.turn('inspect').status == 'failed'
    assert service.recover_turn('analysis', 'inspect', complete_presentation=True) == view
    assert service.submit_turn('analysis', 'inspect', utterance,
        expected_generation=0, registered_input=binding) == view
    assert calls == ['inspect_scATAC'] and len(models) == 1


def interrupted_inspection_service(source, tmp_path, monkeypatch):
    """Leave successful inspection linked before its application acceptance."""
    registry = build_default_tool_registry()
    specification = registry.get('inspect_scATAC')
    calls = []

    def counted(**arguments):
        calls.append('inspect_scATAC')
        return specification.function(**arguments)

    registry = ToolRegistry(tuple(
        replace(registry.get(name), function=counted)
        if name == 'inspect_scATAC' else registry.get(name)
        for name in registry.names()
    ))
    service, models = service_for(tmp_path, target='inspect_scATAC', registry=registry)
    resource = register(service, source)
    binding = service.resources.resolve(resource.resource_id, tool_name='inspect_scATAC')
    service.create_session('analysis')
    from agent.application.sessions import AnalysisSessions
    class Interrupted(BaseException):
        pass

    def interrupt(*args, **kwargs):
        raise Interrupted()

    utterance = 'Inspect this registered matrix.'
    with monkeypatch.context() as stopped:
        stopped.setattr(AnalysisSessions, '_record_result', interrupt)
        with pytest.raises(Interrupted):
            service.submit_turn('analysis', 'interrupted', utterance,
                expected_generation=0, registered_input=binding)
    state = service._application.sessions.load('analysis')
    assert not state.revisions and state.turn('interrupted').status == 'linked'
    run = service._application.run_store.load(state.turn('interrupted').run_id)
    assert run.lifecycle_status.value == 'SUCCEEDED'
    return service, models, calls, binding


def test_interrupted_inspection_cannot_accept_changed_source_on_recovery(local_source, tmp_path, monkeypatch):
    source, _ = local_source
    service, models, calls, binding = interrupted_inspection_service(source, tmp_path, monkeypatch)
    source.write_bytes(b'Source replaced before registered inspection acceptance.\n')
    view = service.recover_turn('analysis', 'interrupted', complete_presentation=True)
    assert view.status == 'failed' and view.revision_id is None, view
    assert view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    state = service._application.sessions.load('analysis')
    assert state.generation == 0 and not state.revisions
    assert state.turn('interrupted').status == 'failed'
    assert service.recover_turn('analysis', 'interrupted', complete_presentation=True) == view
    assert service.submit_turn('analysis', 'interrupted', 'Inspect this registered matrix.',
        expected_generation=0, registered_input=binding) == view
    assert calls == ['inspect_scATAC'] and len(models) == 1


def test_inspection_integrity_failure_survives_interruption_before_facade_presentation(
        local_source, tmp_path, monkeypatch):
    source, _ = local_source
    service, models, calls, binding = interrupted_inspection_service(source, tmp_path, monkeypatch)
    source.write_bytes(b'Source replaced before registered inspection acceptance.\n')
    from agent.application.sessions import AnalysisSessions
    original = AnalysisSessions._record_result

    class Interrupted(BaseException):
        pass

    def interrupt_after_failure(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ResourceAdmissionError:
            raise Interrupted()

    with monkeypatch.context() as stopped:
        stopped.setattr(AnalysisSessions, '_record_result', interrupt_after_failure)
        with pytest.raises(Interrupted):
            service.recover_turn('analysis', 'interrupted', complete_presentation=True)

    reopened, reopened_models = service_for(tmp_path, target='inspect_scATAC')
    view = reopened.turn('analysis', 'interrupted')
    assert view.status == 'failed' and view.revision_id is None, view
    assert view.error is not None and view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    state = reopened._application.sessions.load('analysis')
    assert state.generation == 0 and not state.revisions
    assert state.turn('interrupted').status == state.interactions[0].status == 'failed'
    assert reopened.recover_turn('analysis', 'interrupted', complete_presentation=True) == view
    assert reopened.submit_turn('analysis', 'interrupted', 'Inspect this registered matrix.',
        expected_generation=0, registered_input=binding) == view
    assert calls == ['inspect_scATAC'] and len(models) == 1 and not reopened_models
    assert reopened._application.sessions.load('analysis') == state


@pytest.mark.parametrize('public_sessions', [False, True])
def test_inspection_drift_during_recovery_composition_is_rejected_at_acceptance(
        local_source, tmp_path, monkeypatch, public_sessions):
    source, _ = local_source
    service, models, calls, _ = interrupted_inspection_service(source, tmp_path, monkeypatch)
    original = service._application._complete
    completions = []

    def changed_during_composition(run_result, paths):
        result = original(run_result, paths)
        completions.append(result.status.value)
        source.write_bytes(b'Source changed during post-run composition.\n')
        return result

    monkeypatch.setattr(service._application, '_complete', changed_during_composition)
    if public_sessions:
        with pytest.raises(ResourceAdmissionError) as error:
            service._application.sessions.complete_presentation('analysis', 'interrupted')
        assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    else:
        view = service.recover_turn('analysis', 'interrupted', complete_presentation=True)
        assert view.status == 'failed' and view.revision_id is None, view
        assert view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    state = service._application.sessions.load('analysis')
    assert state.generation == 0 and not state.revisions
    assert state.turn('interrupted').status == 'failed'
    assert state.interactions[0].status == 'failed'
    assert completions == ['SUCCEEDED'] and calls == ['inspect_scATAC'] and len(models) == 1


def test_prior_failure_presentation_survives_registered_inspection_recovery_failure(
        local_source, tmp_path, monkeypatch):
    source, _ = local_source
    service, models = service_for(tmp_path, target='inspect_scATAC')
    resource = register(service, source)
    binding = service.resources.resolve(resource.resource_id, tool_name='inspect_scATAC')
    service.create_session('analysis')
    from agent.application import service as service_module
    original = service_module.build_analysis_report

    def fail_report(*args, **kwargs):
        raise RuntimeError('Synthetic report-composition failure')

    monkeypatch.setattr(service_module, 'build_analysis_report', fail_report)
    view = service.submit_turn('analysis', 'inspect', 'Inspect this registered matrix.',
        expected_generation=0, registered_input=binding)
    assert view.status == 'finalizing' and view.response.error.code == 'APP_REPORT_FAILED', view
    initial = service._application.sessions.load('analysis')
    assert initial.turn('inspect').status == 'run_succeeded' and not initial.revisions
    shown = initial.interactions[0].presentation
    assert shown is not None
    monkeypatch.setattr(service_module, 'build_analysis_report', original)
    source.write_bytes(b'Source changed after immutable failure presentation.\n')
    with pytest.raises(ResourceAdmissionError) as error:
        service.recover_turn('analysis', 'inspect', complete_presentation=True)
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    state = service._application.sessions.load('analysis')
    assert state.generation == 0 and not state.revisions and state.turn('inspect').status == 'failed'
    assert state.interactions[0].presentation == shown
    assert service.turn('analysis', 'inspect').response == view.response
    assert len(models) == 1


def test_inspection_drift_is_preserved_when_application_composition_also_fails(
        local_source, tmp_path, monkeypatch):
    source, _ = local_source
    service, models = service_for(tmp_path, target='inspect_scATAC')
    resource = register(service, source)
    binding = service.resources.resolve(resource.resource_id, tool_name='inspect_scATAC')
    service.create_session('analysis')
    from agent.application import service as service_module

    def changed_and_failed_report(*args, **kwargs):
        source.write_bytes(b'Changed source during failed report composition.\n')
        raise RuntimeError('Synthetic report-composition failure')

    monkeypatch.setattr(service_module, 'build_analysis_report', changed_and_failed_report)
    view = service.submit_turn('analysis', 'inspect', 'Inspect this registered matrix.',
        expected_generation=0, registered_input=binding)
    assert view.status == 'failed' and view.revision_id is None, view
    assert view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    state = service._application.sessions.load('analysis')
    assert state.generation == 0 and not state.revisions and state.turn('inspect').status == 'failed'
    assert state.interactions[0].status == 'failed'
    assert len(models) == 1


@pytest.mark.parametrize('mutation,code', [
    ('order', 'MATRIX_REFERENCE_MISMATCH'),
    ('negative', 'MATRIX_INTEGER_INVALID'),
    ('float', 'MATRIX_CSR_INVALID'),
    ('schema', 'MATRIX_H5AD_INVALID'),
    ('binary', 'MATRIX_INTEGER_INVALID'),
    ('reference', 'MATRIX_REFERENCE_MISMATCH'),
    ('corrupt', 'TOOL_EXCEPTION'),
    ('missing_reference', 'RESOURCE_NOT_FOUND'),
])
def test_scientific_failure_remains_owned_and_structured(local_source, tmp_path, monkeypatch, mutation, code):
    source, declarations = local_source
    declarations = dict(declarations)
    if mutation == 'corrupt':
        source.write_bytes(b'This completed source is not readable HDF5.\n')
    elif mutation in ('order', 'negative', 'float', 'schema'):
        with h5py.File(source, 'r+') as file:
            if mutation == 'order':
                file['var/_index'][:] = ['chr1:10-20', 'chr1:0-10', 'chr1:30-40']
            elif mutation == 'negative':
                file['X/data'][0] = -1
            elif mutation == 'float':
                values = file['X/data'][:].astype(float)
                del file['X/data']
                file['X'].create_dataset('data', data=values)
            else:
                file.attrs['encoding-type'] = 'unqualified-format'
    elif mutation == 'binary':
        declarations['matrix_semantics'] = 'binary_accessibility'
    elif mutation == 'reference':
        declarations.update(species='mouse', assembly='mm10')
    else:
        declarations['reference_manifest_path'] = str(source.parent / 'missing-reference.json')
    service, models = service_for(tmp_path)
    calls = count_science(monkeypatch)
    resource = register(service, source)
    binding = bind(service, resource, declarations)
    assert not models and calls == {'production': 0, 'verification': 0}
    service.create_session('analysis')
    view = submit(service, binding)
    assert_failed_science(service, view, code)
    assert calls == {'production': 1, 'verification': 0}
    assert submit(service, binding) == view
    assert calls == {'production': 1, 'verification': 0} and len(models) == 1


def test_existing_matrix_size_limit_propagates_without_oversized_fixture(local_source, tmp_path, monkeypatch):
    source, declarations = local_source
    service, _ = service_for(tmp_path)
    resource = register(service, source)
    binding = bind(service, resource, declarations)
    original = production.build
    from agent.tools.data._cell_by_ccre_io import MatrixLimits

    def limited(args, output, **kwargs):
        return original(args, output, limits=MatrixLimits(max_scratch_bytes=1), **kwargs)

    monkeypatch.setattr(production, 'build', limited)
    monkeypatch.setattr(verification, '_verify_external', lambda *a, **k: pytest.fail('Rejected source was verified'))
    service.create_session('analysis')
    assert_failed_science(service, submit(service, binding), 'MATRIX_RESOURCE_LIMIT')


def test_source_drift_after_resolution_and_completed_retry_policy(local_source, tmp_path, monkeypatch):
    source, declarations = local_source
    service, models = service_for(tmp_path)
    calls = count_science(monkeypatch)
    resource = register(service, source)
    binding = bind(service, resource, declarations)
    service.create_session('analysis')
    view = submit(service, binding)
    assert view.status == 'succeeded', view
    # A registered source is hash pinned. Historical accepted results retain their existing policy.
    source.write_bytes(b'Replaced original source.\n')
    assert submit(service, binding) == view
    assert calls == {'production': 1, 'verification': 1} and len(models) == 1
    with pytest.raises(ValueError) as error:
        bind(service, resource, declarations)
    assert getattr(error.value, 'code', None) == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    with pytest.raises((ValueError, InteractiveBoundaryError)) as error:
        submit(service, binding, turn='new', generation=1)
    actual = getattr(error.value, 'error', error.value)
    assert actual.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    assert len(models) == 1 and calls == {'production': 1, 'verification': 1}
    script = '''
import json, sys
from agent.application import InteractiveAgentApplication
from agent.application.local_resources import RegisteredInput, ResourceAdmissionError
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.tools.data import _external_matrix_io, cell_by_ccre_verifier
def forbidden(*args, **kwargs): raise AssertionError('Historical retry repeated model or science')
_external_matrix_io.build = cell_by_ccre_verifier._verify_external = forbidden
profile = PlanningModelProfile('local-scripted', 'scripted', 'scripted/local-resource')
app = InteractiveAgentApplication(sys.argv[1], model_profiles=(profile,), default_profile_id=profile.profile_id,
    planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
state = app._application.sessions.load('analysis')
submission = _serialize(state.interactions[0].submission)
binding = RegisteredInput(**submission['registered_input'], execution_inputs=submission['execution_inputs'])
retry = app.submit_turn('analysis', 'adopt', sys.argv[2], expected_generation=0, registered_input=binding)
try:
    app.resources.resolve(binding.resource_id, tool_name=binding.tool_name,
        scientific_inputs=json.loads(sys.argv[3]))
except ResourceAdmissionError as error:
    assert error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
else:
    raise AssertionError('Changed source resolved as fresh input')
try:
    app.submit_turn('analysis', 'fresh-child', sys.argv[2], expected_generation=1, registered_input=binding)
except ResourceAdmissionError as error:
    assert error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
else:
    raise AssertionError('Changed source consumed by a new turn')
assert app._application.sessions.load('analysis') == state
print(json.dumps(retry.to_dict()))
'''
    child = subprocess.run([sys.executable, '-B', '-c', script,
        str(service._application.workspace_root), UTTERANCE, json.dumps(declarations)],
        env=os.environ.copy(), capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == view.to_dict()


def test_source_drift_during_output_selection_stops_before_science(local_source, tmp_path, monkeypatch):
    source, declarations = local_source
    service, models = service_for(tmp_path, output_callback=lambda: source.write_bytes(b'Changed during planning.\n'))
    calls = count_science(monkeypatch)
    resource = register(service, source)
    binding = bind(service, resource, declarations)
    service.create_session('analysis')
    view = submit(service, binding)
    assert view.status == 'failed' and view.revision_id is None, view
    assert view.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    state = service._application.sessions.load('analysis')
    assert state.generation == 0 and not state.revisions
    assert len(models) == 1 and calls == {'production': 0, 'verification': 0}


def test_registration_binding_cannot_mix_or_forge_existing_inputs(local_source, tmp_path, monkeypatch):
    source, declarations = local_source
    service, models = service_for(tmp_path)
    calls = count_science(monkeypatch)
    resource = register(service, source)
    binding = bind(service, resource, declarations)
    service.create_session('analysis')
    with pytest.raises((ValueError, InteractiveBoundaryError)):
        service.submit_turn('analysis', 'mixed', UTTERANCE, expected_generation=0,
            execution_inputs=dict(binding.execution_inputs), registered_input=binding)
    forged = replace(binding, record_sha256='0' * 64)
    with pytest.raises((ValueError, InteractiveBoundaryError)):
        submit(service, forged, turn='forged')
    assert not models and calls == {'production': 0, 'verification': 0}
    assert not service._application.sessions.load('analysis').interactions


@pytest.mark.parametrize('declaration', ['missing', 'invalid_semantics', 'unsupported_tool'])
def test_missing_or_unsupported_declarations_are_not_guessed(local_source, tmp_path, monkeypatch, declaration):
    source, declarations = local_source
    service, models = service_for(tmp_path)
    calls = count_science(monkeypatch)
    resource = register(service, source)
    values = dict(declarations)
    tool = ADOPT
    if declaration == 'missing':
        del values['matrix_semantics']
    elif declaration == 'invalid_semantics':
        values['matrix_semantics'] = 'normalized_signal'
    else:
        tool = 'invented_scientific_operation'
    if declaration == 'invalid_semantics':
        binding = service.resources.resolve(resource.resource_id, tool_name=tool, scientific_inputs=values)
        service.create_session('analysis')
        view = submit(service, binding)
        assert view.status == 'failed' and view.revision_id is None, view
        run = service._application.run_store.load(view.run_id)
        assert run.errors and view.error.code == run.errors[0].code
        assert view.error.code != 'INTERACTIVE_APPLICATION_FAILED'
        assert service._application.sessions.load('analysis').generation == 0
    else:
        with pytest.raises(ValueError) as error:
            service.resources.resolve(resource.resource_id, tool_name=tool, scientific_inputs=values)
        assert error.value.code == ('LOCAL_RESOURCE_DECLARATION_REQUIRED' if declaration == 'missing'
                                    else 'LOCAL_RESOURCE_OPERATION_UNSUPPORTED')
    assert (len(models) == 1 if declaration == 'invalid_semantics' else not models)
    assert calls == {'production': 0, 'verification': 0}
