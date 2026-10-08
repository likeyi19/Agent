"""Registered FASTQs use the qualified scientific owners through ordinary turns.

Semantic decisions are scripted. The guarded lifecycle tests run the actual
qualified Chromap, packaging tools, source scans and independent verifier.
"""
from dataclasses import replace
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agent.application import InteractiveAgentApplication
from agent.application.local_resources import ResourceAdmissionError
from agent.orchestration import PlanningModelProfile, ToolRegistry, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.tools.data import _chromap as chromap
from agent.tools.data import chromap_reference_index, fastq_fragment_manifest
from agent.tools.data import fastq_fragments, fastq_fragments_verifier
from agent.tools.data import raw_scatac, raw_scatac_manifest, scatac_library_context as library
from agent.tools.data import scatac_fragments_v2

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / 'chromap'))
from chromap.conftest import BC, executables, reads, tiny
from fragments_helpers import backend


PROFILE = PlanningModelProfile('fastq-scripted', 'scripted', 'scripted/registered-fastq')
INSPECT = 'inspect_raw_scATAC'
PREPARE = 'prepare_scATAC_fragments'
UTTERANCE = 'Execute the declared operation on this registered FASTQ collection using the supplied scientific inputs.'


class Model:
    """The existing interpreter, capability selection and planner interfaces."""
    model_id = PROFILE.model_id

    def __init__(self, target):
        self.target = target
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
            return json.dumps(dict(outputs=[dict(name='intake' if self.target == INSPECT else 'fragments',
                step_id=step_id, output_key='manifest_path')]))
        ports = ([('raw_input', 'raw_input_paths')] if self.target == INSPECT else
                 [('intake', 'intake_manifest_path'), ('library_context', 'library_context_path'),
                  ('reference', 'reference_bundle_path')])
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[dict(
            step_id=step_id, tool=self.target, control_dependencies=[],
            sources=[dict(target=port, source=dict(kind='input', input=name)) for port, name in ports])])))


def application(root, *, registry=None):
    models = []
    def factory(profile):
        assert profile == PROFILE
        model = Model(INSPECT if not models else PREPARE)
        models.append(model)
        return model
    app = InteractiveAgentApplication(root / 'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, approved_source_roots=(root,), registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    return app, models


def register(app, groups):
    records = []
    for group in groups:
        for role, path in group.files:
            records.append(app.resources.register(path, path, input_type='fastq',
                label=f'{Path(path).parent.name} {role.value}',
                attribution='Explicit operator-selected synthetic FASTQ source.'))
    return records


def members(records):
    return [dict(resource_id=record.resource_id, record_sha256=record.record_sha256) for record in records]


def declarations(layout):
    return dict(species='human', raw_assay='TENX_ATAC', fastq_layout=layout.value)


def producer_inputs(tiny, intake, selected_ids, whitelist, selection_mode):
    """Explicitly provision prerequisites through their existing owners."""
    context = library.build_scatac_library_processing_context(
        intake_manifest_path=intake['manifest_path'], expected_intake_sha256=intake['manifest_sha256'],
        libraries=(library.LibraryDeclaration(namespace='explicit_library', group_ids=selected_ids,
            membership_basis=(library.MembershipBasis.SINGLE_GROUP if len(selected_ids) == 1
                              else library.MembershipBasis.CALLER_SHARED_LIBRARY),
            barcode_interpretation=library.BarcodeInterpretation.RAW_SEQUENCE,
            correction_policy=library.CorrectionPolicy.WHITELIST_REQUIRED, whitelist=whitelist),),
        selection_mode=selection_mode)
    pointer = library.publish_scatac_library_processing_context(context, tiny['root'] / 'context.json')
    return dict(intake_manifest_path=intake['manifest_path'], intake_manifest_sha256=intake['manifest_sha256'],
        library_context_path=pointer['manifest_path'], library_context_sha256=pointer['manifest_sha256'],
        reference_bundle_path=tiny['pointer']['manifest_path'],
        reference_bundle_sha256=tiny['pointer']['manifest_sha256'])


def instrument(monkeypatch):
    calls = dict(intake=0, production=0, verification=0, alignment=0)
    registry = build_default_tool_registry()
    original_inspect = registry.get(INSPECT).function
    original_prepare = fastq_fragments.prepare_fastq_fragments
    original_verify = fastq_fragments_verifier.verify_stream
    original_stage = fastq_fragments.run_stage
    def inspect(**arguments):
        calls['intake'] += 1
        return original_inspect(**arguments)
    def prepare(**arguments):
        calls['production'] += 1
        return original_prepare(**arguments)
    def verify(*args, **kwargs):
        calls['verification'] += 1
        return original_verify(*args, **kwargs)
    def stage(argv, **kwargs):
        if '--preset' in argv:
            calls['alignment'] += 1
        return original_stage(argv, **kwargs)
    monkeypatch.setattr(fastq_fragments, 'prepare_fastq_fragments', prepare)
    monkeypatch.setattr(fastq_fragments_verifier, 'verify_stream', verify)
    monkeypatch.setattr(fastq_fragments, 'run_stage', stage)
    return ToolRegistry(tuple(replace(registry.get(name), function=inspect) if name == INSPECT
                              else registry.get(name) for name in registry.names())), calls


def step_for(app, view):
    step, = app._application.run_store.load(view.run_id).steps
    return step


def assert_attribution(app, view, binding):
    state = app._application.sessions.load('session')
    interaction = next(item for item in state.interactions if item.turn_id == view.turn_id)
    assert _serialize(interaction.submission['registered_input']) == binding.attribution()
    assert _serialize(interaction.submission['execution_inputs']) == _serialize(binding.execution_inputs)
    assert str(app._application.workspace_root.parent) not in json.dumps(view.to_dict())


@pytest.mark.parametrize('selection_mode,layout', [
    (library.SelectionMode.ALL_GROUPS, chromap.FastqLayout.A),
    (library.SelectionMode.EXPLICIT_SUBSET, chromap.FastqLayout.A),
    (library.SelectionMode.EXPLICIT_SUBSET, chromap.FastqLayout.B),
])
def test_registered_fastq_qualified_session_lifecycle(tiny, reads, executables, monkeypatch, selection_mode, layout):
    first, whitelist = reads([(100, 200, BC, 3)], name='lane_one', layout=layout)
    second, _ = reads([(1100, 1200, BC, 2)], name='lane_two', layout=layout)
    registry, calls = instrument(monkeypatch)
    app, models = application(tiny['root'], registry=registry)
    records = register(app, [first, second])
    inspection_binding = app.resources.resolve_collection(members(records), tool_name=INSPECT,
        scientific_inputs=declarations(layout))
    assert set(inspection_binding.execution_inputs['raw_input_paths']) == {record.source_path for record in records}
    assert not models and calls == dict(intake=0, production=0, verification=0, alignment=0)
    assert not list(app._application.sessions._store.root.glob('*.json'))
    app.create_session('session')
    inspected = app.submit_turn('session', 'inspect', UTTERANCE, expected_generation=0,
        registered_input=inspection_binding)
    assert inspected.status == 'succeeded', inspected
    intake_step = step_for(app, inspected)
    intake = intake_step.result
    assert intake['readiness'] == 'READY' and intake['n_files'] == 8 and intake['n_groups'] == 2
    assert intake_step.verification.passed and intake_step.verification.artifact_authority is None
    _, manifest, _ = raw_scatac_manifest.load_raw_intake_manifest(intake['manifest_path'],
        expected_sha256=intake['manifest_sha256'])
    assert {source.path for source in manifest.files} == {record.source_path for record in records}
    assert_attribution(app, inspected, inspection_binding)
    selected_groups = manifest.groups
    if selection_mode is library.SelectionMode.EXPLICIT_SUBSET:
        file_ids = {source.id for source in manifest.files if source.path in {path for _, path in first.files}}
        selected_groups = tuple(group for group in manifest.groups if set(group.file_ids) == file_ids)
        assert len(selected_groups) == 1
    selected_ids = tuple(group.id for group in selected_groups)
    scientific_inputs = producer_inputs(tiny, intake, selected_ids, whitelist, selection_mode)
    executable = executables['candidate']['path']
    indexes = tiny['root'] / 'indexes'
    indexes.mkdir()
    chromap_reference_index.prepare_chromap_reference_index(
        reference_manifest_path=tiny['pointer']['manifest_path'],
        expected_reference_sha256=tiny['pointer']['manifest_sha256'], executable=executable,
        backend=backend(), output_dir=tiny['root'] / 'prepared-index')
    # The operator index catalog contains only admitted index directories.
    # Preparation's mechanical lock remains outside that catalog.
    (tiny['root'] / 'prepared-index').rename(indexes / 'accepted')
    monkeypatch.setenv('AGENT_CHROMAP_BIN', executable)
    monkeypatch.setenv('AGENT_CHROMAP_INDEX_ROOT', str(indexes))
    producer_binding = app.resources.resolve_collection(members(records), tool_name=PREPARE,
        scientific_inputs=scientific_inputs)
    assert producer_binding.collection_sha256 == inspection_binding.collection_sha256
    assert set(producer_binding.execution_inputs) == set(scientific_inputs)
    prepared = app.submit_turn('session', 'prepare', UTTERANCE, expected_generation=1,
        registered_input=producer_binding)
    assert prepared.status == 'succeeded', prepared
    step = step_for(app, prepared)
    assert step.verification.passed
    expected_support = 5 if selection_mode is library.SelectionMode.ALL_GROUPS else 3
    expected_records = 2 if selection_mode is library.SelectionMode.ALL_GROUPS else 1
    assert step.result['total_support'] == expected_support
    assert step.result['n_fragment_records'] == expected_records and step.result['n_libraries'] == 1
    assert step.result['artifact_schema_version'] == 2 and step.result['contract_version'] == 'scatac-fragments.v2'
    authority = step.verification.artifact_authority
    assert authority['schema_version'] == 2 and authority['verifier']['id'] == 'agent.fastq-fragments-independent'
    assert authority['producer_qualification']['scope'] == 'fastq_source_and_producer_record.v1'
    assert authority['source_policy'] == 'historical_verified_sources.v1'
    fragments = fastq_fragment_manifest.load_manifest(step.result['manifest_path'],
        expected_sha256=step.result['manifest_sha256'])
    provenance = fragments['libraries'][0]['provenance']
    assert provenance['source_selection'] == 'producer_subset'
    actual_sources = [item['resource'] for item in provenance['sources'] if item['role'] == 'fastq']
    selected_file_ids = {file_id for group in selected_groups for file_id in group.file_ids}
    expected_paths = {source.path for source in manifest.files if source.id in selected_file_ids}
    assert {source['path'] for source in actual_sources} == expected_paths
    registered_sources = {record.source_path: dict(path=record.source_path,
        sha256=record.source_sha256, size_bytes=record.size_bytes) for record in records}
    assert all(source == registered_sources[source['path']] for source in actual_sources)
    assert all(source in authority['historical_sources'] for source in actual_sources)
    assert any('_I1_' in source['path'] for source in actual_sources)
    if selection_mode is library.SelectionMode.EXPLICIT_SUBSET:
        assert len(actual_sources) == 4 < len(records)
    assert_attribution(app, prepared, producer_binding)
    assert app.evidence('session', prepared.revision_id, 'fragments').status == 'available'
    state = app._application.sessions.load('session')
    assert len(state.revisions) == 2 and state.revisions[-1].parent_revision_id == inspected.revision_id
    assert state.active_revision_id == prepared.revision_id
    assert app.submit_turn('session', 'prepare', UTTERANCE, expected_generation=1,
        registered_input=producer_binding) == prepared
    assert calls == dict(intake=1, production=1, verification=1, alignment=1)
    assert len(models) == 2 and all(len(model.calls) == 4 for model in models)
    submission = state.interactions[-1].submission
    # These are deliberately forged application result fixtures, not newly
    # verified scientific artifacts. Their canonical v2 shape and pinned digest
    # pass the loader, so rejection must come from registered byte association.
    for field in ('sha256', 'size_bytes', 'path'):
        forged = deepcopy(fragments)
        sources = forged['libraries'][0]['provenance']['sources']
        resource = next(item['resource'] for item in sources if item['role'] == 'fastq')
        resource[field] = ('0' * 64 if field == 'sha256' else resource[field] + 1
                           if field == 'size_bytes' else str(tiny['root'] / 'unregistered.fastq'))
        sources.sort(key=lambda item: (item['role'], item['resource']['sha256'], item['resource']['path']))
        forged['fragments_identity_sha256'] = scatac_fragments_v2.fragments_identity(forged)
        payload = scatac_fragments_v2.canonical_fragments_manifest_v2_bytes(forged)
        path = tiny['root'] / f'forged-{field}.json'
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        assert fastq_fragment_manifest.load_manifest(path, expected_sha256=digest) == forged
        forged_step = replace(step, result=dict(_serialize(step.result),
            manifest_path=str(path), manifest_sha256=digest))
        with pytest.raises(ResourceAdmissionError) as rejected:
            app.resources.validate_result(submission, (forged_step,))
        assert rejected.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'
    # A genuine verified result must still match registered source bytes after
    # its provenance has been read, including changes during that final read.
    source = Path(records[0].source_path)
    original_bytes = source.read_bytes()
    original_load = fastq_fragment_manifest.load_manifest
    def changed_after_load(*args, **kwargs):
        manifest = original_load(*args, **kwargs)
        source.write_bytes(original_bytes + b'changed during provenance loading\n')
        return manifest
    try:
        with monkeypatch.context() as changed:
            changed.setattr(fastq_fragment_manifest, 'load_manifest', changed_after_load)
            with pytest.raises(ResourceAdmissionError) as rejected:
                app.resources.validate_result(submission, (step,))
            assert rejected.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    finally:
        source.write_bytes(original_bytes)
    assert app._application.sessions.load('session') == state
    assert calls == dict(intake=1, production=1, verification=1, alignment=1)
    unrelated, _ = reads([(2100, 2200, BC, 1)], name='unregistered', layout=layout)
    unrelated_intake = raw_scatac.inspect_raw_scATAC([path for _, path in unrelated.files],
        tiny['root'] / 'unrelated-intake', **declarations(layout))
    forged_arguments = dict(_serialize(step.resolved_arguments),
        intake_manifest_path=unrelated_intake['manifest_path'],
        intake_manifest_sha256=unrelated_intake['manifest_sha256'])
    with pytest.raises(ResourceAdmissionError) as rejected:
        app.resources.validate_result(submission, (replace(step, resolved_arguments=forged_arguments),))
    assert rejected.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'
    assert app._application.sessions.load('session') == state
    assert calls == dict(intake=1, production=1, verification=1, alignment=1)
    # Historical presentation and exact retries must not consume original reads.
    for record in records:
        Path(record.source_path).unlink()
    script = '''
import json, sys
from agent.application import InteractiveAgentApplication
from agent.application.local_resources import RegisteredInputCollection
from agent.orchestration import PlanningModelProfile, raw_scatac_verifier
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.tools.data import raw_scatac, _raw_fastq, fastq_fragments, fastq_fragments_verifier, _chromap
def forbidden(*args, **kwargs): raise AssertionError('Accepted lifecycle replayed providers or science')
raw_scatac._reconstruct = raw_scatac_verifier._reconstruct = _raw_fastq.inspect_fastq_inputs = forbidden
fastq_fragments.prepare_fastq_fragments = fastq_fragments.run_stage = forbidden
fastq_fragments_verifier.verify_stream = _chromap.identify_backend = forbidden
profile = PlanningModelProfile('fastq-scripted', 'scripted', 'scripted/registered-fastq')
app = InteractiveAgentApplication(sys.argv[1], model_profiles=(profile,), default_profile_id=profile.profile_id,
    planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
initial = app._application.sessions.load('session')
assert len(initial.revisions) == 2
submission = _serialize(initial.interactions[1].submission)
binding = RegisteredInputCollection(**submission['registered_input'], execution_inputs=submission['execution_inputs'])
view = app.recover_turn('session', 'prepare', complete_presentation=True)
assert view.status == 'succeeded' and app.reopen_session('session').active_revision_id == view.revision_id
assert app.revision('session', view.revision_id).revision_id == view.revision_id
assert app.evidence('session', view.revision_id, 'fragments').status == 'available'
assert app.submit_turn('session', 'prepare', sys.argv[2], expected_generation=1, registered_input=binding) == view
assert app._application.sessions.load('session') == initial
print(json.dumps(view.to_dict()))
'''
    child_env = {key: value for key, value in os.environ.items()
                 if key not in {'AGENT_CHROMAP_BIN', 'AGENT_CHROMAP_INDEX_ROOT'}}
    child = subprocess.run([sys.executable, '-B', '-c', script,
        str(app._application.workspace_root), UTTERANCE], env=child_env,
        capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == prepared.to_dict()
    assert calls == dict(intake=1, production=1, verification=1, alignment=1)


@pytest.mark.parametrize('mismatch', ['unrelated', 'missing_member', 'extra_member', 'intake_digest', 'source_drift'])
def test_registered_fastq_producer_requires_exact_intake_membership(tiny, reads, monkeypatch, mismatch):
    first, whitelist = reads([(100, 200, BC, 1)], name='first')
    second, _ = reads([(1100, 1200, BC, 1)], name='second')
    intake_groups = [second] if mismatch == 'unrelated' else [first]
    intake = raw_scatac.inspect_raw_scATAC([path for group in intake_groups for _, path in group.files],
        tiny['root'] / 'intake-output', **declarations(first.layout))
    _, manifest, _ = raw_scatac_manifest.load_raw_intake_manifest(intake['manifest_path'],
        expected_sha256=intake['manifest_sha256'])
    scientific_inputs = producer_inputs(tiny, intake, tuple(group.id for group in manifest.groups),
        whitelist, library.SelectionMode.ALL_GROUPS)
    registry, calls = instrument(monkeypatch)
    app, models = application(tiny['root'], registry=registry)
    records = register(app, [first, second] if mismatch == 'extra_member' else [first])
    if mismatch == 'missing_member':
        records.pop()
    elif mismatch == 'intake_digest':
        scientific_inputs['intake_manifest_sha256'] = '0' * 64
    elif mismatch == 'source_drift':
        Path(records[0].source_path).write_bytes(b'changed after registration')
    with pytest.raises(ResourceAdmissionError) as rejected:
        app.resources.resolve_collection(members(records), tool_name=PREPARE,
            scientific_inputs=scientific_inputs)
    assert rejected.value.code == ('LOCAL_RESOURCE_INTEGRITY_INVALID' if mismatch == 'source_drift'
                                   else 'LOCAL_RESOURCE_BINDING_INVALID')
    assert not models and calls == dict(intake=0, production=0, verification=0, alignment=0)
    assert not list(app._application.sessions._store.root.glob('*.json'))
