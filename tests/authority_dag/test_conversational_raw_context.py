"""Ordinary conversational planning reuses qualified raw scientific outputs.

The interpreter and Planner are scripted semantic actors. Scientific production,
verification, authority, compilation and durable Session updates are real.
The bounded synthetic thresholds below are explicit acceptance declarations.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.prior_output import PriorOutputRef

from test_application import application_chain
from test_prior_outputs import source, all_work
from test_propagation import chain


PROFILE = PlanningModelProfile('raw-context-scripted', 'scripted', 'scripted/raw-context')
SELECT = 'Using the accepted QC, select cells with depth at least 1 and TSS enrichment at least 0.'
MATRIX = SELECT + ' Also rebuild the canonical matrix.'
MISSING = 'Select candidate cells from the accepted QC.'
ANSWER = 'Require depth at least 1 and TSS enrichment at least 0.'


def declarations(utterance):
    result = []
    for name, literal in (('min_qc_fragment_records', '1'), ('min_tss_enrichment', '0')):
        start = utterance.index(literal)
        result.append(dict(tool='select_scATAC_cells', argument=name, literal=literal,
                           start=start, end=start + len(literal)))
    return result


class RawContextModel:
    model_id = PROFILE.model_id

    def __init__(self, calls, *, matrix=False, handle_override=None, decision=None, race=None):
        self.calls, self.matrix = calls, matrix
        self.handle_override, self.decision, self.race = handle_override, decision, race

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append((value, response_schema, prompt))
        if 'turn_schema_version' in value:
            utterance = value['utterance']
            decision = self.decision
            if decision is None:
                if utterance == ANSWER:
                    decision = dict(kind='answer_parameters', pending='@parameters', arguments=declarations(utterance))
                else:
                    decision = dict(kind='execute_plan', context='current',
                        target='build_scATAC_cell_by_ccre' if self.matrix else 'select_scATAC_cells')
                    if utterance != MISSING:
                        decision['arguments'] = declarations(utterance)
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        if 'selection_schema_version' in response_schema.get('properties', {}):
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=['raw_preprocessing'])))
        if 'output_selection_schema_version' in value:
            if self.race is not None:
                self.race()
                self.race = None
            outputs = [dict(name='selection', step_id='new_selection', output_key='manifest_path')]
            if self.matrix:
                outputs.append(dict(name='matrix', step_id='new_matrix', output_key='manifest_path'))
            retained = value.get('retained_outputs', [])
            retain = [item['handle'] for item in retained if
                item['producer_tool'] == 'compute_scATAC_qc' or item['source_port'] == 'fragments']
            assert len(retain) == 2, 'The raw base must explicitly offer its captured fragments and QC.'
            return json.dumps(dict(outputs=outputs, retain=retain))
        offered = value.get('active_outputs', [])
        candidates = [item for item in offered if item['producer_tool'] == 'compute_scATAC_qc']
        assert len(candidates) == 1 or (not candidates and self.handle_override), (
            'The ordinary Planner did not receive its exact captured QC authority.')
        qc = self.handle_override or candidates[0]['handle']
        steps = [dict(step_id='new_selection', tool='select_scATAC_cells',
            sources=[dict(target='barcode_qc', source=dict(kind='context', handle=qc))], control_dependencies=[])]
        if self.matrix:
            fragments, = [item for item in offered if item['source_port'] == 'fragments']
            steps.append(dict(step_id='new_matrix', tool='build_scATAC_cell_by_ccre', sources=[
                dict(target='fragments', source=dict(kind='context', handle=fragments['handle'])),
                dict(target='selected_cells', source=dict(kind='step', step='new_selection'))], control_dependencies=[]))
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


def interactive(source, **options):
    _, _, old = source
    calls, models = [], []
    def factory(profile):
        assert profile == PROFILE
        model = RawContextModel(calls, **options)
        models.append(model)
        return model
    service = InteractiveAgentApplication(old.workspace_root, model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    return service, calls, models


def science(counts):
    return {name: value for name, value in counts.items() if name.startswith(('production.', 'owner.'))}


def source_files(source):
    _, _, app = source
    run = app.run_store.load('authority-request:run')
    names = {entry['path'] for step in run.steps for entry in step.verification.artifact_authority['files']}
    return {Path(name): Path(name).read_bytes() for name in names}


@pytest.mark.parametrize('matrix', [False, True])
def test_ordinary_interactive_raw_context_uses_real_owners_and_exact_history(source, matrix):
    _, args, old = source
    service, calls, models = interactive(source, matrix=matrix)
    initial = old.sessions.load('analysis')
    original = source_files(source)
    inputs = {} if not matrix else dict(reference_manifest_path=args['reference_bundle_path'],
                                       reference_manifest_sha256=args['reference_bundle_sha256'])
    utterance = MATRIX if matrix else SELECT
    with all_work() as counts:
        view = service.submit_turn('analysis', 'ordinary', utterance, expected_generation=1,
                                   execution_inputs=inputs)
    assert view.response.status == 'activated', view.to_dict()
    assert view.response.kind == 'execute'
    expected = {'production.selection': 1, 'owner.selection': 1}
    if matrix:
        expected.update({'production.matrix': 1, 'owner.matrix': 1})
    assert science(counts) == expected
    assert counts['integrity'] > 0
    state = service._application.sessions.load('analysis')
    run = service._application.run_store.load(view.run_id)
    selection = run.steps[0]
    refs = [value for value in run.plan.steps[0].arguments.values() if isinstance(value, PriorOutputRef)]
    assert len(refs) == 2 and refs[0].binding == refs[1].binding
    assert refs[0].binding.run_id == 'authority-request:run' and refs[0].binding.step_id == 'qc'
    assert selection.resolved_arguments['min_qc_fragment_records'] == 1
    assert selection.resolved_arguments['min_tss_enrichment'] == 0
    assert [dict(item) for item in state.interactions[-1].admitted['arguments']] == declarations(utterance)
    assert selection.verification.artifact_authority['schema_version'] == 2
    assert len(state.revisions) == 2 and state.generation == 2
    assert state.revisions[0] == initial.revisions[0]
    assert state.revisions[-1].parent_revision_id == initial.active_revision_id
    assert {item.name for item in state.revisions[-1].outputs} == (
        {'fragments', 'qc', 'selection', 'matrix'} if matrix else {'fragments', 'qc', 'selection'})
    assert {item.name for item in state.turn('ordinary').retained_outputs} == {'fragments', 'qc'}
    for retained in state.turn('ordinary').retained_outputs:
        assert retained in initial.revisions[0].outputs
    if matrix:
        ref = run.plan.steps[1].arguments['fragments_manifest_path']
        assert isinstance(ref, PriorOutputRef) and ref.binding.step_id == 'fragments'
        assert run.plan.steps[1].arguments['selected_cells_manifest_path'].step_id == 'new_selection'
    for payload, _, prompt in calls:
        assert str(old.workspace_root) not in prompt
        assert 'authority-request:run' not in prompt
        if 'active_outputs' in payload:
            assert all(set(item) == {'handle', 'semantic_type', 'producer_tool', 'source_port', 'verification_mode'}
                       for item in payload['active_outputs'])
    assert len(models) == 1
    before_calls = len(calls)
    with all_work() as retried:
        assert service.submit_turn('analysis', 'ordinary', utterance, expected_generation=1,
                                   execution_inputs=inputs) == view
        assert service.reopen_session('analysis').generation == 2
    assert not science(retried) and len(calls) == before_calls and len(models) == 1
    code = '''
import json, sys
from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
def forbidden(*args, **kwargs): raise AssertionError('Unexpected provider or scientific work')
profile=PlanningModelProfile('raw-context-scripted','scripted','scripted/raw-context')
app=InteractiveAgentApplication(sys.argv[1],model_profiles=(profile,),default_profile_id=profile.profile_id,
    planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':forbidden}))
app._application.run=app._application.resume=app._application._complete=forbidden
state=app.reopen_session('analysis')
assert state.generation==2 and len(state.revisions)==2
retry=app.submit_turn('analysis','ordinary',sys.argv[2],expected_generation=1,
    execution_inputs=json.loads(sys.argv[3]))
assert retry.revision_id==state.active_revision_id
print(json.dumps(retry.to_dict()))
'''
    restored = subprocess.run([sys.executable, '-B', '-c', code, str(old.workspace_root),
        utterance, json.dumps(inputs)], env=os.environ.copy(), capture_output=True, text=True)
    assert restored.returncode == 0, restored.stderr
    assert json.loads(restored.stdout) == view.to_dict()
    assert all(path.read_bytes() == content for path, content in original.items())
    witness = dict(kind=source[0], matrix=matrix, counts=dict(counts),
        captured_base_revision_id=initial.active_revision_id, captured_generation=1,
        source_bindings=[reference.binding.to_dict() for reference in refs],
        new_run_id=run.run_id, new_revision_id=state.active_revision_id,
        retained_outputs=[{name: getattr(item, name) for name in item.__dataclass_fields__}
                          for item in state.turn('ordinary').retained_outputs],
        production_tools=[step.tool_name for step in run.steps])
    (old.workspace_root.parent / 'ordinary-raw-context-witness.json').write_text(
        json.dumps(witness, indent=2, sort_keys=True) + '\n')
    print('ordinary raw counts:', json.dumps(dict(kind=source[0], matrix=matrix, counts=dict(counts)), sort_keys=True))


def test_ordinary_raw_missing_thresholds_continue_the_exact_pending_objective(source):
    _, _, old = source
    service, calls, _ = interactive(source)
    with all_work() as pending:
        first = service.submit_turn('analysis', 'missing', MISSING, expected_generation=1)
    assert first.response.kind == 'clarify' and first.response.clarification['reason'] == 'missing_parameter_value'
    assert set(first.response.clarification['choices']) == {'min_qc_fragment_records', 'min_tss_enrichment'}
    assert not science(pending) and first.revision_id is None
    before = old.sessions.load('analysis')
    assert before.generation == 1 and len(before.revisions) == 1
    with all_work() as completed:
        view = service.submit_turn('analysis', 'answer', ANSWER, expected_generation=1)
    assert view.response.status == 'activated', view.to_dict()
    assert science(completed) == {'production.selection': 1, 'owner.selection': 1}
    planning = [value for value, _, _ in calls if 'active_outputs' in value]
    assert planning[-1]['user_request'] == MISSING
    state = old.sessions.load('analysis')
    assert state.interactions[-1].admitted['continuation']['origin_turn_id'] == 'missing'
    assert state.interactions[-1].base_revision_id == before.active_revision_id


def test_unknown_raw_historical_handle_fails_without_science_or_new_revision(source):
    _, _, old = source
    service, _, _ = interactive(source, handle_override='ctx.999')
    before = old.sessions.load('analysis')
    with all_work() as counts:
        view = service.submit_turn('analysis', 'forged', SELECT, expected_generation=1)
    assert view.response.status == 'failed'
    assert not science(counts)
    after = old.sessions.load('analysis')
    assert after.revisions == before.revisions and after.generation == before.generation


def test_default_context_none_does_not_grant_legacy_decisions_historical_inputs(source):
    _, _, old = source
    decision = dict(kind='execute_plan', target='select_scATAC_cells', arguments=declarations(SELECT))
    service, calls, _ = interactive(source, decision=decision, handle_override='ctx.0')
    before = old.sessions.load('analysis')
    with all_work() as counts:
        view = service.submit_turn('analysis', 'no-context', SELECT, expected_generation=1)
    assert view.response.status == 'failed'
    assert not science(counts)
    after = old.sessions.load('analysis')
    assert after.revisions == before.revisions and after.generation == before.generation
    assert not any(payload.get('active_outputs') for payload, _, _ in calls)


def test_raw_pending_answer_cannot_follow_generation_navigation(source):
    _, _, old = source
    service, _, _ = interactive(source)
    first = service.submit_turn('analysis', 'pending', MISSING, expected_generation=1)
    assert first.response.kind == 'clarify'
    old.sessions.switch('analysis', 'navigate-current', old.sessions.load('analysis').active_revision_id,
                        expected_generation=1)
    with all_work() as counts:
        view = service.submit_turn('analysis', 'stale-answer', ANSWER, expected_generation=2)
    assert view.response.kind == 'clarify' and view.response.status != 'activated'
    assert not science(counts)
    assert old.sessions.load('analysis').generation == 2
