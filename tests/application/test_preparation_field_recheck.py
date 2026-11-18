"""Exact field answers survive fresh task rechecks without a new source identity.

Regression for the genuine final-integration A/E witness: the grouping Answer
is complete before a missing gene reference is supplied by an explicit recheck.
Scientific execution is withheld at the ordinary shared Executor boundary.
"""
import json

import pytest

from agent.application import ResearchAgentApplication
from test_preparation_source_contracts import app_fixture, source_file


def pending_gene_reference(tmp_path, monkeypatch):
    app, model, executor, source, refs = app_fixture(tmp_path, monkeypatch)
    complete = model.complete
    def intent(**kwargs):
        value = json.loads(kwargs['prompt'])
        if 'turn_schema_version' in value:
            model.calls.append(value)
            return json.dumps(dict(turn_schema_version=1, decision=dict(
                kind='execute_plan', target='annotate_scATAC_cell_types',
                roles=[dict(tool='annotate_scATAC_cell_types', port='source', source='source.0')],
                arguments=[dict(tool='annotate_scATAC_cell_types', argument='species', literal='Human', evidence=value['utterance'])])))
        return complete(**kwargs)
    model.complete = intent
    first = app.submit_turn('session', 'task', 'Annotate Human existing groups.', expected_generation=0,
        registered_input=app.resources.compose_group([source.resource_id], attachment_candidates=True))
    field = next(c for c in first.clarification_choices if c.get('value') == 'other_groups')
    answered = app.submit_turn('session', 'field', '', expected_generation=0,
        clarification_choice=dict(question_turn_id='task', handle=field['handle']))
    assert answered.status == 'clarification', answered.to_dict()
    state = app._application.sessions.load('session')
    assert [r['argument'] for r in state.interactions[-1].admitted['preparation_result']['requirements']] == ['gene_anno_path']
    assert not executor.plans and not state.turns
    return app, model, executor, source, refs, state


@pytest.mark.parametrize('attachment', ['none', 'unused'])
def test_field_answer_recheck_ready_and_exact_authorization(tmp_path, monkeypatch, attachment):
    app, model, executor, source, refs, before = pending_gene_reference(tmp_path, monkeypatch)
    supplied = {}
    if attachment == 'unused':
        extra = tmp_path/'extra';extra.mkdir()
        path = source_file(extra)
        added = app.resources.register('unused', path, input_type='h5ad', label='Unused matrix',
            session_id='session', attribution='Different offered identity')
        supplied['attachment_ids'] = [added.resource_id]
    calls = len(model.calls)
    checked = app.submit_turn('session', 'recheck', '', expected_generation=0,
        resume_task_id=before.interactions[-1].preparation['task_id'], input_sets=refs, **supplied)
    assert checked.status == 'answered', checked.to_dict()
    assert checked.preparation_recheck
    state = app._application.sessions.load('session')
    assert state.interactions[-1].admitted['preparation_result']['status'] == 'READY'
    assert state.interactions[:len(before.interactions)] == before.interactions
    assert len({i.preparation['candidate_id'] for i in state.interactions}) == 1
    assert len({i.preparation['task_id'] for i in state.interactions}) == 1
    assert not executor.plans and not state.turns and len(model.calls) == calls
    members = state.interactions[-1].snapshot['dialogue']['source_view']['registered_input']['members']
    assert len(members) == (0 if attachment == 'none' else 1)
    if members:
        assert members[0]['resource_id'] == added.resource_id
        assert members[0]['resource_id'] != source.resource_id
    choice = next(c for c in checked.clarification_choices if c.get('action') == 'execute_preparation')
    done = app.submit_turn('session', 'authorize', '', expected_generation=0,
        clarification_choice=dict(question_turn_id='recheck', handle=choice['handle']))
    assert done.error.code == 'TEST_SCIENCE_WITHHELD', done.to_dict()
    args = executor.plans[0].steps[0].arguments
    assert args['input_path'] == source.source_path and args['group_column'] == 'other_groups'
    assert args['gene_anno_path'] == refs[0].execution_inputs['gene_anno_path']
    assert sum('semantic_prompt_version' in c for c in model.calls) == 1
    assert sum('turn_schema_version' in c for c in model.calls) == 1
    assert ResearchAgentApplication(app._application.workspace_root).sessions.load('session') == app._application.sessions.load('session')


def test_field_answer_recheck_rejects_changed_original_bytes(tmp_path, monkeypatch):
    app, model, executor, source, refs, before = pending_gene_reference(tmp_path, monkeypatch)
    from pathlib import Path
    path = Path(source.source_path)
    path.write_bytes(path.read_bytes() + b'changed-original-source')
    calls = len(model.calls)
    checked = app.submit_turn('session', 'recheck', '', expected_generation=0,
        resume_task_id=before.interactions[-1].preparation['task_id'], input_sets=refs)
    assert checked.error.code == 'PREPARATION_SOURCE_CHANGED', checked.to_dict()
    state = app._application.sessions.load('session')
    assert state.interactions[-1].admitted['preparation_result']['status'] == 'INVALID'
    assert state.interactions[:len(before.interactions)] == before.interactions
    assert not state.turns and not executor.plans and len(model.calls) == calls


@pytest.mark.parametrize('mismatch', ['source', 'consumer', 'axis'])
def test_captured_field_offer_stays_bound_to_current_consumer_operand(tmp_path, monkeypatch, mismatch):
    from dataclasses import replace
    from agent.application.scientific_preparation import source_field_offers
    from agent.schemas.orchestration import _serialize
    app, model, executor, source, refs, before = pending_gene_reference(tmp_path, monkeypatch)
    question = before.interactions[0]
    answer = before.interactions[-1].preparation['answers'][0]['value']
    registry = app._application.registry
    tool = registry.get('annotate_scATAC_cell_types')
    arguments = dict(input_path=source.source_path)
    if mismatch == 'source':
        other = tmp_path/'other';other.mkdir()
        arguments['input_path'] = str(source_file(other))
    elif mismatch == 'consumer':
        tool = registry.get('run_scanpy')
    else:
        spec = tool.optional_arguments['group_column']
        tool = replace(tool, optional_arguments=dict(tool.optional_arguments,
            group_column=replace(spec, planning=replace(spec.planning, observation_axis='var'))))
    offers = source_field_offers(registry, question, before.interactions, tool,
        'group_column', arguments, {})
    assert _serialize(answer) not in offers
    assert not executor.plans
