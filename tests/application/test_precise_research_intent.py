"""Precise downstream requests are typed intent, never scientific authority."""
from dataclasses import asdict, replace
import json

import pytest

from agent.application.raw_handoff import empty
from agent.application.turn_decisions import (
    Execute, IntentError, RequestRoles, RetainedParameter, ScientificQuestion, ScientificTarget,
    decision_schema, parse_decision,
)
from agent.schemas.orchestration import _serialize


def parse(value):
    return parse_decision(json.dumps(dict(turn_schema_version=1, decision=value)))


def comparison(section='category_association'):
    return dict(left='new', right='g3', detail=None if section is None else dict(
        section=section, field='result:leiden', other_field='result:leiden', offset=0, limit=8))


@pytest.mark.parametrize('section', [None, 'category_counts', 'category_association'])
def test_exact_comparison_implies_delivery_without_changing_operands(section):
    selected = comparison(section)
    decision = parse(dict(kind='execute_plan', target='cluster_cells', comparison=selected))
    assert isinstance(decision, Execute)
    assert _serialize(decision.comparison) == selected
    assert decision.delivery == ('comparison',)
    assert decision.references == ()  # Capture binds exact references at its existing owner.
    with pytest.raises(TypeError):
        decision.comparison['left'] = 'g8'


@pytest.mark.parametrize('operand', ['new', 'g3'])
@pytest.mark.parametrize('section', ['category_counts', 'category_association'])
def test_one_result_can_compare_distinct_source_and_computed_fields(operand, section):
    selected = comparison(section)
    selected.update(left=operand, right=operand)
    selected['detail']['other_field'] = 'source:cell_type'
    decision = parse(dict(kind='execute_plan', target='cluster_cells', comparison=selected))
    assert _serialize(decision.comparison) == selected
    assert decision.delivery == ('comparison',)


def test_missing_file_request_retains_explicit_accepted_parameter_reference():
    value = dict(kind='request_roles', target='pca_embed_cells', roles=[],
        required_roles=[dict(tool='pca_embed_cells', port='dataset')],
        retain_parameters=[dict(result='g3', argument='n_components')])
    assert parse(value).retain_parameters == (RetainedParameter('g3', 'n_components'),)
    with pytest.raises(IntentError, match='invalid_decision'):
        parse(value | dict(retain_parameters=value['retain_parameters'] * 2))


@pytest.mark.parametrize('field,target', [
    ('fastq', 'prepare_scATAC_fragments'), ('bam', 'prepare_scATAC_bam_fragments'),
    ('fragments', 'import_scATAC_fragments'), ('annotation', 'annotate_scATAC_cell_types'),
    ('epizoo', 'epizoo_embed_cells'), (None, 'inspect_scATAC'),
])
def test_role_prerequisite_retains_the_same_precise_research_intent(field, target):
    visual = dict(coordinates='new', labels='source.0', field='source:cell_type')
    value = dict(kind='request_roles', target=target, roles=[], required_roles=[],
        comparison=comparison(), visualization=visual, references=['g3'],
        delivery=['explanation'], unfulfilled=['separately unavailable outcome'])
    if field is None:
        value['required_roles'] = [dict(tool=target, port='dataset')]
    else:
        declared = empty(field)
        if field == 'epizoo':
            declared = replace(declared, operation='inference')
        value[field] = asdict(declared)
    decision = parse(value)
    assert isinstance(decision, RequestRoles)
    assert _serialize(decision.comparison) == comparison()
    assert _serialize(decision.visualization) == visual
    assert decision.delivery == ('explanation', 'comparison', 'visualization')
    assert decision.references == ('g3',)
    assert decision.unfulfilled == ('separately unavailable outcome',)


@pytest.mark.parametrize('change', [
    dict(left='g3'), dict(right='new'), dict(left='source.0'), dict(left='@current_result'),
    dict(left='step.1'), dict(right=True), dict(extra='selector'), dict(detail={}),
    dict(detail=dict(section='categorical_fields', field=None, other_field=None, offset=0, limit=8)),
    dict(detail=dict(section='category_counts', field='result:leiden', other_field=None, offset=0, limit=8)),
    dict(detail=dict(section='category_association', field='leiden', other_field='result:leiden', offset=0, limit=8)),
    dict(detail=dict(section='category_counts', field='result:leiden', other_field='source:', offset=0, limit=8)),
    dict(detail=dict(section='category_counts', field='result:leiden', other_field='result:leiden', offset=True, limit=8)),
    dict(detail=dict(section='category_counts', field='result:leiden', other_field='result:leiden', offset=0, limit=33)),
])
def test_comparison_rejects_unselected_ambiguous_and_unbounded_operands(change):
    with pytest.raises(IntentError, match='invalid_decision'):
        parse(dict(kind='execute_plan', target='cluster_cells', comparison=comparison() | change))


@pytest.mark.parametrize('coordinates,labels,field', [
    ('new', 'source.0', 'source:cell_type'), ('g2', 'new', 'result:leiden'),
    ('g2', 'g3', 'result:leiden'), ('new', 'new', 'result:leiden'),
])
def test_precise_figure_distinguishes_accepted_source_and_planned_operands(coordinates, labels, field):
    visual = dict(coordinates=coordinates, labels=labels, field=field)
    decision = parse(dict(kind='execute_plan', target='compute_cell_umap', visualization=visual))
    assert _serialize(decision.visualization) == visual
    assert decision.delivery == ('visualization',)


@pytest.mark.parametrize('change', [
    dict(coordinates='source.0'), dict(coordinates='@most_recently_created'),
    dict(labels='source.16'), dict(labels='source.0', field='result:cell_type'),
    dict(field='cell_type'), dict(field='source:'), dict(field='/obs/cell_type'), dict(extra=True),
])
def test_precise_figure_rejects_ambiguous_and_wrong_role_operands(change):
    visual = dict(coordinates='new', labels='source.0', field='source:cell_type') | change
    with pytest.raises(IntentError, match='invalid_decision'):
        parse(dict(kind='execute_plan', target='compute_cell_umap', visualization=visual))


@pytest.mark.parametrize('labels,field', [('g3', 'result:leiden'), ('source.0', 'source:cell_type')])
def test_read_only_figure_request_keeps_accepted_operands_without_fictitious_science(labels, field):
    visual = dict(coordinates='g2', labels=labels, field=field)
    value = dict(kind='answer_scientific', target=dict(output='g2', subject=None),
        comparison=None, focus='question', visualization=visual)
    decision = parse(value)
    assert decision.scientific == ScientificQuestion(ScientificTarget('g2'), visualization=visual)
    for change in (dict(coordinates='new'), dict(labels='new')):
        with pytest.raises(IntentError, match='invalid_decision'):
            parse(value | dict(visualization=visual | change))


def test_historical_coarse_decision_does_not_invent_precise_operands():
    decision = parse(dict(kind='execute_plan', target='compute_cell_umap',
        delivery=['comparison', 'visualization']))
    assert decision.comparison is decision.visualization is None
    assert decision.delivery == ('comparison', 'visualization')
    assert parse(dict(kind='answer_scientific', target=dict(output='g2', subject=None),
        comparison=None, focus='question')).scientific.visualization is None


def test_model_schema_exposes_exact_requests_on_execution_read_and_role_capture():
    public = dict(bases={}, relations=[], dialogue=dict(outputs=[],
        tools=['cluster_cells', 'compute_cell_umap', 'inspect_scATAC'],
        result_groups=[dict(handle='g2', tool='compute_cell_umap'),
                       dict(handle='g3', tool='cluster_cells', parameters=[dict(argument='resolution')])],
        selected_sources=[dict(source='source.0')], input_roles=[dict(tool='inspect_scATAC', port='dataset')],
        attachment_candidates=True, fastq_resources=[]))
    variants = decision_schema(public)['properties']['decision']['anyOf']
    execution, = [v['properties'] for v in variants if v['properties']['kind']['enum'] == ['execute_plan']]
    drawing = execution['visualization']['anyOf'][0]['properties']
    assert drawing['coordinates']['enum'] == ['new', 'g2']
    assert drawing['labels']['enum'] == ['new', 'g2', 'g3', 'source.0']
    assert execution['comparison']['anyOf'][0]['properties']['left']['enum'] == ['new', 'g2', 'g3']
    for variant in variants:
        properties = variant['properties']
        if properties['kind']['enum'] == ['request_roles']:
            assert {'delivery', 'unfulfilled', 'visualization', 'comparison', 'references', 'retain_parameters'} <= set(properties)
    scientific, = [v['properties'] for v in variants if v['properties']['kind']['enum'] == ['answer_scientific']]
    accepted = scientific['visualization']['anyOf'][0]['properties']
    assert accepted['coordinates']['enum'] == ['g2']
    assert 'new' not in accepted['labels']['enum']


def test_first_turn_schema_captures_new_coordinates_and_explicit_source_coloring():
    public = dict(bases={}, relations=[], dialogue=dict(outputs=[], tools=['compute_cell_umap'],
        selected_sources=[dict(source='source.0')]))
    variants = decision_schema(public)['properties']['decision']['anyOf']
    execution, = [v['properties'] for v in variants if v['properties']['kind']['enum'] == ['execute_plan']]
    compared = execution['comparison']['anyOf'][0]['properties']
    assert compared['left']['enum'] == compared['right']['enum'] == ['new']
    drawing = execution['visualization']['anyOf'][0]['properties']
    assert drawing['coordinates']['enum'] == ['new']
    assert drawing['labels']['enum'] == ['new', 'source.0']
    scientific, = [v['properties'] for v in variants if v['properties']['kind']['enum'] == ['answer_scientific']]
    assert 'visualization' not in scientific
