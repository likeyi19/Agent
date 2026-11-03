"""Stable proposals, complete Registry discovery and deterministic admission.

These are offline contract witnesses, not claims about provider understanding.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import re

import pytest
from agent.application.turn_decisions import (
    IntentError, decision_schema, interpret, parse_decision,
    _interpretation_public,
)
from agent.application.dialogue_execution import (
    scientific_parameters, scientific_capabilities, _bind_arguments, _check_parameter_plan,
)
from agent.application.local_resources import _GROUP_PORTS
from agent.orchestration import build_default_tool_registry, AgentPlan, PlanStep


@pytest.fixture(scope='module')
def registry():
    return build_default_tool_registry()


def context(registry, sources=1):
    return dict(bases={}, relations=['current'], dialogue=dict(outputs=[],
        tools=list(registry.names()), scientific_parameters=scientific_parameters(registry),
        capabilities=scientific_capabilities(registry),
        input_roles=[dict(tool=t, port=p) for t, p in _GROUP_PORTS],
        selected_sources=[dict(source=f'source.{i}', input_type='h5ad') for i in range(sources)],
        source_fields=[dict(source=f'source.{i}', fields=[f'field_{i}_{j}' for j in range(64)], complete=True)
                       for i in range(sources)]))


def accepts(schema, value, root=None):
    """Independent interpreter of the JSON Schema subset used by this protocol."""
    root = schema if root is None else root
    if '$ref' in schema:
        return accepts(root['$defs'][schema['$ref'].split('/')[-1]], value, root)
    if 'anyOf' in schema:
        return any(accepts(s, value, root) for s in schema['anyOf'])
    kind = schema['type']
    if kind == 'object':
        return (type(value) is dict and set(value) == set(schema['required'])
                and all(accepts(s, value[k], root) for k, s in schema['properties'].items()))
    if kind == 'array':
        return (type(value) is list and schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', float('inf'))
                and all(accepts(schema['items'], v, root) for v in value))
    if kind == 'null': return value is None
    types = dict(string=str, integer=int, number=(int, float), boolean=bool)
    if kind == 'number': valid = type(value) in types[kind]
    else: valid = type(value) is types[kind]
    if not valid or 'enum' in schema and value not in schema['enum']: return False
    if kind == 'string':
        return (schema.get('minLength', 0) <= len(value) <= schema.get('maxLength', float('inf'))
                and ('pattern' not in schema or re.search(schema['pattern'], value) is not None))
    return schema.get('minimum', -float('inf')) <= value <= schema.get('maximum', float('inf'))


def execute(arguments, **changes):
    return dict(kind='execute_plan', target='run_scanpy', arguments=arguments, context='none',
        roles=[dict(tool='run_scanpy', port='dataset', source='source.0')], references=[], delivery=[],
        unfulfilled=[], visualization=None, comparison=None, retain_parameters=[], missing_parameters=[]) | changes


def interpret_value(value, utterance, public):
    class Model:
        calls = 0
        def complete(self, *, prompt, response_schema):
            self.calls += 1
            assert response_schema == decision_schema()
            return json.dumps(dict(turn_schema_version=1, decision=value))
    model = Model()
    result = interpret(model, utterance, public)
    assert model.calls == 1
    return result


def test_one_structurally_stable_schema_and_complete_inventory(registry):
    schema = decision_schema()
    frozen = json.dumps(schema, sort_keys=True)
    for count in (0, 1, 2, 8, 16):
        public = context(registry, count)
        before = deepcopy(public)
        projection = _interpretation_public(public, 'Explore the supplied data.')['dialogue']
        capabilities = projection['capabilities']
        assert set(capabilities) == set(registry.names())
        assert len(capabilities) == 43
        assert sum(len(c.get('functions', {})) for c in capabilities.values()) == 69
        assert sum(len(c['parameters']) for c in capabilities.values()) == 164
        for row in scientific_parameters(registry):
            actual=dict(capabilities[row['tool']]['parameters'][row['argument']])
            expected={k:v for k,v in row.items() if k not in ('tool','argument')}
            if 'member_names' in actual:
                assert set(actual.pop('member_names')) == set(expected['members'])
                assert actual.pop('members') == {}
                expected.pop('members')
            assert actual == expected
        assert not {'tools','scientific_parameters','input_roles','parameter_catalog'} & projection.keys()
        assert public == before
        assert json.dumps(decision_schema(), sort_keys=True) == frozen
    def visit(node):
        if isinstance(node, dict):
            if node.get('type') == 'object':
                assert set(node['required']) == set(node['properties']) and node['additionalProperties'] is False
            for value in node.values(): visit(value)
        elif isinstance(node, list):
            for value in node: visit(value)
    visit(schema)
    # Inventory choices and arbitrary source-field growth never enter the schema.
    for word in ('run_scanpy', 'pp.pca', 'source.0', 'field_0_0', 'n_comps'):
        assert word not in frozen
    assert len(frozen) < 40000


def test_explicit_api_detail_is_exact_without_filtering_discovery(registry):
    public = context(registry)
    for utterance, expected in [
        ('Run scanpy.pp.pca with n_comps=10.', {'run_scanpy': {'pp.pca'}}),
        ('Run scanpy.pp.pca then scanpy.pp.neighbors.', {'run_scanpy': {'pp.pca', 'pp.neighbors'}}),
        ('Run chromVAR::getBackgroundPeaks.', {'run_chromvar': {'getBackgroundPeaks'}}),
    ]:
        capabilities = _interpretation_public(public, utterance)['dialogue']['capabilities']
        assert set(capabilities) == set(registry.names())
        for tool, metadata in capabilities.items():
            if not metadata.get('functions'): continue
            wanted = {k for fn in expected.get(tool, ()) for k in metadata['functions'][fn]}
            kwargs = metadata['parameters']['kwargs']
            assert set(kwargs['members']) == wanted
            assert set(kwargs['member_names']) == set(dict(registry.get(tool).optional_arguments['kwargs'].planning.conversational_members))
    # A package-only request and unsupported function never guess a scientific route.
    c = _interpretation_public(public, 'Use Scanpy to investigate these cells.')['dialogue']['capabilities']
    assert c['run_scanpy']['parameters']['kwargs']['members']
    assert 'member_names' not in c['run_scanpy']['parameters']['kwargs']
    c = _interpretation_public(public, 'Run scanpy.pp.invented.')['dialogue']['capabilities']
    assert 'pp.invented' not in c['run_scanpy']['functions']
    assert c['run_scanpy']['parameters']['kwargs']['members'] == {}


@pytest.mark.parametrize('literal,evidence,utterance', [
    ('10', 'n_comps=10', 'Run scanpy.pp.pca with n_comps=10.'),
    ('10', '10 principal components', 'Run scanpy.pp.pca with 10 principal components.'),
])
def test_complete_protocol_native_declaration_is_grounded_and_bound(registry, literal, evidence, utterance):
    arguments = [dict(tool='run_scanpy', argument='function_name', native_function='pp.pca',
                     literal='pp.pca', evidence='scanpy.pp.pca', value='pp.pca'),
                 dict(tool='run_scanpy', argument='kwargs', native_function='pp.pca',
                      members=[dict(name='n_comps', kind='number', literal=literal, evidence=evidence)])]
    value = execute(arguments)
    assert accepts(decision_schema(), dict(turn_schema_version=1, decision=value))
    decision = interpret_value(value, utterance, context(registry))
    values = _bind_arguments(registry, utterance, {}, decision.arguments)
    assert values['run_scanpy__pp_pca__kwargs'] == {'n_comps': 10}
    plan = AgentPlan('p','r',utterance,(PlanStep('pca','run_scanpy',dict(function_name='pp.pca',kwargs={'n_comps':10})),))
    _check_parameter_plan(registry, plan, dict(inputs=values, arguments=[asdict(d) for d in decision.arguments]), ())


@pytest.mark.parametrize('change', ['tool','argument','function','port','source','member','value','function_member'])
def test_model_strings_never_authorize_invented_or_crossed_identities(registry, tmp_path, change):
    utterance = 'Run scanpy.pp.pca with n_comps=10.'
    argument = dict(tool='run_scanpy',argument='kwargs',native_function='pp.pca',
        members=[dict(name='n_comps',kind='number',literal='10',evidence='n_comps=10')])
    value = execute([argument])
    if change == 'tool': value['target']='invented'
    elif change == 'argument': argument['argument']='invented'
    elif change == 'function': argument['native_function']='tl.invented'
    elif change == 'port': value['roles'][0]['port']='invented'
    elif change == 'source': value['roles'][0]['source']='source.9'
    elif change == 'member': argument['members'][0]['name']='invented'
    elif change == 'value': argument['members'][0]['literal']='11'
    else: argument['members'][0]['name']='n_neighbors'
    import anndata as ad
    import numpy as np
    from agent.application import InteractiveAgentApplication
    from agent.application.local_resources import LocalResourceAdmission
    from agent.providers import PlanningModelFactoryRegistry
    from agent.orchestration import PlanningModelProfile
    source=tmp_path/'source.h5ad'
    ad.AnnData(np.ones((8,8))).write_h5ad(source)
    calls=[]
    class Model:
        model_id='scripted/proposal-rejection'
        def complete(self, *, prompt, response_schema):
            calls.append(json.loads(prompt))
            assert 'turn_schema_version' in calls[-1], 'Invalid proposal reached the Planner'
            return json.dumps(dict(turn_schema_version=1,decision=value))
    profile=PlanningModelProfile('test','scripted','scripted/proposal-rejection')
    root=tmp_path/'workspace'
    app=InteractiveAgentApplication(root,model_profiles=(profile,),default_profile_id='test',approved_source_roots=(tmp_path,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda p:Model()}))
    app.create_session('negative')
    local=LocalResourceAdmission(root,approved_source_roots=(tmp_path,))
    record=local.register('data',source,input_type='h5ad',label='data',attribution='simulated')
    group=local.compose_group([record.resource_id],attachment_candidates=True)
    result=app.submit_turn('negative','first',utterance,expected_generation=0,registered_input=group)
    state=app._application.sessions.load('negative')
    assert not state.turns and not state.revisions and len(calls)==1, result.to_dict()
    assert result.status in ('failed','clarification')


@pytest.mark.parametrize('source,mode,value,valid', [
    ('source.0','offered','field_0_0',True), ('source.1','offered','field_0_0',False),
    ('source.0','offered','invented',False), ('source.0','literal','field_0_0',True),
    ('source.9','literal','field_0_0',False), ('source.0','literal','different',False),
])
def test_observation_fields_retain_exact_captured_source(registry, source, mode, value, valid):
    utterance='Evaluate against field_0_0.'
    raw=dict(tool='evaluate_cell_clustering',argument='label_key',source=source,mode=mode,value=value,
             literal='field_0_0',evidence='field_0_0')
    public=context(registry,2)
    decision=parse_decision(json.dumps(dict(turn_schema_version=1,decision=execute([raw],target='evaluate_cell_clustering',roles=[]))),utterance=utterance)
    from types import SimpleNamespace
    from agent.application.local_resources import RegisteredInputGroup
    from agent.application.dialogue_execution import _source_field_roles, _argument_fields
    members=[dict(resource_id='local-'+str(i)*64,record_sha256=str(i+2)*64,source_sha256=str(i+4)*64,
                  source_path='/captured/'+str(i)+'.h5ad',input_type='h5ad',label=str(i)) for i in range(2)]
    binding=RegisteredInputGroup(members,{},attachment_candidates=True)
    rows=[row | dict(resource_id=members[i]['resource_id'],source_sha256=members[i]['source_sha256'])
          for i,row in enumerate(public['dialogue']['source_fields'])]
    interaction=SimpleNamespace(snapshot={'dialogue':{'source_fields':rows}},admitted=None)
    def bind():
        roles=_source_field_roles(registry,(),decision.arguments)
        fields=_argument_fields(registry,interaction,binding,roles,declarations=decision.arguments)
        return _bind_arguments(registry,utterance,{},decision.arguments,field_choices=fields)
    if valid: assert bind()['label_key']=='field_0_0'
    else:
        with pytest.raises(ValueError): bind()


def test_missing_field_projection_does_not_authorize_normalized_observation(registry):
    raw=dict(tool='evaluate_cell_clustering',argument='label_key',literal='labels',evidence='labels',
             source='source.0', mode='offered', value='invented')
    utterance='Evaluate against labels.'
    decision=interpret_value(execute([raw],target='evaluate_cell_clustering',roles=[]),utterance,context(registry))
    with pytest.raises(IntentError):
        _bind_arguments(registry,utterance,{},decision.arguments,field_choices={})


def test_retired_initialization_parameters_stay_readable_but_cannot_be_new_intent(registry):
    from agent.application.turn_decisions import ScopedArgument
    declaration=ScopedArgument('adapt_epizoo_species','max_steps','10',0,2)
    with pytest.raises(IntentError):_bind_arguments(registry,'10',{},(declaration,))
    assert _bind_arguments(registry,'10',{},(declaration,),historical=True)['max_steps']==10
