"""Joint live-schema/Parser witnesses; no provider or scientific execution."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from agent.application.dialogue_evidence import DetailRequest
from agent.application.raw_handoff import empty
from agent.application.turn_decisions import (
    IntentError, ScalarChoiceArgument, decision_schema, interpret, parse_decision,
)
from test_interpretation_protocol import accepts, context, execute, registry
from test_interpreter_contract import strict_structure
from test_specialized_declaration_targets import proposal, G1Model
from test_preparation_owner_contracts import setup

SAVED = json.loads((Path(__file__).parent/'fixtures/icta_g1_qc_null_interpretation.json').read_text())


def offered(registry, sources=1, groups=0, pending=None):
    public = context(registry, sources)
    public['dialogue']['outputs'] = [dict(handle=f'r{i}', subjects=dict(candidates=[])) for i in range(groups)]
    public['dialogue']['result_groups'] = [dict(handle=f'g{i}', tool='run_scanpy', outputs=[f'r{i}'], canonical_output=f'r{i}') for i in range(groups)]
    if pending:
        public['dialogue']['pending_prerequisite'] = dict(handle='@'+pending, field=pending)
    return public


def question(**changes):
    return dict(kind='answer_scientific', target=dict(output='g0', subject=None), comparison=None,
                focus='question', detail=None, visualization=None, export=None) | changes


def detail(section, field=None, other=None, **changes):
    return dict(section=section, field=field, other_field=other, offset=0, limit=8) | changes


def joint(public, value, *, valid=True, parsed=True, utterance='Analyze these data.', version=1):
    schema = decision_schema(public, utterance=utterance)
    strict_structure(schema)
    wire = dict(turn_schema_version=version, decision=value)
    assert accepts(schema, wire) is valid
    if not parsed:
        with pytest.raises(IntentError):
            parse_decision(json.dumps(wire), utterance=utterance)
        return
    result = parse_decision(json.dumps(wire), utterance=utterance)
    if valid:
        class SavedResponse:
            def complete(self, *, prompt, response_schema):
                assert response_schema == schema
                assert accepts(response_schema, wire)
                return json.dumps(wire)
        assert interpret(SavedResponse(), utterance, public) == result
    return result


def test_exact_preserved_g1_and_explicit_future_delivery(registry):
    assert hashlib.sha256(SAVED['response'].encode()).hexdigest() == SAVED['response_sha256']
    wire = json.loads(SAVED['response'])
    joint(offered(registry), wire['decision'], valid=False, parsed=False, utterance=SAVED['utterance'])
    valid = deepcopy(wire['decision'])
    valid.update(visualization=None, delivery=['explanation', 'visualization'])
    parsed = joint(offered(registry), valid, utterance=SAVED['utterance'])
    assert parsed.bam.source == 'source.0' and parsed.bam.species == wire['decision']['bam']['species']
    assert parsed.delivery == ('explanation', 'visualization')


@pytest.mark.parametrize('sources', [0, 1, 2, 16])
@pytest.mark.parametrize('groups', [0, 1, 2])
@pytest.mark.parametrize('kind', ['qc', 'marker_activity_umap'])
def test_required_visualization_offers(registry, sources, groups, kind):
    visualization = (dict(kind=kind, source='g0' if groups else None) if kind == 'qc' else
                     dict(kind=kind, coordinates='g0' if groups else None, activity='g0' if groups else None))
    joint(offered(registry, sources, groups), execute([], roles=[], visualization=visualization),
          valid=bool(groups), parsed=bool(groups))
    if groups:
        joint(offered(registry, sources, groups), question(visualization=visualization))


def test_planned_new_categorical_operands_remain_available(registry):
    joint(offered(registry, 0), execute([], roles=[], visualization=dict(coordinates='new', labels='new', field='result:cluster')))
    joint(offered(registry), execute([], visualization=dict(coordinates='new', labels='source.0', field='source:cluster')))


@pytest.mark.parametrize('section,field,other,second,valid', [
    ('categorical_fields', None, None, False, True),
    ('categorical_fields', 'result:a', None, False, True),
    ('categorical_fields', None, 'result:b', False, False),
    ('category_counts', None, None, False, False),
    ('category_counts', 'result:a', None, False, True),
    ('category_counts', 'result:a', 'result:b', False, False),
    ('category_counts', 'result:a', 'result:b', True, True),
    ('category_association', 'result:a', None, True, False),
    ('category_association', None, 'result:b', True, False),
    ('category_association', 'result:a', 'result:b', False, True),
    ('category_association', 'result:a', 'result:b', True, True),
])
def test_categorical_detail_relationships(registry, section, field, other, second, valid):
    joint(offered(registry, groups=2), question(detail=detail(section, field, other),
          comparison=dict(output='g1', subject=None) if second else None), valid=valid, parsed=valid)


@pytest.mark.parametrize('section', DetailRequest.CATEGORICAL_SECTIONS)
@pytest.mark.parametrize('field,other', [(None, None), ('result:a', None), ('result:a', 'source:b')])
def test_execution_comparison_uses_paired_fields(registry, section, field, other):
    valid = section != 'categorical_fields' and field is not None and other is not None
    joint(offered(registry, groups=2), execute([], comparison=dict(left='g0', right='g1',
          detail=detail(section, field, other))), valid=valid, parsed=valid)


@pytest.mark.parametrize('n,m', [(0,0), (1,0), (0,1), (32,0), (0,32), (32,1), (32,32)])
def test_export_nonempty_and_deterministic_aggregate_maximum(registry, n, m):
    row = lambda i, field: dict(output='g0', field=field, destination=f'field_{i}')
    export = dict(measurement='g0', observations=[row(i, 'result:cluster') for i in range(n)],
                  representations=[row(i, 'obsm:X_pca') for i in range(m)])
    joint(offered(registry, groups=1), question(export=export), valid=n+m>0, parsed=1<=n+m<=32)


@pytest.mark.parametrize('targets,candidate,valid', [([], None, True), (['g0'], None, True),
    ([], 'offer.0', True), (['g0'], 'offer.0', False)])
def test_guidance_alternatives(registry, targets, candidate, valid):
    public = offered(registry, groups=1)
    public['dialogue']['guidance_predecessor'] = [dict(references=['offer.0'])]
    joint(public, dict(kind='answer_guidance', targets=[dict(output=h, subject=None) for h in targets],
                      candidate=candidate), valid=valid, parsed=valid)


def test_ordinary_role_answer_must_answer_something(registry):
    public = offered(registry, pending='roles')
    value = dict(kind='answer_roles', pending='@roles', roles=[], arguments=[])
    joint(public, value, valid=False, parsed=False)
    joint(public, value | dict(roles=[dict(tool='run_scanpy', port='dataset', source='source.0')]))
    joint(public, value | dict(arguments=[dict(tool='run_scanpy', argument='function_name', native_function='pp.scale',
          value='pp.scale', literal='pp.scale', evidence='pp.scale')]), utterance='Run pp.scale.')


@pytest.mark.parametrize('field', ['bam', 'fastq', 'fragments', 'annotation', 'epizoo'])
def test_specialized_role_answers_retain_declarations(registry, field):
    value = asdict(empty(field))
    if field == 'fastq': value['members'] = []
    value['species' if field != 'epizoo' else 'scientific_name'] = 'human' if field != 'epizoo' else 'Homo sapiens'
    joint(offered(registry, pending='roles'), dict(kind='answer_roles', pending='@roles', roles=[], arguments=[],
          **{field:value}, **({'annotation':None} if field in ('fastq','bam','fragments') else {}), unknown_fields=[]))


@pytest.mark.parametrize('delivery,valid,parsed', [(['files_only'],True,True),
    (['explanation','visualization'],True,True), (['files_only','explanation'],False,False),
    (['explanation','explanation'],True,False), (['files_only','files_only'],False,False)])
def test_delivery_exclusivity_and_deterministic_uniqueness(registry, delivery, valid, parsed):
    joint(offered(registry), execute([], delivery=delivery), valid=valid, parsed=parsed)


@pytest.mark.parametrize('field,key,bound', [('bam','species',128), ('bam','assembly',128),
    ('fastq','species',128), ('fragments','species',128), ('fragments','assembly',128), ('fastq','namespace',64)])
def test_declaration_text_domains(registry, field, key, bound):
    value = proposal(field, None, 'run_chromap' if field=='fastq' else 'run_snapatac2')['decision']
    value[field][key] = 'a' * bound
    joint(offered(registry), value)
    value[field][key] += 'a'
    joint(offered(registry), value, valid=False, parsed=False)


@pytest.mark.parametrize('length', [80,81,256])
def test_offered_field_domain_does_not_relax_literal_or_scalar_choices(registry, length):
    name = 'x' * length
    public = offered(registry)
    public['dialogue']['source_fields'][0]['fields'] = [name]
    arg = dict(tool='annotate_scATAC_cell_types', argument='group_column', source='source.0', mode='offered',
               value=name, literal=name, evidence=name)
    joint(public, execute([arg], target='annotate_scATAC_cell_types'), utterance='Use '+name+'.')
    arg['mode'] = 'literal'
    joint(public, execute([arg], target='annotate_scATAC_cell_types'), utterance='Use '+name+'.',
          valid=length<=80, parsed=length<=80)
    if length>80:
        with pytest.raises(ValueError): ScalarChoiceArgument('tool','argument','x',0,1,name)


@pytest.mark.parametrize('value', [0.5, True, float('inf'), float('-inf'), float('nan')])
def test_protocol_integer_rejects_nonintegers_and_booleans(value):
    with pytest.raises(IntentError): parse_decision(json.dumps(dict(turn_schema_version=value, decision=dict(kind='clarify',reason='ambiguous_subject',choices=[]))))
    for field in ('offset','limit'):
        with pytest.raises(ValueError): DetailRequest('categorical_fields', **{field:value})


def test_integral_protocol_numbers_normalize_and_keep_ranges(registry):
    joint(offered(registry,0), dict(kind='clarify',reason='ambiguous_subject',choices=[]), version=1.0)
    result = joint(offered(registry,groups=1), question(detail=detail('categorical_fields', offset=0.0, limit=32.0)))
    assert type(result.scientific.detail['offset']) is type(result.scientific.detail['limit']) is int
    result = joint(offered(registry,groups=1), execute([], comparison=dict(left='new',right='g0',
        detail=detail('category_counts','result:a','result:b',offset=0.0,limit=8.0))))
    assert type(result.comparison['detail']['limit']) is int
    for offset,limit in [(-1.0,8),(0,0.0),(0,33.0)]:
        joint(offered(registry,groups=1),question(detail=detail('categorical_fields',offset=offset,limit=limit)),valid=False,parsed=False)


@pytest.mark.parametrize('value', [True,False,None])
@pytest.mark.parametrize('answer', [False,True])
def test_annotation_umap_tristate_on_initial_and_pending_declarations(registry,value,answer):
    decision = proposal('annotation',None,'annotate_scATAC_cell_types')['decision']
    decision['annotation']['generate_umap'] = value
    if answer:
        decision=dict(kind='answer_roles', pending='@roles', roles=[], arguments=[], annotation=decision['annotation'], unknown_fields=[])
    result = joint(offered(registry,pending='roles' if answer else None),decision)
    assert result.annotation.generate_umap is value


def test_valid_bam_future_visualization_reaches_candidate_without_science(tmp_path):
    class Model(G1Model):
        def complete(self, *, prompt, response_schema):
            data=json.loads(prompt)
            if 'turn_schema_version' in data:
                self.calls.append(data)
                wire=json.loads(SAVED['response'])
                wire['decision'].update(visualization=None,delivery=['explanation','visualization'])
                assert accepts(response_schema,wire)
                return json.dumps(wire)
            return super().complete(prompt=prompt,response_schema=response_schema)
    model=Model();app,executor=setup(tmp_path,model)
    path=tmp_path/'opaque.attachment';path.write_bytes(b'Offline identity fixture; never decode BAM')
    record=app.resources.register('source',path,input_type='attachment',label='input.bam',
        attribution='Offline contract witness',session_id='session')
    result=app.submit_turn('session','g1',SAVED['utterance'],expected_generation=0,
        registered_input=app.resources.compose_group([record.resource_id],attachment_candidates=True))
    assert result.status=='clarification', result.to_dict()
    state=app._application.sessions.load('session');current=state.interactions[-1]
    assert current.preparation['candidate_id'] and current.utterance==SAVED['utterance']
    assert current.admitted['delivery']==('explanation','visualization')
    assert not executor.plans and not state.turns and not state.revisions


@pytest.mark.parametrize('token', ['1.0000000000000001','0.99999999999999999','1e-400'])
def test_raw_wire_fractional_protocol_numbers_cannot_round_into_integers(token):
    raw = '{"turn_schema_version":'+token+',"decision":{"kind":"clarify","reason":"ambiguous_subject","choices":[]}}'
    with pytest.raises(IntentError): parse_decision(raw)
    wire = dict(turn_schema_version=1,decision=question(detail=detail('categorical_fields')))
    for key in ('offset','limit'):
        raw = json.dumps(wire).replace('"'+key+'": '+str(wire['decision']['detail'][key]), '"'+key+'": '+token)
        with pytest.raises(IntentError): parse_decision(raw)


def test_raw_wire_large_integral_offset_preserves_its_exact_value():
    wire=dict(turn_schema_version=1,decision=question(detail=detail('categorical_fields')))
    raw=json.dumps(wire).replace('"offset": 0','"offset": 9007199254740993.0')
    parsed=parse_decision(raw)
    assert parsed.scientific.detail['offset']==9007199254740993
    assert type(parsed.scientific.detail['offset']) is int


def test_protocol_normalization_does_not_change_scientific_json_or_span_coordinates(registry):
    # A scientific JSON member still belongs to the existing scientific decoder.
    from agent.application.dialogue_execution import _bind_arguments
    literal='{"max_value": 4.0}'
    utterance='Run run_scanpy pp.scale with '+literal+'.'
    argument=dict(tool='run_scanpy',argument='kwargs',native_function='pp.scale',literal=literal,evidence=literal)
    result=joint(offered(registry),execute([argument]),utterance=utterance)
    value=_bind_arguments(registry,utterance,{},result.arguments)['run_scanpy__pp_scale__kwargs']['max_value']
    assert type(value) is float and value==4.0
    from agent.application.turn_decisions import ScopedArgument
    with pytest.raises(ValueError): ScopedArgument('run_scanpy','kwargs','x',0.0,1)


def test_finite_protocol_exponent_and_existing_integer_decoder_bound():
    wire=dict(turn_schema_version=1,decision=question(detail=detail('categorical_fields')))
    raw=json.dumps(wire).replace('"offset": 0','"offset": 1e400')
    assert parse_decision(raw).scientific.detail['offset']==10**400
    for token in ('1e4300','1e99999999999999999999999999999'):
        raw=json.dumps(wire).replace('"offset": 0','"offset": '+token)
        with pytest.raises(IntentError): parse_decision(raw)


@pytest.mark.parametrize('key,value', [
    ('references',['g0','g0']), ('input_results',['g0','g0']), ('unfulfilled',['Optional clause','Optional clause']),
    ('retain_parameters',[dict(result='g0',argument='kwargs')]*2),
    ('roles',[dict(tool='run_scanpy',port='dataset',source='source.0')]*2),
])
def test_collection_duplicates_remain_deterministic_rejections(registry,key,value):
    joint(offered(registry,groups=1),execute([],**{key:value}),valid=True,parsed=False)


@pytest.mark.parametrize('structured',[False,True])
def test_compound_argument_identities_remain_deterministic(registry,structured):
    member=dict(name='max_value',kind='number',literal='4',evidence='max_value=4')
    argument=dict(tool='run_scanpy',argument='kwargs',native_function='pp.scale',members=[member])
    arguments=[argument,deepcopy(argument)]
    if structured:
        argument['members'].append(deepcopy(member));arguments=[argument]
    joint(offered(registry),execute(arguments),valid=True,parsed=False,utterance='Run run_scanpy pp.scale with max_value=4.')


def test_pending_choice_uniqueness_and_exact_grounding_are_still_required(registry):
    public=offered(registry,pending='parameters')
    public['dialogue']['pending_prerequisite'].update(tool='run_scanpy',missing=[dict(tool_name='run_scanpy',argument='kwargs')])
    joint(public,dict(kind='clarify',reason='ambiguous_parameter',choices=['run_scanpy.kwargs']*2),valid=True,parsed=False)
    argument=dict(tool='run_scanpy',argument='kwargs',native_function='pp.scale',
        members=[dict(name='max_value',kind='number',literal='4',evidence='max_value=4')])
    joint(offered(registry),execute([argument]),valid=True,parsed=False,utterance='Run run_scanpy pp.scale with max_value requested.')


def test_source_free_pending_roles_keeps_parameter_answer_but_no_source_association(registry):
    public=offered(registry,sources=0,pending='roles')
    argument=dict(tool='run_scanpy',argument='function_name',native_function='pp.scale',
        value='pp.scale',literal='pp.scale',evidence='pp.scale')
    joint(public,dict(kind='answer_roles',pending='@roles',roles=[],arguments=[argument]),utterance='Run pp.scale.')


@pytest.mark.parametrize('field',['bam','fastq','fragments'])
@pytest.mark.parametrize('value',[True,False,None])
def test_downstream_annotation_exposes_the_same_umap_declaration(registry,field,value):
    decision=proposal(field,None,'run_chromap' if field=='fastq' else 'run_snapatac2')['decision']
    decision['annotation']=asdict(empty('annotation'))|dict(generate_umap=value)
    assert joint(offered(registry),decision).annotation.generate_umap is value
