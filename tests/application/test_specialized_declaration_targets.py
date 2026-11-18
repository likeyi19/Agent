"""Saved ICTA failure and legal entry targets, without providers or native science."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from agent.application.raw_handoff import empty, schema as declaration_schema
from agent.application.turn_decisions import IntentError, decision_schema, parse_decision
from agent.schemas.orchestration import _serialize
from agent.orchestration import build_default_tool_registry
from test_interpretation_protocol import accepts, context
from test_preparation_owner_contracts import Model, setup

SAVED = json.loads((Path(__file__).parent/'fixtures/icta_g1_failed_interpretation.json').read_text())
CASES = [('bam', None, 'run_snapatac2'), ('fragments', None, 'run_snapatac2'),
         ('fastq', None, 'run_chromap'), ('annotation', None, 'annotate_scATAC_cell_types'),
         ('epizoo', 'inference', 'epizoo_embed_cells'),
         ('epizoo', 'initialization', 'adapt_epizoo_species'),
         ('epizoo', 'posttraining_initialized', 'posttrain_epizoo_model')]


def proposal(field, operation, target):
    declared={k:v for k,v in asdict(empty(field)).items() if k in declaration_schema(field)['properties']}
    if field=='fastq': declared['members']=[]
    if operation is not None: declared['operation']=operation
    decision=dict(kind='request_roles',target=target,roles=[],required_roles=[],arguments=[],
        input_results=[],references=[],delivery=['explanation'],unfulfilled=[],visualization=None,
        comparison=None,retain_parameters=[],**{field:declared})
    if field in ('bam','fragments','fastq'): decision['annotation']=None
    if field=='annotation': decision['context']='none'
    return dict(turn_schema_version=1,decision=decision)


@pytest.fixture(scope='module')
def registry():return build_default_tool_registry()


@pytest.mark.parametrize('field,operation,target',CASES)
@pytest.mark.parametrize('live',[False,True])
def test_schema_and_parser_agree_on_all_entry_target_pairs(registry,field,operation,target,live):
    public=context(registry) if live else None
    schema=decision_schema(public,utterance='Analyze the supplied data.')
    valid=proposal(field,operation,target)
    assert accepts(schema,valid)
    assert parse_decision(json.dumps(valid)).target==target
    # Every other registered target must fail both contracts, including real
    # scientific capabilities which are legal in ordinary composed plans.
    for other in registry.names():
        if other==target:continue
        invalid=deepcopy(valid);invalid['decision']['target']=other
        assert not accepts(schema,invalid),(field,operation,other)
        with pytest.raises(IntentError):parse_decision(json.dumps(invalid))


def test_incomplete_epizoo_operation_is_not_an_initial_task(registry):
    value=proposal('epizoo',None,'epizoo_embed_cells')
    assert not accepts(decision_schema(context(registry)),value)
    with pytest.raises(IntentError):parse_decision(json.dumps(value))
    # Partial metadata remains legal for an actual pending continuation.
    answer=dict(turn_schema_version=1,decision=dict(kind='answer_roles',pending='@roles',roles=[],
        arguments=[],epizoo=asdict(empty('epizoo')),unknown_fields=[]))
    assert parse_decision(json.dumps(answer)).epizoo.operation is None


@pytest.mark.parametrize('field',['bam','fastq','fragments'])
def test_downstream_annotation_does_not_replace_raw_entry_target(registry,field):
    expected='run_chromap' if field=='fastq' else 'run_snapatac2'
    value=proposal(field,None,expected)
    value['decision']['annotation']=proposal('annotation',None,'annotate_scATAC_cell_types')['decision']['annotation']|dict(species='human')
    assert accepts(decision_schema(context(registry)),value)
    assert parse_decision(json.dumps(value)).target==expected
    value['decision']['target']='annotate_scATAC_cell_types'
    assert not accepts(decision_schema(context(registry)),value)
    with pytest.raises(IntentError):parse_decision(json.dumps(value))


def test_owner_target_unavailable_in_registry_is_not_offered(registry):
    public=context(registry);public['dialogue']['tools'].remove('run_snapatac2')
    for field in ('bam','fragments'):
        assert not accepts(decision_schema(public),proposal(field,None,'run_snapatac2'))


def test_saved_failure_is_rejected_and_legal_bam_wire_preserves_metadata(registry):
    assert hashlib.sha256(SAVED['response'].encode()).hexdigest()==SAVED['response_sha256']
    actual=json.loads(SAVED['response']);schema=decision_schema(context(registry),utterance=SAVED['utterance'])
    assert not accepts(schema,actual)
    with pytest.raises(IntentError):parse_decision(SAVED['response'],utterance=SAVED['utterance'])
    # A separate valid fixture, never an automatic production correction.
    valid=deepcopy(actual);valid['decision']['target']='run_snapatac2'
    assert accepts(schema,valid)
    parsed=parse_decision(json.dumps(valid),utterance=SAVED['utterance'])
    assert parsed.bam.source=='source.0' and parsed.bam.species=='human'
    assert parsed.bam.assay=='SCATAC' and parsed.bam.assembly is None


def test_retired_raw_capability_and_implementation_are_absent(registry):
    import agent.tools as tools
    import agent.tools.data as data
    assert 'inspect_raw_scATAC' not in registry.names()
    assert not hasattr(tools,'inspect_raw_scATAC') and not hasattr(data,'RawScATACInspection')
    for name in ('raw_scatac.py','raw_scatac_manifest.py','_raw_bam.py'):
        assert not (Path(data.__file__).parent/name).exists()
    assert {'run_snapatac2','run_chromap','inspect_scATAC','annotate_scATAC_cell_types'}<=set(registry.names())
    from agent.tools.data import _raw_fastq
    assert not hasattr(_raw_fastq,'inspect_fastq_inputs') and not hasattr(_raw_fastq,'observe_fastq_sources')


class G1Model(Model):
    def complete(self,*,prompt,response_schema):
        value=json.loads(prompt)
        if 'turn_schema_version' in value:
            self.calls.append(value)
            valid=json.loads(SAVED['response']);valid['decision']['target']='run_snapatac2'
            assert accepts(response_schema,valid)
            return json.dumps(valid,ensure_ascii=False)
        if 'semantic_prompt_version' in value:
            self.calls.append(value)
            def step(name,function,parent=None):
                source=dict(kind='input',input='input_path') if parent is None else dict(kind='step_port',step=parent,source_port='fragments' if parent=='extract' else 'dataset')
                return dict(step_id=name,tool='run_snapatac2',sources=[dict(target='dataset',source=source),
                    dict(target='function_name',source=dict(kind='registered_choice',choice=function))] + ([dict(target='input_sha256',source=dict(kind='input',input='input_sha256'))] if name=='extract' else []),control_dependencies=[])
            steps=[step('extract','pp.make_fragment_file'),step('import','pp.import_fragments','extract'),
                   step('tiles','pp.add_tile_matrix','import'),dict(step_id='explore',tool='inspect_scATAC',
                   sources=[dict(target='dataset',source=dict(kind='step_port',step='tiles',source_port='dataset'))],control_dependencies=[])]
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=steps)))
        return super().complete(prompt=prompt,response_schema=response_schema)


def test_complete_g1_objective_and_exact_source_reach_candidate_preparation(tmp_path):
    model=G1Model();app,executor=setup(tmp_path,model)
    source=tmp_path/'opaque.attachment';source.write_bytes(b'Offline source identity fixture; never decode as BAM')
    record=app.resources.register('source',source,input_type='attachment',label='input.bam',
        attribution='Offline contract witness',session_id='session')
    result=app.submit_turn('session','g1',SAVED['utterance'],expected_generation=0,
        registered_input=app.resources.compose_group([record.resource_id],attachment_candidates=True))
    assert result.status=='clarification',result.to_dict()
    state=app._application.sessions.load('session');interaction=state.interactions[-1]
    assert interaction.utterance==SAVED['utterance']
    assert interaction.admitted['bam']['source']=='source.0'
    assert interaction.admitted['bam']['species']=='human'
    assert interaction.preparation['candidate_id']
    assert not executor.plans and not state.turns and not state.revisions
    interpreter=next(c for c in model.calls if 'turn_schema_version' in c)
    planner=next(c for c in model.calls if 'semantic_prompt_version' in c)
    assert interpreter['utterance']==SAVED['utterance']
    assert SAVED['utterance'] in json.dumps(planner,ensure_ascii=False)
    assert str(source) in json.dumps(_serialize(interaction.admitted),ensure_ascii=False)
    assert 'inspect_scATAC' in json.dumps(_serialize(interaction.preparation),ensure_ascii=False)
