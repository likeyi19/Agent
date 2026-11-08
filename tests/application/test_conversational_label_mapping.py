"""Grounded natural-language mapping proposals and real saved comparison reuse.

Provider proposals alone are scripted; scientific execution and publication are real.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import pandas as pd
import pytest

from agent.application import InteractiveAgentApplication
from agent.application.dialogue_execution import _bind_arguments, _check_parameter_plan, scientific_parameters
from agent.application.turn_decisions import parse_decision, decision_schema, IntentError, admit_scientific_argument
from agent.orchestration import build_default_tool_registry, AgentPlan, PlanStep, PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.tools.analysis import cell_label_comparison as owner

TOOL=owner.TOOL
TEXT='Map CD4 Naive T and CD4 Memory T in my predictions to CD4 T, and map CD4 T cell in the reference to CD4 T. Leave the other labels unchanged.'


def rules():
    return [dict(tool=TOOL,argument='prediction_mapping',members=[
        dict(name=label,name_evidence=TEXT.split(',')[0],kind='string',literal='CD4 T',evidence='predictions to CD4 T')
        for label in ('CD4 Naive T','CD4 Memory T')]),
        dict(tool=TOOL,argument='reference_mapping',members=[dict(name='CD4 T cell',name_evidence='map CD4 T cell in the reference',
            kind='string',literal='CD4 T',evidence='reference to CD4 T')])]


def wire(decision):return json.dumps(dict(turn_schema_version=1,decision=decision))


def test_natural_language_mapping_both_keys_and_values_are_grounded_and_replay():
    registry=build_default_tool_registry()
    raw=wire(dict(kind='execute_plan',target=TOOL,arguments=rules()))
    decision=parse_decision(raw,utterance=TEXT)
    values=_bind_arguments(registry,TEXT,{},decision.arguments)
    assert values==dict(prediction_mapping={'CD4 Naive T':'CD4 T','CD4 Memory T':'CD4 T'},reference_mapping={'CD4 T cell':'CD4 T'})
    replay=parse_decision(wire(dict(kind='execute_plan',target=TOOL,arguments=[asdict(d) for d in decision.arguments])))
    assert _bind_arguments(registry,TEXT,{},replay.arguments)==values
    plan=AgentPlan('p','r',TEXT,(PlanStep('comparison',TOOL,values),))
    _check_parameter_plan(registry,plan,dict(inputs=values,arguments=[asdict(d) for d in decision.arguments]),())
    rows=[r for r in scientific_parameters(registry) if r['tool']==TOOL and r['argument'].endswith('_mapping')]
    assert len(rows)==2 and all(r['json_object'] and r['string_mapping'] for r in rows)
    # A planner cannot substitute a different mapping for the admitted instruction.
    with pytest.raises(IntentError):
        _check_parameter_plan(registry,replace(plan,steps=(replace(plan.steps[0],arguments=dict(values,prediction_mapping={'B':'T'})),)),
            dict(inputs=values,arguments=[asdict(d) for d in decision.arguments]),())


@pytest.mark.parametrize('variant',['invented_source','invented_target','conflicting_source','missing_key_evidence','wrong_key_span'])
def test_mapping_never_invents_or_conflicts_in_literal_admission(variant):
    proposal=deepcopy(rules()[0])
    if variant=='invented_source':proposal['members'][0]['name']='Not in the user request'
    elif variant=='invented_target':proposal['members'][0]['literal']='NK cell'
    elif variant=='conflicting_source':proposal['members'][1]['name']='CD4 Naive T'
    elif variant=='missing_key_evidence':proposal['members'][0].pop('name_evidence')
    if variant=='wrong_key_span':
        decision=parse_decision(wire(dict(kind='execute_plan',target=TOOL,arguments=[proposal])),utterance=TEXT)
        d=decision.arguments[0];m=replace(d.members[0],name_start=d.members[1].name_start,name_end=d.members[1].name_end)
        with pytest.raises(IntentError):admit_scientific_argument(replace(d,members=(m,d.members[1])),TEXT,
            build_default_tool_registry().get(TOOL).optional_arguments['prediction_mapping'])
    else:
        with pytest.raises((IntentError,ValueError)):
            decision=parse_decision(wire(dict(kind='execute_plan',target=TOOL,arguments=[proposal])),utterance=TEXT)
            _bind_arguments(build_default_tool_registry(),TEXT,{},decision.arguments)


@pytest.mark.parametrize('raw',[ '{"B":"T","B":"NK"}', '{"B":["T","NK"]}', '{"B":null}' ])
def test_explicit_json_conflicts_and_split_rules_are_not_repaired(raw):
    # Duplicate keys are rejected at admission; invalid owner values remain invalid.
    try:
        decision=parse_decision(wire(dict(kind='execute_plan',target=TOOL,arguments=[
            dict(tool=TOOL,argument='prediction_mapping',literal=raw,start=0,end=len(raw))])))
        values=_bind_arguments(build_default_tool_registry(),raw,{},decision.arguments)
    except IntentError:
        assert raw.startswith('{"B":"T",')
    else:
        with pytest.raises(ValueError):owner._mapping(['B'],values['prediction_mapping'],'prediction')


class Model:
    model_id='scripted/mapping'
    def __init__(self,decisions):self.decisions=decisions;self.calls=[]
    def complete(self,*,prompt,response_schema):
        value=json.loads(prompt);self.calls.append(value)
        if 'turn_schema_version' in value:
            if value['utterance'] == 'Compare the same labels using a common vocabulary.':
                group=value['dialogue']['result_groups'][0]
                assert group['label_vocabulary']['prediction']==['B','CD4 Memory T','CD4 Naive T']
                assert group['label_vocabulary']['reference']==['B','CD4 T cell']
                assert 'confusion' not in group
            assert 'name_evidence' in json.dumps(response_schema)
            return wire(self.decisions[value['utterance']])
        if 'output_selection_schema_version' in value:
            selected = dict(outputs=[dict(name='comparison',step_id='comparison',output_key='manifest_path')])
            if 'retain' in response_schema['properties']:
                selected['retain'] = []
            return json.dumps(selected)
        if 'dialogue_schema_version' in value:
            return json.dumps(dict(support='unavailable',paragraphs=[]))
        if 'grounding_review_version' in value:
            return json.dumps(dict(verdicts={}))
        # Ordinary v4 Planner: bind each reviewed parameter through its input.
        inputs=value['catalog']['request_inputs']
        names=[k for k in ('prediction_path','reference_path','prediction_label_key','reference_label_key','shared_library_id') if k in inputs]
        names += [k for k in ('prediction_mapping','reference_mapping','prediction_sha256','reference_sha256') if k in inputs]
        return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='comparison',tool=TOOL,
            sources=[dict(target=k.replace('_path','_dataset') if k.endswith('_path') else k,
                source=dict(kind='input',input=k)) for k in names],control_dependencies=[])])))


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("with_library", [False, True])
def test_missing_mapping_continues_via_existing_parameter_owner_and_reuses_result(tmp_path,monkeypatch,registered,with_library):
    p=tmp_path/'prediction.h5ad';r=tmp_path/'reference.h5ad'
    ad.AnnData(obs=pd.DataFrame({'label':pd.Categorical(['CD4 Naive T','CD4 Memory T','B',None])},index=list('abcd'))).write_h5ad(p)
    ad.AnnData(obs=pd.DataFrame({'label':['CD4 T cell','CD4 T cell','B','B']},index=list('abcd'))).write_h5ad(r)
    inputs=dict(prediction_path=str(p),reference_path=str(r),prediction_label_key='label',reference_label_key='label',shared_library_id='fixture-library')
    if not with_library:inputs.pop('shared_library_id')
    decisions={
        'Compare the supplied fixed labels strictly.':dict(kind='execute_plan',target=TOOL),
        'Compare the same labels using a common vocabulary.':dict(kind='execute_plan',target=TOOL,
            missing_parameters=['prediction_mapping','reference_mapping']),
        TEXT:dict(kind='answer_parameters',pending='@parameters',arguments=rules())}
    if registered:
        roles=[dict(tool=TOOL,port=side+'_dataset',source='source.'+str(n)) for n,side in enumerate(('prediction','reference'))]
        for text in ('Compare the supplied fixed labels strictly.','Compare the same labels using a common vocabulary.'):
            decisions[text]['roles']=roles
        decisions['Compare the same labels using a common vocabulary.']['retain_parameters']=[
            dict(result='g0',argument=k) for k in ('prediction_label_key','reference_label_key','shared_library_id') if k in inputs]
    model=Model(decisions)
    profile=PlanningModelProfile('test','scripted',model.model_id)
    service=InteractiveAgentApplication(tmp_path/'workspace',model_profiles=(profile,),default_profile_id='test',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda p:model}),approved_source_roots=(tmp_path,))
    service.create_session('session')
    options=dict(execution_inputs=inputs)
    if registered:
        from agent.application.local_resources import LocalResourceAdmission
        local=LocalResourceAdmission(tmp_path/'workspace',approved_source_roots=(tmp_path,))
        records=[local.register(side,path,input_type='h5ad',label=side,attribution='simulated technical fixture',session_id='session')
                 for side,path in (('prediction',p),('reference',r))]
        options=dict(attachment_ids=[r.resource_id for r in records],registered_input=local.compose_group([r.resource_id for r in records],
            {k:v for k,v in inputs.items() if not k.endswith('_path')},attachment_candidates=True))
    strict=service.submit_turn('session','strict','Compare the supplied fixed labels strictly.',expected_generation=0,**options).to_dict()
    assert strict['status']=='succeeded',strict
    app=service._application
    before=app.run_store.load(strict['run_id']).to_dict()
    pending=service.submit_turn('session','pending','Compare the same labels using a common vocabulary.',expected_generation=1,**({} if registered else dict(execution_inputs=inputs))).to_dict()
    assert pending['status']=='clarification',pending
    assert len(app.sessions.load('session').turns)==1
    mapped=service.submit_turn('session','mapped',TEXT,expected_generation=1).to_dict()
    assert mapped['status']=='succeeded',mapped
    state=app.sessions.load('session');assert len(state.revisions)==len(state.turns)==2
    step=app.run_store.load(mapped['run_id']).steps[0]
    assert step.result['accuracy']==.75 and step.result['original_metrics']['accuracy']==.25
    assert step.verification.passed
    assert app.run_store.load(strict['run_id']).to_dict()==before
    from agent.application.native_delivery import accepted_artifacts
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    revision=state.revisions[-1];origin=state.turn('mapped')
    artifacts=accepted_artifacts(SimpleNamespace(_application=app),'session',revision,origin,revision.outputs[0])
    assert {a.reference.filename for a in artifacts}==set(owner.DELIVERY_FILES)
    for artifact in artifacts:
        with MatrixDownloadPreparer().prepare(artifact) as download:payload=b''.join(download.iter_chunks())
        if artifact.role=='comparison.json':
            delivered=json.loads(payload);assert delivered['summary']['original_metrics']['accuracy']==.25
        if artifact.role=='aligned-labels.tsv':assert b'mapped_predicted_label' in payload
    monkeypatch.setattr(owner,'compare',lambda *a,**k:pytest.fail('Reuse reran comparison'))
    fresh=InteractiveAgentApplication(tmp_path/'workspace',model_profiles=(profile,),default_profile_id='test',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda p:model}))
    evidence=fresh.evidence('session',revision.revision_id,revision.outputs[0].name)
    facts={f.field:_serialize(f.value) for f in evidence.facts}
    assert facts['mapping']['prediction']['rules'][0]['common_label']=='CD4 T'
    assert facts['original_metrics']['accuracy']==.25 and facts['label_policy']=='explicit_harmonized_labels'
    assert fresh._application.run_store.load(strict['run_id']).to_dict()==before
    from agent.application.scientific_dialogue import context
    output=revision.outputs[0]
    target=dict(revision_id=revision.revision_id,output_name=output.name,accepted_step_sha256=output.accepted_step_sha256,subject=None)
    public,claims=context(fresh._application.sessions,'session',dict(target=target,comparison=None))
    assert {'mapping','original_metrics','vocabulary','metric_denominators'} <= {c.field for c in claims}
    assert next(c.value for c in claims if c.field=='original_metrics')['accuracy']==.25



def test_mapping_question_uses_researcher_language_without_json():
    from agent.application.responses import clarification_text
    from agent.application.turn_decisions import Clarify
    text=clarification_text(Clarify('missing_parameter_value',
        ('compare_cell_annotations.prediction_mapping','compare_cell_annotations.reference_mapping'),True))
    assert 'original labels' in text and 'common categories' in text
    assert 'predictions and reference' in text and 'own words' in text
    assert 'compare_cell_annotations' not in text and 'JSON' not in text


def test_historical_report_order_and_new_whole_fact_order_are_version_aware():
    from agent.report.analysis_report import report_fact_fields
    assert report_fact_fields(TOOL,{})==owner.FACT_FIELDS
    order=report_fact_fields(TOOL,{'vocabulary':{}})
    assert order[:2]==('label_policy','vocabulary')
    assert 'classification_disagreements' not in order
    order=report_fact_fields(TOOL,{k:{} for k in owner.EXTENDED_FACT_FIELDS})
    assert order[:1+len(owner.EXTENDED_FACT_FIELDS)]==('label_policy',*owner.EXTENDED_FACT_FIELDS)


def test_quoted_mapping_target_is_not_the_prefix_inside_another_quoted_label():
    text='In predictions map "CD4 Naive T" to "CD4 T" and "CD4 Memory T" to "CD4 T". In reference map "CD4 T cell" to "CD4 T".'
    arguments=[]
    for side,labels in [('prediction',['CD4 Naive T','CD4 Memory T']),('reference',['CD4 T cell'])]:
        members=[]
        for label in labels:
            clause='"'+label+'" to "CD4 T"'
            members.append(dict(name=label,name_evidence=clause,kind='string',literal='CD4 T',evidence=clause))
        arguments.append(dict(tool=TOOL,argument=side+'_mapping',native_function=None,members=members))
    decision=parse_decision(wire(dict(kind='execute_plan',target=TOOL,arguments=arguments)),utterance=text)
    values=_bind_arguments(build_default_tool_registry(),text,{},decision.arguments)
    assert values['reference_mapping']=={'CD4 T cell':'CD4 T'}
    assert values['prediction_mapping']=={'CD4 Naive T':'CD4 T','CD4 Memory T':'CD4 T'}
    member=decision.arguments[1].members[0]
    assert text[member.start-1:member.end+1]=='"CD4 T"'


def test_null_native_scope_never_masks_a_mapping_grounding_clarification():
    from agent.application.turn_decisions import DeclarationGroundingError
    text='Map CD4 T cell to CD4 T.'
    proposal=dict(tool=TOOL,argument='reference_mapping',native_function=None,members=[
        dict(name='CD4 T cell',name_evidence=text,kind='string',literal='CD4 T',evidence=text)])
    with pytest.raises(DeclarationGroundingError,match='ambiguous_parameter'):
        parse_decision(wire(dict(kind='execute_plan',target=TOOL,arguments=[proposal])),utterance=text)


def test_rejected_mapping_literal_leaves_a_closed_nonexecuting_interaction(tmp_path,monkeypatch):
    p=tmp_path/'prediction.h5ad';r=tmp_path/'reference.h5ad'
    for path in (p,r):ad.AnnData(obs=pd.DataFrame({'label':['B']},index=['cell'])).write_h5ad(path)
    proposal=rules()
    proposal[0]['native_function']=None
    proposal[0]['members'][0].update(literal='invented replacement',evidence=TEXT)
    model=Model({TEXT:dict(kind='execute_plan',target=TOOL,arguments=proposal)})
    profile=PlanningModelProfile('test','scripted',model.model_id)
    service=InteractiveAgentApplication(tmp_path/'workspace',model_profiles=(profile,),default_profile_id='test',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda p:model}))
    service.create_session('session')
    monkeypatch.setattr(owner,'compare',lambda *a,**k:pytest.fail('Rejected proposal executed science'))
    from agent.application import turns
    import traceback
    escaped=[]
    original=turns.respond
    def observed(*a,**kw):
        try:return original(*a,**kw)
        except Exception as exc:
            escaped.append(traceback.format_exc())
            raise
    monkeypatch.setattr(turns,'respond',observed)
    view=service.submit_turn('session','rejected',TEXT,expected_generation=0,execution_inputs=dict(
        prediction_path=str(p),reference_path=str(r),prediction_label_key='label',reference_label_key='label',
        shared_library_id='fixture-library')).to_dict()
    state=service._application.sessions.load('session')
    assert not state.turns and not state.revisions
    assert not escaped, '\n'.join(escaped)
    assert view['status'] in ('failed','clarification'),view
    assert state.interactions[-1].status in ('failed','clarification'),state.interactions[-1]
    assert view.get('error',{}).get('code')!='INTERACTIVE_APPLICATION_FAILED' if view.get('error') else True


def retained_columns(result='g0', *extra):
    return [dict(result=result,argument=k) for k in (
        'prediction_label_key','reference_label_key','shared_library_id',*extra)]


@pytest.mark.parametrize('mode',['direct','question','reference_then_prediction','reordered_candidates','explicit_source'])
def test_accepted_column_retention_keeps_original_operands_without_repeated_roles(tmp_path,mode,monkeypatch):
    """Witness Phase 0 #3 and #9: correct retained parameters but no roles."""
    p=tmp_path/'pred.h5ad';r=tmp_path/'ref.h5ad'
    for path,key,values in [(p,'label',['CD4 Naive T','CD4 Memory T','B',None]),
                            (r,'label',['CD4 T cell','CD4 T cell','B','B'])]:
        ad.AnnData(obs=pd.DataFrame({key:pd.Categorical(values)},index=list('abcd'))).write_h5ad(path)
    initial='Compare the supplied fixed labels strictly.'
    missing='Compare the same labels using a common vocabulary.'
    base=dict(kind='execute_plan',target=TOOL,context='current',retain_parameters=retained_columns())
    decisions={initial:dict(kind='execute_plan',target=TOOL,roles=[
        dict(tool=TOOL,port=side+'_dataset',source='source.'+str(n)) for n,side in enumerate(('prediction','reference'))])}
    if mode=='question':
        decisions[missing]=dict(base,missing_parameters=['prediction_mapping','reference_mapping'])
        decisions[TEXT]=dict(kind='answer_parameters',pending='@parameters',arguments=rules())
    elif mode=='reference_then_prediction':
        decisions[TEXT+' First apply only the reference-side rule.']=dict(base,arguments=rules()[1:])
        decisions[TEXT]=dict(base,arguments=rules()[:1],retain_parameters=retained_columns('g0','reference_mapping'))
    else:decisions[TEXT]=dict(base,arguments=rules())
    model=Model(decisions);profile=PlanningModelProfile('test','scripted',model.model_id)
    service=InteractiveAgentApplication(tmp_path/'app',model_profiles=(profile,),default_profile_id='test',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda p:model}),approved_source_roots=(tmp_path,))
    service.create_session('session')
    local=service.resources
    records=[local.register(side,path,input_type='h5ad',label=side,attribution='simulated',session_id='session')
        for side,path in [('prediction',p),('reference',r)]]
    group=local.compose_group([x.resource_id for x in records],dict(prediction_label_key='label',reference_label_key='label',
        shared_library_id='fixture-library'),attachment_candidates=True)
    strict=service.submit_turn('session','strict',initial,expected_generation=0,registered_input=group,
        attachment_ids=[x.resource_id for x in records]).to_dict()
    assert strict['status']=='succeeded',strict
    before=service._application.run_store.load(strict['run_id']).to_dict()
    generation=1
    extra={}
    if mode=='reference_then_prediction':
        ref=service.submit_turn('session','reference-only',TEXT+' First apply only the reference-side rule.',expected_generation=1).to_dict()
        assert ref['status']=='succeeded',ref
        generation=2
    elif mode in ('reordered_candidates','explicit_source'):
        folder=tmp_path/'other';folder.mkdir()
        other=folder/p.name
        data=ad.read_h5ad(p);data.uns['unrelated']='different bytes, same cells and label vocabulary';data.write_h5ad(other)
        record=local.register('other',other,input_type='h5ad',label=p.name,attribution='unrelated simulated upload',session_id='session')
        extra=dict(attachment_ids=[record.resource_id])
        if mode=='explicit_source':
            decisions[TEXT]['roles']=[dict(tool=TOOL,port='prediction_dataset',source='source.0')]
    if mode=='question':
        q=service.submit_turn('session','question',missing,expected_generation=1).to_dict()
        assert q['status']=='clarification',q
        assert len(service._application.sessions.load('session').turns)==1
    view=service.submit_turn('session','mapped',TEXT,expected_generation=generation,**extra).to_dict()
    assert view['status']=='succeeded',view
    step=service._application.run_store.load(view['run_id']).steps[0]
    assert step.resolved_arguments['prediction_path']==str(other if mode=='explicit_source' else p)
    assert step.resolved_arguments['reference_path']==str(r)
    assert step.result['accuracy']==.75 and step.result['original_metrics']['accuracy']==.25
    assert step.verification.passed
    assert service._application.run_store.load(strict['run_id']).to_dict()==before
    state=service._application.sessions.load('session')
    assert len(state.turns)==len(state.revisions)==generation+1
    if mode=='direct':
        from copy import copy
        from agent.application import dialogue_execution as execution
        from agent.application.local_resources import RegisteredInputGroup
        current=state.interactions[-1]
        offered=execution.captured_input_group(current,state.interactions[:-1])
        scoped=copy(service._application.sessions);scoped._interaction_session_id='session'
        monkeypatch.setattr(owner,'compare',lambda *a,**k:pytest.fail('Source retention executed science'))
        for key in ('resource_id','record_sha256','source_path','source_sha256'):
            altered=dict(offered.members[0])
            altered[key]=('local-'+'f'*64 if key=='resource_id' else str(tmp_path/'other'/p.name)
                if key=='source_path' else 'f'*64)
            changed=RegisteredInputGroup([altered,*offered.members[1:]],{},attachment_candidates=True)
            monkeypatch.setattr(execution,'captured_input_group',lambda *a,**k:changed)
            with pytest.raises(IntentError,match='unavailable_context'):
                execution._retained_field_roles(scoped,current,current.admitted['retained_parameters'],())
    if mode=='reordered_candidates':
        assert [r['source'] for r in state.interactions[-1].admitted['roles']]==['source.1','source.2']
    if mode=='reference_then_prediction':
        assert dict(step.resolved_arguments['reference_mapping'])=={'CD4 T cell':'CD4 T'}
        replacement='In the reference, replace the earlier rule: map "CD4 T cell" to "B". Keep the prediction rules.'
        decisions[replacement]=dict(base,retain_parameters=retained_columns('g0','prediction_mapping'),arguments=[
            dict(tool=TOOL,argument='reference_mapping',members=[dict(name='CD4 T cell',name_evidence=replacement,
                kind='string',literal='B',evidence=replacement)])])
        accepted_before=service._application.run_store.load(view['run_id']).to_dict()
        replaced=service.submit_turn('session','replaced',replacement,expected_generation=3).to_dict()
        assert replaced['status']=='succeeded',replaced
        final=service._application.run_store.load(replaced['run_id']).steps[0]
        assert dict(final.resolved_arguments['reference_mapping'])=={'CD4 T cell':'B'}
        assert final.resolved_arguments['prediction_mapping']==step.resolved_arguments['prediction_mapping']
        assert final.result['accuracy']==final.result['original_metrics']['accuracy']==.25
        assert service._application.run_store.load(view['run_id']).to_dict()==accepted_before


def test_identity_mapping_excludes_exact_key_span_but_retains_real_target_ambiguity():
    from agent.application.turn_decisions import DeclarationGroundingError
    text='In reference map "Plasma" to "Plasma".'
    member=dict(name='Plasma',name_evidence='map "Plasma" to',kind='string',literal='Plasma',evidence='map "Plasma" to "Plasma"')
    decision=parse_decision(wire(dict(kind='execute_plan',target=TOOL,arguments=[
        dict(tool=TOOL,argument='reference_mapping',members=[member])])),utterance=text)
    value=_bind_arguments(build_default_tool_registry(),text,{},decision.arguments)
    assert value['reference_mapping']=={'Plasma':'Plasma'}
    assert decision.arguments[0].members[0].start>decision.arguments[0].members[0].name_end
    text='In reference map "Plasma" to "Plasma" or "Plasma".'
    member['evidence']='map "Plasma" to "Plasma" or "Plasma"'
    with pytest.raises(DeclarationGroundingError,match='ambiguous_parameter'):
        parse_decision(wire(dict(kind='execute_plan',target=TOOL,arguments=[
            dict(tool=TOOL,argument='reference_mapping',members=[member])])),utterance=text)


def test_saved_identity_mapping_rejection_uses_original_locator_semantics():
    from agent.application.dialogue_execution import _repair_declarations
    text='In reference map "Plasma" to "Plasma".'
    raw=wire(dict(kind='execute_plan',target=TOOL,arguments=[dict(tool=TOOL,argument='reference_mapping',members=[
        dict(name='Plasma',name_evidence='map "Plasma" to',kind='string',literal='Plasma',evidence='map "Plasma" to "Plasma"')])]))
    assert parse_decision(raw,utterance=text).arguments[0].members[0].literal=='Plasma'
    safe,rejected=_repair_declarations(raw,text,build_default_tool_registry(),historical_missing={(TOOL,'reference_mapping')})
    assert rejected==[(TOOL,'reference_mapping')] and not safe.arguments
