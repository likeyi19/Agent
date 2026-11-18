"""Exact accepted metadata projection; these fixtures qualify no new science."""
from dataclasses import replace
import hashlib

import pytest

from agent.application import ApplicationResult, ApplicationStatus, ArtifactReference, OutputSelection
from agent.application.session_state import canonical, digest
from agent.report import evidence as ev
from agent.schemas import AgentPlan, AgentRequest, PlanStep, StepExecutionResult, StepOutputRef, StepStatus
from agent.schemas.run_state import (PersistedRunState, RecoveryPolicySnapshot, RunLifecycleStatus,
    ToolRecoveryPolicySnapshot, fingerprint_plan, fingerprint_recovery_policy)
from test_dialogue_evidence import app, accepted, files, forbid_work, passed


@pytest.mark.parametrize('mode', ['offered', 'literal'])
def test_mixed_source_field_and_numeric_provenance_reuse_captured_scope_without_io(app, monkeypatch, mode):
    """Attribution-only fixture; native acceptance is covered by the HTTP test."""
    from types import SimpleNamespace
    import h5py
    from agent.application.dialogue_evidence import _explicit_user_parameter_facts
    from agent.application.local_resources import RegisteredInputGroup
    from agent.application.session_state import Interaction
    field = 'Reference identity β'
    literal = field if mode == 'literal' else 'provided annotations'
    text = f'Use resolution 0.6 and compare with {literal}.'
    start = text.index(literal)
    member = dict(resource_id='local-' + 'a'*64, record_sha256='b'*64,
        source_sha256='c'*64, source_path='/accepted/reference.h5ad', input_type='h5ad', label='Reference')
    unbound = RegisteredInputGroup((member,), {}, attachment_candidates=True)
    roles = [dict(tool='evaluate_cell_clustering', port='ground_truth_dataset', source='source.0')]
    values = dict(run_scanpy__tl_leiden__kwargs={'resolution':.6}, label_key=field, reference_h5ad_path=member['source_path'])
    arguments = [dict(tool='run_scanpy',native_function='tl.leiden',argument='kwargs',literal='0.6',start=15,end=18,
        members=[dict(name='resolution',kind='number',literal='0.6',start=15,end=18)]),
        dict(tool='evaluate_cell_clustering',argument='label_key',source='source.0',mode=mode,
             value=field,literal=literal,start=start,end=start+len(literal))]
    arguments[0]['start'] = text.index('0.6'); arguments[0]['end'] = arguments[0]['start'] + 3
    snapshot = dict(dialogue=dict(source_view=dict(execution_inputs={},registered_input=unbound.attribution()),
        source_fields=[dict(source='source.0',resource_id=member['resource_id'],source_sha256=member['source_sha256'],
                            fields=[field] if mode == 'offered' else [],complete=mode == 'offered')]))
    admitted = dict(kind='execute',operation='plan',request_id='science',inputs=values,roles=roles,
        registered_input=RegisteredInputGroup((member,),values,roles=roles,attachment_candidates=True).attribution(),
        arguments=arguments)
    interaction = Interaction('turn',text,None,0,snapshot,status='admitted',admitted=admitted)
    planned = (PlanStep('cluster','run_scanpy',dict(function_name='tl.leiden',kwargs={'resolution':.6})),
               PlanStep('evaluate','evaluate_cell_clustering',dict(label_key=field,reference_h5ad_path=member['source_path'])))
    run = SimpleNamespace(run_id='science:run',request_id='science',
                          plan=SimpleNamespace(steps=planned))
    state = SimpleNamespace(turns=(SimpleNamespace(run_id=run.run_id,turn_id='turn'),),interactions=(interaction,))
    monkeypatch.setattr(h5py,'File',lambda *a,**k: pytest.fail('Evidence attribution reopened a scientific source.'))
    before = files(app)
    for target, name in zip(planned,('kwargs','label_key')):
        step = SimpleNamespace(tool_name=target.tool_name,step_id=target.step_id,resolved_arguments=target.arguments)
        fact, = _explicit_user_parameter_facts(app.sessions,state,run,step,target)
        assert fact.value['parameters'] == {name:'explicit_user_instruction'}
        assert fact.artifact_name == 'accepted_session_interactions'
    assert files(app) == before

    # The same scope owner must still reject wrong source and admitted/accepted
    # values; adding field support cannot weaken parameter provenance.
    from agent.application.dialogue_evidence import _Unavailable
    target = planned[1]
    step = SimpleNamespace(tool_name=target.tool_name,step_id=target.step_id,resolved_arguments=target.arguments)
    for mutation in ('source','admitted','planned','resolved'):
        changed = dict(admitted)
        bad_target, bad_step = target, step
        if mutation == 'source':
            changed['arguments'] = [arguments[0],dict(arguments[1],source='source.1')]
        elif mutation == 'admitted':
            changed['inputs'] = dict(values,label_key='different')
        elif mutation == 'planned':
            bad_target = PlanStep('evaluate','evaluate_cell_clustering',dict(label_key='different'))
        else:
            bad_step = SimpleNamespace(tool_name=target.tool_name,step_id=target.step_id,
                                       resolved_arguments=dict(label_key='different'))
        bad_interaction = replace(interaction,admitted=changed)
        bad_state = SimpleNamespace(turns=state.turns,interactions=(bad_interaction,))
        with pytest.raises(_Unavailable,match='session_parameter_binding_mismatch'):
            _explicit_user_parameter_facts(app.sessions,bad_state,run,bad_step,bad_target)


def analysis(app, *, resolution=.7, binding='exact', mutation=None):
    """Accept two distinct native clustering steps; select requested clustering and UMAP."""
    request = AgentRequest('analysis', 'Accepted processed-analysis metadata fixture.', {})
    app.sessions.start_turn('session', 'analysis', request,
        (OutputSelection('result', 'umap', 'matrix_path'), OutputSelection('clustering', 'requested', 'matrix_path')), expected_generation=0)
    app.sessions.link_run('session', 'analysis')
    effective, paths = app._prepare_request(request)
    definitions = []
    spec=app.registry.get('run_scanpy')
    defaults={str:'fixture',int:8,float:.5,bool:True,dict:{},list:[]}
    def result(function, kwargs, path):
        row={k:defaults[t[0]] for k,t in spec.result_contract.required_fields.items()}
        row.update(function_name=function,matrix_path=path,parameters=dict(kwargs=kwargs,
            effective_native_parameters=dict(resolution=1.0 if resolution is None else resolution,random_state=0)
                if function=='tl.leiden' else dict(min_dist=.5,random_state=0)))
        return row
    for name,value in (('unrelated',.2),('requested',resolution)):
        kwargs={} if value is None else dict(resolution=value)
        arguments=dict(input_path='/accepted/neighbors.h5ad',output_dir='/accepted',function_name='tl.leiden',kwargs=kwargs)
        definitions.append((PlanStep(name,'run_scanpy',arguments),result('tl.leiden',kwargs,f'/accepted/{name}.h5ad')))
    source=StepOutputRef('requested','matrix_path') if binding=='exact' else '/accepted/requested.h5ad'
    definitions.append((PlanStep('umap','run_scanpy',dict(input_path=source,output_dir='/accepted',function_name='tl.umap'),
        depends_on=('requested',)),result('tl.umap',{},'/accepted/umap.h5ad')))
    plan = AgentPlan('analysis:plan', 'analysis', 'scripted', tuple(step for step, _ in definitions))
    steps = []
    for step, result in definitions:
        arguments = dict(step.arguments)
        if step.step_id == 'umap':
            arguments['input_path'] = '/accepted/requested.h5ad'
        if mutation == 'binding' and step.step_id == 'umap':
            arguments['input_path'] = '/accepted/unrelated.h5ad'
        if mutation == 'argument' and step.step_id == 'requested':
            arguments['kwargs'] = {'resolution':.9}
        steps.append(StepExecutionResult(step.step_id, step.tool_name, StepStatus.SUCCEEDED,
            attempt_count=1, result=result, resolved_arguments=arguments, verification=passed('step', step.step_id),
            started_at='2026-09-23T00:00:00+00:00', finished_at='2026-09-23T00:00:00+00:00',
            duration_seconds=0.0))
    policies = tuple(ToolRecoveryPolicySnapshot(tool, app.registry.get(tool).recovery_policy_version)
        for tool in ('run_scanpy',))
    policy = RecoveryPolicySnapshot('fixture', 1, policies, fingerprint_recovery_policy('fixture', 1, policies))
    stored = PersistedRunState(3, 0, 'analysis:run', effective, RunLifecycleStatus.SUCCEEDED,
        '2026-09-23T00:00:00+00:00', '2026-09-23T00:00:00+00:00', plan=plan,
        plan_fingerprint=fingerprint_plan(plan), recovery_policy_snapshot=policy,
        preflight_verification=passed('plan', plan.plan_id), steps=tuple(steps),
        run_verification=passed('run', plan.plan_id))
    app.run_store.create(stored)
    run = stored.to_run_result()
    payload = dict(schema_version=1, artifact_type=ev.ANALYSIS_EVIDENCE_ARTIFACT_TYPE, status='success',
        run=dict(run_id=run.run_id, request_id=run.request_id, plan_id=plan.plan_id,
            planner_name=plan.planner_name, source_run_status='SUCCEEDED', plan_sha256=digest(plan.to_dict()),
            source_run_result_sha256=digest(run.to_dict())),
        workflow=dict(ordered_steps=[dict(step_id=s.step_id, tool_name=s.tool_name,
            depends_on=list(s.depends_on)) for s in plan.steps]),
        steps=[dict(step_id=s.step_id, tool_name=s.tool_name, attempt_count=1,
            facts={key: s.to_dict()['result'][key] for key in ev._TOOL_PROJECTIONS[s.tool_name].fact_fields},
            verification=dict(passed=True, freshly_verified=True, check_names=['fixture']),
            recovery_identity=app.registry.get(s.tool_name).recovery_policy_version) for s in steps],
        artifacts=[], provenance=dict(trace_sha256=digest([]), evidence_projection_version=1,
            registry_tool_identities=[dict(tool_name=tool, result_contract=app.registry.get(tool).result_contract.name,
                recovery_identity=app.registry.get(tool).recovery_policy_version)
                for tool in ('run_scanpy',)]))
    path = paths.evidence / ev.ANALYSIS_EVIDENCE_FILENAME
    path.write_bytes(canonical(payload))
    report = paths.report / 'historical-report.txt'
    report.write_bytes(b'Original accepted report bytes.\n')
    completed = ApplicationResult('analysis', run.run_id, ApplicationStatus.SUCCEEDED, run.status,
        str(paths.root), run, ArtifactReference(ev.ANALYSIS_EVIDENCE_ARTIFACT_TYPE, str(path), digest(payload)),
        report=ArtifactReference('fixture', str(report), hashlib.sha256(report.read_bytes()).hexdigest()))
    app.sessions._record_result('session', 'analysis', completed)
    state = app.sessions.recover('session', 'analysis')
    return state.turn('analysis').revision_id, path, payload, run


def repin(app, path, payload):
    """A pinned malformed summary still must match its accepted scientific source."""
    raw = canonical(payload)
    path.write_bytes(raw)
    state = app.sessions.load('session')
    app.sessions._store._write(replace(state, turns=tuple(replace(turn, completion_files=tuple(
        replace(item, sha256=hashlib.sha256(raw).hexdigest()) if item.path == str(path) else item
        for item in turn.completion_files)) for turn in state.turns)))


def native_parameters(resolution=.7):
    return dict(kwargs={} if resolution is None else {'resolution':resolution},
        effective_native_parameters={'resolution':1.0 if resolution is None else resolution,'random_state':4})


@pytest.mark.parametrize('resolution,origin', [(.7, 'explicit_execution_argument'),
    (1.0, 'explicit_execution_argument'), (None, 'existing_owner_default')])
def test_selected_cluster_resolution_origin_reads_exact_accepted_metadata(app, monkeypatch, resolution, origin):
    parameters=native_parameters(resolution)
    rid,path,payload=accepted(app,'run_scanpy',arguments=dict(function_name='tl.leiden',kwargs=parameters['kwargs']),
        overrides=dict(function_name='tl.leiden',parameters=parameters))
    guarded=forbid_work(app,monkeypatch);before=files(app)
    view=guarded.sessions.evidence('session',rid,'output0',fields=('parameters',))
    assert view.status=='available',view
    fact,=view.facts
    assert fact.value==parameters
    assert fact.source_pointer=='/steps/0/facts/parameters'
    assert ('resolution' in fact.value['kwargs']) == (origin=='explicit_execution_argument')
    assert files(app)==before and path.read_bytes()==canonical(payload)


def test_historical_effective_value_is_never_replaced_by_current_owner_default(app,monkeypatch):
    parameters=dict(kwargs={},effective_native_parameters={'resolution':1.3,'random_state':7})
    rid,_,_=accepted(app,'run_scanpy',overrides=dict(function_name='tl.leiden',parameters=parameters))
    view=forbid_work(app,monkeypatch).sessions.evidence('session',rid,'output0',fields=('parameters',))
    assert view.status=='available' and view.facts[0].value==parameters


def test_corrupt_boolean_explicit_argument_is_not_equated_with_recorded_numeric_resolution(app,monkeypatch):
    rid,path,payload=accepted(app,'run_scanpy',overrides=dict(function_name='tl.leiden',parameters=native_parameters(1.0)))
    payload['steps'][0]['facts']['parameters']['kwargs']['resolution']=True
    repin(app,path,payload)
    view=forbid_work(app,monkeypatch).sessions.evidence('session',rid,'output0')
    assert view.status=='unavailable' and view.source is None and not view.facts


@pytest.mark.parametrize('resolution,expected,origin', [(.7,.7,'explicit_execution_argument'),(None,1.0,'existing_owner_default')])
def test_selected_cluster_beside_umap_preserves_exact_step_and_history(app,monkeypatch,resolution,expected,origin):
    rid,path,payload,run=analysis(app,resolution=resolution)
    guard=forbid_work(app,monkeypatch);before=files(app)
    cluster=guard.sessions.evidence('session',rid,'clustering',fields=('parameters',))
    assert cluster.status=='available',cluster
    assert cluster.source.output_locator['step_id']=='requested'
    assert cluster.source.source_run_result_sha256==digest(run.to_dict())
    assert cluster.facts[0].source_pointer=='/steps/1/facts/parameters'
    assert cluster.facts[0].value['effective_native_parameters']['resolution']==expected
    assert ('resolution' in cluster.facts[0].value['kwargs'])==(origin=='explicit_execution_argument')
    umap=guard.sessions.evidence('session',rid,'result',fields=('parameters',))
    assert umap.status=='available' and 'resolution' not in umap.facts[0].value['effective_native_parameters']
    assert files(app)==before and path.read_bytes()==canonical(payload)


def test_matching_path_does_not_infer_clustering_source(app,monkeypatch):
    rid,_,_,_=analysis(app,binding='path')
    view=forbid_work(app,monkeypatch).sessions.evidence('session',rid,'result',fields=('parameters','clustering_parameters'))
    assert view.status=='available',view
    assert 'resolution' not in view.facts[0].value['effective_native_parameters']
    assert view.facts[1].status=='unavailable' and view.facts[1].value is None


@pytest.mark.parametrize('mutation',['binding','argument'])
def test_inconsistent_accepted_clustering_source_is_not_projected(app,monkeypatch,mutation):
    rid,path,payload,run=analysis(app)
    # Pin a changed claimed workflow/source. Exact accepted run remains immutable.
    if mutation=='binding':payload['workflow']['ordered_steps'][1]['depends_on']=['unrelated']
    else:payload['steps'][1]['facts']['parameters']['kwargs']['resolution']=.9
    repin(app,path,payload)
    view=forbid_work(app,monkeypatch).sessions.evidence('session',rid,'clustering')
    assert view.status=='unavailable' and view.source is None and not view.facts


@pytest.mark.parametrize('mutation',['missing_resolution','wrong_resolution','missing_summary','wrong_contract','verification'])
def test_missing_or_tampered_pinned_upstream_summary_does_not_supply_parameters(app,monkeypatch,mutation):
    rid,path,payload,_=analysis(app)
    if mutation=='missing_resolution':del payload['steps'][1]['facts']['parameters']['effective_native_parameters']['resolution']
    elif mutation=='wrong_resolution':payload['steps'][1]['facts']['parameters']['effective_native_parameters']['resolution']=.9
    elif mutation=='missing_summary':del payload['steps'][1]
    elif mutation=='verification':payload['steps'][1]['verification']['passed']=False
    else:payload['provenance']['registry_tool_identities'][0]['result_contract']='unsupported'
    repin(app,path,payload)
    view=forbid_work(app,monkeypatch).sessions.evidence('session',rid,'clustering')
    assert view.status in ('unavailable','unsupported') and view.source is None and not view.facts


def test_projected_parameter_fields_obey_existing_fact_bounds(app,monkeypatch):
    rid,_,_,_=analysis(app)
    guard=forbid_work(app,monkeypatch)
    from agent.application import dialogue_evidence
    monkeypatch.setattr(dialogue_evidence,'MAX_FACT_BYTES',8)
    view=guard.sessions.evidence('session',rid,'clustering',fields=('parameters',))
    assert view.status=='available'
    assert all(f.status=='omitted' and f.value is None and f.reason=='field_size_limit' for f in view.facts)


def test_native_parameter_followup_uses_accepted_values_without_science(app,monkeypatch):
    from test_scientific_dialogue import Model,question
    rid,path,_=accepted(app,'run_scanpy',overrides=dict(function_name='tl.leiden',parameters=native_parameters(.7)))
    guard=forbid_work(app,monkeypatch);pinned=path.read_bytes()
    fields=('parameters',)
    model=Model(question(),fields)
    response=guard.sessions.respond('session','parameters','What resolution did you use for clustering?',interpreter=model)
    assert response.status=='answered',response
    claim,=response.scientific.claims
    assert claim.value==native_parameters(.7)
    assert claim.source['pointer']=='/steps/0/facts/parameters'
    assert claim.source['revision_id']==rid
    assert path.read_bytes()==pinned


@pytest.mark.parametrize('location',['planned','resolved'])
@pytest.mark.parametrize('changed',[True,.9])
def test_native_numeric_attribution_rejects_type_or_value_substitution(app,monkeypatch,location,changed):
    from types import SimpleNamespace as N
    import h5py
    from agent.application.dialogue_evidence import _explicit_user_parameter_facts,_Unavailable
    from agent.application.session_state import Interaction
    text='Use resolution 1.0.'
    start=text.index('1.0')
    declaration=dict(tool='run_scanpy',native_function='tl.leiden',argument='kwargs',literal='1.0',start=start,end=start+3,
        members=[dict(name='resolution',kind='number',literal='1.0',start=start,end=start+3)])
    admitted=dict(kind='execute',operation='plan',request_id='science',
        inputs={'run_scanpy__tl_leiden__kwargs':{'resolution':1.0}},arguments=[declaration])
    interaction=Interaction('turn',text,None,0,{},status='admitted',admitted=admitted)
    planned=PlanStep('cluster','run_scanpy',dict(function_name='tl.leiden',kwargs={'resolution':1.0}))
    run=N(run_id='science:run',request_id='science',plan=N(steps=(planned,)))
    state=N(turns=(N(run_id=run.run_id,turn_id='turn'),),interactions=(interaction,))
    step=N(tool_name=planned.tool_name,step_id=planned.step_id,resolved_arguments=planned.arguments)
    monkeypatch.setattr(h5py,'File',lambda *a,**k:pytest.fail('Metadata attribution reopened science'))
    fact,=_explicit_user_parameter_facts(app.sessions,state,run,step,planned)
    assert fact.value['parameters']=={'kwargs':'explicit_user_instruction'}
    before=files(app)
    if location=='planned':planned=replace(planned,arguments=dict(function_name='tl.leiden',kwargs={'resolution':changed}))
    else:step=N(tool_name=planned.tool_name,step_id=planned.step_id,resolved_arguments=dict(function_name='tl.leiden',kwargs={'resolution':changed}))
    with pytest.raises(_Unavailable,match='session_parameter_binding_mismatch'):
        _explicit_user_parameter_facts(app.sessions,state,run,step,planned)
    assert files(app)==before


@pytest.mark.parametrize('changed',[None,'planned','resolved','type'])
def test_repeated_native_parameter_evidence_uses_each_step_value(app,monkeypatch,changed):
    from types import SimpleNamespace
    from agent.application.session_state import Interaction
    from agent.application.dialogue_execution import _bind_arguments
    from agent.application.turn_decisions import _scientific_arguments
    from agent.application.dialogue_evidence import _explicit_user_parameter_facts, _Unavailable
    import h5py
    text='Use resolution 0.7 for left and 1.0 for right.'
    rows=[]
    for sid,value in [('left','0.7'),('right','1.0')]:
        start=text.index(value)
        rows.append(dict(tool='run_scanpy',native_function='tl.leiden',argument='kwargs',step_id=sid,
            literal=value,start=start,end=start+3,
            members=[dict(name='resolution',kind='number',literal=value,start=start,end=start+3)]))
    values=_bind_arguments(app.registry,text,{},_scientific_arguments(rows))
    admitted=dict(kind='execute',operation='plan',request_id='science',inputs=values,arguments=rows)
    interaction=Interaction('turn',text,None,0,{},status='admitted',admitted=admitted)
    planned=tuple(PlanStep(sid,'run_scanpy',dict(function_name='tl.leiden',kwargs={'resolution':value}))
                  for sid,value in [('left',.7),('right',1.0)])
    run=SimpleNamespace(run_id='science:run',request_id='science',plan=SimpleNamespace(steps=planned))
    state=SimpleNamespace(turns=(SimpleNamespace(run_id=run.run_id,turn_id='turn'),),interactions=(interaction,))
    monkeypatch.setattr(h5py,'File',lambda *a,**k:pytest.fail('Parameter evidence reopened source science.'))
    for target in planned:
        step=SimpleNamespace(tool_name=target.tool_name,step_id=target.step_id,resolved_arguments=target.arguments)
        if changed=='planned':
            target=replace(target,arguments={**target.arguments,'kwargs':{'resolution':.9}})
        elif changed:
            step.resolved_arguments={**step.resolved_arguments,'kwargs':{'resolution':True if changed=='type' else .9}}
        if changed:
            with pytest.raises(_Unavailable,match='session_parameter_binding_mismatch'):
                _explicit_user_parameter_facts(app.sessions,state,run,step,target)
        else:
            fact,=_explicit_user_parameter_facts(app.sessions,state,run,step,target)
            assert fact.value==dict(tool='run_scanpy',step_id=target.step_id,parameters={'kwargs':'explicit_user_instruction'})
