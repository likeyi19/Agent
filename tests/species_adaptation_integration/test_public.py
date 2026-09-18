from pathlib import Path
import json
import pytest
from agent.application import ResearchAgentApplication
from agent.orchestration import AgentRequest,AgentPlan,PlanStep,RunMode,LLMPlanner,build_default_tool_registry
from agent.tools.data import neutral_matrix_tool as public


class Planner:
    def __init__(self,args):self.args=args
    def plan(self,request,registry):
        return AgentPlan('neutral-plan',request.request_id,'Exact neutral adoption.',(
            PlanStep('adopt','adopt_scATAC_cell_by_features',self.args),))


def public_args(case):return {k:case[k] for k in public.ARGUMENTS}


def test_adoption_application(case,tmp_path,monkeypatch):
    from agent.tools.data import cell_by_ccre_verifier as owner
    calls=[];original=owner._verify_external
    def observed(*a,**k):calls.append(1);return original(*a,**k)
    monkeypatch.setattr(owner,'_verify_external',observed)
    app=ResearchAgentApplication(tmp_path/'app',planner=Planner(public_args(case)))
    result=app.run(AgentRequest('neutral','Adopt this explicit peak matrix.',{}))
    if result.error:
        from agent.report.evidence import build_analysis_evidence
        from agent.orchestration.verification_authority import accepted_authorities
        with accepted_authorities(app.run_store,result.run_id):
            build_analysis_evidence(result.run_result,tmp_path/'debug-evidence',registry=build_default_tool_registry())
    assert result.status.value=='SUCCEEDED',(result.run_result.errors,result.error)
    assert calls==[1]
    assert result.evidence and result.report and not result.visualization
    report_text=Path(result.report.path).read_text()
    assert 'curated-cCRE' in report_text and 'Artifact contract' in report_text
    assert 'Fragment contract' not in report_text
    monkeypatch.setattr(owner,'_verify_external',lambda *a,**k:pytest.fail('Repeated source conservation'))
    Path(case['source_path']).unlink()
    assert app.resume(result.run_id)==result
    with Path(result.run_result.steps[0].result['matrix_path']).open('ab') as f:f.write(b'mutated')
    assert app.resume(result.run_id).status.value=='FAILED'


class Model:
    model_id='scripted-species-integration'
    def __init__(self,payload):self.payload=payload;self.calls=[]
    def complete(self,*,prompt,response_schema):
        self.calls.append(prompt)
        if 'selection_schema_version' in response_schema.get('properties',{}):
            return json.dumps(dict(selection_schema_version=1,decision=dict(kind='select',capability_ids=['species_adaptation'])))
        return json.dumps(self.payload)


def wire():
    def inp(target,key):return dict(target=target,source=dict(kind='input',input=key))
    return dict(schema_version=4,decision=dict(kind='plan',steps=[
        dict(step_id='adopt',tool='adopt_scATAC_cell_by_features',sources=[inp('source','source_path'),inp('reference','reference_manifest_path'),inp('matrix_semantics','matrix_semantics')],control_dependencies=[]),
        dict(step_id='adapt',tool='adapt_epizoo_species',sources=[dict(target='matrix',source=dict(kind='step_port',step='adopt',source_port='matrix')),
            inp('specification','adaptation_spec_path'),inp('strategy','strategy')],control_dependencies=[])]))


@pytest.mark.parametrize('text',['Adapt EpiZoo to this new species using this cell-by-peak matrix.',
                                  'Use this zebrafish matrix and genome to post-train EpiZoo.'])
def test_plan_only(case,tmp_path,monkeypatch,text):
    import h5py
    from agent.tools.models.epizoo_adaptation import resources,backend
    monkeypatch.setattr(h5py,'File',lambda *a,**k:pytest.fail('PLAN_ONLY opened matrix'))
    monkeypatch.setattr(resources,'load_bundle',lambda *a,**k:pytest.fail('PLAN_ONLY loaded model'))
    monkeypatch.setattr(backend,'sequences_and_embeddings',lambda *a,**k:pytest.fail('PLAN_ONLY ran SEAM'))
    inputs=public_args(case);inputs.pop('output_dir')
    inputs.update(adaptation_spec_path='/private/DO-NOT-PROMPT.json',adaptation_spec_sha256='a'*64,strategy='de_novo')
    model=Model(wire());app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(model))
    result=app.run(AgentRequest('plan',text,inputs,RunMode.PLAN_ONLY))
    assert result.status.value=='PLANNED',result.run_result.errors
    assert not result.evidence and not result.report
    assert 'DO-NOT-PROMPT' not in json.dumps(model.calls)
    assert [s.tool_name for s in result.run_result.plan.steps]==['adopt_scATAC_cell_by_features','adapt_epizoo_species']


def test_registry_scope():
    from agent.orchestration.planning_scope import capability_index
    r=build_default_tool_registry();assert len(r.names())==22
    index=capability_index(r)
    assert set(index['species_adaptation'])=={'adopt_scATAC_cell_by_features','adapt_epizoo_species'}
    assert 'adapt_epizoo_species' not in index['embedding_analysis']


@pytest.mark.parametrize('integration',['fragment_counts','binary_accessibility'],indirect=True)
def test_adaptation_application(integration,tmp_path,monkeypatch):
    from agent.tools.data import cell_by_ccre_verifier as owner
    inputs,counts,resource=integration
    matrix_calls=[];original=owner._verify_external
    def observed(*a,**kw):matrix_calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(owner,'_verify_external',observed)
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire())))
    result=app.run(AgentRequest('adapt','Adapt EpiZoo to the supplied species.',inputs))
    if result.error:
        from agent.orchestration.verification_authority import accepted_authorities
        from agent.report.evidence import build_analysis_evidence
        with accepted_authorities(app.run_store,result.run_id):
            build_analysis_evidence(result.run_result,tmp_path/'debug',registry=build_default_tool_registry())
    assert result.status.value=='SUCCEEDED',(result.run_result.errors,result.error)
    assert counts==dict(production=1,owner=1) and matrix_calls==[1]
    assert result.report and result.evidence and not result.visualization
    assert 'applied optimizer steps' in Path(result.report.path).read_text()
    assert all(s.verification.artifact_authority['schema_version']==2 for s in result.run_result.steps)
    assert app.resume(result.run_id)==result
    assert counts==dict(production=1,owner=1) and matrix_calls==[1]
    resource.write_text('changed resource')
    assert app.resume(result.run_id).status.value=='FAILED'


@pytest.mark.parametrize('missing',['reference_manifest_path','adaptation_spec_path','strategy'])
def test_missing_request_binding_fails_plan(case,tmp_path,missing):
    inputs=public_args(case);inputs.pop('output_dir')
    inputs.update(adaptation_spec_path='/missing/spec.json',adaptation_spec_sha256='a'*64,strategy='de_novo')
    inputs.pop(missing)
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire())))
    result=app.run(AgentRequest('missing','Adapt EpiZoo.',inputs,RunMode.PLAN_ONLY))
    assert result.status.value=='FAILED'


@pytest.mark.parametrize('change',['mapping','assembly','species','reference','binary','spec_hash'])
def test_incompatible_resources_fail_before_backend(integration,tmp_path,change):
    inputs,counts,_=integration
    if change=='mapping':inputs['strategy']='mapped_reference'
    elif change=='binary':inputs['matrix_semantics']='binary_accessibility'
    elif change=='spec_hash':inputs['adaptation_spec_sha256']='0'*64
    else:
        from agent.tools.models.epizoo_adaptation import contract as c
        path=Path(inputs['adaptation_spec_path']);value=c.read_json(path)
        if change=='assembly':value['assembly']='wrong'
        elif change=='species':value['species']['scientific_name']='Danio rerio'
        else:value['reference_identity_sha256']='0'*64
        path.write_bytes(c.canonical(value));inputs['adaptation_spec_sha256']=c.file_record(path)['sha256']
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire())))
    result=app.run(AgentRequest('bad','Adapt with explicit resources.',inputs))
    assert result.status.value=='FAILED'
    assert counts==dict(production=0,owner=0)


def test_wrong_upstream_port(case,tmp_path):
    inputs=public_args(case);inputs.pop('output_dir')
    inputs.update(adaptation_spec_path='/missing/spec.json',adaptation_spec_sha256='a'*64,strategy='de_novo')
    value=wire();value['decision']['steps'][1]['sources'][0]['source']['source_port']='dataset'
    result=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(value))).run(
        AgentRequest('wrong-port','Adapt EpiZoo.',inputs,RunMode.PLAN_ONLY))
    assert result.status.value=='FAILED'


@pytest.mark.parametrize('integration',['fragment_counts','binary_accessibility'],indirect=True)
def test_recover_publication_after_crash(integration,tmp_path,monkeypatch):
    from agent.orchestration import AgentRuntime,FileRunStore,StepStatus
    from agent.tools.models.epizoo_adaptation import publication as backend
    inputs,counts,_=integration
    store=FileRunStore(tmp_path/'store')
    class Crash(BaseException):pass
    class StopStore:
        def __getattr__(self,name):return getattr(store,name)
        def update(self,state,*,expected_revision):
            if any(s.tool_name=='adapt_epizoo_species' and s.status is StepStatus.SUCCEEDED for s in state.steps):raise Crash()
            return store.update(state,expected_revision=expected_revision)
    # Supply a managed output root through normal request mechanics.
    inputs['output_dir']=str(tmp_path/'outputs')
    with pytest.raises(Crash):
        AgentRuntime(planner=LLMPlanner(Model(wire())),run_store=StopStore()).run(AgentRequest('crash','Adapt.',inputs))
    monkeypatch.setattr(backend,'_build',lambda *a:pytest.fail('Recovery repeated production'))
    result=AgentRuntime(run_store=store).resume('crash:run')
    assert result.status.value=='SUCCEEDED',result.errors
    assert counts['production']==1
    assert result.steps[-1].verification.artifact_authority['schema_version']==2


def test_cancelled_training_is_not_resumed(integration,tmp_path,monkeypatch):
    from agent.tools.models.epizoo_adaptation import publication as backend
    from agent.tools._cancellation import ToolWorkCancelled
    inputs,counts,_=integration
    def cancelled(args,root):
        root.mkdir()
        app.cancel('cancel:run')
        from agent.tools._cancellation import cancellation_checkpoint
        cancellation_checkpoint()
    monkeypatch.setattr(backend,'_build',cancelled)
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire())))
    result=app.run(AgentRequest('cancel','Adapt.',inputs))
    assert result.status.value=='CANCELLED',result.run_result.errors
    monkeypatch.setattr(backend,'_build',lambda *a:pytest.fail('Cancellation silently restarted training'))
    assert app.resume(result.run_id).status.value=='CANCELLED'
    assert not list((tmp_path/'app').rglob('.matrix-attempt-*'))


@pytest.mark.parametrize('species',['human','mouse'])
def test_existing_embedding_request_not_redirected(tmp_path,species):
    class EmbeddingModel:
        model_id='scripted-existing-embedding'
        def complete(self,*,prompt,response_schema):
            if 'selection_schema_version' in response_schema.get('properties',{}):
                return json.dumps(dict(selection_schema_version=1,decision=dict(kind='select',capability_ids=['embedding_analysis'])))
            assert 'adapt_epizoo_species' not in prompt
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='embed',tool='epizoo_embed_cells',
                sources=[dict(target=target,source=dict(kind='input',input=key)) for target,key in [('dataset','input_path'),('species','species')]],control_dependencies=[])])))
    result=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(EmbeddingModel())).run(
        AgentRequest('embed',f'Embed this {species} scATAC matrix using EpiZoo.',dict(input_path='/explicit/matrix.h5ad',species=species),RunMode.PLAN_ONLY))
    assert result.status.value=='PLANNED',result.run_result.errors
    assert [s.tool_name for s in result.run_result.plan.steps]==['epizoo_embed_cells']


def test_v3_direct_semantics(case,tmp_path):
    from agent.orchestration import PlanningWireMode
    args=public_args(case)
    class V3Model:
        model_id='scripted-v3-neutral'
        def complete(self,**kw):
            return json.dumps(dict(schema_version=3,status='plan',reason=None,steps=[dict(step_id='adopt',tool_name='adopt_scATAC_cell_by_features',
                arguments={k:dict(binding_type='input',input_name=k) for k in args},depends_on=[],description='Exact neutral adoption.')]))
    plan=LLMPlanner(V3Model(),wire_mode=PlanningWireMode.V3).plan(AgentRequest('v3','Adopt this exact matrix.',args,RunMode.PLAN_ONLY),build_default_tool_registry())
    assert dict(plan.steps[0].arguments)==args


@pytest.mark.parametrize('key',['source_bundle','seam_bundle'])
def test_missing_bundle_fails_before_execution(integration,tmp_path,key):
    from agent.tools.models.epizoo_adaptation import contract as c
    inputs,counts,_=integration
    spec=c.read_json(inputs['adaptation_spec_path'])
    Path(spec[key]['path']).unlink()
    result=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire()))).run(
        AgentRequest('missing-bundle','Adapt.',inputs))
    assert result.status.value=='FAILED'
    assert counts==dict(production=0,owner=0)


def test_backend_revision_change_invalidates_accepted_reuse(integration,tmp_path,monkeypatch):
    from agent.tools.models.epizoo_adaptation import resources
    inputs,counts,_=integration
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire())))
    result=app.run(AgentRequest('revision','Adapt.',inputs))
    assert result.status.value=='SUCCEEDED'
    monkeypatch.setattr(resources,'code_bundle',lambda:dict(files=[],revision='different'))
    assert app.resume(result.run_id).status.value=='FAILED'
    assert counts==dict(production=1,owner=1)


@pytest.mark.parametrize('integration',['binary_accessibility'],indirect=True)
@pytest.mark.parametrize('change',['fragment_profile','fragment_declaration','insertion_declaration','mapped','unknown_profile'])
def test_binary_public_mismatches_fail_before_adaptation(integration,tmp_path,change):
    from agent.tools.models.epizoo_adaptation import contract as c
    inputs,counts,_=integration
    path=Path(inputs['adaptation_spec_path']);spec=c.read_json(path)
    if change=='fragment_profile':spec['execution_profile']='qualification.v1'
    elif change=='unknown_profile':spec['execution_profile']='binary-production-candidate.v1'
    elif change=='fragment_declaration':inputs['matrix_semantics']='fragment_counts'
    elif change=='insertion_declaration':inputs['matrix_semantics']='insertion_counts'
    elif change=='mapped':inputs['strategy']='mapped_reference'
    path.write_bytes(c.canonical(spec));inputs['adaptation_spec_sha256']=c.file_record(path)['sha256']
    result=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire()))).run(
        AgentRequest('binary-mismatch','Adapt this explicitly bound binary matrix.',inputs))
    assert result.status.value=='FAILED'
    assert counts==dict(production=0,owner=0)


@pytest.mark.parametrize('integration',['binary_accessibility'],indirect=True)
def test_binary_profile_binds_receipt_authority_and_result(integration,tmp_path):
    from copy import deepcopy
    from agent.tools.models.epizoo_adaptation import contract as c,publication
    from agent.orchestration.species_adaptation_registry import validate_adaptation
    from agent.orchestration.registry import ToolResultContractError
    inputs,_,_=integration
    result=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire()))).run(
        AgentRequest('binary-binding','Adapt.',inputs))
    assert result.status.value=='SUCCEEDED'
    step=result.run_result.steps[-1];value=dict(step.result)
    assert value['profile_sha256']==c.BINARY_PROFILE_SHA256
    manifest=publication.load_manifest(value['manifest_path'],value['manifest_sha256'])
    assert manifest['arguments']['profile']['matrix_semantics']=='binary_accessibility'
    receipt=c.read_json(Path(value['manifest_path']).parent.parent/'receipt.json')
    assert receipt['profile_sha256']==c.BINARY_PROFILE_SHA256
    authority=step.verification.artifact_authority
    assert authority['science_profile']==c.BINARY_PROFILE_SHA256
    for change in ('mapped','production','hash'):
        forged=dict(value)
        if change=='mapped':forged.update(strategy='mapped_reference',mapping_identity_sha256='f'*64)
        elif change=='production':forged['purpose']='production'
        else:forged['profile_sha256']='f'*64
        with pytest.raises(ToolResultContractError):validate_adaptation(forged)
    forged=deepcopy(manifest);forged['profile_sha256']=c.PROFILE_SHA256
    bad=c.write_json(tmp_path/'forged-profile.json',forged)
    with pytest.raises(ValueError,match='Unsupported target-model contract'):
        publication.load_manifest(bad['path'],bad['sha256'])
