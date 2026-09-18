import json
from pathlib import Path
import numpy as np
import anndata as ad
import pytest
from agent.application import ResearchAgentApplication
from agent.orchestration import AgentRequest, LLMPlanner, RunMode, build_default_tool_registry
from agent.tools.data import fragment_feature_tool as public

TOOL = 'build_scATAC_cell_by_features'


def wire(adapt=False):
    def inp(target,key):return dict(target=target,source=dict(kind='input',input=key))
    steps=[dict(step_id='matrix',tool=TOOL,control_dependencies=[],sources=[inp(p,k) for p,k in (
        ('fragments','fragments_manifest_path'),('explicit_cells','explicit_cells_manifest_path'),('reference','reference_manifest_path'))])]
    if adapt:steps.append(dict(step_id='adapt',tool='adapt_epizoo_species',control_dependencies=[],sources=[
        dict(target='matrix',source=dict(kind='step',step='matrix')),inp('specification','adaptation_spec_path'),inp('strategy','strategy')]))
    return dict(schema_version=4,decision=dict(kind='plan',steps=steps))


class Model:
    model_id='scripted-fragment-features'
    def __init__(self,payload):self.payload=payload;self.prompts=[]
    def complete(self,*,prompt,response_schema):
        self.prompts.append(prompt)
        if 'selection_schema_version' in response_schema.get('properties',{}):
            return json.dumps(dict(selection_schema_version=1,decision=dict(kind='select',capability_ids=['species_adaptation'])))
        return json.dumps(self.payload)


def inputs(args):return {k:v for k,v in args.items() if k!='output_dir'}


def test_application(factory,tmp_path,monkeypatch):
    from agent.tools.data import cell_by_ccre_verifier as owner
    args,_,_=factory();calls=[];original=owner.reconstruct
    def observed(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(owner,'reconstruct',observed)
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire())))
    run=app.run(AgentRequest('matrix','Build a zebrafish cell-by-peak matrix from these fragments and exact barcodes.',inputs(args)))
    assert run.status.value=='SUCCEEDED',(run.error,run.run_result.errors)
    assert calls==[1]
    assert run.evidence and run.report and not run.visualization
    result=run.run_result.steps[0].result
    matrix=ad.read_h5ad(result['matrix_path'])
    assert matrix.obs['barcode_identifier'].tolist()==['B','A','Z']
    assert matrix.var_names.tolist()==['chr1:30-40','chr1:0-20','chr1:10-25','chr1:90-100']
    np.testing.assert_array_equal(matrix.X.toarray(),[[1,1,1,0],[0,2,2,0],[0,0,0,0]])
    authority=run.run_result.steps[0].verification.artifact_authority
    assert authority['schema_version']==2 and authority['artifact_contract']=='scatac-cell-by-features.v1'
    assert set(authority['upstream'])=={'fragments','cells'}
    text=Path(run.report.path).read_text()
    assert 'no QC' in text and 'peak_set' in text and 'fragment-record' in text
    assert authority['execution_identity'] in text
    assert Path(result['manifest_path']).parent.parent.name.removeprefix('cell-by-ccre-') in text
    assert app.resume(run.run_id)==run and calls==[1]
    Path(args['explicit_cells_manifest_path']).write_bytes(b'changed')
    assert app.resume(run.run_id).status.value=='FAILED'


@pytest.mark.parametrize('adapt',[False,True])
def test_plan_only_no_scientific_io(tmp_path,monkeypatch,adapt):
    from agent.tools.data import fragment_feature_matrix as owner
    from agent.tools.data import _fragment_feature_binding as binding
    def forbidden(*a,**kw):pytest.fail('Scientific IO during PLAN_ONLY')
    monkeypatch.setattr(owner,'build_cell_by_features',forbidden)
    monkeypatch.setattr(binding,'bind',forbidden)
    monkeypatch.setattr(ad,'read_h5ad',forbidden)
    args={k:('a'*64 if k.endswith('_sha256') else '/PRIVATE/'+k) for k in public.ARGUMENTS if k!='output_dir'}
    if adapt:args.update(adaptation_spec_path='/PRIVATE/spec',adaptation_spec_sha256='b'*64,strategy='de_novo')
    model=Model(wire(adapt));app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(model))
    run=app.run(AgentRequest('plan','Build the target-species matrix and adapt EpiZoo de novo.' if adapt else 'Build a cell-by-peak matrix.',args,RunMode.PLAN_ONLY))
    assert run.status.value=='PLANNED',run.run_result.errors
    assert not run.report and not run.evidence
    assert '/PRIVATE' not in json.dumps(model.prompts)
    assert [s.tool_name for s in run.run_result.plan.steps]==([TOOL,'adapt_epizoo_species'] if adapt else [TOOL])
    if adapt:assert run.run_result.plan.steps[1].arguments['matrix_manifest_path'].step_id=='matrix'


def test_registry():
    registry=build_default_tool_registry();assert len(registry.names())==23
    spec=registry.get(TOOL)
    cell_fields = {'explicit_cells_manifest_path','explicit_cells_manifest_sha256'}
    assert set(spec.required_arguments)==set(public.ARGUMENTS)-cell_fields
    assert set(spec.optional_arguments)==cell_fields | {'selected_cells_manifest_path','selected_cells_manifest_sha256'}
    assert [p.name for p in spec.semantic_planning.consumer_ports]==['fragments','explicit_cells','selected_cells','reference','output_dir']
    assert spec.semantic_planning.producer_ports[0].semantic_type=='scatac_cell_by_features.v1'


@pytest.mark.parametrize('missing',['fragments_manifest_path','explicit_cells_manifest_path','reference_manifest_path'])
def test_missing_inputs(tmp_path,missing):
    args={k:('a'*64 if k.endswith('_sha256') else '/PRIVATE/'+k) for k in public.ARGUMENTS if k!='output_dir'}
    args.pop(missing)
    run=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire()))).run(
        AgentRequest('missing','Build from fragments and peaks.',args,RunMode.PLAN_ONLY))
    assert run.status.value=='FAILED'


@pytest.mark.parametrize('change',['qc_cells','legacy_fragment_producer','wrong_matrix_port','unknown_knob'])
def test_wrong_ports_fail_closed(tmp_path,change):
    args={k:('a'*64 if k.endswith('_sha256') else '/PRIVATE/'+k) for k in public.ARGUMENTS if k!='output_dir'}
    payload=wire(change=='wrong_matrix_port');step=payload['decision']['steps'][0]
    if change=='qc_cells':step['sources'][1]['target']='selected_cells'
    elif change=='unknown_knob':step['sources'].append(dict(target='count_mode',source=dict(kind='input',input='count_mode')));args['count_mode']='support'
    elif change=='wrong_matrix_port':
        payload['decision']['steps'][1]['sources'][0]['source']=dict(kind='step_port',step='matrix',source_port='dataset')
        args.update(adaptation_spec_path='/PRIVATE/spec',adaptation_spec_sha256='b'*64,strategy='de_novo')
    else:
        step['sources'][0]['source']=dict(kind='step',step='legacy')
        payload['decision']['steps'].insert(0,dict(step_id='legacy',tool='import_scATAC_fragments',sources=[],control_dependencies=[]))
    with pytest.raises(ValueError):LLMPlanner(Model(payload)).plan(AgentRequest('wrong','Build.',args,RunMode.PLAN_ONLY),build_default_tool_registry())


@pytest.mark.parametrize('change',['species','assembly','genome','namespace','absent','duplicate','qc_contract','hash'])
def test_invalid_prerequisites_precede_counting(factory,tmp_path,monkeypatch,change):
    from agent.tools.data import regulatory_feature_reference as ref, explicit_cells, _cell_by_ccre_production as production
    if change=='duplicate':
        with pytest.raises(ValueError):factory(cells=[('lib','A'),('lib','A')])
        return
    args,resources,_=factory()
    if change in ('namespace','absent'):
        cells=[('wrong','A')] if change=='namespace' else [('lib','absent')]
        ptr=explicit_cells.publish_explicit_cells(cells=cells,declaration='Caller supplied',output_dir=tmp_path/'other-cells')
        args.update(explicit_cells_manifest_path=ptr['manifest_path'],explicit_cells_manifest_sha256=ptr['manifest_sha256'])
    elif change in ('species','assembly','genome'):
        if change=='species':resources['species']={'scientific_name':'Macaca mulatta','taxonomy_id':9544}
        elif change=='assembly':resources['target_assembly']='different'
        else:
            fa=tmp_path/'other.fa';fa.write_text('>chr1\n'+'C'*100+'\n');resources['fasta_path']=fa
        ptr=ref.publish_regulatory_feature_reference(ref.build_regulatory_feature_reference(**resources),tmp_path/'other-ref.json')
        args.update(reference_manifest_path=ptr['manifest_path'],reference_manifest_sha256=ptr['manifest_sha256'])
    elif change=='hash':args['fragments_manifest_sha256']='0'*64
    else:
        p=tmp_path/'wrong.json';p.write_text('{"contract_version":"scatac-cell-selection.v1"}')
        import hashlib
        args.update(explicit_cells_manifest_path=str(p),explicit_cells_manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    calls=[];original=production.construct_counts
    def observed(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(production,'construct_counts',observed)
    run=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire()))).run(AgentRequest('bad','Build.',inputs(args)))
    assert run.status.value=='FAILED'
    # Missing records are rejected by the accepted engine's streaming validation;
    # reference/namespace/schema errors fail before the counting engine begins.
    if change!='absent':assert calls==[]
    assert not list((tmp_path/'app').rglob('matrix.h5ad'))


@pytest.mark.parametrize('change',['source','matrix','reference','cells','receipt'])
def test_mutation_rejects_authority_reuse(factory,tmp_path,change):
    args,resources,source= factory()
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire())))
    run=app.run(AgentRequest('mutation','Build.',inputs(args)));assert run.status.value=='SUCCEEDED'
    result=run.run_result.steps[0].result
    paths=dict(source=source['source_path'],matrix=result['matrix_path'],reference=resources['feature_bed_path'],
        cells=args['explicit_cells_manifest_path'],receipt=Path(result['manifest_path']).parent.parent/'receipt.json')
    path=Path(paths[change]);before=path.read_bytes();path.write_bytes(before+b'changed')
    assert app.resume(run.run_id).status.value=='FAILED'


def test_crash_after_publication_recovers_authority(factory,tmp_path,monkeypatch):
    from agent.orchestration import AgentRuntime,FileRunStore,StepStatus
    from agent.tools.data import _cell_by_ccre_production as production
    args,_,_=factory();store=FileRunStore(tmp_path/'store')
    class Crash(BaseException):pass
    class StopStore:
        def __getattr__(self,k):return getattr(store,k)
        def update(self,state,*,expected_revision):
            if any(s.status is StepStatus.SUCCEEDED for s in state.steps):raise Crash()
            return store.update(state,expected_revision=expected_revision)
    with pytest.raises(Crash):AgentRuntime(planner=LLMPlanner(Model(wire())),run_store=StopStore()).run(AgentRequest('crash','Build.',args))
    monkeypatch.setattr(production,'construct_counts',lambda *a,**k:pytest.fail('Recovery repeated production'))
    run=AgentRuntime(run_store=store).resume('crash:run')
    assert run.status.value=='SUCCEEDED',run.errors
    assert run.steps[0].verification.artifact_authority['schema_version']==2
    assert AgentRuntime(run_store=store).resume('crash:run').status.value=='SUCCEEDED'


@pytest.mark.parametrize('zero_row',[False,True])
def test_real_corpus_binding_preserves_fragment_authority(factory,zero_row):
    from agent.tools.models.epizoo_adaptation.preprocessing import corpus
    from agent.tools.data.authority_context import VerificationContext,authority_operation
    from agent.tools.data import scatac_matrix_contract as m
    args,_,_=factory(cells=[('lib','B'),('lib','A'),('lib','Z')] if zero_row else [('lib','B'),('lib','A')])
    with authority_operation(VerificationContext()):
        result=public.execute(args)
        binding=dict(manifest_path=result['manifest_path'],manifest_sha256=result['manifest_sha256'])
        if zero_row:
            with pytest.raises(ValueError,match='positive and inaccessible'):corpus([binding])
        else:
            joint,cells,manifests,reference=corpus([binding])
            np.testing.assert_array_equal(joint.X.toarray(),[[1,1,1,0],[0,2,2,0]])
            assert manifests[0]['contract_version']=='scatac-cell-by-features.v1'
            assert set(manifests[0]['upstream'])=={'fragments','cells','reference'}
            assert reference.species.scientific_name=='Danio rerio'
        assert m.load_manifest_bytes(Path(result['manifest_path']).read_bytes())['contract_version']=='scatac-cell-by-features.v1'


def test_public_adaptation_binding_without_model_execution(factory,tmp_path,monkeypatch):
    from agent.tools.models import species_adaptation as adapt
    from agent.tools.models.epizoo_adaptation import contract as c,resources
    args,_,_=factory(cells=[('lib','B'),('lib','A')]);result=public.execute(args)
    reference=json.loads(Path(args['reference_manifest_path']).read_text())
    bundle=c.write_json(tmp_path/'bundle.json',dict(code=dict(files=[])))
    pointer={k:bundle[k] for k in ('path','sha256')}
    monkeypatch.setattr(resources,'code_bundle',lambda:dict(files=[]))
    spec=dict(contract_version=adapt.SPECIFICATION,reference_identity_sha256=reference['reference_identity_sha256'],
        species=reference['species'],assembly=reference['target_assembly'],source_bundle=pointer,seam_bundle=pointer,
        mapping=None,execution_profile='qualification.v1')
    def expand(value):
        record=c.write_json(tmp_path/(c.digest(value)+'.json'),value)
        return adapt.expand(dict(matrix_manifest_path=result['manifest_path'],matrix_manifest_sha256=result['manifest_sha256'],
            adaptation_spec_path=record['path'],adaptation_spec_sha256=record['sha256'],strategy='de_novo',output_dir=str(tmp_path/'model')))
    expanded=expand(spec)
    assert expanded['matrices'][0]['manifest_sha256']==result['manifest_sha256']
    for change in (dict(species={'scientific_name':'Mus musculus','taxonomy_id':10090}),dict(assembly='wrong'),
                   dict(reference_identity_sha256='0'*64),dict(execution_profile='binary-de-novo-qualification.v1')):
        with pytest.raises(ValueError):expand(spec|change)


@pytest.mark.parametrize('change',[{'n_cells':True},{'matrix_semantics':'binary_accessibility'},
    {'contract_version':'scatac-cell-by-features.external.v1'}, {'species':'human'}, {'nnz':0}])
def test_strict_result_contract(factory,change):
    args,_,_=factory();result=public.execute(args)
    registry=build_default_tool_registry();registry.validate_result(TOOL,result)
    with pytest.raises(ValueError):registry.validate_result(TOOL,result|change)
    with pytest.raises(ValueError):registry.validate_result(TOOL,result|{'diagnostic':{
        k.replace('any_feature','any_ccre'):v for k,v in result['diagnostic'].items()}})


def test_v3_static_bindings(tmp_path):
    from agent.orchestration import PlanningWireMode
    args={k:('a'*64 if k.endswith('_sha256') else '/PRIVATE/'+k) for k in public.ARGUMENTS}
    class V3:
        model_id='v3-fragments'
        def complete(self,**kw):return json.dumps(dict(schema_version=3,status='plan',reason=None,steps=[
            dict(step_id='matrix',tool_name=TOOL,arguments={k:dict(binding_type='input',input_name=k) for k in args} |
                {'selected_cells_manifest_path':None,'selected_cells_manifest_sha256':None},depends_on=[],description='Build.')]))
    plan=LLMPlanner(V3(),wire_mode=PlanningWireMode.V3).plan(AgentRequest('v3','Build.',args,RunMode.PLAN_ONLY),build_default_tool_registry())
    assert dict(plan.steps[0].arguments)==args


def test_cancelled_public_run_does_not_publish(factory,tmp_path,monkeypatch):
    from agent.tools.data import _cell_by_ccre_production as production
    from agent.tools._cancellation import cancellation_checkpoint
    args,_,_=factory()
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire())))
    def cancel(*a,**kw):app.cancel('cancel:run');cancellation_checkpoint()
    monkeypatch.setattr(production,'construct_counts',cancel)
    run=app.run(AgentRequest('cancel','Build.',inputs(args)))
    assert run.status.value=='CANCELLED'
    assert not list((tmp_path/'app').rglob('matrix.h5ad'))
    assert not list((tmp_path/'app').rglob('.matrix-attempt-*'))
    assert app.resume(run.run_id).status.value=='CANCELLED'
