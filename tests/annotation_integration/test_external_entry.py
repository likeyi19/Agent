"""Direct external peak input qualification; synthetic native backend is labelled."""
import json
from pathlib import Path
import anndata as ad
import pytest
from agent.application import ResearchAgentApplication
from agent.orchestration import AgentRequest, AgentPlan, PlanStep
from agent.schemas import StepOutputRef
from agent.tools.analysis import annotation_contract as c, marker_annotation as science
from agent.tools.analysis import external_marker_annotation as external
from test_annotation import case
_DIRECT_RUNNER = external.annotate_cell_groups
from fastapi.testclient import TestClient
from agent.application import InteractiveAgentApplication
from agent.application.input_sets import ScientificInputSet
from agent.application.uploads import H5ADUploadAdmission
from agent.providers import PlanningModelFactoryRegistry
from agent.orchestration import PlanningModelProfile
from agent.web.app import create_app
import time
import io
import h5py

ROLES = [dict(tool='annotate_scATAC_cell_types', port='source', source='source.0'),
         dict(tool='annotate_scATAC_cell_types', port='groups', source='source.1')]
OBJECTIVE = 'Annotate my external human hg38 PBMC Cell Ranger ATAC cut-site counts using MAESTRO. I will supply independent groups.'


def wire():
    pairs=[('source','source_path'),('specification','annotation_spec_path'),('groups','groups_path'),
           ('annotation_method','annotation_method'),('count_slot','count_slot')]
    return dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='annotation',
        tool='annotate_scATAC_cell_types',control_dependencies=[],sources=[dict(target=t,source=dict(kind='input',input=k)) for t,k in pairs])]))


class Model:
    model_id = 'scripted-external-annotation'
    def __init__(self, prompts): self.prompts = prompts
    def complete(self, *, prompt, response_schema):
        p = json.loads(prompt); self.prompts.append(p)
        if 'turn_schema_version' in p:
            decision = (dict(kind='request_roles', target='annotate_scATAC_cell_types',
                required_roles=[{k: r[k] for k in ('tool', 'port')} for r in ROLES],
                roles=ROLES[:1], arguments=[], context='none') if p['utterance'] == OBJECTIVE else
                dict(kind='answer_roles', pending='@roles', roles=ROLES[1:], arguments=[]))
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        if 'selection_schema_version' in p:
            return json.dumps(dict(selection_schema_version=1, decision=dict(kind='select',
                capability_ids=['marker_annotation'])))
        if 'output_selection_schema_version' in p:
            return json.dumps(dict(outputs=[dict(name='annotation', step_id='annotation', output_key='manifest_path')]))
        return json.dumps(wire())


def wait_turn(client, turn):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        value = client.get('/api/v1/sessions/upload/turns/' + turn).json()
        if value.get('response') is not None and value['status'] not in (
                'pending', 'running', 'processing', 'submitted', 'admitted', 'interpreting'):
            return value
        time.sleep(.02)
    pytest.fail(str(value))


def test_uploaded_matrix_then_group_file_continues_original_objective(external_case, tmp_path):
    annotation, calls = external_case
    prompts = []
    root = tmp_path/'web'; upload_root = root/'uploads'; upload_root.mkdir(parents=True)
    profile = PlanningModelProfile('scripted', 'scripted', Model.model_id)
    app = InteractiveAgentApplication(root, model_profiles=(profile,), default_profile_id='scripted',
        approved_source_roots=(upload_root,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:Model(prompts)}))
    inputs = {k:v for k,v in annotation.items() if k.startswith('annotation_') or k == 'count_slot'}
    choice = ScientificInputSet('reviewed', 'Human hg38 synthetic cut-site counts; independent groups', inputs)
    uploads = H5ADUploadAdmission(app.resources, upload_root)
    with TestClient(create_app(app, input_sets=(choice,), uploads=uploads)) as client:
        assert client.post('/api/v1/sessions',json={'session_id':'upload'}).status_code == 201
        response = client.put('/api/v1/uploads/matrix?filename=external.h5ad&input_type=h5ad',
            content=Path(annotation['source_path']).read_bytes(), headers={'Content-Type':'application/octet-stream'})
        assert response.status_code == 201, response.text
        matrix_id = response.json()['resource_id']
        response = client.post('/api/v1/sessions/upload/turns',json=dict(turn_id='first', expected_generation=0,
            utterance=OBJECTIVE, resource_ids=[matrix_id], input_set_id='reviewed'))
        assert response.status_code == 202, response.text
        first = wait_turn(client, 'first')
        assert first['status'] == 'clarification', first
        state = app._application.sessions.load('upload')
        assert state.interactions[-1].prerequisite['field'] == 'roles'
        response = client.put('/api/v1/uploads/groups?filename=groups.tsv&input_type=groups',
            content=Path(annotation['groups_path']).read_bytes(), headers={'Content-Type':'application/octet-stream'})
        assert response.status_code == 201, response.text
        groups_id = response.json()['resource_id']
        response = client.post('/api/v1/sessions/upload/turns',json=dict(turn_id='answer', expected_generation=0,
            utterance='The second file is the exact ordered independent cell grouping TSV for this matrix.',
            resource_ids=[matrix_id, groups_id], input_set_id='reviewed'))
        assert response.status_code == 202, response.text
        result = wait_turn(client, 'answer')
        assert result['status'] == 'succeeded', result
        run = app._application.run_store.load(result['run_id'])
        assert run.request.prompt == OBJECTIVE
        assert len(calls) == 1
        base = '/api/v1/sessions/upload/revisions/' + result['revision_id']
        response = client.get(base + '/scientific-artifacts')
        assert response.status_code == 200, response.text
        artifact, = response.json()['artifacts']
        downloaded = client.get(base + '/scientific-artifacts/' + artifact['handle'])
        assert downloaded.status_code == 200, downloaded.text
        with h5py.File(io.BytesIO(downloaded.content)) as exported:
            assert 'matrix_manifest_path' not in exported['uns/cell_type_annotation']
            assert 'source_path' not in exported['uns/cell_type_annotation']
            assert exported['uns/measurement'].asstr()[()] == c.EXTERNAL_SEMANTICS
            assert exported['uns/cell_type_annotation/rp_n_features'][()] > 0
            assert 'Simulated' in exported['uns/cell_type_annotation/rp_interpretation'].asstr()[()]
            assert 'supplied_groups' in exported['obs'] and 'raw_counts' in exported['layers']
        detail = client.get(base + '/evidence?output_name=annotation&detail_section=annotation_rationale')
        assert detail.status_code == 200
        assert 'feature_participation.n_features' in detail.text
        assert 'Simulated participation' in detail.text
        assert all(str(upload_root) not in json.dumps(p) for p in prompts)


@pytest.fixture
def external_case(case,tmp_path,monkeypatch):
    from scipy import sparse
    from agent.tools.analysis import scatac_annotation as owner
    args,_,_,run=case
    source=ad.read_h5ad(run.steps[-1].result['matrix_path'])
    source.uns=dict(species='human',assembly='hg38',measurement=c.EXTERNAL_SEMANTICS,
                    fixture='synthetic counts only; no biological qualification')
    source.obs['supplied_groups']=['first']+['second']*(source.n_obs-1)
    source.layers['raw_counts']=source.X.copy()
    source.obsm['existing_embedding']=__import__('numpy').zeros((source.n_obs,2))
    path=tmp_path/'external.h5ad';source.write_h5ad(path,convert_strings_to_categoricals=False)
    old=json.loads(Path(args['annotation_spec_path']).read_bytes())
    qc=next(s.resolved_arguments for s in run.steps if 'qc_reference_manifest_path' in s.resolved_arguments)
    spec={k:v for k,v in old.items() if k not in ('source_run_id','source_step_id','groups')}
    spec.update(schema=c.EXTERNAL_SPEC,method=c.EXTERNAL_PRIMARY,measurement=c.EXTERNAL_SEMANTICS,
        coordinate_system='zero-based-half-open',grouping_provenance='Explicit independently supplied synthetic grouping',
        contig_scope=dict(path=qc['qc_reference_manifest_path'],sha256=qc['qc_reference_manifest_sha256']))
    resource=tmp_path/'external-resources.json';resource.write_bytes(c.matrix.canonical(spec))
    groups=tmp_path/'external-groups.tsv'
    import pandas as pd
    pd.DataFrame(dict(cell_id=source.obs_names,group=source.obs['supplied_groups'])).to_csv(groups,sep='\t',index=False)
    annotation=dict(source_path=str(path),source_sha256=science.sha256(path),annotation_spec_path=str(resource),
        annotation_spec_sha256=science.sha256(resource),annotation_method=c.EXTERNAL_PRIMARY,count_slot='X',
        groups_path=str(groups),groups_sha256=science.sha256(groups),output_dir=str(tmp_path/'new-annotation'))
    calls=[]
    def component(*,binding,specification,runtime,output_dir, _matrix=None):
        calls.append(1);a = _matrix if _matrix is not None else external.load_matrix(binding)[0];rows=external.group_rows(a,specification)
        scores=science.score_candidates(a.obs_names.tolist(),rows,['G','H'],[(rows[0][1],'G',1.)],science.read_signatures(specification['signature_resource']['path']))
        output=Path(output_dir);output.mkdir()
        for key,values in scores.items():(output/(key+'.json')).write_text(json.dumps(values))
        sparse.save_npz(output/'rp.npz',sparse.csr_matrix([[1.,0.]]*a.n_obs))
        for name in ('markers-native.tsv.gz','markers-maestro-filtered.tsv','markers-signature-input.tsv','group-sizes.tsv'):
            pd.DataFrame({'group':[rows[0][1]],'gene':['G'],'effect':[1.]}).to_csv(output/name,sep='\t',index=False)
        (output/'genes.tsv').write_text('G\nH\n');(output/'cells.tsv').write_text('\n'.join(a.obs_names)+'\n')
        pd.DataFrame(rows,columns=['cell_id','group']).to_csv(output/'groups.tsv',sep='\t',index=False)
        (output/'marker-runtime.txt').write_text('synthetic native backend only\n')
        # Synthetic native output includes the new publication observation. Real
        # weight mathematics is exercised separately with the pinned backend.
        participation = dict(n_features=a.n_vars, unmatched_contig_features=0,
            unmatched_contig_counts=0, zero_transcript_weight_features=0,
            zero_transcript_weight_counts=0, interpretation='Simulated participation; not native scientific evidence.')
        pd.DataFrame(dict(feature=a.var_names, supplied_counts=__import__('numpy').asarray(a.X.sum(axis=0)).ravel(),
            exact_gene_contig_present=True, native_transcript_weight_nnz=1)).to_csv(
                output/'feature-participation.tsv', sep='\t', index=False)
        value=dict(binding=binding,inputs=specification,feature_participation=participation,
            sidecars={p.name:science.sha256(p) for p in output.iterdir()})
        (output/'result.json').write_text(json.dumps(value));return value
    monkeypatch.setattr(external,'annotate_cell_groups',component)
    return annotation,calls


class Planner:
    def __init__(self,args):self.args=args
    def plan(self,request,registry):
        return AgentPlan('external-annotation',request.request_id,'Explicit direct peak annotation.',(PlanStep('annotation','annotate_scATAC_cell_types',self.args),))


def test_external_owner_and_terminal_reuse(external_case,tmp_path,monkeypatch):
    args,calls=external_case
    app=ResearchAgentApplication(tmp_path/'app',planner=Planner(args))
    result=app.run(AgentRequest('external','Annotate my declared cut-site peak counts.',{}))
    assert result.status.value=='SUCCEEDED',result
    assert len(calls)==1
    step=result.run_result.steps[0]
    assert step.result['annotation_profile']==c.EXTERNAL_PRIMARY
    assert 'matrix' not in step.verification.artifact_authority['upstream']
    output=ad.read_h5ad(step.result['annotated_h5ad_path']);original=ad.read_h5ad(args['source_path'])
    assert (original.layers['raw_counts']!=output.layers['raw_counts']).nnz==0
    assert original.var.equals(output.var)
    monkeypatch.setattr(external,'annotate_cell_groups',lambda **kw:pytest.fail('Terminal replayed science'))
    assert app.resume(result.run_id)==result


@pytest.mark.parametrize('semantics',['fragment_counts','binary_accessibility','tfidf'])
def test_external_incompatible_semantics_rejected(external_case,semantics):
    args,calls=external_case
    path=Path(args['annotation_spec_path']);spec=json.loads(path.read_text());spec['measurement']=semantics
    path.write_bytes(c.matrix.canonical(spec));args['annotation_spec_sha256']=science.sha256(path)
    with pytest.raises(ValueError):c.resolved_specification(args)
    assert not calls


def test_explicit_group_column_and_count_layer(external_case,tmp_path):
    args,calls=external_case
    args={k:v for k,v in args.items() if not k.startswith('groups_')}
    args.update(group_column='supplied_groups',count_slot='layers/raw_counts')
    result=ResearchAgentApplication(tmp_path/'app',planner=Planner(args)).run(AgentRequest('columns','Annotate selected counts and groups.',{}))
    assert result.status.value=='SUCCEEDED',result
    assert len(calls)==1
    assert c.load_manifest(result.run_result.steps[0].result['manifest_path'])['matrix_binding']['count_slot']=='layers/raw_counts'


@pytest.mark.parametrize('problem',['negative','fractional','nonfinite','overflow','duplicate','coordinate','group','assembly','metadata','source_hash','group_hash','one_group','missing_column'])
def test_direct_input_rejects(external_case,problem):
    import numpy as np
    args,calls=external_case
    a=ad.read_h5ad(args['source_path'])
    if problem in ('negative','fractional','nonfinite','overflow'):
        a.X=a.X.astype(np.float64);a.X.data[0]={'negative':-1,'fractional':.5,'nonfinite':np.inf,'overflow':2**53}[problem]
    elif problem=='duplicate':a.obs_names=[a.obs_names[0]]*a.n_obs
    elif problem=='coordinate':a.var_names=['chrM:2-1']+list(a.var_names[1:])
    elif problem=='metadata':a.uns['measurement']='binary_accessibility'
    elif problem=='one_group':
        args={k:v for k,v in args.items() if not k.startswith('groups_')};args['group_column']='supplied_groups';a.obs['supplied_groups']='one'
    elif problem=='missing_column':
        args={k:v for k,v in args.items() if not k.startswith('groups_')};args['group_column']='absent'
    a.write_h5ad(args['source_path'],convert_strings_to_categoricals=False);args['source_sha256']=science.sha256(args['source_path'])
    if problem=='source_hash':args['source_sha256']='0'*64
    if problem=='group_hash':args['groups_sha256']='0'*64
    if problem=='group':
        p=Path(args['groups_path']);p.write_text(p.read_text().replace(a.obs_names[0],'unknown'));args['groups_sha256']=science.sha256(p)
    if problem=='assembly':
        p=Path(args['annotation_spec_path']);v=json.loads(p.read_text());v['assembly']='hg19';p.write_bytes(c.matrix.canonical(v));args['annotation_spec_sha256']=science.sha256(p)
    with pytest.raises((ValueError,KeyError)):
        external.qualify(c.arguments(args),c.resolved_specification(args))
    assert not calls


def test_method_cannot_be_selected_by_group_argument(external_case):
    args,_=external_case
    args.pop('annotation_method')
    with pytest.raises(ValueError):c.arguments(args)


def test_owner_replay_rejects_modified_output(external_case,tmp_path):
    from agent.tools.analysis import scatac_annotation as owner
    args,calls=external_case
    result=owner.execute_annotation(args,'tamper')
    p=Path(result['annotated_h5ad_path']);a=ad.read_h5ad(p);a.layers['raw_counts'].data[0]+=1;a.write_h5ad(p)
    with pytest.raises(ValueError):owner.verify_annotation(result['manifest_path'],expected_sha256=result['manifest_sha256'])


def test_private_metadata_is_not_a_scientific_admission_gate(external_case):
    args,calls=external_case
    a=ad.read_h5ad(args['source_path']);a.uns['private']='/private/operator/config';a.write_h5ad(args['source_path'])
    args['source_sha256']=science.sha256(args['source_path'])
    external.qualify(args,c.resolved_specification(args))
    assert not calls


def test_exact_count_and_group_declarations_use_shared_admission():
    from agent.orchestration import build_default_tool_registry
    from agent.application.dialogue_execution import scientific_parameters, _bind_arguments
    from agent.application.turn_decisions import ScopedArgument, IntentError
    registry=build_default_tool_registry()
    offered={r['argument']:r for r in scientific_parameters(registry) if r['tool']=='annotate_scATAC_cell_types'}
    assert set(offered)=={'count_slot','group_column'}
    assert all(r['exact_string'] for r in offered.values())
    text='Use "layers/raw_counts" and the obs column "supplied_groups".'
    declarations=[]
    for key,value in (('count_slot','layers/raw_counts'),('group_column','supplied_groups')):
        start=text.index(value);declarations.append(ScopedArgument('annotate_scATAC_cell_types',key,value,start,start+len(value)))
    assert _bind_arguments(registry,text,{},declarations)==dict(count_slot='layers/raw_counts',group_column='supplied_groups')
    with pytest.raises(IntentError):_bind_arguments(registry,text,{'group_column':'other'},declarations)


def test_direct_native_child_cancellation_reaps(external_case,tmp_path,monkeypatch):
    import os,sys
    from scipy import sparse
    from agent.tools._cancellation import cancellation_scope,ToolWorkCancelled
    args,_=external_case;spec=c.resolved_specification(args);_,_,binding=external.qualify(args,spec)
    a,_=external.load_matrix(binding)
    upstream=tmp_path/'cancel-upstream'
    for name in science.SOURCES:
        p=upstream/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic cancellation fixture')
        monkeypatch.setitem(science.SOURCES,name,science.sha256(p))
    monkeypatch.setattr(external,'write_participation',lambda *a: {})
    monkeypatch.setattr(external,'weighted_rp',lambda *a,**kw:(sparse.csr_matrix([[1.,1.]]*len(a[2])),['G','H']))
    started=tmp_path/'child.pid';exe=tmp_path/'rscript'
    exe.write_text('#!'+sys.executable+'\nimport os,time\nfrom pathlib import Path\nPath('+repr(str(started))+').write_text(str(os.getpid()))\ntime.sleep(30)\n');exe.chmod(0o700)
    with cancellation_scope(lambda:started.exists()),pytest.raises(ToolWorkCancelled):
        _DIRECT_RUNNER(binding=binding,specification=spec,runtime=science.Runtime(str(upstream),str(exe),str(tmp_path)),output_dir=tmp_path/'cancel-output')
    with pytest.raises(ProcessLookupError):os.kill(int(started.read_text()),0)
    assert science.sha256(args['source_path'])==args['source_sha256']


def test_http_explicit_obs_column_uses_shared_literal_binding(external_case,tmp_path):
    args,calls=external_case;prompts=[]
    utterance='Annotate my PBMC counts using "layers/raw_counts" and the independent group column "supplied_groups".'
    class ColumnModel(Model):
        def complete(self,*,prompt,response_schema):
            p=json.loads(prompt)
            if 'turn_schema_version' in p:
                declarations=[dict(tool='annotate_scATAC_cell_types',argument=key,literal=value,evidence='"'+value+'"') for key,value in (('count_slot','layers/raw_counts'),('group_column','supplied_groups'))]
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='execute_plan',target='annotate_scATAC_cell_types',roles=ROLES[:1],arguments=declarations)))
            if 'semantic_prompt_version' in p:
                value=wire();sources=value['decision']['steps'][0]['sources']
                sources[:]=[r for r in sources if r['target']!='groups']+[dict(target='group_column',source=dict(kind='input',input='group_column'))]
                return json.dumps(value)
            return super().complete(prompt=prompt,response_schema=response_schema)
    root=tmp_path/'columns-web';uploads=root/'uploads';uploads.mkdir(parents=True)
    profile=PlanningModelProfile('scripted','scripted',Model.model_id)
    app=InteractiveAgentApplication(root,model_profiles=(profile,),default_profile_id='scripted',approved_source_roots=(uploads,),planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:ColumnModel(prompts)}))
    choice=ScientificInputSet('reviewed','Human PBMC cut-site counts and annotation resources',{k:v for k,v in args.items() if k.startswith('annotation_')})
    with TestClient(create_app(app,input_sets=(choice,),uploads=H5ADUploadAdmission(app.resources,uploads))) as client:
        client.post('/api/v1/sessions',json={'session_id':'upload'})
        registered=client.put('/api/v1/uploads/column-source?filename=peaks.h5ad',content=Path(args['source_path']).read_bytes(),headers={'Content-Type':'application/octet-stream'})
        assert registered.status_code==201,registered.text
        submitted=client.post('/api/v1/sessions/upload/turns',json=dict(turn_id='column',expected_generation=0,utterance=utterance,resource_ids=[registered.json()['resource_id']],input_set_id='reviewed'))
        assert submitted.status_code==202,submitted.text
        result=wait_turn(client,'column');assert result['status']=='succeeded',result
        run=app._application.run_store.load(result['run_id']);bound=run.steps[0].resolved_arguments
        assert bound['count_slot']=='layers/raw_counts' and bound['group_column']=='supplied_groups' and 'groups_path' not in bound
        assert len(calls)==1


def test_selected_resources_empty_offer_does_not_assert_missing_reference(external_case,tmp_path):
    """A provider's resource question is not proof that selected resources are absent."""
    args,calls=external_case
    prompts=[]
    class ResourceQuestion(Model):
        def complete(self,*,prompt,response_schema):
            value=json.loads(prompt);prompts.append(value)
            assert 'turn_schema_version' in value, 'Clarification must not invoke planning'
            return json.dumps(dict(turn_schema_version=1,decision=dict(
                kind='request_resources',target='annotate_scATAC_cell_types',
                roles=ROLES[:1],arguments=[])))
    root=tmp_path/'resource-question';uploads=root/'uploads';uploads.mkdir(parents=True)
    profile=PlanningModelProfile('scripted','scripted',Model.model_id)
    app=InteractiveAgentApplication(root,model_profiles=(profile,),default_profile_id='scripted',
        approved_source_roots=(uploads,),planning_model_factory_registry=PlanningModelFactoryRegistry(
            {'scripted':lambda _:ResourceQuestion(prompts)}))
    inputs={k:v for k,v in args.items() if k.startswith('annotation_') or k=='count_slot'}
    choice=ScientificInputSet('reviewed','Reviewed human annotation resources',inputs)
    with TestClient(create_app(app,input_sets=(choice,),uploads=H5ADUploadAdmission(app.resources,uploads))) as client:
        client.post('/api/v1/sessions',json={'session_id':'upload'})
        registered=client.put('/api/v1/uploads/source?filename=peaks.h5ad',
            content=Path(args['source_path']).read_bytes(),headers={'Content-Type':'application/octet-stream'})
        assert registered.status_code==201,registered.text
        body=dict(turn_id='resources',expected_generation=0,utterance=OBJECTIVE,
            resource_ids=[registered.json()['resource_id']],input_set_id='reviewed')
        assert client.post('/api/v1/sessions/upload/turns',json=body).status_code==202
        view=wait_turn(client,'resources')
        assert view['status']=='clarification',view
        question=view['response']['clarification']
        assert question['reason']=='missing_resources' and not question['choices']
        assert 'No additional reviewed resource choice' in view['response']['text']
        assert 'must provision' not in view['response']['text']
        state=app._application.sessions.load('upload')
        assert state.generation==0 and not state.revisions and not calls and len(prompts)==1
        interaction=state.interactions[-1]
        assert interaction.utterance==OBJECTIVE and interaction.prerequisite['field']=='resources'
        assert all(interaction.admitted['inputs'][k]==v for k,v in inputs.items())
        assert interaction.admitted['inputs']['source_sha256']==args['source_sha256']
        assert client.post('/api/v1/sessions/upload/turns',json=body).status_code==202
        assert wait_turn(client,'resources')==view
        assert len(prompts)==1 and not calls


def test_nonprimary_and_unmatched_peaks_reach_native_owner(external_case, tmp_path):
    args, calls = external_case
    a = ad.read_h5ad(args['source_path'])
    a.var_names = ['KI270728.1:232298-233201'] + list(a.var_names[1:])
    a.write_h5ad(args['source_path'], convert_strings_to_categoricals=False)
    args['source_sha256'] = science.sha256(args['source_path'])
    result = ResearchAgentApplication(tmp_path/'scope', planner=Planner(args)).run(
        AgentRequest('outside-scope', 'Annotate declared cut-site counts.', {}))
    assert result.status.value == 'SUCCEEDED', result
    assert len(calls) == 1  # Simulated backend, not biological qualification.
    output = ad.read_h5ad(result.run_result.steps[0].result['annotated_h5ad_path'])
    assert output.var_names.equals(a.var_names)
    assert (output.X != a.X).nnz == 0


def test_cellranger_h5_renaming_does_not_convert_to_h5ad(tmp_path):
    from agent.tools.analysis.peak_h5ad import inspect
    from agent.orchestration.error_policy import safe_message_for
    path = tmp_path/'renamed.h5ad'
    with h5py.File(path, 'w') as f:
        f.create_group('matrix')
    with pytest.raises(ValueError, match='Cell Ranger .h5 is unsupported'):
        inspect(path)
    assert 'renaming the extension is not conversion' in safe_message_for('ANNOTATION_PREREQUISITE_INVALID')


@pytest.mark.parametrize('feature', ['chr1:5-5', 'chr1:9-2', 'chr1:-1-2', 'chr1:0-4503599627370497', 'chr1:01-9', 'bad chr:1-3'])
def test_native_coordinate_syntax_still_rejects_invalid_intervals(feature):
    with pytest.raises(ValueError):
        external.coordinates([feature], {})


def test_native_coordinates_preserve_unmatched_and_nonprimary_names():
    assert external.coordinates(['KI270728.1:232298-233201', 'chrM:1-10', 'chrUn_KI270442v1:0-10'], {}) == [
        b'KI270728.1_232298_233201', b'chrM_1_10', b'chrUn_KI270442v1_0_10']


def test_optional_genome_bounds_still_reject_out_of_bounds(external_case, monkeypatch):
    monkeypatch.setattr(external, 'contig_lengths', lambda spec: {'chr1': 10})
    with pytest.raises(ValueError, match='contig length'):
        external.coordinates(['chr1:1-11'], {})


def test_participation_uses_native_weights_and_exact_count_totals(tmp_path):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    gene = tmp_path/'genes.tsv'; gene.write_text('chrom\nchr1\n')
    x = sparse.csr_matrix([[1, 2, 3], [2, 1, 4]], dtype=np.int64)
    before = x.copy()
    summary = external.write_participation(x, [b'chr1_1_2', b'chr1_3_4', b'KI270728.1_1_2'],
        gene, {'transcript_weight_nnz': np.array([2, 0, 0])}, tmp_path)
    assert summary['unmatched_contig_features'] == 1
    assert summary['unmatched_contig_counts'] == 7
    assert summary['zero_transcript_weight_features'] == 2
    assert summary['zero_transcript_weight_counts'] == 10
    assert (x != before).nnz == 0
    rows = pd.read_csv(tmp_path/'feature-participation.tsv', sep='\t')
    assert rows.feature.tolist() == ['chr1:1-2', 'chr1:3-4', 'KI270728.1:1-2']
    assert rows.native_transcript_weight_nnz.tolist() == [2, 0, 0]


def test_participation_tampering_fails_even_with_rehashed_metadata(external_case):
    from agent.tools.analysis import scatac_annotation as owner
    args, _ = external_case
    result = owner.execute_annotation(args, 'participation-tamper')
    path = Path(result['manifest_path'])
    value = json.loads(path.read_text())
    value['feature_participation']['unmatched_contig_counts'] = 999
    path.write_bytes(c.matrix.canonical(value))
    with pytest.raises(ValueError, match='participation'):
        owner.verify_annotation(path, expected_sha256=science.sha256(path))
