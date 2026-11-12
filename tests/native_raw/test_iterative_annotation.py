"""Opt-in ICTA-4A: real Runtime/RunStore and installed native child packages.

The internal ToolSpec is confined to this harness. Production MAESTRO is intact.
"""
from dataclasses import replace
from collections.abc import Mapping
import json
import os
from pathlib import Path
import runpy

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.orchestration import AgentRuntime, FileRunStore
from agent.orchestration.annotation_runs import continuation_anchor
from agent.orchestration.prior_outputs import bind_output, validate_binding
from agent.orchestration.registry import (build_default_tool_registry, ToolRegistry, ToolSpec,
    ArgumentSpec, ResultContract, DurableToolHooks, ScientificAuthorityHooks,
    SemanticToolSpec, SemanticConsumerPortSpec, SemanticProducerPortSpec, SemanticPortMember)
from agent.orchestration.native_matrix_registry import classify
from agent.schemas import AgentPlan, AgentRequest, PlanStep, TraceEventType
from agent.schemas.prior_output import PriorOutputRef
from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError
from agent.tools._file_identity import file_sha256
from agent.tools.analysis import iterative_annotation as coordinator, hierarchical_annotation as owner
from agent.tools.data.cellmarker import CellMarkerEvidence
from agent.tools.analysis.sequence_prediction import library


def registry(monkeypatch):
    baseline = build_default_tool_registry()
    pub = owner.publication
    required = {k: ArgumentSpec((str,)) for k in ('input_path', 'input_sha256', 'grouping_path', 'cellmarker_id', 'output_dir')}
    optional = {k: ArgumentSpec((Mapping,) if t == (dict,) else t) for k, t in dict(activity_path=(str,), activity_source_path=(str,), activity_method=(str,),
        gene_anno_path=(str,), gene_anno_sha256=(str,), activity_kwargs=(dict,), activity_count_slot=(str,),
        cellmarker_identity=(dict,), maximum_levels=(int,), local_kwargs=(dict,), differential_kwargs=(dict,),
        parameter_selection=(str,), species=(str,), marker_alpha=(float,), ora_alpha=(float,),
        adjusted_p_threshold=(float,), minimum_peak_features=(int,), marker_context=(dict,),
        identity_kind=(str,), identity_column=(str,)).items()}
    def validate(result):
        assert pub.verify_publication(result['manifest_path'], expected_sha256=result['manifest_sha256']) == dict(result)
    spec = ToolSpec(owner.TOOL, lambda **a: pytest.fail('requires durable coordinator'), required, optional,
        ResultContract('HierarchicalAnnotation', {k:(str,) for k in ('manifest_path','manifest_sha256','annotated_h5ad_path','annotated_h5ad_sha256')}, validator=validate),
        classify, durable_hooks=DurableToolHooks(coordinator.coordinate, owner.recover),
        scientific_authority=ScientificAuthorityHooks(owner.KIND, ('dataset',), pub.describe,
            pub.publication_record, pub.reuse_result, pub.proof_metadata),
        semantic_planning=SemanticToolSpec(consumer_ports=(
            *(SemanticConsumerPortSpec(k, (SemanticPortMember('dataset', k + '_path'),), k == 'grouping',
                accepted_upstream_types=('native_matrix_h5ad.v1',)) for k in ('grouping', 'activity', 'activity_source')),
            next(p for p in baseline.get('run_episcanpy').semantic_planning.consumer_ports if p.name == 'gene_anno')),
            producer_ports=(SemanticProducerPortSpec('dataset', 'hierarchical_annotation_h5ad.v1',
                (SemanticPortMember('dataset','annotated_h5ad_path'), SemanticPortMember('sha256','annotated_h5ad_sha256'))),)),
        annotation_children=coordinator.CHILDREN)
    result = ToolRegistry(tuple(baseline.get(n) for n in baseline.names()) + (spec,))
    monkeypatch.setattr('agent.orchestration.registry.build_default_tool_registry', lambda: result)
    monkeypatch.setattr('agent.orchestration.verification_authority.build_default_tool_registry', lambda: result)
    return result


def prepare(name, root, args):
    request = AgentRequest(name, None, {})
    return request, AgentPlan(name + ':plan', name, 'Internal annotation qualification',
        (PlanStep('annotate', owner.TOOL, dict(args, output_dir=str(root/name))),))


def reserved(state):
    return [e.details for e in state.trace if e.event_type is TraceEventType.ANNOTATION_PROGRESS and e.details['event'] == 'reserved']


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION') != '1':
        pytest.skip('Explicit installed native qualification')
    patch = pytest.MonkeyPatch()
    reg = registry(patch)
    root = tmp_path_factory.mktemp('iterative-annotation')
    store = FileRunStore(root/'run_state')
    runtime = lambda s=store: AgentRuntime(registry=reg, run_store=s)
    rng = np.random.default_rng(2026)
    probability = np.full((140,120), .2)
    probability[:60,:20] = .8; probability[:60,20:40] = .04
    probability[60:120,:20] = .04; probability[60:120,20:40] = .8
    x = rng.binomial(1,probability).astype('uint32')
    data = ad.AnnData(sparse.csr_matrix(x), obs=pd.DataFrame({'original': pd.Categorical(['kept']*140)}, index=[f'cell-{i}' for i in range(140)]),
        var=pd.DataFrame(index=[f'chr1:{1000+i*200}-{1100+i*200}' for i in range(120)]))
    data.layers['original_counts'] = data.X.copy()
    data.raw = data.copy()
    data.uns['matrix_semantics'] = 'peak_counts'
    data.uns['scientific_note'] = 'transparent Bernoulli synthetic accessibility; no biological validation'
    data.obsp['distances'] = sparse.block_diag([np.ones((120,120))-np.eye(120),np.ones((20,20))-np.eye(20)],format='csr')
    source = root/'original.h5ad'; data.write_h5ad(source)
    before = file_sha256(source)
    request=AgentRequest('initial-groups',None,{})
    plan=AgentPlan('initial-groups:plan',request.request_id,'Accepted initial native grouping',
        (PlanStep('groups','run_snapatac2',dict(input_path=str(source),output_dir=str(root/'initial-groups'),
            function_name='tl.leiden',kwargs=dict(resolution=.05))),))
    result=runtime().run_prepared(request,plan)
    assert result.status.value=='SUCCEEDED',result.errors
    grouping=bind_output(store,reg,result.run_id,'groups','dataset')
    genes=['Cd3d','Cd3e','Trac','Cd247']+[f'fixture_gene_{i}' for i in range(116)]
    gtf=root/'genes.gtf'
    gtf.write_text(''.join(f'chr1\tHAVANA\tgene\t{1001+i*200}\t{1100+i*200}\t.\t+\t.\tgene_id "g{i}"; gene_name "{gene}";\n' for i,gene in enumerate(genes)))
    lib=library(); entry=next(e for e in lib.entries() if e['kind']=='cellmarker')
    evidence=CellMarkerEvidence(lib,entry['id'])
    args=dict(input_path=str(source),input_sha256=before,grouping_path=PriorOutputRef(grouping,'matrix_path'),
        activity_method='episcanpy.tl.geneactivity',activity_kwargs={'upstream':0},
        gene_anno_path=str(gtf),gene_anno_sha256=file_sha256(gtf),cellmarker_id=entry['id'],
        cellmarker_identity=evidence.identity,species='Mouse',maximum_levels=2,
        local_kwargs={'pp.select_features':{'n_features':100},'tl.spectral':{'n_comps':8,'random_state':0},'pp.knn':{'n_neighbors':10}})
    yield dict(root=root,store=store,registry=reg,runtime=runtime,args=args,source=source,sha=before,grouping=grouping)
    assert file_sha256(source)==before
    patch.undo()


def test_genuine_coordinator_interrupt_continue_publish_and_recover(native, monkeypatch):
    f=native
    # Reuse ICTA-1's fault injector; it writes only through FileRunStore.
    lifecycle=runpy.run_path(str(Path(__file__).parents[1]/'unit/orchestration/test_annotation_child_runs.py'))
    FaultStore,Crash=lifecycle['FaultStore'],lifecycle['Crash']
    def boundary(state):
        return state.run_id=='first:run' and any(e.event_type is TraceEventType.ANNOTATION_PROGRESS
            and e.details.get('event')=='outcome' and e.details.get('operation_id')=='markers' for e in state.trace)
    fault=FaultStore(f['store'],boundary)
    with pytest.raises(Crash):
        f['runtime'](fault).run_prepared(*prepare('first',f['root'],f['args']))
    saved=f['store'].load('first:run')
    old_children=[r['request']['request_id']+':run' for r in reserved(saved)]
    exact={rid:f['store'].load(rid).to_dict() for rid in old_children}
    f['runtime']().resume(saved.run_id)
    anchor=continuation_anchor(f['store'],saved.run_id,'annotate')
    with pytest.raises(AuthorityError,match='exact captured'):
        f['runtime']().continue_annotation(*prepare('changed-budget',f['root'],dict(f['args'],maximum_levels=3)),previous=anchor)
    final_fault=FaultStore(f['store'],lambda state:state.run_id=='continued:run' and bool(state.steps)
                          and state.steps[0].status.value=='SUCCEEDED',before=True)
    with pytest.raises(Crash):
        f['runtime'](final_fault).continue_annotation(*prepare('continued',f['root'],f['args']),previous=anchor)
    pending=f['store'].load('continued:run')
    assert pending.steps[0].status.value=='RUNNING' and pending.steps[0].verification is None
    all_children=old_children+[r['request']['request_id']+':run' for r in reserved(pending)]
    child_files={str(p):file_sha256(p) for p in f['root'].rglob('*')
                 if p.is_file() and (any(part.endswith('.children') for part in p.parts)
                    or p in [f['store'].state_path(rid) for rid in all_children])}
    completed={rid:f['store'].load(rid).to_dict() for rid in all_children}
    publication_files={str(p):file_sha256(p) for p in (f['root']/'continued').iterdir() if p.is_file()}
    from agent.tools import _native_process
    with monkeypatch.context() as recovery_patch:
        recovery_patch.setattr(_native_process,'call',lambda *a,**k:pytest.fail('publication recovery repeated science'))
        recovery_patch.setattr(coordinator,'annotate_markers',lambda *a,**k:pytest.fail('recovery repeated ORA'))
        recovery_patch.setattr(coordinator,'assess_subdivision',lambda *a,**k:pytest.fail('recovery repeated subdivision'))
        result=f['runtime']().resume('continued:run')
        authority=result.steps[0].verification.artifact_authority
        assert authority is not None and authority['schema_version']==2
        from agent.orchestration.durable_tool_recovery import execution_identity
        assert authority['execution_identity']==execution_identity(pending.run_id,pending.plan,
            pending.plan.steps[0],f['registry'].get(owner.TOOL))
        binding=bind_output(f['store'],f['registry'],result.run_id,'annotate','dataset')
        validate_binding(binding,store=f['store'],registry=f['registry'],integrity=True)
    assert result.status.value=='SUCCEEDED', result.to_dict()
    assert all(f['store'].load(rid).to_dict()==state for rid,state in completed.items())
    assert all(file_sha256(p)==sha for p,sha in {**child_files,**publication_files}.items())
    assert all(f['store'].load(rid).to_dict()==state for rid,state in exact.items())
    value=_serialize(result.steps[0].result)
    hierarchy=json.loads(Path(value['hierarchy_path']).read_text())
    assert any(b['level']==2 for b in hierarchy['branches'])
    assert any(b['stopping_reason']=='maximum_levels' for b in hierarchy['branches'])
    methods=[r['native_result']['function_name'] for r in hierarchy['operations']]
    assert methods.count('episcanpy.tl.geneactivity')==1
    assert 'scanpy.tl.rank_genes_groups' in methods and 'snapatac2.tl.diff_test' in methods
    assert all(m in methods for m in coordinator.LOCAL_METHODS)
    assert any(r['native_result']['parameters'].get('parameter_selection',{}).get('origin')=='automatic' for r in hierarchy['operations'])
    assert any(r['native_result']['parameters'].get('grouping_projection') for r in hierarchy['operations'])
    matrix=ad.read_h5ad(value['annotated_h5ad_path']); original=ad.read_h5ad(f['source'])
    assert matrix.obs_names.tolist()==original.obs_names.tolist()
    assert (matrix.X != original.X).nnz==0
    assert (matrix.layers['original_counts'] != original.layers['original_counts']).nnz==0
    assert matrix.obs.annotation_depth.max()==2
    from agent.tools.analysis import peak_h5ad
    peak_h5ad.inspect(value['annotated_h5ad_path'])
    pd.testing.assert_frame_equal(matrix.uns[owner.UNS]['branches'],owner.portable_hierarchy(hierarchy)['branches'])
    from agent.tools import _native_process
    monkeypatch.setattr(_native_process,'call',lambda *a,**k:pytest.fail('accepted recovery repeated science'))
    monkeypatch.setattr(coordinator,'coordinate',lambda *a,**k:pytest.fail('completed parent reentered'))
    recovered=f['runtime']().resume(result.run_id)
    assert recovered.status.value=='SUCCEEDED'
    assert recovered.steps[0].result==result.steps[0].result
    f['completed']=value
    (f['root']/'acceptance.json').write_text(json.dumps(dict(result=value,old_children=old_children,
        recovered_authority=_serialize(authority),recovered_binding=binding.to_dict(),
        completed_children=all_children,
        unchanged_child_files=child_files,unchanged_publication_files=publication_files,
        branch_summary=[{k:b[k] for k in ('branch_id','parent','level','n_cells','stopping_reason')} for b in hierarchy['branches']]),indent=2))


def test_completed_publication_rejects_tampering(native):
    f=native; value=f['completed']
    for path in (Path(value['annotated_h5ad_path']),Path(value['manifest_path']),f['source']):
        before=path.read_bytes()
        try:
            path.write_bytes(before+b'tamper')
            with pytest.raises((AuthorityError,ValueError)):
                owner.publication.verify_publication(value['manifest_path'],expected_sha256=value['manifest_sha256'])
        finally:
            path.write_bytes(before)
    hierarchy=json.loads(Path(value['hierarchy_path']).read_text())
    path=Path(hierarchy['operations'][0]['native_result']['matrix_path']); before=path.read_bytes()
    try:
        path.write_bytes(before+b'tamper')
        with pytest.raises(AuthorityError):
            owner.publication.verify_publication(value['manifest_path'],expected_sha256=value['manifest_sha256'])
    finally:
        path.write_bytes(before)


def test_accepted_activity_reuse_and_native_failure_remains_unfinished(native):
    f=native; hierarchy=json.loads(Path(f['completed']['hierarchy_path']).read_text())
    from agent.schemas.prior_output import PriorOutputBinding
    activity=PriorOutputBinding(**hierarchy['activity'])
    args={k:v for k,v in f['args'].items() if k not in {'activity_method','activity_kwargs','gene_anno_path','gene_anno_sha256'}}
    args.update(activity_path=PriorOutputRef(activity,'matrix_path'),maximum_levels=1)
    result=f['runtime']().run_prepared(*prepare('reuse-activity',f['root'],args))
    assert result.status.value=='SUCCEEDED',result.to_dict()
    rows=reserved(f['store'].load(result.run_id))
    assert [r['operation_id'] for r in rows]==['markers']
    args['maximum_levels']=2
    args['local_kwargs']={'pp.select_features':{'n_features':1.5}}
    failed=f['runtime']().run_prepared(*prepare('native-failure',f['root'],args))
    assert failed.status.value=='FAILED'
    state=f['store'].load(failed.run_id)
    assert not (f['root']/'native-failure'/'manifest.json').exists()
    children=[f['store'].load(r['request']['request_id']+':run') for r in reserved(state)]
    assert children[0].lifecycle_status.value=='SUCCEEDED'
    assert children[-1].lifecycle_status.value=='FAILED'
    assert any(e.code=='NATIVE_FUNCTION_REJECTED' for e in children[-1].errors)
    exact=children[-1].to_dict()
    resumed=f['runtime']().continue_annotation(*prepare('failed-continuation',f['root'],args),
        previous=continuation_anchor(f['store'],failed.run_id,'annotate'))
    assert resumed.status.value=='FAILED'
    assert f['store'].load(children[-1].run_id).to_dict()==exact
    (f['root']/'native-failure-evidence.json').write_text(json.dumps(dict(parent=failed.to_dict(),child=exact),indent=2))


def test_gene_only_current_round_is_a_complete_limited_result(native):
    from agent.schemas.prior_output import PriorOutputBinding
    f=native; h=json.loads(Path(f['completed']['hierarchy_path']).read_text())
    activity=PriorOutputBinding(**h['activity'])
    value=next(s for s in f['store'].load(activity.run_id).steps if s.step_id==activity.step_id).result
    request=AgentRequest('gene-grouping',None,{})
    plan=AgentPlan('gene-grouping:plan',request.request_id,'Gene-only grouping',
        (PlanStep('groups','run_snapatac2',dict(input_path=PriorOutputRef(activity,'matrix_path'),
            output_dir=str(f['root']/'gene-grouping'),function_name='tl.leiden',kwargs={'resolution':.05})),))
    result=f['runtime']().run_prepared(request,plan)
    assert result.status.value=='SUCCEEDED',result.to_dict()
    grouping=bind_output(f['store'],f['registry'],result.run_id,'groups','dataset')
    args=dict(input_path=value['matrix_path'],input_sha256=value['matrix_sha256'],
        grouping_path=PriorOutputRef(grouping,'matrix_path'),activity_path=PriorOutputRef(activity,'matrix_path'),
        cellmarker_id=f['args']['cellmarker_id'],cellmarker_identity=f['args']['cellmarker_identity'],species='Mouse')
    result=f['runtime']().run_prepared(*prepare('gene-only',f['root'],args))
    assert result.status.value=='SUCCEEDED',result.to_dict()
    hierarchy=json.loads(Path(result.steps[0].result['hierarchy_path']).read_text())
    assert all(b['stopping_reason']=='missing_required_peak_space' for b in hierarchy['branches'])
    assert [r['operation_id'] for r in hierarchy['operations']]==['markers']
    assert hierarchy['maximum_levels']==5
    (f['root']/'gene-only-evidence.json').write_text(json.dumps(_serialize(result.steps[0].result),indent=2))


def test_cancellation_after_accepted_marker_and_exact_continuation(native,monkeypatch):
    from agent.orchestration.annotation_runs import _AnnotationInvocation
    from agent.schemas.prior_output import PriorOutputBinding
    from agent.tools import _native_process
    f=native; h=json.loads(Path(f['completed']['hierarchy_path']).read_text())
    args={k:v for k,v in f['args'].items() if k not in {'activity_method','activity_kwargs','gene_anno_path','gene_anno_sha256'}}
    args.update(activity_path=PriorOutputRef(PriorOutputBinding(**h['activity']),'matrix_path'),maximum_levels=1)
    original=_AnnotationInvocation.invoke
    def cancel_after(inv,**kw):
        result=original(inv,**kw)
        if kw['operation_id']=='markers': f['runtime']().cancel('cancel-parent:run')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(_AnnotationInvocation,'invoke',cancel_after)
        result=f['runtime']().run_prepared(*prepare('cancel-parent',f['root'],args))
    assert result.status.value=='CANCELLED',result.to_dict()
    state=f['store'].load(result.run_id)
    rows=reserved(state)
    assert [r['operation_id'] for r in rows]==['markers']
    child_id=rows[0]['request']['request_id']+':run'; before=f['store'].load(child_id).to_dict()
    assert before['lifecycle_status']=='SUCCEEDED'
    assert not (f['root']/'cancel-parent'/'manifest.json').exists()
    monkeypatch.setattr(_native_process,'call',lambda *a,**k:pytest.fail('continuation repeated an accepted child'))
    result=f['runtime']().continue_annotation(*prepare('cancel-continued',f['root'],args),
        previous=continuation_anchor(f['store'],state.run_id,'annotate'))
    assert result.status.value=='SUCCEEDED',result.to_dict()
    assert f['store'].load(child_id).to_dict()==before
    (f['root']/'cancellation.json').write_text(json.dumps(dict(cancelled_parent=state.to_dict(),
        accepted_child=child_id,continued_parent=result.to_dict()),indent=2))


@pytest.fixture(scope='module')
def fragment_handoff(tmp_path_factory):
    """One small native source DAG; constructors choose independent accepted inputs."""
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION') != '1':
        pytest.skip('Explicit installed native qualification')
    patch=pytest.MonkeyPatch(); reg=registry(patch)
    root=tmp_path_factory.mktemp('fragment-handoff'); store=FileRunStore(root/'run_state')
    runtime=lambda:AgentRuntime(registry=reg,run_store=store)
    fragments=root/'fragments.tsv'; peaks=root/'peaks.bed'; gtf=root/'genes.gtf'
    lines=[]; regions=[]; genes=[]
    names=['Cd3d','Cd3e','Trac','Cd247']+[f'fixture_gene_{i}' for i in range(16)]
    for j,name in enumerate(names):
        start=1000+j*500
        regions.append(f'chr1\t{start}\t{start+200}\n')
        for kind in ('gene','transcript'):
            genes.append(f'chr1\tHAVANA\t{kind}\t{start+1}\t{start+200}\t.\t+\t.\tgene_id "G{j}"; transcript_id "T{j}"; gene_name "{name}";\n')
    for i in range(24):
        for j in range(20):
            copies=5+(i%3) if (i<12 and j<5) or (i>=12 and 5<=j<10) else 1
            for k in range(copies):
                start=1000+j*500+10+k*3
                lines.append(f'chr1\t{start}\t{start+40}\tc{i:02}\t1\n')
    fragments.write_text(''.join(lines)); peaks.write_text(''.join(regions)); gtf.write_text(''.join(genes))
    request=AgentRequest('fragment-prerequisites',None,{})
    from agent.schemas import StepOutputRef
    plan=AgentPlan('fragment-prerequisites:plan',request.request_id,'Native fragment, peak and grouping prerequisites',(
        PlanStep('import','run_snapatac2',dict(input_path=str(fragments),output_dir=str(root/'import'),
            function_name='pp.import_fragments',kwargs=dict(chrom_sizes={'chr1':12000},min_num_fragments=0))),
        PlanStep('peaks','run_snapatac2',dict(input_path=StepOutputRef('import','matrix_path'),output_dir=str(root/'peaks'),
            function_name='pp.make_peak_matrix',peaks_path=str(peaks),peaks_sha256=file_sha256(peaks)),('import',)),
        PlanStep('neighbors','run_scanpy',dict(input_path=StepOutputRef('peaks','matrix_path'),output_dir=str(root/'neighbors'),
            function_name='pp.neighbors',kwargs=dict(use_rep='X',n_neighbors=5,random_state=0)),('peaks',)),
        PlanStep('groups','run_scanpy',dict(input_path=StepOutputRef('neighbors','matrix_path'),output_dir=str(root/'groups'),
            function_name='tl.leiden',kwargs=dict(resolution=.3,random_state=0)),('neighbors',))))
    result=runtime().run_prepared(request,plan)
    assert result.status.value=='SUCCEEDED',result.to_dict()
    bindings={step:bind_output(store,reg,result.run_id,step,'dataset') for step in ('import','peaks','groups')}
    native={step.step_id:_serialize(step.result) for step in result.steps}
    imported=ad.read_h5ad(native['import']['matrix_path']); matrix=ad.read_h5ad(native['peaks']['matrix_path'])
    assert 'fragment_paired' in imported.obsm and 'fragment_paired' not in matrix.obsm
    assert matrix.obs_names.tolist()==imported.obs_names.tolist()
    assert len(ad.read_h5ad(native['groups']['matrix_path']).obs.leiden.cat.categories)>1
    lib=library(); entry=next(e for e in lib.entries() if e['kind']=='cellmarker')
    evidence=CellMarkerEvidence(lib,entry['id'])
    args=dict(input_path=native['peaks']['matrix_path'],input_sha256=native['peaks']['matrix_sha256'],
        grouping_path=PriorOutputRef(bindings['groups'],'matrix_path'),
        activity_source_path=PriorOutputRef(bindings['import'],'matrix_path'),
        activity_method='snapatac2.pp.make_gene_matrix',activity_kwargs={'upstream':0},
        gene_anno_path=str(gtf),gene_anno_sha256=file_sha256(gtf),cellmarker_id=entry['id'],
        cellmarker_identity=evidence.identity,species='Mouse',maximum_levels=1)
    yield dict(root=root,store=store,registry=reg,runtime=runtime,args=args,bindings=bindings,native=native,fragments=fragments)
    assert all(file_sha256(v['matrix_path'])==v['matrix_sha256'] for v in native.values())
    patch.undo()


def test_small_recovered_hierarchy_is_an_accepted_native_input(fragment_handoff, monkeypatch):
    f=fragment_handoff
    lifecycle=runpy.run_path(str(Path(__file__).parents[1]/'unit/orchestration/test_annotation_child_runs.py'))
    FaultStore,Crash=lifecycle['FaultStore'],lifecycle['Crash']
    fault=FaultStore(f['store'],lambda s:s.run_id=='recovery:run' and bool(s.steps)
                    and s.steps[0].status.value=='SUCCEEDED',before=True)
    with pytest.raises(Crash):
        AgentRuntime(registry=f['registry'],run_store=fault).run_prepared(*prepare('recovery',f['root'],f['args']))
    pending=f['store'].load('recovery:run')
    assert pending.steps[0].status.value=='RUNNING' and pending.steps[0].verification is None
    rows=reserved(pending)
    children={r['request']['request_id']+':run':f['store'].load(r['request']['request_id']+':run').to_dict() for r in rows}
    pins={str(p):file_sha256(p) for r in rows for p in Path(r['output_dir']).rglob('*') if p.is_file()}
    pins.update({str(f['store'].state_path(rid)):file_sha256(f['store'].state_path(rid)) for rid in children})
    pins.update({str(p):file_sha256(p) for p in (f['root']/'recovery').iterdir() if p.is_file()})
    pins.update({v['matrix_path']:v['matrix_sha256'] for v in f['native'].values()})
    from agent.tools import _native_process
    from agent.orchestration.durable_tool_recovery import execution_identity
    with monkeypatch.context() as patch:
        patch.setattr(_native_process,'call',lambda *a,**k:pytest.fail('recovery repeated native science'))
        patch.setattr(coordinator,'annotate_markers',lambda *a,**k:pytest.fail('recovery repeated ORA'))
        patch.setattr(coordinator,'assess_subdivision',lambda *a,**k:pytest.fail('recovery repeated subdivision'))
        recovered=f['runtime']().resume(pending.run_id)
        assert recovered.status.value=='SUCCEEDED',recovered.to_dict()
        authority=recovered.steps[0].verification.artifact_authority
        assert authority is not None and authority['schema_version']==2
        assert authority['execution_identity']==execution_identity(pending.run_id,pending.plan,
            pending.plan.steps[0],f['registry'].get(owner.TOOL))
        binding=bind_output(f['store'],f['registry'],pending.run_id,'annotate','dataset')
        validate_binding(binding,store=f['store'],registry=f['registry'],integrity=True)
    value=_serialize(recovered.steps[0].result)
    matrix=ad.read_h5ad(value['annotated_h5ad_path'])
    original=ad.read_h5ad(f['native']['peaks']['matrix_path'])
    assert matrix.obs_names.equals(original.obs_names) and (matrix.X!=original.X).nnz==0
    hierarchy=json.loads(Path(value['hierarchy_path']).read_text())
    assert hierarchy['completion']=='complete' and matrix.obs.annotation_depth.tolist()==[1]*24
    assert [op['native_result']['function_name'] for op in hierarchy['operations']]==[
        'snapatac2.pp.make_gene_matrix','scanpy.tl.rank_genes_groups']
    # The coordinator remains test-only. Enable exactly its test semantic channel
    # to the existing Scanpy owner, without changing production Native Functions.
    with monkeypatch.context() as patch:
        scanpy=f['registry'].get('run_scanpy'); semantic=scanpy.semantic_planning
        scanpy=replace(scanpy,semantic_planning=replace(semantic,consumer_ports=tuple(
            replace(p,accepted_upstream_types=(*p.accepted_upstream_types,'hierarchical_annotation_h5ad.v1'))
            if p.name=='dataset' else p for p in semantic.consumer_ports)))
        consumers=ToolRegistry(tuple(scanpy if n=='run_scanpy' else f['registry'].get(n) for n in f['registry'].names()))
        patch.setattr('agent.orchestration.registry.build_default_tool_registry',lambda:consumers)
        patch.setattr('agent.orchestration.verification_authority.build_default_tool_registry',lambda:consumers)
        request=AgentRequest('consume-recovered',None,{})
        plan=AgentPlan('consume-recovered:plan',request.request_id,'Native consumer of recovered hierarchy',(
            PlanStep('consumer','run_scanpy',dict(input_path=PriorOutputRef(binding,'annotated_h5ad_path'),
                output_dir=str(f['root']/'consume-recovered'),function_name='pp.log1p')),))
        consumed=AgentRuntime(registry=consumers,run_store=f['store']).run_prepared(request,plan)
        assert consumed.status.value=='SUCCEEDED',consumed.to_dict()
        downstream=bind_output(f['store'],consumers,consumed.run_id,'consumer','dataset')
        validate_binding(downstream,store=f['store'],registry=consumers,integrity=True)
        assert consumed.steps[0].result['source_identity']['sha256']==value['annotated_h5ad_sha256']
        output=ad.read_h5ad(consumed.steps[0].result['matrix_path'])
        assert output.obs.equals(matrix.obs)
        np.testing.assert_allclose(output.X.toarray(),np.log1p(matrix.X.toarray()))
    assert all(f['store'].load(rid).to_dict()==state for rid,state in children.items())
    assert all(file_sha256(p)==sha for p,sha in pins.items())
    (f['root']/'recovery-acceptance.json').write_text(json.dumps(dict(
        before_step=pending.steps[0].to_dict(),recovered=recovered.to_dict(),binding=binding.to_dict(),
        downstream=consumed.to_dict(),unchanged_files=pins,completed_children=list(children)),indent=2))


def test_captured_fragment_constructor_and_existing_peak_routes(fragment_handoff):
    f=fragment_handoff; records={}
    cases=[('fragment',f['args']),
        ('explicit-x',dict(f['args'],activity_source_path=PriorOutputRef(f['bindings']['peaks'],'matrix_path'),
                           activity_kwargs={'upstream':0,'use_x':True})),
        ('episcanpy',{k:v for k,v in dict(f['args'],activity_method='episcanpy.tl.geneactivity').items()
                     if k!='activity_source_path'})]
    for name,args in cases:
        result=f['runtime']().run_prepared(*prepare('handoff-'+name,f['root'],args))
        assert result.status.value=='SUCCEEDED',result.to_dict()
        h=json.loads(Path(result.steps[0].result['hierarchy_path']).read_text())
        activity,markers=h['operations']
        expected=f['bindings']['import' if name=='fragment' else 'peaks' if name=='explicit-x' else 'groups']
        state=f['store'].load(activity['binding']['run_id'])
        assert state.plan.steps[0].arguments['input_path']==PriorOutputRef(expected,'matrix_path')
        assert _serialize(state.steps[0].resolved_arguments['kwargs'])==args['activity_kwargs']
        a=activity['native_result']; m=markers['native_result']
        assert a['source_identity']['sha256']==f['native']['import' if name=='fragment' else 'peaks' if name=='explicit-x' else 'groups']['matrix_sha256']
        if name!='episcanpy':
            assert a['parameters']['effective_native_parameters']['use_x']==(name=='explicit-x')
        projection=m['parameters']['grouping_projection']
        assert projection['activity_positions']==list(range(24))
        assert projection['accepted_publication']=={k:f['native']['groups'][k] for k in ('manifest_path','manifest_sha256')}
        assert m['function_name']=='scanpy.tl.rank_genes_groups'
        evidence=json.loads(Path(result.steps[0].result['annotations_path']).read_text())['initial']
        assert evidence['schema']=='marker-candidates.v1' and evidence['reference']['resource']==h['cellmarker']
        assert len(evidence['clusters'])>1
        annotated=ad.read_h5ad(result.steps[0].result['annotated_h5ad_path'])
        original=ad.read_h5ad(f['native']['peaks']['matrix_path'])
        assert annotated.obs_names.tolist()==original.obs_names.tolist()
        assert (annotated.X!=original.X).nnz==0 and annotated.obs.annotation_depth.tolist()==[1]*24
        assert all(b['stopping_reason']=='maximum_levels' for b in h['branches'])
        records[name]=dict(result=_serialize(result.steps[0].result),hierarchy=h)
    from agent.schemas.prior_output import PriorOutputBinding
    args={k:v for k,v in f['args'].items() if k not in {'activity_source_path','activity_method','activity_kwargs','gene_anno_path','gene_anno_sha256'}}
    args['activity_path']=PriorOutputRef(PriorOutputBinding(**records['episcanpy']['hierarchy']['activity']),'matrix_path')
    reused=f['runtime']().run_prepared(*prepare('handoff-reuse',f['root'],args))
    assert reused.status.value=='SUCCEEDED',reused.to_dict()
    assert [r['operation_id'] for r in reserved(f['store'].load(reused.run_id))]==['markers']
    records['reuse']=dict(result=_serialize(reused.steps[0].result))
    (f['root']/'handoff-evidence.json').write_text(json.dumps(records,indent=2))


def test_wrong_unrelated_and_tampered_constructor_sources(fragment_handoff,monkeypatch):
    f=fragment_handoff
    # A captured plain peak source with default use_x=False reaches the genuine
    # native refusal. The coordinator must not flip use_x or substitute a source.
    args=dict(f['args'],activity_source_path=PriorOutputRef(f['bindings']['peaks'],'matrix_path'))
    failed=f['runtime']().run_prepared(*prepare('handoff-wrong',f['root'],args))
    assert failed.status.value=='FAILED'
    rows=reserved(f['store'].load(failed.run_id)); assert len(rows)==1
    child=f['store'].load(rows[0]['request']['request_id']+':run')
    assert child.lifecycle_status.value=='FAILED' and any(e.code=='NATIVE_FUNCTION_REJECTED' for e in child.errors)
    assert 'use_x' not in child.steps[0].resolved_arguments['kwargs']
    # Same literal cell names, independently imported file: no accepted shared
    # ancestry, even though the native constructor itself can process the file.
    unrelated=f['root']/'unrelated.tsv'; unrelated.write_bytes(f['fragments'].read_bytes())
    request=AgentRequest('unrelated-import',None,{})
    plan=AgentPlan('unrelated-import:plan',request.request_id,'Independent source',(
        PlanStep('import','run_snapatac2',dict(input_path=str(unrelated),output_dir=str(f['root']/'unrelated-import'),
            function_name='pp.import_fragments',kwargs=dict(chrom_sizes={'chr1':12000},min_num_fragments=0))),))
    imported=f['runtime']().run_prepared(request,plan); assert imported.status.value=='SUCCEEDED'
    binding=bind_output(f['store'],f['registry'],imported.run_id,'import','dataset')
    args=dict(f['args'],activity_source_path=PriorOutputRef(binding,'matrix_path'))
    failed=f['runtime']().run_prepared(*prepare('handoff-unrelated',f['root'],args))
    assert failed.status.value=='FAILED'
    children=[f['store'].load(r['request']['request_id']+':run') for r in reserved(f['store'].load(failed.run_id))]
    assert len(children)==2 and children[0].lifecycle_status.value=='SUCCEEDED' and children[1].lifecycle_status.value=='FAILED'
    assert any(e.code=='NATIVE_FUNCTION_PUBLICATION_INVALID' and e.exception_type=='AuthorityError' for e in children[1].errors)
    assert not (f['root']/'handoff-unrelated'/'manifest.json').exists()
    # Existing authority rejects tampered accepted source bytes before invocation.
    path=Path(f['native']['import']['matrix_path']); before=path.read_bytes()
    from agent.orchestration.prior_outputs import validate_binding
    from agent.tools import _native_process
    try:
        path.write_bytes(before+b'tamper')
        with monkeypatch.context() as patch:
            patch.setattr(_native_process,'call',lambda *a,**k:pytest.fail('tampered source reached native execution'))
            with pytest.raises(AuthorityError):
                validate_binding(f['bindings']['import'],store=f['store'],registry=f['registry'],integrity=True)
            result=f['runtime']().run_prepared(*prepare('handoff-tampered',f['root'],f['args']))
            assert result.status.value=='FAILED' and not reserved(f['store'].load(result.run_id))
    finally:path.write_bytes(before)
    (f['root']/'source-rejections.json').write_text(json.dumps(dict(wrong_native_child=child.to_dict(),
        unrelated_parent=failed.to_dict(),unrelated_children=[c.to_dict() for c in children],tamper_rejected=True),indent=2))


def test_accepted_multilevel_publication_downstream_compatibility(tmp_path, monkeypatch):
    """Republish accepted science only; no native annotation step may be repeated."""
    import shutil
    import h5py
    from agent.orchestration.annotation_runs import annotation_invocation
    from agent.orchestration.durable_tool_recovery import execution_identity
    from agent.schemas.prior_output import PriorOutputBinding
    from agent.tools import _native_process
    from agent.tools.analysis import peak_h5ad
    from agent.tools.analysis.sequence_prediction import read
    source_root=os.environ.get('ICTA4A_ACCEPTED_FIXTURE')
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION')!='1' or not source_root:
        pytest.skip('Explicit accepted multilevel fixture and native qualification')
    source_root=Path(source_root)
    before={str(p):file_sha256(p) for p in source_root.rglob('*') if p.is_file()}
    shutil.copytree(source_root/'run_state',tmp_path/'run_state')
    store=FileRunStore(tmp_path/'run_state');reg=registry(monkeypatch)
    previous=store.load('continued:run');old=previous.steps[0].result
    binding=bind_output(store,reg,previous.run_id,'annotate','dataset')
    validate_binding(binding,store=store,registry=reg,integrity=True)
    hierarchy=read(old['hierarchy_path'],old['hierarchy_sha256'])
    assert len(json.dumps(hierarchy,sort_keys=True,ensure_ascii=False).encode())>65536
    assert max(b['level'] for b in hierarchy['branches'])==2
    peak_h5ad.inspect(old['annotated_h5ad_path'])
    old_metadata=ad.read_h5ad(old['annotated_h5ad_path']).uns[owner.UNS]
    assert isinstance(old_metadata,str) and len(old_metadata.encode())>65536
    science=[PriorOutputBinding(**r['binding']) for r in hierarchy['operations']]
    child_states={b.run_id:store.load(b.run_id).to_dict() for b in science}
    lib=library();evidence=CellMarkerEvidence(lib,hierarchy['cellmarker']['resource_id'])
    assert evidence.identity==hierarchy['cellmarker']
    # Publication-only acceptance harness. The complete hierarchy is an exact
    # captured accepted result, not reconstructed or reinterpreted science.
    def publish_accepted(args,identity):
        captured=annotation_invocation().input_binding('accepted_hierarchy_path')
        assert captured==binding
        accepted=validate_binding(captured,store=store,registry=reg,integrity=True).result
        for child in science:validate_binding(child,store=store,registry=reg,integrity=True)
        return owner.publish(owner.arguments(args),identity,
            read(accepted['hierarchy_path'],accepted['hierarchy_sha256']),
            read(accepted['annotations_path'],accepted['annotations_sha256']),
            read(accepted['subdivisions_path'],accepted['subdivisions_sha256']),resource_files=evidence.entry['files'])
    spec=reg.get(owner.TOOL)
    spec=replace(spec,optional_arguments={**spec.optional_arguments,'accepted_hierarchy_path':ArgumentSpec((str,))},
        durable_hooks=DurableToolHooks(publish_accepted,owner.recover),
        semantic_planning=replace(spec.semantic_planning,consumer_ports=(*spec.semantic_planning.consumer_ports,
            SemanticConsumerPortSpec('accepted_hierarchy',(SemanticPortMember('dataset','accepted_hierarchy_path'),),False,
                accepted_upstream_types=('hierarchical_annotation_h5ad.v1',)))))
    scanpy=reg.get('run_scanpy')
    scanpy=replace(scanpy,semantic_planning=replace(scanpy.semantic_planning,consumer_ports=tuple(
        replace(p,accepted_upstream_types=(*p.accepted_upstream_types,'hierarchical_annotation_h5ad.v1'))
        if p.name=='dataset' else p for p in scanpy.semantic_planning.consumer_ports)))
    reg=ToolRegistry(tuple(spec if n==owner.TOOL else scanpy if n=='run_scanpy' else reg.get(n) for n in reg.names()))
    monkeypatch.setattr('agent.orchestration.registry.build_default_tool_registry',lambda:reg)
    monkeypatch.setattr('agent.orchestration.verification_authority.build_default_tool_registry',lambda:reg)
    args=dict(previous.plan.steps[0].arguments,accepted_hierarchy_path=PriorOutputRef(binding,'annotated_h5ad_path'))
    lifecycle=runpy.run_path(str(Path(__file__).parents[1]/'unit/orchestration/test_annotation_child_runs.py'))
    FaultStore,Crash=lifecycle['FaultStore'],lifecycle['Crash']
    fault=FaultStore(store,lambda s:s.run_id=='compact:run' and bool(s.steps)
        and s.steps[0].status.value=='SUCCEEDED',before=True)
    with monkeypatch.context() as patch:
        patch.setattr(_native_process,'call',lambda *a,**k:pytest.fail('accepted annotation science repeated'))
        patch.setattr(coordinator,'annotate_markers',lambda *a,**k:pytest.fail('ORA repeated'))
        patch.setattr(coordinator,'assess_subdivision',lambda *a,**k:pytest.fail('subdivision repeated'))
        with pytest.raises(Crash):
            AgentRuntime(registry=reg,run_store=fault).run_prepared(*prepare('compact',tmp_path,args))
        pending=store.load('compact:run')
        assert pending.steps[0].status.value=='RUNNING' and pending.steps[0].verification is None
        pins={str(p):file_sha256(p) for p in (tmp_path/'compact').iterdir() if p.is_file()}
        runtime=AgentRuntime(registry=reg,run_store=store)
        result=runtime.resume(pending.run_id)
        assert result.status.value=='SUCCEEDED',result.to_dict()
        authority=result.steps[0].verification.artifact_authority
        assert authority['schema_version']==2 and authority['execution_identity']==execution_identity(
            pending.run_id,pending.plan,pending.plan.steps[0],reg.get(owner.TOOL))
        recovered=bind_output(store,reg,result.run_id,'annotate','dataset')
        validate_binding(recovered,store=store,registry=reg,integrity=True)
        terminal=store.load(result.run_id).to_dict()
        assert runtime.resume(result.run_id).status.value=='SUCCEEDED'
        assert store.load(result.run_id).to_dict()==terminal
    value=result.steps[0].result
    matrix=ad.read_h5ad(value['annotated_h5ad_path']);original=ad.read_h5ad(hierarchy['source']['path'])
    peak_h5ad.inspect(value['annotated_h5ad_path'])
    owner.verify_h5ad(value['annotated_h5ad_path'],hierarchy)
    pd.testing.assert_frame_equal(matrix.obs,pd.concat([original.obs,owner.observation_columns(original.obs,hierarchy)],axis=1))
    pd.testing.assert_frame_equal(matrix.uns[owner.UNS]['branches'],owner.portable_hierarchy(hierarchy)['branches'])
    with h5py.File(value['annotated_h5ad_path']) as output,h5py.File(hierarchy['source']['path']) as original_file:
        assert all(owner._equal_node(original_file[k],output[k]) for k in original_file if k not in {'obs','uns'})
        assert output['uns'][owner.UNS].attrs['encoding-type']=='dict'
        assert output['uns'][owner.UNS]['branches'].attrs['encoding-type']=='dataframe'
    for key in ('hierarchy','annotations','subdivisions'):
        assert Path(value[key+'_path']).read_bytes()==Path(old[key+'_path']).read_bytes()
        assert value[key+'_sha256']==old[key+'_sha256']
    request=AgentRequest('consume-compact',None,{})
    plan=AgentPlan('consume-compact:plan',request.request_id,'Native consumption of complete multilevel hierarchy',(
        PlanStep('consumer','run_scanpy',dict(input_path=PriorOutputRef(recovered,'annotated_h5ad_path'),
            output_dir=str(tmp_path/'consumer'),function_name='pp.log1p')),))
    native_call=_native_process.call;calls=[]
    def only_consumer(operation,arguments,local):
        assert operation=='matrix' and arguments['package']=='scanpy'
        assert arguments['values']['input_path']==value['annotated_h5ad_path']
        assert arguments['values']['function_name']=='pp.log1p'
        calls.append(arguments['values']['function_name'])
        return native_call(operation,arguments,local)
    monkeypatch.setattr(_native_process,'call',only_consumer)
    consumed=runtime.run_prepared(request,plan)
    assert consumed.status.value=='SUCCEEDED',consumed.to_dict()
    downstream=bind_output(store,reg,consumed.run_id,'consumer','dataset')
    validate_binding(downstream,store=store,registry=reg,integrity=True)
    output=ad.read_h5ad(consumed.steps[0].result['matrix_path'])
    assert consumed.steps[0].result['source_identity']['sha256']==value['annotated_h5ad_sha256']
    pd.testing.assert_frame_equal(output.obs,matrix.obs)
    np.testing.assert_allclose(output.X.toarray(),np.log1p(matrix.X.toarray()))
    assert calls==['pp.log1p']
    # The generic reader also admits the unmodified historical long string.
    # This is an independent consumer, never a retry or rewrite of that source.
    monkeypatch.setattr(_native_process,'call',native_call)
    legacy_request=AgentRequest('consume-long-string',None,{})
    legacy_plan=AgentPlan('consume-long-string:plan',legacy_request.request_id,'Native consumption of complete long metadata',(
        PlanStep('consumer','run_scanpy',dict(input_path=PriorOutputRef(binding,'annotated_h5ad_path'),
            output_dir=str(tmp_path/'long-string-consumer'),function_name='pp.log1p')),))
    legacy=runtime.run_prepared(legacy_request,legacy_plan)
    assert legacy.status.value=='SUCCEEDED',legacy.to_dict()
    legacy_binding=bind_output(store,reg,legacy.run_id,'consumer','dataset')
    validate_binding(legacy_binding,store=store,registry=reg,integrity=True)
    legacy_matrix=ad.read_h5ad(legacy.steps[0].result['matrix_path'])
    assert legacy_matrix.uns[owner.UNS]==old_metadata
    assert legacy.steps[0].result['source_identity']['sha256']==old['annotated_h5ad_sha256']
    assert all(store.load(rid).to_dict()==saved for rid,saved in child_states.items())
    assert all(file_sha256(p)==sha for p,sha in {**before,**pins}.items())
    (tmp_path/'compatibility-evidence.json').write_text(json.dumps(dict(
        previous_binding=binding.to_dict(),recovered=result.to_dict(),binding=recovered.to_dict(),
        downstream=consumed.to_dict(),long_string_downstream=legacy.to_dict(),native_calls=calls+['pp.log1p'],
        unchanged_children=list(child_states),
        unchanged_old_files=before,unchanged_publication=pins,complete_hierarchy_bytes=Path(value['hierarchy_path']).stat().st_size),indent=2))
