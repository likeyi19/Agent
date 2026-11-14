"""Opt-in genuine two-branch ICTA science, resource overlap and equivalence."""
import json
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.orchestration import AgentRuntime, FileRunStore, build_default_tool_registry
from agent.schemas import AgentPlan, AgentRequest, PlanStep
from agent.tools._file_identity import file_sha256
from agent.tools.analysis.sequence_prediction import library
from agent.tools.data.cellmarker import CellMarkerEvidence

pytestmark = pytest.mark.skipif(os.environ.get('RUN_RAW_NATIVE_QUALIFICATION') != '1',
    reason='Explicit installed native scientific qualification')


def test_genuine_parallel_annotation_matches_serial(tmp_path, monkeypatch):
    root = Path(os.environ.get('ICTA_PARALLEL_QUALIFICATION_ROOT', tmp_path))
    root.mkdir(exist_ok=True, parents=True)
    if all((root/(mode+'-result.json')).is_file() for mode in ('serial','parallel')):
        _verify_accepted(root)
        return
    rng = np.random.default_rng(2026)
    probability = np.full((200,160), .2)
    for offset in (0,100):
        probability[offset:offset+50,:20]=.8; probability[offset:offset+50,20:40]=.04
        probability[offset+50:offset+100,:20]=.04; probability[offset+50:offset+100,20:40]=.8
    x=rng.binomial(1,probability).astype('uint32')
    source=root/'original.h5ad'
    data=ad.AnnData(sparse.csr_matrix(x),obs=pd.DataFrame({'original':pd.Categorical(['A']*100+['B']*100)},
        index=[f'cell-{i}' for i in range(200)]),var=pd.DataFrame(index=[f'chr1:{1000+i*200}-{1100+i*200}' for i in range(160)]))
    data.layers['original_counts']=data.X.copy(); data.raw=data.copy()
    data.uns['fixture']='Transparent Bernoulli accessibility, two known independent parent branches; no biological validation'
    data.write_h5ad(source); before=file_sha256(source)
    genes=['Cd3d','Cd3e','Trac','Cd247']+[f'fixture_gene_{i}' for i in range(156)]
    gtf=root/'genes.gtf'
    gtf.write_text(''.join(f'chr1\tHAVANA\tgene\t{1001+i*200}\t{1100+i*200}\t.\t+\t.\tgene_id "g{i}"; gene_name "{gene}";\n' for i,gene in enumerate(genes)))
    lib=library();entry=next(e for e in lib.entries() if e['kind']=='cellmarker');evidence=CellMarkerEvidence(lib,entry['id'])
    args=dict(input_path=str(source),input_sha256=before,group_column='original',
        activity_method='episcanpy.tl.geneactivity',activity_kwargs={'upstream':0},
        gene_anno_path=str(gtf),gene_anno_sha256=file_sha256(gtf),cellmarker_id=entry['id'],
        cellmarker_identity=evidence.identity,species='mouse',maximum_levels=2,
        local_kwargs={'pp.select_features':{'n_features':140},'tl.spectral':{'n_comps':8,'random_state':0},'pp.knn':{'n_neighbors':10}})
    memory=4*1024**3
    monkeypatch.setenv('AGENT_NATIVE_MEMORY_BYTES',str(memory));monkeypatch.setenv('AGENT_NATIVE_CPU_COUNT','4')
    store=FileRunStore(root/'state');reg=build_default_tool_registry();runtime=AgentRuntime(registry=reg,run_store=store)
    runs={};results={};measurements={}
    for mode,total in [('serial',memory),('parallel',memory*2)]:
        monkeypatch.setenv('AGENT_NATIVE_TOTAL_MEMORY_BYTES',str(total))
        request=AgentRequest('icta-native-'+mode,None,{})
        plan=AgentPlan(request.request_id+':plan',request.request_id,'Focused native concurrency qualification',
            (PlanStep('annotate','annotate_scATAC_cell_types',dict(args,output_dir=str(root/mode))),))
        started=time.monotonic();result=runtime.run_prepared(request,plan)
        measurements[mode]={'wall_seconds':time.monotonic()-started}
        (root/(mode+'-result.json')).write_text(json.dumps(result.to_dict(),indent=2)+'\n')
        assert result.status.value=='SUCCEEDED',result.errors
        output=result.steps[0].result
        results[mode]=ad.read_h5ad(output['annotated_h5ad_path'])
        runs[mode]=json.loads((root/mode/'hierarchy.json').read_text())
        assert file_sha256(source)==before
    _verify_accepted(root)


def _verify_accepted(root):
    source=root/'original.h5ad';before=file_sha256(source);data=ad.read_h5ad(source)
    results={mode:ad.read_h5ad(root/mode/'annotated.h5ad') for mode in ('serial','parallel')}
    runs={mode:json.loads((root/mode/'hierarchy.json').read_text()) for mode in results}
    for mode in results:
        from agent.tools.analysis.hierarchical_annotation import read_evidence
        annotations=read_evidence(root/mode/'annotations.json',file_sha256(root/mode/'annotations.json'))
        assert all(a['reference']['species_binding']=={'declared':'mouse','resolved':'Mouse'} for a in annotations.values())
    store=FileRunStore(root/'state');runtime=AgentRuntime(registry=build_default_tool_registry(),run_store=store)
    from datetime import datetime
    measurements={}
    for mode in results:
        state=store.load('icta-native-'+mode+':run')
        measurements[mode]={'run_lifetime_seconds':(datetime.fromisoformat(state.updated_at)-datetime.fromisoformat(state.created_at)).total_seconds()}
    pd.testing.assert_frame_equal(results['serial'].obs,results['parallel'].obs)
    assert (results['parallel'].X!=data.X).nnz==0
    assert (results['parallel'].layers['original_counts']!=data.layers['original_counts']).nnz==0
    assert (results['parallel'].raw.X!=data.raw.X).nnz==0
    branch_fields=('branch_id','parent','cluster','level','positions','residual_positions','annotation','stopping_reason')
    assert [{k:b[k] for k in branch_fields} for b in runs['serial']['branches']]==[{k:b[k] for k in branch_fields} for b in runs['parallel']['branches']]
    scientific=[]
    for a,b in zip(runs['serial']['operations'],runs['parallel']['operations'],strict=True):
        assert (a['branch_id'],a['operation_id'])==(b['branch_id'],b['operation_id'])
        aa,bb=a['native_result'],b['native_result']
        assert aa['function_name']==bb['function_name']
        assert aa['parameters'].get('effective_native_parameters')==bb['parameters'].get('effective_native_parameters')
        da,db=(ad.read_h5ad(n['matrix_path']) for n in (aa,bb))
        assert da.obs_names.equals(db.obs_names) and da.var_names.equals(db.var_names)
        assert (sparse.csr_matrix(da.X)!=sparse.csr_matrix(db.X)).nnz==0
        if 'X_spectral' in da.obsm:np.testing.assert_allclose(da.obsm['X_spectral'],db.obsm['X_spectral'],rtol=1e-9,atol=1e-9)
        if aa['function_name'].endswith('tl.diff_test'):
            pd.testing.assert_frame_equal(da.uns['agent_native_return']['value'],db.uns['agent_native_return']['value'])
        scientific.append({'branch_id':a['branch_id'],'operation_id':a['operation_id'],'equivalent':True})
    from datetime import datetime
    diagnostics={}
    for mode in runs:
        diagnostics[mode]=[]
        for operation in runs[mode]['operations']:
            n=operation['native_result'];p=Path(n['matrix_path']).parent
            d=json.loads((p.parent/('.'+p.name+'-execution.json')).read_text())
            d.update(branch_id=operation['branch_id'],operation_id=operation['operation_id'])
            diagnostics[mode].append(d)
    overlaps=[]
    for a in diagnostics['parallel']:
        for b in diagnostics['parallel']:
            if a['branch_id']>=b['branch_id'] or 'initial' in (a['branch_id'],b['branch_id']):continue
            lo=max(datetime.fromisoformat(a['started_at']),datetime.fromisoformat(b['started_at']))
            hi=min(datetime.fromisoformat(a['finished_at']),datetime.fromisoformat(b['finished_at']))
            if hi>lo:overlaps.append({'a':a['unit'],'b':b['unit'],'seconds':(hi-lo).total_seconds()})
    assert overlaps
    for mode in diagnostics:
        assert all(d['reservation']['reserved_memory_bytes']<=d['reservation']['aggregate_memory_bytes'] for d in diagnostics[mode])
    (root/'parallel-evidence.json').write_text(json.dumps(dict(measurements=measurements,overlaps=overlaps,
        operations=scientific,diagnostics=diagnostics,source_sha256=before,source_unchanged=file_sha256(source)==before,
        deterministic_hierarchy=True),indent=2)+'\n')
    exact=store.load('icta-native-parallel:run').to_dict()
    assert runtime.resume('icta-native-parallel:run').status.value=='SUCCEEDED'
    assert store.load('icta-native-parallel:run').to_dict()==exact


def test_genuine_native_call_overlap_and_resource_usage(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime
    root=Path(os.environ.get('ICTA_NATIVE_INTERVAL_ROOT',tmp_path));root.mkdir(exist_ok=True,parents=True)
    memory=4*1024**3
    monkeypatch.setenv('AGENT_NATIVE_MEMORY_BYTES',str(memory));monkeypatch.setenv('AGENT_NATIVE_TOTAL_MEMORY_BYTES',str(memory*2))
    rng=np.random.default_rng(109)
    inputs=[];hashes=[]
    for branch in ('A','B'):
        probability=np.full((2000,1000),.2)
        probability[:1000,:500]=.8;probability[1000:,500:]=.8
        data=ad.AnnData(sparse.csr_matrix(rng.binomial(1,probability).astype('uint32')),
            var=pd.DataFrame(index=[f'chr1:{1000+i*200}-{1100+i*200}' for i in range(1000)]))
        p=root/(branch+'.h5ad');data.write_h5ad(p);inputs.append(p);hashes.append(file_sha256(p))
    runtime=AgentRuntime(registry=build_default_tool_registry(),run_store=FileRunStore(root/'state'))
    def operation(mode,i):
        name=mode+'-'+str(i);request=AgentRequest(name,None,{})
        plan=AgentPlan(name+':plan',name,'Independent native comparison fixture',(PlanStep('diff','run_snapatac2',dict(
            input_path=str(inputs[i]),output_dir=str(root/name),function_name='tl.diff_test',
            kwargs=dict(cell_group1=list(range(1000)),cell_group2=list(range(1000,2000)),direction='positive',min_log_fc=.25,min_pct=.05))),))
        result=runtime.run_prepared(request,plan)
        assert result.status.value=='SUCCEEDED',result.errors
        return result.steps[0].result
    started=time.perf_counter();serial=[operation('serial',i) for i in range(2)];serial_seconds=time.perf_counter()-started
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:parallel=list(pool.map(lambda i:operation('parallel',i),range(2)))
    parallel_seconds=time.perf_counter()-started
    records=[];receipts=[]
    for i,(a,b) in enumerate(zip(serial,parallel,strict=True)):
        left,right=(ad.read_h5ad(n['matrix_path']).uns['agent_native_return']['value'] for n in (a,b))
        pd.testing.assert_frame_equal(left,right)
        assert file_sha256(inputs[i])==hashes[i]
        out=Path(b['matrix_path']).parent
        receipts.append(json.loads((out/'native-receipt.json').read_text()))
        records.append(json.loads((out.parent/('.'+out.name+'-execution.json')).read_text()))
    overlap=(min(datetime.fromisoformat(x['native_finished_at']) for x in receipts)-
        max(datetime.fromisoformat(x['native_started_at']) for x in receipts)).total_seconds()
    assert overlap>0
    assert all(x['worker_resource_usage']['descendants']['max_rss_kib']>0 for x in records)
    (root/'native-interval-evidence.json').write_text(json.dumps(dict(serial_seconds=serial_seconds,parallel_seconds=parallel_seconds,
        native_call_overlap_seconds=overlap,scientific_equivalence=True,source_hashes=hashes,receipts=receipts,execution=records),indent=2)+'\n')


@pytest.mark.parametrize('mode',['cancel','failure'])
def test_genuine_parallel_children_cancel_or_fail_without_losing_sibling(tmp_path,monkeypatch,mode):
    import runpy,threading
    from dataclasses import replace
    from agent.orchestration.registry import ToolRegistry,DurableToolHooks
    from agent.orchestration.annotation_runs import annotation_invocation,continuation_anchor
    from agent.orchestration.annotation_registry import classify
    native_registry=build_default_tool_registry()
    fixture=runpy.run_path(str(Path(__file__).parents[1]/'unit/orchestration/test_annotation_child_runs.py'))
    h=fixture['harness'].__wrapped__(tmp_path,monkeypatch)
    rng=np.random.default_rng(51)
    p=np.full((2000,1000),.2);p[:1000,:500]=.8;p[1000:,500:]=.8
    data=ad.AnnData(sparse.csr_matrix(rng.binomial(1,p).astype('uint32')))
    source=tmp_path/'accessibility.h5ad';data.write_h5ad(source);h.source=str(source);h.sha=file_sha256(source)
    monkeypatch.setenv('AGENT_NATIVE_MEMORY_BYTES',str(4*1024**3));monkeypatch.setenv('AGENT_NATIVE_TOTAL_MEMORY_BYTES',str(8*1024**3))
    def coordinate(args,identity):
        inv=annotation_invocation()
        def child(label):
            kwargs=dict(cell_group1=list(range(1000)),cell_group2=list(range(1000,2000)),direction='positive')
            if mode=='failure' and label=='a':kwargs['covariates']=['unimplemented']
            return inv.invoke(branch_id=label,operation_id='diff',tool_name='run_snapatac2',
                arguments=dict(function_name='tl.diff_test',kwargs=kwargs),
                parent_inputs=dict(input_path='input_path',input_sha256='input_sha256'))
        inv.map_branches(child,['a','b'])
        raise AssertionError('An unresolved failure/cancellation cannot publish a result')
    parent=replace(h.registry.get('parent'),annotation_children=frozenset({'run_snapatac2'}),
        durable_hooks=DurableToolHooks(coordinate,h.registry.get('parent').durable_hooks.recover),exception_classifier=classify)
    h.registry=ToolRegistry((parent,native_registry.get('run_snapatac2')))
    results=[];errors=[]
    def execute():
        try:results.append(h.runtime().run_prepared(*h.prepare()))
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=execute);thread.start()
    if mode=='cancel':
        ready=[];deadline=time.monotonic()+45
        while time.monotonic()<deadline and thread.is_alive():
            ready=[]
            for path in (Path('/run/user')/str(os.getuid())).glob('agent-native-*/membership.json'):
                job=json.loads((path.parent/'request.json').read_text())
                if job['operation']=='matrix' and job['arguments']['values']['output_dir'].startswith(str(tmp_path/'parent-1.children')):
                    ready.append(json.loads(path.read_text()))
            if len(ready)==2:break
            time.sleep(.02)
        assert len(ready)==2,(ready,errors)
        h.runtime().cancel('parent-1:run')
    thread.join(90);assert not thread.is_alive() and not errors,errors
    assert results[0].status.value==('CANCELLED' if mode=='cancel' else 'FAILED')
    assert results[0].steps[0].result is None
    rows=fixture['reservations'](h.store.load('parent-1:run'))
    children=[h.store.load(r['request']['request_id']+':run') for r in rows]
    if mode=='cancel':
        assert all(c.lifecycle_status.value=='CANCELLED' for c in children)
        assert all(not Path(record['cgroup']).exists() for record in ready)
    else:
        successful=[c for c in children if c.lifecycle_status.value=='SUCCEEDED'];assert len(successful)==1
        original=successful[0].to_dict()
        anchor=continuation_anchor(h.store,'parent-1:run','annotate')
        result=h.runtime().continue_annotation(*h.prepare('parent-2'),previous=anchor)
        assert result.status.value=='FAILED'
        assert h.store.load(successful[0].run_id).to_dict()==original
    assert file_sha256(source)==h.sha
    assert not list((Path('/run/user')/str(os.getuid())/'agent-native-reservations').glob('*.json'))
