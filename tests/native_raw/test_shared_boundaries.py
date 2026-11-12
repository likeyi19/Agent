"""Bounded ICTA-2 native boundary cleanup; actual installed scientific calls."""
import hashlib
import io
import json
import os
from pathlib import Path
from types import MappingProxyType

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.tools.analysis.native_matrix import SCANPY, EPISCANPY, SNAPATAC2, native_kwargs
from agent.tools.analysis.native_matrix_worker import returned_data
from agent.tools.analysis import categorical, peak_h5ad


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def native():
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION') != '1':
        pytest.skip('Explicit installed native runtime qualification')


@pytest.fixture
def source(tmp_path):
    a = ad.AnnData(sparse.csr_matrix([[2., 1.], [4., 2.], [1., 8.], [2., 6.]]),
        obs=pd.DataFrame({'group': pd.Categorical(['a', 'a', 'b', 'b'])}, index=['c0', 'c1', 'c2', 'c3']),
        var=pd.DataFrame(index=['chr1:100-120', 'chr1:900-920']))
    a.uns.update(species='human', assembly='stored-assembly', matrix_semantics='peak_counts')
    path=tmp_path/'input.h5ad';a.write_h5ad(path)
    return path


def execute(owner, source, root, function, **extra):
    before=sha(source)
    args=dict(input_path=str(source), input_sha256=before, output_dir=str(root), function_name=function, **extra)
    from agent.application import ResearchAgentApplication
    from agent.orchestration import AgentRuntime, AgentRequest, AgentPlan, PlanStep
    from agent.orchestration.prior_outputs import accepted_step_digest
    from agent.schemas.orchestration import _serialize
    app=ResearchAgentApplication(root.parent/(root.name+'-app'))
    runtime=AgentRuntime(registry=app.registry,run_store=app.run_store)
    run=runtime.run_prepared(AgentRequest('boundary',None,{}),AgentPlan('boundary','boundary','Native boundary fixture',(
        PlanStep('native',owner.TOOL,args),)))
    assert run.status.value=='SUCCEEDED', run.errors
    result=_serialize(run.steps[0].result)
    assert sha(source)==before
    assert result['source_identity']['sha256']==before
    from agent.orchestration.durable_tool_recovery import execution_identity
    state=app.run_store.load('boundary:run')
    identity=execution_identity(state.run_id,state.plan,state.plan.steps[0],app.registry.get(owner.TOOL))
    assert owner.recover(_serialize(state.steps[0].resolved_arguments), identity)==result
    assert result['parameters']['kwargs']==extra.get('kwargs', {})
    result['_fixture_accepted_step_sha256']=accepted_step_digest(run.steps[0])
    return result,ad.read_h5ad(result['matrix_path'])


def export(result, target):
    from agent.application.statistical_delivery import create_statistical_export
    path=Path(result['matrix_path'])
    with path.open('rb') as source:
        create_statistical_export(source, target, dict(tool_name='run_scanpy',accepted_step_sha256=result['_fixture_accepted_step_sha256']), source_sha256=result['matrix_sha256'],
            source_size_bytes=path.stat().st_size, manifest_sha256=result['manifest_sha256'], max_bytes=1024**3)
    return ad.read_h5ad(target)


def test_native_long_kwargs_reach_installed_qc(native, source, tmp_path):
    a=ad.read_h5ad(source)
    names=['scientific_feature_set_'+str(i)+'_'+'x'*140 for i in range(129)]
    a.var=pd.DataFrame({name:[True,False] for name in names},index=a.var_names)
    a.write_h5ad(source)
    kwargs=dict(qc_vars=names,percent_top=None,log1p=False,inplace=False)
    assert len(json.dumps(kwargs))>8192 and len(names)>128 and len(names[0])>128
    result,b=execute(SCANPY,source,tmp_path/'qc','pp.calculate_qc_metrics',kwargs=kwargs)
    returned=b.uns['agent_native_return']['value']['0']
    for name in names:
        np.testing.assert_array_equal(returned['total_counts_'+name],a.X.toarray()[:,0])
    np.testing.assert_array_equal(b.X.toarray(),a.X.toarray())


def test_native_sparse_centered_scale(native, source, tmp_path):
    x=ad.read_h5ad(source).X.toarray()
    result,b=execute(SCANPY,source,tmp_path/'scale','pp.scale',kwargs=dict(zero_center=True,copy=True))
    assert isinstance(b.X,np.ndarray)
    np.testing.assert_allclose(b.X,(x-x.mean(axis=0))/x.std(axis=0,ddof=1))
    assert sparse.issparse(ad.read_h5ad(source).X)
    assert result['parameters']['effective_native_parameters']['zero_center'] is True


@pytest.mark.parametrize('owner', [EPISCANPY,SNAPATAC2],ids=['episcanpy','snapatac2'])
def test_native_conflicting_declarations_preserve_both(native, source, tmp_path, owner):
    gtf=tmp_path/'genes.gtf'
    gtf.write_text(''.join(f'chr1\tHAVANA\t{kind}\t{start}\t{end}\t.\t+\t.\tgene_id "{gene}"; transcript_id "{gene}-tx"; gene_name "{gene}";\n'
        for kind in ('gene','transcript') for start,end,gene in [(101,200,'Alpha'),(801,1000,'Beta')]))
    if owner is SNAPATAC2:
        a=ad.read_h5ad(source);a.X=a.X.astype(np.uint32);a.write_h5ad(source)
    supplied=dict(species='mouse',assembly='supplied-assembly',matrix_semantics='different_declared_measurement')
    result,b=execute(owner,source,tmp_path/'activity','tl.geneactivity' if owner is EPISCANPY else 'pp.make_gene_matrix',
        gene_anno_path=str(gtf),gene_anno_sha256=sha(gtf),kwargs=dict(upstream=0,**({'use_x':True} if owner is SNAPATAC2 else {})),**supplied)
    assert result['source_identity']['supplied_declarations']==supplied
    desc=result['parameters']['gene_activity']
    assert desc['source_measurement']=='peak_counts'
    assert desc['declarations']==dict(species='human',assembly='stored-assembly')
    assert b.uns['species']=='human' and b.uns['assembly']=='stored-assembly'
    np.testing.assert_array_equal(b.X.toarray() if sparse.issparse(b.X) else b.X,ad.read_h5ad(source).X.toarray())
    manifest=json.loads(Path(result['manifest_path']).read_text())
    assert all(manifest['arguments'][k]==v for k,v in supplied.items())


def test_native_wide_record_publication_and_delivery(native, source, tmp_path):
    a=ad.read_h5ad(source)
    records=np.zeros(2,dtype=[('group_'+str(i),'f8') for i in range(129)])
    for i,name in enumerate(records.dtype.names): records[name]=[i,i+0.5]
    a.uns['native_statistics']=records;a.write_h5ad(source)
    result,b=execute(SCANPY,source,tmp_path/'wide','pp.log1p',kwargs=dict(base=2))
    np.testing.assert_array_equal(b.uns['native_statistics'],records)
    np.testing.assert_array_equal(export(result,io.BytesIO()).uns['native_statistics'],records)


def test_native_complex_result_publication_and_delivery(native, source, tmp_path):
    a=ad.read_h5ad(source);a.X=a.X.toarray().astype(complex)+0.5j;a.write_h5ad(source)
    result,b=execute(SCANPY,source,tmp_path/'complex','pp.log1p',kwargs=dict(base=2))
    assert b.X.dtype.kind=='c'
    np.testing.assert_allclose(b.X,np.log1p(a.X)/np.log(2))
    np.testing.assert_array_equal(export(result,io.BytesIO()).X,b.X)


def test_native_repeated_population_reading_and_association(native, source, tmp_path):
    result,b=execute(SCANPY,source,tmp_path/'repeated','pp.calculate_qc_metrics',
        cell_subset=dict(positions=[3,1,3]),kwargs=dict(percent_top=None,log1p=False))
    ids,labels=categorical.observations(result['matrix_path'],result['matrix_sha256'],'group')
    assert ids==('c3','c1','c3') and labels.tolist()==['b','a','b']
    assert categorical.counts(labels)['n_entities']==3
    associated=categorical.association(ids,labels,ids,labels,left_source=result['matrix_sha256'],right_source=result['matrix_sha256'])
    assert associated['n_paired']==3
    with pytest.raises(ValueError,match='same pinned artifact'):
        categorical.association(ids,labels,ids,labels,left_source=result['matrix_sha256'],right_source='0'*64)


def test_data_only_json_has_no_arbitrary_depth_or_collection_bound():
    value=dict(leaf='line\nvalue')
    for _ in range(10): value=dict(nested=value)
    kwargs=dict(metric_kwds=dict(value,**{str(i):i for i in range(150)}))
    assert native_kwargs({'fn':('metric_kwds',)},dict(function_name='fn',kwargs=kwargs))==kwargs


@pytest.mark.parametrize('value',[float('nan'),float('inf'),object(),lambda:None])
def test_non_json_arguments_remain_rejected(value):
    with pytest.raises(ValueError):
        native_kwargs({'fn':('value',)},dict(function_name='fn',kwargs=dict(value=value)))


def test_data_only_mapping_return_preserves_keys_and_arrays(tmp_path):
    value=MappingProxyType({np.str_('values'):np.array([1+2j,3+4j])})
    a=ad.AnnData(np.ones((1,1)));a.uns['return']=returned_data(value)
    path=tmp_path/'return.h5ad';a.write_h5ad(path);peak_h5ad.inspect(path)
    np.testing.assert_array_equal(ad.read_h5ad(path).uns['return']['values'],value['values'])
    with pytest.raises(TypeError,match='string keys'):
        returned_data({1:'one','1':'different'})


def test_inspector_keeps_external_reference_and_host_allocation_bound(source,tmp_path):
    with pytest.raises(ValueError,match='storage limit'):
        peak_h5ad.inspect(source,max_logical_bytes=1)
    with h5py.File(source,'r+') as handle:
        handle['uns']['external']=h5py.ExternalLink(str(tmp_path/'untrusted.h5'),'data')
    with pytest.raises(ValueError,match='Indirect HDF5'):
        peak_h5ad.inspect(source)


@pytest.mark.parametrize('fixed', [False, True])
def test_complete_long_utf8_scalar_vector_and_record_strings(source, fixed):
    text='细胞🙂'*12000+'完整结尾'
    encoded=text.encode('utf-8');assert len(encoded)>65536
    dtype=h5py.string_dtype('utf-8',length=len(encoded) if fixed else None)
    with h5py.File(source,'r+') as handle:
        scalar=handle['uns'].create_dataset('long_scalar',data=encoded,dtype=dtype)
        vector=handle['uns'].create_dataset('long_vector',data=[encoded,b'short',encoded],dtype=dtype)
        records=np.empty(2,dtype=[('names',dtype),('score','f8')])
        records['names']=[encoded,b'short'];records['score']=[1.,2.]
        record=handle['uns'].create_dataset('long_record',data=records)
        assert peak_h5ad.strings(scalar,())==[text]
        assert peak_h5ad.strings(vector,slice(0,3))==[text,'short',text]
        assert peak_h5ad.strings(vector,slice(1,3))==['short',text]
        assert list(peak_h5ad.record_strings(record,'names'))==[text,'short']
    before=source.read_bytes()
    size=peak_h5ad.inspect(source,privacy=True)
    assert size>=4*len(encoded)
    with pytest.raises(ValueError,match='storage limit'):
        peak_h5ad.inspect(source,max_logical_bytes=len(encoded))
    assert source.read_bytes()==before


def test_large_attributes_use_aggregate_budget_without_field_caps(source):
    text='注释🙂'*12000+'结尾';numbers=np.arange(10000,dtype='int64')
    with h5py.File(source,'r+') as handle:
        group=handle['uns'].create_group('metadata',track_order=True)
        group.attrs['long_text']=text
        group.attrs.create('long_array',np.array([text,'末尾'],dtype=object),dtype=h5py.string_dtype('utf-8'))
        group.attrs['numeric_metadata']=numbers
        assert group.attrs.get_id('numeric_metadata').get_storage_size()>65536
    before=source.read_bytes()
    size=peak_h5ad.inspect(source,privacy=True)
    assert size>=2*len(text.encode('utf-8'))+numbers.nbytes
    with pytest.raises(ValueError,match='storage limit'):
        peak_h5ad.inspect(source,max_logical_bytes=100000)
    with h5py.File(source) as handle:
        attrs=handle['uns/metadata'].attrs
        assert attrs['long_text']==text and attrs['long_array'].tolist()==[text,'末尾']
        np.testing.assert_array_equal(attrs['numeric_metadata'],numbers)
    assert source.read_bytes()==before


@pytest.mark.parametrize('nested',[False,True])
@pytest.mark.parametrize('vector',[False,True])
@pytest.mark.parametrize('private',[False,True])
def test_compound_attribute_complete_strings_accounting_and_privacy(source,nested,vector,private):
    baseline=peak_h5ad.inspect(source)
    text='细胞🙂'*12000+(' /private/research/hidden.h5ad' if private else '完整结尾')
    dtype=np.dtype([('text',h5py.string_dtype('utf-8')),('scores','f8',(3,)),('count','i8')])
    values=np.zeros(2 if vector else (),dtype=[('metadata',dtype)] if nested else dtype)
    fields=values['metadata'] if nested else values
    fields['text']=text.encode('utf-8');fields['scores']=[1.,2.,3.];fields['count']=7
    with h5py.File(source,'r+') as handle:handle['uns'].attrs.create('compound',values)
    before=source.read_bytes()
    expected=baseline+fields.size*(len(text.encode('utf-8'))+8)+fields['scores'].nbytes+fields['count'].nbytes
    assert expected-baseline>65536 and values.nbytes<expected-baseline
    assert peak_h5ad.inspect(source,max_logical_bytes=expected)==expected
    with pytest.raises(ValueError,match='storage limit'):
        peak_h5ad.inspect(source,max_logical_bytes=expected-1)
    if private:
        with pytest.raises(ValueError,match='Private locator'):
            peak_h5ad.inspect(source,privacy=True)
    else:assert peak_h5ad.inspect(source,privacy=True)==expected
    with h5py.File(source) as handle:
        saved=np.asarray(handle['uns'].attrs['compound'])
        saved=saved['metadata'] if nested else saved
        assert [v.decode('utf-8') for v in saved['text'].ravel()]==[text]*fields.size
        np.testing.assert_array_equal(saved['scores'],fields['scores'])
        np.testing.assert_array_equal(saved['count'],fields['count'])
    assert source.read_bytes()==before


def test_compound_attribute_strings_remain_strict_utf8(source):
    values=np.array([(b'bad\xff',)],dtype=[('text',h5py.string_dtype('utf-8'))])
    with h5py.File(source,'r+') as handle:handle['uns'].attrs.create('compound',values)
    with pytest.raises(UnicodeDecodeError):peak_h5ad.inspect(source)


@pytest.mark.parametrize('kind',['scalar','vector','record','attribute'])
def test_long_string_privacy_inspects_the_complete_tail(source, kind):
    text='细胞'*12000+' /private/research/hidden.h5ad'
    with h5py.File(source,'r+') as handle:
        if kind=='attribute':handle['uns'].attrs['long']=text
        elif kind=='record':
            values=np.empty(1,dtype=[('name',h5py.string_dtype('utf-8'))]);values['name']=[text.encode()]
            handle['uns'].create_dataset('long',data=values)
        else:handle['uns'].create_dataset('long',data=text if kind=='scalar' else ['safe',text],dtype=h5py.string_dtype('utf-8'))
    peak_h5ad.inspect(source)
    with pytest.raises(ValueError,match='Private locator'):
        peak_h5ad.inspect(source,privacy=True)
    if kind in {'scalar','vector'}:
        peak_h5ad.inspect(source,privacy=True,private_paths=('/uns/long',))


@pytest.mark.parametrize('record',[False,True])
def test_string_decoding_remains_strict_utf8(source,record):
    with h5py.File(source,'r+') as handle:
        value=np.array([(b'bad\xff',)],dtype=[('name','S4')]) if record else np.bytes_(b'bad\xff')
        handle['uns'].create_dataset('invalid',data=value)
    with pytest.raises(UnicodeDecodeError):peak_h5ad.inspect(source)


@pytest.mark.parametrize('storage',['soft_link','virtual','external','filter','reference','chunk'])
def test_string_limit_removal_preserves_storage_safety(source,tmp_path,storage):
    with h5py.File(source,'r+') as handle:
        group=handle['uns']
        if storage=='soft_link':group['unsafe']=h5py.SoftLink('/X')
        elif storage=='virtual':
            layout=h5py.VirtualLayout(shape=(2,),dtype='f8')
            layout[:]=h5py.VirtualSource(str(tmp_path/'other.h5'),'data',shape=(2,))
            group.create_virtual_dataset('unsafe',layout)
        elif storage=='external':
            group.create_dataset('unsafe',shape=(2,),dtype='f8',external=[(str(tmp_path/'other.bin'),0,h5py.h5f.UNLIMITED)])
        elif storage=='filter':group.create_dataset('unsafe',data=[1.,2.],compression='lzf')
        elif storage=='reference':group.create_dataset('unsafe',shape=(1,),dtype=h5py.ref_dtype)
        else:group.create_dataset('unsafe',shape=(100,),chunks=(100,),dtype='f8')
    with peak_h5ad.inspection_limits(1024**2,512),pytest.raises(ValueError,match='Indirect|layout|filter|dtype|chunk'):
        peak_h5ad.inspect(source)
