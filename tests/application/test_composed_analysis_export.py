"""OPT-2C storage/preparation checks with simulated recipe metadata and real H5AD.

The normal admission/lineage integration is tested separately. These fixtures
exercise the on-demand owner without executing scientific algorithms.
"""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import analysis_delivery as delivery
from agent.application import matrix_delivery as matrix
from test_analysis_export import export_fixture, sha
from test_dialogue_evidence import app


def composed_fixture(tmp_path, *, encoding='csr', legacy=False):
    old = export_fixture(tmp_path/'old', encoding=encoding)
    if legacy:
        from anndata.io import write_elem
        with h5py.File(old.source, 'r+') as handle:
            obs = handle['obs']; obs.attrs['encoding-version'] = '0.1.0'
            categories = obs.create_group('__categories')
            for key in ('cohort', 'leiden'):
                values = old.original.obs[key]
                del obs[key]
                write_elem(obs, key, values.cat.codes.to_numpy())
                write_elem(categories, key, np.asarray(values.cat.categories, dtype=object))
                obs[key].attrs['categories'] = categories[key].ref
                categories[key].attrs['ordered'] = values.cat.ordered
        old = export_fixture(tmp_path/'old-legacy', source=old.source)
    new = export_fixture(tmp_path/'new', stage='clustering', source=old.source)
    data = ad.read_h5ad(new.native)
    data.obs['leiden'] = pd.Categorical(['B', 'A', 'B', 'A'])
    data.write_h5ad(new.native)
    source = lambda path: dict(path=str(path), managed_root=str(path.parent),
        sha256=sha(path), size_bytes=path.stat().st_size)
    sources = [source(old.native), source(new.native)]
    locator = lambda i: dict(name='analysis', run_id=f'accepted-{i}', step_id='science',
        output_key='analysis_path', accepted_step_sha256=str(i)*64, revision_id=f'revision-{i}')
    rows = [dict(slot='obs', source_index=1, accepted_output=locator(2), field='result:leiden',
                 destination='selected_leiden', method={'clustering': {'method':'Leiden','parameters': {'resolution': .75}}},
                 observation_role='computational_assignment'),
            dict(slot='obs', source_index=0, accepted_output=locator(1), field='source:cohort',
                 destination='selected_supplied_cohort', method={}, observation_role='supplied_source_observation'),
            dict(slot='obsm', source_index=0, accepted_output=locator(1), field='obsm:X_umap',
                 destination='X_umap_previous', method={'umap': {'method':'UMAP'}}),
            dict(slot='obsm', source_index=1, accepted_output=locator(2), field='obsm:X_pca',
                 destination='X_pca_selected', method={'embedding': {'method':'PCA'}})]
    contents = dict(policy=delivery.COMPOSED_CONTRACT,
        measurement=dict(origin='accepted_output', accepted_output=locator(1), source_sha256=sha(old.source)),
        observations=[{k:v for k,v in r.items() if k not in ('slot','source_index')} for r in rows if r['slot']=='obs'],
        representations=[{k:v for k,v in r.items() if k not in ('slot','source_index')} for r in rows if r['slot']=='obsm'],
        omitted_information=['Unselected derived results are not added.'], preparation='on_demand',
        payload_checks='performed_on_download')
    for row in rows:
        path=old.source if row['field'].startswith('source:') else Path(sources[row['source_index']]['path'])
        with h5py.File(path) as handle:
            row['logical_bytes']=delivery._field_logical_bytes(handle,row['slot'],row['field'].split(':',1)[1])
    recipe = dict(original_source=source(old.source), sources=sources, components=rows,
        contents=contents, recipe_identity={'policy':delivery.COMPOSED_CONTRACT, 'selected_fields': contents})
    artifact = SimpleNamespace(path=old.source, managed_root=old.source.parent,
        sha256=sha(old.source), size_bytes=old.source.stat().st_size,
        contract=delivery.COMPOSED_CONTRACT, role=delivery.COMPOSED_ROLE, manifest=recipe,
        reference=SimpleNamespace(filename=delivery.COMPOSED_FILENAME))
    return artifact, old, new


def prepare(artifact):
    preparer = matrix.MatrixDownloadPreparer()
    with preparer.prepare(artifact) as result:
        raw = b''.join(result.iter_chunks())
        assert result.sha256 == hashlib.sha256(raw).hexdigest()
        assert result.size_bytes == len(raw) and result.source_sha256 == artifact.sha256
        assert result.filename == delivery.COMPOSED_FILENAME
    assert preparer.active_count == preparer._reserved_bytes == 0
    return ad.read_h5ad(io.BytesIO(raw)), raw


@pytest.mark.parametrize('encoding', ['csr', 'csc', 'dense'])
def test_composed_preserves_original_storage_and_only_explicit_cross_result_fields(tmp_path,monkeypatch,encoding):
    artifact, old, new = composed_fixture(tmp_path, encoding=encoding)
    before = {p:p.read_bytes() for p in (old.source,old.native,new.native)}
    def no_densification(*a,**k):
        pytest.fail('Export densified the original sparse matrix.')
    monkeypatch.setattr(sparse.csr_matrix,'toarray',no_densification)
    monkeypatch.setattr(sparse.csc_matrix,'toarray',no_densification)
    result, raw = prepare(artifact)
    original = old.original
    assert type(result.X) is type(original.X)
    if sparse.issparse(original.X):
        for field in ('data','indices','indptr'):
            np.testing.assert_array_equal(getattr(result.X,field),getattr(original.X,field))
    else:
        np.testing.assert_array_equal(result.X,original.X)
    assert result.obs_names.equals(original.obs_names) and result.var_names.equals(original.var_names)
    pd.testing.assert_frame_equal(result.var,original.var)
    pd.testing.assert_series_equal(result.obs['leiden'],original.obs['leiden'])
    pd.testing.assert_series_equal(result.obs['selected_leiden'],ad.read_h5ad(new.native).obs['leiden'],check_names=False)
    pd.testing.assert_series_equal(result.obs['selected_supplied_cohort'],original.obs['cohort'],check_names=False)
    np.testing.assert_array_equal(result.obsm['X_umap_previous'],old.result.obsm['X_umap'])
    np.testing.assert_array_equal(result.obsm['X_pca_selected'],new.result.obsm['X_pca'])
    np.testing.assert_array_equal(result.obsm['X_pca'],original.obsm['X_pca'])
    assert 'X_umap' not in result.obsm and 'agent_leiden' not in result.obs
    assert result.raw.var_names.equals(original.raw.var_names)
    assert set(result.layers)==set(original.layers) and set(result.obsp)==set(original.obsp)
    assert set(result.varm)==set(original.varm)
    metadata=result.uns[delivery.EXPORT_KEY]
    assert metadata['contract']==delivery.COMPOSED_CONTRACT
    selected=json.loads(metadata['selection_json'])
    assert selected['measurement']['source_sha256']==artifact.sha256
    assert selected['observations'][0]['accepted_output']['revision_id']=='revision-2'
    assert selected['representations'][0]['accepted_output']['revision_id']=='revision-1'
    assert 'does not issue scientific authority' in metadata['authority_scope']
    assert b'/private/' not in raw and b'must-not-leak' not in raw
    with h5py.File(old.source) as source, h5py.File(io.BytesIO(raw)) as target:
        for name in ('X','layers/accessibility'):
            key=name if encoding=='dense' else name+'/data'
            assert source[key].compression==target[key].compression=='gzip'
    assert all(p.read_bytes()==value for p,value in before.items())


def test_composed_legacy_original_categories_rebind_without_full_obs_decoding(tmp_path,monkeypatch):
    artifact,old,new=composed_fixture(tmp_path,legacy=True)
    from anndata._io.specs import methods
    # Native selected-column AnnData reader owns the legacy category encoding.
    result,raw=prepare(artifact)
    pd.testing.assert_series_equal(result.obs['cohort'],old.original.obs['cohort'])
    pd.testing.assert_series_equal(result.obs['selected_supplied_cohort'],old.original.obs['cohort'],check_names=False)
    with h5py.File(io.BytesIO(raw)) as handle:
        assert handle[handle['obs/cohort'].attrs['categories']].name=='/obs/__categories/cohort'


@pytest.mark.parametrize('failure',['original_bytes','accepted_bytes','reordered','missing','duplicate','collision','missing_field','private_labels'])
def test_composed_failed_snapshot_or_field_validation_releases_all_capacity(tmp_path,failure):
    artifact,old,new=composed_fixture(tmp_path)
    if failure in ('original_bytes','accepted_bytes'):
        path=old.source if failure=='original_bytes' else new.native
        path.write_bytes(path.read_bytes()+b'changed')
    elif failure in ('reordered','missing','duplicate','private_labels'):
        data=ad.read_h5ad(new.native)
        if failure=='reordered': data=data[[3,2,1,0]].copy()
        if failure=='missing': data=data[:3].copy()
        if failure=='duplicate': data.obs_names=['cell-a']*4
        if failure=='private_labels': data.obs['leiden']=pd.Categorical(['/home/operator/private']*4)
        data.write_h5ad(new.native)
        artifact.manifest['sources'][1].update(sha256=sha(new.native),size_bytes=new.native.stat().st_size)
    elif failure=='collision': artifact.manifest['components'][0]['destination']='leiden'
    else: artifact.manifest['components'][0]['field']='result:unavailable'
    preparer=matrix.MatrixDownloadPreparer()
    with pytest.raises(matrix.MatrixDeliveryError) as error:
        preparer.prepare(artifact)
    assert error.value.code=='SCIENTIFIC_DOWNLOAD_UNAVAILABLE'
    assert preparer.active_count==preparer._reserved_bytes==0


def test_composed_and_native_transfers_share_capacity_and_cleanup_on_early_close(tmp_path):
    artifact,old,new=composed_fixture(tmp_path)
    preparer=matrix.MatrixDownloadPreparer()
    one=preparer.prepare(artifact); two=preparer.prepare(artifact)
    try:
        with pytest.raises(matrix.MatrixDeliveryError) as error:
            preparer.prepare(artifact)
        assert error.value.code=='SCIENTIFIC_DOWNLOAD_BUSY'
        assert preparer.active_count==2
        chunks=one.iter_chunks(); next(chunks); chunks.close()
    finally:
        one.close(); two.close()
    assert preparer.active_count==preparer._reserved_bytes==0


def test_explicit_numeric_source_covariate_keeps_dtype_and_missingness(tmp_path):
    artifact,old,new=composed_fixture(tmp_path)
    original=ad.read_h5ad(old.source)
    original.obs['quality']=np.array([1.5,np.nan,0.,2.5],dtype='float32')
    original.write_h5ad(old.source)
    artifact.sha256=sha(old.source); artifact.size_bytes=old.source.stat().st_size
    artifact.manifest['original_source'].update(sha256=artifact.sha256,size_bytes=artifact.size_bytes)
    artifact.manifest['contents']['measurement']['source_sha256']=artifact.sha256
    row=dict(artifact.manifest['components'][1],field='source:quality',destination='selected_quality',
             observation_role='supplied_source_observation; biological validity not assessed')
    with h5py.File(old.source) as handle:
        row['logical_bytes']=delivery._field_logical_bytes(handle,'obs','quality')
    artifact.manifest['components'].append(row)
    artifact.manifest['contents']['observations'].append({k:v for k,v in row.items() if k not in ('slot','source_index','logical_bytes')})
    result,_=prepare(artifact)
    pd.testing.assert_series_equal(result.obs['selected_quality'],original.obs['quality'],check_names=False)
    assert result.obs['selected_quality'].dtype==np.dtype('float32')


def test_repeated_large_representation_reserves_each_destination_not_only_unique_inputs(tmp_path):
    artifact,old,new=composed_fixture(tmp_path)
    values=np.arange(4*20000,dtype='float32').reshape(4,20000)
    with h5py.File(new.native,'r+') as handle:
        from anndata.io import write_elem
        del handle['obsm/X_pca']; write_elem(handle['obsm'],'X_pca',values)
    artifact.manifest['sources'][1].update(sha256=sha(new.native),size_bytes=new.native.stat().st_size)
    row=artifact.manifest['components'][3]
    with h5py.File(new.native) as handle:
        logical=delivery._field_logical_bytes(handle,'obsm','X_pca')
    artifact.manifest['components']=[dict(row,destination=f'X_copy_{i}',logical_bytes=logical) for i in range(12)]
    artifact.manifest['contents']['observations']=[]
    artifact.manifest['contents']['representations']=[{k:v for k,v in r.items() if k not in ('slot','source_index','logical_bytes')}
        for r in artifact.manifest['components']]
    preparer=matrix.MatrixDownloadPreparer()
    with preparer.prepare(artifact) as snapshot:
        assert preparer._reserved_bytes >= artifact.size_bytes + sum(s['size_bytes'] for s in artifact.manifest['sources']) + 12*values.nbytes
        raw=b''.join(snapshot.iter_chunks())
    result=ad.read_h5ad(io.BytesIO(raw))
    for i in range(12): np.testing.assert_array_equal(result.obsm[f'X_copy_{i}'],values)
    assert len(raw)>sum(s['size_bytes'] for s in artifact.manifest['sources'])
    assert preparer.active_count==preparer._reserved_bytes==0


def test_composed_resource_limit_fails_before_snapshot_or_capacity_mutation(tmp_path,monkeypatch):
    artifact,old,new=composed_fixture(tmp_path)
    artifact.manifest['components'][0]['logical_bytes']=matrix.MAX_MATRIX_BYTES
    monkeypatch.setattr(matrix.tempfile,'TemporaryFile',lambda *a,**k:pytest.fail('Capacity limit created a temporary file.'))
    preparer=matrix.MatrixDownloadPreparer()
    with pytest.raises(matrix.MatrixDeliveryError) as error: preparer.prepare(artifact)
    assert error.value.code=='SCIENTIFIC_DOWNLOAD_LIMIT'
    assert preparer.active_count==preparer._reserved_bytes==0


def test_failed_bounded_hdf5_writer_sanitizes_close_error_and_releases_inputs(tmp_path,monkeypatch):
    artifact,old,new=composed_fixture(tmp_path)
    writer=delivery.create_composed_export
    def too_small(original,sources,destination,manifest,*,max_bytes):
        return writer(original,sources,destination,manifest,max_bytes=16)
    monkeypatch.setattr(delivery,'create_composed_export',too_small)
    preparer=matrix.MatrixDownloadPreparer()
    with pytest.raises(matrix.MatrixDeliveryError) as error: preparer.prepare(artifact)
    assert error.value.code=='SCIENTIFIC_DOWNLOAD_UNAVAILABLE'
    assert preparer.active_count==preparer._reserved_bytes==0


def test_exact_accepted_digest_change_cannot_select_an_export_source(app,monkeypatch):
    from dataclasses import asdict
    from test_composed_export_interaction import case
    current=case(app,monkeypatch)
    state=current.facade._application.sessions.load('session')
    old,new=state.revisions
    locator=lambda r:dict(asdict(r.outputs[0]),revision_id=r.revision_id)
    measurement=locator(old); changed=locator(new); changed['accepted_step_sha256']='0'*64
    selection=dict(measurement={'accepted_output':measurement},observations=[dict(
        accepted_output=changed,field='result:leiden',destination='selected_leiden')],representations=[])
    with pytest.raises(ValueError,match='exact accepted output'):
        delivery.composed_analysis(current.facade._application.sessions,'session',selection,turn_id='digest-change')


def test_discovery_uses_accepted_field_names_but_selected_native_methods_require_exact_bytes(app,monkeypatch):
    from dataclasses import asdict
    from test_composed_export_interaction import case
    current=case(app,monkeypatch)
    sessions=current.facade._application.sessions
    state=sessions.load('session'); old,new=state.revisions
    locator=lambda r:dict(asdict(r.outputs[0]),revision_id=r.revision_id)
    target=dict(revision_id=old.revision_id, output_name=old.outputs[0].name,
        accepted_step_sha256=old.outputs[0].accepted_step_sha256)
    before=delivery.export_fields(sessions,'session',target)
    with h5py.File(current.old.native,'r+') as handle:
        handle['uns/agent_milestone6/parameters/umap/min_dist'][()]=.999
        handle['uns/agent_milestone6/embedding/representation'][()]='X_unaccepted'
        handle['obsm'].copy('X_pca','X_unaccepted')
    assert delivery.export_fields(sessions,'session',target)==before
    assert {r['field'] for r in before if r['role'].startswith('accepted_')}=={
        'obsm:X_pca','obsm:X_umap','result:leiden'}
    assert all(set(row)=={'field','role'} for row in before)
    selection=dict(measurement={'accepted_output':locator(old)},observations=[],representations=[dict(
        accepted_output=locator(old),field='obsm:X_umap',destination='X_selected')])
    with pytest.raises(ValueError,match='digest changed'):
        delivery.composed_analysis(sessions,'session',selection,turn_id='changed-method')
    assert sessions.load('session')==state


def test_field_discovery_reads_no_native_methods_or_dataset_payloads(app,monkeypatch):
    from agent.tools.analysis import embedding_analysis
    from test_composed_export_interaction import case
    current=case(app,monkeypatch)
    sessions=current.facade._application.sessions
    state=sessions.load('session')
    protected={p:p.read_bytes() for p in (current.old.source,current.old.native,current.new.native)}
    def no_payload(*args,**kwargs):
        pytest.fail('Offering export field names read or hashed scientific payloads.')
    monkeypatch.setattr(delivery,'_accepted_metadata',no_payload)
    monkeypatch.setattr(embedding_analysis,'scientific_methods',no_payload)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',no_payload)
    for revision in state.revisions:
        output=revision.outputs[0]
        rows=delivery.export_fields(sessions,'session',dict(revision_id=revision.revision_id,
            output_name=output.name,accepted_step_sha256=output.accepted_step_sha256))
        assert {'obsm:X_pca','result:leiden','source:cohort'} <= {row['field'] for row in rows}
        assert all(set(row)=={'field','role'} for row in rows)
    assert sessions.load('session')==state
    assert all(p.read_bytes()==raw for p,raw in protected.items())


def test_field_discovery_preserves_numeric_source_names_and_omits_private_names(app,monkeypatch):
    from anndata.io import write_elem
    from test_composed_export_interaction import case
    current=case(app,monkeypatch)
    sessions=current.facade._application.sessions
    state=sessions.load('session'); revision=state.revisions[0]; output=revision.outputs[0]
    with h5py.File(current.old.source,'r+') as handle:
        obs=handle['obs']; names=list(obs.attrs['column-order'])
        additions=dict(depth=np.array([1.,np.nan,3.,4.]), credentials=np.arange(4),
            operator_path=np.arange(4))
        for name,values in additions.items():
            write_elem(obs,name,values)
        obs.attrs['column-order']=np.asarray(names+list(additions),dtype=object)
    target=dict(revision_id=revision.revision_id,output_name=output.name,
        accepted_step_sha256=output.accepted_step_sha256)
    rows=delivery.export_fields(sessions,'session',target)
    assert 'source:depth' in {row['field'] for row in rows}
    assert not {'source:credentials','source:operator_path'} & {row['field'] for row in rows}
    assert 'accepted_observation' not in {row['role'] for row in rows if row['field']=='source:depth'}
    assert sessions.load('session')==state


@pytest.mark.parametrize('layout',['indirect_observations','excessive_column_names'])
def test_field_discovery_uses_existing_bounded_observation_metadata_owner(app,monkeypatch,layout):
    from test_composed_export_interaction import case
    current=case(app,monkeypatch)
    sessions=current.facade._application.sessions
    state=sessions.load('session'); revision=state.revisions[0]; output=revision.outputs[0]
    with h5py.File(current.old.source,'r+') as handle:
        if layout=='indirect_observations':
            del handle['obs']; handle['obs']=h5py.ExternalLink(str(current.old.native),'/obs')
        else:
            handle['obs'].attrs['column-order']=np.asarray(['c']*4097,dtype='S1')
    with pytest.raises(ValueError,match='Indirect|Excessive'):
        delivery.export_fields(sessions,'session',dict(revision_id=revision.revision_id,
            output_name=output.name,accepted_step_sha256=output.accepted_step_sha256))
    assert sessions.load('session')==state
