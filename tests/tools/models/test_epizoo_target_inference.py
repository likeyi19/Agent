"""Target eligibility: small simulated counts establish software behavior only."""
from copy import deepcopy
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from agent.tools.models import epizoo_target_inference as target


@pytest.fixture
def case():
    data = ad.AnnData(sparse.csr_matrix([[0., 1., 3.], [2., 0., 0.]]),
        obs=pd.DataFrame(index=['b', 'a']), var=pd.DataFrame(index=['chr1:4-5', 'chr1:2-3', 'chr1:0-1']))
    manifest = dict(reference=dict(species=dict(scientific_name='Test species', taxonomy_id=123),
        target_assembly='v1'), arguments=dict(profile=dict(matrix_semantics='fragment_counts')))
    args = dict(species='Test species', taxonomy_id=123, assembly='v1', matrix_semantics='fragment_counts')
    return data, manifest, list(reversed(data.var_names)), args


def test_lossless_permutation_preserves_rows_and_values(case):
    data, manifest, vocabulary, args = case
    original = data.copy()
    aligned, changed = target.qualify_input(data, manifest, vocabulary, **args)
    assert changed and aligned.obs_names.tolist() == ['b', 'a']
    assert aligned.var_names.tolist() == vocabulary
    assert (aligned.X != original.X[:, ::-1]).nnz == 0
    assert (data.X != original.X).nnz == 0
    assert data.var_names.tolist() == original.var_names.tolist()
    second, changed = target.qualify_input(aligned, manifest, vocabulary, **args)
    assert not changed and (second.X != aligned.X).nnz == 0


@pytest.mark.parametrize('field,value', [('species', 'Other species'), ('taxonomy_id', 124),
    ('taxonomy_id', True), ('assembly', 'v2'), ('matrix_semantics', 'insertion_counts'),
    ('matrix_semantics', 'binary_accessibility')])
def test_scientific_mismatch(case, field, value):
    data, manifest, vocabulary, args = case
    args[field] = value
    with pytest.raises(ValueError, match='mismatch'):
        target.qualify_input(data, manifest, vocabulary, **args)


@pytest.mark.parametrize('bad', [-1., .5, float('nan'), float('inf'), float(2**53)])
def test_invalid_measurements(case, bad):
    data, manifest, vocabulary, args = case
    data.X.data[0] = bad
    with pytest.raises(ValueError, match='measurements'):
        target.qualify_input(data, manifest, vocabulary, **args)


@pytest.mark.parametrize('kind', ['missing', 'different', 'duplicate', 'duplicate_cells', 'dense', 'metadata'])
def test_incompatible_inputs(case, kind):
    data, manifest, vocabulary, args = case
    if kind == 'missing': data = data[:, :2].copy()
    if kind == 'different': data.var_names = ['chr1:4-6', *data.var_names[1:]]
    if kind == 'duplicate': data.var_names = ['x', 'x', 'y']
    if kind == 'duplicate_cells': data.obs_names = ['a', 'a']
    if kind == 'dense': data.X = np.zeros(data.shape)
    if kind == 'metadata': data.uns['assembly'] = 'wrong'
    with pytest.raises(ValueError):
        target.qualify_input(data, manifest, vocabulary, **args)


def test_binary_is_distinct(case):
    data, manifest, vocabulary, args = case
    manifest['arguments']['profile']['matrix_semantics'] = args['matrix_semantics'] = 'binary_accessibility'
    with pytest.raises(ValueError, match='measurements'):
        target.qualify_input(data, manifest, vocabulary, **args)
    data.X.data[:] = 1
    aligned, _ = target.qualify_input(data, manifest, vocabulary, **args)
    assert aligned.X.nnz == data.X.nnz


def test_model_pair_and_legacy_conflict_fail_before_loading(tmp_path):
    from agent.tools.analysis.epizoo_embedding import epizoo_embed_cells
    for options in [dict(target_model_manifest_path='/missing'),
                    dict(target_model_manifest_path='/missing', target_model_manifest_sha256='a'*64,
                         checkpoint_path='/another')]:
        with pytest.raises(ValueError, match='complete target model'):
            epizoo_embed_cells('/missing.h5ad', tmp_path, species='Test species', **options)


def test_target_resource_inputset_does_not_select_legacy_default():
    from agent.application.local_resources import select_epizoo_resource, ResourceAdmissionError
    values = dict(species='Test species', target_model_manifest_path='/private/model.json',
                  target_model_manifest_sha256='a'*64)
    assert select_epizoo_resource(values, None, ()) == (values, None)
    with pytest.raises(ResourceAdmissionError):
        select_epizoo_resource(values, 'legacy', ())
    del values['target_model_manifest_sha256']
    with pytest.raises(ResourceAdmissionError):
        select_epizoo_resource(values, None, ())


def test_registry_target_edge_and_legacy_port():
    from agent.orchestration.registry import build_default_tool_registry
    registry = build_default_tool_registry()
    spec = registry.get('epizoo_embed_cells')
    ports = {p.name: p for p in spec.semantic_planning.consumer_ports}
    assert ports['target_model'].accepted_upstream_types == ('epizoo_target_model.v1',)
    assert ports['dataset'].accepted_upstream_types == ('raw_scatac_dataset.v1', 'scatac_matrix_h5ad.v1')
    spec.required_arguments['species'].validate('species', {'scientific_name':'Test species','taxonomy_id':123})


@pytest.mark.parametrize('mutation', [None, 'assembly', 'semantics', 'features', 'reordered', 'cells', 'embedding'])
def test_persisted_inference_checks_exact_consumed_identities(case, tmp_path, monkeypatch, mutation):
    data, manifest, vocabulary, args = case
    from agent.tools.models.epizoo_adaptation import resources
    manifest.update(files={'checkpoint.pth':{'sha256':'a'*64}},
        preprocessing={'ordered_feature_sha256':'b'*64}, execution={'completed_step':10})
    manifest['arguments']['profile']['purpose']='qualification'
    path=tmp_path/'manifest.json';path.write_text('{}')
    (tmp_path/'features.txt').write_text('\n'.join(vocabulary)+'\n')
    source=tmp_path/'input.h5ad';data.write_h5ad(source)
    embedding=tmp_path/'embedding.npy';np.save(embedding,np.ones((2,512),dtype=np.float32))
    cells=tmp_path/'cells.txt';cells.write_text('b\na\n')
    runtime=dict(cpu_threads=4,interop_threads=2)
    monkeypatch.setattr(target,'bind_model',lambda *a:'authority')
    monkeypatch.setattr(target.publication,'load_manifest',lambda *a:manifest)
    monkeypatch.setattr(resources,'runtime',lambda:runtime)
    arguments=dict(species=manifest['reference']['species'],assembly='v1',matrix_semantics='fragment_counts',
        target_model_manifest_path=str(path),target_model_manifest_sha256='c'*64)
    p=dict(method=target.METHOD,manifest_path=str(path),manifest_sha256='c'*64,
        checkpoint_sha256='a'*64,species=manifest['reference']['species'],assembly='v1',
        matrix_semantics='fragment_counts',ordered_feature_sha256='b'*64,training_purpose='qualification',
        training_steps=10,settings=dict(target.SETTINGS,device='cuda:0'),accepted_model_authority='authority',
        features_reordered=True,runtime=runtime)
    result=dict(target_model_provenance=p,input_path=str(source),embedding_path=str(embedding),
        cell_ids_path=str(cells),checkpoint_path=str(tmp_path/'checkpoint.pth'),n_cells=2)
    if mutation=='assembly':arguments['assembly']=p['assembly']='wrong'
    if mutation=='semantics':arguments['matrix_semantics']=p['matrix_semantics']='binary_accessibility'
    if mutation=='features':
        data.var_names=['wrong',*data.var_names[1:]];data.write_h5ad(source)
    if mutation=='reordered':p['features_reordered']=False
    if mutation=='cells':cells.write_text('a\nb\n')
    if mutation=='embedding':np.save(embedding,np.ones((2,511),dtype=np.float32))
    for key,field in [('input_path','input_sha256'),('embedding_path','embedding_sha256'),('cell_ids_path','cell_ids_sha256')]:
        p[field]=target.c.file_record(result[key])['sha256']
    if mutation is None:target.verify_result(arguments,result)
    else:
        with pytest.raises(ValueError):target.verify_result(arguments,result)
