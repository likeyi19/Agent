import hashlib
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
import pytest
from agent.schemas.verification_authority import AuthorityError, VerifiedArtifactAuthority
from agent.tools.data import regulatory_matrix_adoption as adoption, regulatory_matrix_contract as c
from agent.tools.data import external_matrix_contract as legacy, scatac_matrix_contract as m
from agent.tools.data import cell_by_ccre_verifier as verifier, scientific_authority as authority
from agent.tools.data.authority_context import VerificationContext, authority_operation


def rehash(case):
    case['source_sha256'] = hashlib.sha256(Path(case['source_path']).read_bytes()).hexdigest()


def test_exact_adoption_and_recovery(case):
    result = adoption.adopt_cell_by_features(**case, execution_identity='a'*64)
    value = m.load_manifest_bytes(Path(result['manifest_path']).read_bytes())
    assert value['contract_version'] == c.CONTRACT
    assert value['artifact_type'] == c.ARTIFACT
    assert value['source_logical_matrix_sha256'] == value['logical_matrix_sha256']
    assert value['species'] == case['species'] and value['assembly'] == case['assembly']
    assert not {'upstream','ordered_selected_sha256','diagnostic','model_readiness'} & value.keys()
    output = ad.read_h5ad(result['matrix_path'])
    source = ad.read_h5ad(case['source_path'])
    assert output.obs_names.tolist() == ['cell-z','cell-a','cell-empty']
    assert output.var_names.tolist() == source.var_names.tolist()
    for name in ('data','indices','indptr'):
        np.testing.assert_array_equal(getattr(output.X, name), getattr(source.X, name))
    assert output.X.dtype == np.int64 and output.X.has_canonical_format
    assert result['zero_row_count'] == 1 and result['total_count'] == 6
    with pytest.raises(ValueError): legacy.validate(value)
    assert adoption.recover_matrix(case, 'a'*64) == result
    with pytest.raises(ValueError): adoption.recover_matrix(case, 'b'*64)
    with pytest.raises(ValueError): adoption.adopt_cell_by_features(**case, execution_identity='a'*64)


@pytest.mark.parametrize('semantics', c.SEMANTICS)
@pytest.mark.parametrize('empty', [False, True])
def test_values_and_empty(case, semantics, empty):
    source = ad.read_h5ad(case['source_path'])
    if empty: source = source[:0].copy()
    elif semantics == 'binary_accessibility': source.X.data[:] = 1
    source.write_h5ad(case['source_path']); rehash(case)
    case['matrix_semantics'] = semantics
    result = adoption.adopt_cell_by_features(**case)
    assert result['matrix_semantics'] == semantics
    assert result['n_cells'] == (0 if empty else 3) and result['n_features'] == 3
    assert result['readiness'] == ('no_source_cells' if empty else 'matrix_available')
    verifier.verify_cell_by_ccre(result['manifest_path'], expected_sha256=result['manifest_sha256'])


@pytest.mark.parametrize('mutation', ['order','partial','extra','duplicate_cell','negative','zero','float',
    'overflow','duplicate_entry','unsorted','bad_index','bad_pointer','species','assembly','sha','binary',
    'reference_identity','species_metadata','coordinates','reference_sha'])
def test_fail_closed(case, mutation):
    with h5py.File(case['source_path'], 'r+') as f:
        if mutation == 'order': f['var/_index'][:] = ['chr1:0-20','chr1:30-40','chr1:10-25']
        elif mutation in ('partial','extra'): f['X'].attrs['shape'] = [3,2 if mutation=='partial' else 4]
        elif mutation == 'duplicate_cell': f['obs/_index'][1] = 'cell-z'
        elif mutation in ('negative','zero','overflow'): f['X/data'][0] = {'negative':-1,'zero':0,'overflow':2**63-1}[mutation]
        elif mutation == 'float':
            v = f['X/data'][:].astype(float); del f['X/data']; f['X'].create_dataset('data',data=v)
        elif mutation == 'duplicate_entry': f['X/indices'][1] = 0
        elif mutation == 'unsorted': f['X/indices'][:2] = [1,0]
        elif mutation == 'bad_index': f['X/indices'][0] = 3
        elif mutation == 'bad_pointer': f['X/indptr'][1] = 10
    if mutation in ('reference_identity','species_metadata','coordinates'):
        a = ad.read_h5ad(case['source_path'])
        if mutation == 'reference_identity': a.uns['reference_identity_sha256'] = '0'*64
        elif mutation == 'species_metadata': a.uns['species'] = 'macaque'
        else:
            a.var['chrom'] = ['chr1']*3; a.var['start'] = [31,0,10]; a.var['end'] = [40,20,25]
        a.write_h5ad(case['source_path'])
    rehash(case)
    if mutation == 'species': case['species'] = {'scientific_name':'Macaca mulatta','taxonomy_id':9544}
    elif mutation == 'assembly': case['assembly'] = 'other'
    elif mutation == 'sha': case['source_sha256'] = '0'*64
    elif mutation == 'reference_sha': case['reference_manifest_sha256'] = '0'*64
    elif mutation == 'binary': case['matrix_semantics'] = 'binary_accessibility'
    with pytest.raises(ValueError): adoption.adopt_cell_by_features(**case)
    out = Path(case['output_dir'])
    assert not out.exists() or not list(out.glob('cell-by-ccre-*'))


def test_rehashed_value_forgery(case):
    result = adoption.adopt_cell_by_features(**case)
    path = Path(result['manifest_path']); payload = Path(result['matrix_path'])
    with h5py.File(payload, 'r+') as f: f['X/data'][:] = [2,1,1,2]
    value = m.load_manifest_bytes(path.read_bytes())
    forged = legacy.logical_identity(3,3,[([0,1],[2,1]),([0,1],[1,2]),([],[])],'fragment_counts')
    value.update(forged); value['source_logical_matrix_sha256'] = forged['logical_matrix_sha256']
    value['matrix'].update(sha256=hashlib.sha256(payload.read_bytes()).hexdigest(), size_bytes=payload.stat().st_size)
    value['identity_sha256'] = c.identity(value); raw = m.canonical(value); path.write_bytes(raw)
    with pytest.raises(ValueError, match='MATRIX_CONSERVATION_MISMATCH'):
        verifier.verify_cell_by_ccre(path, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_source_mutation_after_publication(case):
    result = adoption.adopt_cell_by_features(**case)
    with h5py.File(case['source_path'], 'r+') as f: f['X/data'][0] = 3
    with pytest.raises(ValueError): adoption.verify_public_result(case, result)


def test_owner_proof_issuance_and_reuse(case, monkeypatch):
    context = VerificationContext(); identity = 'a'*64
    context.register_execution('matrix', case['output_dir'], identity)
    calls = []
    original = verifier._verify_external
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(verifier, '_verify_external', counted)
    with authority_operation(context):
        result = adoption.adopt_cell_by_features(**case, execution_identity=identity)
        adoption.verify_public_result(case, result)
        record = authority.issue(context, 'adopt_cell_by_features', case, result, identity)
        assert len(calls) == 1
        assert record.record['schema_version'] == 2
        assert record.record['science_profile'] == c.PROFILE_SHA256
        assert record.record['historical_sources'][0]['sha256'] == case['source_sha256']
        assert case['source_path'] not in [f['path'] for f in record.record['files']]
        assert record.record['producer_qualification'] == {}
        # Historical source identity is distinct from current artifact/resource integrity.
        Path(case['source_path']).unlink()
        adoption.verify_public_result(case, result)
        assert len(calls) == 1
        # Parsed schema-2 bytes alone cannot issue or restore a scientific proof.
        assert VerifiedArtifactAuthority(record.to_dict()) == record
        with pytest.raises(AuthorityError):
            authority.issue(VerificationContext(), 'adopt_cell_by_features', case, result, identity)
        with h5py.File(result['matrix_path'], 'r+') as f: f['X/data'][0] = 7
        with pytest.raises(ValueError): adoption.verify_public_result(case, result)
        assert len(calls) == 1


def test_resource_mutation_invalidates_reuse(case, resources):
    context = VerificationContext()
    with authority_operation(context):
        result = adoption.adopt_cell_by_features(**case)
        resources['feature_bed_path'].write_text('chr1\t30\t41\nchr1\t0\t20\nchr1\t10\t25\n')
        with pytest.raises(ValueError): adoption.verify_public_result(case, result)


def test_no_registry_expansion():
    from agent.orchestration.registry import build_default_tool_registry, UnknownToolError
    registry = build_default_tool_registry()
    with pytest.raises(UnknownToolError): registry.get('adopt_cell_by_features')


def test_wrong_execution_cannot_issue(case):
    context = VerificationContext()
    context.register_execution('matrix', case['output_dir'], 'a'*64)
    with authority_operation(context):
        result = adoption.adopt_cell_by_features(**case, execution_identity='a'*64)
        with pytest.raises(AuthorityError):
            authority.issue(context, 'adopt_cell_by_features', case, result, 'b'*64)


def test_receipt_mutation_rejected(case):
    result = adoption.adopt_cell_by_features(**case, execution_identity='a'*64)
    receipt = Path(result['manifest_path']).parent.parent/'receipt.json'
    import json
    value = json.loads(receipt.read_bytes()); value['matrix_profile_sha256'] = '0'*64
    receipt.write_bytes(m.canonical(value))
    with pytest.raises(ValueError): adoption.recover_matrix(case, 'a'*64)


def test_cancellation_before_work(case):
    from agent.tools._cancellation import cancellation_scope, ToolWorkCancelled
    with cancellation_scope(lambda: True), pytest.raises(ToolWorkCancelled):
        adoption.adopt_cell_by_features(**case)
    assert not Path(case['output_dir']).exists()


def test_reference_identity_required_not_just_contig_names(case, resources, tmp_path):
    from agent.tools.data import regulatory_feature_reference as r
    source = ad.read_h5ad(case['source_path'])
    _, first, _ = r.load_regulatory_feature_reference(case['reference_manifest_path'])
    source.uns['reference_identity_sha256'] = first.reference_identity_sha256
    source.write_h5ad(case['source_path']); rehash(case)
    second = r.build_regulatory_feature_reference(**(resources | {'target_assembly':'different-release'}))
    ptr = r.publish_regulatory_feature_reference(second, tmp_path/'second.json')
    case.update(reference_manifest_path=ptr['manifest_path'], reference_manifest_sha256=ptr['manifest_sha256'],
                assembly='different-release')
    with pytest.raises(ValueError, match='MATRIX_REFERENCE_MISMATCH'):
        adoption.adopt_cell_by_features(**case)
