from pathlib import Path
import anndata as ad
import numpy as np
import hashlib
import json
import h5py
import pytest

from agent.tools.data import fragment_feature_matrix as owner, scientific_authority
from agent.tools.data import cell_by_ccre_verifier as verifier
from agent.tools.data.authority_context import VerificationContext, authority_operation
from agent.tools.data import scatac_matrix_contract as m, fragment_feature_matrix_contract as contract
from agent.tools.data import regulatory_feature_reference as reference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_bounded_acceptance(factory, monkeypatch):
    args, _, _ = factory()
    calls = []; original = verifier.reconstruct
    def counted(*a, **kw):
        calls.append(1)
        return original(*a, **kw)
    monkeypatch.setattr(verifier, 'reconstruct', counted)
    context = VerificationContext(); execution = 'a'*64
    context.register_execution('matrix', args['output_dir'], execution)
    with authority_operation(context):
        result = owner.build_cell_by_features(**args, execution_identity=execution)
        output = ad.read_h5ad(result['matrix_path'])
        np.testing.assert_array_equal(output.X.toarray(), [[1,1,1,0], [0,2,2,0], [0,0,0,0]])
        assert output.obs['barcode_identifier'].tolist() == ['B','A','Z']
        assert output.var_names.tolist() == ['chr1:30-40','chr1:0-20','chr1:10-25','chr1:90-100']
        assert output.X.dtype == np.int64 and output.X.has_canonical_format
        assert (output.X.data > 0).all()
        assert result['total_count'] == 7 and result['zero_row_count'] == 1
        owner.verify_public_result(args, result)
        authority = scientific_authority.issue(context, 'build_cell_by_features', args, result, execution)
        assert authority.record['schema_version'] == 2
        assert set(authority.record['upstream']) == {'fragments', 'cells'}
        assert owner.recover_matrix(args, execution) == result
        assert len(calls) == 1
    # Standalone recovery performs one independent owner reconstruction, no production.
    from agent.tools.data import _cell_by_ccre_production as production
    def forbidden(*a, **kw):
        pytest.fail('Recovery reran matrix production')
    monkeypatch.setattr(production, 'construct_counts', forbidden)
    assert owner.recover_matrix(args, execution) == result
    assert len(calls) == 2
    assert Path(result['manifest_path']).is_file()
    evidence = dict(result=result, authority=authority.to_dict(),
        expected=[[1,1,1,0],[0,2,2,0],[0,0,0,0]], produced=output.X.toarray().tolist(),
        owner_reconstructions_before_standalone_recovery=1, owner_reconstructions_after=2,
        publication='verified_atomic', context_recovery='proof_reused', standalone_recovery='independently_verified')
    (Path(args['output_dir']).parent/'acceptance.json').write_bytes(m.canonical(evidence))


@pytest.mark.parametrize('cells', [[], [('lib','Z')], [('lib','A'),('lib','B')]])
def test_empty_zero_and_order(factory, cells):
    args, _, _ = factory(cells=cells)
    result = owner.build_cell_by_features(**args)
    output = ad.read_h5ad(result['matrix_path'])
    expected = {'A':[0,2,2,0], 'B':[1,1,1,0], 'Z':[0,0,0,0]}
    np.testing.assert_array_equal(output.X.toarray(), np.array([expected[b] for _,b in cells], dtype=np.int64).reshape(len(cells),4))
    assert output.shape == (len(cells),4)


@pytest.mark.parametrize('cells,error', [([('wrong','A')],'MATRIX_NAMESPACE_MISMATCH'),
                                        ([('lib','absent')],'MATRIX_SELECTED_CELL_ABSENT')])
def test_absent_rejected(factory, cells, error):
    args, _, _ = factory(cells=cells)
    with pytest.raises(ValueError, match=error): owner.build_cell_by_features(**args)
    assert not list(Path(args['output_dir']).glob('cell-by-ccre-*'))


def test_support_and_strand_do_not_weight_counts(factory):
    results = [owner.build_cell_by_features(**factory(support=s, strand=strand)[0])
               for s, strand in [(1,False), (2**64-1,True)]]
    assert results[0]['logical_matrix_sha256'] == results[1]['logical_matrix_sha256']
    assert results[0]['total_count'] == results[1]['total_count'] == 7


@pytest.mark.parametrize('change', ['species', 'assembly', 'sequence', 'contig', 'length'])
def test_exact_reference_compatibility(factory, change):
    args, resources, _ = factory()
    root = Path(args['output_dir']).parent
    other = dict(resources)
    if change == 'species': other['species'] = {'scientific_name':'Macaca mulatta', 'taxonomy_id':9544}
    if change == 'assembly': other['target_assembly'] = 'different-assembly'
    if change in ('sequence','contig','length'):
        fa = root/'other.fa'; fai = root/'other.fa.fai'; bed = root/'other.bed'
        name = 'chr2' if change == 'contig' else 'chr1'; length = 101 if change == 'length' else 100
        fa.write_text('>'+name+'\n'+('C' if change == 'sequence' else 'A')*length+'\n')
        fai.write_text(f'{name}\t{length}\t6\t{length}\t{length+1}\n')
        bed.write_text(f'{name}\t0\t20\n')
        other.update(fasta_path=fa, fai_path=fai, feature_bed_path=bed)
    ptr = reference.publish_regulatory_feature_reference(reference.build_regulatory_feature_reference(**other), root/'other.json')
    args.update(reference_manifest_path=ptr['manifest_path'], reference_manifest_sha256=ptr['manifest_sha256'])
    with pytest.raises(ValueError, match='MATRIX_REFERENCE_MISMATCH'): owner.build_cell_by_features(**args)


def test_alternative_vocabulary_same_exact_genome(factory):
    args, resources, _ = factory()
    root = Path(args['output_dir']).parent
    bed = root/'different-features.bed'; bed.write_text('chr1\t50\t60\nchr1\t10\t15\n')
    ptr = reference.publish_regulatory_feature_reference(reference.build_regulatory_feature_reference(
        **(resources | {'feature_bed_path':bed})), root/'different-reference.json')
    args.update(reference_manifest_path=ptr['manifest_path'], reference_manifest_sha256=ptr['manifest_sha256'])
    result = owner.build_cell_by_features(**args)
    np.testing.assert_array_equal(ad.read_h5ad(result['matrix_path']).X.toarray(), [[1,1],[0,1],[0,0]])


@pytest.mark.parametrize('mutation', ['source','bgzf','index','fragment_manifest','cell_table','cell_manifest',
                                     'reference_manifest','fasta','fai','bed','matrix','receipt'])
def test_mutation_invalidates_reuse(factory, mutation):
    args, resources, source_args = factory()
    context = VerificationContext()
    with authority_operation(context):
        result = owner.build_cell_by_features(**args)
        fragments = json.loads(Path(args['fragments_manifest_path']).read_bytes())
        fragment_root = Path(args['fragments_manifest_path']).parent
        paths = dict(source=source_args['source_path'],
            bgzf=fragment_root/fragments['libraries'][0]['bgzf']['path'],
            index=fragment_root/fragments['libraries'][0]['tabix']['path'],
            fragment_manifest=args['fragments_manifest_path'], cell_manifest=args['explicit_cells_manifest_path'],
            cell_table=Path(args['explicit_cells_manifest_path']).parent/'cells.tsv.gz',
            reference_manifest=args['reference_manifest_path'], fasta=resources['fasta_path'],
            fai=resources['fai_path'], bed=resources['feature_bed_path'], matrix=result['matrix_path'],
            receipt=Path(result['manifest_path']).parent.parent/'receipt.json')
        path = Path(paths[mutation]); path.write_bytes(path.read_bytes()+b'changed')
        with pytest.raises(ValueError): owner.recover_matrix(args, None)


@pytest.mark.parametrize('mutation', ['value','marginals','rows','features','negative','zero','unsorted','duplicate',
                                     'float','fragments','cells','reference','profile','external'])
def test_rehashed_forgery_rejected(factory, mutation):
    args, _, _ = factory(extra_rows=[(11,19,'B')] if mutation == 'marginals' else ())
    result = owner.build_cell_by_features(**args)
    path = Path(result['manifest_path']); payload = Path(result['matrix_path'])
    value = json.loads(path.read_bytes())
    with h5py.File(payload,'r+') as f:
        if mutation == 'value': f['X/data'][0] = 2
        elif mutation == 'marginals':
            # Equal row totals, column totals, nnz and overall total; wrong exact values.
            f['X/data'][:] = [1,3,1,1,3]
        elif mutation == 'rows': f['obs/barcode_identifier'][:2] = ['A','B']
        elif mutation == 'features': f['var/_index'][:2] = ['chr1:0-20','chr1:30-40']
        elif mutation == 'negative': f['X/data'][0] = -1
        elif mutation == 'zero': f['X/data'][0] = 0
        elif mutation == 'unsorted': f['X/indices'][:2] = [1,0]
        elif mutation == 'duplicate': f['X/indices'][1] = 0
        elif mutation == 'float':
            data = f['X/data'][:].astype(float); del f['X/data']; f['X'].create_dataset('data',data=data)
        if mutation == 'value':
            summary = m.logical_matrix_identity(3,4,[([0,1,2],[2,1,1]),([1,2],[2,2]),([],[])])
            value.update(summary); f['uns/logical_matrix_sha256'][()] = summary['logical_matrix_sha256']
        if mutation == 'marginals':
            summary = m.logical_matrix_identity(3,4,[([0,1,2],[1,3,1]),([1,2],[1,3]),([],[])])
            assert all(summary[k] == value[k] for k in ('nnz','total_count','zero_row_count'))
            value.update(summary); f['uns/logical_matrix_sha256'][()] = summary['logical_matrix_sha256']
    if mutation in ('fragments','cells','reference'): value['upstream'][mutation]['identity_sha256'] = '0'*64
    if mutation == 'profile': value['profile_sha256'] = '0'*64
    if mutation == 'external': value['contract_version'] = 'scatac-cell-by-features.external.v1'
    value['matrix'].update(sha256=sha(payload),size_bytes=payload.stat().st_size)
    value['identity_sha256'] = contract.manifest_identity(value); path.write_bytes(m.canonical(value))
    if mutation in ('value','marginals'): m.load_manifest_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='MATRIX_SCIENCE_MISMATCH' if mutation in ('value','marginals') else None):
        verifier.verify_cell_by_ccre(path, expected_sha256=sha(path), bedtools_path='/usr/bin/bedtools')


def test_no_unowned_authority_and_no_registry_expansion(factory):
    args, _, _ = factory(); result = owner.build_cell_by_features(**args, execution_identity='b'*64)
    with pytest.raises(ValueError):
        scientific_authority.issue(VerificationContext(), 'build_cell_by_features', args, result, 'b'*64)
    from agent.orchestration.registry import build_default_tool_registry, UnknownToolError
    registry = build_default_tool_registry()
    assert len(registry.names()) == 23
    with pytest.raises(UnknownToolError): registry.get('build_cell_by_features')
    assert registry.get('build_scATAC_cell_by_features').name == 'build_scATAC_cell_by_features'


def test_cancellation_and_conflicts(factory):
    args, _, _ = factory()
    from agent.tools._cancellation import cancellation_scope, ToolWorkCancelled
    with cancellation_scope(lambda: True), pytest.raises(ToolWorkCancelled): owner.build_cell_by_features(**args)
    assert not Path(args['output_dir']).exists()
    owner.build_cell_by_features(**args, execution_identity='c'*64)
    with pytest.raises(ValueError): owner.build_cell_by_features(**args, execution_identity='c'*64)
    with pytest.raises(ValueError): owner.recover_matrix(args, 'd'*64)
