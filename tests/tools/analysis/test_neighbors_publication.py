"""Scientific owner qualification, separate from generic historical admission."""

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools.analysis import embedding_analysis as science
from agent.tools.analysis import neighbors_publication as owner


@pytest.fixture
def publication(tmp_path):
    embeddings = tmp_path / 'embeddings.npy'
    cells = tmp_path / 'cells.txt'
    np.save(embeddings, np.random.default_rng(7).normal(size=(20, 512)).astype(np.float32))
    cells.write_text(''.join(f'cell-{n}\n' for n in range(20)))
    return science.build_cell_neighbors(embeddings, cells, tmp_path / 'output', n_neighbors=5)


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _modify_artifact(result, change):
    path = Path(result['analysis_path'])
    artifact = ad.read_h5ad(path)
    change(artifact)
    artifact.write_h5ad(path)
    manifest_path = Path(result['manifest_path'])
    manifest, _ = owner.load_manifest(manifest_path)
    record = owner.file_closure([path])[0]
    manifest['analysis'] = dict(record, path=path.name)
    return _write(manifest_path, manifest)


def test_real_publication_binds_exact_payload_sources_and_owner_profile(publication):
    manifest, sha = owner.load_manifest(publication['manifest_path'], publication['manifest_sha256'])
    assert manifest['execution_identity'] is None
    assert manifest['analysis']['sha256'] == hashlib.sha256(Path(publication['analysis_path']).read_bytes()).hexdigest()
    assert manifest['profile_sha256'] == owner.PROFILE_SHA256
    for record in manifest['sources'].values():
        assert record['sha256'] == hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()
    assert owner.verify_neighbors_publication(publication['manifest_path'], expected_sha256=sha) == publication
    assert owner._summary(manifest, publication['manifest_path'], sha) == publication


def test_publication_summary_is_json_lightweight(publication):
    assert len(json.dumps(publication, allow_nan=False)) < 5000
    assert not any(isinstance(value, np.ndarray) for value in publication.values())


def test_registered_result_must_match_owner_publication(publication):
    manifest, _ = owner.load_manifest(publication['manifest_path'])
    assert owner.verify_result(manifest['arguments'], publication) == publication
    forged = dict(publication, n_cells=500)
    with pytest.raises(AuthorityError, match='result differs'):
        owner.verify_result(manifest['arguments'], forged)
    with pytest.raises(AuthorityError, match='declared execution'):
        owner.verify_result(dict(manifest['arguments'], n_neighbors=6), publication)
    with pytest.raises(AuthorityError, match='Incomplete'):
        owner.verify_result(manifest['arguments'], {name: value for name, value in publication.items()
                                                    if name != 'manifest_sha256'})


def test_accepted_description_keeps_historical_sources_when_replaced_by_symlinks(publication, tmp_path):
    from agent.tools.data.authority_context import VerificationContext
    path = Path(publication['embedding_path'])
    old = path.with_name('retained-original.npy')
    path.rename(old)
    replacement = tmp_path / 'different.npy'
    np.save(replacement, np.zeros((20, 512), np.float32))
    path.symlink_to(replacement)
    description = owner.describe(publication['manifest_path'], publication['manifest_sha256'], VerificationContext())
    assert str(path) in {source['path'] for source in description['historical_sources']}
    with pytest.raises(ValueError, match='SOURCE_CHANGED'):
        owner.verify_neighbors_publication(publication['manifest_path'])


def test_initial_source_alias_is_recorded_without_becoming_historical_freshness(tmp_path):
    embedding = tmp_path / 'actual-embeddings.npy'
    cells = tmp_path / 'actual-cells.txt'
    np.save(embedding, np.random.default_rng(11).normal(size=(20, 512)).astype(np.float32))
    cells.write_text(''.join(f'cell-{n}\n' for n in range(20)))
    alias = tmp_path / 'embedding-alias.npy'
    alias.symlink_to(embedding)
    result = science.build_cell_neighbors(alias, cells, tmp_path / 'output', n_neighbors=5)
    manifest, _ = owner.load_manifest(result['manifest_path'])
    assert manifest['arguments']['embedding_path'] == str(alias)
    assert manifest['sources']['embedding']['path'] == str(embedding)
    assert owner.verify_result(manifest['arguments'], result) == result


def test_standalone_publication_cannot_gain_a_different_accepted_execution(publication):
    from agent.tools.data.authority_context import VerificationContext
    manifest, _ = owner.load_manifest(publication['manifest_path'])
    with pytest.raises(AuthorityError, match='exact producing execution'):
        owner.publication_record(VerificationContext(), 'build_cell_neighbors', manifest['arguments'],
                                 publication, digest('different-execution'), {})


@pytest.mark.parametrize('source', ('embedding_path', 'cell_ids_path'))
def test_initial_owner_verification_checks_current_source_bytes(publication, source):
    path = Path(publication[source])
    with path.open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(AuthorityError, match='source bytes'):
        owner.verify_neighbors_publication(publication['manifest_path'])


def test_modified_payload_fails_before_scientific_reconstruction(publication):
    path = Path(publication['analysis_path'])
    with path.open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(AuthorityError, match='artifact bytes'):
        owner.verify_neighbors_publication(publication['manifest_path'])


def test_pin_alone_does_not_qualify_changed_representation(publication):
    sha = _modify_artifact(publication, lambda artifact: artifact.obsm['X_epizoo'].__setitem__((0, 0), 400))
    with pytest.raises(AuthorityError, match='aligned source'):
        owner.verify_neighbors_publication(publication['manifest_path'], expected_sha256=sha)


def test_pin_alone_does_not_qualify_changed_order(publication):
    def reverse(artifact):
        artifact.obs_names = list(reversed(artifact.obs_names))
        artifact.uns[science.PROVENANCE_KEY]['cell_order_sha256'] = science._cell_order_digest(tuple(artifact.obs_names))
    sha = _modify_artifact(publication, reverse)
    with pytest.raises(AuthorityError, match='ordered cell identity'):
        owner.verify_neighbors_publication(publication['manifest_path'], expected_sha256=sha)


def test_pin_alone_does_not_qualify_invalid_scientific_provenance(publication):
    def forge(artifact):
        artifact.uns[science.PROVENANCE_KEY]['parameters']['neighbors']['metric'] = 'cosine'
    sha = _modify_artifact(publication, forge)
    with pytest.raises(AuthorityError, match='provenance or parameters'):
        owner.verify_neighbors_publication(publication['manifest_path'], expected_sha256=sha)


def test_scanpy_parameters_must_match_declared_production(publication):
    def forge(artifact):
        artifact.uns['neighbors']['params']['n_neighbors'] = 8
    sha = _modify_artifact(publication, forge)
    with pytest.raises(AuthorityError, match='graph parameters'):
        owner.verify_neighbors_publication(publication['manifest_path'], expected_sha256=sha)


def test_initial_software_provenance_is_required(publication):
    def remove(artifact):
        del artifact.uns[science.PROVENANCE_KEY]['software_versions']
    sha = _modify_artifact(publication, remove)
    with pytest.raises(AuthorityError, match='software provenance'):
        owner.verify_neighbors_publication(publication['manifest_path'], expected_sha256=sha)


@pytest.mark.parametrize('change', ('membership', 'extra', 'profile', 'source'))
def test_publication_metadata_contract_is_closed(publication, change):
    path = Path(publication['manifest_path'])
    manifest, _ = owner.load_manifest(path)
    if change == 'membership':
        manifest['analysis']['path'] = '../another.h5ad'
    elif change == 'extra':
        manifest['undeclared_fact'] = True
    elif change == 'profile':
        manifest['profile_sha256'] = digest('another-science')
    else:
        manifest['sources']['embedding']['path'] = manifest['sources']['cell_ids']['path']
    sha = _write(path, manifest)
    with pytest.raises(AuthorityError):
        owner.load_manifest(path, sha)


def test_source_change_during_real_production_has_no_accepted_publication(tmp_path, monkeypatch):
    import scanpy as sc
    embeddings = tmp_path / 'embeddings.npy'
    cells = tmp_path / 'cells.txt'
    np.save(embeddings, np.random.default_rng(9).normal(size=(20, 512)).astype(np.float32))
    cells.write_text(''.join(f'cell-{n}\n' for n in range(20)))
    original = sc.pp.neighbors
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        cells.write_text(''.join(f'different-{n}\n' for n in range(20)))
        return result
    monkeypatch.setattr(sc.pp, 'neighbors', mutate)
    with pytest.raises(ValueError, match='SOURCE_CHANGED'):
        science.build_cell_neighbors(embeddings, cells, tmp_path / 'output', n_neighbors=5)
    assert not list((tmp_path / 'output').glob('*.manifest.json'))


def test_source_mutation_during_exact_alignment_is_rejected(publication, monkeypatch):
    original = science._load_embeddings
    def mutate(path):
        result = original(path)
        with Path(publication['cell_ids_path']).open('ab') as handle:
            handle.write(b'extra\n')
        return result
    monkeypatch.setattr(science, '_load_embeddings', mutate)
    with pytest.raises(ValueError, match='SOURCE_CHANGED'):
        owner.verify_neighbors_publication(publication['manifest_path'])


def test_payload_mutation_during_owner_review_is_rejected(publication, monkeypatch):
    original = science._read_analysis
    def mutate(path):
        artifact = original(path)
        with path.open('ab') as handle:
            handle.write(b'changed-during-review')
        return artifact
    monkeypatch.setattr(science, '_read_analysis', mutate)
    with pytest.raises(ValueError, match='SOURCE_CHANGED'):
        owner.verify_neighbors_publication(publication['manifest_path'])


def test_uppercase_npy_retains_exact_source_in_current_publication(tmp_path):
    embedding = tmp_path / 'embeddings.NPY'
    cells = tmp_path / 'cells.txt'
    values = np.random.default_rng(17).normal(size=(20, 512)).astype(np.float32)
    with embedding.open('wb') as stream:
        np.save(stream, values)
    cells.write_text(''.join(f'cell-{i}\n' for i in range(20)))
    result = science.build_cell_neighbors(embedding, cells, tmp_path / 'output', n_neighbors=5)
    manifest, _ = owner.load_manifest(result['manifest_path'])
    assert manifest['contract_version'] == owner.CONTRACT
    assert manifest['profile_sha256'] == owner.PROFILE_SHA256
    np.testing.assert_array_equal(ad.read_h5ad(result['analysis_path']).obsm['X_epizoo'], values)
    assert owner.verify_result(manifest['arguments'], result) == result


def test_legacy_v1_reader_preserves_format_identity_and_safe_export(publication):
    """Simulated historical layout; real persisted examples are checked externally."""
    from io import BytesIO
    from agent.tools.analysis import neighbors_delivery
    path = Path(publication['analysis_path'])
    data = ad.read_h5ad(path)
    del data.uns[science.PROVENANCE_KEY]['embedding']
    data.write_h5ad(path)
    manifest_path = Path(publication['manifest_path'])
    manifest, _ = owner.load_manifest(manifest_path)
    manifest.pop('embedding')
    manifest.update(contract_version=owner.LEGACY_CONTRACT,
                    profile_sha256=owner.PROFILES[owner.LEGACY_CONTRACT])
    manifest['analysis'] = dict(owner.file_closure([path])[0], path=path.name)
    sha = _write(manifest_path, manifest)
    before = {p: p.read_bytes() for p in (path, manifest_path)}
    result = owner.verify_neighbors_publication(manifest_path, expected_sha256=sha)
    assert result['contract_version'] == 'cell-neighbors-publication.v1'
    args = manifest['arguments']
    _, _, _, token = owner._publication(args, None, sources=manifest['sources'],
                                        contract_version=owner.LEGACY_CONTRACT)
    assert token == digest(dict(arguments=args, execution_identity=None,
        contract='cell-neighbors-publication.v1', profile=manifest['profile_sha256']))
    output = BytesIO()
    with path.open('rb') as stream:
        neighbors_delivery.create_neighbors_export(stream, output, manifest,
            source_sha256=manifest['analysis']['sha256'], source_size_bytes=path.stat().st_size,
            manifest_sha256=sha, max_bytes=path.stat().st_size + 65_536)
    exported = ad.read_h5ad(output)
    assert exported.uns['agent_scientific_export']['source_contract'] == owner.LEGACY_CONTRACT
    np.testing.assert_array_equal(exported.obsm['X_epizoo'], data.obsm['X_epizoo'])
    assert all(p.read_bytes() == raw for p, raw in before.items())
