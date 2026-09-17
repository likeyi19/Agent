import hashlib
import pytest
from agent.tools.data import regulatory_feature_reference as r, scatac_reference as legacy


def test_identity_and_order(resources, tmp_path):
    bundle = r.build_regulatory_feature_reference(**resources)
    assert bundle == r.build_regulatory_feature_reference(**resources)
    assert bundle.features.feature_count == 3
    expected = b'chr1:30-40\nchr1:0-20\nchr1:10-25\n'
    assert bundle.features.ordered_feature_sha256 == hashlib.sha256(expected).hexdigest()
    assert bundle.features.category == 'peak_set'
    assert bundle.features.bed.provenance.source == 'explicit test peak set'
    assert bundle.genome.fasta.sha256 == hashlib.sha256(resources['fasta_path'].read_bytes()).hexdigest()
    assert bundle.genome.fai.sha256 == hashlib.sha256(resources['fai_path'].read_bytes()).hexdigest()
    p = r.publish_regulatory_feature_reference(bundle, tmp_path/'reference.json')
    assert r.load_regulatory_feature_reference(p['manifest_path'], expected_sha256=p['manifest_sha256'])[1] == bundle
    assert r.reinspect_regulatory_feature_reference(bundle) == bundle
    with pytest.raises(ValueError): legacy.load_scatac_reference_bundle(p['manifest_path'])
    with pytest.raises(ValueError): r.publish_regulatory_feature_reference(bundle, tmp_path/'reference.json')
    with pytest.raises(ValueError): r.publish_regulatory_feature_reference(bundle, resources['fasta_path'])


@pytest.mark.parametrize('field,value', [
    ('species', {'scientific_name':'Macaca mulatta','taxonomy_id':9544}),
    ('species', {'scientific_name':'Macaca fascicularis','taxonomy_id':9544}),
    ('target_assembly','different-explicit-assembly'), ('feature_category','curated_ccre')])
def test_identity_binds_declarations(resources, field, value):
    old = r.build_regulatory_feature_reference(**resources)
    new = r.build_regulatory_feature_reference(**(resources | {field:value}))
    assert old.reference_identity_sha256 != new.reference_identity_sha256
    assert old.features.ordered_feature_sha256 == new.features.ordered_feature_sha256


@pytest.mark.parametrize('species', ['macaque', {'scientific_name':'macaque','taxonomy_id':9541},
    {'scientific_name':'Macaca fascicularis','taxonomy_id':True},
    {'scientific_name':'Macaca fascicularis','taxonomy_id':0},
    {'scientific_name':'Macaca fascicularis'},
    {'scientific_name':'Macaca fascicularis','taxonomy_id':9541,'extra':'x'}])
def test_species_closed(resources, species):
    with pytest.raises(ValueError): r.build_regulatory_feature_reference(**(resources | {'species':species}))


@pytest.mark.parametrize('bed', ['chr1\t0\t10\nchr1\t0\t10\n', 'chr1\t0\t101\n',
    'chr1\t-1\t10\n', 'chr1\t10\t10\n', 'chr1\t20\t10\n', '1\t0\t10\n'])
def test_bad_features(resources, bed):
    resources['feature_bed_path'].write_text(bed)
    with pytest.raises(ValueError): r.build_regulatory_feature_reference(**resources)


@pytest.mark.parametrize('fai', ['chr1\t99\t6\t100\t101\n', 'chr1\t100\t7\t100\t101\n'])
def test_fai_sequence_consistency(resources, fai):
    resources['fai_path'].write_text(fai)
    with pytest.raises(ValueError): r.build_regulatory_feature_reference(**resources)


def test_complete_contig_order(resources):
    resources['fasta_path'].write_text('>chr2\nAAAA\n>chr1\n'+'A'*100+'\n')
    resources['fai_path'].write_text('chr2\t4\t6\t4\t5\nchr1\t100\t17\t100\t101\n')
    b = r.build_regulatory_feature_reference(**resources)
    assert b.genome.ordered_contig_sha256 == hashlib.sha256(b'chr2\t4\nchr1\t100\n').hexdigest()
    resources['fai_path'].write_text('chr1\t100\t17\t100\t101\nchr2\t4\t6\t4\t5\n')
    with pytest.raises(ValueError): r.build_regulatory_feature_reference(**resources)


@pytest.mark.parametrize('resource', ['fasta_path','fai_path','feature_bed_path'])
def test_source_mutation(resources, resource):
    b = r.build_regulatory_feature_reference(**resources)
    p = resources[resource]
    p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError): r.reinspect_regulatory_feature_reference(b)


def test_manifest_mutation(resources, tmp_path):
    b = r.build_regulatory_feature_reference(**resources)
    ptr = r.publish_regulatory_feature_reference(b, tmp_path/'ref.json')
    value = b.to_dict(); value['features']['category'] = 'curated_ccre'
    (tmp_path/'ref.json').write_bytes(legacy._json_bytes(value))
    with pytest.raises(ValueError): r.load_regulatory_feature_reference(ptr['manifest_path'], expected_sha256=ptr['manifest_sha256'])
    with pytest.raises(ValueError): r.load_regulatory_feature_reference(ptr['manifest_path'])


def test_feature_bound(resources, monkeypatch):
    monkeypatch.setattr(r, 'MAX_FEATURES', 2)
    with pytest.raises(ValueError): r.build_regulatory_feature_reference(**resources)


def test_reordering_is_not_silently_sorted(resources):
    first = r.build_regulatory_feature_reference(**resources)
    resources['feature_bed_path'].write_text('chr1\t0\t20\nchr1\t10\t25\nchr1\t30\t40\n')
    second = r.build_regulatory_feature_reference(**resources)
    assert first.features.ordered_feature_sha256 != second.features.ordered_feature_sha256
    assert first.reference_identity_sha256 != second.reference_identity_sha256


def test_extra_labels_are_only_source_provenance(resources):
    first = r.build_regulatory_feature_reference(**resources)
    resources['feature_bed_path'].write_text('chr1\t30\t40\tcustom-z\nchr1\t0\t20\tcustom-a\nchr1\t10\t25\toverlap\n')
    second = r.build_regulatory_feature_reference(**resources)
    assert first.features.ordered_feature_sha256 == second.features.ordered_feature_sha256
    assert first.features.bed.sha256 != second.features.bed.sha256
    assert second.features.category == 'peak_set'
