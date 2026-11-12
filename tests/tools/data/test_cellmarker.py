"""Small synthetic reference witnesses; genuine CellMarker acceptance is external."""
from copy import deepcopy
import csv
import io
from pathlib import Path
import sqlite3
import zipfile

import pytest

from agent.application.shared_resources import ScientificResourceLibrary, discover
from agent.application.local_resources import LocalResourceAdmission
from agent.tools._file_identity import file_closure
from agent.tools.data import cellmarker as c


def pin(path):
    return file_closure([path])[0]


def fixture_resource(root, rows):
    root.mkdir(parents=True)
    archives = {}
    for name in c.ARCHIVES:
        buffer = io.StringIO(newline='')
        writer = csv.writer(buffer, delimiter='\t', lineterminator='\n')
        writer.writerow(c.COLUMNS)
        writer.writerows([[row[k] for k in c.COLUMNS] for row in rows])
        with zipfile.ZipFile(root/name, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            # Deliberately synthetic: the four nonoperational archives are only
            # pinned provenance fixtures, never concatenated or imported.
            z.writestr(name.replace('.zip', '.txt'), buffer.getvalue())
        archives[name] = pin(root/name)
    index = c.build_index(root/c.ARCHIVES[0], root/'cellmarker3.sqlite',
                          expected_sha256=archives[c.ARCHIVES[0]]['sha256'])
    return dict(schema=c.RESOURCE, index_format=c.FORMAT,
                record_url='fixture:synthetic-cellmarker-not-biological-evidence',
                archives=archives, index=index)


@pytest.fixture
def sample(tmp_path):
    base = dict.fromkeys(c.COLUMNS, '')
    base.update(species='Mouse', tissue_class='Adipose tissue', tissue_type='Adipose tissue',
        disease_category='Normal', disease='Normal', cell_name_class='Endothelial cell',
        cell_name='Endothelial cell', uberon_id='UBERON_0001013', cellontology_id='CL_0000115',
        marker='Gpr116', symbol='Adgrf5', gene_id='224792.0', gene_type='protein_coding',
        gene_name='adhesion G protein-coupled receptor F5', uniprot_id='G5E8Q8',
        technology_seq='FACS', marker_source='Method', pmid='40670619.0',
        title='Synthetic title', journal='Synthetic journal', year='2025.0',
        series_id='Figshare_40670619', method_details='CelliD')
    rows = [base, dict(base), dict(base, method_details='COSG'),
        dict(base, tissue_type='Lung', cell_name='Macrophage', marker_source='Experiment',
             pmid='123.0', method_details=''),
        dict(base, species='Human', marker='GPR116', symbol='ADGRF5'),
        dict(base, marker='Mystery', symbol='', gene_id='', pmid='', method_details='', marker_source='Company'),
        dict(base, species='Human', marker='CD3D', symbol='CD3D'),
        dict(base, pmid='40670620.0'),
        dict(base, marker='LargeID', gene_id='9007199254740993.0',
             title=' leading "quoted"\tβ title\n', gene_name='未知')]
    resource = fixture_resource(tmp_path/'reference', rows)
    library = ScientificResourceLibrary(tmp_path/'library')
    entry = library.install('cellmarker', resource, label='Synthetic CellMarker', version='test-1',
        provenance='Synthetic test fixture', usage='Test lookup only')
    return rows, resource, library, entry


def items(lookup, *args, **kwargs):
    return [item for page in lookup.query(*args, **kwargs) for item in page['items']]


def test_all_strings_positions_and_reproducible_immutable_build(sample, tmp_path):
    rows, resource, _, _ = sample
    archive = resource['archives'][c.ARCHIVES[0]]
    index = resource['index']
    with sqlite3.connect(index['path']) as db:
        actual = db.execute('SELECT * FROM evidence ORDER BY source_row').fetchall()
    assert actual == [(i, *(row[k] for k in c.COLUMNS)) for i,row in enumerate(rows,1)]
    assert len(actual) == len(rows)  # only All, not five-archive concatenation
    again = c.build_index(archive['path'], index['path'], expected_sha256=archive['sha256'])
    assert again == index
    second = c.build_index(archive['path'], tmp_path/'second.sqlite', expected_sha256=archive['sha256'])
    assert second['sha256'] == index['sha256']
    changed = fixture_resource(tmp_path/'another', [dict(rows[0], marker='Other')])
    with pytest.raises(ValueError, match='different source'):
        c.build_index(changed['archives'][c.ARCHIVES[0]]['path'],index['path'],
                      expected_sha256=changed['archives'][c.ARCHIVES[0]]['sha256'])
    assert pin(index['path']) == index


def test_exact_fields_duplicates_context_and_batch_matches(sample):
    _, _, library, entry = sample
    lookup = c.CellMarkerEvidence(library, entry['id'])
    result = items(lookup, ['Gpr116', 'Adgrf5', 'Gpr116'], species='Mouse', page_size=1)
    first = result[0]
    assert first['fields']['marker'] == 'Gpr116' and first['fields']['symbol'] == 'Adgrf5'
    assert first['source_rows'] == [1, 2] and first['multiplicity'] == 2
    assert [(m['position'],m['field']) for m in first['matches']] == [(0,'marker'),(2,'marker'),(1,'symbol')]
    assert [r['source_rows'] for r in result] == [[1,2],[3],[4],[8],[9]]
    assert len({pos for r in result for pos in r['source_rows']}) == sum(r['multiplicity'] for r in result)
    assert {r['fields']['marker_source'] for r in result} == {'Method','Experiment'}
    assert {r['fields']['method_details'] for r in result} == {'CelliD','COSG',''}
    assert {r['fields']['pmid'] for r in result} >= {'123.0','40670619.0','40670620.0'}
    # Matching marker and symbol on one source record does not duplicate that row.
    same = items(lookup, 'CD3D', species='Human')
    assert len(same) == 1 and same[0]['source_rows'] == [7]
    assert [m['field'] for m in same[0]['matches']] == ['marker','symbol']


def test_missing_unmatched_species_ids_and_exact_filters(sample):
    _, _, library, entry = sample
    lookup = c.CellMarkerEvidence(library, entry['id'])
    missing = items(lookup, 'Mystery')[0]
    assert missing['fields']['symbol'] == missing['fields']['pmid'] == missing['fields']['method_details'] == ''
    assert missing['matches'][0]['field'] == 'marker'
    assert not items(lookup, 'gpr116') and not items(lookup, 'Gpr116', species='mouse')
    assert items(lookup, 'GPR116', species='Human')[0]['fields']['species'] == 'Human'
    assert not items(lookup, '', gene_ids=[''])
    result = list(lookup.query(['NoSuchMarker']))
    assert len(result) == 1 and result[0]['items'] == [] and result[0]['complete']
    assert result[0]['query']['names'] == ['NoSuchMarker']
    assert result[0]['resource']['resource_id'] == entry['id']
    assert not items(lookup, gene_ids=['224792'])  # no implicit numeric/namespace conversion
    exact = items(lookup, gene_ids=['9007199254740993.0'])
    assert exact[0]['fields']['gene_id'] == '9007199254740993.0'
    assert exact[0]['matches'] == [dict(kind='gene_id',position=0,value='9007199254740993.0',field='gene_id')]
    filtered = items(lookup, 'Gpr116', filters=dict(tissue_type='Lung', tissue_class='Adipose tissue',
        disease='Normal', disease_category='Normal', cell_name='Macrophage',
        marker_source='Experiment', technology_seq='FACS', method_details=''))
    assert [r['source_rows'] for r in filtered] == [[4]]
    with pytest.raises(ValueError): list(lookup.query('Gpr116', filters={'nonexistent':'x'}))
    with pytest.raises(ValueError): list(lookup.query(gene_ids=[224792.0]))


def test_large_batch_and_interleaved_technical_pages(sample):
    _, _, library, entry = sample
    lookup = c.CellMarkerEvidence(library, entry['id'])
    names = [f'Absent{i}' for i in range(33000)] + ['Gpr116','Mystery']
    batch = list(lookup.query(names, species='Mouse',page_size=2))
    singles = items(lookup, ['Gpr116','Mystery'],species='Mouse')
    assert [r['source_rows'] for page in batch for r in page['items']] == [r['source_rows'] for r in singles]
    assert [page['page'] for page in batch] == list(range(len(batch)))
    assert all(not p['complete'] for p in batch[:-1]) and batch[-1]['complete']
    a, b = lookup.query('Gpr116',page_size=1), lookup.query('Mystery',page_size=1)
    assert next(a)['items'][0]['source_rows'] == [1,2]
    assert next(b)['items'][0]['source_rows'] == [6]
    assert next(a)['items'][0]['source_rows'] == [3]
    a.close(); b.close()


def test_idempotent_library_restart_discovery_and_operation_reuse(sample, tmp_path, monkeypatch):
    _, resource, library, entry = sample
    monkeypatch.setattr(c, 'qualify_resource', lambda _:pytest.fail('Repeated qualification'))
    assert library.install('cellmarker', resource, label='Another label',version='test-1',
        provenance='Synthetic',usage='Test only') == entry
    restarted = ScientificResourceLibrary(library.root)
    lookup = c.CellMarkerEvidence(restarted, entry['id'])
    from agent.tools import _file_identity
    monkeypatch.setattr(_file_identity,'file_sha256',lambda _:pytest.fail('Lookup rehashed the file collection'))
    assert items(lookup, 'Mystery')
    assert items(lookup, 'Gpr116')
    admission = LocalResourceAdmission(tmp_path/'workspace', approved_source_roots=(tmp_path,),
                                        scientific_resource_root=library.root)
    assert discover(admission,'any') == () and restarted.models() == ()


@pytest.mark.parametrize('target', ['index','archive'])
def test_altered_bytes_fail_resolved_and_fresh_handles(sample,target):
    _, resource, library, entry = sample
    lookup = c.CellMarkerEvidence(library, entry['id'])
    path = Path(resource['index']['path'] if target=='index' else resource['archives'][c.ARCHIVES[-1]]['path'])
    with path.open('r+b') as f:
        f.seek(-1,2); byte=f.read(1);f.seek(-1,2);f.write(bytes([byte[0]^1]))
    with pytest.raises(ValueError): list(lookup.query('Gpr116'))
    with pytest.raises(ValueError): c.CellMarkerEvidence(library,entry['id'])


def test_tamper_between_pages(sample):
    _, resource, library, entry = sample
    lookup = c.CellMarkerEvidence(library,entry['id'])
    pages = lookup.query('Gpr116',page_size=1)
    next(pages)
    Path(resource['archives'][c.ARCHIVES[-1]]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):next(pages)


def test_pages_do_not_share_mutable_identity_and_main_database_is_read_only(sample):
    _, resource, library, entry = sample
    lookup = c.CellMarkerEvidence(library,entry['id'])
    pages = lookup.query('Gpr116',page_size=1)
    first = next(pages)
    first['query']['names'][0] = 'changed by consumer'
    first['resource']['archive']['sha256'] = 'changed by consumer'
    second = next(pages)
    assert second['query']['names'] == ['Gpr116']
    assert second['resource']['archive']['sha256'] == resource['archives'][c.ARCHIVES[0]]['sha256']
    pages.close()
    db = c._read(resource['index']['path'])
    try:
        with pytest.raises(sqlite3.OperationalError,match='readonly'):
            db.execute("UPDATE evidence SET marker='changed'")
    finally:
        db.close()


def test_forged_index_cannot_be_qualified_or_overwrite_existing(sample,tmp_path):
    _, resource, library, _ = sample
    index = Path(resource['index']['path']); archive=resource['archives'][c.ARCHIVES[0]]
    with sqlite3.connect(index) as db:
        db.execute("UPDATE evidence SET marker='FORGED' WHERE source_row=2")
    altered=pin(index); resource=deepcopy(resource);resource['index']=altered
    with pytest.raises(ValueError,match='source correspondence'):
        library.install('cellmarker',resource,label='Forged',version='test-2',provenance='Synthetic',usage='Test')
    with pytest.raises(ValueError,match='source correspondence'):
        c.build_index(archive['path'],index,expected_sha256=archive['sha256'])
    assert pin(index)==altered
    assert len(library.entries())==1


def test_failed_import_does_not_publish_partial_index(tmp_path):
    path=tmp_path/c.ARCHIVES[0]
    with zipfile.ZipFile(path,'w') as z:
        z.writestr(c.MEMBER,'\t'.join(c.COLUMNS)+'\n'+'too\tfew\n')
    target=tmp_path/'cellmarker3.sqlite'
    with pytest.raises(ValueError,match='field count'):
        c.build_index(path,target,expected_sha256=pin(path)['sha256'])
    assert not target.exists() and not list(tmp_path.glob('.cellmarker-*'))


def test_version_coexistence_and_unaffected_historical_resource(sample,tmp_path):
    rows, _, library, first = sample
    reference=tmp_path/'genes.gtf';reference.write_text('Synthetic pre-existing gene reference')
    gene=library.install('gene_annotation',dict(species='human',assembly='test',file={
        k:v for k,v in pin(reference).items() if k!='size_bytes'}),label='Historical fixture',version='1',
        provenance='Synthetic',usage='Test')
    before=library.choices()
    second_resource=fixture_resource(tmp_path/'v2', [dict(rows[0],marker='VersionTwo')])
    second=library.install('cellmarker',second_resource,label='Second',version='test-2',
        provenance='Synthetic',usage='Test')
    assert first['id'] != second['id']
    assert library.check(gene)==gene['resource'] and library.choices()==before
    assert items(c.CellMarkerEvidence(library,first['id']),'Gpr116')
    assert items(c.CellMarkerEvidence(library,second['id']),'VersionTwo')
