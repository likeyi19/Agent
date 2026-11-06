"""M11.4a synthetic resource and real narrow BEDTools qualification."""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from agent.tools.data import scatac_reference as parent
from agent.tools.data import scatac_qc_reference as q
from agent.tools.data import scatac_qc_profile as science


@pytest.fixture
def inputs(tmp_path):
    # FAI order is deliberately chr2, chr1, organelle: no name inference.
    fa = tmp_path / 'genome.fa'
    fa.write_text('>chr2\n' + 'A' * 6001 + '\n>chr1\n' + 'A' * 6001 + '\n>organelle\n' + 'A' * 6001 + '\n')
    fai = tmp_path / 'genome.fa.fai'
    fai.write_text('chr2\t6001\t6\t6001\t6002\nchr1\t6001\t6014\t6001\t6002\norganelle\t6001\t12027\t6001\t6002\n')
    bed = tmp_path / 'ccre.bed'; bed.write_text('chr2\t1\t3\n')
    p = parent.build_scatac_reference_bundle(species='human', target_assembly='hg38',
        fasta_path=fa, fai_path=fai, ccre_bed_path=bed)
    pr = parent.publish_scatac_reference_bundle(p, tmp_path / 'parent.json')
    source = tmp_path / 'synthetic.tsv'
    source.write_text('chr1\t3000\t3500\t+\tg1\tt1\tprotein_coding\n'
        'chr2\t2500\t3001\t-\tg2\tt2\tprotein_coding\n'
        'chr2\t3000\t3500\t+\tg3\tt3\tprotein_coding\n'
        'chr2\t3000\t3600\t+\tg3\tt4\tprotein_coding\n'
        'chr2\t1999\t3000\t+\tg4\tt5\tprotein_coding\n'
        'organelle\t3000\t3500\t+\tg5\tt6\tprotein_coding\n'
        'chr2\t3000\t3500\t+\tg6\tt7\tlncRNA\n')
    return dict(parent_manifest_path=pr['manifest_path'], parent_manifest_sha256=pr['manifest_sha256'],
        annotation_path=source, annotation_source='hand-derived-fixture', annotation_release='1',
        classifications=(q.QCContig('chr2',6001,'primary_nuclear_qc'), q.QCContig('chr1',6001,'primary_nuclear_qc'),
                         q.QCContig('organelle',6001,'mitochondrial')),
        classification_source='explicit-fixture-classification', output_dir=tmp_path / 'qc')


def test_roundtrip_independent_identity_and_lineage(inputs, tmp_path, monkeypatch):
    b = q.build_scatac_qc_reference_bundle(**inputs)
    assert b.tss_rows == 3
    assert b.construction == q.QCConstructionSummary(7,1,1,1,4,1)
    exact = b'chr2\t3000\t3001\t+\nchr2\t3000\t3001\t-\nchr1\t3000\t3001\t+\n'
    assert Path(b.tss.path).read_bytes() == exact
    assert b.ordered_tss_sha256 == hashlib.sha256(q.TSS_DOMAIN + exact).hexdigest()
    assert len(Path(b.lineage.path).read_text().splitlines()) == 4
    assert q.load_scatac_qc_reference_bundle(inputs['output_dir'] / 'manifest.json')[1] == b
    assert list(q.iter_canonical_tss_records(b)) == [('chr2',3000,'+'),('chr2',3000,'-'),('chr1',3000,'+')]
    with pytest.raises(FrozenInstanceError):
        b.species = 'mouse'
    monkeypatch.setattr(q, 'build_scatac_qc_reference_bundle', lambda **_: pytest.fail('production reuse'))
    assert q.reinspect_scatac_qc_reference_bundle_sources(b) == b
    pointer = q.publish_scatac_qc_reference_bundle(b, tmp_path / 'copy.json')
    assert q.load_scatac_qc_reference_bundle(pointer['manifest_path'], expected_sha256=pointer['manifest_sha256'])[1] == b
    with pytest.raises(science.ScATACQCError):
        q.publish_scatac_qc_reference_bundle(b, pointer['manifest_path'])


def test_mouse_and_parent_change(inputs, tmp_path):
    path, p, _ = parent.load_scatac_reference_bundle(inputs['parent_manifest_path'])
    mouse = replace(p, species='mouse', target_assembly='mm10')
    mouse = replace(mouse, reference_identity_sha256=parent._reference_digest(mouse))
    pointer = parent.publish_scatac_reference_bundle(mouse, tmp_path / 'mouse.json')
    inputs.update(parent_manifest_path=pointer['manifest_path'], parent_manifest_sha256=pointer['manifest_sha256'])
    b = q.build_scatac_qc_reference_bundle(**inputs)
    assert (b.species,b.assembly) == ('mouse','mm10')
    forged = replace(b, parent_reference_identity_sha256=p.reference_identity_sha256)
    forged = replace(forged, resource_identity_sha256=q._identity(forged))
    with pytest.raises(science.ScATACQCError, match='QC_PARENT_MISMATCH'):
        q.reinspect_scatac_qc_reference_bundle_sources(forged)


@pytest.mark.parametrize('key,value', [('species','rat'),('assembly','mm39'),('schema_version',True),
    ('scientific_profile_sha256','0'*64),('resource_identity_sha256','0'*64),('construction_profile','other'),
    ('tss_rows',0),('tss_rows',True)])
def test_invalid_manifest(inputs,key,value):
    b=q.build_scatac_qc_reference_bundle(**inputs).to_dict();b[key]=value
    with pytest.raises(science.ScATACQCError):q.validate_scatac_qc_reference_bundle(b)


@pytest.mark.parametrize('mode',['unknown','duplicate','subset','order','no_qc','bad_class'])
def test_classifications(inputs,mode):
    cs=list(inputs['classifications'])
    if mode=='unknown':cs[0]=replace(cs[0],name='unknown')
    if mode=='duplicate':cs[1]=cs[0]
    if mode=='subset':cs.pop()
    if mode=='order':cs.reverse()
    if mode=='no_qc':cs=[replace(c,classification='other') for c in cs]
    if mode=='bad_class':cs[0]=replace(cs[0],classification='guess')
    inputs['classifications']=cs
    with pytest.raises(science.ScATACQCError):q.build_scatac_qc_reference_bundle(**inputs)
    assert not inputs['output_dir'].exists()


@pytest.mark.parametrize('position,expected',[(1999,False),(2000,True),(4000,True),(4001,False)])
@pytest.mark.parametrize('strand',['+','-'])
def test_complete_windows(inputs,position,expected,strand):
    start,end=(position,position+1)
    inputs['annotation_path'].write_text(f'chr2\t{start}\t{end}\t{strand}\tg\tt\tprotein_coding\n')
    if expected:
        b=q.build_scatac_qc_reference_bundle(**inputs)
        assert list(q.iter_canonical_tss_records(b))==[('chr2',position,strand)]
    else:
        with pytest.raises(science.ScATACQCError,match='QC_TSS_EMPTY'):q.build_scatac_qc_reference_bundle(**inputs)


@pytest.mark.parametrize('row',[
    'unknown\t3000\t3100\t+\tg\tt\tprotein_coding\n',
    'chr2\t6000\t6002\t+\tg\tt\tprotein_coding\n',
    'chr2\t03000\t3100\t+\tg\tt\tprotein_coding\n',
    'chr2\t3000\t3000\t+\tg\tt\tprotein_coding\n',
    'chr2\t3000\t3100\t.\tg\tt\tprotein_coding\n',
    'chr2\t3000\t3100\t+\tg\tt\tprotein_coding',
    'x'*4097+'\n'])
def test_invalid_source(inputs,row):
    inputs['annotation_path'].write_text(row)
    with pytest.raises(science.ScATACQCError):q.build_scatac_qc_reference_bundle(**inputs)


def test_no_production_parser_or_path_fallback(inputs):
    with pytest.raises(science.ScATACQCError,match='PARSER_UNQUALIFIED'):
        q.build_scatac_qc_reference_bundle(**(inputs|{'parser_profile':'gencode-gtf.v1'}))


@pytest.mark.parametrize('resource',['annotation','tss','lineage','parent'])
def test_source_mutations(inputs,resource):
    b=q.build_scatac_qc_reference_bundle(**inputs)
    path={'annotation':b.annotation.resource.path,'tss':b.tss.path,'lineage':b.lineage.path,'parent':b.parent_manifest.path}[resource]
    with open(path,'ab') as f:f.write(b'\n')
    with pytest.raises(science.ScATACQCError):q.reinspect_scatac_qc_reference_bundle_sources(b)


def test_forged_self_consistent_tss_rejected(inputs):
    b=q.build_scatac_qc_reference_bundle(**inputs)
    p=Path(b.tss.path);p.write_bytes(p.read_bytes().replace(b'3000\t3001',b'3001\t3002'))
    h=hashlib.sha256(q.TSS_DOMAIN+p.read_bytes()).hexdigest()
    forged=replace(b,tss=q._file(p),ordered_tss_sha256=h)
    forged=replace(forged,resource_identity_sha256=q._identity(forged))
    q.validate_scatac_qc_reference_bundle(forged)
    with pytest.raises(science.ScATACQCError,match='DERIVATION_MISMATCH'):
        q.reinspect_scatac_qc_reference_bundle_sources(forged)


def test_io_free_validation_and_strict_json(inputs,monkeypatch,tmp_path):
    b=q.build_scatac_qc_reference_bundle(**inputs)
    monkeypatch.setattr(q,'_file',lambda _:pytest.fail('scientific IO'))
    assert q.validate_scatac_qc_reference_bundle(b)==b
    for text in ['{"x":1,"x":2}','{"x":NaN}','[']:
        p=tmp_path/'bad.json';p.write_text(text)
        with pytest.raises(science.ScATACQCError):q.load_scatac_qc_reference_bundle(p)
    with pytest.raises(science.ScATACQCError):
        q.load_scatac_qc_reference_bundle(inputs['output_dir']/'manifest.json',expected_sha256='0'*64)


def test_order_identity():
    rows=[('chr2',2000,'+'),('chr1',2000,'-')]
    assert q.ordered_tss_sha256(rows)!=q.ordered_tss_sha256(rows[::-1])
    assert q.ordered_tss_sha256(rows)!=q.ordered_tss_sha256([('chr2',2001,'+'),rows[1]])
    assert q.ordered_tss_sha256(rows)!=q.ordered_tss_sha256([('chr2',2000,'-'),rows[1]])
    with pytest.raises(science.ScATACQCError):q.ordered_tss_sha256([])


def _gtf(inputs):
    rows = inputs['annotation_path'].read_text().splitlines()
    lines = ['##description: synthetic GENCODE-shaped fixture; not biological provenance']
    for row in rows:
        chrom,start,end,strand,gene,tx,bio = row.split('\t')
        lines.append(f'{chrom}\tTEST\ttranscript\t{int(start)+1}\t{end}\t.\t{strand}\t.\t'
            f'gene_id "{gene}"; transcript_id "{tx}"; gene_type "{bio}"; tag "basic"; tag "other";')
    inputs['annotation_path'].write_text('\n'.join(lines)+'\n')
    inputs.update(parser_profile=q.GTF_PROFILE, annotation_source='GENCODE', annotation_release='synthetic-test')


def test_qualified_mature_gtf_parser(inputs):
    _gtf(inputs)
    b = q.build_scatac_qc_reference_bundle(**inputs)
    assert b.construction == q.QCConstructionSummary(7,1,1,1,4,1)
    assert list(q.iter_canonical_tss_records(b)) == [('chr2',3000,'+'),('chr2',3000,'-'),('chr1',3000,'+')]
    assert b.annotation.qualification == 'source_declared_not_production_qualified'
    assert b.annotation.parser_identity_sha256 == q.GTF_RUNTIME_SHA256


@pytest.mark.parametrize('old,new', [
    ('gene_type "protein_coding";', ''),
    ('gene_id "g1";', 'gene_id "g1"; gene_id "duplicate";'),
    ('\t3001\t3500\t', '\t0\t3500\t'),
    ('\t3001\t3500\t', '\t4000\t3500\t'),
    ('\t+\t.', '\t.\t.'),
    ('chr1\tTEST', 'unknown\tTEST'),
    ('chr1\tTEST\ttranscript', 'chr1\ttranscript'),
])
def test_gtf_malformed_or_ambiguous(inputs,old,new):
    _gtf(inputs)
    p=inputs['annotation_path']; p.write_text(p.read_text().replace(old,new))
    with pytest.raises(science.ScATACQCError):
        q.build_scatac_qc_reference_bundle(**inputs)
    assert not inputs['output_dir'].exists()


def test_parser_identity_mismatch(inputs,monkeypatch):
    from agent.tools.data import _qc_gtf
    _gtf(inputs)
    monkeypatch.setattr(_qc_gtf, '_sha', lambda _: '0'*64)
    with pytest.raises(science.ScATACQCError,match='PARSER_UNQUALIFIED'):
        q.build_scatac_qc_reference_bundle(**inputs)


def test_duplicate_transcript_rejected(inputs):
    p=inputs['annotation_path']; p.write_text(p.read_text()+p.read_text().splitlines()[0]+'\n')
    with pytest.raises(science.ScATACQCError):q.build_scatac_qc_reference_bundle(**inputs)


def test_inconsistent_gene_identity_rejected(inputs):
    p=inputs['annotation_path'];p.write_text(p.read_text().replace('g6\tt7','g3\tt7'))
    with pytest.raises(science.ScATACQCError,match='GENE_IDENTITY_INVALID'):
        q.build_scatac_qc_reference_bundle(**inputs)


@pytest.mark.parametrize('resource',['lineage','summary'])
def test_self_consistent_false_derivation(inputs,resource):
    b=q.build_scatac_qc_reference_bundle(**inputs)
    if resource=='lineage':
        p=Path(b.lineage.path);p.write_text(p.read_text().replace('g1','fake'))
        b=replace(b,lineage=q._file(p))
    else:
        b=replace(b,construction=replace(b.construction,non_protein_coding=0,outside_qc_contigs=2))
    b=replace(b,resource_identity_sha256=q._identity(b))
    q.validate_scatac_qc_reference_bundle(b)
    with pytest.raises(science.ScATACQCError,match='DERIVATION_MISMATCH'):
        q.reinspect_scatac_qc_reference_bundle_sources(b)


def test_bounds(inputs,monkeypatch,tmp_path):
    monkeypatch.setattr(q,'MAX_TRANSCRIPTS',2)
    with pytest.raises(science.ScATACQCError,match='RESOURCE_LIMIT'):
        q.build_scatac_qc_reference_bundle(**inputs)
    monkeypatch.setattr(q,'MAX_MANIFEST_BYTES',10)
    p=tmp_path/'large.json';p.write_bytes(b' '*11)
    with pytest.raises(science.ScATACQCError,match='MANIFEST_LIMIT'):
        q.load_scatac_qc_reference_bundle(p)


@pytest.mark.parametrize('suffix', [
    ' broken "unterminated;', ' broken unquoted;', ' broken "extra"quote";',
    ' broken "escaped\\quote";', ' broken "semi;colon";', ' garbage',
])
@pytest.mark.parametrize('feature', ['transcript', 'gene'])
def test_gtf_rejects_permissive_attribute_repairs(inputs,suffix,feature):
    _gtf(inputs)
    p=inputs['annotation_path']
    malformed=('chr2\tTEST\t'+feature+'\t3001\t3500\t.\t+\t.\t'
        'gene_id "g"; transcript_id "t"; gene_type "protein_coding";'+suffix+'\n')
    p.write_text(p.read_text()+malformed)
    with pytest.raises(science.ScATACQCError,match='QC_GTF_INVALID'):
        q.build_scatac_qc_reference_bundle(**inputs)
    assert not inputs['output_dir'].exists()


@pytest.mark.parametrize('old,new', [
    ('gene_id "g1";', 'gene_id g1;'),
    ('tag "other";\n', 'tag "other"\n'),
    ('gene_id "g1"; transcript_id', 'gene_id "g1";  transcript_id'),
])
def test_gtf_narrow_attribute_dialect(inputs,old,new):
    _gtf(inputs)
    p=inputs['annotation_path'];p.write_text(p.read_text().replace(old,new))
    with pytest.raises(science.ScATACQCError,match='QC_GTF_INVALID'):
        q.build_scatac_qc_reference_bundle(**inputs)


def test_gtf_accepts_integer_metadata_and_quoted_spaces(inputs):
    _gtf(inputs)
    p=inputs['annotation_path']
    p.write_text(p.read_text().replace('tag "other";', 'tag "other"; level 2; note "two words";'))
    b=q.build_scatac_qc_reference_bundle(**inputs)
    assert b.construction == q.QCConstructionSummary(7,1,1,1,4,1)


def test_all_derived_resources_forged_together(inputs):
    b=q.build_scatac_qc_reference_bundle(**inputs)
    tss=Path(b.tss.path); lineage=Path(b.lineage.path)
    tss.write_text('chr2\t3100\t3101\t+\n')
    lineage.write_text('chr2\t3100\t+\tfake_gene\tfake_transcript\n')
    forged=replace(b,tss=q._file(tss),lineage=q._file(lineage),tss_rows=1,
        ordered_tss_sha256=q.ordered_tss_sha256([('chr2',3100,'+')]),
        construction=q.QCConstructionSummary(7,4,1,1,1,0))
    forged=replace(forged,resource_identity_sha256=q._identity(forged))
    q.validate_scatac_qc_reference_bundle(forged)
    with pytest.raises(science.ScATACQCError,match='DERIVATION_MISMATCH'):
        q.reinspect_scatac_qc_reference_bundle_sources(forged)


@pytest.mark.parametrize('size,accepted', [
    (2 * 1024**3 + 1, True), (4 * 1024**3, True),
    (4 * 1024**3 + 1, False), (True, False), (0, False),
    ('3323462848', False), (3323462848.0, False),
])
def test_gtf_annotation_manifest_size_boundary(inputs, size, accepted):
    _gtf(inputs)
    b = q.build_scatac_qc_reference_bundle(**inputs)
    b = replace(b, annotation=replace(b.annotation,
        resource=replace(b.annotation.resource, size_bytes=size)))
    b = replace(b, resource_identity_sha256=q._identity(b))
    if accepted:
        assert q.validate_scatac_qc_reference_bundle(b) == b
        assert json.loads(q.canonical_qc_reference_bundle_bytes(b))['annotation']['resource']['size_bytes'] == size
        # Plausible metadata alone does not establish real source integrity.
        with pytest.raises(science.ScATACQCError, match='QC_RESOURCE_MISMATCH'):
            q.reinspect_scatac_qc_reference_bundle_sources(b)
    else:
        with pytest.raises(science.ScATACQCError):
            q.validate_scatac_qc_reference_bundle(b)


@pytest.mark.parametrize('role', ['parent_manifest', 'tss', 'lineage', 'synthetic'])
def test_non_gtf_resource_manifest_bound_unchanged(inputs, role):
    b = q.build_scatac_qc_reference_bundle(**inputs)
    for size, accepted in ((q.MAX_SIDECAR_BYTES, True), (q.MAX_SIDECAR_BYTES + 1, False)):
        if role == 'synthetic':
            forged = replace(b, annotation=replace(b.annotation,
                resource=replace(b.annotation.resource, size_bytes=size)))
        else:
            forged = replace(b, **{role: replace(getattr(b, role), size_bytes=size)})
        forged = replace(forged, resource_identity_sha256=q._identity(forged))
        if accepted:
            q.validate_scatac_qc_reference_bundle(forged)
        else:
            with pytest.raises(science.ScATACQCError):
                q.validate_scatac_qc_reference_bundle(forged)


def test_sparse_source_size_gate_before_hashing(tmp_path, monkeypatch):
    path = tmp_path / 'sparse.gtf'
    hashed = []
    monkeypatch.setattr(parent, '_file_hash', lambda p: hashed.append(p) or '0'*64)
    for size in (q.MAX_SIDECAR_BYTES + 1, q.MAX_ANNOTATION_BYTES):
        with path.open('wb') as f:
            f.truncate(size)
        assert q._file(path, max_bytes=q._annotation_limit(q.GTF_PROFILE)).size_bytes == size
        with pytest.raises(science.ScATACQCError, match='QC_RESOURCE_LIMIT'):
            q._file(path)
    with path.open('wb') as f:
        f.truncate(q.MAX_ANNOTATION_BYTES + 1)
    with pytest.raises(science.ScATACQCError, match='QC_RESOURCE_LIMIT'):
        q._file(path, max_bytes=q._annotation_limit(q.GTF_PROFILE))
    assert len(hashed) == 2


def test_gtf_builder_load_and_reinspection_share_bound(inputs, monkeypatch):
    _gtf(inputs)
    # A bounded comment makes the annotation larger than the parent manifest.
    path = inputs['annotation_path']
    path.write_bytes(b'#' + b'x' * 3000 + b'\n' + path.read_bytes())
    size = path.stat().st_size
    monkeypatch.setattr(q, 'MAX_SIDECAR_BYTES', size - 1)
    monkeypatch.setattr(q, 'MAX_ANNOTATION_BYTES', size)
    b = q.build_scatac_qc_reference_bundle(**inputs)
    loaded = q.load_scatac_qc_reference_bundle(inputs['output_dir'] / 'manifest.json')[1]
    assert q.reinspect_scatac_qc_reference_bundle_sources(loaded) == b
    monkeypatch.setattr(q, 'MAX_ANNOTATION_BYTES', size - 1)
    with pytest.raises(science.ScATACQCError):
        q.load_scatac_qc_reference_bundle(inputs['output_dir'] / 'manifest.json')
    with pytest.raises(science.ScATACQCError):
        q.reinspect_scatac_qc_reference_bundle_sources(b)
    with pytest.raises(science.ScATACQCError, match='QC_RESOURCE_LIMIT'):
        q.build_scatac_qc_reference_bundle(**(inputs | {'output_dir': inputs['output_dir'].with_name('other')}))


@pytest.mark.parametrize('field,value', [('format', 'gtf'), ('parser_profile', q.GTF_PROFILE),
    ('qualification', 'source_declared_not_production_qualified')])
def test_forged_annotation_profile_cannot_widen_bound(inputs, field, value):
    b = q.build_scatac_qc_reference_bundle(**inputs)
    annotation = replace(b.annotation, **{field: value},
        resource=replace(b.annotation.resource, size_bytes=q.MAX_SIDECAR_BYTES + 1))
    b = replace(b, annotation=annotation)
    b = replace(b, resource_identity_sha256=q._identity(b))
    with pytest.raises(science.ScATACQCError):
        q.validate_scatac_qc_reference_bundle(b)
    malformed = b.to_dict()
    malformed['annotation']['resource']['max_bytes'] = q.MAX_ANNOTATION_BYTES
    with pytest.raises(science.ScATACQCError):
        q.validate_scatac_qc_reference_bundle(malformed)
