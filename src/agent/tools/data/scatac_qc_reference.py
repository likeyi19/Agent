"""Immutable QC resources with independently reconstructed transcript lineage.

The synthetic transcript TSV is NOT a GTF parser or a production annotation
profile. Narrow GENCODE GTF decoding uses qualified pysam. Reinspection uses a
separate SQLite query path. No registered tool, fragment IO or cell selection.
"""
from dataclasses import asdict, dataclass, fields, replace
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

from . import scatac_reference as parent
from .scatac_qc_profile import PROFILE_SHA256, canonical, fail, ScATACQCError

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_ROW_BYTES = 4096
MAX_TSS_ROWS = 1_000_000
MAX_TRANSCRIPTS = 5_000_000
MAX_SIDECAR_BYTES = 2 * 1024**3
TSS_DOMAIN = b'agent.qc-ordered-tss.v1\0'
CONSTRUCTION_PROFILE = 'protein-coding-all-transcript-tss.v1'
SYNTHETIC_PROFILE = 'synthetic-transcript-tsv.v1'
GTF_PROFILE = 'pysam-0.24.1-gencode-transcript-gtf.v1'
GTF_RUNTIME_SHA256 = 'a8fbc8d3910a63708245bc81c6eeb81961c393bde3ef0b1d70ab2345de2c672a'
SYNTHETIC_PARSER_SHA256 = hashlib.sha256(SYNTHETIC_PROFILE.encode()).hexdigest()


@dataclass(frozen=True)
class QCResource:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class QCAnnotationSource:
    resource: QCResource
    format: str
    source: str
    release: str
    qualification: str = 'synthetic_only'
    parser_profile: str = SYNTHETIC_PROFILE
    parser_identity_sha256: str = SYNTHETIC_PARSER_SHA256


@dataclass(frozen=True)
class QCContig:
    name: str
    length: int
    classification: str


@dataclass(frozen=True)
class QCConstructionSummary:
    transcript_rows: int
    non_protein_coding: int
    outside_qc_contigs: int
    incomplete_window: int
    eligible_transcripts: int
    collapsed_tss_duplicates: int


@dataclass(frozen=True)
class ScATACQCReferenceBundle:
    species: str
    assembly: str
    parent_manifest: QCResource
    parent_reference_identity_sha256: str
    ordered_contig_sha256: str
    contigs: tuple[QCContig, ...]
    classification_source: str
    annotation: QCAnnotationSource
    tss: QCResource
    lineage: QCResource
    tss_rows: int
    ordered_tss_sha256: str
    construction: QCConstructionSummary
    resource_identity_sha256: str
    scientific_profile_sha256: str = PROFILE_SHA256
    construction_profile: str = CONSTRUCTION_PROFILE
    artifact_type: str = 'agent.scatac-qc-reference-bundle'
    schema_version: int = 1
    contract_version: str = 'scatac-qc-reference-bundle.v1'

    def to_dict(self):
        return asdict(self)


def _shape(value, cls):
    if type(value) is not dict or set(value) != {x.name for x in fields(cls)}:
        fail()
    return dict(value)


def _text(value):
    if type(value) is not str or not value or len(value.encode('utf-8')) > 2048 or any(ord(c) < 33 or ord(c) == 127 for c in value):
        fail()


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        fail('QC_DIGEST_INVALID')


def _integer(value, low=0, high=MAX_TRANSCRIPTS):
    if type(value) is not int or not low <= value <= high:
        fail()


def _resource(value):
    d = _shape(value, QCResource)
    parent._path_text(d['path'])
    _sha(d['sha256'])
    _integer(d['size_bytes'], 1, MAX_SIDECAR_BYTES)
    return QCResource(**d)


def _identity(bundle):
    d = bundle.to_dict()
    del d['resource_identity_sha256']
    # Exact parent manifest bytes remain bound; own resource locators are portable.
    for r in (d['parent_manifest'], d['annotation']['resource'], d['tss'], d['lineage']):
        del r['path']
    return hashlib.sha256(b'agent.qc-reference-identity.v1\0' + canonical(d)).hexdigest()


def validate_scatac_qc_reference_bundle(value):
    """Closed, IO-free schema/profile validation; not biological qualification."""
    try:
        d = _shape(value.to_dict() if type(value) is ScATACQCReferenceBundle else value, ScATACQCReferenceBundle)
        if len(canonical(d)) > MAX_MANIFEST_BYTES:
            fail('QC_MANIFEST_LIMIT')
        if ((d['species'], d['assembly']) not in (('human', 'hg38'), ('mouse', 'mm10'))
                or type(d['schema_version']) is not int or d['schema_version'] != 1
                or d['artifact_type'] != 'agent.scatac-qc-reference-bundle'
                or d['contract_version'] != 'scatac-qc-reference-bundle.v1'
                or d['scientific_profile_sha256'] != PROFILE_SHA256
                or d['construction_profile'] != CONSTRUCTION_PROFILE):
            fail()
        for k in ('parent_reference_identity_sha256', 'ordered_contig_sha256', 'ordered_tss_sha256', 'resource_identity_sha256'):
            _sha(d[k])
        _text(d['classification_source'])
        if type(d['contigs']) not in (list, tuple) or not 1 <= len(d['contigs']) <= 4096:
            fail('QC_CONTIG_INVALID')
        contigs = []
        for c in d['contigs']:
            c = QCContig(**_shape(c, QCContig))
            _text(c.name); _integer(c.length, 1, 2**63 - 1)
            if c.classification not in ('primary_nuclear_qc', 'mitochondrial', 'other'):
                fail('QC_CONTIG_INVALID')
            contigs.append(c)
        if len({c.name for c in contigs}) != len(contigs) or not any(c.classification == 'primary_nuclear_qc' for c in contigs):
            fail('QC_CONTIG_INVALID')
        digest = hashlib.sha256(''.join(f'{c.name}\t{c.length}\n' for c in contigs).encode()).hexdigest()
        if digest != d['ordered_contig_sha256']:
            fail('QC_CONTIG_INVALID')
        d['contigs'] = tuple(contigs)
        for k in ('parent_manifest', 'tss', 'lineage'):
            d[k] = _resource(d[k])
        a = _shape(d['annotation'], QCAnnotationSource)
        a['resource'] = _resource(a['resource'])
        _text(a['source']); _text(a['release'])
        parser = (a['format'], a['qualification'], a['parser_profile'], a['parser_identity_sha256'])
        if parser not in (
                ('synthetic_transcript_tsv', 'synthetic_only', SYNTHETIC_PROFILE, SYNTHETIC_PARSER_SHA256),
                ('gtf', 'source_declared_not_production_qualified', GTF_PROFILE, GTF_RUNTIME_SHA256)):
            fail('QC_ANNOTATION_PARSER_UNQUALIFIED')
        if a['parser_profile'] == GTF_PROFILE and a['source'] != 'GENCODE':
            fail('QC_ANNOTATION_SOURCE_INVALID')
        d['annotation'] = QCAnnotationSource(**a)
        s = QCConstructionSummary(**_shape(d['construction'], QCConstructionSummary))
        for v in asdict(s).values():
            _integer(v)
        _integer(d['tss_rows'], 1, MAX_TSS_ROWS)
        if (s.transcript_rows != s.non_protein_coding + s.outside_qc_contigs + s.incomplete_window + s.eligible_transcripts
                or s.eligible_transcripts != d['tss_rows'] + s.collapsed_tss_duplicates):
            fail('QC_CONSTRUCTION_INVALID')
        d['construction'] = s
        bundle = ScATACQCReferenceBundle(**d)
        if _identity(bundle) != bundle.resource_identity_sha256:
            fail('QC_IDENTITY_MISMATCH')
        return bundle
    except ScATACQCError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError):
        fail()


def canonical_qc_reference_bundle_bytes(value):
    return canonical(validate_scatac_qc_reference_bundle(value).to_dict())


def _file(path):
    p = parent._source_path(path)
    if p.stat().st_size > MAX_SIDECAR_BYTES:
        fail('QC_RESOURCE_LIMIT')
    return QCResource(str(p), parent._file_hash(p), p.stat().st_size)


def _lines(path):
    with open(path, 'rb') as f:
        while line := f.readline(MAX_ROW_BYTES + 1):
            if len(line) > MAX_ROW_BYTES or not line.endswith(b'\n') or b'\r' in line or b'\0' in line:
                fail('QC_ROW_INVALID')
            yield line


def _decimal(value):
    if re.fullmatch('0|[1-9][0-9]{0,18}', value) is None or int(value) > 2**63 - 1:
        fail('QC_ROW_INVALID')
    return int(value)


def _transcripts(path, parser_profile=SYNTHETIC_PROFILE):
    """Strict fixture format only: contig,start,end,strand,gene,transcript,biotype."""
    if parser_profile == GTF_PROFILE:
        from ._qc_gtf import gencode_transcripts
        yield from gencode_transcripts(path)
        return
    if parser_profile != SYNTHETIC_PROFILE:
        fail('QC_ANNOTATION_PARSER_UNQUALIFIED')
    for n, line in enumerate(_lines(path), 1):
        if n > MAX_TRANSCRIPTS:
            fail('QC_RESOURCE_LIMIT')
        a = line.decode('utf-8').rstrip('\n').split('\t')
        if len(a) != 7:
            fail('QC_ROW_INVALID')
        for field in a:
            _text(field)
        chrom, start, end, strand, gene, tx, bio = a
        start, end = _decimal(start), _decimal(end)
        if start >= end or strand not in ('+', '-'):
            fail('QC_ROW_INVALID')
        yield chrom, start, end, strand, gene, tx, bio


def _parent(resource):
    _, bundle, _ = parent.load_scatac_reference_bundle(resource.path, expected_sha256=resource.sha256)
    parent.reinspect_scatac_reference_bundle_sources(bundle)
    dictionary, _, _ = parent._inspect_fai(Path(bundle.genome.fai.path), Path(bundle.genome.fasta.path).stat().st_size)
    return bundle, dictionary


def ordered_tss_sha256(records):
    """Order-sensitive digest over exact BED4: contig,position,position+1,strand LF."""
    h = hashlib.sha256(TSS_DOMAIN)
    n = 0
    for chrom, position, strand in records:
        _text(chrom); _integer(position, 0, 2**63 - 2)
        if strand not in ('+', '-'):
            fail('QC_TSS_INVALID')
        h.update(f'{chrom}\t{position}\t{position+1}\t{strand}\n'.encode())
        n += 1
        if n > MAX_TSS_ROWS:
            fail('QC_RESOURCE_LIMIT')
    if not n:
        fail('QC_TSS_EMPTY')
    return h.hexdigest()


def _db(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA cache_size=-8192')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE tx (rank INTEGER, chrom TEXT, pos INTEGER, strand TEXT, gene TEXT, transcript TEXT UNIQUE)')
    return db


def build_scatac_qc_reference_bundle(*, parent_manifest_path, parent_manifest_sha256,
        annotation_path, annotation_source, annotation_release, classifications,
        classification_source, output_dir, parser_profile=SYNTHETIC_PROFILE):
    """Publish derived resources, without claiming canonical production qualification.

    classifications is an explicit tuple of QCContig in exact FAI order.
    GENCODE requires the explicitly selected pinned parser profile and source.
    """
    if parser_profile not in (SYNTHETIC_PROFILE, GTF_PROFILE):
        fail('QC_ANNOTATION_PARSER_UNQUALIFIED')
    if parser_profile == GTF_PROFILE and annotation_source != 'GENCODE':
        fail('QC_ANNOTATION_SOURCE_INVALID')
    stage = None
    try:
        pr = _file(parent_manifest_path)
        if pr.sha256 != parent_manifest_sha256:
            fail('QC_PARENT_MISMATCH')
        p, dictionary = _parent(pr)
        cs = tuple(classifications)
        if any(type(c) is not QCContig for c in cs) or [(c.name, c.length) for c in cs] != list(dictionary.items()):
            fail('QC_CONTIG_INVALID')
        annotation = _file(annotation_path)
        before = [parent._snapshot(Path(x.path)) for x in (pr, annotation)]
        final = Path(output_dir).absolute()
        if final.exists() or final.is_symlink():
            fail('QC_OUTPUT_CONFLICT')
        final = final.resolve()
        stage = Path(tempfile.mkdtemp(prefix='.qc-reference-', dir=final.parent))
        lookup = {c.name: (i, c) for i, c in enumerate(cs)}
        counts = [0, 0, 0, 0, 0, 0]
        with closing(_db(stage / 'work.sqlite')) as db:
            for chrom, start, end, strand, gene, tx, bio in _transcripts(annotation.path, parser_profile):
                counts[0] += 1
                if chrom not in lookup or end > dictionary[chrom]:
                    fail('QC_ANNOTATION_COORDINATE_INVALID')
                rank, contig = lookup[chrom]
                pos = start if strand == '+' else end - 1
                if bio != 'protein_coding':
                    counts[1] += 1
                elif contig.classification != 'primary_nuclear_qc':
                    counts[2] += 1
                elif pos < 2000 or pos + 2000 >= contig.length:
                    counts[3] += 1
                else:
                    counts[4] += 1
                    db.execute('INSERT INTO tx VALUES (?,?,?,?,?,?)', (rank, chrom, pos, strand, gene, tx))
            with (stage / 'tss.bed').open('wb') as tss, (stage / 'lineage.tsv').open('wb') as lineage:
                previous = None; n = 0
                for rank, chrom, pos, strand, gene, tx in db.execute('SELECT * FROM tx ORDER BY rank,pos,strand,gene,transcript'):
                    key = (rank, pos, strand)
                    if key != previous:
                        tss.write(f'{chrom}\t{pos}\t{pos+1}\t{strand}\n'.encode()); n += 1
                        if n > MAX_TSS_ROWS:
                            fail('QC_RESOURCE_LIMIT')
                    lineage.write(f'{chrom}\t{pos}\t{strand}\t{gene}\t{tx}\n'.encode())
                    previous = key
                counts[5] = counts[4] - n
        (stage / 'work.sqlite').unlink()
        if not n:
            fail('QC_TSS_EMPTY')
        tss_resource = _file(stage / 'tss.bed')
        ordered = hashlib.sha256(TSS_DOMAIN)
        for line in _lines(tss_resource.path):
            ordered.update(line)
        bundle = ScATACQCReferenceBundle(p.species, p.target_assembly, pr,
            p.reference_identity_sha256, p.genome.ordered_contig_sha256, cs, classification_source,
            (QCAnnotationSource(annotation, 'synthetic_transcript_tsv', annotation_source, annotation_release)
             if parser_profile == SYNTHETIC_PROFILE else
             QCAnnotationSource(annotation, 'gtf', annotation_source, annotation_release,
                 'source_declared_not_production_qualified', GTF_PROFILE, GTF_RUNTIME_SHA256)),
            tss_resource, _file(stage / 'lineage.tsv'), n, ordered.hexdigest(),
            QCConstructionSummary(*counts), '')
        bundle = replace(bundle, resource_identity_sha256=_identity(bundle))
        reinspect_scatac_qc_reference_bundle_sources(bundle)
        if before != [parent._snapshot(Path(x.path)) for x in (pr, annotation)]:
            fail('QC_SOURCE_CHANGED')
        bundle = replace(bundle, tss=replace(bundle.tss, path=str(final / 'tss.bed')),
                         lineage=replace(bundle.lineage, path=str(final / 'lineage.tsv')))
        (stage / 'manifest.json').write_bytes(canonical_qc_reference_bundle_bytes(bundle))
        for name in ('tss.bed', 'lineage.tsv', 'manifest.json'):
            with (stage / name).open('rb') as f:
                os.fsync(f.fileno())
        _fsync_dir(stage)
        os.rename(stage, final); stage = None
        _fsync_dir(final.parent)
        return bundle
    except ScATACQCError:
        raise
    except (OSError, ValueError, TypeError, AttributeError, sqlite3.Error):
        fail('QC_CONSTRUCTION_FAILED')
    finally:
        if stage is not None:
            shutil.rmtree(stage)


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def iter_canonical_tss_records(bundle):
    """Freshly verify the full resource chain, then stream TSSs with mutation checks."""
    b = reinspect_scatac_qc_reference_bundle_sources(bundle)
    before = parent._snapshot(Path(b.tss.path))
    h = hashlib.sha256()
    try:
        for line in _lines(b.tss.path):
            h.update(line)
            chrom, start, end, strand = line.decode().rstrip('\n').split('\t')
            yield chrom, int(start), strand
        if h.hexdigest() != b.tss.sha256:
            fail('QC_SOURCE_CHANGED')
    finally:
        if parent._snapshot(Path(b.tss.path)) != before:
            fail('QC_SOURCE_CHANGED')


def reinspect_scatac_qc_reference_bundle_sources(value):
    """Independent derivation check; never calls the builder or trusts its summaries."""
    b = validate_scatac_qc_reference_bundle(value)
    try:
        resources = (b.parent_manifest, b.annotation.resource, b.tss, b.lineage)
        before = [parent._snapshot(Path(r.path)) for r in resources]
        if any(_file(r.path) != r for r in resources):
            fail('QC_RESOURCE_MISMATCH')
        p, dictionary = _parent(b.parent_manifest)
        if (p.reference_identity_sha256 != b.parent_reference_identity_sha256
                or (p.species, p.target_assembly) != (b.species, b.assembly)
                or list(dictionary.items()) != [(c.name, c.length) for c in b.contigs]):
            fail('QC_PARENT_MISMATCH')
        lookup = {c.name: (i, c) for i, c in enumerate(b.contigs)}
        # The source parser is shared syntax only. Eligibility, extraction,
        # grouping, canonical serialization and statistics are reconstructed here.
        with tempfile.TemporaryDirectory(prefix='qc-verify-') as tmp:
            db = sqlite3.connect(Path(tmp) / 'verify.sqlite')
            try:
                db.execute('PRAGMA cache_size=-8192'); db.execute('PRAGMA temp_store=FILE')
                db.execute('CREATE TABLE source (rank INTEGER, chrom TEXT, start INTEGER, end INTEGER, strand TEXT, gene TEXT, transcript TEXT UNIQUE, coding INTEGER, qc INTEGER, size INTEGER)')
                for chrom, start, end, strand, gene, tx, bio in _transcripts(b.annotation.resource.path, b.annotation.parser_profile):
                    if chrom not in lookup or end > lookup[chrom][1].length:
                        fail('QC_ANNOTATION_COORDINATE_INVALID')
                    rank, c = lookup[chrom]
                    db.execute('INSERT INTO source VALUES (?,?,?,?,?,?,?,?,?,?)',
                        (rank, chrom, start, end, strand, gene, tx, int(bio == 'protein_coding'), int(c.classification == 'primary_nuclear_qc'), c.length))
                if db.execute('SELECT gene FROM source GROUP BY gene HAVING count(DISTINCT chrom)>1 OR count(DISTINCT strand)>1 OR count(DISTINCT coding)>1 LIMIT 1').fetchone():
                    fail('QC_GENE_IDENTITY_INVALID')
                db.execute("CREATE VIEW positions AS SELECT *, CASE strand WHEN '+' THEN start ELSE end-1 END AS pos FROM source")
                db.execute('CREATE VIEW eligible AS SELECT * FROM positions WHERE coding=1 AND qc=1 AND pos>=2000 AND pos<size-2000')
                stats = tuple(db.execute('SELECT count(*),sum(coding=0),sum(coding=1 AND qc=0),sum(coding=1 AND qc=1 AND (pos<2000 OR pos>=size-2000)) FROM positions').fetchone())
                eligible = db.execute('SELECT count(*) FROM eligible').fetchone()[0]
                tss_hash = hashlib.sha256(); order_hash = hashlib.sha256(TSS_DOMAIN); n = 0
                for chrom, pos, strand in db.execute('SELECT chrom,pos,strand FROM eligible GROUP BY rank,chrom,pos,strand ORDER BY rank,pos,strand'):
                    line = f'{chrom}\t{pos}\t{pos+1}\t{strand}\n'.encode()
                    tss_hash.update(line); order_hash.update(line); n += 1
                lineage_hash = hashlib.sha256()
                for chrom, pos, strand, gene, tx in db.execute('SELECT chrom,pos,strand,gene,transcript FROM eligible ORDER BY rank,pos,strand,gene,transcript'):
                    lineage_hash.update(f'{chrom}\t{pos}\t{strand}\t{gene}\t{tx}\n'.encode())
                expected = QCConstructionSummary(*(x or 0 for x in stats), eligible, eligible-n)
                if (expected != b.construction or n != b.tss_rows
                        or tss_hash.hexdigest() != b.tss.sha256 or order_hash.hexdigest() != b.ordered_tss_sha256
                        or lineage_hash.hexdigest() != b.lineage.sha256):
                    fail('QC_DERIVATION_MISMATCH')
            finally:
                db.close()
        if before != [parent._snapshot(Path(r.path)) for r in resources]:
            fail('QC_SOURCE_CHANGED')
        return b
    except ScATACQCError:
        raise
    except (OSError, ValueError, TypeError, sqlite3.Error):
        fail('QC_REINSPECTION_FAILED')


def _pairs(items):
    d = {}
    for k, v in items:
        if k in d:
            fail('QC_JSON_INVALID')
        d[k] = v
    return d


def load_scatac_qc_reference_bundle(path, *, expected_sha256=None):
    """Bounded manifest-only IO; no declared resource access."""
    try:
        p = parent._source_path(path)
        with p.open('rb') as f:
            raw = f.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            fail('QC_MANIFEST_LIMIT')
        sha = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and sha != expected_sha256:
            fail('QC_MANIFEST_MISMATCH')
        d = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs,
                       parse_constant=lambda _: fail('QC_JSON_INVALID'))
        return p, validate_scatac_qc_reference_bundle(d), sha
    except ScATACQCError:
        raise
    except (OSError, ValueError, TypeError, RecursionError):
        fail('QC_JSON_INVALID')


def publish_scatac_qc_reference_bundle(value, path):
    """Publish a new manifest without overwriting any existing file or resource."""
    raw = canonical_qc_reference_bundle_bytes(value)
    destination = Path(path).absolute()
    if destination.exists() or destination.is_symlink():
        fail('QC_OUTPUT_CONFLICT')
    fd, temporary = tempfile.mkstemp(prefix='.qc-manifest-', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.link(temporary, destination)
        _fsync_dir(destination.parent)
    except OSError:
        fail('QC_PUBLICATION_FAILED')
    finally:
        os.unlink(temporary)
    return {'manifest_path': str(destination.resolve()), 'manifest_sha256': hashlib.sha256(raw).hexdigest()}
