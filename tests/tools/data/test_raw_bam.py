"""Synthetic M10.3 BAM acceptance; all BAM/index files are temporary fixtures."""
from dataclasses import replace
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent.tools.data import _raw_bam as b
from agent.tools.data import raw_scatac_manifest as m

try:
    import pysam
except ImportError:
    pysam = None


def backend():
    if pysam is None:
        pytest.skip('Install optional requirements-bam.txt for BAM fixtures')
    return pysam


def header(*, species='Homo sapiens', assembly='GRCh38', sort='coordinate', hd=True,
           rgs=None, pgs=None, refs=1, m5=False):
    result = {'SQ': [{'SN': f'chr{i + 1}', 'LN': 100000} for i in range(refs)]}
    for entry in result['SQ']:
        if species is not None:
            entry['SP'] = species
        if assembly is not None:
            entry['AS'] = assembly
        if m5:
            entry['M5'] = 'a' * 32
    if hd:
        result['HD'] = {'VN': '1.6', **({'SO': sort} if sort is not None else {})}
    if rgs is not None:
        result['RG'] = rgs
    if pgs is not None:
        result['PG'] = pgs
    return result


def bam(path, *, hd=None, specs=None, tags=None):
    ps = backend()
    path.parent.mkdir(parents=True, exist_ok=True)
    if hd is None:
        hd = header()
    if specs is None:
        specs = [{}, {}]
    if tags is None:
        tags = [('CB', 'SECRET_CELL-1', 'Z')]
    with ps.AlignmentFile(str(path), 'wb', header=hd) as out:
        for i, spec in enumerate(specs):
            r = ps.AlignedSegment(out.header)
            r.query_name = spec.get('name', f'SECRET_READ_{i}')
            r.query_sequence = spec.get('sequence', 'ACGTACGT')
            r.query_qualities = ps.qualitystring_to_array('I' * len(r.query_sequence))
            r.flag = spec.get('flag', 0)
            r.reference_id = spec.get('tid', -1 if r.is_unmapped else 0)
            r.reference_start = spec.get('pos', -1 if r.is_unmapped else i * 20)
            if not r.is_unmapped:
                r.cigarstring = f'{len(r.query_sequence)}M'
                r.mapping_quality = 30
            for key, value, kind in spec.get('tags', tags):
                r.set_tag(key, value, value_type=kind, replace=False)
            out.write(r)
    return path


def inspect(path, **kwargs):
    kwargs.setdefault('assay', b.BamAssay.SCATAC)
    return b.inspect_bam_inputs(path, **kwargs)


def source(path):
    s = path.stat()
    return m.FileRecord(str(path), m.InputKind.BAM, s.st_size, s.st_mtime_ns)


def observe(path):
    return b.observe_bam_source(source(path))


def codes(result):
    return {i.code for i in result.issues}


def facts(result, name):
    return [json.loads(e.value) for e in result.evidence if e.value.startswith('{')
            and json.loads(e.value).get('fact') == name]


def tag(observation, name):
    return next(t for t in observation.tags if t.tag == name)


def test_optional_backend_version_and_dependency_declaration():
    assert backend().__version__ == '0.24.1'
    assert b._backend().version.__htslib_version__ == '1.24'
    lines = Path('requirements-bam.txt').read_text().splitlines()
    assert [v for v in lines if v and not v.startswith('#')] == ['pysam==0.24.1']


def test_imports_and_fastq_work_without_pysam(tmp_path):
    code = '''
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pysam' or fullname.startswith('pysam.'):
            raise ModuleNotFoundError('blocked optional backend')
sys.meta_path.insert(0, Reject())
import agent
from agent.tools.data import raw_scatac_manifest, _raw_fastq, _raw_bam
from agent.tools.data import inspect_scATAC
assert 'pysam' not in sys.modules
from pathlib import Path
path=Path(sys.argv[1]) / 'input.fastq'
path.write_bytes(b'@read\\nACGT\\n+\\nIIII\\n')
assert _raw_fastq.inspect_fastq_inputs(path).files
try:
    _raw_bam.inspect_bam_inputs(path)
except _raw_bam.BamInspectionError as exc:
    assert exc.code == 'RAW_BAM_DEPENDENCY_UNAVAILABLE'
else:
    raise AssertionError('BAM backend absence ignored')
print('lazy imports and FASTQ: PASS')
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'PASS' in result.stdout


def test_backend_version_mismatch_is_sanitized(monkeypatch):
    monkeypatch.setattr(b.importlib, 'import_module', lambda name: SimpleNamespace(__version__='other'))
    with pytest.raises(b.BamInspectionError, match='RAW_BAM_BACKEND_VERSION_UNSUPPORTED'):
        b.inspect_bam_inputs('/not/opened.bam')


@pytest.mark.parametrize('directory', [True, False])
def test_explicit_and_directory_bam_selection(tmp_path, directory):
    p = bam(tmp_path / 'input.bam')
    result = inspect(tmp_path if directory else p)
    assert result.readiness is m.Readiness.READY
    assert len(result.files) == len(result.groups) == 1
    assert result.files[0].selection is (m.SelectionBasis.DIRECTORY_DISCOVERY if directory else m.SelectionBasis.EXPLICIT)
    assert result.groups[0].route is m.IntakeRoute.BAM_TO_CCRE


def test_multiple_bams_deterministic_nonrecursive_independent_groups(tmp_path):
    a, c = bam(tmp_path / 'a.bam'), bam(tmp_path / 'c.bam')
    bam(tmp_path / 'nested' / 'b.bam')
    (tmp_path / 'note.txt').write_text('ignore')
    (tmp_path / 'orphan.bai').write_bytes(b'ignore')
    (tmp_path / 'orphan.csi').write_bytes(b'ignore')
    x = inspect(tmp_path)
    y = inspect((c, a))
    assert len(x.files) == len(x.groups) == 2
    assert [s.file.path for s in b._discover(tmp_path)] == [str(a), str(c)]
    assert m.canonical_manifest_bytes(y) == m.canonical_manifest_bytes(inspect((a, c)))
    assert len({g.barcode_identity_scope_id for g in x.groups}) == 2


def test_independent_root_same_names_never_merge(tmp_path):
    a, c = bam(tmp_path / 'a' / 'sample.bam'), bam(tmp_path / 'b' / 'sample.bam')
    result = inspect((a.parent, c.parent))
    assert result.readiness is m.Readiness.READY and len(result.groups) == 2


def test_duplicate_selections_and_canonical_symlink_alias_collapse(tmp_path):
    p = bam(tmp_path / 'source.bam')
    alias = tmp_path / 'alias.bam'
    alias.symlink_to(p)
    assert inspect((p, p, alias)) == inspect(p)


def test_directory_symlink_candidate_and_hardlink_alias_are_rejected(tmp_path):
    p = bam(tmp_path / 'source.bam')
    alias = tmp_path / 'alias.bam'
    alias.symlink_to(p)
    with pytest.raises(b.BamInspectionError, match='DISCOVERY_SYMLINK'):
        inspect(tmp_path)
    alias.unlink()
    os.link(p, alias)
    with pytest.raises(b.BamInspectionError, match='ALIAS_CONFLICT'):
        inspect((p, alias))


@pytest.mark.parametrize('selection', [(), [], 42, 'https://host/source.bam', 'bad\0file.bam'])
def test_invalid_selection_is_operational(selection):
    backend()
    with pytest.raises(b.BamInspectionError, match='SELECTION_INVALID'):
        inspect(selection)


def test_missing_and_no_candidates(tmp_path):
    backend()
    with pytest.raises(b.BamInspectionError, match='SELECTION_INVALID'):
        inspect(tmp_path / 'absent.bam')
    (tmp_path / 'x.bai').write_bytes(b'not a BAM')
    with pytest.raises(b.BamInspectionError, match='NO_CANDIDATES'):
        inspect(tmp_path)


@pytest.mark.parametrize('suffix', ['.sam', '.cram', '.bai', '.csi'])
def test_explicit_non_bam_suffix_unsupported(tmp_path, suffix):
    backend()
    p = tmp_path / ('x' + suffix)
    p.write_bytes(b'not a selected BAM')
    with pytest.raises(b.BamInspectionError, match='SUFFIX_UNSUPPORTED'):
        inspect(p)


@pytest.mark.parametrize('which', ['files', 'entries', 'selections'])
def test_discovery_limits(tmp_path, monkeypatch, which):
    paths = [bam(tmp_path / f'{i}.bam') for i in range(3)]
    monkeypatch.setattr(b, {'files': 'MAX_CANDIDATE_FILES', 'entries': 'MAX_DIRECTORY_ENTRIES',
                           'selections': 'MAX_SELECTIONS'}[which], 2)
    with pytest.raises(b.BamInspectionError):
        inspect(paths if which == 'selections' else tmp_path)


def test_special_file_not_opened(tmp_path):
    backend()
    path = tmp_path / 'fifo.bam'
    os.mkfifo(path)
    with pytest.raises(b.BamInspectionError, match='NOT_REGULAR_FILE'):
        inspect(path)


@pytest.mark.parametrize('hd_present', [False, True])
def test_hd_optional_reference_dictionary_and_m5(tmp_path, hd_present):
    p = bam(tmp_path / 'input.bam', hd=header(hd=hd_present, refs=2, m5=True))
    obs = observe(p)
    assert obs.header.hd_present is hd_present
    assert obs.header.reference_count == obs.header.m5_count == 2
    assert len(obs.header.reference_sha256) == len(obs.header.m5_sha256) == 64
    assert inspect(p).readiness is m.Readiness.READY


def test_dictionary_fingerprint_is_ordered_and_tracks_m5(tmp_path):
    h = header(refs=2, m5=True)
    a = observe(bam(tmp_path / 'a.bam', hd=h)).header
    h['SQ'].reverse()
    c = observe(bam(tmp_path / 'c.bam', hd=h)).header
    assert a.reference_sha256 != c.reference_sha256
    h['SQ'][0]['M5'] = 'b' * 32
    d = observe(bam(tmp_path / 'd.bam', hd=h)).header
    assert c.m5_sha256 != d.m5_sha256 and c.reference_sha256 != d.reference_sha256


@pytest.mark.parametrize('field,limit', [('references', 1), ('groups', 1), ('programs', 1), ('bytes', 8)])
def test_header_projection_limits_are_nonready_not_false_byte_guarantees(tmp_path, monkeypatch, field, limit):
    h = header(refs=2, rgs=[{'ID': 'a'}, {'ID': 'b'}], pgs=[{'ID': 'a'}, {'ID': 'b'}])
    p = bam(tmp_path / 'input.bam', hd=h)
    monkeypatch.setattr(b, {'references':'MAX_REFERENCES', 'groups':'MAX_READ_GROUPS',
                           'programs':'MAX_PROGRAMS', 'bytes':'MAX_HEADER_BYTES'}[field], limit)
    result = inspect(p)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert 'BAM_HEADER_LIMIT' in codes(result)
    assert result.coverage[0].decoded_bytes_inspected is None
    assert result.coverage[0].decoded_byte_limit is None
    assert result.coverage[0].stop_reason is m.StopReason.BUDGET


def test_metadata_collection_limit(tmp_path, monkeypatch):
    h = header(refs=2)
    h['SQ'][1]['AS'] = 'hg19'
    p = bam(tmp_path / 'input.bam', hd=h)
    monkeypatch.setattr(b, 'MAX_METADATA_VALUES', 1)
    assert 'BAM_HEADER_LIMIT' in codes(inspect(p))


@pytest.mark.parametrize('species,assembly,target', [('human','hg38','hg38'), ('mouse','mm10','mm10')])
def test_explicit_species_and_source_declarations(tmp_path, species, assembly, target):
    result = inspect(bam(tmp_path / 'input.bam', hd=header(species=None, assembly=None)),
                     species=species, source_genome_assembly=assembly)
    g = result.groups[0]
    assert result.readiness is m.Readiness.READY
    assert g.assembly_compatibility is m.AssemblyCompatibility.MATCH
    assert g.source_genome_assembly.value == assembly
    assert g.target_genome_assembly.value == target
    assert g.barcode.source is m.BarcodeSource.BAM_CELL_IDENTIFIER


@pytest.mark.parametrize('sp,assembly,species,target', [
    ('Homo sapiens','GRCh38','human','hg38'), ('human','hg38','human','hg38'),
    ('Mus musculus','GRCm38','mouse','mm10'), ('mouse','mm10','mouse','mm10'),
])
def test_reviewed_header_metadata_normalization(tmp_path, sp, assembly, species, target):
    result = inspect(bam(tmp_path / 'input.bam', hd=header(species=sp, assembly=assembly)))
    assert result.readiness is m.Readiness.READY
    g = result.groups[0]
    assert g.species.value == species and g.source_genome_assembly.value == target
    assert g.target_genome_assembly.value == target
    assert g.species.assertions[0].source is m.EvidenceSource.AUTHORITATIVE_METADATA


@pytest.mark.parametrize('assembly,canonical', [('hg19','hg19'), ('GRCh37','grch37')])
def test_mismatch_never_promises_supported_harmonization(tmp_path, assembly, canonical):
    result = inspect(bam(tmp_path / 'input.bam', hd=header(assembly=assembly)))
    g = result.groups[0]
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert g.source_genome_assembly.value == canonical
    assert g.target_genome_assembly.value == 'hg38'
    assert g.assembly_compatibility is m.AssemblyCompatibility.MISMATCH and g.harmonization_required
    assert result.repairs[0].admissibility is m.PreparationAdmissibility.UNRESOLVED
    assert m.InformationCode.ROUTE_ADMISSIBILITY in {r.code for r in result.required_information}
    assert result.repairs[0].route is None


@pytest.mark.parametrize('contig', ['chr1', '1', 'hg38_chr1'])
def test_species_and_contig_names_cannot_establish_source_assembly(tmp_path, contig):
    h = header(assembly=None)
    h['SQ'][0]['SN'] = contig
    result = inspect(bam(tmp_path / 'human_hg38_atac.bam', hd=h))
    g = result.groups[0]
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert g.source_genome_assembly.state is m.ResolutionState.UNKNOWN
    assert g.target_genome_assembly.value == 'hg38'
    assert m.InformationCode.SOURCE_ASSEMBLY in {r.code for r in result.required_information}


def test_species_not_inferred_from_filename_or_sequence(tmp_path):
    result = inspect(bam(tmp_path / 'human_hg38_atac.bam', hd=header(species=None)))
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert result.groups[0].species.state is m.ResolutionState.UNKNOWN
    assert result.groups[0].target_genome_assembly.state is m.TargetState.UNRESOLVED


@pytest.mark.parametrize('key,value', [('AS','hg19'), ('SP','Mus musculus')])
def test_conflicting_header_metadata_is_retained(tmp_path, key, value):
    h = header(refs=2)
    h['SQ'][1][key] = value
    result = inspect(bam(tmp_path / 'input.bam', hd=h))
    g = result.groups[0]
    metadata = g.species if key == 'SP' else g.source_genome_assembly
    assert metadata.state is m.ResolutionState.CONFLICT
    assert len(metadata.assertions) == 2 and result.readiness is m.Readiness.INVALID


@pytest.mark.parametrize('kwargs', [{'species':'mouse'}, {'source_genome_assembly':'hg19'}])
def test_declaration_does_not_override_contradictory_header(tmp_path, kwargs):
    result = inspect(bam(tmp_path / 'input.bam'), **kwargs)
    assert result.readiness is m.Readiness.INVALID
    assert {'SPECIES_CONFLICT', 'SOURCE_ASSEMBLY_CONFLICT'} & codes(result)


def test_partial_sq_metadata_is_observation_only_but_conflicts_survive(tmp_path):
    h = header(refs=2)
    del h['SQ'][1]['AS']
    p = bam(tmp_path / 'input.bam', hd=h)
    result = inspect(p)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert result.groups[0].source_genome_assembly.state is m.ResolutionState.UNKNOWN
    assert result.groups[0].source_genome_assembly.assertions[0].source is m.EvidenceSource.OBSERVATION
    assert inspect(p, source_genome_assembly='hg38').readiness is m.Readiness.READY
    assert inspect(p, source_genome_assembly='hg19').readiness is m.Readiness.INVALID


def test_unreviewed_species_and_assembly_are_not_fuzzy_matched(tmp_path):
    h = header(species='Homo sapiens-like', assembly='GRCh38 custom reference')
    result = inspect(bam(tmp_path / 'input.bam', hd=h))
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert 'BAM_METADATA_UNREVIEWED' in codes(result)
    assert result.groups[0].species.value is None
    assert result.groups[0].source_genome_assembly.value is None


@pytest.mark.parametrize('index_kind', ['none', 'bai', 'csi', 'corrupt', 'multiple', 'short-bai', 'short-csi'])
def test_index_observation_and_sequential_independence(tmp_path, index_kind):
    p = bam(tmp_path / 'input.bam')
    baseline = observe(p)
    if index_kind in ('bai','short-bai','multiple'):
        backend().index(str(p))
        if index_kind == 'short-bai':
            Path(str(p) + '.bai').rename(p.with_suffix('.bai'))
    if index_kind in ('csi','short-csi','multiple'):
        backend().index('-c', str(p))
        if index_kind == 'short-csi':
            Path(str(p) + '.csi').rename(p.with_suffix('.csi'))
    if index_kind == 'corrupt':
        Path(str(p) + '.bai').write_bytes(b'not an index')
    expected = b.IndexState.ABSENT if index_kind == 'none' else b.IndexState.UNUSABLE if index_kind == 'corrupt' else (
        b.IndexState.MULTIPLE if index_kind == 'multiple' else b.IndexState.OPENABLE)
    value = observe(p)
    assert value.index.state is expected
    assert value.counts == baseline.counts
    assert value.tags == baseline.tags
    assert value.normalized_observation_sha256 == baseline.normalized_observation_sha256
    result = inspect(p)
    assert result.readiness is m.Readiness.READY
    assert not result.repairs
    assert len(result.files) == len(result.groups) == 1


def test_index_symlink_not_followed(tmp_path):
    p = bam(tmp_path / 'input.bam')
    external = tmp_path / 'arbitrary'
    external.write_bytes(b'not a reference or index')
    Path(str(p) + '.bai').symlink_to(external)
    assert observe(p).index.state is b.IndexState.UNUSABLE
    assert inspect(p).readiness is m.Readiness.READY


@pytest.mark.parametrize('sort,positions,expected', [
    ('coordinate', [0,20], b.ObservedOrder.CONSISTENT),
    ('unsorted', [0,20], b.ObservedOrder.CONSISTENT),
    ('coordinate', [20,0], b.ObservedOrder.INVERSION),
    ('unsorted', [20,0], b.ObservedOrder.INVERSION),
    (None, [0], b.ObservedOrder.INSUFFICIENT),
])
def test_sort_declaration_and_sample_observation_are_separate(tmp_path, sort, positions, expected):
    p = bam(tmp_path / 'input.bam', hd=header(sort=sort), specs=[{'pos':v} for v in positions])
    observation = observe(p)
    assert observation.order is expected
    assert observation.header.sort_declaration.value == (sort or 'absent')
    result = inspect(p)
    assert result.readiness is m.Readiness.READY
    assert not result.repairs
    if expected is b.ObservedOrder.INVERSION:
        assert ('BAM_SORT_DECLARATION_CONFLICT' if sort == 'coordinate' else 'BAM_SORT_INVERSION_OBSERVED') in codes(result)


def test_secondary_and_supplementary_positions_do_not_define_primary_sort(tmp_path):
    p = bam(tmp_path / 'input.bam', specs=[{'pos':10}, {'pos':0,'flag':256}, {'pos':0,'flag':2048}, {'pos':20}])
    value = observe(p)
    assert value.order is b.ObservedOrder.CONSISTENT
    assert value.coordinate_comparisons == 1


def test_flag_counts_and_tag_denominators(tmp_path):
    tags = [('CB', 'SECRET_CELL-1', 'Z')]
    specs = [{'flag':99}, {'flag':147}, {'flag':4}, {'flag':256}, {'flag':2048},
             {'flag':1024 | 512, 'tags':[]}]
    value = observe(bam(tmp_path / 'input.bam', specs=specs, tags=tags))
    c = value.counts
    assert (c.records,c.primary,c.secondary,c.supplementary,c.mapped,c.unmapped,c.mapped_primary) == (6,4,1,1,5,1,3)
    assert (c.paired,c.unpaired,c.read1,c.read2,c.proper_pair,c.duplicate,c.qc_fail) == (2,4,1,1,2,1,1)
    cb = tag(value, 'CB')
    assert (cb.all.n, cb.all.present, cb.all.usable) == (6,5,5)
    assert (cb.primary.n, cb.primary.usable) == (4,3)
    assert (cb.mapped_primary.n, cb.mapped_primary.usable) == (3,2)


def test_record_budget_preserves_unmapped_records_and_does_not_peek_next(tmp_path, monkeypatch):
    monkeypatch.setattr(b,'MAX_RECORDS',2)
    p = bam(tmp_path / 'input.bam', specs=[{}, {'flag':4}, {'tags':[('CB', 2, 'i')]}])
    value = observe(p)
    assert value.counts.records == 2 and value.counts.unmapped == 1
    assert value.stop is b.ObservationStop.BUDGET
    result = inspect(p)
    assert result.readiness is m.Readiness.READY
    c = result.coverage[0]
    assert c.scope is m.CoverageScope.SAMPLE and not c.eof_observed
    assert c.record_limit == 2 and c.decoded_bytes_inspected is c.decoded_byte_limit is None


def test_small_complete_bam_and_header_only_bam(tmp_path):
    complete = inspect(bam(tmp_path / 'complete.bam'))
    assert complete.coverage[0].scope is m.CoverageScope.COMPLETE
    assert complete.coverage[0].stop_reason is m.StopReason.EOF and complete.coverage[0].eof_observed
    empty = inspect(bam(tmp_path / 'empty.bam', specs=[]))
    assert empty.readiness is m.Readiness.INVALID
    assert 'BAM_EMPTY' in codes(empty) and empty.coverage[0].records_inspected == 0


def test_no_arbitrary_mapping_or_barcode_prevalence_threshold(tmp_path):
    p = bam(tmp_path / 'input.bam', specs=[{}] + [{'flag':4,'tags':[]} for _ in range(99)])
    assert inspect(p).readiness is m.Readiness.READY
    assert observe(p).counts.mapped_primary == 1
    all_unmapped = bam(tmp_path / 'unmapped.bam', specs=[{'flag':4}])
    assert inspect(all_unmapped).readiness is m.Readiness.NEEDS_USER_INPUT


@pytest.mark.parametrize('tags,expected,locator', [
    ([('CB','SECRET_CELL-1','Z')],m.BarcodeSource.BAM_CELL_IDENTIFIER,'CB'),
    ([('CR','ACGT','Z')],m.BarcodeSource.BAM_RAW_SEQUENCE,'CR'),
    ([('CR','ACGT','Z'),('CY','IIII','Z')],m.BarcodeSource.BAM_RAW_SEQUENCE,'CR'),
    ([('CB','SECRET_CELL-1','Z'),('CR','ACGT','Z'),('CY','IIII','Z')],m.BarcodeSource.BAM_CELL_IDENTIFIER,'CB'),
    ([('BC','ACGT','Z'),('QT','IIII','Z')],m.BarcodeSource.UNKNOWN,None),
    ([],m.BarcodeSource.UNKNOWN,None),
])
def test_barcode_selection_and_raw_evidence(tmp_path, tags, expected, locator):
    result = inspect(bam(tmp_path / 'input.bam', tags=tags))
    g = result.groups[0]
    assert g.barcode.source is expected and g.barcode.locator == locator
    assert result.readiness is (m.Readiness.NEEDS_USER_INPUT if locator is None else m.Readiness.READY)
    assert bool(g.barcode.quality_evidence_ids) == any(t[0] == 'CY' for t in tags)
    assert any(p.code is m.PrerequisiteCode.BARCODE_PROCESSING for p in result.prerequisites) == (locator == 'CR')
    assert len(facts(result, 'tag')) == 6
    assert 'corrected' not in result.to_dict()['groups'][0]['barcode']


@pytest.mark.parametrize('tag_name', b.TAGS)
def test_wrong_standard_tag_types_are_invalid(tmp_path, tag_name):
    result = inspect(bam(tmp_path / 'input.bam', tags=[(tag_name, 3, 'i')]))
    assert result.readiness is m.Readiness.INVALID
    assert 'BAM_TAG_TYPE_INVALID' in codes(result)
    observation = observe(Path(result.files[0].path))
    counts = tag(observation, tag_name).all
    assert counts.n == counts.present == counts.wrong_type == 2
    assert counts.usable == 0


@pytest.mark.parametrize('tag_name', b.TAGS)
def test_empty_standard_tags_are_invalid_not_usable(tmp_path, tag_name):
    result = inspect(bam(tmp_path / 'input.bam', tags=[(tag_name,'','Z')]))
    assert result.readiness is m.Readiness.INVALID
    assert 'BAM_TAG_VALUE_INVALID' in codes(result)


def test_duplicate_cb_tags_not_silently_first_matched(tmp_path):
    result = inspect(bam(tmp_path / 'input.bam', tags=[('CB','A-1','Z'),('CB','B-1','Z')]))
    assert result.readiness is m.Readiness.INVALID
    assert 'BAM_TAG_DUPLICATE' in codes(result)
    assert result.groups[0].barcode.source is m.BarcodeSource.UNKNOWN


def test_tag_absence_is_sample_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(b,'MAX_RECORDS',1)
    p = bam(tmp_path / 'input.bam', specs=[{'tags':[]}, {}])
    result = inspect(p)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    cb = next(v for v in facts(result,'tag') if v['tag'] == 'CB')
    assert cb['columns'] == 'n,present,usable,wrong_type,invalid_value,duplicate'
    assert cb['all'] == [1,0,0,0,0,0]
    assert result.coverage[0].scope is m.CoverageScope.SAMPLE


def test_rg_never_used_as_cell_barcode(tmp_path):
    h = header(rgs=[{'ID':'rg1','LB':'lib1','SM':'sample1'}])
    result = inspect(bam(tmp_path / 'input.bam', hd=h, tags=[('RG','rg1','Z')]))
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert result.groups[0].barcode.source is m.BarcodeSource.UNKNOWN
    assert result.groups[0].library_id == 'lib1'
    assert result.groups[0].syntactic_sample_token is None


def test_undeclared_rg_is_a_distinct_invalid_finding(tmp_path):
    result = inspect(bam(tmp_path / 'input.bam', tags=[('CB','A-1','Z'),('RG','unknown','Z')]))
    assert result.readiness is m.Readiness.INVALID and 'BAM_RG_UNDECLARED' in codes(result)


@pytest.mark.parametrize('locator', ['CB','CR'])
def test_multi_library_scope_handling(tmp_path, locator):
    h = header(rgs=[{'ID':'rg1','LB':'lib1','SM':'sample1'}, {'ID':'rg2','LB':'lib2','SM':'sample2'}])
    specs = [{'tags':[(locator,'SECRET_CELL-1','Z'),('RG','rg1','Z')]},
             {'tags':[(locator,'SECRET_CELL-1','Z'),('RG','rg2','Z')]}]
    result = inspect(bam(tmp_path / 'input.bam', hd=h, specs=specs))
    g = result.groups[0]
    assert g.library_id is None
    assert facts(result,'read-groups')[0]['libraries'] == 2
    assert g.barcode.identity_scope is (m.BarcodeIdentityScope.GROUP_LOCAL if locator == 'CB' else m.BarcodeIdentityScope.UNRESOLVED)
    assert result.readiness is (m.Readiness.READY if locator == 'CB' else m.Readiness.NEEDS_USER_INPUT)


def test_multiple_rg_without_library_evidence_does_not_invent_multiple_libraries(tmp_path):
    h = header(rgs=[{'ID':'rg1'}, {'ID':'rg2'}])
    result = inspect(bam(tmp_path / 'input.bam', hd=h, tags=[('CR','ACGT','Z')]))
    assert result.readiness is m.Readiness.READY
    assert result.groups[0].barcode.identity_scope is m.BarcodeIdentityScope.GROUP_LOCAL
    assert result.groups[0].barcode.namespace is None


@pytest.mark.parametrize('producer', [
    {'ID':'pipeline','PN':'cellranger-atac','VN':'2.1.0'},
    {'ID':'cellranger-atac','VN':'2.1.0'},
])
def test_reviewed_cellranger_atac_producer_authority(tmp_path, producer):
    producer = {**producer,'CL':'SECRET_COMMAND /private/checkpoint --token SECRET_TOKEN'}
    result = inspect(bam(tmp_path / 'input.bam', hd=header(pgs=[producer])), assay=None)
    assert result.readiness is m.Readiness.READY
    assert result.groups[0].barcode.producer == 'cellranger-atac'
    assert facts(result,'producer-barcode-definitions')
    assert b'SECRET_COMMAND' not in m.canonical_manifest_bytes(result)
    assert b'SECRET_TOKEN' not in m.canonical_manifest_bytes(result)


@pytest.mark.parametrize('program', [
    {'ID':'bwa','PN':'bwa','VN':'0.7.17'}, {'ID':'bowtie2','PN':'bowtie2'},
    {'ID':'cellranger-atac','PN':'bwa'}, {'ID':'other','CL':'cellranger-atac count'},
    {'ID':'cellranger','PN':'cellranger'}, {'ID':'other','PN':'cellranger-atac-like'},
])
def test_generic_or_ambiguous_producer_does_not_establish_atac(tmp_path, program):
    result = inspect(bam(tmp_path / 'human_atac.bam', hd=header(pgs=[program])), assay=None)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert 'BAM_ASSAY_UNRESOLVED' in codes(result)
    assert result.groups[0].barcode.producer is None
    assert not facts(result,'producer-barcode-definitions')


def test_explicit_tenx_assay_is_not_proof_of_corrected_cb_producer(tmp_path):
    result = inspect(bam(tmp_path / 'input.bam'), assay=b.BamAssay.TENX_ATAC)
    assert result.readiness is m.Readiness.READY
    assert result.groups[0].barcode.producer is None
    assert not facts(result,'producer-barcode-definitions')


def test_sensitive_header_and_read_fields_not_persisted_or_hashed_for_identity(tmp_path):
    h = header(pgs=[{'ID':'bwa','PN':'bwa','VN':'0.7.17','CL':'SECRET_COMMAND'}])
    h['SQ'][0]['UR'] = '/private/SECRET_REFERENCE'
    p = bam(tmp_path / 'a.bam', hd=h)
    first = observe(p)
    payload = m.canonical_manifest_bytes(inspect(p))
    for secret in (b'SECRET_COMMAND', b'SECRET_REFERENCE', b'SECRET_CELL', b'SECRET_READ', b'ACGTACGT', b'IIIIIIII'):
        assert secret not in payload
    h['PG'][0]['CL'] = 'DIFFERENT_COMMAND'
    h['SQ'][0]['UR'] = '/different/reference'
    q = bam(tmp_path / 'b.bam', hd=h, tags=[('CB','DIFFERENT_CELL-99','Z')], specs=[{'name':'different'},{'name':'also-different'}])
    second = observe(q)
    assert first.normalized_observation_sha256 == second.normalized_observation_sha256
    assert first.header.reference_sha256 == second.header.reference_sha256


@pytest.mark.parametrize('payload', [b'not BAM', b'', b'@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:1000\n'])
def test_non_bam_contents_are_invalid(tmp_path, payload):
    backend()
    p = tmp_path / 'input.bam'
    p.write_bytes(payload)
    result = inspect(p)
    assert result.readiness is m.Readiness.INVALID
    assert {'BAM_NOT_BAM','BAM_CONTENT_INVALID'} & codes(result)


@pytest.mark.parametrize('cut', [1, 28, 40, 100])
def test_truncated_bam_is_invalid(tmp_path, cut):
    p = bam(tmp_path / 'input.bam')
    p.write_bytes(p.read_bytes()[:-cut])
    result = inspect(p)
    assert result.readiness is m.Readiness.INVALID
    assert {'BAM_TRUNCATION_OBSERVED','BAM_CONTENT_INVALID'} & codes(result)
    assert result.coverage[0].stop_reason is m.StopReason.ERROR
    assert not result.coverage[0].eof_observed


def test_corrupted_alignment_region_is_invalid(tmp_path):
    p = bam(tmp_path / 'input.bam', specs=[{} for _ in range(20)])
    with backend().AlignmentFile(str(p),'rb') as handle:
        start = handle.tell() >> 16
    data = bytearray(p.read_bytes())
    # Test-only corruption of the data BGZF block; production never decodes bytes.
    data[start + 20] ^= 255
    p.write_bytes(data)
    assert inspect(p).readiness is m.Readiness.INVALID


def test_header_labels_accept_sam_strings_without_identifier_token_guesses(tmp_path):
    p = bam(tmp_path / 'input.bam', hd=header(rgs=[{'ID': 'lane/1', 'LB': 'library one', 'SM': 'sample/one'}]),
            tags=[('CB', 'cell:one-1', 'Z'), ('RG', 'lane/1', 'Z')])
    result = inspect(p)
    assert result.readiness is m.Readiness.READY
    assert result.groups[0].library_id == 'library one'


def test_unprojectable_library_label_preserves_header_summary(tmp_path):
    p = bam(tmp_path / 'input.bam', hd=header(rgs=[{'ID': 'rg1', 'LB': 'x' * 129}]))
    result = inspect(p)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert result.groups[0].library_id is None
    assert observe(p).header.library_count == 1
    assert 'BAM_METADATA_UNREVIEWED' in codes(result)


def test_corruption_beyond_record_sample_is_not_certified(tmp_path, monkeypatch):
    specs = [{} for _ in range(4000)]
    p = bam(tmp_path / 'input.bam', specs=specs)
    with backend().AlignmentFile(str(p),'rb') as handle:
        it = handle.fetch(until_eof=True)
        for _ in range(3000):
            next(it)
        later = handle.tell() >> 16
    data = bytearray(p.read_bytes())
    data[later + 20] ^= 255
    p.write_bytes(data)
    monkeypatch.setattr(b,'MAX_RECORDS',1)
    result = inspect(p)
    assert result.readiness is m.Readiness.READY
    assert result.coverage[0].scope is m.CoverageScope.SAMPLE
    monkeypatch.setattr(b,'MAX_RECORDS',5000)
    assert inspect(p).readiness is m.Readiness.INVALID


def test_reinspection_and_manifest_are_deterministic_readonly(tmp_path):
    p = bam(tmp_path / 'input.bam')
    initial = {v.name:(v.read_bytes(),v.stat().st_mtime_ns) for v in tmp_path.iterdir()}
    result = inspect(p)
    a, c = b.observe_bam_source(result.files[0]), b.observe_bam_source(result.files[0])
    assert a == c
    assert a.normalized_observation_sha256 == result.coverage[0].observed_region_sha256
    assert m.canonical_manifest_bytes(result) == m.canonical_manifest_bytes(inspect(p))
    assert {v.name:(v.read_bytes(),v.stat().st_mtime_ns) for v in tmp_path.iterdir()} == initial
    assert result.coverage[0].decoded_bytes_inspected is None
    assert {x.name for x in result.inspection.backends} == {'raw-bam','pysam','htslib'}


def test_source_drift_before_reinspection_is_operational(tmp_path):
    p = bam(tmp_path / 'input.bam')
    s = source(p)
    bam(p, specs=[{}])
    with pytest.raises(b.BamInspectionError, match='SOURCE_CHANGED'):
        b.observe_bam_source(s)


def test_source_change_during_observation_is_operational(tmp_path, monkeypatch):
    p = bam(tmp_path / 'input.bam')
    original = b._sample
    def mutate(*args):
        result = original(*args)
        os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 1000000))
        return result
    monkeypatch.setattr(b,'_sample',mutate)
    with pytest.raises(b.BamInspectionError, match='SOURCE_CHANGED'):
        inspect(p)


def test_index_change_during_observation_is_detected(tmp_path, monkeypatch):
    p = bam(tmp_path / 'input.bam')
    original = b._sample
    def mutate(*args):
        result = original(*args)
        Path(str(p) + '.bai').write_bytes(b'new sidecar')
        return result
    monkeypatch.setattr(b,'_sample',mutate)
    with pytest.raises(b.BamInspectionError, match='SOURCE_CHANGED'):
        inspect(p)


def test_read_failure_is_sanitized_operational_error(tmp_path, monkeypatch):
    p = bam(tmp_path / 'input.bam')
    def denied(*args, **kwargs):
        raise PermissionError(errno.EACCES,'SECRET_PATH')
    monkeypatch.setattr(b.os,'open',denied)
    with pytest.raises(b.BamInspectionError) as error:
        inspect(p)
    assert error.value.code == 'RAW_BAM_READ_FAILED'
    assert 'SECRET_PATH' not in str(error.value)


def test_valid_group_survives_other_invalid_group(tmp_path):
    a = bam(tmp_path / 'valid.bam')
    bad = tmp_path / 'bad.bam'
    bad.write_bytes(b'not a BAM')
    result = inspect((bad,a))
    assert len(result.files) == len(result.groups) == 2
    assert result.readiness is m.Readiness.INVALID
    assert {v.readiness for v in result.group_readiness} == {m.Readiness.READY,m.Readiness.INVALID}


def test_fixed_contract_and_compact_evidence(tmp_path):
    p = bam(tmp_path / 'input.bam', specs=[{} for _ in range(b.MAX_RECORDS)])
    result = inspect(p)
    assert (b.MAX_CANDIDATE_FILES,b.MAX_SELECTIONS,b.MAX_DIRECTORY_ENTRIES,b.MAX_RECORDS) == (128,128,16384,256)
    assert (b.MAX_HEADER_BYTES,b.MAX_REFERENCES,b.MAX_READ_GROUPS,b.MAX_PROGRAMS,b.MAX_METADATA_VALUES) == (1048576,4096,256,128,4)
    assert all(len(e.value) <= 256 for e in result.evidence)
    assert result.schema_version == 1
    assert m.validate_raw_intake_manifest(result) == result
    cb = next(v for v in facts(result,'tag') if v['tag'] == 'CB')
    assert cb['all'] == [256,256,256,0,0,0]
    assert cb['primary'] == cb['mapped_primary'] == cb['all']
    assert result.coverage[0].scope is m.CoverageScope.SAMPLE  # No read beyond budget for EOF.


def test_no_public_data_tool_export():
    import agent.tools.data as data
    assert data.__all__ == ['ScATACInspection', 'inspect_scATAC', 'RawScATACInspection', 'inspect_raw_scATAC']
