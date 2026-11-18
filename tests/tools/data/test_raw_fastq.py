"""M10.2 synthetic FASTQ acceptance; no biological fixtures or dependencies."""
import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from agent.tools.data import _raw_fastq as f
from agent.tools.data import raw_scatac_manifest as m


def records(ids=("read0", "read1"), seq=b"ACNT", *, repeated=False, ending=b"\n"):
    return b"".join(ending.join((b"@" + name.encode(), seq,
        b"+" + (name.encode() if repeated else b""), b"I" * len(seq))) + ending for name in ids)


def write(path, payload=None, *, compressed=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if payload is None:
        payload = records()
    if compressed is None:
        compressed = path.name.endswith(".gz")
    path.write_bytes(gzip.compress(payload, mtime=0) if compressed else payload)
    return path


def readset(root, roles=("R1", "R2", "R3"), *, sample="sample", number="1", lane="001",
            chunk="001", suffix=".fastq", payload=None):
    stem = f"{sample}_S{number}" + (f"_L{lane}" if lane else "")
    return tuple(write(root / f"{stem}_{role}_{chunk}{suffix}", payload) for role in roles)


def inspect(inputs, *, species="human", assay=f.FastqAssay.TENX_ATAC, declared_layout=None):
    return f.inspect_fastq_inputs(inputs, species=species, assay=assay, declared_layout=declared_layout)


def facts(manifest, kind):
    return [json.loads(e.value) for e in manifest.evidence if e.value.startswith('{')
            and json.loads(e.value).get("fact") == kind]


def codes(manifest):
    return {i.code for i in manifest.issues}


def source(path):
    value = path.stat()
    return m.FileRecord(str(path), m.InputKind.FASTQ, value.st_size, value.st_mtime_ns)


def observe(path):
    return f.observe_fastq_sources((source(path),)).sources[0]


@pytest.mark.parametrize("suffix", f.SUFFIXES)
@pytest.mark.parametrize("directory", [False, True])
def test_plain_gzip_and_fq_selection(tmp_path, suffix, directory):
    paths = readset(tmp_path, suffix=suffix)
    result = inspect(tmp_path if directory else paths)
    assert result.readiness is m.Readiness.READY
    assert len(result.files) == 3
    assert all(c.scope is m.CoverageScope.COMPLETE and c.eof_observed for c in result.coverage)
    assert all(c.records_inspected == 2 for c in result.coverage)
    assert {p['compression'] for p in facts(result, 'parse')} == {
        'gzip' if suffix.endswith('.gz') else 'plain'}
    assert {s.selection for s in result.files} == {
        m.SelectionBasis.DIRECTORY_DISCOVERY if directory else m.SelectionBasis.EXPLICIT}
    assert {s.selection_root for s in result.files} == {str(tmp_path) if directory else None}


def test_explicit_single_file_is_inspected_without_invented_grouping(tmp_path):
    path = write(tmp_path / "unknown.fastq")
    result = inspect(path)
    assert len(result.files) == len(result.groups) == 1
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert result.coverage[0].records_inspected == 2
    assert result.groups[0].grouping_basis is m.GroupingBasis.UNRESOLVED
    assert result.groups[0].syntactic_sample_token is None
    assert not facts(result, "name")


def test_nonrecursive_unrelated_files_and_explicit_leaf_directories(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    readset(a)
    readset(b)
    readset(a / "nested")
    write(a / "unrelated.txt", b"ignore")
    (a / "directory_link").symlink_to(b, target_is_directory=True)
    first = inspect((b, a))
    assert len(first.files) == 6 and len(first.groups) == 2
    assert first.readiness is m.Readiness.READY
    assert m.canonical_manifest_bytes(first) == m.canonical_manifest_bytes(inspect((a, b)))
    assert not any("nested" in s.path for s in first.files)


def test_duplicate_selection_and_explicit_symlink_alias_collapse(tmp_path):
    paths = readset(tmp_path)
    alias = tmp_path / "alias.fq"
    alias.symlink_to(paths[0])
    a = inspect(paths)
    b = inspect((*reversed(paths), paths[0], alias))
    assert a == b
    assert len(b.files) == 3


def test_explicit_selection_wins_over_discovery_deterministically(tmp_path):
    paths = readset(tmp_path)
    a = inspect((tmp_path, *paths))
    b = inspect((*reversed(paths), tmp_path))
    assert a == b == inspect(paths)


def test_hardlink_aliases_rejected_without_role_choice(tmp_path):
    paths = readset(tmp_path)
    alias = tmp_path / "sample_S1_L001_I1_001.fastq"
    os.link(paths[0], alias)
    with pytest.raises(f.FastqInspectionError, match="RAW_FASTQ_ALIAS_CONFLICT"):
        inspect(tmp_path)


def test_directory_candidate_symlink_rejected(tmp_path):
    paths = readset(tmp_path)
    (tmp_path / "alias.fastq").symlink_to(paths[0])
    with pytest.raises(f.FastqInspectionError, match="RAW_FASTQ_DISCOVERY_SYMLINK"):
        inspect(tmp_path)


@pytest.mark.parametrize("selection,code", [
    ("missing.fastq", "SELECTION_INVALID"), ("https://example.org/a.fastq", "SELECTION_INVALID"),
    ("bad\x00path.fastq", "SELECTION_INVALID"),
    ((), "SELECTION_INVALID"), ([], "SELECTION_INVALID"), (42, "SELECTION_INVALID"),
])
def test_selection_errors(tmp_path, selection, code):
    if selection == "missing.fastq":
        selection = tmp_path / selection
    with pytest.raises(f.FastqInspectionError) as error:
        inspect(selection)
    assert error.value.code == "RAW_FASTQ_" + code
    assert str(error.value) == error.value.code


def test_no_candidates_and_explicit_unsupported_suffix(tmp_path):
    path = write(tmp_path / "notes.txt")
    with pytest.raises(f.FastqInspectionError, match="NO_CANDIDATES"):
        inspect(tmp_path)
    with pytest.raises(f.FastqInspectionError, match="SUFFIX_UNSUPPORTED"):
        inspect(path)


@pytest.mark.parametrize("directory", [False, True])
def test_discovery_file_budget(tmp_path, monkeypatch, directory):
    paths = readset(tmp_path)
    monkeypatch.setattr(f, "MAX_CANDIDATE_FILES", 2)
    with pytest.raises(f.FastqInspectionError, match="DISCOVERY_LIMIT"):
        inspect(tmp_path if directory else paths)


def test_directory_entry_budget_includes_unrelated_files(tmp_path, monkeypatch):
    readset(tmp_path)
    write(tmp_path / "notes.txt")
    monkeypatch.setattr(f, "MAX_DIRECTORY_ENTRIES", 3)
    with pytest.raises(f.FastqInspectionError, match="DISCOVERY_LIMIT"):
        inspect(tmp_path)


def test_candidate_special_file_rejected_without_opening(tmp_path):
    path = tmp_path / "input.fastq"
    os.mkfifo(path)
    with pytest.raises(f.FastqInspectionError, match="NOT_REGULAR_FILE"):
        inspect(path)


@pytest.mark.parametrize("lane", [None, "001", "002"])
def test_reviewed_name_grammar_underscore_sample_and_no_invented_lane(tmp_path, lane):
    paths = readset(tmp_path, sample="human_sample_with_under_scores", lane=lane, number="23", chunk="004")
    result = inspect(paths)
    group = result.groups[0]
    assert group.syntactic_sample_token == "human_sample_with_under_scores"
    assert group.lane == lane and group.chunk == "004"
    assert group.library_id is None
    assert {v['sample_number'] for v in facts(result, "name")} == {"23"}
    assert {v['role'] for v in facts(result, "name")} == {"R1", "R2", "R3"}


@pytest.mark.parametrize("difference", ["sample", "number", "lane", "chunk", "root", "no_lane"])
def test_grouping_never_merges_distinct_syntactic_keys(tmp_path, difference):
    a = readset(tmp_path / "a")
    kwargs = {"sample": "different"} if difference == "sample" else {
        "number": "2"} if difference == "number" else {"lane": "002"} if difference == "lane" else {
        "chunk": "002"} if difference == "chunk" else {"lane": None} if difference == "no_lane" else {}
    b = readset(tmp_path / ("b" if difference == "root" else "a"), **kwargs)
    result = inspect((*a, *b))
    assert len(result.groups) == 2
    assert result.readiness is m.Readiness.READY
    assert len({g.barcode_identity_scope_id for g in result.groups}) == 2


@pytest.mark.parametrize("name", ["foo_R1.fastq", "foo_R2.fastq", "foo_S1_L1_R1_001.fastq",
    "foo_S1_L001_R4_001.fastq", "foo_S1_L001_R1_1.fastq", "foo_S1_L001_r1_001.fastq"])
def test_unreviewed_names_do_not_receive_roles(tmp_path, name):
    result = inspect(write(tmp_path / name))
    assert result.groups[0].grouping_basis is m.GroupingBasis.UNRESOLVED
    assert result.groups[0].barcode.source is m.BarcodeSource.UNKNOWN
    assert not facts(result, "name")


@pytest.mark.parametrize("roles,layout,locator", [
    (("R1", "R2", "R3"), f.FastqLayout.A, "R2"),
    (("R1", "R2", "R3", "I1"), f.FastqLayout.A, "R2"),
    (("R1", "I2", "R2"), f.FastqLayout.B, "I2"),
    (("R1", "I2", "R2", "I1"), f.FastqLayout.B, "I2"),
])
@pytest.mark.parametrize("species,target", [("human", "hg38"), ("mouse", "mm10")])
def test_declared_supported_layouts_and_species(tmp_path, roles, layout, locator, species, target):
    result = inspect(readset(tmp_path, roles), species=species)
    group = result.groups[0]
    assert result.readiness is m.Readiness.READY
    assert group.structure is m.StructureState.SUPPORTED
    assert {v['value'] for v in facts(result, "layout")} == {layout.value}
    assert group.barcode.source is m.BarcodeSource.FASTQ_READ
    assert group.barcode.locator == locator
    assert group.barcode.identity_scope is m.BarcodeIdentityScope.GROUP_LOCAL
    assert group.barcode.namespace is group.barcode.producer is None
    assert group.target_genome_assembly.value == target
    assert group.source_genome_assembly == m.MetadataResolution(applicable=False)
    assert not group.harmonization_required and not result.repairs
    assert {p.code for p in result.prerequisites} == {m.PrerequisiteCode.ALIGNMENT, m.PrerequisiteCode.TARGET_REFERENCE}
    assert not result.required_information
    assert "corrected" not in result.to_dict()['groups'][0]['barcode']
    assert any(e.source is m.EvidenceSource.USER_DECLARATION and 'TENX_ATAC' in e.value for e in result.evidence)
    assert all(e.coverage_id is not None for e in result.evidence if e.claim is m.EvidenceClaim.BARCODE_SOURCE)


@pytest.mark.parametrize("roles", [("R1", "R2", "R3"), ("R1", "R2", "I2"), ("R1", "R2")])
@pytest.mark.parametrize("length", [16, 50])
def test_no_assay_inference_from_names_or_lengths(tmp_path, roles, length):
    result = inspect(readset(tmp_path, roles, payload=records(seq=b"N" * length)), assay=None)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert result.groups[0].barcode.source is m.BarcodeSource.UNKNOWN
    assert result.groups[0].structure is m.StructureState.UNRESOLVED
    assert {v['value'] for v in facts(result, "layout")} == {"unresolved"}
    assert result.groups[0].target_genome_assembly.value == "hg38"
    assert "FASTQ_ASSAY_REQUIRED" in codes(result)
    assert not facts(result, 'assay')


@pytest.mark.parametrize("roles", [("R1", "R2"), ("R1", "R3"), ("R2", "R3"), ("R1", "I2"), ("I1",)])
def test_missing_roles_are_blocking_information(tmp_path, roles):
    result = inspect(readset(tmp_path, roles))
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert "FASTQ_REQUIRED_ROLE_MISSING" in codes(result)
    assert result.groups[0].barcode.source is m.BarcodeSource.UNKNOWN


def test_duplicate_role_is_invalid(tmp_path):
    paths = readset(tmp_path)
    duplicate = write(tmp_path / "sample_S1_L001_R1_001.fq.gz")
    result = inspect((*paths, duplicate))
    assert len(result.groups) == 1
    assert result.readiness is m.Readiness.INVALID
    assert "FASTQ_DUPLICATE_ROLE" in codes(result)
    assert result.groups[0].barcode.source is m.BarcodeSource.UNKNOWN


def test_incompatible_mixed_layout_is_unsupported(tmp_path):
    result = inspect(readset(tmp_path, ("R1", "R2", "R3", "I2")))
    assert result.readiness is m.Readiness.UNSUPPORTED
    assert "FASTQ_ROLE_SET_UNSUPPORTED" in codes(result)
    assert result.groups[0].barcode.source is m.BarcodeSource.UNKNOWN


@pytest.mark.parametrize("payload,finding", [
    (b"", f.ParseFinding.EMPTY), (b"@a\n", f.ParseFinding.TRUNCATED),
    (b"@a\nACT\n", f.ParseFinding.TRUNCATED), (b"@a\nACT\n+\n", f.ParseFinding.TRUNCATED),
    (b"@a\nACT\n+\nIII", f.ParseFinding.TRUNCATED),
    (b"a\nACT\n+\nIII\n", f.ParseFinding.HEADER),
    (b"@\nACT\n+\nIII\n", f.ParseFinding.HEADER),
    (b"@ a\nACT\n+\nIII\n", f.ParseFinding.HEADER),
    (b"@\x0b\nACT\n+\nIII\n", f.ParseFinding.HEADER),
    (b"@a\nACT\n-\nIII\n", f.ParseFinding.SEPARATOR),
    (b"@a\nACT\n+\nII\n", f.ParseFinding.LENGTH),
    (b"@a\nA\x00T\n+\nIII\n", f.ParseFinding.CHARACTERS),
    (b"@a\nA T\n+\nIII\n", f.ParseFinding.CHARACTERS),
    (b"@a\nACT\n+\nI\x7fI\n", f.ParseFinding.CHARACTERS),
    (b"@a\x00\nACT\n+\nIII\n", f.ParseFinding.CHARACTERS),
    (b"@a\n\n+\n\n", f.ParseFinding.CHARACTERS),
    (b"@a\nACT\n+other\nIII\n", f.ParseFinding.REPEATED_HEADER),
])
def test_malformed_content_returns_truthful_manifest(tmp_path, payload, finding):
    paths = readset(tmp_path)
    write(paths[0], payload)
    result = inspect(paths)
    observed = observe(paths[0])
    assert observed.finding is finding
    assert result.readiness is m.Readiness.INVALID
    assert "FASTQ_CONTENT_" + finding.name in codes(result)
    assert len(result.files) == 3
    assert result.groups[0].target_genome_assembly.value == 'hg38'


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
@pytest.mark.parametrize("repeated", [False, True])
def test_printable_non_acgt_sequences_and_repeated_header(tmp_path, ending, repeated):
    result = inspect(readset(tmp_path, payload=records(("read1 1:N:0:0",), seq=b"NRY.-", ending=ending, repeated=repeated)))
    assert result.readiness is m.Readiness.READY


def test_variable_length_compact_summary_only(tmp_path):
    payload = records(("a",), b"AC") + records(("b",), b"ACNNRY")
    result = inspect(readset(tmp_path, payload=payload))
    assert {(v['minimum'], v['maximum'], v['constant']) for v in facts(result, 'length')} == {(2, 6, False)}
    text = m.canonical_manifest_bytes(result)
    assert b'ACNNRY' not in text and b'IIIIII' not in text


@pytest.mark.parametrize("compressed", [False, True])
def test_component_limit_is_not_malformed_content(tmp_path, monkeypatch, compressed):
    monkeypatch.setattr(f, 'MAX_LINE_BYTES', 16)
    path = write(tmp_path / 'unknown.fastq', b'@' + b'x' * 100000, compressed=compressed)
    observed = observe(path)
    assert observed.finding is f.ParseFinding.LIMIT
    assert observed.stop is f.ParseStop.LINE_LIMIT
    assert observed.decoded_bytes == 16
    result = inspect(path)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert result.coverage[0].stop_reason is m.StopReason.BUDGET


def test_exact_line_bound_accepts_newline_at_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(f, 'MAX_LINE_BYTES', 8)
    assert observe(write(tmp_path / 'x.fastq', records(('a',), b'A' * 7))).finding is f.ParseFinding.VALID


@pytest.mark.parametrize("suffix", ['.fastq', '.fastq.gz'])
def test_record_budget_sample_and_uninspected_malformed_tail(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(f, 'MAX_RECORDS', 2)
    payload = records() + b'not a valid subsequent record\n'
    result = inspect(readset(tmp_path, suffix=suffix, payload=payload))
    assert result.readiness is m.Readiness.READY
    assert {c.scope for c in result.coverage} == {m.CoverageScope.SAMPLE}
    assert all(not c.eof_observed and c.stop_reason is m.StopReason.BUDGET for c in result.coverage)
    assert {v['stop'] for v in facts(result, 'parse')} == {'record-limit'}
    assert {v['state'] for v in facts(result, 'sync')} == {'synchronized-prefix'}
    assert {c.observed_region_sha256 for c in result.coverage} == {hashlib.sha256(records()).hexdigest()}


@pytest.mark.parametrize("budget", [1, 7, 20, 31, 60])
def test_decoded_byte_budget_is_hard_and_exact(tmp_path, monkeypatch, budget):
    monkeypatch.setattr(f, 'MAX_DECODED_BYTES', budget)
    payload = records(('a', 'b', 'c', 'd'), seq=b'A' * 10)
    path = write(tmp_path / 'x.fastq.gz', payload)
    observed = observe(path)
    assert observed.decoded_bytes == budget
    assert observed.stop is f.ParseStop.BYTE_LIMIT and not observed.eof
    assert observed.observed_region_sha256 == hashlib.sha256(payload[:budget]).hexdigest()
    result = inspect(path)
    assert result.coverage[0].decoded_byte_limit == budget
    assert result.coverage[0].decoded_bytes_inspected == budget


@pytest.mark.parametrize("defect", ['trailer', 'truncated', 'header', 'deflate'])
def test_gzip_corruption_observed_is_invalid(tmp_path, defect):
    payload = gzip.compress(records(), mtime=0)
    if defect == 'trailer':
        payload = payload[:-8] + bytes([payload[-8] ^ 1]) + payload[-7:]
    elif defect == 'truncated':
        payload = payload[:-5]
    elif defect == 'header':
        payload = b'\x1f\x8bgarbage'
    else:
        payload = payload[:10] + b'\xff' + payload[11:]
    path = write(tmp_path / 'x.fastq.gz', payload, compressed=False)
    result = inspect(path)
    assert result.readiness is m.Readiness.INVALID
    assert 'FASTQ_CONTENT_GZIP' in codes(result)
    assert result.coverage[0].stop_reason is m.StopReason.ERROR
    assert not result.coverage[0].eof_observed


def test_gzip_corrupt_trailer_after_record_budget_is_not_inspected(tmp_path, monkeypatch):
    monkeypatch.setattr(f, 'MAX_RECORDS', 2)
    data = gzip.compress(records(), mtime=0)
    data = data[:-8] + bytes([data[-8] ^ 1]) + data[-7:]
    path = write(tmp_path / 'x.fastq.gz', data, compressed=False)
    observed = observe(path)
    assert observed.finding is f.ParseFinding.VALID
    assert observed.stop is f.ParseStop.RECORD_LIMIT
    assert not observed.eof
    monkeypatch.setattr(f, 'MAX_RECORDS', 3)
    assert observe(path).finding is f.ParseFinding.GZIP


def test_gzip_later_corrupt_member_not_claimed_inside_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(f, 'MAX_RECORDS', 2)
    path = write(tmp_path / 'x.fastq.gz', gzip.compress(records(), mtime=0) + b'\x1f\x8bBROKEN', compressed=False)
    assert observe(path).stop is f.ParseStop.RECORD_LIMIT
    monkeypatch.setattr(f, 'MAX_RECORDS', 3)
    assert observe(path).finding is f.ParseFinding.GZIP


def test_gzip_concatenated_valid_members(tmp_path):
    path = write(tmp_path / 'x.fastq.gz', gzip.compress(records(('a',))) + gzip.compress(records(('b',))), compressed=False)
    observation = observe(path)
    assert observation.eof and observation.records == 2
    assert observation.finding is f.ParseFinding.VALID


def test_pathological_gzip_header_and_empty_members_have_physical_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(f, 'MAX_ENCODED_BYTES', 64)
    # FNAME with no terminating NUL; gzip's own header loop must be bounded.
    name_header = b'\x1f\x8b\x08\x08' + b'\0' * 6 + b'a' * 1000
    for payload in (name_header, gzip.compress(b'', mtime=0) * 100):
        path = write(tmp_path / 'x.fastq.gz', payload, compressed=False)
        value = observe(path)
        assert value.stop is f.ParseStop.ENCODED_LIMIT
        assert value.finding is f.ParseFinding.LIMIT and not value.eof
        assert inspect(path).readiness is m.Readiness.NEEDS_USER_INPUT


@pytest.mark.parametrize("suffix,compressed", [('.fastq', True), ('.fastq.gz', False)])
def test_suffix_content_mismatch_is_advisory_with_actual_parser_authority(tmp_path, suffix, compressed):
    paths = readset(tmp_path, suffix=suffix)
    for path in paths:
        write(path, compressed=compressed)
    result = inspect(paths)
    assert result.readiness is m.Readiness.READY
    assert 'FASTQ_COMPRESSION_SUFFIX_MISMATCH' in codes(result)
    assert all(i.effect is m.ReadinessEffect.ADVISORY for i in result.issues)
    assert not result.required_information and not result.repairs
    assert result.groups[0].barcode.locator == 'R2'
    assert {v['compression'] for v in facts(result, 'parse')} == {'gzip' if compressed else 'plain'}
    assert all(not v['suffix_matches'] for v in facts(result, 'parse'))
    assert all(c.eof_observed for c in result.coverage)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    assert m.canonical_manifest_bytes(result) == m.canonical_manifest_bytes(inspect(tuple(reversed(paths))))
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths} == before


def test_narrow_header_normalization_and_no_read_ids_persisted(tmp_path):
    paths = readset(tmp_path)
    for path, suffix in zip(paths, ('/1 1:N:0:0', '/2 2:N:0:0', ' 3:N:0:0')):
        write(path, records(('secret_identifier' + suffix,)))
    result = inspect(paths)
    assert result.readiness is m.Readiness.READY
    assert {v['state'] for v in facts(result, 'sync')} == {'synchronized-complete'}
    assert b'secret_identifier' not in m.canonical_manifest_bytes(result)
    assert 'secret_identifier' not in repr(f.observe_fastq_sources(result.files))


@pytest.mark.parametrize("changed", ['different', 'read0/3', 'READ0', 'read0:1'])
def test_read_id_mismatch_has_no_fuzzy_repair(tmp_path, changed):
    paths = readset(tmp_path)
    write(paths[1], records((changed, 'read1')))
    result = inspect(paths)
    assert result.readiness is m.Readiness.INVALID
    assert 'FASTQ_NAME_MISMATCH' in codes(result)
    assert {v['compared'] for v in facts(result, 'sync')} == {1}
    assert {v['synchronized'] for v in facts(result, 'sync')} == {0}
    assert result.groups[0].target_genome_assembly.value == 'hg38'


@pytest.mark.parametrize("budget", [None, 2])
def test_observed_early_eof_and_unequal_counts(tmp_path, monkeypatch, budget):
    if budget:
        monkeypatch.setattr(f, 'MAX_RECORDS', budget)
    paths = readset(tmp_path, payload=records(('a', 'b', 'c')))
    write(paths[1], records(('a',)))
    result = inspect(paths)
    assert result.readiness is m.Readiness.INVALID
    assert 'FASTQ_COUNT_MISMATCH' in codes(result)
    assert {v['synchronized'] for v in facts(result, 'sync')} == {1}


def test_shorter_byte_limited_sample_is_not_premature_eof(tmp_path, monkeypatch):
    monkeypatch.setattr(f, 'MAX_DECODED_BYTES', 60)
    paths = readset(tmp_path, payload=records(('a', 'b', 'c', 'd'), seq=b'A'))
    write(paths[1], records(('a', 'b', 'c', 'd'), seq=b'A' * 10))
    result = inspect(paths)
    assert result.readiness is m.Readiness.READY
    assert 'FASTQ_COUNT_MISMATCH' not in codes(result)
    assert {v['state'] for v in facts(result, 'sync')} == {'synchronized-prefix'}


def test_unknown_species_and_no_filename_inference(tmp_path):
    result = inspect(readset(tmp_path, sample='human_hg38'), species=None)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    group = result.groups[0]
    assert group.species.state is m.ResolutionState.UNKNOWN
    assert group.target_genome_assembly == m.TargetAssembly(m.TargetState.UNRESOLVED, None)
    assert group.source_genome_assembly.state is m.ResolutionState.NOT_APPLICABLE
    assert {r.code for r in result.required_information} == {m.InformationCode.SPECIES}
    assert group.barcode.source is m.BarcodeSource.FASTQ_READ


def test_explicit_unsupported_species_uses_m10_1_gate(tmp_path):
    result = inspect(readset(tmp_path), species='rat')
    assert result.readiness is m.Readiness.UNSUPPORTED
    assert result.groups[0].target_genome_assembly.state is m.TargetState.UNSUPPORTED


@pytest.mark.parametrize("declaration", [True, 'TENX_ATAC', '10x', 1])
def test_assay_api_is_closed(tmp_path, declaration):
    with pytest.raises(f.FastqInspectionError, match='DECLARATION_INVALID'):
        inspect(readset(tmp_path), assay=declaration)


@pytest.mark.parametrize("species", ['Human', '', ' human', 1, True, 'hg38\n'])
def test_species_declaration_must_be_explicit_normalized_token(tmp_path, species):
    with pytest.raises(f.FastqInspectionError, match='DECLARATION_INVALID'):
        inspect(readset(tmp_path), species=species)


def test_readonly_deterministic_manifest_and_reinspection(tmp_path):
    paths = readset(tmp_path, suffix='.fastq.gz')
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    initial_names = sorted(tmp_path.iterdir())
    a, b = inspect(paths), inspect(tuple(reversed(paths)))
    assert m.canonical_manifest_bytes(a) == m.canonical_manifest_bytes(b)
    observations = f.observe_fastq_sources(a.files)
    assert observations == f.observe_fastq_sources(tuple(reversed(a.files)))
    by_file = {c.file_ids[0]: c for c in a.coverage}
    for observation in observations.sources:
        c = by_file[observation.file.id]
        assert observation.observed_region_sha256 == c.observed_region_sha256
        assert observation.decoded_bytes == c.decoded_bytes_inspected
        assert observation.records == c.records_inspected
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths} == before
    assert sorted(tmp_path.iterdir()) == initial_names
    assert a.inspection.backends == (m.BackendIdentity('raw-fastq', 'raw-fastq.v1'),)


def test_reinspection_detects_recorded_snapshot_drift(tmp_path):
    result = inspect(readset(tmp_path))
    path = Path(result.files[0].path)
    write(path, records(('changed',)))
    with pytest.raises(f.FastqInspectionError, match='SOURCE_CHANGED'):
        f.observe_fastq_sources(result.files)


def test_reinspection_reobserves_same_size_mtime_content_without_trusting_metadata(tmp_path):
    result = inspect(readset(tmp_path))
    before = f.observe_fastq_sources(result.files)
    path = Path(result.files[0].path)
    old_stat = path.stat()
    payload = path.read_bytes().replace(b'read0', b'other')
    path.write_bytes(payload)
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    after = f.observe_fastq_sources(result.files)
    assert before != after
    assert after.synchronization is f.SyncState.NAME_MISMATCH
    old = {s.file.path: s.observed_region_sha256 for s in before.sources}
    new = {s.file.path: s.observed_region_sha256 for s in after.sources}
    assert old[str(path)] != new[str(path)]


def test_source_change_during_parse_is_operational_failure(tmp_path, monkeypatch):
    paths = readset(tmp_path)
    original = f._parse
    def mutate(stream, compressed):
        result = original(stream, compressed)
        paths[0].write_bytes(records(('changed',)))
        return result
    monkeypatch.setattr(f, '_parse', mutate)
    with pytest.raises(f.FastqInspectionError, match='SOURCE_CHANGED'):
        inspect(paths)


def test_read_failure_is_sanitized_operational_error(tmp_path, monkeypatch):
    paths = readset(tmp_path)
    def fail(*args, **kwargs):
        raise PermissionError('private-path-and-sensitive-details')
    monkeypatch.setattr(f.os, 'open', fail)
    with pytest.raises(f.FastqInspectionError) as error:
        inspect(paths)
    assert error.value.code == 'RAW_FASTQ_READ_FAILED'
    assert 'private' not in str(error.value)


def test_one_invalid_group_does_not_discard_other_findings(tmp_path):
    valid = readset(tmp_path / 'valid')
    invalid = readset(tmp_path / 'invalid')
    write(invalid[0], b'broken\n')
    result = inspect((*valid, *invalid))
    assert len(result.files) == 6 and len(result.groups) == 2
    assert result.readiness is m.Readiness.INVALID
    assert {g.readiness for g in result.group_readiness} == {m.Readiness.INVALID, m.Readiness.READY}
    by_id = {g.id: g for g in result.groups}
    ready = next(by_id[g.group_id] for g in result.group_readiness if g.readiness is m.Readiness.READY)
    assert ready.barcode.locator == 'R2' and ready.target_genome_assembly.value == 'hg38'


def test_canonical_evidence_vocabulary_and_shapes_frozen(tmp_path):
    result = inspect(readset(tmp_path))
    values = {e.value for e in result.evidence}
    assert '{"contract":"raw-fastq.v1","fact":"assay","value":"TENX_ATAC"}' in values
    assert '{"chunk":"001","contract":"raw-fastq.v1","fact":"name","lane":"001","role":"R2","sample_number":"1"}' in values
    assert '{"contract":"raw-fastq.v1","fact":"layout","value":"tenx-atac-r1-r2-r3.v1"}' in values
    assert '{"compression":"plain","contract":"raw-fastq.v1","fact":"parse","finding":"valid-observed-records","stop":"eof","suffix_matches":true}' in values
    assert '{"compared":2,"contract":"raw-fastq.v1","fact":"sync","state":"synchronized-complete","synchronized":2}' in values
    assert '{"constant":true,"contract":"raw-fastq.v1","fact":"length","maximum":4,"minimum":4}' in values
    assert '{"contract":"raw-fastq.v1","encoded_bytes":4194304,"fact":"limits","line_bytes":16384}' in values
    assert {v.value for v in f.FastqLayout} == {'tenx-atac-r1-r2-r3.v1', 'tenx-atac-r1-i2-r2.v1', 'unresolved', 'unsupported-role-set'}
    assert all(len(e.value) <= 256 for e in result.evidence)
    assert m.validate_raw_intake_manifest(result) == result
    assert (f.MAX_CANDIDATE_FILES, f.MAX_RECORDS, f.MAX_DECODED_BYTES, f.MAX_LINE_BYTES,
            f.MAX_ENCODED_BYTES, f.MAX_SELECTIONS, f.MAX_DIRECTORY_ENTRIES) == (
        128, 256, 2097152, 16384, 4194304, 128, 16384)


def test_no_existing_integration_exports_added():
    import agent.tools.data as data
    assert data.__all__ == ['ScATACInspection', 'inspect_scATAC', 'RawScATACInspection', 'inspect_raw_scATAC']


@pytest.mark.parametrize('roles,meanings', [
    (('R1', 'R2', 'R3', 'I1'), {'R1': 'genomic-read-1', 'R2': 'raw-cell-barcode-i5',
                               'R3': 'genomic-read-2', 'I1': 'sample-index'}),
    (('R1', 'R2', 'I2', 'I1'), {'R1': 'genomic-read-1', 'R2': 'genomic-read-2',
                               'I2': 'raw-cell-barcode-i5', 'I1': 'sample-index'}),
])
def test_read_meanings_are_reviewed_and_require_assay_authority(tmp_path, roles, meanings):
    paths = readset(tmp_path, roles)
    declared = inspect(paths)
    assert {v['role']: v['meaning'] for v in facts(declared, 'read-meaning')} == meanings
    assert not facts(inspect(paths, assay=None), 'read-meaning')
    assert '{"contract":"raw-fastq.v1","fact":"read-meaning","meaning":"sample-index","role":"I1"}' in {
        e.value for e in declared.evidence}


def test_complete_malformed_record_counted_but_not_synchronized(tmp_path):
    paths = readset(tmp_path)
    write(paths[0], records(('read0',)) + b'@read1\nACT\n+\nII\n')
    result = inspect(paths)
    c = next(c for c in result.coverage if c.file_ids == (m.file_identity(str(paths[0])),))
    assert c.records_inspected == 2 and c.stop_reason is m.StopReason.ERROR
    assert {v['compared'] for v in facts(result, 'sync')} == {1}
    assert {v['synchronized'] for v in facts(result, 'sync')} == {1}
    assert result.readiness is m.Readiness.INVALID


def test_partial_final_record_not_counted(tmp_path):
    payload = records(('read0',)) + b'@read1\nACT\n'
    path = write(tmp_path / 'x.fastq', payload)
    value = observe(path)
    assert value.records == 1 and value.finding is f.ParseFinding.TRUNCATED
    assert value.decoded_bytes == len(payload)
    assert value.observed_region_sha256 == hashlib.sha256(payload).hexdigest()


def test_empty_normalized_header_is_invalid(tmp_path):
    value = observe(write(tmp_path / 'x.fastq', records(('/1',))))
    assert value.finding is f.ParseFinding.HEADER


def test_no_unbounded_parser_read_requests(tmp_path, monkeypatch):
    import io
    class Guarded(io.BytesIO):
        def read(self, size=-1):
            assert 0 <= size <= f.MAX_LINE_BYTES
            return super().read(size)
        def readline(self, size=-1):
            raise AssertionError('unreviewed line reader')
    monkeypatch.setattr(f, 'MAX_LINE_BYTES', 32)
    value = f._parse(Guarded(b'@' + b'x' * 1000000), False)
    assert value[1] is f.ParseStop.LINE_LIMIT and value[3] == 32


def test_candidate_limit_fits_unchanged_manifest_size_and_collection_contract(tmp_path):
    for number in range(f.MAX_CANDIDATE_FILES):
        write(tmp_path / f'unknown_{number:03}.fastq')
    result = inspect(tmp_path)
    assert len(result.files) == len(result.groups) == f.MAX_CANDIDATE_FILES
    assert len(result.evidence) <= m.MAX_COLLECTION_ITEMS
    assert len(m.canonical_manifest_bytes(result)) <= m.MAX_MANIFEST_BYTES


def test_longest_reviewed_sample_token_remains_representable(tmp_path):
    result = inspect(readset(tmp_path, sample='x' * 128, number='9' * 9))
    assert result.readiness is m.Readiness.READY
    assert len(result.groups[0].syntactic_sample_token) == 128
    assert all(len(e.value) <= 256 for e in result.evidence)


def test_unknown_similar_filenames_remain_independent_singletons(tmp_path):
    paths = (write(tmp_path / 'sample_R1.fastq'), write(tmp_path / 'sample_R2.fastq'))
    result = inspect(paths)
    assert len(result.groups) == 2
    assert all(len(g.file_ids) == 1 and g.grouping_basis is m.GroupingBasis.UNRESOLVED for g in result.groups)
    assert not facts(result, 'name')


def test_directory_enumeration_order_does_not_affect_inventory_or_manifest(tmp_path, monkeypatch):
    readset(tmp_path, sample='b')
    readset(tmp_path, sample='a')
    expected = inspect(tmp_path)
    original = os.scandir
    class Reversed:
        def __init__(self, path):
            with original(path) as entries:
                self.entries = list(entries)[::-1]
        def __enter__(self):
            return iter(self.entries)
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(f.os, 'scandir', Reversed)
    assert m.canonical_manifest_bytes(inspect(tmp_path)) == m.canonical_manifest_bytes(expected)
    assert [s.file.path for s in f._discover(tmp_path)] == sorted(s.path for s in expected.files)


def test_source_replaced_with_same_size_mtime_during_inspection_detected(tmp_path, monkeypatch):
    paths = readset(tmp_path)
    original = f._parse
    def replace_source(stream, compressed):
        value = original(stream, compressed)
        path = paths[0]
        metadata = path.stat()
        staging = tmp_path / 'replacement'
        staging.write_bytes(path.read_bytes())
        os.utime(staging, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        os.replace(staging, path)
        return value
    monkeypatch.setattr(f, '_parse', replace_source)
    with pytest.raises(f.FastqInspectionError, match='SOURCE_CHANGED'):
        inspect(paths)


def test_earlier_group_source_drift_is_checked_at_final_boundary(tmp_path, monkeypatch):
    first = readset(tmp_path / 'a')
    second = readset(tmp_path / 'b')
    original = f._read_set
    def change_earlier(selected):
        result = original(selected)
        if selected[0].file.path.startswith(str(tmp_path / 'b')):
            write(first[0], records(('different',)))
        return result
    monkeypatch.setattr(f, '_read_set', change_earlier)
    with pytest.raises(f.FastqInspectionError, match='SOURCE_CHANGED'):
        inspect((*first, *second))


@pytest.mark.parametrize('sources', [[], (), (None,), ('not-a-record',)])
def test_reinspection_requires_explicit_source_records(sources):
    with pytest.raises(f.FastqInspectionError, match='SELECTION_INVALID'):
        f.observe_fastq_sources(sources)


def test_reinspection_rejects_duplicate_records(tmp_path):
    record = source(write(tmp_path / 'x.fastq'))
    with pytest.raises(f.FastqInspectionError, match='ALIAS_CONFLICT'):
        f.observe_fastq_sources((record, record))


def test_discovered_unrepresentable_path_has_stable_operational_error(tmp_path):
    write(tmp_path / 'bad\nname.fastq')
    with pytest.raises(f.FastqInspectionError, match='SELECTION_INVALID'):
        inspect(tmp_path)


@pytest.mark.parametrize('declared_layout,roles,missing_role', [
    (f.FastqLayout.A, ('R1', 'R2'), 'R3'),
    (f.FastqLayout.A, ('R1', 'R3'), 'R2'),
    (f.FastqLayout.B, ('R1', 'R2'), 'I2'),
])
def test_explicit_layout_missing_role_has_exact_artifact_requirement(tmp_path, declared_layout, roles, missing_role):
    paths = readset(tmp_path, roles)
    result = inspect(paths, declared_layout=declared_layout)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert 'FASTQ_REQUIRED_ROLE_MISSING' in codes(result)
    assert 'FASTQ_LAYOUT_UNRESOLVED' not in codes(result)
    assert 'FASTQ_DUPLICATE_ROLE' not in codes(result)
    assert result.groups[0].barcode.source is m.BarcodeSource.UNKNOWN
    assert result.groups[0].target_genome_assembly.value == 'hg38'
    assert result.groups[0].source_genome_assembly.state is m.ResolutionState.NOT_APPLICABLE
    expected = {'contract': 'raw-fastq.v1', 'fact': 'missing-role',
                'candidate_layout': declared_layout.value, 'role': missing_role, 'basis': 'declared-layout'}
    assert facts(result, 'missing-role') == [expected] * len(roles)
    issue = next(i for i in result.issues if i.code == 'FASTQ_REQUIRED_ROLE_MISSING')
    evidence = {e.id: e for e in result.evidence}
    assert len(issue.evidence_ids) == len(roles)
    assert all(json.loads(evidence[e].value) == expected for e in issue.evidence_ids)
    assert all(evidence[e].coverage_id is not None for e in issue.evidence_ids)
    declarations = [e for e in result.evidence if e.value.startswith('{')
                    and json.loads(e.value)['fact'] == 'layout-declaration']
    assert len(declarations) == 1 and declarations[0].source is m.EvidenceSource.USER_DECLARATION
    assert json.loads(declarations[0].value)['value'] == declared_layout.value
    assert m.canonical_manifest_bytes(result) == m.canonical_manifest_bytes(
        inspect(tuple(reversed(paths)), declared_layout=declared_layout))
    assert len(result.files) == len(roles)  # No synthesized missing source.


def test_missing_role_canonical_encoding_is_frozen(tmp_path):
    result = inspect(readset(tmp_path, ('R1', 'R2')), declared_layout=f.FastqLayout.A)
    assert ('{"basis":"declared-layout","candidate_layout":"tenx-atac-r1-r2-r3.v1",'
            '"contract":"raw-fastq.v1","fact":"missing-role","role":"R3"}') in {
                e.value for e in result.evidence}
    assert all(len(e.value) <= 256 for e in result.evidence)


@pytest.mark.parametrize('roles,layout,missing_role', [
    (('R1', 'R3'), f.FastqLayout.A, 'R2'),
    (('R2', 'R3'), f.FastqLayout.A, 'R1'),
    (('R1', 'I2'), f.FastqLayout.B, 'R2'),
])
def test_unique_incomplete_candidate_identifies_absent_role(tmp_path, roles, layout, missing_role):
    result = inspect(readset(tmp_path, roles))
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert 'FASTQ_REQUIRED_ROLE_MISSING' in codes(result)
    assert 'FASTQ_LAYOUT_UNRESOLVED' not in codes(result)
    assert {(v['candidate_layout'], v['role'], v['basis']) for v in facts(result, 'missing-role')} == {
        (layout.value, missing_role, 'unique-candidate')}


def test_r1_r2_without_layout_declaration_preserves_both_missing_role_alternatives(tmp_path):
    result = inspect(readset(tmp_path, ('R1', 'R2')))
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert codes(result) >= {'FASTQ_REQUIRED_ROLE_MISSING', 'FASTQ_LAYOUT_UNRESOLVED'}
    assert {(v['candidate_layout'], v['role'], v['basis']) for v in facts(result, 'missing-role')} == {
        (f.FastqLayout.A.value, 'R3', 'alternative-candidate'),
        (f.FastqLayout.B.value, 'I2', 'alternative-candidate')}
    assert result.groups[0].barcode.source is m.BarcodeSource.UNKNOWN


@pytest.mark.parametrize('declared_layout', [None, f.FastqLayout.A, f.FastqLayout.B])
def test_no_assay_authority_never_asserts_precise_missing_role(tmp_path, declared_layout):
    result = inspect(readset(tmp_path, ('R1', 'R2')), assay=None, declared_layout=declared_layout)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert 'FASTQ_ASSAY_REQUIRED' in codes(result)
    assert 'FASTQ_REQUIRED_ROLE_MISSING' not in codes(result)
    assert not facts(result, 'missing-role')
    assert result.groups[0].barcode.source is m.BarcodeSource.UNKNOWN


@pytest.mark.parametrize('layout,roles,locator', [
    (f.FastqLayout.A, ('R1', 'R2', 'R3'), 'R2'),
    (f.FastqLayout.B, ('R1', 'R2', 'I2'), 'I2'),
])
def test_explicit_layout_without_optional_i1_still_ready(tmp_path, layout, roles, locator):
    result = inspect(readset(tmp_path, roles), declared_layout=layout)
    assert result.readiness is m.Readiness.READY
    assert result.groups[0].barcode.locator == locator
    assert not facts(result, 'missing-role')
    assert 'FASTQ_REQUIRED_ROLE_MISSING' not in codes(result)


@pytest.mark.parametrize('layout', [f.FastqLayout.A, f.FastqLayout.B])
def test_duplicate_required_role_remains_distinct_from_missing_role(tmp_path, layout):
    roles = ('R1', 'R2', 'R3') if layout is f.FastqLayout.A else ('R1', 'R2', 'I2')
    paths = readset(tmp_path, roles)
    duplicate = write(tmp_path / 'sample_S1_L001_R1_001.fq')
    result = inspect((*paths, duplicate), declared_layout=layout)
    assert result.readiness is m.Readiness.INVALID
    assert 'FASTQ_DUPLICATE_ROLE' in codes(result)
    assert 'FASTQ_REQUIRED_ROLE_MISSING' not in codes(result)
    assert not facts(result, 'missing-role')


@pytest.mark.parametrize('declared_layout', [None, f.FastqLayout.A, f.FastqLayout.B])
def test_incompatible_mixture_never_reported_as_merely_missing_input(tmp_path, declared_layout):
    result = inspect(readset(tmp_path, ('R1', 'R2', 'R3', 'I2')), declared_layout=declared_layout)
    assert result.readiness is m.Readiness.UNSUPPORTED
    assert 'FASTQ_ROLE_SET_UNSUPPORTED' in codes(result)
    assert 'FASTQ_REQUIRED_ROLE_MISSING' not in codes(result)
    assert not facts(result, 'missing-role')


def test_declared_layout_conflict_is_not_silently_overridden(tmp_path):
    result = inspect(readset(tmp_path, ('R1', 'R2', 'I2')), declared_layout=f.FastqLayout.A)
    assert result.readiness is m.Readiness.UNSUPPORTED
    assert 'FASTQ_ROLE_SET_UNSUPPORTED' in codes(result)
    assert result.groups[0].barcode.source is m.BarcodeSource.UNKNOWN
    assert not facts(result, 'missing-role')


def test_unresolved_grouping_does_not_prove_missing_roles_even_with_layout_declaration(tmp_path):
    result = inspect(write(tmp_path / 'unknown.fastq'), declared_layout=f.FastqLayout.A)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert 'FASTQ_LAYOUT_UNRESOLVED' in codes(result)
    assert 'FASTQ_REQUIRED_ROLE_MISSING' not in codes(result)
    assert not facts(result, 'missing-role')


@pytest.mark.parametrize('layout', [f.FastqLayout.UNRESOLVED, f.FastqLayout.UNSUPPORTED,
                                    'tenx-atac-r1-r2-r3.v1', True])
def test_layout_declaration_is_closed_to_explicit_supported_identities(tmp_path, layout):
    with pytest.raises(f.FastqInspectionError, match='DECLARATION_INVALID'):
        inspect(readset(tmp_path), declared_layout=layout)


@pytest.mark.parametrize('suffix,compressed', [('.fastq.gz', False), ('.fastq', True)])
def test_misleading_suffix_does_not_weaken_fastq_corruption_findings(tmp_path, suffix, compressed):
    paths = readset(tmp_path, suffix=suffix)
    for path in paths:
        write(path, b'@read\nACT\n+\nII\n', compressed=compressed)
    result = inspect(paths)
    assert result.readiness is m.Readiness.INVALID
    assert 'FASTQ_CONTENT_LENGTH' in codes(result)
    mismatch = [i for i in result.issues if i.code == 'FASTQ_COMPRESSION_SUFFIX_MISMATCH']
    assert len(mismatch) == len(paths)
    assert all(i.effect is m.ReadinessEffect.ADVISORY for i in mismatch)


def test_gzip_magic_with_corruption_and_plain_suffix_remains_invalid(tmp_path):
    paths = readset(tmp_path)
    payload = gzip.compress(records(), mtime=0)
    corrupt = payload[:-8] + bytes([payload[-8] ^ 1]) + payload[-7:]
    write(paths[0], corrupt, compressed=False)
    result = inspect(paths)
    assert result.readiness is m.Readiness.INVALID
    assert 'FASTQ_CONTENT_GZIP' in codes(result)
    assert 'FASTQ_COMPRESSION_SUFFIX_MISMATCH' in codes(result)
    assert all(i.effect is m.ReadinessEffect.ADVISORY for i in result.issues
               if i.code == 'FASTQ_COMPRESSION_SUFFIX_MISMATCH')


def test_suffix_advisory_does_not_remove_real_information_requirements(tmp_path):
    paths = readset(tmp_path, ('R1', 'R2'))
    for path in paths:
        write(path, compressed=True)
    result = inspect(paths, declared_layout=f.FastqLayout.A)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert codes(result) >= {'FASTQ_REQUIRED_ROLE_MISSING', 'FASTQ_COMPRESSION_SUFFIX_MISMATCH'}
    assert {v['role'] for v in facts(result, 'missing-role')} == {'R3'}
