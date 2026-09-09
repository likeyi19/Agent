"""Tiny resource tests; no model, scientific tool, or external runtime calls."""

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import pytest

from agent.tools.data import scatac_reference as ref
from agent.tools.data._ordered_identity import ordered_identity_sha256


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def sources(tmp_path):
    # BED order intentionally differs from both FAI and coordinate order.
    fasta = tmp_path / "unknown_Lu2025.fa"
    fasta.write_bytes(b">chr1\nACGTACGTACGT\n>chr2\nACGTACGT\n")
    fai = tmp_path / "genome.fa.fai"
    fai.write_bytes(b"chr1\t12\t6\t12\t13\nchr2\t8\t25\t8\t9\n")
    bed = tmp_path / "ccre.bed"
    bed.write_bytes(b"chr2\t1\t5\textra\nchr1\t8\t12\nchr1\t0\t4\n")
    return dict(species="human", target_assembly="hg38", fasta_path=fasta,
                fai_path=fai, ccre_bed_path=bed)


@pytest.fixture
def bundle(sources):
    return ref.build_scatac_reference_bundle(**sources)


def test_exact_contract_and_independent_digests(sources, bundle):
    assert bundle.artifact_type == "agent.scatac-reference-bundle"
    assert bundle.schema_version == 1
    assert bundle.contract_version == "scatac-reference-bundle.v1"
    assert bundle.ordered_identity_algorithm == "sha256-utf8-lf.v1"
    assert bundle.assembly_binding == "explicit_declaration"
    assert bundle.ccre.feature_count == 3  # Every row, with no cell/use filtering.
    assert bundle.genome.contig_count == 2
    assert bundle.ccre.ordered_feature_sha256 == sha(b"chr2:1-5\nchr1:8-12\nchr1:0-4\n")
    assert bundle.genome.ordered_contig_sha256 == sha(b"chr1\t12\nchr2\t8\n")
    assert bundle.ccre.bed.sha256 == sha(sources["ccre_bed_path"].read_bytes())
    assert bundle.genome.fasta.sha256 == sha(sources["fasta_path"].read_bytes())
    assert bundle.genome.fai.sha256 == sha(sources["fai_path"].read_bytes())
    assert ref.validate_scatac_reference_bundle(bundle.to_dict()) == bundle
    assert ref.build_scatac_reference_bundle(**sources) == bundle
    with pytest.raises(FrozenInstanceError):
        bundle.species = "mouse"
    with pytest.raises(FrozenInstanceError):
        bundle.ccre.feature_count = 1


def test_ordered_helper_exact_unicode_no_normalization():
    names = ["β", "chr1:0-2", "e\u0301"]
    assert ordered_identity_sha256(iter(names)) == sha("β\nchr1:0-2\ne\u0301\n".encode())
    assert ordered_identity_sha256(names) != ordered_identity_sha256(names[::-1])
    assert ordered_identity_sha256(["é"]) != ordered_identity_sha256(["e\u0301"])


@pytest.mark.parametrize("names", [[], [""], ["x", "x"], ["x\ny"], ["x\ry"],
                                        ["\ud800"], [1], [None], "abc", b"abc"])
def test_ordered_helper_rejects_ambiguous_records(names):
    with pytest.raises(ValueError):
        ordered_identity_sha256(names)


@pytest.mark.parametrize("species,assembly", [("human", "hg38"), ("mouse", "mm10")])
def test_shared_species_contract(sources, species, assembly):
    sources.update(species=species, target_assembly=assembly)
    assert ref.build_scatac_reference_bundle(**sources).species == species


@pytest.mark.parametrize("species,assembly", [("human", "mm10"), ("mouse", "hg38"),
    ("rat", "rn6"), ("Human", "hg38"), ("human", "GRCh38"), (None, "hg38")])
def test_unsupported_species_before_source_io(sources, monkeypatch, species, assembly):
    sources.update(species=species, target_assembly=assembly)
    monkeypatch.setattr(ref, "_source_path", lambda *_: pytest.fail("unexpected source IO"))
    with pytest.raises(ref.ScATACReferenceError, match="species/assembly"):
        ref.build_scatac_reference_bundle(**sources)


@pytest.mark.parametrize("path,value", [
    (("artifact_type",), "other"), (("schema_version",), 2), (("schema_version",), True),
    (("contract_version",), "v2"), (("species",), "rat"), (("target_assembly",), "mm10"),
    (("assembly_binding",), "inferred"), (("ordered_identity_algorithm",), "sorted"),
    (("reference_identity_sha256",), "a" * 64),
    (("genome", "fasta", "sha256"), "A" * 64),
    (("genome", "fai", "sha256"), "0" * 63),
    (("genome", "ordered_contig_sha256"), "x" * 64),
    (("genome", "contig_count"), False), (("genome", "contig_count"), 0),
    (("ccre", "feature_count"), 3.0), (("ccre", "feature_count"), -1),
    (("ccre", "coordinate_convention"), "one-based"),
    (("ccre", "feature_name_convention"), "column4"), (("ccre", "bounds_checked"), False),
    (("ccre", "bed", "path"), "relative.bed"),
    (("ccre", "bed", "path"), "/tmp/../wrong.bed"),
    (("ccre", "bed", "path"), "/tmp//wrong.bed"),
    (("ccre", "bed", "provenance", "basis"), "verified_from_filename"),
    (("ccre", "bed", "provenance", "source"), "inferred"),
])
def test_invalid_contract_fields(bundle, path, value):
    data = bundle.to_dict()
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ref.ScATACReferenceError):
        ref.validate_scatac_reference_bundle(data)


@pytest.mark.parametrize("path", [(), ("genome",), ("genome", "fasta"),
    ("ccre",), ("ccre", "bed"), ("ccre", "bed", "provenance")])
@pytest.mark.parametrize("extra", [True, False])
def test_closed_shapes_at_every_level(bundle, path, extra):
    data = bundle.to_dict()
    node = data
    for key in path:
        node = node[key]
    if extra:
        node["unexpected"] = 1
    else:
        del node[next(iter(node))]
    with pytest.raises(ref.ScATACReferenceError):
        ref.validate_scatac_reference_bundle(data)


@pytest.mark.parametrize("bed", [b"", b"chr1\t0\n", b"\t0\t2\n", b"chr1\t-1\t2\n",
    b"chr1\t1.0\t2\n", b"chr1\t01\t2\n", b"chr1\t+1\t2\n", b"chr1\t 1\t2\n",
    b"chr1\t2\t2\n", b"chr1\t3\t2\n", b"chr1\t0\t13\n", b"other\t0\t2\n",
    b"1\t0\t2\n", b"chr1\t0\t2\r\n", b"chr\n1\t0\t2\n", b"chr1\t0\t2\n\n",
    b"chr1\t0\t2\nchr1\t0\t2\tsecond label\n", b"chr1\t0\t9223372036854775808\n",
    b"#comment\n", b"track name=test\n", b"chr1\t0\t2\t\x00\n", b"\xff\t0\t2\n",
    b"chr:1\t0\t2\n", b"chr 1\t0\t2\n", b"chr1\t0\t2\t" + b"x" * ref.MAX_LINE_BYTES])
def test_bed_fails_closed(sources, bed):
    sources["ccre_bed_path"].write_bytes(bed)
    with pytest.raises(ref.ScATACReferenceError):
        ref.build_scatac_reference_bundle(**sources)


def test_bed_order_coordinate_and_extra_column_identities(sources, bundle):
    bed = sources["ccre_bed_path"]
    original = bed.read_bytes()
    bed.write_bytes(b"\n".join(original.rstrip(b"\n").split(b"\n")[::-1]))
    reordered = ref.build_scatac_reference_bundle(**sources)
    assert reordered.ccre.feature_count == 3
    assert reordered.ccre.ordered_feature_sha256 != bundle.ccre.ordered_feature_sha256
    bed.write_bytes(original.replace(b"extra", b"different"))
    extras = ref.build_scatac_reference_bundle(**sources)
    assert extras.ccre.ordered_feature_sha256 == bundle.ccre.ordered_feature_sha256
    assert extras.reference_identity_sha256 != bundle.reference_identity_sha256
    bed.write_bytes(original.replace(b"\t1\t5", b"\t2\t5"))
    assert ref.build_scatac_reference_bundle(**sources).ccre.ordered_feature_sha256 != bundle.ccre.ordered_feature_sha256


@pytest.mark.parametrize("fai", [b"", b"chr1\t12\n", b"chr1\t0\t6\t12\t13\n",
    b"chr1\t-1\t6\t12\t13\n", b"chr1\t12.0\t6\t12\t13\n", b"chr1\t12\t6\t0\t1\n",
    b"chr1\t12\t6\t12\t11\n", b"chr1\t12\t1000\t12\t13\n",
    b"chr1\t12\t6\t12\t13\nchr1\t12\t6\t12\t13\n", b"\t12\t6\t12\t13\n",
    b"chr1\t12\t6\t12\t13\r\n", b"chr1\t12\t6\t12\t13\textra\n"])
def test_fai_fails_closed(sources, fai):
    sources["fai_path"].write_bytes(fai)
    with pytest.raises(ref.ScATACReferenceError):
        ref.build_scatac_reference_bundle(**sources)


def test_contig_order_length_and_name_are_exact(sources, bundle):
    fai = sources["fai_path"]
    original = fai.read_bytes()
    fai.write_bytes(b"\n".join(original.rstrip(b"\n").split(b"\n")[::-1]) + b"\n")
    changed = ref.build_scatac_reference_bundle(**sources)
    assert changed.genome.ordered_contig_sha256 == sha(b"chr2\t8\nchr1\t12\n")
    assert changed.genome.ordered_contig_sha256 != bundle.genome.ordered_contig_sha256
    fai.write_bytes(original.replace(b"chr2\t8", b"chr2\t7"))
    assert ref.build_scatac_reference_bundle(**sources).genome.ordered_contig_sha256 != bundle.genome.ordered_contig_sha256
    fai.write_bytes(original.replace(b"chr2", b"2"))
    sources["ccre_bed_path"].write_bytes(sources["ccre_bed_path"].read_bytes().replace(b"chr2", b"2"))
    assert ref.build_scatac_reference_bundle(**sources).genome.ordered_contig_sha256 != bundle.genome.ordered_contig_sha256


def test_expected_vocabulary_binding(sources, bundle):
    assert ref.build_scatac_reference_bundle(**sources, expected_feature_count=3,
        expected_ordered_feature_sha256=bundle.ccre.ordered_feature_sha256) == bundle
    for expectation in ({"expected_feature_count": 2}, {"expected_ordered_feature_sha256": "0" * 64}):
        with pytest.raises(ref.ScATACReferenceError) as error:
            ref.build_scatac_reference_bundle(**sources, **expectation)
        assert error.value.code == "REFERENCE_EXPECTATION_MISMATCH"


def test_unknown_provenance_and_portable_identity(sources, bundle, tmp_path):
    assert bundle.genome.fasta.provenance == ref.SourceProvenance()
    assert bundle.ccre.bed.provenance == ref.SourceProvenance()
    assert bundle.annotation is None
    claim = ref.SourceProvenance(basis="caller_supplied", source="explicit caller assertion")
    claimed = ref.build_scatac_reference_bundle(**sources, ccre_provenance=claim)
    assert claimed.reference_identity_sha256 == bundle.reference_identity_sha256
    assert ref.canonical_reference_bundle_bytes(claimed) != ref.canonical_reference_bundle_bytes(bundle)
    new_path = tmp_path / "copy.bed"
    new_path.write_bytes(sources["ccre_bed_path"].read_bytes())
    sources["ccre_bed_path"] = new_path
    relocated = ref.build_scatac_reference_bundle(**sources)
    assert relocated.reference_identity_sha256 == bundle.reference_identity_sha256
    assert ref.canonical_reference_bundle_bytes(relocated) != ref.canonical_reference_bundle_bytes(bundle)


def test_annotation_identity_and_reinspection(sources, bundle, tmp_path):
    annotation = tmp_path / "annotation.bed"
    annotation.write_bytes(b"chr1\t0\t1\n")
    annotated = ref.build_scatac_reference_bundle(**sources, annotation_path=annotation, annotation_format="bed3")
    assert annotated.annotation.resource.sha256 == sha(annotation.read_bytes())
    assert annotated.reference_identity_sha256 != bundle.reference_identity_sha256
    assert ref.reinspect_scatac_reference_bundle_sources(annotated) == annotated
    annotation.write_bytes(b"chr1\t0\t2\n")
    with pytest.raises(ref.ScATACReferenceError) as error:
        ref.reinspect_scatac_reference_bundle_sources(annotated)
    assert error.value.code == "REFERENCE_SOURCE_CHANGED"


@pytest.mark.parametrize("resource", ["fasta_path", "fai_path", "ccre_bed_path"])
def test_reinspection_detects_changed_resources(sources, bundle, resource):
    assert ref.reinspect_scatac_reference_bundle_sources(bundle) == bundle
    path = sources[resource]
    changes = {"fasta_path": (b"ACGT", b"TGCA"), "fai_path": (b"chr2\t8", b"chr2\t7"),
               "ccre_bed_path": (b"extra", b"other")}
    path.write_bytes(path.read_bytes().replace(*changes[resource]))
    with pytest.raises(ref.ScATACReferenceError) as error:
        ref.reinspect_scatac_reference_bundle_sources(bundle)
    assert error.value.code == "REFERENCE_SOURCE_CHANGED"


def test_load_publish_validate_do_not_reinspect(sources, bundle, tmp_path, monkeypatch):
    for key in ("fasta_path", "fai_path", "ccre_bed_path"):
        sources[key].unlink()
    monkeypatch.setattr(ref, "_file_hash", lambda *_a, **_k: pytest.fail("unexpected hashing"))
    output = tmp_path / "reference.json"
    pointer = ref.publish_scatac_reference_bundle(bundle, output)
    path, loaded, digest = ref.load_scatac_reference_bundle(output, expected_sha256=pointer["manifest_sha256"])
    assert path == output and loaded == bundle and digest == sha(output.read_bytes())
    assert pointer["artifact_schema_version"] == 1
    with pytest.raises(ref.ScATACReferenceError):
        ref.reinspect_scatac_reference_bundle_sources(bundle)


def test_canonical_json_and_atomic_roundtrip(bundle, tmp_path):
    output = tmp_path / "reference.json"
    pointer = ref.publish_scatac_reference_bundle(bundle, output)
    expected = json.dumps(bundle.to_dict(), ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(",", ":")).encode()
    assert output.read_bytes() == expected
    assert ref.load_scatac_reference_bundle(output)[1] == bundle
    with pytest.raises(ref.ScATACReferenceError) as error:
        ref.publish_scatac_reference_bundle(bundle, output)
    assert error.value.code == "REFERENCE_OUTPUT_CONFLICT"
    assert ref.publish_scatac_reference_bundle(bundle, output, overwrite=True) == pointer
    assert not list(tmp_path.glob(".scatac-reference-*.tmp"))


def test_reference_digest_has_documented_portable_payload(bundle):
    data = bundle.to_dict()
    del data["reference_identity_sha256"]
    for resource in (data["genome"]["fasta"], data["genome"]["fai"], data["ccre"]["bed"]):
        del resource["path"]
        del resource["provenance"]
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert bundle.reference_identity_sha256 == sha(b"agent.scatac-reference-identity.v1\0" + payload)


@pytest.mark.parametrize("kwargs", [
    {"annotation_format": "gtf"}, {"annotation_format": "guessed"},
    {"annotation_provenance": ref.SourceProvenance("caller_supplied", source="claim")},
    {"ccre_provenance": ref.SourceProvenance("caller_supplied")},
    {"ccre_provenance": ref.SourceProvenance("unknown", accession="fabricated")},
    {"ccre_provenance": {"basis": "unknown"}},
    {"expected_feature_count": True}, {"expected_feature_count": 0},
    {"expected_ordered_feature_sha256": "invalid"},
])
def test_invalid_build_declarations_fail_before_io(sources, monkeypatch, kwargs):
    monkeypatch.setattr(ref, "_source_path", lambda *_: pytest.fail("unexpected source IO"))
    with pytest.raises(ref.ScATACReferenceError):
        ref.build_scatac_reference_bundle(**sources, **kwargs)


def test_partial_staging_artifact_is_not_published(bundle, tmp_path, monkeypatch):
    output = tmp_path / "reference.json"
    output.write_bytes(b"previous")
    real_load = ref.load_scatac_reference_bundle
    def corrupt_then_load(path, **kwargs):
        Path(path).write_bytes(b'{"artifact_type":')
        return real_load(path, **kwargs)
    monkeypatch.setattr(ref, "load_scatac_reference_bundle", corrupt_then_load)
    with pytest.raises(ref.ScATACReferenceError):
        ref.publish_scatac_reference_bundle(bundle, output, overwrite=True)
    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".scatac-reference-*.tmp"))


@pytest.mark.parametrize("payload", [b"{", b"[]", b"null", b"{}", b"\xff", b"{\"x\":1,\"x\":2}",
    b'{"nested":{"x":1,"x":2}}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}',
    b"[" * 1100 + b"]" * 1100, b" " * (ref.MAX_MANIFEST_BYTES + 1)])
def test_invalid_json_fails_closed(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_bytes(payload)
    with pytest.raises(ref.ScATACReferenceError):
        ref.load_scatac_reference_bundle(path)


def test_manifest_corruption_and_expected_byte_hash(bundle, tmp_path):
    output = tmp_path / "reference.json"
    pointer = ref.publish_scatac_reference_bundle(bundle, output)
    data = bundle.to_dict()
    data["ccre"]["feature_count"] += 1
    output.write_text(json.dumps(data))
    with pytest.raises(ref.ScATACReferenceError) as error:
        ref.load_scatac_reference_bundle(output)
    assert error.value.code == "REFERENCE_DIGEST_MISMATCH"
    output.write_bytes(ref.canonical_reference_bundle_bytes(bundle) + b"\n")
    assert ref.load_scatac_reference_bundle(output)[1] == bundle
    with pytest.raises(ref.ScATACReferenceError):
        ref.load_scatac_reference_bundle(output, expected_sha256=pointer["manifest_sha256"])


@pytest.mark.parametrize("existing", [False, True])
def test_failed_publication_preserves_previous_artifact(bundle, tmp_path, monkeypatch, existing):
    output = tmp_path / "reference.json"
    if existing:
        output.write_bytes(b"previous")
    def fail(*_args):
        raise OSError("injected failure before publication")
    monkeypatch.setattr(ref.os, "fsync", fail)
    with pytest.raises(ref.ScATACReferenceError):
        ref.publish_scatac_reference_bundle(bundle, output, overwrite=existing)
    assert output.read_bytes() == b"previous" if existing else not output.exists()
    assert not list(tmp_path.glob(".scatac-reference-*.tmp"))


def test_atomic_no_clobber_competing_writer(bundle, tmp_path, monkeypatch):
    output = tmp_path / "reference.json"
    real_link = ref.os.link
    def competitor(source, destination):
        Path(destination).write_bytes(b"competing artifact")
        real_link(source, destination)
    monkeypatch.setattr(ref.os, "link", competitor)
    with pytest.raises(ref.ScATACReferenceError) as error:
        ref.publish_scatac_reference_bundle(bundle, output)
    assert error.value.code == "REFERENCE_OUTPUT_CONFLICT"
    assert output.read_bytes() == b"competing artifact"
    assert not list(tmp_path.glob(".scatac-reference-*.tmp"))


def test_publication_never_replaces_source_or_alias(sources, bundle, tmp_path):
    bed = sources["ccre_bed_path"]
    original = bed.read_bytes()
    alias = tmp_path / "alias.json"
    alias.hardlink_to(bed)
    symbolic = tmp_path / "symlink.json"
    symbolic.symlink_to(bed)
    for path in (bed, alias, symbolic, tmp_path):
        with pytest.raises(ref.ScATACReferenceError):
            ref.publish_scatac_reference_bundle(bundle, path, overwrite=True)
    assert bed.read_bytes() == original


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink", "not_fasta"])
def test_invalid_source_paths(sources, tmp_path, kind):
    path = tmp_path / "bad"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        path.symlink_to(sources["fasta_path"])
    elif kind == "not_fasta":
        path.write_bytes(b"not a FASTA")
    sources["fasta_path"] = path
    with pytest.raises(ref.ScATACReferenceError):
        ref.build_scatac_reference_bundle(**sources)


def test_source_mutation_during_build_fails(sources, monkeypatch):
    real_hash = ref._file_hash
    def mutate(path, **kwargs):
        digest = real_hash(path, **kwargs)
        path.write_bytes(path.read_bytes() + b"\n")
        return digest
    monkeypatch.setattr(ref, "_file_hash", mutate)
    with pytest.raises(ref.ScATACReferenceError) as error:
        ref.build_scatac_reference_bundle(**sources)
    assert error.value.code == "REFERENCE_SOURCE_CHANGED"
