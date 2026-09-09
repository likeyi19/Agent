"""M11.1b synthetic artifact acceptance; no raw reads or external executables."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import pytest

from agent.tools.data import raw_scatac_manifest as m
from agent.tools.data import scatac_library_context as c


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def intake_factory(tmp_path):
    def make(specs=None, name="intake.json"):
        specs = specs or [{}]
        files, groups, evidence = [], [], []
        for i, spec in enumerate(specs):
            kind = spec.get("kind", m.InputKind.FASTQ)
            file = m.FileRecord(str(tmp_path / f"same_sample_L00{i + 1}.raw"), kind)
            gid = m.group_identity(kind, (file.id,))
            def claim(topic, value, authority=m.EvidenceSource.USER_DECLARATION):
                item = m.EvidenceRecord(authority, topic, value, group_id=gid)
                evidence.append(item)
                return (item.id,)
            library = spec.get("library_id")
            library_refs = () if library is None else claim(m.EvidenceClaim.LIBRARY, library,
                spec.get("library_authority", m.EvidenceSource.USER_DECLARATION))
            source = spec.get("source", m.BarcodeSource.FASTQ_READ if kind is m.InputKind.FASTQ
                              else m.BarcodeSource.BAM_CELL_IDENTIFIER)
            locator = spec.get("locator", "R2" if kind is m.InputKind.FASTQ else "CB")
            barcode = m.BarcodeProvenance() if source is m.BarcodeSource.UNKNOWN else m.BarcodeProvenance(
                source, locator, claim(m.EvidenceClaim.BARCODE_SOURCE, source.value + ":" + locator),
                identity_scope=spec.get("scope", m.BarcodeIdentityScope.GROUP_LOCAL))
            grouping = spec.get("grouping", m.GroupingBasis.SYNTACTIC)
            group = m.GroupRecord(kind, (file.id,),
                route=m.IntakeRoute.FASTQ_TO_CCRE if kind is m.InputKind.FASTQ else m.IntakeRoute.BAM_TO_CCRE,
                grouping_basis=grouping,
                grouping_evidence_ids=claim(m.EvidenceClaim.GROUPING, "same_sample",
                    m.EvidenceSource.FILENAME if grouping is m.GroupingBasis.SYNTACTIC
                    else m.EvidenceSource.USER_DECLARATION),
                syntactic_sample_token="same_sample", lane=str(i + 1),
                library_id=library, library_evidence_ids=library_refs,
                structure=m.StructureState.SUPPORTED,
                structure_evidence_ids=claim(m.EvidenceClaim.STRUCTURE, "supported"),
                source_genome_assembly=m.MetadataResolution(applicable=kind is not m.InputKind.FASTQ),
                barcode=barcode)
            files.append(file)
            groups.append(group)
        manifest = m.build_raw_intake_manifest(files=tuple(files), groups=tuple(groups), evidence=tuple(evidence))
        pointer = m.publish_raw_intake_manifest(manifest, tmp_path / name)
        return manifest, pointer
    return make


@pytest.fixture
def whitelist(tmp_path):
    path = tmp_path / "737K-arc-v1.txt"
    path.write_bytes(b"TTTT\nAAAA\nACGT\n")
    return c.inspect_barcode_whitelist(path)


def declare(ids, whitelist=None, namespace="library_A", **kwargs):
    options = dict(namespace=namespace, group_ids=tuple(ids), membership_basis=c.MembershipBasis.SINGLE_GROUP,
        barcode_interpretation=c.BarcodeInterpretation.RAW_SEQUENCE,
        correction_policy=c.CorrectionPolicy.WHITELIST_REQUIRED, whitelist=whitelist)
    options.update(kwargs)
    return c.LibraryDeclaration(**options)


def build(pointer, declarations, selection=c.SelectionMode.ALL_GROUPS):
    return c.build_scatac_library_processing_context(intake_manifest_path=pointer["manifest_path"],
        expected_intake_sha256=pointer["manifest_sha256"], libraries=tuple(declarations), selection_mode=selection)


@pytest.fixture
def context(intake_factory, whitelist):
    manifest, pointer = intake_factory()
    return build(pointer, [declare([manifest.groups[0].id], whitelist)])


def test_contract_exact_binding_and_frozen_records(context):
    assert context.artifact_type == "agent.scatac-library-processing-context"
    assert context.schema_version == 1
    assert context.contract_version == "scatac-library-processing-context.v1"
    assert context.intake.artifact_type == m.RAW_INTAKE_ARTIFACT_TYPE
    assert context.intake.contract_version == m.RAW_INTAKE_CONTRACT_VERSION
    assert context.intake.manifest_sha256 == sha(Path(context.intake.manifest_path).read_bytes())
    assert c.validate_library_context_intake_binding(context) == context
    assert c.validate_scatac_library_processing_context(context.to_dict()) == context
    with pytest.raises(FrozenInstanceError):
        context.libraries[0].namespace = "wrong"
    data = c.canonical_library_processing_context_bytes(context)
    assert all(token not in data for token in (b"reference_identity", b"hg38", b"mm10", b"chemistry", b"AAAA"))


def test_canonical_portable_identity_has_exact_documented_payload(context):
    payload = context.to_dict()
    del payload["context_identity_sha256"]
    del payload["intake"]["manifest_path"]
    for lib in payload["libraries"]:
        del lib["whitelist"]["resource"]["path"]
        del lib["whitelist"]["resource"]["provenance"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert context.context_identity_sha256 == sha(b"agent.scatac-library-processing-context.v1\0" + encoded)


@pytest.mark.parametrize("authority", [m.EvidenceSource.USER_DECLARATION, m.EvidenceSource.AUTHORITATIVE_METADATA])
def test_authoritative_library_can_merge_lanes(intake_factory, whitelist, authority):
    manifest, pointer = intake_factory([{"library_id": "lib", "library_authority": authority}] * 2)
    declaration = declare([g.id for g in manifest.groups], whitelist,
                          membership_basis=c.MembershipBasis.INTAKE_LIBRARY_ID)
    context = build(pointer, [declaration])
    assert len(context.libraries) == 1
    assert context.libraries[0].source_library_id == "lib"
    assert c.validate_library_context_intake_binding(context) == context
    assert build(pointer, [replace(declaration, group_ids=declaration.group_ids[::-1])]) == context


def test_explicit_shared_mapping_without_intake_library_id(intake_factory, whitelist):
    manifest, pointer = intake_factory([{}, {}])
    context = build(pointer, [declare([g.id for g in manifest.groups], whitelist,
        membership_basis=c.MembershipBasis.CALLER_SHARED_LIBRARY)])
    assert context.libraries[0].source_library_id is None
    assert context.libraries[0].membership_basis is c.MembershipBasis.CALLER_SHARED_LIBRARY


@pytest.mark.parametrize("basis", [c.MembershipBasis.SINGLE_GROUP, c.MembershipBasis.INTAKE_LIBRARY_ID])
def test_syntactic_similarity_does_not_authorize_merge(intake_factory, whitelist, basis):
    manifest, pointer = intake_factory([{}, {}])
    with pytest.raises(c.LibraryContextError):
        build(pointer, [declare([g.id for g in manifest.groups], whitelist, membership_basis=basis)])


@pytest.mark.parametrize("basis", [c.MembershipBasis.INTAKE_LIBRARY_ID, c.MembershipBasis.CALLER_SHARED_LIBRARY])
def test_conflicting_authoritative_library_ids_cannot_merge(intake_factory, whitelist, basis):
    manifest, pointer = intake_factory([{"library_id": "A"}, {"library_id": "B"}])
    with pytest.raises(c.LibraryContextError):
        build(pointer, [declare([g.id for g in manifest.groups], whitelist, membership_basis=basis)])


def test_selected_known_library_cannot_split_namespaces(intake_factory, whitelist):
    manifest, pointer = intake_factory([{"library_id": "same"}] * 2)
    with pytest.raises(c.LibraryContextError, match="shared library"):
        build(pointer, [declare([g.id], whitelist, namespace=f"lib{i}") for i, g in enumerate(manifest.groups)])


def test_caller_mapping_can_cover_one_known_and_one_undeclared_group(intake_factory, whitelist):
    manifest, pointer = intake_factory([{"library_id": "known"}, {}])
    declaration = declare([g.id for g in manifest.groups], whitelist,
                          membership_basis=c.MembershipBasis.CALLER_SHARED_LIBRARY)
    assert build(pointer, [declaration]).libraries[0].source_library_id == "known"
    with pytest.raises(c.LibraryContextError):
        build(pointer, [replace(declaration, membership_basis=c.MembershipBasis.INTAKE_LIBRARY_ID)])


def test_independent_namespaces_and_collection_order(intake_factory, whitelist):
    manifest, pointer = intake_factory([{}, {}])
    declarations = [declare([g.id], whitelist, namespace=f"library_{i}") for i, g in enumerate(manifest.groups)]
    first = build(pointer, declarations)
    second = build(pointer, declarations[::-1])
    assert c.canonical_library_processing_context_bytes(first) == c.canonical_library_processing_context_bytes(second)
    assert (first.libraries[0].namespace, "AAAA") != (first.libraries[1].namespace, "AAAA")
    assert all(lib.namespace != lib.groups[0].group_id for lib in first.libraries)
    with pytest.raises(c.LibraryContextError):
        build(pointer, [declarations[0], replace(declarations[1], namespace=declarations[0].namespace)])


@pytest.mark.parametrize("namespace", ["", " ", " bad", "bad ", "bad\n", "bad\r", "bad\x00", "a/b", "a:b", None])
def test_namespace_requires_explicit_safe_declaration(intake_factory, whitelist, namespace):
    manifest, pointer = intake_factory()
    with pytest.raises(c.LibraryContextError):
        build(pointer, [declare([manifest.groups[0].id], whitelist, namespace=namespace)])


def test_partial_group_selection_is_explicit(intake_factory, whitelist):
    manifest, pointer = intake_factory([{"library_id": "same"}] * 2)
    declaration = declare([manifest.groups[0].id], whitelist)
    assert build(pointer, [declaration], c.SelectionMode.EXPLICIT_SUBSET).selection_mode is c.SelectionMode.EXPLICIT_SUBSET
    with pytest.raises(c.LibraryContextError):
        build(pointer, [declaration])
    all_groups = replace(declaration, group_ids=tuple(g.id for g in manifest.groups),
                         membership_basis=c.MembershipBasis.INTAKE_LIBRARY_ID)
    with pytest.raises(c.LibraryContextError):
        build(pointer, [all_groups], c.SelectionMode.EXPLICIT_SUBSET)


@pytest.mark.parametrize("case", ["unknown", "duplicate", "twice", "wrong_manifest"])
def test_exact_group_membership(intake_factory, whitelist, case):
    manifest, pointer = intake_factory()
    gid = manifest.groups[0].id
    declarations = [declare([gid], whitelist)]
    if case == "unknown":
        declarations = [declare(["group:" + "0" * 64], whitelist)]
    elif case == "duplicate":
        declarations = [declare([gid, gid], whitelist, membership_basis=c.MembershipBasis.CALLER_SHARED_LIBRARY)]
    elif case == "twice":
        declarations.append(declare([gid], whitelist, namespace="other"))
    else:
        other, _ = intake_factory([{}, {}], name="other.json")
        declarations = [declare([next(g.id for g in other.groups if g.id != gid)], whitelist)]
    with pytest.raises(c.LibraryContextError):
        build(pointer, declarations)


def test_byte_changed_intake_binding_fails_even_if_semantics_same(context):
    path = Path(context.intake.manifest_path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(c.LibraryContextError):
        c.validate_library_context_intake_binding(context)


@pytest.mark.parametrize("locator", ["R2", "I2"])
def test_fastq_locator_is_projected_from_m10(intake_factory, whitelist, locator):
    manifest, pointer = intake_factory([{"locator": locator}])
    gid = manifest.groups[0].id
    source = c.GroupBarcodeBinding(gid, m.BarcodeSource.FASTQ_READ, locator)
    declaration = declare([gid], whitelist, expected_barcode_bindings=(source,))
    context = build(pointer, [declaration])
    assert context.libraries[0].groups == (source,)
    with pytest.raises(c.LibraryContextError):
        build(pointer, [replace(declaration, expected_barcode_bindings=(replace(source, locator="wrong"),))])


@pytest.mark.parametrize("spec", [{"source": m.BarcodeSource.UNKNOWN},
    {"scope": m.BarcodeIdentityScope.UNRESOLVED}])
def test_unresolved_barcode_facts_fail_closed(intake_factory, whitelist, spec):
    manifest, pointer = intake_factory([spec])
    with pytest.raises(c.LibraryContextError):
        build(pointer, [declare([manifest.groups[0].id], whitelist)])


@pytest.mark.parametrize("source,locator", [(m.BarcodeSource.BAM_CELL_IDENTIFIER, "CB"),
                                           (m.BarcodeSource.BAM_RAW_SEQUENCE, "CR")])
def test_bam_interpretation_is_explicit_not_tag_inference(intake_factory, source, locator):
    manifest, pointer = intake_factory([{"kind": m.InputKind.BAM, "source": source, "locator": locator}])
    gid = manifest.groups[0].id
    corrected = declare([gid], barcode_interpretation=c.BarcodeInterpretation.CORRECTED_IDENTIFIER,
                        correction_policy=c.CorrectionPolicy.ALREADY_CORRECTED)
    assert build(pointer, [corrected]).libraries[0].interpretation_basis == "caller_declaration"
    with pytest.raises(c.LibraryContextError):
        build(pointer, [replace(corrected, barcode_interpretation=None)])
    raw = replace(corrected, barcode_interpretation=c.BarcodeInterpretation.RAW_SEQUENCE,
                  correction_policy=c.CorrectionPolicy.UNQUALIFIED_RAW_BAM)
    assert build(pointer, [raw]).libraries[0].correction_policy is c.CorrectionPolicy.UNQUALIFIED_RAW_BAM


@pytest.mark.parametrize("kind", [m.InputKind.FASTQ, m.InputKind.BAM])
@pytest.mark.parametrize("interpretation", list(c.BarcodeInterpretation))
@pytest.mark.parametrize("policy", list(c.CorrectionPolicy))
@pytest.mark.parametrize("with_whitelist", [False, True])
def test_entire_barcode_policy_legality_table(intake_factory, whitelist, kind, interpretation, policy, with_whitelist):
    manifest, pointer = intake_factory([{"kind": kind}])
    declaration = declare([manifest.groups[0].id], whitelist if with_whitelist else None,
                          barcode_interpretation=interpretation, correction_policy=policy)
    raw = interpretation is c.BarcodeInterpretation.RAW_SEQUENCE
    legal = ((kind is m.InputKind.FASTQ and raw and policy is c.CorrectionPolicy.WHITELIST_REQUIRED and with_whitelist)
             or (kind is m.InputKind.BAM and raw and policy is c.CorrectionPolicy.UNQUALIFIED_RAW_BAM)
             or (kind is m.InputKind.BAM and not raw and policy is c.CorrectionPolicy.ALREADY_CORRECTED and not with_whitelist))
    if legal:
        assert build(pointer, [declaration]).libraries[0].correction_policy is policy
    else:
        with pytest.raises(c.LibraryContextError):
            build(pointer, [declaration])


def test_no_auto_or_inferred_policy(intake_factory, whitelist):
    manifest, pointer = intake_factory()
    for policy in (None, "AUTO", "no_correction"):
        with pytest.raises(c.LibraryContextError):
            build(pointer, [declare([manifest.groups[0].id], whitelist, correction_policy=policy)])


def test_authoritative_barcode_length_contradiction(intake_factory, whitelist):
    manifest, pointer = intake_factory()
    declaration = declare([manifest.groups[0].id], whitelist, declared_barcode_length=4)
    assert build(pointer, [declaration]).libraries[0].declared_barcode_length == 4
    for length in (5, 0, True, -1):
        with pytest.raises(c.LibraryContextError):
            build(pointer, [replace(declaration, declared_barcode_length=length)])


def test_whitelist_byte_and_set_identities(whitelist):
    assert whitelist.resource.sha256 == sha(b"TTTT\nAAAA\nACGT\n")
    assert whitelist.barcode_set_sha256 == sha(b"AAAA\nACGT\nTTTT\n")
    assert whitelist.n_barcodes == 3 and whitelist.barcode_length == 4
    assert whitelist.set_identity_algorithm == "sha256-ascii-sorted-lf.v1"
    path = Path(whitelist.resource.path)
    path.write_bytes(b"AAAA\nACGT\nTTTT")
    changed = c.inspect_barcode_whitelist(path)
    assert changed.barcode_set_sha256 == whitelist.barcode_set_sha256
    assert changed.resource.sha256 != whitelist.resource.sha256


@pytest.mark.parametrize("payload", [b"", b"\n", b"AAAA\n\n", b"AAAA\r\n", b"aaaa\n", b"AAAA \n",
    b" AAAA\n", b"AAAA\tTTTT\n", b"AAAA-1\n", b"AAAN\n", b"AAAU\n", b"AAAA\nAAAA\n",
    b"AAA\nAAAA\n", b"A" * 257 + b"\n", b"AA\x00A\n", b"\xff\n"])
def test_whitelist_rejects_malformed_records(tmp_path, payload):
    path = tmp_path / "whitelist.txt"
    path.write_bytes(payload)
    with pytest.raises(c.LibraryContextError):
        c.inspect_barcode_whitelist(path)


def test_whitelist_change_is_detected_by_reinspection_and_build(context):
    path = Path(context.libraries[0].whitelist.resource.path)
    path.write_bytes(b"AAAA\nACGT\nTTTT\n")  # Same candidates, different file identity.
    with pytest.raises(c.LibraryContextError) as error:
        c.reinspect_barcode_resources(context)
    assert error.value.code == "LIBRARY_RESOURCE_CHANGED"
    declaration = declare([context.libraries[0].groups[0].group_id], context.libraries[0].whitelist)
    with pytest.raises(c.LibraryContextError):
        build({"manifest_path": context.intake.manifest_path,
               "manifest_sha256": context.intake.manifest_sha256}, [declaration])


def test_unknown_and_caller_supplied_whitelist_history(whitelist, intake_factory):
    assert whitelist.resource.provenance == c.SourceProvenance()
    provenance = c.SourceProvenance("caller_supplied", source="caller claim", accession="explicit-accession")
    claimed = c.inspect_barcode_whitelist(whitelist.resource.path, provenance=provenance)
    assert claimed.resource.provenance == provenance
    manifest, pointer = intake_factory()
    original = build(pointer, [declare([manifest.groups[0].id], whitelist)])
    changed = build(pointer, [declare([manifest.groups[0].id], claimed)])
    assert original.context_identity_sha256 == changed.context_identity_sha256
    assert c.canonical_library_processing_context_bytes(original) != c.canonical_library_processing_context_bytes(changed)


def test_build_binding_load_and_publish_never_open_raw(context, tmp_path, monkeypatch):
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".raw":
            pytest.fail("raw source opened")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    declaration = declare([context.libraries[0].groups[0].group_id], context.libraries[0].whitelist)
    assert build({"manifest_path": context.intake.manifest_path,
                  "manifest_sha256": context.intake.manifest_sha256}, [declaration]) == context
    output = tmp_path / "context.json"
    c.publish_scatac_library_processing_context(context, output)
    Path(context.intake.manifest_path).unlink()
    Path(context.libraries[0].whitelist.resource.path).unlink()
    assert c.load_scatac_library_processing_context(output)[1] == context
    with pytest.raises(c.LibraryContextError):
        c.validate_library_context_intake_binding(context)


def test_serialization_and_publication_roundtrip(context, tmp_path):
    output = tmp_path / "context.json"
    pointer = c.publish_scatac_library_processing_context(context, output)
    assert pointer["manifest_sha256"] == sha(output.read_bytes())
    assert output.read_bytes() == c.canonical_library_processing_context_bytes(context)
    assert c.load_scatac_library_processing_context(output, expected_sha256=pointer["manifest_sha256"])[1] == context
    with pytest.raises(c.LibraryContextError) as error:
        c.publish_scatac_library_processing_context(context, output)
    assert error.value.code == "LIBRARY_OUTPUT_CONFLICT"
    assert c.publish_scatac_library_processing_context(context, output, overwrite=True) == pointer


@pytest.mark.parametrize("path,value", [
    (("artifact_type",), "wrong"), (("schema_version",), True), (("schema_version",), 2),
    (("contract_version",), "wrong"), (("context_identity_sha256",), "0" * 64),
    (("selection_mode",), "AUTO"), (("intake", "manifest_sha256"), "A" * 64),
    (("intake", "manifest_path"), "relative"), (("intake", "schema_version"), True),
    (("intake", "contract_version"), "wrong"), (("libraries", 0, "namespace_basis"), "filename"),
    (("libraries", 0, "interpretation_basis"), "observed_CB"),
    (("libraries", 0, "whitelist", "n_barcodes"), 0),
    (("libraries", 0, "whitelist", "barcode_length"), True),
    (("libraries", 0, "whitelist", "barcode_set_sha256"), "bad"),
    (("libraries", 0, "whitelist", "syntax"), "arbitrary"),
    (("libraries", 0, "whitelist", "resource", "provenance", "source"), "inferred vendor"),
])
def test_invalid_domain_fields(context, path, value):
    data = json.loads(c.canonical_library_processing_context_bytes(context))
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(c.LibraryContextError):
        c.validate_scatac_library_processing_context(data)


@pytest.mark.parametrize("path", [(), ("intake",), ("libraries", 0), ("libraries", 0, "groups", 0),
    ("libraries", 0, "whitelist"), ("libraries", 0, "whitelist", "resource"),
    ("libraries", 0, "whitelist", "resource", "provenance")])
@pytest.mark.parametrize("extra", [True, False])
def test_strict_closed_shapes(context, path, extra):
    data = json.loads(c.canonical_library_processing_context_bytes(context))
    node = data
    for key in path:
        node = node[key]
    if extra:
        node["unexpected"] = True
    else:
        del node[next(iter(node))]
    with pytest.raises(c.LibraryContextError):
        c.validate_scatac_library_processing_context(data)


@pytest.mark.parametrize("payload", [b"{", b"[]", b"null", b"{}", b"\xff", b'{"x":1,"x":2}',
    b'{"x":{"a":1,"a":2}}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}',
    b"[" * 1100 + b"]" * 1100, b" " * (c.MAX_CONTEXT_BYTES + 1)])
def test_malformed_json(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_bytes(payload)
    with pytest.raises(c.LibraryContextError):
        c.load_scatac_library_processing_context(path)


def test_modified_manifest_bytes_fail_expected_digest(context, tmp_path):
    output = tmp_path / "context.json"
    pointer = c.publish_scatac_library_processing_context(context, output)
    output.write_bytes(output.read_bytes() + b"\n")
    assert c.load_scatac_library_processing_context(output)[1] == context
    with pytest.raises(c.LibraryContextError):
        c.load_scatac_library_processing_context(output, expected_sha256=pointer["manifest_sha256"])


def test_publication_protects_intake_whitelist_and_raw_paths(context, tmp_path):
    _, manifest, _ = m.load_raw_intake_manifest(context.intake.manifest_path)
    raw = Path(manifest.files[0].path)
    raw.write_bytes(b"synthetic input")
    alias = tmp_path / "alias.json"
    alias.hardlink_to(raw)
    for path in (context.intake.manifest_path, context.libraries[0].whitelist.resource.path, raw, alias):
        with pytest.raises(c.LibraryContextError):
            c.publish_scatac_library_processing_context(context, path, overwrite=True)
    assert raw.read_bytes() == b"synthetic input"


@pytest.mark.parametrize("failure", ["fsync", "corrupt_staging"])
def test_failed_atomic_publication_preserves_previous(context, tmp_path, monkeypatch, failure):
    output = tmp_path / "context.json"
    output.write_bytes(b"previous")
    if failure == "fsync":
        def fail(*args):
            raise OSError("injected fsync failure")
        monkeypatch.setattr(c.os, "fsync", fail)
    else:
        real_load = c.load_scatac_library_processing_context
        def corrupt(path, **kwargs):
            Path(path).write_bytes(b'{"partial":')
            return real_load(path, **kwargs)
        monkeypatch.setattr(c, "load_scatac_library_processing_context", corrupt)
    with pytest.raises(c.LibraryContextError):
        c.publish_scatac_library_processing_context(context, output, overwrite=True)
    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".library-context-*.tmp"))


def test_concurrent_identical_publication_never_clobbers(context, tmp_path, monkeypatch):
    output = tmp_path / "context.json"
    real_link = c.os.link
    def competitor(source, destination):
        Path(destination).write_bytes(c.canonical_library_processing_context_bytes(context))
        real_link(source, destination)
    monkeypatch.setattr(c.os, "link", competitor)
    with pytest.raises(c.LibraryContextError) as error:
        c.publish_scatac_library_processing_context(context, output)
    assert error.value.code == "LIBRARY_OUTPUT_CONFLICT"
    assert c.load_scatac_library_processing_context(output)[1] == context
    assert not list(tmp_path.glob(".library-context-*.tmp"))


@pytest.mark.parametrize("field,value", [("source_library_id", "invented"), ("locator", "I2")])
def test_self_consistent_forged_projection_still_fails_m10_binding(context, field, value):
    library = context.libraries[0]
    if field == "locator":
        library = replace(library, groups=(replace(library.groups[0], locator=value),))
    else:
        library = replace(library, source_library_id=value)
    forged = replace(context, libraries=(library,))
    forged = replace(forged, context_identity_sha256=c._context_digest(forged))
    assert c.validate_scatac_library_processing_context(forged) == forged
    with pytest.raises(c.LibraryContextError) as error:
        c.validate_library_context_intake_binding(forged)
    assert error.value.code == "LIBRARY_INTAKE_INVALID"


def test_changed_intake_cannot_be_substituted_on_explicit_binding(context, tmp_path):
    alternate = tmp_path / "alternate.json"
    data = Path(context.intake.manifest_path).read_bytes()
    alternate.write_bytes(data)
    assert c.validate_library_context_intake_binding(context, intake_manifest_path=alternate) == context
    alternate.write_bytes(data + b"\n")
    with pytest.raises(c.LibraryContextError):
        c.validate_library_context_intake_binding(context, intake_manifest_path=alternate)


def test_relocation_preserves_context_identity(context, tmp_path):
    whitelist = context.libraries[0].whitelist
    alternate = tmp_path / "relocated-whitelist.txt"
    alternate.write_bytes(Path(whitelist.resource.path).read_bytes())
    relocated = c.inspect_barcode_whitelist(alternate)
    manifest = tmp_path / "relocated-intake.json"
    manifest.write_bytes(Path(context.intake.manifest_path).read_bytes())
    rebuilt = build({"manifest_path": str(manifest), "manifest_sha256": context.intake.manifest_sha256},
                    [declare([context.libraries[0].groups[0].group_id], relocated)])
    assert rebuilt.context_identity_sha256 == context.context_identity_sha256
    assert c.canonical_library_processing_context_bytes(rebuilt) != c.canonical_library_processing_context_bytes(context)


def test_mixed_kinds_can_be_independent_but_not_one_library(intake_factory, whitelist):
    manifest, pointer = intake_factory([{}, {"kind": m.InputKind.BAM}])
    declarations = []
    for group in manifest.groups:
        declarations.append(declare([group.id], whitelist) if group.input_kind is m.InputKind.FASTQ else
            declare([group.id], namespace="bam_library",
                    barcode_interpretation=c.BarcodeInterpretation.CORRECTED_IDENTIFIER,
                    correction_policy=c.CorrectionPolicy.ALREADY_CORRECTED))
    assert len(build(pointer, declarations).libraries) == 2
    with pytest.raises(c.LibraryContextError):
        build(pointer, [declare([g.id for g in manifest.groups], whitelist,
                                membership_basis=c.MembershipBasis.CALLER_SHARED_LIBRARY)])


def test_invalid_m10_file_kind_is_rejected_at_binding(intake_factory, whitelist, tmp_path):
    manifest, pointer = intake_factory()
    data = manifest.to_dict()
    data["files"][0]["input_kind"] = "bam"
    malformed = tmp_path / "bad-intake.json"
    payload = json.dumps(data).encode()
    malformed.write_bytes(payload)
    with pytest.raises(c.LibraryContextError) as error:
        build({"manifest_path": str(malformed), "manifest_sha256": sha(payload)},
              [declare([manifest.groups[0].id], whitelist)])
    assert error.value.code == "LIBRARY_INTAKE_INVALID"


@pytest.mark.parametrize("layout", ["A", "B"])
def test_context_consumes_actual_m10_layout_evidence(tmp_path, whitelist, layout):
    from agent.tools.data import _raw_fastq as fastq
    reviewed_layout = getattr(fastq.FastqLayout, layout)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Layout inventory comes from the existing M10 authority even in this test.
    for role in fastq.LAYOUT_ROLES[reviewed_layout]:
        (raw_dir / f"sample_S1_L001_{role.value}_001.fastq").write_bytes(b"@read\nACGT\n+\nIIII\n")
    manifest = fastq.inspect_fastq_inputs(raw_dir, species="human", assay=fastq.FastqAssay.TENX_ATAC,
                                         declared_layout=reviewed_layout)
    pointer = m.publish_raw_intake_manifest(manifest, tmp_path / "intake.json")
    for path in raw_dir.iterdir():
        path.unlink()  # M11 consumes the published M10 evidence, not raw inputs.
    context = build(pointer, [declare([manifest.groups[0].id], whitelist)])
    assert context.libraries[0].groups[0].locator == manifest.groups[0].barcode.locator


def test_whitelist_mutation_during_inspection_fails(tmp_path, monkeypatch):
    path = tmp_path / "whitelist.txt"
    path.write_bytes(b"AAAA\nTTTT\n")
    real_snapshot = c._snapshot
    calls = 0
    def changed(p):
        nonlocal calls
        calls += 1
        if calls == 3:
            p.write_bytes(p.read_bytes() + b"ACGT\n")
        return real_snapshot(p)
    monkeypatch.setattr(c, "_snapshot", changed)
    with pytest.raises(c.LibraryContextError) as error:
        c.inspect_barcode_whitelist(path)
    assert error.value.code == "LIBRARY_RESOURCE_CHANGED"


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink", "remote"])
def test_whitelist_requires_local_regular_file(tmp_path, kind):
    path = tmp_path / "whitelist.txt"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target.txt"
        target.write_bytes(b"AAAA\n")
        path.symlink_to(target)
    elif kind == "remote":
        path = "https://example.invalid/whitelist.txt"
    with pytest.raises(c.LibraryContextError):
        c.inspect_barcode_whitelist(path)
