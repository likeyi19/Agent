"""Offline domain tests: supplied observations, never FASTQ/BAM parsing."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from agent.tools.data import raw_scatac_manifest as m


def example(kind=m.InputKind.FASTQ, species="human", assembly=None, *, path="/synthetic/sample.raw"):
    file = m.FileRecord(path, kind, size_bytes=128, mtime_ns=123)
    gid = m.group_identity(kind, (file.id,))
    coverage = m.CoverageRecord(gid, (file.id,), m.CoverageMethod.PREFIX,
                               m.CoverageScope.SAMPLE, 2, 128, 10, 1024,
                               False, m.StopReason.BUDGET, "a" * 64)
    evidence = []

    def claim(claim, value, source=m.EvidenceSource.USER_DECLARATION):
        record = m.EvidenceRecord(source, claim, value, group_id=gid,
                                 coverage_id=coverage.id if source is m.EvidenceSource.OBSERVATION else None)
        evidence.append(record)
        return (record.id,)

    def metadata(value, topic):
        if value is None:
            return m.MetadataResolution()
        values = value if isinstance(value, tuple) else (value,)
        return m.MetadataResolution(tuple(m.MetadataAssertion(
            v, m.EvidenceSource.USER_DECLARATION, claim(topic, v)) for v in values))

    barcode_source = m.BarcodeSource.FASTQ_READ if kind is m.InputKind.FASTQ else m.BarcodeSource.BAM_CELL_IDENTIFIER
    barcode = m.BarcodeProvenance(
        barcode_source, "synthetic_locator",
        claim(m.EvidenceClaim.BARCODE_SOURCE, barcode_source.value + ":synthetic_locator"),
        namespace="library-1", namespace_evidence_ids=claim(m.EvidenceClaim.BARCODE_NAMESPACE, "library-1"),
    )
    group = m.GroupRecord(
        kind, (file.id,),
        m.IntakeRoute.FASTQ_TO_CCRE if kind is m.InputKind.FASTQ else m.IntakeRoute.BAM_TO_CCRE,
        grouping_basis=m.GroupingBasis.EXPLICIT,
        grouping_evidence_ids=claim(m.EvidenceClaim.GROUPING, "explicit-membership"),
        structure=m.StructureState.SUPPORTED,
        structure_evidence_ids=claim(m.EvidenceClaim.STRUCTURE, "supported", m.EvidenceSource.OBSERVATION),
        species=metadata(species, m.EvidenceClaim.SPECIES),
        source_genome_assembly=(m.MetadataResolution(applicable=False) if kind is m.InputKind.FASTQ
                                else metadata(assembly, m.EvidenceClaim.SOURCE_ASSEMBLY)),
        barcode=barcode,
    )
    return dict(files=(file,), groups=(group,), evidence=tuple(evidence), coverage=(coverage,))


def build(**kwargs):
    return m.build_raw_intake_manifest(**example(**kwargs))


def assessed_preparation(data, code=m.PreparationCode.INPUT_PREPARATION,
                         admissibility=m.PreparationAdmissibility.SUPPORTED):
    """Synthetic future route assessment, not real inspection/admissibility logic."""
    group = data["groups"][0]
    evidence = m.EvidenceRecord(m.EvidenceSource.OBSERVATION,
        m.EvidenceClaim.PREPARATION_ADMISSIBILITY,
        code.value + ":" + group.route.value + ":" + admissibility.value,
        group_id=group.id, coverage_id=data["coverage"][0].id)
    data["evidence"] += (evidence,)
    return m.PreparationRequirement(group.id, code, admissibility=admissibility,
        route=group.route, admissibility_evidence_ids=(evidence.id,))


def test_frozen_identity_and_vocabulary():
    manifest = build()
    assert (manifest.artifact_type, manifest.schema_version, manifest.intake_contract_version) == (
        "agent.raw-scatac-intake", 1, "raw-scatac-intake.v1")
    assert manifest.inspection == m.InspectionConfiguration(
        "raw-scatac-inspection.v1", "raw-scatac-inspection-budget.v1")
    assert [r.value for r in m.IntakeRoute] == ["fastq-to-cell-by-ccre.v1", "bam-to-cell-by-ccre.v1"]
    assert [r.value for r in m.READINESS_PRECEDENCE] == [
        "READY", "READY_WITH_REPAIRS", "NEEDS_USER_INPUT", "UNSUPPORTED", "INVALID"]
    assert [e.value for e in m.ReadinessEffect] == [
        "advisory", "repair_required", "information_required", "unsupported", "invalid"]
    assert [s.value for s in m.Severity] == ["info", "warning", "error"]


def test_minimal_uninspected_manifest_is_valid_but_not_ready():
    file = m.FileRecord("/synthetic/input", m.InputKind.UNKNOWN)
    group = m.GroupRecord(m.InputKind.UNKNOWN, (file.id,))
    manifest = m.build_raw_intake_manifest(files=(file,), groups=(group,))
    assert manifest.readiness is m.Readiness.NEEDS_USER_INPUT
    assert manifest.groups[0].target_genome_assembly.value is None
    assert not manifest.repairs
    assert {r.code for r in manifest.required_information} >= {
        m.InformationCode.SPECIES, m.InformationCode.ROUTE, m.InformationCode.STRUCTURE,
        m.InformationCode.GROUPING, m.InformationCode.BARCODE_SOURCE, m.InformationCode.INSPECTION_COVERAGE}


@pytest.mark.parametrize("kind,species,source,target,compatibility,readiness", [
    (m.InputKind.FASTQ, "human", None, "hg38", "not_applicable", "READY"),
    (m.InputKind.FASTQ, "mouse", None, "mm10", "not_applicable", "READY"),
    (m.InputKind.BAM, "human", "hg38", "hg38", "match", "READY"),
    (m.InputKind.BAM, "human", "hg19", "hg38", "mismatch", "NEEDS_USER_INPUT"),
    (m.InputKind.BAM, "mouse", "mm10", "mm10", "match", "READY"),
    (m.InputKind.BAM, "human", None, "hg38", "source_unknown", "NEEDS_USER_INPUT"),
    (m.InputKind.BAM, "mouse", None, "mm10", "source_unknown", "NEEDS_USER_INPUT"),
    (m.InputKind.BAM, None, "hg38", None, "target_unresolved", "NEEDS_USER_INPUT"),
    (m.InputKind.FASTQ, None, None, None, "not_applicable", "NEEDS_USER_INPUT"),
    (m.InputKind.BAM, "human", ("hg19", "hg38"), "hg38", "source_conflict", "INVALID"),
    (m.InputKind.BAM, ("human", "mouse"), "hg38", None, "target_unresolved", "INVALID"),
    (m.InputKind.BAM, "rat", "rn6", None, "unsupported", "UNSUPPORTED"),
])
def test_assembly_and_species_semantics(kind, species, source, target, compatibility, readiness):
    manifest = build(kind=kind, species=species, assembly=source)
    group = manifest.groups[0]
    assert group.target_genome_assembly.value == target
    assert group.assembly_compatibility.value == compatibility
    assert manifest.readiness.value == readiness
    assert group.harmonization_required == (compatibility == "mismatch")
    assert bool(manifest.repairs) == (compatibility == "mismatch")
    if kind is m.InputKind.FASTQ:
        assert group.source_genome_assembly.value is None
        assert group.source_genome_assembly.state is m.ResolutionState.NOT_APPLICABLE
        assert m.PrerequisiteCode.ALIGNMENT in {r.code for r in manifest.prerequisites}
        assert m.InformationCode.SOURCE_ASSEMBLY not in {r.code for r in manifest.required_information}
    if source is None and kind is m.InputKind.BAM:
        assert m.InformationCode.SOURCE_ASSEMBLY in {r.code for r in manifest.required_information}
    if isinstance(source, tuple):
        assert {a.value for a in group.source_genome_assembly.assertions} == set(source)
        assert group.source_genome_assembly.value is None


def test_required_information_repairs_and_prerequisites_remain_distinct():
    unknown = build(kind=m.InputKind.BAM, assembly=None)
    mismatch = build(kind=m.InputKind.BAM, assembly="hg19")
    fastq = build()
    assert {r.code for r in unknown.required_information} == {m.InformationCode.SOURCE_ASSEMBLY}
    assert not unknown.repairs
    assert {r.code for r in mismatch.repairs} == {m.PreparationCode.ASSEMBLY_HARMONIZATION}
    assert {r.code for r in mismatch.required_information} == {m.InformationCode.ROUTE_ADMISSIBILITY}
    assert fastq.prerequisites and not fastq.repairs and not fastq.required_information
    assert fastq.readiness is m.Readiness.READY


@pytest.mark.parametrize("source", [m.EvidenceSource.FILENAME, m.EvidenceSource.OBSERVATION])
def test_nonauthoritative_assembly_assertion_cannot_resolve_build(source):
    data = example(kind=m.InputKind.BAM, assembly=None)
    group = data["groups"][0]
    e = m.EvidenceRecord(source, m.EvidenceClaim.SOURCE_ASSEMBLY, "hg38", group_id=group.id,
                         coverage_id=data["coverage"][0].id if source is m.EvidenceSource.OBSERVATION else None)
    metadata = m.MetadataResolution((m.MetadataAssertion("hg38", source, (e.id,)),))
    data["groups"] = (replace(group, source_genome_assembly=metadata),)
    data["evidence"] += (e,)
    manifest = m.build_raw_intake_manifest(**data)
    assert manifest.groups[0].source_genome_assembly.state is m.ResolutionState.UNKNOWN
    assert manifest.groups[0].assembly_compatibility is m.AssemblyCompatibility.SOURCE_UNKNOWN
    assert manifest.readiness is m.Readiness.NEEDS_USER_INPUT


def test_explicit_metadata_does_not_hide_conflicting_lower_authority_assertion():
    data = example(kind=m.InputKind.BAM, assembly="hg38")
    group = data["groups"][0]
    e = m.EvidenceRecord(m.EvidenceSource.FILENAME, m.EvidenceClaim.SOURCE_ASSEMBLY,
                         "hg19", group_id=group.id)
    assertions = group.source_genome_assembly.assertions + (
        m.MetadataAssertion("hg19", m.EvidenceSource.FILENAME, (e.id,)),)
    data["groups"] = (replace(group, source_genome_assembly=m.MetadataResolution(assertions)),)
    data["evidence"] += (e,)
    result = m.build_raw_intake_manifest(**data)
    assert result.readiness is m.Readiness.INVALID
    assert len(result.groups[0].source_genome_assembly.assertions) == 2


def test_historical_fastq_reference_is_provenance_only():
    data = example()
    e = m.EvidenceRecord(m.EvidenceSource.USER_DECLARATION, m.EvidenceClaim.HISTORICAL_REFERENCE,
                         "hg19", group_id=data["groups"][0].id)
    result = m.build_raw_intake_manifest(**{**data, "evidence": data["evidence"] + (e,)})
    assert result.readiness is m.Readiness.READY
    assert result.groups[0].target_genome_assembly.value == "hg38"
    assert not result.repairs


@pytest.mark.parametrize("effect,state", [
    (m.ReadinessEffect.ADVISORY, m.Readiness.READY),
    (m.ReadinessEffect.REPAIR_REQUIRED, m.Readiness.READY_WITH_REPAIRS),
    (m.ReadinessEffect.INFORMATION_REQUIRED, m.Readiness.NEEDS_USER_INPUT),
    (m.ReadinessEffect.UNSUPPORTED, m.Readiness.UNSUPPORTED),
    (m.ReadinessEffect.INVALID, m.Readiness.INVALID),
])
@pytest.mark.parametrize("severity", list(m.Severity))
def test_severity_does_not_determine_readiness(effect, state, severity):
    data = example()
    gid = data["groups"][0].id
    repair = assessed_preparation(data)
    ids = (repair.id,) if effect is m.ReadinessEffect.REPAIR_REQUIRED else ()
    issue = m.IntakeIssue("SYNTHETIC_FINDING", severity, gid, effect, requirement_ids=ids)
    result = m.build_raw_intake_manifest(**data, issues=(issue,), repairs=(repair,) if ids else ())
    assert result.readiness is state


def test_precedence_preserves_all_findings_and_group_statuses():
    left, right = example(), example(path="/another-root/sample.raw")
    gid = right["groups"][0].id
    info = m.RequiredInformation(gid, m.InformationCode.ROUTE_ADMISSIBILITY)
    repair = assessed_preparation(right)
    issues = tuple(m.IntakeIssue("FINDING_" + e.name, m.Severity.INFO, gid, e,
                                 requirement_ids=(repair.id,) if e is m.ReadinessEffect.REPAIR_REQUIRED else ())
                   for e in m.ReadinessEffect)
    data = {key: left[key] + right[key] for key in left}
    result = m.build_raw_intake_manifest(**data, issues=issues,
                                         required_information=(info,), repairs=(repair,))
    assert result.readiness is m.Readiness.INVALID
    assert {s.group_id: s.readiness for s in result.group_readiness} == {
        left["groups"][0].id: m.Readiness.READY, gid: m.Readiness.INVALID}
    assert {i.id for i in result.issues} == {i.id for i in issues}
    assert info in result.required_information and repair in result.repairs


@pytest.mark.parametrize("state", list(m.Readiness)[1:])
def test_forged_readiness_without_corresponding_findings_rejected(state):
    payload = build().to_dict()
    payload["readiness"] = state.value
    with pytest.raises(m.RawIntakeManifestError):
        m.validate_raw_intake_manifest(payload)


def test_caller_cannot_hide_required_information_or_blocked_group():
    payload = build(kind=m.InputKind.BAM, assembly=None).to_dict()
    payload["required_information"] = []
    payload["readiness"] = "READY"
    payload["group_readiness"][0]["readiness"] = "READY"
    with pytest.raises(m.RawIntakeManifestError, match="mandatory"):
        m.validate_raw_intake_manifest(payload)


def test_repair_effect_requires_actual_linked_preparation():
    data = example()
    issue = m.IntakeIssue("REPAIR", m.Severity.WARNING, data["groups"][0].id,
                          m.ReadinessEffect.REPAIR_REQUIRED)
    with pytest.raises(m.RawIntakeManifestError, match="actual linked"):
        m.build_raw_intake_manifest(**data, issues=(issue,))


@pytest.mark.parametrize("field,value", [
    ("artifact_type", "agent.scATAC"), ("schema_version", 2), ("schema_version", True),
    ("intake_contract_version", "raw-scatac-intake.v2"), ("unknown", "extra"),
    ("readiness", "PASSED"), ("files", {}), ("issues", "none"),
])
def test_root_schema_rejects_wrong_identity_shape_or_type(field, value):
    payload = build().to_dict()
    payload[field] = value
    with pytest.raises(m.RawIntakeManifestError):
        m.validate_raw_intake_manifest(payload)


@pytest.mark.parametrize("mutate", [
    lambda p: p["groups"][0].update(genome_assembly="hg38"),
    lambda p: p["groups"][0].update(biological_replicate="sample"),
    lambda p: p["groups"][0]["barcode"].update(corrected=True),
    lambda p: p["groups"][0]["barcode"].update(source="canonical_corrected"),
    lambda p: p["groups"][0]["species"].update(value="mouse"),
    lambda p: p["groups"][0]["source_genome_assembly"].update(state="resolved", value="hg38"),
    lambda p: p["groups"][0]["target_genome_assembly"].update(value="hg19"),
    lambda p: p["groups"][0].update(harmonization_required=True),
    lambda p: p["files"][0].update(size_bytes="128"),
    lambda p: p["files"][0].update(size_bytes=True),
    lambda p: p["files"][0].update(size_bytes=-1),
    lambda p: p["files"][0].update(mtime_ns=-1),
    lambda p: p["coverage"][0].update(records_inspected=-1),
    lambda p: p["coverage"][0].update(decoded_bytes_inspected=-1),
    lambda p: p["coverage"][0].update(full_file_sha256="a" * 64),
    lambda p: p["inspection"].update(budget_version="arbitrary"),
])
def test_nested_invalid_and_forged_values_rejected(mutate):
    payload = build().to_dict()
    mutate(payload)
    with pytest.raises(m.RawIntakeManifestError):
        m.validate_raw_intake_manifest(payload)


@pytest.mark.parametrize("collection", ["files", "groups", "evidence", "coverage", "prerequisites"])
def test_duplicate_ids_rejected(collection):
    payload = build().to_dict()
    payload[collection].append(payload[collection][0])
    with pytest.raises(m.RawIntakeManifestError):
        m.validate_raw_intake_manifest(payload)


def test_dangling_group_file_evidence_and_coverage_references():
    data = example()
    group = data["groups"][0]
    changes = [
        {"groups": (replace(group, file_ids=("file:missing",)),)},
        {"groups": (replace(group, grouping_evidence_ids=("evidence:missing",)),)},
        {"issues": (m.IntakeIssue("FINDING", m.Severity.ERROR, "group:missing", m.ReadinessEffect.INVALID),)},
        {"evidence": data["evidence"] + (m.EvidenceRecord(m.EvidenceSource.FILENAME,
             m.EvidenceClaim.INPUT_BINDING, "token", file_id="file:missing"),)},
        {"evidence": data["evidence"] + (m.EvidenceRecord(m.EvidenceSource.OBSERVATION,
             m.EvidenceClaim.STRUCTURE, "supported", group_id=group.id, coverage_id="coverage:missing"),)},
    ]
    for change in changes:
        with pytest.raises(m.RawIntakeManifestError):
            m.build_raw_intake_manifest(**{**data, **change})


def test_assertion_authority_and_evidence_value_must_agree():
    data = example()
    group = data["groups"][0]
    assertion = group.species.assertions[0]
    for replacement in (replace(assertion, value="mouse"),
                        replace(assertion, source=m.EvidenceSource.AUTHORITATIVE_METADATA),
                        replace(assertion, evidence_ids=group.grouping_evidence_ids)):
        data["groups"] = (replace(group, species=m.MetadataResolution((replacement,))),)
        with pytest.raises(m.RawIntakeManifestError):
            m.build_raw_intake_manifest(**data)


def test_conflict_cannot_be_serialized_as_resolved():
    payload = build(kind=m.InputKind.BAM, assembly=("hg19", "hg38")).to_dict()
    payload["groups"][0]["source_genome_assembly"].update(state="resolved", value="hg38")
    with pytest.raises(m.RawIntakeManifestError):
        m.validate_raw_intake_manifest(payload)


def test_group_identity_depends_on_membership_not_biological_labels():
    first = m.FileRecord("/root-1/sample.raw", m.InputKind.FASTQ)
    second = m.FileRecord("/root-2/sample.raw", m.InputKind.FASTQ)
    assert first.id != second.id
    assert m.group_identity(m.InputKind.FASTQ, (first.id, second.id)) == m.group_identity(
        m.InputKind.FASTQ, (second.id, first.id))
    assert m.group_identity(m.InputKind.FASTQ, (first.id,)) != m.group_identity(m.InputKind.FASTQ, (second.id,))


def test_groups_with_identical_namespace_are_not_collapsed():
    a, b = example(path="/root-1/sample.raw"), example(path="/root-2/sample.raw")
    result = m.build_raw_intake_manifest(**{k: a[k] + b[k] for k in a})
    assert len(result.groups) == 2
    assert {g.barcode.namespace for g in result.groups} == {"library-1"}
    assert len({g.id for g in result.groups}) == 2


def test_duplicate_membership_across_groups_rejected():
    data = example()
    extra = m.FileRecord("/synthetic/second.raw", m.InputKind.FASTQ)
    data["files"] += (extra,)
    data["groups"] += (m.GroupRecord(m.InputKind.FASTQ, (data["files"][0].id, extra.id),
                                     source_genome_assembly=m.MetadataResolution(applicable=False)),)
    with pytest.raises(m.RawIntakeManifestError, match="Duplicate"):
        m.build_raw_intake_manifest(**data)


def test_syntactic_grouping_never_establishes_library_identity():
    data = example()
    g = data["groups"][0]
    e = m.EvidenceRecord(m.EvidenceSource.FILENAME, m.EvidenceClaim.GROUPING,
                         "sample_S1", group_id=g.id)
    data["evidence"] += (e,)
    data["groups"] = (replace(g, grouping_basis=m.GroupingBasis.SYNTACTIC,
                              grouping_evidence_ids=(e.id,), syntactic_sample_token="sample_S1"),)
    result = m.build_raw_intake_manifest(**data)
    assert result.groups[0].syntactic_sample_token == "sample_S1"
    assert result.groups[0].library_id is None


@pytest.mark.parametrize("source", list(m.BarcodeSource))
def test_barcode_provenance_is_conservative(source):
    kind = m.InputKind.FASTQ if source is m.BarcodeSource.FASTQ_READ else m.InputKind.BAM
    data = example(kind=kind, assembly="hg38")
    g = data["groups"][0]
    if source is m.BarcodeSource.UNKNOWN:
        barcode = m.BarcodeProvenance()
    else:
        e = m.EvidenceRecord(m.EvidenceSource.OBSERVATION, m.EvidenceClaim.BARCODE_SOURCE,
                             source.value + ":test", group_id=g.id, coverage_id=data["coverage"][0].id)
        data["evidence"] += (e,)
        barcode = m.BarcodeProvenance(source, "test", (e.id,))
    data["groups"] = (replace(g, barcode=barcode),)
    result = m.build_raw_intake_manifest(**data)
    assert result.groups[0].barcode.source is source
    assert result.groups[0].barcode.producer is None
    assert result.groups[0].barcode.namespace is None
    assert result.readiness is (m.Readiness.NEEDS_USER_INPUT if source is m.BarcodeSource.UNKNOWN
                                else m.Readiness.READY)
    assert "corrected" not in result.to_dict()["groups"][0]["barcode"]


def test_barcode_producer_namespace_and_quality_evidence():
    data = example(kind=m.InputKind.BAM, assembly="hg38")
    g = data["groups"][0]
    producer = m.EvidenceRecord(m.EvidenceSource.AUTHORITATIVE_METADATA,
                                m.EvidenceClaim.BARCODE_PRODUCER, "producer.v1", group_id=g.id)
    quality = m.EvidenceRecord(m.EvidenceSource.OBSERVATION, m.EvidenceClaim.BARCODE_QUALITIES,
                               "qualities_present", group_id=g.id, coverage_id=data["coverage"][0].id)
    data["evidence"] += (producer, quality)
    data["groups"] = (replace(g, barcode=replace(g.barcode, producer="producer.v1",
        producer_evidence_ids=(producer.id,), quality_evidence_ids=(quality.id,))),)
    result = m.build_raw_intake_manifest(**data)
    assert result.groups[0].barcode.producer == "producer.v1"
    assert result.groups[0].barcode.namespace == "library-1"


def coverage_manifest(**changes):
    data = example()
    old = data["coverage"][0]
    new = replace(old, **changes)
    # Keep IDs/evidence honest so these tests reach coverage invariants.
    evidence = tuple(replace(e, coverage_id=new.id) if e.coverage_id == old.id else e for e in data["evidence"])
    replacements = {old_e.id: new_e.id for old_e, new_e in zip(data["evidence"], evidence, strict=True)}
    group = replace(data["groups"][0], structure_evidence_ids=tuple(
        replacements[e] for e in data["groups"][0].structure_evidence_ids))
    return m.build_raw_intake_manifest(**{**data, "groups": (group,), "evidence": evidence, "coverage": (new,)})


def test_complete_coverage_is_explicit_and_not_physical_file_integrity():
    result = coverage_manifest(method=m.CoverageMethod.SEQUENTIAL, scope=m.CoverageScope.COMPLETE,
                                eof_observed=True, stop_reason=m.StopReason.EOF)
    assert result.coverage[0].scope is m.CoverageScope.COMPLETE
    assert "full_file_sha256" not in result.to_dict()["coverage"][0]
    assert result.coverage[0].observed_region_sha256 == "a" * 64


@pytest.mark.parametrize("changes", [
    {"scope": m.CoverageScope.COMPLETE},
    {"eof_observed": True},
    {"stop_reason": m.StopReason.EOF},
    {"scope": m.CoverageScope.COMPLETE, "stop_reason": m.StopReason.EOF, "eof_observed": True},
    {"records_inspected": -1}, {"decoded_bytes_inspected": -1},
    {"records_inspected": 11}, {"decoded_bytes_inspected": 1025},
    {"record_limit": None, "decoded_byte_limit": None},
    {"record_limit": 0}, {"scope": m.CoverageScope.NONE},
    {"method": m.CoverageMethod.NOT_INSPECTED}, {"observed_region_sha256": "broken"},
    {"stop_reason": "random"},
])
def test_contradictory_or_invalid_coverage_rejected(changes):
    with pytest.raises(m.RawIntakeManifestError):
        coverage_manifest(**changes)


def test_coverage_error_cannot_silently_become_ready():
    result = coverage_manifest(stop_reason=m.StopReason.ERROR)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert m.InformationCode.INSPECTION_COVERAGE in {r.code for r in result.required_information}


def test_canonical_serialization_normalizes_every_record_and_reference_collection():
    a = example(kind=m.InputKind.BAM, assembly=("hg19", "hg38"))
    b = example(path="/another-root/sample.raw")
    result = m.build_raw_intake_manifest(**{k: a[k] + b[k] for k in a})

    def reverse(value):
        if isinstance(value, dict):
            return {k: reverse(v) for k, v in reversed(list(value.items()))}
        if isinstance(value, list):
            return [reverse(v) for v in reversed(value)]
        return value

    first = m.canonical_manifest_bytes(result)
    assert first == m.canonical_manifest_bytes(reverse(result.to_dict()))
    assert first == m.canonical_manifest_bytes(result)
    assert json.loads(first) == result.to_dict()
    assert b"timestamp" not in first


@pytest.mark.parametrize("content", [
    b"not-json", b"[]", b"null", b'{"artifact_type":1,"artifact_type":2}',
    b'{"value":NaN}', b'{"value":Infinity}', b'{"value":-Infinity}',
    b'{"value":1e9999}', b"\xff", b"{" * 1000,
])
def test_strict_loading_rejects_invalid_json(tmp_path, content):
    path = tmp_path / "intake.json"
    path.write_bytes(content)
    with pytest.raises(m.RawIntakeManifestError):
        m.load_raw_intake_manifest(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_serialization_rejects_nonfinite(value):
    payload = build().to_dict()
    payload["files"][0]["size_bytes"] = value
    with pytest.raises(m.RawIntakeManifestError):
        m.canonical_manifest_bytes(payload)


def test_publication_strict_roundtrip_digest_and_overwrite(tmp_path):
    manifest = build()
    path = tmp_path / "intake.json"
    reference = m.publish_raw_intake_manifest(manifest, path)
    actual_path, loaded, digest = m.load_raw_intake_manifest(path, expected_sha256=reference["manifest_sha256"])
    assert actual_path == path
    assert loaded.to_dict() == manifest.to_dict()
    assert path.read_bytes() == m.canonical_manifest_bytes(manifest)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(json.dumps(reference)) == reference
    with pytest.raises(FileExistsError):
        m.publish_raw_intake_manifest(manifest, path)
    assert m.publish_raw_intake_manifest(manifest, path, overwrite=True) == reference
    assert sorted(p.name for p in tmp_path.iterdir()) == ["intake.json"]


def test_load_hashes_actual_bytes_and_checks_identity(tmp_path):
    path = tmp_path / "intake.json"
    value = build().to_dict()
    path.write_text(json.dumps(value, indent=2))
    _, loaded, digest = m.load_raw_intake_manifest(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.to_dict() == value
    with pytest.raises(m.RawIntakeManifestError, match="SHA-256"):
        m.load_raw_intake_manifest(path, expected_sha256="0" * 64)
    value["schema_version"] = 999
    path.write_text(json.dumps(value))
    with pytest.raises(m.RawIntakeManifestError):
        m.load_raw_intake_manifest(path)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("boundary", ["fsync", "replace", "validation"])
def test_failed_publication_preserves_destination_and_cleans_staging(tmp_path, monkeypatch, existing, boundary):
    path = tmp_path / "intake.json"
    if existing:
        path.write_bytes(b"old-authoritative-content")
    def fail(*args, **kwargs):
        raise OSError("simulated failure")
    if boundary == "validation":
        monkeypatch.setattr(m, "load_raw_intake_manifest", fail)
    else:
        monkeypatch.setattr(m.os, boundary, fail)
    with pytest.raises(OSError):
        m.publish_raw_intake_manifest(build(), path, overwrite=existing)
    assert path.read_bytes() == b"old-authoritative-content" if existing else not path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == (["intake.json"] if existing else [])


def test_no_source_file_access_or_dependency_needed(tmp_path, monkeypatch):
    # The declared file need not exist. Any future parser would fail this test.
    def fail(*args, **kwargs):
        raise AssertionError("domain layer accessed source filesystem")
    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "stat", fail)
        guarded.setattr(Path, "open", fail)
        manifest = build()
        payload = m.canonical_manifest_bytes(manifest)
    assert payload
    m.publish_raw_intake_manifest(manifest, tmp_path / "manifest.json")


@pytest.mark.parametrize("path", ["relative.fastq", "/a/../b", "/a//b", "/a/./b", "//host/file", "/bad\npath"])
def test_source_path_contract_performs_no_hidden_resolution(path):
    with pytest.raises(m.RawIntakeManifestError):
        m.FileRecord(path, m.InputKind.FASTQ)


@pytest.mark.parametrize("message", ["unsafe\nmessage", "\x1b[31m", "x" * 257, " leading", "hidden\u202evalue"])
def test_bounded_plain_messages(message):
    data = example()
    issue = m.IntakeIssue("NOTE", m.Severity.INFO, data["groups"][0].id,
                          m.ReadinessEffect.ADVISORY, message=message)
    with pytest.raises(m.RawIntakeManifestError):
        m.build_raw_intake_manifest(**data, issues=(issue,))


def test_manifest_size_and_collection_limits(tmp_path, monkeypatch):
    manifest = build()
    path = tmp_path / "manifest.json"
    path.write_bytes(m.canonical_manifest_bytes(manifest))
    monkeypatch.setattr(m, "MAX_MANIFEST_BYTES", 100)
    with pytest.raises(m.RawIntakeManifestError, match="byte limit"):
        m.canonical_manifest_bytes(manifest)
    with pytest.raises(m.RawIntakeManifestError, match="byte limit"):
        m.load_raw_intake_manifest(path)
    monkeypatch.setattr(m, "MAX_MANIFEST_BYTES", 4 * 1024 * 1024)
    monkeypatch.setattr(m, "MAX_COLLECTION_ITEMS", 1)
    with pytest.raises(m.RawIntakeManifestError, match="collection"):
        m.validate_raw_intake_manifest(manifest)


@pytest.mark.parametrize("species", ["human", "mouse"])
def test_unselected_route_leaves_target_unresolved(species):
    data = example(species=species)
    data["groups"] = (replace(data["groups"][0], route=None),)
    result = m.build_raw_intake_manifest(**data)
    assert result.groups[0].target_genome_assembly == m.TargetAssembly(m.TargetState.UNRESOLVED, None)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert m.InformationCode.ROUTE in {r.code for r in result.required_information}


@pytest.mark.parametrize("source", [m.EvidenceSource.USER_DECLARATION, m.EvidenceSource.AUTHORITATIVE_METADATA])
def test_matching_authoritative_assertions_corroborate_without_conflict(source):
    data = example(kind=m.InputKind.BAM, assembly="hg38")
    group = data["groups"][0]
    e = m.EvidenceRecord(source, m.EvidenceClaim.SOURCE_ASSEMBLY, "hg38", group_id=group.id,
                         file_id=data["files"][0].id)
    data["evidence"] += (e,)
    data["groups"] = (replace(group, source_genome_assembly=m.MetadataResolution(
        group.source_genome_assembly.assertions + (m.MetadataAssertion("hg38", source, (e.id,)),))),)
    result = m.build_raw_intake_manifest(**data)
    assert result.groups[0].source_genome_assembly.state is m.ResolutionState.RESOLVED
    assert result.groups[0].source_genome_assembly.value == "hg38"
    assert len(result.groups[0].source_genome_assembly.assertions) == 2


@pytest.mark.parametrize("first,second", [(a, b) for a in m.ReadinessEffect for b in m.ReadinessEffect])
def test_every_readiness_effect_pair_has_explicit_precedence(first, second):
    data = example()
    gid = data["groups"][0].id
    repair = assessed_preparation(data)
    issues = tuple(m.IntakeIssue("PAIR_" + str(index), m.Severity.INFO, gid, effect,
        requirement_ids=(repair.id,) if effect is m.ReadinessEffect.REPAIR_REQUIRED else ())
        for index, effect in enumerate((first, second)))
    states = {
        m.ReadinessEffect.ADVISORY: "READY",
        m.ReadinessEffect.REPAIR_REQUIRED: "READY_WITH_REPAIRS",
        m.ReadinessEffect.INFORMATION_REQUIRED: "NEEDS_USER_INPUT",
        m.ReadinessEffect.UNSUPPORTED: "UNSUPPORTED",
        m.ReadinessEffect.INVALID: "INVALID",
    }
    order = ["READY", "READY_WITH_REPAIRS", "NEEDS_USER_INPUT", "UNSUPPORTED", "INVALID"]
    expected = max((states[first], states[second]), key=order.index)
    manifest = m.build_raw_intake_manifest(**data, issues=issues,
        repairs=(repair,) if m.ReadinessEffect.REPAIR_REQUIRED in (first, second) else ())
    assert manifest.readiness.value == expected
    assert len(manifest.issues) == 2


def test_cross_group_evidence_cannot_support_species_or_barcode():
    a, b = example(), example(path="/other/input.raw")
    data = {k: a[k] + b[k] for k in a}
    first, second = data["groups"]
    for changed in (replace(first, species=second.species),
                    replace(first, barcode=second.barcode),
                    replace(first, grouping_evidence_ids=second.grouping_evidence_ids)):
        with pytest.raises(m.RawIntakeManifestError, match="different raw-input group"):
            m.build_raw_intake_manifest(**{**data, "groups": (changed, second)})


def test_false_grouping_authority_is_rejected():
    data = example()
    group = data["groups"][0]
    e = m.EvidenceRecord(m.EvidenceSource.FILENAME, m.EvidenceClaim.GROUPING,
                         "sample_L001", group_id=group.id)
    with pytest.raises(m.RawIntakeManifestError, match="Grouping basis"):
        m.build_raw_intake_manifest(**{**data, "evidence": data["evidence"] + (e,),
            "groups": (replace(group, grouping_evidence_ids=(e.id,)),)})


def test_assembly_harmonization_not_allowed_for_unbound_or_unknown_source():
    for data in (example(), example(kind=m.InputKind.BAM)):
        repair = m.PreparationRequirement(data["groups"][0].id, m.PreparationCode.ASSEMBLY_HARMONIZATION)
        with pytest.raises(m.RawIntakeManifestError, match="known assembly-bound"):
            m.build_raw_intake_manifest(**data, repairs=(repair,))


def test_domain_structure_findings_derive_readiness():
    for structure in (m.StructureState.UNSUPPORTED, m.StructureState.INVALID):
        data = example()
        group = data["groups"][0]
        e = m.EvidenceRecord(m.EvidenceSource.OBSERVATION, m.EvidenceClaim.STRUCTURE,
                             structure.value, group_id=group.id, coverage_id=data["coverage"][0].id)
        result = m.build_raw_intake_manifest(**{**data, "evidence": data["evidence"] + (e,),
            "groups": (replace(group, structure=structure, structure_evidence_ids=(e.id,)),)})
        assert result.readiness.value == structure.value.upper()
        assert e.id in result.issues[0].evidence_ids


def test_observation_requires_coverage_reference():
    data = example()
    observation = m.EvidenceRecord(m.EvidenceSource.OBSERVATION, m.EvidenceClaim.STRUCTURE,
                                   "unresolved", group_id=data["groups"][0].id)
    with pytest.raises(m.RawIntakeManifestError, match="require explicit coverage"):
        m.build_raw_intake_manifest(**{**data, "evidence": data["evidence"] + (observation,)})


def test_not_inspected_coverage_is_representable_without_claiming_validation():
    data = example()
    group = data["groups"][0]
    not_started = m.CoverageRecord(group.id, group.file_ids, m.CoverageMethod.NOT_INSPECTED,
        m.CoverageScope.NONE, 0, None, None, None, False, m.StopReason.NOT_STARTED)
    result = m.build_raw_intake_manifest(**{**data, "coverage": (not_started,),
        "groups": (replace(group, structure=m.StructureState.UNRESOLVED, structure_evidence_ids=()),),
        "evidence": tuple(e for e in data["evidence"] if e.source is not m.EvidenceSource.OBSERVATION)})
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert result.coverage[0].scope is m.CoverageScope.NONE


def test_source_files_are_preserved_and_cannot_be_overwritten(tmp_path):
    source = tmp_path / "raw.input"
    source.write_bytes(b"synthetic source bytes")
    before = source.read_bytes(), source.stat().st_mtime_ns
    manifest = build(path=str(source))
    with pytest.raises(m.RawIntakeManifestError, match="cannot replace"):
        m.publish_raw_intake_manifest(manifest, source, overwrite=True)
    output = tmp_path / "manifest.json"
    m.publish_raw_intake_manifest(manifest, output)
    m.load_raw_intake_manifest(output)
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json", "raw.input"]


def test_output_symlink_rejected_without_mutating_target(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"preserve")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(m.RawIntakeManifestError):
        m.publish_raw_intake_manifest(build(), link, overwrite=True)
    assert target.read_bytes() == b"preserve"


def test_post_replace_fsync_failure_leaves_complete_valid_artifact(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    original = m.os.fsync
    calls = 0
    def fail_directory(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        return original(fd)
    monkeypatch.setattr(m.os, "fsync", fail_directory)
    manifest = build()
    with pytest.raises(OSError, match="directory fsync"):
        m.publish_raw_intake_manifest(manifest, path)
    assert path.read_bytes() == m.canonical_manifest_bytes(manifest)
    assert m.load_raw_intake_manifest(path)[1].to_dict() == manifest.to_dict()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_serialized_collections_support_frozen_request_representations():
    from agent.schemas.orchestration import freeze_json_mapping

    manifest = build()
    frozen = freeze_json_mapping(manifest.to_dict(), "manifest")
    assert m.validate_raw_intake_manifest(frozen).to_dict() == manifest.to_dict()


def test_bad_typed_builder_inputs_fail_with_domain_error():
    with pytest.raises(m.RawIntakeManifestError):
        m.build_raw_intake_manifest(files=("not-a-file-record",), groups=("not-a-group",))


def test_hardening_vocabularies_are_closed():
    assert [v.value for v in m.PreparationAdmissibility] == ["unresolved", "supported", "unsupported"]
    assert [v.value for v in m.BarcodeIdentityScope] == ["group_local", "unresolved"]


@pytest.mark.parametrize("decision,expected", [
    (m.PreparationAdmissibility.UNRESOLVED, m.Readiness.NEEDS_USER_INPUT),
    (m.PreparationAdmissibility.SUPPORTED, m.Readiness.READY_WITH_REPAIRS),
    (m.PreparationAdmissibility.UNSUPPORTED, m.Readiness.UNSUPPORTED),
])
def test_same_assembly_mismatch_supports_all_later_route_decisions(decision, expected, tmp_path):
    data = example(kind=m.InputKind.BAM, assembly="hg19")
    repairs = () if decision is m.PreparationAdmissibility.UNRESOLVED else (
        assessed_preparation(data, m.PreparationCode.ASSEMBLY_HARMONIZATION, decision),)
    result = m.build_raw_intake_manifest(**data, repairs=repairs)
    group = result.groups[0]
    assert group.source_genome_assembly.value == "hg19"
    assert group.target_genome_assembly.value == "hg38"
    assert group.assembly_compatibility is m.AssemblyCompatibility.MISMATCH
    assert group.harmonization_required
    assert len(result.repairs) == 1 and result.repairs[0].admissibility is decision
    assert result.readiness is expected
    assert result.schema_version == 1
    assert bool(result.required_information) == (decision is m.PreparationAdmissibility.UNRESOLVED)
    payload = m.canonical_manifest_bytes(result)
    assert payload == m.canonical_manifest_bytes(result.to_dict())
    path = tmp_path / "manifest.json"
    reference = m.publish_raw_intake_manifest(result, path)
    assert m.load_raw_intake_manifest(path, expected_sha256=reference["manifest_sha256"])[1] == result


def test_mismatch_cannot_forge_supported_repairs_by_removing_uncertainty():
    manifest = build(kind=m.InputKind.BAM, assembly="hg19")
    assert manifest.repairs[0].admissibility is m.PreparationAdmissibility.UNRESOLVED
    forged = manifest.to_dict()
    forged["required_information"] = []
    forged["readiness"] = "READY_WITH_REPAIRS"
    forged["group_readiness"][0]["readiness"] = "READY_WITH_REPAIRS"
    with pytest.raises(m.RawIntakeManifestError):
        m.validate_raw_intake_manifest(forged)


@pytest.mark.parametrize("defect", ["no_route", "wrong_route", "no_evidence", "wrong_claim", "wrong_value",
                                    "user_assertion", "filename", "other_group"])
def test_preparation_admissibility_requires_route_bound_evidence(defect):
    data = example(kind=m.InputKind.BAM, assembly="hg19")
    repair = assessed_preparation(data, m.PreparationCode.ASSEMBLY_HARMONIZATION)
    if defect == "no_route":
        repair = replace(repair, route=None)
    elif defect == "wrong_route":
        repair = replace(repair, route=m.IntakeRoute.FASTQ_TO_CCRE)
    elif defect == "no_evidence":
        repair = replace(repair, admissibility_evidence_ids=())
    else:
        e = data["evidence"][-1]
        e = replace(e, **{
            "wrong_claim": {"claim": m.EvidenceClaim.SOURCE_ASSEMBLY},
            "wrong_value": {"value": "mismatch_only"},
            "user_assertion": {"source": m.EvidenceSource.USER_DECLARATION},
            "filename": {"source": m.EvidenceSource.FILENAME},
            "other_group": {"group_id": "group:another"},
        }[defect])
        data["evidence"] = data["evidence"][:-1] + (e,)
        repair = replace(repair, admissibility_evidence_ids=(e.id,))
    with pytest.raises(m.RawIntakeManifestError):
        m.build_raw_intake_manifest(**data, repairs=(repair,))


def test_mixed_preparation_admissibility_cannot_be_promoted_by_one_supported_repair():
    data = example(kind=m.InputKind.BAM, assembly="hg19")
    supported = assessed_preparation(data, m.PreparationCode.INPUT_PREPARATION)
    result = m.build_raw_intake_manifest(**data, repairs=(supported,))
    assert len(result.repairs) == 2
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT


def test_conflicting_assessments_for_same_preparation_are_rejected():
    data = example(kind=m.InputKind.BAM, assembly="hg19")
    supported = assessed_preparation(data, m.PreparationCode.ASSEMBLY_HARMONIZATION)
    unsupported = assessed_preparation(data, m.PreparationCode.ASSEMBLY_HARMONIZATION,
                                       m.PreparationAdmissibility.UNSUPPORTED)
    with pytest.raises(m.RawIntakeManifestError, match="Duplicate"):
        m.build_raw_intake_manifest(**data, repairs=(supported, unsupported))


def without_namespace(data, scope=m.BarcodeIdentityScope.GROUP_LOCAL):
    group = data["groups"][0]
    return {**data, "groups": (replace(group, barcode=replace(group.barcode,
        namespace=None, namespace_evidence_ids=(), identity_scope=scope)),)}


def test_missing_external_namespace_does_not_block_group_local_identity():
    result = m.build_raw_intake_manifest(**without_namespace(example()))
    group = result.groups[0]
    assert result.readiness is m.Readiness.READY
    assert group.barcode.namespace is None
    assert group.barcode.identity_scope is m.BarcodeIdentityScope.GROUP_LOCAL
    assert group.barcode_identity_scope_id is not None
    assert not result.required_information


@pytest.mark.parametrize("explicit_namespace", [False, True])
def test_independent_groups_never_merge_identical_barcodes_or_namespace_labels(explicit_namespace):
    a, b = example(path="/first/sample.raw"), example(path="/second/sample.raw")
    if not explicit_namespace:
        a, b = without_namespace(a), without_namespace(b)
    result = m.build_raw_intake_manifest(**{key: a[key] + b[key] for key in a})
    assert result.readiness is m.Readiness.READY
    scopes = [g.barcode_identity_scope_id for g in result.groups]
    assert len(set(scopes)) == 2
    # Conceptual identity pairs only: no extraction or barcode vectors in JSON.
    assert (scopes[0], "AAAC") != (scopes[1], "AAAC")
    assert b"AAAC" not in m.canonical_manifest_bytes(result)
    if explicit_namespace:
        assert {g.barcode.namespace for g in result.groups} == {"library-1"}


def test_ambiguous_composition_scope_blocks_even_with_explicit_namespace():
    data = example()
    group = data["groups"][0]
    data["groups"] = (replace(group, barcode=replace(group.barcode,
        identity_scope=m.BarcodeIdentityScope.UNRESOLVED)),)
    result = m.build_raw_intake_manifest(**data)
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT
    assert result.groups[0].barcode.namespace == "library-1"
    assert result.groups[0].barcode_identity_scope_id is None
    assert {r.code for r in result.required_information} == {m.InformationCode.BARCODE_IDENTITY_SCOPE}


def test_group_local_scope_does_not_resolve_unknown_barcode_provenance():
    data = example()
    data["groups"] = (replace(data["groups"][0], barcode=m.BarcodeProvenance()),)
    result = m.build_raw_intake_manifest(**data)
    assert result.groups[0].barcode_identity_scope_id is not None
    assert {r.code for r in result.required_information} == {m.InformationCode.BARCODE_SOURCE}
    assert result.readiness is m.Readiness.NEEDS_USER_INPUT


def test_cell_identity_scope_is_derived_from_group_not_external_label():
    data = example()
    original = m.build_raw_intake_manifest(**data)
    unlabeled = m.build_raw_intake_manifest(**without_namespace(data))
    assert original.groups[0].barcode_identity_scope_id == unlabeled.groups[0].barcode_identity_scope_id
    forged = original.to_dict()
    forged["groups"][0]["barcode_identity_scope_id"] = "global:library-1"
    with pytest.raises(m.RawIntakeManifestError):
        m.validate_raw_intake_manifest(forged)
