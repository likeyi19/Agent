from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from agent.orchestration import AgentRuntime, ToolRegistry
from agent.schemas import AgentRequest
from agent.report import AnalysisEvidenceError, build_analysis_evidence, verify_analysis_evidence
from agent.report import evidence as e
from agent.tools.data import raw_scatac_manifest as m


@pytest.mark.parametrize("case", ["human_fastq", "mouse_fastq", "missing_species", "missing_role",
    "invalid_fastq", "unsupported_fastq", "bounded_fastq", "human_bam", "mismatch_bam", "unknown_bam"])
def test_verified_projection_matches_authoritative_manifest(raw_run, tmp_path, case):
    ctx = raw_run(case)
    result = build_analysis_evidence(ctx.run, tmp_path / "evidence", registry=ctx.context.registry)
    payload = json.loads(Path(result["evidence_path"]).read_bytes())
    assert payload["artifact_type"] == "agent.analysis-evidence" and payload["schema_version"] == 1
    step = payload["steps"][0]
    facts = step["facts"]
    raw = ctx.run.steps[0].result
    _, manifest, sha = m.load_raw_intake_manifest(raw["manifest_path"])
    assert step["recovery_identity"] == "inspect-raw-scatac-v1"
    assert step["verification"]["freshly_verified"] is True
    for field in e._RAW_INTAKE_FACT_FIELDS:
        assert facts[field] == raw[field]
    assert facts["group_readiness_counts"] == {
        state.value: sum(g.readiness is state for g in manifest.group_readiness) for state in m.Readiness}
    for field, attr in [("species_summary", "species"), ("source_assembly_summary", "source_genome_assembly"),
                        ("target_assembly_summary", "target_genome_assembly")]:
        expected = {}
        for group in manifest.groups:
            value = getattr(group, attr)
            key = (value.state.value, value.value)
            expected[key] = expected.get(key, 0) + 1
        assert {(r["state"], r["value"]): r["count"] for r in facts[field]} == expected
    for field, vocabulary, values in [
        ("assembly_compatibility_counts", m.AssemblyCompatibility, [g.assembly_compatibility for g in manifest.groups]),
        ("structure_counts", m.StructureState, [g.structure for g in manifest.groups]),
        ("barcode_source_counts", m.BarcodeSource, [g.barcode.source for g in manifest.groups]),
        ("barcode_identity_scope_counts", m.BarcodeIdentityScope, [g.barcode.identity_scope for g in manifest.groups]),
    ]:
        assert facts[field] == {state.value: values.count(state) for state in vocabulary}
    assert facts["n_harmonization_required"] == sum(g.harmonization_required for g in manifest.groups)
    assert facts["issue_codes"] == sorted({i.code for i in manifest.issues})
    assert facts["required_information_codes"] == sorted({i.code.value for i in manifest.required_information})
    assert facts["prerequisite_codes"] == sorted({i.code.value for i in manifest.prerequisites})
    assert sum(r["count"] for r in facts["preparation_summary"]) == len(manifest.repairs)
    if case == "mismatch_bam":
        assert facts["preparation_summary"] == [{"code": "assembly_harmonization", "admissibility": "unresolved", "count": 1}]
    coverage = facts["coverage_summary"]
    assert coverage["scope_counts"] == {s.value: sum(c.scope is s for c in manifest.coverage) for s in m.CoverageScope}
    assert coverage["method_counts"] == {s.value: sum(c.method is s for c in manifest.coverage) for s in m.CoverageMethod}
    sampled = [c for c in manifest.coverage if c.scope is m.CoverageScope.SAMPLE]
    assert coverage["any_sample_scoped"] == bool(sampled)
    assert coverage["n_sample_scoped_groups"] == len({c.group_id for c in sampled})
    assert coverage["n_sample_scoped_files"] == len({f for c in sampled for f in c.file_ids})
    inspected = [c for c in manifest.coverage if c.scope is not m.CoverageScope.NONE]
    assert coverage["n_groups_without_inspection"] == len({g.id for g in manifest.groups} - {c.group_id for c in inspected})
    assert coverage["n_files_without_inspection"] == len({f.id for f in manifest.files} - {f for c in inspected for f in c.file_ids})
    assert coverage["n_groups_with_insufficient_coverage"] == len({
        r.group_id for r in manifest.required_information if r.code is m.InformationCode.INSPECTION_COVERAGE})
    artifact, = payload["artifacts"]
    assert artifact["artifact_path"] == raw["manifest_path"]
    assert artifact["artifact_kind"] == "raw_scatac_intake_manifest_json"
    assert artifact["integrity"]["authoritative_digest"] == {
        "algorithm": "sha256", "value": sha, "source_result_field": "manifest_sha256"}
    assert sha == hashlib.sha256(Path(raw["manifest_path"]).read_bytes()).hexdigest()
    assert "authoritative_manifest_sha256" in artifact["integrity"]["verification_basis"]
    assert "manifest_path" not in facts and "status" not in facts
    serialized = json.dumps(payload)
    for forbidden in ("PRIVATE_READ", "PRIVATE_BARCODE", "ACGT", "IIII", "query_sequence", "assertions", "syntactic_sample_token"):
        assert forbidden not in serialized
    assert not set(("files", "groups", "evidence", "coverage")) & facts.keys()
    assert verify_analysis_evidence(ctx.run, result, registry=ctx.context.registry).passed
    second = build_analysis_evidence(ctx.run, tmp_path / "repeat", registry=ctx.context.registry)
    assert Path(second["evidence_path"]).read_bytes() == Path(result["evidence_path"]).read_bytes()
    assert ctx.context.production.call_count == 1
    assert len(list(Path(raw["manifest_path"]).parent.iterdir())) == 1


@pytest.mark.parametrize("mutation", ["source", "manifest", "summary", "declaration", "index"])
def test_fresh_evidence_revalidation_rejects_drift(raw_run, tmp_path, mutation):
    ctx = raw_run("human_bam" if mutation == "index" else "human_fastq")
    run = ctx.run
    evidence = build_analysis_evidence(run, tmp_path / "before", registry=ctx.context.registry)
    if mutation == "source":
        next(ctx.source.folder.iterdir()).write_bytes(b"changed source")
    elif mutation == "manifest":
        Path(run.steps[0].result["manifest_path"]).write_bytes(b"{}")
    elif mutation == "summary":
        run = replace(run, steps=(replace(run.steps[0], result={**run.steps[0].result, "readiness": "INVALID"}),))
    elif mutation == "declaration":
        args = {**run.plan.steps[0].arguments, "species": "mouse"}
        run = replace(run, plan=replace(run.plan, steps=(replace(run.plan.steps[0], arguments=args),)),
            steps=(replace(run.steps[0], resolved_arguments=args),))
    else:
        (ctx.source.folder / "input.bam.bai").write_bytes(b"new unusable index")
    with pytest.raises(AnalysisEvidenceError, match="Fresh source-step verification"):
        build_analysis_evidence(run, tmp_path / "after", registry=ctx.context.registry)
    assert not verify_analysis_evidence(run, evidence, registry=ctx.context.registry).passed
    assert not (tmp_path / "after" / "analysis_evidence.json").exists()


@pytest.mark.parametrize("mutation", ["recovery", "fields"])
def test_projection_rejects_registry_contract_drift(raw_run, tmp_path, mutation):
    ctx = raw_run()
    registry = ctx.context.registry
    spec = registry.get("inspect_raw_scATAC")
    if mutation == "recovery":
        spec = replace(spec, recovery_policy_version="inspect-raw-scatac-v2")
    else:
        spec = replace(spec, result_contract=replace(spec.result_contract,
            required_fields={**spec.result_contract.required_fields, "future_field": (str,)}))
    changed = ToolRegistry(tuple(spec if n == spec.name else registry.get(n) for n in registry.names()))
    with pytest.raises(AnalysisEvidenceError) as error:
        build_analysis_evidence(ctx.run, tmp_path / "evidence", registry=changed)
    assert error.value.code == "EVIDENCE_TOOL_SCHEMA_INCOMPATIBLE"


@pytest.mark.parametrize("mutation", ["readiness", "aggregate", "digest", "run", "noncanonical"])
def test_evidence_bytes_and_derived_facts_are_authoritative(raw_run, tmp_path, mutation):
    ctx = raw_run()
    evidence = build_analysis_evidence(ctx.run, tmp_path / "evidence", registry=ctx.context.registry)
    path = Path(evidence["evidence_path"])
    payload = json.loads(path.read_bytes())
    if mutation == "readiness":
        payload["steps"][0]["facts"]["readiness"] = "INVALID"
    elif mutation == "aggregate":
        payload["steps"][0]["facts"]["target_assembly_summary"][0]["value"] = "hg19"
    elif mutation == "digest":
        payload["artifacts"][0]["integrity"]["authoritative_digest"]["value"] = "0" * 64
    elif mutation == "run":
        payload["run"]["run_id"] = "forged:run"
    path.write_bytes(json.dumps(payload).encode() if mutation == "noncanonical" else e._canonical_json_bytes(payload))
    # Path-only verification must reconstruct expected content, even without a supplied digest.
    assert not verify_analysis_evidence(ctx.run, path, registry=ctx.context.registry).passed


def test_independent_group_aggregation_is_compact_and_order_independent(raw_factory, raw_context, tmp_path):
    sources = [raw_factory("bam", root=f"group-{i}", assembly="hg19" if i == 2 else "hg38") for i in range(3)]
    args = {**sources[0], "raw_input_paths": [s["raw_input_paths"] for s in reversed(sources)]}
    run = AgentRuntime(planner=raw_context.planner, registry=raw_context.registry).run(AgentRequest("groups", "Inspect raw inputs", args))
    assert run.status.value == "SUCCEEDED", run.errors
    evidence = build_analysis_evidence(run, tmp_path / "evidence", registry=raw_context.registry)
    facts = json.loads(Path(evidence["evidence_path"]).read_bytes())["steps"][0]["facts"]
    assert facts["n_files"] == facts["n_groups"] == 3
    assert facts["group_readiness_counts"] == {"READY": 2, "READY_WITH_REPAIRS": 0,
        "NEEDS_USER_INPUT": 1, "UNSUPPORTED": 0, "INVALID": 0}
    assert facts["species_summary"] == [{"state": "resolved", "value": "human", "count": 3}]
    assert facts["source_assembly_summary"] == [{"state": "resolved", "value": "hg19", "count": 1},
        {"state": "resolved", "value": "hg38", "count": 2}]
    assert facts["barcode_identity_scope_counts"] == {"group_local": 3, "unresolved": 0}
    assert facts["barcode_source_counts"]["bam_cell_identifier"] == 3
    assert facts["required_information_codes"] == ["route_admissibility"]
    # Reordering the verified domain records cannot reorder aggregate facts.
    raw = run.steps[0].result
    _, manifest, _ = m.load_raw_intake_manifest(raw["manifest_path"])
    changed = replace(manifest, groups=tuple(reversed(manifest.groups)), files=tuple(reversed(manifest.files)))
    alternate = m.publish_raw_intake_manifest(changed, tmp_path / "reordered.json")
    assert e._raw_intake_derived_facts(raw) == e._raw_intake_derived_facts(alternate)


def test_manifest_changed_between_fresh_verification_and_projection_fails(raw_run, tmp_path, monkeypatch):
    ctx = raw_run()
    original = e.verify_step
    def verified_then_changed(*args, **kwargs):
        verification = original(*args, **kwargs)
        assert verification.passed
        Path(ctx.run.steps[0].result["manifest_path"]).write_bytes(b"{}")
        return verification
    monkeypatch.setattr(e, "verify_step", verified_then_changed)
    with pytest.raises(AnalysisEvidenceError) as error:
        build_analysis_evidence(ctx.run, tmp_path / "evidence", registry=ctx.context.registry)
    assert error.value.code == "EVIDENCE_SOURCE_RESULT_INVALID"
    assert not (tmp_path / "evidence" / "analysis_evidence.json").exists()
