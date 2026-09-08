import hashlib
import json
from pathlib import Path

import pytest

from agent.report import (build_analysis_evidence, build_analysis_report,
    get_supported_visualization_kinds, verify_analysis_report)
from agent.report import analysis_report as r


@pytest.fixture
def raw_report(raw_run, tmp_path):
    def make(case="human_fastq"):
        ctx = raw_run(case)
        evidence = build_analysis_evidence(ctx.run, tmp_path / "evidence", registry=ctx.context.registry)
        report = build_analysis_report(ctx.run, evidence, tmp_path / "report", registry=ctx.context.registry)
        return ctx, evidence, report
    return make


@pytest.mark.parametrize("case,phrase", [
    ("human_fastq", "satisfies the current preprocessing intake contract"),
    ("mouse_fastq", "satisfies the current preprocessing intake contract"),
    ("missing_species", "Additional required information or input artifacts"),
    ("missing_role", "Additional required information or input artifacts"),
    ("invalid_fastq", "invalid or internally inconsistent raw input"),
    ("unsupported_fastq", "outside the currently supported preprocessing intake contract"),
    ("human_bam", "satisfies the current preprocessing intake contract"),
    ("mismatch_bam", "Additional required information or input artifacts"),
    ("unknown_bam", "Additional required information or input artifacts"),
    ("bounded_fastq", "satisfies the current preprocessing intake contract"),
])
def test_raw_report_is_readiness_aware_figureless_and_verified(raw_report, tmp_path, case, phrase):
    ctx, evidence, report = raw_report(case)
    text = Path(report["report_path"]).read_text()
    manifest = json.loads(Path(report["manifest_path"]).read_bytes())
    assert "## Raw scATAC Intake\n" in text
    assert phrase in text.split("## Raw scATAC Intake")[0]
    assert "completed successfully and was independently verified" in text
    assert f'Preprocessing readiness is ` "{ctx.source.readiness}" `' in text
    assert "#### Reference / assembly compatibility" in text
    assert "#### Barcode identity" in text
    assert "- Raw input groups: ` 1 `" in text
    assert "Biological groups" not in text
    assert "- Coverage records by scope:" in text
    assert "- Files without inspection:" in text
    if case == "human_fastq":
        assert '- Species resolution: ` "human" ` (resolved): 1 group(s)' in text
        assert '- Source coordinate assembly: ` "not_applicable" `: 1 group(s)' in text
        assert '- Target assembly: ` "hg38" ` (resolved): 1 group(s)' in text
    assert "#### Outstanding findings" in text
    assert "Required information / input codes" in text
    assert "Preparation / repair admissibility" in text
    assert "Normal downstream prerequisite codes" in text
    assert "Source coordinates and target references are separate" in text
    assert "does not establish a supported transformation route" in text
    assert "PRIVATE_BARCODE" not in text and "PRIVATE_READ" not in text
    assert "## Figures" not in text and "![" not in text
    assert get_supported_visualization_kinds(ctx.run, evidence, registry=ctx.context.registry) == ()
    assert not (Path(report["report_path"]).parent / "figures").exists()
    raw = ctx.run.steps[0].result
    assert raw["manifest_path"] in text and raw["manifest_sha256"] in text
    assert text.count(raw["manifest_path"]) == 1
    assert "authoritative_manifest_sha256" in text
    if case == "bounded_fastq":
        assert "bounded and sample-scoped; this is not whole-file certification" in text
    if case == "missing_role":
        assert "FASTQ_REQUIRED_ROLE_MISSING" in text
    if case == "missing_species":
        facts = json.loads(Path(evidence["evidence_path"]).read_bytes())["steps"][0]["facts"]
        assert facts["required_information_codes"] == ["species"]
        assert facts["target_assembly_summary"] == [{"state": "unresolved", "value": None, "count": 1}]
    assert verify_analysis_report(ctx.run, evidence, report, registry=ctx.context.registry).passed
    second = build_analysis_report(ctx.run, evidence, tmp_path / "repeat", registry=ctx.context.registry)
    assert Path(second["report_path"]).read_bytes() == Path(report["report_path"]).read_bytes()
    assert Path(second["manifest_path"]).read_bytes() == Path(report["manifest_path"]).read_bytes()
    assert manifest["schema_version"] == 1
    assert ctx.context.production.call_count == 1


@pytest.mark.parametrize("mutation", ["readiness", "assembly", "requirement", "digest", "path", "figure"])
def test_raw_report_tampering_fails_even_with_rehashed_markdown(raw_report, mutation):
    ctx, evidence, report = raw_report("mismatch_bam" if mutation == "requirement" else "human_fastq")
    path = Path(report["report_path"])
    text = path.read_text()
    if mutation == "readiness":
        changed = text.replace('Preprocessing readiness is ` "READY" `', 'Preprocessing readiness is ` "INVALID" `')
    elif mutation == "assembly":
        changed = text.replace("hg38", "hg19")
    elif mutation == "requirement":
        changed = text.replace("route_admissibility", "invented_requirement")
    elif mutation == "digest":
        changed = text.replace(ctx.run.steps[0].result["manifest_sha256"], "0" * 64)
    elif mutation == "path":
        changed = text.replace(ctx.run.steps[0].result["manifest_path"], "/forged/manifest.json")
    else:
        changed = text + "\n![Invented QC](figures/forged.png)\n"
    assert changed != text
    path.write_text(changed)
    manifest_path = Path(report["manifest_path"])
    payload = json.loads(manifest_path.read_bytes())
    # Rehash the markdown wherever its old digest occurs; deterministic reconstruction
    # must still reject changed scientific prose rather than trust a self-consistent bundle.
    old_digest = hashlib.sha256(text.encode()).hexdigest()
    new_digest = hashlib.sha256(changed.encode()).hexdigest()
    payload = json.loads(json.dumps(payload).replace(old_digest, new_digest))
    manifest_path.write_bytes(r._canonical_json_bytes(payload))
    assert not verify_analysis_report(ctx.run, evidence, manifest_path, registry=ctx.context.registry).passed


def test_ready_with_repairs_has_distinct_wording():
    # Presentation coverage only: no currently implemented inspector claims a new repair route.
    step = r._StepFacts("raw", "inspect_raw_scATAC", (
        r._Fact("F0001", "raw", "inspect_raw_scATAC", "readiness", "READY_WITH_REPAIRS"),))
    text = r._raw_readiness_sentence(step)
    assert "Supported preparation actions remain" in text
    assert "completed successfully and was independently verified" in text


def test_reporting_cannot_call_public_tool_or_publish_raw_manifest(raw_run, tmp_path, monkeypatch):
    from agent.tools.data import raw_scatac, raw_scatac_manifest
    ctx = raw_run()
    def forbidden(*args, **kwargs):
        raise AssertionError("Reporting invoked raw production/publication")
    ctx.context.production.side_effect = forbidden
    monkeypatch.setattr(raw_scatac, "inspect_raw_scATAC", forbidden)
    monkeypatch.setattr(raw_scatac_manifest, "publish_raw_intake_manifest", forbidden)
    evidence = build_analysis_evidence(ctx.run, tmp_path / "evidence", registry=ctx.context.registry)
    report = build_analysis_report(ctx.run, evidence, tmp_path / "report", registry=ctx.context.registry)
    assert verify_analysis_report(ctx.run, evidence, report, registry=ctx.context.registry).passed
    assert ctx.context.production.call_count == 1
