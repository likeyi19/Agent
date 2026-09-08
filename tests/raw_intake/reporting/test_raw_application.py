import hashlib
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from agent.application import ResearchAgentApplication
from agent.application import service
from agent.report import ANALYSIS_REPORT_MANIFEST_FILENAME, verify_analysis_report
from agent.schemas import AgentRequest, RunMode


@pytest.mark.parametrize("case", ["human_fastq", "mouse_fastq", "missing_species", "missing_role",
    "invalid_fastq", "unsupported_fastq", "human_bam", "mismatch_bam", "unknown_bam"])
def test_full_raw_application_and_resume(intake_case, raw_context, tmp_path, case):
    source = intake_case(case)
    inputs = {k: v for k, v in source.args.items() if k != "output_dir"}
    app = ResearchAgentApplication(tmp_path / "workspace", planner=raw_context.planner, registry=raw_context.registry)
    result = app.run(AgentRequest("raw-app", "Inspect raw scATAC inputs and report preprocessing readiness.", inputs))
    assert result.status.value == result.run_status.value == "SUCCEEDED", result
    raw = result.run_result.steps[0].result
    assert raw["readiness"] == source.readiness
    assert result.evidence is not None and result.report is not None and result.visualization is None
    for artifact in (result.evidence, result.report):
        assert Path(artifact.path).is_file()
        assert artifact.sha256 == hashlib.sha256(Path(artifact.path).read_bytes()).hexdigest()
    workspace = app._workspace.run_paths(result.run_id)
    assert not list(workspace.visualizations.iterdir())
    manifest_path = Path(result.report.path).parent / ANALYSIS_REPORT_MANIFEST_FILENAME
    assert verify_analysis_report(result.run_result, result.evidence.path, manifest_path, registry=app.registry).passed
    facts = json.loads(Path(result.evidence.path).read_bytes())["steps"][0]["facts"]
    if case in ("human_fastq", "mouse_fastq"):
        assert facts["source_assembly_summary"] == [{"state": "not_applicable", "value": None, "count": 1}]
        assert facts["target_assembly_summary"] == [{"state": "resolved", "value": "mm10" if case == "mouse_fastq" else "hg38", "count": 1}]
        assert "alignment" in facts["prerequisite_codes"]
        assert "alignment" not in facts["required_information_codes"]
        assert facts["barcode_source_counts"]["fastq_read"] == 1
    if case == "human_bam":
        assert facts["assembly_compatibility_counts"]["match"] == 1
        assert facts["barcode_source_counts"]["bam_cell_identifier"] == 1
    if case == "mismatch_bam":
        assert facts["assembly_compatibility_counts"]["mismatch"] == 1
        assert facts["n_harmonization_required"] == 1
        assert facts["preparation_summary"] == [{"code": "assembly_harmonization", "admissibility": "unresolved", "count": 1}]
    if case == "unknown_bam":
        assert facts["source_assembly_summary"] == [{"state": "unknown", "value": None, "count": 1}]
        assert facts["target_assembly_summary"] == [{"state": "resolved", "value": "hg38", "count": 1}]
    before = {p: p.read_bytes() for p in Path(result.workspace_path).rglob("*") if p.is_file()}
    forbidden = Mock()
    forbidden.plan.side_effect = AssertionError("Planner called during resume")
    resumed = ResearchAgentApplication(tmp_path / "workspace", planner=forbidden, registry=raw_context.registry).resume(result.run_id)
    assert resumed == result
    assert before == {p: p.read_bytes() for p in Path(result.workspace_path).rglob("*") if p.is_file()}
    assert raw_context.production.call_count == raw_context.model.calls == 1
    forbidden.plan.assert_not_called()
    assert len(list(workspace.scientific.glob("raw-scatac-intake-*.json"))) == 1


@pytest.mark.parametrize("when", ["before_composition", "resume"])
def test_application_source_drift_fails_composition(intake_case, raw_context, tmp_path, monkeypatch, when):
    source = intake_case()
    app = ResearchAgentApplication(tmp_path / "workspace", planner=raw_context.planner, registry=raw_context.registry)
    def drift():
        next(source.folder.iterdir()).write_bytes(b"changed after verified production")
    original = service.build_analysis_evidence
    if when == "before_composition":
        def changed(*args, **kwargs):
            drift()
            return original(*args, **kwargs)
        monkeypatch.setattr(service, "build_analysis_evidence", changed)
    result = app.run(AgentRequest("raw-app", "Inspect raw inputs", {k: v for k, v in source.args.items() if k != "output_dir"}))
    if when == "resume":
        assert result.status.value == "SUCCEEDED"
        drift()
        result = app.resume(result.run_id)
    assert result.status.value == "FAILED" and result.run_status.value == "SUCCEEDED"
    assert result.error.code == "APP_EVIDENCE_FAILED"
    assert result.report is None
    assert raw_context.production.call_count == 1


def test_raw_application_plan_only_still_has_no_artifacts(raw_context, tmp_path):
    app = ResearchAgentApplication(tmp_path / "workspace", planner=raw_context.planner, registry=raw_context.registry)
    result = app.run(AgentRequest("planned", "Inspect raw BAM", {"raw_input_paths": "/missing/file.bam"}, RunMode.PLAN_ONLY))
    assert result.status.value == "PLANNED" and not result.run_result.steps
    assert result.evidence is result.report is result.visualization is None
    assert raw_context.production.call_count == 0
    assert not list(tmp_path.rglob("raw-scatac-intake-*.json"))


def test_full_fastq_application_without_pysam(intake_case, tmp_path):
    source = intake_case()
    inputs = {k: v for k, v in source.args.items() if k != "output_dir"}
    code = r'''
import importlib.abc, json, sys
class NoPysam(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pysam' or fullname.startswith('pysam.'):
            raise ModuleNotFoundError('optional dependency intentionally blocked')
sys.meta_path.insert(0, NoPysam())
from agent.application import ResearchAgentApplication
from agent.orchestration import LLMPlanner
from agent.schemas import AgentRequest
class Model:
    model_id = 'scripted'
    def complete(self, **kwargs):
        return json.dumps({'schema_version': 4, 'decision': {'kind':'plan', 'steps':[
            {'step_id':'raw','tool':'inspect_raw_scATAC','sources':[], 'control_dependencies':[]}]}})
result = ResearchAgentApplication(sys.argv[1], planner=LLMPlanner(Model())).run(
    AgentRequest('no-pysam', 'Inspect raw FASTQ readiness', json.loads(sys.argv[2])))
assert result.status.value == 'SUCCEEDED', result
assert result.evidence and result.report and result.visualization is None
assert 'pysam' not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "workspace"), json.dumps(inputs)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_application_preserves_supported_preparation_readiness(intake_case, raw_context, tmp_path, monkeypatch):
    """Synthetic assessed M10.1 preparation; does not add a production repair route."""
    from agent.tools.data import raw_scatac, raw_scatac_manifest as m
    from agent.orchestration import raw_scatac_verifier
    original = raw_scatac._reconstruct
    def assessed(*args, **kwargs):
        manifest = original(*args, **kwargs)  # Still freshly observe current sources.
        group = manifest.groups[0]
        code = m.PreparationCode.INPUT_PREPARATION
        support = m.PreparationAdmissibility.SUPPORTED
        evidence = m.EvidenceRecord(m.EvidenceSource.OBSERVATION, m.EvidenceClaim.PREPARATION_ADMISSIBILITY,
            code.value + ":" + group.route.value + ":" + support.value,
            group_id=group.id, coverage_id=manifest.coverage[0].id)
        preparation = m.PreparationRequirement(group.id, code, admissibility=support,
            route=group.route, admissibility_evidence_ids=(evidence.id,))
        return m.validate_raw_intake_manifest(replace(manifest,
            evidence=manifest.evidence + (evidence,), repairs=(preparation,)))
    monkeypatch.setattr(raw_scatac, "_reconstruct", assessed)
    monkeypatch.setattr(raw_scatac_verifier, "_reconstruct", assessed)
    source = intake_case()
    app = ResearchAgentApplication(tmp_path / "workspace", planner=raw_context.planner, registry=raw_context.registry)
    result = app.run(AgentRequest("assessed", "Inspect raw inputs", {k: v for k, v in source.args.items() if k != "output_dir"}))
    assert result.status.value == "SUCCEEDED", result
    assert result.run_result.steps[0].result["readiness"] == "READY_WITH_REPAIRS"
    assert result.evidence and result.report and result.visualization is None
    assert "Supported preparation actions remain" in Path(result.report.path).read_text()
    facts = json.loads(Path(result.evidence.path).read_bytes())["steps"][0]["facts"]
    assert facts["group_readiness_counts"]["READY_WITH_REPAIRS"] == 1
    assert facts["preparation_summary"] == [{"code": "input_preparation", "admissibility": "supported", "count": 1}]
    assert raw_context.production.call_count == 1
