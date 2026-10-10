"""Real tiny owner fixtures and an execution bound for UA3.5.4a.

No provider is constructed or scripted here. Gaussian embeddings qualify only
downstream software behavior, and synthetic QC makes no biological claim.
Fixture bodies are reused from the accepted offline tests without their model
harnesses. All generated artifacts belong in a fresh disposable directory.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path

from agent.orchestration.executor import ExecutionOutcome, PlanExecutor
from agent.schemas import AgentError, ErrorCategory, ExecutionTraceEvent, TraceEventType
from agent.schemas.orchestration import _serialize


_REPO = Path(__file__).resolve().parents[2]
_ALLOWED = frozenset({"cluster_cells", "select_scATAC_cells", "inspect_scATAC"})


def _fresh(root):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("UA3.5.4a fixture root must be fresh and empty.")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fixture_module(relative_path, name):
    """Load accepted fixture source; do not import any scripted model harness."""
    spec = importlib.util.spec_from_file_location(name, _REPO / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The accepted fixture source is unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def synthetic_qc_environment(*, bedtools="/usr/bin/bedtools"):
    """Temporarily admit only the explicit synthetic QC qualification.

    Use this context for both fixture construction and later selection turns;
    the existing owner rechecks QC qualification during ordinary execution.
    Operator configuration and environment are restored when the context exits.
    """
    names = ("AGENT_QC_BEDTOOLS", "AGENT_QC_ALLOW_SYNTHETIC", "AGENT_QC_RESOURCE_CATALOG")
    previous = {name: os.environ.get(name) for name in names}
    os.environ["AGENT_QC_BEDTOOLS"] = str(bedtools)
    os.environ["AGENT_QC_ALLOW_SYNTHETIC"] = "1"
    os.environ.pop("AGENT_QC_RESOURCE_CATALOG", None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def build_neighbors_fixture(root):
    """Create 32 ordered Gaussian rows, then run the real neighbors owner once."""
    from agent.tools.analysis.embedding_analysis import build_cell_neighbors

    root = _fresh(root)
    module = _fixture_module(
        "tests/tools/analysis/test_embedding_analysis.py", "ua354a_embedding_fixture_source")
    embedding_path, cell_ids_path, cell_ids = module.embedding_artifacts.__wrapped__(root)
    result = build_cell_neighbors(embedding_path, cell_ids_path, root / "neighbors")
    return _serialize(dict(
        execution_inputs=dict(analysis_path=result["analysis_path"]),
        result=result,
        metadata=dict(
            fixture="ua353-accepted-downstream-software-embedding",
            fixture_source="tests/tools/analysis/test_embedding_analysis.py:embedding_artifacts",
            producer="build_cell_neighbors",
            synthetic_embeddings=True,
            epizoo_inference=False,
            biological_qualification=False,
            ordered_cell_ids=cell_ids,
            n_cells=32,
            embedding_dim=512,
        ),
    ))


def build_selection_fixture(root, *, bedtools="/usr/bin/bedtools"):
    """Create genuine synthetic reference/import/QC artifacts without thresholds."""
    from agent.tools.data.scatac_barcode_qc import compute_scATAC_qc

    root = _fresh(root)
    module = _fixture_module("tests/barcode_qc/conftest.py", "ua354a_qc_fixture_source")
    with synthetic_qc_environment(bedtools=bedtools):
        make = module.fixture_factory.__wrapped__(root)
        arguments, rows, _reference, _qc, _import_arguments = make()
        result = compute_scATAC_qc(**arguments)
    if result["resource_qualification"] != "synthetic_only":
        raise ValueError("The fixture must retain its explicit synthetic qualification.")
    return _serialize(dict(
        execution_inputs=dict(
            barcode_qc_manifest_path=result["manifest_path"],
            barcode_qc_manifest_sha256=result["manifest_sha256"],
        ),
        result=result,
        metadata=dict(
            fixture="ua353-accepted-synthetic-barcode-qc",
            fixture_source="tests/barcode_qc/conftest.py:fixture_factory",
            producers=["import_scATAC_fragments", "compute_scATAC_qc"],
            resource_qualification="synthetic_only",
            biological_qualification=False,
            canonical_source_records=len(rows),
            omitted_required_parameters=["min_qc_fragment_records", "min_tss_enrichment"],
            supplied_scientific_thresholds=[],
        ),
    ))


class BoundedScienceExecutor(PlanExecutor):
    """Keep real preflight; run only the named tiny evaluation science.

    A plan containing another tool is withheld in full before any execution.
    The explicit evaluation error is evidence of this bound, never a semantic
    failure or an assertion that the requested science is unsupported by Agent.
    """

    def __init__(self, registry):
        super().__init__(registry)
        self.observations = []

    def execute(self, plan, **kwargs):
        preflight = self.preflight(plan)
        tools = tuple(step.tool_name for step in plan.steps)
        withheld = tuple(tool for tool in tools if tool not in _ALLOWED)
        delegated = not withheld or not preflight.passed
        self.observations.append(_serialize(dict(
            plan=plan.to_dict(), preflight=preflight.to_dict(),
            delegated=delegated, tool_names=tools, withheld_tools=withheld,
            evaluation_boundary="after_compilation_admission_and_preflight_before_tools",
        )))
        if delegated:
            return super().execute(plan, **kwargs)
        error = AgentError(
            category=ErrorCategory.ENVIRONMENT_ERROR,
            code="EVALUATION_EXECUTION_WITHHELD",
            message="UA3.5.4a captured successful preflight and withheld this plan before tools.",
            details=dict(evaluation_only=True, withheld_tools=withheld,
                         scientific_execution_count=0),
        )
        event = ExecutionTraceEvent(
            sequence=0,
            event_type=TraceEventType.PLAN_VALIDATION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=error.message,
            details=dict(preflight_passed=True, evaluation_only=True,
                         error_code=error.code, withheld_tools=withheld),
        )
        return ExecutionOutcome(step_results=(), errors=(error,), trace=(event,))
