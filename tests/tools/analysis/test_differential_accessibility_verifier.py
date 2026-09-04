"""Independent Milestone 8.2-C verification and tamper regressions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.orchestration import differential_accessibility_verifier as verifier
from agent.application import ApplicationStatus, ResearchAgentApplication
from agent.orchestration import (
    AgentPlan,
    AgentRequest,
    AgentRuntime,
    DeterministicPlanner,
    ErrorCategory,
    FileRunStore,
    PlanStep,
    RecoveryDisposition,
    RecoveryPolicyIncompatibleError,
    RunLifecycleStatus,
    RunMode,
    RunStatus,
    StepStatus,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.report import verify_analysis_report
from agent.tools.analysis import differential_accessibility_backend as production
from agent.tools.analysis.replicate_pseudobulk import (
    build_replicate_pseudobulk,
    validate_scATAC_feature_space,
)


R_SCRIPT = Path("/home/likeyi/anaconda3/envs/agent-edger/bin/Rscript")
GROUP = "T"
NUMERATOR = "treated"
DENOMINATOR = "control"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_counts(n_samples: int = 6) -> np.ndarray:
    counts = np.zeros((n_samples, 40), dtype=np.int64)
    for sample in range(n_samples):
        counts[sample, :5] = 1
        counts[sample, 5:15] = 45 + sample + np.arange(10) % 3
        if sample % 2:
            counts[sample, 15:25] = 120 + sample + np.arange(10) % 5
            counts[sample, 25:35] = 25 + sample + np.arange(10) % 4
        else:
            counts[sample, 15:25] = 30 + sample + np.arange(10) % 5
            counts[sample, 25:35] = 100 + sample + np.arange(10) % 4
        counts[sample, 35:40] = 40 + sample + np.arange(5)
    return counts


def _source_and_pseudobulk(
    root: Path, *, design_type: str = "independent", low_replication: bool = False
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    n_selected = 4 if low_replication else 6
    selected = _selected_counts(n_selected)
    units: list[tuple[str, str, str, np.ndarray, int]] = [
        ("B", "excluded-group", DENOMINATOR, np.full(40, 20), 2),
    ]
    for index, values in enumerate(selected):
        condition = DENOMINATOR if index % 2 == 0 else NUMERATOR
        if design_type == "paired":
            replicate = f"pair-{index // 2 + 1}"
        else:
            replicate = f"{condition}-{index // 2 + 1}"
        cells = 1 if low_replication and index == 0 else 2
        units.append((GROUP, replicate, condition, values, cells))
    units.insert(
        2,
        (GROUP, "excluded-condition", "other", np.full(40, 30), 2),
    )
    rows: list[np.ndarray] = []
    observations: list[dict[str, str]] = []
    cell_ids: list[str] = []
    for unit_index, (group, replicate, condition, values, n_cells) in enumerate(units):
        pieces = [values] if n_cells == 1 else [values // 2, values - values // 2]
        for cell_index, row in enumerate(pieces):
            rows.append(np.asarray(row, dtype=np.int64))
            observations.append(
                {"group": group, "replicate": replicate, "condition": condition}
            )
            cell_ids.append(f"cell-{unit_index:02d}-{cell_index}")
    raw_path = root / "raw.h5ad"
    var = pd.DataFrame(
        {
            "chromosome": ["chr1"] * 20 + ["chr2"] * 20,
            "start": np.arange(40, dtype=np.int64) * 100,
            "end": np.arange(40, dtype=np.int64) * 100 + 50,
        },
        index=[f"peak-{index:03d}" for index in range(40)],
    )
    ad.AnnData(
        X=sparse.csr_matrix(np.asarray(rows, dtype=np.int64)),
        obs=pd.DataFrame(observations, index=cell_ids),
        var=var,
    ).write_h5ad(raw_path)
    feature = validate_scATAC_feature_space(
        raw_path,
        root / "feature",
        matrix_source="X",
        matrix_semantics="fragment_counts",
        species="human",
        genome_assembly="hg38",
        coordinate_source="var_columns",
        feature_chrom_key="chromosome",
        feature_start_key="start",
        feature_end_key="end",
        coordinate_system="zero_based_half_open",
    )
    pseudobulk = build_replicate_pseudobulk(
        feature["feature_space_path"],
        "replicate",
        "group",
        "condition",
        root / "pseudobulk",
        group_source="raw_obs",
    )
    return raw_path, Path(pseudobulk["pseudobulk_path"])


def _arguments(pseudobulk: Path, output: Path, design_type: str) -> dict[str, object]:
    return {
        "pseudobulk_path": str(pseudobulk),
        "group_value": GROUP,
        "condition_key": "condition",
        "numerator_condition": NUMERATOR,
        "denominator_condition": DENOMINATOR,
        "design_type": design_type,
        "output_dir": str(output),
    }


def _produce(
    root: Path, *, design_type: str = "independent", low_replication: bool = False
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    raw, pseudobulk = _source_and_pseudobulk(
        root, design_type=design_type, low_replication=low_replication
    )
    arguments = _arguments(pseudobulk, root / "da", design_type)
    result = dict(production.run_replicate_differential_accessibility(**arguments))
    return raw, pseudobulk, arguments, result


@pytest.fixture(autouse=True)
def _pinned_r(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("RUN_EDGER_INTEGRATION") != "1":
        pytest.skip("set RUN_EDGER_INTEGRATION=1 for M8.2-C verifier tests")
    if not R_SCRIPT.is_file():
        pytest.skip("the isolated agent-edger Rscript is unavailable")
    monkeypatch.setenv(production.EDGER_RSCRIPT_ENVIRONMENT_VARIABLE, str(R_SCRIPT))


@pytest.mark.parametrize("design_type", ["independent", "paired"])
def test_independent_verifier_reconstructs_and_recomputes_valid_artifact(
    tmp_path: Path, design_type: str
) -> None:
    _, pseudobulk, arguments, result = _produce(
        tmp_path, design_type=design_type
    )
    before = _sha(pseudobulk)

    metadata = verifier.verify_replicate_differential_accessibility(
        arguments, result
    )

    assert metadata.preparation_sha256 == result["preparation_sha256"]
    assert metadata.analysis_sha256 == result["analysis_sha256"]
    assert len(metadata.result_sha256) == 64
    assert metadata.verifier_r_script_sha256 == _sha(verifier.VERIFICATION_R_SCRIPT)
    assert metadata.verifier_r_script_sha256 != result["production_r_script_sha256"]
    assert _sha(pseudobulk) == before


def test_low_replication_and_one_cell_warnings_are_reconstructed(
    tmp_path: Path,
) -> None:
    _, _, arguments, result = _produce(tmp_path, low_replication=True)
    assert result["warning_codes"] == [
        "DA_LOW_REPLICATION",
        "DA_ONE_CELL_PSEUDOBULK",
    ]
    verifier.verify_replicate_differential_accessibility(arguments, result)


def _mutate_da(path: Path, mutation: str) -> None:
    artifact = ad.read_h5ad(path)
    provenance = artifact.uns[production.DA_PROVENANCE_KEY]
    if mutation == "coordinate":
        artifact.var.iloc[0, artifact.var.columns.get_loc("start")] += 1
    elif mutation == "inclusion":
        artifact.obs.iloc[0, artifact.obs.columns.get_loc("da_analysis_included")] = True
    elif mutation == "warning":
        provenance["comparison"]["warnings"] = []
    elif mutation == "design_digest":
        provenance["preparation"]["design_sha256"] = "0" * 64
    elif mutation == "contrast_digest":
        provenance["preparation"]["contrast_sha256"] = "0" * 64
    elif mutation == "filter_mask":
        artifact.var.iloc[0, artifact.var.columns.get_loc("da_status")] = "tested"
    elif mutation == "postfilter_library":
        artifact.obs.iloc[1, artifact.obs.columns.get_loc("da_postfilter_library_size")] += 1
    elif mutation == "tmm_factor":
        artifact.obs.iloc[1, artifact.obs.columns.get_loc("da_tmm_normalization_factor")] += 0.01
    elif mutation == "effective_library":
        artifact.obs.iloc[1, artifact.obs.columns.get_loc("da_effective_library_size")] += 1
    elif mutation in {"logFC", "logCPM", "F", "PValue", "FDR"}:
        tested = np.flatnonzero(artifact.var["da_status"].astype(str) == "tested")
        artifact.var.iloc[tested[0], artifact.var.columns.get_loc(mutation)] += 0.01
    elif mutation == "effect_direction":
        tested = np.flatnonzero(artifact.var["da_status"].astype(str) == "tested")
        artifact.var.iloc[
            tested[0], artifact.var.columns.get_loc("effect_direction")
        ] = "no_change"
    elif mutation == "production_script":
        provenance["backend"]["production_r_script_sha256"] = "0" * 64
    elif mutation == "backend_version":
        provenance["backend"]["versions"]["edger"] = "0.0.0"
    elif mutation == "unexpected_x":
        artifact.X = sparse.csr_matrix((artifact.n_obs, artifact.n_vars))
    elif mutation == "unexpected_layer":
        artifact.layers["counts"] = sparse.csr_matrix(
            (artifact.n_obs, artifact.n_vars)
        )
    elif mutation == "unexpected_slot":
        artifact.obsm["unexpected"] = np.zeros((artifact.n_obs, 1))
    else:  # pragma: no cover - parameter list is fixed
        raise AssertionError(mutation)
    artifact.uns[production.DA_PROVENANCE_KEY] = provenance
    artifact.write_h5ad(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "coordinate",
        "inclusion",
        "warning",
        "design_digest",
        "contrast_digest",
        "filter_mask",
        "postfilter_library",
        "tmm_factor",
        "effective_library",
        "logFC",
        "logCPM",
        "F",
        "PValue",
        "FDR",
        "effect_direction",
        "production_script",
        "backend_version",
        "unexpected_x",
        "unexpected_layer",
        "unexpected_slot",
    ],
)
def test_da_semantic_and_statistical_tampering_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _, _, arguments, result = _produce(fixture)
    output = tmp_path / "tampered"
    output.mkdir()
    copied = output / Path(str(result["da_path"])).name
    shutil.copyfile(str(result["da_path"]), copied)
    _mutate_da(copied, mutation)
    changed_arguments = dict(arguments)
    changed_arguments["output_dir"] = str(output)
    changed_result = deepcopy(result)
    changed_result["da_path"] = str(copied)
    changed_result["da_sha256"] = _sha(copied)

    with pytest.raises(verifier.DAVerificationError):
        verifier.verify_replicate_differential_accessibility(
            changed_arguments, changed_result
        )


def test_whole_file_tampering_fails_against_authoritative_result_digest(
    tmp_path: Path,
) -> None:
    _, _, arguments, result = _produce(tmp_path)
    Path(str(result["da_path"])).write_bytes(
        Path(str(result["da_path"])).read_bytes() + b"tamper"
    )
    with pytest.raises(verifier.DAVerificationError) as caught:
        verifier.verify_replicate_differential_accessibility(arguments, result)
    assert caught.value.code == "ARTIFACT_SHA256_MISMATCH"


@pytest.mark.parametrize("mutation", ["raw", "manifest", "row_order", "feature_order"])
def test_authoritative_m81_source_drift_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    raw, pseudobulk, arguments, result = _produce(tmp_path)
    if mutation == "raw":
        source = ad.read_h5ad(raw)
        source.uns["tampered"] = True
        source.write_h5ad(raw)
    elif mutation == "manifest":
        pseudobulk_artifact = ad.read_h5ad(pseudobulk)
        feature_path = Path(
            pseudobulk_artifact.uns["agent_milestone8_pseudobulk"][
                "source"
            ]["feature_space_path"]
        )
        feature_path.write_bytes(feature_path.read_bytes() + b" ")
    else:
        source = ad.read_h5ad(pseudobulk)
        if mutation == "row_order":
            source = source[list(reversed(range(source.n_obs))), :].copy()
        else:
            source = source[:, list(reversed(range(source.n_vars)))].copy()
        source.write_h5ad(pseudobulk)
    with pytest.raises(verifier.DAVerificationError):
        verifier.verify_replicate_differential_accessibility(arguments, result)


def test_source_mutation_during_verification_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pseudobulk, arguments, result = _produce(tmp_path)
    original = verifier._invoke_verifier_r

    def mutate_after_r(directory, preparation):
        recomputed = original(directory, preparation)
        pseudobulk.write_bytes(pseudobulk.read_bytes() + b"concurrent-change")
        return recomputed

    monkeypatch.setattr(verifier, "_invoke_verifier_r", mutate_after_r)
    with pytest.raises(verifier.DAVerificationError) as caught:
        verifier.verify_replicate_differential_accessibility(arguments, result)
    assert caught.value.code == "SOURCE_CHANGED_DURING_READ"


def test_verifier_never_invokes_production_callable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, arguments, result = _produce(tmp_path)
    monkeypatch.setattr(
        production,
        "run_replicate_differential_accessibility",
        lambda **_: pytest.fail("independent verifier invoked production DA"),
    )
    verifier.verify_replicate_differential_accessibility(arguments, result)


def test_verifier_script_identity_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, arguments, result = _produce(tmp_path)
    monkeypatch.setattr(
        verifier, "EXPECTED_VERIFICATION_R_SCRIPT_SHA256", "0" * 64
    )
    with pytest.raises(verifier.DAVerificationError) as caught:
        verifier.verify_replicate_differential_accessibility(arguments, result)
    assert caught.value.code == "R_PACKAGE_VERSION_INCOMPATIBLE"


class _FixedPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan_value = plan
        self.calls = 0

    def plan(self, request, registry):
        self.calls += 1
        return self.plan_value


class _InterruptedAfterDASuccess(BaseException):
    pass


class _InterruptAfterDASuccessStore:
    def __init__(self, delegate: FileRunStore) -> None:
        self.delegate = delegate
        self.triggered = False

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)

    def update(self, state, *, expected_revision):
        saved = self.delegate.update(state, expected_revision=expected_revision)
        if (
            not self.triggered
            and saved.lifecycle_status is RunLifecycleStatus.RUNNING
            and len(saved.steps) == 1
            and saved.steps[0].status is StepStatus.SUCCEEDED
        ):
            self.triggered = True
            raise _InterruptedAfterDASuccess()
        return saved


class _InterruptWhileDARunningStore:
    def __init__(self, delegate: FileRunStore) -> None:
        self.delegate = delegate
        self.triggered = False

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)

    def update(self, state, *, expected_revision):
        saved = self.delegate.update(state, expected_revision=expected_revision)
        if (
            not self.triggered
            and saved.lifecycle_status is RunLifecycleStatus.RUNNING
            and len(saved.steps) == 1
            and saved.steps[0].status is StepStatus.RUNNING
        ):
            self.triggered = True
            raise _InterruptedAfterDASuccess()
        return saved


def _da_plan(request_id: str, arguments: dict[str, object]) -> AgentPlan:
    return AgentPlan(
        f"{request_id}:da-plan",
        request_id,
        "fixed-da",
        (
            PlanStep(
                "da",
                "run_replicate_differential_accessibility",
                arguments,
            ),
        ),
    )


def _counting_da_registry(call: Mock) -> ToolRegistry:
    default = build_default_tool_registry()
    return ToolRegistry(
        tuple(
            replace(default.get(name), function=call)
            if name == "run_replicate_differential_accessibility"
            else default.get(name)
            for name in default.names()
        )
    )


def test_verified_da_checkpoint_is_reused_on_nonterminal_resume(
    tmp_path: Path,
) -> None:
    _, pseudobulk = _source_and_pseudobulk(tmp_path / "fixture")
    arguments = _arguments(pseudobulk, tmp_path / "output", "independent")
    request_id = "m82-resume"
    plan = _da_plan(request_id, arguments)
    default = build_default_tool_registry()
    call = Mock(
        wraps=default.get("run_replicate_differential_accessibility").function
    )
    registry = _counting_da_registry(call)
    store = FileRunStore(tmp_path / "store")
    with pytest.raises(_InterruptedAfterDASuccess):
        AgentRuntime(
            planner=_FixedPlanner(plan),
            registry=registry,
            run_store=_InterruptAfterDASuccessStore(store),
        ).run(AgentRequest(request_id, "fixed", {}))

    planner = _FixedPlanner(plan)
    result = AgentRuntime(
        planner=planner, registry=registry, run_store=store
    ).resume(f"{request_id}:run")

    assert result.status is RunStatus.SUCCEEDED
    assert result.steps[0].attempt_count == 1
    assert planner.calls == 0
    call.assert_called_once()


@pytest.mark.parametrize("mutation", ["source", "artifact", "backend"])
def test_nonterminal_resume_blocks_drift_without_reinvoking_production(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pseudobulk = _source_and_pseudobulk(tmp_path / "fixture")
    arguments = _arguments(pseudobulk, tmp_path / "output", "independent")
    request_id = f"m82-drift-{mutation}"
    plan = _da_plan(request_id, arguments)
    default = build_default_tool_registry()
    call = Mock(
        wraps=default.get("run_replicate_differential_accessibility").function
    )
    registry = _counting_da_registry(call)
    store = FileRunStore(tmp_path / "store")
    with pytest.raises(_InterruptedAfterDASuccess):
        AgentRuntime(
            planner=_FixedPlanner(plan),
            registry=registry,
            run_store=_InterruptAfterDASuccessStore(store),
        ).run(AgentRequest(request_id, "fixed", {}))
    state = store.load(f"{request_id}:run")
    if mutation == "source":
        pseudobulk.write_bytes(pseudobulk.read_bytes() + b"drift")
    elif mutation == "artifact":
        da_path = Path(state.steps[0].result["da_path"])
        da_path.write_bytes(da_path.read_bytes() + b"drift")
    else:
        monkeypatch.setattr(
            verifier,
            "_invoke_verifier_r",
            Mock(
                side_effect=verifier.DAVerificationError(
                    "EDGER_VERSION_UNSUPPORTED",
                    "compatible pinned edgeR is unavailable",
                )
            ),
        )

    result = AgentRuntime(
        planner=_FixedPlanner(plan), registry=registry, run_store=store
    ).resume(f"{request_id}:run")

    assert result.status is RunStatus.FAILED
    if mutation == "backend":
        assert len(result.errors) == 1
        error = result.errors[0]
        assert error.code == "EDGER_VERSION_UNSUPPORTED"
        assert error.category is ErrorCategory.ENVIRONMENT_ERROR
        assert (
            error.recovery_disposition
            is RecoveryDisposition.RESUME_WITH_COMPATIBLE_RUNTIME
        )
    else:
        assert any(
            error.code == "PERSISTED_STEP_REVALIDATION_FAILED"
            for error in result.errors
        )
    call.assert_called_once()


def test_stale_running_da_requires_manual_reconciliation(
    tmp_path: Path,
) -> None:
    _, pseudobulk = _source_and_pseudobulk(tmp_path / "fixture")
    arguments = _arguments(pseudobulk, tmp_path / "output", "independent")
    request_id = "m82-stale-running"
    plan = _da_plan(request_id, arguments)
    call = Mock(side_effect=AssertionError("stale DA was reinvoked"))
    registry = _counting_da_registry(call)
    store = FileRunStore(tmp_path / "store")
    with pytest.raises(_InterruptedAfterDASuccess):
        AgentRuntime(
            planner=_FixedPlanner(plan),
            registry=registry,
            run_store=_InterruptWhileDARunningStore(store),
        ).run(AgentRequest(request_id, "fixed", {}))

    result = AgentRuntime(
        planner=_FixedPlanner(plan), registry=registry, run_store=store
    ).resume(f"{request_id}:run")

    assert result.status is RunStatus.FAILED
    assert result.steps[0].status is StepStatus.FAILED
    assert any(
        error.recovery_disposition is RecoveryDisposition.MANUAL_RECONCILIATION
        for error in result.errors
    )
    call.assert_not_called()


def test_recovery_identity_drift_blocks_resume_before_execution(
    tmp_path: Path,
) -> None:
    _, pseudobulk = _source_and_pseudobulk(tmp_path / "fixture")
    arguments = _arguments(pseudobulk, tmp_path / "output", "independent")
    request_id = "m82-policy-drift"
    plan = _da_plan(request_id, arguments)
    default = build_default_tool_registry()
    call = Mock(
        wraps=default.get("run_replicate_differential_accessibility").function
    )
    registry = _counting_da_registry(call)
    store = FileRunStore(tmp_path / "store")
    with pytest.raises(_InterruptedAfterDASuccess):
        AgentRuntime(
            planner=_FixedPlanner(plan),
            registry=registry,
            run_store=_InterruptAfterDASuccessStore(store),
        ).run(AgentRequest(request_id, "fixed", {}))
    drifted = ToolRegistry(
        tuple(
            replace(
                registry.get(name),
                recovery_policy_version="run-replicate-da-v2",
            )
            if name == "run_replicate_differential_accessibility"
            else registry.get(name)
            for name in registry.names()
        )
    )
    with pytest.raises(RecoveryPolicyIncompatibleError):
        AgentRuntime(
            planner=_FixedPlanner(plan), registry=drifted, run_store=store
        ).resume(f"{request_id}:run")
    call.assert_called_once()


def test_plan_only_executes_no_python_or_r_da(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pseudobulk = _source_and_pseudobulk(tmp_path / "fixture")
    arguments = _arguments(pseudobulk, tmp_path / "output", "independent")
    request_id = "m82-plan-only"
    call = Mock(side_effect=AssertionError("PLAN_ONLY invoked Python DA"))
    registry = _counting_da_registry(call)
    r_call = Mock(side_effect=AssertionError("PLAN_ONLY invoked verifier R"))
    monkeypatch.setattr(verifier, "_invoke_verifier_r", r_call)

    result = AgentRuntime(
        planner=_FixedPlanner(_da_plan(request_id, arguments)), registry=registry
    ).run(AgentRequest(request_id, "fixed", {}, RunMode.PLAN_ONLY))

    assert result.status is RunStatus.PLANNED
    call.assert_not_called()
    r_call.assert_not_called()


def test_cancellation_during_da_checkpoints_success_then_wins(
    tmp_path: Path,
) -> None:
    _, pseudobulk = _source_and_pseudobulk(tmp_path / "fixture")
    arguments = _arguments(pseudobulk, tmp_path / "output", "independent")
    request_id = "m82-cancel-during"
    plan = _da_plan(request_id, arguments)
    store = FileRunStore(tmp_path / "store")
    default = build_default_tool_registry()
    production_call = Mock(
        wraps=default.get("run_replicate_differential_accessibility").function
    )

    def run_then_cancel(**kwargs):
        result = production_call(**kwargs)
        store.request_cancellation(f"{request_id}:run")
        return result

    registry = _counting_da_registry(run_then_cancel)
    result = AgentRuntime(
        planner=_FixedPlanner(plan), registry=registry, run_store=store
    ).run(AgentRequest(request_id, "fixed", {}))

    assert result.status is RunStatus.CANCELLED
    assert result.steps[0].status is StepStatus.SUCCEEDED
    production_call.assert_called_once()
    assert AgentRuntime(
        planner=_FixedPlanner(plan), registry=registry, run_store=store
    ).resume(f"{request_id}:run").status is RunStatus.CANCELLED


def test_cancellation_after_verified_predecessor_prevents_da_and_r(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pseudobulk = _source_and_pseudobulk(tmp_path / "fixture")
    arguments = _arguments(pseudobulk, tmp_path / "output", "independent")
    request_id = "m82-cancel-before-da"
    store = FileRunStore(tmp_path / "store")
    default = build_default_tool_registry()

    def inspect_then_cancel(**kwargs):
        result = default.get("inspect_scATAC").function(**kwargs)
        store.request_cancellation(f"{request_id}:run")
        return result

    da_call = Mock(side_effect=AssertionError("cancelled run invoked Python DA"))
    r_call = Mock(side_effect=AssertionError("cancelled run invoked verifier R"))
    monkeypatch.setattr(verifier, "_invoke_verifier_r", r_call)
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=inspect_then_cancel)
            if name == "inspect_scATAC"
            else replace(default.get(name), function=da_call)
            if name == "run_replicate_differential_accessibility"
            else default.get(name)
            for name in default.names()
        )
    )
    plan = AgentPlan(
        f"{request_id}:plan",
        request_id,
        "fixed",
        (
            PlanStep("inspect", "inspect_scATAC", {"path": str(pseudobulk)}),
            PlanStep(
                "da",
                "run_replicate_differential_accessibility",
                arguments,
                ("inspect",),
            ),
        ),
    )
    runtime = AgentRuntime(
        planner=_FixedPlanner(plan), registry=registry, run_store=store
    )
    result = runtime.run(AgentRequest(request_id, "fixed", {}))

    assert result.status is RunStatus.CANCELLED
    assert tuple(step.status for step in result.steps) == (
        StepStatus.SUCCEEDED,
        StepStatus.SKIPPED,
    )
    da_call.assert_not_called()
    r_call.assert_not_called()
    first = runtime.cancel(result.run_id)
    second = runtime.cancel(result.run_id)
    assert first.disposition == second.disposition


def test_figureless_evidence_report_and_application_paths(
    tmp_path: Path,
) -> None:
    _, pseudobulk = _source_and_pseudobulk(
        tmp_path / "fixture", low_replication=True
    )
    application = ResearchAgentApplication(
        tmp_path / "workspace", planner=DeterministicPlanner()
    )
    request = AgentRequest(
        "m82-application",
        "Run replicate-aware differential accessibility and produce a report.",
        {
            "pseudobulk_path": str(pseudobulk),
            "group_value": GROUP,
            "condition_key": "condition",
            "numerator_condition": NUMERATOR,
            "denominator_condition": DENOMINATOR,
            "design_type": "independent",
        },
    )

    result = application.run(request)

    assert result.status is ApplicationStatus.SUCCEEDED
    assert result.evidence is not None
    assert result.visualization is None
    assert result.report is not None
    evidence = json.loads(Path(result.evidence.path).read_text(encoding="utf-8"))
    da_step = next(
        step
        for step in evidence["steps"]
        if step["tool_name"] == "run_replicate_differential_accessibility"
    )
    facts = da_step["facts"]
    assert facts["positive_logfc_meaning"] == "higher_in_numerator"
    assert len(facts["result_sha256"]) == 64
    assert not {"logFC", "logCPM", "F", "PValue", "FDR"}.intersection(facts)
    report = Path(result.report.path).read_text(encoding="utf-8")
    assert "## Replicate-aware Differential Accessibility" in report
    assert '"DA_LOW_REPLICATION"' in report
    assert '"DA_ONE_CELL_PSEUDOBULK"' in report
    assert "significant peaks" not in report.casefold()
    assert verify_analysis_report(
        result.run_result,
        result.evidence.path,
        Path(result.report.path).parent / "report_manifest.json",
        registry=application.registry,
    ).passed

    report_bytes = Path(result.report.path).read_bytes()
    resumed = application.resume(result.run_id)
    assert resumed.status is ApplicationStatus.SUCCEEDED
    assert resumed.visualization is None
    assert Path(resumed.report.path).read_bytes() == report_bytes


def test_complete_raw_to_da_application_chain_is_figureless(
    tmp_path: Path,
) -> None:
    raw, _ = _source_and_pseudobulk(tmp_path / "fixture")
    application = ResearchAgentApplication(
        tmp_path / "workspace", planner=DeterministicPlanner()
    )
    result = application.run(
        AgentRequest(
            "m82-chained-application",
            "Run differential accessibility from raw scATAC and report it.",
            {
                "input_path": str(raw),
                "matrix_source": "X",
                "matrix_semantics": "fragment_counts",
                "species": "human",
                "genome_assembly": "hg38",
                "coordinate_source": "var_columns",
                "feature_chrom_key": "chromosome",
                "feature_start_key": "start",
                "feature_end_key": "end",
                "coordinate_system": "zero_based_half_open",
                "replicate_key": "replicate",
                "group_key": "group",
                "condition_key": "condition",
                "group_source": "raw_obs",
                "group_value": GROUP,
                "numerator_condition": NUMERATOR,
                "denominator_condition": DENOMINATOR,
                "design_type": "independent",
            },
        )
    )

    assert result.status is ApplicationStatus.SUCCEEDED
    assert result.visualization is None
    assert tuple(
        step.tool_name for step in result.run_result.plan.steps
    ) == (
        "validate_scATAC_feature_space",
        "build_replicate_pseudobulk",
        "run_replicate_differential_accessibility",
    )


@pytest.mark.parametrize("mutation", ["source", "artifact"])
def test_terminal_application_composition_rejects_da_drift(
    tmp_path: Path, mutation: str
) -> None:
    _, pseudobulk = _source_and_pseudobulk(tmp_path / "fixture")
    application = ResearchAgentApplication(
        tmp_path / "workspace", planner=DeterministicPlanner()
    )
    result = application.run(
        AgentRequest(
            f"m82-composition-{mutation}",
            "Run differential accessibility and report it.",
            {
                "pseudobulk_path": str(pseudobulk),
                "group_value": GROUP,
                "condition_key": "condition",
                "numerator_condition": NUMERATOR,
                "denominator_condition": DENOMINATOR,
                "design_type": "independent",
            },
        )
    )
    assert result.status is ApplicationStatus.SUCCEEDED
    if mutation == "source":
        target = pseudobulk
    else:
        target = Path(result.run_result.steps[0].result["da_path"])
    target.write_bytes(target.read_bytes() + b"drift")

    resumed = application.resume(result.run_id)

    assert resumed.status is ApplicationStatus.FAILED
    assert resumed.error is not None
    assert resumed.error.code == "APP_EVIDENCE_FAILED"
