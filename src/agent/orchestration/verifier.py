"""Lightweight orchestration verification for tool steps and completed runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

from agent.schemas import (
    AgentError,
    AgentPlan,
    ErrorCategory,
    PlanStep,
    StepExecutionResult,
    StepOutputRef,
    StepStatus,
    VerificationCheck,
    VerificationResult,
)

from .registry import ToolRegistry, UnknownToolError


class _VerificationChecks:
    def __init__(self) -> None:
        self.checks: list[VerificationCheck] = []
        self.failures: list[tuple[str, str, str]] = []

    def add(
        self,
        name: str,
        passed: bool,
        success_message: str,
        failure_message: str,
        error_code: str,
    ) -> None:
        message = success_message if passed else failure_message
        self.checks.append(VerificationCheck(name=name, passed=passed, message=message))
        if not passed:
            self.failures.append((error_code, failure_message, name))

    def result(
        self,
        *,
        target_type: str,
        target_id: str,
        step_id: str | None = None,
        tool_name: str | None = None,
    ) -> VerificationResult:
        if not self.failures:
            return VerificationResult(
                passed=True,
                target_type=target_type,
                target_id=target_id,
                checks=tuple(self.checks),
            )
        code, _, _ = self.failures[0]
        failed_names = tuple(name for _, _, name in self.failures)
        messages = "; ".join(message for _, message, _ in self.failures)
        return VerificationResult(
            passed=False,
            target_type=target_type,
            target_id=target_id,
            checks=tuple(self.checks),
            error=AgentError(
                category=ErrorCategory.VERIFICATION_ERROR,
                code=code,
                message=messages,
                step_id=step_id,
                tool_name=tool_name,
                details={"failed_checks": failed_names},
            ),
        )


def _plain_json_value(value: object, path: str = "result") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key.")
            converted[key] = _plain_json_value(nested, f"{path}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            _plain_json_value(nested, f"{path}[{index}]")
            for index, nested in enumerate(value)
        ]
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}.")


def _path_corresponds(left: object, right: object) -> bool:
    if not isinstance(left, (str, Path)) or not isinstance(right, (str, Path)):
        return False
    if not str(left).strip() or not str(right).strip():
        return False
    try:
        return Path(left).expanduser().resolve(strict=False) == Path(
            right
        ).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False


def _positive_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _verify_common_step(
    step: PlanStep,
    result: object,
    registry: ToolRegistry,
    checks: _VerificationChecks,
) -> Mapping[str, object] | None:
    try:
        spec = registry.get(step.tool_name)
    except UnknownToolError:
        checks.add(
            "tool_registered",
            False,
            "Tool is registered.",
            f"Tool {step.tool_name!r} is not registered for execution.",
            "UNKNOWN_TOOL",
        )
        try:
            _plain_json_value(result)
        except (TypeError, ValueError):
            checks.add(
                "result_lightweight",
                False,
                "Result is lightweight and JSON-safe.",
                "Result is not lightweight and JSON-safe.",
                "RESULT_NOT_LIGHTWEIGHT",
            )
        else:
            checks.add(
                "result_lightweight",
                True,
                "Result is lightweight and JSON-safe.",
                "Result is not lightweight and JSON-safe.",
                "RESULT_NOT_LIGHTWEIGHT",
            )
        return None

    checks.add(
        "tool_registered",
        True,
        f"Tool {step.tool_name!r} is registered.",
        "Tool is not registered.",
        "UNKNOWN_TOOL",
    )
    checks.add(
        "tool_identity",
        spec.name == step.tool_name,
        "Plan step tool identity matches the registered specification.",
        "Plan step tool identity does not match the registered specification.",
        "TOOL_IDENTITY_MISMATCH",
    )

    plain_result: Mapping[str, object] | None = None
    try:
        converted = _plain_json_value(result)
        if not isinstance(converted, Mapping):
            raise TypeError("result must be a mapping")
        plain_result = converted
    except (TypeError, ValueError):
        checks.add(
            "result_lightweight",
            False,
            "Result is lightweight and JSON-safe.",
            "Result is not a lightweight JSON-safe mapping.",
            "RESULT_NOT_LIGHTWEIGHT",
        )
    else:
        checks.add(
            "result_lightweight",
            True,
            "Result is a lightweight JSON-safe mapping.",
            "Result is not a lightweight JSON-safe mapping.",
            "RESULT_NOT_LIGHTWEIGHT",
        )

    try:
        contract_result = plain_result if plain_result is not None else result
        registry.validate_result(step.tool_name, contract_result)
    except (TypeError, ValueError):
        checks.add(
            "result_contract",
            False,
            "Result satisfies the authoritative registry contract.",
            "Result violates the authoritative registry contract.",
            "RESULT_CONTRACT_INVALID",
        )
    else:
        checks.add(
            "result_contract",
            True,
            "Result satisfies the authoritative registry contract.",
            "Result violates the authoritative registry contract.",
            "RESULT_CONTRACT_INVALID",
        )
    return plain_result


def _verify_inspection(
    resolved_arguments: Mapping[str, object],
    result: Mapping[str, object],
    checks: _VerificationChecks,
) -> None:
    checks.add(
        "input_path_matches",
        _path_corresponds(result.get("input_path"), resolved_arguments.get("path")),
        "Inspection result path matches the resolved input path.",
        "Inspection result input_path does not match the resolved path argument.",
        "RESULT_PATH_MISMATCH",
    )
    checks.add(
        "positive_cell_count",
        _positive_integer(result.get("n_cells")),
        "Inspection reports a positive cell count.",
        "Inspection n_cells must be a positive integer.",
        "RESULT_VALUE_INVALID",
    )
    checks.add(
        "positive_feature_count",
        _positive_integer(result.get("n_features")),
        "Inspection reports a positive feature count.",
        "Inspection n_features must be a positive integer.",
        "RESULT_VALUE_INVALID",
    )

    nnz = result.get("nnz")
    nnz_valid = nnz is None or _nonnegative_integer(nnz)
    checks.add(
        "nonnegative_nnz",
        nnz_valid,
        "Inspection nnz is absent or nonnegative.",
        "Inspection nnz must be None or a nonnegative integer.",
        "RESULT_VALUE_INVALID",
    )

    density = result.get("density")
    density_valid = density is None or (
        not isinstance(density, bool)
        and isinstance(density, (int, float))
        and math.isfinite(float(density))
        and 0.0 <= float(density) <= 1.0
    )
    checks.add(
        "density_range",
        density_valid,
        "Inspection density is absent or finite within [0, 1].",
        "Inspection density must be None or finite within [0, 1].",
        "RESULT_VALUE_INVALID",
    )

    is_sparse = result.get("x_is_sparse")
    if is_sparse is True:
        sparse_metadata_valid = (
            _nonnegative_integer(nnz) and density_valid and density is not None
        )
        n_cells = result.get("n_cells")
        n_features = result.get("n_features")
        if (
            sparse_metadata_valid
            and _positive_integer(n_cells)
            and _positive_integer(n_features)
        ):
            expected_density = int(nnz) / (int(n_cells) * int(n_features))
            sparse_metadata_valid = math.isclose(
                float(density), expected_density, rel_tol=1e-12, abs_tol=1e-15
            )
    elif is_sparse is False:
        sparse_metadata_valid = nnz is None and density is None
    else:
        sparse_metadata_valid = False
    checks.add(
        "sparse_metadata_coherent",
        sparse_metadata_valid,
        "Sparse storage metadata is internally coherent.",
        "Sparse storage metadata is internally inconsistent.",
        "RESULT_METADATA_INCONSISTENT",
    )


def _artifact_checks(
    result: Mapping[str, object], checks: _VerificationChecks
) -> None:
    for field_name in ("embedding_path", "cell_ids_path"):
        value = result.get(field_name)
        path = Path(value) if isinstance(value, str) and value.strip() else None
        try:
            exists_as_file = path is not None and path.is_file()
        except OSError:
            exists_as_file = False
        checks.add(
            f"{field_name}_exists",
            exists_as_file,
            f"Artifact {field_name} exists as a regular file.",
            f"Artifact {field_name} is missing or is not a regular file.",
            "ARTIFACT_MISSING",
        )
        nonempty = False
        if exists_as_file and path is not None:
            try:
                nonempty = path.stat().st_size > 0
            except OSError:
                nonempty = False
        checks.add(
            f"{field_name}_nonempty",
            nonempty,
            f"Artifact {field_name} is nonempty.",
            f"Artifact {field_name} is empty or unavailable.",
            "ARTIFACT_EMPTY",
        )


def _verify_embedding(
    step: PlanStep,
    resolved_arguments: Mapping[str, object],
    result: Mapping[str, object],
    dependency_results: Mapping[str, Mapping[str, object]],
    checks: _VerificationChecks,
) -> None:
    checks.add(
        "backend_identity",
        result.get("backend") == "EpiZoo",
        "Embedding result identifies the registered EpiZoo backend.",
        "Embedding result backend identity is not EpiZoo.",
        "TOOL_IDENTITY_MISMATCH",
    )
    if "species" in resolved_arguments:
        resolved_species = resolved_arguments.get("species")
        species_matches = (
            isinstance(resolved_species, str)
            and isinstance(result.get("species"), str)
            and result["species"].casefold() == resolved_species.strip().casefold()
        )
        checks.add(
            "species_matches",
            species_matches,
            "Embedding result species matches the resolved argument.",
            "Embedding result species does not match the resolved argument.",
            "RESULT_IDENTITY_MISMATCH",
        )
    checks.add(
        "success_status",
        result.get("status") == "success",
        "Embedding tool reports success.",
        "Embedding result status must be 'success'.",
        "RESULT_STATUS_INVALID",
    )
    checks.add(
        "input_path_matches",
        _path_corresponds(
            result.get("input_path"), resolved_arguments.get("input_path")
        ),
        "Embedding result path matches the resolved input path.",
        "Embedding result input_path does not match the resolved input_path argument.",
        "RESULT_PATH_MISMATCH",
    )
    checks.add(
        "positive_cell_count",
        _positive_integer(result.get("n_cells")),
        "Embedding result reports a positive cell count.",
        "Embedding n_cells must be a positive integer.",
        "RESULT_VALUE_INVALID",
    )
    checks.add(
        "positive_embedding_dimension",
        _positive_integer(result.get("embedding_dim")),
        "Embedding dimension is positive.",
        "Embedding dimension must be a positive integer.",
        "RESULT_VALUE_INVALID",
    )
    checks.add(
        "finite_embeddings",
        result.get("finite") is True,
        "Embedding result reports finite values.",
        "Embedding result does not report finite values.",
        "RESULT_VALUE_INVALID",
    )
    checks.add(
        "cell_order_preserved",
        result.get("cell_order_preserved") is True,
        "Embedding result reports preserved cell order.",
        "Embedding result does not report preserved cell order.",
        "RESULT_VALUE_INVALID",
    )
    _artifact_checks(result, checks)

    input_reference = step.arguments.get("input_path")
    if (
        isinstance(input_reference, StepOutputRef)
        and input_reference.output_key == "input_path"
        and input_reference.step_id in step.depends_on
    ):
        inspection_step_id = input_reference.step_id
    elif "inspect" in step.depends_on:
        inspection_step_id = "inspect"
    else:
        return
    inspection = dependency_results.get(inspection_step_id)
    checks.add(
        "inspection_dependency_available",
        isinstance(inspection, Mapping),
        "Verified inspection dependency result is available.",
        "Verified inspection dependency result is unavailable.",
        "DEPENDENCY_RESULT_MISSING",
    )
    if not isinstance(inspection, Mapping):
        return
    checks.add(
        "dependency_input_path_matches",
        _path_corresponds(result.get("input_path"), inspection.get("input_path")),
        "Embedding and inspection input paths match.",
        "Embedding input_path does not match inspection input_path.",
        "RESULT_PATH_MISMATCH",
    )
    checks.add(
        "dependency_cell_count_matches",
        result.get("n_cells") == inspection.get("n_cells"),
        "Embedding and inspection cell counts match.",
        "Embedding n_cells does not match inspection n_cells.",
        "CELL_COUNT_MISMATCH",
    )


def verify_step(
    step: PlanStep,
    resolved_arguments: Mapping[str, object],
    result: object,
    registry: ToolRegistry,
    *,
    dependency_results: Mapping[str, Mapping[str, object]] | None = None,
) -> VerificationResult:
    """Verify one lightweight result without invoking scientific tools."""

    if not isinstance(step, PlanStep):
        raise TypeError("`step` must be a PlanStep.")
    if not isinstance(resolved_arguments, Mapping):
        raise TypeError("`resolved_arguments` must be a mapping.")
    if not isinstance(registry, ToolRegistry):
        raise TypeError("`registry` must be a ToolRegistry.")
    if dependency_results is None:
        dependency_results = {}
    elif not isinstance(dependency_results, Mapping):
        raise TypeError("`dependency_results` must be a mapping.")

    checks = _VerificationChecks()
    plain_result = _verify_common_step(step, result, registry, checks)
    if plain_result is not None:
        if step.tool_name == "inspect_scATAC":
            _verify_inspection(resolved_arguments, plain_result, checks)
        elif step.tool_name == "epizoo_embed_cells":
            _verify_embedding(
                step,
                resolved_arguments,
                plain_result,
                dependency_results,
                checks,
            )
    return checks.result(
        target_type="step",
        target_id=step.step_id,
        step_id=step.step_id,
        tool_name=step.tool_name,
    )


def _status_is_consistent(result: StepExecutionResult) -> bool:
    if result.status is StepStatus.SUCCEEDED:
        return result.result is not None and result.error is None
    if result.status is StepStatus.FAILED:
        if result.error is None:
            return False
        if result.verification is not None:
            return not result.verification.passed
        return result.result is None
    if result.status is StepStatus.SKIPPED:
        return result.result is None and (
            result.verification is None or not result.verification.passed
        )
    return False


def verify_run(
    plan: AgentPlan, step_results: Sequence[StepExecutionResult]
) -> VerificationResult:
    """Verify completed-run consistency without inspecting scientific files."""

    if not isinstance(plan, AgentPlan):
        raise TypeError("`plan` must be an AgentPlan.")
    if not isinstance(step_results, Sequence):
        raise TypeError("`step_results` must be a sequence.")

    checks = _VerificationChecks()
    if not all(isinstance(result, StepExecutionResult) for result in step_results):
        checks.add(
            "step_result_types",
            False,
            "All step results have the expected type.",
            "Every run result must be a StepExecutionResult.",
            "RUN_RESULT_INVALID",
        )
        return checks.result(target_type="run", target_id=plan.plan_id)
    checks.add(
        "step_result_types",
        True,
        "All step results have the expected type.",
        "Every run result must be a StepExecutionResult.",
        "RUN_RESULT_INVALID",
    )

    plan_by_id = {step.step_id: step for step in plan.steps}
    grouped: dict[str, list[StepExecutionResult]] = {}
    for result in step_results:
        grouped.setdefault(result.step_id, []).append(result)

    unexpected = tuple(step_id for step_id in grouped if step_id not in plan_by_id)
    checks.add(
        "no_unexpected_step_results",
        not unexpected,
        "No unexpected step results are present.",
        f"Unexpected step results are present: {unexpected}.",
        "UNEXPECTED_STEP_RESULT",
    )
    duplicates = tuple(step_id for step_id, values in grouped.items() if len(values) > 1)
    checks.add(
        "one_result_per_step",
        not duplicates,
        "No duplicate step results are present.",
        f"Duplicate step results are present: {duplicates}.",
        "DUPLICATE_STEP_RESULT",
    )
    missing = tuple(step_id for step_id in plan_by_id if step_id not in grouped)
    checks.add(
        "all_plan_steps_reported",
        not missing,
        "Every plan step has a result.",
        f"Plan steps are missing results: {missing}.",
        "MISSING_STEP_RESULT",
    )

    unique_results = {
        step_id: values[0]
        for step_id, values in grouped.items()
        if step_id in plan_by_id and len(values) == 1
    }
    for step in plan.steps:
        result = unique_results.get(step.step_id)
        if result is None:
            continue
        checks.add(
            f"{step.step_id}_tool_identity",
            result.tool_name == step.tool_name,
            f"Step {step.step_id!r} tool identity matches its plan.",
            f"Step {step.step_id!r} tool name does not match its plan.",
            "STEP_IDENTITY_MISMATCH",
        )
        checks.add(
            f"{step.step_id}_status_consistent",
            _status_is_consistent(result),
            f"Step {step.step_id!r} status and payload are internally consistent.",
            f"Step {step.step_id!r} status and payload are inconsistent.",
            "STEP_STATUS_INCONSISTENT",
        )
        checks.add(
            f"{step.step_id}_succeeded",
            result.status is StepStatus.SUCCEEDED,
            f"Step {step.step_id!r} succeeded.",
            f"Step {step.step_id!r} did not succeed.",
            "STEP_NOT_SUCCEEDED",
        )
        verification_passed = (
            result.verification is not None
            and result.verification.passed
            and result.verification.target_type == "step"
            and result.verification.target_id == step.step_id
        )
        checks.add(
            f"{step.step_id}_verification_passed",
            verification_passed,
            f"Step {step.step_id!r} has passed verification.",
            f"Step {step.step_id!r} lacks matching passed verification.",
            "STEP_VERIFICATION_FAILED",
        )

        dependency_results = [unique_results.get(name) for name in step.depends_on]
        if result.status is StepStatus.SUCCEEDED:
            dependency_consistent = all(
                dependency is not None and dependency.status is StepStatus.SUCCEEDED
                for dependency in dependency_results
            )
        elif result.status is StepStatus.SKIPPED and step.depends_on:
            dependency_consistent = any(
                dependency is not None and dependency.status is not StepStatus.SUCCEEDED
                for dependency in dependency_results
            )
        else:
            dependency_consistent = True
        checks.add(
            f"{step.step_id}_dependencies_consistent",
            dependency_consistent,
            f"Step {step.step_id!r} dependency completion is consistent.",
            f"Step {step.step_id!r} dependency completion is inconsistent.",
            "DEPENDENCY_INCONSISTENT",
        )

    return checks.result(target_type="run", target_id=plan.plan_id)


__all__ = ["verify_run", "verify_step"]
