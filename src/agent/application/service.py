"""Thin end-to-end composition of runtime and verified reporting APIs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from agent.orchestration import (
    AgentRuntime,
    FileRunStore,
    LLMPlanner,
    PlanExecutor,
    Planner,
    PlannerError,
    PlanningModelError,
    PlanningModelProfile,
    PlanningRecoveryPolicy,
    ToolRegistry,
)
from agent.providers import (
    PlanningModelFactoryError,
    PlanningModelFactoryRegistry,
    build_default_planning_model_factory_registry,
)
from agent.report import (
    ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
    ANALYSIS_EVIDENCE_FILENAME,
    ANALYSIS_REPORT_ARTIFACT_TYPE,
    ANALYSIS_REPORT_BUNDLE_DIRNAME,
    ANALYSIS_REPORT_FILENAME,
    ANALYSIS_REPORT_MANIFEST_FILENAME,
    ANALYSIS_VISUALIZATION_ARTIFACT_TYPE,
    ANALYSIS_VISUALIZATION_BUNDLE_DIRNAME,
    ANALYSIS_VISUALIZATION_MANIFEST_FILENAME,
    build_analysis_evidence,
    build_analysis_report,
    build_analysis_visualizations,
    get_supported_visualization_kinds,
    verify_analysis_evidence,
    verify_analysis_report,
    verify_analysis_visualizations,
)
from agent.schemas import (
    AgentPlan,
    AgentRequest,
    AgentRunResult,
    CancellationReceipt,
    ErrorCategory,
    RunStatus,
)

from .schemas import (
    ApplicationError,
    ApplicationResult,
    ApplicationStage,
    ApplicationStatus,
    ArtifactReference,
)
from .workspace import ApplicationWorkspaceError, ManagedWorkspace, RunWorkspace


RESERVED_APPLICATION_INPUTS = frozenset(
    {
        "output_dir",
        "workspace_root",
        "run_store_root",
        "run_state_dir",
        "scientific_output_dir",
        "evidence_output_dir",
        "visualization_output_dir",
        "report_output_dir",
        "composition_lock",
    }
)


_APPLICATION_MESSAGES = {
    "APP_REQUEST_INVALID": "Application request configuration is invalid.",
    "APP_WORKSPACE_INVALID": "Application workspace is invalid or unavailable.",
    "APP_OUTPUT_CONFLICT": "Application output conflicts with existing managed data.",
    "APP_COMPOSITION_ACTIVE": "Post-run composition is already active for this run.",
    "APP_EVIDENCE_FAILED": "Verified analysis evidence could not be completed.",
    "APP_VISUALIZATION_FAILED": "Verified analysis visualization could not be completed.",
    "APP_REPORT_FAILED": "Verified analysis report could not be completed.",
    "APP_PLANNING_CONFIGURATION_CONFLICT": (
        "Explicit Planner injection cannot be combined with application-owned "
        "planning-model configuration."
    ),
    "PLANNING_MODEL_PROFILE_REQUIRED": (
        "A primary planning-model profile is required for an LLM-planned new run."
    ),
    "PLANNING_MODEL_PROVIDER_UNKNOWN": (
        "The selected planning-model provider is not registered."
    ),
    "PLANNING_MODEL_PROFILE_DISABLED": (
        "The selected planning-model profile is disabled."
    ),
    "PLANNING_MODEL_CAPABILITY_UNSUPPORTED": (
        "The selected planning model lacks required structured output."
    ),
    "PLANNING_PROVIDER_DEPENDENCY_MISSING": (
        "A required planning-provider dependency is unavailable."
    ),
    "PLANNING_PROVIDER_CONFIGURATION_FAILED": (
        "The planning provider could not be initialized from its configuration."
    ),
}

_STABLE_PLANNING_CONFIGURATION_CODES = frozenset(
    {
        "PLANNING_MODEL_PROVIDER_UNKNOWN",
        "PLANNING_MODEL_PROFILE_DISABLED",
        "PLANNING_MODEL_CAPABILITY_UNSUPPORTED",
        "PLANNING_PROVIDER_DEPENDENCY_MISSING",
        "PLANNING_PROVIDER_CONFIGURATION_FAILED",
    }
)


class ApplicationServiceError(ValueError):
    """Pre-runtime application validation failure with sanitized details."""

    def __init__(self, error: ApplicationError) -> None:
        self.error = error
        super().__init__(error.message)


class _StageFailure(Exception):
    def __init__(
        self,
        code: str,
        stage: ApplicationStage,
        cause: BaseException | None = None,
    ) -> None:
        self.code = code
        self.stage = stage
        self.cause = cause
        super().__init__(
            _APPLICATION_MESSAGES.get(code, "Application processing failed.")
        )


class _PlanningProfileRequiredPlanner:
    """Fail closed if callers bypass the application new-run configuration gate."""

    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        del request, registry
        raise PlannerError(
            "PLANNING_MODEL_PROFILE_REQUIRED",
            _APPLICATION_MESSAGES["PLANNING_MODEL_PROFILE_REQUIRED"],
            category=ErrorCategory.ENVIRONMENT_ERROR,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _app_error(code: str, stage: ApplicationStage) -> ApplicationError:
    message = _APPLICATION_MESSAGES.get(code, "Application processing failed safely.")
    return ApplicationError(code, message, stage)


def _planning_configuration_error(code: str) -> ApplicationServiceError:
    stable_code = (
        code
        if code in _STABLE_PLANNING_CONFIGURATION_CODES
        else "PLANNING_PROVIDER_CONFIGURATION_FAILED"
    )
    return ApplicationServiceError(_app_error(stable_code, ApplicationStage.RUNTIME))


def _validate_recovery_profile(
    profile: PlanningModelProfile,
    registry: PlanningModelFactoryRegistry,
) -> None:
    if not profile.enabled:
        raise _planning_configuration_error("PLANNING_MODEL_PROFILE_DISABLED")
    if not profile.supports_structured_output:
        raise _planning_configuration_error(
            "PLANNING_MODEL_CAPABILITY_UNSUPPORTED"
        )
    if profile.provider_id not in registry.provider_ids:
        raise _planning_configuration_error("PLANNING_MODEL_PROVIDER_UNKNOWN")


def _application_llm_planner(
    primary_profile: PlanningModelProfile,
    *,
    recovery_profile: PlanningModelProfile | None,
    model_factory_registry: PlanningModelFactoryRegistry | None,
    recovery_policy: PlanningRecoveryPolicy | None,
) -> LLMPlanner:
    if not isinstance(primary_profile, PlanningModelProfile):
        raise _planning_configuration_error(
            "PLANNING_PROVIDER_CONFIGURATION_FAILED"
        )
    if recovery_profile is not None and not isinstance(
        recovery_profile, PlanningModelProfile
    ):
        raise _planning_configuration_error(
            "PLANNING_PROVIDER_CONFIGURATION_FAILED"
        )
    if model_factory_registry is not None and not isinstance(
        model_factory_registry, PlanningModelFactoryRegistry
    ):
        raise _planning_configuration_error(
            "PLANNING_PROVIDER_CONFIGURATION_FAILED"
        )
    if recovery_policy is not None and not isinstance(
        recovery_policy, PlanningRecoveryPolicy
    ):
        raise _planning_configuration_error(
            "PLANNING_PROVIDER_CONFIGURATION_FAILED"
        )

    registry = (
        build_default_planning_model_factory_registry()
        if model_factory_registry is None
        else model_factory_registry
    )
    if recovery_profile is not None:
        _validate_recovery_profile(recovery_profile, registry)
    try:
        model = registry.create(primary_profile)
        return LLMPlanner(
            model,
            profile=primary_profile,
            recovery_policy=recovery_policy,
            recovery_profiles=(
                () if recovery_profile is None else (recovery_profile,)
            ),
            model_factory_registry=(
                None if recovery_profile is None else registry
            ),
        )
    except ApplicationServiceError:
        raise
    except (PlanningModelFactoryError, PlanningModelError) as exc:
        raise _planning_configuration_error(exc.code) from exc
    except Exception as exc:
        raise _planning_configuration_error(
            "PLANNING_PROVIDER_CONFIGURATION_FAILED"
        ) from exc


class ResearchAgentApplication:
    """Own one durable runtime and compose its verified post-run products."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        planner: Planner | None = None,
        primary_planning_profile: PlanningModelProfile | None = None,
        recovery_planning_profile: PlanningModelProfile | None = None,
        planning_model_factory_registry: PlanningModelFactoryRegistry | None = None,
        planning_recovery_policy: PlanningRecoveryPolicy | None = None,
        registry: ToolRegistry | None = None,
        executor: PlanExecutor | None = None,
    ) -> None:
        try:
            workspace = ManagedWorkspace(workspace_root)
        except ApplicationWorkspaceError as exc:
            raise ApplicationServiceError(
                _app_error(exc.code, ApplicationStage.RUNTIME)
            ) from exc
        application_planning_configured = any(
            value is not None
            for value in (
                primary_planning_profile,
                recovery_planning_profile,
                planning_model_factory_registry,
                planning_recovery_policy,
            )
        )
        if planner is not None and application_planning_configured:
            raise ApplicationServiceError(
                _app_error(
                    "APP_PLANNING_CONFIGURATION_CONFLICT",
                    ApplicationStage.RUNTIME,
                )
            )
        missing_primary_profile = planner is None and primary_planning_profile is None
        if planner is None and primary_planning_profile is not None:
            planner = _application_llm_planner(
                primary_planning_profile,
                recovery_profile=recovery_planning_profile,
                model_factory_registry=planning_model_factory_registry,
                recovery_policy=planning_recovery_policy,
            )
        elif missing_primary_profile:
            planner = _PlanningProfileRequiredPlanner()
        store = FileRunStore(workspace.run_state)
        runtime = AgentRuntime(
            planner=planner,
            registry=registry,
            executor=executor,
            run_store=store,
        )
        self._workspace = workspace
        self._run_store = store
        self._runtime = runtime
        self._missing_primary_profile = missing_primary_profile

    @property
    def workspace_root(self) -> Path:
        return self._workspace.root

    @property
    def runtime(self) -> AgentRuntime:
        return self._runtime

    @property
    def registry(self) -> ToolRegistry:
        return self._runtime.registry

    @property
    def run_store(self) -> FileRunStore:
        return self._run_store

    def run(self, request: AgentRequest) -> ApplicationResult:
        """Plan, execute, and compose verified reporting for one request."""

        if self._missing_primary_profile:
            raise ApplicationServiceError(
                _app_error(
                    "PLANNING_MODEL_PROFILE_REQUIRED",
                    ApplicationStage.RUNTIME,
                )
            )
        effective, run = self._prepare_request(request)
        result = self._runtime.run(effective)
        return self._complete(result, run)

    def resume(self, run_id: str) -> ApplicationResult:
        """Resume durable execution without planning, then compose reporting."""

        try:
            run = self._workspace.run_paths(run_id)
        except ApplicationWorkspaceError as exc:
            raise ApplicationServiceError(
                _app_error(exc.code, ApplicationStage.RUNTIME)
            ) from exc
        result = self._runtime.resume(run_id)
        return self._complete(result, run)

    def cancel(self, run_id: str) -> CancellationReceipt:
        """Delegate cooperative runtime cancellation without another state system."""

        return self._runtime.cancel(run_id)

    def _prepare_request(self, request: AgentRequest) -> tuple[AgentRequest, RunWorkspace]:
        if not isinstance(request, AgentRequest):
            raise ApplicationServiceError(
                _app_error("APP_REQUEST_INVALID", ApplicationStage.RUNTIME)
            )
        conflicting = RESERVED_APPLICATION_INPUTS.intersection(request.inputs)
        if conflicting or request.inputs.get("overwrite") is True:
            raise ApplicationServiceError(
                _app_error("APP_REQUEST_INVALID", ApplicationStage.RUNTIME)
            )
        run_id = f"{request.request_id}:run"
        try:
            run = self._workspace.run_paths(run_id)
        except ApplicationWorkspaceError as exc:
            raise ApplicationServiceError(
                _app_error(exc.code, ApplicationStage.RUNTIME)
            ) from exc
        inputs = dict(request.inputs)
        inputs["output_dir"] = str(run.scientific)
        return (
            AgentRequest(request.request_id, request.prompt, inputs, request.mode),
            run,
        )

    def _complete(
        self, run_result: AgentRunResult, run: RunWorkspace
    ) -> ApplicationResult:
        if run_result.status is RunStatus.PLANNED:
            return self._base_result(
                run_result, run, status=ApplicationStatus.PLANNED
            )
        if run_result.status is RunStatus.CANCELLED:
            return self._base_result(
                run_result, run, status=ApplicationStatus.CANCELLED
            )
        if run_result.status is not RunStatus.SUCCEEDED:
            return self._base_result(run_result, run, status=ApplicationStatus.FAILED)

        evidence: ArtifactReference | None = None
        visualization: ArtifactReference | None = None
        report: ArtifactReference | None = None
        failure: _StageFailure | None = None
        try:
            with self._workspace.composition_lease(run):
                evidence_path, evidence = self._evidence(run_result, run)
                try:
                    kinds = get_supported_visualization_kinds(
                        run_result, evidence_path, registry=self.registry
                    )
                except Exception as exc:
                    raise _StageFailure(
                        "APP_VISUALIZATION_FAILED",
                        ApplicationStage.VISUALIZATION,
                        exc,
                    ) from exc
                visualization_path: Path | None = None
                if kinds:
                    visualization_path, visualization = self._visualization(
                        run_result, evidence_path, run
                    )
                else:
                    self._empty_or_fail(
                        run.visualizations, ApplicationStage.VISUALIZATION
                    )
                report_path, report_candidate = self._report(
                    run_result,
                    evidence_path,
                    visualization_path,
                    run,
                )
                try:
                    verification = verify_analysis_report(
                        run_result,
                        evidence_path,
                        report_path,
                        registry=self.registry,
                        visualization=visualization_path,
                    )
                except Exception as exc:
                    raise _StageFailure(
                        "APP_REPORT_FAILED", ApplicationStage.REPORT, exc
                    ) from exc
                if not verification.passed:
                    raise _StageFailure(
                        "APP_REPORT_FAILED", ApplicationStage.REPORT
                    )
                report = report_candidate
        except ApplicationWorkspaceError as exc:
            failure = _StageFailure(
                exc.code,
                ApplicationStage.EVIDENCE,
                exc,
            )
        except _StageFailure as exc:
            failure = exc
        if failure is not None:
            return self._base_result(
                run_result,
                run,
                status=ApplicationStatus.FAILED,
                evidence=evidence,
                visualization=visualization,
                report=report,
                error=_app_error(failure.code, failure.stage),
            )
        return self._base_result(
            run_result,
            run,
            status=ApplicationStatus.SUCCEEDED,
            evidence=evidence,
            visualization=visualization,
            report=report,
        )

    def _evidence(
        self, run_result: AgentRunResult, run: RunWorkspace
    ) -> tuple[Path, ArtifactReference]:
        path = run.evidence / ANALYSIS_EVIDENCE_FILENAME
        try:
            if path.exists() or path.is_symlink():
                self._workspace.require_regular_file(path)
            else:
                self._empty_or_fail(run.evidence, ApplicationStage.EVIDENCE)
                result = build_analysis_evidence(
                    run_result, run.evidence, registry=self.registry
                )
                if Path(result["evidence_path"]).resolve() != path.resolve():
                    raise ValueError("Unexpected evidence output path.")
            verification = verify_analysis_evidence(
                run_result, path, registry=self.registry
            )
            if not verification.passed:
                raise ValueError("Evidence verification failed.")
            self._workspace.require_regular_file(path)
            return path, ArtifactReference(
                ANALYSIS_EVIDENCE_ARTIFACT_TYPE, str(path), _sha256_file(path)
            )
        except ApplicationWorkspaceError as exc:
            raise _StageFailure(
                exc.code, ApplicationStage.EVIDENCE, exc
            ) from exc
        except _StageFailure:
            raise
        except Exception as exc:
            raise _StageFailure(
                "APP_EVIDENCE_FAILED", ApplicationStage.EVIDENCE, exc
            ) from exc

    def _visualization(
        self,
        run_result: AgentRunResult,
        evidence_path: Path,
        run: RunWorkspace,
    ) -> tuple[Path, ArtifactReference]:
        bundle = run.visualizations / ANALYSIS_VISUALIZATION_BUNDLE_DIRNAME
        path = bundle / ANALYSIS_VISUALIZATION_MANIFEST_FILENAME
        try:
            if path.exists() or path.is_symlink():
                self._workspace.require_regular_file(path)
            else:
                self._empty_or_fail(
                    run.visualizations, ApplicationStage.VISUALIZATION
                )
                result = build_analysis_visualizations(
                    run_result,
                    evidence_path,
                    run.visualizations,
                    registry=self.registry,
                )
                if Path(result["manifest_path"]).resolve() != path.resolve():
                    raise ValueError("Unexpected visualization output path.")
            verification = verify_analysis_visualizations(
                run_result,
                evidence_path,
                path,
                registry=self.registry,
            )
            if not verification.passed:
                raise ValueError("Visualization verification failed.")
            self._workspace.require_regular_file(path)
            return path, ArtifactReference(
                ANALYSIS_VISUALIZATION_ARTIFACT_TYPE,
                str(path),
                _sha256_file(path),
            )
        except ApplicationWorkspaceError as exc:
            raise _StageFailure(
                exc.code, ApplicationStage.VISUALIZATION, exc
            ) from exc
        except _StageFailure:
            raise
        except Exception as exc:
            raise _StageFailure(
                "APP_VISUALIZATION_FAILED", ApplicationStage.VISUALIZATION, exc
            ) from exc

    def _report(
        self,
        run_result: AgentRunResult,
        evidence_path: Path,
        visualization_path: Path | None,
        run: RunWorkspace,
    ) -> tuple[Path, ArtifactReference]:
        bundle = run.report / ANALYSIS_REPORT_BUNDLE_DIRNAME
        manifest = bundle / ANALYSIS_REPORT_MANIFEST_FILENAME
        document = bundle / ANALYSIS_REPORT_FILENAME
        try:
            if manifest.exists() or manifest.is_symlink():
                self._workspace.require_regular_file(manifest)
            else:
                self._empty_or_fail(run.report, ApplicationStage.REPORT)
                result = build_analysis_report(
                    run_result,
                    evidence_path,
                    run.report,
                    registry=self.registry,
                    visualization=visualization_path,
                )
                if Path(result["manifest_path"]).resolve() != manifest.resolve():
                    raise ValueError("Unexpected report output path.")
            verification = verify_analysis_report(
                run_result,
                evidence_path,
                manifest,
                registry=self.registry,
                visualization=visualization_path,
            )
            if not verification.passed:
                raise ValueError("Report verification failed.")
            self._workspace.require_regular_file(manifest)
            self._workspace.require_regular_file(document)
            return manifest, ArtifactReference(
                ANALYSIS_REPORT_ARTIFACT_TYPE,
                str(document),
                _sha256_file(document),
            )
        except ApplicationWorkspaceError as exc:
            raise _StageFailure(
                exc.code, ApplicationStage.REPORT, exc
            ) from exc
        except _StageFailure:
            raise
        except Exception as exc:
            raise _StageFailure(
                "APP_REPORT_FAILED", ApplicationStage.REPORT, exc
            ) from exc

    def _empty_or_fail(self, path: Path, stage: ApplicationStage) -> None:
        try:
            self._workspace.require_empty_directory(path)
        except ApplicationWorkspaceError as exc:
            raise _StageFailure(
                exc.code,
                stage,
                exc,
            ) from exc

    @staticmethod
    def _base_result(
        run_result: AgentRunResult,
        run: RunWorkspace,
        *,
        status: ApplicationStatus,
        evidence: ArtifactReference | None = None,
        visualization: ArtifactReference | None = None,
        report: ArtifactReference | None = None,
        error: ApplicationError | None = None,
    ) -> ApplicationResult:
        return ApplicationResult(
            request_id=run_result.request_id,
            run_id=run_result.run_id,
            status=status,
            run_status=run_result.status,
            workspace_path=str(run.root),
            run_result=run_result,
            evidence=evidence,
            visualization=visualization,
            report=report,
            error=error,
        )


__all__ = [
    "ApplicationServiceError",
    "RESERVED_APPLICATION_INPUTS",
    "ResearchAgentApplication",
]
