"""Thin end-to-end composition of runtime and verified reporting APIs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from agent.orchestration import (
    AgentRuntime,
    FileRunStore,
    PlanExecutor,
    Planner,
    ToolRegistry,
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
from agent.schemas import AgentRequest, AgentRunResult, CancellationReceipt, RunStatus

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
}


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _app_error(code: str, stage: ApplicationStage) -> ApplicationError:
    message = _APPLICATION_MESSAGES.get(code, "Application processing failed safely.")
    return ApplicationError(code, message, stage)


class ResearchAgentApplication:
    """Own one durable runtime and compose its verified post-run products."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        planner: Planner | None = None,
        registry: ToolRegistry | None = None,
        executor: PlanExecutor | None = None,
    ) -> None:
        try:
            workspace = ManagedWorkspace(workspace_root)
        except ApplicationWorkspaceError as exc:
            raise ApplicationServiceError(
                _app_error(exc.code, ApplicationStage.RUNTIME)
            ) from exc
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
