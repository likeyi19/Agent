"""Compact JSON-safe contracts for the research application boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent.schemas import AgentRunResult, RunStatus


class ApplicationStatus(str, Enum):
    PLANNED = "PLANNED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ApplicationStage(str, Enum):
    RUNTIME = "RUNTIME"
    EVIDENCE = "EVIDENCE"
    VISUALIZATION = "VISUALIZATION"
    REPORT = "REPORT"
    COMPLETE = "COMPLETE"


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a non-empty string.")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"`{name}` must be a SHA-256 hex digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"`{name}` must be a SHA-256 hex digest.") from exc
    return value


@dataclass(frozen=True)
class ArtifactReference:
    """One compact identity for an application-produced artifact."""

    artifact_type: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _non_empty(self.artifact_type, "artifact_type")
        _non_empty(self.path, "path")
        _sha256(self.sha256, "sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_type": self.artifact_type,
            "path": self.path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ApplicationError:
    """Sanitized application-composition failure."""

    code: str
    message: str
    stage: ApplicationStage

    def __post_init__(self) -> None:
        _non_empty(self.code, "code")
        _non_empty(self.message, "message")
        if not isinstance(self.stage, ApplicationStage):
            raise TypeError("`stage` must be an ApplicationStage.")

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage.value,
        }


@dataclass(frozen=True)
class ApplicationResult:
    """Lightweight result of one application run or resume operation."""

    request_id: str
    run_id: str
    status: ApplicationStatus
    run_status: RunStatus
    workspace_path: str
    run_result: AgentRunResult
    evidence: ArtifactReference | None = None
    visualization: ArtifactReference | None = None
    report: ArtifactReference | None = None
    error: ApplicationError | None = None

    def __post_init__(self) -> None:
        _non_empty(self.request_id, "request_id")
        _non_empty(self.run_id, "run_id")
        _non_empty(self.workspace_path, "workspace_path")
        if not isinstance(self.status, ApplicationStatus):
            raise TypeError("`status` must be an ApplicationStatus.")
        if not isinstance(self.run_status, RunStatus):
            raise TypeError("`run_status` must be a RunStatus.")
        if not isinstance(self.run_result, AgentRunResult):
            raise TypeError("`run_result` must be an AgentRunResult.")
        if self.request_id != self.run_result.request_id:
            raise ValueError("Application and runtime request IDs must match.")
        if self.run_id != self.run_result.run_id:
            raise ValueError("Application and runtime run IDs must match.")
        if self.run_status is not self.run_result.status:
            raise ValueError("Application and runtime statuses must match.")
        expected_runtime_status = {
            ApplicationStatus.PLANNED: RunStatus.PLANNED,
            ApplicationStatus.SUCCEEDED: RunStatus.SUCCEEDED,
            ApplicationStatus.CANCELLED: RunStatus.CANCELLED,
        }.get(self.status)
        if (
            expected_runtime_status is not None
            and self.run_status is not expected_runtime_status
        ):
            raise ValueError("Application status is inconsistent with runtime status.")
        for name in ("evidence", "visualization", "report"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ArtifactReference):
                raise TypeError(f"`{name}` must be an ArtifactReference or None.")
        if self.error is not None and not isinstance(self.error, ApplicationError):
            raise TypeError("`error` must be an ApplicationError or None.")
        if self.status in {
            ApplicationStatus.PLANNED,
            ApplicationStatus.CANCELLED,
        } and any(
            value is not None
            for value in (self.evidence, self.visualization, self.report)
        ):
            raise ValueError("Planned and cancelled results cannot contain report artifacts.")
        if self.status is ApplicationStatus.SUCCEEDED and (
            self.evidence is None or self.report is None or self.error is not None
        ):
            raise ValueError("Successful application results require evidence and a report.")
        if self.error is not None and self.status is not ApplicationStatus.FAILED:
            raise ValueError("Application errors are only valid for failed results.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "run_status": self.run_status.value,
            "workspace_path": self.workspace_path,
            "run_result": self.run_result.to_dict(),
            "evidence": None if self.evidence is None else self.evidence.to_dict(),
            "visualization": (
                None if self.visualization is None else self.visualization.to_dict()
            ),
            "report": None if self.report is None else self.report.to_dict(),
            "error": None if self.error is None else self.error.to_dict(),
        }


__all__ = [
    "ApplicationError",
    "ApplicationResult",
    "ApplicationStage",
    "ApplicationStatus",
    "ArtifactReference",
]
