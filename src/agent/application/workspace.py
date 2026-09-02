"""Deterministic trusted-local workspace management for the application."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import stat
from typing import Iterator


class ApplicationWorkspaceError(ValueError):
    """Sanitized managed-workspace failure with a stable application code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class RunWorkspace:
    root: Path
    scientific: Path
    evidence: Path
    visualizations: Path
    report: Path
    composition_lock: Path


class ManagedWorkspace:
    """Manage fixed paths beneath one caller-approved local workspace.

    The workspace is intended for a trusted local filesystem.  Fixed names,
    full run-ID hashes, symlink checks, and containment checks reduce accidental
    escape; this is not a claim of race-free operation against a hostile user
    concurrently mutating the filesystem.
    """

    def __init__(self, root: str | Path) -> None:
        if not isinstance(root, (str, Path)):
            raise ApplicationWorkspaceError(
                "APP_WORKSPACE_INVALID",
                "Application workspace must be a filesystem path.",
            )
        candidate = Path(root).expanduser()
        if candidate.is_symlink():
            raise ApplicationWorkspaceError(
                "APP_WORKSPACE_INVALID",
                "Application workspace must not be a symbolic link.",
            )
        try:
            candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ApplicationWorkspaceError(
                "APP_WORKSPACE_INVALID",
                "Application workspace could not be initialized safely.",
            ) from exc
        if not resolved.is_dir():
            raise ApplicationWorkspaceError(
                "APP_WORKSPACE_INVALID",
                "Application workspace must be a directory.",
            )
        self._root = resolved
        self._run_state = self._ensure_directory(resolved / "run_state")
        self._runs = self._ensure_directory(resolved / "runs")

    @property
    def root(self) -> Path:
        return self._root

    @property
    def run_state(self) -> Path:
        return self._run_state

    @property
    def runs(self) -> Path:
        return self._runs

    @staticmethod
    def run_digest(run_id: str) -> str:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ApplicationWorkspaceError(
                "APP_REQUEST_INVALID",
                "Application run ID must be a non-empty string.",
            )
        return hashlib.sha256(run_id.encode("utf-8")).hexdigest()

    def run_paths(self, run_id: str) -> RunWorkspace:
        digest = self.run_digest(run_id)
        root = self._ensure_directory(self._runs / digest)
        scientific = self._ensure_directory(root / "scientific")
        evidence = self._ensure_directory(root / "evidence")
        visualizations = self._ensure_directory(root / "visualizations")
        report = self._ensure_directory(root / "report")
        lock = root / "composition.lock"
        self._assert_contained(lock)
        if lock.is_symlink() or (lock.exists() and not lock.is_file()):
            raise ApplicationWorkspaceError(
                "APP_WORKSPACE_INVALID",
                "Application composition lock has an invalid filesystem type.",
            )
        return RunWorkspace(root, scientific, evidence, visualizations, report, lock)

    def require_regular_file(self, path: Path) -> Path:
        self._assert_contained(path)
        if path.is_symlink() or not path.is_file():
            raise ApplicationWorkspaceError(
                "APP_OUTPUT_CONFLICT",
                "A managed application artifact has an unexpected filesystem type.",
            )
        return path

    def require_empty_directory(self, path: Path) -> None:
        directory = self._ensure_directory(path)
        try:
            occupied = next(directory.iterdir(), None) is not None
        except OSError as exc:
            raise ApplicationWorkspaceError(
                "APP_WORKSPACE_INVALID",
                "A managed application directory could not be inspected safely.",
            ) from exc
        if occupied:
            raise ApplicationWorkspaceError(
                "APP_OUTPUT_CONFLICT",
                "A managed application output contains incomplete or conflicting data.",
            )

    @contextmanager
    def composition_lease(self, run: RunWorkspace) -> Iterator[None]:
        """Acquire the fail-fast local lock for post-run composition."""

        self._assert_contained(run.composition_lock)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(run.composition_lock, flags, 0o600)
        except OSError as exc:
            raise ApplicationWorkspaceError(
                "APP_WORKSPACE_INVALID",
                "Application composition lock could not be opened safely.",
            ) from exc
        acquired = False
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ApplicationWorkspaceError(
                    "APP_WORKSPACE_INVALID",
                    "Application composition lock must be a regular file.",
                )
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise ApplicationWorkspaceError(
                        "APP_WORKSPACE_INVALID",
                        "Application composition lock could not be acquired safely.",
                    ) from exc
                raise ApplicationWorkspaceError(
                    "APP_COMPOSITION_ACTIVE",
                    "Application post-run composition is already active for this run.",
                ) from exc
            yield
        finally:
            try:
                if acquired:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _assert_contained(self, path: Path) -> None:
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(self._root)
        except (OSError, ValueError) as exc:
            raise ApplicationWorkspaceError(
                "APP_WORKSPACE_INVALID",
                "Application-managed path escapes the approved workspace.",
            ) from exc

    def _ensure_directory(self, path: Path) -> Path:
        self._assert_contained(path)
        if path.is_symlink():
            raise ApplicationWorkspaceError(
                "APP_WORKSPACE_INVALID",
                "Application-managed directories must not be symbolic links.",
            )
        try:
            if path.exists():
                if not path.is_dir():
                    raise ApplicationWorkspaceError(
                        "APP_WORKSPACE_INVALID",
                        "An application-managed path is not a directory.",
                    )
            else:
                path.mkdir(mode=0o700)
        except ApplicationWorkspaceError:
            raise
        except OSError as exc:
            raise ApplicationWorkspaceError(
                "APP_WORKSPACE_INVALID",
                "Application-managed directory could not be initialized safely.",
            ) from exc
        if path.is_symlink():
            raise ApplicationWorkspaceError(
                "APP_WORKSPACE_INVALID",
                "Application-managed directories must not be symbolic links.",
            )
        resolved = path.resolve(strict=True)
        self._assert_contained(resolved)
        return resolved


__all__ = [
    "ApplicationWorkspaceError",
    "ManagedWorkspace",
    "RunWorkspace",
]
