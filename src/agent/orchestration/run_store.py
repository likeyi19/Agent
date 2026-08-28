"""Process-safe local persistence for durable Agent run state."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import ContextManager, Iterator, Mapping, Protocol, runtime_checkable

from agent.schemas.run_state import (
    PersistedRunState,
    RUN_STATE_SCHEMA_VERSION,
    RunLifecycleStatus,
)
from agent.schemas import StepStatus


RUN_STATE_FORMAT = "agent.run-state"


class RunStoreError(RuntimeError):
    """Base class for durable run-store failures."""


class RunNotFoundError(RunStoreError):
    """Raised when a requested durable run does not exist."""


class RunAlreadyExistsError(RunStoreError):
    """Raised when a durable run ID has already been created."""


class RunAlreadyActiveError(RunStoreError):
    """Raised when another runtime currently owns a durable run."""


class RunStateCorruptionError(RunStoreError):
    """Raised when persisted state is malformed or internally inconsistent."""


class RunStateVersionError(RunStoreError):
    """Raised when persisted state uses an unsupported schema version."""


class RunStateConflictError(RunStoreError):
    """Raised when an optimistic revision check detects a concurrent update."""


@runtime_checkable
class RunStore(Protocol):
    """Minimal persistence contract used by AgentRuntime."""

    def execution_lease(self, run_id: str) -> ContextManager[None]: ...

    def create(self, state: PersistedRunState) -> PersistedRunState: ...

    def load(self, run_id: str) -> PersistedRunState: ...

    def update(
        self,
        state: PersistedRunState,
        *,
        expected_revision: int,
    ) -> PersistedRunState: ...


class _DuplicateJsonKey(ValueError):
    pass


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant {value!r} is not permitted.")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _record_digest(record: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(record)).hexdigest()


def _safe_run_name(run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("`run_id` must be a non-empty string.")
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()


_RUN_TRANSITIONS = {
    RunLifecycleStatus.PLANNING: frozenset(
        {
            RunLifecycleStatus.PLANNING,
            RunLifecycleStatus.VALIDATED,
            RunLifecycleStatus.PLANNED,
            RunLifecycleStatus.FAILED,
        }
    ),
    RunLifecycleStatus.VALIDATED: frozenset(
        {
            RunLifecycleStatus.VALIDATED,
            RunLifecycleStatus.RUNNING,
            RunLifecycleStatus.FAILED,
        }
    ),
    RunLifecycleStatus.RUNNING: frozenset(
        {
            RunLifecycleStatus.RUNNING,
            RunLifecycleStatus.SUCCEEDED,
            RunLifecycleStatus.FAILED,
            RunLifecycleStatus.INTERRUPTED,
        }
    ),
    RunLifecycleStatus.FAILED: frozenset({RunLifecycleStatus.FAILED}),
    RunLifecycleStatus.PLANNED: frozenset(),
    RunLifecycleStatus.SUCCEEDED: frozenset(),
    RunLifecycleStatus.INTERRUPTED: frozenset(),
}


_STEP_TRANSITIONS = {
    StepStatus.PENDING: frozenset(
        {StepStatus.PENDING, StepStatus.RUNNING, StepStatus.FAILED, StepStatus.SKIPPED}
    ),
    StepStatus.RUNNING: frozenset(
        {StepStatus.RUNNING, StepStatus.SUCCEEDED, StepStatus.FAILED}
    ),
    StepStatus.SUCCEEDED: frozenset({StepStatus.SUCCEEDED}),
    StepStatus.FAILED: frozenset({StepStatus.FAILED}),
    StepStatus.SKIPPED: frozenset({StepStatus.SKIPPED}),
}


def _validate_update_transition(
    current: PersistedRunState, next_state: PersistedRunState
) -> None:
    if current.request != next_state.request:
        raise RunStateConflictError("Durable request identity cannot change.")
    if current.plan is not None and current.plan != next_state.plan:
        raise RunStateConflictError("A persisted plan cannot be removed or changed.")
    if (
        current.plan_fingerprint is not None
        and current.plan_fingerprint != next_state.plan_fingerprint
    ):
        raise RunStateConflictError("A persisted plan fingerprint cannot change.")
    if next_state.lifecycle_status not in _RUN_TRANSITIONS[current.lifecycle_status]:
        raise RunStateConflictError(
            f"Illegal durable lifecycle transition {current.lifecycle_status.value} "
            f"-> {next_state.lifecycle_status.value}."
        )
    if len(next_state.trace) < len(current.trace) or next_state.trace[
        : len(current.trace)
    ] != current.trace:
        raise RunStateConflictError("Durable trace history cannot be changed or removed.")
    if not current.steps:
        return
    if len(current.steps) != len(next_state.steps):
        raise RunStateConflictError("Persisted step set cannot change after validation.")
    next_by_id = {result.step_id: result for result in next_state.steps}
    for previous in current.steps:
        following = next_by_id.get(previous.step_id)
        if following is None or following.tool_name != previous.tool_name:
            raise RunStateConflictError("Persisted step identity cannot change.")
        if following.status not in _STEP_TRANSITIONS[previous.status]:
            raise RunStateConflictError(
                f"Illegal persisted step transition for {previous.step_id!r}: "
                f"{previous.status.value} -> {following.status.value}."
            )


class FileRunStore:
    """Versioned JSON RunStore with stable locks and atomic replacement."""

    def __init__(self, root: str | Path) -> None:
        if not isinstance(root, (str, Path)):
            raise TypeError("`root` must be a string or pathlib.Path.")
        resolved = Path(root).expanduser().resolve()
        if resolved.exists() and not resolved.is_dir():
            raise ValueError(f"Run-store root is not a directory: {resolved}")
        resolved.mkdir(parents=True, exist_ok=True)
        self._root = resolved

    @property
    def root(self) -> Path:
        return self._root

    def state_path(self, run_id: str) -> Path:
        """Return the safe JSON location for a run ID."""

        return self._root / f"{_safe_run_name(run_id)}.json"

    def lock_path(self, run_id: str) -> Path:
        """Return the stable lock-file location for a run ID."""

        return self._root / f"{_safe_run_name(run_id)}.lock"

    def lease_path(self, run_id: str) -> Path:
        """Return the stable execution-lease location for a run ID."""

        return self._root / f"{_safe_run_name(run_id)}.lease"

    @contextmanager
    def execution_lease(self, run_id: str) -> Iterator[None]:
        """Fail fast unless this process can exclusively own run execution.

        The lease removes its in-process reference on context exit. Its stable
        file may remain, but the operating system releases the flock on normal
        exit, exception unwinding, or process death.
        """

        path = self.lease_path(run_id)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        acquired = False
        try:
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                raise RunAlreadyActiveError(
                    f"Durable run {run_id!r} is already active in another runtime."
                ) from exc
            yield
        finally:
            try:
                if acquired:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    @contextmanager
    def _lock(self, run_id: str, *, exclusive: bool) -> Iterator[None]:
        path = self.lock_path(run_id)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(descriptor, mode)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def create(self, state: PersistedRunState) -> PersistedRunState:
        if not isinstance(state, PersistedRunState):
            raise TypeError("`state` must be a PersistedRunState.")
        if state.revision != 0:
            raise RunStateConflictError("A newly created run must use revision 0.")
        path = self.state_path(state.run_id)
        with self._lock(state.run_id, exclusive=True):
            if path.exists():
                raise RunAlreadyExistsError(
                    f"Durable run {state.run_id!r} already exists."
                )
            self._write_atomic(path, state)
        return state

    def load(self, run_id: str) -> PersistedRunState:
        path = self.state_path(run_id)
        with self._lock(run_id, exclusive=False):
            return self._load_path(path, expected_run_id=run_id)

    def update(
        self,
        state: PersistedRunState,
        *,
        expected_revision: int,
    ) -> PersistedRunState:
        if not isinstance(state, PersistedRunState):
            raise TypeError("`state` must be a PersistedRunState.")
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("`expected_revision` must be a nonnegative integer.")
        if state.revision != expected_revision + 1:
            raise RunStateConflictError(
                "Updated state revision must be exactly expected_revision + 1."
            )
        path = self.state_path(state.run_id)
        with self._lock(state.run_id, exclusive=True):
            current = self._load_path(path, expected_run_id=state.run_id)
            if current.revision != expected_revision:
                raise RunStateConflictError(
                    f"Durable run {state.run_id!r} is at revision "
                    f"{current.revision}, not expected revision {expected_revision}."
                )
            if current.created_at != state.created_at:
                raise RunStateConflictError("Durable run creation identity changed.")
            _validate_update_transition(current, state)
            self._write_atomic(path, state)
        return state

    def _load_path(self, path: Path, *, expected_run_id: str) -> PersistedRunState:
        try:
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            raise RunNotFoundError(
                f"Durable run {expected_run_id!r} was not found."
            ) from exc
        except OSError as exc:
            raise RunStoreError(f"Unable to read durable run state: {exc}") from exc
        try:
            decoded = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            _DuplicateJsonKey,
            RecursionError,
            ValueError,
        ) as exc:
            raise RunStateCorruptionError(
                f"Durable run state is malformed JSON: {exc}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise RunStateCorruptionError("Durable run envelope must be a JSON object.")
        expected_fields = {"format", "schema_version", "integrity", "record"}
        if set(decoded) != expected_fields:
            raise RunStateCorruptionError(
                "Durable run envelope fields do not match schema."
            )
        if decoded["format"] != RUN_STATE_FORMAT:
            raise RunStateCorruptionError("Durable run format identifier is invalid.")
        version = decoded["schema_version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != RUN_STATE_SCHEMA_VERSION
        ):
            raise RunStateVersionError(
                f"Unsupported durable run schema version {version!r}."
            )
        integrity = decoded["integrity"]
        if not isinstance(integrity, Mapping) or set(integrity) != {
            "algorithm",
            "digest",
        }:
            raise RunStateCorruptionError("Durable run integrity block is invalid.")
        if integrity["algorithm"] != "sha256" or not isinstance(
            integrity["digest"], str
        ):
            raise RunStateCorruptionError("Durable run integrity metadata is invalid.")
        record = decoded["record"]
        if _record_digest(record) != integrity["digest"]:
            raise RunStateCorruptionError("Durable run integrity digest does not match.")
        try:
            state = PersistedRunState.from_dict(record)
        except (TypeError, ValueError, KeyError) as exc:
            if "schema version" in str(exc).lower():
                raise RunStateVersionError(str(exc)) from exc
            raise RunStateCorruptionError(
                f"Durable run record violates its schema: {exc}"
            ) from exc
        if state.schema_version != version:
            raise RunStateVersionError(
                "Envelope and record schema versions do not match."
            )
        if state.run_id != expected_run_id:
            raise RunStateCorruptionError(
                "Durable run identity does not match its storage location."
            )
        return state

    def _write_atomic(self, path: Path, state: PersistedRunState) -> None:
        record = state.to_dict()
        envelope = {
            "format": RUN_STATE_FORMAT,
            "schema_version": RUN_STATE_SCHEMA_VERSION,
            "integrity": {
                "algorithm": "sha256",
                "digest": _record_digest(record),
            },
            "record": record,
        }
        payload = _canonical_json_bytes(envelope)
        temporary: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                dir=self._root,
                prefix=f".{path.stem}.",
                suffix=".tmp",
            )
            temporary = Path(name)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                raise
            os.replace(temporary, path)
            temporary = None
            os.chmod(path, 0o600)
            directory_descriptor = os.open(self._root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


__all__ = [
    "FileRunStore",
    "RUN_STATE_FORMAT",
    "RunAlreadyActiveError",
    "RunAlreadyExistsError",
    "RunNotFoundError",
    "RunStateConflictError",
    "RunStateCorruptionError",
    "RunStateVersionError",
    "RunStore",
    "RunStoreError",
]
