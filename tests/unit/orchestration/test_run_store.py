"""Offline tests for strict, atomic durable run-state persistence."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pytest

from agent.orchestration import (
    AgentPlan,
    AgentRequest,
    FileRunStore,
    PersistedRunState,
    PlanStep,
    RunAlreadyExistsError,
    RunLifecycleStatus,
    RunStateConflictError,
    RunStateCorruptionError,
    RunStateVersionError,
)
from agent.schemas import RUN_STATE_SCHEMA_VERSION, fingerprint_plan


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _planning_state(request_id: str = "request-1") -> PersistedRunState:
    timestamp = _now()
    return PersistedRunState(
        schema_version=RUN_STATE_SCHEMA_VERSION,
        revision=0,
        run_id=f"{request_id}:run",
        request=AgentRequest(request_id, "inspect", {}),
        lifecycle_status=RunLifecycleStatus.PLANNING,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _planned_state() -> PersistedRunState:
    state = _planning_state()
    plan = AgentPlan(
        "plan-1",
        state.request.request_id,
        "fixed",
        (PlanStep("step", "tool", {}),),
    )
    return replace(state, plan=plan, plan_fingerprint=fingerprint_plan(plan))


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _rewrite_record(path: Path, mutate, *, recompute_digest: bool = True) -> None:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    mutate(envelope)
    if recompute_digest:
        envelope["integrity"]["digest"] = hashlib.sha256(
            _canonical(envelope["record"])
        ).hexdigest()
    path.write_bytes(_canonical(envelope))


def test_create_and_load_valid_state(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    original = _planning_state()

    stored = store.create(original)
    loaded = store.load(original.run_id)

    assert stored == original
    assert loaded == original
    assert store.state_path(original.run_id).stat().st_mode & 0o777 == 0o600
    assert original.run_id not in store.state_path(original.run_id).name


def test_duplicate_run_id_is_rejected(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    state = _planning_state()
    store.create(state)

    with pytest.raises(RunAlreadyExistsError, match="already exists"):
        store.create(state)


@pytest.mark.parametrize("payload", [b"{", b"not-json", b"\xff"])
def test_malformed_or_truncated_json_is_rejected(
    tmp_path: Path, payload: bytes
) -> None:
    store = FileRunStore(tmp_path)
    state = _planning_state()
    store.create(state)
    store.state_path(state.run_id).write_bytes(payload)

    with pytest.raises(RunStateCorruptionError, match="malformed JSON"):
        store.load(state.run_id)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    state = _planning_state()
    store.create(state)
    store.state_path(state.run_id).write_text(
        '{"format":"agent.run-state","format":"agent.run-state"}',
        encoding="utf-8",
    )

    with pytest.raises(RunStateCorruptionError, match="Duplicate JSON key"):
        store.load(state.run_id)


def test_bad_integrity_digest_is_rejected(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    state = _planning_state()
    store.create(state)
    _rewrite_record(
        store.state_path(state.run_id),
        lambda envelope: envelope["integrity"].__setitem__("digest", "0" * 64),
        recompute_digest=False,
    )

    with pytest.raises(RunStateCorruptionError, match="digest does not match"):
        store.load(state.run_id)


def test_unsupported_schema_version_is_rejected(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    state = _planning_state()
    store.create(state)
    _rewrite_record(
        store.state_path(state.run_id),
        lambda envelope: envelope.__setitem__("schema_version", 999),
    )

    with pytest.raises(RunStateVersionError, match="Unsupported"):
        store.load(state.run_id)


def test_illegal_lifecycle_combination_is_rejected(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    state = _planning_state()
    store.create(state)
    _rewrite_record(
        store.state_path(state.run_id),
        lambda envelope: envelope["record"].__setitem__(
            "lifecycle_status", "SUCCEEDED"
        ),
    )

    with pytest.raises(RunStateCorruptionError, match="requires a persisted plan"):
        store.load(state.run_id)


def test_plan_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    state = _planned_state()
    store.create(state)
    _rewrite_record(
        store.state_path(state.run_id),
        lambda envelope: envelope["record"]["plan"].__setitem__(
            "plan_id", "tampered-plan"
        ),
    )

    with pytest.raises(RunStateCorruptionError, match="fingerprint"):
        store.load(state.run_id)


def test_request_plan_and_run_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    state = _planned_state()
    store.create(state)

    def mutate(envelope) -> None:
        envelope["record"]["request"]["request_id"] = "different-request"
        plan = envelope["record"]["plan"]
        plan["request_id"] = "different-request"
        envelope["record"]["plan_fingerprint"] = hashlib.sha256(
            _canonical(plan)
        ).hexdigest()

    _rewrite_record(store.state_path(state.run_id), mutate)

    with pytest.raises(RunStateCorruptionError, match="run identity"):
        store.load(state.run_id)


def test_trace_sequence_corruption_is_rejected(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    state = _planning_state()
    store.create(state)

    def mutate(envelope) -> None:
        envelope["record"]["trace"] = [
            {
                "sequence": 4,
                "event_type": "PLANNING",
                "timestamp": _now(),
                "message": "bad sequence",
                "step_id": None,
                "attempt": None,
                "details": {},
            }
        ]

    _rewrite_record(store.state_path(state.run_id), mutate)

    with pytest.raises(RunStateCorruptionError, match="contiguous"):
        store.load(state.run_id)


def test_stale_revision_is_rejected(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    state = store.create(_planning_state())
    revision_one = replace(state, revision=1, updated_at=_now())
    store.update(revision_one, expected_revision=0)
    stale_update = replace(state, revision=1, updated_at=_now())

    with pytest.raises(RunStateConflictError, match="not expected revision"):
        store.update(stale_update, expected_revision=0)


def test_failed_replace_leaves_previous_revision_readable(
    tmp_path: Path, monkeypatch
) -> None:
    store = FileRunStore(tmp_path)
    state = store.create(_planning_state())
    revision_one = replace(state, revision=1, updated_at=_now())

    def fail_replace(source, destination) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        store.update(revision_one, expected_revision=0)

    assert store.load(state.run_id) == state
    assert not tuple(tmp_path.glob("*.tmp"))
