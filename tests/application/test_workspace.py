from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent.application import ApplicationWorkspaceError, ManagedWorkspace


def test_workspace_uses_fixed_full_digest_layout(tmp_path: Path) -> None:
    workspace = ManagedWorkspace(tmp_path / "workspace")
    run_id = "../request/with/separators:run"
    run = workspace.run_paths(run_id)
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()

    assert run.root == workspace.root / "runs" / digest
    assert len(run.root.name) == 64
    assert run.scientific == run.root / "scientific"
    assert run.evidence == run.root / "evidence"
    assert run.visualizations == run.root / "visualizations"
    assert run.report == run.root / "report"
    assert run.composition_lock == run.root / "composition.lock"
    assert run.root.is_relative_to(workspace.root)


def test_same_run_id_maps_to_same_workspace(tmp_path: Path) -> None:
    workspace = ManagedWorkspace(tmp_path / "workspace")
    assert workspace.run_paths("request:run") == workspace.run_paths("request:run")


def test_distinct_run_ids_do_not_collide(tmp_path: Path) -> None:
    workspace = ManagedWorkspace(tmp_path / "workspace")
    assert workspace.run_paths("one:run").root != workspace.run_paths("two:run").root


def test_managed_directory_symlink_is_rejected(tmp_path: Path) -> None:
    workspace = ManagedWorkspace(tmp_path / "workspace")
    digest = workspace.run_digest("unsafe:run")
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace.runs / digest).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ApplicationWorkspaceError) as caught:
        workspace.run_paths("unsafe:run")

    assert caught.value.code == "APP_WORKSPACE_INVALID"


def test_managed_wrong_type_is_rejected(tmp_path: Path) -> None:
    workspace = ManagedWorkspace(tmp_path / "workspace")
    digest = workspace.run_digest("wrong-type:run")
    (workspace.runs / digest).write_text("not a directory", encoding="utf-8")

    with pytest.raises(ApplicationWorkspaceError) as caught:
        workspace.run_paths("wrong-type:run")

    assert caught.value.code == "APP_WORKSPACE_INVALID"


def test_workspace_root_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ApplicationWorkspaceError) as caught:
        ManagedWorkspace(link)

    assert caught.value.code == "APP_WORKSPACE_INVALID"


def test_composition_lease_is_fail_fast(tmp_path: Path) -> None:
    workspace = ManagedWorkspace(tmp_path / "workspace")
    run = workspace.run_paths("locked:run")

    with workspace.composition_lease(run):
        with pytest.raises(ApplicationWorkspaceError) as caught:
            with workspace.composition_lease(run):
                pass

    assert caught.value.code == "APP_COMPOSITION_ACTIVE"


def test_nonempty_stage_directory_is_an_output_conflict(tmp_path: Path) -> None:
    workspace = ManagedWorkspace(tmp_path / "workspace")
    run = workspace.run_paths("occupied:run")
    (run.report / "unexpected").write_text("partial", encoding="utf-8")

    with pytest.raises(ApplicationWorkspaceError) as caught:
        workspace.require_empty_directory(run.report)

    assert caught.value.code == "APP_OUTPUT_CONFLICT"
