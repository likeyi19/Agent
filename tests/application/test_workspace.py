from __future__ import annotations

import hashlib
import json
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


def test_interaction_presentation_identity_is_scoped_and_read_only_until_created(tmp_path: Path) -> None:
    workspace = ManagedWorkspace(tmp_path / 'workspace')
    session, turn = '../a/session', '../a/turn'
    expected = hashlib.sha256(json.dumps([session, turn], ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()
    root = workspace.interaction_presentation_root(session, turn)
    assert root == workspace.root / 'presentations' / expected
    assert not root.parent.exists()
    assert workspace.interaction_presentation_root(session, turn, create=True) == root
    assert root.is_dir() and root.stat().st_mode & 0o777 == 0o700
    sentinel = root / 'uncommitted-bundle'
    sentinel.write_bytes(b'No stored response grants no delivery authority.')
    assert workspace.interaction_presentation_root(session, turn) == root
    assert sentinel.read_bytes() == b'No stored response grants no delivery authority.'
    assert workspace.interaction_presentation_root('different-session', turn) != root
    assert workspace.interaction_presentation_root(session, 'different-turn') != root
    assert workspace.interaction_presentation_root('ab', 'c') != workspace.interaction_presentation_root('a', 'bc')


@pytest.mark.parametrize('level', ['parent', 'owner'])
@pytest.mark.parametrize('create', [False, True])
def test_interaction_presentation_rejects_symlink_directories(tmp_path: Path, level: str, create: bool) -> None:
    workspace = ManagedWorkspace(tmp_path / 'workspace')
    root = workspace.interaction_presentation_root('session', 'turn')
    target = workspace.root / 'unrelated'
    target.mkdir()
    if level == 'owner':
        root.parent.mkdir()
    (root if level == 'owner' else root.parent).symlink_to(target, target_is_directory=True)
    with pytest.raises(ApplicationWorkspaceError):
        workspace.interaction_presentation_root('session', 'turn', create=create)
    assert list(target.iterdir()) == []


@pytest.mark.parametrize('level', ['parent', 'owner'])
def test_interaction_presentation_rejects_wrong_directory_types(tmp_path: Path, level: str) -> None:
    workspace = ManagedWorkspace(tmp_path / 'workspace')
    root = workspace.interaction_presentation_root('session', 'turn')
    if level == 'owner':
        root.parent.mkdir()
    target = root if level == 'owner' else root.parent
    target.write_bytes(b'Conflicting file')
    for create in (False, True):
        with pytest.raises(ApplicationWorkspaceError):
            workspace.interaction_presentation_root('session', 'turn', create=create)
    assert target.read_bytes() == b'Conflicting file'
