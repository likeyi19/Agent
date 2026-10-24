"""Interaction publication uses real rendering over simulated accepted metadata.

The existing W3 source fixture establishes software/identity boundaries only.
No native scientific algorithm or paid provider executes in these checks.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.application.session_state import SessionConflictError, SessionError
from agent.application.sessions import AnalysisSessions
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.report import visualization
from agent.web.app import create_app
from test_dialogue_evidence import forbid_work
from test_interactive_boundary import PROFILES
from test_scientific_completion import overlay_service


REQUEST = 'Show the accepted UMAP colored by its exact cohort categories without recalculating it.'


def case(tmp_path, monkeypatch):
    seed, coords, _, _, science, _ = overlay_service(tmp_path, monkeypatch)
    guarded = forbid_work(seed._application, monkeypatch)
    models, rendered = [], []
    build = visualization.build_cell_overlay_visualizations

    def counted(*args, **kwargs):
        result = build(*args, **kwargs)
        rendered.append(result)
        return result

    monkeypatch.setattr(visualization, 'build_cell_overlay_visualizations', counted)

    class Model:
        def __init__(self, profile):
            self.model_id, self.phases = profile.model_id, []
            models.append(self)

        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            if 'turn_schema_version' in value:
                self.phases.append('interpretation')
                group, = [g for g in value['dialogue']['result_groups'] if g['tool'] == 'compute_cell_umap']
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind='answer_scientific',
                    target=dict(output=group['handle'], subject=None), comparison=None, focus='question',
                    visualization=dict(coordinates=group['handle'], labels=group['handle'], field='result:cohort'))))
            if 'dialogue_schema_version' in value:
                self.phases.append('generation')
                return json.dumps(dict(support='insufficient_evidence', paragraphs=[dict(
                    kind='limitation', text='No biological identity is established.', citations=[], comparisons=[])]))
            if 'grounding_review_version' in value:
                self.phases.append('review')
                return json.dumps(dict(integrity='intact', verdicts={p['id']: dict(verdict='unsupported', independent=True) for p in value['paragraphs']}))
            raise AssertionError('Read-only presentation entered a planning/provider phase')

    factory = PlanningModelFactoryRegistry({'scripted': Model})
    facade = InteractiveAgentApplication(guarded.workspace_root, model_profiles=PROFILES,
        default_profile_id='alpha', registry=guarded.registry, planning_model_factory_registry=factory)
    protected = {str(p): p.read_bytes() for root in (guarded._workspace.runs, guarded._workspace.run_state)
                 for p in root.rglob('*') if p.is_file()}
    before = guarded.sessions.load('session')
    return SimpleNamespace(facade=facade, coords=coords, source_bytes=coords.read_bytes(), science=science,
        models=models, rendered=rendered, before=before, protected=protected)


def submit(current):
    return current.facade.submit_turn('session', 'draw', REQUEST, expected_generation=1)


def unchanged_science(current):
    state = current.facade._application.sessions.load('session')
    assert state.turns == current.before.turns and state.revisions == current.before.revisions
    assert state.generation == current.before.generation and state.active_revision_id == current.before.active_revision_id
    assert current.coords.read_bytes() == current.source_bytes and not current.science
    assert all(Path(path).read_bytes() == content for path, content in current.protected.items())


def restarted(current):
    def forbidden(profile):
        pytest.fail('A persisted presentation read constructed a provider')
    return InteractiveAgentApplication(current.facade._application.workspace_root,
        model_profiles=PROFILES, default_profile_id='alpha', registry=current.facade._application.registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))


def test_standalone_figure_survives_review_withholding_save_reopen_and_duplicate(tmp_path, monkeypatch):
    current = case(tmp_path, monkeypatch)
    view = submit(current)
    assert view.response.version == 3 and len(view.response.artifacts) == 1, view.to_dict()
    assert view.status == 'answered' and len(current.rendered) == 1
    assert current.models[0].phases == ['interpretation', 'generation', 'review']
    assert 'could not ground' in view.response.text
    reference, = current.facade.presentation_artifact_handles('session', 'draw')
    assert reference.revision_id is None and reference.turn_id == 'draw'
    content = current.facade.presentation_artifact_content('session', 'draw', reference.handle)
    assert content.data.startswith(b'\x89PNG\r\n\x1a\n')
    assert hashlib.sha256(content.data).hexdigest() == reference.sha256
    assert reference.figure['publisher'] == dict(kind='interaction', session_id='session', turn_id='draw')
    assert str(tmp_path) not in json.dumps(view.to_dict()) + json.dumps(reference.to_dict())
    with TestClient(create_app(current.facade)) as client:
        endpoint = '/api/v1/sessions/session/turns/draw/artifacts'
        inventory = client.get(endpoint)
        assert inventory.status_code == 200 and inventory.json() == {'artifacts': [reference.to_dict()]}
        assert str(tmp_path) not in inventory.text
        delivered = client.get(f'{endpoint}/{reference.handle}')
        assert delivered.status_code == 200 and delivered.content == content.data
        assert delivered.headers['content-type'] == 'image/png'
        assert delivered.headers['x-artifact-source-sha256'] == reference.sha256
        assert delivered.headers['x-artifact-content-sha256'] == hashlib.sha256(delivered.content).hexdigest()
        assert delivered.headers['x-artifact-presentation'] == 'accepted_png'
        assert delivered.headers['content-disposition'].startswith('inline;')
        download = client.get(f'{endpoint}/{reference.handle}', params={'download': True})
        assert download.content == content.data and download.headers['content-disposition'].startswith('attachment;')
        wrong_owner = client.get(f'/api/v1/sessions/session/turns/other/artifacts/{reference.handle}')
        assert wrong_owner.status_code == 404
    unchanged_science(current)
    monkeypatch.setattr(visualization, 'build_cell_overlay_visualizations',
        lambda *a, **k: pytest.fail('Completed history regenerated a figure'))
    restored = restarted(current)
    assert restored.turn('session', 'draw') == view
    assert restored.submit_turn('session', 'draw', REQUEST, expected_generation=1) == view
    assert restored.recover_turn('session', 'draw', complete_presentation=True) == view
    assert restored.presentation_artifact_content('session', 'draw', reference.handle).data == content.data
    state = current.facade._application.sessions.load('session')
    interaction = state.interactions[-1]
    with pytest.raises(SessionConflictError):
        current.facade._application.sessions.compose_visualization('session', interaction)
    changed = view.response.to_dict() | dict(text='Changed historical display.')
    with pytest.raises(SessionConflictError):
        current.facade._application.sessions.store_presentation('session', 'draw', changed)
    assert current.facade.turn('session', 'draw') == view
    unchanged_science(current)


@pytest.mark.parametrize('mutation', ['png', 'manifest', 'png_symlink', 'bundle_symlink'])
def test_pinned_presentation_rejects_changed_bytes_and_no_follow_substitution(tmp_path, monkeypatch, mutation):
    current = case(tmp_path, monkeypatch)
    assert submit(current).response.artifacts
    reference, = current.facade.presentation_artifact_handles('session', 'draw')
    result, = current.rendered
    manifest, png = Path(result['manifest_path']), Path(result['figures'][0]['figure_path'])
    original_manifest, original_png = manifest.read_bytes(), png.read_bytes()
    moved = None
    try:
        if mutation == 'png':
            png.write_bytes(original_png + b'changed')
        elif mutation == 'manifest':
            manifest.write_bytes(original_manifest + b' ')
        elif mutation == 'png_symlink':
            copied = tmp_path / 'same-png-bytes.png'
            copied.write_bytes(original_png)
            png.unlink()
            png.symlink_to(copied)
        else:
            bundle = manifest.parent
            moved = tmp_path / 'detached-bundle'
            bundle.rename(moved)
            bundle.symlink_to(moved, target_is_directory=True)
        with pytest.raises(InteractiveBoundaryError):
            current.facade.presentation_artifact_content('session', 'draw', reference.handle)
    finally:
        if moved is not None:
            manifest.parent.unlink()
            moved.rename(manifest.parent)
        else:
            if png.is_symlink():
                png.unlink()
            png.write_bytes(original_png)
            manifest.write_bytes(original_manifest)
    assert current.facade.presentation_artifact_content('session', 'draw', reference.handle).data == original_png
    unchanged_science(current)


def test_explicit_pins_cannot_substitute_artifact_or_publisher_identity(tmp_path, monkeypatch):
    current = case(tmp_path, monkeypatch)
    view = submit(current)
    sessions = current.facade._application.sessions
    pins = _serialize(view.response.artifacts)
    with pytest.raises(SessionError):
        sessions.presentation_figure_inventory('session', 'draw', artifacts=[pins[0] | dict(artifact_id='f' * 64)])
    manifest = Path(current.rendered[0]['manifest_path'])
    original = manifest.read_bytes()
    payload = json.loads(original)
    payload['publisher']['turn_id'] = 'another-interaction'
    raw = json.dumps(payload).encode()
    forged = pins[0] | dict(manifest_sha256=hashlib.sha256(raw).hexdigest())
    forged['artifact_id'] = sessions._presentation_artifact_id('session', 'draw', forged)
    try:
        manifest.write_bytes(raw)
        with pytest.raises(SessionError, match='publisher'):
            sessions.presentation_figure_inventory('session', 'draw', artifacts=[forged])
    finally:
        manifest.write_bytes(original)
    unchanged_science(current)


def test_uncommitted_bundle_is_not_adopted_by_history_or_recovery(tmp_path, monkeypatch):
    current = case(tmp_path, monkeypatch)
    with monkeypatch.context() as interrupted:
        def fail_before_save(*args, **kwargs):
            raise RuntimeError('Scripted interruption before immutable response save')
        interrupted.setattr(AnalysisSessions, 'store_presentation', fail_before_save)
        with pytest.raises(InteractiveBoundaryError):
            submit(current)
    assert len(current.rendered) == 1
    state = current.facade._application.sessions.load('session')
    assert state.interactions[-1].presentation is None
    manifest = Path(current.rendered[0]['manifest_path'])
    assert manifest.is_file()
    assert current.facade.presentation_artifact_handles('session', 'draw') == ()
    pin = dict(manifest_sha256=current.rendered[0]['manifest_sha256'],
        png_sha256=current.rendered[0]['figures'][0]['png_sha256'],
        figure_id=current.rendered[0]['figures'][0]['figure_id'])
    artifact_id = current.facade._application.sessions._presentation_artifact_id('session', 'draw', pin)
    with pytest.raises(InteractiveBoundaryError):
        current.facade.presentation_artifact_content('session', 'draw', artifact_id)
    monkeypatch.setattr(visualization, 'build_cell_overlay_visualizations',
        lambda *a, **k: pytest.fail('Recovery regenerated or adopted an uncommitted bundle'))
    restored = restarted(current)
    closed = restored.recover_turn('session', 'draw', complete_presentation=True)
    assert closed.response.artifacts is None
    assert restored.presentation_artifact_handles('session', 'draw') == ()
    assert restored.submit_turn('session', 'draw', REQUEST, expected_generation=1) == closed
    assert manifest.is_file()  # Closing the interaction grants no artifact authority.
    unchanged_science(current)
