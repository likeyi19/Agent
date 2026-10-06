"""M18.4 projects accepted state through the facade without scientific work."""
from dataclasses import replace
import hashlib
import io
import json

import pytest

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.application.interactive_schemas import PresentedResponse
from agent.application.session_state import CompletionFile
from agent.providers import PlanningModelFactoryRegistry
from test_dialogue_evidence import app, accepted, forbid_work, files
from test_dialogue_detail import publication
from test_interactive_boundary import PROFILES, boundary, first
from test_service import _tiny_h5ad, _counting_registry
from test_scientific_dialogue import Model as ScientificModel, question


def read_only_boundary(application):
    def forbidden(profile):
        pytest.fail('Rich scientific reads constructed a provider.')
    return InteractiveAgentApplication(application.workspace_root,
        model_profiles=PROFILES, default_profile_id='alpha', registry=application.registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))


def test_revision_projection_navigation_and_exact_parent_metadata(app, monkeypatch):
    earlier = accepted(app, 'inspect_scATAC', overrides={'n_cells': 3})[0]
    later = accepted(app, 'inspect_scATAC', name='later', overrides={'n_cells': 5})[0]
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    before = files(app)
    old = facade.revision('session', earlier)
    current = facade.revision('session', later)
    assert not old.is_active and current.is_active
    assert current.parent_revision_id == earlier and current.base_generation == 1
    assert current.session_generation == 2 and current.run_id == 'later:run'
    assert current.outputs == ('output0',) and current.evidence_outputs == current.outputs
    assert current.steps[0].tool_name == 'inspect_scATAC' and current.status == 'succeeded'
    assert str(app.workspace_root) not in json.dumps(current.to_dict())
    assert files(app) == before
    switched = facade.activate_revision('session', 'return-earlier', earlier, expected_generation=2)
    assert switched.active_revision_id == earlier and switched.generation == 3
    assert [r.revision_id for r in switched.revisions] == [earlier, later]
    assert facade.revision('session', earlier).is_active
    nav = facade.turn('session', 'return-earlier')
    assert nav.status == 'navigated' and nav.response is None and nav.error is None and nav.utterance == ''
    assert facade.activate_revision('session', 'return-earlier', earlier, expected_generation=2) == switched
    assert app.sessions.load('session').revisions == facade._application.sessions.load('session').revisions
    assert len(app.sessions.load('session').navigation) == 3


@pytest.mark.parametrize('kind', ['stale', 'unknown_revision', 'turn_conflict', 'invalid_generation'])
def test_navigation_fails_closed_without_science(app, monkeypatch, kind):
    revision = accepted(app, 'inspect_scATAC')[0]
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    kwargs = dict(turn_id='nav', revision_id=revision, expected_generation=1)
    code = 'INTERACTIVE_GENERATION_CONFLICT'
    if kind == 'stale': kwargs['expected_generation'] = 0
    if kind == 'unknown_revision':
        kwargs['revision_id'], code = 'unknown', 'INTERACTIVE_REFERENCE_INVALID'
    if kind == 'turn_conflict':
        kwargs['turn_id'], code = 'initial', 'INTERACTIVE_TURN_CONFLICT'
    if kind == 'invalid_generation':
        kwargs['expected_generation'], code = True, 'INTERACTIVE_INPUT_INVALID'
    state = app.sessions.load('session')
    with pytest.raises(InteractiveBoundaryError) as error:
        facade.activate_revision('session', **kwargs)
    assert error.value.error.code == code and app.sessions.load('session') == state


def test_navigation_generation_race_keeps_the_generation_error_identity(app, monkeypatch):
    revision = accepted(app, 'inspect_scATAC')[0]
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    from agent.application.sessions import AnalysisSessions
    original = AnalysisSessions.switch
    def raced(sessions, session_id, turn_id, revision_id, *, expected_generation):
        original(sessions, session_id, 'concurrent-navigation', revision_id, expected_generation=expected_generation)
        return original(sessions, session_id, turn_id, revision_id, expected_generation=expected_generation)
    monkeypatch.setattr(AnalysisSessions, 'switch', raced)
    with pytest.raises(InteractiveBoundaryError) as error:
        facade.activate_revision('session', 'nav', revision, expected_generation=1)
    assert error.value.error.code == 'INTERACTIVE_GENERATION_CONFLICT'
    assert app.sessions.load('session').generation == 2


@pytest.mark.parametrize('no_claims', [False, True])
def test_historical_answer_exposes_actual_target_separately_from_captured_base(app, monkeypatch, no_claims):
    earlier = accepted(app, 'inspect_scATAC', overrides={'n_cells': 3})[0]
    later = accepted(app, 'inspect_scATAC', name='later', overrides={'n_cells': 5})[0]
    guarded = forbid_work(app, monkeypatch)
    models = []
    def factory(profile):
        response = dict(support='insufficient_evidence', paragraphs=[dict(parts=[
            dict(kind='text', text='No supported claim is included in this response.')])]) if no_claims else None
        model = ScientificModel(question('r1'), ['n_cells'], response=response)
        model.model_id = profile.model_id
        models.append(model)
        return model
    facade = InteractiveAgentApplication(app.workspace_root, model_profiles=PROFILES,
        default_profile_id='alpha', registry=guarded.registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    answered = facade.submit_turn('session', 'historical-question', 'What did the earlier result show?',
        expected_generation=2)
    assert answered.response.status == 'answered' and answered.base_revision_id == later
    if no_claims:
        assert answered.response.scientific['support'] == 'insufficient_evidence'
    assert answered.response.to_dict()['scientific']['targets'] == [
        dict(revision_id=earlier, output_name='output0', subject=None)]
    restarted = read_only_boundary(guarded)
    assert restarted.turn('session', 'historical-question') == answered
    assert len(models) == 1 and app.sessions.load('session').generation == 2


def test_evidence_preserves_provenance_checks_and_omits_whole_unsafe_values(app, monkeypatch):
    revision, _, _ = accepted(app, 'inspect_scATAC', overrides={
        'n_cells': 7, 'input_path': '/private/server/source.h5ad'},
        extra_facts={'nested_source': {'source_path': '/private/server/data', 'sha256': 'a' * 64},
            'api_key': 'private-key', 'credentials': 'private-credentials',
            'authority_payload': {'opaque': 'private-authority'}, 'provider_prompt': 'private-prompt',
            'nested_configuration': {'api_key': 'private-nested-key'}})
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    before = files(app)
    view = facade.evidence('session', revision, 'output0')
    assert view.status == 'available' and view.is_active and view.generation == 1
    assert view.source.output_locator['run_id'] == 'initial:run'
    assert view.source.output_locator['step_id'] == 'step0'
    assert view.source.verification_checks == ('fixture',) and view.source.evidence_sha256
    facts = {f.field: f for f in view.facts}
    assert facts['n_cells'].value == 7 and facts['n_cells'].status == 'available'
    for key in ('input_path', 'nested_source', 'api_key', 'credentials', 'authority_payload',
                'provider_prompt', 'nested_configuration'):
        assert facts[key].status == 'omitted' and facts[key].value is None
        assert facts[key].reason == 'client_safe_content_omitted' and facts[key].source_pointer
    assert '/private/server' not in json.dumps(view.to_dict())
    assert 'private-key' not in json.dumps(view.to_dict()) and 'private-prompt' not in json.dumps(view.to_dict())
    assert view.supported_details == () and files(app) == before
    missing = facade.evidence('session', revision, 'unknown')
    assert missing.status == 'unavailable' and missing.source is None and missing.facts == ()


@pytest.mark.parametrize('qc,section,subject', [
    (True, 'length_histogram', None), (False, 'annotation_rationale', '3'),
])
def test_existing_supported_detail_owner_is_reused_without_science(app, monkeypatch, qc, section, subject):
    revision, _ = publication(app, qc=qc)
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    before = files(app)
    basic = facade.evidence('session', revision, 'output0')
    assert basic.supported_details == (section,)
    detail = facade.evidence('session', revision, 'output0', detail_section=section, subject=subject, limit=1)
    assert detail.status == 'available' and detail.evidence_scope == 'accepted_summary_and_published_detail'
    assert detail.detail[0].artifact_sha256 and detail.detail[0].source_pointer
    assert any(f.field.endswith('records_omitted') for f in detail.detail)
    unsupported = facade.evidence('session', revision, 'output0', detail_section='unsupported')
    assert unsupported.status == 'unsupported' and unsupported.source is None and unsupported.detail == ()
    assert files(app) == before


def test_real_report_delivery_is_labeled_safe_projection_and_pins_both_digests(tmp_path):
    calls = []
    facade, models = boundary(tmp_path, registry=_counting_registry(calls))
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    facade.create_session('session')
    result = first(facade, source)
    handle = facade.artifact_handles('session', result.revision_id)[0]
    before = files(facade._application)
    content = facade.artifact_content('session', result.revision_id, handle.handle)
    assert content.reference == handle and content.reference.sha256 == handle.sha256
    assert content.content_type == 'text/plain; charset=utf-8' and content.filename.endswith('.txt')
    assert content.presentation == 'client_safe_report_projection'
    assert content.data.startswith(b'Client-safe report projection\n')
    assert handle.sha256.encode() in content.data
    assert b'n_cells [available]: 2' in content.data
    assert str(tmp_path).encode() not in content.data and b'input_path [omitted]' in content.data
    assert hashlib.sha256(content.data).hexdigest() == content.content_sha256 != handle.sha256
    assert calls == ['inspect_scATAC'] and len(models) == 1
    assert files(facade._application) == before
    assert facade.artifact_content('session', result.revision_id, handle.handle) == content


def accepted_png(app, *, text_metadata=False, valid=True):
    """A pinned presentation fixture tests delivery, not biological plotting."""
    from PIL import Image, PngImagePlugin
    revision = accepted(app, 'inspect_scATAC')[0]
    turn = app.sessions.load('session').turn('initial')
    root = app._workspace.runs / app._workspace.run_digest(turn.run_id) / 'visualizations'
    path = root / 'fixture.png'
    stream = io.BytesIO()
    metadata = PngImagePlugin.PngInfo()
    if text_metadata:
        metadata.add_text('Description', '/private/server/figure-input')
    Image.new('RGB', (2, 2), color='white').save(stream, format='PNG', pnginfo=metadata)
    data = stream.getvalue() if valid else b'not a png'
    path.write_bytes(data)
    file = CompletionFile(str(path), hashlib.sha256(data).hexdigest())
    state = app.sessions.load('session')
    app.sessions._store._write(replace(state, turns=tuple(
        replace(t, completion_files=t.completion_files + (file,)) if t.turn_id == 'initial' else t
        for t in state.turns)))
    return revision, path, data


def test_png_delivery_keeps_exact_pinned_accepted_bytes(app, monkeypatch):
    revision, _, data = accepted_png(app)
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    before = files(app)
    handle = facade.artifact_handles('session', revision)[0]
    content = facade.artifact_content('session', revision, handle.handle)
    assert content.data == data and content.content_type == 'image/png'
    assert content.presentation == 'accepted_png' and content.filename.endswith('.png')
    assert content.content_sha256 == handle.sha256 and files(app) == before


@pytest.mark.parametrize('attack', ['digest', 'unknown', 'wrong_revision', 'wrong_session', 'format', 'metadata', 'symlink'])
def test_artifact_delivery_rejects_invalid_handles_bytes_and_metadata(app, monkeypatch, attack):
    revision, path, data = accepted_png(app, text_metadata=attack == 'metadata', valid=attack != 'format')
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    handle = facade.artifact_handles('session', revision)[0].handle
    session = 'session'
    if attack == 'digest': path.write_bytes(data + b'changed')
    if attack == 'unknown': handle = 'f' * 64
    if attack == 'wrong_revision': revision = 'other'
    if attack == 'wrong_session': session = 'other'
    if attack == 'symlink':
        elsewhere = path.with_name('elsewhere.png')
        path.rename(elsewhere)
        path.symlink_to(elsewhere)
    with pytest.raises(InteractiveBoundaryError):
        facade.artifact_content(session, revision, handle)


def test_missing_presentation_directory_read_never_repairs_workspace(app, monkeypatch):
    revision, path, _ = accepted_png(app)
    path.unlink()
    path.parent.rmdir()
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    before = files(app)
    handle = facade.artifact_handles('session', revision)[0]
    assert facade.revision('session', revision).artifacts == (handle,)
    with pytest.raises(InteractiveBoundaryError):
        facade.artifact_content('session', revision, handle.handle)
    assert not path.parent.exists() and files(app) == before


def test_guidance_legacy_and_new_readiness_shapes_remain_closed_and_exact():
    old = dict(candidate_id='c' * 64, origin_turn_id='guidance', option=1,
        capability='cluster_cells', text='Conditional explanation.', support='insufficient_evidence', limitations=[])
    legacy = PresentedResponse('answer', 'answered', 'Advice.', guidance={'candidates': [old], 'limitations': []})
    assert PresentedResponse.from_dict(legacy.to_dict()) == legacy
    assert 'readiness' not in legacy.to_dict()['guidance']['candidates'][0]
    readiness = dict(capability_registered=True, readiness='not_fully_checked',
        request_scope='no_execution_inputs_bound', accepted_evidence_handles=['r0'],
        required_ports_without_supplied_request_source=['graph_path'],
        required_scientific_parameters=['resolution'], explicit_request_source_choices=[])
    new = old | dict(base_revision_id='revision', readiness=readiness)
    rich = PresentedResponse('answer', 'answered', 'Advice.', guidance={'candidates': [new], 'limitations': []})
    assert rich.to_dict()['guidance']['candidates'][0]['readiness'] == readiness
    assert PresentedResponse.from_dict(rich.to_dict()) == rich
    for changed in (old | {'readiness': readiness}, new | {'plan': {}},
                    new | {'readiness': readiness | {'readiness': 'ready'}}):
        with pytest.raises(ValueError):
            PresentedResponse('answer', 'answered', 'Advice.', guidance={'candidates': [changed], 'limitations': []})
