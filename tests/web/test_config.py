"""Operator configuration stays immutable, strict, and private to the server."""
from dataclasses import replace
import json

import pytest

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
from agent.web.config import (
    QualifiedEpiZooResource, ScientificInputSet, WebConfiguration, WebConfigurationError,
    build_interactive_application, load_web_configuration,
)


def configuration_json():
    return {
        "workspace_root": "workspace",
        "default_profile_id": "primary",
        "model_profiles": [{"profile_id": "primary", "provider_id": "scripted",
                            "model_id": "model/primary", "display_label": "Lab model"}],
        "input_sets": [{"input_set_id": "tiny", "label": "Tiny matrix",
                        "execution_inputs": {"input_path": "/operator/tiny.h5ad", "species": "human"}}],
    }


def write_configuration(tmp_path, value):
    path = tmp_path / "operator.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_operator_load_preserves_profiles_paths_and_private_inputs(tmp_path):
    config = load_web_configuration(write_configuration(tmp_path, configuration_json()))
    assert config.workspace_root == tmp_path / "workspace"
    assert config.model_profiles == (PlanningModelProfile("primary", "scripted", "model/primary"),)
    assert config.default_profile_id == "primary" and config.display_labels["primary"] == "Lab model"
    input_set = config.input_sets[0]
    assert input_set.choice() == {"input_set_id": "tiny", "display_label": "Tiny matrix"}
    assert "/operator" not in json.dumps(input_set.choice())
    assert input_set.inputs()["input_path"] == "/operator/tiny.h5ad"
    changed = input_set.inputs()
    changed["input_path"] = "/different.h5ad"
    assert input_set.inputs()["input_path"] == "/operator/tiny.h5ad"
    with pytest.raises(TypeError):
        input_set.execution_inputs["species"] = "mouse"
    with pytest.raises(TypeError):
        config.display_labels["primary"] = "Changed"


def test_composition_uses_existing_factory_without_constructing_clients(tmp_path):
    config = load_web_configuration(write_configuration(tmp_path, configuration_json()))
    calls = []
    def forbidden(profile):
        calls.append(profile)
        raise AssertionError("Provider client must be lazy.")
    app = build_interactive_application(config,
        planning_model_factory_registry=PlanningModelFactoryRegistry({"scripted": forbidden}))
    assert isinstance(app, InteractiveAgentApplication)
    assert calls == []
    assert app.model_choices()[0].to_dict() == {
        "profile_id": "primary", "display_label": "Lab model", "is_default": True,
        "provider_id": "scripted", "model_id": "model/primary",
    }
    session = app.create_session("known")
    assert session.generation == 0 and app.reopen_session("known") == session and calls == []


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(api_key="private"),
    lambda value: value.update(default_profile_id="unknown"),
    lambda value: value.update(model_profiles=[]),
    lambda value: value["model_profiles"][0].update(endpoint="https://example.invalid"),
    lambda value: value["model_profiles"][0].update(enabled=False),
    lambda value: value["model_profiles"][0].update(supports_structured_output=False),
    lambda value: value["model_profiles"][0].update(request_timeout_seconds=True),
    lambda value: value["model_profiles"].append(dict(value["model_profiles"][0])),
    lambda value: value["input_sets"].append(dict(value["input_sets"][0])),
    lambda value: value["input_sets"][0].update(extra=True),
])
def test_strict_operator_configuration_rejects_invalid_or_extra_fields(tmp_path, mutation):
    value = configuration_json()
    mutation(value)
    with pytest.raises(WebConfigurationError):
        load_web_configuration(write_configuration(tmp_path, value))


@pytest.mark.parametrize("text", [
    '{"workspace_root":"first","workspace_root":"second"}',
    '{"workspace_root":NaN}',
    '{"workspace_root":"a","model_profiles":[],"default_profile_id":"x"}',
    '[]',
    '{bad}',
])
def test_unsafe_json_is_rejected(tmp_path, text):
    path = tmp_path / "operator.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(WebConfigurationError):
        load_web_configuration(path)


@pytest.mark.parametrize("inputs", [
    None, [], {"overwrite": True}, {"output_dir": "/browser/output"},
    {"input_path": {"$prior_output": "private"}}, {"other": {"authority_payload": {}}},
    {"text": "a" * 65_536},
])
def test_operator_input_sets_use_the_existing_application_admission(inputs):
    with pytest.raises(WebConfigurationError):
        ScientificInputSet("tiny", "Tiny", inputs)


def test_unregistered_profile_rejected_before_workspace_construction(tmp_path):
    config = load_web_configuration(write_configuration(tmp_path, configuration_json()))
    with pytest.raises(WebConfigurationError):
        build_interactive_application(config, planning_model_factory_registry=PlanningModelFactoryRegistry({}))
    assert not config.workspace_root.exists()


def test_composition_preserves_existing_workspace_symlink_rejection(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "workspace-link"
    link.symlink_to(target, target_is_directory=True)
    value = configuration_json()
    value["workspace_root"] = "workspace-link"
    config = load_web_configuration(write_configuration(tmp_path, value))
    assert config.workspace_root == link and config.workspace_root.is_symlink()
    factory = PlanningModelFactoryRegistry({"scripted": lambda profile: None})
    with pytest.raises(InteractiveBoundaryError) as error:
        build_interactive_application(config, planning_model_factory_registry=factory)
    assert error.value.error.code == "INTERACTIVE_APPLICATION_FAILED"
    assert list(target.iterdir()) == []


def test_configuration_size_and_identity_bounds(tmp_path):
    path = tmp_path / "operator.json"
    path.write_bytes(b" " * 1_048_577)
    with pytest.raises(WebConfigurationError):
        load_web_configuration(path)
    profile = PlanningModelProfile("primary", "scripted", "model/primary")
    config = WebConfiguration(tmp_path / "workspace", (profile,), "primary")
    with pytest.raises(WebConfigurationError):
        replace(config, display_labels={"unknown": "Other"})
    with pytest.raises(WebConfigurationError):
        replace(config, model_profiles=[profile])
    with pytest.raises(WebConfigurationError):
        ScientificInputSet("../server-path", "Tiny", {})


def qualified_resource(**changes):
    values = dict(resource_id='mouse-reviewed', label='Reviewed mouse resource', species='mouse',
        checkpoint_path='/operator/epizoo.pt', checkpoint_sha256='a' * 64,
        frequencies_sha256='b' * 64, filter_indices_sha256='c' * 64,
        qualification='Reviewed resource identities for the pinned execution', default=True)
    values.update(changes)
    return values


def test_typed_companion_and_reviewed_epizoo_configuration_are_explicit_and_private(tmp_path):
    value = configuration_json()
    value['input_sets'] = [dict(input_set_id='mouse', label='Mouse declarations',
        execution_inputs={'species': 'mouse', 'resolution': 0.7}, h5ad_companion=True)]
    value['epizoo_resources'] = [qualified_resource()]
    config = load_web_configuration(write_configuration(tmp_path, value))
    companion = config.input_sets[0]
    assert companion.choice() == dict(input_set_id='mouse', display_label='Mouse declarations', h5ad_companion=True)
    assert companion.inputs() == {'species': 'mouse', 'resolution': 0.7}
    resource = config.epizoo_resources[0]
    assert resource.choice() == dict(resource_id='mouse-reviewed', display_label='Reviewed mouse resource',
                                     species='mouse', is_default=True)
    assert '/operator' not in json.dumps(resource.choice())
    assert 'sha256' not in json.dumps(resource.choice())
    assert resource.inputs() == dict(checkpoint_path='/operator/epizoo.pt', expected_resource_identity={
        'resource_id': 'mouse-reviewed', 'checkpoint_sha256': 'a' * 64,
        'frequencies_sha256': 'b' * 64, 'filter_indices_sha256': 'c' * 64})
    changed = resource.inputs()
    changed['expected_resource_identity']['checkpoint_sha256'] = 'd' * 64
    assert resource.checkpoint_sha256 == 'a' * 64


@pytest.mark.parametrize('changes', [
    {'resource_id': '../checkpoint'}, {'species': 'macaque'}, {'checkpoint_path': ''},
    {'checkpoint_sha256': 'a' * 63}, {'frequencies_sha256': 'B' * 64},
    {'filter_indices_sha256': 123}, {'qualification': ''}, {'default': 1},
])
def test_epizoo_configuration_requires_explicit_review_and_exact_pins(changes):
    with pytest.raises(WebConfigurationError):
        QualifiedEpiZooResource(**qualified_resource(**changes))


@pytest.mark.parametrize('change', [
    lambda value: value['epizoo_resources'][0].update(auxiliary_directory='/operator/unreviewed'),
    lambda value: value['epizoo_resources'][0].pop('qualification'),
    lambda value: value['epizoo_resources'].append(dict(value['epizoo_resources'][0])),
    lambda value: value['input_sets'][0].update(h5ad_companion='yes'),
])
def test_epizoo_config_loader_rejects_extra_missing_and_duplicate_fields(tmp_path, change):
    value = configuration_json()
    value['epizoo_resources'] = [qualified_resource()]
    change(value)
    with pytest.raises(WebConfigurationError):
        load_web_configuration(write_configuration(tmp_path, value))


@pytest.mark.parametrize('field,value', [
    ('checkpoint_path', '/operator/unreviewed.pt'),
    ('expected_resource_identity', {'checkpoint_sha256': 'a' * 64}),
])
def test_companion_cannot_bypass_reviewed_resource_selection(field, value):
    with pytest.raises(WebConfigurationError):
        ScientificInputSet('mouse', 'Mouse', {'species': 'mouse', field: value}, h5ad_companion=True)


def test_upload_configuration_is_opt_in_and_approves_only_controlled_workspace_root(tmp_path):
    value = configuration_json()
    default = load_web_configuration(write_configuration(tmp_path, value))
    assert default.upload_root is None
    assert default.upload_max_bytes == 256 * 1024 * 1024 and default.upload_max_concurrent == 2
    value.update(upload_root='inputs/uploads', upload_max_bytes=8192, upload_max_concurrent=1)
    configured = load_web_configuration(write_configuration(tmp_path, value))
    assert configured.upload_root == tmp_path / 'workspace' / 'inputs' / 'uploads'
    assert not configured.workspace_root.exists()
    calls = []
    def forbidden(profile):
        calls.append(profile)
        raise AssertionError('Configuration must not construct a provider.')
    application = build_interactive_application(configured,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
    assert configured.upload_root.is_dir() and not calls
    source = configured.upload_root / 'complete.h5ad'
    source.write_bytes(b'bytes are not science')
    record = application.resources.register('completed', source, label='tiny.h5ad', attribution='test')
    assert record.input_type == 'h5ad'
    outside = tmp_path / 'outside.h5ad'
    outside.write_bytes(b'not approved')
    from agent.application.local_resources import ResourceAdmissionError
    with pytest.raises(ResourceAdmissionError) as error:
        application.resources.register('outside', outside, label='outside.h5ad', attribution='test')
    assert error.value.code == 'LOCAL_RESOURCE_ACCESS_INVALID'


@pytest.mark.parametrize('changes', [
    {'upload_root': '../outside'}, {'upload_root': '.'}, {'upload_root': ''},
    {'upload_root': 1}, {'upload_root': []},
    {'upload_max_bytes': True}, {'upload_max_bytes': 0}, {'upload_max_bytes': 1024 ** 3 + 1},
    {'upload_max_concurrent': True}, {'upload_max_concurrent': 0}, {'upload_max_concurrent': 5},
])
def test_upload_configuration_rejects_unsafe_roots_and_unbounded_limits(tmp_path, changes):
    value = configuration_json()
    value.update(changes)
    with pytest.raises(WebConfigurationError):
        load_web_configuration(write_configuration(tmp_path, value))
    assert not (tmp_path / 'workspace').exists()


def test_absolute_upload_root_must_be_a_workspace_descendant(tmp_path):
    value = configuration_json()
    value['upload_root'] = str(tmp_path / 'outside')
    with pytest.raises(WebConfigurationError):
        load_web_configuration(write_configuration(tmp_path, value))
    value['upload_root'] = str(tmp_path / 'workspace' / 'uploads')
    configuration = load_web_configuration(write_configuration(tmp_path, value))
    assert configuration.upload_root == tmp_path / 'workspace' / 'uploads'


def test_upload_root_symlink_is_rejected_before_admission(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (workspace / 'uploads').symlink_to(outside, target_is_directory=True)
    value = configuration_json()
    value['upload_root'] = 'uploads'
    configuration = load_web_configuration(write_configuration(tmp_path, value))
    with pytest.raises(WebConfigurationError):
        build_interactive_application(configuration,
            planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda profile: None}))
    assert list(outside.iterdir()) == []


def test_upload_parent_alias_is_rejected_before_admission(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    parent = workspace / 'target'
    parent.mkdir()
    (workspace / 'alias').symlink_to(parent, target_is_directory=True)
    value = configuration_json()
    value['upload_root'] = 'alias/uploads'
    configuration = load_web_configuration(write_configuration(tmp_path, value))
    with pytest.raises(WebConfigurationError):
        build_interactive_application(configuration,
            planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda profile: None}))
    assert list(parent.iterdir()) == []
