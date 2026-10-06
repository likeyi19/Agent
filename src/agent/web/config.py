"""Operator-owned composition for the thin local HTTP application.

Profiles use the existing immutable planning factory registry. Named input sets
are deployment configuration: browsers receive labels and identifiers only.
Credentials are read by the existing provider adapters from server environment.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from types import MappingProxyType

from agent.application import InteractiveAgentApplication
from agent.application.interactive import _inputs
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry, build_default_planning_model_factory_registry
from agent.schemas.orchestration import _serialize, freeze_json_mapping


_CONFIGURATION_BYTES = 1_048_576
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class WebConfigurationError(ValueError):
    """Invalid operator configuration; never returned as browser diagnostics."""


def _label(value):
    if (type(value) is not str or not value.strip() or len(value) > 512
            or any(not c.isprintable() for c in value)):
        raise WebConfigurationError("Display labels must be nonempty printable text.")
    return value


@dataclass(frozen=True)
class ScientificInputSet:
    """One explicit operator-admitted structured science input mapping."""

    input_set_id: str
    label: str
    execution_inputs: Mapping[str, object]

    def __post_init__(self):
        if type(self.input_set_id) is not str or not _IDENTIFIER.fullmatch(self.input_set_id):
            raise WebConfigurationError("Input-set identifiers must be safe lowercase identifiers.")
        _label(self.label)
        try:
            if not isinstance(self.execution_inputs, Mapping):
                raise TypeError("Input-set structured inputs must be a mapping.")
            inputs = _inputs(self.execution_inputs)
            if len(json.dumps(inputs, allow_nan=False).encode("utf-8")) > 65_536:
                raise ValueError("Input-set mapping exceeds the submission bound.")
            object.__setattr__(self, "execution_inputs", freeze_json_mapping(inputs, "execution_inputs"))
        except (ValueError, TypeError, RecursionError) as exc:
            raise WebConfigurationError("Input-set structured inputs are invalid.") from exc

    def choice(self):
        """Client-safe metadata; deliberately contains no scientific paths."""
        return {"input_set_id": self.input_set_id, "display_label": self.label}

    def inputs(self):
        """Return a fresh mapping suitable for the existing submit contract."""
        return _serialize(self.execution_inputs)


@dataclass(frozen=True)
class WebConfiguration:
    workspace_root: Path
    model_profiles: tuple[PlanningModelProfile, ...]
    default_profile_id: str
    display_labels: Mapping[str, str] = field(default_factory=dict)
    input_sets: tuple[ScientificInputSet, ...] = ()

    def __post_init__(self):
        if not isinstance(self.workspace_root, (str, Path)) or not str(self.workspace_root).strip():
            raise WebConfigurationError("A server workspace is required.")
        # Preserve a final symlink for the existing ManagedWorkspace rejection;
        # resolving it here would erase a deployment safety check owned there.
        object.__setattr__(self, "workspace_root", Path(self.workspace_root).expanduser().absolute())
        profiles = self.model_profiles
        if (type(profiles) is not tuple or not 1 <= len(profiles) <= 64
                or any(not isinstance(p, PlanningModelProfile) for p in profiles)
                or len({p.profile_id for p in profiles}) != len(profiles)):
            raise WebConfigurationError("Profiles must be a bounded unique tuple of existing model profiles.")
        default = next((p for p in profiles if p.profile_id == self.default_profile_id), None)
        if default is None or not default.enabled or not default.supports_structured_output:
            raise WebConfigurationError("The default profile must be enabled and support structured output.")
        if not isinstance(self.display_labels, Mapping):
            raise WebConfigurationError("Profile labels must be a mapping.")
        labels = dict(self.display_labels)
        if set(labels) - {p.profile_id for p in profiles}:
            raise WebConfigurationError("Profile labels reference an unconfigured profile.")
        for label in labels.values():
            _label(label)
        object.__setattr__(self, "display_labels", MappingProxyType(labels))
        if (type(self.input_sets) is not tuple or len(self.input_sets) > 64
                or any(not isinstance(s, ScientificInputSet) for s in self.input_sets)
                or len({s.input_set_id for s in self.input_sets}) != len(self.input_sets)):
            raise WebConfigurationError("Input sets must have bounded unique identities.")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise WebConfigurationError("Operator JSON contains duplicate keys.")
        result[key] = value
    return result


def _reject_constant(value):
    raise WebConfigurationError("Operator JSON contains a non-finite number.")


def load_web_configuration(path):
    """Load strict JSON. Relative workspace paths resolve beside the config file.

    Scientific input paths are preserved exactly as declared by the operator;
    this loader does not infer or silently repair scientific input locations.
    """
    try:
        path = Path(path).expanduser().resolve()
        with path.open("rb") as stream:
            content = stream.read(_CONFIGURATION_BYTES + 1)
        if len(content) > _CONFIGURATION_BYTES:
            raise WebConfigurationError("Operator configuration exceeds one MiB.")
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant)
        allowed = {"workspace_root", "model_profiles", "default_profile_id", "input_sets"}
        required = {"workspace_root", "model_profiles", "default_profile_id"}
        if type(value) is not dict or set(value) - allowed or required - set(value):
            raise WebConfigurationError("Operator configuration fields are invalid.")
        workspace = value["workspace_root"]
        if type(workspace) is not str or not workspace.strip():
            raise WebConfigurationError("A server workspace path is required.")
        workspace = Path(workspace).expanduser()
        if not workspace.is_absolute():
            workspace = path.parent / workspace
        profile_values = value["model_profiles"]
        if type(profile_values) is not list or not 1 <= len(profile_values) <= 64:
            raise WebConfigurationError("Operator model profiles must be a bounded list.")
        profiles, labels = [], {}
        profile_keys = {"profile_id", "provider_id", "model_id", "enabled", "supports_structured_output",
                        "request_timeout_seconds", "display_label"}
        for item in profile_values:
            if type(item) is not dict or set(item) - profile_keys:
                raise WebConfigurationError("Operator profile fields are invalid.")
            values = {key: val for key, val in item.items() if key != "display_label"}
            profile = PlanningModelProfile(**values)
            profiles.append(profile)
            if "display_label" in item:
                labels[profile.profile_id] = _label(item["display_label"])
        input_values = value.get("input_sets", [])
        if type(input_values) is not list or len(input_values) > 64:
            raise WebConfigurationError("Operator input sets must be a bounded list.")
        input_sets = []
        for item in input_values:
            if type(item) is not dict or set(item) != {"input_set_id", "label", "execution_inputs"}:
                raise WebConfigurationError("Operator input-set fields are invalid.")
            input_sets.append(ScientificInputSet(**item))
        return WebConfiguration(workspace, tuple(profiles), value["default_profile_id"], labels, tuple(input_sets))
    except WebConfigurationError:
        raise
    except (ValueError, TypeError, OSError, UnicodeError, RecursionError) as exc:
        raise WebConfigurationError("Operator configuration could not be loaded safely.") from exc


def build_interactive_application(configuration, *, planning_model_factory_registry=None,
                                  registry=None, executor=None):
    """Compose the existing facade without constructing any provider client."""
    if not isinstance(configuration, WebConfiguration):
        raise WebConfigurationError("Expected WebConfiguration.")
    factory = (build_default_planning_model_factory_registry()
               if planning_model_factory_registry is None else planning_model_factory_registry)
    if not isinstance(factory, PlanningModelFactoryRegistry):
        raise WebConfigurationError("Expected the existing planning-model factory registry.")
    if any(p.enabled and p.supports_structured_output and p.provider_id not in factory.provider_ids
           for p in configuration.model_profiles):
        raise WebConfigurationError("An enabled model profile uses an unregistered provider.")
    return InteractiveAgentApplication(configuration.workspace_root,
        model_profiles=configuration.model_profiles, default_profile_id=configuration.default_profile_id,
        display_labels=configuration.display_labels, planning_model_factory_registry=factory,
        registry=registry, executor=executor)
