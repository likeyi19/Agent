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
from agent.application.workspace import ManagedWorkspace, ApplicationWorkspaceError
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry, build_default_planning_model_factory_registry
from agent.schemas.orchestration import _serialize, freeze_json_mapping
from agent.tools.analysis.epizoo_embedding import validate_expected_resource_identity


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
    h5ad_companion: bool = False
    fragments_companion: bool = False
    fragments_default: bool = False
    bam_companion: bool = False
    bam_default: bool = False
    fastq_companion: bool = False
    fastq_default: bool = False

    def __post_init__(self):
        if type(self.input_set_id) is not str or not _IDENTIFIER.fullmatch(self.input_set_id):
            raise WebConfigurationError("Input-set identifiers must be safe lowercase identifiers.")
        _label(self.label)
        if type(self.h5ad_companion) is not bool:
            raise WebConfigurationError("H5AD companion designation must be boolean.")
        if type(self.fragments_companion) is not bool or type(self.fragments_default) is not bool:
            raise WebConfigurationError("Fragments companion and default designations must be boolean.")
        if type(self.bam_companion) is not bool or type(self.bam_default) is not bool:
            raise WebConfigurationError("BAM companion and default designations must be boolean.")
        if type(self.fastq_companion) is not bool or type(self.fastq_default) is not bool:
            raise WebConfigurationError("FASTQ companion and default designations must be boolean.")
        if sum((self.h5ad_companion, self.fragments_companion, self.bam_companion,
                self.fastq_companion)) > 1:
            raise WebConfigurationError("An input set must have one explicit scientific companion role.")
        if self.fragments_default and not self.fragments_companion:
            raise WebConfigurationError("A fragments default must designate a fragments companion.")
        if self.bam_default and not self.bam_companion:
            raise WebConfigurationError("A BAM default must designate a BAM companion.")
        if self.fastq_default and not self.fastq_companion:
            raise WebConfigurationError("A FASTQ default must designate a FASTQ companion.")
        try:
            if not isinstance(self.execution_inputs, Mapping):
                raise TypeError("Input-set structured inputs must be a mapping.")
            inputs = _inputs(self.execution_inputs)
            if self.h5ad_companion and {"checkpoint_path", "expected_resource_identity"}.intersection(inputs):
                raise ValueError("Qualified resources must use the separate resource selection.")
            if self.fragments_companion and {"source_path", "source_sha256", "source_index_path",
                                              "source_index_sha256"}.intersection(inputs):
                raise ValueError("Fragments source identities must come from the selected registration.")
            if self.bam_companion and {"source_path", "source_sha256", "raw_input_paths",
                                      "source_index_path", "source_index_sha256"}.intersection(inputs):
                raise ValueError("BAM source identities must come from the selected registration.")
            if self.fastq_companion and {"source_path", "source_sha256", "raw_input_paths",
                    "source_index_path", "source_index_sha256", "raw_assay", "fastq_layout"}.intersection(inputs):
                raise ValueError("FASTQ source, assay and layout declarations must come from the completed library.")
            if len(json.dumps(inputs, allow_nan=False).encode("utf-8")) > 65_536:
                raise ValueError("Input-set mapping exceeds the submission bound.")
            object.__setattr__(self, "execution_inputs", freeze_json_mapping(inputs, "execution_inputs"))
        except (ValueError, TypeError, RecursionError) as exc:
            raise WebConfigurationError("Input-set structured inputs are invalid.") from exc

    def choice(self):
        """Client-safe metadata; deliberately contains no scientific paths."""
        choice = {"input_set_id": self.input_set_id, "display_label": self.label}
        if self.h5ad_companion:
            choice["h5ad_companion"] = True
        if self.fragments_companion:
            choice["fragments_companion"] = True
        if self.fragments_default:
            choice["fragments_default"] = True
        if self.bam_companion:
            choice["bam_companion"] = True
        if self.bam_default:
            choice["bam_default"] = True
        if self.fastq_companion:
            choice["fastq_companion"] = True
        if self.fastq_default:
            choice["fastq_default"] = True
        return choice

    def inputs(self):
        """Return a fresh mapping suitable for the existing submit contract."""
        return _serialize(self.execution_inputs)


@dataclass(frozen=True)
class QualifiedEpiZooResource:
    """An operator-reviewed exact resource choice for the existing embedding owner.

    Configuration records a review attribution and content pins; it does not
    qualify unfamiliar files or select auxiliary directories on the operator's
    behalf. The scientific owner checks these pins against consumed resources.
    """

    resource_id: str
    label: str
    species: str
    checkpoint_path: str
    checkpoint_sha256: str
    frequencies_sha256: str
    filter_indices_sha256: str
    qualification: str
    default: bool = False

    def __post_init__(self):
        if type(self.resource_id) is not str or not _IDENTIFIER.fullmatch(self.resource_id):
            raise WebConfigurationError("EpiZoo resource identifiers must be safe lowercase identifiers.")
        _label(self.label)
        _label(self.qualification)
        if self.species not in ("human", "mouse") or type(self.species) is not str:
            raise WebConfigurationError("EpiZoo resources require an explicit supported species.")
        if (type(self.checkpoint_path) is not str or not self.checkpoint_path.strip()
                or len(self.checkpoint_path) > 4096 or "\x00" in self.checkpoint_path):
            raise WebConfigurationError("EpiZoo checkpoint paths must be explicitly configured.")
        try:
            validate_expected_resource_identity(self.inputs()['expected_resource_identity'])
        except (ValueError, TypeError) as exc:
            raise WebConfigurationError("EpiZoo resource content identities must be exact approved SHA-256 pins.") from exc
        if type(self.default) is not bool:
            raise WebConfigurationError("EpiZoo default designation must be boolean.")

    def choice(self):
        return {"resource_id": self.resource_id, "display_label": self.label,
                "species": self.species, "is_default": self.default}

    def inputs(self):
        return {"checkpoint_path": self.checkpoint_path, "expected_resource_identity": {
            "resource_id": self.resource_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "frequencies_sha256": self.frequencies_sha256,
            "filter_indices_sha256": self.filter_indices_sha256,
        }}


@dataclass(frozen=True)
class WebConfiguration:
    workspace_root: Path
    model_profiles: tuple[PlanningModelProfile, ...]
    default_profile_id: str
    display_labels: Mapping[str, str] = field(default_factory=dict)
    input_sets: tuple[ScientificInputSet, ...] = ()
    upload_root: Path | None = None
    upload_max_bytes: int = 256 * 1024 * 1024
    upload_max_concurrent: int = 2
    epizoo_resources: tuple[QualifiedEpiZooResource, ...] = ()

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
        if (type(self.epizoo_resources) is not tuple or len(self.epizoo_resources) > 64
                or any(not isinstance(r, QualifiedEpiZooResource) for r in self.epizoo_resources)
                or len({r.resource_id for r in self.epizoo_resources}) != len(self.epizoo_resources)):
            raise WebConfigurationError("EpiZoo resources must have bounded unique identities.")
        if type(self.upload_max_bytes) is not int or not 1 <= self.upload_max_bytes <= 1024 ** 3:
            raise WebConfigurationError("The attachment size limit must be between one byte and one GiB.")
        if type(self.upload_max_concurrent) is not int or not 1 <= self.upload_max_concurrent <= 4:
            raise WebConfigurationError("Concurrent attachments must be between one and four.")
        if self.upload_root is not None:
            if not isinstance(self.upload_root, (str, Path)) or not str(self.upload_root).strip():
                raise WebConfigurationError("An attachment root must be a workspace directory.")
            root = Path(self.upload_root).expanduser()
            if '..' in root.parts:
                raise WebConfigurationError("The attachment root must not contain parent traversal.")
            if not root.is_absolute():
                root = self.workspace_root / root
            root = root.absolute()
            if root == self.workspace_root or not root.is_relative_to(self.workspace_root):
                raise WebConfigurationError("The attachment root must be contained below the workspace.")
            object.__setattr__(self, 'upload_root', root)


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
        allowed = {"workspace_root", "model_profiles", "default_profile_id", "input_sets",
                   "upload_root", "upload_max_bytes", "upload_max_concurrent", "epizoo_resources"}
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
            if (type(item) is not dict or set(item) - {"input_set_id", "label", "execution_inputs", "h5ad_companion",
                                                     "fragments_companion", "fragments_default",
                                                     "bam_companion", "bam_default",
                                                     "fastq_companion", "fastq_default"}
                    or {"input_set_id", "label", "execution_inputs"} - set(item)):
                raise WebConfigurationError("Operator input-set fields are invalid.")
            input_sets.append(ScientificInputSet(**item))
        resource_values = value.get("epizoo_resources", [])
        if type(resource_values) is not list or len(resource_values) > 64:
            raise WebConfigurationError("Operator EpiZoo resources must be a bounded list.")
        resources = []
        resource_keys = {"resource_id", "label", "species", "checkpoint_path", "checkpoint_sha256",
                         "frequencies_sha256", "filter_indices_sha256", "qualification", "default"}
        for item in resource_values:
            if type(item) is not dict or set(item) - resource_keys or resource_keys - {"default"} - set(item):
                raise WebConfigurationError("Operator EpiZoo resource fields are invalid.")
            resources.append(QualifiedEpiZooResource(**item))
        return WebConfiguration(workspace, tuple(profiles), value["default_profile_id"], labels, tuple(input_sets),
            value.get('upload_root'), value.get('upload_max_bytes', 256 * 1024 * 1024),
            value.get('upload_max_concurrent', 2), tuple(resources))
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
    approved_source_roots = ()
    if configuration.upload_root is not None:
        try:
            if configuration.upload_root != configuration.upload_root.resolve(strict=False):
                raise ValueError('Noncanonical attachment root.')
            workspace = ManagedWorkspace(configuration.workspace_root)
            # Walk the declared descendant using the existing workspace owner,
            # checking every parent before a registration can approve this root.
            root = workspace.root
            for component in configuration.upload_root.relative_to(configuration.workspace_root).parts:
                root = workspace._ensure_directory(root / component)
            if root != root.resolve(strict=True):
                raise ValueError('Noncanonical attachment root.')
            approved_source_roots = (root,)
        except (ApplicationWorkspaceError, ValueError, OSError, RuntimeError) as exc:
            raise WebConfigurationError('The attachment root could not be initialized safely.') from exc
    return InteractiveAgentApplication(configuration.workspace_root,
        model_profiles=configuration.model_profiles, default_profile_id=configuration.default_profile_id,
        display_labels=configuration.display_labels, planning_model_factory_registry=factory,
        registry=registry, executor=executor, approved_source_roots=approved_source_roots)
