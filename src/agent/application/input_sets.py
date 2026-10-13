"""Existing operator InputSet declarations shared by HTTP and dialogue admission."""
from collections.abc import Mapping
from dataclasses import dataclass
import json
import re
from agent.schemas.orchestration import _serialize, freeze_json_mapping
from .session_state import digest
from .local_resources import ResourceConfigurationError

WebConfigurationError = ResourceConfigurationError
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")

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
        from .interactive import _inputs
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

    def resource_choice(self):
        """Return exact operator choice identity, separate from scientific inputs.

        The definition digest captures all declarations and selection policy. It
        records a configured choice and does not qualify that choice scientifically.
        """
        definition = {name: _serialize(getattr(self, name)) for name in self.__dataclass_fields__}
        return dict(input_set_id=self.input_set_id, definition_sha256=digest(definition))


