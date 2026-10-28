"""Bounded presentation contracts for transport-independent interactive clients.

These views contain display history and stable identities, never scientific
authority, executable bindings, provider configuration, or storage locations.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
import json
import re

from agent.schemas.orchestration import _JsonModel, freeze_json_mapping, _serialize
from .dialogue_evidence import DialogueEvidence, EvidenceFact, EvidenceSource


MAX_IDENTIFIER_LENGTH = 256
MAX_UTTERANCE_LENGTH = 16_384
MAX_PRESENTATION_LENGTH = 131_072
MAX_PRESENTATION_BYTES = 262_144
MAX_HISTORY_ITEMS = 1_000
MAX_PRESENTATION_ARTIFACTS = 32


def _text(value, name, maximum, *, empty=False):
    if (type(value) is not str or len(value) > maximum
            or not empty and not value.strip()):
        raise ValueError(f'Invalid interactive {name}.')
    return value


def _identifier(value, name):
    _text(value, name, MAX_IDENTIFIER_LENGTH)
    if any(not char.isprintable() for char in value):
        raise ValueError(f'Invalid interactive {name}.')


def _optional_identifier(value, name):
    if value is not None:
        _identifier(value, name)


def _sha256(value, name):
    if (type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None):
        raise ValueError(f'Invalid interactive {name} digest.')


def _natural(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'Invalid interactive {name}.')


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f'Invalid interactive {name}.')


def _shape(value, keys, name):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f'Invalid interactive {name} fields.')


def _strings(value, name, *, maximum=128):
    if not isinstance(value, (tuple, list)) or len(value) > maximum:
        raise ValueError(f'Invalid interactive {name}.')
    for item in value:
        _text(item, name, 4_096)


def _collection(values, cls, key, name):
    if type(values) is not tuple or len(values) > MAX_HISTORY_ITEMS:
        raise ValueError(f'Invalid interactive {name}.')
    if any(not isinstance(value, cls) for value in values):
        raise TypeError(f'Invalid interactive {name} item.')
    if len({getattr(value, key) for value in values}) != len(values):
        raise ValueError(f'Duplicate interactive {name} identity.')


def _support(value):
    if value not in ('supported', 'insufficient_evidence'):
        raise ValueError('Invalid interactive scientific support.')


def _clarification(value):
    _shape(value, ('reason', 'choices', 'value_required'), 'clarification')
    _identifier(value['reason'], 'clarification reason')
    _strings(value['choices'], 'clarification choices')
    _boolean(value['value_required'], 'value_required')


def _scientific(value, *, version=1):
    legacy = {'support', 'limitations', 'evidence_scope'}
    richer = legacy | {'targets'}
    current = richer | {'availability', 'citations'}
    if not isinstance(value, Mapping) or set(value) not in ((legacy, richer) if version == 1 else (current,)):
        raise ValueError('Invalid interactive scientific presentation fields.')
    _support(value['support'])
    _strings(value['limitations'], 'scientific limitations')
    _identifier(value['evidence_scope'], 'evidence_scope')
    if 'targets' in value:
        targets = value['targets']
        if not isinstance(targets, (tuple, list)) or len(targets) > 64:
            raise ValueError('Invalid scientific presentation targets.')
        identities = []
        for target in targets:
            _shape(target, ('revision_id', 'output_name', 'subject'), 'scientific target')
            _identifier(target['revision_id'], 'target revision')
            _identifier(target['output_name'], 'target output')
            _optional_identifier(target['subject'], 'target subject')
            identities.append((target['revision_id'], target['output_name'], target['subject']))
        if len(set(identities)) != len(identities):
            raise ValueError('Duplicate scientific presentation target.')
    if version >= 2:
        if value['availability'] not in ('generated', 'deterministic', 'unavailable'):
            raise ValueError('Invalid scientific explanation availability.')
        citations = value['citations']
        if not isinstance(citations, (tuple, list)) or len(citations) > 64:
            raise ValueError('Invalid scientific citations.')
        for citation in citations:
            _shape(citation, ('claim_id', 'field', 'source'), 'scientific citation')
            _identifier(citation['claim_id'], 'claim identity')
            _text(citation['field'], 'claim field', 4096)
            if not isinstance(citation['source'], Mapping):
                raise ValueError('Invalid scientific citation source.')
        if value['availability'] == 'unavailable' and citations:
            raise ValueError('Unavailable explanations cannot cite generated claims.')


def _guidance(value):
    _shape(value, ('candidates', 'limitations'), 'guidance presentation')
    _strings(value['limitations'], 'guidance limitations')
    candidates = value['candidates']
    if not isinstance(candidates, (list, tuple)) or len(candidates) > 4:
        raise ValueError('Invalid interactive guidance candidates.')
    for item in candidates:
        legacy = {'candidate_id', 'origin_turn_id', 'option', 'capability', 'text', 'support', 'limitations'}
        richer = legacy | {'base_revision_id', 'readiness'}
        if not isinstance(item, Mapping) or set(item) not in (legacy, richer):
            raise ValueError('Invalid interactive guidance candidate fields.')
        for key in ('candidate_id', 'origin_turn_id', 'capability'):
            _identifier(item[key], key)
        if type(item['option']) is not int or not 1 <= item['option'] <= 4:
            raise ValueError('Invalid interactive guidance option.')
        _text(item['text'], 'candidate text', MAX_PRESENTATION_LENGTH, empty=True)
        _support(item['support'])
        _strings(item['limitations'], 'candidate limitations')
        if set(item) == richer:
            _optional_identifier(item['base_revision_id'], 'candidate base revision')
            readiness = item['readiness']
            _shape(readiness, ('capability_registered', 'readiness', 'request_scope',
                'accepted_evidence_handles', 'required_ports_without_supplied_request_source',
                'required_scientific_parameters', 'explicit_request_source_choices'), 'candidate readiness')
            if (readiness['capability_registered'] is not True
                    or readiness['readiness'] != 'not_fully_checked'
                    or readiness['request_scope'] != 'no_execution_inputs_bound'):
                raise ValueError('Invalid persisted candidate readiness.')
            for name in ('accepted_evidence_handles', 'required_ports_without_supplied_request_source',
                         'required_scientific_parameters', 'explicit_request_source_choices'):
                _strings(readiness[name], name, maximum=128)
    if (len({item['candidate_id'] for item in candidates}) != len(candidates)
            or len({item['option'] for item in candidates}) != len(candidates)):
        raise ValueError('Duplicate interactive guidance candidate.')


@dataclass(frozen=True)
class ModelChoice(_JsonModel):
    """An exact operator-admitted profile; metadata is display-only."""

    profile_id: str
    display_label: str
    is_default: bool
    provider_id: str | None = None
    model_id: str | None = None

    def __post_init__(self):
        _identifier(self.profile_id, 'profile_id')
        _text(self.display_label, 'display_label', 512)
        _boolean(self.is_default, 'is_default')
        _optional_identifier(self.provider_id, 'provider_id')
        _optional_identifier(self.model_id, 'model_id')


@dataclass(frozen=True)
class ClientError(_JsonModel):
    """Deterministic client failure, with no exception or provider payload."""

    code: str
    message: str

    def __post_init__(self):
        _identifier(self.code, 'error code')
        _text(self.message, 'error message', 4_096)

    @classmethod
    def from_dict(cls, value):
        _shape(value, ('code', 'message'), 'error')
        return cls(**value)


@dataclass(frozen=True)
class PresentedResponse(_JsonModel):
    """The exact immutable display of one completed semantic outcome.

    Structured branches retain rendering metadata and guidance references.
    Scientific claim values and authority sources remain with existing owners.
    """

    kind: str
    status: str
    text: str
    clarification: object = None
    scientific: object = None
    guidance: object = None
    error: ClientError | None = None
    version: int = 2
    artifacts: object = dataclass_field(default=None, metadata={'omit_none': True})

    def __post_init__(self):
        if type(self.version) is not int or self.version not in (1, 2, 3):
            raise ValueError('Unsupported interactive presentation version.')
        if self.kind not in ('answer', 'clarify', 'navigate', 'execute'):
            raise ValueError('Invalid interactive response kind.')
        _text(self.status, 'response status', 64)
        _text(self.text, 'response text', MAX_PRESENTATION_LENGTH, empty=True)
        if self.error is not None and not isinstance(self.error, ClientError):
            raise TypeError('Interactive response error must be a ClientError.')
        if self.scientific is not None and self.guidance is not None:
            raise ValueError('A response cannot contain both scientific and guidance presentations.')
        for name, validator, kind in (
                ('clarification', _clarification, 'clarify'),
                ('scientific', _scientific, 'answer'),
                ('guidance', _guidance, 'answer')):
            value = getattr(self, name)
            if value is not None:
                if self.kind != kind and not (name == 'scientific' and self.version >= 2 and self.kind == 'execute'):
                    raise ValueError(f'Invalid {name} response kind.')
                if name == 'scientific':
                    validator(value, version=self.version)
                else:
                    validator(value)
                object.__setattr__(self, name, freeze_json_mapping(value, name))
        if self.version == 3:
            if (not isinstance(self.artifacts, (tuple, list))
                    or not 1 <= len(self.artifacts) <= MAX_PRESENTATION_ARTIFACTS):
                raise ValueError('Invalid interactive presentation artifacts.')
            pins = []
            for artifact in self.artifacts:
                _shape(artifact, ('artifact_id', 'manifest_sha256', 'png_sha256', 'figure_id'), 'presentation artifact')
                for key in ('artifact_id', 'manifest_sha256', 'png_sha256'):
                    _sha256(artifact[key], key)
                figure_id = artifact['figure_id']
                _identifier(figure_id, 'presentation figure identity')
                if re.fullmatch(r'[A-Za-z0-9_-]+', figure_id) is None:
                    raise ValueError('Invalid interactive presentation figure identity.')
                pins.append(freeze_json_mapping(artifact, 'presentation artifact'))
            if len({pin['artifact_id'] for pin in pins}) != len(pins):
                raise ValueError('Duplicate interactive presentation artifact identity.')
            object.__setattr__(self, 'artifacts', tuple(pins))
        elif self.artifacts is not None:
            raise ValueError('Presentation artifacts require version 3.')
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False).encode('utf-8')
        if len(encoded) > MAX_PRESENTATION_BYTES:
            raise ValueError('Interactive presentation exceeds its byte limit.')

    @classmethod
    def from_dict(cls, value):
        keys = ('version', 'kind', 'status', 'text', 'clarification', 'scientific', 'guidance', 'error')
        if isinstance(value, Mapping) and value.get('version') == 3:
            keys += ('artifacts',)
        _shape(value, keys, 'presentation')
        data = dict(value)
        if data['error'] is not None:
            data['error'] = ClientError.from_dict(data['error'])
        return cls(**data)


@dataclass(frozen=True)
class StepView(_JsonModel):
    """A persisted execution checkpoint with optional resolved native training notice."""

    step_id: str
    tool_name: str
    status: str
    attempts: int
    training_notice: str | None = dataclass_field(default=None, metadata={'omit_none': True})

    def __post_init__(self):
        _identifier(self.step_id, 'step_id')
        _identifier(self.tool_name, 'tool_name')
        if self.status not in ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'SKIPPED'):
            raise ValueError('Invalid interactive step status.')
        _natural(self.attempts, 'step attempts')
        if self.training_notice is not None:
            if self.tool_name != 'adapt_epizoo_species' or self.status != 'RUNNING':
                raise ValueError('Training notice requires a running adaptation step.')
            _text(self.training_notice, 'training notice', 2048)


@dataclass(frozen=True)
class TurnView(_JsonModel):
    """A bounded turn projection combining stored presentation and checkpoints."""

    session_id: str
    turn_id: str
    utterance: str
    base_generation: int
    profile_id: str | None = None
    run_id: str | None = None
    revision_id: str | None = None
    status: str = 'submitted'
    response: PresentedResponse | None = None
    error: ClientError | None = None
    steps: tuple[StepView, ...] = ()
    state_revision: int | None = None
    base_revision_id: str | None = None
    attachments: tuple = ()
    export_requested: bool | None = dataclass_field(default=None, metadata={'omit_none': True})

    def __post_init__(self):
        if type(self.attachments) is not tuple or len(self.attachments) > 16:
            raise ValueError('Invalid turn attachments.')
        for item in self.attachments:
            _shape(item, ('resource_id', 'label', 'input_type', 'use'), 'attachment')
            _identifier(item['resource_id'], 'attachment resource')
            _text(item['label'], 'attachment label', 160)
            _identifier(item['input_type'], 'attachment type')
            if item['use'] not in ('staged', 'offered_for_analysis', 'consumed'):
                raise ValueError('Invalid attachment use.')
        object.__setattr__(self, 'attachments', tuple(freeze_json_mapping(a, 'attachment') for a in self.attachments))
        _identifier(self.session_id, 'session_id')
        _identifier(self.turn_id, 'turn_id')
        _text(self.utterance, 'utterance', MAX_UTTERANCE_LENGTH, empty=True)
        _natural(self.base_generation, 'base_generation')
        for name in ('profile_id', 'run_id', 'revision_id', 'base_revision_id'):
            _optional_identifier(getattr(self, name), name)
        _text(self.status, 'turn status', 64)
        if self.response is not None and not isinstance(self.response, PresentedResponse):
            raise TypeError('Interactive turn response must be a PresentedResponse.')
        if self.error is not None and not isinstance(self.error, ClientError):
            raise TypeError('Interactive turn error must be a ClientError.')
        _collection(self.steps, StepView, 'step_id', 'steps')
        if self.state_revision is not None:
            _natural(self.state_revision, 'state_revision')
        if self.export_requested is not None:
            _boolean(self.export_requested, 'export requested')


@dataclass(frozen=True)
class RevisionView(_JsonModel):
    revision_id: str
    parent_revision_id: str | None
    turn_id: str
    is_active: bool
    outputs: tuple[str, ...] = ()
    run_id: str | None = None
    base_generation: int | None = None
    session_generation: int | None = None
    status: str | None = None
    retained_outputs: tuple[str, ...] = ()
    result: PresentedResponse | None = None
    steps: tuple[StepView, ...] = ()
    artifacts: tuple[ArtifactHandle, ...] = ()
    evidence_outputs: tuple[str, ...] = ()
    scientific_artifacts: tuple[ScientificArtifactHandle, ...] = ()
    result_groups: tuple = ()

    def __post_init__(self):
        _identifier(self.revision_id, 'revision_id')
        _optional_identifier(self.parent_revision_id, 'parent_revision_id')
        _identifier(self.turn_id, 'turn_id')
        _boolean(self.is_active, 'is_active')
        if type(self.outputs) is not tuple or len(self.outputs) > MAX_HISTORY_ITEMS:
            raise ValueError('Invalid interactive revision outputs.')
        for name in self.outputs:
            _identifier(name, 'output name')
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError('Duplicate interactive revision output name.')
        _optional_identifier(self.run_id, 'run_id')
        for name in ('base_generation', 'session_generation'):
            if getattr(self, name) is not None:
                _natural(getattr(self, name), name)
        if self.status is not None:
            _identifier(self.status, 'revision status')
        for name in ('retained_outputs', 'evidence_outputs'):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) > MAX_HISTORY_ITEMS or len(set(values)) != len(values):
                raise ValueError('Invalid interactive revision output inventory.')
            for value in values:
                _identifier(value, name)
            if set(values) - set(self.outputs):
                raise ValueError('Revision inventory references an unselected output.')
        if self.result is not None and not isinstance(self.result, PresentedResponse):
            raise TypeError('Invalid interactive revision presentation.')
        _collection(self.steps, StepView, 'step_id', 'steps')
        _collection(self.artifacts, ArtifactHandle, 'handle', 'artifacts')
        if any(a.revision_id != self.revision_id or a.turn_id != self.turn_id for a in self.artifacts):
            raise ValueError('Artifact inventory belongs to a different revision.')
        _collection(self.scientific_artifacts, ScientificArtifactHandle, 'handle', 'scientific artifacts')
        if any(a.revision_id != self.revision_id or a.turn_id != self.turn_id
               for a in self.scientific_artifacts):
            raise ValueError('Scientific artifact inventory belongs to a different revision.')
        if type(self.result_groups) is not tuple or len(self.result_groups) > 32:
            raise ValueError('Invalid revision result groups.')
        for group in self.result_groups:
            if group['canonical_output'] not in self.outputs or any(o['name'] not in self.outputs for o in group['outputs']):
                raise ValueError('Result group references an unselected output.')
        object.__setattr__(self, 'result_groups', tuple(freeze_json_mapping(g, 'result group') for g in self.result_groups))


@dataclass(frozen=True)
class SessionView(_JsonModel):
    session_id: str
    generation: int
    active_revision_id: str | None
    revisions: tuple[RevisionView, ...] = ()
    turns: tuple[TurnView, ...] = ()
    history_truncated: bool = False

    def __post_init__(self):
        _identifier(self.session_id, 'session_id')
        _natural(self.generation, 'generation')
        _optional_identifier(self.active_revision_id, 'active_revision_id')
        _collection(self.revisions, RevisionView, 'revision_id', 'revisions')
        _collection(self.turns, TurnView, 'turn_id', 'turns')
        _boolean(self.history_truncated, 'history_truncated')
        if any(turn.session_id != self.session_id for turn in self.turns):
            raise ValueError('Interactive turn belongs to a different session.')
        if any(revision.is_active != (revision.revision_id == self.active_revision_id)
               for revision in self.revisions):
            raise ValueError('Interactive revision active marker is inconsistent.')


@dataclass(frozen=True)
class ArtifactHandle(_JsonModel):
    """An application-issued artifact identity; resolution stays server-side."""

    handle: str
    artifact_type: str
    sha256: str
    revision_id: str | None
    turn_id: str
    figure: object = dataclass_field(default=None, kw_only=True, metadata={'omit_none': True})

    def __post_init__(self):
        for name in ('handle', 'artifact_type', 'turn_id'):
            _identifier(getattr(self, name), name)
        _optional_identifier(self.revision_id, 'revision_id')
        if self.revision_id is None and self.artifact_type not in ('analysis_figure', 'scientific_composition'):
            raise ValueError('Only Interaction presentations and recipes can omit a scientific revision.')
        _sha256(self.sha256, 'artifact')
        if self.figure is not None:
            if self.artifact_type != 'analysis_figure':
                raise ValueError('Figure provenance belongs to a scientific figure.')
            object.__setattr__(self, 'figure', freeze_json_mapping(self.figure, 'figure provenance'))


@dataclass(frozen=True)
class ScientificArtifactHandle(ArtifactHandle):
    """Eligible accepted identity; payload integrity is checked on download."""

    size_bytes: int | None
    filename: str = 'matrix.h5ad'
    label: str = 'Original accepted matrix'
    title: str = dataclass_field(init=False)
    purpose: str = dataclass_field(init=False)
    description: str = dataclass_field(init=False)
    contents: object = dataclass_field(default=None, kw_only=True, metadata={'omit_none': True})

    def __post_init__(self):
        super().__post_init__()
        if self.revision_id is None and self.artifact_type != 'scientific_composition':
            raise ValueError('Scientific artifacts require an accepted revision.')
        if self.artifact_type == 'scientific_composition' and self.revision_id is not None:
            raise ValueError('A composed export recipe belongs to its Interaction.')
        # An unprepared integrated derivative has no known delivered size.
        # Server-only payload sizes separately own snapshot/capacity admission.
        if self.size_bytes is not None or self.artifact_type not in ('scientific_integrated', 'scientific_composition'):
            _natural(self.size_bytes, 'scientific artifact size')
        reviewed = {
            ('scientific_matrix', 'matrix.h5ad', 'Original accepted matrix'),
            ('scientific_qc_table', 'barcodes.tsv.gz', 'Barcode QC'),
            ('scientific_qc_table', 'lengths.tsv.gz', 'Fragment length distribution'),
            ('scientific_selection_table', 'decisions.tsv.gz', 'Cell-selection decisions'),
            ('scientific_selection_table', 'selected.tsv.gz', 'Selected cells'),
            ('scientific_analysis', 'neighbors.h5ad', 'Cell neighbors — safe export'),
            ('scientific_analysis', 'analysis.h5ad', 'Embeddings, Leiden and UMAP — safe export'),
            ('scientific_analysis', 'analysis.h5ad', 'Embeddings and UMAP — safe export'),
            ('scientific_analysis', 'embeddings.h5ad', 'Cell embeddings — safe export'),
            ('scientific_analysis', 'clusters.h5ad', 'Embeddings and Leiden — safe export'),
            ('scientific_annotation', 'annotated.h5ad', 'Annotated cells — safe export'),
            ('scientific_reference_evidence', 'reference-evidence.tsv', 'Cell-type reference evidence'),
            ('scientific_integrated', 'integrated-embeddings.h5ad', 'Original data with accepted analysis'),
            ('scientific_integrated', 'integrated-clusters.h5ad', 'Original data with accepted analysis'),
            ('scientific_integrated', 'integrated-analysis.h5ad', 'Original data with accepted analysis'),
            ('scientific_composition', 'composed-analysis.h5ad', 'Original data with selected accepted results'),
            ('scientific_model', 'epizoox-checkpoint.pth', 'EpiZooX model checkpoint'),
            ('scientific_model', 'epizoox-model.json', 'Model configuration and reuse requirements'),
            ('scientific_model', 'epizoox-features.txt', 'Exact ordered EpiZooX feature vocabulary'),
            ('scientific_model', 'epizoox-document-frequency.npy', 'EpiZooX input-corpus document frequency'),
        }
        reviewed.update((kind, filename, prefix + ': ' + label)
            for kind, filename, label in tuple(reviewed) if kind == 'scientific_model'
            for prefix in ('Initialized', 'Post-trained'))
        if (self.size_bytes == 0
                or (self.artifact_type, self.filename, self.label) not in reviewed):
            raise ValueError('Unsupported scientific artifact descriptor.')
        if self.contents is not None:
            from .dialogue_evidence import _unsafe_fact_value
            contents = freeze_json_mapping(self.contents, 'export contents')
            if (self.artifact_type != 'scientific_composition'
                    or len(json.dumps(_serialize(contents), ensure_ascii=False).encode()) > 65_536
                    or _unsafe_fact_value(contents)):
                raise ValueError('Unsafe or unbounded composed export description.')
            object.__setattr__(self, 'contents', contents)
        elif self.artifact_type == 'scientific_composition':
            raise ValueError('Composed exports require an exact content description.')
        for name, value in self._description().items():
            object.__setattr__(self, name, value)

    def _description(self):
        """One client descriptor owns purpose/wording for chat and result details.

        These are transient delivery descriptions, never accepted result state.
        Existing labels and handles remain compatible with historical clients.
        """
        if self.artifact_type == 'scientific_composition':
            title, purpose, description = ('Selected results with original measurements (H5AD)',
                'Integrated annotated H5AD', 'Prepared when downloaded. Original measurements and features '
                'with only the explicitly selected accepted fields added; source metadata follows the existing privacy policy.')
        elif self.artifact_type == 'scientific_integrated':
            stage = {'integrated-embeddings.h5ad': 'cell embeddings', 'integrated-clusters.h5ad': 'Leiden clusters',
                     'integrated-analysis.h5ad': 'UMAP analysis'}[self.filename]
            title, purpose, description = ('Original data with ' + stage + ' (H5AD)', 'Integrated annotated H5AD',
                'Integrated annotated H5AD — prepared when downloaded. '
                'Continue analysis with original measurements, features and safe metadata plus this accepted result.')
        elif self.artifact_type == 'scientific_analysis':
            stage = {'embeddings.h5ad': 'Cell embeddings', 'clusters.h5ad': 'Leiden clusters',
                     'analysis.h5ad': 'UMAP analysis', 'neighbors.h5ad': 'Neighbor graph'}[self.filename]
            title, purpose, description = (stage + ' (compact H5AD)', 'Compact analysis-result H5AD',
                'Lightweight cell analysis and recorded methods. No original matrix or feature axis; X=None.')
        elif self.artifact_type == 'scientific_annotation':
            title, purpose, description = ('Annotated original data (H5AD)', 'Integrated annotated H5AD',
                'Original measurements and features with accepted cell annotations; private metadata omitted.')
        elif self.artifact_type == 'scientific_matrix':
            title, purpose, description = (self.label, 'Scientific datasets and tables',
                'Accepted scientific matrix with its original measurement semantics.')
        else:
            title, purpose, description = (self.label, 'Scientific datasets and tables',
                'Accepted tool output; provenance and method details are available with the result.')
        return dict(title=title, purpose=purpose, description=description)


@dataclass(frozen=True)
class EvidenceView(DialogueEvidence):
    """The existing accepted evidence contract with client-safe whole fields.

    Path/configuration-bearing values are explicitly omitted by the facade.
    Source anchors, coverage, detail provenance and scientific limitations retain
    their existing meanings; this adds no evidence authority or reconstruction.
    """
    supported_details: tuple[str, ...] = ()
    field_labels: Mapping[str, str] = dataclass_field(default_factory=dict)

    def __post_init__(self):
        for name in ('session_id', 'revision_id', 'output_name', 'evidence_scope'):
            _identifier(getattr(self, name), name)
        if self.status not in ('available', 'unavailable', 'unsupported'):
            raise ValueError('Invalid interactive evidence availability.')
        if self.generation is not None:
            _natural(self.generation, 'generation')
        if self.is_active is not None:
            _boolean(self.is_active, 'is_active')
        if self.source is not None and not isinstance(self.source, EvidenceSource):
            raise TypeError('Invalid interactive evidence source.')
        for name in ('facts', 'coverage', 'detail'):
            _collection(getattr(self, name), EvidenceFact, 'field', name)
        _strings(self.unselected_fields, 'unselected_fields', maximum=128)
        _strings(self.limitations, 'limitations', maximum=128)
        if (type(self.supported_details) is not tuple or len(self.supported_details) > 5
                or len(set(self.supported_details)) != len(self.supported_details)
                or any(s not in ('annotation_rationale', 'reference_associations', 'length_histogram', 'categorical_fields',
                                 'category_counts', 'category_association') for s in self.supported_details)):
            raise ValueError('Invalid supported evidence detail inventory.')
        available = {fact.field for name in ('facts', 'coverage', 'detail') for fact in getattr(self, name)}
        if not isinstance(self.field_labels, Mapping) or set(self.field_labels) - available:
            raise ValueError('Invalid evidence field labels.')
        for name, label in self.field_labels.items():
            _identifier(name, 'evidence field')
            _text(label, 'evidence label', 4096)
        object.__setattr__(self, 'field_labels', freeze_json_mapping(self.field_labels, 'field_labels'))
        if len(json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False).encode('utf-8')) > 131_072:
            raise ValueError('Interactive evidence exceeds the client byte limit.')


@dataclass(frozen=True)
class ArtifactContent:
    """Bounded server-controlled delivery, with distinct source/delivery digests."""
    reference: ArtifactHandle
    content_type: str
    filename: str
    data: bytes
    content_sha256: str
    presentation: str
    limitations: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.reference, ArtifactHandle):
            raise TypeError('Invalid artifact delivery reference.')
        if self.content_type not in ('text/plain; charset=utf-8', 'image/png'):
            raise ValueError('Unsupported artifact delivery content type.')
        if (type(self.filename) is not str or not self.filename.startswith('agent-')
                or len(self.filename) > 128 or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-.' for c in self.filename)):
            raise ValueError('Invalid server-controlled artifact filename.')
        if type(self.data) is not bytes or len(self.data) > 16 * 1024 * 1024:
            raise ValueError('Artifact delivery exceeds the byte bound.')
        if self.presentation not in ('client_safe_report_projection', 'accepted_png'):
            raise ValueError('Invalid artifact presentation kind.')
        expected = ('analysis_figure', 'image/png') if self.presentation == 'accepted_png' else (
            'analysis_report', 'text/plain; charset=utf-8')
        if (self.reference.artifact_type, self.content_type) != expected:
            raise ValueError('Artifact delivery type and presentation disagree.')
        import hashlib
        if hashlib.sha256(self.data).hexdigest() != self.content_sha256:
            raise ValueError('Artifact delivery digest mismatch.')
        if self.presentation == 'accepted_png' and self.content_sha256 != self.reference.sha256:
            raise ValueError('Accepted figure delivery changed its pinned bytes.')
        _strings(self.limitations, 'artifact limitations', maximum=32)


__all__ = ['ArtifactHandle', 'ScientificArtifactHandle', 'ArtifactContent', 'EvidenceView', 'ClientError', 'ModelChoice', 'PresentedResponse',
           'RevisionView', 'SessionView', 'StepView', 'TurnView']
