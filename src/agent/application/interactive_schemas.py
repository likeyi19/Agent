"""Bounded presentation contracts for transport-independent interactive clients.

These views contain display history and stable identities, never scientific
authority, executable bindings, provider configuration, or storage locations.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json

from agent.schemas.orchestration import _JsonModel, freeze_json_mapping


MAX_IDENTIFIER_LENGTH = 256
MAX_UTTERANCE_LENGTH = 16_384
MAX_PRESENTATION_LENGTH = 131_072
MAX_PRESENTATION_BYTES = 262_144
MAX_HISTORY_ITEMS = 1_000


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


def _scientific(value):
    _shape(value, ('support', 'limitations', 'evidence_scope'), 'scientific presentation')
    _support(value['support'])
    _strings(value['limitations'], 'scientific limitations')
    _identifier(value['evidence_scope'], 'evidence_scope')


def _guidance(value):
    _shape(value, ('candidates', 'limitations'), 'guidance presentation')
    _strings(value['limitations'], 'guidance limitations')
    candidates = value['candidates']
    if not isinstance(candidates, (list, tuple)) or len(candidates) > 4:
        raise ValueError('Invalid interactive guidance candidates.')
    for item in candidates:
        _shape(item, ('candidate_id', 'origin_turn_id', 'option', 'capability',
                      'text', 'support', 'limitations'), 'guidance candidate')
        for key in ('candidate_id', 'origin_turn_id', 'capability'):
            _identifier(item[key], key)
        if type(item['option']) is not int or not 1 <= item['option'] <= 4:
            raise ValueError('Invalid interactive guidance option.')
        _text(item['text'], 'candidate text', MAX_PRESENTATION_LENGTH, empty=True)
        _support(item['support'])
        _strings(item['limitations'], 'candidate limitations')
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
    version: int = 1

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
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
                if self.kind != kind:
                    raise ValueError(f'Invalid {name} response kind.')
                validator(value)
                object.__setattr__(self, name, freeze_json_mapping(value, name))
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False).encode('utf-8')
        if len(encoded) > MAX_PRESENTATION_BYTES:
            raise ValueError('Interactive presentation exceeds its byte limit.')

    @classmethod
    def from_dict(cls, value):
        _shape(value, ('version', 'kind', 'status', 'text', 'clarification',
                      'scientific', 'guidance', 'error'), 'presentation')
        data = dict(value)
        if data['error'] is not None:
            data['error'] = ClientError.from_dict(data['error'])
        return cls(**data)


@dataclass(frozen=True)
class StepView(_JsonModel):
    """A persisted execution checkpoint; no parameters or liveness inference."""

    step_id: str
    tool_name: str
    status: str
    attempts: int

    def __post_init__(self):
        _identifier(self.step_id, 'step_id')
        _identifier(self.tool_name, 'tool_name')
        if self.status not in ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'SKIPPED'):
            raise ValueError('Invalid interactive step status.')
        _natural(self.attempts, 'step attempts')


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

    def __post_init__(self):
        _identifier(self.session_id, 'session_id')
        _identifier(self.turn_id, 'turn_id')
        _text(self.utterance, 'utterance', MAX_UTTERANCE_LENGTH, empty=True)
        _natural(self.base_generation, 'base_generation')
        for name in ('profile_id', 'run_id', 'revision_id'):
            _optional_identifier(getattr(self, name), name)
        _text(self.status, 'turn status', 64)
        if self.response is not None and not isinstance(self.response, PresentedResponse):
            raise TypeError('Interactive turn response must be a PresentedResponse.')
        if self.error is not None and not isinstance(self.error, ClientError):
            raise TypeError('Interactive turn error must be a ClientError.')
        _collection(self.steps, StepView, 'step_id', 'steps')
        if self.state_revision is not None:
            _natural(self.state_revision, 'state_revision')


@dataclass(frozen=True)
class RevisionView(_JsonModel):
    revision_id: str
    parent_revision_id: str | None
    turn_id: str
    is_active: bool
    outputs: tuple[str, ...] = ()

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
    revision_id: str
    turn_id: str

    def __post_init__(self):
        for name in ('handle', 'artifact_type', 'revision_id', 'turn_id'):
            _identifier(getattr(self, name), name)
        if (type(self.sha256) is not str or len(self.sha256) != 64
                or any(char not in '0123456789abcdef' for char in self.sha256)):
            raise ValueError('Invalid interactive artifact digest.')


__all__ = ['ArtifactHandle', 'ClientError', 'ModelChoice', 'PresentedResponse',
           'RevisionView', 'SessionView', 'StepView', 'TurnView']
