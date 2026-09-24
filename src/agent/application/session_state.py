"""Versioned navigation metadata. These records never establish scientific authority."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from agent.schemas.orchestration import freeze_json_mapping, _serialize


class SessionError(ValueError):
    """Invalid, unavailable, or corrupt session metadata."""


class SessionConflictError(SessionError):
    """An identity or captured generation conflicts with durable state."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def text(value):
    if type(value) is not str or not value.strip():
        raise SessionError('Expected nonempty session identifier.')


def sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise SessionError('Invalid session digest.')


def natural(value):
    if type(value) is not int or value < 0:
        raise SessionError('Invalid session generation.')


@dataclass(frozen=True)
class OutputSelection:
    """Explicit result member to retain; never an executable plan input."""
    name: str
    step_id: str
    output_key: str

    def __post_init__(self):
        for value in (self.name, self.step_id, self.output_key):
            text(value)


@dataclass(frozen=True)
class OutputLocator:
    name: str
    run_id: str
    step_id: str
    output_key: str
    accepted_step_sha256: str

    def __post_init__(self):
        for value in (self.name, self.run_id, self.step_id, self.output_key):
            text(value)
        sha(self.accepted_step_sha256)


@dataclass(frozen=True)
class AnalysisRevision:
    revision_id: str
    parent_revision_id: str | None
    turn_id: str
    request_id: str
    run_id: str
    run_result_sha256: str
    outputs: tuple[OutputLocator, ...]

    def __post_init__(self):
        for value in (self.revision_id, self.turn_id, self.request_id, self.run_id):
            text(value)
        if self.parent_revision_id is not None:
            text(self.parent_revision_id)
        sha(self.run_result_sha256)
        _unique(self.outputs, OutputLocator, 'name')
        if not self.outputs:
            raise SessionError('Revision requires explicit outputs.')


@dataclass(frozen=True)
class CompletionFile:
    path: str
    sha256: str

    def __post_init__(self):
        text(self.path)
        sha(self.sha256)


@dataclass(frozen=True)
class SessionTurn:
    turn_id: str
    base_revision_id: str | None
    base_generation: int
    request_id: str | None
    request_sha256: str | None
    run_id: str | None
    status: str
    selections: tuple[OutputSelection, ...] = ()
    revision_id: str | None = None
    run_result_sha256: str | None = None
    completion_files: tuple[CompletionFile, ...] = ()
    retained_outputs: tuple[OutputLocator, ...] = ()

    def __post_init__(self):
        text(self.turn_id)
        natural(self.base_generation)
        for value in (self.base_revision_id, self.request_id, self.run_id, self.revision_id):
            if value is not None:
                text(value)
        for value in (self.request_sha256, self.run_result_sha256):
            if value is not None:
                sha(value)
        _unique(self.selections, OutputSelection, 'name')
        _unique(self.completion_files, CompletionFile, 'path')
        _unique(self.retained_outputs, OutputLocator, 'name')
        if {o.name for o in self.retained_outputs} & {o.name for o in self.selections}:
            raise SessionError('Overlapping new and retained active outputs.')
        if self.status not in {'started', 'linked', 'run_succeeded', 'ready', 'activated',
                               'stale', 'failed', 'cancelled', 'planned', 'clarification'}:
            raise SessionError('Unknown turn status.')
        if self.request_id is None:
            if (self.request_sha256 is not None or self.run_id is not None or self.selections
                    or self.completion_files or self.retained_outputs or self.run_result_sha256 is not None
                    or self.status not in {'activated', 'clarification', 'failed', 'cancelled'}
                    or (self.status != 'activated' and self.revision_id is not None)):
                raise SessionError('Invalid navigation-only turn.')
        else:
            if self.request_sha256 is None or not self.selections:
                raise SessionError('Scientific turn requires a request anchor and explicit selections.')
            if self.status == 'clarification':
                raise SessionError('Clarification cannot be associated with scientific execution.')
            if self.status == 'started':
                if self.run_id is not None:
                    raise SessionError('Started turn cannot already have a run.')
            elif self.run_id != self.request_id + ':run':
                raise SessionError('Turn/run identity mismatch.')
            if self.status in {'ready', 'activated', 'stale'}:
                if self.run_result_sha256 is None or not self.completion_files:
                    raise SessionError('Activation requires recorded application completion.')
            elif self.completion_files or self.run_result_sha256 is not None or self.revision_id is not None:
                raise SessionError('Premature completion data.')
        if self.status in {'activated', 'stale'} and self.revision_id is None:
            raise SessionError('Completed activation decision requires a revision.')
        if self.status == 'ready' and self.revision_id is not None:
            raise SessionError('Ready turn cannot already have a revision.')


@dataclass(frozen=True)
class Navigation:
    turn_id: str
    from_revision_id: str | None
    to_revision_id: str
    generation: int

    def __post_init__(self):
        text(self.turn_id)
        text(self.to_revision_id)
        if self.from_revision_id is not None:
            text(self.from_revision_id)
        natural(self.generation)


def _unique(values, cls, key):
    if type(values) is not tuple or any(not isinstance(v, cls) for v in values):
        raise SessionError('Invalid session record collection.')
    if len({getattr(v, key) for v in values}) != len(values):
        raise SessionError('Duplicate session record identity.')


@dataclass(frozen=True)
class Interaction:
    turn_id: str
    utterance: str
    base_revision_id: str
    base_generation: int
    snapshot: object
    status: str = 'interpreting'
    admitted: object = None
    guidance_candidates: object = None

    def __post_init__(self):
        for v in (self.turn_id, self.utterance, self.base_revision_id): text(v)
        natural(self.base_generation)
        if self.status not in {'interpreting', 'admitted', 'submitted', 'clarification', 'navigated', 'answered', 'failed'}:
            raise SessionError('Invalid interaction state.')
        object.__setattr__(self, 'snapshot', freeze_json_mapping(self.snapshot, 'interaction.snapshot'))
        if self.admitted is not None:
            object.__setattr__(self, 'admitted', freeze_json_mapping(self.admitted, 'interaction.admitted'))
        if self.guidance_candidates is not None:
            refs = freeze_json_mapping({'refs': self.guidance_candidates}, 'guidance references')['refs']
            object.__setattr__(self, 'guidance_candidates', refs)
            if self.admitted is None or self.admitted.get('intent') != 'guidance' or self.status != 'answered':
                raise SessionError('Candidate references require an answered guidance interaction.')


@dataclass(frozen=True)
class AnalysisSession:
    session_id: str
    schema_version: int = 1
    active_revision_id: str | None = None
    generation: int = 0
    revisions: tuple[AnalysisRevision, ...] = ()
    turns: tuple[SessionTurn, ...] = ()
    navigation: tuple[Navigation, ...] = ()
    interactions: tuple[Interaction, ...] = ()

    def __post_init__(self):
        text(self.session_id)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise SessionError('Unsupported session schema.')
        natural(self.generation)
        _unique(self.revisions, AnalysisRevision, 'revision_id')
        _unique(self.turns, SessionTurn, 'turn_id')
        _unique(self.navigation, Navigation, 'turn_id')
        _unique(self.interactions, Interaction, 'turn_id')
        revisions = {}
        turns = {t.turn_id: t for t in self.turns}
        requests = [t.request_id for t in self.turns if t.request_id is not None]
        if len(requests) != len(set(requests)):
            raise SessionError('Request belongs to multiple session turns.')
        for r in self.revisions:
            if r.revision_id != digest({'session_id': self.session_id, 'turn_id': r.turn_id}):
                raise SessionError('Revision identity does not match its originating turn.')
            if r.parent_revision_id is not None and r.parent_revision_id not in revisions:
                raise SessionError('Unknown or nonhistorical revision parent.')
            t = turns.get(r.turn_id)
            if (t is None or t.status not in {'activated', 'stale'} or t.revision_id != r.revision_id
                    or t.request_id != r.request_id or t.run_id != r.run_id
                    or t.base_revision_id != r.parent_revision_id
                    or t.run_result_sha256 != r.run_result_sha256
                    or tuple((o.name, o.step_id, o.output_key) for o in r.outputs[:len(t.selections)])
                    != tuple((s.name, s.step_id, s.output_key) for s in t.selections)
                    or any(o.run_id != r.run_id for o in r.outputs[:len(t.selections)])
                    or r.outputs[len(t.selections):] != t.retained_outputs):
                raise SessionError('Revision/turn binding mismatch.')
            revisions[r.revision_id] = r
        for t in self.turns:
            if t.retained_outputs:
                parent = revisions.get(t.base_revision_id)
                if parent is None or any(o not in parent.outputs for o in t.retained_outputs):
                    raise SessionError('Retained output was not in the captured base revision.')
        active = None
        history = [None]
        for i, event in enumerate(self.navigation, 1):
            t = turns.get(event.turn_id)
            if (event.generation != i or event.from_revision_id != active
                    or event.to_revision_id not in revisions or t is None
                    or t.status != 'activated' or t.revision_id != event.to_revision_id
                    or t.base_generation != i - 1 or t.base_revision_id != active):
                raise SessionError('Invalid navigation history.')
            active = event.to_revision_id
            history.append(active)
        if self.generation != len(self.navigation) or self.active_revision_id != active:
            raise SessionError('Active pointer/generation does not match history.')
        activated = {e.turn_id for e in self.navigation}
        for t in self.turns:
            if (t.base_generation > self.generation or history[t.base_generation] != t.base_revision_id
                    or (t.status == 'activated') != (t.turn_id in activated)
                    or (t.revision_id is not None and t.revision_id not in revisions)):
                raise SessionError('Invalid turn base or outcome.')
        for interaction_index, interaction in enumerate(self.interactions):
            if (interaction.base_generation > self.generation
                    or history[interaction.base_generation] != interaction.base_revision_id):
                raise SessionError('Invalid captured interaction base.')
            captured = _serialize(interaction.snapshot)
            if set(captured) not in ({'relations', 'bases'}, {'relations', 'bases', 'dialogue'}):
                raise SessionError('Invalid interaction snapshot.')
            expected = {'current': interaction.base_revision_id}
            parent = revisions[interaction.base_revision_id].parent_revision_id
            if parent is not None: expected['parent'] = parent
            previous = next((e.from_revision_id for e in reversed(self.navigation[:interaction.base_generation])
                             if e.to_revision_id != e.from_revision_id), None)
            if previous is not None: expected['previous_active'] = previous
            if captured['relations'] != expected or set(captured['bases']) != set(expected.values()):
                raise SessionError('Interaction revision relations changed.')
            for rid, base in captured['bases'].items():
                if base.get('revision_id') != rid or base.get('outputs') != [asdict(o) for o in revisions[rid].outputs]:
                    raise SessionError('Interaction outputs differ from captured revisions.')
            if 'dialogue' in captured:
                from .scientific_dialogue import validate_capture
                validate_capture(self, interaction, self.interactions[:interaction_index])

    def to_dict(self):
        # MappingProxyType in frozen interaction payloads is not deepcopyable.
        from dataclasses import fields
        value = {f.name: tuple(asdict(v) for v in getattr(self, f.name))
                 if f.name in {'revisions', 'turns', 'navigation'} else getattr(self, f.name)
                 for f in fields(self) if f.name != 'interactions'}
        if self.interactions:
            value['interactions'] = tuple({f.name: _serialize(getattr(i, f.name)) for f in fields(i)
                if f.name != 'guidance_candidates' or i.guidance_candidates is not None} for i in self.interactions)
        for turn in value['turns']:
            if not turn['retained_outputs']:
                del turn['retained_outputs']
        return value

    def turn(self, turn_id):
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        raise SessionError('Unknown session turn.')

    @classmethod
    def from_dict(cls, value):
        def decode(model, data, children=None):
            if model is cls and type(data) is dict and 'interactions' not in data:
                data = dict(data, interactions=[])
            if model is SessionTurn and type(data) is dict and 'retained_outputs' not in data:
                data = dict(data, retained_outputs=[])
            if model is Interaction and type(data) is dict and 'guidance_candidates' not in data:
                data = dict(data, guidance_candidates=None)
            if type(data) is not dict or set(data) != set(model.__dataclass_fields__):
                raise SessionError('Invalid session record shape.')
            data = dict(data)
            for key, (child, nested) in (children or {}).items():
                if type(data[key]) is not list:
                    raise SessionError('Invalid session record array.')
                data[key] = tuple(decode(child, v, nested) for v in data[key])
            return model(**data)
        return decode(cls, value, {
            'revisions': (AnalysisRevision, {'outputs': (OutputLocator, None)}),
            'turns': (SessionTurn, {'selections': (OutputSelection, None),
                                    'completion_files': (CompletionFile, None),
                                    'retained_outputs': (OutputLocator, None)}),
            'navigation': (Navigation, None),
            'interactions': (Interaction, None),
        })
