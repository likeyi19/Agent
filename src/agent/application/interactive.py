"""Transport-independent client of the existing durable Agent application.

Submission/presentation records belong to the Session. This facade owns neither
scientific state nor a job queue; callers may run its synchronous submit method
in a server worker and poll the existing durable checkpoints.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields as dataclass_fields, replace
import hashlib
import io
import json
from pathlib import Path
from types import MappingProxyType

from agent.orchestration import PlanningModelProfile, PlanningRecoveryPolicy, PlanningWireMode
from agent.orchestration.error_policy import safe_message_for
from agent.orchestration.run_store import RunNotFoundError
from agent.providers import PlanningModelFactoryRegistry, build_default_planning_model_factory_registry
from agent.schemas.orchestration import _serialize, freeze_json_mapping

from .interactive_schemas import (
    ArtifactHandle, ClientError, ModelChoice, PresentedResponse, RevisionView,
    SessionView, StepView, TurnView, EvidenceView, ArtifactContent, ScientificArtifactHandle,
)
from .service import RESERVED_APPLICATION_INPUTS, ResearchAgentApplication
from .session_state import SessionConflictError, SessionError, canonical, digest
from .local_resources import (LocalResourceAdmission, RegisteredInput, RegisteredInputCollection, ResourceAdmissionError,
    qualified_epizoo_resources)


_MESSAGES = {
    'INTERACTIVE_INPUT_INVALID': 'The interactive request is invalid.',
    'INTERACTIVE_SESSION_INVALID': 'The session is unavailable or invalid.',
    'INTERACTIVE_GENERATION_CONFLICT': 'The session changed. Reopen it before submitting a new turn.',
    'INTERACTIVE_TURN_CONFLICT': 'This turn identifier already belongs to a different submission.',
    'INTERACTIVE_REFERENCE_INVALID': 'The requested turn, revision, or reference is unavailable.',
    'INTERACTIVE_MODEL_UNAVAILABLE': 'The requested model profile is not offered or is unavailable.',
    'INTERACTIVE_OPERATION_ACTIVE': 'This turn is already being processed.',
    'INTERACTIVE_APPLICATION_FAILED': 'The application could not complete this turn.',
    'INTERACTIVE_PRESENTATION_UNAVAILABLE': 'No stored displayed response is available for this historical turn.',
    'INTERACTIVE_ARTIFACT_UNAVAILABLE': 'The accepted presentation artifact is unavailable.',
    'INTERACTIVE_SCIENTIFIC_ARTIFACT_UNAVAILABLE': 'The accepted scientific file is unavailable.',
}


_MATRIX_CONTRACTS = {
    'build_scATAC_cell_by_ccre': frozenset({'scatac-cell-by-ccre.v1'}),
    'adopt_scATAC_cell_by_ccre': frozenset({'scatac-cell-by-ccre.external.v1'}),
    'build_scATAC_cell_by_features': frozenset({
        'scatac-cell-by-features.v1', 'scatac-cell-by-features.qc-selected.v1'}),
    'adopt_scATAC_cell_by_features': frozenset({'scatac-cell-by-features.external.v1'}),
}

_TABLE_TOOLS = {
    'compute_scATAC_qc': ('scatac-barcode-qc.v1', 'barcode_qc',
        (('table', 'scientific_qc_table', 'barcodes.tsv.gz', 'Barcode QC'),
         ('histogram', 'scientific_qc_table', 'lengths.tsv.gz', 'Fragment length distribution'))),
    'select_scATAC_cells': ('scatac-cell-selection.v1', 'selected_cells',
        (('decisions', 'scientific_selection_table', 'decisions.tsv.gz', 'Cell-selection decisions'),
         ('selected', 'scientific_selection_table', 'selected.tsv.gz', 'Selected cells'))),
}


@dataclass(frozen=True)
class _ScientificMatrixArtifact:
    """Server-only accepted locator, never a client-supplied path selector."""

    reference: ScientificArtifactHandle
    path: Path
    managed_root: Path
    sha256: str
    size_bytes: int
    contract: str
    manifest: Mapping
    role: str = 'matrix'


class InteractiveBoundaryError(ValueError):
    """A safe boundary error; the original typed exception remains its cause."""

    def __init__(self, code):
        self.error = ClientError(code, _MESSAGES[code])
        super().__init__(self.error.message)


def _fail(code):
    return InteractiveBoundaryError(code)


def _identifier(value):
    if (type(value) is not str or not value.strip() or len(value) > 256
            or any(not c.isprintable() for c in value)):
        raise _fail('INTERACTIVE_INPUT_INVALID')
    return value


def _inputs(values):
    """Accept ordinary structured science inputs, never client binding objects."""
    try:
        frozen = freeze_json_mapping({} if values is None else values, 'execution_inputs')
        values = _serialize(frozen)
        if RESERVED_APPLICATION_INPUTS.intersection(values) or values.get('overwrite') is True:
            raise ValueError('Reserved application inputs.')
        def check(value):
            if isinstance(value, dict):
                if any(k in value for k in ('$ref', '$prior_output', 'authority_payload')):
                    raise ValueError('Internal binding input.')
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
        check(values)
        return values
    except (ValueError, TypeError, RecursionError) as exc:
        raise _fail('INTERACTIVE_INPUT_INVALID') from exc


def _unsafe_fact_field(field):
    return field == 'path' or field.endswith('_path') or field in {
        'workspace_root', 'authority_payload', 'api_key', 'credentials',
        'provider_endpoint', 'base_url', 'provider_prompt',
    }


def _unsafe_fact_value(value):
    """Omit a whole unsafe field; never rewrite or reinterpret scientific values."""
    if isinstance(value, Mapping):
        return any(_unsafe_fact_field(k) or _unsafe_fact_value(child) for k, child in value.items())
    if isinstance(value, (tuple, list)):
        return any(_unsafe_fact_value(child) for child in value)
    return isinstance(value, str) and (value.startswith(('/', '~/', 'file://'))
        or len(value) > 2 and value[0].isalpha() and value[1:3] in (':\\', ':/'))


def _present(outcome, state=None, *, turn_id=None):
    """Store the displayed response, not a second copy of scientific evidence."""
    if outcome.presentation is not None:
        return outcome.presentation
    text = outcome.text
    clarification = None if outcome.clarification is None else asdict(outcome.clarification)
    scientific = None
    if outcome.scientific is not None:
        value = outcome.scientific
        targets = tuple(dict.fromkeys((c.source['revision_id'], c.source['output_locator']['name'], c.subject)
            for c in value.claims))
        interaction = None if state is None else next((i for i in state.interactions if i.turn_id == turn_id), None)
        admitted = None if interaction is None else interaction.admitted
        if admitted is not None and admitted.get('intent') == 'scientific':
            # A response can use no claim slots while still addressing an exact
            # admitted historical target. Attribution comes from admission,
            # never from prose or the currently active revision.
            targets = tuple(dict.fromkeys((admitted[k]['revision_id'], admitted[k]['output_name'],
                admitted[k]['subject']) for k in ('target', 'comparison') if admitted.get(k) is not None))
        scientific = dict(support=value.support, limitations=value.limitations,
                          evidence_scope=value.evidence_scope,
                          targets=[dict(revision_id=revision, output_name=output, subject=subject)
                                   for revision, output, subject in targets])
    guidance = None
    if outcome.guidance is not None:
        def ordinal(candidate, fallback):
            origin = None if state is None else next((i for i in state.interactions
                if i.turn_id == candidate.reference['origin_turn_id']), None)
            refs = () if origin is None else origin.guidance_candidates or ()
            return next((n + 1 for n, r in enumerate(refs)
                if r['candidate_id'] == candidate.reference['candidate_id']), fallback)
        guidance = dict(candidates=[dict(candidate_id=c.reference['candidate_id'],
            origin_turn_id=c.reference['origin_turn_id'], option=ordinal(c, n + 1),
            **(dict(base_revision_id=c.reference['base_revision_id'], readiness=_serialize(c.readiness))
               if 'base_revision_id' in c.reference else {}),
            capability=c.reference['capability'], text=c.explanation.explanation,
            support=c.explanation.support, limitations=c.explanation.limitations)
            for n, c in enumerate(outcome.guidance.candidates)],
            limitations=outcome.guidance.limitations)
        # Rationale follow-ups can contain a subset. Their display headings use
        # the original inventory ordinals, just like explicit selection does.
        for n, candidate in enumerate(outcome.guidance.candidates):
            text = text.replace(f'Option {n + 1}: {candidate.reference["capability"]}\n',
                f'Option {ordinal(candidate, n + 1)}: {candidate.reference["capability"]}\n')
    return PresentedResponse(kind=outcome.kind, status=outcome.status, text=text,
                             clarification=clarification, scientific=scientific, guidance=guidance,
                             error=None if outcome.error is None else ClientError(outcome.error.code, outcome.error.message))


class InteractiveAgentApplication:
    """Plain Python interactive boundary with exact operator-admitted profiles.

    Each new turn constructs its own configured application/client bundle. Reads
    and completed retries require no provider client. Scientific recovery remains
    explicit and uses the existing planner-free application API.
    """

    def __init__(self, workspace_root, *, model_profiles, default_profile_id,
                 planning_model_factory_registry=None, display_labels=None,
                 registry=None, executor=None, planning_wire_mode=None,
                 recovery_planning_profile=None, planning_recovery_policy=None,
                 approved_source_roots=(), epizoo_resources=()):
        self._epizoo_resources = qualified_epizoo_resources(epizoo_resources)
        self._factory = (build_default_planning_model_factory_registry()
                         if planning_model_factory_registry is None else planning_model_factory_registry)
        if (not isinstance(self._factory, PlanningModelFactoryRegistry)
                or planning_wire_mode is not None and not isinstance(planning_wire_mode, PlanningWireMode)
                or recovery_planning_profile is not None and not isinstance(recovery_planning_profile, PlanningModelProfile)
                or planning_recovery_policy is not None and not isinstance(planning_recovery_policy, PlanningRecoveryPolicy)):
            raise _fail('INTERACTIVE_MODEL_UNAVAILABLE')
        if not isinstance(model_profiles, tuple) or not all(isinstance(p, PlanningModelProfile) for p in model_profiles):
            raise _fail('INTERACTIVE_MODEL_UNAVAILABLE')
        self._profiles = MappingProxyType({p.profile_id: p for p in model_profiles})
        if len(self._profiles) != len(model_profiles):
            raise _fail('INTERACTIVE_MODEL_UNAVAILABLE')
        self._default = default_profile_id
        self._labels = MappingProxyType(dict(display_labels or {}))
        self._config = dict(registry=registry, executor=executor)
        self._planning = dict(planning_wire_mode=planning_wire_mode,
            recovery_planning_profile=recovery_planning_profile,
            planning_recovery_policy=planning_recovery_policy)
        self._profile(default_profile_id)
        try:
            self._application = ResearchAgentApplication(workspace_root, **self._config)
        except (ValueError, RuntimeError, OSError) as exc:
            raise _fail('INTERACTIVE_APPLICATION_FAILED') from exc
        self._config = dict(registry=self._application.registry, executor=self._application.runtime.executor)
        self.resources = LocalResourceAdmission(self._application._workspace,
            approved_source_roots=approved_source_roots, registry=self._application.registry)
        from .matrix_delivery import MatrixDownloadPreparer
        self._matrix_downloads = MatrixDownloadPreparer()

    def _profile(self, profile_id):
        profile = self._profiles.get(profile_id) if type(profile_id) is str else None
        if (profile is None or not profile.enabled or not profile.supports_structured_output
                or profile.provider_id not in self._factory.provider_ids):
            raise _fail('INTERACTIVE_MODEL_UNAVAILABLE')
        return profile

    def model_choices(self):
        return tuple(ModelChoice(p.profile_id, self._labels.get(p.profile_id, p.model_id),
                     p.profile_id == self._default, p.provider_id, p.model_id)
                     for p in self._profiles.values() if p.enabled and p.supports_structured_output
                     and p.provider_id in self._factory.provider_ids)

    def _load(self, session_id):
        _identifier(session_id)
        try:
            return self._application.sessions.load(session_id)
        except (SessionError, ValueError, RuntimeError, OSError) as exc:
            raise _fail('INTERACTIVE_SESSION_INVALID') from exc

    def create_session(self, session_id):
        _identifier(session_id)
        try:
            self._application.sessions.create(session_id)
        except SessionConflictError as exc:
            raise _fail('INTERACTIVE_TURN_CONFLICT') from exc
        except (SessionError, ValueError, RuntimeError, OSError) as exc:
            raise _fail('INTERACTIVE_SESSION_INVALID') from exc
        return self.reopen_session(session_id)

    def reopen_session(self, session_id, *, limit=100):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise _fail('INTERACTIVE_INPUT_INVALID')
        state = self._load(session_id)
        try:
            revisions = tuple(RevisionView(r.revision_id, r.parent_revision_id, r.turn_id,
                r.revision_id == state.active_revision_id, tuple(o.name for o in r.outputs))
                for r in state.revisions[-limit:])
            interaction_ids = [i.turn_id for i in state.interactions]
            # Legacy metadata has no conversation chronology. Keep it separate
            # at the front so bounded display history retains recent captures.
            ids = [t.turn_id for t in state.turns if t.turn_id not in interaction_ids] + interaction_ids
            return SessionView(state.session_id, state.generation, state.active_revision_id,
                revisions, tuple(self._view(state, key) for key in ids[-limit:]),
                len(state.revisions) > limit or len(ids) > limit)
        except InteractiveBoundaryError:
            raise
        except (ValueError, TypeError, KeyError) as exc:
            raise _fail('INTERACTIVE_SESSION_INVALID') from exc

    def revision(self, session_id, revision_id):
        """Project one exact accepted revision, never reconstruct its science."""
        _identifier(revision_id)
        state = self._load(session_id)
        revision = next((r for r in state.revisions if r.revision_id == revision_id), None)
        if revision is None:
            raise _fail('INTERACTIVE_REFERENCE_INVALID')
        try:
            self._application.sessions._validate_revision(revision)
            turn = state.turn(revision.turn_id)
            view = self._view(state, revision.turn_id)
            try:
                scientific_artifacts = self.scientific_artifact_handles(session_id, revision_id)
            except InteractiveBoundaryError as exc:
                if exc.error.code != 'INTERACTIVE_SCIENTIFIC_ARTIFACT_UNAVAILABLE':
                    raise
                # Download ineligibility never removes the existing scientific
                # result, evidence or presentation details.
                scientific_artifacts = ()
            return RevisionView(revision.revision_id, revision.parent_revision_id, revision.turn_id,
                revision.revision_id == state.active_revision_id, tuple(o.name for o in revision.outputs),
                run_id=revision.run_id, base_generation=turn.base_generation, session_generation=state.generation,
                status=view.status, retained_outputs=tuple(o.name for o in turn.retained_outputs),
                result=view.response, steps=view.steps, artifacts=self.artifact_handles(session_id, revision_id),
                evidence_outputs=tuple(o.name for o in revision.outputs),
                scientific_artifacts=scientific_artifacts)
        except InteractiveBoundaryError:
            raise
        except (SessionError, ValueError, RuntimeError, OSError) as exc:
            raise _fail('INTERACTIVE_REFERENCE_INVALID') from exc

    def activate_revision(self, session_id, turn_id, revision_id, *, expected_generation):
        """Expose existing generation-checked navigation; record no new science."""
        _identifier(turn_id)
        _identifier(revision_id)
        if type(expected_generation) is not int or expected_generation < 0:
            raise _fail('INTERACTIVE_INPUT_INVALID')
        self._load(session_id)
        sessions = self._application.sessions
        try:
            with sessions.processing_lease(session_id, turn_id):
                state = self._load(session_id)
                if any(i.turn_id == turn_id for i in state.interactions):
                    raise _fail('INTERACTIVE_TURN_CONFLICT')
                try:
                    sessions.switch(session_id, turn_id, revision_id, expected_generation=expected_generation)
                except SessionConflictError as exc:
                    latest = self._load(session_id)
                    existing = any(t.turn_id == turn_id for t in latest.turns)
                    code = ('INTERACTIVE_GENERATION_CONFLICT' if not existing
                            and latest.generation != expected_generation else 'INTERACTIVE_TURN_CONFLICT')
                    raise _fail(code) from exc
                return self.reopen_session(session_id)
        except InteractiveBoundaryError:
            raise
        except SessionConflictError as exc:
            raise _fail('INTERACTIVE_OPERATION_ACTIVE') from exc
        except (SessionError, ValueError, RuntimeError, OSError) as exc:
            raise _fail('INTERACTIVE_REFERENCE_INVALID') from exc

    def evidence(self, session_id, revision_id, output_name, *, fields=None,
                 detail_section=None, subject=None, limit=8, candidate=None):
        """Read the existing accepted evidence/detail owner through a safe view."""
        _identifier(revision_id)
        _identifier(output_name)
        self._load(session_id)
        from .dialogue_evidence import DetailRequest, DialogueEvidence
        try:
            if detail_section is None and (subject is not None or candidate is not None or limit != 8):
                raise ValueError('Detail operands require an explicit section.')
            detail = (None if detail_section is None else DetailRequest(detail_section, subject, limit, candidate))
            original = self._application.sessions.evidence(session_id, revision_id, output_name,
                fields=fields, detail=detail)
            values = {f.name: getattr(original, f.name) for f in dataclass_fields(DialogueEvidence)}
            omitted = False
            for name in ('facts', 'coverage', 'detail'):
                safe = []
                for fact in values[name]:
                    if fact.status == 'available' and (_unsafe_fact_field(fact.field)
                            or _unsafe_fact_value(fact.value)):
                        fact = replace(fact, status='omitted', value=None, reason='client_safe_content_omitted')
                        omitted = True
                    safe.append(fact)
                values[name] = tuple(safe)
            if omitted:
                values['limitations'] += ('Path or private configuration fields are explicitly omitted from this client view.',)
            sections = {'annotate_scATAC_cell_types': ('annotation_rationale',),
                        'compute_scATAC_qc': ('length_histogram',)}
            return EvidenceView(**values, supported_details=() if original.source is None
                else sections.get(original.source.tool_name, ()))
        except (ValueError, TypeError, RecursionError) as exc:
            raise _fail('INTERACTIVE_INPUT_INVALID') from exc

    def _submission(self, profile, generation, inputs, predecessor, registered_input=None):
        submission = dict(profile_id=profile.profile_id,
            configuration_sha256=digest(dict(profile=asdict(profile),
                wire_mode=None if self._planning['planning_wire_mode'] is None else self._planning['planning_wire_mode'].value,
                recovery_profile=None if self._planning['recovery_planning_profile'] is None else asdict(self._planning['recovery_planning_profile']),
                recovery_policy=None if self._planning['planning_recovery_policy'] is None else self._planning['planning_recovery_policy'].to_dict())),
            expected_generation=generation, execution_inputs=inputs, predecessor_turn_id=predecessor)
        if registered_input is not None:
            submission['registered_input'] = registered_input.attribution()
        return submission

    @staticmethod
    def _duplicate(state, turn_id, utterance, submission):
        existing = next((i for i in state.interactions if i.turn_id == turn_id), None)
        if existing is not None:
            if existing.utterance != utterance or _serialize(existing.submission) != submission:
                raise _fail('INTERACTIVE_TURN_CONFLICT')
            return True
        if any(t.turn_id == turn_id for t in state.turns):
            raise _fail('INTERACTIVE_TURN_CONFLICT')
        return False

    def _checked_submission(self, session_id, turn_id, utterance, *, expected_generation,
                            execution_inputs=None, predecessor_turn_id=None, profile_id=None,
                            registered_input=None):
        _identifier(turn_id)
        if (type(expected_generation) is not int or expected_generation < 0
                or type(utterance) is not str or not utterance.strip() or len(utterance) > 4096):
            raise _fail('INTERACTIVE_INPUT_INVALID')
        if predecessor_turn_id is not None:
            _identifier(predecessor_turn_id)
        profile = self._profile(self._default if profile_id is None else profile_id)
        if registered_input is not None:
            if execution_inputs is not None or type(registered_input) not in (RegisteredInput, RegisteredInputCollection):
                raise ResourceAdmissionError('LOCAL_RESOURCE_BINDING_INVALID')
            execution_inputs = registered_input.execution_inputs
        inputs = _inputs(execution_inputs)
        submission = self._submission(profile, expected_generation, inputs, predecessor_turn_id, registered_input)
        try:
            if len(canonical(submission)) > 65536:
                raise ValueError('Submission bound exceeded.')
        except (ValueError, TypeError, UnicodeError) as exc:
            raise _fail('INTERACTIVE_INPUT_INVALID') from exc
        state = self._load(session_id)
        duplicate = self._duplicate(state, turn_id, utterance, submission)
        if not duplicate and state.generation != expected_generation:
            raise _fail('INTERACTIVE_GENERATION_CONFLICT')
        if not duplicate and predecessor_turn_id is not None:
            prior = next((i for i in state.interactions if i.turn_id == predecessor_turn_id), None)
            if (prior is None or prior.status != 'answered' or prior.admitted is None
                    or prior.admitted.get('intent') not in {'scientific', 'guidance'}):
                raise _fail('INTERACTIVE_REFERENCE_INVALID')
        if not duplicate and registered_input is not None:
            self.resources.validate_binding(registered_input, verify_source=False)
        return profile, inputs, submission, state, duplicate

    def validate_submission(self, session_id, turn_id, utterance, *, expected_generation,
                            execution_inputs=None, predecessor_turn_id=None, profile_id=None,
                            registered_input=None, epizoo_resources=None):
        """Check a submission without providers, persistence, or execution.

        The normalized fingerprint permits a local transport to coalesce in-flight
        requests. It reserves no identity and grants no execution authority;
        ``submit_turn`` repeats admission under the existing processing lease.
        """
        resources = self._epizoo_resources if epizoo_resources is None else qualified_epizoo_resources(epizoo_resources)
        _, _, submission, _, duplicate = self._checked_submission(session_id, turn_id, utterance,
            expected_generation=expected_generation, execution_inputs=execution_inputs,
            predecessor_turn_id=predecessor_turn_id, profile_id=profile_id, registered_input=registered_input)
        if not duplicate and registered_input is not None:
            self.resources.validate_binding(registered_input)
        return digest(dict(session_id=session_id, turn_id=turn_id,
                           utterance=utterance, submission=submission))

    def submit_turn(self, session_id, turn_id, utterance, *, expected_generation,
                    execution_inputs=None, predecessor_turn_id=None, profile_id=None,
                    registered_input=None, epizoo_resources=None):
        resources = self._epizoo_resources if epizoo_resources is None else qualified_epizoo_resources(epizoo_resources)
        profile, inputs, submission, state, duplicate = self._checked_submission(
            session_id, turn_id, utterance, expected_generation=expected_generation,
            execution_inputs=execution_inputs, predecessor_turn_id=predecessor_turn_id,
            profile_id=profile_id, registered_input=registered_input)
        if duplicate:
            return self.turn(session_id, turn_id)
        sessions = self._application.sessions
        try:
            with sessions.processing_lease(session_id, turn_id):
                state = self._load(session_id)
                if self._duplicate(state, turn_id, utterance, submission):
                    return self.turn(session_id, turn_id)
                if state.generation != expected_generation:
                    raise _fail('INTERACTIVE_GENERATION_CONFLICT')
                if registered_input is not None:
                    self.resources.validate_binding(registered_input)
                try:
                    if predecessor_turn_id is not None:
                        prior = next((i for i in state.interactions if i.turn_id == predecessor_turn_id), None)
                        if (prior is None or prior.status != 'answered' or prior.admitted is None
                                or prior.admitted.get('intent') not in {'scientific', 'guidance'}):
                            raise _fail('INTERACTIVE_REFERENCE_INVALID')
                    app = ResearchAgentApplication(self._application.workspace_root,
                        primary_planning_profile=profile, planning_model_factory_registry=self._factory,
                        **self._config, **self._planning)
                except InteractiveBoundaryError:
                    raise
                except (ValueError, RuntimeError, OSError) as exc:
                    raise _fail('INTERACTIVE_MODEL_UNAVAILABLE') from exc
                model = app.runtime.planner.model
                try:
                    outcome = app.sessions.respond(session_id, turn_id, utterance,
                        interpreter=model, answerer=model, expected_generation=expected_generation,
                        execution_inputs=inputs, predecessor_turn_id=predecessor_turn_id,
                        submission=submission, epizoo_resources=resources)
                    presentation = _present(outcome, self._load(session_id), turn_id=turn_id)
                except SessionConflictError as exc:
                    code = ('INTERACTIVE_GENERATION_CONFLICT'
                        if self._load(session_id).generation != expected_generation
                        and not any(i.turn_id == turn_id for i in self._load(session_id).interactions)
                        else 'INTERACTIVE_TURN_CONFLICT')
                    raise _fail(code) from exc
                except Exception as exc:
                    # Sanitized completion is stored only for a captured turn.
                    if not any(i.turn_id == turn_id for i in self._load(session_id).interactions):
                        raise _fail('INTERACTIVE_APPLICATION_FAILED') from exc
                    error = _fail('INTERACTIVE_APPLICATION_FAILED').error
                    presentation = PresentedResponse(kind='execute', status='failed',
                        text=error.message, error=error)
                try:
                    sessions.store_presentation(session_id, turn_id, presentation.to_dict())
                except (SessionError, ValueError, RuntimeError, OSError) as exc:
                    raise _fail('INTERACTIVE_APPLICATION_FAILED') from exc
                return self.turn(session_id, turn_id)
        except SessionConflictError as exc:
            state = self._load(session_id)
            if self._duplicate(state, turn_id, utterance, submission):
                return self._view(state, turn_id)
            raise _fail('INTERACTIVE_OPERATION_ACTIVE') from exc
        except InteractiveBoundaryError:
            raise
        except ResourceAdmissionError:
            raise
        except (SessionError, ValueError, RuntimeError, OSError) as exc:
            raise _fail('INTERACTIVE_APPLICATION_FAILED') from exc

    def _view(self, state, turn_id):
        interaction = next((i for i in state.interactions if i.turn_id == turn_id), None)
        turn = next((t for t in state.turns if t.turn_id == turn_id), None)
        if interaction is None and turn is None:
            raise _fail('INTERACTIVE_REFERENCE_INVALID')
        response = (None if interaction is None or interaction.presentation is None
                    else PresentedResponse.from_dict(_serialize(interaction.presentation)))
        admitted = {} if interaction is None or interaction.admitted is None else interaction.admitted
        run_id = turn.run_id if turn is not None else (
            admitted['request_id'] + ':run' if admitted.get('kind') == 'execute' else None)
        status = turn.status if turn is not None else interaction.status
        steps, state_revision, error = (), None, None
        if run_id is not None:
            try:
                run = self._application.run_store.load(run_id)
                if turn is not None and turn.request_id is not None:
                    self._application.sessions._check_request(turn, run)
            except RunNotFoundError:
                run = None
                if turn is not None and turn.status in {'activated','stale','ready','run_succeeded'}:
                    error = _fail('INTERACTIVE_REFERENCE_INVALID').error
                    status = 'unavailable'
            except (ValueError, RuntimeError, OSError) as exc:
                raise _fail('INTERACTIVE_APPLICATION_FAILED') from exc
            if run is not None:
                status, state_revision = run.lifecycle_status.value.lower(), run.revision
                steps = tuple(StepView(s.step_id, s.tool_name, s.status.value, s.attempt_count) for s in run.steps)
                if status == 'succeeded' and (turn is None or turn.status in {'linked','run_succeeded','ready'}):
                    status = 'finalizing'
                elif status == 'succeeded' and turn.status == 'failed':
                    status = 'failed'
                    error = _fail('INTERACTIVE_APPLICATION_FAILED').error
                if run.errors:
                    error = ClientError(run.errors[0].code, safe_message_for(run.errors[0].code))
        if response is None and ((interaction is not None and interaction.status in {'answered', 'clarification', 'navigated', 'failed'})
                                or (turn is not None and turn.status in {'activated','stale','failed','cancelled','planned','clarification'})):
            error = _fail('INTERACTIVE_PRESENTATION_UNAVAILABLE').error if error is None else error
        if (interaction is None and turn is not None and turn.request_id is None
                and turn.status == 'activated' and turn.revision_id is not None):
            # Metadata-only navigation has no assistant wording to regenerate.
            status, error = 'navigated', None
        if response is not None and response.error is not None:
            error = response.error
        if interaction is not None and interaction.prerequisite is not None:
            status, error = 'clarification', None
        return TurnView(session_id=state.session_id, turn_id=turn_id,
            utterance='' if interaction is None else interaction.utterance,
            base_generation=turn.base_generation if interaction is None else interaction.base_generation,
            base_revision_id=turn.base_revision_id if interaction is None else interaction.base_revision_id,
            profile_id=None if interaction is None or interaction.submission is None else interaction.submission.get('profile_id'),
            run_id=run_id, revision_id=None if turn is None else turn.revision_id,
            status=status, response=response, error=error, steps=steps, state_revision=state_revision)

    def turn(self, session_id, turn_id):
        _identifier(turn_id)
        try:
            return self._view(self._load(session_id), turn_id)
        except InteractiveBoundaryError:
            raise
        except (ValueError, TypeError, KeyError) as exc:
            raise _fail('INTERACTIVE_APPLICATION_FAILED') from exc

    def status(self, session_id, turn_id):
        return self.turn(session_id, turn_id)

    def recover_turn(self, session_id, turn_id, *, complete_presentation=False):
        """Explicit reconciliation; never regenerate a historical model answer."""
        if type(complete_presentation) is not bool:
            raise _fail('INTERACTIVE_INPUT_INVALID')
        state = self._load(session_id)
        self._view(state, turn_id)
        sessions = self._application.sessions
        try:
            with sessions.processing_lease(session_id, turn_id):
                state = self._load(session_id)
                if any(t.turn_id == turn_id for t in state.turns):
                    state = sessions.recover(session_id, turn_id)
                    if complete_presentation:
                        try:
                            state = sessions.complete_presentation(session_id, turn_id)
                        except ResourceAdmissionError as exc:
                            failed = self._load(session_id)
                            interaction = next((i for i in failed.interactions if i.turn_id == turn_id), None)
                            from .turns import TurnOutcome
                            outcome = TurnOutcome('execute', 'failed', text=exc.message, error=exc.error)
                            if interaction is None or _serialize(interaction.presentation) != _present(outcome).to_dict():
                                raise  # Preserve a different immutable historical display.
                            return self.turn(session_id, turn_id)
                    interaction = next((i for i in state.interactions if i.turn_id == turn_id), None)
                    if (interaction is not None and interaction.submission is not None
                            and interaction.presentation is None and state.turn(turn_id).request_id is not None
                            and state.turn(turn_id).status in {'activated','stale','failed','cancelled','planned'}):
                        from .responses import summarize
                        from .turns import TurnOutcome
                        if interaction.admitted.get('operation') == 'plan':
                            from .dialogue_execution import completion_outcome
                            outcome = completion_outcome(state, turn_id)
                        else:
                            outcome = summarize(sessions, session_id, turn_id,
                                TurnOutcome('execute', state.turn(turn_id).status))
                        sessions.store_presentation(session_id, turn_id, _present(outcome).to_dict())
                return self.turn(session_id, turn_id)
        except SessionConflictError as exc:
            raise _fail('INTERACTIVE_OPERATION_ACTIVE') from exc
        except ResourceAdmissionError:
            raise
        except (SessionError, ValueError, RuntimeError, OSError) as exc:
            raise _fail('INTERACTIVE_APPLICATION_FAILED') from exc

    def cancel_turn(self, session_id, turn_id):
        view = self.turn(session_id, turn_id)
        if view.run_id is None:
            raise _fail('INTERACTIVE_REFERENCE_INVALID')
        try:
            return self._application.cancel(view.run_id)
        except (ValueError, RuntimeError, OSError) as exc:
            raise _fail('INTERACTIVE_APPLICATION_FAILED') from exc

    def scientific_artifact_handles(self, session_id, revision_id, *, turn_id=None):
        """Inventory reviewed accepted scientific identities without payload reads.

        This eligibility projection does not attest current payload integrity or
        privacy. Those checks are performed against a private download snapshot.
        """
        return tuple(item.reference for item in
                     self._scientific_artifacts(session_id, revision_id, turn_id=turn_id))

    def scientific_artifacts_for_turn(self, session_id, turn_id):
        """Offer reviewed files produced by this exact accepted execution.

        Navigation, questions, failed executions and retained upstream results
        have no newly created downloadable file in their conversation message.
        """
        _identifier(turn_id)
        state = self._load(session_id)
        try:
            turn = state.turn(turn_id)
        except SessionError:
            return ()
        if (turn.request_id is None or turn.status not in {'activated', 'stale'}
                or turn.revision_id is None or turn.run_id is None):
            return ()
        interaction = next((i for i in state.interactions if i.turn_id == turn_id), None)
        if (interaction is None or interaction.presentation is None or interaction.admitted is None
                or interaction.admitted.get('kind') != 'execute'
                or interaction.admitted.get('request_id') != turn.request_id):
            return ()
        response = PresentedResponse.from_dict(_serialize(interaction.presentation))
        if (response.kind != 'execute' or response.error is not None
                or response.status not in {'activated', 'stale'}):
            return ()
        revisions = [r for r in state.revisions if r.revision_id == turn.revision_id]
        if (len(revisions) != 1 or revisions[0].turn_id != turn_id
                or revisions[0].run_id != turn.run_id):
            raise _fail('INTERACTIVE_SCIENTIFIC_ARTIFACT_UNAVAILABLE')
        return self.scientific_artifact_handles(session_id, turn.revision_id, turn_id=turn_id)

    def _scientific_artifacts(self, session_id, revision_id, *, turn_id=None):
        _identifier(revision_id)
        if turn_id is not None:
            _identifier(turn_id)
        state = self._load(session_id)
        revisions = [r for r in state.revisions if r.revision_id == revision_id]
        if len(revisions) != 1:
            raise _fail('INTERACTIVE_REFERENCE_INVALID')
        revision = revisions[0]
        try:
            self._application.sessions._validate_revision(revision)
            origin = state.turn(revision.turn_id)
            if (origin.request_id is None or origin.status not in {'activated', 'stale'}
                    or origin.revision_id != revision_id or origin.run_id != revision.run_id
                    or origin.run_result_sha256 != revision.run_result_sha256
                    or origin.request_id != revision.request_id):
                raise ValueError('Revision is not an accepted scientific result.')
            if turn_id is not None and turn_id != origin.turn_id:
                raise ValueError('Turn does not identify this created result.')
            selections = {(s.name, s.step_id, s.output_key) for s in origin.selections}
            items = []
            represented = set()
            for output in revision.outputs:
                if turn_id is not None and (output.run_id != origin.run_id
                        or (output.name, output.step_id, output.output_key) not in selections):
                    continue
                run = self._application.run_store.load(output.run_id)
                step = next(s for s in run.steps if s.step_id == output.step_id)
                if step.tool_name in _MATRIX_CONTRACTS:
                    items.append(self._accepted_matrix(session_id, revision, origin, output))
                elif step.tool_name in _TABLE_TOOLS:
                    try:
                        if output.run_id == origin.run_id:
                            # Current-request membership belongs to the exact
                            # accepted plan step, even when selected display
                            # outputs contain only its summary statistics.
                            items.extend(self._accepted_tables(session_id, revision, origin, None,
                                step_id=output.step_id))
                        else:
                            items.extend(self._accepted_tables(session_id, revision, origin, output))
                    except (ValueError, RuntimeError, OSError, KeyError, TypeError, StopIteration):
                        # Eligibility of an optional table never removes a
                        # separately accepted matrix or existing result view.
                        pass
                represented.add((output.run_id, output.step_id))
            # The captured run-result digest includes its exact plan and every
            # accepted step. It establishes current-request membership without
            # turning historical input references into new result attachments.
            run = self._application.run_store.load(revision.run_id)
            if (run.lifecycle_status.value != 'SUCCEEDED' or run.plan is None
                    or run.request.request_id != revision.request_id):
                raise ValueError('Created result does not identify a successful execution.')
            for planned in run.plan.steps:
                if (run.run_id, planned.step_id) in represented:
                    continue
                if planned.tool_name in _MATRIX_CONTRACTS:
                    items.append(self._accepted_matrix(session_id, revision, origin, None,
                        step_id=planned.step_id))
                elif planned.tool_name in _TABLE_TOOLS:
                    try:
                        items.extend(self._accepted_tables(session_id, revision, origin, None,
                            step_id=planned.step_id))
                    except (ValueError, RuntimeError, OSError, KeyError, TypeError, StopIteration):
                        pass
            self._application.sessions._validate_revision(revision)
            unique = {}
            for item in items:
                handle = item.reference.handle
                if handle in unique:
                    if unique[handle] != item:
                        raise ValueError('Conflicting accepted scientific artifact identity.')
                else:
                    unique[handle] = item
            return tuple(unique.values())
        except (SessionError, ValueError, RuntimeError, OSError, KeyError, TypeError, StopIteration) as exc:
            raise _fail('INTERACTIVE_SCIENTIFIC_ARTIFACT_UNAVAILABLE') from exc

    def _accepted_matrix(self, session_id, revision, origin, output, *, step_id=None):
        from agent.orchestration.prior_outputs import accepted_step_digest
        from agent.orchestration.verification_authority import _accepted
        from agent.tools.data import scatac_matrix_contract as contract
        from .matrix_delivery import _open_matrix_source

        store = self._application.run_store
        run_id = revision.run_id if output is None else output.run_id
        step_id = step_id if output is None else output.step_id
        step, authority, _, anchor = _accepted(store, run_id, step_id)
        result, record = _serialize(step.result), authority.record
        if (output is not None and accepted_step_digest(step) != output.accepted_step_sha256
                or result['contract_version'] not in _MATRIX_CONTRACTS[step.tool_name]
                or record['schema_version'] != 2
                or record['scope'] != 'scientific_correctness.v1'
                or record['completion'] != 'succeeded'):
            raise ValueError('Unsupported accepted matrix authority.')
        spec = self._application.registry.get(step.tool_name)
        spec.result_contract.validate(result)
        output_key = 'manifest_path' if output is None else output.output_key
        ports = [p for p in spec.semantic_planning.producer_ports
                 if output_key in {member.field_name for member in p.members}]
        if len(ports) != 1 or ports[0].name not in {'matrix', 'dataset'}:
            raise ValueError('Output does not select a reviewed matrix artifact port.')
        manifest_path = Path(result['manifest_path'])
        workspace = self._application._workspace
        root = workspace.runs / workspace.run_digest(run_id) / 'scientific'
        output_dir = Path(step.resolved_arguments['output_dir'])
        for path in (root, output_dir, manifest_path):
            if not path.is_absolute() or path != path.resolve():
                raise ValueError('Unsafe accepted matrix publication path.')
        output_dir.relative_to(root)
        manifest_path.relative_to(root)
        if (manifest_path.name != 'manifest.json' or manifest_path.parent.name != 'artifact'
                or manifest_path.parent.parent.parent != output_dir):
            raise ValueError('Matrix publication differs from the managed output directory.')
        # Inventory reads only the bounded pinned manifest. No matrix open,
        # payload hash, authority-DAG reconstruction, or scientific IO occurs.
        with _open_matrix_source(manifest_path, root) as stream:
            raw = stream.read(contract.MAX_MANIFEST_BYTES + 1)
        if (len(raw) > contract.MAX_MANIFEST_BYTES
                or hashlib.sha256(raw).hexdigest() != result['manifest_sha256']):
            raise ValueError('Changed matrix manifest.')
        manifest = contract.load_manifest_bytes(raw)
        if (record['publication_path'] != str(manifest_path)
                or record['manifest_sha256'] != result['manifest_sha256']
                or record['artifact_contract'] != manifest['contract_version']
                or record['artifact_type'] != manifest['artifact_type']
                or record['science_profile'] != manifest['profile_sha256']
                or _serialize(record['resources']['manifest_identity']) != manifest):
            raise ValueError('Matrix result, manifest and authority differ.')
        if step.tool_name == 'build_scATAC_cell_by_ccre':
            from agent.tools.data.scatac_matrix import _summary
            expected = _summary(manifest, manifest_path, result['manifest_sha256'])
        elif step.tool_name == 'build_scATAC_cell_by_features':
            from agent.tools.data.fragment_feature_matrix import _summary
            expected = _summary(manifest, manifest_path, result['manifest_sha256'])
        else:
            from agent.tools.data.scatac_matrix_adoption import _summary
            from agent.tools.data.external_matrix_contract import contract_for
            expected = _summary(manifest, manifest_path, result['manifest_sha256'],
                                contract=contract_for(manifest))
        if result != expected:
            raise ValueError('Accepted result differs from the matrix owner summary.')
        payload = manifest_path.parent / manifest['matrix']['path']
        if str(payload) != result['matrix_path'] or result['matrix_sha256'] != manifest['matrix']['sha256']:
            raise ValueError('Matrix payload locator or digest differs.')
        files = {f['path']: f for f in record['files']}
        if (_serialize(files.get(str(manifest_path))) != dict(path=str(manifest_path),
                sha256=result['manifest_sha256'], size_bytes=len(raw))
                or _serialize(files.get(str(payload))) != dict(path=str(payload),
                sha256=manifest['matrix']['sha256'], size_bytes=manifest['matrix']['size_bytes'])):
            raise ValueError('Authority does not pin the matrix payload and manifest.')
        proof = record['resources']['result_metadata']
        for key in ('identity_sha256', 'logical_matrix_sha256', 'nnz', 'total_count', 'zero_row_count'):
            if proof[key] != manifest[key]:
                raise ValueError('Scientific proof metadata differs from the matrix manifest.')
        if 'diagnostic' in manifest and _serialize(proof['diagnostic']) != manifest['diagnostic']:
            raise ValueError('Scientific proof diagnostic differs from the matrix manifest.')
        if _accepted(store, run_id, step_id)[3] != anchor:
            raise ValueError('Accepted matrix anchor changed during inventory.')
        identity = dict(session_id=session_id, revision_id=revision.revision_id,
            turn_id=origin.turn_id, output=None if output is None else asdict(output), artifact_role='scientific_matrix',
            authority_sha256=authority.identity_sha256, manifest_sha256=result['manifest_sha256'],
            payload_sha256=manifest['matrix']['sha256'], size_bytes=manifest['matrix']['size_bytes'])
        if output is None:
            identity['accepted_step'] = dict(run_id=run_id, step_id=step_id,
                accepted_step_sha256=accepted_step_digest(step))
        handle = digest(identity)
        reference = ScientificArtifactHandle(handle, 'scientific_matrix', manifest['matrix']['sha256'],
            revision.revision_id, origin.turn_id, manifest['matrix']['size_bytes'])
        return _ScientificMatrixArtifact(reference, payload, root, reference.sha256,
            reference.size_bytes, manifest['contract_version'], manifest)

    def _accepted_tables(self, session_id, revision, origin, output, *, step_id=None):
        """Resolve the two closed owner roles from their pinned manifest only."""
        from agent.orchestration.prior_outputs import accepted_step_digest
        from agent.orchestration.verification_authority import _accepted
        from agent.tools.data import _barcode_qc_contract as qc, _cell_selection_contract as selection
        from agent.tools.data import scatac_barcode_qc, scatac_cell_selection
        from .matrix_delivery import _open_matrix_source

        store = self._application.run_store
        run_id = revision.run_id if output is None else output.run_id
        step_id = step_id if output is None else output.step_id
        step, authority, execution, anchor = _accepted(store, run_id, step_id)
        result, record = _serialize(step.result), authority.record
        version, port, roles = _TABLE_TOOLS[step.tool_name]
        if (output is not None and accepted_step_digest(step) != output.accepted_step_sha256
                or result['contract_version'] != version or record['schema_version'] != 2
                or record['scope'] != 'scientific_correctness.v1' or record['completion'] != 'succeeded'):
            raise ValueError('Unsupported accepted table authority.')
        spec = self._application.registry.get(step.tool_name)
        spec.result_contract.validate(result)
        output_key = 'manifest_path' if output is None else output.output_key
        ports = [p for p in spec.semantic_planning.producer_ports
                 if output_key in {member.field_name for member in p.members}]
        if len(ports) != 1 or ports[0].name != port:
            raise ValueError('Output does not identify a reviewed table artifact port.')
        manifest_path = Path(result['manifest_path'])
        workspace = self._application._workspace
        root = workspace.runs / workspace.run_digest(run_id) / 'scientific'
        output_dir = Path(step.resolved_arguments['output_dir'])
        for path in (root, output_dir, manifest_path):
            if not path.is_absolute() or path != path.resolve():
                raise ValueError('Unsafe accepted table publication path.')
        output_dir.relative_to(root)
        manifest_path.relative_to(root)
        prefix = 'barcode-qc-' if step.tool_name == 'compute_scATAC_qc' else 'cell-selection-'
        if (manifest_path.name != 'manifest.json' or manifest_path.parent.parent != output_dir
                or not manifest_path.parent.name.startswith(prefix)):
            raise ValueError('Table publication differs from the managed output directory.')
        with _open_matrix_source(manifest_path, root) as stream:
            raw = stream.read(qc.MAX_MANIFEST + 1)
        if len(raw) > qc.MAX_MANIFEST or hashlib.sha256(raw).hexdigest() != result['manifest_sha256']:
            raise ValueError('Changed table manifest.')
        is_qc = step.tool_name == 'compute_scATAC_qc'
        owner = scatac_barcode_qc if is_qc else scatac_cell_selection
        manifest = (qc.BarcodeQCManifest(raw) if is_qc else selection.CellSelectionManifest(raw)).to_dict()
        arguments = (qc.validate_arguments(step.resolved_arguments) if is_qc
                     else selection.arguments(step.resolved_arguments))
        token = (owner._publication_token(arguments, execution, manifest['qc_resource_identity_sha256'],
                    manifest['resource_qualification'], manifest['backend_identity']) if is_qc
                 else owner._publication(arguments, execution)[2])
        profile = manifest['science_profile_sha256' if is_qc else 'selection_profile_sha256']
        if (record['publication_path'] != str(manifest_path)
                or record['manifest_sha256'] != result['manifest_sha256']
                or record['artifact_contract'] != version or record['artifact_type'] != manifest['artifact_type']
                or record['science_profile'] != profile
                or _serialize(record['resources']['manifest_identity']) != manifest
                or _serialize(record['resources']['result_metadata']) != {}
                or manifest_path.parent.name != prefix + token
                or record['arguments_sha256'] != qc.digest(arguments)
                or manifest['arguments'] != arguments
                or result != owner._summary(manifest, manifest_path, result['manifest_sha256'])):
            raise ValueError('Accepted result, arguments, manifest and table authority differ.')
        files = {f['path']: _serialize(f) for f in record['files']}
        expected = [dict(path=str(manifest_path), sha256=result['manifest_sha256'], size_bytes=len(raw))]
        expected.extend(dict(path=str(manifest_path.parent / manifest[role]['path']),
            sha256=manifest[role]['sha256'], size_bytes=manifest[role]['size_bytes']) for role, *_ in roles)
        if any(files.get(f['path']) != f for f in expected):
            raise ValueError('Authority does not pin the table payloads and manifest.')
        if _accepted(store, run_id, step_id)[3] != anchor:
            raise ValueError('Accepted table anchor changed during inventory.')
        items = []
        for role, artifact_type, filename, label in roles:
            payload = manifest[role]
            identity = dict(session_id=session_id, revision_id=revision.revision_id, turn_id=origin.turn_id,
                accepted_step=dict(run_id=run_id, step_id=step_id, accepted_step_sha256=accepted_step_digest(step)),
                artifact_role=role, authority_sha256=authority.identity_sha256,
                manifest_sha256=result['manifest_sha256'], payload_sha256=payload['sha256'],
                size_bytes=payload['size_bytes'])
            reference = ScientificArtifactHandle(digest(identity), artifact_type, payload['sha256'],
                revision.revision_id, origin.turn_id, payload['size_bytes'], filename, label)
            items.append(_ScientificMatrixArtifact(reference, manifest_path.parent / payload['path'], root,
                reference.sha256, reference.size_bytes, version, manifest, role))
        return tuple(items)

    def _scientific_artifact(self, session_id, revision_id, handle):
        _identifier(handle)
        matches = [item for item in self._scientific_artifacts(session_id, revision_id)
                   if item.reference.handle == handle]
        if len(matches) != 1:
            raise _fail('INTERACTIVE_SCIENTIFIC_ARTIFACT_UNAVAILABLE')
        return matches[0]

    def prepare_scientific_artifact(self, session_id, revision_id, handle):
        """Prepare bounded verified original bytes without provider or science."""
        return self._matrix_downloads.prepare(self._scientific_artifact(session_id, revision_id, handle))

    def artifact_handles(self, session_id, revision_id):
        """References only to the exact accepted application completion files."""
        return tuple(handle for handle, _ in self._artifacts(session_id, revision_id))

    def _artifacts(self, session_id, revision_id):
        state = self._load(session_id)
        revision = next((r for r in state.revisions if r.revision_id == revision_id), None)
        if revision is None:
            raise _fail('INTERACTIVE_REFERENCE_INVALID')
        try:
            self._application.sessions._validate_revision(revision)
            turn = state.turn(revision.turn_id)
            workspace = self._application._workspace
            root = workspace.runs / workspace.run_digest(turn.run_id)
            items = []
            for file in turn.completion_files:
                path = Path(file.path)
                # Evidence and manifests remain application-domain projections;
                # only rendered presentations are offered as downloadable bytes.
                if path.suffix not in {'.md', '.png'}:
                    continue
                relative = path.relative_to(root)
                if relative.parts[0] not in {'evidence','visualizations','report'}:
                    raise ValueError('Unowned completion file.')
                kind = ('analysis_figure' if path.suffix == '.png' else
                        'analysis_report' if path.suffix == '.md' else relative.parts[0])
                handle = digest(dict(session_id=session_id, revision_id=revision_id,
                    turn_id=turn.turn_id, relative=str(relative), sha256=file.sha256))
                items.append((ArtifactHandle(handle, kind, file.sha256, revision_id, turn.turn_id), file))
            return tuple(items)
        except (SessionError, ValueError, RuntimeError, OSError) as exc:
            raise _fail('INTERACTIVE_ARTIFACT_UNAVAILABLE') from exc

    def resolve_artifact(self, session_id, revision_id, handle):
        """Validate an exact handle and pinned bytes; return no path or contents.

        Raw reports can contain internal paths. Supported byte delivery uses
        ``artifact_content`` with its separate safe presentation contract.
        """
        return self._read_artifact(session_id, revision_id, handle)[0]

    def _read_artifact(self, session_id, revision_id, handle):
        _identifier(handle)
        matches = [(h, f) for h, f in self._artifacts(session_id, revision_id) if h.handle == handle]
        if len(matches) != 1:
            raise _fail('INTERACTIVE_ARTIFACT_UNAVAILABLE')
        try:
            reference, file = matches[0]
            path = Path(file.path)
            self._application._workspace.require_regular_file(path)
            if path.resolve() != path or path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError('Unsafe presentation file.')
            with path.open('rb') as stream:
                data = stream.read(16 * 1024 * 1024 + 1)
            if len(data) > 16 * 1024 * 1024:
                raise ValueError('Presentation size limit.')
            if hashlib.sha256(data).hexdigest() != file.sha256:
                raise ValueError('Changed presentation bytes.')
            return reference, data
        except (ValueError, RuntimeError, OSError) as exc:
            raise _fail('INTERACTIVE_ARTIFACT_UNAVAILABLE') from exc

    def artifact_content(self, session_id, revision_id, handle):
        """Deliver pinned PNG or a labeled safe report projection, never paths.

        Reports retain their original source digest while delivery has its own
        digest. The report projection reads only the existing accepted evidence
        owner and adds no scientific claims, report regeneration or verification.
        """
        reference, original = self._read_artifact(session_id, revision_id, handle)
        try:
            if reference.artifact_type == 'analysis_figure':
                from PIL import Image
                if (len(original) < 33 or not original.startswith(b'\x89PNG\r\n\x1a\n')
                        or original[12:16] != b'IHDR'):
                    raise ValueError('Unsupported figure content.')
                width, height = (int.from_bytes(original[start:start + 4], 'big') for start in (16, 20))
                if not 0 < width * height <= 40_000_000:
                    raise ValueError('Unsupported figure bounds.')
                with Image.open(io.BytesIO(original)) as image:
                    if image.format != 'PNG' or not 0 < image.width * image.height <= 40_000_000:
                        raise ValueError('Unsupported figure bounds.')
                    if set(image.info) - {'Software', 'dpi', 'srgb', 'gamma', 'transparency'}:
                        raise ValueError('Unsupported figure metadata.')
                    if 'Software' in image.info and image.info['Software'] != 'Agent plotting spec v1':
                        raise ValueError('Unsupported figure metadata.')
                    image.verify()
                return ArtifactContent(reference, 'image/png', f'agent-figure-{handle[:16]}.png',
                    original, reference.sha256, 'accepted_png',
                    ('This is the accepted persisted figure, not fresh scientific reconstruction.',))
            if reference.artifact_type != 'analysis_report':
                raise ValueError('Unsupported artifact delivery.')
            revision = self.revision(session_id, revision_id)
            lines = ['Client-safe report projection', '', f'Revision: {revision_id}',
                f'Run: {revision.run_id}', f'Turn: {revision.turn_id}',
                f'Original accepted report SHA-256: {reference.sha256}', '',
                'This text projects accepted revision output evidence. It is not the original Markdown report.',
                'No scientific reconstruction or fresh scientific verification was performed.', '']
            size = sum(len(line.encode('utf-8')) + 1 for line in lines)
            for output in revision.outputs:
                start = len(lines)
                view = self.evidence(session_id, revision_id, output)
                lines.extend([f'Output: {output}', f'Evidence availability: {view.status}'])
                if view.reason is not None:
                    lines.append(f'Availability reason: {view.reason}')
                if view.source is not None:
                    lines.extend([f'Tool: {view.source.tool_name}', f'Result contract: {view.source.result_contract}',
                        f'Evidence SHA-256: {view.source.evidence_sha256}',
                        'Accepted verification checks: ' + ', '.join(view.source.verification_checks)])
                for fact in view.facts:
                    rendered = (json.dumps(_serialize(fact.value), ensure_ascii=False, allow_nan=False)
                                if fact.status == 'available' else fact.reason or fact.status)
                    lines.append(f'{fact.field} [{fact.status}]: {rendered}')
                lines.extend(['Limitations:'] + list(view.limitations) + [''])
                size += sum(len(line.encode('utf-8')) + 1 for line in lines[start:])
                if size > 1_048_576:
                    raise ValueError('Report projection exceeds the byte bound.')
            data = ('\n'.join(lines) + '\n').encode('utf-8')
            if len(data) > 1_048_576:
                raise ValueError('Report projection exceeds the byte bound.')
            return ArtifactContent(reference, 'text/plain; charset=utf-8', f'agent-report-{handle[:16]}.txt',
                data, hashlib.sha256(data).hexdigest(), 'client_safe_report_projection',
                ('The original accepted report is digest-checked but its raw bytes are not delivered.',
                 'The delivered text is a bounded presentation of existing accepted evidence, not scientific authority.'))
        except (ValueError, TypeError, RuntimeError, OSError, ImportError, SyntaxError) as exc:
            raise _fail('INTERACTIVE_ARTIFACT_UNAVAILABLE') from exc


__all__ = ['InteractiveAgentApplication', 'InteractiveBoundaryError']
