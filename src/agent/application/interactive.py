"""Transport-independent client of the existing durable Agent application.

Submission/presentation records belong to the Session. This facade owns neither
scientific state nor a job queue; callers may run its synchronous submit method
in a server worker and poll the existing durable checkpoints.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from types import MappingProxyType

from agent.orchestration import PlanningModelProfile, PlanningRecoveryPolicy, PlanningWireMode
from agent.orchestration.error_policy import safe_message_for
from agent.orchestration.run_store import RunNotFoundError
from agent.providers import PlanningModelFactoryRegistry, build_default_planning_model_factory_registry
from agent.schemas.orchestration import _serialize, freeze_json_mapping

from .interactive_schemas import (
    ArtifactHandle, ClientError, ModelChoice, PresentedResponse, RevisionView,
    SessionView, StepView, TurnView,
)
from .service import RESERVED_APPLICATION_INPUTS, ResearchAgentApplication
from .session_state import SessionConflictError, SessionError, canonical, digest


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
}


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


def _present(outcome, state=None):
    """Store the displayed response, not a second copy of scientific evidence."""
    if outcome.presentation is not None:
        return outcome.presentation
    text = outcome.text
    clarification = None if outcome.clarification is None else asdict(outcome.clarification)
    scientific = None
    if outcome.scientific is not None:
        value = outcome.scientific
        scientific = dict(support=value.support, limitations=value.limitations,
                          evidence_scope=value.evidence_scope)
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
                 recovery_planning_profile=None, planning_recovery_policy=None):
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

    def _submission(self, profile, generation, inputs, predecessor):
        return dict(profile_id=profile.profile_id,
            configuration_sha256=digest(dict(profile=asdict(profile),
                wire_mode=None if self._planning['planning_wire_mode'] is None else self._planning['planning_wire_mode'].value,
                recovery_profile=None if self._planning['recovery_planning_profile'] is None else asdict(self._planning['recovery_planning_profile']),
                recovery_policy=None if self._planning['planning_recovery_policy'] is None else asdict(self._planning['planning_recovery_policy']))),
            expected_generation=generation, execution_inputs=inputs, predecessor_turn_id=predecessor)

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
                            execution_inputs=None, predecessor_turn_id=None, profile_id=None):
        _identifier(turn_id)
        if (type(expected_generation) is not int or expected_generation < 0
                or type(utterance) is not str or not utterance.strip() or len(utterance) > 4096):
            raise _fail('INTERACTIVE_INPUT_INVALID')
        if predecessor_turn_id is not None:
            _identifier(predecessor_turn_id)
        profile = self._profile(self._default if profile_id is None else profile_id)
        inputs = _inputs(execution_inputs)
        submission = self._submission(profile, expected_generation, inputs, predecessor_turn_id)
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
        return profile, inputs, submission, state, duplicate

    def validate_submission(self, session_id, turn_id, utterance, *, expected_generation,
                            execution_inputs=None, predecessor_turn_id=None, profile_id=None):
        """Check a submission without providers, persistence, or execution.

        The normalized fingerprint permits a local transport to coalesce in-flight
        requests. It reserves no identity and grants no execution authority;
        ``submit_turn`` repeats admission under the existing processing lease.
        """
        _, _, submission, _, _ = self._checked_submission(session_id, turn_id, utterance,
            expected_generation=expected_generation, execution_inputs=execution_inputs,
            predecessor_turn_id=predecessor_turn_id, profile_id=profile_id)
        return digest(dict(session_id=session_id, turn_id=turn_id,
                           utterance=utterance, submission=submission))

    def submit_turn(self, session_id, turn_id, utterance, *, expected_generation,
                    execution_inputs=None, predecessor_turn_id=None, profile_id=None):
        profile, inputs, submission, state, duplicate = self._checked_submission(
            session_id, turn_id, utterance, expected_generation=expected_generation,
            execution_inputs=execution_inputs, predecessor_turn_id=predecessor_turn_id,
            profile_id=profile_id)
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
                        submission=submission)
                    presentation = _present(outcome, self._load(session_id))
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
        if response is not None and response.error is not None:
            error = response.error
        return TurnView(session_id=state.session_id, turn_id=turn_id,
            utterance='' if interaction is None else interaction.utterance,
            base_generation=turn.base_generation if interaction is None else interaction.base_generation,
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
                        state = sessions.complete_presentation(session_id, turn_id)
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
            root = self._application._workspace.run_paths(turn.run_id)
            items = []
            for file in turn.completion_files:
                path = Path(file.path)
                # Evidence and manifests remain application-domain projections;
                # only rendered presentations are offered as downloadable bytes.
                if path.suffix not in {'.md', '.png'}:
                    continue
                relative = path.relative_to(root.root)
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

        Raw reports can contain internal paths. Their delivery belongs to a later
        transport boundary, rather than this client-safe identity contract.
        """
        matches = [(h, f) for h, f in self._artifacts(session_id, revision_id) if h.handle == handle]
        if len(matches) != 1:
            raise _fail('INTERACTIVE_ARTIFACT_UNAVAILABLE')
        try:
            reference, file = matches[0]
            path = Path(file.path)
            self._application._workspace.require_regular_file(path)
            if path.resolve() != path or path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError('Unsafe presentation file.')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != file.sha256:
                raise ValueError('Changed presentation bytes.')
            return reference
        except (ValueError, RuntimeError, OSError) as exc:
            raise _fail('INTERACTIVE_ARTIFACT_UNAVAILABLE') from exc


__all__ = ['InteractiveAgentApplication', 'InteractiveBoundaryError']
