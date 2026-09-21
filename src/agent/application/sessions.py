"""Explicit session API above the unchanged one-request application API.

Recovery/navigation never invoke the runtime, scientific verification, or presentation.
Completing unfinished presentation is a separate explicit operation.
"""
from dataclasses import replace
import hashlib
from pathlib import Path

from agent.orchestration.run_store import RunNotFoundError
from agent.schemas import AgentRequest, RunMode, RunStatus, StepStatus
from .schemas import ApplicationStatus
from .session_state import (AnalysisRevision, CompletionFile, Navigation, OutputLocator,
                            OutputSelection, SessionConflictError, SessionError, SessionTurn,
                            digest, natural, text)
from .session_store import FileSessionStore


def _replace_turn(state, turn):
    return replace(state, turns=tuple(turn if t.turn_id == turn.turn_id else t for t in state.turns))


class AnalysisSessions:
    """Application-owned navigation with an opt-in bounded planning projection.

    New selections and explicitly retained base outputs form a complete snapshot.
    Scientific authority and compatibility remain owned by orchestration/tools.
    Returned records describe historical acceptance, not current scientific validity.
    """

    def __init__(self, application, root):
        self._application = application
        self._store = FileSessionStore(root)

    def create(self, session_id):
        return self._store.create(session_id)

    def load(self, session_id):
        return self._store.load(session_id)

    def record_no_run_turn(self, session_id, turn_id, *, outcome, expected_generation):
        """Record an explicit non-execution outcome; no linguistic interpretation."""
        text(turn_id)
        natural(expected_generation)
        if outcome not in {'clarification', 'failed', 'cancelled'}:
            raise SessionError('Invalid non-execution outcome.')
        def change(state):
            previous = next((t for t in state.turns if t.turn_id == turn_id), None)
            if previous is not None:
                if (previous.request_id is not None or previous.status != outcome
                        or previous.base_generation != expected_generation):
                    raise SessionConflictError('Non-execution turn identity conflict.')
                return state
            if state.generation != expected_generation:
                raise SessionConflictError('Session generation changed.')
            turn = SessionTurn(turn_id, state.active_revision_id, state.generation,
                               None, None, None, outcome)
            return replace(state, turns=state.turns + (turn,))
        return self._store._update(session_id, change)

    def start_turn(self, session_id, turn_id, request, outputs, *, expected_generation, retain=()):
        """Capture a base and explicit output selections before starting any work."""
        if not isinstance(request, AgentRequest):
            raise SessionError('Session turn requires an AgentRequest.')
        text(turn_id)
        natural(expected_generation)
        selections = tuple(outputs)
        retain = tuple(retain)
        if len(set(retain)) != len(retain) or any(type(n) is not str for n in retain):
            raise SessionError('Invalid explicit retained output names.')
        if not selections or any(not isinstance(s, OutputSelection) for s in selections):
            raise SessionError('Explicit output selections are required.')
        # Apply precisely the existing managed request transformation. No planner runs.
        effective, _ = self._application._prepare_request(request)
        anchor = digest(effective.to_dict())
        def change(state):
            previous = next((t for t in state.turns if t.turn_id == turn_id), None)
            if previous is not None:
                if (previous.request_id != request.request_id or previous.request_sha256 != anchor
                        or previous.selections != selections or previous.base_generation != expected_generation
                        or tuple(o.name for o in previous.retained_outputs) != retain):
                    raise SessionConflictError('Turn ID already belongs to a different request/base.')
                return state
            if state.generation != expected_generation:
                raise SessionConflictError('Session generation changed.')
            if any(t.request_id == request.request_id for t in state.turns):
                raise SessionConflictError('Request already belongs to another turn in this session.')
            base = next((r for r in state.revisions if r.revision_id == state.active_revision_id), None)
            available = {} if base is None else {o.name: o for o in base.outputs}
            if any(name not in available for name in retain):
                raise SessionError('Retained output is not in the captured active revision.')
            turn = SessionTurn(turn_id, state.active_revision_id, state.generation,
                               request.request_id, anchor, None, 'started', selections,
                               retained_outputs=tuple(available[name] for name in retain))
            return replace(state, turns=state.turns + (turn,))
        return self._store._update(session_id, change)

    def link_run(self, session_id, turn_id):
        """Reserve the exact request-derived run ID, including before run creation."""
        def change(state):
            turn = state.turn(turn_id)
            if turn.status != 'started':
                return state
            return _replace_turn(state, replace(turn, run_id=turn.request_id + ':run', status='linked'))
        return self._store._update(session_id, change)

    def run(self, session_id, turn_id, request, outputs, *, expected_generation,
            use_active_context=False, retain=()):
        """Explicit session-aware new run; duplicate calls never restart linked work.

        Returns session metadata. The ordinary run/result remains accessible through
        the associated run ID. For an interrupted linked run, use ordinary resume
        explicitly, then complete_presentation; recover itself never executes work.
        """
        self.start_turn(session_id, turn_id, request, outputs, expected_generation=expected_generation, retain=retain)
        # Claim submission under the metadata lock. A crash after this claim may
        # leave a linked but absent run; retry never silently issues another run.
        claimed = []
        def claim(state):
            turn = state.turn(turn_id)
            if turn.status != 'started':
                return state
            claimed.append(True)
            return _replace_turn(state, replace(turn, run_id=turn.request_id + ':run', status='linked'))
        state = self._store._update(session_id, claim)
        if not claimed:
            return self.recover(session_id, turn_id)
        turn = state.turn(turn_id)
        try:
            existing = self._application.run_store.load(turn.run_id)
        except RunNotFoundError:
            existing = None
        if existing is not None:
            self._check_request(turn, existing)
            return self.recover(session_id, turn_id)
        from contextlib import nullcontext
        from agent.orchestration.active_context import planning_context
        try:
            frozen = self.active_context(session_id, turn_id) if use_active_context else None
            with planning_context(frozen) if frozen is not None else nullcontext():
                result = self._application.run(request)
        except Exception:
            # Configuration/submission can fail before a scientific run exists.
            # Preserve the outcome without claiming anything about a live run.
            def failed_submission(state):
                current = state.turn(turn_id)
                try:
                    self._run(current)
                except RunNotFoundError:
                    if current.status == 'linked':
                        return _replace_turn(state, replace(current, status='failed'))
                return state
            self._store._update(session_id, failed_submission)
            raise
        self._record_result(session_id, turn_id, result)
        return self.recover(session_id, turn_id)

    def _check_request(self, turn, run):
        if run.run_id != turn.run_id or digest(run.request.to_dict()) != turn.request_sha256:
            raise SessionConflictError('Associated scientific request changed.')

    def _run(self, turn):
        run = self._application.run_store.load(turn.run_id)
        self._check_request(turn, run)
        return run

    def _completion_files(self, result):
        """Pin only completed presentation bytes, never scientific artifacts."""
        run = self._application._workspace.run_paths(result.run_id)
        files = []
        for root in (run.evidence, run.visualizations, run.report):
            for path in sorted(root.rglob('*')):
                if path.is_symlink():
                    raise SessionError('Unsafe presentation path.')
                if path.is_dir():
                    continue
                self._application._workspace.require_regular_file(path)
                files.append(CompletionFile(str(path), hashlib.sha256(path.read_bytes()).hexdigest()))
        if not result.evidence or not result.report or not files:
            raise SessionError('Missing completed application presentation.')
        for reference in (result.evidence, result.visualization, result.report):
            if reference is not None and CompletionFile(reference.path, reference.sha256) not in files:
                raise SessionError('Application artifact differs from completion result.')
        return tuple(files)

    def _record_result(self, session_id, turn_id, result):
        turn = self.load(session_id).turn(turn_id)
        run = self._run(turn)
        if run.to_run_result() != result.run_result:
            raise SessionError('Application result differs from persisted scientific run.')
        if result.status is ApplicationStatus.SUCCEEDED:
            files = self._completion_files(result)
            anchor = digest(result.run_result.to_dict())
            locators = self._locators(turn, run)
            if turn.retained_outputs:
                from agent.orchestration.prior_outputs import validate_active_outputs
                validate_active_outputs(locators, self._application.run_store, self._application.registry)
            status = 'ready'
        else:
            files, anchor = (), None
            status = ('run_succeeded' if result.run_status is RunStatus.SUCCEEDED else
                      'cancelled' if result.status is ApplicationStatus.CANCELLED else
                      'planned' if result.status is ApplicationStatus.PLANNED else 'failed')
        def change(state):
            current = state.turn(turn_id)
            if current.status in {'activated', 'stale', 'ready', 'failed', 'cancelled', 'planned'}:
                return state
            return _replace_turn(state, replace(current, status=status,
                                completion_files=files, run_result_sha256=anchor))
        return self._store._update(session_id, change)

    def complete_presentation(self, session_id, turn_id):
        """Explicitly compose a terminal successful run, then record/activate it.

        This is application presentation, NOT storage-only session recovery. Existing
        presentation APIs may independently verify science under their own contracts.
        No nonterminal runtime is resumed here and no production is rerun.
        """
        state = self.recover(session_id, turn_id)
        turn = state.turn(turn_id)
        if turn.status != 'run_succeeded':
            return state
        run = self._run(turn)
        result = self._application._complete(run.to_run_result(),
                                             self._application._workspace.run_paths(turn.run_id))
        self._record_result(session_id, turn_id, result)
        return self.recover(session_id, turn_id)

    def _locators(self, turn, run):
        result = run.to_run_result()
        if result.status is not RunStatus.SUCCEEDED or run.request.mode is not RunMode.EXECUTE:
            raise SessionError('Revision requires a successful executed run.')
        steps = {s.step_id: s for s in run.steps}
        locators = []
        for selection in turn.selections:
            step = steps.get(selection.step_id)
            if (step is None or step.status is not StepStatus.SUCCEEDED or step.result is None
                    or step.verification is None or not step.verification.passed
                    or selection.output_key not in step.result):
                raise SessionError('Output selection does not locate an accepted step result.')
            locators.append(OutputLocator(selection.name, run.run_id, step.step_id,
                                           selection.output_key, digest(step.to_dict())))
        for output in turn.retained_outputs:
            self._validate_locator(output)
        return tuple(locators) + turn.retained_outputs

    def recover(self, session_id, turn_id):
        """Storage-only reconciliation and idempotent conditional activation.

        Without a persisted application completion, scientific success is exposed as
        run_succeeded, requiring explicit complete_presentation. Never infer success
        from artifact existence, replay science, or call application.resume here.
        """
        def change(state):
            turn = state.turn(turn_id)
            if turn.status in {'started', 'activated', 'stale', 'failed', 'cancelled', 'planned', 'clarification'}:
                return state
            try:
                run = self._run(turn)
            except RunNotFoundError:
                if turn.status != 'linked':
                    raise SessionError('Previously completed run is unavailable.') from None
                return state
            result = run.to_run_result() if run.lifecycle_status.value in {
                'SUCCEEDED', 'FAILED', 'CANCELLED', 'INTERRUPTED', 'PLANNED'} else None
            if turn.status != 'ready':
                if result is None:
                    return state
                status = ('run_succeeded' if result.status is RunStatus.SUCCEEDED else
                          'cancelled' if result.status is RunStatus.CANCELLED else
                          'planned' if result.status is RunStatus.PLANNED else 'failed')
                return _replace_turn(state, replace(turn, status=status))
            if result is None or digest(result.to_dict()) != turn.run_result_sha256:
                raise SessionError('Completed run anchor changed.')
            for item in turn.completion_files:
                path = Path(item.path)
                # Completion records are metadata, not permission to read arbitrary paths.
                root = self._application._workspace.run_paths(turn.run_id)
                if not any(path.is_relative_to(p) for p in (root.evidence, root.visualizations, root.report)):
                    raise SessionError('Completion path escapes presentation directories.')
                self._application._workspace.require_regular_file(path)
                if path.resolve() != path or hashlib.sha256(path.read_bytes()).hexdigest() != item.sha256:
                    raise SessionError('Completed presentation changed.')
            outputs = self._locators(turn, run)
            revision_id = digest({'session_id': session_id, 'turn_id': turn_id})
            revision = AnalysisRevision(revision_id, turn.base_revision_id, turn.turn_id,
                                        turn.request_id, turn.run_id, turn.run_result_sha256, outputs)
            valid = state.generation == turn.base_generation and state.active_revision_id == turn.base_revision_id
            updated = replace(turn, revision_id=revision_id, status='activated' if valid else 'stale')
            turns = tuple(updated if t.turn_id == turn_id else t for t in state.turns)
            changes = dict(revisions=state.revisions + (revision,), turns=turns)
            if valid:
                changes.update(active_revision_id=revision_id, generation=state.generation + 1,
                    navigation=state.navigation + (Navigation(turn_id, state.active_revision_id,
                                                               revision_id, state.generation + 1),))
            return replace(state, **changes)
        return self._store._update(session_id, change)

    def switch(self, session_id, turn_id, revision_id, *, expected_generation):
        """Navigate existing history using metadata only; record previous-active state."""
        text(turn_id)
        text(revision_id)
        natural(expected_generation)
        def change(state):
            previous = next((t for t in state.turns if t.turn_id == turn_id), None)
            if previous is not None:
                if (previous.request_id is not None or previous.revision_id != revision_id
                        or previous.base_generation != expected_generation):
                    raise SessionConflictError('Navigation turn identity conflict.')
                return state
            if state.generation != expected_generation:
                raise SessionConflictError('Session generation changed.')
            revision = next((r for r in state.revisions if r.revision_id == revision_id), None)
            if revision is None:
                raise SessionError('Unknown revision.')
            self._validate_revision(revision)
            turn = SessionTurn(turn_id, state.active_revision_id, state.generation,
                               None, None, None, 'activated', revision_id=revision_id)
            return replace(state, active_revision_id=revision_id, generation=state.generation + 1,
                           turns=state.turns + (turn,), navigation=state.navigation + (
                               Navigation(turn_id, state.active_revision_id, revision_id, state.generation + 1),))
        return self._store._update(session_id, change)

    def _validate_revision(self, revision):
        run = self._application.run_store.load(revision.run_id)
        if digest(run.to_run_result().to_dict()) != revision.run_result_sha256:
            raise SessionError('Revision historical run anchor changed.')
        for output in revision.outputs:
            self._validate_locator(output)

    def _validate_locator(self, output):
        run = self._application.run_store.load(output.run_id)
        if run.lifecycle_status.value != 'SUCCEEDED':
            raise SessionError('Historical locator run is not successful.')
        step = next((s for s in run.steps if s.step_id == output.step_id), None)
        if (step is None or digest(step.to_dict()) != output.accepted_step_sha256
                or step.result is None or output.output_key not in step.result):
            raise SessionError('Revision output anchor changed.')

    def active_context(self, session_id, turn_id):
        """Freeze the captured revision, never the mutable active pointer."""
        from agent.orchestration.active_context import ActivePlanningContext, ActiveContextItem
        from agent.orchestration.prior_outputs import binding_for_locator, validate_binding
        state = self.load(session_id)
        turn = state.turn(turn_id)
        revision = next((r for r in state.revisions if r.revision_id == turn.base_revision_id), None)
        if revision is None:
            raise SessionError('No captured analysis revision for active context.')
        items = []
        bindings = set()
        for output in revision.outputs:
            self._validate_locator(output)
            binding = binding_for_locator(output, self._application.run_store, self._application.registry)
            validate_binding(binding, store=self._application.run_store,
                             registry=self._application.registry, integrity=True)
            if binding not in bindings:
                items.append(ActiveContextItem(f'ctx.{len(items)}', binding))
                bindings.add(binding)
        return ActivePlanningContext(session_id, revision.revision_id, turn.base_generation, tuple(items))
