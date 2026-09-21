"""Reviewed, metadata-only conversational projection. Never invokes scientific owners.

These records describe persisted acceptance, not a fresh scientific verification.
No prose, arbitrary tables, or artifact payloads enter the projection.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

from agent.schemas import PriorOutputRef
from agent.schemas.orchestration import _serialize
from agent.orchestration.prior_outputs import binding_for_locator, validate_binding
from .session_state import SessionError
from .turn_context import MATRIX, SELECTION, parameter_specs, stored_step

ROLES = {
    'prepare_scATAC_fragments': 'fragments',
    'import_scATAC_fragments': 'fragments',
    'prepare_scATAC_bam_fragments': 'fragments',
    'compute_scATAC_qc': 'QC', SELECTION: 'cell selection', MATRIX: 'matrix',
}
LABELS = {
    'min_tss_enrichment': 'Minimum TSS enrichment',
    'min_qc_fragment_records': 'Minimum QC fragment depth',
    'min_tss_flank_evidence': 'Minimum TSS flank evidence',
    'max_qc_fragment_records': 'Maximum QC fragment depth',
    'max_nucleosome_signal': 'Maximum nucleosome signal',
}


@dataclass(frozen=True)
class OutputFact:
    role: str
    run_id: str
    step_id: str
    accepted_step_sha256: str
    manifest_sha256: str
    authority_sha256: str
    verification_mode: str


@dataclass(frozen=True)
class WorkFact:
    role: str
    step_id: str
    tool_attempts: int
    successful_verification_events: int
    recovery_events: int


@dataclass(frozen=True)
class ReuseFact:
    output: OutputFact
    consumer_step_id: str
    production_step_absent: bool
    accepted_authority_bound: bool
    historical_integrity_check: str = 'required_by_accepted_execution_contract'
    owner_reconstruction_calls: int | None = None


@dataclass(frozen=True)
class LineageFact:
    consumer_role: str
    source_role: str
    source_run_id: str
    source_step_id: str


@dataclass(frozen=True)
class TurnResponseFacts:
    intent: str
    revision_id: str
    version: int
    is_active: bool
    relation: str
    outputs: tuple[OutputFact, ...]
    parameters: tuple[tuple[str, object], ...]
    comparison_revision_id: str | None = None
    comparison_version: int | None = None
    parameter_changes: tuple[tuple[str, object, object], ...] = ()
    output_changes: tuple[tuple[str, str], ...] = ()
    work: tuple[WorkFact, ...] = ()
    reuse: tuple[ReuseFact, ...] = ()
    lineage: tuple[LineageFact, ...] = ()
    reference_sha256: str | None = None
    retained_outputs: tuple[OutputFact, ...] = ()
    evidence_files_validated: int = 0
    # No scientific counts are inferred from plan absence or authority presence.
    production_calls: int | None = None
    owner_reconstruction_calls: int | None = None
    owner_verification_calls: int | None = None
    presentation_reused: bool | None = None
    limitations: tuple[str, ...] = (
        'Historical production and owner-call totals were not recorded.',
        'Recorded acceptance is not a fresh scientific or artifact-integrity verification.',
    )


def _output(binding):
    return OutputFact(ROLES[binding.tool_name], binding.run_id, binding.step_id,
        binding.accepted_step_sha256, binding.manifest_sha256, binding.authority_sha256,
        binding.verification_mode)


def _revision(sessions, state, rid):
    revision = next((r for r in state.revisions if r.revision_id == rid), None)
    if revision is None: raise SessionError('Missing requested revision.')
    sessions._validate_revision(revision)
    outputs, parameters, steps = [], {}, {}
    for locator in revision.outputs:
        step = stored_step(sessions, locator)
        role = ROLES.get(step.tool_name)
        if role is None:
            raise SessionError('Output has no reviewed conversational projection.')
        binding = binding_for_locator(locator, sessions._application.run_store, sessions._application.registry)
        output = _output(binding)
        if output in outputs: continue
        if role in steps: raise SessionError('Ambiguous active scientific role.')
        outputs.append(output)
        steps[role] = step
        if step.tool_name == SELECTION:
            specs = parameter_specs(sessions._application.registry)
            required = set(specs).intersection(sessions._application.registry.get(SELECTION).required_arguments)
            if not required <= set(step.resolved_arguments): raise SessionError('Missing accepted parameters.')
            for key, spec in specs.items():
                if key in step.resolved_arguments:
                    value = _serialize(step.resolved_arguments[key])
                    spec.validate(key, value)
                    parameters[key] = value
    return revision, tuple(outputs), parameters, steps


def _work(sessions, revision):
    app = sessions._application
    run = app.run_store.load(revision.run_id)
    work, reuse = [], []
    for step in run.steps:
        if step.tool_name not in ROLES: continue
        events = [e for e in run.trace if e.step_id == step.step_id]
        attempts = sum(e.event_type.value == 'STEP_EXECUTION' and
                       e.message == 'Registered tool attempt started.' for e in events)
        verified = sum(e.event_type.value == 'VERIFICATION' and e.details.get('passed') is True for e in events)
        recovered = sum(e.event_type.value == 'RECOVERY' for e in events)
        work.append(WorkFact(ROLES[step.tool_name], step.step_id, attempts, verified, recovered))
    if run.plan is None: raise SessionError('Missing accepted plan.')
    for step in run.plan.steps:
        for key, ref in step.arguments.items():
            if not isinstance(ref, PriorOutputRef): continue
            source = validate_binding(ref.binding, store=app.run_store, registry=app.registry)
            if source.tool_name not in ROLES: raise SessionError('Unsupported historical role.')
            result = next(s for s in run.steps if s.step_id == step.step_id)
            if result.resolved_arguments.get(key) != source.result.get(ref.output_key):
                raise SessionError('Prior output differs from accepted resolved argument.')
            # The exact reference must have been resolved, not merely proposed.
            if not any(e.step_id == step.step_id and e.details.get('argument_name') == key
                       and _serialize(e.details.get('prior_output')) == ref.to_dict() for e in run.trace):
                raise SessionError('Missing historical-resolution trace.')
            fact = ReuseFact(_output(ref.binding), step.step_id,
                            not any(s.tool_name == source.tool_name for s in run.plan.steps), True)
            if fact not in reuse: reuse.append(fact)
    return tuple(work), tuple(reuse)


def _lineage(outputs, steps):
    """Three reviewed input pairs only; never search history or infer step order."""
    edges = []
    for consumer, source_role, prefix in (
        ('matrix', 'cell selection', 'selected_cells_manifest_'),
        ('matrix', 'fragments', 'fragments_manifest_'),
        ('cell selection', 'QC', 'barcode_qc_manifest_'),
        ('QC', 'fragments', 'fragments_manifest_'),
    ):
        if consumer not in steps: continue
        source = steps.get(source_role)
        args = steps[consumer].resolved_arguments
        if source is None or any(args.get(prefix + suffix) != source.result.get('manifest_' + suffix)
                                 for suffix in ('path', 'sha256')):
            raise SessionError('Active lineage cannot be established from exact inputs.')
        binding = next(o for o in outputs if o.role == source_role)
        edges.append(LineageFact(consumer, source_role, binding.run_id, binding.step_id))
    return tuple(edges)


def _evidence_integrity(sessions, state, revision):
    """Check pinned presentation evidence only; never rebuild or interpret it."""
    turn = state.turn(revision.turn_id)
    root = sessions._application._workspace.run_paths(revision.run_id).evidence
    files = [f for f in turn.completion_files if Path(f.path).is_relative_to(root)]
    if not files: raise SessionError('Missing accepted evidence identity.')
    for item in files:
        path = Path(item.path)
        sessions._application._workspace.require_regular_file(path)
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''): sha.update(block)
        if sha.hexdigest() != item.sha256: raise SessionError('Evidence digest changed.')
    return len(files)


def project(sessions, state, request, *, revision_id=None, comparison_id=None):
    """Compose bounded facts from exact immutable revisions and accepted run metadata."""
    rid = state.active_revision_id if revision_id is None else revision_id
    revision, outputs, parameters, steps = _revision(sessions, state, rid)
    work, reuse = _work(sessions, revision)
    lineage = _lineage(outputs, steps)
    changes, output_changes = [], []
    other = None
    if request.intent == 'changes': comparison_id = revision.parent_revision_id
    if comparison_id is not None:
        other, old_outputs, old_parameters, _ = _revision(sessions, state, comparison_id)
        for key in sorted(set(parameters) | set(old_parameters)):
            if (key in parameters) != (key in old_parameters) or parameters.get(key) != old_parameters.get(key):
                changes.append((key, old_parameters.get(key, 'not present'), parameters.get(key, 'not present')))
        before, after = {o.role:o for o in old_outputs}, {o.role:o for o in outputs}
        for role in sorted(set(before) | set(after)):
            status = ('removed' if role not in after else 'added' if role not in before
                      else 'unchanged' if before[role] == after[role] else 'changed')
            output_changes.append((role, status))
    evidence_count = _evidence_integrity(sessions, state, revision) if request.intent == 'verification' else 0
    reference = steps.get('matrix')
    return TurnResponseFacts(request.intent, rid, state.revisions.index(revision) + 1,
        rid == state.active_revision_id, request.relation, outputs, tuple(parameters.items()),
        None if other is None else other.revision_id,
        None if other is None else state.revisions.index(other) + 1,
        tuple(changes), tuple(output_changes), work, reuse, lineage,
        None if reference is None else reference.resolved_arguments.get('reference_manifest_sha256'),
        tuple(o for o in outputs if o.run_id != revision.run_id), evidence_count)
