"""Read accepted presentation evidence without entering scientific verification.

This is a trusted-local, ephemeral view, not authority or a provider prompt. The
accepted session completion pins the evidence bytes; the accepted RunStore pins
their source. Optional reviewed sidecars are resolved only through that accepted
result's pinned publication. No discovery or scientific fallback is permitted.
In particular, do not call build/verify_analysis_evidence here.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Literal, Mapping

from agent.report.evidence import (
    ANALYSIS_EVIDENCE_ARTIFACT_TYPE, ANALYSIS_EVIDENCE_FILENAME,
    ANALYSIS_EVIDENCE_SCHEMA_VERSION, _TOOL_PROJECTIONS,
    _reject_duplicate_keys, _reject_constant,
)
from agent.schemas import PriorOutputRef, StepOutputRef
from agent.schemas.orchestration import _JsonModel, _serialize, freeze_json_mapping
from agent.schemas.verification_authority import VerifiedArtifactAuthority
from .session_state import digest, text

MAX_EVIDENCE_BYTES = 8 * 1024 * 1024
MAX_FACT_BYTES = 16 * 1024
MAX_VIEW_BYTES = 64 * 1024
MAX_FIELDS = 128
MAX_DETAIL_BYTES = 8 * 1024 * 1024


# Reviewed scope explanations only, not new scores, biological interpretations,
# or method implementations. Values and method parameters remain evidence facts.
RESULT_LIMITATIONS = {
    'compute_scATAC_qc': 'QC describes observed barcodes; it does not select barcodes or statistically call cells.',
    'select_scATAC_cells': 'QC selection applies explicit thresholds. Selected barcodes are candidate cells, not a statistical cell-calling result. Thresholds can be adjusted; native defaults are pragmatic starting values, not universal biological optima or ArchR-equivalent scores. An empty selection means no barcode met these criteria; thresholds are not automatically lowered.',
    'annotate_scATAC_cell_types': 'The compact summary records assignments and statuses. It does not contain marker rationale or calibrated confidence; omitted groups are not evidence of absent groups.',
    'build_scATAC_cell_by_ccre': 'Matrix dimensions and count summaries describe structure and value semantics, not biological quality.',
    'build_scATAC_cell_by_features': 'Matrix dimensions and count summaries describe structure and value semantics, not biological quality.',
    'adapt_epizoo_species': 'Adaptation completion and structural qualification do not establish improved biological performance.',
    'cluster_cells': 'Clustering summaries record method parameters and cluster counts. They do not establish cluster correspondence, cell-type identity, biological superiority, significance or causality.',
    'compute_cell_umap': 'UMAP parameters and output dimensions describe a layout. Independently computed coordinates have no established alignment: rotation, reflection or scale differences do not establish biological change. No coordinate comparison or biological superiority is established.',
    'inspect_scATAC': 'Inspection describes the stored matrix structure; it does not establish biological quality.',
}


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


@dataclass(frozen=True)
class DetailRequest:
    """Exact reviewed section, optional subject, and owner-order record bound."""
    section: str
    subject: str | None = None
    limit: int = 8
    candidate: str | None = None
    field: str | None = None
    other_field: str | None = None
    offset: int = 0
    comparison: Mapping | None = None

    def __post_init__(self):
        if (type(self.section) is not str or not 0 < len(self.section) <= 128
                or self.subject is not None and (type(self.subject) is not str or not 0 < len(self.subject) <= 128)
                or self.candidate is not None and (type(self.candidate) is not str or not 0 < len(self.candidate) <= 128)
                or type(self.limit) is not int or not 1 <= self.limit <= 32):
            raise ValueError('Invalid bounded detail request.')
        if (type(self.offset) is not int or self.offset < 0
                or any(v is not None and (type(v) is not str or not 0 < len(v) <= 256)
                       for v in (self.field, self.other_field))):
            raise ValueError('Invalid categorical detail operands.')
        categorical = self.section in ('category_counts', 'category_association', 'categorical_fields')
        if categorical:
            if self.subject is not None or self.candidate is not None:
                raise ValueError('Categorical fields are distinct from annotation group subjects.')
            if self.section != 'categorical_fields' and self.field is None:
                raise ValueError('Select an exact categorical field.')
            if self.section == 'category_association' and self.other_field is None:
                raise ValueError('Select both categorical fields.')
        elif self.field is not None or self.other_field is not None or self.offset or self.comparison is not None:
            raise ValueError('Unsupported detail operands.')
        if self.section != 'category_association' and (self.other_field is not None or self.comparison is not None):
            raise ValueError('Only association reads accept a second source.')
        if self.comparison is not None:
            object.__setattr__(self, 'comparison', freeze_json_mapping(self.comparison, 'comparison'))


@dataclass(frozen=True)
class EvidenceFact(_JsonModel):
    """A whole field, never a silently sliced list. Available null is a value."""
    field: str
    status: Literal['available', 'unavailable', 'omitted']
    value: object = None
    source_pointer: str | None = None
    reason: str | None = None
    artifact_sha256: str | None = None
    artifact_name: str | None = None
    derivation: Mapping | None = None

    def __post_init__(self):
        object.__setattr__(self, 'value', freeze_json_mapping({'value': self.value}, 'fact')['value'])
        if self.derivation is not None:
            object.__setattr__(self, 'derivation', freeze_json_mapping(self.derivation, 'derivation'))


@dataclass(frozen=True)
class EvidenceSource(_JsonModel):
    output_locator: Mapping
    source_run_result_sha256: str
    tool_name: str
    result_contract: str
    recovery_identity: str
    evidence_sha256: str
    evidence_schema_version: int
    step_pointer: str
    verification_checks: tuple[str, ...]
    authority_sha256: str | None
    authority_scope: str | None
    depends_on: tuple[str, ...]
    prior_outputs: tuple[Mapping, ...]

    def __post_init__(self):
        object.__setattr__(self, 'output_locator', freeze_json_mapping(self.output_locator, 'locator'))
        object.__setattr__(self, 'prior_outputs', tuple(
            freeze_json_mapping(item, 'prior_output') for item in self.prior_outputs))


@dataclass(frozen=True)
class DialogueEvidence(_JsonModel):
    session_id: str
    revision_id: str
    output_name: str
    status: Literal['available', 'unavailable', 'unsupported']
    reason: str | None = None
    generation: int | None = None
    is_active: bool | None = None
    source: EvidenceSource | None = None
    facts: tuple[EvidenceFact, ...] = ()
    coverage: tuple[EvidenceFact, ...] = ()
    unselected_fields: tuple[str, ...] = ()
    evidence_scope: str = 'accepted_persisted_summary'
    detail: tuple[EvidenceFact, ...] = ()
    limitations: tuple[str, ...] = (
        'This view is not scientific authority or fresh artifact verification.',
        'Facts describe the accepted source result, not current artifact validity.',
        'A compact summary is not the complete scientific result; missing detail is not scientific absence.',
        'Detailed sidecars and raw artifacts are not read.',
        'Active status is relative to the captured session generation.',
    )


class _Unavailable(ValueError):
    pass


class _Unsupported(ValueError):
    pass


def _one(items):
    if len(items) != 1:
        raise _Unavailable('missing_or_ambiguous_binding')
    return items[0]


def _bytes(value):
    return json.dumps(_serialize(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _accepted(sessions, state, revision, output):
    app = sessions._application
    sessions._validate_revision(revision)
    run = app.run_store.load(output.run_id).to_run_result()
    turn = _one([t for t in state.turns if t.run_id == output.run_id and t.completion_files])
    if (turn.status not in {'activated', 'stale'} or run.status.value != 'SUCCEEDED'
            or run.planning_only or run.plan is None or run.verification is None
            or not run.verification.passed or run.verification.target_type != 'run'
            or run.verification.target_id != run.plan.plan_id or run.errors
            or digest(run.to_dict()) != turn.run_result_sha256):
        raise _Unavailable('source_run_not_accepted')
    step = _one([s for s in run.steps if s.step_id == output.step_id])
    planned = _one([s for s in run.plan.steps if s.step_id == output.step_id])
    if (step.status.value != 'SUCCEEDED' or step.result is None or step.error is not None
            or step.tool_name != planned.tool_name or step.verification is None
            or not step.verification.passed or step.verification.target_type != 'step'
            or step.verification.target_id != step.step_id
            or digest(step.to_dict()) != output.accepted_step_sha256
            or output.output_key not in step.result):
        raise _Unavailable('source_step_not_accepted')

    # run_paths() creates missing directories; derive this existing canonical path
    # using the workspace's identity function instead. Reads must not repair state.
    workspace = app._workspace
    path = workspace.runs / workspace.run_digest(run.run_id) / 'evidence' / ANALYSIS_EVIDENCE_FILENAME
    completion = _one([f for f in turn.completion_files if f.path == str(path)])
    workspace.require_regular_file(path)
    if path.resolve(strict=True) != path:
        raise _Unavailable('unsafe_evidence_path')
    with path.open('rb') as stream:
        raw = stream.read(MAX_EVIDENCE_BYTES + 1)
    if len(raw) > MAX_EVIDENCE_BYTES:
        raise _Unavailable('evidence_size_limit')
    if hashlib.sha256(raw).hexdigest() != completion.sha256:
        raise _Unavailable('evidence_digest_mismatch')
    value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    if (type(value) is not dict or type(value.get('schema_version')) is not int
            or value['schema_version'] != ANALYSIS_EVIDENCE_SCHEMA_VERSION
            or value.get('artifact_type') != ANALYSIS_EVIDENCE_ARTIFACT_TYPE):
        raise _Unsupported('unsupported_evidence_contract')
    if (value.get('status') != 'success' or value.get('run') != dict(
            run_id=run.run_id, request_id=run.request_id, plan_id=run.plan.plan_id,
            planner_name=run.plan.planner_name, source_run_status=run.status.value,
            plan_sha256=digest(run.plan.to_dict()), source_run_result_sha256=digest(run.to_dict()))):
        raise _Unavailable('evidence_source_mismatch')
    return run, step, planned, value, completion.sha256


def _step_summary(sessions, run, step, planned, value):
    """Read one exact accepted step summary, including a named same-run source."""
    if (step.status.value != 'SUCCEEDED' or step.result is None or step.error is not None
            or step.tool_name != planned.tool_name or step.verification is None
            or not step.verification.passed or step.verification.target_type != 'step'
            or step.verification.target_id != step.step_id):
        raise _Unavailable('source_step_not_accepted')
    projection = _TOOL_PROJECTIONS.get(step.tool_name)
    if projection is None or step.tool_name not in sessions._application.registry.names():
        raise _Unsupported('unsupported_tool')
    spec = sessions._application.registry.get(step.tool_name)
    if (frozenset(spec.result_contract.required_fields) != projection.contract_fields
            or spec.recovery_policy_version != projection.recovery_identity):
        raise _Unsupported('unsupported_result_contract')
    identities = {_bytes(i): i for i in value['provenance']['registry_tool_identities']
                  if i['tool_name'] == step.tool_name}
    # A tool can legitimately occur more than once in a run; its contract must
    # still have exactly one identity. Steps are resolved separately by exact ID.
    identity = _one(list(identities.values()))
    if identity != dict(tool_name=step.tool_name, result_contract=spec.result_contract.name,
                        recovery_identity=projection.recovery_identity):
        raise _Unsupported('unsupported_result_contract')
    if (value['provenance']['evidence_projection_version'] != ANALYSIS_EVIDENCE_SCHEMA_VERSION
            or value['provenance']['trace_sha256'] != digest([e.to_dict() for e in run.trace])):
        raise _Unavailable('evidence_provenance_mismatch')
    index, evidence = _one([(i, s) for i, s in enumerate(value['steps']) if s['step_id'] == step.step_id])
    if (evidence['tool_name'] != step.tool_name or evidence['recovery_identity'] != projection.recovery_identity
            or evidence['attempt_count'] != step.attempt_count
            or evidence['verification']['passed'] is not True
            or evidence['verification']['freshly_verified'] is not True
            or not all(type(c) is str for c in evidence['verification']['check_names'])
            or type(evidence['facts']) is not dict):
        raise _Unavailable('evidence_step_mismatch')
    facts = evidence['facts']
    # Base projected fields must agree with the exact accepted lightweight result.
    # Additional facts are the already-reviewed derived projection in the pinned
    # evidence, not fields discovered in arbitrary result payloads or raw files.
    if any(k not in facts or k not in step.result or _bytes(facts[k]) != _bytes(step.result[k])
           for k in projection.fact_fields):
        raise _Unavailable('evidence_fact_mismatch')
    if step.tool_name == 'compute_cell_umap' and (
            'clustering_present' in facts or 'clustering_present' in step.result):
        if facts.get('clustering_present') is not False or step.result.get('clustering_present') is not False:
            raise _Unavailable('evidence_fact_mismatch')
    if len(facts) > MAX_FIELDS:
        raise _Unavailable('evidence_field_limit')
    workflow = _one([s for s in value['workflow']['ordered_steps'] if s['step_id'] == step.step_id])
    if workflow != dict(step_id=step.step_id, tool_name=step.tool_name, depends_on=list(planned.depends_on)):
        raise _Unavailable('evidence_lineage_mismatch')
    return spec, projection, index, evidence


def _clustering_source(run, step, planned):
    """UMAP attribution requires its exact accepted same-plan clustering binding."""
    if step.tool_name not in ('cluster_cells', 'compute_cell_umap'):
        return None
    if step.tool_name == 'compute_cell_umap':
        reference = planned.arguments.get('analysis_path')
        if not isinstance(reference, StepOutputRef) or reference.output_key != 'analysis_path':
            return None
        source_step = _one([s for s in run.steps if s.step_id == reference.step_id])
        source_plan = _one([s for s in run.plan.steps if s.step_id == reference.step_id])
        if source_step.tool_name != 'cluster_cells' or source_plan.tool_name != 'cluster_cells':
            return None
        if (reference.step_id not in planned.depends_on or source_step.result is None
                or step.resolved_arguments.get('analysis_path') != source_step.result.get('analysis_path')):
            raise _Unavailable('clustering_source_binding_mismatch')
        step, planned = source_step, source_plan
    return step, planned


def _clustering_parameter_facts(sessions, run, step, planned, value):
    """Project recorded values only; defaults never come from current code."""
    if step.tool_name not in ('cluster_cells', 'compute_cell_umap'):
        return ()
    if step.tool_name == 'compute_cell_umap' and step.result.get('clustering_present') is False:
        return tuple(EvidenceFact(name, 'unavailable', reason='clustering_not_performed')
                     for name in ('clustering_parameters', 'clustering_resolution_origin'))
    source = _clustering_source(run, step, planned)
    if source is None:
        return tuple(EvidenceFact(name, 'unavailable', reason='exact_clustering_source_not_recorded')
                     for name in ('clustering_parameters', 'clustering_resolution_origin'))
    step, planned = source
    spec, _, index, evidence = _step_summary(sessions, run, step, planned, value)
    fields = evidence['facts']
    explicit = 'resolution' in step.resolved_arguments
    if explicit != ('resolution' in planned.arguments):
        raise _Unavailable('clustering_arguments_mismatch')
    if explicit:
        argument = step.resolved_arguments['resolution']
        try:
            spec.optional_arguments['resolution'].validate('resolution', planned.arguments['resolution'])
            spec.optional_arguments['resolution'].validate('resolution', argument)
        except (ValueError, TypeError) as exc:
            raise _Unavailable('clustering_arguments_invalid') from exc
        if (isinstance(argument, (PriorOutputRef, StepOutputRef))
                or _bytes(argument) != _bytes(planned.arguments['resolution'])
                or argument != fields['resolution']):
            raise _Unavailable('clustering_arguments_mismatch')
    run_index = _one([i for i, item in enumerate(run.steps) if item.step_id == step.step_id])
    return (
        EvidenceFact('clustering_parameters', 'available',
            {'resolution': fields['resolution'], 'random_seed': fields['random_seed']},
            f'/steps/{index}/facts', reason='accepted_clustering_summary'),
        EvidenceFact('clustering_resolution_origin', 'available',
            'explicit_execution_argument' if explicit else 'existing_owner_default',
            f'/steps/{run_index}/resolved_arguments', reason='accepted_execution_argument_presence',
            artifact_sha256=digest(run.to_dict()), artifact_name='accepted_run_result'),
    )


def _clustering_neighbors_facts(sessions, run, step, planned):
    """Recorded qualified input identity, without owner work or artifact reads.

    Equality means the same accepted graph publication/population, not matching
    cluster labels. A literal path or an unqualified legacy graph supplies no
    compatibility evidence.
    """
    unclustered_umap = step.tool_name == 'compute_cell_umap' and step.result.get('clustering_present') is False
    source = (step, planned) if unclustered_umap else _clustering_source(run, step, planned)
    if source is None:
        return ()
    step, planned = source
    reference = planned.arguments.get('analysis_path')
    from agent.orchestration.prior_outputs import bind_output, validate_binding
    from agent.schemas.verification_authority import AuthorityError
    store, registry = sessions._application.run_store, sessions._application.registry
    if isinstance(reference, StepOutputRef):
        producer = _one([s for s in run.steps if s.step_id == reference.step_id])
        if producer.tool_name != 'build_cell_neighbors':
            return ()
        if reference.step_id not in planned.depends_on or reference.output_key != 'analysis_path':
            raise _Unavailable('clustering_neighbors_binding_mismatch')
        # Existing unqualified results can still supply their independent facts.
        if producer.verification.artifact_authority is None:
            return ()
        try:
            binding = bind_output(store, registry, run.run_id, reference.step_id, 'neighbors')
        except AuthorityError as exc:
            raise _Unavailable('clustering_neighbors_authority_mismatch') from exc
        consumption = 'same_plan_qualified_output'
    elif isinstance(reference, PriorOutputRef):
        binding = reference.binding
        if (binding.tool_name != 'build_cell_neighbors' or binding.source_port != 'neighbors'
                or reference.output_key != 'analysis_path'):
            raise _Unavailable('clustering_neighbors_binding_mismatch')
        try:
            producer = validate_binding(binding, store=store, registry=registry)
        except AuthorityError as exc:
            raise _Unavailable('clustering_neighbors_authority_mismatch') from exc
        consumption = 'qualified_historical_output'
    else:
        return ()
    if step.resolved_arguments.get('analysis_path') != producer.result.get('analysis_path'):
        raise _Unavailable('clustering_neighbors_binding_mismatch')
    index = _one([i for i, s in enumerate(run.plan.steps) if s.step_id == step.step_id])
    pointer = f'/plan/steps/{index}/arguments/analysis_path'
    return tuple(EvidenceFact(name, 'available', value, pointer,
        artifact_name='accepted_run_result', artifact_sha256=digest(run.to_dict()))
        for name, value in (
            ('umap_neighbors_identity_sha256' if unclustered_umap else 'clustering_neighbors_identity_sha256', digest(binding.to_dict())),
            ('umap_neighbors_consumption' if unclustered_umap else 'clustering_neighbors_consumption', consumption)))


def _selection_parameter_facts(sessions, run, step, planned, evidence):
    """Attribute the pinned owner's exact thresholds without inventing defaults."""
    if step.tool_name != 'select_scATAC_cells':
        return ()
    from agent.tools.data.scatac_selection_profile import REQUIRED, OPTIONAL, thresholds
    names = (*REQUIRED, *OPTIONAL)
    effective = evidence['facts'].get('effective_thresholds')
    # Historical summaries can omit the owner threshold projection. They remain
    # readable, but cannot establish the additional parameter-origin facts.
    if not isinstance(effective, Mapping) or set(effective) != set(names):
        return ()
    spec = sessions._application.registry.get(step.tool_name)
    try:
        for name in names:
            declaration = name in planned.arguments
            if declaration != (name in step.resolved_arguments):
                raise ValueError('Threshold argument presence differs.')
            if declaration:
                argument = (spec.required_arguments | spec.optional_arguments)[name]
                argument.validate(name, planned.arguments[name])
                argument.validate(name, step.resolved_arguments[name])
                if _bytes(planned.arguments[name]) != _bytes(step.resolved_arguments[name]):
                    raise ValueError('Threshold argument differs.')
        # The existing owner performs exact decimal/rational normalization and
        # optional omission handling. No conversational threshold rules exist.
        if _bytes(thresholds(step.resolved_arguments)) != _bytes(effective):
            raise ValueError('Effective owner threshold differs.')
    except (ValueError, TypeError, KeyError) as exc:
        raise _Unavailable('selection_arguments_mismatch') from exc
    run_index = _one([i for i, item in enumerate(run.steps) if item.step_id == step.step_id])
    return (EvidenceFact('selection_threshold_origins', 'available',
        {name: 'explicit_execution_argument' if name in step.resolved_arguments
         else 'existing_owner_default' for name in names},
        f'/steps/{run_index}/resolved_arguments', reason='accepted_execution_argument_presence',
        artifact_sha256=digest(run.to_dict()), artifact_name='accepted_run_result'),)


def _explicit_user_parameter_facts(sessions, state, run, step, planned):
    """A Session declaration proves user origin only for its exact admitted run.

    Execution argument presence alone never establishes how a historical value
    was supplied. Follow only recorded continuation links; conversation order or
    matching values in unrelated turns confer no attribution.
    """
    if step.tool_name == 'compute_cell_umap':
        source = _clustering_source(run, step, planned)
        if source is None:
            return ()
        step, planned = source
    turn = _one([t for t in state.turns if t.run_id == run.run_id])
    matches = [i for i in state.interactions if i.turn_id == turn.turn_id]
    if not matches:
        return ()
    interaction = _one(matches)
    admitted = interaction.admitted or {}
    if (admitted.get('kind') != 'execute' or admitted.get('operation') != 'plan'
            or admitted.get('request_id') != run.request_id):
        return ()
    from .dialogue_execution import _parameter_input
    from .turn_decisions import _scientific_arguments, admit_scientific_argument
    indices = {item.turn_id: n for n, item in enumerate(state.interactions)}
    linked, current = [], interaction
    while current is not None:
        if len(linked) >= MAX_FIELDS:
            raise _Unavailable('session_parameter_record_limit')
        linked.append(current)
        record = current.admitted or {}
        continuation = record.get('parameter_continuation', record.get('continuation'))
        if continuation is None:
            break
        parent_id = continuation.get('binding_turn_id', continuation.get('origin_turn_id'))
        parent = _one([i for i in state.interactions if i.turn_id == parent_id])
        if (indices[parent.turn_id] >= indices[current.turn_id]
                or parent.base_generation != interaction.base_generation
                or parent.base_revision_id != interaction.base_revision_id):
            raise _Unavailable('session_parameter_binding_mismatch')
        current = parent
    linked.reverse()
    declarations = {}
    try:
        for current in linked:
            record = current.admitted or {}
            arguments = ([] if record.get('argument') is None else [record['argument']])
            arguments += list(record.get('arguments', ()))
            for encoded in arguments:
                declaration, = _scientific_arguments([_serialize(encoded)])
                name, specification = _parameter_input(sessions._application.registry,
                    declaration.tool, declaration.argument)
                argument = admit_scientific_argument(declaration, current.utterance, specification)
                if (name not in record.get('inputs', {})
                        or _bytes(record['inputs'][name]) != _bytes(argument)):
                    raise ValueError('Declaration differs from its admitted binding.')
                declarations[declaration.tool, declaration.argument] = (name, argument)
        names = {}
        for (tool, argument_name), (input_name, argument) in declarations.items():
            if tool != step.tool_name:
                continue
            if (len([p for p in run.plan.steps if p.tool_name == tool]) != 1
                    or input_name not in admitted.get('inputs', {})
                    or _bytes(admitted['inputs'][input_name]) != _bytes(argument)
                    or argument_name not in planned.arguments
                    or _bytes(planned.arguments[argument_name]) != _bytes(argument)
                    or argument_name not in step.resolved_arguments
                    or _bytes(step.resolved_arguments[argument_name]) != _bytes(argument)):
                raise ValueError('Declaration differs from exact accepted execution.')
            names[argument_name] = 'explicit_user_instruction'
    except (ValueError, TypeError, KeyError) as exc:
        raise _Unavailable('session_parameter_binding_mismatch') from exc
    if not names:
        return ()
    # Presentation/status completion does not change an admitted declaration.
    # Pin only the immutable received text, admitted binding and captured scope.
    records = {'interactions': [dict(turn_id=i.turn_id, utterance=i.utterance,
        base_revision_id=i.base_revision_id, base_generation=i.base_generation,
        admitted=_serialize(i.admitted)) for i in linked]}
    return (EvidenceFact('explicit_user_parameters', 'available',
        dict(tool=step.tool_name, step_id=step.step_id, parameters=names), '/interactions',
        reason='accepted_utterance_bound_session_declarations',
        artifact_sha256=digest(records), artifact_name='accepted_session_interactions'),)


def _project(sessions, state, revision, output, fields, detail):
    run, step, planned, value, evidence_sha = _accepted(sessions, state, revision, output)
    spec, projection, index, evidence = _step_summary(sessions, run, step, planned, value)
    facts = evidence['facts']
    parameters = {fact.field: fact for fact in (
        *_clustering_parameter_facts(sessions, run, step, planned, value),
        *_clustering_neighbors_facts(sessions, run, step, planned),
        *_selection_parameter_facts(sessions, run, step, planned, evidence),
        *_explicit_user_parameter_facts(sessions, state, run, step, planned))}
    if len(set(facts) | set(parameters)) > MAX_FIELDS:
        raise _Unavailable('evidence_field_limit')
    authority = None if step.verification.artifact_authority is None else VerifiedArtifactAuthority(
        step.verification.artifact_authority)
    prior = tuple(dict(argument=k, run_id=r.binding.run_id, step_id=r.binding.step_id,
                       output_key=r.output_key, accepted_step_sha256=r.binding.accepted_step_sha256,
                       authority_sha256=r.binding.authority_sha256)
                  for k, r in planned.arguments.items() if isinstance(r, PriorOutputRef))
    source = EvidenceSource(asdict(output), digest(run.to_dict()), step.tool_name,
        spec.result_contract.name, projection.recovery_identity, evidence_sha,
        value['schema_version'], f'/steps/{index}', tuple(evidence['verification']['check_names']),
        None if authority is None else authority.identity_sha256,
        None if authority is None else authority.record['scope'], planned.depends_on, prior)

    def fact(key):
        if key in parameters:
            parameter = parameters[key]
            if parameter.status == 'available' and len(_bytes(parameter.value)) > MAX_FACT_BYTES:
                return replace(parameter, status='omitted', value=None, reason='field_size_limit')
            return parameter
        if key not in facts:
            return EvidenceFact(key, 'unavailable', reason='not_in_accepted_summary')
        pointer = f'/steps/{index}/facts/' + key.replace('~', '~0').replace('/', '~1')
        if len(_bytes(facts[key])) > MAX_FACT_BYTES:
            return EvidenceFact(key, 'omitted', source_pointer=pointer, reason='field_size_limit')
        return EvidenceFact(key, 'available', facts[key], pointer)

    selected = tuple(sorted(set(facts) | set(parameters))) if fields is None else fields
    # Preserve the existing annotation coverage counter even for field-selected
    # access. No group lookup or annotation semantics are implemented here.
    coverage = (fact('groups_omitted'),) if step.tool_name == 'annotate_scATAC_cell_types' else ()
    result = DialogueEvidence(state.session_id, revision.revision_id, output.name, 'available',
        generation=state.generation, is_active=state.active_revision_id == revision.revision_id,
        source=source, facts=tuple(fact(k) for k in selected), coverage=coverage,
        unselected_fields=tuple(sorted((set(facts) | set(parameters)) - set(selected))))
    if step.tool_name in RESULT_LIMITATIONS:
        result = replace(result, limitations=result.limitations + (RESULT_LIMITATIONS[step.tool_name],))
    if step.tool_name == 'annotate_scATAC_cell_types' and step.resolved_arguments.get('source_path'):
        result = replace(result, limitations=result.limitations + (
            'Native MAESTRO uses exact supplied contig names. Peaks without native transcript weights do not contribute; this is not biological absence. No alias correction is performed. Feature participation, when recorded, is available in annotation details. Annotation is a computational result, not independent biological validation.',))
    if detail is not None:
        from .result_reading import read
        if detail.section not in detail_sections(step.tool_name):
            raise _Unsupported('unsupported_detail_section')
        categorical = detail.section in ('categorical_fields', 'category_counts', 'category_association')
        result = replace(result, detail=(
            read(sessions, state, run, step, detail)
            if categorical
            else _detail(sessions, step, detail)),
            evidence_scope='accepted_summary_and_on_demand_facts' if categorical else 'accepted_summary_and_published_detail',
            limitations=tuple(x for x in result.limitations if x != 'Detailed sidecars and raw artifacts are not read.') + (
                'Detail is a bounded owner-order subset, not a ranking; omissions are not scientific absence.',) + (
                ('Categorical metadata and statistics are read after acceptance; derived statistics were not part of the original historical publication.',)
                if categorical else ()) + (
                ('Whole-result annotation support includes only the published primary candidate for each displayed group; other candidates require a group-specific question.',)
                if detail.section == 'annotation_rationale' and detail.subject is None else ()))
    if len(_bytes(result)) > MAX_VIEW_BYTES:
        raise _Unavailable('evidence_view_size_limit')
    return result


def detail_sections(tool):
    """One section inventory for application answering and interactive clients."""
    from .result_reading import CATEGORICAL_TOOLS
    return ({'annotate_scATAC_cell_types': ('annotation_rationale',),
             'compute_scATAC_qc': ('length_histogram',)}.get(tool, ())
            + (('categorical_fields', 'category_counts', 'category_association')
               if tool in CATEGORICAL_TOOLS else ()))


def detail_request(view, subject):
    """Reviewed result shapes only; no language/intent routing or scientific work."""
    if view.source.tool_name == 'annotate_scATAC_cell_types':
        if subject is None:
            return DetailRequest('annotation_rationale')
        rows = next((f.value for f in view.facts if f.field == 'group_summary' and f.status == 'available'), ())
        labels = [row.get('primary_annotation') for row in rows if row['group'] == subject]
        return DetailRequest('annotation_rationale', subject, candidate=labels[0] if len(labels) == 1 else None)
    if view.source.tool_name == 'compute_scATAC_qc' and subject is None:
        return DetailRequest('length_histogram')
    return None


def _detail(sessions, step, request):
    if request.section not in detail_sections(step.tool_name):
        raise _Unsupported('unsupported_detail_section')
    if request.section != 'annotation_rationale' and request.subject is not None:
        raise _Unsupported('unsupported_detail_subject')
    if (request.section != 'annotation_rationale' or request.subject is None) and request.candidate is not None:
        raise _Unsupported('unsupported_detail_candidate')
    result = step.result

    def read(path, sha, size=None):
        path = Path(path)
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise _Unavailable('unsafe_detail_path')
        import stat
        if not stat.S_ISREG(path.stat().st_mode):
            raise _Unavailable('unsafe_detail_path')
        # Read at most the declared bound, including corrupt/enlarged files.
        with path.open('rb') as stream:
            raw = stream.read(MAX_DETAIL_BYTES + 1)
        if len(raw) > MAX_DETAIL_BYTES:
            raise _Unsupported('detail_artifact_size_limit')
        if (size is not None and len(raw) != size) or hashlib.sha256(raw).hexdigest() != sha:
            raise _Unavailable('detail_digest_mismatch')
        return raw

    def parse(raw):
        return json.loads(raw, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)

    path = Path(result['manifest_path'])
    authority = step.verification.artifact_authority
    if authority is not None:
        authority = VerifiedArtifactAuthority(authority).record
        if (authority['publication_path'] != str(path)
                or authority['manifest_sha256'] != result['manifest_sha256']
                or authority['artifact_contract'] != result['contract_version']):
            raise _Unavailable('detail_authority_binding_mismatch')
    manifest = parse(read(path, result['manifest_sha256']))
    if (type(manifest['schema_version']) is not int
            or manifest['contract_version'] != result['contract_version']
            or manifest['artifact_type'] != result['artifact_type']
            or manifest['schema_version'] != result['artifact_schema_version']):
        raise _Unavailable('detail_result_binding_mismatch')
    facts = []

    def sidecar(name, binding):
        # Names come only from reviewed contracts below, never filesystem discovery.
        if authority is not None:
            entry = _one([f for f in authority['files'] if f['path'] == str(path.parent / name)])
            if entry['sha256'] != binding['sha256'] or entry['size_bytes'] != binding['size_bytes']:
                raise _Unavailable('detail_authority_binding_mismatch')
        raw = read(path.parent / name, binding['sha256'], binding['size_bytes'])
        return raw

    def emit(field, value, name, sha, pointer):
        if len(_bytes(value)) > MAX_FACT_BYTES:
            facts.append(EvidenceFact(field, 'omitted', source_pointer=pointer,
                reason='field_size_limit', artifact_sha256=sha, artifact_name=name))
        else:
            facts.append(EvidenceFact(field, 'available', value, pointer,
                artifact_sha256=sha, artifact_name=name))

    def records(name, predicate, field):
        binding = manifest['sidecars'][name]
        rows = parse(sidecar(name, binding))
        if type(rows) is not list or any(type(row) is not dict for row in rows):
            raise _Unavailable('invalid_detail_records')
        selected = [(i, row) for i, row in enumerate(rows) if predicate(row)]
        for n, (i, row) in enumerate(selected[:request.limit]):
            emit(f'{field}.{n}', row, name, binding['sha256'], f'/{i}')
        facts.append(EvidenceFact(field + '.records_omitted', 'available', max(0, len(selected)-request.limit),
            reason='presentation_coverage_count', artifact_name=name, artifact_sha256=binding['sha256']))
        return selected

    if request.section == 'annotation_rationale':
        from agent.tools.analysis.annotation_contract import CONTRACT, profile_sha
        if (manifest['contract_version'] != CONTRACT or manifest['profile_sha256'] !=
                profile_sha(manifest['arguments'], manifest['schema_version'])):
            raise _Unsupported('unsupported_detail_profile')
        name = 'primary/result.json'
        if 'primary/feature-participation.tsv' in manifest['sidecars']:
            binding = manifest['sidecars'][name]
            primary = parse(sidecar(name, binding))
            for key, value in primary['feature_participation'].items():
                emit('feature_participation.' + key, value, name, binding['sha256'], '/feature_participation/' + key)
        groups = records('primary/groups.json', lambda row: request.subject is None or row['group'] == request.subject, 'assignment')
        if request.subject is not None and len(groups) != 1:
            raise _Unavailable('unknown_or_ambiguous_detail_subject')
        # Whole-result questions receive published primary-candidate support for
        # the same bounded groups, never inferred labels or a new ranking.
        selected = {row['group']: row['primary_annotation'] for _, row in groups[:request.limit]}
        candidates = records('primary/candidate_evidence.json', lambda row:
            (row['group'] in selected and row['candidate'] == selected[row['group']]) if request.subject is None else
            (row['group'] == request.subject and (request.candidate is None or row['candidate'] == request.candidate)), 'candidate')
        if request.candidate is not None and len(candidates) != 1:
            raise _Unavailable('unknown_or_ambiguous_detail_candidate')
        coverage = records('primary/signature_coverage.json', lambda row:
            row['candidate'] in set(selected.values()) if request.subject is None else
            (request.candidate is None or row['candidate'] == request.candidate), 'signature_coverage')
        if request.candidate is not None and len(coverage) != 1:
            raise _Unavailable('unknown_or_ambiguous_detail_candidate')
        if request.candidate is not None:
            facts.append(EvidenceFact('candidate.selection', 'available', request.candidate,
                reason='presentation_selection'))
    else:
        from agent.tools.data._barcode_qc_contract import CONTRACT
        from agent.tools.data.scatac_qc_profile import PROFILE_SHA256
        if manifest['contract_version'] != CONTRACT or manifest['science_profile_sha256'] != PROFILE_SHA256:
            raise _Unsupported('unsupported_detail_profile')
        import gzip
        import io
        name = 'lengths.tsv.gz'
        binding = manifest['histogram']
        if binding['path'] != name:
            raise _Unavailable('detail_result_binding_mismatch')
        with gzip.GzipFile(fileobj=io.BytesIO(sidecar(name, binding))) as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise _Unsupported('detail_decoded_size_limit')
        rows = raw.decode('ascii').splitlines()
        if len(rows) != 1001:
            raise _Unavailable('invalid_detail_records')
        for i, row in enumerate(rows):
            key, count = row.split('\t')
            if key != str(i+1) or not count.isascii() or not count.isdecimal():
                raise _Unavailable('invalid_detail_records')
            if i < request.limit:
                emit(f'length_histogram.{key}', dict(length_bin=key, fragment_records=int(count)),
                     name, binding['sha256'], f'line:{i+1}')
        facts.append(EvidenceFact('length_histogram.records_omitted', 'available', len(rows)-request.limit,
            reason='presentation_coverage_count', artifact_name=name, artifact_sha256=binding['sha256']))
    return tuple(facts)


def read_evidence(sessions, session_id, revision_id, output_name, *, fields=None, detail=None):
    """Resolve an exact revision/output name; never search for a replacement.

    `fields` optionally selects whole top-level evidence fields. Missing fields
    are unavailable in this summary, not absent scientifically. IDs and active
    status refer to one captured session snapshot. Failure returns no facts.
    """
    for value in (session_id, revision_id, output_name):
        text(value)
    if detail is not None and not isinstance(detail, DetailRequest):
        raise ValueError('detail must be a DetailRequest.')
    if fields is not None:
        if (not isinstance(fields, (tuple, list)) or len(fields) > MAX_FIELDS
                or any(type(k) is not str or not k or len(k) > 256 for k in fields)
                or len(set(fields)) != len(fields)):
            raise ValueError('fields must be a bounded sequence of unique field names.')
        fields = tuple(fields)
    try:
        state = sessions.load(session_id)
        revision = _one([r for r in state.revisions if r.revision_id == revision_id])
        output = _one([o for o in revision.outputs if o.name == output_name])
        return _project(sessions, state, revision, output, fields, detail)
    except _Unsupported as exc:
        return DialogueEvidence(session_id, revision_id, output_name, 'unsupported', str(exc))
    except _Unavailable as exc:
        return DialogueEvidence(session_id, revision_id, output_name, 'unavailable', str(exc))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, AttributeError, RecursionError):
        return DialogueEvidence(session_id, revision_id, output_name, 'unavailable', 'invalid_accepted_evidence')


def annotation_field_semantics(field):
    """Reviewed native summary meanings; no values or duplicate result authority."""
    return {
        'n_groups': dict(kind='scientific_result', unit='input groups', label='Supplied annotation groups',
            description='Number of independently supplied groups annotated by the computational method; not proof of distinct biological cell types.'),
        'groups_omitted': dict(kind='summary_coverage', unit='group records',
            description='Number of group records omitted from the bounded display summary: max(0, n_groups - 50). This is not a count of unidentified or unsuccessfully annotated groups.'),
        'assigned_cells': dict(kind='scientific_result', unit='cells',
            description='Number of cells assigned a label by the computational annotation method; not independently validated biological identities.'),
        'unassigned_cells': dict(kind='scientific_result', unit='cells',
            description='Number of cells without a computationally assigned annotation; distinct from omitted display records.'),
        'validation_state': dict(kind='interpretation_limit', unit=None,
            description='Independent validation status. not_assessed means computational annotations have not established independently validated biological cell identities.'),
    }.get(field)


def field_semantics(registry, tool, field):
    """Shared reviewed meanings for R5 and interactive evidence; no new facts."""
    from agent.report.analysis_report import _FIELD_LABELS, _REPORT_FIELDS
    spec = registry.get(tool)
    semantics = {}
    parameter_labels = {'clustering_parameters': 'Clustering parameters',
                        'selection_threshold_origins': 'Selection threshold origins',
                        'explicit_user_parameters': 'Explicit user parameters',
                        'clustering_parameters.resolution': 'Leiden resolution',
                        'clustering_parameters.random_seed': 'Leiden random seed',
                        'clustering_resolution_origin': 'Leiden resolution origin'}
    if field in parameter_labels:
        semantics['label'] = parameter_labels[field]
        semantics['label_source'] = 'reviewed_clustering_parameter_projection'
    if field == 'clustering_resolution_origin':
        semantics['description'] = ('This origin is established by accepted execution arguments. '
            'An explicit execution argument alone does not establish how a historical user supplied it. '
            'The effective value comes from the accepted clustering owner result, not current defaults.')
        semantics['description_source'] = 'reviewed_clustering_parameter_projection'
    if field == 'clustering_neighbors_consumption':
        semantics['description'] = ('The accepted clustering argument consumed a qualified neighbors publication. '
            'same_plan_qualified_output records a direct output binding in its plan; '
            'qualified_historical_output records an exact accepted PriorOutputRef. '
            'This does not report cumulative scientific-owner call counts.')
        semantics['description_source'] = 'reviewed_clustering_input_projection'
    if field in ('detail.category_counts', 'detail.category_association'):
        semantics.update(kind='on_demand_descriptive_statistic',
            description='Read after acceptance from exact pinned observation fields. Denominators, missing values, unused categories and pagination are explicit. Source observations retain their publisher/source attribution and are not ground truth. No native scientific analysis was rerun.',
            description_source='scientific_categorical_reader')
    if field == 'detail.categorical_fields':
        semantics.update(kind='observation_field_inventory',
            description='Bounded eligible field-name metadata, not category statistics or biological facts. offset/next_offset page field names. Selecting a field still requires exact source-byte and owner type checks.',
            description_source='scientific_categorical_reader')
    if field.startswith('selection_threshold_origins.'):
        argument_name = field.split('.', 1)[1]
        argument = {**spec.required_arguments, **spec.optional_arguments}.get(argument_name)
        if argument is not None:
            semantics['label'] = argument_name + ' origin'
            semantics['label_source'] = 'registry_argument_identity'
            semantics['description'] = ('The origin comes from exact accepted execution argument presence. '
                f'The accepted effective_thresholds.{argument_name} records the owner-effective value: '
                'a non-null threshold is enabled, including a native default when the argument was omitted; '
                'null denotes a disabled optional filter. Use that accepted value, not current defaults. '
                'This origin alone does not establish a user instruction.')
            semantics['description_source'] = 'reviewed_selection_parameter_projection'
    if field.startswith('explicit_user_parameters.parameters.'):
        semantics['label'] = field.rsplit('.', 1)[1] + ' user origin'
        semantics['label_source'] = 'registry_argument_identity'
        semantics['description'] = ('An exact utterance-bound Session declaration, its admitted input, '
            'and the accepted planned and resolved operation argument agree. '
            'This records an explicit user instruction; effective scientific values remain owner evidence facts.')
        semantics['description_source'] = 'reviewed_session_parameter_projection'
    if field in _REPORT_FIELDS.get(tool, ()) and field in _FIELD_LABELS:
        semantics['label'] = _FIELD_LABELS[field]
        semantics['label_source'] = 'reviewed_report_field'
    description = spec.result_contract.planning_fields.get(field)
    if description is not None:
        semantics['description'] = description.description
        semantics['description_source'] = 'registry_result_contract'
    if tool == 'annotate_scATAC_cell_types':
        reviewed = annotation_field_semantics(field)
        if reviewed is not None:
            semantics.update(reviewed, description_source='reviewed_native_annotation_summary')
    semantics.setdefault('kind', 'execution_parameter' if field in spec.optional_arguments
                         else 'unclassified_accepted_field')
    return semantics


def public_citation_source(source):
    """Project the actual R5 citation contract, never arbitrary owner payloads."""
    keys = ('revision_id', 'evidence_sha256', 'source_run_result_sha256', 'artifact_name',
            'artifact_sha256', 'pointer', 'session_id')
    result = {k:_serialize(source[k]) for k in keys if k in source}
    if 'output_locator' in source:
        result['output_locator'] = {k:source['output_locator'][k] for k in
            ('name','run_id','step_id','output_key','accepted_step_sha256')}
    if 'derivation' in source:
        value = source['derivation']
        result['derivation'] = {k:_serialize(value[k]) for k in ('read_version','cell_order_sha256') if k in value}
        result['derivation']['sources'] = [{k:s[k] for k in (
            'artifact_sha256','field','location','role','accepted_step_sha256','run_id','step_id',
            'source_sha256','neighbors_identity_sha256') if k in s} for s in value.get('sources', ())]
        for key, allowed in (('request', ('section','field','other_field','offset','limit')),
                             ('comparison', ('revision_id','output_name','accepted_step_sha256','subject'))):
            child = value.get(key)
            result['derivation'][key] = None if child is None else {k:child[k] for k in allowed if k in child}
    return result


def figure_metadata(record, *, manifest_sha256, run_id=None, revision=None, publisher=None):
    """One safe projection for historical Run and Interaction figure owners."""
    sources = []
    composite = record.get('presentation', {}).get('delivery') == 'accepted_cell_overlay'
    for item in record['source_artifacts']:
        source = {k:item[k] for k in ('tool_name', 'artifact_kind', 'result_field', 'field', 'role',
                  'source_sha256', 'neighbors_identity_sha256', 'observation_role') if k in item}
        if 'accepted_output' in item:
            locator = item['accepted_output']
            source['output'] = {k:locator[k] for k in ('name', 'run_id', 'step_id', 'output_key',
                                                     'accepted_step_sha256', 'revision_id')}
            if source['output']['revision_id'] is None and revision is not None and locator['run_id'] == run_id:
                source['output']['revision_id'] = revision.revision_id
            source.update(run_id=locator['run_id'], step_id=locator['step_id'], sha256=item['integrity']['sha256'])
        elif 'registered_source' in item:
            source['registered_source'] = {k:item['registered_source'][k]
                for k in ('resource_id', 'record_sha256', 'source_sha256')}
            source['sha256'] = item['integrity']['sha256']
        else:
            source.update(run_id=run_id, step_id=item['producing_step_id'])
        sources.append(source)
    outputs = () if revision is None else revision.outputs
    if composite:
        label = next((s.get('output') for s in sources if s['role'] == 'categories'), None)
        names = [] if label is None else [o.name for o in outputs if all(getattr(o, k) == label[k]
            for k in ('run_id', 'step_id', 'accepted_step_sha256'))]
    else:
        names = [o.name for o in outputs if o.run_id == run_id and o.step_id in record['source_step_ids']]
    public = dict(kind=record['figure_kind'], figure_id=record['figure_id'], manifest_sha256=manifest_sha256,
        revision_id=None if revision is None else revision.revision_id,
        step_ids=record['source_step_ids'], output_names=names, sources=sources)
    if run_id is not None:
        public['run_id'] = run_id
    if publisher is not None:
        public['publisher'] = {k:publisher[k] for k in ('kind', 'session_id', 'turn_id')}
    if composite:
        public['coordinate_reused'] = True
        public['correspondence'] = {k:record['presentation']['correspondence'][k]
            for k in ('basis', 'n_cells', 'cell_order_sha256')}
    return public


def figure_inventory(sessions, turn, revision=None):
    """Distinct accepted figures from pinned manifests, including report copies.

    This replaces file-suffix accounting and the Interactive-only manifest reader.
    Internal manifest storage metadata is never copied to the public projection.
    """
    from agent.report.visualization import ANALYSIS_VISUALIZATION_MANIFEST_FILENAME, _sha256_file
    workspace = sessions._application._workspace
    pinned = {f.path:f.sha256 for f in turn.completion_files}
    figures = {}
    for file in turn.completion_files:
        path = Path(file.path)
        if path.name != ANALYSIS_VISUALIZATION_MANIFEST_FILENAME:
            continue
        workspace.require_regular_file(path)
        if path.stat().st_size > MAX_EVIDENCE_BYTES:
            raise ValueError('Unbounded accepted figure manifest.')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != file.sha256:
            raise ValueError('Changed accepted figure manifest.')
        manifest = json.loads(raw, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
        if manifest['source']['run_id'] != turn.run_id:
            raise ValueError('Changed figure publishing identity.')
        for record in manifest['figures']:
            target = path.parent / record['relative_path']
            target.relative_to(path.parent)
            workspace.require_regular_file(target)
            if (target.resolve() != target or pinned.get(str(target)) != record['png_sha256']
                    or _sha256_file(target) != record['png_sha256']):
                raise ValueError('Changed accepted figure bytes.')
            public = figure_metadata(record, manifest_sha256=file.sha256, run_id=turn.run_id, revision=revision)
            identity = (file.sha256, record['figure_id'])
            figures[identity] = dict(path=str(target), sha256=record['png_sha256'], metadata=public)
    return tuple(figures.values())
