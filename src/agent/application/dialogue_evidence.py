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
from agent.schemas import PriorOutputRef
from agent.schemas.orchestration import _JsonModel, _serialize, freeze_json_mapping
from agent.schemas.verification_authority import VerifiedArtifactAuthority
from .session_state import digest, text

MAX_EVIDENCE_BYTES = 8 * 1024 * 1024
MAX_FACT_BYTES = 16 * 1024
MAX_VIEW_BYTES = 64 * 1024
MAX_FIELDS = 128
MAX_DETAIL_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class DetailRequest:
    """Exact reviewed section, optional subject, and owner-order record bound."""
    section: str
    subject: str | None = None
    limit: int = 8
    candidate: str | None = None

    def __post_init__(self):
        if (type(self.section) is not str or not 0 < len(self.section) <= 128
                or self.subject is not None and (type(self.subject) is not str or not 0 < len(self.subject) <= 128)
                or self.candidate is not None and (type(self.candidate) is not str or not 0 < len(self.candidate) <= 128)
                or type(self.limit) is not int or not 1 <= self.limit <= 32):
            raise ValueError('Invalid bounded detail request.')


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

    def __post_init__(self):
        object.__setattr__(self, 'value', freeze_json_mapping({'value': self.value}, 'fact')['value'])


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


def _project(sessions, state, revision, output, fields, detail):
    run, step, planned, value, evidence_sha = _accepted(sessions, state, revision, output)
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
    if len(facts) > MAX_FIELDS:
        raise _Unavailable('evidence_field_limit')
    workflow = _one([s for s in value['workflow']['ordered_steps'] if s['step_id'] == step.step_id])
    if workflow != dict(step_id=step.step_id, tool_name=step.tool_name, depends_on=list(planned.depends_on)):
        raise _Unavailable('evidence_lineage_mismatch')
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
        if key not in facts:
            return EvidenceFact(key, 'unavailable', reason='not_in_accepted_summary')
        pointer = f'/steps/{index}/facts/' + key.replace('~', '~0').replace('/', '~1')
        if len(_bytes(facts[key])) > MAX_FACT_BYTES:
            return EvidenceFact(key, 'omitted', source_pointer=pointer, reason='field_size_limit')
        return EvidenceFact(key, 'available', facts[key], pointer)

    selected = tuple(sorted(facts)) if fields is None else fields
    # Preserve the existing annotation coverage counter even for field-selected
    # access. No group lookup or annotation semantics are implemented here.
    coverage = (fact('groups_omitted'),) if step.tool_name == 'annotate_scATAC_cell_types' else ()
    result = DialogueEvidence(state.session_id, revision.revision_id, output.name, 'available',
        generation=state.generation, is_active=state.active_revision_id == revision.revision_id,
        source=source, facts=tuple(fact(k) for k in selected), coverage=coverage,
        unselected_fields=tuple(sorted(set(facts) - set(selected))))
    if detail is not None:
        result = replace(result, detail=_detail(sessions, step, detail),
            evidence_scope='accepted_summary_and_published_detail',
            limitations=tuple(x for x in result.limitations if x != 'Detailed sidecars and raw artifacts are not read.') + (
                'Detail is a bounded owner-order subset, not a ranking; omissions are not scientific absence.',))
    if len(_bytes(result)) > MAX_VIEW_BYTES:
        raise _Unavailable('evidence_view_size_limit')
    return result


def detail_request(view, subject):
    """Reviewed result shapes only; no language/intent routing or scientific work."""
    if view.source.tool_name == 'annotate_scATAC_cell_types' and subject is not None:
        rows = next((f.value for f in view.facts if f.field == 'group_summary' and f.status == 'available'), ())
        labels = [row.get('primary_annotation') for row in rows if row['group'] == subject]
        return DetailRequest('annotation_rationale', subject, candidate=labels[0] if len(labels) == 1 else None)
    if view.source.tool_name == 'compute_scATAC_qc' and subject is None:
        return DetailRequest('length_histogram')
    return None


def _detail(sessions, step, request):
    expected = {'annotate_scATAC_cell_types': 'annotation_rationale',
                'compute_scATAC_qc': 'length_histogram'}
    if expected.get(step.tool_name) != request.section:
        raise _Unsupported('unsupported_detail_section')
    if (request.section == 'annotation_rationale') != (request.subject is not None):
        raise _Unsupported('unsupported_detail_subject')
    if request.section != 'annotation_rationale' and request.candidate is not None:
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
        from agent.tools.analysis.annotation_contract import CONTRACT, PROFILE_SHA256
        if manifest['contract_version'] != CONTRACT or manifest['profile_sha256'] != PROFILE_SHA256:
            raise _Unsupported('unsupported_detail_profile')
        groups = records('primary/groups.json', lambda row: row['group'] == request.subject, 'assignment')
        if len(groups) != 1:
            raise _Unavailable('unknown_or_ambiguous_detail_subject')
        candidates = records('primary/candidate_evidence.json', lambda row: row['group'] == request.subject
            and (request.candidate is None or row['candidate'] == request.candidate), 'candidate')
        if request.candidate is not None and len(candidates) != 1:
            raise _Unavailable('unknown_or_ambiguous_detail_candidate')
        coverage = records('primary/signature_coverage.json', lambda row: request.candidate is None
            or row['candidate'] == request.candidate, 'signature_coverage')
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
