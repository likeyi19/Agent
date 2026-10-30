"""Accepted sequence artifacts through the existing private snapshot delivery."""
import csv
import io
from pathlib import Path

from agent.schemas.orchestration import _serialize
from agent.tools.analysis import sequence_prediction as owner
from .session_state import digest
from agent.tools.analysis import sequence_optimization, sequence_generation, scatac_regulatory_sequences

CONTRACTS = (owner.CONTRACT, owner.EVAL_CONTRACT, sequence_optimization.CONTRACT, sequence_generation.CONTRACT, scatac_regulatory_sequences.CONTRACT)

FILES = {
    'regions.bed': ('Selected source peak intervals', 'text/plain; charset=utf-8'),
    'windows.bed': ('Extracted reference DNA windows', 'text/plain; charset=utf-8'),
    'selection.tsv': ('scATAC region selection and observed evidence', 'text/tab-separated-values; charset=utf-8'),
    'candidates.tsv': ('Synthetic candidate sequence characteristics', 'text/tab-separated-values; charset=utf-8'),
    'composition.png': ('Synthetic candidate composition', 'image/png'),
    'originals.fasta': ('Original DNA sequences', 'text/plain; charset=utf-8'),
    'design-scores.tsv': ('Original and candidate accessibility scores', 'text/tab-separated-values; charset=utf-8'),
    'mutations.tsv': ('Actual nucleotide substitutions', 'text/tab-separated-values; charset=utf-8'),
    'trajectory.json': ('Native optimization trajectory', 'application/json'),
    'design.png': ('Optimization objective trajectory', 'image/png'),
    'sequences.fasta': ('DNA sequences', 'text/plain; charset=utf-8'),
    'scores.tsv': ('DNase prediction scores', 'text/tab-separated-values; charset=utf-8'),
    'metadata.json': ('Sequence model and measurement metadata', 'application/json'),
    'scores.png': ('DNase score comparison', 'image/png'),
    'metrics.json': ('Matched DNase evaluation', 'application/json'),
}


def _accepted_publication(app, revision, origin, output, *, step_id=None):
    from agent.orchestration.verification_authority import _accepted
    from agent.orchestration.prior_outputs import accepted_step_digest
    store = app._application.run_store
    run_id = revision.run_id if output is None else output.run_id
    step_id = step_id if output is None else output.step_id
    step, authority, execution, anchor = _accepted(store, run_id, step_id)
    result, record = _serialize(step.result), authority.to_dict()
    workspace = app._application._workspace
    root = workspace.runs / workspace.run_digest(run_id) / 'scientific'
    path = Path(result['manifest_path']); path.relative_to(root)
    manifest, sha = owner.load(path, result['manifest_sha256'])
    if (step.tool_name != owner.publication_type(manifest['kind'])['tool']
            or output is not None and accepted_step_digest(step) != output.accepted_step_sha256
            or record['artifact_contract'] != manifest['contract'] or record['science_profile'] != manifest['profile']
            or record['verifier'] != owner.VERIFIER or record['scope'] != 'scientific_correctness.v1'
            or manifest['arguments'] != owner.arguments(step.resolved_arguments)
            or manifest['execution_identity'] != execution
            or record['resources']['manifest_identity'] != manifest
            or record['resources']['result_metadata'] != result or owner.result(manifest,path,sha) != result):
        raise ValueError('Sequence artifacts differ from accepted publication')
    return store, run_id, step_id, step, authority, anchor, root, manifest, sha, record


def accepted_sequences(app, session_id, revision, origin, output, *, step_id=None):
    from agent.orchestration.verification_authority import _accepted
    from agent.orchestration.prior_outputs import accepted_step_digest
    from .interactive_schemas import ScientificArtifactHandle
    from .interactive import _ScientificMatrixArtifact
    store, run_id, step_id, step, authority, anchor, root, manifest, sha, record = _accepted_publication(
        app, revision, origin, output, step_id=step_id)
    items = []
    for pin in manifest['files']:
        payload = Path(pin['path']); name = payload.name
        if name not in FILES: continue
        payload.relative_to(root)
        if pin not in record['files']: raise ValueError('Sequence payload is not authority-bound')
        identity = dict(session_id=session_id, revision_id=revision.revision_id, turn_id=origin.turn_id,
            run_id=run_id, step_id=step_id, accepted_step_sha256=accepted_step_digest(step),
            authority_sha256=authority.identity_sha256, manifest_sha256=sha, payload=pin)
        ref = ScientificArtifactHandle(digest(identity), 'scientific_sequence', pin['sha256'],
            revision.revision_id, origin.turn_id, pin['size_bytes'], name, FILES[name][0])
        items.append(_ScientificMatrixArtifact(ref,payload,root,pin['sha256'],pin['size_bytes'],manifest['contract'],manifest,name,sha))
    if _accepted(store,run_id,step_id)[3] != anchor:
        raise ValueError('Accepted sequence anchor changed during inventory')
    return items


def prepare(stream, role):
    """Redact path-like original labels only at delivery, retaining ordinal mapping.

    Scientific private headers remain exact. Returned bytes have a separate
    delivered checksum via the shared snapshot owner.
    """
    from .dialogue_evidence import _unsafe_fact_value
    raw = stream.read()
    if role in ('scores.tsv', 'selection.tsv'):
        reader = csv.DictReader(io.StringIO(raw.decode()), delimiter='\t')
        rows = list(reader)
        fields = owner.TABLE_FIELDS if role == 'scores.tsv' else scatac_regulatory_sequences.TABLE_FIELDS
        if reader.fieldnames != list(fields): raise ValueError('Sequence table schema changed')
        for row in rows:
            key = 'original_header' if role == 'scores.tsv' else 'feature_id'
            label = row[key]
            if _unsafe_fact_value(label) or label.startswith(('=', '+', '-', '@')):
                row[key] = '[private label omitted; see sequence_id]'
        target = io.StringIO()
        writer = csv.DictWriter(target,fieldnames=fields,delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
        raw = target.getvalue().encode()
    elif role in ('sequences.fasta','originals.fasta'):
        from Bio import SeqIO
        records = list(SeqIO.parse(io.StringIO(raw.decode()), 'fasta'))
        # Ordinal IDs are stable even for repeated, empty, or private headers.
        import re
        raw = ''.join('>' + (r.id if re.fullmatch(r'sequence-\d+', r.id) else f'sequence-{i+1}') + ('' if _unsafe_fact_value(r.description) else ' '+r.description)
                      + '\n' + str(r.seq) + '\n' for i,r in enumerate(records)).encode()
    elif role == 'metadata.json':
        import json
        def public(value):
            if isinstance(value, dict):
                return {k:public(v) for k,v in value.items() if not (k == 'path' or k.endswith('_path'))}
            if isinstance(value, list):
                return [public(v) for v in value]
            if isinstance(value, str) and _unsafe_fact_value(value):
                return '[private label omitted]'
            return value
        raw = json.dumps(public(json.loads(raw)),ensure_ascii=False,sort_keys=True,allow_nan=False).encode()
    elif role not in FILES:
        raise ValueError('Unknown sequence delivery role')
    return raw


def accepted_figures(sessions, turn, revision):
    """Project declared native PNG roles from the same accepted publication owner."""
    from agent.orchestration.verification_authority import _accepted
    from agent.orchestration.prior_outputs import accepted_step_digest
    if revision is None:
        return ()
    result, seen = [], set()
    for output in revision.outputs:
        if output.run_id != turn.run_id: continue
        key = (output.run_id, output.step_id)
        if key in seen: continue
        seen.add(key)
        run = sessions._application.run_store.load(output.run_id)
        step = next(s for s in run.steps if s.step_id == output.step_id)
        if step.tool_name not in (owner.TOOL, sequence_optimization.TOOL, sequence_generation.TOOL): continue
        store, run_id, step_id, step, authority, anchor, root, manifest, sha, record = _accepted_publication(
            sessions, revision, turn, output)
        name, figure_kind = {owner.TOOL: ('scores.png','dnase_score_heatmap'),
            sequence_optimization.TOOL: ('design.png','sequence_optimization_trajectory'),
            sequence_generation.TOOL: ('composition.png','sequence_generation_composition')}[step.tool_name]
        pin = next(p for p in manifest['files'] if Path(p['path']).name == name)
        path = Path(pin['path']); path.relative_to(root)
        if pin not in record['files'] or owner.file_closure([path]) != [pin]:
            raise ValueError('Native figure differs from accepted scientific bytes')
        if _accepted(store, run_id, step_id)[3] != anchor:
            raise ValueError('Native figure accepted anchor changed')
        source = dict(tool_name=step.tool_name, role='native_scientific_figure',
            output=dict(name=output.name, run_id=run_id, step_id=step_id,
                accepted_step_sha256=accepted_step_digest(step), revision_id=revision.revision_id))
        result.append(dict(path=str(path),sha256=pin['sha256'],metadata=dict(
            kind=figure_kind,
            figure_id=name,manifest_sha256=sha,revision_id=revision.revision_id,run_id=run_id,
            step_ids=[step_id],output_names=[o.name for o in revision.outputs if (o.run_id,o.step_id)==key],
            sources=[source])))
    return tuple(result)
