"""Accepted reference evidence inventory and bounded path-free table delivery."""
import csv
import io
from pathlib import Path
from agent.tools.analysis import annotation_corroboration as owner
from agent.schemas.orchestration import _serialize
from .session_state import digest

ROLE='reference_evidence'
FILENAME='reference-evidence.tsv'
LABEL='Cell-type reference evidence'


def accepted_corroboration(app,session_id,revision,origin,output,*,step_id=None):
    from agent.orchestration.verification_authority import _accepted
    from agent.orchestration.prior_outputs import accepted_step_digest
    from .interactive_schemas import ScientificArtifactHandle
    from .interactive import _ScientificMatrixArtifact
    store=app._application.run_store
    run_id=revision.run_id if output is None else output.run_id
    step_id=step_id if output is None else output.step_id
    step,authority,execution,anchor=_accepted(store,run_id,step_id)
    result,record=_serialize(step.result),authority.to_dict()
    root=app._application._workspace.runs/app._application._workspace.run_digest(run_id)/'scientific'
    path=Path(result['manifest_path']);path.relative_to(root)
    manifest,sha=owner.load(path,result['manifest_sha256'])
    app._application.registry.get(step.tool_name).result_contract.validate(result)
    if (step.tool_name!=owner.TOOL or output is not None and accepted_step_digest(step)!=output.accepted_step_sha256
            or record['artifact_contract']!=owner.CONTRACT or record['science_profile']!=owner.PROFILE
            or record['verifier']!=owner.VERIFIER or record['scope']!='scientific_correctness.v1'
            or manifest['arguments']!=owner.arguments(step.resolved_arguments) or manifest['execution_identity']!=execution
            or record['resources']['manifest_identity']!=manifest or record['resources']['result_metadata']!=result
            or owner.result(manifest,path,sha)!=result):
        raise ValueError('Reference evidence differs from accepted publication')
    payload=Path(result['evidence_path']);payload.relative_to(root)
    pin=next(f for f in manifest['files'] if f['path']==str(payload))
    if pin not in record['files'] or _accepted(store,run_id,step_id)[3]!=anchor:
        raise ValueError('Reference evidence authority changed')
    identity=dict(session_id=session_id,revision_id=revision.revision_id,turn_id=origin.turn_id,
        run_id=run_id,step_id=step_id,accepted_step_sha256=accepted_step_digest(step),
        authority_sha256=authority.identity_sha256,manifest_sha256=sha,payload=pin,role=ROLE)
    reference=ScientificArtifactHandle(digest(identity),'scientific_reference_evidence',pin['sha256'],
        revision.revision_id,origin.turn_id,pin['size_bytes'],FILENAME,LABEL)
    return _ScientificMatrixArtifact(reference,payload,root,pin['sha256'],pin['size_bytes'],owner.CONTRACT,manifest,ROLE,sha)


def validate_table(stream,manifest):
    """Privacy belongs to delivery; scientific identities are never rewritten."""
    from .dialogue_evidence import _unsafe_fact_value
    stream.seek(0)
    raw=stream.read(64*1024**2+1)
    if len(raw)>64*1024**2: raise ValueError('Reference table exceeds delivery bound')
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8')),delimiter='\t')
    rows=list(reader)
    if reader.fieldnames!=list(owner.TABLE_FIELDS) or len(rows)!=manifest['result']['n_records']:
        raise ValueError('Reference table schema changed')
    if any(None in row or None in row.values() or _unsafe_fact_value(row) for row in rows):
        raise ValueError('Reference table contains unsupported private values')
    stream.seek(0)
