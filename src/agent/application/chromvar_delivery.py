"""Motif artifacts reuse accepted authority and private snapshot delivery."""
import csv
import io
from pathlib import Path
from agent.tools.analysis import chromvar as owner
from agent.schemas.orchestration import _serialize

FILES = {
    'motif-accessibility.h5ad': ('Cell-by-motif deviations and separate z-scores','application/octet-stream'),
    'peaks.tsv': ('Original peak mapping, exclusions, GC and sequence identities','text/tab-separated-values; charset=utf-8'),
    'cells.tsv': ('Original ordered cells','text/tab-separated-values; charset=utf-8'),
    'motifs.tsv': ('Versioned motif annotations and native match fractions','text/tab-separated-values; charset=utf-8'),
    'motif-summary.tsv': ('Descriptive motif variability','text/tab-separated-values; charset=utf-8'),
    'group-summary.tsv': ('Descriptive supplied-group motif means','text/tab-separated-values; charset=utf-8'),
    'motif-matches.npz': ('Sparse retained-peak by motif matches','application/octet-stream'),
    'background-peaks.npy': ('Native background retained-peak indices','application/octet-stream'),
    'patterns.png': ('Motif-associated accessibility patterns','image/png'),
    'metadata.json': ('Motif method, source and resource metadata','application/json'),
    'native-receipt.json': ('Native completion and software versions','application/json'),
}


def _publication(app,revision,origin,output,step_id=None):
    from agent.orchestration.verification_authority import _accepted
    from agent.orchestration.prior_outputs import accepted_step_digest
    store=app._application.run_store;run_id=revision.run_id if output is None else output.run_id
    step_id=step_id if output is None else output.step_id
    step,authority,execution,anchor=_accepted(store,run_id,step_id)
    result=_serialize(step.result);record=authority.to_dict();path=Path(result['manifest_path'])
    workspace=app._application._workspace;root=workspace.runs/workspace.run_digest(run_id)/'scientific';path.relative_to(root)
    manifest,sha=owner.load(path,result['manifest_sha256'])
    if (step.tool_name!=owner.TOOL or output is not None and accepted_step_digest(step)!=output.accepted_step_sha256
            or record['artifact_contract']!=owner.CONTRACT or record['science_profile']!=owner.PROFILE or record['verifier']!=owner.VERIFIER
            or manifest['arguments']!=owner.arguments(step.resolved_arguments) or manifest['execution_identity']!=execution
            or record['resources']['manifest_identity']!=manifest or owner.result(manifest,path,sha)!=result):
        raise ValueError('Motif artifacts differ from accepted publication')
    return store,run_id,step_id,step,authority,anchor,root,manifest,sha,record


def accepted_artifacts(app,session_id,revision,origin,output,*,step_id=None):
    from agent.orchestration.verification_authority import _accepted
    from agent.orchestration.prior_outputs import accepted_step_digest
    from .interactive import _ScientificMatrixArtifact
    from .interactive_schemas import ScientificArtifactHandle
    from .session_state import digest
    store,run_id,step_id,step,authority,anchor,root,manifest,sha,record=_publication(app,revision,origin,output,step_id)
    items=[]
    for pin in manifest['files']:
        path=Path(pin['path']);name=path.name;path.relative_to(root)
        if name not in FILES or pin not in record['files']:raise ValueError('Unbound motif artifact')
        identity=dict(session_id=session_id,revision_id=revision.revision_id,turn_id=origin.turn_id,run_id=run_id,
            step_id=step_id,accepted_step_sha256=accepted_step_digest(step),authority_sha256=authority.identity_sha256,manifest_sha256=sha,payload=pin)
        ref=ScientificArtifactHandle(digest(identity),'scientific_motifs',pin['sha256'],revision.revision_id,origin.turn_id,pin['size_bytes'],name,FILES[name][0])
        delivery=dict(manifest,tool_name=owner.TOOL,accepted_step_sha256=accepted_step_digest(step))
        items.append(_ScientificMatrixArtifact(ref,path,root,pin['sha256'],pin['size_bytes'],owner.CONTRACT,delivery,name,sha))
    if _accepted(store,run_id,step_id)[3]!=anchor:raise ValueError('Motif accepted anchor changed')
    return items


def prepare(stream,role):
    if role.endswith('.json'):
        import json
        from .sequence_delivery import prepare as metadata_prepare
        value=json.load(stream)
        # Raw sessionInfo contains embedded runtime paths; exact package versions
        # remain public, while the full environment receipt stays private.
        value.pop('session',None)
        if isinstance(value.get('receipt'),dict):value['receipt'].pop('session',None)
        return metadata_prepare(io.BytesIO(json.dumps(value).encode()),'metadata.json')
    raw=stream.read()
    if role.endswith('.tsv'):
        from .dialogue_evidence import _unsafe_fact_value
        rows=list(csv.reader(io.StringIO(raw.decode()),delimiter='\t'))
        for row in rows[1:]:
            for i,text in enumerate(row):
                try:float(text);numeric=True
                except ValueError:numeric=False
                if _unsafe_fact_value(text) or not numeric and text.startswith(('=','+','-','@')):
                    row[i]='[private label omitted; use ordered index]'
        target=io.StringIO();csv.writer(target,delimiter='\t').writerows(rows);raw=target.getvalue().encode()
    elif role not in FILES:raise ValueError('Unknown motif artifact')
    return raw


def accepted_figures(app,turn,revision):
    from agent.orchestration.verification_authority import _accepted
    from agent.orchestration.prior_outputs import accepted_step_digest
    if revision is None:return ()
    figures=[];seen=set()
    for output in revision.outputs:
        key=(output.run_id,output.step_id)
        if output.run_id!=turn.run_id or key in seen:continue
        seen.add(key);run=app._application.run_store.load(output.run_id)
        step=next(s for s in run.steps if s.step_id==output.step_id)
        if step.tool_name!=owner.TOOL:continue
        store,run_id,step_id,step,authority,anchor,root,manifest,sha,record=_publication(app,revision,turn,output)
        pin=next(p for p in manifest['files'] if Path(p['path']).name=='patterns.png');path=Path(pin['path']);path.relative_to(root)
        if pin not in record['files'] or owner.file_closure([path])!=[pin]:raise ValueError('Motif figure integrity changed')
        if _accepted(store,run_id,step_id)[3]!=anchor:raise ValueError('Motif figure anchor changed')
        source=dict(tool_name=owner.TOOL,role='native_scientific_figure',output=dict(name=output.name,run_id=run_id,
            step_id=step_id,accepted_step_sha256=accepted_step_digest(step),revision_id=revision.revision_id))
        figures.append(dict(path=str(path),sha256=pin['sha256'],metadata=dict(kind='motif_accessibility_heatmap',figure_id='patterns.png',
            manifest_sha256=sha,revision_id=revision.revision_id,run_id=run_id,step_ids=[step_id],
            output_names=[o.name for o in revision.outputs if (o.run_id,o.step_id)==key],sources=[source])))
    return tuple(figures)
