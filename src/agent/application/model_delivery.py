"""Read-only delivery of accepted target weights and a closed metadata projection.

This consumes recorded scientific authority, never runs the adaptation verifier,
loads tensors, or renews compatibility with a changed training implementation.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import canonical, digest

CONTRACT = 'epizoo-target-model.v1'
CHECKPOINT = 'model_checkpoint'
METADATA = 'model_metadata'
FEATURES = 'model_features'
FREQUENCY = 'model_frequency'
# Full-vocabulary accepted M14.4/M14.5 targets are 4.60/3.74 GiB.
# Five GiB covers both without changing the 1 GiB limit of other roles.
MAX_CHECKPOINT_BYTES = 5 * 1024**3
MAX_METADATA_BYTES = 8 * 1024**2
FILES = {
    CHECKPOINT: ('epizoox-checkpoint.pth', 'EpiZooX model checkpoint'),
    METADATA: ('epizoox-model.json', 'Model configuration and reuse requirements'),
    FEATURES: ('epizoox-features.txt', 'Exact ordered EpiZooX feature vocabulary'),
    FREQUENCY: ('epizoox-document-frequency.npy', 'EpiZooX input-corpus document frequency'),
}


def read_json(stream):
    from agent.tools.data.scatac_fragments_v2 import _pairs
    raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError('Model metadata exceeds the bound.')
    value = json.loads(raw, object_pairs_hook=_pairs)
    if canonical(value) != raw:
        raise ValueError('Noncanonical accepted model metadata.')
    return raw, value


def metadata(manifest, sha, *, root=None):
    """No paths, cells, corpus tokens, raw matrices or arbitrary provenance text."""
    p = manifest['arguments']['profile']
    result = dict(format='epizoox-download-metadata.v1', portable_bundle=False,
        model_manifest_sha256=sha, method=p['id'], method_sha256=manifest['profile_sha256'],
        checkpoint=dict(filename=FILES[CHECKPOINT][0], **manifest['files']['checkpoint.pth']),
        species=manifest['reference']['species'], assembly=manifest['reference']['target_assembly'],
        matrix_semantics=p['matrix_semantics'], architecture=p['architecture'],
        trainer=p['trainer'], execution_settings={k:p[k] for k in
            ('batch_size','sequence_batch_size','seed','max_length','num_workers','dtype','autocast')},
        strategy=manifest['arguments']['strategy'], source_model_identity=manifest['source_identity'],
        seam_identity=manifest['seam_identity'],
        ordered_feature_sha256=manifest['preprocessing']['ordered_feature_sha256'],
        n_features=manifest['preprocessing']['n_features'],
        training_cell_count=manifest['preprocessing']['n_cells'],
        training_attempts=manifest['execution']['completed_step'],
        optimizer_updates=manifest['execution']['optimizer_steps'],
        required_not_included={name:manifest['files'][name] for name in
                               ('features.txt','document_frequency.npy')},
        reuse_requirements=[
            'This checkpoint and metadata are not a standalone inference bundle.',
            'Native EpiZooX strict_target needs EpiZooXConfig(vocab_size=n_features+4, **architecture).',
            'Obtain the exact ordered features.txt and training-corpus document_frequency.npy from the model owner; their hashes are listed. They are not included in these downloads.',
            'Use the training_cell_count for native compute_tfidf with scale_factor=10000 and float32; generate_cell_sentences uses species=None and base_offset=4.',
            'Inference requires the pinned EpiZoo implementation and compatible numerical dependencies, plus an exact compatible sparse query matrix and explicit species, assembly and measurement declarations.',
            'Agent target inference uses max_length=8192, random_sample=True, seed=0, batch_size=1, num_workers=0, float32, CUDA float16 autocast.',
            'Model export does not export Agent execution authority, authorize global sharing, or establish biological convergence.',
        ], runtime={k:manifest['runtime'][k] for k in ('python','packages','cuda','cudnn')})
    if manifest['schema_version']==2:
        from .matrix_delivery import _open_matrix_source
        with _open_matrix_source(Path(root)/'transfer-decision.json', Path(root)) as stream:
            decision_bytes, decision = read_json(stream)
        pin = manifest['files']['transfer-decision.json']
        if (hashlib.sha256(decision_bytes).hexdigest()!=pin['sha256']
                or len(decision_bytes)!=pin['size_bytes']):
            raise ValueError('Transfer decision differs from the accepted model.')
        result['format']='epizoox-download-metadata.v2'
        result['requested_strategy']=result['strategy']
        result['strategy']=decision['selected']
        result['transfer_assessment']=dict(decision,candidates=[dict(v,
            chain={k:x for k,x in v['chain'].items() if k!='path'}) for v in decision['candidates']])
        result['model_state']=p['purpose']
        parent=manifest['arguments']['parent']
        result['parent_initialized_manifest_sha256']=parent['manifest_sha256'] if parent else None
        result['training_resume']=('checkpoint-warm-start;no-optimizer-state-resume'
            if parent else 'not-trained;parameter-transfer-only')
        result['available_companion_downloads']={FILES[role][0]:manifest['files'][name]
            for role,name in ((FEATURES,'features.txt'),(FREQUENCY,'document_frequency.npy'))}
        result['required_not_included']={}
        result['reuse_requirements'][2]='Download the exact feature vocabulary and document-frequency companion files listed here with this checkpoint.'
        result['document_frequency_scope']=manifest['preprocessing']['df_scope']
        result['preprocessing_cell_count']=result.pop('training_cell_count')
        result['reuse_requirements'][3]='Use preprocessing_cell_count with the pinned document-frequency vector for native compute_tfidf, scale_factor=10000 and float32; generate_cell_sentences uses species=None and base_offset=4.'
    # Closed owner configurations contain no locators; reject path-like declared
    # species/assembly instead of silently exporting a private user string.
    from .matrix_delivery import _PATH_TEXT
    def check(value):
        if isinstance(value, str):
            if _PATH_TEXT.search(value) or any(ord(c) < 32 for c in value):
                raise ValueError('Private text in model metadata.')
        elif isinstance(value, dict):
            for child in value.values(): check(child)
        elif isinstance(value, list):
            for child in value: check(child)
    check(result)
    return canonical(result)


def accepted_models(service, session_id, revision, origin, output, *, step_id=None):
    from agent.orchestration.verification_authority import _accepted
    from agent.orchestration.prior_outputs import accepted_step_digest
    from agent.tools.models.epizoo_adaptation import publication as owner, contract as c
    from .interactive import _ScientificMatrixArtifact
    from .interactive_schemas import ScientificArtifactHandle
    from .matrix_delivery import _open_matrix_source
    store = service._application.run_store
    run_id = revision.run_id if output is None else output.run_id
    step_id = step_id if output is None else output.step_id
    step, authority, execution, anchor = _accepted(store, run_id, step_id)
    record = authority.to_dict(); result = _serialize(step.result)
    if (store.load(run_id).lifecycle_status.value != 'SUCCEEDED'
            or step.tool_name not in ('adapt_epizoo_species','posttrain_epizoo_model')
            or output is not None and (output.output_key not in ('manifest_path','manifest_sha256')
                or accepted_step_digest(step) != output.accepted_step_sha256)
            or record['schema_version'] != 2 or record['scope'] != 'scientific_correctness.v1'
            or record['completion'] != 'succeeded' or record['artifact_contract'] not in (CONTRACT,c.LIFECYCLE_CONTRACT)
            or record['artifact_type'] != c.ARTIFACT
            or record['verifier'] != dict(id='epizoo.species-adaptation-structural', compatibility_version='1')):
        raise ValueError('No exact accepted target model.')
    service._application.registry.get(step.tool_name).result_contract.validate(result)
    workspace = service._application._workspace
    root = workspace.runs / workspace.run_digest(run_id) / 'scientific'
    path = Path(result['manifest_path'])
    if not path.is_absolute() or path != path.resolve() or root != root.resolve():
        raise ValueError('Unsafe model publication location.')
    path.relative_to(root)
    with _open_matrix_source(path, root) as stream:
        raw, manifest = read_json(stream)
    sha = hashlib.sha256(raw).hexdigest()
    args, destination, token = owner._publication(manifest['arguments'], execution)
    if path != destination/'artifact'/'manifest.json':
        raise ValueError('Model publication differs from accepted execution.')
    receipt_path = destination/'receipt.json'
    with _open_matrix_source(receipt_path, root) as stream:
        receipt_raw, receipt = read_json(stream)
    if (sha != result['manifest_sha256'] or record['publication_path'] != str(path)
            or record['manifest_sha256'] != sha or record['science_profile'] != manifest['profile_sha256']
            or manifest['profile_sha256'] != c.profile_sha256(args['profile'])
            or record['arguments_sha256'] != digest(args)
            or record['receipt_sha256'] != hashlib.sha256(receipt_raw).hexdigest()
            or receipt != owner._make_receipt(args,token,sha)
            or record['resources']['manifest_identity'] != manifest
            or any(result[k] != v for k,v in owner._summary(manifest,path,sha).items())
            or args['output_dir'] != str(step.resolved_arguments['output_dir'])):
        raise ValueError('Model authority/publication/receipt mismatch.')
    files = {entry['path']:entry for entry in record['files']}
    checkpoint = path.parent/'checkpoint.pth'; pin = manifest['files']['checkpoint.pth']
    for entry in (dict(path=str(path),sha256=sha,size_bytes=len(raw)),
                  dict(path=str(receipt_path),sha256=hashlib.sha256(receipt_raw).hexdigest(),size_bytes=len(receipt_raw)),
                  dict(path=str(checkpoint),**pin)):
        if files.get(entry['path']) != entry: raise ValueError('Unpinned model publication.')
    # Inventory does not read 1.2 GiB. Download hashes a private snapshot before
    # sending any headers. Metadata uses the pinned manifest and decision bytes.
    projected = metadata(manifest,sha,root=path.parent)
    items=[]
    payloads=[(CHECKPOINT,checkpoint,pin['sha256'],pin['size_bytes']),(METADATA,path,sha,len(raw))]
    if manifest['schema_version']==2:
        for role,name in ((FEATURES,'features.txt'),(FREQUENCY,'document_frequency.npy')):
            payload=path.parent/name;record=manifest['files'][name]
            if files.get(str(payload))!=dict(path=str(payload),**record): raise ValueError('Unpinned model companion.')
            payloads.append((role,payload,record['sha256'],record['size_bytes']))
    for role, payload, payload_sha, size in payloads:
        filename,label=FILES[role]
        if manifest['schema_version']==2:
            state_label = ('Initialized' if manifest['arguments']['profile']['purpose']=='initialization'
                           else 'Post-trained')
            label = state_label + ': ' + label
        identity=dict(session_id=session_id,revision_id=revision.revision_id,turn_id=origin.turn_id,
            output=None if output is None else asdict(output),run_id=run_id,step_id=step_id,
            accepted_step_sha256=accepted_step_digest(step),authority_sha256=authority.identity_sha256,
            role=role,manifest_sha256=sha,payload_sha256=payload_sha,size_bytes=size,
            projection_sha256=hashlib.sha256(projected).hexdigest() if role==METADATA else None)
        reference=ScientificArtifactHandle(digest(identity),'scientific_model',payload_sha,
            revision.revision_id,origin.turn_id,size,filename,label)
        items.append(_ScientificMatrixArtifact(reference,payload,root,payload_sha,size,CONTRACT,manifest,role,sha))
    if _accepted(store,run_id,step_id)[3] != anchor: raise ValueError('Accepted model changed.')
    return items
