"""Data-layer adaptation lifecycle using the existing publication/authority owner."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import os
import numpy as np
import torch

from agent.tools.data import scatac_matrix as publication
from agent.tools.data.authority_context import current,authority_operation,VerificationContext,owned_verification
from agent.tools._cancellation import cancellation_checkpoint
from . import contract as c,resources,preprocessing,backend


def arguments(value):
    args=deepcopy(value)
    if set(args)!={'matrices','source_bundle','seam_bundle','strategy','mapping','profile','device','output_dir'}:
        raise ValueError('Invalid adaptation arguments.')
    c.validate_strategy(args['profile'], args['strategy'])
    if args['device']!='cuda:0': raise ValueError('Initial execution profile requires explicit cuda:0; no fallback.')
    if args['strategy'] not in ('de_novo','mapped_reference'):
        raise ValueError('Unknown adaptation strategy.')
    if (args['strategy']=='de_novo')!=(args['mapping'] is None):
        raise ValueError('Mapping resources must agree with the explicit strategy.')
    for key in ('source_bundle','seam_bundle'):
        c.checks.shape(args[key],('path','sha256'))
        c.checks.absolute_path(args[key]['path']); c.checks.sha(args[key]['sha256'])
    c.checks.absolute_path(args['output_dir'])
    if not isinstance(args['matrices'],list) or not 1<=len(args['matrices'])<=64:
        raise ValueError('Explicit ordered matrix bindings required.')
    for binding in args['matrices']:
        c.checks.shape(binding,('manifest_path','manifest_sha256'))
        c.checks.absolute_path(binding['manifest_path']);c.checks.sha(binding['manifest_sha256'])
    return args


def _publication(args,execution_identity):
    args=arguments(args)
    c.checks.sha(execution_identity)
    token=c.digest(dict(arguments=args,execution_identity=execution_identity,policy=c.POLICY,profile=c.profile_sha256(args['profile'])))
    return args,Path(args['output_dir'])/('epizoo-adaptation-'+token),token


def _make_receipt(args,token,sha):
    return dict(artifact_type='agent.epizoo-adaptation-receipt',schema_version=1,policy=c.POLICY,
        execution_identity=token,arguments_sha256=c.digest(args),manifest_sha256=sha,profile_sha256=c.profile_sha256(args['profile']))


def _receipt(destination,args,token=None):
    value=c.read_json(destination/'receipt.json')
    c.checks.sha(value['manifest_sha256']); c.checks.sha(value['execution_identity'])
    if value!=_make_receipt(args,value['execution_identity'],value['manifest_sha256']):
        raise ValueError('Adaptation receipt differs.')
    if destination.name!='epizoo-adaptation-'+value['execution_identity'] or token is not None and token!=value['execution_identity']:
        raise ValueError('Wrong completed publication execution.')
    return value


def load_manifest(path,sha):
    value=c.read_json(path,sha)
    c.checks.shape(value,('artifact_type','contract_version','schema_version','profile_sha256','arguments',
        'reference','source_identity','seam_identity','preprocessing','execution','files','runtime'))
    args=arguments(value['arguments'])
    if (value['artifact_type'],value['contract_version'],value['schema_version'],value['profile_sha256'])!=(c.ARTIFACT,c.CONTRACT,1,c.profile_sha256(args['profile'])):
        raise ValueError('Unsupported target-model contract.')
    required={'checkpoint.pth','sequence_embeddings.npy','sequences.txt','document_frequency.npy','sentence_tokens.npy',
              'sentence_indptr.npy','cells.jsonl','features.txt','preprocessing.json','execution.json','training/training_log.csv'}
    if value['arguments']['strategy']=='mapped_reference':
        required|={'mapping-target.bed','mapping-retained-source.bed','correspondence.json'}
    if set(value['files'])!=required: raise ValueError('Incomplete/unexpected target model sidecars.')
    for name,record in value['files'].items():
        c.checks.shape(record,('sha256','size_bytes'));c.checks.sha(record['sha256']);c.checks.integer(record['size_bytes'],0)
    return value


def _summary(value,path,sha):
    return dict(status='success',artifact_type=c.ARTIFACT,contract_version=c.CONTRACT,
        manifest_path=str(path),manifest_sha256=sha,checkpoint_path=str(path.parent/'checkpoint.pth'),
        checkpoint_sha256=value['files']['checkpoint.pth']['sha256'],completed_step=value['execution']['completed_step'],
        purpose=value['arguments']['profile']['purpose'],n_cells=value['preprocessing']['n_cells'],
        n_features=value['preprocessing']['n_features'])


def _build(args,root):
    root.mkdir()
    source=resources.load_bundle(**args['source_bundle'])
    seam=resources.load_bundle(**args['seam_bundle'])
    if source['contract_version']!='epizoo-source-bundle.v1' or seam['contract_version']!='epizoo-seam-bundle.v1':
        raise ValueError('Wrong source/SEAM bundle type.')
    if not torch.cuda.is_available(): raise ValueError('Requested CUDA unavailable.')
    joint,cells,manifests,reference=preprocessing.corpus(args['matrices'], profile=args['profile'], strategy=args['strategy'])
    torch.cuda.reset_peak_memory_stats(args['device'])
    with backend.seeded(args['profile']['seed']):
        df,sentences=preprocessing.preprocess(joint)
        metadata=preprocessing.save_preprocessing(root,joint,cells,manifests,df,sentences)
        seq=backend.sequences_and_embeddings(reference,joint.var_names.tolist(),seam,args['profile'],root,args['device'])
        mapping=backend.mapping(args['mapping'],source,reference,joint.var_names.tolist(),root) if args['mapping'] else None
        execution=backend.train(source,seq,sentences,args['profile'],args['strategy'],mapping,
            args['mapping']['source_species'] if args['mapping'] else None,root,args['device'])
    files={str(p.relative_to(root)):{k:v for k,v in c.file_record(p).items() if k!='path'}
           for p in sorted(root.rglob('*')) if p.is_file()}
    value=dict(artifact_type=c.ARTIFACT,contract_version=c.CONTRACT,schema_version=1,profile_sha256=c.profile_sha256(args['profile']),
        arguments=args,reference=reference.to_dict(),source_identity=source['identity_sha256'],
        seam_identity=seam['identity_sha256'],preprocessing=metadata,execution=execution,files=files,runtime=resources.runtime())
    record=c.write_json(root/'manifest.json',value)
    verify_target(root/'manifest.json',expected_sha256=record['sha256'])
    for path in root.rglob('*'):
        if path.is_file():
            with path.open('rb') as f: os.fsync(f.fileno())
    from agent.tools.data.scatac_qc_reference import _fsync_dir
    _fsync_dir(root/'training'); _fsync_dir(root)
    return dict(manifest_path=str(root/'manifest.json'),manifest_sha256=record['sha256'])


@owned_verification('epizoo_adaptation')
def verify_target(path,*,expected_sha256):
    path=Path(path);root=path.parent;value=load_manifest(path,expected_sha256);args=value['arguments']
    for name,record in value['files'].items(): c.check_file(dict(path=str(root/name),**record))
    if value['runtime']!=resources.runtime(): raise ValueError('Target runtime compatibility differs.')
    source=resources.load_bundle(**args['source_bundle']);seam=resources.load_bundle(**args['seam_bundle'])
    if (value['source_identity'],value['seam_identity'])!=(source['identity_sha256'],seam['identity_sha256']):
        raise ValueError('Target model resource lineage differs.')
    joint,cells,manifests,reference=preprocessing.corpus(args['matrices'], profile=args['profile'], strategy=args['strategy'])
    if reference.to_dict()!=value['reference']: raise ValueError('Target reference changed.')
    if args['mapping']:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='.verify-map-',dir=root.parent) as scratch:
            expected=backend.mapping(args['mapping'],source,reference,joint.var_names.tolist(),Path(scratch))
            if c.read_json(root/'correspondence.json')!={str(k):v for k,v in expected.items()}:
                raise ValueError('Mapped correspondence differs from public EpiZoo mapping.')
            for name in ('mapping-target.bed','mapping-retained-source.bed'):
                if (root/name).read_bytes()!=(Path(scratch)/name).read_bytes():
                    raise ValueError('Mapped vocabulary order differs.')
    df,sentences=preprocessing.preprocess(joint)
    stored_df=np.load(root/'document_frequency.npy',allow_pickle=False)
    if stored_df.dtype!=np.int64 or not np.array_equal(df,stored_df):
        raise ValueError('Target DF resource is inconsistent with its training corpus.')
    tokens=np.load(root/'sentence_tokens.npy',allow_pickle=False); indptr=np.load(root/'sentence_indptr.npy',allow_pickle=False)
    if tokens.dtype!=np.int64 or indptr.dtype!=np.int64:
        raise ValueError('Sentence resources require signed int64 storage.')
    if not np.array_equal(indptr,np.concatenate(([0],np.cumsum([len(x) for x in sentences],dtype=np.int64)))):
        raise ValueError('Sentence row boundaries differ.')
    if not np.array_equal(tokens,np.asarray([x for row in sentences for x in row],dtype=np.int64)):
        raise ValueError('Sentence ranking differs from EpiZoo output.')
    if (root/'features.txt').read_text().splitlines()!=joint.var_names.tolist(): raise ValueError('Target feature order differs.')
    import json
    if [json.loads(line) for line in (root/'cells.jsonl').read_text().splitlines()]!=cells:
        raise ValueError('Target dataset/cell row identity differs.')
    from epizoo.data.ccre import extract_dna_sequences
    sequences=extract_dna_sequences(reference.genome.fasta.path,joint.var_names.tolist(),fix_chrom_name=False,on_error='raise',show_progress=False)
    if (root/'sequences.txt').read_text().splitlines()!=sequences: raise ValueError('Target DNA sequence order/content differs.')
    seq=np.load(root/'sequence_embeddings.npy',allow_pickle=False)
    if seq.dtype!=np.float32 or seq.shape!=(joint.n_vars,512) or not np.isfinite(seq).all():
        raise ValueError('Invalid sequence embedding resource.')
    execution=c.read_json(root/'execution.json')
    if execution!=value['execution'] or execution['completed_step']!=args['profile']['trainer']['max_steps'] or execution['finite_loss_steps']!=execution['completed_step']:
        raise ValueError('Incomplete target training execution.')
    if execution['optimizer_steps']<1 or execution['amp_skipped_steps']<0 or execution['optimizer_steps']+execution['amp_skipped_steps']!=execution['completed_step']:
        raise ValueError('Invalid optimizer-step accounting.')
    if execution['mid_training_resume'] is not False or not all(np.isfinite(execution['last_losses'][k]) for k in ('loss','sr_loss','cca_loss')):
        raise ValueError('Invalid training execution record.')
    if c.read_json(root/'preprocessing.json')!=value['preprocessing'] or value['preprocessing']!=preprocessing.describe_preprocessing(root,joint,manifests):
        raise ValueError('Preprocessing record differs.')
    from epizoo.models.epizoo_x import EpiZooXConfig
    from epizoo.qualification import strict_target,parameter_scopes,check_forward
    state=torch.load(root/'checkpoint.pth',map_location='cpu',weights_only=True,mmap=True)
    model=strict_target(state,EpiZooXConfig(vocab_size=joint.n_vars+4,**args['profile']['architecture']))
    if parameter_scopes(model)!=execution['parameter_scopes'] or not torch.equal(model.seq_emb.weight[4:],torch.as_tensor(seq)):
        raise ValueError('Trainability or frozen sequence embeddings differ.')
    with backend.seeded(args['profile']['seed']):
        from epizoo.data.datasets import add_special_tokens,truncate_cell
        sample=add_special_tokens(truncate_cell(np.asarray(sentences[0]),8192,True)).unsqueeze(0)
        forward=check_forward(model,sample,device=args['device'],use_amp=True)
    if forward!=execution['expected_forward']: raise ValueError('Target strict reload/forward differs.')
    del model,state
    import gc
    gc.collect();torch.cuda.empty_cache()
    return _summary(value,path,expected_sha256)


def verify_public_result(args,result):
    args=arguments(args);path=Path(result['manifest_path']);destination=path.parent.parent
    if path!=path.resolve() or path!=destination/'artifact/manifest.json' or destination.parent!=Path(args['output_dir']):
        raise ValueError('Target publication path differs.')
    receipt=_receipt(destination,args)
    if receipt['manifest_sha256']!=result['manifest_sha256']: raise ValueError('Target receipt hash differs.')
    value=load_manifest(path,result['manifest_sha256'])
    if value['arguments']!=args or _summary(value,path,result['manifest_sha256'])!=result:
        raise ValueError('Target result/arguments differ.')
    verify_target(path,expected_sha256=result['manifest_sha256'])
    return value


def adapt_epizoo_species(*,execution_identity,**kwargs):
    cancellation_checkpoint(); args,destination,token=_publication(kwargs,execution_identity)
    operation=nullcontext(current()) if current() else authority_operation(VerificationContext())
    with operation as context:
        context.register_execution('epizoo_adaptation',args['output_dir'],execution_identity)
        result=publication._execute_publication(args,destination,token,lambda root:_build(args,root),
            _summary,c.POLICY,c.profile_sha256(args['profile']),load_manifest=load_manifest,receipt_factory=_make_receipt)
        from agent.tools.data.scientific_authority import issue
        # A record is provenance. Trusted context/accepted execution anchors reuse.
        authority=issue(context,'adapt_epizoo_species',args,result,execution_identity)
        return dict(result=result,authority=authority.to_dict())


def recover_adaptation(args,execution_identity):
    args,destination,token=_publication(args,execution_identity)
    receipt=_receipt(destination,args,token);path=destination/'artifact/manifest.json'
    value=load_manifest(path,receipt['manifest_sha256'])
    result=_summary(value,path,receipt['manifest_sha256'])
    verify_public_result(args,result)
    return result
