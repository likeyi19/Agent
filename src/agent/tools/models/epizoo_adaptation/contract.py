"""Closed M14.2 scientific and execution contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from agent.schemas.verification_authority import canonical, digest
from agent.tools.data import scatac_matrix_contract as checks
from agent.tools._cancellation import cancellation_checkpoint

CONTRACT = 'epizoo-target-model.v1'
ARTIFACT = 'agent.epizoo-target-model'
POLICY = 'adapt-epizoo-species-v1'
PROFILE = dict(id='epizoo-species-posttraining.v1', matrix_semantics='fragment_counts',
    df='epizoo.compute_document_frequency;all-participating-cells',
    tfidf='epizoo.compute_tfidf;scale=10000;float32', sentences='epizoo.generate_cell_sentences;offset=4;species=None',
    ranking='numpy.argsort(-values);runtime-pinned;no-new-secondary-rule',
    max_length=8192, random_sample=True, cca_alpha=1.0,
    architecture=dict(emb_dim=512,num_layers=30,num_heads=8,max_rank=8192,use_moe=True,
        num_experts=4,top_k=2,use_flash_attn=True,normalize_topk=True,hidden_dropout=0.1,
        attn_dropout=0.1,layer_norm_eps=1e-12,hidden_act='gelu',cca_hidden_dim=128,
        cca_pos_weight=1.0,signal_pos_weight=100.0,init_range=0.02,pad_token_id=0,ccre_offset=4),
    trainer=dict(mode='sr_cca',max_steps=500000,save_steps=4000,log_steps=500,keep_last=1,
        lr=5e-5,weight_decay=0.01,warmup_steps=1000,epoch_decay=0.9,sr_weight=1.0,
        cca_weight=1.0,use_amp=True,grad_clip=None,max_cca_metric_samples=10000,freeze_seq_emb=True),
    optimizer='torch.optim.AdamW;betas=(0.9,0.999);eps=1e-8;amsgrad=False;runtime-default-kernels',
    scheduler='public-LambdaLR:step/warmup-then-epoch_decay**floor(step/len(loader))',
    batch_size=4,sequence_batch_size=128,sequence_max_length=512,num_workers=0,
    dtype='float32',autocast='cuda-float16',frozen=['seq_emb.weight'],seed=0,
    amp_step_accounting='public-global-step-counts-attempts;record-applied-and-skipped;require-applied-step',
    training_resume='completed-publication-only;no-optimizer-scheduler-RNG-checkpoint',
    production_schedule_status='public-default-candidate;biological-convergence-not-qualified')
PROFILE_SHA256 = digest(PROFILE)


def execution_profile(*, purpose, seed, batch_size, sequence_batch_size, max_steps, save_steps, log_steps):
    if purpose not in ('qualification','production'):
        raise ValueError('Explicit execution purpose required.')
    for name,value in locals().copy().items():
        if name not in ('purpose',) and (type(value) is not int or value < (0 if name=='seed' else 1)):
            raise ValueError(f'Invalid execution setting: {name}')
    if seed >= 2**32 or batch_size > 4 or sequence_batch_size > 128:
        raise ValueError('Execution exceeds the qualified batch/seed bounds.')
    if purpose=='qualification' and max_steps > 10:
        raise ValueError('Qualification is bounded to ten steps.')
    if purpose=='production' and (batch_size,max_steps,save_steps,log_steps)!=(4,500000,4000,500):
        raise ValueError('Production profile settings differ.')
    if save_steps > max_steps or log_steps > max_steps:
        raise ValueError('Intervals exceed execution length.')
    value=deepcopy(PROFILE)
    value.update(purpose=purpose,seed=seed,batch_size=batch_size,sequence_batch_size=sequence_batch_size)
    value['trainer'].update(max_steps=max_steps,save_steps=save_steps,log_steps=log_steps)
    return value


def file_record(path):
    from agent.tools.data._fragments_common import snapshot
    path=Path(path).resolve(strict=True)
    before=snapshot(path)
    size=path.stat().st_size
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            cancellation_checkpoint(); h.update(chunk)
    if before != snapshot(path): raise ValueError('Resource changed while hashing.')
    return dict(path=str(path),sha256=h.hexdigest(),size_bytes=size)


def check_file(record):
    checks.shape(record,('path','sha256','size_bytes'))
    checks.absolute_path(record['path']); checks.sha(record['sha256']); checks.integer(record['size_bytes'],0)
    if file_record(record['path']) != record: raise ValueError('Resource identity changed.')


def write_json(path,value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value)); stream.flush()
        import os
        os.fsync(stream.fileno())
    return file_record(path)


def read_json(path,sha=None):
    from agent.tools.data.scatac_fragments_v2 import _pairs
    with Path(path).open('rb') as stream: raw=stream.read(8*1024*1024+1)
    if len(raw)>8*1024*1024 or sha is not None and hashlib.sha256(raw).hexdigest()!=sha:
        raise ValueError('Manifest size/hash mismatch.')
    value=json.loads(raw,object_pairs_hook=_pairs)
    if canonical(value)!=raw: raise ValueError('Noncanonical manifest.')
    return value


def validate_execution(value):
    expected=execution_profile(purpose=value['purpose'],seed=value['seed'],batch_size=value['batch_size'],
        sequence_batch_size=value['sequence_batch_size'],**{k:value['trainer'][k] for k in ('max_steps','save_steps','log_steps')})
    if value!=expected: raise ValueError('Unqualified scientific profile.')
    return value
