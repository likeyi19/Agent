"""Thin calls into public EpiZoo; qualification does not implement its science."""
from contextlib import contextmanager
from pathlib import Path
import gc
import random
import numpy as np
import torch

from agent.tools._cancellation import cancellation_checkpoint
from .contract import check_file,file_record,read_json,write_json


@contextmanager
def seeded(seed):
    python=random.getstate(); numpy=np.random.get_state(); cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark)
    deterministic=torch.are_deterministic_algorithms_enabled()
    try:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if cuda is not None: torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(False)
        yield
    finally:
        random.setstate(python); np.random.set_state(numpy); torch.set_rng_state(cpu)
        if cuda is not None: torch.cuda.set_rng_state_all(cuda)
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark=flags
        torch.use_deterministic_algorithms(deterministic)


class CheckedLoader:
    def __init__(self,loader): self.loader=loader
    def __len__(self): return len(self.loader)
    def __iter__(self):
        for batch in self.loader:
            cancellation_checkpoint()
            yield batch


def sequences_and_embeddings(reference,features,seam,profile,root,device):
    from epizoo.data.ccre import extract_dna_sequences,parse_region
    from epizoo.data.datasets import SEAMDataset,collate_fn_seam
    from epizoo.models.seam import SEAM,SEAMConfig
    from epizoo.inference.embeddings import extract_seq_embeddings
    from torch.utils.data import DataLoader
    check_file(seam['checkpoint'])
    sequences=extract_dna_sequences(reference.genome.fasta.path,features,
        fix_chrom_name=False,on_error='raise',show_progress=False)
    if len(sequences)!=len(features) or any(len(seq)!=parse_region(region)[2]-parse_region(region)[1] for region,seq in zip(features,sequences)):
        raise ValueError('Sequence extraction clipped, skipped or reordered target features.')
    (root/'sequences.txt').write_text('\n'.join(sequences)+'\n')
    dataset=SEAMDataset(sequences,seam['config_dir'],max_length=profile['sequence_max_length'],
                        trust_remote_code=True,return_index=True)
    loader=DataLoader(dataset,batch_size=profile['sequence_batch_size'],shuffle=False,num_workers=0,collate_fn=collate_fn_seam)
    model=SEAM(SEAMConfig(seam['config_dir'],trust_remote_code=True))
    state=torch.load(seam['checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True); model.requires_grad_(False)
    result=extract_seq_embeddings(model,CheckedLoader(loader),device=device,use_amp=True,show_progress=False)
    result=np.asarray(result,dtype=np.float32)
    if result.shape!=(len(features),512) or not np.isfinite(result).all():
        raise ValueError('SEAM output dimensions/finiteness differ.')
    np.save(root/'sequence_embeddings.npy',result,allow_pickle=False)
    del model,state,loader,dataset
    gc.collect(); torch.cuda.empty_cache()
    cancellation_checkpoint()
    return result


def mapping_tool_versions(liftover,bedtools):
    import subprocess
    result={}
    for name,command in (('liftover',[liftover]),('bedtools',[bedtools,'--version'])):
        output=subprocess.run(command,capture_output=True,text=True,timeout=10)
        text=(output.stdout+output.stderr).strip()
        if not text or (name=='bedtools' and output.returncode!=0):
            raise ValueError('Unable to identify mapping runtime.')
        result[name]=text.splitlines()[0]
    return result


def mapping(spec,source,reference,features,root):
    from epizoo.data.ccre import build_ccre_map
    from agent.tools.data.scatac_reference import load_scatac_reference_bundle,reinspect_scatac_reference_bundle_sources
    import pandas as pd
    import gzip
    if set(spec)!={'source_species','source_reference','target_reference_identity','chain','liftover','bedtools','min_overlap','direction','tool_versions'}:
        raise ValueError('Invalid mapped resource specification.')
    if spec['source_species'] not in ('human','mouse') or spec['direction']!='target-to-source':
        raise ValueError('Explicit target-to-source mapping required.')
    if spec['target_reference_identity']!=reference.reference_identity_sha256:
        raise ValueError('Mapping target reference differs.')
    if type(spec['min_overlap']) is not int or spec['min_overlap']<1:
        raise ValueError('Minimum overlap must be a positive base count.')
    for key in ('chain','liftover','bedtools','source_reference'): check_file(spec[key])
    if mapping_tool_versions(spec['liftover']['path'],spec['bedtools']['path'])!=spec['tool_versions']:
        raise ValueError('Mapping tool versions differ.')
    _,ref,_=load_scatac_reference_bundle(spec['source_reference']['path'],expected_sha256=spec['source_reference']['sha256'])
    reinspect_scatac_reference_bundle_sources(ref)
    if ref.species!=spec['source_species']: raise ValueError('Mapping source species differs.')
    def lengths(path):
        return {row[0]:int(row[1]) for row in (x.split('\t') for x in Path(path).read_text().splitlines())}
    target_lengths=lengths(reference.genome.fai.path); source_lengths=lengths(ref.genome.fai.path)
    chain_path=spec['chain']['path']; count=0
    with open(chain_path,'rb') as probe: compressed=probe.read(2)==b'\x1f\x8b'
    opener=gzip.open if compressed else open
    with opener(chain_path,'rt') as f:
        for line in f:
            if line.startswith('chain '):
                parts=line.split()
                if len(parts)!=13 or target_lengths.get(parts[2])!=int(parts[3]) or source_lengths.get(parts[7])!=int(parts[8]):
                    raise ValueError('Chain direction/contig lengths incompatible with bound references.')
                count+=1
    if not count: raise ValueError('No chain headers.')
    frame=pd.read_csv(source['retained'][ref.species]['filter']['path'],index_col=0)
    retained=frame['cCRE'].tolist()
    expected_positions=dict(zip(frame['idx'],retained))
    matched=0
    with open(ref.ccre.bed.path) as f:
        for index,line in enumerate(f):
            if index in expected_positions:
                chrom,start,end=line.split()[:3]
                if f'{chrom}:{start}-{end}'!=expected_positions[index]:
                    raise ValueError('Source reference does not match retained checkpoint rows.')
                matched+=1
    if matched!=len(retained): raise ValueError('Incomplete retained source reference vocabulary.')
    new=root/'mapping-target.bed'; old=root/'mapping-retained-source.bed'
    for path,names in ((new,features),(old,retained)):
        with path.open('x') as f:
            for name in names:
                chrom,interval=name.rsplit(':',1); start,end=interval.split('-')
                f.write(f'{chrom}\t{start}\t{end}\n')
    cancellation_checkpoint()
    result=build_ccre_map(str(new),str(old),chain_path,liftover_bin=spec['liftover']['path'],
        bedtools_bin=spec['bedtools']['path'],min_overlap=spec['min_overlap'],tmp_dir=str(root),
        cancel_check=cancellation_checkpoint)
    cancellation_checkpoint()
    write_json(root/'correspondence.json',{str(k):int(v) for k,v in sorted(result.items())})
    return result


def train(source,seq,sentences,profile,strategy,correspondence,source_species,root,device):
    from epizoo.models.epizoo import EpiZooConfig
    from epizoo.models.epizoo_x import EpiZooXConfig
    from epizoo.qualification import qualify_source,initialize_target,parameter_scopes
    from epizoo.data.datasets import CellDatasetX,collate_fn_x
    from epizoo.train.posttrain import EpiZooXPostTrainer,EpiZooXPostTrainConfig
    from torch.utils.data import DataLoader
    state=torch.load(source['checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
    state=qualify_source(state,EpiZooConfig(**source['architecture']))
    cfg=EpiZooXConfig(vocab_size=len(seq)+4,**profile['architecture'])
    model=initialize_target(state,seq,cfg,strategy=strategy,mapping=correspondence,source_species=source_species)
    scopes=parameter_scopes(model)
    if scopes['frozen']!=['seq_emb.weight']: raise ValueError('Unexpected frozen modules.')
    del state
    dataset=CellDatasetX(sentences,num_ccres=len(seq),max_length=profile['max_length'],
                        random_sample=profile['random_sample'],cca_alpha=profile['cca_alpha'])
    generator=torch.Generator().manual_seed(profile['seed'])
    loader=DataLoader(dataset,batch_size=profile['batch_size'],shuffle=True,num_workers=0,
                      generator=generator,collate_fn=collate_fn_x)
    observed_steps=0
    skipped_steps=0
    last_losses={}
    class QualifiedTrainer(EpiZooXPostTrainer):
        def train_step(self,batch):
            nonlocal observed_steps,last_losses,skipped_steps
            cancellation_checkpoint()
            scale=self.scaler.get_scale()
            out=super().train_step(batch)
            if self.scaler.get_scale()<scale:
                skipped_steps+=1
            if not all(np.isfinite(out[k]) for k in ('loss','sr_loss','cca_loss')):
                raise ValueError('Nonfinite post-training loss; publication prohibited.')
            observed_steps+=1
            last_losses={k:out[k] for k in ('loss','sr_loss','cca_loss')}
            cancellation_checkpoint()
            return out
    cfg_train=EpiZooXPostTrainConfig(**profile['trainer'],device=device,output_dir=str(root/'training'))
    trainer=QualifiedTrainer(model,CheckedLoader(loader),cfg_train)
    trainer.train()
    if trainer.global_step!=profile['trainer']['max_steps']: raise ValueError('Training incomplete.')
    if observed_steps==skipped_steps: raise ValueError('No optimizer step applied; backend training not qualified.')
    final_scale=trainer.scaler.get_scale()
    checkpoint=root/'checkpoint.pth'
    torch.save(model.state_dict(),checkpoint)
    # Keep public training logs, but do not retain redundant large periodic checkpoints.
    for path in (root/'training').glob('*.pth'): path.unlink()
    del trainer,model
    gc.collect(); torch.cuda.empty_cache()
    record=dict(completed_step=observed_steps,finite_loss_steps=observed_steps,last_losses=last_losses,
        optimizer_steps=observed_steps-skipped_steps,amp_skipped_steps=skipped_steps,final_amp_scale=final_scale,
        parameter_scopes=scopes,expected_forward=dict(cells=1,embedding_dim=512,decoder_size=len(seq)),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device),
        device=torch.cuda.get_device_name(device),mid_training_resume=False)
    write_json(root/'execution.json',record)
    return record
