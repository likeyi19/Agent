"""Pretrained native DNA-Diffusion generation and bounded sequence descriptors."""
import csv
import hashlib
import itertools
import os
from pathlib import Path
import secrets
import subprocess
import tempfile

import numpy as np

from . import sequence_prediction as shared
from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools._cancellation import cancellation_checkpoint
from agent.tools.data.authority_context import owned_verification
from agent.tools.data.fastq_verification_authority import file_closure

TOOL = 'generate_regulatory_sequences'
KIND = 'regulatory_sequence_generation'
CONTRACT = 'regulatory-sequence-generation.v1'
RESOURCE = 'dna-diffusion-sequence-generator.v1'
POLICY = 'regulatory-sequence-generation-native-v1'
MAPPING = [dict(condition=c,index=i,label=label) for i,c,label in (
    (1,'GM12878','GM12878_ENCLB441ZZZ'), (2,'HepG2','HepG2_ENCLB029COU'),
    (3,'K562','K562_ENCLB843GMH'), (4,'hESCT0','hESCT0_ENCLB449ZZZ'))]
SCHEDULE = dict(timesteps=50,beta_start=0.0001,beta_end=0.2)
PROFILE = digest(dict(contract=CONTRACT,method='DNA-Diffusion native pretrained conditional sampling',
                      context=200,alphabet='ACGT',schedule=SCHEDULE))
LIMITATIONS = ('Synthetic regulatory DNA candidates from a model trained on human accessible DNA in four '
    'released sample conditions. Conditioning is not predicted or measured activity. GC, base-composition '
    'entropy, homopolymer runs and position-wise Hamming diversity describe this batch only; no training-set '
    'novelty, functional cCRE identity, enhancer activity or endogenous regulation is established. Exact '
    'duplicates are retained and uniqueness is not guaranteed. Entropy and homopolymers are simple '
    'low-complexity descriptors, not repeat-family annotation. Hamming diversity does not align sequences '
    'or reverse complements. Stochastic output depends on seed, batching, precision, software and device; '
    'a seed does not guarantee identical results across environments. DHS64 prediction is optional separate '
    'in silico evidence with native trailing zero padding to 500 positions; DeepPBMC requires 2114 real bases.')
FILES = ('sequence-records.json','sequences.fasta','candidates.tsv','metadata.json','composition.png')
OUTPUTS = (('sequences','sequences.fasta'),('characteristics','candidates.tsv'))
TABLE_FIELDS = ('sequence_id','condition','checkpoint_label','length','gc_fraction','entropy_bits',
                'longest_homopolymer','identical_sequence_count','sequence_sha256')
RESULT_FIELDS = dict(status=(str,),contract_version=(str,),manifest_path=(str,),manifest_sha256=(str,),
    sequences_path=(str,),sequences_sha256=(str,),characteristics_path=(str,),characteristics_sha256=(str,),
    n_sequences=(int,),requested_sequences=(int,),condition=(str,),checkpoint_label=(str,),condition_index=(int,),
    sequence_length=(int,),valid_alphabet=(bool,),unique_sequences=(int,),duplicate_sequences=(int,),
    gc_fraction=(dict,),entropy_bits=(dict,),longest_homopolymer=(int,),pairwise_hamming=(dict,),
    seed=(int,),guidance_scale=(float,),sampling_schedule=(dict,),candidate_examples=(list,),examples_omitted=(int,),
    model_release=(str,),limitations=(str,))


def qualify_resource(resource):
    if (resource['schema'] != RESOURCE or resource['family'] != 'DNA-Diffusion'
            or resource['sequence_length'] != 200 or resource['condition_mapping'] != MAPPING
            or resource['conditions'] != [r['condition'] for r in MAPPING] or resource['schedule'] != SCHEDULE):
        raise ValueError('Unsupported released DNA-Diffusion specification or condition mapping')
    metadata = shared.read(resource['metadata']['path'])
    if metadata['condition_mapping'] != MAPPING or not metadata.get('training_labels_sha256'):
        raise ValueError('Exact released training-label provenance is required')
    config = shared.read(resource['config']['path'])
    if config.get('dim') != 200 or config.get('num_classes') != 10 or config.get('channels') != 1:
        raise ValueError('Released checkpoint architecture differs')
    if (Path(resource['checkpoint']['path']).name != 'model.safetensors'
            or Path(resource['checkpoint']['path']).parent != Path(resource['config']['path']).parent):
        raise ValueError('Native local pretrained checkpoint and config must share their model directory')
    settings = resource['runtime']['settings']
    if (not isinstance(settings['device'],str) or not settings['device']
            or type(settings['batch_size']) is not int or settings['batch_size'] < 1
            or type(settings['threads']) is not int or settings['threads'] < 1
            or not Path(resource['runtime']['python']).is_file()):
        raise ValueError('Provision an explicit isolated native runtime, device, batch size and thread count')
    return resource


def resolve_model():
    library = shared.library()
    matches = [e for e in library.entries() if e['kind']=='sequence_generator' and e['resource']['family']=='DNA-Diffusion']
    if len(matches) != 1: raise ValueError('A unique qualified DNA-Diffusion resource must be installed by the operator')
    return library.check(matches[0]),matches[0]


def parameters(args,resource):
    match = [r for r in resource['condition_mapping'] if args.get('condition')==r['condition']]
    if len(match)!=1: raise ValueError('Choose a supported DNA-Diffusion condition; no substitute was selected')
    n=args.get('number_of_sequences',20)
    if type(n) is not int or not 1 <= n <= 1000: raise ValueError('Request 1–1000 candidate sequences')
    seed=args.get('seed')
    if seed is None: seed=secrets.randbits(32)
    if type(seed) is not int or not 0 <= seed < 2**32: raise ValueError('Seed must be an integer from 0 through 4294967295')
    guidance=args.get('guidance_scale',1.0)
    if isinstance(guidance,bool) or not isinstance(guidance,(float,int)) or not np.isfinite(guidance) or guidance<0:
        raise ValueError('Guidance must be finite and nonnegative')
    return dict(number_of_sequences=n,seed=seed,guidance_scale=float(guidance),condition_index=match[0]['index']),match[0]


def native_generate(resource,effective,stage):
    from agent.tools.models import diffusion_runtime
    request=stage/'native.json'
    shared.write(request,dict(model_directory=str(Path(resource['checkpoint']['path']).parent),
        settings=resource['runtime']['settings'],sampling=effective,schedule=resource['schedule'],condition_mapping=resource['condition_mapping']))
    environment=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1')
    environment.pop('PYTHONPATH',None);environment.pop('PYTHONHOME',None)
    with (stage/'native.log').open('wb') as log:
        process=subprocess.Popen([resource['runtime']['python'],'-I','-B',str(Path(diffusion_runtime.__file__).resolve()),str(request)],
            cwd=stage,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
        try:
            while True:
                cancellation_checkpoint()
                try: code=process.wait(timeout=.2);break
                except subprocess.TimeoutExpired: pass
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        if code: raise RuntimeError('Native DNA-Diffusion sampling failed; no checkpoint, condition, algorithm or device was substituted')
    output=shared.read(request.with_suffix('.result.json'));receipt=output['receipt']
    if (receipt['completion']!='succeeded' or receipt['settings']!=resource['runtime']['settings']
            or receipt['sampling']!=effective or receipt['schedule']!=resource['schedule']):
        raise AuthorityError('Native generation completion or effective parameters differ')
    return output


def characteristics(sequences,condition,expected_count):
    from collections import Counter
    from Bio.SeqUtils import gc_fraction
    from scipy.stats import entropy
    from scipy.spatial.distance import pdist
    if len(sequences)!=expected_count or any(not isinstance(s,str) or len(s)!=200 or set(s)-set('ACGT') for s in sequences):
        raise AuthorityError('Native generated count, 200-base length or ACGT alphabet differs')
    counts=Counter(sequences)
    rows=[dict(sequence_id=f'sequence-{i+1}',condition=condition['condition'],checkpoint_label=condition['label'],
        length=len(s),gc_fraction=float(gc_fraction(s)),entropy_bits=float(entropy([s.count(base) for base in "ACGT"],base=2)),
        longest_homopolymer=max(sum(1 for _ in run) for _,run in itertools.groupby(s)),
        identical_sequence_count=counts[s],sequence_sha256=hashlib.sha256(s.encode()).hexdigest()) for i,s in enumerate(sequences)]
    def summary(values):return dict(min=float(min(values)),mean=float(np.mean(values)),max=float(max(values)))
    # At most 499,500 distances for the 1,000-record execution bound. Fixed-length
    # aligned-position Hamming is explicitly not evolutionary or alignment distance.
    distances=pdist(np.asarray([list(s.encode()) for s in sequences],dtype=np.uint8),metric='hamming')*200
    diversity=dict(pairs=len(distances),**(summary(distances) if len(distances) else dict(min=None,mean=None,max=None)),
                   unit='differing positions out of 200; all unordered pairs, no alignment or reverse-complement matching')
    return rows,dict(unique_sequences=len(counts),duplicate_sequences=len(sequences)-len(counts),
        gc_fraction=summary([r['gc_fraction'] for r in rows]),entropy_bits=summary([r['entropy_bits'] for r in rows]),
        longest_homopolymer=max(r['longest_homopolymer'] for r in rows),pairwise_hamming=diversity)


def execute(values,execution_identity=None):
    args=shared.arguments(values);resource,entry=resolve_model();effective,condition=parameters(args,resource)
    root=Path(args['output_dir']);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise ValueError('Generation output is not empty; no overwrite or automatic retry')
    sources=entry['files']
    with tempfile.TemporaryDirectory(prefix='.generation-',dir=root) as temporary:
        stage=Path(temporary);native=native_generate(resource,effective,stage)
        rows,stats=characteristics(native['sequences'],condition,effective['number_of_sequences'])
        records=[dict(r,original_header=r['sequence_id'],sequence=s) for r,s in zip(rows,native['sequences'],strict=True)]
        shared.write(stage/'sequence-records.json',records)
        with (stage/'sequences.fasta').open('x') as stream:
            for r in records:stream.write(f">{r['sequence_id']}\n{r['sequence']}\n")
        with (stage/'candidates.tsv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=TABLE_FIELDS,delimiter='\t');writer.writeheader();writer.writerows(rows)
        summary=dict(status='success',contract_version=CONTRACT,n_sequences=len(rows),requested_sequences=effective['number_of_sequences'],
            condition=condition['condition'],checkpoint_label=condition['label'],condition_index=condition['index'],sequence_length=200,
            valid_alphabet=True,**stats,seed=effective['seed'],guidance_scale=effective['guidance_scale'],sampling_schedule=resource['schedule'],
            candidate_examples=rows[:12],examples_omitted=max(0,len(rows)-12),model_release=resource['release'],limitations=LIMITATIONS)
        shared.write(stage/'metadata.json',dict(summary,candidates=rows,
            model={k:resource[k] for k in ('family','release','conditions','condition_mapping','sequence_length','training_domain','license','source')},
            checkpoint={k:v for k,v in resource['checkpoint'].items() if k!='path'},
            label_provenance=shared.read(resource['metadata']['path']),
            software={k:v for k,v in resource['software'].items() if k!='source_root'},runtime=native['receipt']))
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        fig=Figure(figsize=(10,4));axes=fig.subplots(1,2)
        axes[0].hist([r['gc_fraction'] for r in rows],bins=np.linspace(0,1,21));axes[0].set(xlabel='GC fraction',ylabel='Candidate count')
        axes[1].scatter([r['entropy_bits'] for r in rows],[r['longest_homopolymer'] for r in rows]);axes[1].set(xlabel='Base-composition entropy (bits; maximum 2)',ylabel='Longest homopolymer (bases)')
        fig.suptitle(f"{condition['condition']}: synthetic candidates; composition is not activity validation")
        fig.tight_layout();fig.savefig(stage/'composition.png',dpi=120)
        cancellation_checkpoint()
        if file_closure(r['path'] for r in sources)!=sources:raise AuthorityError('Generator resource changed during sampling')
        for name in FILES:os.link(stage/name,root/name)
    return shared.publish(args,execution_identity,KIND,CONTRACT,summary,sources,FILES,dict(model=entry,runtime=native['receipt']))


@owned_verification(KIND)
def verify_generation(path,*,expected_sha256):
    return shared._verify(path,expected_sha256,KIND)


def generate_regulatory_sequences(output_dir,condition,number_of_sequences=None,seed=None,guidance_scale=None):
    return execute({k:v for k,v in locals().items() if v is not None})
