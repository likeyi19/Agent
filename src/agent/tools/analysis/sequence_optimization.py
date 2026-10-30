"""DeepPBMC constrained native editing with exact sequence/trajectory checks."""
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from . import sequence_prediction as shared
from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools._cancellation import cancellation_checkpoint
from agent.tools.data.authority_context import owned_verification
from agent.tools.data.fastq_verification_authority import file_closure

TOOL = 'optimize_regulatory_sequences'
KIND = 'regulatory_sequence_optimization'
CONTRACT = 'regulatory-sequence-optimization.v1'
RESOURCE = 'deeppbmc-sequence-optimizer.v1'
POLICY = 'regulatory-sequence-optimization-native-v1'
MEASUREMENT = 'DeepPBMC_normalized_peak_accessibility'
OBJECTIVE = 'target_score - sum(comparison_weight * comparison_score) / 7; denominator is all seven model outputs even when only one comparison condition is selected'
PROFILE = digest(dict(contract=CONTRACT, method='CREsted native ISE weighted_difference',
    context=2114, alphabet='ACGT', objective='target minus one seventh weighted comparison scores'))
LIMITATIONS = (OBJECTIVE + '. DeepPBMC predicts normalized human PBMC peak accessibility, not enhancer activity, '
    'gene expression or endogenous regulatory effects. Native weighted difference penalizes comparison '
    'score increases and rewards decreases; it does not bound absolute comparison changes. Iterations '
    'are substitution steps, not final edit distance. Native steps may worsen the objective. One final '
    'candidate per starting sequence; no uniqueness guarantee. In silico improvement is not experimental '
    'validation. Scores are incompatible with DHS64 DNase scores.')
FILES = ('sequence-records.json', 'sequences.fasta', 'originals.fasta', 'design-scores.tsv',
         'mutations.tsv', 'trajectory.json', 'metadata.json', 'design.png')
OUTPUTS = (('sequences','sequences.fasta'), ('scores','design-scores.tsv'), ('mutations','mutations.tsv'))
RESULT_FIELDS = dict(status=(str,), contract_version=(str,), manifest_path=(str,), manifest_sha256=(str,),
    sequences_path=(str,), sequences_sha256=(str,), scores_path=(str,), scores_sha256=(str,),
    mutations_path=(str,), mutations_sha256=(str,), n_sequences=(int,), conditions=(list,),
    target_condition=(str,), comparison_conditions=(list,), measurement=(str,),
    iterations=(int,), max_changes=(int,), edit_region=(list,), protected_ranges=(list,),
    comparison_weight=(float,), candidate_examples=(list,), examples_omitted=(int,),
    improved_candidates=(int,), constraints_satisfied=(bool,), model_release=(str,), limitations=(str,))
SCORE_FIELDS = ('sequence_id','condition','measurement','original','candidate','change')
MUTATION_FIELDS = ('sequence_id','position_0based','original','candidate')


def qualify_resource(resource):
    if (resource['schema'] != RESOURCE or resource['family'] != 'DeepPBMC'
            or resource['context_bases'] != 2114 or resource['padding'] != 'none'
            or resource['measurements'] != dict(accessibility=MEASUREMENT)
            or resource['training_assembly'] != 'hg38' or len(resource['checkpoints']) != 1):
        raise ValueError('Unsupported DeepPBMC model specification')
    labels = Path(resource['metadata']['path']).read_text().splitlines()
    if labels != resource['conditions'] or len(labels) != 7 or len(set(labels)) != 7:
        raise ValueError('DeepPBMC ordered class metadata differs')
    if not Path(resource['runtime']['python']).is_file():
        raise ValueError('Provision the pinned isolated CREsted runtime')
    settings = resource['runtime']['settings']
    if (settings['device'] not in ('cpu','cuda:0') or type(settings['threads']) is not int
            or settings['threads'] < 1 or type(settings['seed']) is not int):
        raise ValueError('Invalid explicit native execution settings')
    return resource


def resolve_model():
    library = shared.library()
    matches = [e for e in library.entries() if e['kind'] == 'sequence_predictor'
               and e['resource']['family'] == 'DeepPBMC']
    if len(matches) != 1:
        raise ValueError('A unique qualified DeepPBMC resource must be installed by the operator')
    return library.check(matches[0]), matches[0]


def parameters(args, labels):
    target = args.get('condition')
    if target not in labels:
        raise ValueError('Choose an exact supported DeepPBMC target cell condition; no substitute was selected')
    comparisons = ([s.strip() for s in args['comparison_conditions'].split(',')]
                   if args.get('comparison_conditions') is not None else [s for s in labels if s != target])
    if len(set(comparisons)) != len(comparisons) or any(s not in labels or s == target for s in comparisons):
        raise ValueError('Comparison conditions must be distinct exact DeepPBMC names other than the target')
    budget = args.get('max_changes', 5)
    iterations = args.get('iterations', budget)
    if any(type(n) is not int or not 1 <= n <= 100 for n in (budget, iterations)) or iterations > budget:
        raise ValueError('Use 1–100 iterations and max_changes; iterations cannot exceed the requested edit budget')
    start, end = args.get('edit_start',0), args.get('edit_end',2114)
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= 2114:
        raise ValueError('Edit interval must be 0-based half-open within the 2,114-base supplied sequence')
    protected = []
    if args.get('protected_regions') is not None:
        try: protected = [[int(x) for x in r.strip().split(':')] for r in args['protected_regions'].split(',')]
        except (ValueError, TypeError): raise ValueError('Protected regions use start:end, separated by commas') from None
        if any(len(r) != 2 or not 0 <= r[0] < r[1] <= 2114 for r in protected):
            raise ValueError('Protected intervals must be 0-based half-open within 2,114 bases')
    allowed = set(range(start,end)) - {i for a,b in protected for i in range(a,b)}
    if not allowed: raise ValueError('No editable positions remain after protection')
    weight = args.get('comparison_weight',1.0)
    if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not np.isfinite(weight) or weight < 0:
        raise ValueError('Comparison penalty weight must be finite and nonnegative')
    effective = dict(iterations=iterations, max_changes=budget, target_index=labels.index(target),
        comparison_conditions=comparisons, comparison_indices=[labels.index(s) for s in comparisons], comparison_weight=float(weight),
        penalty_weights=[float(weight) if s in comparisons else 0.0 for s in labels],
        flanks=[start,2114-end], protected_ranges=protected)
    return effective, allowed


def native_optimize(resource, rows, effective, stage):
    from agent.tools.models import crested_runtime
    request = stage/'native.json'
    shared.write(request, dict(checkpoint=resource['checkpoints'][0]['path'],
        settings=resource['runtime']['settings'], optimization=effective, sequences=[r['sequence'] for r in rows]))
    environment = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', KERAS_BACKEND='torch')
    environment.pop('PYTHONPATH',None); environment.pop('PYTHONHOME',None)
    with (stage/'native.log').open('wb') as log:
        process = subprocess.Popen([resource['runtime']['python'],'-I','-B',str(Path(crested_runtime.__file__).resolve()),str(request)],
            cwd=stage, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            while True:
                cancellation_checkpoint()
                try: code=process.wait(timeout=.2); break
                except subprocess.TimeoutExpired: pass
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        if code:
            raise RuntimeError('Native CREsted optimization failed; no model, device or objective was substituted')
    output = shared.read(request.with_suffix('.result.json'))
    if (output['receipt']['completion'] != 'succeeded' or output['receipt']['settings'] != resource['runtime']['settings']
            or output['receipt']['optimization'] != effective):
        raise AuthorityError('Native optimization completion or effective settings differ')
    return output


def validate_candidates(rows, native, effective, allowed):
    """Check actual DNA and native trajectory, never rerun scientific search."""
    if len(native['candidates']) != len(rows) or len(native['trajectories']) != len(rows):
        raise AuthorityError('Native candidate count differs from starting sequences')
    summaries, scores, mutations = [], [], []
    for row, candidate, trajectory in zip(rows,native['candidates'],native['trajectories'],strict=True):
        original = row['sequence']; sid = row['sequence_id']
        if len(candidate) != len(original) or set(candidate)-set('ACGT'):
            raise AuthorityError('Native candidate sequence length/alphabet differs')
        changed = [i for i,(a,b) in enumerate(zip(original,candidate,strict=True)) if a != b]
        if len(changed)>effective['max_changes'] or set(changed)-allowed:
            raise AuthorityError('Returned candidate violates actual edit budget or protected positions')
        changes = trajectory['changes']; predictions = np.asarray(trajectory['predictions'],dtype=float)
        if (trajectory['initial_sequence'] != original or trajectory['designed_sequence'] != candidate
                or len(changes) != effective['iterations']+1 or list(changes[0]) != [-1,'N']
                or predictions.shape != (effective['iterations']+1,7) or not np.isfinite(predictions).all()):
            raise AuthorityError('Native trajectory identity or predictions differ')
        reconstructed = list(original)
        for position,base in changes[1:]:
            if type(position) is not int or position not in allowed or base not in ('A','C','G','T') or reconstructed[position] == base:
                raise AuthorityError('Native trajectory contains an invalid substitution')
            reconstructed[position] = base
        if ''.join(reconstructed) != candidate:
            raise AuthorityError('Native trajectory does not reconstruct returned DNA')
        # Native weighted difference is linear: aggregate changes telescope.
        target = effective['target_index']; weights = np.asarray(effective['penalty_weights'])
        objective = predictions[:,target] - predictions @ weights / 7
        summaries.append(dict(sequence_id=sid, original_sha256=row['sequence_sha256'],
            candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(), changed_bases=len(changed),
            changed_positions=changed, substitutions=[dict(position_0based=i,original=original[i],candidate=candidate[i]) for i in changed],
            comparison_scores=[dict(condition=condition,original=float(predictions[0,index]),candidate=float(predictions[-1,index]),change=float(predictions[-1,index]-predictions[0,index])) for condition,index in zip(effective['comparison_conditions'],effective['comparison_indices'],strict=True)],
            original_target=float(predictions[0,target]), candidate_target=float(predictions[-1,target]),
            original_objective=float(objective[0]),candidate_objective=float(objective[-1]),
            objective_change=float(objective[-1]-objective[0]), improved=bool(objective[-1]>objective[0]),
            non_improving_steps=int(np.sum(np.diff(objective)<=0)),constraints_satisfied=True))
        for i in changed: mutations.append(dict(sequence_id=sid,position_0based=i,original=original[i],candidate=candidate[i]))
        scores.append(predictions)
    return summaries, scores, mutations


def execute(values, execution_identity=None):
    args = shared.arguments(values)
    resource, entry = resolve_model(); labels = resource['conditions']
    effective, allowed = parameters(args,labels)
    rows,sources,reference = shared.records(args,context_bases=2114,exact_length=True,max_records=32)
    root=Path(args['output_dir']); root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()): raise ValueError('Optimization output is not empty; no overwrite or automatic retry')
    sources=sorted({r['path']:r for r in sources+entry['files']}.values(),key=lambda r:r['path'])
    with tempfile.TemporaryDirectory(prefix='.design-',dir=root) as temporary:
        stage=Path(temporary)
        native=native_optimize(resource,rows,effective,stage)
        summaries,predictions,mutations=validate_candidates(rows,native,effective,allowed)
        records=[dict(r,candidate_sequence=c) for r,c in zip(rows,native['candidates'],strict=True)]
        shared.write(stage/'sequence-records.json',records)
        shared.write(stage/'trajectory.json',native['trajectories'])
        for name,key in (('sequences.fasta','candidate_sequence'),('originals.fasta','sequence')):
            with (stage/name).open('x') as stream:
                for r in records: stream.write(f">{r['original_header']}\n{r[key]}\n")
        table=[dict(sequence_id=r['sequence_id'],condition=label,measurement=MEASUREMENT,
                    original=float(p[0,j]),candidate=float(p[-1,j]),change=float(p[-1,j]-p[0,j]))
               for r,p in zip(rows,predictions,strict=True) for j,label in enumerate(labels)]
        for name,fields,data in (('design-scores.tsv',SCORE_FIELDS,table),('mutations.tsv',MUTATION_FIELDS,mutations)):
            with (stage/name).open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(data)
        summary=dict(status='success',contract_version=CONTRACT,n_sequences=len(rows),conditions=labels,
            target_condition=args['condition'],comparison_conditions=effective['comparison_conditions'],measurement=MEASUREMENT,
            iterations=effective['iterations'],max_changes=effective['max_changes'],edit_region=[effective['flanks'][0],2114-effective['flanks'][1]],
            protected_ranges=effective['protected_ranges'],comparison_weight=effective['comparison_weight'],
            candidate_examples=summaries[:12],examples_omitted=max(0,len(rows)-12),
            improved_candidates=sum(r['improved'] for r in summaries),constraints_satisfied=True,
            model_release=resource['release'],limitations=LIMITATIONS)
        shared.write(stage/'metadata.json',dict(summary,candidates=summaries,optimization=effective,
            model={k:resource[k] for k in ('family','release','conditions','measurements','context_bases','padding','training_assembly','training_domain','license','source')},
            checkpoints=[{k:v for k,v in r.items() if k!='path'} for r in resource['checkpoints']],
            software={k:v for k,v in resource['software'].items() if k!='source_root'},runtime=native['receipt'],
            source_sha256=args.get('sequence_sha256'),reference=None if reference is None else dict(assembly=reference['assembly'],fasta_sha256=reference['genome']['fasta']['sha256']),
            coordinate_policy='0-based half-open positions in the oriented supplied sequence; no resizing, padding or genomic expansion'))
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        fig=Figure(figsize=(9,4));ax=fig.subplots()
        for row,p in zip(rows[:12],predictions[:12],strict=True):
            objective=p[:,effective['target_index']] - p@np.asarray(effective['penalty_weights'])/7
            ax.plot(range(len(p)),objective-objective[0],marker='.',label=row['sequence_id'])
        ax.axhline(0,color='gray',linewidth=.8);ax.set(xlabel='Native substitution iteration',ylabel='Objective change from original',title='DeepPBMC native ISE; model scores, not experimental validation')
        ax.legend(fontsize=7);fig.tight_layout();fig.savefig(stage/'design.png',dpi=120)
        cancellation_checkpoint()
        if file_closure(r['path'] for r in sources) != sources: raise AuthorityError('Source or model changed during optimization')
        for name in FILES: os.link(stage/name,root/name)
    return shared.publish(args,execution_identity,KIND,CONTRACT,summary,sources,FILES,
                          dict(model=entry,reference=reference,runtime=native['receipt']))


@owned_verification(KIND)
def verify_optimization(path, *, expected_sha256):
    return shared._verify(path,expected_sha256,KIND)


def optimize_regulatory_sequences(output_dir,condition,sequence_path=None,sequence_sha256=None,sequence=None,
        assembly=None,comparison_conditions=None,comparison_weight=None,max_changes=None,iterations=None,
        edit_start=None,edit_end=None,protected_regions=None,sequence_index=None):
    return execute({k:v for k,v in locals().items() if v is not None})
