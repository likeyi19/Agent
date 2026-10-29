"""Native DHS64 prediction and matched-observation evaluation publications.

Sequence parsing and indexed extraction use Biopython/pandas/pysam. This owner
checks model applicability once, calls the isolated native runtime, and publishes
ordered sequences separately from scores. Reads never repeat inference.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError, VerifiedArtifactAuthority, digest
from agent.tools._cancellation import cancellation_checkpoint
from agent.tools.data.authority_context import current, owned_verification
from agent.tools.data.fastq_verification_authority import file_closure
from . import annotation_contract as io

TOOL = 'predict_regulatory_sequences'
EVALUATE = 'evaluate_regulatory_predictions'
KIND = 'regulatory_sequence_prediction'
EVAL_KIND = 'regulatory_sequence_evaluation'
CONTRACT = 'regulatory-sequence-prediction.v1'
EVAL_CONTRACT = 'regulatory-sequence-evaluation.v1'
RESOURCE = 'dhs64-sequence-predictor.v1'
POLICY = 'regulatory-sequence-native-v1'
VERIFIER = dict(id='agent.regulatory-sequence-completion', compatibility_version='1')
PROFILE = digest(dict(contract=CONTRACT, folds=[0, 1, 3], mean='arithmetic',
    padding='native left: bases then zero vectors to 500', alphabet='ACGT; uppercase',
    measurement='native two-head DHS64; no assay conversion'))
MEASUREMENTS = dict(continuous='log10_quantile_normalized_DNase_signal', binary='DNase_peak_call')
LIMITATIONS = ('Human DHS64 DNase-accessibility prediction, not measured enhancer activity, ATAC '
    'accessibility, MPRA activity or gene expression. Continuous scores are transformed signal, '
    'not probabilities; negative values are valid. The sigmoid head predicts peak calls, not '
    'enhancer functionality. Fold standard deviation is disagreement, not calibrated uncertainty '
    'or observed error. Folds share training data; their mean is not independent validation of '
    'designs optimized against these models. Short sequences use zero padding, not genomic flanks. '
    'Novel sequences and training overlap have not been biologically validated.')
RESULT_FIELDS = dict(status=(str,), contract_version=(str,), manifest_path=(str,), manifest_sha256=(str,),
    sequences_path=(str,), sequences_sha256=(str,), scores_path=(str,), scores_sha256=(str,),
    n_sequences=(int,), n_conditions=(int,), conditions=(list,), head=(str,), measurement=(str,),
    folds=(list,), score_min=(float,), score_max=(float,), score_examples=(list,), examples_omitted=(int,),
    model_release=(str,), normalization=(str,), limitations=(str,))
EVAL_FIELDS = dict(status=(str,), contract_version=(str,), manifest_path=(str,), manifest_sha256=(str,),
    metrics_path=(str,), metrics_sha256=(str,), n_observations=(int,), measurement=(str,),
    metrics=(list,), limitations=(str,))
TABLE_FIELDS = ('sequence_id', 'original_header', 'sequence_sha256', 'condition_index', 'condition',
                'measurement', 'mean', 'fold_std', 'fold_0', 'fold_1', 'fold_3')


def read(path, sha=None):
    return io.read_json(path, sha, limit=64 * 1024**2)


def write(path, value):
    data = io.matrix.canonical(value)
    with Path(path).open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def qualify_resource(resource):
    """Installation-time metadata qualification; native parity is a separate gate."""
    if resource['schema'] != RESOURCE or resource['family'] != 'DHS64':
        raise ValueError('Only the qualified DHS64 accessibility family is supported')
    if ([r['fold'] for r in resource['checkpoints']] != [0, 1, 3]
            or resource['context_bases'] != 500 or resource['measurements'] != MEASUREMENTS
            or resource['padding'] != 'left' or resource['training_assembly'] != 'hg38'):
        raise ValueError('Unsupported DHS64 science or fold configuration')
    import pandas as pd
    metadata = pd.read_excel(resource['metadata']['path'])
    labels = metadata['Biosample name'].tolist()
    if len(labels) != 64 or labels != resource['conditions'] or len(set(labels)) != 64:
        raise ValueError('DHS64 ordered output conditions differ from released metadata')
    if not Path(resource['runtime']['python']).is_file() or not Path(resource['software']['source_root']).is_dir():
        raise ValueError('Provision an isolated native DHS runtime and pinned upstream source')
    return resource


def library():
    from agent.application.shared_resources import current_scientific_library, ScientificResourceLibrary
    value = current_scientific_library()
    if value is None:
        root = os.environ.get('AGENT_SCIENTIFIC_RESOURCE_ROOT')
        if not root:
            raise ValueError('The operator must configure a scientific resource library')
        value = ScientificResourceLibrary(root)
    return value


def resolve_model():
    lib = library()
    choices = [r for r in lib.entries() if r['kind'] == 'sequence_predictor' and r['resource']['family'] == 'DHS64']
    if len(choices) != 1:
        raise ValueError('A unique qualified DHS64 resource is required in the configured library')
    return lib.check(choices[0]), choices[0]


def arguments(values):
    values = dict(values)
    for key, value in values.items():
        if key.endswith('_path') or key == 'output_dir':
            values[key] = str(value); io.matrix.absolute_path(values[key])
        elif key.endswith('_sha256'):
            io.matrix.sha(value)
    if 'output_dir' not in values:
        raise ValueError('Missing managed output directory')
    return values


def records(args, *, context_bases=500, exact_length=False, max_records=10000):
    """Preserve original labels, order, source bytes and exact extraction semantics."""
    from Bio import SeqIO
    from Bio.Seq import Seq
    rows, sources, reference = [], [], None
    inline = args.get('sequence')
    path = args.get('sequence_path')
    if (inline is None) == (path is None):
        raise ValueError('Supply one exact DNA sequence or one FASTA/BED source')
    if inline is not None:
        if args.get('sequence_sha256') is not None:
            raise ValueError('Inline DNA does not accept a file identity')
        rows = [dict(original_header='inline sequence', original_sequence=inline, sequence=inline)]
    else:
        if Path(path).stat().st_size > 32 * 1024**2:
            raise ValueError('Sequence input exceeds the 32 MiB execution bound')
        sources = file_closure([path])
        if sources[0]['sha256'] != args.get('sequence_sha256'):
            raise ValueError('Sequence input changed from its captured identity')
        with open(path) as stream:
            first = next((line for line in stream if line.strip()), '')
        if first.startswith('>'):
            for r in SeqIO.parse(path, 'fasta'):
                rows.append(dict(original_header=r.description, original_sequence=str(r.seq), sequence=str(r.seq)))
                if len(rows) > max_records:
                    raise ValueError(f'Execution supports at most {max_records} records per request')
        else:
            import pandas as pd
            import pysam
            if not args.get('assembly'):
                raise ValueError('BED coordinates require an explicit genome assembly')
            lib = library()
            matches = [r for r in lib.entries() if r['kind'] == 'genome'
                       and r['resource']['assembly'] == args['assembly']]
            if len(matches) != 1:
                raise ValueError('BED requires a unique qualified genome for the declared assembly')
            genome = lib.check(matches[0])
            if genome['species']['taxonomy_id'] != 9606:
                raise ValueError('This model is qualified for human conditions; nonhuman BED input is not supported')
            reference = genome
            sources += matches[0]['files']
            table = pd.read_csv(path, sep='\t', comment='#', header=None, dtype=str, keep_default_na=False)
            if not 3 <= table.shape[1] <= 6 or not 0 < len(table) <= max_records:
                raise ValueError(f'Supply BED3 through BED6 contiguous intervals, at most {max_records} records; BED12 blocks are not contiguous sequences')
            with pysam.FastaFile(genome['genome']['fasta']['path'], filepath_index=genome['genome']['fai']['path']) as fasta:
                for i, r in table.iterrows():
                    chrom, start, end = r[0], int(r[1]), int(r[2])
                    strand = r[5] if len(r) >= 6 else '+'
                    if strand == '.': strand = '+'
                    if (strand not in ('+', '-') or chrom not in fasta.references
                            or not 0 <= start < end <= fasta.get_reference_length(chrom)):
                        raise ValueError('Invalid BED strand or out-of-bounds interval; no clipping or liftOver is performed')
                    if end - start > context_bases:
                        raise ValueError(f'This model accepts at most {context_bases} bases per interval; select an explicit compatible interval')
                    seq = fasta.fetch(chrom, start, end)
                    if strand == '-': seq = str(Seq(seq).reverse_complement())
                    rows.append(dict(original_header=r[3] if len(r) >= 4 else f'{chrom}:{start}-{end}',
                        original_sequence=seq, sequence=seq,
                        interval=dict(chrom=chrom, start=start, end=end, strand=strand,
                                      coordinates='0-based half-open; exact interval; no genomic expansion')))
    if not rows or len(rows) > max_records:
        raise ValueError(f'Supply between one and {max_records} sequence records')
    for i, row in enumerate(rows):
        seq = row['sequence'].upper()
        # One model-specific guard against the native encoder's silent truncation
        # and unknown-base zero vectors. Other future models own their own policy.
        if not (context_bases if exact_length else 1) <= len(seq) <= context_bases or set(seq) - set('ACGT'):
            raise ValueError(f'This model requires {context_bases if exact_length else str(1)+"–"+str(context_bases)} A/C/G/T bases; ambiguous bases need resolved sequence; inputs are never padded with invented flanks or truncated')
        row.update(sequence=seq, sequence_id=f'sequence-{i + 1}',
                   sequence_sha256=hashlib.sha256(seq.encode('ascii')).hexdigest(),
                   ordinal=i + 1, native_zero_padding=context_bases - len(seq))
    return rows, sorted(sources, key=lambda r: r['path']), reference


def native_predict(resource, rows, stage):
    from agent.tools.models import dhs64_runtime
    settings = resource['runtime']['settings']
    if (settings['device'] not in ('cpu', 'gpu') or any(type(settings[k]) is not int or settings[k] < 1
            for k in ('batch_size', 'intra_threads', 'inter_threads'))):
        raise ValueError('Invalid configured execution settings')
    request = stage / 'native.json'
    write(request, dict(source_root=resource['software']['source_root'], settings=settings,
        checkpoints=[r['path'] for r in resource['checkpoints']], sequences=[r['sequence'] for r in rows]))
    environment = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                       TF_CPP_MIN_LOG_LEVEL='2')
    environment.pop('PYTHONPATH', None); environment.pop('PYTHONHOME', None)
    with (stage / 'native.log').open('wb') as log:
        process = subprocess.Popen([resource['runtime']['python'], '-I', '-B', str(Path(dhs64_runtime.__file__).resolve()), str(request)],
            cwd=stage, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            while True:
                cancellation_checkpoint()
                try:
                    code = process.wait(timeout=.2); break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
    if code != 0:
        raise RuntimeError('Native DHS64 runtime failed; no model, device or objective was substituted')
    with np.load(request.with_suffix('.npz'), allow_pickle=False) as data:
        folds = data['folds']
    if (folds.shape != (3, 2, len(rows), 64) or not np.isfinite(folds).all()
            or np.any(folds[:, 1] < 0) or np.any(folds[:, 1] > 1)):
        raise ValueError('Native DHS64 head dimensions, finite values or sigmoid bounds differ')
    receipt = read(request.with_suffix('.receipt.json'))
    if receipt['completion'] != 'succeeded' or receipt['settings'] != settings:
        raise ValueError('Native completion differs from configured execution')
    return folds, receipt


def execute(values, execution_identity=None):
    args = arguments(values)
    resource, entry = resolve_model()
    head = args.get('head', 'continuous')
    if head not in MEASUREMENTS:
        raise ValueError('Choose continuous DNase signal or binary peak-call head')
    condition = args.get('condition')
    labels = resource['conditions']
    if condition is not None and condition not in labels:
        raise ValueError('Requested condition is not an exact DHS64 biosample; no substitute was selected')
    indices = [labels.index(condition)] if condition is not None else list(range(64))
    rows, sources, reference = records(args)
    root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Sequence output directory is not empty; cannot overwrite published or interrupted work')
    sources = sorted({r['path']: r for r in sources + entry['files']}.values(), key=lambda r:r['path'])
    with tempfile.TemporaryDirectory(prefix='.sequence-', dir=root) as temporary:
        stage = Path(temporary)
        folds, receipt = native_predict(resource, rows, stage)
        selected = folds[:, 0 if head == 'continuous' else 1, :, :][:, :, indices]
        mean, std = np.mean(selected, axis=0), np.std(selected, axis=0)
        scores = []
        for i, row in enumerate(rows):
            for j, index in enumerate(indices):
                scores.append(dict(sequence_id=row['sequence_id'], original_header=row['original_header'],
                    sequence_sha256=row['sequence_sha256'], condition_index=index, condition=labels[index],
                    measurement=MEASUREMENTS[head], mean=float(mean[i, j]), fold_std=float(std[i, j]),
                    **{f'fold_{fold}':float(selected[f, i, j]) for f, fold in enumerate((0, 1, 3))}))
        write(stage / 'sequence-records.json', rows)
        with (stage / 'sequences.fasta').open('x') as stream:
            for row in rows: stream.write(f">{row['original_header']}\n{row['sequence']}\n")
        with (stage / 'scores.tsv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS, delimiter='\t')
            writer.writeheader(); writer.writerows(scores)
        np.savez(stage / 'predictions.npz', folds=folds)
        summary = dict(status='success', contract_version=CONTRACT, n_sequences=len(rows),
            n_conditions=len(indices), conditions=[labels[i] for i in indices], head=head,
            measurement=MEASUREMENTS[head], folds=[0, 1, 3], score_min=float(mean.min()),
            score_max=float(mean.max()), score_examples=[{k:v for k,v in s.items() if k != 'original_header'} for s in scores[:12]],
            examples_omitted=max(0, len(scores)-12), model_release=resource['release'],
            normalization='Uppercase ACGT; native left placement followed by zero vectors to 500 positions; no truncation',
            limitations=LIMITATIONS)
        metadata = dict(summary, model={k:resource[k] for k in ('family', 'release', 'conditions', 'measurements',
            'context_bases', 'padding', 'training_assembly', 'training_domain', 'license', 'source')},
            checkpoints=[{k:v for k,v in r.items() if k != 'path'} for r in resource['checkpoints']],
            condition_metadata_sha256=resource['metadata']['sha256'],
            sequence_records=[dict(sequence_id=r['sequence_id'],sequence_sha256=r['sequence_sha256'],
                length=len(r['sequence']),native_zero_padding=r['native_zero_padding'],
                **({'interval':r['interval']} if 'interval' in r else {})) for r in rows],
            software={k:v for k,v in resource['software'].items() if k != 'source_root'}, runtime=receipt,
            source_sha256=args.get('sequence_sha256'), reference=None if reference is None else
            dict(assembly=reference['assembly'], fasta_sha256=reference['genome']['fasta']['sha256']),
            sequence_id_policy='Ordinal identity; original headers retained in private records and FASTA. Duplicate headers remain distinct.')
        write(stage / 'metadata.json', metadata)
        # A bounded useful figure is part of this publication, never fresh science.
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        fig = Figure(figsize=(max(5, min(16, len(indices)*.3)), min(10, 2+min(len(rows), 30)*.22)))
        ax = fig.subplots(); plot = ax.imshow(mean[:30], aspect='auto', cmap='viridis')
        ax.set_yticks(range(min(len(rows),30)), [r['sequence_id'] for r in rows[:30]])
        ax.set_xticks(range(len(indices)), [labels[i] for i in indices], rotation=90, fontsize=7)
        ax.set_title(f'DHS64 {head}: fold mean; first {min(len(rows),30)} sequences')
        fig.colorbar(plot, ax=ax, label=MEASUREMENTS[head]); fig.tight_layout()
        fig.savefig(stage / 'scores.png', dpi=120)
        cancellation_checkpoint()
        if file_closure(r['path'] for r in sources) != sources:
            raise AuthorityError('Sequence or selected scientific resource changed during prediction')
        names = ('sequence-records.json','sequences.fasta','scores.tsv','predictions.npz','metadata.json','scores.png')
        for name in names: os.link(stage / name, root / name)
    return publish(args, execution_identity, KIND, CONTRACT, summary, sources, names,
                   dict(model=entry, reference=reference, runtime=receipt))


def publication_type(kind):
    if kind in (KIND, EVAL_KIND):
        names = (('sequence-records.json','sequences.fasta','scores.tsv','predictions.npz','metadata.json','scores.png')
                 if kind == KIND else ('metrics.json',))
        return dict(contract=CONTRACT if kind == KIND else EVAL_CONTRACT, profile=PROFILE,
                    tool=TOOL if kind == KIND else EVALUATE, names=names,
                    outputs=(('sequences','sequences.fasta'),('scores','scores.tsv')) if kind == KIND else (('metrics','metrics.json'),))
    from . import sequence_optimization as optimization
    if kind != optimization.KIND:
        raise AuthorityError('Unknown sequence publication')
    return dict(contract=optimization.CONTRACT, profile=optimization.PROFILE, tool=optimization.TOOL,
                names=optimization.FILES, outputs=optimization.OUTPUTS)


def publish(args, execution_identity, kind, contract, summary, sources, names, provenance):
    root = Path(args['output_dir'])
    value = dict(contract=contract, kind=kind, profile=publication_type(kind)['profile'], arguments=args,
        execution_identity=execution_identity, result=summary,
        files=file_closure(root / name for name in names), sources=sources, provenance=provenance)
    path = root / 'manifest.json'; sha = write(path, value)
    if current() is not None: current().register_execution(kind, root, execution_identity)
    return verifier(kind)(path, expected_sha256=sha)


def load(path, sha=None):
    value = read(path, sha)
    definition = publication_type(value['kind'])
    if value['contract'] != definition['contract']:
        raise AuthorityError('Unknown sequence publication')
    if value['profile'] != definition['profile'] or Path(path) != Path(value['arguments']['output_dir']) / 'manifest.json':
        raise AuthorityError('Sequence publication identity differs')
    expected = set(definition['names'])
    if {f['path'] for f in value['files']} != {str(Path(path).parent / n) for n in expected} or len(value['files']) != len(expected):
        raise AuthorityError('Sequence publication membership differs')
    return value, file_closure([path])[0]['sha256']


def result(value, path, sha):
    pins = {Path(r['path']).name:r for r in value['files']}
    output = dict(value['result'], manifest_path=str(path), manifest_sha256=sha)
    for prefix, name in publication_type(value['kind'])['outputs']:
        output.update({prefix+'_path':pins[name]['path'], prefix+'_sha256':pins[name]['sha256']})
    return output


def _verify(path, sha, kind):
    value, actual = load(path, sha)
    if value['kind'] != kind or file_closure(r['path'] for r in value['files'] + value['sources']) != sorted(value['files'] + value['sources'], key=lambda r:r['path']):
        raise AuthorityError('Sequence publication or source integrity changed')
    # The native receipt is bound to this producing execution and exact artifacts.
    # No inference, extraction, or observed-data metrics are rerun on reads.
    if kind != EVAL_KIND and value['provenance']['runtime']['completion'] != 'succeeded':
        raise AuthorityError('Missing completed native prediction')
    return result(value, path, actual)


@owned_verification(KIND)
def verify_prediction(path, *, expected_sha256):
    return _verify(path, expected_sha256, KIND)


@owned_verification(EVAL_KIND)
def verify_evaluation(path, *, expected_sha256):
    return _verify(path, expected_sha256, EVAL_KIND)


def verifier(kind):
    if kind == KIND: return verify_prediction
    if kind == EVAL_KIND: return verify_evaluation
    from .sequence_optimization import verify_optimization
    publication_type(kind)
    return verify_optimization


def verify_result(args, output):
    value, sha = load(output['manifest_path'], output['manifest_sha256'])
    if value['arguments'] != arguments(args) or _serialize(output) != verifier(value['kind'])(output['manifest_path'], expected_sha256=sha):
        raise AuthorityError('Sequence result differs from its request')


def recover(args, execution_identity):
    args = arguments(args); path = Path(args['output_dir']) / 'manifest.json'
    value, sha = load(path)
    if value['arguments'] != args or value['execution_identity'] != execution_identity:
        raise AuthorityError('Sequence recovery identity differs; no native retry is authorized')
    return verifier(value['kind'])(path, expected_sha256=sha)


def predict_regulatory_sequences(output_dir, sequence_path=None, sequence_sha256=None, sequence=None,
                                  condition=None, head=None, assembly=None):
    return execute({k:v for k,v in locals().items() if v is not None})


def describe(path, sha, context):
    value, actual = load(path, sha)
    files = context.files([path, *(r['path'] for r in value['files'])])
    if [r for r in files if r['path'] != str(path)] != value['files']:
        raise AuthorityError('Sequence artifact integrity changed')
    identity = dict(kind=value['kind'], manifest=value, files=files, historical_sources=value['sources'],
        execution_identity=context.execution_for(value['kind'], path), profile=value['profile'], verifier=VERIFIER)
    return dict(kind=value['kind'], manifest_path=str(path), manifest_sha256=actual, manifest=value,
        proof_identity=identity, files=files, historical_sources=value['sources'], upstream={},
        profile=value['profile'], verifier=VERIFIER, qualification={})


def reuse_result(kind, description, proof, kwargs):
    return result(description['manifest'], description['manifest_path'], description['manifest_sha256'])


def proof_metadata(kind, output, context):
    return dict(output)


def publication_record(context, tool_name, args, output, execution_identity, result_metadata):
    value, sha = load(output['manifest_path'], output['manifest_sha256'])
    if (tool_name != publication_type(value['kind'])['tool']
            or value['execution_identity'] != execution_identity or value['arguments'] != arguments(args)
            or _serialize(result_metadata) != _serialize(output) or result(value, output['manifest_path'], sha) != _serialize(output)):
        raise AuthorityError('Sequence publication lacks its exact producing execution')
    description, key = context.describe(value['kind'], Path(output['manifest_path']), sha)
    return VerifiedArtifactAuthority(dict(schema_version=2, artifact_type='agent.'+value['kind'],
        artifact_contract=value['contract'], publication_path=output['manifest_path'], manifest_sha256=sha,
        files=description['files'], execution_identity=execution_identity, arguments_sha256=digest(arguments(args)),
        receipt_sha256=sha, upstream={'producer_inputs':digest(description['proof_identity'])},
        resources=dict(scientific_proof_sha256=key, manifest_identity=value, result_metadata=_serialize(output)),
        science_profile=value['profile'], producer_qualification={}, verifier=VERIFIER, scope='scientific_correctness.v1',
        completion='succeeded', integrity_scope='artifact_integrity_lineage_historical_sources.v1',
        source_policy='historical_verified_sources.v1', historical_sources=description['historical_sources']))


def execute_evaluation(values, execution_identity=None):
    args = arguments(values)
    prediction = verify_prediction(args['prediction_manifest_path'], expected_sha256=args['prediction_manifest_sha256'])
    truth_path = args['observations_path']
    sources = file_closure([args['prediction_manifest_path'], prediction['scores_path'], truth_path])
    if next(r['sha256'] for r in sources if r['path'] == truth_path) != args['observations_sha256']:
        raise ValueError('Observed measurement table changed')
    with open(prediction['scores_path'], newline='') as stream:
        scores = {(r['sequence_id'], r['condition']):r for r in csv.DictReader(stream, delimiter='\t')}
    with open(truth_path, newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not {'sequence_id','condition','measurement','value','provenance'} <= set(reader.fieldnames or ()):
            raise ValueError('Observed TSV needs exact sequence_id, condition, measurement, value and experiment provenance')
        observations = list(reader)
    if not observations or len(observations) > len(scores):
        raise ValueError('Observed table has no supported unique prediction correspondence')
    grouped, seen = {}, set()
    for row in observations:
        key = (row['sequence_id'], row['condition'])
        if key in seen or key not in scores or row['measurement'] != prediction['measurement'] or not row['provenance'].strip():
            raise ValueError('Observations require unique exact sequence/condition identity and compatible DNase measurement and scale; MPRA/ATAC are not DNase truth')
        if row.get('sequence_sha256') and row['sequence_sha256'] != scores[key]['sequence_sha256']:
            raise ValueError('Supplied sequence checksum differs from the explicitly selected prediction')
        value = float(row['value'])
        if not np.isfinite(value) or prediction['head'] == 'binary' and value not in (0, 1):
            raise ValueError('Observed values are invalid for the selected head')
        seen.add(key)
        grouped.setdefault(row['condition'], []).append((value, float(scores[key]['mean'])))
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
    metrics = []
    for condition, pairs in grouped.items():
        y, p = np.asarray(pairs).T
        row = dict(condition=condition, n=len(y))
        if prediction['head'] == 'continuous':
            variable = len(y) >= 2 and np.ptp(y) > 0 and np.ptp(p) > 0
            row.update(rmse=float(np.sqrt(np.mean((y-p)**2))), mae=float(np.mean(np.abs(y-p))),
                pearson=float(pearsonr(y,p).statistic) if variable else None,
                spearman=float(spearmanr(y,p).statistic) if variable else None)
        else:
            both = len(np.unique(y)) == 2
            row.update(auroc=float(roc_auc_score(y,p)) if both else None,
                average_precision=float(average_precision_score(y,p)) if both else None,
                brier=float(brier_score_loss(y,p)), log_loss=float(log_loss(y,p,labels=[0,1])))
        metrics.append(row)
    limitations = ('Evaluation uses only supplied matched observations and their declared exact model measurement/transform. '
        'It does not independently establish experiment provenance, absence of training overlap, calibration, '
        'biological validation or independence of overlapping sequences. Undefined correlations or single-class '
        'discrimination metrics are null. No cross-assay conversion or label approximation is performed.')
    summary = dict(status='success', contract_version=EVAL_CONTRACT, n_observations=len(observations),
        measurement=prediction['measurement'], metrics=metrics, limitations=limitations)
    root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()): raise ValueError('Evaluation output already exists')
    write(root / 'metrics.json', summary)
    if file_closure(r['path'] for r in sources) != sources: raise AuthorityError('Evaluation inputs changed')
    cancellation_checkpoint()
    return publish(args, execution_identity, EVAL_KIND, EVAL_CONTRACT, summary, sources, ('metrics.json',),
                   dict(prediction_manifest_sha256=args['prediction_manifest_sha256']))


def evaluate_regulatory_predictions(prediction_manifest_path, prediction_manifest_sha256,
                                    observations_path, observations_sha256, output_dir):
    return execute_evaluation(locals())
