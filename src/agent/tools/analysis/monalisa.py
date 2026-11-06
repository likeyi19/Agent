"""Native monaLisa sequence-bin enrichment; no inferred differential analysis."""
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import pandas as pd
import pysam
from anndata.io import read_elem

from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools._native_process import run, r_environment
from agent.tools.data.authority_context import current
from agent.tools._file_identity import file_closure
from . import motifmatchr, peak_h5ad, sequence_reference
from .native_matrix import arguments, native_kwargs, NativeInvocationError
from .native_publication import NativePublication
from .sequence_prediction import library, read, write

package = 'monaLisa'
TOOL = 'run_monalisa'
KIND = 'native_monalisa_enrichment'
CONTRACT = 'native-monalisa-enrichment.v1'
POLICY = 'native-monalisa-enrichment-v1'
PROFILE = digest(dict(contract=CONTRACT, version='1.16.0', representation='DNAStringSet, exact factor bins, native PWMatrixList and SummarizedExperiment.v1'))
VERIFIER = dict(id='agent.' + KIND, compatibility_version='1')
guidance = 'monaLisa calcBinnedMotifEnrR: native motif enrichment across supplied genomic sequence bins, with sequence-composition correction; not motif matching, chromVAR or topic modeling.'
NATIVE_ARGUMENTS = (
    ('background', 'Native otherBins, allBins or zeroBin; genome background requires an unimplemented sampling contract.'),
    ('test', 'Native one-sided fisher or binomial enrichment test.'),
    ('maxFracN', 'Native maximum sequence N fraction; native exclusions remain recorded.'),
    ('maxKmerSize', 'Native sequence composition correction, default 3; execution cap 4.'),
    ('min.score', 'Native motif hit threshold, default 10.'),
    ('matchMethod', 'Native motif matching method.'),
    ('GCbreaks', 'Native GC-bin boundaries.'),
    ('pseudocount.log2enr', 'Native log2 enrichment pseudocount, default 8.'),
    ('p.adjust.method', 'Native multiple-testing correction, default BH.'),
    ('verbose', 'Native diagnostic verbosity.'),
)
FUNCTIONS = {'calcBinnedMotifEnrR': tuple(k for k, _ in NATIVE_ARGUMENTS)}
FACT_FIELDS = ('function_name', 'n_sequences', 'n_motifs', 'n_bins', 'parameters', 'source_identity', 'software_versions', 'measurement', 'limitations')
RESULT_FIELDS = dict(status=(str,), contract_version=(str,), manifest_path=(str,), manifest_sha256=(str,),
    table_path=(str,), table_sha256=(str,), function_name=(str,), n_sequences=(int,), n_motifs=(int,), n_bins=(int,),
    parameters=(dict,), source_identity=(dict,), software_versions=(dict,), measurement=(str,), limitations=(str,))
DELIVERY_FILES = {
    'result.tsv': ('Native motif-by-bin assays, including negative-log10 adjusted significance', 'text/tab-separated-values'),
    'native-result.rds': ('Exact native SummarizedExperiment; download only', 'application/octet-stream'),
    'sequences.fa': ('Exact reference sequences in source feature order', 'text/plain'),
    'bins.tsv': ('Exact source features and supplied sequence bins', 'text/tab-separated-values'),
    'motifs.tsv': ('Native motif accessions and labels', 'text/tab-separated-values'),
    'metadata.json': ('Scientific inputs, parameters and output semantics', 'application/json'),
    'native-receipt.json': ('Native completion and version evidence', 'application/json'),
}
LIMITATIONS = ('Native sequence-bin motif enrichment, not TF binding, expression, causality, topic modeling or biological validation. '
    'Bins are supplied scientific labels, not inferred differential accessibility. Reference sequences retain exact source order. '
    'Native background weighting, exclusions, undefined statistics and adjusted significance remain in the SummarizedExperiment. '
    'No native figure is generated. RDS is a download, never an executable uploaded input.')


def native_arguments(args):
    kw = native_kwargs(FUNCTIONS, args)
    if kw.get('background') == 'genome':
        raise ValueError('monaLisa genome-background sampling is not supported; no alternative background is substituted')
    if type(kw.get('maxKmerSize')) in (int, float) and kw['maxKmerSize'] > 4:
        raise ValueError('maxKmerSize exceeds the execution bound 4')
    return kw


def parameter_prerequisites(args):
    missing = () if args.get('bin_key') or args.get('bins_path') else ('bin_key',)
    return missing + motifmatchr.parameter_prerequisites(args)


def prepare(args, stage):
    effective, peaks = motifmatchr.prepare_input(args, stage)
    if len(peaks) > 100_000 or (peaks.end - peaks.start).sum() > 100_000_000:
        raise ValueError('monaLisa sequence input exceeds 100000 regions or 100 million bases')
    if args.get('bin_key') and args.get('bins_path'):
        raise ValueError('Select one exact bin field or bin table, not both')
    if args.get('bins_path'):
        if Path(args['bins_path']).stat().st_size > 32 * 1024**2:
            raise ValueError('Sequence bin table exceeds 32 MiB')
        bins = pd.read_csv(args['bins_path'], sep='\t', dtype=str, keep_default_na=False)
        if list(bins) != ['feature_id', 'bin'] or bins.feature_id.tolist() != peaks.feature_id.tolist():
            raise ValueError('Bin table must contain feature_id and bin in exact source feature order; no silent alignment')
        levels = sorted(set(bins['bin']))
    else:
        with h5py.File(args['input_path'], 'r') as h:
            column = read_elem(h['var'])[args['bin_key']]
        if column.isna().any():
            raise ValueError('Every source sequence requires its supplied bin')
        bins = pd.DataFrame({'feature_id': peaks.feature_id, 'bin': column.astype(str).to_numpy()})
        levels = list(map(str, column.cat.categories)) if isinstance(column.dtype, pd.CategoricalDtype) else sorted(set(bins['bin']))
    if (not 1 <= len(levels) <= 128 or any(not v or len(v) > 128 or any(c in v for c in '\t\r\n') for v in levels)):
        raise ValueError('Native sequence bins require bounded nonempty labels')
    bins.to_csv(stage / 'bins.tsv', sep='\t', index=False)
    fasta_path, genome, motif, entry, resources = sequence_reference.resolve_motif_resources(effective, peaks, stage)
    with pysam.FastaFile(fasta_path) as fasta, (stage / 'sequences.fa').open('w') as out:
        for p in peaks.itertuples():
            # Zero-based half-open coordinates were qualified by the shared owner.
            out.write('>' + p.feature_id + '\n' + sequence_reference.extract(fasta, p.chrom, int(p.start), int(p.end)) + '\n')
    return levels, genome, motif, entry, resources


def execute(values, execution_identity=None):
    args = arguments(values); kw = native_arguments(args)
    missing = parameter_prerequisites(args)
    if missing:
        raise ValueError('Missing actual enrichment inputs: ' + ', '.join(missing))
    root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Native output must be empty; no retry authority')
    sources = file_closure(args[k] for k in ('input_path', 'bins_path') if args.get(k))
    for prefix in ('input', 'bins'):
        if args.get(prefix + '_sha256') and next(p['sha256'] for p in sources if p['path'] == args[prefix + '_path']) != args[prefix + '_sha256']:
            raise AuthorityError('Selected enrichment source changed')
    executable = Path(os.environ.get('AGENT_MONALISA_RSCRIPT') or library().root.parent / 'runtimes/archr-1/env/bin/Rscript')
    script = Path(__file__).parent / 'r/monalisa_v1.R'
    implementation = file_closure([executable, script])
    with tempfile.TemporaryDirectory(prefix='.monalisa-', dir=root) as temp:
        stage = Path(temp); levels, genome, motif, entry, resources = prepare(args, stage)
        sha = write(stage / 'request.json', dict(stage=str(stage), function_name=args['function_name'], kwargs=kw, levels=levels, motifs=motif['pfm']['path']))
        code = run([str(executable), '--vanilla', str(script), str(stage / 'request.json')], stage / 'native.log', environment=r_environment())
        shutil.copyfile(stage / 'native.log', root / ('native-failure.log' if code else 'native.log'))
        if code:
            message = read(stage / 'native-error.json').get('message', '')[:2000] if (stage / 'native-error.json').exists() else ''
            if not message or peak_h5ad._PATH.search(message) or any(ord(c) < 32 for c in message):
                message = 'Native monaLisa failed; exact diagnostic retained privately'
            raise NativeInvocationError(message)
        receipt = read(stage / 'native-receipt.json')
        if receipt['request_sha256'] != sha or receipt['completion'] != 'succeeded' or receipt['function_name'] != args['function_name'] or receipt['versions']['monaLisa'] != '1.16.0':
            raise AuthorityError('monaLisa completion differs from requested invocation')
        table = pd.read_csv(stage / 'result.tsv', sep='\t')
        if len(table) != receipt['n_motifs'] * receipt['n_bins']:
            raise AuthorityError('Native motif/bin result dimensions differ')
        receipt['implementation'] = implementation
        summary = dict(status='success', contract_version=CONTRACT, function_name='monaLisa::' + args['function_name'],
            n_sequences=receipt['n_sequences'], n_motifs=receipt['n_motifs'], n_bins=receipt['n_bins'],
            parameters=dict(kwargs=kw, bin_levels=levels, bin_key=args.get('bin_key'), omitted_optional_arguments='native defaults', assays=receipt['assays']),
            source_identity=dict(inputs=[{k: p[k] for k in ('sha256', 'size_bytes')} for p in sources], genome=genome, motif_collection=entry['id']),
            software_versions=receipt['versions'], measurement='native motif-by-bin enrichment assays; negLog10Padj is negative log10 adjusted P, not a probability', limitations=LIMITATIONS)
        for name in DELIVERY_FILES:
            if name not in ('metadata.json', 'native-receipt.json'):
                shutil.copyfile(stage / name, root / name)
        write(root / 'metadata.json', dict(summary=summary, sources=sources, motif=motif, receipt=receipt))
        write(root / 'native-receipt.json', receipt)
    expected = sorted({p['path']: p for p in sources + resources + implementation}.values(), key=lambda p: p['path'])
    if file_closure(p['path'] for p in expected) != expected:
        raise AuthorityError('Source, reference or native runtime changed')
    value = dict(contract=CONTRACT, profile=PROFILE, arguments=args, execution_identity=execution_identity,
        result=summary, files=file_closure(root / n for n in DELIVERY_FILES), sources=expected, native_completion='succeeded')
    path = root / 'manifest.json'; sha = write(path, value)
    if current() is not None:
        current().register_execution(KIND, root, execution_identity)
    return verify_publication(path, expected_sha256=sha)


def invoke(input_path, output_dir, function_name, kwargs=None, input_sha256=None, bin_key=None, bins_path=None,
           bins_sha256=None, assembly=None, coordinate_system=None, coordinate_source=None, genome_resource=None,
           motif_collection=None, taxonomy_id=None, genome_path=None, genome_sha256=None,
           feature_chrom_key=None, feature_start_key=None, feature_end_key=None):
    return execute({k: v for k, v in locals().items() if v is not None})


publication = NativePublication(tool=TOOL, kind=KIND, contract=CONTRACT, profile=PROFILE, verifier=VERIFIER,
    files=tuple(DELIVERY_FILES), arguments=arguments, outputs={'table': 'result.tsv'})
load = publication.load
result = publication.result
verify_publication = publication.verify_publication
verify_result = publication.verify_result
recover = publication.recover
