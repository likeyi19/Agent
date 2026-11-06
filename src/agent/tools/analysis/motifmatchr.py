"""Bounded matchMotifs invocation on ordered genomic peaks, without count analysis."""
import math
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy import sparse
from scipy.io import mmread

from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools.data.authority_context import current
from agent.tools._file_identity import file_closure
from .sequence_prediction import read, write, library, arguments as _arguments
from . import sequence_reference, peak_h5ad

TOOL = 'run_motifmatchr'
KIND = 'native_motif_matches'
CONTRACT = 'motifmatchr-matches.v1'
POLICY = 'motifmatchr-native-v1'
HISTORICAL_PROFILE = digest(dict(contract=CONTRACT, function='motifmatchr::matchMotifs',
    subject='ordered GRanges; all supplied peaks', output='native matches'))
PROFILE = digest(dict(contract=CONTRACT, adaptation='native matches, scores and genomic positions.v2'))
VERIFIER = dict(id='agent.motifmatchr-native-completion', compatibility_version='1')
FILES = ('matches.mtx','motif-matches.npz','peaks.tsv','motifs.tsv','metadata.json','native-receipt.json')
NATIVE_ARGUMENTS = (
    ('p.cutoff', 'Native p-value cutoff for motif matching.'),
    ('bg', 'Native nucleotide background: subject, genome, even or an explicit frequency vector.'),
    ('out', 'Native output form; matches, scores or positions; preserve the requested native representation.'),
)
LIMITATIONS = ('Sequence motif matches are not chromVAR deviations, enrichment, TF expression or binding, '
    'causality or experimental validation. All supplied peaks are scanned in original order, including '
    'zero-count peaks. Counts and cells are not analyzed. Omitted optional arguments use the installed '
    'native method defaults. Scores include native matches, scores and match counts; positions are genomic 1-based closed ranges with strands and scores, not a boolean matrix.')
FACT_FIELDS = ('function_name','n_peaks','n_motifs','n_matches','assembly','motif_collection',
    'parameters','measurement','ordering','software_versions','limitations')
RESULT_FIELDS = dict(status=(str,),contract_version=(str,),manifest_path=(str,),manifest_sha256=(str,),
    matches_path=(str,type(None)),matches_sha256=(str,type(None)),
    function_name=(str,),n_peaks=(int,),n_motifs=(int,),n_matches=(int,type(None)),
    assembly=(str,),motif_collection=(dict,),parameters=(dict,),measurement=(str,),ordering=(str,),
    software_versions=(dict,),limitations=(str,))


class NativeInvocationError(ValueError):
    """A bounded, locator-free native condition; the full log remains private."""


def arguments(values):
    return _arguments(_serialize(values))


def native_arguments(args):
    if args.get('function_name') != 'matchMotifs':
        raise ValueError('Only motifmatchr::matchMotifs is supported; no other function is substituted')
    kwargs = args.get('kwargs', {})
    if type(kwargs) is not dict or set(kwargs) - {name for name, _ in NATIVE_ARGUMENTS}:
        raise ValueError('Supported native kwargs are p.cutoff, bg and out; input objects and genome come from authorized sources')
    def scalar(value):
        return (value is None or type(value) in (bool,int) or
            type(value) is float and math.isfinite(value) or
            type(value) is str and len(value) <= 128 and not any(ord(c)<32 for c in value))
    for key,value in kwargs.items():
        vector = key == 'bg' and type(value) in (dict,list) and len(value) <= 4
        if vector:
            if type(value) is dict and (set(value)-set('ACGT')):
                raise ValueError('A named native background vector uses A, C, G and T keys')
            values = value.values() if type(value) is dict else value
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
                raise ValueError('Native background vectors require finite JSON numbers')
        elif not scalar(value):
            raise ValueError('Native kwargs require bounded JSON scalar data or a background frequency vector')
    if 'out' in kwargs and kwargs['out'] not in ('matches', 'scores', 'positions'):
        raise ValueError('Supported native outputs are matches, scores and positions; requested output is not substituted')
    return kwargs


def declarations(path, args):
    with h5py.File(path,'r') as handle:
        var = read_elem(handle['var'])
        effective = sequence_reference.metadata_declarations(handle,'X',args,var)
    # Matrix measurements belong to count consumers, not genomic ranges.
    effective.pop('matrix_semantics',None)
    return var,effective


def parameter_prerequisites(inputs):
    path = inputs.get('input_path')
    if not isinstance(path,(str,Path)):
        return ()
    _,effective = declarations(path,inputs)
    missing = tuple(key for key in ('assembly','coordinate_system') if not effective.get(key))
    if missing:
        return missing
    # Catalog discovery only. Execution still checks exact closure, coordinates
    # and compatibility through the existing resource owner.
    entries = library().entries()
    taxonomy = effective.get('taxonomy_id')
    if effective.get('genome_path'):
        if taxonomy is None:
            return ('taxonomy_id',)
    else:
        genomes = [e for e in entries if e['kind']=='genome' and e['resource']['assembly']==effective['assembly']
            and (not effective.get('genome_resource') or effective['genome_resource'] in (e['id'],e['label']))]
        if len(genomes)!=1:
            return ('genome_resource',)
        taxonomy = genomes[0]['resource']['species']['taxonomy_id']
    motifs = [e for e in entries if e['kind']=='motif_collection' and taxonomy in e['resource']['applicable_taxonomy_ids']
        and (not effective.get('motif_collection') or effective['motif_collection'] in (e['id'],e['label']))]
    return () if len(motifs)==1 else ('motif_collection',)


def prepare_input(args,stage):
    from .sequence_reference import coordinates, check_declarations
    from .replicate_pseudobulk import _strict_ids
    peak_h5ad.inspect(args['input_path'],require_matrix=False,max_logical_bytes=None)
    var,effective = declarations(args['input_path'],args)
    check_declarations(args['input_path'],effective)
    features = _strict_ids(var.index,'feature')
    if not features or any(any(c in s for c in '\t\r\n') for s in features):
        raise ValueError('Nonempty unique peak labels without tabs/newlines are required for native exchange')
    effective.setdefault('coordinate_source','var_columns' if {'chrom','start','end'}<=set(var) else 'feature_names')
    intervals,_ = coordinates(var,effective)
    peaks = pd.DataFrame(intervals,columns=['chrom','start','end'])
    peaks.insert(0,'feature_id',features);peaks.insert(0,'source_index',np.arange(len(var)))
    peaks.to_csv(stage/'peaks.tsv',sep='\t',index=False)
    return effective,peaks


def native(args,stage,fasta,motif):
    kwargs = native_arguments(args)
    rscript = os.environ.get('AGENT_CHROMVAR_RSCRIPT',str(library().root.parent/'runtimes/chromvar/env/bin/Rscript'))
    if not Path(rscript).is_file():
        raise ValueError('The operator must provision the existing optional motifmatchr R runtime')
    script = Path(__file__).parent/'r/motifmatchr_v1.R'
    implementation = file_closure([rscript,script])
    request = dict(stage=str(stage),function_name=args['function_name'],fasta=fasta,motifs=motif['pfm']['path'],kwargs=kwargs)
    request_sha = write(stage/'request.json',request)
    from agent.tools._native_process import run, r_environment
    code = run([rscript, '--vanilla', str(script), str(stage/'request.json')],
        stage/'native.log', environment=r_environment())
    if code:
        shutil.copyfile(stage/'native.log',stage.parent/'native-failure.log')
        diagnostic = 'See the retained private runtime diagnostic; no arguments were changed'
        if (stage/'native-error.json').is_file():
            diagnostic = read(stage/'native-error.json')['message'][:2000]
            # Native conditions may quote a bound reference path. Public errors
            # keep useful package wording while internal locators stay private.
            for path in (fasta,motif['pfm']['path'],str(stage),rscript,str(script)):
                diagnostic = diagnostic.replace(path,'[bound resource]')
        if peak_h5ad._PATH.search(diagnostic) or any(ord(c)<32 for c in diagnostic):
            diagnostic = 'The native condition contains private source information; consult the retained private diagnostic.'
        raise NativeInvocationError(diagnostic)
    receipt = read(stage/'native-receipt.json')
    if (receipt['completion']!='succeeded' or receipt['request_sha256']!=request_sha
            or receipt['function_name']!=args['function_name'] or receipt['chromVAR_loaded']):
        raise AuthorityError('Native completion does not identify the requested matchMotifs invocation')
    if file_closure([rscript,script])!=implementation:
        raise AuthorityError('Native implementation changed during execution')
    receipt['implementation'] = implementation
    return receipt


def execute(values,execution_identity=None):
    args = arguments(values);kwargs = native_arguments(args)
    root = Path(args['output_dir']);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Output directory is not empty; native retry requires a new execution')
    source = file_closure([args['input_path']])[0]
    if args.get('input_sha256') and source['sha256']!=args['input_sha256']:
        raise AuthorityError('Selected peak source bytes changed')
    with tempfile.TemporaryDirectory(prefix='.matchmotifs-',dir=root) as temp:
        stage = Path(temp)
        effective,peaks = prepare_input(args,stage)
        fasta,genome,motif,entry,sources = sequence_reference.resolve_motif_resources(effective,peaks,stage)
        receipt = native(effective,stage,fasta,motif)
        motifs = pd.read_csv(stage/'motifs.tsv',sep='\t',keep_default_na=False)
        mode = kwargs.get('out', 'matches')
        if mode == 'positions':
            positions = pd.read_csv(stage/'positions.tsv', sep='\t', keep_default_na=False)
            if (receipt['n_positions'] != len(positions)
                    or list(positions) != ['motif_index', 'native_name', 'seqnames', 'start', 'end', 'width', 'strand', 'score']
                    or not positions.motif_index.between(0, len(motifs)-1).all()
                    or not (positions.end - positions.start + 1 == positions.width).all()):
                raise AuthorityError('Native genomic positions differ from completion or motif axes')
            n_matches = None
            measurement = 'native genomic motif positions with strand and score; 1-based closed'
        else:
            matches = mmread(stage/'matches.mtx').tocsc()
            assays = 'motifMatches' if mode == 'matches' else ['motifMatches', 'motifScores', 'motifCounts']
            if (matches.shape != (len(peaks), len(motifs)) or receipt['dim'] != list(matches.shape)
                    or receipt['nnz'] != matches.nnz or not np.all(matches.data == 1)
                    or receipt['rows_preserved'] is not True or (receipt['assays'] != assays if mode == 'matches' else set(receipt['assays']) != set(assays))):
                raise AuthorityError('Native match matrix, axes or completion dimensions differ')
            sparse.save_npz(root/'motif-matches.npz', matches.astype(bool))
            n_matches = matches.nnz
            measurement = 'native boolean sequence motif matches'
            if mode == 'scores':
                for name in ('scores.mtx', 'counts.mtx'):
                    if mmread(stage/name).shape != matches.shape:
                        raise AuthorityError('Native score/count axes differ from motif matches')
                measurement = 'native motif matches, maximum match scores and match counts'
        # Accessions/version and TF labels remain distinct, in native motif order.
        metadata = pd.read_csv(motif['metadata']['path'], sep='\t', keep_default_na=False)
        annotations = metadata.set_index('matrix_id').loc[motifs.motif_id]
        for key in annotations:
            if key not in motifs:
                motifs[key] = annotations[key].to_numpy()
        motifs.to_csv(root/'motifs.tsv', sep='\t', index=False)
        for name in publication_files(dict(arguments=args)):
            if name not in ('motifs.tsv', 'metadata.json', 'native-receipt.json', 'motif-matches.npz'):
                shutil.copyfile(stage/name, root/name)
        collection = dict(resource_id=entry['id'],release=motif['release'],collection=motif['collection'],
            tax_group=motif['tax_group'],citation=motif['citation'],license=motif['license'])
        summary = dict(status='success',contract_version=CONTRACT,function_name='motifmatchr::matchMotifs',
            n_peaks=len(peaks),n_motifs=len(motifs),n_matches=n_matches,assembly=genome['assembly'],
            motif_collection=collection,parameters=dict(kwargs=kwargs,omitted_optional_arguments='native defaults'),
            measurement=measurement,ordering=('native motif order then native genomic-hit order; peaks.tsv retains source peak order' if mode == 'positions' else 'all source peaks × native motif order'),
            software_versions=receipt['versions'],limitations=LIMITATIONS)
        if mode == 'positions':
            summary['n_positions'] = len(positions)
        write(root/'native-receipt.json',receipt)
        write(root/'metadata.json',dict(summary=summary,source=source,genome=genome,motif=motif,
            coordinate_system='zero_based_half_open',native_coordinates='GRanges: start + 1, end unchanged',
            receipt=receipt))
        expected = {r['path']:r for r in [source,*sources,*receipt['implementation']]}
        sources = file_closure(expected)
        if sources!=sorted(expected.values(),key=lambda r:r['path']):
            raise AuthorityError('Input or reference changed during native invocation')
    value = dict(contract=CONTRACT,profile=PROFILE,arguments=args,execution_identity=execution_identity,
        result=summary,files=file_closure(root/name for name in publication_files(dict(arguments=args))),sources=sources,native_completion='succeeded')
    path = root/'manifest.json';sha = write(path,value)
    if current() is not None:
        current().register_execution(KIND,root,execution_identity)
    return verify_publication(path,expected_sha256=sha)


def run_motifmatchr(input_path,output_dir,function_name,kwargs=None,input_sha256=None,assembly=None,
    coordinate_source=None,coordinate_system=None,feature_chrom_key=None,feature_start_key=None,feature_end_key=None,
    genome_path=None,genome_sha256=None,genome_resource=None,taxonomy_id=None,motif_collection=None):
    return execute({k:v for k,v in locals().items() if v is not None})


def publication_files(value):
    mode = value['arguments'].get('kwargs', {}).get('out', 'matches')
    if value.get('profile') == HISTORICAL_PROFILE or mode == 'matches':
        return FILES
    common = ('peaks.tsv', 'motifs.tsv', 'metadata.json', 'native-receipt.json', 'native-result.rds')
    if mode == 'scores':
        return common + ('matches.mtx', 'motif-matches.npz', 'scores.mtx', 'counts.mtx')
    if mode == 'positions':
        return common + ('positions.tsv',)
    raise AuthorityError('Unrepresented native motif output')


from .native_publication import NativePublication
publication = NativePublication(tool=TOOL, kind=KIND, contract=CONTRACT, profile=PROFILE, verifier=VERIFIER,
    files=publication_files, arguments=arguments, historical_profiles=(HISTORICAL_PROFILE,),
    outputs=lambda v: {'matches': None if v['arguments'].get('kwargs', {}).get('out') == 'positions' else 'motif-matches.npz'})
load = publication.load
result = publication.result
verify_publication = publication.verify_publication
verify_result = publication.verify_result
recover = publication.recover
describe = publication.describe
reuse_result = publication.reuse_result
proof_metadata = publication.proof_metadata
publication_record = publication.publication_record


DELIVERY_FILES = {
    'native-result.rds': ('Exact native R result, including ranges and attributes', 'application/octet-stream'),
    'positions.tsv': ('Native motif hits: 1-based closed genomic ranges, strand and score', 'text/tab-separated-values'),
    'scores.mtx': ('Native peak-by-motif maximum match scores', 'application/octet-stream'),
    'counts.mtx': ('Native peak-by-motif match counts', 'application/octet-stream'),
    'matches.mtx': ('Native peak-by-motif matches (Matrix Market)', 'application/octet-stream'),
    'motif-matches.npz': ('Native peak-by-motif matches (SciPy sparse)', 'application/octet-stream'),
    'peaks.tsv': ('All original ordered genomic peaks', 'text/tab-separated-values; charset=utf-8'),
    'motifs.tsv': ('Native ordered motif names and versioned annotations', 'text/tab-separated-values; charset=utf-8'),
    'metadata.json': ('Native motif matching source, parameters and resources', 'application/json'),
    'native-receipt.json': ('Native matchMotifs completion and software versions', 'application/json'),
}
