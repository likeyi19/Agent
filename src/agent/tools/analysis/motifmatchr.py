"""Bounded matchMotifs invocation on ordered genomic peaks, without count analysis."""
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy import sparse
from scipy.io import mmread

from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError, VerifiedArtifactAuthority, digest
from agent.tools._cancellation import cancellation_checkpoint
from agent.tools.data.authority_context import current, owned_verification
from agent.tools.data.fastq_verification_authority import file_closure
from .sequence_prediction import read, write, library, arguments as _arguments
from . import chromvar, peak_h5ad

TOOL = 'run_motifmatchr'
KIND = 'native_motif_matches'
CONTRACT = 'motifmatchr-matches.v1'
POLICY = 'motifmatchr-native-v1'
PROFILE = digest(dict(contract=CONTRACT, function='motifmatchr::matchMotifs',
    subject='ordered GRanges; all supplied peaks', output='native matches'))
VERIFIER = dict(id='agent.motifmatchr-native-completion', compatibility_version='1')
FILES = ('matches.mtx','motif-matches.npz','peaks.tsv','motifs.tsv','metadata.json','native-receipt.json')
NATIVE_ARGUMENTS = (
    ('p.cutoff', 'Native p-value cutoff for motif matching.'),
    ('bg', 'Native nucleotide background: subject, genome, even or an explicit frequency vector.'),
    ('out', 'Native output form; only matches has a publication representation. Other explicit forms must not be replaced.'),
)
LIMITATIONS = ('Sequence motif matches are not chromVAR deviations, enrichment, TF expression or binding, '
    'causality or experimental validation. All supplied peaks are scanned in original order, including '
    'zero-count peaks. Counts and cells are not analyzed. Omitted optional arguments use the installed '
    'native method defaults; only matches output is supported.')
FACT_FIELDS = ('function_name','n_peaks','n_motifs','n_matches','assembly','motif_collection',
    'parameters','measurement','ordering','software_versions','limitations')
RESULT_FIELDS = dict(status=(str,),contract_version=(str,),manifest_path=(str,),manifest_sha256=(str,),
    matches_path=(str,),matches_sha256=(str,),function_name=(str,),n_peaks=(int,),n_motifs=(int,),n_matches=(int,),
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
    if 'out' in kwargs and kwargs['out'] != 'matches':
        raise ValueError('Only native out="matches" has a publication representation; requested output is not substituted')
    return kwargs


def declarations(path, args):
    with h5py.File(path,'r') as handle:
        var = read_elem(handle['var'])
        effective = chromvar._metadata_declarations(handle,'X',args,var)
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
    from .scatac_regulatory_sequences import _coordinates, _check_declarations
    from .replicate_pseudobulk import _strict_ids
    peak_h5ad.inspect(args['input_path'],require_matrix=False,max_logical_bytes=None)
    var,effective = declarations(args['input_path'],args)
    _check_declarations(args['input_path'],effective)
    features = _strict_ids(var.index,'feature')
    if not features or any(any(c in s for c in '\t\r\n') for s in features):
        raise ValueError('Nonempty unique peak labels without tabs/newlines are required for native exchange')
    effective.setdefault('coordinate_source','var_columns' if {'chrom','start','end'}<=set(var) else 'feature_names')
    intervals,_ = _coordinates(var,effective)
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
    environment = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',R_ENVIRON_USER='/dev/null',R_PROFILE_USER='/dev/null')
    for key in ('R_LIBS_USER','R_LIBS','R_LIBS_SITE'):
        environment.pop(key,None)
    with (stage/'native.log').open('wb') as log:
        process = subprocess.Popen([rscript,'--vanilla',str(script),str(stage/'request.json')],env=environment,
            stdin=subprocess.DEVNULL,stdout=log,stderr=log)
        try:
            while True:
                cancellation_checkpoint()
                try:
                    code = process.wait(timeout=.2);break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
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
        fasta,genome,motif,entry,sources = chromvar.resolve_resources(effective,peaks,stage)
        receipt = native(effective,stage,fasta,motif)
        motifs = pd.read_csv(stage/'motifs.tsv',sep='\t',keep_default_na=False)
        matches = mmread(stage/'matches.mtx').tocsc()
        if (matches.shape!=(len(peaks),len(motifs)) or receipt['dim']!=list(matches.shape)
                or receipt['nnz']!=matches.nnz or not np.all(matches.data==1)
                or receipt['rows_preserved'] is not True or receipt['assays']!='motifMatches'):
            raise AuthorityError('Native match matrix, axes or completion dimensions differ')
        # Native colnames, accession/version and TF name are retained separately.
        metadata = pd.read_csv(motif['metadata']['path'],sep='\t',keep_default_na=False)
        annotations = metadata.set_index('matrix_id').loc[motifs.motif_id]
        for key in annotations:
            if key not in motifs:
                motifs[key] = annotations[key].to_numpy()
        motifs.to_csv(root/'motifs.tsv',sep='\t',index=False)
        for name in ('peaks.tsv','matches.mtx'):
            shutil.copyfile(stage/name,root/name)
        sparse.save_npz(root/'motif-matches.npz',matches.astype(bool))
        collection = dict(resource_id=entry['id'],release=motif['release'],collection=motif['collection'],
            tax_group=motif['tax_group'],citation=motif['citation'],license=motif['license'])
        summary = dict(status='success',contract_version=CONTRACT,function_name='motifmatchr::matchMotifs',
            n_peaks=len(peaks),n_motifs=len(motifs),n_matches=matches.nnz,assembly=genome['assembly'],
            motif_collection=collection,parameters=dict(kwargs=kwargs,omitted_optional_arguments='native defaults'),
            measurement='native boolean sequence motif matches',ordering='all source peaks × native motif order',
            software_versions=receipt['versions'],limitations=LIMITATIONS)
        write(root/'native-receipt.json',receipt)
        write(root/'metadata.json',dict(summary=summary,source=source,genome=genome,motif=motif,
            coordinate_system='zero_based_half_open',native_coordinates='GRanges: start + 1, end unchanged',
            receipt=receipt))
        expected = {r['path']:r for r in [source,*sources,*receipt['implementation']]}
        sources = file_closure(expected)
        if sources!=sorted(expected.values(),key=lambda r:r['path']):
            raise AuthorityError('Input or reference changed during native invocation')
    value = dict(contract=CONTRACT,profile=PROFILE,arguments=args,execution_identity=execution_identity,
        result=summary,files=file_closure(root/name for name in FILES),sources=sources,native_completion='succeeded')
    path = root/'manifest.json';sha = write(path,value)
    if current() is not None:
        current().register_execution(KIND,root,execution_identity)
    return verify_publication(path,expected_sha256=sha)


def run_motifmatchr(input_path,output_dir,function_name,kwargs=None,input_sha256=None,assembly=None,
    coordinate_source=None,coordinate_system=None,feature_chrom_key=None,feature_start_key=None,feature_end_key=None,
    genome_path=None,genome_sha256=None,genome_resource=None,taxonomy_id=None,motif_collection=None):
    return execute({k:v for k,v in locals().items() if v is not None})


def load(path,sha=None):
    value = read(path,sha);root = Path(path).parent
    if (value['contract']!=CONTRACT or value['profile']!=PROFILE or root!=Path(value['arguments']['output_dir'])
            or Path(path).name!='manifest.json' or value['native_completion']!='succeeded'
            or {r['path'] for r in value['files']}!={str(root/name) for name in FILES} or len(value['files'])!=len(FILES)):
        raise AuthorityError('Invalid native motif publication identity or membership')
    return value,file_closure([path])[0]['sha256']


def result(value,path,sha):
    pin = next(r for r in value['files'] if Path(r['path']).name=='motif-matches.npz')
    return dict(value['result'],manifest_path=str(path),manifest_sha256=sha,matches_path=pin['path'],matches_sha256=pin['sha256'])


@owned_verification(KIND)
def verify_publication(path,*,expected_sha256):
    value,sha = load(path,expected_sha256)
    if file_closure(r['path'] for r in value['files']+value['sources'])!=sorted(value['files']+value['sources'],key=lambda r:r['path']):
        raise AuthorityError('Native motif publication or source integrity changed')
    return result(value,path,sha)


def verify_result(args,output):
    value,sha = load(output['manifest_path'],output['manifest_sha256'])
    if value['arguments']!=arguments(args) or _serialize(output)!=verify_publication(output['manifest_path'],expected_sha256=sha):
        raise AuthorityError('Native motif result differs from its producing request')


def recover(args,execution_identity):
    args = arguments(args);path = Path(args['output_dir'])/'manifest.json';value,sha = load(path)
    if value['arguments']!=args or value['execution_identity']!=execution_identity:
        raise AuthorityError('Recovery cannot authorize a native retry')
    return verify_publication(path,expected_sha256=sha)


def describe(path,sha,context):
    value,actual = load(path,sha);files = context.files([path,*(r['path'] for r in value['files'])])
    if [r for r in files if r['path']!=str(path)]!=value['files']:
        raise AuthorityError('Native motif artifact integrity changed')
    identity = dict(kind=KIND,manifest=value,files=files,historical_sources=value['sources'],upstream={},
        execution_identity=context.execution_for(KIND,path),profile=PROFILE,verifier=VERIFIER,qualification={})
    return dict(kind=KIND,manifest_path=str(path),manifest_sha256=actual,manifest=value,proof_identity=identity,
        files=files,historical_sources=value['sources'],upstream={},profile=PROFILE,verifier=VERIFIER,qualification={})


def reuse_result(kind,description,proof,kwargs):
    return result(description['manifest'],description['manifest_path'],description['manifest_sha256'])


def proof_metadata(kind,output,context):
    return dict(output)


def publication_record(context,tool_name,args,output,execution_identity,result_metadata):
    value,sha = load(output['manifest_path'],output['manifest_sha256'])
    if (tool_name!=TOOL or value['execution_identity']!=execution_identity or value['arguments']!=arguments(args)
            or _serialize(result_metadata)!=_serialize(output) or result(value,output['manifest_path'],sha)!=_serialize(output)):
        raise AuthorityError('Native motif publication lacks its exact producing execution')
    description,key = context.describe(KIND,Path(output['manifest_path']),sha)
    return VerifiedArtifactAuthority(dict(schema_version=2,artifact_type='agent.'+KIND,artifact_contract=CONTRACT,
        publication_path=output['manifest_path'],manifest_sha256=sha,files=description['files'],execution_identity=execution_identity,
        arguments_sha256=digest(arguments(args)),receipt_sha256=sha,upstream={'producer_inputs':digest(description['proof_identity'])},
        resources=dict(scientific_proof_sha256=key,manifest_identity=value,result_metadata=_serialize(output)),science_profile=PROFILE,
        producer_qualification={},verifier=VERIFIER,scope='scientific_correctness.v1',completion='succeeded',
        integrity_scope='artifact_integrity_lineage_historical_sources.v1',source_policy='historical_verified_sources.v1',historical_sources=description['historical_sources']))
