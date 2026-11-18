"""Native single-population MACS3 FRAG calling; no peak or counting algorithm."""
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError, VerifiedArtifactAuthority, digest
from agent.tools.data.authority_context import current, owned_verification
from agent.tools._file_identity import file_closure
from .sequence_prediction import read, write, arguments, library

TOOL = 'call_scATAC_peaks'
KIND = 'scatac_peaks'
LEGACY_CONTRACT = 'macs3-scatac-peaks.v1'
CONTRACT = 'macs3-scatac-peaks.v2'
POLICY = 'macs3-native-fragment-population-v2'
LEGACY_POLICY = 'macs3-frag-unit-record-v1'
LEGACY_PROFILE = digest(dict(contract=LEGACY_CONTRACT, native='MACS3 3.1.0', format='FRAG', max_count=1,
    population='one namespace', coordinates='already Tn5-adjusted; no shift'))
PROFILE = digest(dict(contract=CONTRACT, native='MACS3 3.1.0', format='FRAG', max_count=1,
    population='selected native fragment-backed H5AD', export='snapatac2.ex.export_fragments',
    coordinates='already Tn5-adjusted; no shift'))
VERIFIER = dict(id='agent.macs3-native-completion', compatibility_version='1')
LIMITATIONS = ('Experimental pooled-population enrichment peaks, not replicate-level differential accessibility, '
    'cell-type truth or biological validation. Native p/q fields are -log10 probabilities; narrowPeak score '
    'is a native display score, not a fragment count. Source support is not independent molecule count. '
    'The selected H5AD defines the pooled population; its QC and donor history remain upstream evidence. '
    'No second Tn5 shift, coverage truncation, threshold relaxation or weighted-support inference is performed. '
    'Native fragment storage does not establish original read-pair duplicate support.')
FILES = ('peaks_peaks.narrowPeak', 'peaks_summits.bed', 'peaks_peaks.xls', 'features.bed',
         'distributions.png', 'metadata.json', 'native-receipt.json')
FACT_FIELDS = ('n_peaks','n_barcodes','n_observed_barcodes','n_fragment_records','total_source_support','population',
    'species','assembly','contigs','parameters','native_statistics','source_identity','limitations')
RESULT_FIELDS = dict(status=(str,),contract_version=(str,),manifest_path=(str,),manifest_sha256=(str,),
    peaks_path=(str,),peaks_sha256=(str,),cells_path=(str,),cells_sha256=(str,),
    n_peaks=(int,),n_barcodes=(int,),n_observed_barcodes=(int,),n_fragment_records=(int,),
    total_source_support=(type(None),),population=(str,),species=(str,dict,type(None)),assembly=(str,type(None)),
    contigs=(list,),parameters=(dict,),native_statistics=(dict,),source_identity=(dict,),limitations=(str,))


def parameter_prerequisites(inputs):
    return tuple(k for k in ('effective_genome_size','genome_size_basis') if inputs.get(k) is None)


def settings(args, contigs):
    size = args.get('effective_genome_size')
    if isinstance(size,bool) or not isinstance(size,(int,float)) or not math.isfinite(size) or not 1 <= size <= sum(contigs.values()):
        raise ValueError('Effective genome size must be positive and no larger than the bound genome; total FASTA length is not an effective-size estimate')
    basis = args.get('genome_size_basis')
    if not isinstance(basis,str) or not basis.strip():
        raise ValueError('Declare the scientific basis/applicability of the effective genome size')
    q = args.get('qvalue',0.05)
    if isinstance(q,bool) or not isinstance(q,(int,float)) or not math.isfinite(q) or not 0 < q < 1:
        raise ValueError('Native peak-calling q-value cutoff must be strictly between zero and one')
    excluded = args.get('exclude_contigs',[])
    if not isinstance(excluded,list) or len(set(excluded)) != len(excluded) or any(c not in contigs for c in excluded):
        raise ValueError('Excluded contigs must be distinct exact names in the source genome')
    return dict(format='FRAG',max_count=1,count_policy='one native exported fragment record',effective_genome_size=size,
        genome_size_basis=basis,qvalue=q,exclude_contigs=excluded,shift=0,paired_fragment_mode=True,
        local_background=True,control='none',version='3.1.0',threads=2,process_backend='auto')


def prepare_fragments(args,stage):
    """Use native fragment storage directly; no retired canonical import contract."""
    from . import peak_h5ad
    from .native_callable import executable
    from agent.tools._native_process import run
    source=file_closure([args['input_path']])[0]
    if args.get('input_sha256') is not None and source['sha256']!=args['input_sha256']:
        raise AuthorityError('Selected fragment-backed H5AD changed')
    peak_h5ad.inspect(args['input_path'],require_matrix=False,max_logical_bytes=None)
    worker=Path(__file__).with_name('macs3_fragments_worker.py')
    python=executable('snapatac2')
    implementation=file_closure([worker,python.resolve()])
    request=dict(input_path=args['input_path'],stage=str(stage),exclude_contigs=args.get('exclude_contigs',[]))
    write(stage/'export-request.json',request)
    code=run([str(python),'-I',str(worker),str(stage/'export-request.json')],stage/'export.log',environment=dict(os.environ))
    if code:
        shutil.copyfile(stage/'export.log',stage.parent/'native-export-failure.log')
        raise RuntimeError('Native SnapATAC2 fragment export failed; diagnostics retained')
    exported=read(stage/'export-receipt.json');contigs=exported['contigs']
    parameters=settings(args,contigs)
    summary=dict(n_barcodes=exported['n_cells'],n_observed_barcodes=exported['n_observed_barcodes'],
        n_fragment_records=exported['n_fragment_records'],total_source_support=None,
        population='pooled exact selected native H5AD cells',source_identity=source,
        species=exported['declarations'].get('species'),assembly=exported['declarations'].get('assembly'),
        contigs=exported['analyzed_contigs'],parameters=parameters)
    lineage=dict(source=source,export=exported,native_input=file_closure([stage/'input.frag'])[0],
        preparation='native export; original coordinates and barcode IDs; explicit contig exclusions only; no second shift')
    return contigs,summary,lineage,[source,*implementation]


def native(parameters,stage):
    executable=Path(os.environ.get('AGENT_MACS3_EXECUTABLE') or library().root.parent/'runtimes/macs3/env/bin/macs3').resolve()
    if not executable.is_file():raise ValueError('Provision the optional isolated MACS3 3.1.0 runtime')
    runtime=executable.parent.parent.parent/'runtime.json'
    pin=read(runtime)
    from agent.tools.data.scatac_fragments_v2_verifier import take_snapshots, check_snapshots
    snapshots=take_snapshots([runtime,*(r['path'] for r in pin['files'])])
    if (pin['version']!='3.1.0' or str(executable) not in {p['path'] for p in pin['files']}
            or file_closure(p['path'] for p in pin['files'])!=pin['files']):
        raise AuthorityError('Pinned MACS3 runtime changed')
    version=subprocess.run([str(executable),'--version'],capture_output=True,text=True,check=True).stdout.strip()
    if version!='macs3 3.1.0':raise ValueError('MACS3 release differs from the qualified 3.1.0 runtime')
    command=[str(executable),'callpeak','-t',str(stage/'input.frag'),'-f','FRAG','--max-count','1',
        '--threads','2','--process-backend','auto','-g',str(parameters['effective_genome_size']),'-q',str(parameters['qvalue']),'-n','peaks','--outdir',str(stage)]
    start=time.monotonic()
    from agent.tools._native_process import run
    code = run(command, stage/'native.log', environment=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    if code:
        shutil.copyfile(stage/'native.log',stage.parent/'native-failure.log')
        raise RuntimeError('Native MACS3 failed; diagnostics retained, no fallback or threshold change')
    check_snapshots(snapshots)
    return dict(completion='succeeded',version=version,command=command,parameters=parameters,
        elapsed_seconds=time.monotonic()-start,runtime=dict(version=pin['version'],python=pin['python'],identity=file_closure([runtime])[0]),log_sha256=file_closure([stage/'native.log'])[0]['sha256'])


def inspect_peaks(stage,contigs):
    """Check native output coordinates/fields; never recompute enrichment statistics."""
    rows=[];ids=set()
    with (stage/'peaks_peaks.narrowPeak').open() as stream:
        for line in stream:
            fields=line.rstrip('\n').split('\t')
            if len(fields)!=10:raise AuthorityError('Invalid native narrowPeak record')
            chrom,start,end,name,score,strand,fold,p,q,summit=fields
            start,end,score,summit=map(int,(start,end,score,summit));fold,p,q=map(float,(fold,p,q))
            if (chrom not in contigs or not 0<=start<end<=contigs[chrom] or name in ids or strand!='.'
                or not all(math.isfinite(v) and v>=0 for v in (fold,p,q)) or score<0 or not 0<=summit<end-start):
                raise AuthorityError('Native peaks contain invalid coordinates or statistics')
            ids.add(name);rows.append((chrom,start,end,name,score,fold,p,q,summit))
            if len(rows)>2_000_000:raise ValueError('Peak vocabulary exceeds the existing feature-reference bound')
    summits=(stage/'peaks_summits.bed').read_text().splitlines()
    if len(summits)!=len(rows):raise AuthorityError('Native summit count differs from narrowPeak')
    for row,line in zip(rows,summits):
        f=line.split('\t')
        if len(f)!=5 or f[:4]!=[row[0],str(row[1]+row[8]),str(row[1]+row[8]+1),row[3]]:
            raise AuthorityError('Native summit coordinates/identity differ from narrowPeak')
    # XLS explicitly uses 1-based closed positions, including the absolute summit.
    xls=[l for l in (stage/'peaks_peaks.xls').read_text().splitlines() if l and not l.startswith('#')]
    if not xls or len(xls)-1!=len(rows):raise AuthorityError('Native XLS count differs from narrowPeak')
    for row,line in zip(rows,xls[1:]):
        f=line.split('\t')
        if f[:5]!=[row[0],str(row[1]+1),str(row[2]),str(row[2]-row[1]),str(row[1]+row[8]+1)] or f[-1]!=row[3]:
            raise AuthorityError('Native XLS one-based coordinates differ from narrowPeak')
    return rows


def execute(values,execution_identity=None):
    args=arguments(values);root=Path(args['output_dir']);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise ValueError('Output directory is not empty; recovery does not rerun science')
    with tempfile.TemporaryDirectory(prefix='.macs3-',dir=root) as temp:
        stage=Path(temp);contigs,summary,lineage,sources=prepare_fragments(args,stage)
        receipt=native(summary['parameters'],stage);rows=inspect_peaks(stage,contigs)
        for name in FILES[:3]:shutil.copyfile(stage/name,root/name)
        shutil.copyfile(stage/'cells.tsv',root/'cells.tsv')
        if (stage/'native.log').exists():shutil.copyfile(stage/'native.log',root/'native.log')
        with (root/'features.bed').open('w') as stream:
            for chrom,start,end,*_ in rows:stream.write(f'{chrom}\t{start}\t{end}\t{chrom}:{start}-{end}\n')
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].hist([r[2]-r[1] for r in rows],bins=50);axes[0].set_xlabel('Native peak length (bp)')
        axes[1].hist([r[5] for r in rows],bins=50);axes[1].set_xlabel('Native fold enrichment')
        for ax in axes:ax.set_ylabel('Peaks')
        fig.suptitle(f'MACS3: {len(rows):,} experimental peaks');fig.tight_layout();fig.savefig(root/'distributions.png',dpi=150);plt.close(fig)
        summary.update(status='succeeded',contract_version=CONTRACT,n_peaks=len(rows),limitations=LIMITATIONS,
            native_statistics=dict(narrowPeak_coordinates='zero-based half-open',xls_coordinates='one-based closed',
                probability_fields='-log10(p) and -log10(q)',score='native display score; not counts',
                empty=not bool(rows),feature_order='original native narrowPeak order; coordinate IDs'))
        write(root/'native-receipt.json',receipt)
        write(root/'metadata.json',dict(summary=summary,lineage=lineage))
        pins={p['path']:p for p in sources}
        if file_closure(pins)!=sorted(pins.values(),key=lambda p:p['path']):raise AuthorityError('Sources changed during native execution')
    files=[root/name for name in (*FILES,'cells.tsv')]
    if (root/'native.log').exists():files.append(root/'native.log')
    value=dict(contract=CONTRACT,profile=PROFILE,arguments=args,execution_identity=execution_identity,result=summary,
        lineage=lineage,files=file_closure(files),sources=sorted(pins.values(),key=lambda p:p['path']),native_completion='succeeded')
    for output in files:
        with output.open('rb') as stream:os.fsync(stream.fileno())
    path=root/'manifest.json';sha=write(path,value)
    from agent.tools.data.scatac_qc_reference import _fsync_dir
    _fsync_dir(root)
    if current() is not None:current().register_execution(KIND,root,execution_identity)
    return verify_publication(path,expected_sha256=sha)


def call_scATAC_peaks(*,input_path,effective_genome_size,genome_size_basis,output_dir,
    input_sha256=None,qvalue=None,exclude_contigs=None):
    return execute({k:v for k,v in locals().items() if v is not None})


def load(path,sha=None):
    value=read(path,sha);root=Path(path).parent
    # Historical v1 publications retain their exact files and profile; no old execution path.
    legacy=value['contract']==LEGACY_CONTRACT
    expected={str(root/name) for name in FILES}|({str(root/'population/manifest.json'),str(root/'population/cells.tsv.gz')} if legacy else {str(root/'cells.tsv')})
    if legacy and value['result']['n_peaks']:expected.add(str(root/'reference.json'))
    if any(r['path']==str(root/'native.log') for r in value['files']):expected.add(str(root/'native.log'))
    if (value['contract'] not in (CONTRACT,LEGACY_CONTRACT) or value['profile']!=(LEGACY_PROFILE if legacy else PROFILE) or root!=Path(value['arguments']['output_dir'])
            or Path(path).name!='manifest.json' or value['native_completion']!='succeeded'
            or {r['path'] for r in value['files']}!=expected or len(value['files'])!=len(expected)):
        raise AuthorityError('Invalid peak publication identity or membership')
    return value,file_closure([path])[0]['sha256']


def result(value,path,sha):
    if value['contract']==LEGACY_CONTRACT:
        return dict(value['result'],manifest_path=str(path),manifest_sha256=sha,
            reference_manifest_path=value['reference']['manifest_path'],reference_manifest_sha256=value['reference']['manifest_sha256'],
            cells_manifest_path=value['cells']['manifest_path'],cells_manifest_sha256=value['cells']['manifest_sha256'])
    root=Path(path).parent
    pins={r['path']:r['sha256'] for r in value['files']}
    return dict(value['result'],manifest_path=str(path),manifest_sha256=sha,
        peaks_path=str(root/'features.bed'),peaks_sha256=pins[str(root/'features.bed')],
        cells_path=str(root/'cells.tsv'),cells_sha256=pins[str(root/'cells.tsv')])


@owned_verification(KIND)
def verify_publication(path,*,expected_sha256):
    value,sha=load(path,expected_sha256)
    if file_closure(r['path'] for r in value['files']+value['sources'])!=sorted(value['files']+value['sources'],key=lambda r:r['path']):
        raise AuthorityError('Peak publication or source integrity changed')
    return result(value,path,sha)


def verify_result(args,output):
    value,sha=load(output['manifest_path'],output['manifest_sha256'])
    if value['arguments']!=arguments(args) or _serialize(output)!=verify_publication(output['manifest_path'],expected_sha256=sha):
        raise AuthorityError('Peak result differs from its producing request')


def recover(args,execution_identity):
    args=arguments(args);path=Path(args['output_dir'])/'manifest.json';value,sha=load(path)
    if value['arguments']!=args or value['execution_identity']!=execution_identity:raise AuthorityError('Recovery cannot authorize a native retry')
    return verify_publication(path,expected_sha256=sha)


def describe(path,sha,context):
    value,actual=load(path,sha);files=context.files([path,*(r['path'] for r in value['files'])])
    if [r for r in files if r['path']!=str(path)]!=value['files']:raise AuthorityError('Peak artifact integrity changed')
    identity=dict(kind=KIND,manifest=value,files=files,historical_sources=value['sources'],upstream={},
        execution_identity=context.execution_for(KIND,path),profile=value['profile'],verifier=VERIFIER,qualification={})
    return dict(kind=KIND,manifest_path=str(path),manifest_sha256=actual,manifest=value,proof_identity=identity,
        files=files,historical_sources=value['sources'],upstream={},profile=value['profile'],verifier=VERIFIER,qualification={})


def reuse_result(kind,description,proof,kwargs):return result(description['manifest'],description['manifest_path'],description['manifest_sha256'])
def proof_metadata(kind,output,context):return dict(output)


def publication_record(context,tool_name,args,output,execution_identity,result_metadata):
    value,sha=load(output['manifest_path'],output['manifest_sha256'])
    if (tool_name!=TOOL or value['execution_identity']!=execution_identity or value['arguments']!=arguments(args)
            or _serialize(result_metadata)!=_serialize(output) or result(value,output['manifest_path'],sha)!=_serialize(output)):
        raise AuthorityError('Peak publication lacks its exact producing execution')
    description,key=context.describe(KIND,Path(output['manifest_path']),sha)
    return VerifiedArtifactAuthority(dict(schema_version=2,artifact_type='agent.'+KIND,artifact_contract=value['contract'],
        publication_path=output['manifest_path'],manifest_sha256=sha,files=description['files'],execution_identity=execution_identity,
        arguments_sha256=digest(arguments(args)),receipt_sha256=sha,upstream={'producer_inputs':digest(description['proof_identity'])},
        resources=dict(scientific_proof_sha256=key,manifest_identity=value,result_metadata=_serialize(output)),science_profile=value['profile'],
        producer_qualification={},verifier=VERIFIER,scope='scientific_correctness.v1',completion='succeeded',
        integrity_scope='artifact_integrity_lineage_historical_sources.v1',source_policy='historical_verified_sources.v1',historical_sources=description['historical_sources']))


DELIVERY_FILES = {
    'peaks_peaks.narrowPeak': ('Native MACS3 narrowPeak', 'text/plain; charset=utf-8'),
    'peaks_summits.bed': ('Native peak summits', 'text/plain; charset=utf-8'),
    'peaks_peaks.xls': ('Native peak statistics (private command paths omitted)', 'text/tab-separated-values; charset=utf-8'),
    'features.bed': ('Exact ordered experimental peak features', 'text/plain; charset=utf-8'),
    'cells.tsv': ('Exact ordered native peak-calling population', 'text/tab-separated-values; charset=utf-8'),
    'distributions.png': ('Peak length and enrichment distributions', 'image/png'),
    'metadata.json': ('Peak parameters, population and provenance', 'application/json'),
    'native-receipt.json': ('Native MACS3 completion and runtime identity', 'application/json'),
}


def prepare_operation(arguments, context):
    """Selected scientific owner exposes typed prerequisites on the actual input."""
    from agent.orchestration.scientific_inputs import parameter_requirements, ScientificInputRequired
    observed_args = dict(arguments)
    source = context.existing(arguments.get('input_path'))
    if source is not None:
        observed_args['input_path'] = source.path
    names = parameter_prerequisites(observed_args)
    needs = parameter_requirements(context.tool, names)
    if needs:
        raise ScientificInputRequired(*needs)
    return arguments
