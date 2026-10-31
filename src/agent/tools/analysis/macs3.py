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
from agent.tools._cancellation import cancellation_checkpoint
from agent.tools.data.authority_context import current, owned_verification
from agent.tools.data.fastq_verification_authority import file_closure
from agent.tools.data import explicit_cells
from .sequence_prediction import read, write, arguments, library

TOOL = 'call_scATAC_peaks'
KIND = 'scatac_peaks'
CONTRACT = 'macs3-scatac-peaks.v1'
POLICY = 'macs3-frag-unit-record-v1'
PROFILE = digest(dict(contract=CONTRACT, native='MACS3 3.1.0', format='FRAG', max_count=1,
    population='one namespace', coordinates='already Tn5-adjusted; no shift'))
VERIFIER = dict(id='agent.macs3-native-completion', compatibility_version='1')
LIMITATIONS = ('Experimental pooled-population enrichment peaks, not replicate-level differential accessibility, '
    'cell-type truth or biological validation. Native p/q fields are -log10 probabilities; narrowPeak score '
    'is a native display score, not a fragment count. Source support is not independent molecule count. '
    'Whole-library barcode identities are not QC-called cells. No second Tn5 shift, coverage truncation, '
    'threshold relaxation, biological-replicate pooling or weighted-support inference is performed.')
FILES = ('peaks_peaks.narrowPeak', 'peaks_summits.bed', 'peaks_peaks.xls', 'features.bed',
         'distributions.png', 'metadata.json', 'native-receipt.json')
FACT_FIELDS = ('n_peaks','n_barcodes','n_fragment_records','total_source_support','namespace','population',
    'species','assembly','contigs','parameters','native_statistics','limitations')
RESULT_FIELDS = dict(status=(str,), contract_version=(str,),manifest_path=(str,),manifest_sha256=(str,),
    reference_manifest_path=(str,),reference_manifest_sha256=(str,),
    cells_manifest_path=(str,),cells_manifest_sha256=(str,),n_peaks=(int,),n_barcodes=(int,),
    n_fragment_records=(int,),total_source_support=(int,),namespace=(str,),population=(str,),species=(dict,),
    assembly=(str,),contigs=(list,),parameters=(dict,),native_statistics=(dict,),limitations=(str,))


def parameter_prerequisites(inputs):
    missing=[k for k in ('effective_genome_size', 'genome_size_basis') if inputs.get(k) is None]
    path,sha=inputs.get('fragments_manifest_path'),inputs.get('fragments_manifest_sha256')
    if isinstance(path,(str,Path)) and isinstance(sha,str) and inputs.get('library_namespace') is None:
        from agent.tools.data.scatac_fragments_v2 import load_fragments_manifest_v2
        value=load_fragments_manifest_v2(path,expected_sha256=sha)
        if len(value['libraries'])>1:missing.append('library_namespace')
    return tuple(missing)


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
    return dict(format='FRAG',max_count=1,count_policy='one canonical record',effective_genome_size=size,
        genome_size_basis=basis,qvalue=q,exclude_contigs=excluded,shift=0,paired_fragment_mode=True,
        local_background=True,control='none',version='3.1.0',threads=2,process_backend='auto')


def grouped_population(args, fragments):
    """Reuse accepted clustering/source-axis binding; no equal-size inference."""
    import h5py
    from anndata.io import read_elem
    from agent.tools.data.cell_by_ccre_verifier import verify_cell_by_ccre
    from agent.tools.data.scatac_matrix import _load, executable
    from .embedding_analysis import source_bound_cluster_rows
    keys=('matrix_manifest_path','matrix_manifest_sha256','group_analysis_path','group_analysis_sha256','group_label')
    if not all(k in args for k in keys):raise ValueError('A selected group needs its exact count-matrix publication, clustering and group label')
    path=args['matrix_manifest_path'];sha=args['matrix_manifest_sha256']
    verify_cell_by_ccre(path,expected_sha256=sha,bedtools_path=executable())
    value=_load(Path(path),sha);fp=value['upstream']['fragments']
    if any(fp[k]!=args['fragments_'+k] for k in ('manifest_path','manifest_sha256')):
        raise AuthorityError('Grouping matrix belongs to different source fragments')
    matrix=Path(path).parent/value['matrix']['path'];pin=file_closure([matrix])[0]
    if pin['sha256']!=value['matrix']['sha256']:raise AuthorityError('Grouping count matrix changed')
    with h5py.File(matrix,'r') as f:obs=read_elem(f['obs'])
    if not {'namespace','barcode_identifier'}<=set(obs):raise ValueError('Grouping matrix lacks exact namespace/barcode identities')
    rows=source_bound_cluster_rows(obs.index,args['group_analysis_path'],args['group_analysis_sha256'],pin['sha256'])
    selected=[i for i,(_,label) in enumerate(rows) if label==args['group_label']]
    if not selected:raise ValueError('Selected accepted group is empty or absent')
    return list(zip(obs.iloc[selected]['namespace'].astype(str),obs.iloc[selected]['barcode_identifier'].astype(str))),file_closure([path,matrix,args['group_analysis_path']])


def selected_population(args, fragments):
    """Read the existing cell contracts or an ordinary one-barcode-per-line list."""
    from agent.tools.data._fragment_feature_binding import cell_input
    from agent.tools.data.cell_selection_verifier import verify_cell_selection
    candidates = [k for k in ('selected_cells_manifest_path','explicit_cells_manifest_path','barcodes_path','group_analysis_path') if args.get(k)]
    if len(candidates)>1: raise ValueError('Choose one population: QC-selected cells, explicit cells or supplied barcodes')
    ns = args.get('library_namespace')
    available = {l.namespace for l in fragments.libraries}
    if ns is None and len(available)==1: ns=next(iter(available))
    if ns not in available: raise ValueError('Select exactly one existing library namespace; barcodes alone cannot pool libraries')
    pairs=None; sources=[]; label='complete eligible library; observed barcodes, no QC claim'
    if candidates:
        key=candidates[0]; path=Path(args[key]);sha=args.get(key.replace('_path','_sha256'))
        if not sha: raise ValueError('Population source requires its captured checksum')
        sources=file_closure([path])
        if sources[0]['sha256']!=sha: raise AuthorityError('Population source changed')
        if key=='group_analysis_path':
            pairs,group_sources=grouped_population(args,fragments);sources+=group_sources
            label='exact source-bound accepted cell group '+args['group_label']
        elif key=='barcodes_path':
            if path.stat().st_size>64*1024**2: raise ValueError('Barcode list exceeds the 64 MiB preparation bound')
            rows=path.read_text().splitlines()
            if any(not b or '\t' in b or ' ' in b for b in rows):raise ValueError('Supply one exact barcode per line for the selected namespace')
            pairs=[(ns,b) for b in rows];label='explicit supplied barcode population; no QC claim'
        else:
            declared,contract=cell_input(dict(manifest_path=str(path),manifest_sha256=sha))
            if contract.CELL_CONTRACT==explicit_cells.CONTRACT:
                explicit_cells.verify_explicit_cells(path,expected_sha256=sha);table=path.parent/'cells.tsv.gz'
                label='explicit supplied ordered cells; no QC claim'
            else:
                declared=verify_cell_selection(path,expected_sha256=sha).to_dict();table=path.parent/'selected.tsv.gz'
                bound=declared['qc_lineage']['arguments']
                if any(bound[k]!=args[k] for k in ('fragments_manifest_path','fragments_manifest_sha256')):
                    raise AuthorityError('QC-selected population belongs to different fragments')
                label='accepted source-bound QC-selected cells'
            sources+=file_closure([table]);pairs=[(n,b) for _,n,b,_ in explicit_cells.iter_rows(table)]
        if not pairs or len(pairs)!=len(set(pairs)) or any(n!=ns for n,b in pairs):
            raise ValueError('Population must be nonempty, unique and confined to one exact namespace')
    return ns,pairs,label,sources


def prepare_fragments(args, stage, root):
    from agent.tools.data.scatac_fragment_reader import qualified_fragments
    from agent.tools.data._fragment_reference import load as load_reference
    from agent.tools.data.scatac_reference import species_identity
    fragments,qualification=qualified_fragments(args['fragments_manifest_path'],args['fragments_manifest_sha256'])
    reference=fragments.manifest['reference'];ref=load_reference(reference['manifest_path'],reference['manifest_sha256'])
    contigs=dict(fragments.contigs)
    parameters=settings(args,contigs)
    ns,pairs,label,sources=selected_population(args,fragments)
    selected=None if pairs is None else {b for _,b in pairs};observed=set();n=0;support=0;analyzed=set()
    path=stage/'input.frag'
    with path.open('w') as stream:
        for i,row in enumerate(fragments.iter_fragments(ns)):
            if i%4096==0:cancellation_checkpoint()
            if selected is not None and row.barcode_identifier not in selected:continue
            observed.add(row.barcode_identifier)
            if len(observed)>2_000_000:raise ValueError('Population exceeds the 2 million barcode preparation bound')
            if row.contig in parameters['exclude_contigs']:continue
            # MACS3's parser uses bounded native counts. Unit projection avoids
            # overflow for legitimate uint64 support, without biological reweighting.
            stream.write(f'{row.contig}\t{row.start}\t{row.end}\t{row.barcode_identifier}\t1\n')
            n+=1;support+=row.support;analyzed.add(row.contig)
    if not n or selected is not None and selected != observed:raise ValueError('Empty eligible population or requested barcodes absent from exact source')
    pairs=pairs if pairs is not None else [(ns,b) for b in sorted(observed)]
    cells=explicit_cells.publish_explicit_cells(cells=pairs,declaration=label,output_dir=root/'population')
    fpath=Path(args['fragments_manifest_path'])
    paths=[fpath,reference['manifest_path'],ref.genome.fasta.path,ref.genome.fai.path]
    for item in fragments.manifest['libraries']:
        paths.extend(fpath.parent/item[k]['path'] for k in ('bgzf','tabix'))
        paths.append(item['provenance']['producer_record']['path'])
    sources+=file_closure(paths)
    fragments.verification.check_unchanged()
    species=reference['species'] if isinstance(reference['species'],dict) else species_identity(reference['species'])
    summary=dict(n_barcodes=len(pairs),n_fragment_records=n,total_source_support=support,namespace=ns,
        population=label,species=species,assembly=reference['assembly'],contigs=[c for c in contigs if c in analyzed],parameters=parameters)
    lineage=dict(fragments_manifest_sha256=args['fragments_manifest_sha256'],
        fragments_identity=fragments.manifest['fragments_identity_sha256'],producer=qualification,
        library=next(l for l in fragments.manifest['libraries'] if l['namespace']==ns),cells=cells,
        native_input=file_closure([path])[0],preparation='exact selected records; fifth column unit projection; no shift or contig alias')
    return ref,contigs,summary,lineage,sources,cells


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
    with (stage/'native.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),start_new_session=True)
        try:
            while True:
                cancellation_checkpoint()
                if time.monotonic()-start>7200:raise ValueError('Native peak calling exceeded the two-hour execution bound')
                try:code=process.wait(timeout=.2);break
                except subprocess.TimeoutExpired:pass
        finally:
            if process.poll() is None:
                import signal
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
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
    from agent.tools.data.regulatory_feature_reference import build_regulatory_feature_reference,publish_regulatory_feature_reference
    from agent.tools.data.scatac_reference import SourceProvenance
    args=arguments(values);root=Path(args['output_dir']);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise ValueError('Output directory is not empty; recovery does not rerun science')
    with tempfile.TemporaryDirectory(prefix='.macs3-',dir=root) as temp:
        stage=Path(temp);ref,contigs,summary,lineage,sources,cells=prepare_fragments(args,stage,root)
        receipt=native(summary['parameters'],stage);rows=inspect_peaks(stage,contigs)
        for name in FILES[:3]:shutil.copyfile(stage/name,root/name)
        if (stage/'native.log').exists():shutil.copyfile(stage/'native.log',root/'native.log')
        with (root/'features.bed').open('w') as stream:
            for chrom,start,end,*_ in rows:stream.write(f'{chrom}\t{start}\t{end}\t{chrom}:{start}-{end}\n')
        reference=dict(manifest_path="",manifest_sha256="")
        if rows:
            bundle=build_regulatory_feature_reference(species=summary['species'],target_assembly=summary['assembly'],
                fasta_path=ref.genome.fasta.path,fai_path=ref.genome.fai.path,feature_bed_path=root/'features.bed',
                feature_category='peak_set',genome_provenance=ref.genome.fasta.provenance,
                feature_provenance=SourceProvenance(basis='caller_supplied',source='Native MACS3 experiment-specific peaks; generation lineage in sibling manifest.json'))
            published=publish_regulatory_feature_reference(bundle,root/'reference.json')
            reference=dict(manifest_path=str(root/'reference.json'),manifest_sha256=file_closure([root/'reference.json'])[0]['sha256'])
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
        write(root/'metadata.json',dict(summary=summary,lineage=lineage,reference=reference,cells=cells))
        pins={p['path']:p for p in sources}
        if file_closure(pins)!=sorted(pins.values(),key=lambda p:p['path']):raise AuthorityError('Sources changed during native execution')
    files=[root/name for name in FILES]+[Path(cells['manifest_path']),root/'population/cells.tsv.gz']
    if rows:files.append(root/'reference.json')
    if (root/'native.log').exists():files.append(root/'native.log')
    value=dict(contract=CONTRACT,profile=PROFILE,arguments=args,execution_identity=execution_identity,result=summary,
        reference=reference,cells=cells,lineage=lineage,files=file_closure(files),sources=sorted(pins.values(),key=lambda p:p['path']),native_completion='succeeded')
    for output in files:
        with output.open('rb') as stream:os.fsync(stream.fileno())
    path=root/'manifest.json';sha=write(path,value)
    from agent.tools.data.scatac_qc_reference import _fsync_dir
    _fsync_dir(root)
    if current() is not None:current().register_execution(KIND,root,execution_identity)
    return verify_publication(path,expected_sha256=sha)


def call_scATAC_peaks(*,fragments_manifest_path,fragments_manifest_sha256,effective_genome_size,genome_size_basis,output_dir,
    library_namespace=None,qvalue=None,exclude_contigs=None,selected_cells_manifest_path=None,selected_cells_manifest_sha256=None,
    explicit_cells_manifest_path=None,explicit_cells_manifest_sha256=None,barcodes_path=None,barcodes_sha256=None,
    matrix_manifest_path=None,matrix_manifest_sha256=None,group_analysis_path=None,group_analysis_sha256=None,group_label=None):
    return execute({k:v for k,v in locals().items() if v is not None})


def load(path,sha=None):
    value=read(path,sha);root=Path(path).parent
    expected={str(root/name) for name in FILES}|{str(root/'population/manifest.json'),str(root/'population/cells.tsv.gz')}
    if value['result']['n_peaks']:expected.add(str(root/'reference.json'))
    if any(r['path']==str(root/'native.log') for r in value['files']):expected.add(str(root/'native.log'))
    if (value['contract']!=CONTRACT or value['profile']!=PROFILE or root!=Path(value['arguments']['output_dir'])
            or Path(path).name!='manifest.json' or value['native_completion']!='succeeded'
            or {r['path'] for r in value['files']}!=expected or len(value['files'])!=len(expected)):
        raise AuthorityError('Invalid peak publication identity or membership')
    return value,file_closure([path])[0]['sha256']


def result(value,path,sha):
    return dict(value['result'],manifest_path=str(path),manifest_sha256=sha,
        reference_manifest_path=value['reference']['manifest_path'],reference_manifest_sha256=value['reference']['manifest_sha256'],
        cells_manifest_path=value['cells']['manifest_path'],cells_manifest_sha256=value['cells']['manifest_sha256'])


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
        execution_identity=context.execution_for(KIND,path),profile=PROFILE,verifier=VERIFIER,qualification={})
    return dict(kind=KIND,manifest_path=str(path),manifest_sha256=actual,manifest=value,proof_identity=identity,
        files=files,historical_sources=value['sources'],upstream={},profile=PROFILE,verifier=VERIFIER,qualification={})


def reuse_result(kind,description,proof,kwargs):return result(description['manifest'],description['manifest_path'],description['manifest_sha256'])
def proof_metadata(kind,output,context):return dict(output)


def publication_record(context,tool_name,args,output,execution_identity,result_metadata):
    value,sha=load(output['manifest_path'],output['manifest_sha256'])
    if (tool_name!=TOOL or value['execution_identity']!=execution_identity or value['arguments']!=arguments(args)
            or _serialize(result_metadata)!=_serialize(output) or result(value,output['manifest_path'],sha)!=_serialize(output)):
        raise AuthorityError('Peak publication lacks its exact producing execution')
    description,key=context.describe(KIND,Path(output['manifest_path']),sha)
    return VerifiedArtifactAuthority(dict(schema_version=2,artifact_type='agent.'+KIND,artifact_contract=CONTRACT,
        publication_path=output['manifest_path'],manifest_sha256=sha,files=description['files'],execution_identity=execution_identity,
        arguments_sha256=digest(arguments(args)),receipt_sha256=sha,upstream={'producer_inputs':digest(description['proof_identity'])},
        resources=dict(scientific_proof_sha256=key,manifest_identity=value,result_metadata=_serialize(output)),science_profile=PROFILE,
        producer_qualification={},verifier=VERIFIER,scope='scientific_correctness.v1',completion='succeeded',
        integrity_scope='artifact_integrity_lineage_historical_sources.v1',source_policy='historical_verified_sources.v1',historical_sources=description['historical_sources']))
