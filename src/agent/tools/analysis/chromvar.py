"""Native chromVAR motif deviations; AnnData owns input and R owns the method.

No dataset certification, upstream reconstruction, or scientific replay on reads.
"""
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem, sparse_dataset
from scipy import sparse
from scipy.io import mmread, mmwrite

from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError, VerifiedArtifactAuthority, digest
from agent.tools.data.authority_context import current, owned_verification
from agent.tools._file_identity import file_closure
from .sequence_prediction import read, write, library, arguments
from . import peak_h5ad
from .sequence_reference import resolve_motif_resources, metadata_declarations, coordinates, check_declarations

TOOL = 'compute_motif_accessibility'
KIND = 'motif_accessibility'
CONTRACT = 'chromvar-motif-accessibility.v1'
RESOURCE = 'jaspar-motif-collection.v1'
POLICY = 'chromvar-native-v1'
PROFILE = digest(dict(contract=CONTRACT, method='chromVAR/motifmatchr',
    filtering='native positive-total peaks; no overlap or sample filtering', motif_background='subject'))
VERIFIER = dict(id='agent.chromvar-native-completion', compatibility_version='1')
LIMITATIONS = ('Motif-associated accessibility deviations are not expression, physical TF binding, '
    'regulatory causality or replicate-level condition significance. Related TFs can share motifs. '
    'Z-scores standardize native background deviations; they are not motif-enrichment p-values. '
    'Supplied groups are metadata, not validated cell types. No sample QC or overlap pruning is inferred. '
    'Undefined native values remain NaN; descriptive summaries omit them and record their number.')
FILES = ('motif-accessibility.h5ad','peaks.tsv','cells.tsv','motifs.tsv','motif-summary.tsv',
    'group-summary.tsv','motif-matches.npz','background-peaks.npy','patterns.png','metadata.json','native-receipt.json')
FACT_FIELDS = ('n_cells','n_source_peaks','n_retained_peaks','n_motifs','excluded_peaks','undefined_values',
    'measurement','source_measurement','count_slot','assembly','motif_collection','method','parameters',
    'top_motifs','group_summary','group_key','software_versions','limitations')
RESULT_FIELDS = dict(status=(str,),contract_version=(str,),manifest_path=(str,),manifest_sha256=(str,),
    scores_path=(str,),scores_sha256=(str,),n_cells=(int,),n_source_peaks=(int,),n_retained_peaks=(int,),
    n_motifs=(int,),excluded_peaks=(int,),undefined_values=(dict,),measurement=(str,),source_measurement=(str,),
    count_slot=(str,),assembly=(str,),motif_collection=(dict,),method=(str,),parameters=(dict,),
    top_motifs=(list,),group_summary=(list,),group_key=(str,),software_versions=(dict,),limitations=(str,))


def qualify_resource(resource):
    """A reusable versioned collection; never an experimental-data whitelist."""
    from Bio import motifs
    if (resource['schema'] != RESOURCE or resource['collection'] != 'CORE'
            or not resource['release'] or not resource['tax_group']
            or not resource['applicable_taxonomy_ids']):
        raise ValueError('Versioned CORE motifs and their declared taxonomic applicability are required')
    with open(resource['pfm']['path']) as stream:
        parsed = list(motifs.parse(stream, 'jaspar'))
    ids = [m.matrix_id for m in parsed]
    metadata = pd.read_csv(resource['metadata']['path'], sep='\t', keep_default_na=False)
    selected = metadata[metadata.matrix_id.isin(ids)]
    if (not ids or len(set(ids)) != len(ids) or selected.matrix_id.duplicated().any()
            or set(selected.matrix_id) != set(ids) or set(selected.collection) != {'CORE'}
            or set(selected.tax_group) != {resource['tax_group']}):
        raise ValueError('Motif accessions, versions or taxon metadata differ from the collection')
    for motif in parsed:
        values = np.asarray([motif.counts[k] for k in 'ACGT'])
        if not np.isfinite(values).all() or (values < 0).any() or (values.sum(0) <= 0).any():
            raise ValueError('Invalid native position frequency matrix')
    return resource



def observe_input(path):
    """Read-only declarations for exact offered sources, not a readiness gate."""
    peak_h5ad.inspect(path,require_matrix=False,max_logical_bytes=None)
    with h5py.File(path,'r') as handle:
        var=read_elem(handle['var'])
        known=metadata_declarations(handle,'X',{},var)
        slots=(['X'] if 'X' in handle else [])+['layers/'+key for key in handle.get('layers',{})]
    return dict(default_slot='X',slots=slots,recorded_declarations=known,
        missing_declarations=[key for key in ('assembly','coordinate_system','matrix_semantics') if key not in known],
        scope='source metadata only; not numeric eligibility, execution authority or user instructions')


def _count_declarations(handle, args, var):
    """One owner of slot scope and required declarations, before numeric work."""
    slot=args.get('count_slot','X')
    if slot!='X' and not slot.startswith('layers/'):
        slot='layers/'+slot
    if slot not in handle:
        raise ValueError('Selected count matrix is absent')
    effective=metadata_declarations(handle,slot,args,var)
    effective['count_slot']=slot
    missing=[key for key in ('assembly','coordinate_system') if not effective.get(key)]
    if effective.get('matrix_semantics') not in ('peak_counts','fragment_counts','insertion_counts','binary_accessibility'):
        missing.append('matrix_semantics')
        # A recorded transformed X cannot be the undeclared default count slot.
        # No layer name establishes its measurement or selects it for the user.
        if 'count_slot' not in args and effective.get('matrix_semantics'):
            missing.append('count_slot')
    return effective,tuple(sorted(missing))


def parameter_prerequisites(inputs):
    """Project unresolved declarations for an already bound ordinary input.

    Planned upstream matrices remain with their compiler/execution owners.
    Numeric eligibility, coordinates and resources are not validated twice.
    """
    path=inputs.get('input_path')
    if not isinstance(path,(str,Path)):
        return ()
    with h5py.File(path,'r') as handle:
        _,missing=_count_declarations(handle,inputs,read_elem(handle['var']))
    return missing


def prepare_input(args, stage):
    from .replicate_pseudobulk import _strict_ids
    peak_h5ad.inspect(args['input_path'], require_matrix=False, max_logical_bytes=None)
    with h5py.File(args['input_path'],'r') as f:
        obs,var=read_elem(f['obs']),read_elem(f['var'])
        effective,missing=_count_declarations(f,args,var)
        slot=effective['count_slot']
        check_declarations(args['input_path'],effective)
        if 'matrix_semantics' in missing:
            raise ValueError('Declare actual raw peak/fragment/insertion counts or binary accessibility for the selected slot; normalized or transformed values are not native counts')
        semantics=effective['matrix_semantics']
        cells=_strict_ids(obs.index,'cell');features=_strict_ids(var.index,'feature')
        if any(any(c in s for c in '\t\r\n') for s in cells+features):
            raise ValueError('Tab/newline axis labels exceed the bounded TSV exchange; use ordinary unique labels')
        if 'coordinate_source' not in effective:
            effective['coordinate_source']='var_columns' if {'chrom','start','end'}<=set(var) else 'feature_names'
        intervals,_=coordinates(var,effective)
        peaks=pd.DataFrame(intervals,columns=['chrom','start','end']);peaks.insert(0,'feature_id',features)
        peaks.insert(0,'source_index',np.arange(len(var)))
        group=args.get('group_key')
        if group and group not in obs: raise ValueError('Selected grouping field is absent')
        element=f[slot]
        matrix=sparse_dataset(element) if isinstance(element,h5py.Group) else element
        if matrix.shape!=(len(obs),len(var)):raise ValueError('Count matrix and ordered axes differ')
        x=peak_h5ad.read_native_counts(matrix,semantics)
    mmwrite(stage/'counts.mtx',x.T.tocsc())
    peaks.to_csv(stage/'peaks.tsv',sep='\t',index=False)
    pd.DataFrame(dict(cell_id=cells,source_index=np.arange(len(cells)))).to_csv(stage/'cells.tsv',sep='\t',index=False)
    return effective,obs,peaks,x


def native(args, stage, fasta, motif):
    rscript=os.environ.get('AGENT_CHROMVAR_RSCRIPT',str(library().root.parent/'runtimes/chromvar/env/bin/Rscript'))
    if not Path(rscript).is_file():raise ValueError('Provision the optional isolated chromVAR R runtime; unrelated tools do not require it')
    script=Path(__file__).parent/'r/chromvar_v1.R'
    implementation=file_closure([rscript,script])
    request=dict(stage=str(stage),fasta=fasta,motifs=motif['pfm']['path'],
        seed=args.get('seed',20261030),background_iterations=args.get('background_iterations',50),p_cutoff=args.get('p_cutoff',5e-5))
    if type(request['seed']) is not int or not 0<=request['seed']<2**31:
        raise ValueError('Seed must be an R-compatible nonnegative integer')
    if type(request['background_iterations']) is not int or request['background_iterations'] < 2:
        raise ValueError('Background iterations must be at least two')
    if isinstance(request['p_cutoff'],bool) or not 0<float(request['p_cutoff'])<1:
        raise ValueError('Motif scan probability cutoff must be strictly between zero and one')
    write(stage/'request.json',request)
    from agent.tools._native_process import run, r_environment
    code = run([rscript, '--vanilla', str(script), str(stage/'request.json')],
        stage/'native.log', environment=r_environment())
    if code:
        # Keep native diagnostics private but expose the actionable method failure.
        diagnostic=(stage/'native.log').read_bytes()
        (stage.parent/'native-failure.log').write_bytes(diagnostic)
        if b'the leading minor of order' in diagnostic:
            raise ValueError('Native background selection has singular GC/accessibility covariance; this peak/count set cannot estimate the matched background. No artificial jitter was added.')
        raise RuntimeError('Native chromVAR failed; private runtime diagnostics were retained. No fallback or altered counts were used.')
    receipt=read(stage/'native-receipt.json')
    if receipt['completion']!='succeeded':raise AuthorityError('Missing native completion')
    if any(receipt[k]!=request[k] for k in ('seed','background_iterations','p_cutoff')):
        raise AuthorityError('Native completion parameters differ from the actual request')
    if file_closure([rscript,script])!=implementation:raise AuthorityError('Native implementation changed during execution')
    receipt['implementation']=implementation
    return receipt


def _summaries(z,motifs,obs,group):
    finite=np.isfinite(z);n=finite.sum(0)
    values=np.where(finite,z,0);mean=np.divide(values.sum(0),n,out=np.full(z.shape[1],np.nan),where=n>0)
    centered=np.where(finite,z-mean,0)
    sd=np.sqrt(np.divide((centered**2).sum(0),n-1,out=np.full(z.shape[1],np.nan),where=n>1))
    table=motifs[['motif_id','name']].copy();table['finite_cells']=n;table['mean_z']=mean;table['sd_z']=sd
    order=np.argsort(-np.nan_to_num(sd,nan=-np.inf),kind='stable')
    top=table.iloc[order[:12]].replace([np.inf,-np.inf,np.nan],None).to_dict('records')
    rows=[]
    if group:
        # Group labels are descriptive, including numerical vendor cluster IDs.
        for label in pd.unique(obs[group].dropna()):
            selected=(obs[group]==label).fillna(False).to_numpy(dtype=bool);data=z[selected];ok=np.isfinite(data);counts=ok.sum(0)
            avg=np.divide(np.where(ok,data,0).sum(0),counts,out=np.full(z.shape[1],np.nan),where=counts>0)
            for i in range(len(motifs)):
                rows.append(dict(group=str(label),motif_id=motifs.motif_id.iloc[i],name=motifs.name.iloc[i],
                    cells=int(selected.sum()),finite_cells=int(counts[i]),mean_z=float(avg[i]) if np.isfinite(avg[i]) else None))
    groups=pd.DataFrame(rows,columns=['group','motif_id','name','cells','finite_cells','mean_z'])
    example_rows=groups[groups.motif_id.isin([r['motif_id'] for r in top[:3]])].head(24)
    examples=example_rows.astype(object).where(example_rows.notna(),None).to_dict('records')
    return table,groups,top,examples,order[:20]


def execute(values, execution_identity=None):
    from agent.tools._native_process import call
    return call('chromvar', dict(values=values, execution_identity=execution_identity),
                lambda: _execute(values, execution_identity))


def _execute(values, execution_identity=None):
    import anndata as ad
    args=arguments(values);root=Path(args['output_dir']);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise ValueError('Output directory is not empty; native retry requires a new execution')
    source=file_closure([args['input_path']])[0]
    if args.get('input_sha256') and source['sha256']!=args['input_sha256']:raise AuthorityError('Selected H5AD bytes changed')
    with tempfile.TemporaryDirectory(prefix='.chromvar-',dir=root) as temp:
        stage=Path(temp)
        effective,obs,peaks,x=prepare_input(args,stage)
        fasta,genome,motif,entry,sources=resolve_motif_resources(effective,peaks,stage)
        receipt=native(effective,stage,fasta,motif)
        retained=pd.read_csv(stage/'retained.tsv',sep='\t');motifs=pd.read_csv(stage/'motifs.tsv',sep='\t')
        ids=retained.source_index.to_numpy();m=len(motifs);n=len(obs)
        if (receipt['input_dim']!=[len(peaks),n] or receipt['retained_dim']!=[len(ids),n]
                or len(np.unique(ids))!=len(ids) or not np.array_equal(ids,np.flatnonzero(np.asarray(x.sum(0)).ravel()>0))):
            raise AuthorityError('Native input/filter axes differ from original counts')
        deviations=np.fromfile(stage/'deviations.bin',dtype='<f8').reshape((m,n),order='F').T
        z=np.fromfile(stage/'z.bin',dtype='<f8').reshape((m,n),order='F').T
        bg=np.fromfile(stage/'background.bin',dtype='<i4').reshape((len(ids),receipt['background_iterations']),order='F')-1
        matches=mmread(stage/'matches.mtx').tocsr().astype(bool)
        if matches.shape!=(len(ids),m) or np.any(bg<0) or np.any(bg>=len(ids)):
            raise AuthorityError('Native motif/background axes differ')
        annotations=pd.read_csv(motif['metadata']['path'],sep='\t',keep_default_na=False).set_index('matrix_id')
        for key in ('class','family','tax_id','species'):
            motifs[key]=annotations.loc[motifs.motif_id,key].to_numpy()
        table,groups,top,examples,display=_summaries(z,motifs,obs,args.get('group_key'))
        peaks['retained']=False;peaks.loc[ids,'retained']=True;peaks['gc']=np.nan;peaks.loc[ids,'gc']=retained.gc.to_numpy()
        peaks['retained_index']=-1;peaks.loc[ids,'retained_index']=np.arange(len(ids))
        seq=pd.read_csv(stage/'sequence-identities.tsv',sep='\t').set_index('source_index')
        peaks['sequence_sha256']='';peaks.loc[ids,'sequence_sha256']=seq.loc[ids,'sequence_sha256'].to_numpy()
        peaks['exclusion_reason']=np.where(peaks.retained,'','zero_total_count')
        parameters={k:receipt[k] for k in ('seed','background_iterations','p_cutoff','motif_background','filtering')}
        collection={k:motif[k] for k in ('release','collection','tax_group','license','citation')}
        collection['resource_id']=entry['id']
        summary=dict(status='succeeded',contract_version=CONTRACT,n_cells=n,n_source_peaks=len(peaks),n_retained_peaks=len(ids),n_motifs=m,
            excluded_peaks=len(peaks)-len(ids),undefined_values=dict(deviations=int((~np.isfinite(deviations)).sum()),z_scores=int((~np.isfinite(z)).sum())),
            measurement='chromVAR bias-corrected motif accessibility deviations; separate background-standardized z-scores',
            source_measurement=effective['matrix_semantics'],count_slot=effective['count_slot'],assembly=genome['assembly'],
            motif_collection=collection,method='native chromVAR + motifmatchr',parameters=parameters,
            top_motifs=top,group_summary=examples,group_key=args.get('group_key',''),software_versions=receipt['versions'],limitations=LIMITATIONS)
        derived_obs=pd.DataFrame(index=obs.index.copy());derived_obs['source_cell_index']=np.arange(n)
        if args.get('group_key'):derived_obs['supplied_group']=obs[args['group_key']].astype(str)
        derived=ad.AnnData(deviations,obs=derived_obs,var=motifs.set_index('motif_id'));derived.layers['z_scores']=z
        derived.uns=dict(matrix_semantics='chromVAR_bias_corrected_deviations',layer_semantics=dict(z_scores='chromVAR_background_standardized_z_scores'),
            source_sha256=source['sha256'],source_count_slot=summary['count_slot'],source_measurement=summary['source_measurement'],
            assembly=summary['assembly'],method=summary['method'],seed=parameters['seed'],limitations=LIMITATIONS)
        derived.write_h5ad(root/'motif-accessibility.h5ad',compression='gzip')
        peaks.to_csv(root/'peaks.tsv',sep='\t',index=False);motifs.to_csv(root/'motifs.tsv',sep='\t',index=False)
        (root/'cells.tsv').write_bytes((stage/'cells.tsv').read_bytes())
        table.to_csv(root/'motif-summary.tsv',sep='\t',index=False);groups.to_csv(root/'group-summary.tsv',sep='\t',index=False)
        sparse.save_npz(root/'motif-matches.npz',matches);np.save(root/'background-peaks.npy',bg,allow_pickle=False)
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        shown=z[:,display].T
        labels=[f'{motifs.name.iloc[i]} ({motifs.motif_id.iloc[i]})' for i in display]
        if args.get('group_key'):
            order=np.argsort(obs[args['group_key']].astype('string').fillna('').to_numpy(),kind='stable');shown=shown[:,order]
        fig,ax=plt.subplots(figsize=(12,7));im=ax.imshow(np.ma.masked_invalid(shown),aspect='auto',cmap='RdBu_r',vmin=-3,vmax=3,interpolation='nearest')
        ax.set_yticks(range(len(display)),labels,fontsize=8);ax.set_xlabel('Cells (supplied group order)' if args.get('group_key') else 'Cells (original order)')
        if args.get('group_key'):
            labels_in_order=obs[args['group_key']].astype('string').fillna('[missing]').to_numpy()[order]
            starts=np.r_[0,np.flatnonzero(labels_in_order[1:]!=labels_in_order[:-1])+1];ends=np.r_[starts[1:],n]
            if len(starts)<=40:
                ax.set_xticks((starts+ends-1)/2,labels_in_order[starts],rotation=45,ha='right')
                for boundary in starts[1:]:ax.axvline(boundary-.5,color='black',linewidth=.5)
                ax.set_xlabel('Supplied group labels (cells ordered within group)')
        ax.set_title('Most variable motif-associated accessibility (descriptive)');fig.colorbar(im,ax=ax,label='Native chromVAR z-score (display clipped at ±3)')
        fig.tight_layout();fig.savefig(root/'patterns.png',dpi=150);plt.close(fig)
        write(root/'native-receipt.json',receipt)
        metadata=dict(summary=summary,genome=genome,motif=motif,source=source,coordinate_system='zero_based_half_open',
            background_indices='0-based retained peak index; peaks.tsv maps to original features',matches_axes='retained peaks × motifs.tsv order',
            cells_axes='original cell order, including native undefined values',receipt=receipt)
        write(root/'metadata.json',metadata)
        expected={r['path']:r for r in [source,*sources,*receipt['implementation']]}
        sources=file_closure(expected)
        if sources!=sorted(expected.values(),key=lambda r:r['path']):raise AuthorityError('Input or resource changed during native execution')
    value=dict(contract=CONTRACT,profile=PROFILE,arguments=args,execution_identity=execution_identity,result=summary,
        files=file_closure(root/name for name in FILES),sources=sources,native_completion='succeeded')
    path=root/'manifest.json';sha=write(path,value)
    if current() is not None:current().register_execution(KIND,root,execution_identity)
    return verify_publication(path,expected_sha256=sha)


def compute_motif_accessibility(input_path,output_dir,input_sha256=None,count_slot=None,matrix_semantics=None,assembly=None,
    coordinate_source=None,coordinate_system=None,feature_chrom_key=None,feature_start_key=None,feature_end_key=None,
    group_key=None,genome_path=None,genome_sha256=None,genome_resource=None,taxonomy_id=None,motif_collection=None,seed=None,background_iterations=None,p_cutoff=None):
    return execute({k:v for k,v in locals().items() if v is not None})


def load(path,sha=None):
    value=read(path,sha);root=Path(path).parent
    if (value['contract']!=CONTRACT or value['profile']!=PROFILE or root!=Path(value['arguments']['output_dir'])
            or Path(path).name!='manifest.json' or value['native_completion']!='succeeded'
            or {r['path'] for r in value['files']}!={str(root/name) for name in FILES} or len(value['files'])!=len(FILES)):
        raise AuthorityError('Invalid motif publication identity or membership')
    return value,file_closure([path])[0]['sha256']


def result(value,path,sha):
    pin=next(r for r in value['files'] if Path(r['path']).name=='motif-accessibility.h5ad')
    return dict(value['result'],manifest_path=str(path),manifest_sha256=sha,scores_path=pin['path'],scores_sha256=pin['sha256'])


@owned_verification(KIND)
def verify_publication(path,*,expected_sha256):
    value,sha=load(path,expected_sha256)
    if file_closure(r['path'] for r in value['files']+value['sources'])!=sorted(value['files']+value['sources'],key=lambda r:r['path']):
        raise AuthorityError('Motif publication or source integrity changed')
    return result(value,path,sha)


def verify_result(args,output):
    value,sha=load(output['manifest_path'],output['manifest_sha256'])
    if value['arguments']!=arguments(args) or _serialize(output)!=verify_publication(output['manifest_path'],expected_sha256=sha):
        raise AuthorityError('Motif result differs from its producing request')


def recover(args,execution_identity):
    args=arguments(args);path=Path(args['output_dir'])/'manifest.json';value,sha=load(path)
    if value['arguments']!=args or value['execution_identity']!=execution_identity:raise AuthorityError('Recovery cannot authorize a native retry')
    return verify_publication(path,expected_sha256=sha)


def describe(path,sha,context):
    value,actual=load(path,sha);files=context.files([path,*(r['path'] for r in value['files'])])
    if [r for r in files if r['path']!=str(path)]!=value['files']:raise AuthorityError('Motif artifact integrity changed')
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
        raise AuthorityError('Motif publication lacks its exact producing execution')
    description,key=context.describe(KIND,Path(output['manifest_path']),sha)
    return VerifiedArtifactAuthority(dict(schema_version=2,artifact_type='agent.'+KIND,artifact_contract=CONTRACT,
        publication_path=output['manifest_path'],manifest_sha256=sha,files=description['files'],execution_identity=execution_identity,
        arguments_sha256=digest(arguments(args)),receipt_sha256=sha,upstream={'producer_inputs':digest(description['proof_identity'])},
        resources=dict(scientific_proof_sha256=key,manifest_identity=value,result_metadata=_serialize(output)),science_profile=PROFILE,
        producer_qualification={},verifier=VERIFIER,scope='scientific_correctness.v1',completion='succeeded',
        integrity_scope='artifact_integrity_lineage_historical_sources.v1',source_policy='historical_verified_sources.v1',historical_sources=description['historical_sources']))


DELIVERY_FILES = {
    'motif-accessibility.h5ad': ('Cell-by-motif deviations and separate z-scores','application/octet-stream'),
    'peaks.tsv': ('Original peak mapping, exclusions, GC and sequence identities','text/tab-separated-values; charset=utf-8'),
    'cells.tsv': ('Original ordered cells','text/tab-separated-values; charset=utf-8'),
    'motifs.tsv': ('Versioned motif annotations and native match fractions','text/tab-separated-values; charset=utf-8'),
    'motif-summary.tsv': ('Descriptive motif variability','text/tab-separated-values; charset=utf-8'),
    'group-summary.tsv': ('Descriptive supplied-group motif means','text/tab-separated-values; charset=utf-8'),
    'motif-matches.npz': ('Sparse retained-peak by motif matches','application/octet-stream'),
    'background-peaks.npy': ('Native background retained-peak indices','application/octet-stream'),
    'patterns.png': ('Motif-associated accessibility patterns','image/png'),
    'metadata.json': ('Motif method, source and resource metadata','application/json'),
    'native-receipt.json': ('Native completion and software versions','application/json'),
}
