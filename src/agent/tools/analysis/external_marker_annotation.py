"""Direct peak counts and pinned native MAESTRO; historical science stays immutable."""
import ast
import csv
import json
import os
import re
import subprocess
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite
from . import marker_annotation as legacy, annotation_contract as c, peak_h5ad
from agent.tools._cancellation import cancellation_checkpoint

BINDING_FIELDS = ('method','source_path','source_sha256','count_slot','species','assembly',
                  'value_semantics','ordered_cells_sha256','ordered_features_sha256','logical_matrix_sha256')
_REGION = re.compile(r'([^:\s]+):(0|[1-9][0-9]*)-([1-9][0-9]*)\Z')


class PeakScopeError(ValueError):
    """The declared peak axis extends outside the qualified analysis scope."""


def counts(a, slot):
    if slot == 'X': x = a.X
    elif slot.startswith('layers/') and slot[7:] in a.layers: x = a.layers[slot[7:]]
    else: raise ValueError('Select X or an existing layers/name count slot')
    if not (sparse.isspmatrix_csr(x) or sparse.isspmatrix_csc(x)) or x.shape != a.shape:
        raise ValueError('Count slot must be cell-by-peak sparse CSR or CSC')
    x.check_format(full_check=True)
    if not x.has_canonical_format or x.nnz > 100000000 or x.dtype.kind not in 'iuf':
        raise ValueError('Canonical sparse raw counts within execution bounds required')
    v = x.data
    if not np.isfinite(v).all() or (v < 0).any() or (v > 2**53-1).any() or np.any(v != np.floor(v)):
        raise ValueError('Finite nonnegative integral counts exactly representable in float64 required')
    return x


def logical(x):
    import hashlib
    h=hashlib.sha256(); h.update(str(x.shape).encode()); h.update(x.format.encode()); h.update(x.dtype.str.encode())
    for v in (x.indptr,x.indices,x.data): h.update(v.dtype.str.encode()); h.update(v.tobytes())
    return h.hexdigest()


def load_matrix(binding):
    path=legacy.checked_file(binding['source_path'],binding['source_sha256'])
    if path.stat().st_size > peak_h5ad.MAX_BYTES: raise ValueError('Peak H5AD exceeds 1 GiB source limit')
    peak_h5ad.inspect(path)
    a=ad.read_h5ad(path)
    if not 1 <= a.n_obs <= 100000 or not 1 <= a.n_vars <= 2000000 or not a.obs_names.is_unique or not a.var_names.is_unique:
        raise ValueError('Nonempty unique bounded cell and feature axes required')
    for name in [*a.obs_names,*a.var_names]: legacy.identifier(name)
    if binding['method'] != c.EXTERNAL_PRIMARY or binding['value_semantics'] != c.EXTERNAL_SEMANTICS:
        raise ValueError('Unsupported explicit method or measurement')
    if binding['ordered_cells_sha256'] != legacy.ordered_identity_sha256(a.obs_names.tolist()) or binding['ordered_features_sha256'] != legacy.ordered_identity_sha256(a.var_names.tolist()):
        raise ValueError('Peak H5AD axes changed')
    if binding['logical_matrix_sha256'] != logical(counts(a,binding['count_slot'])): raise ValueError('Peak counts changed')
    return a,path


def group_rows(a, spec):
    selected=spec['groups']
    if 'path' in selected:
        frame=pd.read_csv(legacy.checked_file(selected['path'],selected['sha256']),sep='\t',dtype=str,keep_default_na=False)
        if frame.columns.tolist()!=['cell_id','group']: raise ValueError('Grouping TSV requires exact cell_id/group columns')
        rows=list(frame.itertuples(index=False,name=None))
    else:
        name=selected['obs_column']
        if name not in a.obs or a.obs[name].isna().any(): raise ValueError('Selected grouping column absent or contains missing values')
        values=a.obs[name].tolist()
        if any(not isinstance(v,(str,int,np.integer)) or isinstance(v,bool) for v in values):
            raise ValueError('Grouping requires opaque string or integer labels')
        rows=list(zip(a.obs_names.tolist(),map(str,values)))
    if len(legacy.exact_groups(a.obs_names.tolist(),rows))<2: raise ValueError('At least two explicit groups required')
    return rows


def coordinates(features, spec):
    from agent.tools.data.scatac_qc_reference import load_scatac_qc_reference_bundle
    from agent.tools.data._barcode_qc_binding import resource_qualification
    scope=spec['contig_scope']
    _,bundle,_=load_scatac_qc_reference_bundle(scope['path'],expected_sha256=scope['sha256'])
    resource_qualification(bundle)
    if (bundle.species,bundle.assembly)!=(spec['species'],spec['assembly']): raise ValueError('Incompatible contig scope')
    primary={v.name:v.length for v in bundle.contigs if v.classification=='primary_nuclear_qc'}
    encoded=[]
    for value in features:
        match=_REGION.fullmatch(value)
        if match is None: raise ValueError('Peak var_names require exact chrom:start-end zero-based intervals')
        chrom,start,end=match.groups(); start,end=int(start),int(end)
        if chrom not in primary:
            raise PeakScopeError('Peak outside qualified primary nuclear scope')
        if not 0<=start<end<=primary[chrom]: raise ValueError('Peak outside qualified primary nuclear scope')
        encoded.append(f'{chrom}_{start}_{end}'.encode())
    return encoded


def qualify(args,spec):
    path=legacy.checked_file(args['source_path'],args['source_sha256'])
    if path.stat().st_size>peak_h5ad.MAX_BYTES: raise ValueError('Peak H5AD exceeds 1 GiB limit')
    peak_h5ad.inspect(path,privacy=True)
    a=ad.read_h5ad(path); x=counts(a,args['count_slot'])
    for key,expected in (('species',spec['species']),('assembly',spec['assembly']),
                         ('measurement',spec['measurement']),('matrix_semantics',spec['measurement'])):
        if key in a.uns and a.uns[key] != expected:
            raise ValueError('H5AD metadata conflicts with the explicit scientific declaration: '+key)
    binding=dict(method=c.EXTERNAL_PRIMARY,source_path=str(path),source_sha256=args['source_sha256'],count_slot=args['count_slot'],
        species=spec['species'],assembly=spec['assembly'],value_semantics=spec['measurement'],
        ordered_cells_sha256=legacy.ordered_identity_sha256(a.obs_names.tolist()),
        ordered_features_sha256=legacy.ordered_identity_sha256(a.var_names.tolist()),logical_matrix_sha256=logical(x))
    load_matrix(binding); coordinates(a.var_names.tolist(),spec); group_rows(a,spec)
    qualification=dict(contract='external-peak-annotation-input.v1',binding=binding,declarations=spec,
                       upstream_counting_history='declared_not_reconstructed')
    return c.digest(qualification),dict(shape=list(a.shape),logical_matrix_sha256=logical(x)),binding


def weighted_rp(x,features,cells,gene_path,source):
    legacy.checked_file(source,legacy.SOURCES['MAESTRO/scATAC_Genescore.py'])
    tree=ast.parse(Path(source).read_text()); tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    output={}
    def capture(path,matrix,genes,barcodes,**kwargs): output.update(matrix=matrix,genes=list(genes),cells=list(barcodes))
    ns=dict(np=np,pd=pd,sp_sparse=sparse,write_10X_h5=capture)
    exec(compile(tree,str(source),'exec'),ns)
    ns['calculate_RP_score'](x.astype(np.float64,copy=True).T.tocsr(),features,cells,str(gene_path),10000,'unused','Enhanced')
    rp,genes=output['matrix'].T.tocsr(),output['genes']
    if (output['cells']!=cells or rp.shape!=(len(cells),len(genes)) or not np.isfinite(rp.data).all()
            or (rp.data<0).any() or (np.asarray(rp.sum(axis=1)).ravel()<=0).any()):
        raise ValueError('Native RP returned invalid values or zero-evidence cells')
    legacy.ordered_identity_sha256(genes)
    if len({g.upper() for g in genes})!=len(genes): raise ValueError('Ambiguous RP gene namespace')
    rp.eliminate_zeros(); return rp,genes


def annotate_cell_groups(*,binding,specification,runtime,output_dir):
    """One native invocation; shared outer owner owns atomic publication and replay."""
    root=Path(output_dir);root.mkdir()
    a,path=load_matrix(binding);cells=a.obs_names.tolist();groups=group_rows(a,specification)
    features=coordinates(a.var_names.tolist(),specification);upstream=Path(runtime.upstream_dir)
    for name,sha in legacy.SOURCES.items():legacy.checked_file(upstream/name,sha)
    gene=specification['gene_resource'];signature=specification['signature_resource']
    legacy.checked_file(gene['path'],gene['sha256']);legacy.checked_file(signature['path'],signature['sha256'])
    cancellation_checkpoint()
    rp,genes=weighted_rp(counts(a,binding['count_slot']),features,cells,gene['path'],upstream/'MAESTRO/scATAC_Genescore.py')
    sparse.save_npz(root/'rp.npz',rp);mmwrite(str(root/'rp-genes-by-cells.mtx'),rp.T,precision=17)
    (root/'genes.tsv').write_text('\n'.join(genes)+'\n');(root/'cells.tsv').write_text('\n'.join(cells)+'\n')
    pd.DataFrame(groups,columns=['cell_id','group']).to_csv(root/'groups.tsv',sep='\t',index=False,quoting=csv.QUOTE_NONE)
    script=Path(legacy.__file__).parent/'r/maestro_markers_v1.R'
    env=dict(os.environ,R_LIBS_USER=runtime.r_library,R_ENVIRON_USER='/dev/null',R_PROFILE_USER='/dev/null')
    with (root/'marker-runtime.txt').open('w') as log:
        with subprocess.Popen([runtime.rscript,'--vanilla',str(script),str(root),str(upstream)],env=env,stdout=log,stderr=subprocess.STDOUT) as process:
            try:
                while True:
                    cancellation_checkpoint()
                    try:status=process.wait(timeout=.2);break
                    except subprocess.TimeoutExpired:pass
                if status:raise subprocess.CalledProcessError(status,process.args)
            except BaseException:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
    f=pd.read_csv(root/'markers-signature-input.tsv',sep='\t',dtype={'cluster':str,'gene':str},keep_default_na=False,quoting=csv.QUOTE_NONE)
    result=legacy.score_candidates(cells,groups,genes,list(f[['cluster','gene','avg_logFC']].itertuples(index=False,name=None)),legacy.read_signatures(signature['path']))
    for key,rows in result.items():(root/f'{key}.json').write_text(json.dumps(rows,sort_keys=True,allow_nan=False)+'\n')
    (root/'rp-genes-by-cells.mtx').unlink();legacy.checked_file(path,binding['source_sha256']);cancellation_checkpoint()
    value=dict(schema='agent.peak-marker-annotation.v1',profile=c.EXTERNAL_PROFILE,binding=binding,inputs=specification,
        upstream_revision=legacy.REVISION,upstream_sources=legacy.SOURCES,adapter_sha256=legacy.sha256(__file__),
        r_adapter_sha256=legacy.sha256(script),inherited_policy_adapter_sha256=legacy.sha256(legacy.__file__),
        sidecars={p.name:legacy.sha256(p) for p in sorted(root.iterdir())})
    (root/'result.json').write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');return value
