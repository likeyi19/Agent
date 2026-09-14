"""Constrained H5AD reader and representation-only external matrix publication."""
import hashlib
import os
from pathlib import Path
import tempfile
import h5py
import numpy as np

from . import scatac_matrix_contract as m, external_matrix_contract as e
from . import _cell_by_ccre_io as io, _matrix_bedtools as bed
from .scatac_reference import load_scatac_reference_bundle, reinspect_scatac_reference_bundle_sources
from .scatac_fragments_v2_verifier import take_snapshots, check_snapshots
from .scatac_qc_reference import _fsync_dir


def reference(args, limits):
    snapshots = take_snapshots([args['reference_manifest_path']])
    _, ref, _ = load_scatac_reference_bundle(args['reference_manifest_path'], expected_sha256=args['reference_manifest_sha256'])
    if ((ref.species, ref.target_assembly) != (args['species'], args['assembly'])
            or ref.ccre.feature_count > limits.max_features): m.fail('MATRIX_REFERENCE_MISMATCH')
    resources = [ref.genome.fasta.path, ref.genome.fai.path, ref.ccre.bed.path]
    if ref.annotation is not None: resources.append(ref.annotation.resource.path)
    snapshots = tuple(sorted(set(snapshots + take_snapshots(resources))))
    reinspect_scatac_reference_bundle_sources(ref)
    check_snapshots(snapshots)
    return ref, snapshots


def safe_h5(f):
    """No indirect storage, aliases, cycles or unbounded metadata trees."""
    seen = set(); objects = 0
    def visit(group, depth):
        nonlocal objects
        if depth > 2: m.fail('MATRIX_H5AD_INVALID')
        for key in group:
            objects += 1
            if objects > 64 or not isinstance(group.get(key, getlink=True), h5py.HardLink): m.fail('MATRIX_H5AD_INVALID')
            item = group[key]; address = h5py.h5o.get_info(item.id).addr
            if address in seen: m.fail('MATRIX_H5AD_INVALID')
            seen.add(address)
            if isinstance(item, h5py.Dataset):
                if item.is_virtual or item.external: m.fail('MATRIX_H5AD_INVALID')
            else: visit(item, depth + 1)
    visit(f, 0)
    if (f.attrs.get('encoding-type') != 'anndata' or f.attrs.get('encoding-version') != '0.1.0'
            or not {'X','obs','var'} <= set(f)
            or not set(f) <= {'X','obs','var','uns','layers','obsm','obsp','varm','varp'}): m.fail('MATRIX_H5AD_INVALID')
    for name in ('layers','obsm','obsp','varm','varp'):
        if name in f and (not isinstance(f[name], h5py.Group) or len(f[name])): m.fail('MATRIX_H5AD_INVALID')


def axis(group, length, extra=()):
    index = group.attrs.get('_index')
    if (not isinstance(index, str) or index in extra or set(group) != {index, *extra}
            or group.attrs.get('encoding-type') != 'dataframe' or group.attrs.get('encoding-version') != '0.2.0'
            or list(group.attrs.get('column-order', [])) != list(extra)): m.fail('MATRIX_AXIS_INVALID')
    for key in (index, *extra):
        ds = group[key]
        if not isinstance(ds, h5py.Dataset) or ds.shape != (length,): m.fail('MATRIX_AXIS_INVALID')
        strings = key == index or key == 'chrom'
        if ds.attrs.get('encoding-type') != ('string-array' if strings else 'array') or ds.attrs.get('encoding-version') != '0.2.0': m.fail('MATRIX_AXIS_INVALID')
        if (strings and h5py.check_string_dtype(ds.dtype) is None
                or not strings and ds.dtype.kind not in 'iu'): m.fail('MATRIX_AXIS_INVALID')
    return index


def inspect_axes(f, ref, args, db, budget, *, published=False):
    safe_h5(f)
    if not isinstance(f['X'], h5py.Group): m.fail('MATRIX_CSR_INVALID')
    dims = np.asarray(f['X'].attrs.get('shape', []))
    if dims.shape != (2,) or dims.dtype.kind not in 'iu': m.fail('MATRIX_CSR_INVALID')
    n, p = map(int, dims)
    if n > budget.limits.max_selected or p != ref.ccre.feature_count: m.fail('MATRIX_REFERENCE_MISMATCH')
    oi = axis(f['obs'], n)
    extra = ('chrom','start','end') if 'chrom' in f['var'] else ()
    vi = axis(f['var'], p, extra)
    db.execute('CREATE TABLE IF NOT EXISTS external_cells (row INTEGER PRIMARY KEY, name TEXT UNIQUE)')
    digest = hashlib.sha256(b'agent.external-ordered-cells.v1\0')
    for left in range(0, n, io.CHUNK):
        for i, name in enumerate(f['obs'][oi].asstr()[left:left+io.CHUNK], left):
            if not name or name != name.strip() or len(name.encode('utf-8')) > 4096 or any(ord(c) < 32 or ord(c) == 127 for c in name):
                m.fail('MATRIX_ROW_IDENTITY_INVALID')
            if published:
                if db.execute('SELECT name FROM external_cells WHERE row=?', (i,)).fetchone() != (name,): m.fail('MATRIX_CONSERVATION_MISMATCH')
            else:
                if db.execute('SELECT 1 FROM external_cells WHERE name=?', (name,)).fetchone(): m.fail('MATRIX_ROW_IDENTITY_INVALID')
                db.execute('INSERT INTO external_cells VALUES(?,?)', (i, name))
            raw = name.encode('utf-8'); digest.update(len(raw).to_bytes(8,'big')); digest.update(raw)
        budget.check()
    with open(ref.ccre.bed.path, 'rt') as stream:
        count = 0; feature_digest = hashlib.sha256()
        for left in range(0, p, io.CHUNK):
            names = f['var'][vi].asstr()[left:left+io.CHUNK]
            coords = list(zip(f['var']['chrom'].asstr()[left:left+io.CHUNK],
                              f['var']['start'][left:left+io.CHUNK], f['var']['end'][left:left+io.CHUNK])) if extra else None
            for offset, name in enumerate(names):
                fields = stream.readline(65537).rstrip('\n').split('\t')
                if len(fields) < 3: m.fail('MATRIX_REFERENCE_MISMATCH')
                chrom, start, end = fields[0], int(fields[1]), int(fields[2])
                if name != f'{chrom}:{start}-{end}' or coords is not None and coords[offset] != (chrom,start,end): m.fail('MATRIX_REFERENCE_MISMATCH')
                feature_digest.update((name+'\n').encode()); count += 1
            budget.check()
        if stream.read(1) or count != p or feature_digest.hexdigest() != ref.ccre.ordered_feature_sha256: m.fail('MATRIX_REFERENCE_MISMATCH')
    expected = dict(species=args['species'], assembly=args['assembly'], matrix_semantics=args['matrix_semantics'],
                    reference_identity_sha256=ref.reference_identity_sha256)
    if 'uns' in f:
        if not isinstance(f['uns'], h5py.Group) or not set(f['uns']) <= set(expected): m.fail('MATRIX_H5AD_INVALID')
        for key in f['uns']:
            ds = f['uns'][key]
            if not isinstance(ds,h5py.Dataset) or ds.shape != () or h5py.check_string_dtype(ds.dtype) is None or ds.asstr()[()] != expected[key]: m.fail('MATRIX_REFERENCE_MISMATCH')
    if published:
        if set(f) != {'X','obs','var','uns','layers','obsm','obsp','varm','varp'} or set(f['uns']) != set(expected) or extra: m.fail('MATRIX_H5AD_INVALID')
        for key in ('uns','layers','obsm','obsp','varm','varp'):
            if f[key].attrs.get('encoding-type') != 'dict' or f[key].attrs.get('encoding-version') != '0.1.0': m.fail('MATRIX_H5AD_INVALID')
        for ds in f['uns'].values():
            if ds.attrs.get('encoding-type') != 'string' or ds.attrs.get('encoding-version') != '0.2.0': m.fail('MATRIX_H5AD_INVALID')
    return n, p, digest.hexdigest()


def build(args, output_dir, limits=io.MatrixLimits()):
    """Private-stage writer. Final publication uses the existing matrix envelope."""
    source = Path(args['source_path']); root = Path(output_dir); root.mkdir()
    ref, reference_snapshots = reference(args, limits)
    before = tuple(sorted(set(reference_snapshots + take_snapshots([source]))))
    if source.stat().st_size > limits.max_scratch_bytes: m.fail('MATRIX_RESOURCE_LIMIT')
    if bed.file_sha(source) != args['source_sha256']: m.fail('MATRIX_SOURCE_MISMATCH')
    with tempfile.TemporaryDirectory(prefix='.external-matrix-', dir=root.parent) as scratch:
        budget = io.Budget(root.parent, limits)
        with io.database(Path(scratch)/'axes.sqlite', budget) as db, h5py.File(source,'r') as src:
            n,p,cells = inspect_axes(src,ref,args,db,budget)
            # Validate the complete source before writing any matrix payload.
            summary = e.logical_identity(n,p,io.csr_rows(src,n,p,budget),args['matrix_semantics'])
            with h5py.File(root/'matrix.h5ad','x') as out:
                out.attrs.update({'encoding-type':'anndata','encoding-version':'0.1.0'})
                for key in ('layers','obsm','obsp','varm','varp','uns'):
                    out.create_group(key).attrs.update({'encoding-type':'dict','encoding-version':'0.1.0'})
                io.write_axis(out,'obs',('_index',),('_index',),db.execute('SELECT name FROM external_cells ORDER BY row'),n,budget)
                var = out.create_group('var')
                var.attrs.update({'encoding-type':'dataframe','encoding-version':'0.2.0','_index':'_index','column-order':np.array([],dtype=h5py.string_dtype())})
                ds = var.create_dataset('_index',shape=(p,),dtype=h5py.string_dtype())
                ds.attrs.update({'encoding-type':'string-array','encoding-version':'0.2.0'})
                vi = src['var'].attrs['_index']
                for left in range(0,p,io.CHUNK): ds[left:left+io.CHUNK] = src['var'][vi].asstr()[left:left+io.CHUNK]; budget.check()
                x = out.create_group('X'); x.attrs.update({'encoding-type':'csr_matrix','encoding-version':'0.1.0','shape':np.array([n,p],dtype='int64')})
                for key in ('data','indices','indptr'):
                    source_ds = src['X'][key]
                    ds = x.create_dataset(key,shape=source_ds.shape,dtype='int64',chunks=True,compression='gzip')
                    for left in range(0,len(ds),io.CHUNK): ds[left:left+io.CHUNK] = source_ds[left:left+io.CHUNK]; budget.check()
                for key,value in dict(species=args['species'],assembly=args['assembly'],matrix_semantics=args['matrix_semantics'],reference_identity_sha256=ref.reference_identity_sha256).items():
                    ds = out['uns'].create_dataset(key,data=value,dtype=h5py.string_dtype()); ds.attrs.update({'encoding-type':'string','encoding-version':'0.2.0'})
        payload = root/'matrix.h5ad'
        value = dict(artifact_type=m.ARTIFACT,schema_version=1,contract_version=e.CONTRACT,operation='external_adoption',
            profile=e.PROFILE,profile_sha256=e.PROFILE_SHA256,species=args['species'],assembly=args['assembly'],
            source=dict(path=str(source),sha256=args['source_sha256'],size_bytes=source.stat().st_size),
            reference=dict(manifest_path=args['reference_manifest_path'],manifest_sha256=args['reference_manifest_sha256'],identity_sha256=ref.reference_identity_sha256,contract_version='scatac-reference-bundle.v1'),
            matrix_semantics=args['matrix_semantics'],shape=[n,p],**summary,ordered_cells_sha256=cells,
            ordered_feature_sha256=ref.ccre.ordered_feature_sha256,source_logical_matrix_sha256=summary['logical_matrix_sha256'],
            matrix=dict(path='matrix.h5ad',sha256=bed.file_sha(payload),size_bytes=payload.stat().st_size,format='csr',dtype='int64'),readiness='matrix_available' if n else 'no_source_cells')
        value['identity_sha256'] = e.identity(value)
        raw = m.canonical(e.validate(value)); sha = hashlib.sha256(raw).hexdigest()
        (root/'manifest.json').write_bytes(raw)
        from .cell_by_ccre_verifier import verify_cell_by_ccre
        verify_cell_by_ccre(root/'manifest.json',expected_sha256=sha,limits=limits,scratch_parent=root.parent)
        check_snapshots(before); budget.check()
        for path in (payload,root/'manifest.json'):
            with path.open('rb') as f: os.fsync(f.fileno())
        _fsync_dir(root)
    return dict(manifest_path=str(root/'manifest.json'),manifest_sha256=sha)
