"""Shared qualified genomes, genomic declarations and exact interval extraction."""
import os
import h5py
from pathlib import Path
from anndata.io import read_elem
from agent.schemas.verification_authority import AuthorityError
from agent.tools._file_identity import file_closure
from .sequence_prediction import library


def matching_genome_entries(entries, assembly, species=None):
    """The same exact assembly/species matching for preparation and execution."""
    matches=[e for e in entries if e['kind']=='genome' and e['resource']['assembly']==assembly]
    if len(matches)!=1:raise ValueError('An explicit assembly with a unique qualified matching genome is required')
    identity=matches[0]['resource']['species']
    names={str(v) for v in identity.values()}
    names.update({9606:('human',),10090:('mouse',),7955:('zebrafish',)}.get(identity['taxonomy_id'],()))
    if species is not None and species not in names:raise ValueError('Declared species differs from the qualified reference genome')
    return matches[0]


def resolve_genome(assembly, species=None):
    from .sequence_prediction import library
    lib = library()
    entry=matching_genome_entries(lib.entries(),assembly,species)
    return lib.check(entry),entry


def extract(fasta, chrom, start, end, strand='.'):
    if (strand not in ('+', '-', '.') or chrom not in fasta.references
            or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= fasta.get_reference_length(chrom)):
        raise ValueError('Invalid contig, strand or interval bounds; no clipping, padding or liftOver')
    sequence = fasta.fetch(chrom, start, end).upper()
    if len(sequence) != end - start:
        raise ValueError('Indexed reference returned an incomplete interval')
    if strand == '-':
        from Bio.Seq import Seq
        return str(Seq(sequence).reverse_complement())
    return sequence


def resolve_motif_resources(args, peaks, stage):
    import pysam
    lib = library()
    sources = []
    assembly = args.get('assembly')
    if not assembly:
        raise ValueError('Establish the peak genome assembly from source metadata or a declaration')
    if args.get('genome_path') and args.get('genome_resource'):
        raise ValueError('Select either an installed genome or a task FASTA')
    if args.get('genome_path'):
        # Existing ordinary source binding supplies this private FASTA. Build only
        # its task-local index; never modify the uploaded source or shared library.
        fasta = Path(args['genome_path'])
        pin = file_closure([fasta])[0]
        if args.get('genome_sha256') and pin['sha256'] != args['genome_sha256']:
            raise AuthorityError('Selected genome bytes changed')
        sources.append(pin)
        local = stage/'genome.fa'; os.symlink(fasta,local)
        pysam.faidx(str(local))
        fasta_path = str(local)
        if not args.get('taxonomy_id'):
            raise ValueError('A task FASTA requires its species taxonomy_id to select appropriate motifs')
        genome_identity = dict(assembly=assembly,taxonomy_id=args['taxonomy_id'],fasta_sha256=pin['sha256'],fai_sha256=file_closure([str(local)+'.fai'])[0]['sha256'],source='task FASTA')
    else:
        candidates = [e for e in lib.entries() if e['kind']=='genome' and e['resource']['assembly']==assembly
            and (not args.get('genome_resource') or args['genome_resource'] in (e['id'],e['label']))]
        if len(candidates)!=1:
            raise ValueError('Choose an unambiguous matching genome or supply a task FASTA; no contig renaming is inferred')
        entry=candidates[0]; genome=lib.check(entry); sources.extend(entry['files'])
        fasta_path=genome['genome']['fasta']['path']
        genome_identity=dict(assembly=assembly,taxonomy_id=genome['species']['taxonomy_id'],
            resource_id=entry['id'],fasta_sha256=genome['genome']['fasta']['sha256'],fai_sha256=genome['genome']['fai']['sha256'])
        if args.get('taxonomy_id') is not None and args['taxonomy_id']!=genome_identity['taxonomy_id']:
            raise ValueError('Declared species differs from selected genome')
    with pysam.FastaFile(fasta_path) as fasta:
        lengths=dict(zip(fasta.references,fasta.lengths))
        invalid=[r.feature_id for r in peaks.itertuples() if r.chrom not in lengths or not 0<=r.start<r.end<=lengths[r.chrom]]
        if invalid:
            raise ValueError(f'{len(invalid)} peaks have absent contigs or invalid bounds in this reference; select a compatible FASTA, without clipping or renaming')
    choices=[e for e in lib.entries() if e['kind']=='motif_collection'
        and genome_identity['taxonomy_id'] in e['resource']['applicable_taxonomy_ids']
        and (not args.get('motif_collection') or args['motif_collection'] in (e['id'],e['label']))]
    if len(choices)!=1:
        raise ValueError('Install or select one versioned, taxon-appropriate motif collection in the scientific resource library')
    motif=lib.check(choices[0]);sources.extend(choices[0]['files'])
    return fasta_path,genome_identity,motif,choices[0],sources


def qualified_genome_entries():
    from agent.tools._native_process import NativeExecutionError
    try:
        return library().entries()
    except NativeExecutionError as exc:
        if exc.code != 'NATIVE_EXECUTION_UNAVAILABLE':
            raise
        return ()


def assembly_labels_match(labels, entries):
    """Only exact qualified FASTA/FAI identity can establish label equivalence."""
    labels = set(labels)
    if len(labels) <= 1:
        return True
    identities = {}
    from agent.schemas.verification_authority import digest
    for entry in entries:
        if entry is None:
            continue
        if entry.get('kind') == 'genome':
            resource = entry['resource']
            genome = resource.get('genome', {})
            identity = tuple(genome.get(k, {}).get('sha256') for k in ('fasta', 'fai'))
            species = resource.get('species', {})
        elif entry.get('component_identity_version', 1) >= 2:
            resource = entry
            components = entry.get('components', {})
            fasta, sizes = components.get('reference', {}), components.get('chrom_sizes', {})
            identity = (fasta.get('sha256'), sizes.get('sha256'), sizes.get('ordered_sizes'))
            species = entry.get('species', {})
        else:
            continue
        if all(identity):
            identities.setdefault((identity, digest(species)), set()).add(resource['assembly'])
    return any(labels <= names for names in identities.values())


def reconcile_assembly(observations, declared=None, *, entries=()):
    from agent.orchestration.scientific_inputs import ScientificDiagnostic
    rows = observations['observations']
    labels = {v for row in rows for v in row['values']}
    if declared is not None:
        labels.add(declared)
    if not assembly_labels_match(labels, entries):
        raise ScientificDiagnostic('SOURCE_ASSEMBLY_CONFLICT',
            'Declared assembly conflicts with positive source assembly observations; no exact qualified reference establishes equivalence.')
    if declared is None and labels and not any(row['complete'] and row['values'] for row in rows):
        return None
    return declared or (sorted(labels)[0] if labels else None)


def metadata_declarations(handle, slot, args, var):
    """Read declarations and corroborate all bounded conventional assembly claims."""
    from agent.tools._file_identity import file_sha256
    effective = dict(args)
    uns = handle.get('uns')
    for key in ('coordinate_system', 'matrix_semantics'):
        if key == 'matrix_semantics' and slot != 'X':
            continue
        if key not in effective and uns is not None and key in uns:
            value = read_elem(uns[key])
            if isinstance(value, str):
                effective[key] = value.replace('-', '_') if key == 'coordinate_system' else value
    observed = assembly_observations(handle.filename, file_sha256(handle.filename))
    # Catalog metadata supplies only qualified identity equivalence, never a default.
    labels = {v for row in observed['observations'] for v in row['values']}
    if args.get('assembly') is not None:
        labels.add(args['assembly'])
    entries = qualified_genome_entries() if len(labels) > 1 else ()
    assembly = reconcile_assembly(observed, args.get('assembly'), entries=entries)
    if assembly is not None:
        effective['assembly'] = assembly
    return effective


def assembly_observations(path, sha256):
    """Bounded source claims with their origin; no reference certification.

    Only conventional assembly metadata is decoded. Count matrices and unrelated
    nested metadata are never read to choose a scientific reference.
    """
    from agent.tools._file_identity import checked_file, file_sha256
    from .peak_h5ad import inspect, metadata_item
    path = checked_file(path, sha256)
    rows = []
    with h5py.File(path, 'r') as handle:
        for name in ('uns/assembly', 'var/genome', 'uns/source/assembly', 'uns/source/reference'):
            item = metadata_item(handle, name)
            if item is None:
                continue
            paths = [name]
            if name == 'var/genome' and not (
                    isinstance(item, h5py.Dataset) and item.ndim == 1 and h5py.check_string_dtype(item.dtype) is not None
                    or 'categories' in item.attrs
                    or item.attrs.get('encoding-type') in ('categorical', 'nullable-string-array')):
                rows.append(dict(origin='var.genome', values=[], complete=False))
                continue
            if name != 'var/genome' and (not isinstance(item, h5py.Dataset) or item.ndim != 0):
                rows.append(dict(origin=name.replace('/', '.'), values=[], complete=False))
                continue
            # Older AnnData categoricals point to a sibling categories array.
            if 'categories' in item.attrs:
                paths.append(item.parent.name.strip('/') + '/' + item.attrs['categories'])
            try:
                inspect(handle, require_matrix=False, metadata_paths=paths, max_logical_bytes=8 * 1024**2)
            except ValueError as exc:
                if 'exceed' not in str(exc):
                    raise
                rows.append(dict(origin=name.replace('/', '.'), values=[], complete=False))
                continue
            from .categorical import _series
            value = _series(item) if name == 'var/genome' else read_elem(item)
            values = list(value) if name == 'var/genome' else [value]
            from .peak_h5ad import _PATH
            def usable_label(value):
                return (isinstance(value, str) and value.strip() == value and bool(value)
                        and len(value) <= 128 and not _PATH.search(value) and not any(ord(c) < 32 for c in value))
            observed = sorted({v for v in values if usable_label(v)})
            usable = bool(values) and all(usable_label(v) for v in values)
            rows.append(dict(origin=name.replace('/', '.'), values=observed,
                             complete=bool(usable)))
    if file_sha256(path) != sha256:
        raise ValueError('Assembly observation source bytes changed.')
    return dict(source_sha256=sha256, observations=rows)



def check_declarations(path, args):
    """Corroborate standard source metadata; never substitute it silently."""
    from agent.tools._file_identity import file_sha256
    observed = assembly_observations(path, file_sha256(path))
    labels = {v for row in observed['observations'] for v in row['values']}
    if args.get('assembly') is not None:
        labels.add(args['assembly'])
    reconcile_assembly(observed, args.get('assembly'), entries=qualified_genome_entries() if len(labels) > 1 else ())
    with h5py.File(path, 'r') as handle:
        uns = handle.get('uns')
        if uns is None:
            return
        for key in ('species', 'coordinate_system'):
            if key in uns and args.get(key) is not None:
                saved = read_elem(uns[key])
                if key == 'coordinate_system' and isinstance(saved, str):
                    saved = saved.replace('-', '_')
                if isinstance(saved, str) and saved != args[key]:
                    raise ValueError(f'Declared {key} conflicts with recorded source metadata')
        if args.get('count_slot', 'X') == 'X' and 'matrix_semantics' in uns and args.get('matrix_semantics'):
            saved = read_elem(uns['matrix_semantics'])
            if isinstance(saved, str) and saved != args['matrix_semantics']:
                raise ValueError('Declared X measurement conflicts with source metadata')


def coordinates(var, args):
    from types import SimpleNamespace
    from . import replicate_pseudobulk as features
    source = args.get('coordinate_source')
    keys = [args.get('feature_'+name+'_key') for name in ('chrom', 'start', 'end')]
    if source == 'var_columns' and not any(keys):
        keys = ['chrom', 'start', 'end']
    chrom, start, end, sha = features._coordinates(SimpleNamespace(var=var, var_names=var.index),
        coordinate_source=source, feature_chrom_key=keys[0], feature_start_key=keys[1],
        feature_end_key=keys[2], coordinate_system=args.get('coordinate_system'))
    if chrom is None:
        raise ValueError('Genomic coordinates and their explicit coordinate convention are required')
    offset = int(args['coordinate_system'] == 'one_based_closed')
    return list(zip(chrom, (x-offset for x in start), end)), sha
