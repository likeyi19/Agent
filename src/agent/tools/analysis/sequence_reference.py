"""Shared qualified genomes, genomic declarations and exact interval extraction."""
import os
import h5py
from pathlib import Path
from anndata.io import read_elem
from agent.schemas.verification_authority import AuthorityError
from agent.tools.data.fastq_verification_authority import file_closure
from .sequence_prediction import library


def resolve_genome(assembly, species=None):
    from .sequence_prediction import library
    lib = library()
    matches = [e for e in lib.entries() if e['kind'] == 'genome'
               and e['resource']['assembly'] == assembly]
    if len(matches) != 1:
        raise ValueError('An explicit assembly with a unique qualified matching genome is required')
    genome = lib.check(matches[0])
    identity = genome['species']
    names = {str(v) for v in identity.values()}
    names.update({9606: ('human',), 10090: ('mouse',), 7955: ('zebrafish',)}.get(identity['taxonomy_id'], ()))
    if species is not None and species not in names:
        raise ValueError('Declared species differs from the qualified reference genome')
    return genome, matches[0]


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


def metadata_declarations(handle, slot, args, var):
    """Read explicit AnnData metadata once; observations and execution agree."""
    effective=dict(args);uns=handle.get('uns')
    for key in ('assembly','coordinate_system','matrix_semantics'):
        if key=='matrix_semantics' and slot!='X':continue
        if key not in effective and uns is not None and key in uns:
            value=read_elem(uns[key])
            if isinstance(value,str):effective[key]=value.replace('-','_') if key=='coordinate_system' else value
    if 'assembly' not in effective and 'genome' in var:
        values=var['genome'].dropna().unique()
        if len(values)==1 and isinstance(values[0],str):effective['assembly']=values[0]
    return effective



def check_declarations(path, args):
    """Corroborate standard source metadata; never substitute it silently."""
    with h5py.File(path, 'r') as handle:
        uns = handle.get('uns')
        if uns is None:
            return
        for key in ('species', 'assembly', 'coordinate_system'):
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
