"""Observed scATAC regions to exact reference DNA; no model or DA execution."""
import csv
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem, sparse_dataset

from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools._cancellation import cancellation_checkpoint
from agent.tools.data.authority_context import owned_verification
from agent.tools.data.fastq_verification_authority import file_closure
from . import replicate_pseudobulk as features
from . import sequence_prediction as shared
from .sequence_reference import resolve_genome, extract

TOOL = 'prepare_scatac_regulatory_sequences'
KIND = 'scatac_regulatory_sequences'
CONTRACT = 'scatac-regulatory-sequences.v1'
POLICY = 'scatac-regulatory-sequences-v1'
PROFILE = digest(dict(contract=CONTRACT, selection='explicit; detection_fraction; accepted_DA',
                     extraction='indexed_reference; exact_or_midpoint_expansion; no_clipping'))
LIMITATIONS = ('Detection fractions and their difference are descriptive, not inferential DA or FDR. '
    'Accepted DA statistics apply only to the recorded group and numerator/denominator contrast. '
    'Group labels are supplied metadata or accepted annotations, not validated cell identities. '
    'Reference DNA and observed accessibility do not establish enhancer activity. No sequence model '
    'is invoked here; downstream model conditions, species, lengths and measurements remain independent. '
    'A midpoint is a geometric anchor, not a measured summit; unstranded regions use reference plus orientation.')
FILES = ('sequence-records.json', 'sequences.fasta', 'regions.bed', 'windows.bed', 'selection.tsv', 'metadata.json')
OUTPUTS = (('sequences', 'sequences.fasta'), ('selection', 'selection.tsv'))
FACT_FIELDS = ('n_sequences', 'selection_method', 'selection_parameters', 'source_identity', 'species',
    'assembly', 'reference_identity', 'coordinate_semantics', 'window_policy', 'group_context',
    'candidate_examples', 'examples_omitted', 'limitations')
RESULT_FIELDS = dict(status=(str,), contract_version=(str,), manifest_path=(str,), manifest_sha256=(str,),
    sequences_path=(str,), sequences_sha256=(str,), selection_path=(str,), selection_sha256=(str,),
    n_sequences=(int,), selection_method=(str,), selection_parameters=(dict,), source_identity=(dict,),
    species=(str,), assembly=(str,), reference_identity=(dict,), coordinate_semantics=(str,), window_policy=(str,),
    group_context=(dict,), candidate_examples=(list,), examples_omitted=(int,), limitations=(str,))
TABLE_FIELDS = ('sequence_id', 'feature_id', 'feature_index', 'chrom', 'peak_start', 'peak_end',
    'window_start', 'window_end', 'strand', 'orientation', 'anchor', 'length', 'sequence_sha256',
    'group_cells', 'accessible_cells', 'detection_fraction', 'comparison_cells', 'comparison_accessible_cells',
    'comparison_fraction', 'fraction_difference', 'logFC', 'PValue', 'FDR')


def _pin(path, expected=None):
    pin = file_closure([path])[0]
    if expected is not None and pin['sha256'] != expected:
        raise AuthorityError('Selected source or recorded lineage bytes differ')
    return pin


def _metadata(path):
    from . import peak_h5ad
    # This owner streams matrix chunks; annotation's in-memory matrix budget
    # does not apply. The shared storage/link/filter checks still do.
    peak_h5ad.inspect(path, require_matrix=False, max_logical_bytes=None)
    with h5py.File(path, 'r') as handle:
        return read_elem(handle['obs']), read_elem(handle['var'])


def _check_declarations(path, args):
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


def _coordinates(var, args):
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


def _annotation_groups(args, cells, source_pin):
    from . import scatac_annotation, annotation_contract
    path, sha = args['annotation_manifest_path'], args['annotation_manifest_sha256']
    scatac_annotation.verify_annotation(path, expected_sha256=sha)
    manifest = annotation_contract.load_manifest(path, sha)
    binding = manifest['matrix_binding']
    if binding.get('source_sha256', binding.get('matrix_sha256')) != source_pin['sha256']:
        raise ValueError('Accepted annotation does not identify this exact original matrix')
    rows_path = Path(path).parent / 'primary/cells.json'
    rows = shared.read(rows_path, manifest['sidecars']['primary/cells.json']['sha256'])
    if [r['cell_id'] for r in rows] != list(cells):
        raise ValueError('Annotation and source ordered cell identities differ')
    return np.asarray([r['primary_annotation'] for r in rows], dtype=object), [
        _pin(path, sha), _pin(rows_path, manifest['sidecars']['primary/cells.json']['sha256'])]


def _descriptive(args, obs, var, source_pin):
    semantics = args.get('matrix_semantics')
    if semantics not in ('peak_counts', 'fragment_counts', 'insertion_counts', 'binary_accessibility'):
        raise ValueError('Declare the actual raw peak/fragment/insertion counts or binary accessibility; transformed data are not counts')
    cells = features._strict_ids(obs.index, 'cell')
    group, comparison = args.get('group_value'), args.get('comparison_group')
    sources = []
    if args.get('annotation_manifest_path'):
        if args.get('group_key'):
            raise ValueError('Select one grouping source, obs metadata or accepted annotation')
        labels, sources = _annotation_groups(args, cells, source_pin)
    elif args.get('group_key'):
        key = args['group_key']
        if key not in obs:
            raise ValueError('Selected grouping column is absent')
        # Unassigned cells remain in the original axis; they simply do not match
        # an explicitly named group. Never rename them to a biological class.
        values = obs[key].astype(object).where(obs[key].notna(), None).tolist()
        if any(v is not None and (not isinstance(v, str) or not v.strip() or v != v.strip()) for v in values):
            raise ValueError('Group metadata must contain exact text labels or missing assignments')
        labels = np.asarray(values, dtype=object)
    else:
        labels = None
    if (labels is None) != (group is None) or comparison is not None and (group is None or comparison == group):
        raise ValueError('A group needs an exact grouping source and value; comparison must be distinct')
    selected = np.ones(len(cells), dtype=bool) if labels is None else labels == group
    compared = np.zeros(len(cells), dtype=bool) if comparison is None else labels == comparison
    n, m = int(selected.sum()), int(compared.sum())
    if n == 0 or comparison is not None and m == 0:
        raise ValueError('Requested group has no cells; no cell-type substitution is permitted')
    counts, other = np.zeros(len(var), dtype=np.int64), np.zeros(len(var), dtype=np.int64)
    slot = args.get('count_slot', 'X')
    if slot != 'X' and not slot.startswith('layers/'):
        raise ValueError('Count slot must be X or layers/exact_name')
    with h5py.File(source_pin['path'], 'r') as handle:
        if slot not in handle:
            raise ValueError('Selected matrix slot is absent')
        element = handle[slot]
        matrix = sparse_dataset(element) if isinstance(element, h5py.Group) else element
        if matrix.shape != (len(obs), len(var)):
            raise ValueError('Matrix axes differ from cell/feature metadata')
        # Bound dense chunks by entries, without materializing the entire matrix.
        chunk_size = 256 if isinstance(element, h5py.Group) else max(1, min(4096, 1_000_000 // max(1, len(var))))
        for first in range(0, len(obs), chunk_size):
            cancellation_checkpoint()
            stop = min(len(obs), first + chunk_size)
            chunk = features._canonical_integer_chunk(matrix, first, stop, semantics)
            chunk.data[:] = 1
            counts += np.asarray(chunk[selected[first:stop]].sum(axis=0)).ravel().astype(np.int64)
            if m:
                other += np.asarray(chunk[compared[first:stop]].sum(axis=0)).ravel().astype(np.int64)
    stats = pd.DataFrame(dict(group_cells=n, accessible_cells=counts, detection_fraction=counts/n), index=var.index)
    if m:
        stats['comparison_cells'], stats['comparison_accessible_cells'] = m, other
        stats['comparison_fraction'] = other/m
        stats['fraction_difference'] = counts/n - other/m
    threshold = args.get('min_fraction', 0.0)
    if isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Detection threshold must be between zero and one')
    eligible = np.flatnonzero((counts > 0) & (counts/n >= threshold))
    ranking = stats['fraction_difference' if m else 'detection_fraction'].to_numpy()
    order = sorted(eligible, key=lambda i: (-ranking[i], -counts[i]/n, int(i)))
    return order, stats, dict(group_key=args.get('group_key'), group_value=group, comparison_group=comparison,
        grouping='accepted_annotation' if args.get('annotation_manifest_path') else 'obs' if labels is not None else 'all_cells',
        n_group_cells=n, n_comparison_cells=m), sources


def _da(args):
    from .differential_accessibility_backend import DA_PROVENANCE_KEY, DA_ARTIFACT_TYPE
    path = args['da_path']; source = _pin(path)
    with h5py.File(path, 'r') as handle:
        var = read_elem(handle['var']); provenance = read_elem(handle['uns'])[DA_PROVENANCE_KEY]
    if provenance['artifact_type'] != DA_ARTIFACT_TYPE or provenance['schema_version'] not in (1, 2):
        raise ValueError('Unsupported differential accessibility publication')
    upstream = provenance['source']
    pb_pin = _pin(upstream['pseudobulk_path'], upstream['pseudobulk_sha256'])
    with h5py.File(pb_pin['path'], 'r') as handle:
        pb_var = read_elem(handle['var']); pb = read_elem(handle['uns'])[features.PSEUDOBULK_PROVENANCE_KEY]
    feature_pin = _pin(pb['source']['feature_space_path'], pb['source']['feature_space_sha256'])
    _, manifest, _ = features._load_feature_manifest(feature_pin['path'])
    if (manifest['identities']['feature_space_identity_sha256'] != upstream['feature_space_identity_sha256']
            or pb['source']['feature_space_identity_sha256'] != upstream['feature_space_identity_sha256']
            or manifest['identities']['feature_ids_sha256'] != upstream['feature_ids_sha256']
            or features._ordered_strings_digest(features._strict_ids(var.index, 'feature'),
                domain='agent.regulatory-feature-identities.v1') != upstream['feature_ids_sha256']
            or not var.index.equals(pb_var.index)):
        raise ValueError('DA, pseudobulk and feature-space identities differ')
    lineage_sources = [source, pb_pin, feature_pin]
    if manifest['coordinates']['available']:
        for key in ('chrom', 'start', 'end'):
            if key not in var or not var[key].equals(pb_var[key]):
                raise ValueError('DA genomic coordinates differ from its exact pseudobulk source')
        convention = manifest['coordinates']['coordinate_system']
        if args.get('coordinate_system', convention) != convention:
            raise ValueError('Requested coordinate convention conflicts with accepted feature-space lineage')
        coordinate_args = dict(coordinate_source='var_columns', coordinate_system=convention)
        intervals, coordinate_sha = _coordinates(var, coordinate_args)
        if coordinate_sha != manifest['coordinates']['coordinates_sha256']:
            raise ValueError('DA coordinates differ from the feature-space coordinate identity')
    else:
        # Old DA may have been computed without requesting coordinates. Follow
        # its exact recorded source, not a neighboring filename or row count.
        if not args.get('coordinate_source') or not args.get('coordinate_system'):
            raise ValueError('Accepted DA lineage lacks coordinates; declare the original feature encoding and coordinate convention')
        original = _pin(manifest['source']['input_path'], manifest['source']['source_h5ad_sha256'])
        _, original_var = _metadata(original['path'])
        if not original_var.index.equals(var.index):
            raise ValueError('Original genomic features differ from the accepted DA feature axis')
        _check_declarations(original['path'], args)
        intervals, coordinate_sha = _coordinates(original_var, args)
        lineage_sources.append(original)
    biology = manifest['biology']
    for key, value in (('species', biology['species']), ('assembly', biology['genome_assembly'])):
        if args.get(key, value) != value:
            raise ValueError('Requested biology conflicts with accepted DA lineage')
    contrast = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in provenance['comparison'].items()}
    for key in ('numerator_condition', 'denominator_condition'):
        if args.get(key) != contrast[key]:
            raise ValueError('Select the exact recorded numerator and denominator; a condition contrast is not a cell-type comparison')
    cutoff, minimum, direction = args.get('max_fdr', .05), args.get('min_abs_logfc', 0.0), args.get('direction', 'numerator')
    if (isinstance(cutoff, bool) or not np.isfinite(cutoff) or not 0 <= cutoff <= 1
            or isinstance(minimum, bool) or not np.isfinite(minimum) or minimum < 0
            or direction not in ('numerator', 'denominator')):
        raise ValueError('Invalid DA FDR, effect threshold or direction')
    signed = var.logFC.to_numpy() * (1 if direction == 'numerator' else -1)
    mask = (var.da_status == 'tested').to_numpy() & (var.FDR.to_numpy() <= cutoff) & (signed > minimum)
    indices = np.flatnonzero(mask)
    if not np.isfinite(var.iloc[indices][['logFC','PValue','FDR']].to_numpy()).all():
        raise ValueError('Selected DA statistics are nonfinite')
    order = sorted(indices, key=lambda i: (float(var.FDR.iloc[i]), -signed[i], int(i)))
    identity = dict(source, feature_space_identity_sha256=upstream['feature_space_identity_sha256'],
        feature_ids_sha256=upstream['feature_ids_sha256'], original_source=manifest['source'])
    identity['coordinates_sha256'] = coordinate_sha
    return var, intervals, order, var[['logFC','PValue','FDR']], contrast, lineage_sources, identity, biology


def execute(values, execution_identity=None):
    args = shared.arguments(values)
    method = args.get('selection_method', 'descriptive')
    if method != 'accepted_candidates' and any(k in args for k in ('candidate_manifest_path','candidate_manifest_sha256','candidate_index')):
        raise ValueError('Accepted candidate inputs require accepted_candidates selection')
    top = args.get('top_n', 10)
    if type(top) is not int or not 1 <= top <= 1000:
        raise ValueError('Select between 1 and 1000 candidates per preparation')
    parent_reference = None
    if method == 'accepted_candidates':
        if not args.get('candidate_manifest_path') or not args.get('candidate_manifest_sha256'):
            raise ValueError('Select an exact accepted candidate-region result for continuation')
        if any(k in args for k in ('input_path','da_path','group_key','group_value','comparison_group','feature_ids',
                'annotation_manifest_path','annotation_manifest_sha256','input_sha256','matrix_semantics','count_slot',
                'min_fraction','max_fdr','direction','min_abs_logfc','numerator_condition','denominator_condition',
                'coordinate_source','coordinate_system','feature_chrom_key','feature_start_key','feature_end_key','strand_key')):
            raise ValueError('Accepted candidate continuation cannot change upstream selection or measurements')
        path, sha = args['candidate_manifest_path'], args['candidate_manifest_sha256']
        verify_preparation(path, expected_sha256=sha)
        manifest, _ = shared.load(path, sha)
        record_pin = next(r for r in manifest['files'] if Path(r['path']).name == 'sequence-records.json')
        prior_rows = shared.read(record_pin['path'], record_pin['sha256'])
        index = args.get('candidate_index')
        if index is not None and (type(index) is not int or not 1 <= index <= len(prior_rows)):
            raise ValueError('candidate_index must identify an existing 1-based accepted candidate')
        order = list(range(len(prior_rows))) if index is None else [index-1]
        var = pd.DataFrame(index=[r['feature_id'] for r in prior_rows])
        var['_source_strand'] = [r['strand'] for r in prior_rows]
        intervals = [(r['chrom'],r['peak_start'],r['peak_end']) for r in prior_rows]
        stats = pd.DataFrame([{k:v for k,v in r.items() if k in TABLE_FIELDS[TABLE_FIELDS.index('group_cells'):]} for r in prior_rows], index=var.index)
        prior = manifest['result']
        groups, species, assembly = prior['group_context'], prior['species'], prior['assembly']
        for key, value in (('species',species),('assembly',assembly)):
            if args.get(key,value) != value:
                raise ValueError('Continuation biology conflicts with accepted candidate ancestry')
        identity = dict(parent_manifest_sha256=sha, sha256=sha, original_source=prior['source_identity'],
            original_selection_method=prior['selection_method'], original_selection_parameters=prior['selection_parameters'])
        parent_reference = prior['reference_identity']
        sources = [_pin(path,sha),_pin(record_pin['path'],record_pin['sha256'])]
    elif method == 'differential_accessibility':
        if not args.get('da_path'):
            raise ValueError('Select an accepted replicate-aware differential accessibility result')
        if args.get('input_path') or any(k in args for k in ('group_key','group_value','comparison_group','feature_ids','matrix_semantics','count_slot','annotation_manifest_path')):
            raise ValueError('DA selection consumes its exact accepted result, without a second dataset or descriptive grouping')
        var, intervals, order, stats, groups, sources, identity, biology = _da(args)
        species, assembly = biology['species'], biology['genome_assembly']
    else:
        if method not in ('explicit', 'descriptive') or args.get('da_path'):
            raise ValueError('Choose explicit, descriptive or accepted differential_accessibility selection')
        if any(k in args for k in ('max_fdr','min_abs_logfc','direction','numerator_condition','denominator_condition')):
            raise ValueError('DA thresholds and contrasts cannot apply to descriptive accessibility')
        if not args.get('input_path'):
            raise ValueError('Supply the scATAC H5AD whose genomic features should be selected')
        pin = _pin(args['input_path'], args.get('input_sha256'))
        _check_declarations(pin['path'], args)
        obs, var = _metadata(pin['path'])
        ids = features._strict_ids(var.index, 'feature')
        intervals, coordinate_sha = _coordinates(var, args)
        identity = dict(pin, feature_ids_sha256=features._ordered_strings_digest(ids,
            domain='agent.regulatory-feature-identities.v1'), coordinates_sha256=coordinate_sha)
        species, assembly = args.get('species'), args.get('assembly')
        if not species or not assembly:
            raise ValueError('Declare species and genome assembly before genomic extraction')
        sources = [pin]
        if method == 'explicit':
            if any(k in args for k in ('group_key','group_value','comparison_group','annotation_manifest_path','min_fraction')):
                raise ValueError('Exact feature extraction does not select by groups or descriptive thresholds')
            selected = [v.strip() for v in args.get('feature_ids', '').split(',')]
            if not selected or any(v not in ids for v in selected) or len(set(selected)) != len(selected):
                raise ValueError('Select exact distinct feature IDs present in this dataset')
            if len(selected) > top:
                raise ValueError('top_n is smaller than the explicit requested feature set')
            lookup = {v:i for i,v in enumerate(ids)}
            order, stats, groups = [lookup[v] for v in selected], pd.DataFrame(index=var.index), {}
        else:
            if args.get('feature_ids'):
                raise ValueError('Use explicit selection for named features')
            order, stats, groups, grouping_sources = _descriptive(args, obs, var, pin)
            sources += grouping_sources
            identity.update(cell_ids_sha256=features._ordered_strings_digest(tuple(map(str,obs.index)),
                domain='agent.regulatory-cell-identities.v1'), count_slot=args.get('count_slot','X'),
                matrix_semantics=args['matrix_semantics'])
    maximum = args.get('max_peak_length')
    if maximum is not None:
        if type(maximum) is not int or maximum < 1:
            raise ValueError('Maximum source peak length must be a positive integer')
        if method == 'explicit' and any(intervals[i][2]-intervals[i][1] > maximum for i in order):
            raise ValueError('An explicitly selected peak exceeds the requested maximum length')
        order = [i for i in order if intervals[i][2]-intervals[i][1] <= maximum]
    order = order[:top]
    if not order:
        raise ValueError('No regions satisfy the selected criteria; thresholds were not relaxed')
    genome, entry = resolve_genome(assembly, species)
    if parent_reference is not None and genome['genome']['fasta']['sha256'] != parent_reference['fasta_sha256']:
        raise ValueError('Continuation reference differs from the exact accepted genome')
    sources += entry['files']
    window = args.get('window_size')
    if window is not None and (type(window) is not int or window < 1):
        raise ValueError('Genomic window size must be a positive integer')
    policy = 'exact_peak_interval' if window is None else 'peak_midpoint_floor; start=midpoint-floor(window_size/2); real_reference_expansion'
    rows = []
    import pysam
    with pysam.FastaFile(genome['genome']['fasta']['path'], filepath_index=genome['genome']['fai']['path']) as fasta:
        for ordinal, index in enumerate(order, 1):
            cancellation_checkpoint()
            chrom, start, end = intervals[index]
            strand_key = '_source_strand' if method == 'accepted_candidates' else args.get('strand_key')
            if strand_key and strand_key not in var:
                raise ValueError('Declared strand column is absent')
            strand = str(var.iloc[index][strand_key]) if strand_key else '.'
            anchor = (start+end)//2
            left = start if window is None else anchor-window//2
            right = end if window is None else left+window
            if left > start or right < end:
                raise ValueError('Requested context does not contain the source peak; silent cropping is not allowed')
            sequence = extract(fasta, chrom, int(left), int(right), strand)
            if set(sequence)-set('ACGT'):
                raise ValueError('Reference interval contains unresolved bases; no base substitution is permitted')
            row = dict(sequence_id=prior_rows[index]['sequence_id'] if method == 'accepted_candidates' else f'sequence-{ordinal}',
                feature_id=str(var.index[index]), feature_index=prior_rows[index]['feature_index'] if method == 'accepted_candidates' else int(index),
                chrom=chrom, peak_start=int(start), peak_end=int(end), window_start=int(left), window_end=int(right),
                strand=strand, orientation='reverse_complement' if strand == '-' else 'reference_plus', anchor=anchor,
                length=len(sequence), sequence_sha256=hashlib.sha256(sequence.encode('ascii')).hexdigest())
            row.update({k: v.item() if isinstance(v,np.generic) else v for k,v in stats.iloc[index].items()})
            rows.append(dict(row, sequence=sequence, original_header=row['feature_id']))
    reference = dict(assembly=assembly, species=genome['species'], fasta_sha256=genome['genome']['fasta']['sha256'])
    parameters = dict(top_n=top, max_peak_length=maximum)
    if method == 'descriptive':
        parameters.update(min_fraction=args.get('min_fraction',0.0), ranking='descending fraction_difference then detection_fraction then source feature order' if args.get('comparison_group') else 'descending detection_fraction then source feature order')
    elif method == 'differential_accessibility':
        parameters.update(max_fdr=args.get('max_fdr',.05), min_abs_logfc=args.get('min_abs_logfc',0.0),
            direction=args.get('direction','numerator'), ranking='ascending FDR then descending directional logFC then source feature order')
    examples = [{k:v for k,v in r.items() if k not in ('sequence','original_header')} for r in rows]
    summary = dict(status='success', contract_version=CONTRACT, n_sequences=len(rows), selection_method=method,
        selection_parameters=parameters, source_identity=identity, species=species, assembly=assembly,
        reference_identity=reference, coordinate_semantics='published BED and windows: zero_based_half_open',
        window_policy=policy, group_context=groups, candidate_examples=examples[:12], examples_omitted=max(0,len(rows)-12), limitations=LIMITATIONS)
    root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Preparation output is not empty; no overwrite or automatic retry')
    sources = sorted({r['path']:r for r in sources}.values(), key=lambda r:r['path'])
    with tempfile.TemporaryDirectory(prefix='.regions-', dir=root) as temporary:
        stage = Path(temporary)
        shared.write(stage/'sequence-records.json', rows)
        shared.write(stage/'metadata.json', dict(summary, candidates=examples, source_coordinate_system=args.get('coordinate_system'),
            source_sha256=identity['sha256'], reference=reference))
        with (stage/'sequences.fasta').open('x') as stream:
            for row in rows: stream.write(f">{row['sequence_id']} {row['feature_id']}\n{row['sequence']}\n")
        with (stage/'selection.tsv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS, delimiter='\t', extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows)
        for name, prefix in (('regions.bed','peak'),('windows.bed','window')):
            with (stage/name).open('x') as stream:
                for row in rows:
                    stream.write(f"{row['chrom']}\t{row[prefix+'_start']}\t{row[prefix+'_end']}\t{row['sequence_id']}\t0\t{row['strand']}\n")
        if file_closure(r['path'] for r in sources) != sources:
            raise AuthorityError('Scientific source changed during preparation')
        for name in FILES: os.link(stage/name, root/name)
    return shared.publish(args, execution_identity, KIND, CONTRACT, summary, sources, FILES,
        dict(reference=genome, runtime=dict(completion='succeeded', method='pysam indexed reference extraction; no model inference')))


@owned_verification(KIND)
def verify_preparation(path, *, expected_sha256):
    return shared._verify(path, expected_sha256, KIND)


def prepare_scatac_regulatory_sequences(output_dir, input_path=None, input_sha256=None, da_path=None,
        selection_method=None, species=None, assembly=None, coordinate_source=None, coordinate_system=None,
        feature_chrom_key=None, feature_start_key=None, feature_end_key=None, strand_key=None, feature_ids=None,
        matrix_semantics=None, count_slot=None, group_key=None, group_value=None, comparison_group=None,
        annotation_manifest_path=None, annotation_manifest_sha256=None, top_n=None, min_fraction=None,
        window_size=None, numerator_condition=None, denominator_condition=None, direction=None, max_fdr=None,
        min_abs_logfc=None, max_peak_length=None, candidate_manifest_path=None, candidate_manifest_sha256=None,
        candidate_index=None):
    return execute({k:v for k,v in locals().items() if v is not None})
