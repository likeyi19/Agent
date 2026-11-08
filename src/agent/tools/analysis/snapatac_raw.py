"""File handoffs for native SnapATAC2 preprocessing, without scientific replay.

The package owns parsing, filtering, QC, fragment construction and counting.
These functions join the existing run_snapatac2 catalog and publication owner.
"""
from pathlib import Path
import shutil
import tempfile

import h5py
from anndata.io import read_elem

from agent.schemas.verification_authority import AuthorityError
from agent.tools._cancellation import cancellation_checkpoint
from agent.tools.data.authority_context import current
from agent.tools._file_identity import file_closure

FUNCTIONS = {
    'pp.make_fragment_file': ('is_paired', 'barcode_tag', 'barcode_regex', 'umi_tag', 'umi_regex',
        'shift_left', 'shift_right', 'min_mapq', 'chunk_size', 'chrM', 'source', 'compression', 'compression_level'),
    'pp.import_fragments': ('is_paired', 'min_num_fragments', 'sorted_by_barcode', 'chrM', 'chunk_size', 'n_jobs', 'chrom_sizes', 'whitelist'),
    'metrics.tsse': ('exclude_chroms', 'inplace', 'n_jobs'),
    'metrics.frag_size_distr': ('max_recorded_size', 'add_key', 'inplace', 'n_jobs'),
    'pp.filter_cells': ('min_counts', 'min_tsse', 'max_counts', 'max_tsse', 'inplace', 'n_jobs'),
    'pp.make_peak_matrix': ('chunk_size', 'use_x', 'min_frag_size', 'max_frag_size', 'counting_strategy',
        'value_type', 'summary_type', 'use_rep', 'inplace'),
}
PORTS = {
    'chrom_sizes': ('chrom_sizes_path', 'chrom_sizes_sha256', 'FAI or two-column chromosome sizes used by native fragment import.'),
    'gene_anno': ('gene_anno_path', 'gene_anno_sha256', 'GTF/GFF annotation used by native TSS enrichment.'),
    'peaks': ('peaks_path', 'peaks_sha256', 'BED of the exact requested peak/cCRE/region features, in source coordinates.'),
    'whitelist': ('whitelist_path', 'whitelist_sha256', 'Optional barcode whitelist for native import; no implicit cell calling.'),
}
REQUIRED = {'pp.import_fragments': ('chrom_sizes',), 'metrics.tsse': ('gene_anno',)}
APPLIES = {'pp.import_fragments': ('chrom_sizes', 'whitelist'), 'metrics.tsse': ('gene_anno',),
           'pp.make_peak_matrix': ('peaks',)}
RAW_FILES = {
    'fragments.tsv.gz': ('Native BAM-derived fragments; exact method and support semantics in metadata', 'application/gzip'),
    'fragments.tsv': ('Native BAM-derived fragments; uncompressed native output', 'text/tab-separated-values'),
    'fragments.tsv.zst': ('Native BAM-derived fragments; Zstandard native output', 'application/zstd'),
}
LIMITATIONS = ('Native SnapATAC2 preprocessing, not biological validation. Barcodes are identifiers, not independently established cells. '
    'Import excludes records outside supplied chromosome sizes. Native defaults are min_num_fragments=200 and sorted_by_barcode=True; '
    'task input-order selection is recorded separately and explicit sorted_by_barcode kwargs prevail. '
    'The imported barcode population reflects native import retention, distinct from later filter_cells selection. '
    'BAM filtering/deduplication and endpoint shifts follow the recorded native method, not historical Agent BAM policies; duplicate-flagged reads are excluded. '
    'Native import n_fragment excludes mitochondrial records; count matrices do not weight column-five duplicate support. '
    'Default make_peak_matrix counts paired insertions, not fragment-body overlaps. Native filter_cells defaults to min_counts=1000 and min_tsse=5. '
    'Old Agent TSS enrichment and selection thresholds are different methods. Native fragment-size distribution is library-level, not per-cell nucleosome signal. '
    'Supplied species/assembly and barcode/coordinate history are declarations; no inferred reference correspondence or relabelling.')


def fragment_name(args):
    compression = args.get('kwargs', {}).get('compression')
    return 'fragments.tsv.zst' if compression == 'zstandard' else 'fragments.tsv' if compression is None else 'fragments.tsv.gz'


def files(value):
    args = value['arguments']
    return (fragment_name(args), 'metadata.json', 'native-receipt.json') if args['function_name'] == 'pp.make_fragment_file' else ('result.h5ad', 'metadata.json', 'native-receipt.json')


def outputs(value):
    bam = value['arguments']['function_name'] == 'pp.make_fragment_file'
    return dict(matrix=None if bam else 'result.h5ad', fragments=fragment_name(value['arguments']) if bam else None)


def input_paths(args):
    name = args['function_name']
    paths = [(args['input_path'], args.get('input_sha256'))]
    for port, (path, sha, _) in PORTS.items():
        if args.get(path) is not None:
            if port not in APPLIES.get(name, ()):
                raise ValueError(f'{port} is not an input of snapatac2.{name}')
            paths.append((args[path], args.get(sha)))
        elif args.get(sha) is not None or port in REQUIRED.get(name, ()) and port not in args.get('kwargs', {}):
            raise ValueError(f'snapatac2.{name} requires the {port} input file')
    if args.get('count_slot') is not None:
        raise ValueError('Native raw functions consume their file object; count_slot applies to matrix functions')
    kw = args.get('kwargs', {})
    if 'chrom_sizes' in kw and (type(kw['chrom_sizes']) is not dict or args.get('chrom_sizes_path')):
        raise ValueError('Use one native chromosome-size dictionary or its exact file port')
    if 'whitelist' in kw and (type(kw['whitelist']) is not list or args.get('whitelist_path')):
        raise ValueError('Use one native barcode list or its exact whitelist file port')
    return paths


def axes(path):
    """Only axes/metadata, never load a genome-sized fragment obsm into memory."""
    from . import peak_h5ad
    # Native imports with zero features legitimately omit var and X. Ordinary
    # matrix consumers still require X; this is a fragment-storage boundary.
    # Walk all storage/link declarations before native code can follow them.
    # Numeric fragment arrays are bounded by shape without loading their values.
    peak_h5ad.inspect(path, require_matrix=False, max_logical_bytes=None)
    with h5py.File(path, 'r') as handle:
        obs = read_elem(handle['obs'])
        if 'X' not in handle and not {'fragment_paired', 'fragment_single'}.intersection(handle.get('obsm', {})):
            raise AuthorityError('Native result has neither a matrix nor native fragment storage')
        return obs.index.tolist(), read_elem(handle['var']).index.tolist() if 'var' in handle else []


def execute(owner, args, execution_identity):
    from .native_matrix import arguments
    from .sequence_prediction import write
    args = arguments(args)
    kwargs = owner.native_arguments(args)
    sources = []
    for path, sha in input_paths(args):
        pin = file_closure([path])[0]
        if sha is not None and pin['sha256'] != sha:
            raise AuthorityError('Selected native input bytes changed')
        sources.append(pin)
    root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Output directory is not empty; retry requires a new authorized execution')
    with tempfile.TemporaryDirectory(prefix='.native-snapatac2-', dir=root) as temporary:
        stage = Path(temporary)
        name = args['function_name']
        before = None if name in ('pp.make_fragment_file', 'pp.import_fragments') else axes(args['input_path'])
        receipt = owner.native(args, stage, None)
        is_bam = name == 'pp.make_fragment_file'
        payload = fragment_name(args) if is_bam else 'result.h5ad'
        if not (stage / payload).is_file():
            raise AuthorityError('Native function did not produce its declared output')
        cells, features = (None, None) if is_bam else axes(stage / payload)
        if before is not None:
            prior_cells, prior_features = before
            if name == 'pp.filter_cells' and kwargs.get('inplace', True):
                positions = iter(prior_cells)
                if any(not any(old == cell for old in positions) for cell in cells):
                    raise AuthorityError('Native filtering changed ordered source cell identities')
            elif cells != prior_cells:
                raise AuthorityError('Native function changed ordered source cell identities')
            if name != 'pp.make_peak_matrix' and features != prior_features:
                raise AuthorityError('Native QC/filtering changed feature identities')
        source = sources[0]
        summary = dict(status='success', contract_version=owner.CONTRACT,
            function_name='snapatac2.' + name, n_cells=None if is_bam else len(cells),
            n_features=None if is_bam else len(features), measurement=receipt['measurement'],
            parameters=dict(kwargs=args.get('kwargs', {}), omitted_optional_arguments=(
                'native defaults except recorded Agent input-order selection' if args.get('fragment_order') is not None else 'native defaults'),
                effective_native_parameters=receipt['effective_parameters'], output_storage='native H5AD or fragment file'),
            source_identity=dict(sha256=source['sha256'], size_bytes=source['size_bytes'],
                references=[dict(role=port, sha256=next(p['sha256'] for p in sources if p['path'] == args[path]))
                    for port, (path, _, _) in PORTS.items() if args.get(path)],
                declarations={key: args[key] for key in ('species', 'assembly', 'matrix_semantics') if key in args}),
            software_versions=receipt['versions'], native_output_type=receipt['native_output_type'],
            result_slots=receipt['result_slots'], limitations=LIMITATIONS)
        if args.get('fragment_order') is not None:
            summary['parameters']['fragment_order'] = args['fragment_order']
            summary['parameters']['agent_selected_parameters'] = {
                key: value for key, value in kwargs.items() if key not in args.get('kwargs', {})}
        if args.get('counting_strategy') is not None:
            # A supplied scalar may be a user declaration or registered Planner
            # choice. The captured request/plan owns that attribution; do not
            # mislabel it as a native default or synthetic user instruction.
            summary['parameters']['counting_strategy'] = args['counting_strategy']
            summary['parameters']['omitted_optional_arguments'] = 'native defaults except recorded counting strategy'
        if name == 'pp.import_fragments':
            summary['parameters']['population'] = dict(stage='native_import', retained_barcodes=len(cells),
                min_num_fragments=receipt['effective_parameters']['min_num_fragments'],
                pre_import_barcode_count=None, reason='Native import retention; Agent did not separately count the source barcode population')
        elif name == 'pp.filter_cells':
            summary['parameters']['population'] = dict(stage='native_cell_filter', input_barcodes=len(before[0]),
                selected_barcodes=len(cells) if kwargs.get('inplace', True) else receipt['selected_count'],
                output_barcodes=len(cells), applied=kwargs.get('inplace', True))
        expected = {r['path']: r for r in [*sources, *receipt['implementation']]}
        sources = file_closure(expected)
        if sources != sorted(expected.values(), key=lambda r: r['path']):
            raise AuthorityError('Source or implementation changed during native preprocessing')
        cancellation_checkpoint()
        shutil.copyfile(stage / payload, root / payload)
        write(root / 'native-receipt.json', receipt)
        write(root / 'metadata.json', dict(summary=summary, receipt=receipt, sources=sources))
    value = dict(contract=owner.CONTRACT, profile=owner.PROFILE, arguments=args, execution_identity=execution_identity,
        result=summary, sources=sources, native_completion='succeeded')
    value['files'] = file_closure(root / name for name in files(value))
    cancellation_checkpoint()
    path = root / 'manifest.json'; sha = write(path, value)
    if current() is not None:
        current().register_execution(owner.KIND, root, execution_identity)
    return owner.verify_publication(path, expected_sha256=sha)
