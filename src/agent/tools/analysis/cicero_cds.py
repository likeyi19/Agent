"""Focused Cicero CellDataSet exchange, never executable R serialization.

The bundle represents exprs, pData and fData for the reviewed Cicero functions;
it is not a general Monocle fitted-object serializer.
"""
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import anndata as ad
import numpy as np
from scipy.io import mmwrite

from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools._native_process import run, r_environment
from agent.tools.data.authority_context import current
from agent.tools._file_identity import file_closure
from . import peak_h5ad
from .native_matrix import arguments, NativeInvocationError, native_kwargs
from .native_publication import NativePublication
from .sequence_prediction import library, read, write

package = 'cicero'
guidance = 'Cicero native CellDataSet aggregation and co-accessibility; exact object axes and source identities, no inferred genome lengths or upstream reruns.'
TOOL = 'run_cicero_cds'
KIND = 'native_cicero_cds'
CONTRACT = 'native-cicero-cds.v1'
POLICY = 'native-cicero-cds-v1'
PROFILE = digest(dict(contract=CONTRACT, adaptation='Cicero exprs/pData/fData safe exchange.v1'))
VERIFIER = dict(id='agent.' + KIND, compatibility_version='1')
FUNCTIONS = {
    'make_cicero_cds': ('k', 'summary_stats', 'size_factor_normalize', 'silent', 'return_agg_info'),
    'run_cicero': ('window', 'silent', 'sample_num'),
    'estimate_distance_parameter': ('window', 'maxit', 's', 'sample_num', 'distance_constraint',
        'distance_parameter_convergence', 'max_elements', 'max_sample_windows'),
}
NATIVE_ARGUMENTS = (
    ('k', 'make_cicero_cds neighborhood aggregation size, native default 50.'),
    ('summary_stats', 'make_cicero_cds exact numeric obs fields to summarize.'),
    ('size_factor_normalize', 'make_cicero_cds native aggregate normalization, default true.'),
    ('return_agg_info', 'make_cicero_cds optional original-cell to aggregate mapping; omitted default false.'),
    ('silent', 'Native diagnostic suppression.'),
    ('window', 'Genomic window size, native default 500000.'),
    ('sample_num', 'Native sampled genomic windows.'),
    ('maxit', 'Distance parameter native optimization iterations.'),
    ('s', 'Native distance exponent, default 0.75.'),
    ('distance_constraint', 'Native minimum distance used in parameter estimation.'),
    ('distance_parameter_convergence', 'Native convergence threshold.'),
    ('max_elements', 'Native maximum regions per genomic window.'),
    ('max_sample_windows', 'Native maximum sampled windows.'),
)
FACT_FIELDS = ('function_name', 'n_observations', 'n_regions', 'parameters', 'source_identity',
    'software_versions', 'native_output_type', 'limitations')
RESULT_FIELDS = dict(status=(str,), contract_version=(str,), manifest_path=(str,), manifest_sha256=(str,),
    cds_path=(str,), cds_sha256=(str,), table_path=(str,), table_sha256=(str,), function_name=(str,),
    n_observations=(int,), n_regions=(int,), parameters=(dict,), source_identity=(dict,),
    software_versions=(dict,), native_output_type=(str,), limitations=(str,))
DELIVERY_FILES = {
    'cds.zip': ('Cicero CellDataSet exprs/pData/fData exchange; no executable serialization', 'application/zip'),
    'result.tsv': ('Native connections or distance parameters; make has no connection estimates', 'text/tab-separated-values'),
    'aggregation.tsv': ('Native optional cell aggregation map; native identifiers preserved', 'text/tab-separated-values'),
    'metadata.json': ('Native invocation, object representation and provenance', 'application/json'),
    'native-receipt.json': ('Native completion and runtime', 'application/json'),
}
MEMBERS = ('exprs.mtx', 'object.json')
LIMITATIONS = ('Native Cicero invocation, not biological validation. Aggregates overlap and are not independent cells. '
    'CDS exchange preserves exprs, pData and fData for these functions, not arbitrary fitted Monocle slots. '
    'Region names use native chr_start_end coordinates; no assembly or chromosome lengths are inferred. '
    'Native undefined values and warnings remain unchanged; stochastic behavior follows the recorded seed.')


def native_arguments(args):
    # Reuse the existing ordinary JSON and fixed-function boundary.
    kw = native_kwargs(FUNCTIONS, args)
    seed = args.get('seed')
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2147483647):
        raise ValueError('R seed must fit its native integer representation')
    return kw


def parameter_prerequisites(args):
    if args.get('function_name') == 'make_cicero_cds':
        return () if args.get('reduced_coordinates_key') else ('reduced_coordinates_key',)
    return ('genomic_coords_path',) if args.get('function_name') == 'run_cicero' and not args.get('genomic_coords_path') else ()


def unpack(source, target):
    """Read only two fixed data members; no archive extraction or serialized code."""
    with zipfile.ZipFile(source) as bundle:
        if sorted(bundle.namelist()) != sorted(MEMBERS):
            raise ValueError('Cicero exchange requires exactly exprs.mtx and object.json')
        from agent.tools._native_process import NativeExecutionError
        try:
            budget = int(os.environ.get('AGENT_NATIVE_ARCHIVE_BYTES', str(2 * 1024**3)))
            if budget <= 0:
                raise ValueError()
        except ValueError:
            raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE', 'Configure a positive native archive disk budget') from None
        if sum(i.file_size for i in bundle.infolist()) > budget:
            raise NativeExecutionError('NATIVE_TEMPORARY_LIMIT', 'Archive expansion exceeds the operator temporary-storage budget')
        for name in MEMBERS:
            with bundle.open(name) as inp, (target / name).open('wb') as out:
                shutil.copyfileobj(inp, out)
    # This member contains scientific axes/data, not a control-plane request.
    # Keep the existing strict JSON parser; kernel memory and archive disk
    # budgets govern its decoding instead of an axis-size proxy.
    from .sequence_prediction import io
    path = target / 'object.json'
    obj = io.read_json(path, limit=path.stat().st_size)
    if obj.get('format') != 'cicero-cds-exchange.v1' or obj.get('storage') not in ('dense', 'sparse'):
        raise ValueError('Unsupported Cicero exchange format')
    observations, regions = obj['observations']['ids'], obj['regions']['ids']
    if not isinstance(observations, list) or not isinstance(regions, list):
        raise ValueError('Cicero axes must be explicit arrays')
    with (target / 'exprs.mtx').open() as stream:
        if stream.readline().strip().lower() not in ('%%matrixmarket matrix coordinate real general', '%%matrixmarket matrix coordinate integer general'):
            raise ValueError('Cicero exchange uses sparse real/integer MatrixMarket coordinates')
        for line in stream:
            if not line.startswith('%'):
                rows, cols, nnz = map(int, line.split())
                if (rows, cols) != (len(regions), len(observations)) or nnz < 0:
                    raise ValueError('Cicero matrix dimensions/storage differ from explicit axes')
                break
        else:
            raise ValueError('Missing Cicero MatrixMarket dimensions')
    # Numeric array/object deserialization is performed by jsonlite, never readRDS.
    return obj


def prepare(args, stage):
    if args['function_name'] != 'make_cicero_cds':
        return unpack(args['input_path'], stage)
    peak_h5ad.inspect(args['input_path'], require_matrix=False)
    a = ad.read_h5ad(args['input_path'])
    from .replicate_pseudobulk import _strict_ids
    cells = _strict_ids(a.obs_names, 'cell'); regions = _strict_ids(a.var_names, 'region')
    if any(any(c in s for c in '\t\r\n') for s in cells + regions):
        raise ValueError('Cicero exchange identifiers cannot contain TSV delimiters')
    slot = args.get('count_slot', 'X')
    x = a.X if slot == 'X' else a.layers[slot.removeprefix('layers/')]
    x = peak_h5ad.read_native_matrix(x)
    mmwrite(stage / 'exprs.mtx', x.T)
    coords = a.obsm[args['reduced_coordinates_key']]
    if hasattr(coords, 'to_numpy'):
        coords = coords.to_numpy()
    np.savetxt(stage / 'coordinates.tsv', coords, delimiter='\t', fmt='%.17g')
    # Only requested metadata is required by make_cicero_cds; retain its exact types.
    keys = args.get('kwargs', {}).get('summary_stats') or []
    keys = [keys] if isinstance(keys, str) else keys
    pdata = {'ids': cells, 'columns': {}}
    for key in keys if isinstance(keys, list) else []:
        if key in a.obs:
            series = a.obs[key]
            pdata['columns'][key] = dict(type='logical' if series.dtype.kind == 'b' else 'numeric' if series.dtype.kind in 'iuf' else 'character',
                values=[None if __import__('pandas').isna(v) else v for v in series.tolist()])
    obj = dict(format='cicero-cds-exchange.v1', storage='sparse', observations=pdata,
        regions=dict(ids=regions, columns={}))
    write(stage / 'object.json', obj)
    return obj


def execute(values, execution_identity=None):
    from agent.tools._native_process import call
    return call('cicero_cds', dict(values=values, execution_identity=execution_identity),
                lambda: _execute(values, execution_identity))


def _execute(values, execution_identity=None):
    args = arguments(values); kw = native_arguments(args)
    missing = parameter_prerequisites(args)
    if missing:
        raise ValueError('Missing actual Cicero input: ' + ', '.join(missing))
    root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Native output must be empty; no retry authority')
    sources = file_closure(args[k] for k in ('input_path', 'genomic_coords_path') if args.get(k))
    for prefix in ('input', 'genomic_coords'):
        if args.get(prefix + '_sha256') and next(p['sha256'] for p in sources if p['path'] == args[prefix + '_path']) != args[prefix + '_sha256']:
            raise AuthorityError('Selected Cicero source changed')
    executable = Path(os.environ.get('AGENT_CICERO_RSCRIPT') or library().root.parent / 'runtimes/cicero-1/env/bin/Rscript')
    script = Path(__file__).parent / 'r/cicero_cds_v1.R'
    implementation = file_closure([executable, script])
    with tempfile.TemporaryDirectory(prefix='.cicero-cds-', dir=root) as temp:
        stage = Path(temp); prepare(args, stage)
        if args.get('genomic_coords_path'):
            shutil.copyfile(args['genomic_coords_path'], stage / 'genome.tsv')
        sha = write(stage / 'request.json', dict(stage=str(stage), function_name=args['function_name'], kwargs=kw, seed=args.get('seed')))
        code = run([str(executable), '--vanilla', str(script), str(stage / 'request.json')], stage / 'native.log', environment=r_environment())
        shutil.copyfile(stage / 'native.log', root / ('native-failure.log' if code else 'native.log'))
        if code:
            message = read(stage / 'native-error.json').get('message', '')[:2000] if (stage / 'native-error.json').exists() else ''
            if not message or peak_h5ad._PATH.search(message) or any(ord(c) < 32 for c in message):
                message = 'Native Cicero failed; exact diagnostic retained privately'
            raise NativeInvocationError(message)
        receipt = read(stage / 'native-receipt.json')
        if receipt['request_sha256'] != sha or receipt['completion'] != 'succeeded' or receipt['function_name'] != args['function_name'] or receipt['versions']['cicero'] != '1.28.3':
            raise AuthorityError('Cicero native completion differs')
        receipt['implementation'] = implementation
        with zipfile.ZipFile(root / 'cds.zip', 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in MEMBERS:
                bundle.write(stage / ('out-' + name), name)
        for name in ('result.tsv', 'aggregation.tsv'):
            shutil.copyfile(stage / name, root / name)
        summary = dict(status='success', contract_version=CONTRACT, function_name='cicero::' + args['function_name'],
            n_observations=receipt['n_observations'], n_regions=receipt['n_regions'],
            parameters=dict(kwargs=kw, seed=args.get('seed'), count_slot=args.get('count_slot', 'X'),
                reduced_coordinates_key=args.get('reduced_coordinates_key'),
                genomic_coords=receipt.get('genomic_coords'), omitted_optional_arguments='native defaults'),
            source_identity={k.removesuffix('_path'): {f: p[f] for f in ('sha256', 'size_bytes')} for k in ('input_path', 'genomic_coords_path') if args.get(k) for p in sources if p['path'] == args[k]},
            software_versions=receipt['versions'], native_output_type=receipt['native_output_type'], limitations=LIMITATIONS)
        write(root / 'metadata.json', dict(summary=summary, sources=sources, receipt=receipt))
        write(root / 'native-receipt.json', receipt)
    expected = sorted(sources + implementation, key=lambda p: p['path'])
    if file_closure(p['path'] for p in expected) != expected:
        raise AuthorityError('Source or native runtime changed')
    value = dict(contract=CONTRACT, profile=PROFILE, arguments=args, execution_identity=execution_identity,
        result=summary, files=file_closure(root / n for n in DELIVERY_FILES), sources=expected, native_completion='succeeded')
    path = root / 'manifest.json'; sha = write(path, value)
    if current() is not None:
        current().register_execution(KIND, root, execution_identity)
    return verify_publication(path, expected_sha256=sha)


def invoke(input_path, output_dir, function_name, kwargs=None, input_sha256=None, count_slot=None,
           reduced_coordinates_key=None, genomic_coords_path=None, genomic_coords_sha256=None, seed=None):
    return execute({k: v for k, v in locals().items() if v is not None})


publication = NativePublication(tool=TOOL, kind=KIND, contract=CONTRACT, profile=PROFILE, verifier=VERIFIER,
    files=tuple(DELIVERY_FILES), arguments=arguments, outputs={'cds': 'cds.zip', 'table': 'result.tsv'})
load = publication.load
result = publication.result
verify_publication = publication.verify_publication
verify_result = publication.verify_result
recover = publication.recover
