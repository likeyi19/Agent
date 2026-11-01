"""Reviewed package functions on task-local AnnData/matrix objects.

Packages own their numerical methods. This owner shares only bounded input
adaptation, invocation, exact axes and the existing native publication lifecycle.
"""
import math
import json
import csv
import os
from pathlib import Path
import shutil
import tempfile

import anndata as ad
import h5py
import numpy as np
from anndata.io import read_elem
from scipy import sparse
from scipy.io import mmwrite, mmread

from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools._native_process import run, r_environment
from agent.tools.data.authority_context import current
from agent.tools.data.fastq_verification_authority import file_closure
from . import peak_h5ad
from .native_publication import NativePublication
from .sequence_prediction import arguments as path_arguments, library, read, write
from .sequence_reference import metadata_declarations, check_declarations


FACT_FIELDS = ('function_name', 'n_cells', 'n_features', 'measurement', 'parameters',
    'source_identity', 'software_versions', 'native_output_type', 'result_slots', 'limitations')
RESULT_FIELDS = dict(status=(str,), contract_version=(str,), manifest_path=(str,), manifest_sha256=(str,),
    matrix_path=(str,), matrix_sha256=(str,), function_name=(str,), n_cells=(int,), n_features=(int,),
    measurement=(str,), parameters=(dict,), source_identity=(dict,), software_versions=(dict,),
    native_output_type=(str,), result_slots=(list,), limitations=(str,))
DELIVERY_FILES = {
    'result.h5ad': ('Native result with exact ordered cells and features', 'application/octet-stream'),
    'metadata.json': ('Native arguments, source measurement and output interpretation', 'application/json'),
    'native-receipt.json': ('Installed native function completion and versions', 'application/json'),
}
LIMITATIONS = ('Independent native invocation, not a completed analysis workflow or biological validation. '
    'Source annotations are supplied metadata. Matrix transformations, feature masks and embeddings have distinct meanings. '
    'No inferred filtering, genome conversion or upstream rerun. Omitted native arguments retain installed defaults; '
    'stochastic results are qualified only for the recorded runtime and seed.')
# Exact pre-finalization adapter profiles remain readable, never writable.
HISTORICAL_PROFILES = {
    'episcanpy': '58c0e8efe169db823aec036d9ba6659a1b4ff86e49ed4f1c03d8145a03c1c281',
    'snapatac2': '08d654538cbe449abb3fc4b99ed9b9aa86b855d4852e2a76252eb54d0bc106c0',
    'Signac': 'be212d279ad3840fbadaaf59901b17de595465a0575dd2ed07da5c182cad78ef',
}


class NativeInvocationError(ValueError):
    """The native package failed on the unchanged invocation."""


def arguments(values):
    return path_arguments(_serialize(values))


def declarations(handle, args):
    slot = args.get('count_slot', 'X')
    if slot != 'X' and not slot.startswith('layers/'):
        slot = 'layers/' + slot
    effective = metadata_declarations(handle, slot, args, read_elem(handle['var']))
    effective['count_slot'] = slot
    return effective


def parameter_prerequisites(inputs):
    if not inputs.get('input_path'):
        return ()
    peak_h5ad.inspect(inputs['input_path'])
    with h5py.File(inputs['input_path'], 'r') as handle:
        effective = declarations(handle, inputs)
    return () if effective.get('matrix_semantics') else ('matrix_semantics',)


def prepare_input(args, stage):
    """AnnData owns encoding; only task-local objects can be mutated by packages."""
    from .replicate_pseudobulk import _strict_ids
    peak_h5ad.inspect(args['input_path'])
    with h5py.File(args['input_path'], 'r') as handle:
        effective = declarations(handle, args)
    if not effective.get('matrix_semantics'):
        raise ValueError('Declare the actual measurement in the selected matrix slot')
    check_declarations(args['input_path'], effective)
    original = ad.read_h5ad(args['input_path'])
    cells, features = _strict_ids(original.obs_names, 'cell'), _strict_ids(original.var_names, 'feature')
    if not cells or not features or any(any(c in s for c in '\t\r\n') for s in cells + features):
        raise ValueError('Nonempty unique axes without tab/newline characters are required')
    slot = effective['count_slot']
    x = original.X if slot == 'X' else original.layers[slot.removeprefix('layers/')]
    values = x.data if sparse.issparse(x) else np.asarray(x)
    if values.dtype.kind not in 'biuf' or not np.isfinite(values).all():
        raise ValueError('Native matrix functions require finite numeric measurements')
    if args['function_name'] != 'pp.log1p' and np.any(values < 0):
        raise ValueError('This native matrix function requires nonnegative measurements')
    # The bounded HDF5/AnnData readers admit ordinary serialized annotations,
    # not arbitrary Python objects. Retain them exactly as native AnnData calls
    # do; a retained reduction is not asserted to have been recomputed from X.
    matrix = original.copy()
    matrix.X = x.copy()
    matrix.uns['matrix_semantics'] = effective['matrix_semantics']
    matrix.write_h5ad(stage / 'input.h5ad')
    return matrix, effective


class MatrixPackage:
    """One package interface, several compatible public functions, one lifecycle."""
    FACT_FIELDS, RESULT_FIELDS, DELIVERY_FILES = FACT_FIELDS, RESULT_FIELDS, DELIVERY_FILES
    arguments = staticmethod(arguments)
    parameter_prerequisites = staticmethod(parameter_prerequisites)

    def __init__(self, package, version, runtime, functions, native_arguments, guidance):
        self.package, self.version, self.runtime = package, version, runtime
        self.FUNCTIONS, self.NATIVE_ARGUMENTS, self.guidance = functions, native_arguments, guidance
        self.TOOL = 'run_' + package.lower()
        self.KIND = 'native_' + package.lower() + '_matrix'
        self.CONTRACT = self.KIND.replace('_', '-') + '.v1'
        self.POLICY = self.CONTRACT.replace('.v1', '-v1')
        # The profile identifies the adapter, not its growing function catalog.
        # Exact function, kwargs and installed implementation remain pinned in
        # each publication. Compatible additions need no historical migration.
        self.PROFILE = digest(dict(contract=self.CONTRACT, version=version,
            adaptation='ordinary AnnData preservation; selected X; exact axes; bounded JSON kwargs.v2'))
        self.VERIFIER = dict(id='agent.' + self.KIND, compatibility_version='1')
        self.publication = NativePublication(tool=self.TOOL, kind=self.KIND, contract=self.CONTRACT,
            profile=self.PROFILE, verifier=self.VERIFIER, files=tuple(DELIVERY_FILES),
            arguments=arguments, outputs={'matrix': 'result.h5ad'},
            historical_profiles=(HISTORICAL_PROFILES[package],))
        for name in ('load', 'result', 'verify_publication', 'verify_result', 'recover'):
            setattr(self, name, getattr(self.publication, name))

    def native_arguments(self, args):
        function, kwargs = args.get('function_name'), args.get('kwargs', {})
        if function not in self.FUNCTIONS or type(kwargs) is not dict or set(kwargs) - set(self.FUNCTIONS[function]):
            raise ValueError('Unsupported native function or argument; no method is substituted')
        def ordinary(value, depth=0):
            if depth > 4:
                return False
            if value is None or type(value) in (bool, int):
                return True
            if type(value) is float:
                return math.isfinite(value)
            if type(value) is str:
                return len(value) <= 128 and not any(ord(c) < 32 for c in value)
            if type(value) is list:
                return len(value) <= 128 and all(ordinary(v, depth + 1) for v in value)
            if type(value) is dict:
                return len(value) <= 128 and all(type(k) is str and ordinary(k) and ordinary(v, depth + 1) for k, v in value.items())
            return False
        if not ordinary(kwargs) or len(json.dumps(kwargs).encode()) > 8192:
            raise ValueError('Native kwargs must be bounded JSON data, not executable expressions or input objects')
        # A supplied spectral mask is scientific input, not ordinary kwargs.
        # This interface binds the exact existing var field (or all features).
        if isinstance(kwargs.get('features'), (list, dict)):
            raise ValueError('Bind a spectral feature mask through the input AnnData var field')
        # Bounds govern execution cost, not the native scientific validity range.
        for key, cap in (('n_components', 512), ('n_comps', 512), ('n_iter', 1000), ('max_iter', 10), ('n_jobs', 16), ('num_threads', 32)):
            if type(kwargs.get(key)) in (int, float) and kwargs[key] > cap:
                raise ValueError(f'{key} exceeds the native execution resource bound {cap}')
        for key in ('layer_tfidf_key', 'tf_idf_layer', 'features'):
            value = kwargs.get(key)
            if isinstance(value, str) and ('/' in value or '\\' in value or value in ('.', '..')):
                raise ValueError('Native object keys must be local field names')
        return kwargs

    def native(self, args, stage, matrix):
        root = library().root.parent / 'runtimes' / self.runtime / 'env'
        is_r = self.package == 'Signac'
        executable = Path(os.environ.get('AGENT_' + self.package.upper() + ('_RSCRIPT' if is_r else '_PYTHON'),
            str(root / 'bin' / ('Rscript' if is_r else 'python'))))
        script = Path(__file__).parent / ('r/signac_functions_v1.R' if is_r else 'native_matrix_worker.py')
        if not executable.is_file():
            raise ValueError(f'Provision the isolated {self.package} {self.version} scientific runtime')
        implementation_paths = [executable.resolve(), script]
        if not is_r:
            implementation_paths.append(executable.parent.parent / 'pyvenv.cfg')
        implementation = file_closure(implementation_paths)
        request = dict(stage=str(stage), package=self.package, version=self.version,
            function_name=args['function_name'], kwargs=self.native_arguments(args))
        request_sha = write(stage / 'request.json', request)
        if is_r:
            mmwrite(stage / 'input.mtx', sparse.csc_matrix(matrix.X.T))
            matrix.obs.to_csv(stage / 'cells.tsv', sep='\t', columns=[], index_label='cell_id', quoting=csv.QUOTE_NONE)
            matrix.var.to_csv(stage / 'features.tsv', sep='\t', columns=[], index_label='feature_id', quoting=csv.QUOTE_NONE)
        code = run([str(executable), '--vanilla' if is_r else '-I', str(script), str(stage / 'request.json')],
            stage / 'native.log', environment=r_environment())
        if code:
            shutil.copyfile(stage / 'native.log', stage.parent / 'native-failure.log')
            message = read(stage / 'native-error.json').get('message', '')[:2000] if (stage / 'native-error.json').is_file() else ''
            if not message or peak_h5ad._PATH.search(message) or any(ord(c) < 32 for c in message):
                message = 'Native function failed; consult the retained private diagnostic'
            raise NativeInvocationError(message)
        # Preserve warnings/progress privately; logs confer no publication authority.
        shutil.copyfile(stage / 'native.log', stage.parent / 'native.log')
        receipt = read(stage / 'native-receipt.json')
        if (receipt['completion'] != 'succeeded' or receipt['request_sha256'] != request_sha
                or receipt['function_name'] != args['function_name'] or receipt['package'] != self.package
                or receipt['versions'][self.package] != self.version):
            raise AuthorityError('Native completion differs from the exact requested invocation')
        if file_closure(implementation_paths) != implementation:
            raise AuthorityError('Native implementation changed during invocation')
        receipt['implementation'] = implementation
        if is_r:
            if args['function_name'] == 'FindTopFeatures':
                import pandas as pd
                table = pd.read_csv(stage / 'feature-statistics.tsv', sep='\t', index_col=0,
                    dtype={'feature_id': str}, keep_default_na=False, quoting=csv.QUOTE_NONE)
                if not table.index.is_unique or set(table.index) != set(matrix.var_names):
                    raise AuthorityError('Native feature statistics changed feature identities')
                table['native_rank'] = np.arange(len(table))
                table = table.loc[matrix.var_names]
                for key in table:
                    matrix.var['Signac_' + key] = table[key].to_numpy()
            else:
                matrix.X = mmread(stage / 'output.mtx').T.tocsr()
            matrix.write_h5ad(stage / 'result.h5ad')
        return receipt

    def execute(self, values, execution_identity=None):
        args = arguments(values)
        kwargs = self.native_arguments(args)
        root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError('Output directory is not empty; retry requires a new authorized execution')
        source = file_closure([args['input_path']])[0]
        if args.get('input_sha256') and source['sha256'] != args['input_sha256']:
            raise AuthorityError('Selected input bytes changed')
        with tempfile.TemporaryDirectory(prefix='.native-matrix-', dir=root) as temp:
            stage = Path(temp)
            matrix, effective = prepare_input(args, stage)
            cells, features = matrix.obs_names.tolist(), matrix.var_names.tolist()
            if self.package == 'Signac':
                values = matrix.X.data if sparse.issparse(matrix.X) else np.asarray(matrix.X)
                if np.any(values > 2**53):
                    raise ValueError('Matrix values exceed exact R double precision; no rounding is permitted')
            if args['function_name'] == 'tl.spectral' and kwargs.get('distance_metric', 'cosine') != 'cosine' and matrix.n_obs**2 > 16_000_000:
                raise ValueError('Quadratic native similarity exceeds the 16000000-element execution bound')
            receipt = self.native(effective, stage, matrix)
            peak_h5ad.inspect(stage / 'result.h5ad')
            output = ad.read_h5ad(stage / 'result.h5ad')
            if (output.obs_names.tolist() != cells or output.var_names.tolist() != features):
                raise AuthorityError('Native result changed the exact ordered input axes')
            if args['function_name'] in ('pp.tfidf', 'pp.lsi', 'pp.select_features', 'tl.spectral', 'FindTopFeatures'):
                original = ad.read_h5ad(stage / 'input.h5ad')
                if (sparse.csr_matrix(original.X) != sparse.csr_matrix(output.X)).nnz:
                    raise AuthorityError('Native annotation/embedding function changed source X')
            for slot in receipt['result_slots']:
                group, _, key = slot.partition('.')
                value = output.X if group == 'X' else getattr(output, group)[key]
                if group in ('X', 'layers', 'obsm'):
                    values = value.data if sparse.issparse(value) else np.asarray(value)
                    if values.dtype.kind not in 'biuf' or not np.isfinite(values).all():
                        raise AuthorityError('Native output is nonnumeric or nonfinite; it cannot be published as a usable matrix')
            measurement = receipt.get('measurement') or effective['matrix_semantics']
            output.uns['matrix_semantics'] = measurement
            for slot, semantics in receipt.get('layer_semantics', {}).items():
                output.uns.setdefault('layer_semantics', {})[slot] = semantics
            summary = dict(status='success', contract_version=self.CONTRACT,
                function_name=self.package + ('::' if self.package == 'Signac' else '.') + args['function_name'],
                n_cells=output.n_obs, n_features=output.n_vars, measurement=measurement,
                parameters=dict(kwargs=kwargs, omitted_optional_arguments='native defaults', count_slot=effective['count_slot']),
                source_identity=dict(sha256=source['sha256'], size_bytes=source['size_bytes'], measurement=effective['matrix_semantics']),
                software_versions=receipt['versions'], native_output_type=receipt['native_output_type'],
                result_slots=receipt['result_slots'], limitations=LIMITATIONS)
            output.uns['agent_native_invocation'] = dict(function=summary['function_name'], source_sha256=source['sha256'],
                source_measurement=effective['matrix_semantics'], parameters_json=json.dumps(summary['parameters']),
                annotation_policy='Retained source annotations and reductions are not recomputed; native-written slots replace existing values.')
            output.write_h5ad(root / 'result.h5ad')
            write(root / 'native-receipt.json', receipt)
            write(root / 'metadata.json', dict(summary=summary, source=source, receipt=receipt,
                input_orientation='cells by features; Signac receives features by cells'))
            expected = {r['path']: r for r in [source, *receipt['implementation']]}
            sources = file_closure(expected)
            if sources != sorted(expected.values(), key=lambda r: r['path']):
                raise AuthorityError('Source or runtime changed during invocation')
        value = dict(contract=self.CONTRACT, profile=self.PROFILE, arguments=args, execution_identity=execution_identity,
            result=summary, files=file_closure(root / name for name in DELIVERY_FILES), sources=sources, native_completion='succeeded')
        path = root / 'manifest.json'; sha = write(path, value)
        if current() is not None:
            current().register_execution(self.KIND, root, execution_identity)
        return self.verify_publication(path, expected_sha256=sha)

    def invoke(self, input_path, output_dir, function_name, kwargs=None, input_sha256=None, count_slot=None, matrix_semantics=None):
        return self.execute({k: v for k, v in locals().items() if k != 'self' and v is not None})


EPISCANPY = MatrixPackage('episcanpy', '0.4.0', 'episcanpy-0.4.0', {
    'pp.binarize': ('copy',), 'pp.tfidf': ('norm', 'layer_tfidf_key'),
    'pp.log1p': ('base', 'copy', 'chunked', 'chunk_size'),
    'pp.lsi': ('n_components', 'tf_idf_layer', 'algorithm', 'n_iter', 'random_state'),
}, (
    ('copy', 'binarize/log1p: native copy flag; source is always a managed copy.'),
    ('norm', 'tfidf row normalization; installed default l1, not the docstring l2.'),
    ('layer_tfidf_key', 'tfidf destination layer, default tf-idf; X stays unchanged.'),
    ('base', 'log1p logarithm base; omission uses natural log.'),
    ('chunked', 'log1p native chunked processing flag.'),
    ('chunk_size', 'log1p native cells per processing chunk.'),
    ('n_components', 'lsi component count, default 50; execution cap 512.'),
    ('tf_idf_layer', 'lsi existing TF-IDF layer; default tf-idf. No adapter-generated TF-IDF.'),
    ('algorithm', 'lsi native TruncatedSVD solver, default arpack.'),
    ('n_iter', 'lsi native iterations, default 10; execution cap 1000.'),
    ('random_state', 'lsi native random state, default 42.'),
), 'epiScanpy binarize/log1p change X; tfidf adds a layer; lsi consumes an existing TF-IDF layer and writes X_lsi. Distinct native calls, no automatic workflow.')

SNAPATAC2 = MatrixPackage('snapatac2', '2.9.0', 'native-matrix-1', {
    'pp.select_features': ('n_features', 'filter_lower_quantile', 'filter_upper_quantile', 'max_iter', 'inplace', 'n_jobs', 'verbose'),
    'tl.spectral': ('n_comps', 'features', 'random_state', 'sample_size', 'sample_method', 'distance_metric', 'weighted_by_sd', 'chunk_size', 'inplace', 'num_threads'),
}, (
    ('n_features', 'select_features maximum selected features; no matrix subsetting.'),
    ('filter_lower_quantile', 'select_features lower count quantile exclusion.'),
    ('filter_upper_quantile', 'select_features upper count quantile exclusion.'),
    ('max_iter', 'select_features iterations; default 1; execution cap 10.'),
    ('inplace', 'Native in-place flag; returned arrays are mapped to the same result slots when false.'),
    ('n_jobs', 'select_features parallel jobs; execution cap 16.'),
    ('verbose', 'select_features progress output.'),
    ('n_comps', 'spectral component count; execution cap 512.'),
    ('features', 'spectral exact boolean var field, default selected; null explicitly uses all features.'),
    ('random_state', 'spectral native random seed.'),
    ('sample_size', 'spectral native Nystrom sample size or fraction.'),
    ('sample_method', 'spectral native sampling method.'),
    ('distance_metric', 'spectral native similarity choice.'),
    ('weighted_by_sd', 'spectral native eigenvector scaling.'),
    ('chunk_size', 'spectral native processing chunk size.'),
    ('num_threads', 'spectral native thread count; installed default and execution cap 32.'),
), 'SnapATAC2 select_features records var.selected without subsetting; spectral consumes X and an explicit/default feature mask and records X_spectral plus eigenvalues. No automatic feature selection.')

SIGNAC = MatrixPackage('Signac', '1.16.0', 'signac-1', {
    'RunTFIDF': ('method', 'scale.factor', 'verbose'),
    'BinarizeCounts': ('verbose',), 'FindTopFeatures': ('min.cutoff', 'verbose'),
}, (
    ('method', 'RunTFIDF native method 1/2/3/4; omitted uses 1. Not sklearn TF-IDF.'),
    ('scale.factor', 'RunTFIDF native scaling, default 10000; method-specific effect.'),
    ('verbose', 'RunTFIDF progress; BinarizeCounts/FindTopFeatures matrix methods accept but ignore it.'),
    ('min.cutoff', 'FindTopFeatures matrix method accepts but ignores this assay-only selection argument; returns counts and percentile ranking, no filtering.'),
), 'Signac matrix methods: RunTFIDF transforms X; BinarizeCounts binarizes X; FindTopFeatures returns count/percentile statistics ranked by count, mapped by exact feature ID with native_rank retained. Matrix method does not select variable features or apply min.cutoff. No Seurat object or full LSI workflow.')

PACKAGES = (EPISCANPY, SNAPATAC2, SIGNAC)
