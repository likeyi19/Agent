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
    'stochastic results are qualified only for the recorded runtime and seed. Native nonfinite values are retained, not repaired or claimed usable.')
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


def prepare_input(args, stage):
    """AnnData owns encoding; only task-local objects can be mutated by packages."""
    peak_h5ad.inspect(args['input_path'], require_matrix=False)
    with h5py.File(args['input_path'], 'r') as handle:
        effective = declarations(handle, args)
    check_declarations(args['input_path'], effective)
    original = ad.read_h5ad(args['input_path'])
    slot = effective['count_slot']
    x = original.X if slot == 'X' else original.layers[slot.removeprefix('layers/')]
    # The bounded HDF5/AnnData readers admit ordinary serialized annotations,
    # not arbitrary Python objects. Retain them exactly as native AnnData calls
    # do; a retained reduction is not asserted to have been recomputed from X.
    matrix = original.copy()
    matrix.X = x.copy() if x is not None else None
    effective['matrix_semantics'] = effective.get('matrix_semantics') or 'unspecified_retained_measurement'
    matrix.uns['matrix_semantics'] = effective['matrix_semantics']
    matrix.write_h5ad(stage / 'input.h5ad')
    return matrix, effective


class MatrixPackage:
    """One package interface, several compatible public functions, one lifecycle."""
    FACT_FIELDS, RESULT_FIELDS, DELIVERY_FILES = FACT_FIELDS, RESULT_FIELDS, DELIVERY_FILES
    arguments = staticmethod(arguments)

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
            adaptation='ordinary AnnData preservation; optional measurements; native numerical validation.v3'))
        self.VERIFIER = dict(id='agent.' + self.KIND, compatibility_version='1')
        self.publication = NativePublication(tool=self.TOOL, kind=self.KIND, contract=self.CONTRACT,
            profile=self.PROFILE, verifier=self.VERIFIER, files=tuple(DELIVERY_FILES),
            arguments=arguments, outputs={'matrix': 'result.h5ad'},
            historical_profiles=((HISTORICAL_PROFILES[package], digest(dict(contract=self.CONTRACT, version=version,
                adaptation='ordinary AnnData preservation; selected X; exact axes; bounded JSON kwargs.v2'))) if package in HISTORICAL_PROFILES else ()))
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
        if args['function_name'] == 'tl.umap' and isinstance(kwargs.get('init'), (list, dict)):
            raise ValueError('UMAP initialization arrays are scientific objects, not ordinary kwargs')
        # Bounds govern execution cost, not the native scientific validity range.
        for key, cap in (('n_components', 512), ('n_comps', 512), ('n', 512), ('n_iter', 1000), ('n_iterations', 1000), ('n_epochs', 10000), ('maxiter', 10000), ('n_dcs', 512), ('n_pcs', 512), ('max_iter', 10), ('n_jobs', 16), ('num_threads', 32)):
            if type(kwargs.get(key)) in (int, float) and kwargs[key] > cap:
                raise ValueError(f'{key} exceeds the native execution resource bound {cap}')
        for key in ('layer_tfidf_key', 'tf_idf_layer', 'features', 'use_rep', 'key_added'):
            value = kwargs.get(key)
            if isinstance(value, str) and ('/' in value or '\\' in value or value in ('.', '..')):
                raise ValueError('Native object keys must be local field names')
        return kwargs

    def native(self, args, stage, matrix):
        is_r = self.package == 'Signac'
        override = os.environ.get('AGENT_' + self.package.upper() + ('_RSCRIPT' if is_r else '_PYTHON'))
        executable = Path(override) if override else library().root.parent / 'runtimes' / self.runtime / 'env/bin' / ('Rscript' if is_r else 'python')
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
            elif args['function_name'] == 'RunSVD':
                import pandas as pd
                for filename, names, container, key in (
                    ('embeddings.tsv', matrix.obs_names, matrix.obsm, 'X_lsi'),
                    ('loadings.tsv', matrix.var_names, matrix.varm, 'LSI')):
                    table = pd.read_csv(stage / filename, sep='\t', index_col=0,
                        dtype={'id': str}, keep_default_na=False, quoting=csv.QUOTE_NONE)
                    if table.index.tolist() != names.tolist():
                        raise AuthorityError('Native reduction changed ordered axis identities')
                    container[key] = table.to_numpy(dtype=float)
                matrix.uns['Signac_svd'] = read(stage / 'svd.json')
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
                if any(any(c in s for c in '\t\r\n') for s in cells + features):
                    raise ValueError('R matrix exchange labels cannot contain TSV delimiters')
                values = matrix.X.data if sparse.issparse(matrix.X) else np.asarray(matrix.X)
                peak_h5ad.native_double_values(values)
            if args['function_name'] == 'tl.spectral' and kwargs.get('distance_metric', 'cosine') != 'cosine' and matrix.n_obs**2 > 16_000_000:
                raise ValueError('Quadratic native similarity exceeds the 16000000-element execution bound')
            if args['function_name'] == 'pp.scale' and kwargs.get('zero_center', True):
                selected = matrix.layers[kwargs['layer']] if kwargs.get('layer') else matrix.obsm[kwargs['obsm']] if kwargs.get('obsm') else matrix.X
                if sparse.issparse(selected):
                    raise ValueError('Native centered scale densifies the complete sparse matrix; select zero_center=false explicitly or supply dense data')
            receipt = self.native(effective, stage, matrix)
            peak_h5ad.inspect(stage / 'result.h5ad', require_matrix=False)
            output = ad.read_h5ad(stage / 'result.h5ad')
            if (output.obs_names.tolist() != cells or output.var_names.tolist() != features):
                raise AuthorityError('Native result changed the exact ordered input axes')
            if args['function_name'] not in ('pp.binarize', 'pp.log1p', 'pp.normalize_total', 'RunTFIDF', 'BinarizeCounts', 'pp.scale') and matrix.X is not None:
                original = ad.read_h5ad(stage / 'input.h5ad')
                before, after = (sparse.csr_matrix(x, copy=True) for x in (original.X, output.X))
                for retained in (before, after):
                    retained.sum_duplicates()
                    retained.eliminate_zeros()
                    retained.sort_indices()
                if (before.shape != after.shape
                        or not np.array_equal(before.indptr, after.indptr)
                        or not np.array_equal(before.indices, after.indices)
                        or not np.array_equal(before.data, after.data, equal_nan=True)):
                    raise AuthorityError('Native annotation/embedding function changed source X')
            for slot in receipt['result_slots']:
                group, _, key = slot.partition('.')
                value = output.X if group == 'X' else getattr(output, group)[key]
                if group in ('X', 'layers', 'obsm', 'varm', 'obsp'):
                    values = value.data if sparse.issparse(value) else np.asarray(value)
                    if values.dtype.kind not in 'biuf':
                        raise AuthorityError('Native numeric result changed representation')
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
    'tl.pca': ('n_comps', 'zero_center', 'svd_solver', 'random_state', 'return_info', 'use_highly_variable', 'dtype', 'copy', 'chunked', 'chunk_size'),
    'pp.neighbors': ('n_neighbors', 'n_pcs', 'use_rep', 'knn', 'random_state', 'method', 'metric', 'metric_kwds', 'copy'),
    'tl.dpt': ('n_dcs', 'n_branchings', 'min_group_size', 'allow_kendall_tau_shift', 'neighbors_key', 'copy'),
    'tl.tsne': ('n_pcs', 'use_rep', 'perplexity', 'early_exaggeration', 'learning_rate', 'random_state', 'use_fast_tsne', 'n_jobs', 'copy'),
    'tl.umap': ('min_dist', 'spread', 'n_components', 'maxiter', 'alpha', 'gamma', 'negative_sample_rate', 'init_pos', 'random_state', 'a', 'b', 'copy'),
    'pp.binarize': ('copy',), 'pp.tfidf': ('norm', 'layer_tfidf_key'),
    'pp.log1p': ('base', 'copy', 'chunked', 'chunk_size'),
    'pp.normalize_total': ('target_sum', 'exclude_highly_expressed', 'max_fraction', 'key_added', 'layers', 'layer_norm', 'inplace'),
    'pp.lsi': ('n_components', 'tf_idf_layer', 'algorithm', 'n_iter', 'random_state'),
}, (
    ('n_comps', 'pca native component count; cap 512.'),
    ('zero_center', 'pca native centering; omission uses true.'),
    ('svd_solver', 'pca installed native solver.'),
    ('return_info', 'pca native return_info flag; AnnData results retain native slots.'),
    ('use_highly_variable', 'pca uses the existing var highly_variable flag when requested.'),
    ('dtype', 'pca native result dtype.'),
    ('n_neighbors', 'neighbors native neighborhood size.'),
    ('n_pcs', 'neighbors/tsne component count; native representation selection applies.'),
    ('use_rep', 'neighbors/tsne existing obsm key or native X; omitted uses native representation selection.'),
    ('knn', 'neighbors hard nearest-neighbor threshold versus full kernel.'),
    ('method', 'neighbors native graph construction method, default gauss.'),
    ('metric', 'neighbors metric name; no executable callable.'),
    ('metric_kwds', 'neighbors JSON metric options; no executable callbacks.'),
    ('n_dcs', 'dpt diffusion component count; existing uns.iroot identifies the supplied root cell.'),
    ('n_branchings', 'dpt native branching count.'),
    ('min_group_size', 'dpt minimum group fraction or size.'),
    ('allow_kendall_tau_shift', 'dpt native branching behavior.'),
    ('neighbors_key', 'dpt existing uns neighbors key.'),
    ('perplexity', 'tsne native perplexity.'),
    ('early_exaggeration', 'tsne native early exaggeration.'),
    ('learning_rate', 'tsne native learning rate.'),
    ('use_fast_tsne', 'tsne native implementation choice; no adapter substitution.'),
    ('n_jobs', 'tsne worker count; execution cap 16.'),
    *((k, 'umap native ' + k + '; omitted uses installed default.') for k in ('min_dist', 'spread', 'maxiter', 'alpha', 'gamma', 'negative_sample_rate', 'init_pos', 'a', 'b')),
    ('target_sum', 'normalize_total target library size; omitted uses the native median.'),
    ('exclude_highly_expressed', 'normalize_total excludes high-expression features from normalization-factor calculation.'),
    ('max_fraction', 'normalize_total native high-expression fraction threshold.'),
    ('key_added', 'normalize_total obs field for normalization factors.'),
    ('layers', 'normalize_total exact layer names or all; no external objects.'),
    ('layer_norm', 'normalize_total layer scaling: null, after, or X.'),
    ('inplace', 'normalize_total: false leaves AnnData unchanged; installed epiScanpy discards Scanpy dictionary return.'),
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
), 'epiScanpy pca, neighbors, dpt, tsne and umap retain their native slots and defaults. dpt uses supplied uns.iroot. binarize/log1p change X; tfidf adds a layer; lsi consumes an existing TF-IDF layer. normalize_total scales X/requested layers; inplace=False returns None with no changes in this wrapper. Native internal representation choices apply; Agent adds no preprocessing.')


SNAPATAC2 = MatrixPackage('snapatac2', '2.9.0', 'native-matrix-1', {
    'tl.umap': ('n_comps', 'use_dims', 'use_rep', 'key_added', 'random_state', 'inplace', 'n_neighbors', 'min_dist', 'spread', 'metric', 'n_epochs', 'learning_rate', 'init', 'negative_sample_rate', 'repulsion_strength', 'local_connectivity', 'set_op_mix_ratio', 'a', 'b', 'verbose', 'n_jobs'),
    'pp.knn': ('n_neighbors', 'use_dims', 'use_rep', 'method', 'inplace', 'random_state'),
    'tl.leiden': ('resolution', 'objective_function', 'min_cluster_size', 'n_iterations', 'random_state', 'key_added', 'weighted', 'inplace'),
    'pp.select_features': ('n_features', 'filter_lower_quantile', 'filter_upper_quantile', 'max_iter', 'inplace', 'n_jobs', 'verbose'),
    'tl.spectral': ('n_comps', 'features', 'random_state', 'sample_size', 'sample_method', 'distance_metric', 'weighted_by_sd', 'chunk_size', 'inplace', 'num_threads'),
}, (
    ('use_dims', 'umap/knn leading component count or exact integer component indices.'),
    ('use_rep', 'umap/knn exact existing obsm key, default X_spectral; no implicit reduction.'),
    ('key_added', 'umap output obsm X_<key>, default umap; leiden output obs key, default leiden.'),
    ('n_neighbors', 'umap/knn native neighbor count; defaults 15/50 respectively.'),
    ('method', 'knn kdtree, hora or pynndescent; hora has no reproducible seed.'),
    ('resolution', 'leiden native clustering resolution.'),
    ('objective_function', 'leiden modularity or CPM objective.'),
    ('min_cluster_size', 'leiden minimum cluster size; smaller groups become -1.'),
    ('n_iterations', 'leiden native iterations, -1 until convergence; cap 1000 positive iterations.'),
    ('weighted', 'leiden uses exp(-distance) weights when true.'),
    *((k, 'umap.UMAP native ' + k + '; omitted uses installed default.') for k in ('min_dist', 'spread', 'metric', 'n_epochs', 'learning_rate', 'init', 'negative_sample_rate', 'repulsion_strength', 'local_connectivity', 'set_op_mix_ratio', 'a', 'b')),
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
), 'SnapATAC2 umap and knn consume an existing obsm embedding; knn writes obsp.distances; leiden consumes that distance graph and writes obs labels (including native -1 small groups). No implicit embedding or graph. select_features records var.selected without subsetting; spectral consumes X and an explicit/default feature mask and records X_spectral plus eigenvalues. No automatic feature selection.')

SIGNAC = MatrixPackage('Signac', '1.16.0', 'signac-1', {
    'RunSVD': ('assay', 'n', 'scale.embeddings', 'reduction.key', 'scale.max', 'verbose', 'irlba.work', 'tol'),
    'RunTFIDF': ('method', 'scale.factor', 'verbose'),
    'BinarizeCounts': ('verbose',), 'FindTopFeatures': ('min.cutoff', 'verbose'),
}, (
    ('assay', 'RunSVD assay name recorded by native DimReduc; no assay construction.'),
    ('n', 'RunSVD requested singular components; native clamps to cells minus one; cap 512.'),
    ('scale.embeddings', 'RunSVD native embedding scaling, default true.'),
    ('reduction.key', 'RunSVD native component key, default LSI_.'),
    ('scale.max', 'RunSVD optional symmetric clipping of scaled embeddings.'),
    ('irlba.work', 'RunSVD native IRLBA work space, default n*3.'),
    ('tol', 'RunSVD native IRLBA convergence tolerance.'),
    ('method', 'RunTFIDF native method 1/2/3/4; omitted uses 1. Not sklearn TF-IDF.'),
    ('scale.factor', 'RunTFIDF native scaling, default 10000; method-specific effect.'),
    ('verbose', 'RunTFIDF progress; BinarizeCounts/FindTopFeatures matrix methods accept but ignore it.'),
    ('min.cutoff', 'FindTopFeatures matrix method accepts but ignores this assay-only selection argument; returns counts and percentile ranking, no filtering.'),
), 'Signac RunSVD consumes the selected matrix as-is, publishes cell embeddings, feature loadings, standard deviations and native SVD misc; no automatic TF-IDF. Matrix methods: RunTFIDF transforms X; BinarizeCounts binarizes X; FindTopFeatures returns count/percentile statistics ranked by count, mapped by exact feature ID with native_rank retained. Matrix method does not select variable features or apply min.cutoff. No Seurat object or full LSI workflow.')

SCANPY = MatrixPackage('scanpy', '1.9.8', 'episcanpy-0.4.0', {
    'tl.diffmap': ('n_comps', 'neighbors_key', 'random_state', 'copy'),
    'pp.scale': ('zero_center', 'max_value', 'copy', 'layer', 'obsm'),
}, (
    ('n_comps', 'diffmap native component count, default 15; execution cap 512.'),
    ('neighbors_key', 'diffmap existing uns neighborhood key; default neighbors.'),
    ('random_state', 'diffmap native random state, default 0.'),
    ('copy', 'Native AnnData copy flag; source is always task-local.'),
    ('zero_center', 'scale native centering, default true; sparse X centering materializes dense values.'),
    ('max_value', 'scale native clipping threshold; omission does not clip.'),
    ('layer', 'scale exact existing layer instead of X.'),
    ('obsm', 'scale exact existing obsm field instead of X.'),
), 'Scanpy diffmap consumes the existing neighborhood graph and writes X_diffmap/diffmap_evals. scale transforms selected X, layer or obsm with native variance scaling and optional centering/clipping. No automatic neighbors or inferred preprocessing; installed native defaults remain authoritative.')

PACKAGES = (EPISCANPY, SNAPATAC2, SIGNAC, SCANPY)
