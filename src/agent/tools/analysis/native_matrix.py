"""Reviewed package functions on task-local AnnData/matrix objects.

Packages own their numerical methods. This owner shares only bounded input
adaptation, invocation, exact axes and the existing native publication lifecycle.
"""
import math
import json
import csv
import os
from functools import lru_cache
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
from agent.tools._native_process import run, r_environment, NativeExecutionError
from agent.tools.data.authority_context import current
from agent.tools._file_identity import file_closure
from . import peak_h5ad
from .native_publication import NativePublication
from .sequence_prediction import arguments as path_arguments, library, read, write
from .sequence_reference import metadata_declarations, check_declarations


def native_kwargs(functions, args):
    """Shared bounded JSON/function boundary; scientific object checks stay with owners."""
    function, kwargs = args.get('function_name'), args.get('kwargs', {})
    if function not in functions or type(kwargs) is not dict or set(kwargs) - set(functions[function]):
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
    return kwargs


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
    'Native feature names such as genes do not establish biological feature identity. No inferred filtering, genome conversion or upstream rerun. Omitted native arguments retain installed defaults; '
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
    peak_h5ad.inspect(args['input_path'], require_matrix=False, max_logical_bytes=None)
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


@lru_cache(maxsize=128)
def _installed_parameter_metadata(executable, package, function):
    import subprocess
    script = Path(__file__).with_name('native_matrix_worker.py')
    try:
        result = subprocess.run([executable, '-B', str(script), '--parameters', package, function],
            capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {}


class MatrixPackage:
    """One package interface, several compatible public functions, one lifecycle."""
    FACT_FIELDS, RESULT_FIELDS, DELIVERY_FILES = FACT_FIELDS, RESULT_FIELDS, DELIVERY_FILES
    arguments = staticmethod(arguments)

    def __init__(self, package, version, runtime, functions, native_arguments, guidance):
        self.package, self.version, self.runtime = package, version, runtime
        if package == 'Signac':
            self.invoke = self.invoke_signac
        if package == 'snapatac2':
            from . import snapatac_raw as raw
            self.invoke = self.invoke_snapatac2
            functions = {**functions, **raw.FUNCTIONS}
            native_arguments = tuple(native_arguments) + tuple((key, 'Native ' + key + '; omitted retains its installed default.')
                for key in sorted(set().union(*map(set, raw.FUNCTIONS.values())) - {k for k, _ in native_arguments}))
            self.RESULT_FIELDS = dict(RESULT_FIELDS, matrix_path=(str, type(None)), matrix_sha256=(str, type(None)),
                n_cells=(int, type(None)), n_features=(int, type(None)),
                fragments_path=(str, type(None)), fragments_sha256=(str, type(None)))
            self.DELIVERY_FILES = {**DELIVERY_FILES, **raw.RAW_FILES}
        self.FUNCTIONS, self.NATIVE_ARGUMENTS, self.guidance = functions, native_arguments, guidance
        self.TOOL = 'run_' + package.lower()
        self.KIND = 'native_' + package.lower() + '_matrix'
        self.CONTRACT = self.KIND.replace('_', '-') + '.v1'
        self.POLICY = self.CONTRACT.replace('.v1', '-v1')
        # The profile identifies the adapter, not its growing function catalog.
        # Exact function, kwargs and installed implementation remain pinned in
        # each publication. Compatible additions need no historical migration.
        self.PROFILE = digest(dict(contract=self.CONTRACT, version=version,
            adaptation='ordinary AnnData; native returns and exact subset positions.v4'))
        self.VERIFIER = dict(id='agent.' + self.KIND, compatibility_version='1')
        historical = []
        if package in ('episcanpy', 'snapatac2', 'Signac', 'scanpy'):
            historical.append(digest(dict(contract=self.CONTRACT, version=version,
                adaptation='ordinary AnnData preservation; optional measurements; native numerical validation.v3')))
        if package in HISTORICAL_PROFILES:
            historical.extend((HISTORICAL_PROFILES[package], digest(dict(contract=self.CONTRACT, version=version,
                adaptation='ordinary AnnData preservation; selected X; exact axes; bounded JSON kwargs.v2'))))
        if package == 'Signac':
            historical.append(self.PROFILE)
            self.PROFILE = digest(dict(contract=self.CONTRACT, version=version, adaptation='ordered independent pairwise native matrices.v5'))
        if package == 'snapatac2':
            historical.append(self.PROFILE)
            self.PROFILE = digest(dict(contract=self.CONTRACT, version=version,
                adaptation='native fragment BAM QC filtering and feature matrices; exact bound file ports.v5'))
        self.publication = NativePublication(tool=self.TOOL, kind=self.KIND, contract=self.CONTRACT,
            profile=self.PROFILE, verifier=self.VERIFIER, files=raw.files if package == 'snapatac2' else tuple(DELIVERY_FILES),
            arguments=arguments, outputs=(lambda value: raw.outputs(value) if value['profile'] == self.PROFILE else {'matrix': 'result.h5ad'})
                if package == 'snapatac2' else {'matrix': 'result.h5ad'}, historical_profiles=historical)
        for name in ('load', 'result', 'verify_publication', 'verify_result', 'recover'):
            setattr(self, name, getattr(self.publication, name))

    def native_arguments(self, args):
        function = args.get('function_name')
        kwargs = native_kwargs(self.FUNCTIONS, args)
        if self.package == 'snapatac2':
            from . import snapatac_raw as raw
            if args.get('qc_thresholds') is not None:
                if args['qc_thresholds'] != 'otsu' or function not in ('pp.import_fragments', 'pp.filter_cells'):
                    raise ValueError('Otsu threshold selection applies only to uncalled import or native filtering')
                if function == 'pp.import_fragments':
                    kwargs = dict(kwargs)
                    kwargs.setdefault('min_num_fragments', 0)
            if args.get('counting_strategy') is not None:
                if function not in ('pp.make_peak_matrix', 'pp.add_tile_matrix'):
                    raise ValueError('counting_strategy applies only to native peak or tile matrix construction')
                if 'counting_strategy' in kwargs and kwargs['counting_strategy'] != args['counting_strategy']:
                    raise ValueError('Explicit native kwargs and scalar have conflicting counting_strategy values')
                kwargs = dict(kwargs, counting_strategy=args['counting_strategy'])
            if args.get('fragment_order') is not None:
                if function != 'pp.import_fragments' or args['fragment_order'] not in ('unknown', 'barcode'):
                    raise ValueError('fragment_order is source knowledge for native fragment import only')
                kwargs = dict(kwargs)
                kwargs.setdefault('sorted_by_barcode', args['fragment_order'] == 'barcode')
            if function in raw.FUNCTIONS:
                raw.input_paths(args)
            elif any(args.get(path) is not None or args.get(sha) is not None for path, sha, _ in raw.PORTS.values()):
                raise ValueError('Raw reference files do not apply to this native matrix function')
        if (args.get('second_path') or args.get('second_count_slot') or args.get('second_sha256')) and (self.package != 'Signac' or function not in ('corSparse', 'Jaccard')):
            raise ValueError('A second matrix applies only to Signac corSparse/Jaccard')
        if function == 'Jaccard' and not args.get('second_path'):
            raise ValueError('Signac Jaccard requires two actual matrix sources')
        # A supplied spectral mask is scientific input, not ordinary kwargs.
        # This interface binds the exact existing var field (or all features).
        if args['function_name'] == 'tl.spectral' and isinstance(kwargs.get('features'), (list, dict)):
            raise ValueError('Bind a spectral feature mask through the input AnnData var field')
        if args['function_name'] == 'tl.umap' and any(isinstance(kwargs.get(k), (list, dict)) for k in ('init', 'init_pos')):
            raise ValueError('UMAP initialization arrays are scientific objects, not ordinary kwargs')
        for key in ('layer_tfidf_key', 'tf_idf_layer', 'features', 'use_rep', 'key_added'):
            value = kwargs.get(key)
            if isinstance(value, str) and ('/' in value or '\\' in value or value in ('.', '..')):
                raise ValueError('Native object keys must be local field names')
        return kwargs

    def parameter_metadata(self, function):
        """Inspect the same installed runtime used by native dispatch; run no science."""
        if function not in self.FUNCTIONS or self.package == 'Signac':
            return {}
        override = os.environ.get('AGENT_' + self.package.upper() + '_PYTHON')
        try:
            executable = Path(override) if override else library().root.parent / 'runtimes' / self.runtime / 'env/bin/python'
        except NativeExecutionError:
            return {}
        if not executable.is_file():
            return {}
        return _installed_parameter_metadata(str(executable), self.package, function)

    def native(self, args, stage, matrix):
        is_r = self.package == 'Signac'
        override = os.environ.get('AGENT_' + self.package.upper() + ('_RSCRIPT' if is_r else '_PYTHON'))
        executable = Path(override) if override else library().root.parent / 'runtimes' / self.runtime / 'env/bin' / ('Rscript' if is_r else 'python')
        if is_r:
            from .native_runtimes import rscript
            executable = rscript(self.package)
        from . import snapatac_raw as raw
        raw_call = self.package == 'snapatac2' and args['function_name'] in raw.FUNCTIONS
        script = Path(__file__).parent / ('snapatac_raw_worker.py' if raw_call else 'r/signac_functions_v1.R' if is_r else 'native_matrix_worker.py')
        if not executable.is_file():
            raise NativeExecutionError('NATIVE_EXECUTION_UNAVAILABLE',
                f'Provision the isolated {self.package} {self.version} scientific runtime')
        implementation_paths = [executable.resolve(), script]
        if raw_call:
            implementation_paths.append(script.with_name('native_matrix_worker.py'))
        if not is_r:
            implementation_paths.append(executable.parent.parent / 'pyvenv.cfg')
        implementation = file_closure(implementation_paths)
        request = dict(stage=str(stage), package=self.package, version=self.version,
            function_name=args['function_name'], kwargs=self.native_arguments(args), second_matrix=bool(args.get('second_path')))
        if raw_call:
            request['raw_inputs'] = {key: args[key] for key in ('input_path', 'matrix_semantics', 'species', 'assembly',
                'qc_thresholds',
                *(record[0] for record in raw.PORTS.values())) if key in args}
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
        native_implementation = receipt.get('implementation', [])
        if native_implementation and file_closure(r['path'] for r in native_implementation) != sorted(native_implementation, key=lambda r:r['path']):
            raise AuthorityError('Native package implementation changed during invocation')
        receipt['implementation'] = sorted(implementation + native_implementation, key=lambda r:r['path'])
        if is_r:
            if args.get('second_path'):
                values = mmread(stage/'output.mtx').toarray()
                import pandas as pd
                other_cells = pd.read_csv(stage/'second-cells.tsv', sep='\t', dtype=str, keep_default_na=False).cell_id.tolist()
                if values.shape != (matrix.n_obs, len(other_cells)):
                    raise AuthorityError('Native pairwise axes differ from the exact two inputs')
                matrix.uns['Signac_pairwise'] = dict(values=values, row_cells=matrix.obs_names.to_numpy(dtype=str),
                    column_cells=np.array(other_cells, dtype=str), function_name=args['function_name'])
            elif args['function_name'] == 'corSparse':
                matrix.obsp['Signac_corSparse'] = mmread(stage / 'output.mtx').tocsr()
            elif args['function_name'] == 'FindTopFeatures':
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
        from agent.tools._native_process import call
        return call('matrix', dict(package=self.package, values=values, execution_identity=execution_identity),
                    lambda: self._execute(values, execution_identity))

    def _execute(self, values, execution_identity=None):
        args = arguments(values)
        if self.package == 'snapatac2':
            from . import snapatac_raw as raw
            if args['function_name'] in raw.FUNCTIONS:
                return raw.execute(self, args, execution_identity)
        kwargs = self.native_arguments(args)
        root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError('Output directory is not empty; retry requires a new authorized execution')
        source = file_closure([args['input_path']])[0]
        if args.get('input_sha256') and source['sha256'] != args['input_sha256']:
            raise AuthorityError('Selected input bytes changed')
        second_source = file_closure([args['second_path']])[0] if args.get('second_path') else None
        if second_source and args.get('second_sha256') and second_source['sha256'] != args['second_sha256']:
            raise AuthorityError('Selected second matrix bytes changed')
        with tempfile.TemporaryDirectory(prefix='.native-matrix-', dir=root) as temp:
            stage = Path(temp)
            matrix, effective = prepare_input(args, stage)
            cells, features = matrix.obs_names.tolist(), matrix.var_names.tolist()
            if self.package == 'Signac':
                if any(any(c in s for c in '\t\r\n') for s in cells + features):
                    raise ValueError('R matrix exchange labels cannot contain TSV delimiters')
                values = matrix.X.data if sparse.issparse(matrix.X) else np.asarray(matrix.X)
                peak_h5ad.native_double_values(values)
            if second_source:
                other_stage = stage/'second'; other_stage.mkdir()
                other, other_effective = prepare_input(dict(input_path=args['second_path'],
                    **({'count_slot': args['second_count_slot']} if args.get('second_count_slot') is not None else {})), other_stage)
                if features != other.var_names.tolist():
                    raise ValueError('Two-matrix comparison requires identical ordered feature identities; no implicit alignment')
                peak_h5ad.native_double_values(other.X.data if sparse.issparse(other.X) else np.asarray(other.X))
                mmwrite(stage/'second.mtx', sparse.csc_matrix(other.X.T))
                other.obs.to_csv(stage/'second-cells.tsv', sep='\t', columns=[], index_label='cell_id', quoting=csv.QUOTE_NONE)
                effective['second_path'] = args['second_path']
            if args['function_name'] == 'pp.scale' and kwargs.get('zero_center', True):
                selected = matrix.layers[kwargs['layer']] if kwargs.get('layer') else matrix.obsm[kwargs['obsm']] if kwargs.get('obsm') else matrix.X
                if sparse.issparse(selected):
                    raise ValueError('Native centered scale densifies the complete sparse matrix; select zero_center=false explicitly or supply dense data')
            receipt = self.native(effective, stage, matrix)
            peak_h5ad.inspect(stage / 'result.h5ad', require_matrix=False)
            output = ad.read_h5ad(stage / 'result.h5ad')
            positions = receipt.get('source_axis_positions')
            ci, fi = list(range(len(cells))), list(range(len(features)))
            if positions is not None:
                if self.package != 'scanpy' or args['function_name'] != 'pp.highly_variable_genes' or not kwargs.get('subset'):
                    raise AuthorityError('Unexpected native axis change')
                ci, fi = positions['cells'], positions['features']
                if ci != list(range(len(cells))) or any(type(i) is not int or not 0 <= i < len(features) for i in fi) or fi != sorted(set(fi)):
                    raise AuthorityError('Invalid native subset source positions')
            if output.obs_names.tolist() != [cells[i] for i in ci] or output.var_names.tolist() != [features[i] for i in fi]:
                raise AuthorityError('Native result changed the exact ordered input axes')
            if args['function_name'] not in ('pp.binarize', 'pp.log1p', 'pp.normalize_total', 'RunTFIDF', 'BinarizeCounts', 'pp.scale') and matrix.X is not None:
                original = ad.read_h5ad(stage / 'input.h5ad')[ci, fi]
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
            if 'effective_parameters' in receipt:
                summary['parameters']['effective_native_parameters'] = receipt['effective_parameters']
                summary['parameters']['result_shapes'] = receipt.get('result_shapes', {})
            if second_source:
                summary['source_identity']['second'] = dict(sha256=second_source['sha256'], size_bytes=second_source['size_bytes'],
                    count_slot=other_effective['count_slot'], measurement=other_effective['matrix_semantics'], n_cells=other.n_obs)
            output.uns['agent_native_invocation'] = dict(function=summary['function_name'], source_sha256=source['sha256'],
                source_measurement=effective['matrix_semantics'], parameters_json=json.dumps(summary['parameters']),
                annotation_policy='Retained source annotations and reductions are not recomputed; native-written slots replace existing values.')
            if positions is not None:
                output.uns['agent_native_invocation']['source_axis_positions'] = positions
            output.write_h5ad(root / 'result.h5ad')
            write(root / 'native-receipt.json', receipt)
            write(root / 'metadata.json', dict(summary=summary, source=source, receipt=receipt,
                input_orientation='cells by features; Signac receives features by cells'))
            expected = {r['path']: r for r in [source, *([second_source] if second_source else []), *receipt['implementation']]}
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
        return self.execute({k: v for k, v in locals().items() if k != "self" and v is not None})

    def invoke_signac(self, input_path, output_dir, function_name, kwargs=None, input_sha256=None, count_slot=None, matrix_semantics=None, second_path=None, second_sha256=None, second_count_slot=None):
        return self.execute({k: v for k, v in locals().items() if k != 'self' and v is not None})

    def invoke_snapatac2(self, input_path, output_dir, function_name, kwargs=None, input_sha256=None, count_slot=None,
                        matrix_semantics=None, chrom_sizes_path=None, chrom_sizes_sha256=None, gene_anno_path=None,
                        gene_anno_sha256=None, peaks_path=None, peaks_sha256=None, whitelist_path=None,
                        whitelist_sha256=None, species=None, assembly=None, fragment_order=None, counting_strategy=None,
                        qc_thresholds=None):
        return self.execute({k: v for k, v in locals().items() if k != 'self' and v is not None})


EPISCANPY = MatrixPackage('episcanpy', '0.4.0', 'episcanpy-0.4.0', {
    'tl.pca': ('n_comps', 'zero_center', 'svd_solver', 'random_state', 'return_info', 'use_highly_variable', 'dtype', 'copy', 'chunked', 'chunk_size'),
    'pp.neighbors': ('n_neighbors', 'n_pcs', 'use_rep', 'knn', 'random_state', 'method', 'metric', 'metric_kwds', 'copy'),
    'tl.dpt': ('n_dcs', 'n_branchings', 'min_group_size', 'allow_kendall_tau_shift', 'neighbors_key', 'copy'),
    'tl.tsne': ('n_pcs', 'use_rep', 'perplexity', 'early_exaggeration', 'learning_rate', 'random_state', 'use_fast_tsne', 'n_jobs', 'copy'),
    'tl.umap': ('min_dist', 'spread', 'n_components', 'maxiter', 'alpha', 'gamma', 'negative_sample_rate', 'init_pos', 'random_state', 'a', 'b', 'copy'),
    'pp.binarize': ('copy',),
    'pp.tfidf': ('norm', 'layer_tfidf_key'),
    'pp.log1p': ('base', 'copy', 'chunked', 'chunk_size'),
    'pp.normalize_total': ('target_sum', 'exclude_highly_expressed', 'max_fraction', 'key_added', 'layers', 'layer_norm', 'inplace'),
    'pp.lsi': ('n_components', 'tf_idf_layer', 'algorithm', 'n_iter', 'random_state'),
    'tl.leiden': ('resolution', 'restrict_to', 'random_state', 'key_added', 'directed', 'use_weights', 'n_iterations', 'neighbors_key', 'obsp', 'copy'),
    'tl.louvain': ('resolution', 'random_state', 'restrict_to', 'key_added', 'flavor', 'directed', 'use_weights', 'neighbors_key', 'obsp', 'copy'),
    'tl.kmeans': ('num_clusters',),
    'tl.hc': ('num_clusters',),
    'tl.rank_features': ('groupby', 'omic', 'use_raw', 'groups', 'reference', 'n_features', 'rankby_abs', 'key_added', 'copy', 'method', 'corr_method'),
}, (
    ('n_comps', 'Native n_comps for the selected function; omitted retains its installed default.'),
    ('zero_center', 'pca native centering; omitted retains its installed default.'),
    ('svd_solver', 'pca installed native solver.'),
    ('random_state', 'Native random_state for the selected function; omitted retains its installed default.'),
    ('return_info', 'pca native return_info flag; AnnData results retain native slots.'),
    ('use_highly_variable', 'pca uses the existing var highly_variable flag when requested.'),
    ('dtype', 'pca native result dtype.'),
    ('copy', 'Native copy flag; returned AnnData is retained, source bytes stay immutable.'),
    ('chunked', 'pca/log1p native chunked processing flag.'),
    ('chunk_size', 'pca/log1p native processing chunk size.'),
    ('n_neighbors', 'neighbors native neighborhood size; omitted retains its installed default.'),
    ('n_pcs', 'neighbors/tsne component count; native representation selection applies.'),
    ('use_rep', 'Exact native representation selector; omission retains installed selection/default.'),
    ('knn', 'neighbors hard nearest-neighbor threshold versus full kernel.'),
    ('method', 'Native method for the selected function; omitted retains its installed default.'),
    ('metric', 'neighbors native metric name; no executable callable.'),
    ('metric_kwds', 'neighbors JSON metric options; no executable callbacks.'),
    ('n_dcs', 'dpt diffusion component count; existing uns.iroot identifies the supplied root cell.'),
    ('n_branchings', 'dpt native branching count.'),
    ('min_group_size', 'dpt minimum group fraction or size.'),
    ('allow_kendall_tau_shift', 'dpt native branching behavior.'),
    ('neighbors_key', 'dpt/leiden/louvain existing uns neighborhood key; omitted uses native selection.'),
    ('perplexity', 'tsne native perplexity.'),
    ('early_exaggeration', 'tsne native early exaggeration.'),
    ('learning_rate', 'tsne native learning rate; omitted retains its installed default.'),
    ('use_fast_tsne', 'tsne native implementation choice; no adapter substitution.'),
    ('n_jobs', 'tsne native worker count.'),
    ('min_dist', 'umap native min_dist; omitted uses installed default.'),
    ('spread', 'umap native spread; omitted uses installed default.'),
    ('n_components', 'Native n_components for the selected function; omitted retains its installed default.'),
    ('maxiter', 'umap native maxiter; omitted uses installed default.'),
    ('alpha', 'Native alpha; omitted retains the installed function default.'),
    ('gamma', 'umap native gamma; omitted uses installed default.'),
    ('negative_sample_rate', 'umap native negative_sample_rate; omitted uses installed default.'),
    ('init_pos', 'umap native init_pos; omitted uses installed default.'),
    ('a', 'umap native a; omitted uses installed default.'),
    ('b', 'umap native b; omitted uses installed default.'),
    ('norm', 'tfidf row normalization; installed default l1, not the docstring l2.'),
    ('layer_tfidf_key', 'tfidf destination layer, default tf-idf; X stays unchanged.'),
    ('base', 'log1p logarithm base; omission uses natural log.'),
    ('target_sum', 'normalize_total target library size; omitted uses the native median.'),
    ('exclude_highly_expressed', 'normalize_total excludes high-expression features from normalization-factor calculation.'),
    ('max_fraction', 'normalize_total native high-expression fraction threshold.'),
    ('key_added', 'Native key_added for the selected function; omitted retains its installed default.'),
    ('layers', 'normalize_total exact layer names or all; no external objects.'),
    ('layer_norm', 'normalize_total layer scaling: null, after, or X.'),
    ('inplace', 'normalize_total: false leaves AnnData unchanged; epiScanpy discards the Scanpy return.'),
    ('tf_idf_layer', 'lsi existing TF-IDF layer; default tf-idf. No adapter-generated TF-IDF.'),
    ('algorithm', 'lsi native TruncatedSVD solver, default arpack.'),
    ('n_iter', 'Native n_iter; omitted retains the installed function default.'),
    ('resolution', 'leiden/louvain native clustering resolution.'),
    ('restrict_to', 'Native restrict_to; omitted retains the installed function default.'),
    ('directed', 'Native directed; omitted retains the installed function default.'),
    ('use_weights', 'Native use_weights; omitted retains the installed function default.'),
    ('n_iterations', 'leiden native iterations, -1 until convergence; cap 1000 positive iterations.'),
    ('obsp', 'Native obsp; omitted retains the installed function default.'),
    ('flavor', 'Native flavor; omitted retains the installed function default.'),
    ('num_clusters', 'Native num_clusters; omitted retains the installed function default.'),
    ('groupby', 'Exact obs grouping field; native statistics retain supplied feature semantics.'),
    ('omic', 'Native omic; omitted retains the installed function default.'),
    ('use_raw', 'Native use_raw; omitted retains the installed function default.'),
    ('groups', 'Native groups; omitted retains the installed function default.'),
    ('reference', 'Native reference; omitted retains the installed function default.'),
    ('n_features', 'rank_features number of ranked features per group; no matrix subsetting.'),
    ('rankby_abs', 'Native rankby_abs; omitted retains the installed function default.'),
    ('corr_method', 'Native corr_method; omitted retains the installed function default.'),
), 'Native AnnData transforms, PCA/graphs/embeddings, clustering and feature ranking. kmeans/hc consume X_pca; leiden/louvain consume graphs. rank_features retains upstream method overrides/errors. tfidf adds a layer; lsi uses that supplied layer. normalize_total(inplace=False) is a native no-op returning None. No Agent-added preprocessing.')


SNAPATAC2 = MatrixPackage('snapatac2', '2.9.0', 'native-matrix-1', {
    'tl.umap': ('n_comps', 'use_dims', 'use_rep', 'key_added', 'random_state', 'inplace', 'n_neighbors', 'min_dist', 'spread', 'metric', 'n_epochs', 'learning_rate', 'init', 'negative_sample_rate', 'repulsion_strength', 'local_connectivity', 'set_op_mix_ratio', 'a', 'b', 'verbose', 'n_jobs'),
    'pp.knn': ('n_neighbors', 'use_dims', 'use_rep', 'method', 'inplace', 'random_state'),
    'tl.leiden': ('resolution', 'objective_function', 'min_cluster_size', 'n_iterations', 'random_state', 'key_added', 'weighted', 'inplace'),
    'pp.select_features': ('n_features', 'filter_lower_quantile', 'filter_upper_quantile', 'max_iter', 'inplace', 'n_jobs', 'verbose'),
    'tl.spectral': ('n_comps', 'features', 'random_state', 'sample_size', 'sample_method', 'distance_metric', 'weighted_by_sd', 'chunk_size', 'inplace', 'num_threads'),
    'tl.dbscan': ('eps', 'min_samples', 'leaf_size', 'n_jobs', 'use_rep', 'key_added'),
    'pp.mnc_correct': ('batch', 'n_neighbors', 'n_clusters', 'n_iter', 'use_rep', 'use_dims', 'groupby', 'key_added', 'inplace', 'n_jobs'),
    'tl.marker_regions': ('groupby', 'pvalue'),
    'tl.diff_test': ('cell_group1', 'cell_group2', 'features', 'covariates', 'direction', 'min_log_fc', 'min_pct'),
}, (
    ('n_comps', 'Native n_comps for the selected function; omitted retains its installed default.'),
    ('use_dims', 'umap/knn/mnc_correct leading component count or exact integer indices.'),
    ('use_rep', 'Exact native representation selector; omission retains installed selection/default.'),
    ('key_added', 'Native key_added for the selected function; omitted retains its installed default.'),
    ('random_state', 'Native random_state for the selected function; omitted retains its installed default.'),
    ('inplace', 'Native mutation flag; false preserves independent returns in uns.agent_native_return.'),
    ('n_neighbors', 'umap/knn/mnc_correct native neighbor count; omitted retains installed default.'),
    ('min_dist', 'umap.UMAP native min_dist; omitted uses installed default.'),
    ('spread', 'umap.UMAP native spread; omitted uses installed default.'),
    ('metric', 'umap.UMAP native metric; omitted uses installed default.'),
    ('n_epochs', 'umap.UMAP native n_epochs; omitted uses installed default.'),
    ('learning_rate', 'umap.UMAP native learning_rate; omitted uses installed default.'),
    ('init', 'umap.UMAP native init; omitted uses installed default.'),
    ('negative_sample_rate', 'umap.UMAP native negative_sample_rate; omitted uses installed default.'),
    ('repulsion_strength', 'umap.UMAP native repulsion_strength; omitted uses installed default.'),
    ('local_connectivity', 'umap.UMAP native local_connectivity; omitted uses installed default.'),
    ('set_op_mix_ratio', 'umap.UMAP native set_op_mix_ratio; omitted uses installed default.'),
    ('a', 'umap.UMAP native a; omitted uses installed default.'),
    ('b', 'umap.UMAP native b; omitted uses installed default.'),
    ('verbose', 'Native progress verbosity for the selected function.'),
    ('n_jobs', 'Native parallel jobs for the selected function.'),
    ('method', 'Native method for the selected function; omitted retains its installed default.'),
    ('resolution', 'leiden native clustering resolution.'),
    ('objective_function', 'leiden modularity or CPM objective.'),
    ('min_cluster_size', 'leiden minimum cluster size; smaller groups become -1.'),
    ('n_iterations', 'leiden native iterations, -1 until convergence; cap 1000 positive iterations.'),
    ('weighted', 'leiden uses exp(-distance) weights when true.'),
    ('n_features', 'select_features maximum selected features; no matrix subsetting.'),
    ('filter_lower_quantile', 'select_features lower count quantile exclusion.'),
    ('filter_upper_quantile', 'select_features upper count quantile exclusion.'),
    ('max_iter', 'select_features iterations; default 1.'),
    ('features', 'Native existing mask field for spectral; explicit feature IDs/indices for diff_test.'),
    ('sample_size', 'spectral native Nystrom sample size or fraction.'),
    ('sample_method', 'spectral native sampling method.'),
    ('distance_metric', 'spectral native similarity choice.'),
    ('weighted_by_sd', 'spectral native eigenvector scaling.'),
    ('chunk_size', 'spectral native processing chunk size.'),
    ('num_threads', 'spectral native thread count; installed default 32.'),
    ('n_clusters', 'Native n_clusters; omitted retains the installed function default.'),
    ('eps', 'Native eps; omitted retains the installed function default.'),
    ('min_samples', 'Native min_samples; omitted retains the installed function default.'),
    ('leaf_size', 'Native leaf_size; omitted retains the installed function default.'),
    ('batch', 'Exact obs batch field for mnc_correct; no invented batch labels.'),
    ('n_iter', 'Native n_iter; omitted retains the installed function default.'),
    ('groupby', 'Exact obs grouping field; native statistics retain supplied feature semantics.'),
    ('pvalue', 'Native pvalue; omitted retains the installed function default.'),
    ('cell_group1', 'Explicit ordered source cell names or indices for diff_test.'),
    ('cell_group2', 'Explicit ordered source cell names or indices for diff_test.'),
    ('covariates', 'Native covariates; omitted retains the installed function default.'),
    ('direction', 'Native direction; omitted retains the installed function default.'),
    ('min_log_fc', 'Native min_log_fc; omitted retains the installed function default.'),
    ('min_pct', 'Native min_pct; omitted retains the installed function default.'),
), 'Native BAM/fragments ingestion, QC, filtering and region matrices, alongside AnnData feature selection, spectral embeddings, graphs and clustering. Each call uses its selected native function and actual count unit. Non-inplace returns remain independent in uns.agent_native_return; no automatic embedding or biological validation.')

SIGNAC = MatrixPackage('Signac', '1.16.0', 'signac-1', {
    'RunSVD': ('assay', 'n', 'scale.embeddings', 'reduction.key', 'scale.max', 'verbose', 'irlba.work', 'tol'),
    'RunTFIDF': ('method', 'scale.factor', 'verbose'),
    'BinarizeCounts': ('verbose',),
    'FindTopFeatures': ('min.cutoff', 'verbose'),
    'Jaccard': (),
    'corSparse': ('cov',),
}, (
    ('assay', 'RunSVD assay name recorded by native DimReduc; no assay construction.'),
    ('n', 'RunSVD requested singular components; native clamps to cells minus one; cap 512.'),
    ('scale.embeddings', 'RunSVD native embedding scaling, default true.'),
    ('reduction.key', 'RunSVD native component key, default LSI_.'),
    ('scale.max', 'RunSVD optional symmetric clipping of scaled embeddings.'),
    ('verbose', 'RunTFIDF progress; BinarizeCounts/FindTopFeatures matrix methods accept but ignore it.'),
    ('irlba.work', 'RunSVD native IRLBA work space, default n*3.'),
    ('tol', 'RunSVD native IRLBA convergence tolerance.'),
    ('method', 'Native method for the selected function; omitted retains its installed default.'),
    ('scale.factor', 'RunTFIDF native scaling, default 10000; method-specific effect.'),
    ('min.cutoff', 'FindTopFeatures matrix method accepts but ignores this assay-only selection argument; returns counts and percentile ranking, no filtering.'),
    ('cov', 'corSparse: return covariance instead of correlation; optional second matrix supplies native Y.'),
), 'Native matrix transformations/reductions preserve source axes. corSparse uses features by cells, with optional exact second matrix Y. Jaccard requires two cells-by-features matrices; no automatic binarization. Both require identical ordered features; rectangular cross-cell values and both cell axes are stored in uns.Signac_pairwise. Single-source corSparse remains obsp. No fake Seurat object, hidden TF-IDF or alignment. FindTopFeatures matrix method ignores min.cutoff.')

SCANPY = MatrixPackage('scanpy', '1.9.8', 'episcanpy-0.4.0', {
    'tl.diffmap': ('n_comps', 'neighbors_key', 'random_state', 'copy'),
    'pp.scale': ('zero_center', 'max_value', 'copy', 'layer', 'obsm'),
    'pp.calculate_qc_metrics': ('expr_type', 'var_type', 'qc_vars', 'percent_top', 'layer', 'use_raw', 'inplace', 'log1p', 'parallel'),
    'pp.normalize_total': ('target_sum', 'exclude_highly_expressed', 'max_fraction', 'key_added', 'layer', 'layers', 'layer_norm', 'inplace', 'copy'),
    'pp.log1p': ('base', 'copy', 'chunked', 'chunk_size', 'layer', 'obsm'),
    'pp.highly_variable_genes': ('layer', 'n_top_genes', 'min_disp', 'max_disp', 'min_mean', 'max_mean', 'span', 'n_bins', 'flavor', 'subset', 'inplace', 'batch_key', 'check_values'),
    'pp.pca': ('n_comps', 'zero_center', 'svd_solver', 'random_state', 'return_info', 'use_highly_variable', 'dtype', 'copy', 'chunked', 'chunk_size'),
    'pp.neighbors': ('n_neighbors', 'n_pcs', 'use_rep', 'knn', 'random_state', 'method', 'metric', 'metric_kwds', 'key_added', 'copy'),
    'tl.leiden': ('resolution', 'restrict_to', 'random_state', 'key_added', 'directed', 'use_weights', 'n_iterations', 'neighbors_key', 'obsp', 'copy'),
    'tl.umap': ('min_dist', 'spread', 'n_components', 'maxiter', 'alpha', 'gamma', 'negative_sample_rate', 'init_pos', 'random_state', 'a', 'b', 'copy', 'method', 'neighbors_key'),
    'tl.tsne': ('n_pcs', 'use_rep', 'perplexity', 'early_exaggeration', 'learning_rate', 'random_state', 'use_fast_tsne', 'n_jobs', 'copy', 'metric'),
    'tl.paga': ('groups', 'use_rna_velocity', 'model', 'neighbors_key', 'copy'),
    'tl.rank_genes_groups': ('groupby', 'use_raw', 'groups', 'reference', 'n_genes', 'rankby_abs', 'pts', 'key_added', 'copy', 'method', 'corr_method', 'tie_correct', 'layer'),
    'tl.score_genes': ('gene_list', 'ctrl_size', 'gene_pool', 'n_bins', 'score_name', 'random_state', 'copy', 'use_raw'),
    'tl.dpt': ('n_dcs', 'n_branchings', 'min_group_size', 'allow_kendall_tau_shift', 'neighbors_key', 'copy'),
}, (
    ('n_comps', 'Native n_comps for the selected function; omitted retains its installed default.'),
    ('neighbors_key', 'Existing uns neighborhood key for the selected native function.'),
    ('random_state', 'Native random_state for the selected function; omitted retains its installed default.'),
    ('copy', 'Native copy flag; returned AnnData is retained, source bytes stay immutable.'),
    ('zero_center', 'pca/scale native centering; centered scale materializes sparse values.'),
    ('max_value', 'scale native clipping threshold; omission does not clip.'),
    ('layer', 'Native layer for the selected function; omitted retains its installed default.'),
    ('obsm', 'Native obsm for the selected function; omitted retains its installed default.'),
    ('expr_type', 'Native expr_type; omitted retains the installed function default.'),
    ('var_type', 'Native QC feature noun only, not biological qualification.'),
    ('qc_vars', 'Existing boolean var annotation keys for native QC summaries.'),
    ('percent_top', 'Native QC rank positions; omission retains installed defaults.'),
    ('use_raw', 'Native use_raw; omitted retains the installed function default.'),
    ('inplace', 'Native mutation flag; false preserves independent returns in uns.agent_native_return.'),
    ('log1p', 'Native log1p; omitted retains the installed function default.'),
    ('parallel', 'Native parallel; omitted retains the installed function default.'),
    ('target_sum', 'normalize_total target library size; omitted uses the native median.'),
    ('exclude_highly_expressed', 'normalize_total excludes high-expression features from normalization-factor calculation.'),
    ('max_fraction', 'normalize_total native high-expression fraction threshold.'),
    ('key_added', 'Native key_added for the selected function; omitted retains its installed default.'),
    ('layers', 'normalize_total exact layer names or all; no external objects.'),
    ('layer_norm', 'normalize_total layer scaling: null, after, or X.'),
    ('base', 'log1p logarithm base; omission uses natural log.'),
    ('chunked', 'pca/log1p native chunked processing flag.'),
    ('chunk_size', 'pca/log1p native processing chunk size.'),
    ('n_top_genes', 'Native n_top_genes; omitted retains the installed function default.'),
    ('min_disp', 'Native min_disp; omitted retains the installed function default.'),
    ('max_disp', 'Native max_disp; omitted retains the installed function default.'),
    ('min_mean', 'Native min_mean; omitted retains the installed function default.'),
    ('max_mean', 'Native max_mean; omitted retains the installed function default.'),
    ('span', 'Native span; omitted retains the installed function default.'),
    ('n_bins', 'Native n_bins; omitted retains the installed function default.'),
    ('flavor', 'Native flavor; omitted retains the installed function default.'),
    ('subset', 'Native highly_variable_genes subsetting; resulting feature positions and axes are retained.'),
    ('batch_key', 'Native batch_key; omitted retains the installed function default.'),
    ('check_values', 'Native check_values; omitted retains the installed function default.'),
    ('svd_solver', 'pca installed native solver.'),
    ('return_info', 'pca native return_info flag; AnnData results retain native slots.'),
    ('use_highly_variable', 'pca uses the existing var highly_variable flag when requested.'),
    ('dtype', 'pca native result dtype.'),
    ('n_neighbors', 'neighbors native neighborhood size; omitted retains its installed default.'),
    ('n_pcs', 'neighbors/tsne component count; native representation selection applies.'),
    ('use_rep', 'Exact native representation selector; omission retains installed selection/default.'),
    ('knn', 'neighbors hard nearest-neighbor threshold versus full kernel.'),
    ('method', 'Native method for the selected function; omitted retains its installed default.'),
    ('metric', 'neighbors/tsne native metric name; no executable callable.'),
    ('metric_kwds', 'neighbors JSON metric options; no executable callbacks.'),
    ('resolution', 'leiden native clustering resolution.'),
    ('restrict_to', 'Native restrict_to; omitted retains the installed function default.'),
    ('directed', 'Native directed; omitted retains the installed function default.'),
    ('use_weights', 'Native use_weights; omitted retains the installed function default.'),
    ('n_iterations', 'leiden native iterations, -1 until convergence; cap 1000 positive iterations.'),
    ('obsp', 'Native obsp; omitted retains the installed function default.'),
    ('min_dist', 'umap native min_dist; omitted uses installed default.'),
    ('spread', 'umap native spread; omitted uses installed default.'),
    ('n_components', 'Native n_components for the selected function; omitted retains its installed default.'),
    ('maxiter', 'umap native maxiter; omitted uses installed default.'),
    ('alpha', 'Native alpha; omitted retains the installed function default.'),
    ('gamma', 'umap native gamma; omitted uses installed default.'),
    ('negative_sample_rate', 'umap native negative_sample_rate; omitted uses installed default.'),
    ('init_pos', 'umap native init_pos; omitted uses installed default.'),
    ('a', 'umap native a; omitted uses installed default.'),
    ('b', 'umap native b; omitted uses installed default.'),
    ('perplexity', 'tsne native perplexity.'),
    ('early_exaggeration', 'tsne native early exaggeration.'),
    ('learning_rate', 'tsne native learning rate; omitted retains its installed default.'),
    ('use_fast_tsne', 'tsne native implementation choice; no adapter substitution.'),
    ('n_jobs', 'tsne native worker count.'),
    ('groups', 'Native groups; omitted retains the installed function default.'),
    ('use_rna_velocity', 'Native use_rna_velocity; omitted retains the installed function default.'),
    ('model', 'Native model; omitted retains the installed function default.'),
    ('groupby', 'Exact obs grouping field; native statistics retain supplied feature semantics.'),
    ('reference', 'Native reference; omitted retains the installed function default.'),
    ('n_genes', 'Native n_genes; omitted retains the installed function default.'),
    ('rankby_abs', 'Native rankby_abs; omitted retains the installed function default.'),
    ('pts', 'Native pts; omitted retains the installed function default.'),
    ('corr_method', 'Native corr_method; omitted retains the installed function default.'),
    ('tie_correct', 'Native tie_correct; omitted retains the installed function default.'),
    ('gene_list', 'Exact supplied feature IDs for score_genes; the native name does not establish gene annotation.'),
    ('ctrl_size', 'Native ctrl_size; omitted retains the installed function default.'),
    ('gene_pool', 'Exact background feature IDs for score_genes.'),
    ('score_name', 'Native score_name; omitted retains the installed function default.'),
    ('n_dcs', 'dpt diffusion component count; existing uns.iroot identifies the supplied root cell.'),
    ('n_branchings', 'dpt native branching count.'),
    ('min_group_size', 'dpt minimum group fraction or size.'),
    ('allow_kendall_tau_shift', 'dpt native branching behavior.'),
), 'Native AnnData QC, normalization/log/scale, variable features, PCA/graphs, clustering/embeddings, PAGA, feature ranking/scoring and DPT. Non-inplace returns are separate in uns.agent_native_return; subset records exact source positions. Names such as genes retain supplied feature meaning. Native defaults and internal representation selection apply; no Agent-added preprocessing.')

PACKAGES = (EPISCANPY, SNAPATAC2, SIGNAC, SCANPY)
