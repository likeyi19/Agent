"""Fixed isolated dispatch for native raw preprocessing; no user code."""
import hashlib
import importlib.metadata
from importlib import util
import inspect
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True
import traceback


def otsu_thresholds(obs, kwargs):
    """pycisTopic-style sample thresholds on SnapATAC2's own QC measurements.

    The package supplies Otsu; Agent only selects/transforms its input and
    passes the resulting scalars to native filter_cells. Explicit minima,
    including None, take precedence independently.
    """
    import numpy as np
    from skimage.filters import threshold_otsu
    selected, details = {}, {}
    for key, field, floor, logarithmic in (
            ('min_counts', 'n_fragment', 100, True), ('min_tsse', 'tsse', 1.0, False)):
        if key in kwargs:
            details[key] = dict(method='explicit', value=kwargs[key])
            continue
        values = np.asarray(obs[field], dtype=float)
        fitting = values[np.isfinite(values) & (values >= floor)]
        if not len(fitting):
            raise ValueError('Otsu has no eligible ' + field + ' observations; supply an explicit ' + key)
        transformed = np.log10(fitting) if logarithmic else fitting
        threshold = float(threshold_otsu(transformed, nbins=5000))
        selected[key] = float(10 ** threshold) if logarithmic else threshold
        details[key] = dict(method='skimage.filters.threshold_otsu', field=field,
            transform='log10' if logarithmic else 'identity', fitting_floor=floor,
            fitting_barcodes=len(fitting), nonfinite_barcodes=int((~np.isfinite(values)).sum()),
            value=selected[key], nbins=5000)
    return selected, dict(method='sample-level Otsu', input_barcodes=len(np.asarray(obs['n_fragment'])), thresholds=details,
        measurement='SnapATAC2 n_fragment and tsse; not pycisTopic fragments-in-peaks or TSS scores',
        comparison='native filter_cells inclusive minima (>=)')


def peak_measurement(source_measurement, strategy):
    """Meaning of the native counting choice, shared by planning and publication."""
    return ({'fragment': 'fragment_counts', 'insertion': 'insertion_counts',
             'paired-insertion': 'paired_insertion_counts'}[strategy]
            if source_measurement == 'paired_fragments' else 'snapatac2_' + source_measurement)


def main(request_path):
    raw = Path(request_path).read_bytes(); req = json.loads(raw); stage = Path(req['stage'])
    try:
        import snapatac2 as snap
        import anndata as ad
        import hdf5plugin  # Decode the native H5AD Zstandard storage, without changing it.
        if importlib.metadata.version('snapatac2') != req['version']:
            raise ValueError('Installed package version differs from recorded dispatch')
        sys.path.insert(0, str(Path(__file__).parent))
        from native_callable_worker import functions as installed_functions
        functions = installed_functions('snapatac2')
        name = req['function_name']; fn = functions[name]; kw = req['kwargs']; inputs = req['raw_inputs']
        implementation_paths = [Path(inspect.getsourcefile(fn)), Path(snap._snapatac2.__file__)]
        def implementation():
            return [dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)
                for path in implementation_paths]
        native_implementation = implementation()
        call = dict(kw); a = None; result = None; slots = []; measurement = inputs.get('matrix_semantics')
        declarations = {k: inputs[k] for k in ('species', 'assembly') if inputs.get(k) is not None}
        threshold_selection = None
        activity_source = None
        source_declarations = {}
        if inputs.get('qc_thresholds') == 'otsu' and name == 'pp.filter_cells':
            from skimage.filters import threshold_otsu
            implementation_paths.append(Path(inspect.getsourcefile(threshold_otsu)))
            native_implementation = implementation()
        if name == 'pp.make_fragment_file':
            compression = kw.get('compression')
            filename = 'fragments.tsv.zst' if compression == 'zstandard' else 'fragments.tsv' if compression is None else 'fragments.tsv.gz'
            call.update(bam_file=inputs['input_path'], output_file=str(stage / filename), tempdir=str(stage))
            result = fn(**call); slots = [filename]; measurement = 'native_bam_fragments'
        elif name == 'pp.import_fragments':
            # FAI -> native dictionary is file adaptation, not reference qualification.
            sizes = kw.get('chrom_sizes')
            if sizes is None:
                sizes = {}
                with open(inputs['chrom_sizes_path']) as stream:
                    for line in stream:
                        fields = line.rstrip('\n').split('\t')
                        if len(fields) < 2 or fields[0] in sizes:
                            raise ValueError('Chromosome sizes require unique names and integer lengths')
                        sizes[fields[0]] = int(fields[1])
            call.update(fragment_file=inputs['input_path'], chrom_sizes=sizes, file=str(stage / 'result.h5ad'), tempdir=str(stage))
            if inputs.get('whitelist_path'): call['whitelist'] = inputs['whitelist_path']
            a = fn(**call); result = a
            measurement = 'native_fragment_positions_and_lengths'; slots = ['obsm.fragment_paired' if call.get('is_paired', True) else 'obsm.fragment_single', 'obs.n_fragment', 'obs.frac_dup', 'obs.frac_mito']
        else:
            shutil.copyfile(inputs['input_path'], stage / 'result.h5ad')
            a = snap.read(str(stage / 'result.h5ad'))
            for key in ('species', 'assembly'):
                if key in a.uns:
                    existing = a.uns[key]
                    source_declarations[key] = existing
                    declarations[key] = existing
            if 'matrix_semantics' in a.uns:
                source_declarations['matrix_semantics'] = a.uns['matrix_semantics']
                measurement = source_declarations['matrix_semantics']
            if name in ('metrics.tsse', 'pp.make_gene_matrix'): call['gene_anno'] = inputs['gene_anno_path']
            if name in ('pp.make_peak_matrix', 'pp.add_tile_matrix', 'pp.make_gene_matrix'):
                source_measurement = ('X_region_aggregation' if kw.get('use_x') else 'paired_fragments' if 'fragment_paired' in a.obsm
                    else 'single_end_insertions' if 'fragment_single' in a.obsm else 'native_base_values')
                if inputs.get('peaks_path'): call['peak_file'] = inputs['peaks_path']
                if not kw.get('inplace', name == 'pp.add_tile_matrix'): call['file'] = str(stage / 'matrix.h5ad')
            if name == 'pp.make_gene_matrix':
                from native_gene_activity import source_metadata
                activity_source = source_metadata(inputs['input_path'], inputs,
                    slot='X' if kw.get('use_x') else 'obsm.fragment_paired' if 'fragment_paired' in a.obsm else 'obsm.fragment_single')
                if not kw.get('use_x'):
                    activity_source['measurement'] = source_measurement
            if name == 'pp.filter_cells' and inputs.get('qc_thresholds') == 'otsu':
                thresholds, threshold_selection = otsu_thresholds(a.obs, kw)
                call.update(thresholds)
            result = fn(a, **call)
            if name in ('pp.make_peak_matrix', 'pp.add_tile_matrix', 'pp.make_gene_matrix'):
                if result is not None:
                    a.close(); a = result
                strategy = kw.get('counting_strategy', 'paired-insertion')
                measurement = peak_measurement(source_measurement, strategy)
                if name == 'pp.make_gene_matrix':
                    measurement = ('snapatac2_gene_region_sum' if kw.get('use_x') else 'snapatac2_gene_' + measurement)
                slots = ['X']
            elif name == 'metrics.tsse': slots = ['obs.tsse', 'uns.library_tsse', 'uns.frac_overlap_TSS', 'uns.TSS_profile']
            elif name == 'metrics.frag_size_distr': slots = ['uns.' + kw.get('add_key', 'frag_size_distr')]
            elif name in ('pp.filter_cells', 'pp.call_cells'): slots = ['obs']
            if result is not None and not isinstance(result, type(a)):
                # Preserve inplace=False as an independent native return, never apply its mask.
                spec = util.spec_from_file_location('native_matrix_worker', Path(__file__).with_name('native_matrix_worker.py'))
                shared = util.module_from_spec(spec); spec.loader.exec_module(shared)
                a.uns['agent_native_return'] = shared.returned_data(result); slots = ['uns.agent_native_return']
        native_type = type(result).__module__ + '.' + type(result).__name__
        if a is not None:
            a.close()
            if (stage / 'matrix.h5ad').exists():
                (stage / 'result.h5ad').unlink(); (stage / 'matrix.h5ad').rename(stage / 'result.h5ad')
            # Measurement annotation only; the native sparse matrix and fragment storage stay native.
            import h5py
            from anndata.io import write_elem
            with h5py.File(stage / 'result.h5ad', 'r+') as handle:
                uns = handle.require_group('uns')
                if measurement is None:
                    from anndata.io import read_elem
                    measurement = read_elem(uns['matrix_semantics']) if 'matrix_semantics' in uns else 'native_fragment_positions_and_lengths'
                write_elem(uns, 'matrix_semantics', measurement)
                for key, value in declarations.items(): write_elem(uns, key, value)
                if name in ('pp.make_peak_matrix', 'pp.add_tile_matrix'):
                    # Project native coordinate names, without rebuilding counts
                    # or changing the native feature order.
                    import pandas as pd
                    from anndata.io import read_elem
                    var = read_elem(handle['var'])
                    coordinates = [label.rsplit(':', 1) for label in var.index]
                    if all(len(parts) == 2 and len(parts[1].split('-')) == 2 for parts in coordinates):
                        var['chrom'] = [parts[0] for parts in coordinates]
                        var['start'] = [int(parts[1].split('-')[0]) for parts in coordinates]
                        var['end'] = [int(parts[1].split('-')[1]) for parts in coordinates]
                        write_elem(handle, 'var', var)
        effective = {key: parameter.default for key, parameter in inspect.signature(fn).parameters.items()
            if parameter.default is not inspect.Parameter.empty}
        effective.update(call)
        # Files and large chromosome dictionaries are pinned separately; avoid duplicating private locators.
        effective = {k: v for k, v in effective.items() if k not in ('adata', 'data', 'bam_file', 'fragment_file',
            'output_file', 'file', 'tempdir', 'gene_anno', 'peak_file', 'chrom_sizes', 'whitelist')}
        source = Path(inspect.getsourcefile(fn))
        if implementation() != native_implementation:
            raise ValueError('Native package implementation changed during invocation')
        receipt = dict(completion='succeeded', package='snapatac2', function_name=name,
            request_sha256=hashlib.sha256(raw).hexdigest(), versions={p: importlib.metadata.version(p)
                for p in ('snapatac2', 'anndata', 'numpy', 'scipy', 'hdf5plugin')},
            native_signature=str(inspect.signature(fn)), native_output_type=native_type,
            function_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), result_slots=slots,
            effective_parameters=effective, measurement=measurement, source_declarations=source_declarations, implementation=native_implementation)
        if activity_source is not None:
            from native_gene_activity import axis_identity, source_metadata
            receipt['gene_activity_axes'] = axis_identity(stage / 'result.h5ad')
            receipt['gene_activity_source'] = activity_source
            receipt['inherited_slots'] = sorted(set(activity_source['slots']) & set(
                source_metadata(stage / 'result.h5ad', {})['slots']) - {'uns.matrix_semantics'})
        if threshold_selection is not None:
            receipt['threshold_selection'] = threshold_selection
            receipt['versions']['scikit-image'] = importlib.metadata.version('scikit-image')
        if name in ('pp.filter_cells', 'pp.call_cells') and result is not None:
            receipt['selected_count'] = len(result)
        (stage / 'native-receipt.json').write_text(json.dumps(receipt, indent=2))
        return 0
    except BaseException as exc:
        (stage / 'native-error.json').write_text(json.dumps(dict(message=str(exc), type=type(exc).__name__)))
        traceback.print_exc(); return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
