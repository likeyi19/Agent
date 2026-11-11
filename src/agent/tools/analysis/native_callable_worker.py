"""Installed Python API discovery and data-only invocation in an isolated runtime.

This module is also a script: it imports no Agent application or provider code.
Object adapters are shared across public functions, never scientific pipelines.
"""
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True
import traceback

PUBLIC_NAMESPACES = ('', 'pp', 'tl', 'pl', 'metrics', 'get', 'utils', 'logging')


def functions(package):
    module = importlib.import_module(package)
    found = {}
    for namespace in PUBLIC_NAMESPACES:
        owner = getattr(module, namespace, None) if namespace else module
        if owner is None:
            continue
        for name in dir(owner):
            if name.startswith('_'):
                continue
            value = getattr(owner, name)
            if inspect.isroutine(value) and getattr(value, '__module__', '').split('.')[0] in (package, 'scanpy' if package == 'episcanpy' else package):
                found[(namespace + '.' if namespace else '') + name] = value
    return found


def json_default(value):
    try:
        json.dumps(value, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


def signature_metadata(fn):
    metadata = {}
    import typing
    try:
        annotations = typing.get_type_hints(fn)
    except (NameError, TypeError, AttributeError):
        annotations = {}
    for key, parameter in inspect.signature(fn).parameters.items():
        item = {}
        annotation = annotations.get(key, parameter.annotation)
        if isinstance(annotation, str):
            if annotation in ('bool', 'int', 'float', 'str'):
                item['types'] = [annotation]
            # Native forward annotations may be Literal strings. Resolve only
            # the installed package's own type annotations, never user code.
        else:
            if typing.get_origin(annotation) is typing.Literal:
                item['choices'] = list(typing.get_args(annotation))
            if annotation in (bool, int, float, str):
                item['types'] = [annotation.__name__]
        default = parameter.default
        if default is not inspect.Parameter.empty:
            try:
                serialized = json.loads(json.dumps(default, allow_nan=False))
            except (TypeError, ValueError):
                pass
            else:
                item['default_when_omitted'] = serialized
                if 'types' not in item and type(default) in (bool, int, float, str):
                    item['types'] = [type(default).__name__]
        metadata[key] = item
    return metadata


def metadata(name, function):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return dict(name=name, signature=None, parameters=[], limitation='Installed callable has no inspectable signature.')
    parameters = []
    shared = signature_metadata(function)
    for p in signature.parameters.values():
        default = p.default
        annotation = str(p.annotation) if p.annotation is not inspect.Parameter.empty else ''
        kind = ('boolean' if type(default) is bool else 'integer' if type(default) is int else
                'number' if type(default) is float else 'string' if type(default) is str else
                'list' if type(default) in (list, tuple) else 'value')
        if kind == 'value':
            for hint, candidate in (('bool', 'boolean'), ('int', 'integer'), ('float', 'number'), ('str', 'string')):
                if annotation in (hint, "<class '" + hint + "'>"):
                    kind = candidate
        parameters.append(dict(name=p.name, kind=kind, annotation=annotation, choices=shared.get(p.name, {}).get('choices', []),
            required=default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD),
            parameter_kind=p.kind.name, default_known=default is not inspect.Parameter.empty and json_default(default),
            default=default if default is not inspect.Parameter.empty and json_default(default) else None,
            default_label=str(default) if default is not inspect.Parameter.empty and json_default(default) else 'Native default'))
    return dict(name=name, signature=str(signature), parameters=parameters,
        description=(inspect.getdoc(function) or '').split('\n\n')[0], limitation=None)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class RepresentationError(TypeError):
    pass


def matplotlib_figure(value):
    if 'matplotlib' not in sys.modules:
        return None
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    return value if isinstance(value, Figure) else value.figure if isinstance(value, Axes) else None


def open_figures():
    # Inspect only genuine pyplot-managed figures in this isolated invocation.
    # A closed/unregistered figure cannot be recovered from a None return.
    helpers = sys.modules.get('matplotlib._pylab_helpers')
    return [] if helpers is None else [m.canvas.figure for m in helpers.Gcf.get_all_fig_managers()]


def table_preview(value):
    frame = value.ndim == 2
    small = value.iloc[:8, :8] if frame else value.head(8)
    clip = lambda x: x[:500] if isinstance(x, str) else x
    small = small.applymap(clip) if frame else small.map(clip)
    small.index = small.index.map(clip)
    if frame: small.columns = small.columns.map(clip)
    return small.to_json(orient='split')


def preserve(value, root, name, figures=None):
    """Full typed data or qualified visual artifacts; previews are bounded."""
    import numpy as np
    from scipy import sparse
    import pandas as pd
    import anndata as ad
    typename = type(value).__module__ + '.' + type(value).__qualname__
    result = dict(type=typename)
    if figures is None:
        figures = {}
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not math.isfinite(value):
            result.update(encoding='float', value=repr(value))
        else:
            result.update(encoding='json', value=value)
    elif isinstance(value, np.generic):
        path = root / (name + '.npy')
        np.save(path, value, allow_pickle=False)
        result.update(encoding='numpy', file=path.name, shape=[], dtype=str(value.dtype))
    elif isinstance(value, ad.AnnData):
        path = root / (name + '.h5ad')
        value.write_h5ad(path)
        result.update(encoding='anndata', file=path.name, shape=list(value.shape))
    elif sparse.issparse(value):
        path = root / (name + '.npz')
        sparse.save_npz(path, value)
        result.update(encoding='sparse', file=path.name, shape=list(value.shape), dtype=str(value.dtype), format=value.format,
            preview=repr((value if value.format in ('csr', 'csc') else value.tocsr())[:8, :8].toarray().tolist()))
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise RepresentationError(f'{typename} with object dtype has no safe lossless array representation')
        path = root / (name + '.npy')
        np.save(path, value, allow_pickle=False)
        result.update(encoding='numpy', file=path.name, shape=list(value.shape), dtype=str(value.dtype))
    elif isinstance(value, (pd.DataFrame, pd.Series)):
        # AnnData's typed element encoding retains indexes, categorical values,
        # nullable columns and column order. Fail honestly on unsupported encodings.
        import h5py
        from anndata._io.specs import write_elem
        path = root / (name + '.h5')
        frame = value.to_frame(name='value') if isinstance(value, pd.Series) else value
        with h5py.File(path, 'w') as handle:
            write_elem(handle, 'value', frame)
        result.update(encoding='series' if isinstance(value, pd.Series) else 'dataframe', file=path.name,
            shape=list(value.shape), columns=[str(c) for c in frame.columns],
            index_name=preserve(value.index.name, root, name + '-index-name'),
            columns_name=preserve(frame.columns.name, root, name + '-columns-name'))
        if isinstance(value.index, pd.RangeIndex):
            result['range_index'] = [value.index.start, value.index.stop, value.index.step]
        if isinstance(value, pd.Series):
            result['series_name'] = preserve(value.name, root, name + '-series-name')
        # Refuse encodings which would silently change a pandas dtype or axis.
        restored = restore(result, root)
        try:
            if isinstance(value, pd.Series): pd.testing.assert_series_equal(value, restored)
            else: pd.testing.assert_frame_equal(value, restored)
        except AssertionError as exc:
            raise RepresentationError('The native table has no lossless supported encoding: ' + str(exc)) from exc
        result['preview'] = table_preview(value)
    elif typename == 'polars.dataframe.frame.DataFrame':
        path = root / (name + '.arrow')
        value.write_ipc(path)
        result.update(encoding='polars', file=path.name, shape=list(value.shape))
    elif (figure := matplotlib_figure(value)) is not None:
        if figure not in figures:
            images = {}
            for extension in ('png', 'svg'):
                path = root / (name + '.' + extension)
                figure.savefig(path, format=extension)
                images[extension] = path.name
            figures[figure] = images
        result.update(encoding='matplotlib', file=figures[figure]['png'], images=figures[figure],
            representation='rendered_figure', restorable=False,
            limitation='Rendered images preserve the visual output, not a restorable Python Figure or Axes.')
    elif type(value) in (tuple, list):
        result.update(encoding='tuple' if type(value) is tuple else 'list',
            items=[preserve(v, root, name + '-' + str(i), figures) for i, v in enumerate(value)])
    elif type(value) is dict:
        result.update(encoding='dict', items=[dict(key=preserve(k, root, name + '-key-' + str(i), figures),
            value=preserve(v, root, name + '-value-' + str(i), figures)) for i, (k, v) in enumerate(value.items())])
    else:
        raise RepresentationError(f'Native execution returned {typename}; a faithful result adapter is unavailable')
    if 'items' in result:
        members = ([v for entry in result['items'] for v in (entry['key'], entry['value'])]
            if result['encoding'] == 'dict' else result['items'])
        if any(v.get('restorable') is False for v in members):
            result['restorable'] = False
    return result


def restore(descriptor, root):
    import numpy as np
    from scipy import sparse
    import pandas as pd
    import anndata as ad
    encoding = descriptor['encoding']
    if descriptor.get('restorable') is False:
        raise RepresentationError('The selected visual representation cannot restore the original native object: ' + descriptor['type'])
    path = root / descriptor['file'] if 'file' in descriptor else None
    if encoding == 'json': return descriptor['value']
    if encoding == 'float': return float(descriptor['value'])
    if encoding == 'anndata': return ad.read_h5ad(path)
    if encoding == 'numpy':
        value = np.load(path, allow_pickle=False)
        return value[()] if descriptor['shape'] == [] and descriptor['type'] != 'numpy.ndarray' else value
    if encoding == 'sparse': return sparse.load_npz(path)
    if encoding in ('dataframe', 'series'):
        import h5py
        from anndata._io.specs import read_elem
        with h5py.File(path, 'r') as handle: value = read_elem(handle['value'])
        if 'range_index' in descriptor: value.index = pd.RangeIndex(*descriptor['range_index'])
        if 'index_name' in descriptor: value.index.name = restore(descriptor['index_name'], root)
        if 'columns_name' in descriptor: value.columns.name = restore(descriptor['columns_name'], root)
        if encoding == 'series':
            value = value.iloc[:, 0]
            if 'series_name' in descriptor: value.name = restore(descriptor['series_name'], root)
        return value
    if encoding == 'polars':
        import polars as pl
        return pl.read_ipc(path)
    if encoding in ('list', 'tuple'):
        values = [restore(v, root) for v in descriptor['items']]
        return tuple(values) if encoding == 'tuple' else values
    if encoding == 'dict': return {restore(v['key'], root): restore(v['value'], root) for v in descriptor['items']}
    raise RepresentationError('The selected result representation has no input adapter: ' + encoding)


def construct(binding, work):
    import numpy as np
    import pandas as pd
    import anndata as ad
    from scipy import sparse
    from scipy.io import mmread
    adapter = binding['adapter']
    if adapter == 'value': return binding['value']
    if adapter == 'array': return np.asarray(binding['value'])
    if adapter == 'accepted':
        descriptor = selected_descriptor(binding)
        if descriptor['encoding'] == 'file':
            path = Path(binding['root']) / descriptor['file']
            target = work / descriptor['file']; shutil.copyfile(path, target)
            return str(target)
        return restore(descriptor, Path(binding['root']))
    path = Path(binding['path'])
    if sha(path) != binding['sha256']: raise ValueError('Bound input bytes changed')
    if adapter == 'anndata': return ad.read_h5ad(path)
    if adapter in ('dataframe', 'column', 'matrix'):
        frame = pd.read_csv(path, sep=binding.get('separator', ','), **binding.get('options', {}))
        if adapter == 'column': return frame[binding['column']].tolist()
        if adapter == 'matrix': return frame.to_numpy()
        return frame
    if adapter == 'sparse': return sparse.load_npz(path) if binding.get('format') == 'npz' else mmread(path)
    if adapter == 'numpy': return np.load(path, allow_pickle=False)
    if adapter == 'file':
        # Native readers/writers can only see a task-local copy via this port.
        destination = work / ('input-' + str(len(list(work.iterdir()))) + path.suffix)
        shutil.copyfile(path, destination)
        return str(destination)
    raise RepresentationError('Unsupported input object adapter: ' + adapter)


def selected_descriptor(binding):
    path = Path(binding['result_path'])
    if sha(path) != binding['result_sha256']: raise ValueError('Accepted result representation changed')
    data = json.loads(path.read_text())
    segments = binding['selection'].split('/')
    head = segments.pop(0)
    if head == 'return': value = data['return_value']
    elif head.startswith('modified:'): value = data['modified_inputs'][head.removeprefix('modified:')]
    elif head.startswith('file:'):
        value = dict(type='native_file', encoding='file', file=data['produced_files'][int(head.removeprefix('file:'))]['file'])
    else: raise ValueError('Unknown accepted result selection')
    for segment in segments:
        item = value['items'][int(segment)]
        value = item['value'] if value['encoding'] == 'dict' else item
    return value


def fingerprint(value, root, name):
    directory = root / name
    directory.mkdir()
    try:
        descriptor = preserve(value, directory, 'object')
        # HDF5 allocation metadata can differ between writes. Compare logical
        # encoded datasets and attributes, including complete arrays, instead.
        digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode())
        import h5py
        for path in sorted(directory.iterdir()):
            if path.suffix in ('.h5', '.h5ad'):
                with h5py.File(path, 'r') as handle:
                    def visit(key, item):
                        digest.update(key.encode())
                        digest.update(repr(sorted(item.attrs.items())).encode())
                        if isinstance(item, h5py.Dataset):
                            array = item[()]
                            digest.update(str(item.dtype).encode())
                            digest.update(repr(array.tolist()).encode() if getattr(array, 'dtype', None) is not None and array.dtype.hasobject else array.tobytes() if hasattr(array, 'tobytes') else repr(array).encode())
                    handle.visititems(visit)
            else: digest.update(sha(path).encode())
        return digest.hexdigest()
    finally:
        shutil.rmtree(directory)


def preview(descriptor, root, depth=0):
    result = {k: v for k, v in descriptor.items() if k not in ('value', 'items', 'index_name', 'columns_name', 'series_name')}
    if descriptor['encoding'] == 'json':
        value = descriptor['value']
        result['value'] = value[:500] if isinstance(value, str) else value
        if isinstance(value, str): result['length'] = len(value)
    elif descriptor['encoding'] == 'float':
        result['preview'] = descriptor['value']
    elif 'items' in descriptor:
        result['length'] = len(descriptor['items'])
        if depth < 4:
            result['items'] = [dict(key=preview(v['key'], root, depth+1), value=preview(v['value'], root, depth+1))
                if descriptor['encoding'] == 'dict' else preview(v, root, depth+1) for v in descriptor['items'][:12]]
    elif descriptor['encoding'] in ('numpy', 'polars'):
        if descriptor['encoding'] == 'numpy':
            import numpy as np
            value = np.load(root / descriptor['file'], allow_pickle=False, mmap_mode='r')
        elif descriptor['encoding'] == 'polars':
            import polars as pl
            value = pl.read_ipc(root / descriptor['file'], n_rows=8,
                columns=list(range(min(descriptor['shape'][1], 8))))
        if descriptor['encoding'] == 'polars':
            result['preview'] = [{str(k)[:500]: v[:500] if isinstance(v, str) else v
                for k, v in row.items()} for row in value.select(value.columns[:8]).head(8).to_dicts()]
        else:
            import numpy as np
            values = np.asarray(value).flat[:24].tolist()
            result['preview'] = repr([v[:500] if isinstance(v, str) else v for v in values])
    if 'columns' in result:
        result['columns'] = [c[:500] for c in result['columns'][:8]]
    return result


def invoke(request):
    package, name = request['package'], request['function']
    function = functions(package)[name]
    root = Path(request['stage']); work = root / 'work'; work.mkdir()
    output = root / 'objects'; output.mkdir()
    parameters = dict(request['parameters'])
    inputs = {key: construct(binding, work) for key, binding in request['inputs'].items()}
    if parameters.keys() & inputs.keys(): raise ValueError('An argument cannot be both a parameter and an input')
    signature = inspect.signature(function)
    supplied = {**parameters, **inputs}
    # File outputs use explicit managed destinations, never arbitrary host paths.
    for key, relative in request.get('outputs', {}).items():
        target = work / relative
        if not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Output paths must be relative to the managed execution directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        supplied[key] = str(target)
    positional = []
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            if parameter.name in supplied:
                positional.append(supplied.pop(parameter.name))
            elif parameter.default is not inspect.Parameter.empty:
                # A later supplied positional-only argument needs earlier slots.
                later = list(signature.parameters).index(parameter.name)
                if any(p.name in supplied and p.kind is inspect.Parameter.POSITIONAL_ONLY
                       for p in list(signature.parameters.values())[later + 1:]):
                    positional.append(parameter.default)
    bound = signature.bind(*positional, **supplied)
    import importlib.metadata
    module = importlib.import_module(function.__module__)
    source = inspect.getsourcefile(function) if inspect.isfunction(function) else getattr(module, '__file__', None)
    implementation = [] if source is None else [dict(path=str(Path(source).resolve()), sha256=sha(source), size_bytes=Path(source).stat().st_size)]
    before_files = {p: sha(p) for p in work.rglob('*') if p.is_file()}
    before = {key: fingerprint(value, root, 'before-' + key) for key, value in inputs.items()}
    before_figures = set(open_figures())
    os.chdir(work)
    result = function(*bound.args, **bound.kwargs)
    receipt = dict(native_execution='succeeded', package=package, function=name,
        native_signature=str(signature), supplied_parameters=parameters,
        versions={package: importlib.metadata.version(package)}, implementation=implementation,
        native_output_type=type(result).__module__ + '.' + type(result).__qualname__)
    (root / 'execution.json').write_text(json.dumps(receipt))
    # Successful execution remains recorded even when representation fails.
    representation_errors = []
    figures = {}
    try:
        returned = preserve(result, output, 'return', figures)
    except Exception as exc:
        returned = None
        representation_errors.append(str(exc))
    mutations = {}
    for key, value in inputs.items():
        try:
            if fingerprint(value, root, 'after-' + key) != before[key]:
                mutations[key] = preserve(value, output, 'modified-' + key, figures)
        except Exception as exc:
            representation_errors.append(key + ': ' + str(exc))
    produced = []
    for figure in open_figures():
        if figure in before_figures or figure in figures:
            continue
        try:
            captured = preserve(figure, output, 'figure-' + str(len(figures)), figures)
            for extension, file in captured['images'].items():
                produced.append(dict(name='Captured Matplotlib figure ' + str(len(figures)) + '.' + extension,
                    file=file, representation='rendered_figure', restorable=False))
        except Exception as exc:
            representation_errors.append('Generated figure: ' + str(exc))
    for path in sorted(work.rglob('*')):
        if path.is_symlink(): raise RepresentationError('Native-produced symbolic links are not portable scientific outputs')
        if path.is_file() and before_files.get(path) != sha(path):
            target = output / ('file-' + str(len(produced)) + path.suffix)
            shutil.copyfile(path, target)
            produced.append(dict(name=str(path.relative_to(work)), file=target.name))
    data = dict(return_value=returned, modified_inputs=mutations, produced_files=produced)
    if representation_errors:
        (root / 'unpublished.json').write_text(json.dumps(data, allow_nan=False))
        raise RepresentationError('; '.join(representation_errors))
    (root / 'result.json').write_text(json.dumps(data, allow_nan=False))
    view = dict(return_value=preview(returned, output), modified_inputs={k: preview(v, output) for k, v in mutations.items()}, produced_files=produced)
    (root / 'preview.json').write_text(json.dumps(view, allow_nan=False))
    receipt['result_representation'] = 'succeeded'
    (root / 'execution.json').write_text(json.dumps(receipt))


def main():
    if sys.argv[1] == '--discover':
        package = sys.argv[2]
        selected = (sys.argv[3] or None) if len(sys.argv) > 3 else None
        import importlib.metadata
        values = functions(package)
        encoded = json.dumps(dict(package=package, version=importlib.metadata.version(package),
            functions=[metadata(k, v) if selected else dict(name=k) for k, v in values.items() if selected is None or k == selected]))
        if len(sys.argv) > 4: Path(sys.argv[4]).write_text(encoded)
        else: print(encoded)
        return
    request = json.loads(Path(sys.argv[1]).read_text())
    try:
        invoke(request)
    except Exception as exc:
        root = Path(request['stage'])
        success = (root / 'execution.json').is_file()
        (root / 'error.json').write_text(json.dumps(dict(message=str(exc), type=type(exc).__name__, native_execution='succeeded' if success else 'failed', result_representation='failed' if success else 'not_started')))
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__': main()
