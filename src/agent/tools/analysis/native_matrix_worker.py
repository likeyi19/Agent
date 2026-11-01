"""Isolated, repository-controlled Python package dispatch; no user code."""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import traceback


def main(request_path):
    raw = Path(request_path).read_bytes(); req = json.loads(raw)
    stage = Path(req['stage'])
    try:
        import anndata as ad
        a = ad.read_h5ad(stage / 'input.h5ad')
        package = req['package']; name = req['function_name']; kwargs = req['kwargs']
        version = importlib.metadata.version(package)
        if version != req['version']:
            raise ValueError('Installed package version differs from qualified dispatch')
        if package == 'episcanpy':
            import episcanpy as epi
            dispatch = {'pp.binarize': epi.pp.binarize, 'pp.tfidf': epi.pp.tfidf,
                'pp.log1p': epi.pp.log1p, 'pp.lsi': epi.pp.lsi}
        elif package == 'snapatac2':
            import snapatac2 as snap
            dispatch = {'pp.select_features': snap.pp.select_features, 'tl.spectral': snap.tl.spectral}
        else:
            raise ValueError('Unsupported package')
        function = dispatch[name]
        result = function(a, **kwargs)
        native_type = type(result).__module__ + '.' + type(result).__name__
        if isinstance(result, ad.AnnData):
            a = result
        elif result is not None:
            if name == 'pp.select_features':
                a.var['selected'] = result
            elif name == 'tl.spectral':
                a.uns['spectral_eigenvalue'], a.obsm['X_spectral'] = result
            else:
                raise ValueError('Unrepresented native result type')
        slots = {
            'pp.binarize': ['X'], 'pp.log1p': ['X', 'uns.log1p'],
            'pp.tfidf': ['layers.' + kwargs.get('layer_tfidf_key', 'tf-idf')],
            'pp.lsi': ['obsm.X_lsi'], 'pp.select_features': ['var.selected'],
            'tl.spectral': ['obsm.X_spectral', 'uns.spectral_eigenvalue'],
        }[name]
        if name == 'pp.select_features' and kwargs.get('inplace', True) and 'count' in a.var:
            slots.append('var.count')
        a.write_h5ad(stage / 'result.h5ad')
        versions = {p: importlib.metadata.version(p) for p in (package, 'anndata', 'numpy', 'scipy', 'scikit-learn')}
        if package == 'episcanpy':
            versions['scanpy'] = importlib.metadata.version('scanpy')
        source = Path(inspect.getsourcefile(function))
        receipt = dict(completion='succeeded', package=package, function_name=name,
            request_sha256=hashlib.sha256(raw).hexdigest(), versions=versions,
            native_signature=str(inspect.signature(function)), native_output_type=native_type, result_slots=slots,
            function_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            measurement={'pp.binarize': 'binary_accessibility', 'pp.log1p': 'log1p_transformed_measurement'}.get(name),
            layer_semantics={kwargs.get('layer_tfidf_key', 'tf-idf'): 'episcanpy_tfidf'} if name == 'pp.tfidf' else {})
        (stage / 'native-receipt.json').write_text(json.dumps(receipt, indent=2))
    except Exception as exc:
        (stage / 'native-error.json').write_text(json.dumps(dict(message=str(exc), type=type(exc).__name__)))
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
