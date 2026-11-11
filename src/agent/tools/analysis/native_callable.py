"""Data-only native invocation through the existing publication and process owners."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
from functools import lru_cache
from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import digest, AuthorityError
from agent.tools._file_identity import file_closure
from agent.tools._native_process import call, run, r_environment
from agent.tools.data.authority_context import current
from .native_publication import NativePublication
from .sequence_prediction import write, read
from .native_runtimes import R_RUNTIMES, rscript, macs3

TOOL = 'invoke_native_function'
KIND = 'native_callable'
CONTRACT = 'native-callable.v1'
PROFILE = digest(dict(contract=CONTRACT, adapter='typed objects; distinct returns mutations and files.v1'))
VERIFIER = dict(id='agent.native_callable', compatibility_version='1')
POLICY = 'native-callable-v1'
DELIVERY_FILES = {'outputs.zip': ('Complete native outputs', 'application/zip')}
RESULT_FIELDS = dict(status=(str,), contract_version=(str,), manifest_path=(str,), manifest_sha256=(str,),
    function_name=(str,), preview=(dict,), parameters=(dict,), native_output_type=(str,))


def arguments(values):
    return _serialize(values)


def files(value):
    names = value['output_files']
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
        raise AuthorityError('Native publication files escape their managed destination')
    return names


def delivery_files(value):
    """Describe only actual data files in the accepted managed object closure."""
    import mimetypes
    root = Path(value['arguments']['output_dir']) / 'objects'
    return {Path(pin['path']).name: ('Native output: ' + Path(pin['path']).name,
        mimetypes.guess_type(pin['path'])[0] or 'application/octet-stream')
        for pin in value['files'] if Path(pin['path']).parent == root}


publication = NativePublication(tool=TOOL, kind=KIND, contract=CONTRACT, profile=PROFILE,
    verifier=VERIFIER, files=files, arguments=arguments, outputs={})
load, result, verify_publication, verify_result, recover = (getattr(publication, k) for k in ('load', 'result', 'verify_publication', 'verify_result', 'recover'))


def packages():
    # Existing trusted runtime configuration remains the package boundary.
    from .native_matrix import PACKAGES
    return {**{p.package: p for p in PACKAGES}, **{p: None for p in R_RUNTIMES}, 'MACS3': None}


def executable(package):
    owner = packages()[package]
    if package == 'MACS3': return macs3()
    if package in R_RUNTIMES:
        path = rscript(package)
        if not path.is_file(): raise ValueError('Provision the installed ' + package + ' runtime')
        return path
    from .sequence_prediction import library
    override = os.environ.get('AGENT_' + package.upper() + '_PYTHON')
    path = Path(override) if override else library().root.parent / 'runtimes' / owner.runtime / 'env/bin/python'
    if not path.is_file(): raise ValueError('Provision the installed ' + package + ' runtime')
    return path


@lru_cache(maxsize=128)
def _discover(exe, package, name):
    # No data are loaded and no provider is imported. The shared process owner
    # applies the same resource boundary as execution.
    with tempfile.TemporaryDirectory(prefix='agent-native-metadata-') as temp:
        log = Path(temp) / 'metadata.json'
        is_r = package in R_RUNTIMES
        script = Path(__file__).parent / ('r/native_callable.R' if is_r else 'native_callable_worker.py')
        command = [exe, '--vanilla' if is_r else '-I', str(script), '--discover', package]
        if package == 'MACS3':
            command = [str(Path(exe).parent / 'python'), '-I', str(Path(__file__).with_name('native_cli_worker.py')), '--discover', exe]
        metadata_path = Path(temp) / 'result.json'
        command.extend([name or '', str(metadata_path)])
        code = run(command, log, environment=r_environment())
        if code: raise ValueError('Installed native metadata discovery failed: ' + log.read_text()[-2000:])
        return json.loads(metadata_path.read_text())


def discover(package, function=None):
    return _discover(str(executable(package)), package, function)


def invoke(package, function_name, inputs, parameters, output_dir, outputs=None):
    return execute(dict(package=package, function_name=function_name, inputs=inputs,
        parameters=parameters, output_dir=output_dir, **({'outputs': outputs} if outputs is not None else {})))


def execute(values, execution_identity=None):
    values = arguments(values)
    return call('native_callable', dict(values=values, execution_identity=execution_identity),
        lambda: _execute(values, execution_identity))


def _execute(values, execution_identity):
    args = arguments(values)
    exe = executable(args['package'])
    root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()): raise ValueError('Native invocation destination is not empty; no automatic rerun')
    sources = []
    for binding in args['inputs'].values():
        if 'path' in binding:
            pin = file_closure([binding['path']])[0]
            if pin['sha256'] != binding['sha256']: raise AuthorityError('Selected input bytes changed')
            sources.append(pin)
            if binding['adapter'] == 'anndata' or binding['adapter'] == 'r_matrix' and binding.get('format') == 'h5ad':
                from .peak_h5ad import inspect as inspect_h5ad
                inspect_h5ad(binding['path'], require_matrix=False, max_logical_bytes=None)
        elif binding['adapter'] == 'accepted':
            for pin in binding['files']:
                if file_closure([pin['path']])[0] != pin: raise AuthorityError('Accepted input bytes changed')
                sources.append(pin)
    sources = sorted({p['path']: p for p in sources}.values(), key=lambda p: p['path'])
    is_r = args['package'] in R_RUNTIMES
    script = Path(__file__).parent / ('r/native_callable.R' if is_r else 'native_callable_worker.py')
    if args['package'] == 'MACS3': script = Path(__file__).with_name('native_cli_worker.py')
    implementation = file_closure([script, exe.resolve()])
    with tempfile.TemporaryDirectory(prefix='.invocation-', dir=root) as temp:
        stage = Path(temp)
        request = dict(package=args['package'], function=args['function_name'], inputs=args['inputs'],
            parameters=args['parameters'], outputs=args.get('outputs', {}), stage=str(stage))
        if is_r:
            request['inputs'] = prepare_r_inputs(args['inputs'], stage)
        if args['package'] == 'MACS3': request['executable'] = str(exe)
        write(stage / 'request.json', request)
        argument = str(stage / 'request.json')
        runner = exe.parent / 'python' if args['package'] == 'MACS3' else exe
        code = run([str(runner), '--vanilla' if is_r else '-I', str(script), argument], stage / 'native.log', environment=r_environment())
        shutil.copyfile(stage / 'native.log', root.parent / 'native.log')
        if code:
            error = read(stage / 'error.json') if (stage / 'error.json').is_file() else dict(message='Native process failed; see retained diagnostic', native_execution='failed')
            write(root / 'failure.json', error)
            if (stage / 'execution.json').is_file(): shutil.copyfile(stage / 'execution.json', root / 'execution.json')
            if (stage / 'unpublished.json').is_file():
                shutil.copyfile(stage / 'unpublished.json', root / 'unpublished.json')
                shutil.copytree(stage / 'objects', root / 'unpublished-objects')
            raise ValueError(error['native_execution'] + ': ' + error['message'])
        receipt = read(stage / 'execution.json')
        native_implementation = receipt.get('implementation', [])
        if native_implementation and file_closure(p['path'] for p in native_implementation) != sorted(native_implementation, key=lambda p:p['path']):
            raise AuthorityError('Native implementation changed during execution')
        implementation = sorted({p['path']: p for p in implementation + native_implementation}.values(), key=lambda p:p['path'])
        receipt['implementation'] = implementation
        receipt['input_construction'] = args['inputs']
        for name in ('result.json', 'preview.json'):
            shutil.copyfile(stage / name, root / name)
        shutil.move(str(stage / 'objects'), root / 'objects')
        write(root / 'execution.json', receipt)
    if file_closure(p['path'] for p in sources) != sources or file_closure(p['path'] for p in implementation) != implementation:
        raise AuthorityError('Input or invocation implementation changed during native execution')
    with zipfile.ZipFile(root / 'outputs.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(root / 'result.json', 'result.json')
        for path in sorted((root / 'objects').iterdir()): archive.write(path, 'objects/' + path.name)
    summary = dict(status='success', contract_version=CONTRACT,
        function_name=args['package'] + '.' + args['function_name'], preview=read(root / 'preview.json'),
        parameters=args['parameters'], native_output_type=receipt['native_output_type'])
    output_files = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())
    manifest = dict(contract=CONTRACT, profile=PROFILE, arguments=args, execution_identity=execution_identity,
        native_completion='succeeded', sources=sorted(sources + implementation, key=lambda p: p['path']), output_files=output_files,
        files=file_closure(root / n for n in output_files), result=summary)
    path = root / 'manifest.json'; checksum = write(path, manifest)
    if current() is not None: current().register_execution(KIND, root, execution_identity)
    return verify_publication(path, expected_sha256=checksum)


def prepare_r_inputs(bindings, stage):
    """Explicit structural conversion, using the existing sparse matrix exchange."""
    import anndata as ad
    from scipy import sparse
    from scipy.io import mmwrite
    from .native_callable_worker import construct, selected_descriptor
    work = stage / 'work'; work.mkdir()
    (stage / 'objects').mkdir()
    result = {}
    for key, binding in bindings.items():
        adapter = binding['adapter']
        if adapter == 'accepted' and binding['descriptor']['encoding'] == 'r':
            descriptor = selected_descriptor(binding)
            result[key] = dict(adapter='r_accepted', path=str(Path(binding['root']) / descriptor['file']))
        elif adapter == 'accepted' and binding['descriptor']['encoding'] == 'file':
            result[key] = dict(adapter='file', path=construct(binding, work))
        elif adapter in ('r_matrix', 'anndata', 'matrix', 'array', 'sparse', 'numpy', 'accepted'):
            specification = dict(binding)
            if adapter == 'r_matrix':
                specification['adapter'] = 'anndata' if binding.get('format') == 'h5ad' else 'matrix'
            value = construct(specification, work)
            rows = columns = None
            if isinstance(value, ad.AnnData):
                rows, columns = value.obs_names.tolist(), value.var_names.tolist()
                value = value.X
            if binding.get('orientation') == 'transpose':
                value = value.T; rows, columns = columns, rows
            destination = stage / (key + '.mtx')
            mmwrite(destination, sparse.coo_matrix(value))
            result[key] = dict(adapter='r_matrix', path=str(destination), row_names=rows, column_names=columns)
        elif adapter == 'dataframe':
            if binding.get('options'):
                raise ValueError('R data-frame construction currently supports an explicit separator and header row; extra parsing options are unavailable')
            result[key] = dict(adapter='r_dataframe', path=binding['path'], separator=binding.get('separator', ','))
        elif adapter in ('column', 'value'):
            result[key] = dict(adapter='value', value=construct(binding, work))
        elif adapter == 'file':
            result[key] = dict(adapter='file', path=construct(binding, work))
        else:
            raise ValueError('No R object adapter for ' + adapter)
    return result
