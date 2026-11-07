"""Native Chromap CLI invocation using the shared completion/publication owner.

No FASTQ parser, barcode algorithm, fragment canonicalizer or scientific replay.
Source roles and lane grouping are supplied by the user/Planner, never filenames.
"""
import os
from pathlib import Path
import shutil
import tempfile

from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools._cancellation import cancellation_checkpoint
from agent.tools._native_process import run, r_environment
from agent.tools.analysis.native_matrix import NativeInvocationError, arguments as path_arguments, native_kwargs
from agent.tools.analysis.native_publication import NativePublication
from agent.tools.analysis.sequence_prediction import read, write
from .authority_context import current
from agent.tools._file_identity import file_closure

TOOL = 'run_chromap'
KIND = 'native_chromap'
CONTRACT = 'native-chromap.v1'
POLICY = 'native-chromap-v1'
PROFILE = digest(dict(contract=CONTRACT, adapter='fixed native CLI; supplied roles; original output; completion and integrity.v1'))
VERIFIER = dict(id='agent.native-chromap-completion', compatibility_version='1')
FUNCTIONS = {
    'index': ('kmer', 'window', 'min-frag-length'),
    'map': ('preset', 'error-threshold', 'min-num-seeds', 'max-seed-frequencies', 'max-insert-size',
        'MAPQ-threshold', 'min-read-length', 'trim-adapters', 'remove-pcr-duplicates',
        'remove-pcr-duplicates-at-cell-level', 'remove-pcr-duplicates-at-bulk-level', 'Tn5-shift',
        'low-mem', 'bc-error-threshold', 'bc-probability-threshold', 'num-threads', 'read-format',
        'output-mappings-not-in-whitelist'),
}
NATIVE_ARGUMENTS = tuple((k, 'Native --' + k + '; omission preserves the installed CLI default.')
    for k in sorted(set().union(*map(set, FUNCTIONS.values()))))
SWITCHES = frozenset(('trim-adapters','remove-pcr-duplicates','remove-pcr-duplicates-at-cell-level',
    'remove-pcr-duplicates-at-bulk-level','Tn5-shift','low-mem','output-mappings-not-in-whitelist'))
FACT_FIELDS = ('function_name', 'parameters', 'source_identity', 'software_versions', 'measurement', 'limitations')
RESULT_FIELDS = dict(status=(str,), contract_version=(str,), manifest_path=(str,), manifest_sha256=(str,),
    fragments_path=(str, type(None)), fragments_sha256=(str, type(None)),
    index_path=(str, type(None)), index_sha256=(str, type(None)), function_name=(str,),
    parameters=(dict,), source_identity=(dict,), software_versions=(dict,), measurement=(str,), limitations=(str,))
DELIVERY_FILES = {'fragments.bed': ('Native Chromap BED fragments with original barcodes and support', 'text/plain'),
    'reference.chromap': ('Native reference index', 'application/octet-stream'),
    'metadata.json': ('Exact command parameters and source identities', 'application/json'),
    'native-receipt.json': ('Native completion and executable identity', 'application/json')}
LIMITATIONS = ('Native Chromap output; no independent biological validation, cell calling or source-history inference. '
    'The map interface selects native BED output for interoperability. Explicit native flags are forwarded; omitted flags use native defaults. '
    'Coordinates are Tn5-shifted only when requested by the native preset or --Tn5-shift. Native barcode correction requires its supplied barcode/whitelist inputs. '
    'Duplicate support counts native duplicate groups, not independently validated molecules. Stock Chromap 0.3.2 uses an 8-bit support representation capped at 255; '
    'the recorded support-preserving patched build widens that field. Other builds retain their own native precision; no equality with the historical Agent profile is claimed. '
    'Native output order and format are retained; no compulsory BGZF/TBI packaging or canonicalization.')
PORTS = {'reference': 'reference_path', 'index': 'index_path', 'read1': 'read1_path', 'read2': 'read2_path',
    'barcode': 'barcode_path', 'whitelist': 'whitelist_path'}


def arguments(values):
    lanes = {key: values[key] for key in ('read1_path', 'read2_path', 'barcode_path') if isinstance(values.get(key), (list, tuple))}
    args = path_arguments({key: value for key, value in values.items() if key not in lanes})
    for key, paths in lanes.items():
        args[key] = [path_arguments(dict(input_path=path, output_dir=args['output_dir']))['input_path'] for path in paths]
    return args


def publication_files(value):
    return ('reference.chromap' if value['arguments']['function_name'] == 'index' else 'fragments.bed', 'metadata.json', 'native-receipt.json')


def cli_arguments(kwargs):
    """Represent the native presence-only switches without silently dropping values."""
    command=[]
    for key,value in kwargs.items():
        if key in SWITCHES:
            if value is not True:
                raise ValueError('--'+key+' is a presence-only native switch; false cannot disable a preset. Omit it or select native flags without that preset.')
            command.append('--'+key)
        else:
            if value is None or isinstance(value,(list,dict)):
                raise ValueError('--'+key+' requires one native CLI value')
            command.extend(('--'+key,str(value)))
    return command


publication = NativePublication(tool=TOOL, kind=KIND, contract=CONTRACT, profile=PROFILE, verifier=VERIFIER,
    files=publication_files, arguments=arguments, outputs=lambda value: dict(
        fragments='fragments.bed' if value['arguments']['function_name'] == 'map' else None,
        index='reference.chromap' if value['arguments']['function_name'] == 'index' else None))
load, result = publication.load, publication.result
verify_publication, verify_result, recover = publication.verify_publication, publication.verify_result, publication.recover


def execute(values, execution_identity=None):
    args = arguments(values); kw = native_kwargs(FUNCTIONS, args)
    options=cli_arguments(kw)
    for key in ('num-threads',):
        if type(kw.get(key)) in (int, float) and kw[key] > 32:
            raise ValueError('Chromap threads exceed the execution bound 32')
    # Every external file is a separately captured port, never a CLI expression.
    required = ('reference_path',) if args['function_name'] == 'index' else ('reference_path', 'index_path', 'read1_path', 'read2_path')
    if any(not args.get(key) for key in required):
        raise ValueError('Missing native Chromap input: ' + ', '.join(key for key in required if not args.get(key)))
    sources = []
    for key in PORTS.values():
        if args.get(key) is None:
            continue
        paths = args[key] if isinstance(args[key], list) else [args[key]]
        if len(paths) > 32 or not paths:
            raise ValueError('Native input lane list exceeds execution bound')
        for path in paths:
            if ',' in str(path):
                raise ValueError('Chromap comma-separated source lists cannot represent a path containing a comma')
            pins = file_closure([path])
            expected = args.get(key.removesuffix('_path') + '_sha256')
            if expected is not None and pins[0]['sha256'] != expected:
                raise AuthorityError('Native Chromap input identity changed')
            sources.extend(pins)
    executable = Path(os.environ.get('AGENT_CHROMAP_BIN', ''))
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError('Provision the native Chromap executable through AGENT_CHROMAP_BIN')
    implementation = file_closure([executable.resolve()])
    root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Output directory is not empty; retry requires a new authorized execution')
    with tempfile.TemporaryDirectory(prefix='.native-chromap-', dir=root) as temporary:
        stage = Path(temporary); environment = r_environment()
        version_status = run([str(executable), '--version'], stage / 'version.log', environment=environment, timeout=30)
        if version_status:
            raise NativeInvocationError('Native Chromap --version failed')
        version = (stage / 'version.log').read_text().strip()
        filename = publication_files(dict(arguments=args))[0]
        command = [str(executable)]
        if args['function_name'] == 'index': command.append('--build-index')
        else: command.append('--BED')
        command.extend(options)
        flags = {'reference_path': '--ref', 'index_path': '--index', 'read1_path': '--read1',
            'read2_path': '--read2', 'barcode_path': '--barcode', 'whitelist_path': '--barcode-whitelist'}
        for key, flag in flags.items():
            if args.get(key) is not None:
                value = args[key]
                command.extend((flag, ','.join(map(str, value)) if isinstance(value, list) else str(value)))
        command.extend(('--output', str(stage / filename)))
        status = run(command, stage / 'native.log', environment=environment)
        shutil.copyfile(stage / 'native.log', root / 'native.log')
        if status:
            from agent.tools.analysis import peak_h5ad
            lines=(stage/'native.log').read_text(errors='replace').splitlines()
            message=next((line for line in reversed(lines) if line.strip()),'')[:2000]
            if not message or peak_h5ad._PATH.search(message) or any(ord(c)<32 for c in message):
                message='Native Chromap failed; its private diagnostic is retained'
            raise NativeInvocationError(message)
        if not (stage / filename).is_file():
            raise AuthorityError('Chromap did not produce its declared output')
        expected = {p['path']: p for p in sources + implementation}
        sources = file_closure(expected)
        if sources != sorted(expected.values(), key=lambda p: p['path']):
            raise AuthorityError('Chromap source or executable changed during invocation')
        patched = implementation[0]['sha256'] == '7ba402d0d69d08fedd525a4de83a4dd749670e88d4de507c236b07f03a285649'
        receipt = dict(completion='succeeded', executable=implementation[0], version=version, command=command,
            support_representation='qualified uint64 patch' if patched else 'native build; duplicate-support precision not asserted')
        summary = dict(status='success', contract_version=CONTRACT, function_name='chromap ' + args['function_name'],
            parameters=dict(kwargs=kw, omitted_optional_arguments='native defaults', output_format='native BED' if args['function_name'] == 'map' else 'native index'),
            source_identity=dict(files=[{k:p[k] for k in ('sha256', 'size_bytes')} for p in sources]),
            software_versions=dict(chromap=version, support_representation=receipt['support_representation']),
            measurement='native_chromap_bed_mappings' if args['function_name'] == 'map' else 'native_reference_index', limitations=LIMITATIONS)
        cancellation_checkpoint(); shutil.copyfile(stage / filename, root / filename)
        write(root / 'native-receipt.json', receipt); write(root / 'metadata.json', dict(summary=summary, receipt=receipt))
    value = dict(contract=CONTRACT, profile=PROFILE, arguments=args, execution_identity=execution_identity,
        result=summary, files=file_closure(root / name for name in publication_files(dict(arguments=args))),
        sources=sources, native_completion='succeeded')
    cancellation_checkpoint(); path = root / 'manifest.json'; sha = write(path, value)
    if current() is not None: current().register_execution(KIND, root, execution_identity)
    return verify_publication(path, expected_sha256=sha)


def invoke(reference_path, output_dir, function_name, index_path=None, read1_path=None, read2_path=None,
           barcode_path=None, whitelist_path=None, kwargs=None, reference_sha256=None, index_sha256=None, whitelist_sha256=None):
    return execute({k: v for k, v in locals().items() if v is not None})
