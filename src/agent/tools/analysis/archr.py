"""Focused ArchR Arrow/project snapshots under existing accepted authority.

Only Agent-produced, accepted snapshots can be reopened. Uploaded RDS/ZIP objects
never acquire executable-object authority from a filename or a manifest alone.
"""
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import pysam

from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools._native_process import run, r_environment
from agent.tools.data.authority_context import current
from agent.tools._file_identity import file_closure
from agent.tools.data import fragment_io as fragment_io
from agent.tools.data.fragment_io import parse_record
from . import peak_h5ad, sequence_reference
from .native_matrix import arguments, native_kwargs, NativeInvocationError
from .native_publication import NativePublication
from .sequence_prediction import library, read, write

package = 'ArchR'
TOOL = 'run_archr'
KIND = 'native_archr_project'
CONTRACT = 'native-archr-project.v1'
POLICY = 'native-archr-project-v1'
PROFILE = digest(dict(contract=CONTRACT, version='1.0.3', representation='accepted ArrowFiles and saved ArchRProject closure.v1'))
VERIFIER = dict(id='agent.' + KIND, compatibility_version='1')
guidance = 'ArchR native fragments to ArrowFiles, ArchRProject, gene-score matrix retrieval and available-matrix inspection; true managed projects, not H5AD conversion.'
FUNCTIONS = {
    'createArrowFiles': ('minTSS', 'minFrags', 'maxFrags', 'minFragSize', 'maxFragSize', 'nucLength', 'promoterRegion', 'excludeChr', 'nChunk', 'addTileMat', 'addGeneScoreMat', 'threads', 'subThreading', 'verbose'),
    'ArchRProject': ('showLogo', 'threads'),
    'getMatrixFromProject': ('useMatrix', 'useSeqnames', 'excludeChr', 'binarize', 'threads', 'verbose'),
    'getAvailableMatrices': (),
}
NATIVE_ARGUMENTS = tuple((k, {
    'useMatrix': 'Existing native matrix name; omitted native GeneScoreMatrix. Retrieval never computes a missing matrix.',
    'addGeneScoreMat': 'createArrowFiles native default true; no duplicate addGeneScoreMatrix call.',
    'addTileMat': 'createArrowFiles native default true.',
    'minTSS': 'Native TSS enrichment filtering threshold.',
    'minFrags': 'Native minimum fragment filtering threshold.',
    'threads': 'Native thread count; execution cap 16. Omitted upstream default retained.',
}.get(k, 'Native ArchR ' + k + '; applicable only to its registered functions.')) for k in dict.fromkeys(k for keys in FUNCTIONS.values() for k in keys))
FACT_FIELDS = ('function_name', 'object_type', 'n_cells', 'available_matrices', 'assembly', 'parameters', 'source_identity', 'software_versions', 'measurement', 'limitations')
RESULT_FIELDS = dict(status=(str,), contract_version=(str,), manifest_path=(str,), manifest_sha256=(str,),
    object_path=(str,), object_sha256=(str,), table_path=(str,), table_sha256=(str,), function_name=(str,),
    object_type=(str,), n_cells=(int,), available_matrices=(list,), assembly=(str,), parameters=(dict,),
    source_identity=(dict,), software_versions=(dict,), measurement=(str,), limitations=(str,))
DELIVERY_FILES = {
    'native-object.zip': ('Native Arrow/project snapshot with exact companion files; reopen saved projects with ArchR loadArchRProject', 'application/zip'),
    'result.tsv': ('Native ordered cell identifiers or available matrix names', 'text/tab-separated-values'),
    'matrix.mtx': ('Retrieved native feature-by-cell sparse matrix', 'application/octet-stream'),
    'features.tsv': ('Native ordered matrix row metadata', 'text/tab-separated-values'),
    'native-result.rds': ('Retrieved native SummarizedExperiment; download only', 'application/octet-stream'),
    'metadata.json': ('Native function, sample, reference and snapshot provenance', 'application/json'),
    'native-receipt.json': ('Native completion, object dimensions and runtime', 'application/json'),
}
LIMITATIONS = ('Focused native Arrow/project support, not arbitrary uploaded R objects or every ArchR API. '
    'Gene scores estimate accessibility-related gene activity and are not measured RNA expression. '
    'Native Arrow filtering and matrix defaults retain their upstream meanings. Matrix retrieval does not recompute scores. '
    'No peak-to-gene links, expression integration or automatic figure is claimed. '
    'Saved project snapshots contain the native R object and companion Arrows; only accepted internal snapshots may be reopened by Agent.')


def publication_files(value):
    return ('native-object.zip', 'result.tsv', 'metadata.json', 'native-receipt.json') + (
        ('matrix.mtx', 'features.tsv', 'native-result.rds') if value['arguments']['function_name'] == 'getMatrixFromProject' else ())


def native_arguments(args):
    kw = native_kwargs(FUNCTIONS, args)
    if type(kw.get('threads')) in (int, float) and kw['threads'] > 16:
        raise ValueError('ArchR threads exceeds execution bound 16')
    if type(kw.get('nChunk')) in (int, float) and kw['nChunk'] > 100:
        raise ValueError('ArchR nChunk exceeds execution bound 100')
    return kw


def parameter_prerequisites(args):
    if args.get('function_name') == 'createArrowFiles':
        return tuple(k for k in ('sample_name', 'assembly', 'coordinate_system', 'fragment_endpoints') if args.get(k) is None) + (
            ('second_sample_name',) if args.get('second_path') and not args.get('second_sample_name') else ())
    return ()


def fragment_input(path, target, genome):
    """Exact decoded records, bounded parsing and private BGZF/tabix packaging."""
    with pysam.FastaFile(genome['genome']['fasta']['path']) as ref:
        bounds = {n: (i, size) for i, (n, size) in enumerate(zip(ref.references, ref.lengths))}
    n = 0; decoded = 0
    with pysam.BGZFile(str(target), 'wb') as out:
        for line in fragment_io.lines(path, fragment_io.encoding(path)):
            decoded += len(line)
            if len(line) > 4096 or decoded > 16 * 1024**3:
                raise ValueError('ArchR fragments exceed bounded decoded storage')
            if line.startswith(b'#'):
                out.write(line); continue
            if not line.endswith(b'\n') or b'\r' in line or b'\0' in line:
                raise ValueError('Malformed fragment record')
            _, _, _, barcode, _, _, width = parse_record(line, bounds)
            if width != 5 or '#' in barcode:
                raise ValueError('ArchR entrance requires five-column fragments and barcodes without its sample separator #')
            n += 1
            if n > 200_000_000:
                raise ValueError('ArchR fragments exceed 200 million records')
            out.write(line)
    if not n:
        raise ValueError('Empty fragment input')
    # No sorting, shifting, duplicate collapse or cell selection here.
    pysam.tabix_index(str(target), preset='bed')
    return n


def accepted_snapshot(args, stage):
    context = current(); path = args['input_path']; sha = args.get('input_sha256')
    if context is None or (str(path), sha) not in context.accepted:
        raise AuthorityError('ArchR reuse requires an exact accepted internal publication; uploaded objects are not deserialized')
    authority = context.accepted[str(path), sha]['record']
    if authority['artifact_contract'] != CONTRACT or authority['science_profile'] != PROFILE:
        raise AuthorityError('Selected publication is not the supported ArchR snapshot')
    verify_publication(path, expected_sha256=sha)
    value, _ = load(path, sha)
    needed = 'ArrowFiles' if args['function_name'] == 'ArchRProject' else 'ArchRProject'
    if value['result']['object_type'] != needed:
        raise ValueError('Requested native function requires ' + needed)
    archive = Path(path).parent / 'native-object.zip'
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or len(names) > 1024 or sum(x.file_size for x in bundle.infolist()) > 16 * 1024**3:
            raise AuthorityError('Invalid or oversized native snapshot closure')
        for name in names:
            p = Path(name)
            if p.is_absolute() or '..' in p.parts or len(p.parts) > 4:
                raise AuthorityError('Invalid native snapshot member')
            target = stage / 'snapshot' / p
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as inp, target.open('xb') as out:
                shutil.copyfileobj(inp, out)
    return value, file_closure([path, archive])


def execute(values, execution_identity=None):
    args = arguments(values); kw = native_arguments(args)
    missing = parameter_prerequisites(args)
    if missing:
        raise ValueError('Missing actual ArchR input declarations: ' + ', '.join(missing))
    root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Native output must be empty; no retry authority')
    executable = Path(os.environ.get('AGENT_ARCHR_RSCRIPT') or library().root.parent / 'runtimes/archr-1/env/bin/Rscript')
    script = Path(__file__).parent / 'r/archr_v1.R'
    package_root = executable.parent.parent / 'lib/R/library/ArchR'
    implementation = file_closure([executable, script, package_root / 'DESCRIPTION', *sorted((package_root / 'data').glob('Rdata.*'))])
    sources = file_closure(args[k] for k in ('input_path', 'second_path') if args.get(k))
    for prefix in ('input', 'second'):
        if args.get(prefix + '_sha256') and next(p['sha256'] for p in sources if p['path'] == args[prefix + '_path']) != args[prefix + '_sha256']:
            raise AuthorityError('Selected ArchR source changed')
    with tempfile.TemporaryDirectory(prefix='.archr-', dir=root) as temp:
        stage = Path(temp)
        if args['function_name'] == 'createArrowFiles':
            if args['assembly'] not in ('hg38', 'GRCh38') or args['coordinate_system'] != 'zero_based_half_open' or args['fragment_endpoints'] != 'tn5_adjusted':
                raise ValueError('Qualified ArchR entrance requires declared hg38/GRCh38 zero-based half-open, already Tn5-shifted fragments')
            genome, entry = sequence_reference.resolve_genome('GRCh38')
            resources = entry['files']; assembly = args['assembly']
            samples = [args['sample_name']] + ([args['second_sample_name']] if args.get('second_path') else [])
            if len(set(samples)) != len(samples) or any(not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', s) for s in samples):
                raise ValueError('ArchR sample names must be distinct safe identifiers, without paths or #')
            paths = [args['input_path']] + ([args['second_path']] if args.get('second_path') else [])
            records = [fragment_input(p, stage / f'input-{i}.tsv.gz', genome) for i, p in enumerate(paths)]
            parent = None
        else:
            if any(args.get(k) is not None for k in ('second_path', 'sample_name', 'second_sample_name', 'assembly', 'coordinate_system', 'fragment_endpoints')):
                raise ValueError('Project continuation preserves accepted sample/reference declarations; no source reinterpretation')
            parent, resources = accepted_snapshot(args, stage)
            samples = parent['result']['parameters']['sample_names']; records = []
            assembly = parent['result']['assembly']
        sha = write(stage / 'request.json', dict(stage=str(stage), function_name=args['function_name'], kwargs=kw,
            sample_names=samples, n_inputs=len(records), parent_type=parent['result']['object_type'] if parent else None))
        code = run([str(executable), '--vanilla', str(script), str(stage / 'request.json')], stage / 'native.log', environment=r_environment())
        shutil.copyfile(stage / 'native.log', root / ('native-failure.log' if code else 'native.log'))
        if code:
            message = read(stage / 'native-error.json').get('message', '')[:2000] if (stage / 'native-error.json').exists() else ''
            if not message or peak_h5ad._PATH.search(message) or any(ord(c) < 32 for c in message):
                message = 'Native ArchR failed; exact diagnostic retained privately'
            raise NativeInvocationError(message)
        receipt = read(stage / 'native-receipt.json')
        if receipt['request_sha256'] != sha or receipt['completion'] != 'succeeded' or receipt['function_name'] != args['function_name'] or receipt['versions']['ArchR'] != '1.0.3':
            raise AuthorityError('ArchR native completion differs')
        receipt['implementation'] = implementation
        if args['function_name'] in ('createArrowFiles', 'ArchRProject'):
            snapshot = stage / 'published'
            with zipfile.ZipFile(root / 'native-object.zip', 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
                for p in sorted(snapshot.rglob('*')):
                    if p.is_symlink():
                        raise AuthorityError('Native snapshot cannot contain external links')
                    if p.is_file():
                        bundle.write(p, str(p.relative_to(snapshot)))
        else:
            shutil.copyfile(Path(args['input_path']).parent / 'native-object.zip', root / 'native-object.zip')
        summary = dict(status='success', contract_version=CONTRACT, function_name='ArchR::' + args['function_name'],
            object_type=receipt['object_type'], n_cells=receipt['n_cells'], available_matrices=receipt['available_matrices'], assembly=assembly,
            parameters=dict(kwargs=kw, sample_names=samples, fragment_records=records, omitted_optional_arguments='native defaults',
                native_matrix_dimensions=receipt.get('matrix_dimensions'), reference_annotation='ArchR 1.0.3 bundled Hg38 geneAnnotation/genomeAnnotation, exact installed data bytes pinned'),
            source_identity=dict(inputs=[{k: p[k] for k in ('sha256', 'size_bytes')} for p in sources], parent_publication_sha256=args.get('input_sha256') if parent else None),
            software_versions=receipt['versions'], measurement='Native accessibility gene scores are not measured expression; selected existing matrix meaning is recorded by useMatrix', limitations=LIMITATIONS)
        for name in publication_files(dict(arguments=args)):
            if name not in ('native-object.zip', 'metadata.json', 'native-receipt.json'):
                shutil.copyfile(stage / name, root / name)
        write(root / 'metadata.json', dict(summary=summary, sources=sources, receipt=receipt))
        write(root / 'native-receipt.json', receipt)
    expected = sorted({p['path']: p for p in sources + list(resources) + implementation}.values(), key=lambda p: p['path'])
    if file_closure(p['path'] for p in expected) != expected:
        raise AuthorityError('Source, reference or implementation changed')
    value = dict(contract=CONTRACT, profile=PROFILE, arguments=args, execution_identity=execution_identity,
        result=summary, files=file_closure(root / n for n in publication_files(dict(arguments=args))), sources=expected, native_completion='succeeded')
    path = root / 'manifest.json'; sha = write(path, value)
    if current() is not None:
        current().register_execution(KIND, root, execution_identity)
    return verify_publication(path, expected_sha256=sha)


def invoke(input_path, output_dir, function_name, input_sha256, kwargs=None, sample_name=None,
           second_path=None, second_sha256=None, second_sample_name=None, assembly=None, coordinate_system=None, fragment_endpoints=None):
    return execute({k: v for k, v in locals().items() if v is not None})


publication = NativePublication(tool=TOOL, kind=KIND, contract=CONTRACT, profile=PROFILE, verifier=VERIFIER,
    files=publication_files, arguments=arguments, outputs={'object': 'native-object.zip', 'table': 'result.tsv'})
load = publication.load
result = publication.result
verify_publication = publication.verify_publication
verify_result = publication.verify_result
recover = publication.recover
