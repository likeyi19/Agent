"""Native genomic tables: managed files/objects, shared publication and delivery."""
import csv
import math
import os
from pathlib import Path
import shutil
import tempfile

from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools._native_process import run, r_environment
from agent.tools.data.authority_context import current
from agent.tools.data.fastq_verification_authority import file_closure
from .native_matrix import arguments, NativeInvocationError
from .native_publication import NativePublication
from .sequence_prediction import library, read, write

FACT_FIELDS = ('function_name', 'n_rows', 'result_format', 'measurement', 'parameters',
    'source_identity', 'software_versions', 'limitations')
RESULT_FIELDS = dict(status=(str,), contract_version=(str,), manifest_path=(str,), manifest_sha256=(str,),
    table_path=(str,), table_sha256=(str,), function_name=(str,), n_rows=(int,), result_format=(str,),
    measurement=(str,), parameters=(dict,), source_identity=(dict,), software_versions=(dict,), limitations=(str,))
DELIVERY_FILES = {'result.tsv': ('Native genomic result; format and measurement in metadata', 'text/tab-separated-values'),
    'metadata.json': ('Native parameters and source/result semantics', 'application/json'),
    'native-receipt.json': ('Native completion and runtime identity', 'application/json')}
LIMITATIONS = ('Independent native function, not a complete workflow or biological validation. '
    'Genomic coordinate labels do not qualify a reference assembly. Native scores and CCAN memberships '
    'are not fragment counts, causal regulation or independent experimental validation. '
    'Omitted native arguments retain installed defaults; source files remain unchanged.')


def bedgraph(path, *, empty=False):
    """Protect against the native C parser's silent coordinate/number coercion.

    MACS stores interval ends and would silently fill internal gaps with the
    following score. Require its actual continuous-track input representation.
    A nonzero first start retains native zero-baseline behavior.
    """
    ends = {}; n = 0
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith(('#', 'track', 'browser')):
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError('MACS3 requires four-column bedGraph data')
            chrom, left, right, score = fields
            start, end, value = int(left), int(right), float(score)
            if (not 0 <= start < end <= 2147483647
                    or math.isfinite(value) and abs(value) > 3.4028234e38):
                raise ValueError('bedGraph coordinates/scores exceed native representation')
            if chrom in ends and start != ends[chrom]:
                raise ValueError('MACS3 requires sorted continuous nonoverlapping intervals within each chromosome')
            ends[chrom] = end; n += 1
            if n > 5_000_000:
                raise ValueError('bedGraph exceeds the 5000000-record execution bound')
    if not n and not empty:
        raise ValueError('Empty bedGraph input')
    return n


class TablePackage:
    FACT_FIELDS, RESULT_FIELDS, DELIVERY_FILES = FACT_FIELDS, RESULT_FIELDS, DELIVERY_FILES
    arguments = staticmethod(arguments)

    def __init__(self, package, version, functions, members, guidance):
        self.package, self.version = package, version
        self.invoke = self.invoke_macs3 if package == 'MACS3' else self.invoke_cicero
        self.FUNCTIONS, self.NATIVE_ARGUMENTS, self.guidance = functions, members, guidance
        self.TOOL = 'run_' + package.lower()
        self.KIND = 'native_' + package.lower() + '_table'
        self.CONTRACT = self.KIND.replace('_', '-') + '.v1'
        self.POLICY = self.CONTRACT.replace('.v1', '-v1')
        self.PROFILE = digest(dict(contract=self.CONTRACT, version=version,
            adaptation='exact bound tables; fixed native dispatch; managed output; no scientific replay.v1'))
        self.VERIFIER = dict(id='agent.' + self.KIND, compatibility_version='1')
        self.publication = NativePublication(tool=self.TOOL, kind=self.KIND, contract=self.CONTRACT,
            profile=self.PROFILE, verifier=self.VERIFIER, files=tuple(DELIVERY_FILES),
            arguments=arguments, outputs={'table': 'result.tsv'})
        for name in ('load', 'result', 'verify_publication', 'verify_result', 'recover'):
            setattr(self, name, getattr(self.publication, name))

    def native_arguments(self, args):
        kw = args.get('kwargs', {}); name = args.get('function_name')
        if name not in self.FUNCTIONS or type(kw) is not dict or set(kw) - set(self.FUNCTIONS[name]):
            raise ValueError('Unreviewed native function or argument')
        for key, value in kw.items():
            vector = value if key == 'extra-param' and type(value) is list else [value]
            if len(vector) > 16 or any(not (v is None or type(v) in (bool, int)
                    or type(v) is float and math.isfinite(v)
                    or type(v) is str and 0 < len(v) <= 128 and not any(ord(c) < 32 for c in v)) for v in vector):
                raise ValueError('Native arguments must be bounded ordinary scalars; no files or executable objects')
        if args.get('seed') is not None and (self.package != 'cicero' or name != 'generate_ccans'
                or type(args['seed']) is not int or not 0 <= args['seed'] <= 2147483647):
            raise ValueError('R seed applies to generate_ccans and must fit a native integer')
        if name not in ('bdgcmp', 'cmbreps') and (args.get('control_path') or args.get('control_semantics') or args.get('control_sha256')):
            raise ValueError('The second track is used only by bdgcmp/cmbreps')
        if type(kw.get('threads')) in (int,float) and kw['threads'] > 16:
            raise ValueError('Native threads exceeds execution bound 16')
        if type(kw.get('buffer-size')) in (int,float) and kw['buffer-size'] > 10_000_000:
            raise ValueError('Native parser buffer exceeds execution bound')
        return kw

    def parameter_prerequisites(self, args):
        if self.package != 'MACS3':
            return ()
        return ('control_path',) if args.get('function_name') in ('bdgcmp', 'cmbreps') and not args.get('control_path') else ()

    def native(self, args, stage):
        kw = self.native_arguments(args); name = args['function_name']
        if self.package == 'MACS3':
            executable = Path(os.environ.get('AGENT_MACS3_EXECUTABLE') or library().root.parent / 'runtimes/macs3/env/bin/macs3').resolve()
            runtime_path = executable.parent.parent.parent / 'runtime.json'
            runtime = read(runtime_path)
            if runtime['version'] != self.version or file_closure(p['path'] for p in runtime['files']) != runtime['files']:
                raise AuthorityError('Pinned MACS3 runtime changed')
            implementation = file_closure([runtime_path, executable])
            command = [str(executable), name]
            if name == 'bdgcmp':
                command += ['--tfile', str(stage / 'input.tsv'), '--cfile', str(stage / 'control.tsv')]
            elif name == 'cmbreps':
                command += ['-i', str(stage / 'input.tsv'), str(stage / 'control.tsv')]
            else:
                command += ['--ifile', str(stage / 'input.tsv')]
            for key, value in kw.items():
                if key in ('no-trackline', 'both-direction'):
                    if type(value) is not bool:
                        raise ValueError('CLI switch requires a boolean')
                    if value:
                        command.append('--' + key)
                else:
                    values = value if type(value) is list else [value]
                    # One argv member per native value: no shell, no injected flags.
                    if len(values) == 1:
                        command.append('--' + key + '=' + str(values[0]))
                    else:
                        if any(isinstance(v, str) and v.startswith('-') and not v.lstrip('-').replace('.', '', 1).isdigit() for v in values):
                            raise ValueError('A native vector cannot introduce command-line options')
                        command.append('--' + key); command.extend(str(v) for v in values)
            command += ['--outdir', str(stage), '--ofile', 'result.tsv']
            request_sha = write(stage / 'request.json', dict(function_name=name, kwargs=kw))
        else:
            executable = Path(os.environ.get('AGENT_CICERO_RSCRIPT') or library().root.parent / 'runtimes/cicero-1/env/bin/Rscript')
            script = Path(__file__).parent / 'r/cicero_functions_v1.R'
            implementation = file_closure([executable, script])
            request_sha = write(stage / 'request.json', dict(stage=str(stage), function_name=name,
                version=self.version, kwargs=kw, seed=args.get('seed')))
            command = [str(executable), '--vanilla', str(script), str(stage / 'request.json')]
        code = run(command, stage / 'native.log', environment=r_environment())
        shutil.copyfile(stage / 'native.log', stage.parent / ('native-failure.log' if code else 'native.log'))
        if code:
            from . import peak_h5ad
            message = read(stage / 'native-error.json').get('message', '')[:2000] if (stage / 'native-error.json').exists() else ''
            if not message and self.package == 'MACS3':
                lines = (stage / 'native.log').read_text(errors='replace').splitlines()
                message = next((line for line in reversed(lines) if 'error:' in line or line.startswith(('ValueError:', 'AssertionError:', 'TypeError:'))), '')[:2000]
            if not message or peak_h5ad._PATH.search(message) or any(ord(c) < 32 for c in message):
                message = 'Native function failed; exact diagnostic retained privately, no substituted parameters'
            raise NativeInvocationError(message)
        if file_closure(p['path'] for p in implementation) != implementation:
            raise AuthorityError('Native implementation changed during execution')
        if self.package == 'MACS3':
            if file_closure(p['path'] for p in runtime['files']) != runtime['files']:
                raise AuthorityError('MACS3 runtime changed during execution')
            receipt = dict(completion='succeeded', package=self.package, function_name=name,
                versions={self.package: self.version}, request_sha256=request_sha,
                native_command=command, runtime_sha256=implementation[-1]['sha256'])
        else:
            receipt = read(stage / 'native-receipt.json')
            if (receipt['completion'] != 'succeeded' or receipt['request_sha256'] != request_sha
                    or receipt['function_name'] != name or receipt['versions']['cicero'] != self.version):
                raise AuthorityError('Native completion differs from requested invocation')
        return dict(receipt, implementation=implementation)

    def execute(self, values, execution_identity=None):
        args = arguments(values); kw = self.native_arguments(args)
        if self.parameter_prerequisites(args):
            raise ValueError('Missing native input declarations: ' + ', '.join(self.parameter_prerequisites(args)))
        root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError('Native output directory must be empty; no retry authority')
        sources = file_closure(args[k] for k in ('input_path', 'control_path') if args.get(k))
        for k in ('input', 'control'):
            if args.get(k + '_sha256') and next(p['sha256'] for p in sources if p['path'] == args[k + '_path']) != args[k + '_sha256']:
                raise AuthorityError('Bound input checksum differs')
        with tempfile.TemporaryDirectory(prefix='.native-table-', dir=root) as temp:
            stage = Path(temp)
            for key, target in (('input_path', 'input.tsv'), ('control_path', 'control.tsv')):
                if args.get(key):
                    if Path(args[key]).stat().st_size > 256 * 1024**2:
                        raise ValueError('Native table exceeds 256 MiB execution bound')
                    shutil.copyfile(args[key], stage / target)
                    if self.package == 'MACS3' and args['function_name'] not in ('pileup', 'filterdup'):
                        bedgraph(stage / target)
            receipt = self.native(args, stage)
            result_format, measurement = self.output_semantics(args)
            table = stage / 'result.tsv'
            if table.stat().st_size > 256 * 1024**2:
                raise AuthorityError('Native output exceeds table size bound')
            if result_format == 'bedGraph':
                n_rows = bedgraph(table, empty=True)
            else:
                with table.open() as stream:
                    rows = [r for r in csv.reader(stream, delimiter='\t') if r and not r[0].startswith(('track', '#'))]
                if self.package == 'cicero':
                    header = ['Peak', 'CCAN'] if args['function_name'] == 'generate_ccans' else ['CCAN1', 'CCAN2']
                    if not rows or rows[0] != header or any(len(r) != 2 for r in rows[1:]):
                        raise AuthorityError('Invalid native CCAN table')
                    n_rows = len(rows) - 1
                elif args['function_name'] == 'filterdup':
                    widths = {len(r) for r in rows}
                    if not widths <= {3, 6} or len(widths) > 1:
                        raise AuthorityError('Unrepresented native read output')
                    result_format = 'MACS3 BEDPE' if widths == {3} else 'BED6' if widths else 'native_read_intervals (empty)'
                    n_rows = len(rows)
                else:
                    expected = 10 if result_format == 'narrowPeak' else 15
                    if any(len(r) != expected or not 0 <= int(r[1]) < int(r[2]) for r in rows):
                        raise AuthorityError('Invalid native peak intervals')
                    n_rows = len(rows)
            summary = dict(status='success', contract_version=self.CONTRACT,
                function_name=self.package + ('::' if self.package == 'cicero' else ' ') + args['function_name'],
                n_rows=n_rows, result_format=result_format, measurement=measurement,
                parameters=dict(kwargs=kw, seed=args.get('seed'), omitted_optional_arguments='native defaults'),
                source_identity=dict(files={k: {f: p[f] for f in ('sha256', 'size_bytes')}
                    for k in ('input_path', 'control_path') if args.get(k)
                    for p in sources if p['path'] == args[k]},
                    signal_semantics=args.get('signal_semantics'), control_semantics=args.get('control_semantics'), assembly=args.get('assembly')),
                software_versions=receipt['versions'], limitations=LIMITATIONS)
            shutil.copyfile(table, root / 'result.tsv')
            write(root / 'metadata.json', dict(summary=summary, sources=sources, receipt=receipt))
            write(root / 'native-receipt.json', receipt)
            expected_sources = sorted(sources + receipt['implementation'], key=lambda p: p['path'])
            if file_closure(p['path'] for p in expected_sources) != expected_sources:
                raise AuthorityError('Source or runtime changed during invocation')
        value = dict(contract=self.CONTRACT, profile=self.PROFILE, arguments=args, execution_identity=execution_identity,
            result=summary, files=file_closure(root / name for name in DELIVERY_FILES), sources=expected_sources, native_completion='succeeded')
        path = root / 'manifest.json'; sha = write(path, value)
        if current() is not None:
            current().register_execution(self.KIND, root, execution_identity)
        return self.verify_publication(path, expected_sha256=sha)

    def output_semantics(self, args):
        name = args['function_name']; kw = args.get('kwargs', {})
        if self.package == 'cicero':
            return 'TSV', 'native_CCAN_membership' if name == 'generate_ccans' else 'native_genomic_CCAN_overlaps'
        if name == 'pileup':
            return 'bedGraph', 'native_read_fragment_pileup'
        if name == 'filterdup':
            return 'native_read_intervals', 'native_duplicate_filtered_reads_or_pairs'
        if name == 'cmbreps':
            return 'bedGraph', 'MACS3_' + str(kw.get('method', 'fisher')) + '_combined_tracks'
        if name == 'bdgcmp':
            return 'bedGraph', 'MACS3_' + str(kw.get('method', 'ppois')) + '_comparison'
        if name == 'bdgopt':
            return 'bedGraph', 'MACS3_' + str(kw.get('method', 'p2q')) + '_of_' + args.get('signal_semantics', 'unspecified_scores')
        return ('narrowPeak' if name == 'bdgpeakcall' else 'gappedPeak'), 'native_peaks_from_' + args.get('signal_semantics', 'unspecified_scores')

    def invoke_macs3(self, input_path, output_dir, function_name, kwargs=None, input_sha256=None,
               control_path=None, control_sha256=None, signal_semantics=None, control_semantics=None,
               assembly=None):
        return self.execute({k: v for k, v in locals().items() if k != 'self' and v is not None})

    def invoke_cicero(self, input_path, output_dir, function_name, kwargs=None, input_sha256=None, seed=None):
        return self.execute({k: v for k, v in locals().items() if k != 'self' and v is not None})


MACS3 = TablePackage('MACS3', '3.1.0', {
    'cmbreps': ('method', 'verbose'),
    'pileup': ('format', 'max-count', 'both-direction', 'extsize', 'buffer-size', 'threads', 'verbose'),
    'filterdup': ('format', 'gsize', 'tsize', 'pvalue', 'keep-dup', 'buffer-size', 'threads', 'verbose'),
    'bdgopt': ('method', 'extra-param', 'verbose'),
    'bdgcmp': ('method', 'scaling-factor', 'pseudocount', 'verbose'),
    'bdgpeakcall': ('cutoff', 'min-length', 'max-gap', 'no-trackline', 'verbose'),
    'bdgbroadcall': ('cutoff-peak', 'cutoff-link', 'min-length', 'lvl1-max-gap', 'lvl2-max-gap', 'no-trackline', 'verbose'),
}, (
    ('format', 'Native alignment format; BEDPE/FRAG are MACS formats, not generic matrices.'),
    ('max-count', 'pileup native FRAG multiplicity limit.'),
    ('both-direction', 'pileup native bidirectional extension switch.'),
    ('extsize', 'pileup native read extension in bases.'),
    ('buffer-size', 'Native parser allocation increment.'),
    ('threads', 'Native CPU budget; execution cap 16.'),
    ('gsize', 'filterdup native effective genome size or installed alias.'),
    ('tsize', 'filterdup explicit native tag size.'),
    ('pvalue', 'filterdup native binomial cutoff.'),
    ('keep-dup', 'filterdup native duplicate handling: auto, all or count.'),
    ('method', 'cmbreps fisher/max/mean; bdgopt multiply/add/max/min/p2q; bdgcmp one ppois/qpois/subtract/logFE/FE/logLR/slogLR/max method. Native defaults omitted.'),
    ('extra-param', 'bdgopt native numeric extra parameter (scalar or numeric list).'),
    ('scaling-factor', 'bdgcmp scales treatment and control, native default 1.'),
    ('pseudocount', 'bdgcmp native pseudocount after depth normalization, default 0.'),
    ('cutoff', 'bdgpeakcall score threshold in supplied score units, default 5.'),
    ('min-length', 'Minimum peak length in bases, default 200.'),
    ('max-gap', 'bdgpeakcall maximum gap in bases, default 30.'),
    ('cutoff-peak', 'bdgbroadcall high-score threshold, default 2.'),
    ('cutoff-link', 'bdgbroadcall linking threshold, default 1.'),
    ('lvl1-max-gap', 'bdgbroadcall strong-region gap in bases, default 30.'),
    ('lvl2-max-gap', 'bdgbroadcall linking gap in bases, default 800.'),
    ('no-trackline', 'Native boolean output trackline switch.'),
    ('verbose', 'Native diagnostic verbosity, default 2.'),
), 'MACS3 bedGraph operations and native read pileup/filterdup. cmbreps combines exactly two authorized tracks (dataset/control); fisher expects -log10 p-values. pileup consumes real alignment/read formats; filterdup returns native BED6 or MACS BEDPE. bedGraph inputs require continuous within-chromosome intervals to prevent parser coordinate changes. No callpeak duplication or replicate-level DA claim.')


CICERO = TablePackage('cicero', '1.28.3', {
    'generate_ccans': ('coaccess_cutoff_override', 'tolerance_digits'),
    'find_overlapping_ccans': ('min_overlap',),
}, (
    ('coaccess_cutoff_override', 'generate_ccans optional native coaccessibility cutoff; omitted performs native cutoff search.'),
    ('tolerance_digits', 'generate_ccans native cutoff-search precision, default 2.'),
    ('min_overlap', 'find_overlapping_ccans minimum genomic overlap in bases, default 1.'),
), 'Cicero generate_ccans consumes a TSV with Peak1, Peak2, coaccess and returns Peak/CCAN membership, excluding native communities of size <=2. No CDS, embedding or estimation workflow required. find_overlapping_ccans consumes Peak/CCAN TSV with native chr_start_end peak identities and returns reciprocal CCAN1/CCAN2 overlaps. Native genome ranges are 1-based closed; no coordinate conversion or assembly inference. Optional seed controls native R RNG for generate_ccans.')

PACKAGES = (MACS3, CICERO)
