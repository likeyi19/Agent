"""File handoffs for native SnapATAC2 preprocessing, without scientific replay.

The package owns parsing, filtering, QC, fragment construction and counting.
These functions join the existing run_snapatac2 catalog and publication owner.
"""
from pathlib import Path
import shutil
import tempfile

import h5py
from anndata.io import read_elem

from agent.schemas.verification_authority import AuthorityError
from agent.tools._cancellation import cancellation_checkpoint
from agent.tools.data.authority_context import current
from agent.tools._file_identity import file_closure

FUNCTIONS = {
    'pp.make_fragment_file': ('is_paired', 'barcode_tag', 'barcode_regex', 'umi_tag', 'umi_regex',
        'shift_left', 'shift_right', 'min_mapq', 'chunk_size', 'chrM', 'source', 'compression', 'compression_level'),
    'pp.import_fragments': ('is_paired', 'min_num_fragments', 'sorted_by_barcode', 'chrM', 'chunk_size', 'n_jobs', 'chrom_sizes', 'whitelist'),
    'metrics.tsse': ('exclude_chroms', 'inplace', 'n_jobs'),
    'metrics.frag_size_distr': ('max_recorded_size', 'add_key', 'inplace', 'n_jobs'),
    'pp.filter_cells': ('min_counts', 'min_tsse', 'max_counts', 'max_tsse', 'inplace', 'n_jobs'),
    'pp.call_cells': ('use_rep', 'inplace', 'n_jobs'),
    'pp.add_tile_matrix': ('bin_size', 'chunk_size', 'exclude_chroms', 'min_frag_size', 'max_frag_size',
        'counting_strategy', 'value_type', 'summary_type', 'inplace', 'n_jobs'),
    'pp.make_peak_matrix': ('chunk_size', 'use_x', 'min_frag_size', 'max_frag_size', 'counting_strategy',
        'value_type', 'summary_type', 'use_rep', 'inplace'),
    'pp.make_gene_matrix': ('inplace', 'backend', 'chunk_size', 'use_x', 'id_type', 'upstream', 'downstream',
        'include_gene_body', 'transcript_name_key', 'transcript_id_key', 'gene_name_key', 'gene_id_key',
        'min_frag_size', 'max_frag_size', 'counting_strategy'),
}
PORTS = {
    'chrom_sizes': ('chrom_sizes_path', 'chrom_sizes_sha256', 'FAI or two-column chromosome sizes used by native fragment import.'),
    'gene_anno': ('gene_anno_path', 'gene_anno_sha256', 'Compatible GTF/GFF annotation for the selected native function (epiScanpy requires GTF).'),
    'peaks': ('peaks_path', 'peaks_sha256', 'BED of the exact requested peak/cCRE/region features, in source coordinates.'),
    'whitelist': ('whitelist_path', 'whitelist_sha256', 'Optional barcode whitelist for native import; no implicit cell calling.'),
}
REQUIRED = {'pp.import_fragments': ('chrom_sizes',), 'metrics.tsse': ('gene_anno',), 'pp.make_gene_matrix': ('gene_anno',)}
APPLIES = {'pp.import_fragments': ('chrom_sizes', 'whitelist'), 'metrics.tsse': ('gene_anno',),
           'pp.make_peak_matrix': ('peaks',), 'pp.make_gene_matrix': ('gene_anno',)}
RAW_FILES = {
    'fragments.tsv.gz': ('Native BAM-derived fragments; exact method and support semantics in metadata', 'application/gzip'),
    'fragments.tsv': ('Native BAM-derived fragments; uncompressed native output', 'text/tab-separated-values'),
    'fragments.tsv.zst': ('Native BAM-derived fragments; Zstandard native output', 'application/zstd'),
}
LIMITATIONS = ('Native SnapATAC2 preprocessing, not biological validation. Barcodes are identifiers, not independently established cells. '
    'Import excludes records outside supplied chromosome sizes. Native defaults are min_num_fragments=200 and sorted_by_barcode=True; '
    'task input-order selection is recorded separately and explicit sorted_by_barcode kwargs prevail. '
    'The imported barcode population reflects native import retention, distinct from later filter_cells selection. '
    'Task Otsu removes the depth prefilter and fits omitted filter minima on native n_fragment/log10 and tsse/linear, '
    'not pycisTopic fragments-in-peaks or its TSS calculation; explicit minima prevail. '
    'BAM filtering/deduplication and endpoint shifts follow the recorded native method, not historical Agent BAM policies; duplicate-flagged reads are excluded. '
    'Native import n_fragment excludes mitochondrial records; count matrices do not weight column-five duplicate support. '
    'Default make_peak_matrix counts paired insertions, not fragment-body overlaps. Native filter_cells defaults to min_counts=1000 and min_tsse=5. '
    'Old Agent TSS enrichment and selection thresholds are different methods. Native fragment-size distribution is library-level, not per-cell nucleosome signal. '
    'Supplied species/assembly and barcode/coordinate history are declarations; no inferred reference correspondence or relabelling.')

# Shared by the selected-operation owner and its Registry projection. A native
# signature cannot express this condition because both parameters default to None.
BARCODE_EXTRACTION = dict(argument='kwargs', rule='exactly_one',
    members=('barcode_tag', 'barcode_regex'), effective='nonempty_string')
BARCODE_CHOICES = (
    dict(handle='native-parameter.barcode_tag', label='BAM tag extraction', member='barcode_tag'),
    dict(handle='native-parameter.barcode_regex', label='Read-name regex extraction', member='barcode_regex'),
)


def parameter_metadata(function, metadata):
    """Keep the selected-member editor typed even without runtime discovery."""
    result=dict(metadata)
    for condition in scientific_contract({'function_name':function}).parameters.get('conditions',()):
        for member in condition['members']:
            result[member]=dict(result.get(member,{}),types=['str','NoneType'],
                required=False,default_when_omitted=None,conditional=condition)
    return result


FILTER_DEFAULTS = dict(min_counts=1000, min_tsse=5.0, max_counts=None, max_tsse=None)
FILTER_FIELDS = {'obs.n_fragment': ('min_counts', 'max_counts'), 'obs.tsse': ('min_tsse', 'max_tsse')}
DEFAULT_PEAK_REGION_KEY = 'peaks'
QC_POPULATIONS = ('uncalled_single_sample','study_selected','depth_prefiltered','multiple_samples')
QC_FILTER_MINIMA = ('min_counts','min_tsse')


def active_qc(arguments):
    if arguments.get('qc_thresholds')=='native_defaults':return 'native_defaults'
    decision=arguments.get('qc_selection')
    return decision['method'] if decision is not None else arguments.get('qc_thresholds')


def accepted_qc_population(parameters):
    history=parameters.get('population',{})
    if history.get('stage')=='native_import' and history.get('min_num_fragments',0)>0:
        return 'depth_prefiltered'
    if history.get('applied') and history.get('stage') in ('native_cell_filter','native_ordmag_cell_call'):
        return 'study_selected'
    return parameters.get('qc_population')


def qc_lineage(value, context):
    from agent.schemas import StepOutputRef
    from agent.application.source_observations import identity_parent
    seen=set()
    while isinstance(value,StepOutputRef) and value.step_id not in seen:
        seen.add(value.step_id)
        name, arguments=context.selected.get(value.step_id,(None,{}))
        if name!='run_snapatac2':
            parent=identity_parent(value,context.selected,context.registry)
            if parent is None:break
            value=parent
            continue
        function=arguments.get('function_name')
        kwargs=arguments.get('kwargs',{})
        if function=='pp.import_fragments':
            minimum=kwargs.get('min_num_fragments')
            if (minimum is None and arguments.get('qc_thresholds')=='native_defaults'
                    or isinstance(minimum,(int,float)) and minimum>0):
                return value.step_id,'depth_prefiltered'
            if kwargs.get('whitelist') is not None or arguments.get('whitelist_path') is not None:
                return value.step_id,'restricted_population'
            return value.step_id,arguments.get('qc_population')
        if function in ('pp.filter_cells','pp.call_cells') and kwargs.get('inplace',True):
            return None,'study_selected'
        if function not in ('metrics.tsse','metrics.frag_size_distr','pp.filter_cells','pp.call_cells',
                            'pp.add_tile_matrix','pp.make_peak_matrix','pp.make_gene_matrix'):
            break
        value=arguments.get('input_path')
    if context.observe_existing and (source:=context.existing(value)) is not None:
        return None,accepted_qc_population(source.publication_parameters)
    return None,None


def qc_decision(arguments, context):
    from agent.orchestration.scientific_inputs import ScientificDiagnostic
    request=arguments.get('qc_thresholds')
    if request=='native_defaults':
        return {'method':'native_defaults','reason':'explicit_native_defaults'},qc_lineage(arguments.get('input_path'),context)[0]
    if all(member in arguments.get('kwargs',{}) for member in QC_FILTER_MINIMA):
        return {'method':'native_defaults','reason':'explicit_filter_minima'},qc_lineage(arguments.get('input_path'),context)[0]
    origin,history=qc_lineage(arguments.get('input_path'),context)
    declared=arguments.get('qc_population')
    if history and declared and history!=declared and declared!='uncalled_single_sample':
        raise ScientificDiagnostic('QC_POPULATION_CONFLICT','The declared QC population conflicts with its selected source lineage.')
    population=history or declared
    if population=='uncalled_single_sample':
        return {'method':'otsu','population':population,'reason':'qualified_uncalled_single_sample'},origin
    if population is not None:
        if request=='otsu':
            raise ScientificDiagnostic('OTSU_INAPPLICABLE','The requested Otsu fit requires one uncalled, unprefiltered sample.')
        return {'method':'native_defaults','population':population,'reason':'otsu_inapplicable'},origin
    return {'method':None,'reason':'population_history_unknown'},origin


def coordinate_qc(arguments, context):
    decision,origin=qc_decision(arguments,context)
    values={context.step_id:{'qc_selection':decision}}
    if (origin is not None and decision['method']=='otsu'
            and context.selected[origin][1].get('qc_thresholds')!='native_defaults'):
        values[origin]={'qc_selection':dict(decision,role='retain_fitting_population')}
    elif origin is not None and context.selected[origin][1].get('qc_thresholds')=='otsu':
        from agent.orchestration.scientific_inputs import ScientificDiagnostic
        if decision['method'] is not None:
            raise ScientificDiagnostic('OTSU_SELECTION_CONFLICT','The import requests Otsu but the selected filter does not fit that population; supply import retention explicitly instead.')
    elif origin is not None and context.selected[origin][1].get('qc_selection',{}).get('method')=='otsu':
        values[origin]={'qc_selection':{'method':'native_defaults','reason':'linked_fit_not_selected'}}
    return values


def require_linked_qc_fit(arguments, context):
    if arguments.get('qc_thresholds')!='otsu':return {}
    for _,candidate in context.selected.values():
        if (candidate.get('function_name')=='pp.filter_cells'
                and qc_lineage(candidate.get('input_path'),context)[0]==context.step_id):
            return {}
    from agent.orchestration.scientific_inputs import ScientificDiagnostic
    raise ScientificDiagnostic('OTSU_LINKED_FILTER_REQUIRED',
        'Otsu fits filter minima, not import retention alone. Set min_num_fragments explicitly for an import-only request.')


def peak_region_key(arguments):
    return arguments.get('kwargs', {}).get('use_rep', arguments.get('use_rep'))


def checked_raw_operand(arguments):
    from agent.orchestration.scientific_inputs import ScientificDiagnostic
    function=arguments.get('function_name')
    selected=peak_region_key(arguments)
    if function=='pp.make_peak_matrix':
        if selected is not None and arguments.get('peaks_path') is not None:
            raise ScientificDiagnostic('NATIVE_PEAK_REGIONS_CONFLICT','Use either a BED input or kwargs.use_rep, not both.')
        if selected is not None and not (isinstance(selected,str) and selected or
                isinstance(selected,list) and selected and all(isinstance(region,str) and region for region in selected)):
            raise ScientificDiagnostic('NATIVE_PEAK_REGIONS_INVALID','Select a named region table or supply an ordered nonempty region list.')
    elif function=='pp.call_cells' and selected is not None and not (isinstance(selected,str) and selected or
            isinstance(selected,list) and selected and all(type(value) is int and 0<=value<2**64 for value in selected)):
        raise ScientificDiagnostic('NATIVE_CALL_CELLS_COUNTS_INVALID','Use a numeric observation field or an ordered vector of nonnegative counts.')
    return selected


def use_default_peak_regions(arguments, context):
    if (arguments.get('function_name')!='pp.make_peak_matrix' or arguments.get('peaks_path') is not None
            or peak_region_key(arguments) is not None):
        return arguments
    key='regions.'+DEFAULT_PEAK_REGION_KEY
    if context.properties(arguments.get('input_path'), (key,)).get(key):
        return dict(arguments, kwargs={**arguments.get('kwargs', {}), 'use_rep':DEFAULT_PEAK_REGION_KEY})
    return arguments


def scientific_contract(arguments):
    from agent.orchestration.scientific_inputs import ScientificContract
    from .snapatac_raw_worker import import_storage
    name=arguments.get('function_name'); kwargs=arguments.get('kwargs', {})
    if name=='pp.make_fragment_file':
        return ScientificContract(inputs={'dataset': {'format': 'bam'}},
            parameters={'conditions': (BARCODE_EXTRACTION,)}, outputs={'fragments': {'format': 'fragments', 'barcode_sorted': True}})
    if name=='pp.import_fragments':
        paired=kwargs.get('is_paired', True)
        storage, slot, measurement=import_storage(paired)
        return ScientificContract(inputs={'dataset': {'format': 'fragments'}},
            parameters={'resources': REQUIRED[name], 'optional_resources': ('whitelist',)},
            outputs={'dataset': dict(format='h5ad', x=False, scanpy_auto_rep=False, scanpy_x_rep=False,
                fragment_storage=True, paired_fragments=paired,
                native_values=True, storage_kind=storage, measurement=measurement,
                **{'obsm.'+slot: True, **dict.fromkeys(('obs.n_fragment', 'obs.frac_dup', 'obs.frac_mito'), 'numeric')},
                **{'obs.tsse': False, 'var.selected': False})}, effects={'cell_axis': 'new'},
            selection={'otsu':{'role':'retain_fitting_population','parameter':'min_num_fragments',
                               'applies_if':'linked_qualified_filter'}},resolve_selection=require_linked_qc_fit)
    needs={'format': 'h5ad'}; output={'format': 'h5ad'}; parameters={}; effects={}; selection={}
    if name in REQUIRED:
        parameters['resources']=REQUIRED[name]
    if name in ('pp.add_tile_matrix', 'pp.make_peak_matrix', 'pp.make_gene_matrix'):
        needs['x' if kwargs.get('use_x') else 'native_values']=True
        parameters['storage_by_use_x']={'true': 'x', 'false': 'native_values'} if name!='pp.add_tile_matrix' else {'false': 'native_values'}
        if name=='pp.make_peak_matrix':
            parameters['regions']={'alternatives': ('peaks', 'kwargs.use_rep:list', 'uns[kwargs.use_rep]'),
                                   'native_default': 'uns.'+DEFAULT_PEAK_REGION_KEY}
            selected=checked_raw_operand(arguments)
            if isinstance(selected,str): needs['regions.'+selected]=True
        output.update(x=True, scanpy_auto_rep=True, scanpy_x_rep=True,
                      counting_strategy=kwargs.get('counting_strategy',arguments.get('counting_strategy','paired-insertion')),
                      **{'var.selected': False})
        effects['feature_axis']='replace'
        if not kwargs.get('inplace', name=='pp.add_tile_matrix'):
            output.update(fragment_storage=False, paired_fragments=False, native_values=False,
                representations=(),graphs=(),root=False)
    elif name in ('metrics.tsse', 'metrics.frag_size_distr'):
        needs['fragment_storage']=True
        slots=('obs.tsse','uns.library_tsse','uns.frac_overlap_TSS','uns.TSS_profile') if name=='metrics.tsse' else ('uns.'+kwargs.get('add_key','frag_size_distr'),)
        parameters['inplace_writes']=slots
        if kwargs.get('inplace',True):
            output.update(dict.fromkeys(slots,True))
            if name=='metrics.tsse':output['obs.tsse']='numeric'
        else:effects['independent_return']=True
    elif name=='pp.filter_cells':
        defaults=FILTER_DEFAULTS | kwargs
        needs.update({field:'numeric' for field,bounds in FILTER_FIELDS.items() if any(defaults[key] for key in bounds)})
        if active_qc(arguments)=='otsu':
            if 'min_counts' not in kwargs:needs['obs.n_fragment']='numeric'
            if 'min_tsse' not in kwargs:needs['obs.tsse']='numeric'
        parameters.update(defaults=FILTER_DEFAULTS, reads_if_truthy_bound=FILTER_FIELDS)
        if arguments.get('qc_selection',{}).get('reason')=='population_history_unknown':
            parameters['required_ports']=('qc_population',)
        selection={'otsu':{'role':'fit_omitted_minima','parameters':QC_FILTER_MINIMA,
                           'applies_if':'uncalled_single_sample','unknown':'ask_qc_population',
                           'inapplicable':'native_defaults_unless_explicit_otsu'}}
    elif name=='pp.call_cells':
        parameters['use_rep']={'required': True, 'string_axis': 'obs', 'alternative': 'aligned_vector'}
        selected=checked_raw_operand(arguments)
        if isinstance(selected,str):needs['obs.'+selected]='numeric'
    if name in ('pp.filter_cells','pp.call_cells'):
        effects.update(cell_axis='subset' if kwargs.get('inplace',True) else 'preserve',
                       independent_return=not kwargs.get('inplace',True))
    if name in ('metrics.tsse','metrics.frag_size_distr') or name in ('pp.filter_cells','pp.call_cells') and not kwargs.get('inplace',True):
        effects['graph_storage']='preserve'
    return ScientificContract(inputs={'dataset':needs}, parameters=parameters, outputs={'dataset':output},
                              effects=effects, selection=selection,
                              resolve_selection=coordinate_qc if name=='pp.filter_cells' else None)


def prepare_operation(arguments, context):
    from agent.orchestration.scientific_inputs import ScientificDiagnostic, ScientificInputRequired, ScientificInputRequirement
    contract=scientific_contract(arguments)
    function=arguments.get('function_name');kwargs=arguments.get('kwargs',{})
    if function=='pp.filter_cells' and 'qc_selection' in arguments:
        decision,_=qc_decision(arguments,context)
        if decision['method'] is None:
            raise ScientificInputRequired(ScientificInputRequirement('qc_population','qc_population','scalar',
                choices=QC_POPULATIONS,dependencies=('source:bytes',),
                messages=('What is the cell-calling or selection history of this source population?',)),bindings=arguments)
        arguments=dict(arguments,qc_selection=decision)
    if function=='pp.call_cells' and peak_region_key(arguments) is None:
        raise ScientificInputRequired(ScientificInputRequirement('kwargs','kwargs','scalar',member='use_rep',
            applicability={'expected_types': ('str','list')}, dependencies=('source:bytes',),
            messages=('Supply the exact observation count field or aligned count vector for OrdMag.',)))
    selected=peak_region_key(arguments)
    if function=='pp.call_cells' and isinstance(selected,list):
        count=context.properties(arguments.get('input_path'),('n_obs',)).get('n_obs')
        if count is not None and count!=len(selected):
            raise ScientificDiagnostic('NATIVE_CALL_CELLS_COUNTS_INVALID','Count vector length must match the selected ordered observation axis.')
    for condition in contract.parameters.get('conditions', ()):
        kwargs=arguments.get(condition['argument'], {})
        effective=[name for name in condition['members'] if kwargs.get(name) is not None]
        if len(effective)>1 or any(not isinstance(kwargs[name], str) or not kwargs[name].strip() for name in effective):
            raise ScientificDiagnostic('NATIVE_BARCODE_EXTRACTION_INVALID',
                'Supply exactly one nonempty barcode_tag or barcode_regex for pp.make_fragment_file.')
        if effective:
            continue
        # Only an exact Answer to this Candidate/step/source can select the next
        # member. No filename, assay, reference bundle or historical default is
        # evidence for a BAM barcode mechanism.
        answer=context.answers.get(condition['argument'], {})
        choice=next((row for row in BARCODE_CHOICES if row==answer), None)
        if choice is None:
            raise ScientificInputRequired(ScientificInputRequirement('kwargs', 'kwargs', 'resource',
                offers=BARCODE_CHOICES, dependencies=('source:bytes',), applicability=condition,
                messages=('Choose one barcode-extraction mechanism for the selected BAM.',)))
        raise ScientificInputRequired(ScientificInputRequirement('kwargs', 'kwargs', 'scalar',
            member=choice['member'], dependencies=('source:bytes',),
            applicability=dict(expected_types=('str',), conditional=condition),
            messages=('Supply the exact '+choice['member']+' for the selected BAM.',)))
    return arguments


def fragment_name(args):
    compression = args.get('kwargs', {}).get('compression')
    return 'fragments.tsv.zst' if compression == 'zstandard' else 'fragments.tsv' if compression is None else 'fragments.tsv.gz'


def files(value):
    args = value['arguments']
    return (fragment_name(args), 'metadata.json', 'native-receipt.json') if args['function_name'] == 'pp.make_fragment_file' else ('result.h5ad', 'metadata.json', 'native-receipt.json')


def outputs(value):
    bam = value['arguments']['function_name'] == 'pp.make_fragment_file'
    return dict(matrix=None if bam else 'result.h5ad', fragments=fragment_name(value['arguments']) if bam else None)


def input_paths(args):
    name = args['function_name']
    paths = [(args['input_path'], args.get('input_sha256'))]
    for port, (path, sha, _) in PORTS.items():
        if args.get(path) is not None:
            if port not in APPLIES.get(name, ()):
                raise ValueError(f'{port} is not an input of snapatac2.{name}')
            paths.append((args[path], args.get(sha)))
        elif args.get(sha) is not None or port in REQUIRED.get(name, ()) and port not in args.get('kwargs', {}):
            raise ValueError(f'snapatac2.{name} requires the {port} input file')
    if args.get('count_slot') is not None:
        raise ValueError('Native raw functions consume their file object; count_slot applies to matrix functions')
    kw = args.get('kwargs', {})
    if 'chrom_sizes' in kw and (type(kw['chrom_sizes']) is not dict or args.get('chrom_sizes_path')):
        raise ValueError('Use one native chromosome-size dictionary or its exact file port')
    if 'whitelist' in kw and (type(kw['whitelist']) is not list or args.get('whitelist_path')):
        raise ValueError('Use one native barcode list or its exact whitelist file port')
    return paths


def axes(path):
    """Only axes/metadata, never load a genome-sized fragment obsm into memory."""
    from . import peak_h5ad
    # Native imports with zero features legitimately omit var and X. Ordinary
    # matrix consumers still require X; this is a fragment-storage boundary.
    # Walk all storage/link declarations before native code can follow them.
    # Numeric fragment arrays are bounded by shape without loading their values.
    peak_h5ad.inspect(path, require_matrix=False, max_logical_bytes=None)
    with h5py.File(path, 'r') as handle:
        obs = read_elem(handle['obs'])
        if 'X' not in handle and not {'fragment_paired', 'fragment_single'}.intersection(handle.get('obsm', {})):
            raise AuthorityError('Native result has neither a matrix nor native fragment storage')
        return obs.index.tolist(), read_elem(handle['var']).index.tolist() if 'var' in handle else []


def execute(owner, args, execution_identity):
    from .native_matrix import arguments
    from .sequence_prediction import write
    args = arguments(args)
    kwargs = owner.native_arguments(args)
    sources = []
    for path, sha in input_paths(args):
        pin = file_closure([path])[0]
        if sha is not None and pin['sha256'] != sha:
            raise AuthorityError('Selected native input bytes changed')
        sources.append(pin)
    root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Output directory is not empty; retry requires a new authorized execution')
    with tempfile.TemporaryDirectory(prefix='.native-snapatac2-', dir=root) as temporary:
        stage = Path(temporary)
        name = args['function_name']
        before = None if name in ('pp.make_fragment_file', 'pp.import_fragments') else axes(args['input_path'])
        receipt = owner.native(args, stage, None)
        is_bam = name == 'pp.make_fragment_file'
        payload = fragment_name(args) if is_bam else 'result.h5ad'
        if not (stage / payload).is_file():
            raise AuthorityError('Native function did not produce its declared output')
        cells, features = (None, None) if is_bam else axes(stage / payload)
        if before is not None:
            prior_cells, prior_features = before
            if name in ('pp.filter_cells', 'pp.call_cells') and kwargs.get('inplace', True):
                positions = iter(prior_cells)
                if any(not any(old == cell for old in positions) for cell in cells):
                    raise AuthorityError('Native filtering changed ordered source cell identities')
            elif cells != prior_cells:
                raise AuthorityError('Native function changed ordered source cell identities')
            if name not in ('pp.make_peak_matrix', 'pp.add_tile_matrix', 'pp.make_gene_matrix') and features != prior_features:
                raise AuthorityError('Native QC/filtering changed feature identities')
        source = sources[0]
        summary = dict(status='success', contract_version=owner.CONTRACT,
            function_name='snapatac2.' + name, n_cells=None if is_bam else len(cells),
            n_features=None if is_bam else len(features), measurement=receipt['measurement'],
            parameters=dict(kwargs=args.get('kwargs', {}), omitted_optional_arguments=(
                'native defaults except recorded Agent input-order selection' if args.get('fragment_order') is not None else 'native defaults'),
                effective_native_parameters=receipt['effective_parameters'], output_storage='native H5AD or fragment file'),
            source_identity=dict(sha256=source['sha256'], size_bytes=source['size_bytes'],
                references=[dict(role=port, sha256=next(p['sha256'] for p in sources if p['path'] == args[path]))
                    for port, (path, _, _) in PORTS.items() if args.get(path)],
                declarations=receipt.get('source_declarations', {}),
                supplied_declarations={key: args[key] for key in ('species', 'assembly', 'matrix_semantics') if key in args}),
            software_versions=receipt['versions'], native_output_type=receipt['native_output_type'],
            result_slots=receipt['result_slots'], limitations=LIMITATIONS)
        context=current()
        population=args.get('qc_population')
        decision=args.get('qc_selection')
        if decision:
            summary['parameters']['qc_selection']=decision
            population=decision.get('population') or population
        if context is not None and scientific_contract(args).effects.get('graph_storage')=='preserve':
            context.validate_anchors()
            for proof in context.proofs.values():
                prior=proof.result_metadata
                if (proof.kind in ('native_snapatac2_matrix','native_scanpy_matrix','native_episcanpy_matrix')
                        and prior.get('matrix_path')==source['path'] and prior.get('matrix_sha256')==source['sha256']):
                    if graphs:=prior.get('parameters',{}).get('graph_inputs'):
                        from agent.schemas.orchestration import _serialize
                        summary['parameters']['graph_inputs']=_serialize(graphs)
                    population=accepted_qc_population(prior.get('parameters',{})) or population
                    break
        if population is not None:summary['parameters']['qc_population']=population
        if name == 'pp.make_gene_matrix':
            from .native_gene_activity import publish_description
            publish_description(stage / payload, args, receipt, summary, source)
        if active_qc(args)=='otsu':
            summary['parameters']['qc_thresholds'] = 'otsu'
            if 'threshold_selection' in receipt:
                summary['parameters']['threshold_selection'] = receipt['threshold_selection']
            summary['parameters']['omitted_optional_arguments'] = 'native defaults except recorded Agent choices'
            summary['parameters']['agent_selected_parameters'] = {
                key: value for key, value in receipt['effective_parameters'].items()
                if key in ('min_num_fragments', 'min_counts', 'min_tsse') and key not in args.get('kwargs', {})}
        if args.get('fragment_order') is not None:
            summary['parameters']['fragment_order'] = args['fragment_order']
            summary['parameters'].setdefault('agent_selected_parameters', {}).update({
                key: value for key, value in kwargs.items() if key not in args.get('kwargs', {})})
        if args.get('counting_strategy') is not None:
            # A supplied scalar may be a user declaration or registered Planner
            # choice. The captured request/plan owns that attribution; do not
            # mislabel it as a native default or synthetic user instruction.
            summary['parameters']['counting_strategy'] = args['counting_strategy']
            summary['parameters']['omitted_optional_arguments'] = 'native defaults except recorded counting strategy'
        if name == 'pp.import_fragments':
            summary['parameters']['population'] = dict(stage='native_import', retained_barcodes=len(cells),
                min_num_fragments=receipt['effective_parameters']['min_num_fragments'],
                pre_import_barcode_count=None, reason='Native import retention; Agent did not separately count the source barcode population')
            if args.get('whitelist_path') is not None or kwargs.get('whitelist') is not None:
                summary['parameters']['qc_population']='restricted_population'
        elif name in ('pp.filter_cells', 'pp.call_cells'):
            summary['parameters']['population'] = dict(stage='native_cell_filter' if name == 'pp.filter_cells' else 'native_ordmag_cell_call', input_barcodes=len(before[0]),
                selected_barcodes=len(cells) if kwargs.get('inplace', True) else receipt['selected_count'],
                output_barcodes=len(cells), applied=kwargs.get('inplace', True))
        expected = {r['path']: r for r in [*sources, *receipt['implementation']]}
        sources = file_closure(expected)
        if sources != sorted(expected.values(), key=lambda r: r['path']):
            raise AuthorityError('Source or implementation changed during native preprocessing')
        cancellation_checkpoint()
        shutil.copyfile(stage / payload, root / payload)
        write(root / 'native-receipt.json', receipt)
        write(root / 'metadata.json', dict(summary=summary, receipt=receipt, sources=sources))
    value = dict(contract=owner.CONTRACT, profile=owner.PROFILE, arguments=args, execution_identity=execution_identity,
        result=summary, sources=sources, native_completion='succeeded')
    value['files'] = file_closure(root / name for name in files(value))
    cancellation_checkpoint()
    path = root / 'manifest.json'; sha = write(path, value)
    if current() is not None:
        current().register_execution(owner.KIND, root, execution_identity)
    return owner.verify_publication(path, expected_sha256=sha)
