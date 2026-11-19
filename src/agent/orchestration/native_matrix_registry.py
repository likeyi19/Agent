"""Package-level native matrix interfaces using the existing typed Registry."""
from pathlib import Path
from collections.abc import Mapping
from agent.tools.analysis import native_matrix


def classify(exception):
    from .registry import ErrorClassification
    from agent.schemas import ErrorCategory
    from agent.schemas.verification_authority import AuthorityError
    if isinstance(exception, native_matrix.NativeInvocationError):
        return ErrorClassification(ErrorCategory.USER_INPUT_ERROR, exception.code, str(exception))
    if isinstance(exception, AuthorityError):
        return ErrorClassification(ErrorCategory.VERIFICATION_ERROR, 'NATIVE_FUNCTION_PUBLICATION_INVALID')
    if isinstance(exception, (ValueError, KeyError, FileNotFoundError)):
        return ErrorClassification(ErrorCategory.USER_INPUT_ERROR, 'NATIVE_MATRIX_INPUT_INVALID')
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, 'NATIVE_FUNCTION_EXECUTION_FAILED')


def tool_spec(owner):
    from .registry import (NativeFunctionCatalog, ToolSpec, ResultContract, DurableToolHooks, ScientificAuthorityHooks,
        ArtifactSemanticKind as A, PlanningToolRole, PlanningSourceEligibility as S, SemanticToolSpec,
        SemanticConsumerPortSpec as Consumer, SemanticProducerPortSpec as Producer, _planning_argument as arg,
        _planning_result, _tool_planning, _semantic_member as member, _semantic_request as request,
        _semantic_request_member as rm, _semantic_argument_port as scalar)
    from .native_delivery import delivery_spec
    required = dict(
        input_path=arg((str, Path), ('BAM, original fragments or native H5AD, as required by the selected function.' if owner.package == 'snapatac2' else
            'H5AD with exact cells/features, selected measurement and native layers or feature mask as required.'),
            artifacts=(A.RAW_SCATAC, A.EPIZOO_EMBEDDING), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        output_dir=arg((str, Path), 'Managed native publication destination.'),
        function_name=arg((str,), 'Explicit public ' + owner.package + ' function.', choices=tuple(owner.FUNCTIONS),
            scientific_parameter=True, conversational_choice=True, planner_choice=True))
    optional = dict(
        input_sha256=arg((str,), 'Captured exact source checksum.'),
        kwargs=arg((dict,), 'Native kwargs from grounded assignments or JSON; omit unrequested defaults.',
            scientific_parameter=True, conversational_json_object=True, conversational_members=owner.NATIVE_ARGUMENTS),
        count_slot=arg((str,), 'X by default or exact source layer; this becomes native X without normalization.',
            scientific_parameter=True, conversational_exact_string=True),
        matrix_semantics=arg((str,), 'Optional actual source measurement; no declaration is required for native invocation.',
            scientific_parameter=True, conversational_exact_string=True))
    if owner.package in ('snapatac2', 'scanpy', 'episcanpy'):
        optional['gene_activity'] = arg((bool,),
            'Explicitly identify existing Gene Activity/Gene Score in selected X/layer; no reconstruction or normalization. '
            'Unknown construction stays unknown; uploaded uns metadata is not accepted execution authority.',
            scientific_parameter=True)
        optional['cell_subset'] = arg((Mapping,),
            'Internal native input adaptation for annotation-owned populations. On an exactly bound source, '
            'groupby/groups determines membership; positions may also be supplied. The owner captures source '
            'checksum, positions and cell IDs, verifying any supplied assertions. Do not ask researchers for '
            'these provenance fields. Sliced global graphs/coordinates remain inherited.',
            scientific_parameter=True)
        optional['input_orientation'] = arg((str,),
            'Only an explicit features_by_cells declaration transposes the task-local AnnData before source-slot/cell selection.',
            choices=('cells_by_features', 'features_by_cells'), scientific_parameter=True)
        optional['parameter_selection'] = arg((str,),
            'AI Research automatically selects omitted eligible clustering parameters by bounded ASW after method selection. '
            'Use native_defaults only when the researcher requests native defaults; asw requests bounded selection. '
            'Explicit resolution/num_clusters always prevails. Other functions retain native defaults. '
            'epiScanpy kmeans/hc require num_clusters when native_defaults is requested.',
            choices=('asw', 'native_defaults'), scientific_parameter=True, conversational_choice=True)
        optional['require_asw'] = arg((bool,), 'Internal flag for an explicit ASW request.')
        optional['parameter_origin'] = arg((str,), 'Internal provenance for a carried-forward fitted value.',choices=('retained_fitted',))
    if any('use_rep' in members for members in owner.FUNCTIONS.values()):
        optional['use_rep'] = arg((str,),
            'Exact representation for a function exposing native use_rep; X_epizoo is the compact EpiZoo embedding. '
            'Other exact representations may be supplied in native kwargs. Explicit kwargs prevail. '
            'Do not pass this selector to graph-consuming functions without use_rep, including Scanpy UMAP.',
            choices=('X_spectral', 'X_epizoo', 'X_pca', 'X_lsi'), scientific_parameter=True,
            conversational_choice=True, planner_choice=True)
    if owner.package in ('scanpy','episcanpy'):
        optional['root_cell'] = arg((str,), 'Explicit exact observation ID for DPT; adapted only on the task-local AnnData. Omission retains a valid source uns.iroot. Never infer a root.', scientific_parameter=True, conversational_exact_string=True)
    raw_fields = set()
    if owner.package == 'scanpy':
        optional['group_column'] = arg((str,), 'Explicit source obs grouping for the existing Gene Activity projection; user-supplied, not an accepted clustering claim.', scientific_parameter=True)
        optional['grouping_path'] = arg((str, Path),
            'Accepted native grouping for rank_genes_groups on reused Gene Activity; exact common source ancestry required.',
            source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT)
    if owner.package == 'snapatac2':
        from agent.tools.analysis import snapatac_raw as raw
        optional['qc_thresholds'] = arg((str,),
            'Task choice for uncalled single-sample fragments: otsu retains all observed barcodes on import '
            '(min_num_fragments=0), then derives omitted filter_cells minima from native n_fragment '
            '(log10, floor 100) and tsse (linear, floor 1), skimage Otsu with 5000 bins. '
            'Explicit minima prevail independently. Preserve called cells; no automatic OrdMag stacking. '
            'Explicit native calls omit this choice and retain native defaults.',
            choices=('otsu','native_defaults'), scientific_parameter=True, conversational_choice=True)
        optional['qc_population'] = arg((str,), 'Exact source population history for selected QC: uncalled one sample, study-selected/called, depth-prefiltered, or multiple samples.',
            choices=raw.QC_POPULATIONS, scientific_parameter=True, conversational_choice=True)
        optional['qc_selection'] = arg((Mapping,), 'Internal effective QC strategy and population evidence.')
        optional['counting_strategy'] = arg((str,),
            'Native pp.make_peak_matrix/pp.add_tile_matrix/pp.make_gene_matrix counting strategy. For research tasks select a representation compatible '
            'with the current objective and stated downstream method: fragment gives fragment-body overlaps, '
            'insertion gives endpoints, paired-insertion gives paired endpoints. Preserve explicit kwargs; for explicit native calls '
            'omit an unrequested choice and retain native defaults.',
            choices=('fragment', 'insertion', 'paired-insertion'), scientific_parameter=True,
            conversational_choice=True, planner_choice=True)
        optional['fragment_order'] = arg((str,),
            'Task input-order knowledge for pp.import_fragments only: unknown selects native sorting; '
            'barcode uses established barcode ordering. Explicit kwargs.sorted_by_barcode prevails. '
            'Capture only explicit task source-order declarations; never ask for this optional information. '
            'Omission retains the native default; never add this task input for an explicit native-function request.',
            choices=('unknown', 'barcode'), scientific_parameter=True, conversational_choice=True)
        for port, (path, sha, description) in raw.PORTS.items():
            optional[path] = arg((str, Path), description, artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT)
            optional[sha] = arg((str,), 'Exact captured ' + port + ' checksum.')
            raw_fields.update((path, sha))
        for key in ('species', 'assembly'):
            optional[key] = arg((str,), 'Supplied source ' + key + '; no inferred reference correspondence.',
                scientific_parameter=True, conversational_exact_string=True)
    if owner.package == 'episcanpy':
        from agent.tools.analysis.snapatac_raw import PORTS
        path, sha, description = PORTS['gene_anno']
        optional[path] = arg((str, Path), description, artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT)
        optional[sha] = arg((str,), 'Exact captured gene annotation checksum.')
        raw_fields.update((path, sha))
        for key in ('species', 'assembly'):
            optional[key] = arg((str,), 'Supplied source ' + key + '; no inferred reference correspondence.',
                scientific_parameter=True, conversational_exact_string=True)
    if owner.package == 'Signac':
        optional.update(second_path=arg((str, Path), 'Exact second H5AD for Jaccard or corSparse Y; identical ordered features required.', artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
            second_sha256=arg((str,), 'Captured second source checksum.'),
            second_count_slot=arg((str,), 'Second input X or selected layer, independently bound.', scientific_parameter=True, conversational_exact_string=True))
    from .native_parameters import scalar_port
    base_scalar = scalar
    def scalar(key, **kw):
        if not {**required, **optional}[key].planning.scientific_parameter:
            return base_scalar(key, **kw)
        functions={'root_cell':('tl.dpt',),'qc_population':('pp.import_fragments','pp.filter_cells'),
                   'qc_thresholds':('pp.import_fragments','pp.filter_cells')}.get(key,tuple(owner.FUNCTIONS))
        return scalar_port(owner.TOOL,functions,key,**kw)
    ports = [Consumer('dataset', (member('dataset', 'input_path'),), True,
        request_sources=(request('input_path', rm('dataset', 'input_path')),),
        accepted_upstream_types=('raw_scatac_dataset.v1', 'scatac_matrix_h5ad.v1', 'native_matrix_h5ad.v1', 'cell_embedding_h5ad.v1', 'hierarchical_annotation_h5ad.v1')
            + (('native_fragments_file.v1',) if owner.package == 'snapatac2' else ())),
        *(scalar(k, required=k in required) for k in (*required, *optional)
            if k not in {'input_path', 'second_path', 'grouping_path', 'require_asw', 'qc_selection', 'parameter_origin', *raw_fields})]
    if owner.package == 'scanpy':
        ports.append(Consumer('grouping', (member('dataset', 'grouping_path'),), False,
            request_sources=(request('grouping_path', rm('dataset', 'grouping_path')),),
            accepted_upstream_types=('native_matrix_h5ad.v1',)))
    if owner.package == 'snapatac2':
        from .native_parameters import input_name
        for port, (path, sha, _) in raw.PORTS.items():
            ports.append(Consumer(port, (member('path', path), member('sha256', sha)), False,
                accepted_upstream_types=('scatac_peaks.v2',) if port == 'peaks' else (),
                request_sources=(request(path, rm('path', path), rm('sha256', sha)),) + tuple(
                    request(input_name(owner.TOOL, fn, path),
                        rm('path', input_name(owner.TOOL, fn, path)), rm('sha256', input_name(owner.TOOL, fn, sha)))
                    for fn, applies in raw.APPLIES.items() if port in applies)))
    if owner.package == 'Signac':
        ports.append(Consumer('second', (member('dataset', 'second_path'),), False,
            request_sources=(request('second_path', rm('dataset', 'second_path')),),
            accepted_upstream_types=('raw_scatac_dataset.v1', 'scatac_matrix_h5ad.v1','hierarchical_annotation_h5ad.v1', 'native_matrix_h5ad.v1')))
    if owner.package == 'episcanpy':
        from .native_parameters import input_name
        ports.append(Consumer('gene_anno', (member('path', 'gene_anno_path'), member('sha256', 'gene_anno_sha256')), False,
            request_sources=(request('gene_anno_path', rm('path', 'gene_anno_path'), rm('sha256', 'gene_anno_sha256')),
                request(input_name(owner.TOOL, 'tl.geneactivity', 'gene_anno_path'),
                    rm('path', input_name(owner.TOOL, 'tl.geneactivity', 'gene_anno_path')),
                    rm('sha256', input_name(owner.TOOL, 'tl.geneactivity', 'gene_anno_sha256'))))))
    pub = owner.publication
    raw_guidance = (
        'Native raw functions: pp.make_fragment_file consumes BAM and produces fragments. pp.import_fragments '
        'directly consumes plain/gzip/zstd fragments plus chrom_sizes. For ordinary tasks bind the offered '
        'function-scoped fragment_order input: unknown selects sorted_by_barcode=false and SnapATAC2 performs sorting. '
        'Never ask the researcher for sorting flags or detect/sort order in Agent. '
        'pp.make_fragment_file guarantees barcode-sorted output; its import may omit fragment_order and retain native true. '
        'Chromap BED has no barcode-order guarantee. Explicit sorted_by_barcode kwargs take precedence. '
        'For explicit native-function requests omit task-only fragment_order; omitted kwargs retain native defaults. '
        'TBI is unused. No canonical fragment import or internal source profile is required. '
        'metrics.tsse consumes native fragment H5AD and GTF/GFF gene_anno. pp.filter_cells is a separate native selection '
        'with its own method. pp.make_peak_matrix requires one of its declared region inputs. '
        'Its native default is paired-insertion. Research tasks may select the registered counting_strategy '
        'choice for a scientifically justified downstream method, without inventing a user declaration. '
        'Choose methods and compatible regions from the objective and available references, not integration order; '
        'no annotation method or region vocabulary is mandatory. Explicit native calls retain omitted defaults, '
        'and explicit user kwargs or counting choices prevail. Reuse accepted fragment storage when a later '
        'matrix needs different counting; do not relabel or silently convert an existing measurement. '
        'Population-scoped QC selection is determined by the selected scientific owner. Do not assume uncalled status. '
        'Already called inputs retain established identities unless filtering is requested. '
        'Barcode correction and a whitelist alone do not establish called-cell identities. '
        'pp.call_cells is explicit native OrdMag, never an additional default before or after Otsu. '
        'Omitted task selection on explicit native calls retains native import/filter defaults. '
        'Never substitute old Agent TSS 4 or global fragment-body counting defaults. '
        'A user request to retain low-depth/all barcodes may explicitly bind '
        'min_num_fragments=0; do not lower thresholds just for a small dataset. Imported barcodes and later filtered '
        'barcodes are distinct populations. '
        'Processed compatible H5AD needs no raw reimport or cell calling. '
        'Reuse accepted exact matrices/graphs across turns. '
        'EpiZoo is a separate model-based embedding: consume its dataset port (embedding-only H5AD), '
        'selecting use_rep=X_epizoo for representation consumers. Never reconstruct counts or apply '
        'count-matrix reduction to an embedding-only H5AD. '
        'Preserve supplied barcode/coordinate history and declared source/reference species/assembly. '
        'Use optional native reference ports only for the function that consumes them. '
        'pp.make_gene_matrix consumes native fragment-backed H5AD and compatible gene_anno GTF/GFF, '
        'returning a new cell-by-gene/transcript H5AD; retain the accepted fragment source. '
        'Explicit use_x=true aggregates the existing region values, not reconstructed fragments. '
    ) if owner.package == 'snapatac2' else (
        'tl.geneactivity requires compatible GTF gene_anno and genomic peak coordinate var_names. '
        'Installed epiScanpy 0.4.0 parses underscore-delimited coordinates first, so chromosome names '
        'containing underscores can be rejected by its native coordinate parser. Select a compatible '
        'Gene Activity construction with its actual source; native failures remain visible. '
        'It sums supplied peak values into a new gene/transcript X with native dense intermediates/output; '
        'binary accessibility and peak counts remain different measurements. '
        'Native default annotation=HAVANA selects that GTF source; explicit annotation=null accepts all sources. '
        'Copied graphs and embeddings remain source-derived. No implicit normalization or binarization. '
        'Other matrix functions need no genome. '
    ) if owner.package == 'episcanpy' else 'No genome is required for these matrix functions. '
    notes = tuple(sentence + '.' for sentence in raw_guidance.strip().rstrip('.').split('. '))
    from agent.application.native_raw_handoff import prepare_operation
    return ToolSpec(source_encodings={p.name:((('h5ad','bam','fragments') if owner.package=='snapatac2' else ('h5ad',)) if p.name in ('dataset','second') else {'chrom_sizes':('fai','chromosome_sizes_tsv'),'gene_anno':('gtf','gff'),'peaks':('bed',),'whitelist':('barcode_whitelist',)}[p.name]) for p in ports if p.name in ('dataset','second') or any(m.field_name in raw_fields for m in p.members)}, source_input_types={p.name: (('h5ad','bam','external_fragments') if owner.package=='snapatac2' and p.name=='dataset' else ('h5ad',) if p.name in ('dataset','second') else ('attachment','scientific_reference')) for p in ports if p.name in ('dataset','second') or any(m.field_name in raw_fields for m in p.members)}, preparation_owner=prepare_operation, native_functions=NativeFunctionCatalog(owner.package, tuple(owner.FUNCTIONS.items()), parameter_metadata=owner.parameter_metadata, output_available=owner.output_available),scientific_contract=owner.scientific_contract,name=owner.TOOL, function=owner.invoke, required_arguments=required, optional_arguments=optional,
        application_inputs=frozenset({*raw_fields,'kwargs',
            *({'species','assembly'} if owner.package in ('snapatac2','episcanpy') else set()),
            *({'qc_selection'} if owner.package=='snapatac2' else set())}),
        scientific_delivery=delivery_spec(owner, 'scientific_native_result', h5ad_roles=('result.h5ad',)),
        result_contract=ResultContract('NativeMatrixResult', owner.RESULT_FIELDS, planning_fields={
            k: _planning_result('Exact native publication/file; null outputs are unavailable for this function.', bindable=True, artifact=A.RAW_SCATAC)
            for k in ('manifest_path', 'manifest_sha256', 'matrix_path', 'matrix_sha256',
                *(('fragments_path', 'fragments_sha256') if owner.package == 'snapatac2' else ()))}),
        durable_hooks=DurableToolHooks(owner.execute, owner.recover), exception_classifier=classify,
        recovery_policy_version=owner.POLICY,
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND, producer_ports=('matrix', 'dataset')
                + (('fragments',) if owner.package == 'snapatac2' else ()),
            describe=pub.describe, publication_record=pub.publication_record, reuse_result=pub.reuse_result,
            proof_metadata=pub.proof_metadata, independent_active_publications=True),
        planning=_tool_planning(PlanningToolRole.OPERATION, owner.guidance,
            'Invoke only the named supported native function with explicit kwargs unchanged and unspecified native defaults omitted. '
            'Preserve source bytes, ordered axes and distinct result semantics. '
            'Select native methods by scientific capabilities and researcher intent, with no package priority.',
            'Use the exact representation key for compatible consumers, including X_epizoo. '
            'Never substitute incompatible package graphs or invent a DPT root.',
            'Use only the documented automatic strategy for that operation. Do not plan candidate scans or invent numeric values.', *notes),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports), producer_ports=(
            Producer('matrix', owner.CONTRACT, (member('path', 'manifest_path'), member('sha256', 'manifest_sha256'))),
            Producer('dataset', 'native_matrix_h5ad.v1', (member('dataset','matrix_path'),member('sha256','matrix_sha256')), lineage_from_port='dataset'),
            *((Producer('fragments', 'native_fragments_file.v1', (member('dataset', 'fragments_path'),), lineage_from_port='dataset',
                        observation_sha256_field='fragments_sha256'),)
                if owner.package == 'snapatac2' else ()))))


def tool_specs():
    return tuple(tool_spec(owner) for owner in native_matrix.PACKAGES)
