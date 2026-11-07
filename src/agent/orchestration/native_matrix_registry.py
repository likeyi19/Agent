"""Package-level native matrix interfaces using the existing typed Registry."""
from pathlib import Path
from agent.tools.analysis import native_matrix


def classify(exception):
    from .registry import ErrorClassification
    from agent.schemas import ErrorCategory
    from agent.schemas.verification_authority import AuthorityError
    if isinstance(exception, native_matrix.NativeInvocationError):
        return ErrorClassification(ErrorCategory.USER_INPUT_ERROR, 'NATIVE_FUNCTION_REJECTED', str(exception))
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
            artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
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
    raw_fields = set()
    if owner.package == 'snapatac2':
        from agent.tools.analysis import snapatac_raw as raw
        optional['counting_strategy'] = arg((str,),
            'Native pp.make_peak_matrix counting strategy. For research tasks select a representation compatible '
            'with the current objective and stated downstream method: fragment gives fragment-body overlaps, '
            'insertion gives endpoints, paired-insertion gives paired endpoints. MAESTRO region-count annotation '
            'accepts fragment counts, not insertion counts. Preserve explicit kwargs; for explicit native calls '
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
    if owner.package == 'Signac':
        optional.update(second_path=arg((str, Path), 'Exact second H5AD for Jaccard or corSparse Y; identical ordered features required.', artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
            second_sha256=arg((str,), 'Captured second source checksum.'),
            second_count_slot=arg((str,), 'Second input X or selected layer, independently bound.', scientific_parameter=True, conversational_exact_string=True))
    from .native_parameters import scalar_port
    base_scalar = scalar
    scalar = lambda k, **kw: scalar_port(owner.TOOL, tuple(owner.FUNCTIONS), k, **kw) if ({**required, **optional}[k].planning.scientific_parameter) else base_scalar(k, **kw)
    ports = [Consumer('dataset', (member('dataset', 'input_path'),), True,
        request_sources=(request('input_path', rm('dataset', 'input_path')),),
        accepted_upstream_types=('raw_scatac_dataset.v1', 'scatac_matrix_h5ad.v1', 'native_matrix_h5ad.v1')
            + (('native_fragments_file.v1',) if owner.package == 'snapatac2' else ())),
        *(scalar(k, required=k in required) for k in (*required, *optional)
            if k not in {'input_path', 'second_path', *raw_fields})]
    if owner.package == 'snapatac2':
        from .native_parameters import input_name
        for port, (path, sha, _) in raw.PORTS.items():
            ports.append(Consumer(port, (member('path', path), member('sha256', sha)), False,
                request_sources=(request(path, rm('path', path), rm('sha256', sha)),) + tuple(
                    request(input_name(owner.TOOL, fn, path),
                        rm('path', input_name(owner.TOOL, fn, path)), rm('sha256', input_name(owner.TOOL, fn, sha)))
                    for fn, applies in raw.APPLIES.items() if port in applies)))
    if owner.package == 'Signac':
        ports.append(Consumer('second', (member('dataset', 'second_path'),), False,
            request_sources=(request('second_path', rm('dataset', 'second_path')),),
            accepted_upstream_types=('raw_scatac_dataset.v1', 'scatac_matrix_h5ad.v1', 'native_matrix_h5ad.v1')))
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
        'with its own method. pp.make_peak_matrix consumes imported native H5AD and optional peaks BED. '
        'Its native default is paired-insertion. Research tasks may select the registered counting_strategy '
        'choice for a scientifically justified downstream method, without inventing a user declaration. '
        'Choose methods and compatible regions from the objective and available references, not integration order; '
        'no annotation method or region vocabulary is mandatory. Explicit native calls retain omitted defaults, '
        'and explicit user kwargs or counting choices prevail. Reuse accepted fragment storage when a later '
        'matrix needs different counting; do not relabel or silently convert an existing measurement. '
        'Use native import min_num_fragments=200 and filter_cells min_counts=1000/min_tsse=5.0 by omission. '
        'Never substitute old Agent TSS 4 or global fragment-body counting defaults. '
        'A user request to retain low-depth/all barcodes may explicitly bind '
        'min_num_fragments=0; do not lower thresholds just for a small dataset. Imported barcodes and later filtered '
        'barcodes are distinct populations. Native import has no X measurement: build a matrix before PCA/LSI. '
        'Preserve supplied barcode/coordinate history and declared source/reference species/assembly. '
        'Use optional native reference ports only for the function that consumes them. '
    ) if owner.package == 'snapatac2' else 'No genome is required for these matrix functions. '
    notes = tuple(sentence + '.' for sentence in raw_guidance.strip().rstrip('.').split('. '))
    return ToolSpec(native_functions=NativeFunctionCatalog(owner.package, tuple((name, tuple(members)) for name, members in owner.FUNCTIONS.items())),name=owner.TOOL, function=owner.invoke, required_arguments=required, optional_arguments=optional,
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
            'Preserve source bytes, ordered axes and distinct result semantics.', *notes),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports), producer_ports=(
            Producer('matrix', owner.CONTRACT, (member('path', 'manifest_path'), member('sha256', 'manifest_sha256'))),
            Producer('dataset', 'native_matrix_h5ad.v1', (member('dataset','matrix_path'),member('sha256','matrix_sha256')), lineage_from_port='dataset'),
            *((Producer('fragments', 'native_fragments_file.v1', (member('dataset', 'fragments_path'),)),)
                if owner.package == 'snapatac2' else ()))))


def tool_specs():
    return tuple(tool_spec(owner) for owner in native_matrix.PACKAGES)
