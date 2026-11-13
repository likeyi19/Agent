"""Package-level chromVAR dispatch, through existing typed H5AD source ports."""
from pathlib import Path
from agent.tools.analysis import chromvar_native as owner


def classify(exception):
    from .registry import ErrorClassification
    from agent.schemas import ErrorCategory
    from agent.schemas.verification_authority import AuthorityError
    if isinstance(exception,owner.NativeInvocationError):
        return ErrorClassification(ErrorCategory.USER_INPUT_ERROR,'NATIVE_FUNCTION_REJECTED',str(exception))
    if isinstance(exception,AuthorityError):
        return ErrorClassification(ErrorCategory.VERIFICATION_ERROR,'NATIVE_FUNCTION_PUBLICATION_INVALID')
    if isinstance(exception,(ValueError,KeyError,FileNotFoundError)):
        return ErrorClassification(ErrorCategory.USER_INPUT_ERROR,'CHROMVAR_FUNCTION_INPUT_INVALID')
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR,'NATIVE_FUNCTION_EXECUTION_FAILED')


def tool_spec():
    from .registry import (NativeFunctionCatalog, ToolSpec,ResultContract,DurableToolHooks,ScientificAuthorityHooks,
        ArtifactSemanticKind as A,PlanningToolRole,PlanningSourceEligibility as S,SemanticToolSpec,
        SemanticConsumerPortSpec as Consumer,SemanticProducerPortSpec as Producer,_planning_argument as arg,
        _planning_result,_tool_planning,_semantic_member as member,_semantic_request as request,
        _semantic_request_member as rm,_semantic_argument_port as scalar)
    from .native_delivery import delivery_spec
    required=dict(input_path=arg((str,Path),'Ordinary H5AD counts, or chromVAR deviations X with identified z_scores for computeVariability.',
        artifacts=(A.RAW_SCATAC,A.MOTIF_ACCESSIBILITY),source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        output_dir=arg((str,Path),'Managed native result destination.'),
        function_name=arg((str,),'Explicit native chromVAR function.',choices=tuple(owner.FUNCTIONS),scientific_parameter=True,conversational_choice=True,planner_choice=True))
    optional=dict(bias_key=arg((str,), 'getBackgroundPeaks: exact var column containing actual numeric bias, default bias. No inferred GC bias.', scientific_parameter=True, conversational_exact_string=True),
        input_sha256=arg((str,),'Captured exact source checksum.'),
        kwargs=arg((dict,),'Native kwargs from grounded assignments or JSON; omit unrequested defaults.',
            scientific_parameter=True,conversational_json_object=True,conversational_members=owner.NATIVE_ARGUMENTS),
        count_slot=arg((str,),'Count functions: X by default or exact layer name. Not used for deviation functions.',scientific_parameter=True,conversational_exact_string=True),
        matrix_semantics=arg((str,),'Optional supplied measurement; X metadata may supply it. Not for deviation functions.',
            scientific_parameter=True,conversational_exact_string=True),
        group_key=arg((str,),'Expectations/differential deviations/variability: exact obs group field in original cell order.',scientific_parameter=True,observation_source='dataset'),
        seed=arg((int,),'computeVariability/getBackgroundPeaks: optional R/BiocParallel seed; omission is not reproducible.',scientific_parameter=True))
    from .native_parameters import scalar_port
    base_scalar = scalar
    scalar = lambda k, **kw: scalar_port(owner.TOOL, tuple(owner.FUNCTIONS), k, **kw) if ({**required, **optional}[k].planning.scientific_parameter) else base_scalar(k, **kw)
    ports=[Consumer('dataset',(member('dataset','input_path'),),True,
        request_sources=(request('input_path',rm('dataset','input_path')),),
        accepted_upstream_types=('scatac_matrix_h5ad.v1','hierarchical_annotation_h5ad.v1','chromvar_scores_h5ad.v1','native_matrix_h5ad.v1','raw_scatac_dataset.v1')),
        *(scalar(k,required=k in required) for k in (*required,*optional) if k!='input_path')]
    pub=owner.publication
    return ToolSpec(native_functions=NativeFunctionCatalog('chromVAR', tuple((name, tuple(members)) for name, members in owner.FUNCTIONS.items())),name=owner.TOOL,function=owner.run_chromvar,required_arguments=required,optional_arguments=optional,
        scientific_delivery=delivery_spec(owner,'scientific_native_result'),
        result_contract=ResultContract('NativeChromvarResult',owner.RESULT_FIELDS,planning_fields={k:_planning_result(
            'Native statistics with ordered source identities.',bindable=True,artifact=A.MOTIF_ACCESSIBILITY)
            for k in owner.RESULT_FIELDS if k.endswith(('_path','_sha256'))}),
        durable_hooks=DurableToolHooks(owner.execute,owner.recover),exception_classifier=classify,recovery_policy_version=owner.POLICY,
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND,producer_ports=('statistics',),describe=pub.describe,
            publication_record=pub.publication_record,reuse_result=pub.reuse_result,proof_metadata=pub.proof_metadata,independent_active_publications=True),
        planning=_tool_planning(PlanningToolRole.OPERATION,
            'Invoke explicitly requested chromVAR::getBackgroundPeaks, computeExpectations, getFragmentsPerPeak, getFragmentsPerSample, getTotalFragments, computeVariability, getSampleCorrelation, getSampleDistance, differentialDeviations or differentialVariability.',
            'Matrix functions forward supplied numeric values; no measurement declaration, genome, motifs or upstream analysis is required. computeExpectations accepts norm and obs group_key; count accessors have no kwargs.',
            'computeVariability consumes chromVAR deviations X and z_scores, including accepted scores. It accepts bootstrap_error, bootstrap_samples, bootstrap_quantiles and na.rm, plus optional seed. Reuse existing scores without upstream reruns.',
            'getBackgroundPeaks uses the actual supplied var bias column (bias_key, default bias), forwards niterations/w/bs and optional seed, and publishes native 1-based background indices per ordered original feature and iteration. Never invent GC or sequence bias. ',
            'Sample correlation/distance publish ordered cell-by-cell TSV (cell_i columns map to cells.tsv); totals are scalar. Deviation functions consume X deviations and z_scores; comparisons use supplied groups, not replicate-level accessibility inference. Publish ordered TSV and native provenance. Preserve explicit values and omitted defaults. Independent native calls; never substitute unsupported functions or arguments.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(Producer('statistics','native_chromvar_statistics.v1',
            (member('path','manifest_path'),member('sha256','manifest_sha256'))),)))
