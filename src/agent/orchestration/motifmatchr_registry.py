"""Fixed native motif function dispatch through the existing execution contracts."""
from pathlib import Path
from agent.tools.analysis import motifmatchr as owner


def classify(exception):
    from .registry import ErrorClassification
    from agent.schemas import ErrorCategory
    from agent.schemas.verification_authority import AuthorityError
    if isinstance(exception,owner.NativeInvocationError):
        return ErrorClassification(ErrorCategory.USER_INPUT_ERROR,'NATIVE_FUNCTION_REJECTED',str(exception))
    if isinstance(exception,AuthorityError):
        return ErrorClassification(ErrorCategory.VERIFICATION_ERROR,'NATIVE_FUNCTION_PUBLICATION_INVALID')
    if isinstance(exception,(ValueError,FileNotFoundError)):
        return ErrorClassification(ErrorCategory.USER_INPUT_ERROR,'NATIVE_FUNCTION_INPUT_INVALID')
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR,'NATIVE_FUNCTION_EXECUTION_FAILED')


def tool_spec():
    from .registry import (ToolSpec,ResultContract,DurableToolHooks,ScientificAuthorityHooks,
        ArtifactSemanticKind as A,PlanningToolRole,PlanningSourceEligibility as S,SemanticToolSpec,
        SemanticConsumerPortSpec as Consumer,SemanticProducerPortSpec as Producer,_planning_argument as arg,
        _planning_result,_tool_planning,_semantic_member as member,_semantic_request as request,
        _semantic_request_member as rm,_semantic_argument_port as scalar)
    required = dict(input_path=arg((str,Path),'Ordinary H5AD genomic peaks in var; counts, cells and X measurement are not used.',
        artifacts=(A.RAW_SCATAC,),source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),output_dir=arg((str,Path),'Managed native result destination.'),
        function_name=arg((str,),'Explicitly requested motifmatchr::matchMotifs function.',
            choices=('matchMotifs',),scientific_parameter=True,conversational_choice=True))
    optional = dict(input_sha256=arg((str,),'Exact captured source checksum.'),
        genome_path=arg((str,Path),'Optional authorized assembly-matched FASTA instead of an installed genome.',artifacts=(A.DNA_SEQUENCES,)),
        genome_sha256=arg((str,),'Exact captured task FASTA checksum.'),
        kwargs=arg((dict,),'Explicit native arguments, captured from grounded assignments or supplied JSON. Omit absent optional arguments. No paths, scripts or input objects; no output substitution.',
            scientific_parameter=True,conversational_json_object=True,conversational_members=owner.NATIVE_ARGUMENTS))
    for name,description in (
        ('assembly','Actual peak genome assembly; source uns assembly or unique var genome may supply it.'),
        ('genome_resource','Exact installed genome label/identity if ambiguous; assembly must match.'),
        ('motif_collection','Exact installed taxon-compatible motif collection label/identity if ambiguous.'),
        ('feature_chrom_key','Exact chromosome var column, default chrom.'),
        ('feature_start_key','Exact start var column, default start.'),
        ('feature_end_key','Exact end var column, default end.')):
        optional[name] = arg((str,),description,scientific_parameter=True,conversational_exact_string=True)
    optional['taxonomy_id'] = arg((int,),'NCBI taxonomy ID for a task FASTA; otherwise selected genome supplies it.',scientific_parameter=True)
    optional['coordinate_source'] = arg((str,),'var columns when available, otherwise feature names; preserves original ordered peaks.',
        choices=('var_columns','feature_names'),scientific_parameter=True,conversational_choice=True)
    optional['coordinate_system'] = arg((str,),'Actual peak coordinate convention, required from metadata or researcher.',
        choices=('zero_based_half_open','one_based_closed'),scientific_parameter=True,conversational_choice=True)
    ports = [Consumer('dataset',(member('dataset','input_path'),),True,
        request_sources=(request('input_path',rm('dataset','input_path')),),accepted_upstream_types=('scatac_matrix_h5ad.v1',)),
        Consumer('genome',(member('path','genome_path'),member('sha256','genome_sha256')),False,
            request_sources=(request('genome_path',rm('path','genome_path'),rm('sha256','genome_sha256')),)),
        *(scalar(k,required=k in required) for k in (*required,*optional) if k not in ('input_path','genome_path','genome_sha256'))]
    from .native_delivery import delivery_spec
    return ToolSpec(scientific_delivery=delivery_spec(owner,'scientific_motif_matches'),name=owner.TOOL,function=owner.run_motifmatchr,required_arguments=required,optional_arguments=optional,
        parameter_prerequisites=owner.parameter_prerequisites,
        result_contract=ResultContract('NativeMotifMatchesResult',owner.RESULT_FIELDS,planning_fields={k:_planning_result(
            'Exact native boolean peak-by-motif matches; no cell accessibility analysis.',bindable=True,artifact=A.MOTIF_MATCHES)
            for k in owner.RESULT_FIELDS if k.endswith(('_path','_sha256'))}),
        durable_hooks=DurableToolHooks(owner.execute,owner.recover),exception_classifier=classify,recovery_policy_version=owner.POLICY,
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND,producer_ports=('matches',),describe=owner.describe,
            publication_record=owner.publication_record,reuse_result=owner.reuse_result,proof_metadata=owner.proof_metadata,independent_active_publications=True),
        planning=_tool_planning(PlanningToolRole.OPERATION,'Invoke the explicitly requested native motifmatchr::matchMotifs on ordinary H5AD genomic peak coordinates.',
            'Counts, count measurement, clustering, positive-count filtering, backgrounds and deviations are not prerequisites. All peaks stay in original order. Requires actual coordinate convention, assembly-matched genome and taxon-compatible motifs; unique installed resources are reused.',
            'Forward the captured kwargs mapping unchanged. Only matchMotifs and native matches output are supported. Do not substitute compute_motif_accessibility or another output for an unsupported function/output request.',
            'Publishes a sparse boolean peak-by-motif matrix in Matrix Market and NPZ, ordered peak/motif metadata and native completion. No automatic visualization, deviations or other analysis.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(Producer('matches','native_motif_matches.v1',
            (member('path','manifest_path'),member('sha256','manifest_sha256'))),)))
