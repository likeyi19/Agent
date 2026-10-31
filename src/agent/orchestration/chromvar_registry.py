"""One native motif-deviation operation through ordinary Registry source roles."""
from pathlib import Path
from agent.tools.analysis import chromvar as owner


def classify(exception):
    from .registry import ErrorClassification
    from agent.schemas import ErrorCategory
    from agent.schemas.verification_authority import AuthorityError
    if isinstance(exception,AuthorityError):return ErrorClassification(ErrorCategory.VERIFICATION_ERROR,'MOTIF_PUBLICATION_INVALID')
    if isinstance(exception,(ValueError,FileNotFoundError)):return ErrorClassification(ErrorCategory.USER_INPUT_ERROR,'MOTIF_INPUT_OR_RESOURCE_INVALID')
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR,'MOTIF_NATIVE_EXECUTION_FAILED')


def tool_spec():
    from .registry import (ToolSpec,ResultContract,DurableToolHooks,ScientificAuthorityHooks,
        ArtifactSemanticKind as A,PlanningToolRole,PlanningSourceEligibility as S,SemanticToolSpec,
        SemanticConsumerPortSpec as Consumer,SemanticProducerPortSpec as Producer,_planning_argument as arg,
        _planning_result,_tool_planning,_semantic_member as member,_semantic_request as request,
        _semantic_request_member as rm,_semantic_argument_port as scalar)
    required=dict(input_path=arg((str,Path),'Ordinary cell-by-peak AnnData counts; no prior adoption or model required.',
        artifacts=(A.RAW_SCATAC,),source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),output_dir=arg((str,Path),'Managed result destination.'))
    optional=dict(input_sha256=arg((str,),'Exact captured dataset checksum.',source=S.REQUEST_INPUT_ONLY),
        genome_path=arg((str,Path),'Optional task-provided assembly-matched FASTA, instead of a shared genome.',artifacts=(A.DNA_SEQUENCES,),source=S.REQUEST_INPUT_ONLY),
        genome_sha256=arg((str,),'Exact captured task FASTA checksum.',source=S.REQUEST_INPUT_ONLY))
    for name,choices,description in (
        ('matrix_semantics',('peak_counts','fragment_counts','insertion_counts','binary_accessibility'),'Actual selected raw measurement. Cell Ranger cut-site counts are insertion_counts, not fragment overlap. Standard X metadata can supply this; layers need their own declaration. Transformed matrices are ineligible.'),
        ('coordinate_source',('var_columns','feature_names'),'Coordinates from var columns (default chrom/start/end when present), otherwise chrom:start-end IDs. No inferred coordinate convention.'),
        ('coordinate_system',('zero_based_half_open','one_based_closed'),'Actual peak coordinate convention from metadata or declaration; required to extract exact reference sequence.')):
        optional[name]=arg((str,),description,choices=choices,scientific_parameter=True,conversational_choice=True)
    for name,description in (
        ('count_slot','X by default, or the exact AnnData layer name as supplied by the researcher (e.g. raw_counts); layers/exact_name also works. Source counts and axes are never changed.'),
        ('assembly','Actual genome assembly, matching a configured genome or supplied FASTA. Standard uns assembly or a unique var genome value may supply it.'),
        ('genome_resource','Optional exact installed genome label or identity when multiple genomes match. The declared assembly must still match; this never aliases assemblies.'),
        ('feature_chrom_key','Exact chromosome var column, default chrom.'),('feature_start_key','Exact start var column, default start.'),
        ('feature_end_key','Exact end var column, default end.'),('motif_collection','Optional exact installed motif collection label or identity; omit for the unique taxon-compatible collection.')):
        optional[name]=arg((str,),description,scientific_parameter=True,conversational_exact_string=True)
    optional['group_key']=arg((str,),'Optional obs grouping for descriptive means and figure ordering. Vendor clusters are not validated cell types; no grouping or clustering prerequisite.',scientific_parameter=True,observation_source='dataset')
    for name,types,description in (
        ('taxonomy_id',(int,),'Species NCBI taxonomy ID; required for task FASTA, otherwise provided by the selected genome. Never infer taxonomy from cell names.'),
        ('seed',(int,),'Native background random seed, default 20261030; 0 to 2147483647.'),
        ('background_iterations',(int,),'Native matched-background iterations, default 50; execution bound 2 to 1000.'),
        ('p_cutoff',(float,int),'Native motifmatchr scan probability cutoff, default 0.00005. This is not motif-accessibility significance.')):
        optional[name]=arg(types,description,scientific_parameter=True)
    ports=[Consumer('dataset',(member('dataset','input_path'),),True,request_sources=(request('input_path',rm('dataset','input_path')),),accepted_upstream_types=('scatac_matrix_h5ad.v1',)),
        Consumer('genome',(member('path','genome_path'),member('sha256','genome_sha256')),False,
            request_sources=(request('genome_path',rm('path','genome_path'),rm('sha256','genome_sha256')),)),
        scalar('output_dir',required=True),*(scalar(k,required=False) for k in optional if k not in ('genome_path','genome_sha256'))]
    return ToolSpec(name=owner.TOOL,function=owner.compute_motif_accessibility,required_arguments=required,optional_arguments=optional,
        parameter_prerequisites=owner.parameter_prerequisites,
        result_contract=ResultContract('MotifAccessibilityResult',owner.RESULT_FIELDS,planning_fields={k:_planning_result(
            'Exact native cell-by-motif result; deviations and z-scores are distinct, never peak counts.',bindable=True,artifact=A.MOTIF_ACCESSIBILITY)
            for k in owner.RESULT_FIELDS if k.endswith(('_path','_sha256'))}),
        durable_hooks=DurableToolHooks(owner.execute,owner.recover),exception_classifier=classify,recovery_policy_version=owner.POLICY,
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND,producer_ports=('motifs',),describe=owner.describe,
            publication_record=owner.publication_record,reuse_result=owner.reuse_result,proof_metadata=owner.proof_metadata,independent_active_publications=True),
        planning=_tool_planning(PlanningToolRole.OPERATION,'Examine TF-associated motif accessibility across cells and optional supplied groups using native chromVAR and motifmatchr.',
            'Consumes ordinary dense/CSR/CSC H5AD raw peak counts, including insertion/cut-site counts, fragment counts and declared binary accessibility. No EpiZoo, feature adoption, upstream pipeline, biological replicate or clustering prerequisite.',
            'Requires exact genomic coordinates/convention, assembly-matched genome and taxon-appropriate JASPAR CORE motifs. Use installed resources or a task FASTA. Ask only for missing scientific declarations. Native zero-total peak filtering retains original mappings; zero-total cells retain undefined values.',
            'Publishes per-cell bias-corrected deviations and separate z-scores, matched backgrounds, motif/GC evidence, descriptive summaries, a heatmap and downloads. This is not peak-set motif enrichment, TF expression/binding, causality or replicate-level condition testing.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(Producer('motifs','motif_accessibility.v1',
            (member('path','manifest_path'),member('sha256','manifest_sha256'))),)))
