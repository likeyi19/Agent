"""Sequence-bin enrichment through the existing scientific owners."""
from pathlib import Path
from agent.tools.analysis import monalisa as owner
from .native_matrix_registry import classify


def tool_spec():
    from .registry import (NativeFunctionCatalog, ToolSpec, ResultContract, DurableToolHooks, ScientificAuthorityHooks,
        ArtifactSemanticKind as A, PlanningToolRole, PlanningSourceEligibility as S, SemanticToolSpec,
        SemanticConsumerPortSpec as Consumer, SemanticProducerPortSpec as Producer, _planning_argument as arg,
        _planning_result, _tool_planning, _semantic_member as member, _semantic_request as request,
        _semantic_request_member as rm, _semantic_argument_port as scalar)
    from .native_delivery import delivery_spec
    required = dict(input_path=arg((str, Path), 'H5AD ordered genomic regions in var. Counts/cells are not analyzed. Supply a region bin field or exact ordered bins table.',
        artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        output_dir=arg((str, Path), 'Managed native publication destination.'),
        function_name=arg((str,), 'Native monaLisa function.', choices=tuple(owner.FUNCTIONS), scientific_parameter=True, conversational_choice=True, planner_choice=True))
    optional = dict(input_sha256=arg((str,), 'Exact source checksum.'),
        kwargs=arg((dict,), 'Native ordinary arguments; omitted values retain native defaults.', scientific_parameter=True, conversational_json_object=True, conversational_members=owner.NATIVE_ARGUMENTS),
        bins_path=arg((str, Path), 'Optional TSV feature_id and bin, exactly matching ordered H5AD regions. Alternative to bin_key.', artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        bins_sha256=arg((str,), 'Exact bin table checksum.'),
        genome_path=arg((str, Path), 'Optional assembly-matched genome FASTA instead of installed genome.', artifacts=(A.DNA_SEQUENCES,)),
        genome_sha256=arg((str,), 'Exact genome checksum.'))
    for name, description in (
        ('bin_key', 'Exact var field containing supplied sequence bins; not obs cell groups. Alternative to bins_path.'),
        ('assembly', 'Actual peak genome assembly; source metadata may supply it.'),
        ('genome_resource', 'Installed matching genome identity if ambiguous.'),
        ('motif_collection', 'Installed taxon-compatible motif collection if ambiguous.'),
        ('feature_chrom_key', 'Chromosome var field, default chrom.'),
        ('feature_start_key', 'Start var field, default start.'),
        ('feature_end_key', 'End var field, default end.')):
        optional[name] = arg((str,), description, scientific_parameter=True, conversational_exact_string=True)
    optional['taxonomy_id'] = arg((int,), 'Taxonomy ID for a supplied genome FASTA; otherwise installed genome supplies it.', scientific_parameter=True)
    for name, choices in [('coordinate_source', ('var_columns', 'feature_names')), ('coordinate_system', ('zero_based_half_open', 'one_based_closed'))]:
        optional[name] = arg((str,), 'Actual genomic coordinate declaration, never inferred from filenames.', choices=choices, scientific_parameter=True, conversational_choice=True)
    from .native_parameters import scalar_port
    ports = [Consumer('dataset', (member('dataset', 'input_path'),), True,
        request_sources=(request('input_path', rm('dataset', 'input_path')),), accepted_upstream_types=('scatac_matrix_h5ad.v1', 'native_matrix_h5ad.v1')),
        Consumer('bins', (member('dataset', 'bins_path'),), False, request_sources=(request('bins_path', rm('dataset', 'bins_path')),)),
        Consumer('genome', (member('path', 'genome_path'), member('sha256', 'genome_sha256')), False,
            request_sources=(request('genome_path', rm('path', 'genome_path'), rm('sha256', 'genome_sha256')),))]
    for key, spec in {**required, **optional}.items():
        if key not in ('input_path', 'bins_path', 'genome_path', 'genome_sha256'):
            ports.append(scalar_port(owner.TOOL, tuple(owner.FUNCTIONS), key, required=key in required)
                if spec.planning.scientific_parameter else scalar(key, required=key in required))
    pub = owner.publication
    return ToolSpec(native_functions=NativeFunctionCatalog(owner.package, tuple(owner.FUNCTIONS.items())), name=owner.TOOL, function=owner.invoke,
        required_arguments=required, optional_arguments=optional, parameter_prerequisites=owner.parameter_prerequisites,
        scientific_delivery=delivery_spec(owner, 'scientific_native_result'),
        result_contract=ResultContract('NativeMonaLisaResult', owner.RESULT_FIELDS, planning_fields={k: _planning_result('Exact native enrichment publication.', bindable=True, artifact=A.RAW_SCATAC)
            for k in ('manifest_path', 'manifest_sha256', 'table_path', 'table_sha256')}),
        durable_hooks=DurableToolHooks(owner.execute, owner.recover), exception_classifier=classify, recovery_policy_version=owner.POLICY,
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND, producer_ports=('publication', 'table'), describe=pub.describe,
            publication_record=pub.publication_record, reuse_result=pub.reuse_result, proof_metadata=pub.proof_metadata, independent_active_publications=True),
        planning=_tool_planning(PlanningToolRole.OPERATION, owner.guidance,
            'Requires actual genomic coordinates, matching genome, motifs and supplied per-region bins. Never infer bins, expression or biological truth. Native sequence-composition correction and tests remain upstream. Genome-background sampling is unsupported. No automatic figure.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports), producer_ports=(
            Producer('publication', owner.CONTRACT, (member('path', 'manifest_path'), member('sha256', 'manifest_sha256'))),
            Producer('table', 'native_genomic_table.v1', (member('dataset', 'table_path'),)))))
