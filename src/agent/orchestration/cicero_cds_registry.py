"""Cicero scientific objects through existing Registry and accepted-source ports."""
from pathlib import Path
from agent.tools.analysis import cicero_cds as owner
from .native_matrix_registry import classify


def tool_spec():
    from .registry import (NativeFunctionCatalog, ToolSpec, ResultContract, DurableToolHooks, ScientificAuthorityHooks,
        ArtifactSemanticKind as A, PlanningToolRole, PlanningSourceEligibility as S, SemanticToolSpec,
        SemanticConsumerPortSpec as Consumer, SemanticProducerPortSpec as Producer, _planning_argument as arg,
        _planning_result, _tool_planning, _semantic_member as member, _semantic_request as request,
        _semantic_request_member as rm, _semantic_argument_port as scalar)
    from .native_delivery import delivery_spec
    required = dict(input_path=arg((str, Path), 'make_cicero_cds: H5AD regions by named features and supplied obsm coordinates; later functions: safe Cicero CDS exchange ZIP.', artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        output_dir=arg((str, Path), 'Managed native publication destination.'),
        function_name=arg((str,), 'Native Cicero CDS function.', choices=tuple(owner.FUNCTIONS), scientific_parameter=True, conversational_choice=True, planner_choice=True))
    optional = dict(input_sha256=arg((str,), 'Exact source checksum.'),
        kwargs=arg((dict,), 'Native ordinary arguments; omit native defaults.', scientific_parameter=True, conversational_json_object=True, conversational_members=owner.NATIVE_ARGUMENTS),
        count_slot=arg((str,), 'make_cicero_cds selected X or exact layer; no automatic preprocessing.', scientific_parameter=True, conversational_exact_string=True),
        reduced_coordinates_key=arg((str,), 'make_cicero_cds exact existing obsm field in the same H5AD; no embedding is inferred.', scientific_parameter=True, conversational_exact_string=True),
        genomic_coords_path=arg((str, Path), 'Two-column chromosome lengths without header. Required for run_cicero; optional for estimate_distance_parameter, whose omitted native default is human.hg19.genome, not input reference qualification.', artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        genomic_coords_sha256=arg((str,), 'Exact chromosome-length source checksum.'),
        seed=arg((int,), 'Optional native R RNG seed; omitted retains native stochastic behavior.', scientific_parameter=True))
    from .native_parameters import scalar_port
    base_scalar = scalar
    scalar = lambda k, **kw: scalar_port(owner.TOOL, tuple(owner.FUNCTIONS), k, **kw) if ({**required, **optional}[k].planning.scientific_parameter) else base_scalar(k, **kw)
    ports = [Consumer('dataset', (member('dataset','input_path'),), True,
        request_sources=(request('input_path',rm('dataset','input_path')),), accepted_upstream_types=('raw_scatac_dataset.v1','scatac_matrix_h5ad.v1','native_matrix_h5ad.v1','cicero_cds_exchange.v1')),
        Consumer('genomic_coords',(member('dataset','genomic_coords_path'),),False,
            request_sources=(request('genomic_coords_path',rm('dataset','genomic_coords_path')),)),
        *(scalar(k,required=k in required) for k in (*required,*optional) if k not in ('input_path','genomic_coords_path'))]
    pub = owner.publication
    return ToolSpec(native_functions=NativeFunctionCatalog('cicero', tuple((name, tuple(members)) for name, members in owner.FUNCTIONS.items())),name=owner.TOOL,function=owner.invoke,required_arguments=required,optional_arguments=optional,
        parameter_prerequisites=owner.parameter_prerequisites,scientific_delivery=delivery_spec(owner,'scientific_native_result'),
        result_contract=ResultContract('NativeCiceroCDSResult',owner.RESULT_FIELDS,planning_fields={k:_planning_result('Exact native CDS exchange or connection table.',bindable=True,artifact=A.RAW_SCATAC) for k in ('manifest_path','manifest_sha256','cds_path','cds_sha256','table_path','table_sha256')}),
        durable_hooks=DurableToolHooks(owner.execute,owner.recover),exception_classifier=classify,recovery_policy_version=owner.POLICY,
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND,producer_ports=('publication','cds','table'),describe=pub.describe,
            publication_record=pub.publication_record,reuse_result=pub.reuse_result,proof_metadata=pub.proof_metadata,independent_active_publications=True),
        planning=_tool_planning(PlanningToolRole.OPERATION,
            'Cicero make_cicero_cds constructs native overlapping aggregates from an H5AD selected matrix and exact existing obsm coordinates. Native region names chr_start_end; no automatic reduction. run_cicero and estimate_distance_parameter consume the safe CDS exchange and actual chromosome lengths. No arbitrary RDS or complete Monocle serializer.',
            'Native functions own numerical validation. Aggregation members appear only with return_agg_info=true. run_cicero requires chromosome lengths; estimate_distance_parameter may use its bundled human hg19 default, not inferred input compatibility. No upstream rerun.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
            Producer('publication',owner.CONTRACT,(member('path','manifest_path'),member('sha256','manifest_sha256'))),
            Producer('cds','cicero_cds_exchange.v1',(member('dataset','cds_path'),)),
            Producer('table','native_genomic_table.v1',(member('dataset','table_path'),)))))
