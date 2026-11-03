"""Focused native ArchR objects, with the ordinary accepted publication port."""
from pathlib import Path
from agent.tools.analysis import archr as owner
from .native_matrix_registry import classify


def tool_spec():
    from .registry import (NativeFunctionCatalog, ToolSpec, ResultContract, DurableToolHooks, ScientificAuthorityHooks,
        ArtifactSemanticKind as A, PlanningToolRole, PlanningSourceEligibility as S, SemanticToolSpec,
        SemanticConsumerPortSpec as Consumer, SemanticProducerPortSpec as Producer, _planning_argument as arg,
        _planning_result, _tool_planning, _semantic_member as member, _semantic_request as request,
        _semantic_request_member as rm, _semantic_argument_port as scalar)
    from .native_delivery import delivery_spec
    required = dict(input_path=arg((str, Path), 'createArrowFiles: sorted five-column fragments. Other functions: exact accepted ArchR publication manifest, never an uploaded RDS/H5AD/ZIP.',
        artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        output_dir=arg((str, Path), 'Managed native publication destination.'),
        function_name=arg((str,), 'Native ArchR function.', choices=tuple(owner.FUNCTIONS), scientific_parameter=True, conversational_choice=True, planner_choice=True))
    required['input_sha256'] = arg((str,), 'Exact captured source checksum, including accepted project reuse.')
    optional = dict(kwargs=arg((dict,), 'Native arguments of the selected function. Omitted scientific parameters retain native defaults.', scientific_parameter=True, conversational_json_object=True, conversational_members=owner.NATIVE_ARGUMENTS),
        second_path=arg((str, Path), 'Optional second fragment sample for createArrowFiles only.', artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        second_sha256=arg((str,), 'Exact second source checksum.'),
        fragment_endpoints=arg((str,), 'createArrowFiles: confirm fragments already have Tn5-adjusted endpoints, as native fragments require. No shift is inferred or applied.', choices=('tn5_adjusted','unadjusted'), scientific_parameter=True, conversational_choice=True))
    for key, description in [('sample_name','First native sampleNames entry for createArrowFiles.'),
        ('second_sample_name','Second native sampleNames entry, distinct from the first.'),
        ('assembly','Actual fragment assembly; current qualified native annotation hg38/GRCh38 only.'),
        ('coordinate_system','Actual fragment coordinates; current entrance requires zero_based_half_open.')]:
        optional[key] = arg((str,), description, scientific_parameter=True, conversational_exact_string=True)
    from .native_parameters import scalar_port
    ports = [Consumer('dataset', (member('path','input_path'),member('sha256','input_sha256')), True,
        request_sources=(request('input_path',rm('path','input_path'),rm('sha256','input_sha256')),), accepted_upstream_types=(owner.CONTRACT,)),
        Consumer('second', (member('dataset','second_path'),), False, request_sources=(request('second_path',rm('dataset','second_path')),))]
    for key, spec in {**required, **optional}.items():
        if key not in ('input_path','input_sha256','second_path'):
            ports.append(scalar_port(owner.TOOL,tuple(owner.FUNCTIONS),key,required=key in required)
                if spec.planning.scientific_parameter else scalar(key,required=key in required))
    pub = owner.publication
    return ToolSpec(native_functions=NativeFunctionCatalog(owner.package,tuple(owner.FUNCTIONS.items())),name=owner.TOOL,function=owner.invoke,
        required_arguments=required,optional_arguments=optional,parameter_prerequisites=owner.parameter_prerequisites,
        scientific_delivery=delivery_spec(owner,'scientific_native_result'),
        result_contract=ResultContract('NativeArchRResult',owner.RESULT_FIELDS,planning_fields={k:_planning_result('Exact native ArchR publication/snapshot.',bindable=True,artifact=A.RAW_SCATAC)
            for k in ('manifest_path','manifest_sha256','object_path','object_sha256','table_path','table_sha256')}),
        durable_hooks=DurableToolHooks(owner.execute,owner.recover),exception_classifier=classify,recovery_policy_version=owner.POLICY,
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND,producer_ports=('publication',),describe=pub.describe,
            publication_record=pub.publication_record,reuse_result=pub.reuse_result,proof_metadata=pub.proof_metadata,independent_active_publications=True),
        planning=_tool_planning(PlanningToolRole.OPERATION,owner.guidance,
            'Task dependency: createArrowFiles produces accepted ArrowFiles; ArchRProject consumes that publication and saves a native project; getMatrixFromProject retrieves its existing GeneScoreMatrix by native default. getAvailableMatrices inspects an accepted project.',
            'Only createArrowFiles takes sample names, assembly and fragment declarations. Later functions retain them from exact accepted publications.',
            'Gene scores are accessibility estimates, never measured expression. No addPeak2GeneLinks or addGeneScoreMatrix support, arbitrary uploaded objects or substitute algorithm. Two fragment sample inputs supported; missing scientific declarations require clarification.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
            Producer('publication',owner.CONTRACT,(member('path','manifest_path'),member('sha256','manifest_sha256'))),)))
