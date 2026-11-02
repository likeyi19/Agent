"""Package-level genomic-table ports; shared execution, authority and delivery."""
from pathlib import Path
from agent.tools.analysis.native_tables import PACKAGES
from .native_matrix_registry import classify


def tool_spec(owner):
    from .registry import (ToolSpec, ResultContract, DurableToolHooks, ScientificAuthorityHooks,
        ArtifactSemanticKind as A, PlanningToolRole, PlanningSourceEligibility as S, SemanticToolSpec,
        SemanticConsumerPortSpec as Consumer, SemanticProducerPortSpec as Producer, _planning_argument as arg,
        _planning_result, _tool_planning, _semantic_member as member, _semantic_request as request,
        _semantic_request_member as rm, _semantic_argument_port as scalar)
    from .native_delivery import delivery_spec
    required = dict(input_path=arg((str, Path), 'Exact native genomic table; ordinary attachment or compatible accepted table.',
        artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        output_dir=arg((str, Path), 'Managed native publication destination.'),
        function_name=arg((str,), 'Explicit native function or CLI subcommand.', choices=tuple(owner.FUNCTIONS),
            scientific_parameter=True, conversational_choice=True))
    optional = dict(input_sha256=arg((str,), 'Captured source checksum.'),
        kwargs=arg((dict,), 'Exact native arguments; input files and output locations are bound separately.',
            scientific_parameter=True, conversational_json_object=True, conversational_members=owner.NATIVE_ARGUMENTS))
    if owner.package == 'MACS3':
        optional.update(control_path=arg((str, Path), 'bdgcmp exact control bedGraph.', artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
            control_sha256=arg((str,), 'Captured control checksum.'),
            signal_semantics=arg((str,), 'Actual input score meaning; p2q requires neg_log10_pvalue; ppois/qpois require count_pileup.', scientific_parameter=True, conversational_exact_string=True),
            control_semantics=arg((str,), 'Actual control score meaning; ppois/qpois require expected_count.', scientific_parameter=True, conversational_exact_string=True),
            assembly=arg((str,), 'bdgcmp explicit common assembly declaration for both tracks; labels alone do not establish it.', scientific_parameter=True, conversational_exact_string=True))
    else:
        optional['seed'] = arg((int,), 'Optional native R random seed for generate_ccans; omission retains native RNG.', scientific_parameter=True)
    ports = [Consumer('dataset', (member('dataset', 'input_path'),), True,
        request_sources=(request('input_path', rm('dataset', 'input_path')),), accepted_upstream_types=('native_genomic_table.v1',))]
    if 'control_path' in optional:
        ports.append(Consumer('control', (member('dataset', 'control_path'),), False,
            request_sources=(request('control_path', rm('dataset', 'control_path')),), accepted_upstream_types=('native_genomic_table.v1',)))
    ports.extend(scalar(k, required=k in required) for k in (*required, *optional) if k not in ('input_path', 'control_path'))
    pub = owner.publication
    return ToolSpec(name=owner.TOOL, function=owner.invoke, required_arguments=required, optional_arguments=optional,
        parameter_prerequisites=owner.parameter_prerequisites, scientific_delivery=delivery_spec(owner, 'scientific_native_result'),
        result_contract=ResultContract('NativeGenomicTableResult', owner.RESULT_FIELDS, planning_fields={
            k: _planning_result('Exact native table publication.', bindable=True, artifact=A.RAW_SCATAC)
            for k in ('manifest_path', 'manifest_sha256', 'table_path', 'table_sha256')}),
        durable_hooks=DurableToolHooks(owner.execute, owner.recover), exception_classifier=classify, recovery_policy_version=owner.POLICY,
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND, producer_ports=('publication', 'table'),
            describe=pub.describe, publication_record=pub.publication_record, reuse_result=pub.reuse_result,
            proof_metadata=pub.proof_metadata, independent_active_publications=True),
        planning=_tool_planning(PlanningToolRole.OPERATION, owner.guidance,
            'Forward explicit native kwargs unchanged and leave omitted defaults to the package. '
            'Accepted table bytes are reusable only when their actual format and measurement match the requested consumer; no hidden upstream execution.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports), producer_ports=(
            Producer('publication', owner.CONTRACT, (member('path', 'manifest_path'), member('sha256', 'manifest_sha256'))),
            Producer('table', 'native_genomic_table.v1', (member('dataset', 'table_path'),), lineage_from_port='dataset'))))


def tool_specs():
    return tuple(tool_spec(p) for p in PACKAGES)
