"""Native Chromap file ports, alongside other package-level native tools."""
from pathlib import Path
from agent.tools.data import chromap_native as owner


def tool_spec():
    from .registry import (ToolSpec, NativeFunctionCatalog, ResultContract, DurableToolHooks, ScientificAuthorityHooks,
        ArtifactSemanticKind as A, PlanningToolRole, PlanningSourceEligibility as S, SemanticToolSpec,
        SemanticConsumerPortSpec as C, SemanticProducerPortSpec as P, _planning_argument as arg,
        _planning_result as result, _tool_planning, _semantic_member as member, _semantic_request as request,
        _semantic_request_member as rm, _semantic_argument_port as scalar)
    from .native_parameters import scalar_port
    from .native_matrix_registry import classify
    from .native_delivery import delivery_spec
    required = dict(reference_path=arg((str, Path), 'Exact native FASTA reference.', artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        output_dir=arg((str, Path), 'Managed native publication destination.'),
        function_name=arg((str,), 'Native CLI operation: index or paired-end BED mapping.', choices=tuple(owner.FUNCTIONS),
            scientific_parameter=True, conversational_choice=True, planner_choice=True))
    optional = {field: arg((str, Path, list), 'Exact native ' + port + ' input; read/barcode lists are lanes of one declared library.',
        artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT) for port, field in owner.PORTS.items() if port != 'reference'}
    optional['kwargs'] = arg((dict,), 'Explicit native flags, forwarded unchanged; omitted flags retain native defaults.',
        scientific_parameter=True, conversational_json_object=True, conversational_members=owner.NATIVE_ARGUMENTS)
    ports = tuple(C(port, (member('dataset', field),), port == 'reference',
        request_sources=(request(field, rm('dataset', field)),),
        accepted_upstream_types=('native_chromap_index.v1',) if port == 'index' else ()) for port, field in owner.PORTS.items())
    pub = owner.publication
    return ToolSpec(name=owner.TOOL, function=owner.invoke, required_arguments=required, optional_arguments=optional,
        native_functions=NativeFunctionCatalog('Chromap', tuple(owner.FUNCTIONS.items())),
        result_contract=ResultContract('NativeChromapResult', owner.RESULT_FIELDS, planning_fields={key: result(
            'Exact native publication/file; null means unavailable for this operation.', bindable=True, artifact=A.RAW_SCATAC)
            for key in ('manifest_path', 'manifest_sha256', 'fragments_path', 'fragments_sha256', 'index_path', 'index_sha256')}),
        durable_hooks=DurableToolHooks(owner.execute, owner.recover), exception_classifier=classify, recovery_policy_version=owner.POLICY,
        scientific_delivery=delivery_spec(owner, 'scientific_native_result'),
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND, producer_ports=('fragments', 'index'), describe=pub.describe,
            publication_record=pub.publication_record, reuse_result=pub.reuse_result, proof_metadata=pub.proof_metadata,
            independent_active_publications=True),
        planning=_tool_planning(PlanningToolRole.OPERATION,
            'Native Chromap reference indexing and paired FASTQ mapping with barcode correction and duplicate handling.',
            'Group lanes/chunks only from explicit one-library roles. One native map call receives matching ordered read1/read2/barcode lists. '
            'Native ATAC preset, shift, deduplication and whitelist settings remain explicit choices; no Agent canonicalization or replay. '
            'For supported native function requests forward supplied flags and do not rewrite parameters.'),
        semantic_planning=SemanticToolSpec(consumer_ports=ports + (scalar('output_dir', required=True),
            scalar_port(owner.TOOL, tuple(owner.FUNCTIONS), 'function_name', required=True),
            scalar_port(owner.TOOL, tuple(owner.FUNCTIONS), 'kwargs', required=False)), producer_ports=(
            P('fragments', 'native_fragments_file.v1', (member('dataset', 'fragments_path'),)),
            P('index', 'native_chromap_index.v1', (member('dataset', 'index_path'),)))))
