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
    from .registry import (ToolSpec, ResultContract, DurableToolHooks, ScientificAuthorityHooks,
        ArtifactSemanticKind as A, PlanningToolRole, PlanningSourceEligibility as S, SemanticToolSpec,
        SemanticConsumerPortSpec as Consumer, SemanticProducerPortSpec as Producer, _planning_argument as arg,
        _planning_result, _tool_planning, _semantic_member as member, _semantic_request as request,
        _semantic_request_member as rm, _semantic_argument_port as scalar)
    from .native_delivery import delivery_spec
    required = dict(
        input_path=arg((str, Path), 'H5AD with exact cells/features, selected measurement and native layers or feature mask as required.',
            artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        output_dir=arg((str, Path), 'Managed native publication destination.'),
        function_name=arg((str,), 'Explicit public ' + owner.package + ' function.', choices=tuple(owner.FUNCTIONS),
            scientific_parameter=True, conversational_choice=True))
    optional = dict(
        input_sha256=arg((str,), 'Captured exact source checksum.'),
        kwargs=arg((dict,), 'Native kwargs from grounded assignments or JSON; omit unrequested defaults.',
            scientific_parameter=True, conversational_json_object=True, conversational_members=owner.NATIVE_ARGUMENTS),
        count_slot=arg((str,), 'X by default or exact source layer; this becomes native X without normalization.',
            scientific_parameter=True, conversational_exact_string=True),
        matrix_semantics=arg((str,), 'Optional actual source measurement; no declaration is required for native invocation.',
            scientific_parameter=True, conversational_exact_string=True))
    ports = [Consumer('dataset', (member('dataset', 'input_path'),), True,
        request_sources=(request('input_path', rm('dataset', 'input_path')),),
        accepted_upstream_types=('raw_scatac_dataset.v1', 'scatac_matrix_h5ad.v1', 'native_matrix_h5ad.v1')),
        *(scalar(k, required=k in required) for k in (*required, *optional) if k != 'input_path')]
    pub = owner.publication
    return ToolSpec(name=owner.TOOL, function=owner.invoke, required_arguments=required, optional_arguments=optional,
        scientific_delivery=delivery_spec(owner, 'scientific_native_result', h5ad_roles=('result.h5ad',)),
        result_contract=ResultContract('NativeMatrixResult', owner.RESULT_FIELDS, planning_fields={
            k: _planning_result('Exact native publication or result H5AD.', bindable=True, artifact=A.RAW_SCATAC)
            for k in ('manifest_path', 'manifest_sha256', 'matrix_path', 'matrix_sha256')}),
        durable_hooks=DurableToolHooks(owner.execute, owner.recover), exception_classifier=classify,
        recovery_policy_version=owner.POLICY,
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND, producer_ports=('matrix', 'dataset'),
            describe=pub.describe, publication_record=pub.publication_record, reuse_result=pub.reuse_result,
            proof_metadata=pub.proof_metadata, independent_active_publications=True),
        planning=_tool_planning(PlanningToolRole.OPERATION, owner.guidance,
            'Invoke only the named supported native function with explicit kwargs unchanged and unspecified native defaults omitted. '
            'No genome is required for these matrix functions. Preserve source bytes, ordered axes and distinct result semantics.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports), producer_ports=(
            Producer('matrix', owner.CONTRACT, (member('path', 'manifest_path'), member('sha256', 'manifest_sha256'))),
            Producer('dataset', 'native_matrix_h5ad.v1', (member('dataset', 'matrix_path'),), lineage_from_port='dataset'))))


def tool_specs():
    return tuple(tool_spec(owner) for owner in native_matrix.PACKAGES)
