"""One data-only callable ToolSpec, without an AI planning catalog."""
from pathlib import Path
from collections.abc import Mapping
from agent.tools.analysis import native_callable as owner
from .registry import (ToolSpec, ArgumentSpec, ResultContract, DurableToolHooks,
    ScientificAuthorityHooks, SemanticToolSpec, SemanticProducerPortSpec, SemanticPortMember,
    ErrorClassification)
from agent.schemas import ErrorCategory
from .native_delivery import delivery_spec


def classify(exc):
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, 'NATIVE_INVOCATION_FAILED', str(exc))


def tool_spec():
    pub = owner.publication
    return ToolSpec(name=owner.TOOL, function=owner.invoke,
        required_arguments={k: ArgumentSpec(t) for k, t in dict(package=(str,), function_name=(str,),
            inputs=(Mapping,), parameters=(Mapping,), output_dir=(str, Path)).items()},
        optional_arguments={'outputs': ArgumentSpec((Mapping,))},
        result_contract=ResultContract('NativeCallableResult', owner.RESULT_FIELDS), exception_classifier=classify,
        durable_hooks=DurableToolHooks(owner.execute, owner.recover), recovery_policy_version=owner.POLICY,
        scientific_delivery=delivery_spec(owner, 'scientific_native_output'),
        semantic_planning=SemanticToolSpec(consumer_ports=(), producer_ports=(
            SemanticProducerPortSpec('result', owner.CONTRACT, (SemanticPortMember('path', 'manifest_path'), SemanticPortMember('sha256', 'manifest_sha256'))),)),
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND, producer_ports=('result',),
            describe=pub.describe, publication_record=pub.publication_record, reuse_result=pub.reuse_result,
            proof_metadata=pub.proof_metadata, independent_active_publications=True))
