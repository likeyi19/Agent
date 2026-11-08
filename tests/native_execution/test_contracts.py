"""No native execution: preserve invocation authority and operator distinctions."""
import json
import pytest
from agent.tools.data.authority_context import VerificationContext, ScientificProof, snapshot_worker_context, restore_worker_context
from agent.schemas.orchestration import freeze_json_mapping
from agent.orchestration.prior_outputs import merge_context


def test_private_process_context_preserves_frozen_existing_authority():
    context=VerificationContext()
    context.accepted['/publication','digest']=dict(record=freeze_json_mapping({'files':[{'path':'/x'}]}, 'record'),identity='authority')
    proof=ScientificProof('identity','native_test',freeze_json_mapping({'files':['/x']}, 'description'),freeze_json_mapping({'values':[1,2]}, 'metadata'))
    context.proofs[proof.identity]=proof
    context.locations['native_test','/publication']=('identity','digest')
    context.register_execution('native_test','/output','execution')
    restored=restore_worker_context(json.loads(json.dumps(snapshot_worker_context(context))))
    merge_context(context,restored)  # Lists must not conflict with accepted tuples.
    assert restored.accepted==context.accepted and restored.proofs==context.proofs
    assert restored.execution==context.execution


def test_native_resource_errors_are_operational_and_never_retryable():
    from agent.tools._native_process import NativeExecutionError
    from agent.orchestration import build_default_tool_registry
    registry=build_default_tool_registry()
    for code in ('NATIVE_MEMORY_EXHAUSTED','NATIVE_EXECUTION_TIMEOUT','NATIVE_EXECUTION_BUSY','NATIVE_EXECUTION_UNAVAILABLE'):
        error=registry.classify_exception('annotate_scATAC_cell_types',NativeExecutionError(code,'private diagnostic'))
        assert error.code==code and not error.recoverable
        assert error.category.value in ('RESOURCE_ERROR','ENVIRONMENT_ERROR')
        assert 'private diagnostic' not in error.message
