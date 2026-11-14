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


def test_scoped_library_crosses_worker_boundary_without_global_mutation(tmp_path, monkeypatch):
    import os
    from agent.application.shared_resources import ScientificResourceLibrary, scientific_resource_scope
    from agent.tools._native_process import worker_environment
    monkeypatch.setenv('AGENT_SCIENTIFIC_RESOURCE_ROOT', str(tmp_path/'fallback'))
    with scientific_resource_scope(ScientificResourceLibrary(tmp_path/'configured')):
        assert worker_environment()['AGENT_SCIENTIFIC_RESOURCE_ROOT'] == str(tmp_path/'configured')
        assert os.environ['AGENT_SCIENTIFIC_RESOURCE_ROOT'] == str(tmp_path/'fallback')
    assert worker_environment()['AGENT_SCIENTIFIC_RESOURCE_ROOT'] == str(tmp_path/'fallback')


def test_missing_library_is_environment_error_and_matrix_errors_remain_input_errors(monkeypatch):
    from agent.application.shared_resources import scientific_resource_scope
    from agent.tools.analysis.sequence_prediction import library
    from agent.tools._native_process import NativeExecutionError
    from agent.orchestration import build_default_tool_registry
    from agent.orchestration.registry import _classify_tool_exception
    monkeypatch.delenv('AGENT_SCIENTIFIC_RESOURCE_ROOT', raising=False)
    with scientific_resource_scope(None), pytest.raises(NativeExecutionError) as caught:
        library()
    registry = build_default_tool_registry()
    error = registry.classify_exception('run_snapatac2', caught.value)
    receipt = _classify_tool_exception(caught.value)
    assert error.code == receipt.code == 'NATIVE_EXECUTION_UNAVAILABLE'
    assert error.category.value == receipt.category.value == 'ENVIRONMENT_ERROR'
    assert not error.recoverable
    invalid = registry.classify_exception('run_snapatac2', ValueError('Invalid matrix'))
    assert invalid.code == 'NATIVE_MATRIX_INPUT_INVALID'
    assert invalid.category.value == 'USER_INPUT_ERROR'


def test_native_cpu_default_respects_ancestor_quota_and_single_operation(monkeypatch):
    from agent.tools import _native_process as owner
    monkeypatch.setattr(owner,'_resource_limits',lambda:(8*1024**3,8*1024**3,1))
    monkeypatch.delenv('AGENT_NATIVE_CPU_COUNT',raising=False)
    monkeypatch.setenv('AGENT_NATIVE_MEMORY_BYTES',str(4*1024**3))
    assert owner.native_cpu_count()==1 and owner.resource_capacity()==1
    monkeypatch.setenv('AGENT_NATIVE_CPU_COUNT','2')
    with pytest.raises(owner.NativeExecutionError):owner.native_cpu_count()


@pytest.mark.parametrize('value',[None,'0','invalid'])
def test_invalid_branch_resource_budget_is_an_operational_error(monkeypatch,value):
    from agent.tools import _native_process as owner
    monkeypatch.delenv('AGENT_NATIVE_MEMORY_BYTES',raising=False)
    if value is not None:monkeypatch.setenv('AGENT_NATIVE_MEMORY_BYTES',value)
    with pytest.raises(owner.NativeExecutionError) as error:owner.resource_capacity()
    assert error.value.code=='NATIVE_EXECUTION_UNAVAILABLE'
