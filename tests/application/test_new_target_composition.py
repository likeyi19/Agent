from types import SimpleNamespace as N
import pytest
from agent.application.local_resources import LocalResourceAdmission
from agent.schemas import StepOutputRef, StepStatus


@pytest.mark.parametrize('mutation', [None, 'different_steps', 'wrong_field', 'unrelated_tool', 'no_dependency', 'literal'])
def test_exact_paired_adaptation_output_required(mutation):
    args = dict(target_model_manifest_path=StepOutputRef('train', 'manifest_path'),
                target_model_manifest_sha256=StepOutputRef('train', 'manifest_sha256'))
    producer = N(step_id='train', tool_name='adapt_epizoo_species')
    step = N(arguments=args, depends_on=('train',))
    if mutation == 'different_steps': args['target_model_manifest_sha256'] = StepOutputRef('other', 'manifest_sha256')
    if mutation == 'wrong_field': args['target_model_manifest_path'] = StepOutputRef('train', 'checkpoint_path')
    if mutation == 'unrelated_tool': producer.tool_name = 'inspect_scATAC'
    if mutation == 'no_dependency': step.depends_on = ()
    if mutation == 'literal': args['target_model_manifest_path'] = '/arbitrary/model.json'
    assert LocalResourceAdmission._new_target_binding(step, [producer]) == (mutation is None)


@pytest.mark.parametrize('mutation', [None, 'failed', 'unverified', 'wrong_digest', 'ambiguous'])
def test_resolved_binding_requires_exact_verified_new_result(mutation):
    step = N(resolved_arguments=dict(target_model_manifest_path='/model.json', target_model_manifest_sha256='a'*64))
    producer = N(tool_name='adapt_epizoo_species', status=StepStatus.SUCCEEDED,
                 verification=N(passed=True), result=dict(manifest_path='/model.json', manifest_sha256='a'*64))
    rows = [producer]
    if mutation == 'failed': producer.status = StepStatus.FAILED
    if mutation == 'unverified': producer.verification.passed = False
    if mutation == 'wrong_digest': producer.result['manifest_sha256'] = 'b'*64
    if mutation == 'ambiguous': rows.append(producer)
    assert LocalResourceAdmission._new_target_binding(step, rows, resolved=True) == (mutation is None)
