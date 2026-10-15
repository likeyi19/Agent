"""Owner defaults through real typed admission, planning and verified selection.

Only language decisions are scripted; synthetic inputs establish software scope.
"""
import json
from pathlib import Path

import pytest

from test_scientific_parameter_continuation import (
    TOOL, REQUIRED, OBJECTIVE, argument, selection_harness, selection_inputs,
    selection_work, submit, runtime, fixture_factory, qc_case,
)


@pytest.mark.parametrize('values', [{}, {REQUIRED[0]: 2000, REQUIRED[1]: 5},
    {REQUIRED[0]: 1500}, {REQUIRED[1]: 0}, {REQUIRED[0]: 0, REQUIRED[1]: '0.5'}])
def test_typed_omissions_overrides_zero_and_parameter_evidence(tmp_path, selection_inputs, values):
    text = OBJECTIVE + ''.join(f' Use {key} {value}.' for key, value in values.items())
    declarations = []
    for key, value in values.items():
        token = str(value)
        start = text.index(f'{key} {token}') + len(key) + 1
        declarations.append(argument(text, key, token, start=start, end=start+len(token)))
    harness = selection_harness(tmp_path / 'application', decisions={text:
        dict(kind='execute_plan', target=TOOL, arguments=declarations)})
    harness.service.create_session('session')
    with selection_work() as work:
        view = submit(harness, utterance=text, inputs=selection_inputs)
    assert view['status'] == 'succeeded', view
    assert work == {'production': 1, 'verification': 1}
    step, = harness.service._application.run_store.load(view['run_id']).steps
    assert step.verification.passed and step.verification.artifact_authority['schema_version'] == 2
    expected = dict(min_qc_fragment_records=1000, min_tss_enrichment=4) | values
    manifest = json.loads(Path(step.result['manifest_path']).read_bytes())
    assert {k: manifest['arguments'][k] for k in REQUIRED} == expected
    assert {k: step.resolved_arguments[k] for k in REQUIRED if k in step.resolved_arguments} == values
    facts = {f.field: f for f in harness.service.evidence('session', view['revision_id'], 'selection').facts}
    assert dict(facts['effective_thresholds'].value) == manifest['thresholds']
    for key in REQUIRED:
        assert facts['selection_threshold_origins'].value[key] == (
            'explicit_execution_argument' if key in values else 'existing_owner_default')
    if values:
        assert dict(facts['explicit_user_parameters'].value['parameters']) == {
            key: 'explicit_user_instruction' for key in values}
    if not values:
        assert step.result['n_selected'] == 0 and step.result['readiness'] == 'no_selected_cells'
    before = Path(step.result['manifest_path']).read_bytes()
    restarted = selection_harness(tmp_path / 'application')
    with selection_work() as work:
        assert restarted.service.turn('session', 'first').to_dict() == view
        assert submit(restarted, utterance=text, inputs=selection_inputs) == view
    assert not work and not restarted.calls
    assert Path(step.result['manifest_path']).read_bytes() == before


@pytest.mark.parametrize('key,value', [(REQUIRED[0], -1), (REQUIRED[1], '-1'),
    (REQUIRED[0], True), (REQUIRED[1], None), (REQUIRED[0], None), (REQUIRED[1], 0.5)])
def test_invalid_received_values_never_default_or_execute(tmp_path, selection_inputs, key, value):
    harness = selection_harness(tmp_path / 'application')
    harness.service.create_session('session')
    with selection_work() as work:
        view = submit(harness, inputs=selection_inputs | {key: value})
    assert view['revision_id'] is None and not work
    assert view['status'] != 'succeeded'


def test_defaults_cannot_supply_missing_verified_qc(tmp_path):
    harness = selection_harness(tmp_path / 'application')
    harness.service.create_session('session')
    with selection_work() as work:
        view = submit(harness)
    assert view['revision_id'] is None and not work
    assert view['response']['status'] == 'failed'
