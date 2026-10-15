"""Candidate designation changes task coverage, never source or owner authority."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
from agent.application.turn_decisions import parse_decision
from test_multi_input_group_contract import selection, submission
from test_pending_role_contract import public, variants, wire, request, requirements


ROLE = dict(tool='inspect_scATAC', port='dataset', source='source.0')


def test_single_missing_role_is_offered_only_for_candidate_attachments():
    context = public()
    assert variants(context)['request_roles']['properties']['required_roles']['minItems'] == 2
    context['dialogue']['attachment_candidates'] = True
    assert variants(context)['request_roles']['properties']['required_roles']['minItems'] == 1
    assert len(parse_decision(wire(request(required_roles=requirements()[:1]))).required_roles) == 1


def test_legacy_group_still_requires_all_selected_sources(selection):
    owner, records, original = selection
    before = [record.to_dict() for record in records]
    assert 'attachment_candidates' not in original.attribution()
    with pytest.raises(ResourceAdmissionError):
        owner.bind_group(original, [ROLE])
    candidates = owner.compose_group([r.resource_id for r in records], attachment_candidates=True)
    bound = owner.bind_group(candidates, [ROLE])
    assert bound.execution_inputs == {'input_path': records[0].source_path}
    assert LocalResourceAdmission._submission_binding(submission(bound)) == bound
    second = owner.bind_group(candidates, [ROLE | dict(source='source.1')])
    assert second.execution_inputs == {'input_path': records[1].source_path}
    assert before == [owner.load(r.resource_id).to_dict() for r in records]


def test_staged_file_cannot_be_consumed_through_unassigned_consumer(selection):
    owner, records, original = selection
    candidates = replace(original, attachment_candidates=True)
    bound = owner.bind_group(candidates, [ROLE])
    inspect = SimpleNamespace(tool_name='inspect_scATAC', arguments=dict(path=records[0].source_path))
    owner.validate_plan(submission(bound), SimpleNamespace(steps=[inspect]))
    with pytest.raises(ResourceAdmissionError, match='declared scientific roles'):
        owner.validate_plan(submission(bound), SimpleNamespace(steps=[inspect,
            SimpleNamespace(tool_name='cluster_cells', arguments=dict(analysis_path=records[1].source_path))]))


@pytest.mark.parametrize('flag', [None, 1, 'true'])
def test_candidate_designation_is_closed_and_cannot_be_truthy_coerced(selection, flag):
    owner, records, original = selection
    with pytest.raises(ResourceAdmissionError):
        replace(original, attachment_candidates=flag)


def test_sixteen_candidates_preserve_capture_order_without_expanding_role_count(tmp_path):
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    records = []
    for i in range(17):
        path = tmp_path / f'neutral-{i}'
        path.write_bytes(str(i).encode())
        records.append(owner.register(str(i), path, input_type='attachment', label=f'File {i}', attribution='Simulated neutral bytes'))
    ids = [r.resource_id for r in records]
    candidate = owner.compose_group(ids[:16], attachment_candidates=True)
    assert [m['resource_id'] for m in candidate.members] == ids[:16]
    assert len(owner.group_choices(candidate)['sources']) == 16
    with pytest.raises(ResourceAdmissionError):
        owner.compose_group(ids, attachment_candidates=True)
