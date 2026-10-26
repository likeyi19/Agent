"""Source-qualified field selection reaches existing native scientific owners.

The data and semantic client are simulated. Registration, HTTP worker, role
binding, compiler, Leiden/evaluation and verification use production code.
"""
import json

from fastapi.testclient import TestClient
import pytest

from agent.application import InteractiveAgentApplication
from agent.application.uploads import H5ADUploadAdmission
from agent.providers import PlanningModelFactoryRegistry
from agent.web.app import create_app
from test_multi_input_scientific_roles import (
    LABEL, PROFILE, MultiInputModel, execute, multi_body, multi_input_harness,
    multi_input_work, upload_sources,
)


PHRASE = 'the supplied cell identities'
REQUEST = f'Cluster the analysis at resolution 0.7 and evaluate agreement with {PHRASE} in the other file.'


class SourceModel(MultiInputModel):
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value:
            self.calls.append(dict(value=value, prompt=prompt, schema=response_schema))
            decision = dict(kind='execute_plan', target='evaluate_cell_clustering',
                roles=[dict(tool='cluster_cells', port='analysis', source='source.0')],
                arguments=[dict(tool='cluster_cells', argument='resolution',literal='0.7',evidence='resolution 0.7'),
                    dict(tool='evaluate_cell_clustering', argument='label_key',
                    source='source.1', mode='offered', value=LABEL,
                    literal=PHRASE, evidence=PHRASE)])
            return json.dumps(dict(turn_schema_version=1, decision=decision), ensure_ascii=False)
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='communities',step_id='clustering',output_key='analysis_path'),
                dict(name='evaluation',step_id='evaluation',output_key='report_path')]))
        return super().complete(prompt=prompt, response_schema=response_schema)


@pytest.mark.parametrize('wrong_order', [False, True])
def test_source_field_role_assertion_preserves_native_cell_axis_qualification(tmp_path, wrong_order):
    harness = multi_input_harness(tmp_path, content='wrong_cell_order' if wrong_order else None)
    def factory(profile):
        model = SourceModel(); harness.models.append(model); return model
    harness.service = InteractiveAgentApplication(harness.service._application.workspace_root,
        model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':factory}),
        approved_source_roots=(harness.uploads.root,))
    harness.uploads = H5ADUploadAdmission(harness.service.resources, harness.uploads.root)
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        with multi_input_work() as calls:
            result = execute(client, multi_body([record['resource_id'] for record in records], utterance=REQUEST))
        state = harness.service._application.sessions.load('session')
        interaction, = state.interactions
        roles = {(row['tool'],row['port']):row['source'] for row in interaction.admitted['roles']}
        assert roles == {('cluster_cells','analysis'):'source.0',
                         ('evaluate_cell_clustering','ground_truth_dataset'):'source.1'}
        field, = [row for row in interaction.admitted['arguments'] if row['argument'] == 'label_key']
        assert field['mode'] == 'offered'
        assert interaction.admitted['inputs']['label_key'] == LABEL and LABEL not in REQUEST
        assert calls['interpreter'] == calls['planner'] == calls['compiler'] == 1
        assert calls['production.epizoo'] == calls['production.neighbors'] == 0
        assert calls['production.leiden'] == 1
        run = harness.service._application.run_store.load(result['run_id'])
        if wrong_order:
            assert result['status'] == 'failed' and not state.revisions
            assert calls['production.evaluation'] == calls['source.evaluation'] == 1
            assert calls['verification.evaluation'] == 0
            assert any(error.code == 'INVALID_ARGUMENT' and error.tool_name == 'evaluate_cell_clustering'
                       for error in run.errors)
        else:
            assert result['status'] == 'succeeded' and len(state.revisions) == 1, result
            assert calls['production.evaluation'] == 1
            assert all(step.verification.passed for step in run.steps)
            for name, parameter in (('communities','resolution'),('evaluation','label_key')):
                summary = harness.service.evidence('session',result['revision_id'],name)
                assert summary.status == 'available', summary
                provenance, = [fact for fact in summary.facts if fact.field == 'explicit_user_parameters']
                assert provenance.value['parameters'][parameter] == 'explicit_user_instruction'
        public = harness.models[0].calls[0]['value']['dialogue']
        assert LABEL in public['source_fields'][1]['fields']
        assert LABEL not in public['source_fields'][0]['fields']
