"""The ordinary one-file HTTP upload offers exact source-scoped obs names.

Scripted interpretation exercises upload/worker/admission/compile/preflight.
The executor withholds all native science; these are interface witnesses.
"""
import json
from pathlib import Path

import anndata as ad
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import InteractiveAgentApplication
from agent.application.uploads import H5ADUploadAdmission
from agent.orchestration import build_default_tool_registry
from agent.orchestration.error_policy import classified_agent_error
from agent.orchestration.executor import ExecutionOutcome, PlanExecutor
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import ErrorCategory
from agent.schemas.orchestration import _serialize
from agent.web.app import create_app
from test_uploaded_multiturn import PROFILES, body, submit


LABEL = 'Supplied annotations β'
DIRECT = 'Analyze these cells with 3 components.'
COMPARE = DIRECT + ' Compare groups with the supplied annotations.'
PENDING = 'I will supply cells; use 2 components and compare groups with supplied annotations.'
PENDING_EXACT = PENDING + f' The comparison column is "{LABEL}".'
REPLY = 'Continue the original task using the supplied annotations.'
PORT = dict(tool='pca_embed_cells', port='dataset')
TRUTH = dict(tool='evaluate_cell_clustering', port='ground_truth_dataset')


def upload(client, content, *, upload_id):
    response = client.put('/api/v1/sessions/session/uploads/' + upload_id,
        params={'filename':'Supplied cells.h5ad'}, content=content,
        headers={'Content-Type':'application/octet-stream'})
    assert response.status_code == 201, response.text
    return response.json()


class Withheld(PlanExecutor):
    def __init__(self, registry):
        super().__init__(registry); self.plans = []

    def execute(self, plan, **kwargs):
        assert self.preflight(plan).passed
        self.plans.append(plan)
        error = classified_agent_error(category=ErrorCategory.USER_INPUT_ERROR,
            code='TEST_SCIENCE_WITHHELD', exception_type='TestBoundary')
        return ExecutionOutcome((), (error,), ())


def declaration(text, tool, argument, literal, **extra):
    start = text.index(literal)
    return dict(tool=tool, argument=argument, literal=literal,
                start=start, end=start + len(literal), **extra)


class Model:
    def __init__(self, profile, calls, *, direct=False, pending=False, independent=False, prefile_exact=False):
        self.model_id = profile.model_id
        self.calls, self.direct, self.pending, self.independent = calls, direct, pending, independent
        self.prefile_exact = prefile_exact

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt); self.calls.append(value)
        if 'turn_schema_version' in value:
            text = value['utterance']
            if text in (PENDING, PENDING_EXACT):
                decision = dict(kind='request_roles', target='pca_embed_cells',
                    required_roles=[PORT, TRUTH], roles=[], arguments=[
                        declaration(text, 'pca_embed_cells', 'n_components', '2')])
                if self.prefile_exact:
                    decision['arguments'].append(declaration(text, 'evaluate_cell_clustering', 'label_key', LABEL))
            else:
                source = 'source.1' if self.independent else 'source.0'
                roles = [] if self.direct else [dict(PORT, source=source), dict(TRUTH, source=source)]
                arguments = [] if self.pending and not self.independent else [
                    declaration(text, 'pca_embed_cells', 'n_components', '3')]
                if not self.direct and not self.prefile_exact:
                    arguments.append(declaration(text, 'evaluate_cell_clustering', 'label_key',
                                                 'supplied annotations', value=LABEL, source=source, mode='offered'))
                decision = (dict(kind='answer_roles', pending='@roles', roles=roles, arguments=arguments)
                    if self.pending and not self.independent else
                    dict(kind='execute_plan', target='pca_embed_cells', roles=roles, arguments=arguments))
            return json.dumps(dict(turn_schema_version=1, decision=decision), ensure_ascii=False)
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='accepted result',
                step_id='reduce' if self.direct else 'score',
                output_key='analysis_path' if self.direct else 'report_path')]))
        steps = [dict(step_id='reduce', tool='pca_embed_cells', sources=[], control_dependencies=[])]
        if not self.direct:
            steps.extend([
                dict(step_id='links', tool='build_cell_neighbors', sources=[
                    dict(kind='step', target='embedding', step='reduce')], control_dependencies=[]),
                dict(step_id='partition', tool='cluster_cells', sources=[
                    dict(kind='step', target='analysis', step='links')], control_dependencies=[]),
                dict(step_id='score', tool='evaluate_cell_clustering', sources=[
                    dict(kind='step', target='analysis', step='partition'),
                    dict(kind='input', target='ground_truth_dataset', input='reference_h5ad_path')],
                    control_dependencies=[])])
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


def fixture(tmp_path, **options):
    registry = build_default_tool_registry(); executor = Withheld(registry); calls = []
    root = tmp_path/'workspace'/'uploads'; root.mkdir(parents=True)
    service = InteractiveAgentApplication(tmp_path/'workspace', model_profiles=PROFILES,
        default_profile_id='alpha', registry=registry, executor=executor,
        approved_source_roots=(root,), planning_model_factory_registry=PlanningModelFactoryRegistry(
            {'scripted': lambda profile: Model(profile, calls, **options)}))
    paths = []
    for index in range(2):
        path = tmp_path/f'fixture-{index}.h5ad'
        obs = pd.DataFrame({LABEL:pd.Categorical(['a','a','b','b','c','c'])},
                           index=[f'cell-{index}-{n}' for n in range(6)])
        ad.AnnData(sparse.csr_matrix(np.arange(24).reshape(6,4)%5), obs=obs).write_h5ad(path)
        paths.append(path)
    return service, H5ADUploadAdmission(service.resources, root), executor, calls, paths


@pytest.mark.parametrize('mode', ['direct', 'choice', 'continuation', 'independent'])
def test_single_resource_http_upload_exposes_fields_and_preserves_binding(tmp_path, mode):
    direct, pending, independent = mode == 'direct', mode in {'continuation','independent'}, mode == 'independent'
    service, uploads, executor, calls, paths = fixture(tmp_path,
        direct=direct, pending=pending, independent=independent)
    with TestClient(create_app(service, uploads=uploads)) as client:
        assert client.post('/api/v1/sessions', json={'session_id':'session'}).status_code == 201
        if mode == 'independent':
            old = upload(client, paths[0].read_bytes(), upload_id='old')
            first = submit(client, body(old['resource_id'], turn='pending', utterance=PENDING))
            assert first['status'] == 'clarification' and not executor.plans, first
        elif mode == 'continuation':
            first = submit(client, body(turn='pending', utterance=PENDING))
            assert first['status'] == 'clarification' and not executor.plans, first
            origin, = service._application.sessions.load('session').interactions
            assert 'label_key' not in origin.admitted['inputs']  # A description is not a column name.
        selected = upload(client, paths[1].read_bytes(), upload_id='selected')
        text = REPLY if mode == 'continuation' else DIRECT if direct else COMPARE
        result = submit(client, body(selected['resource_id'], turn='analyze', utterance=text))
        assert result['error']['code'] == 'TEST_SCIENCE_WITHHELD', result
        record = service.resources.load(selected['resource_id'])
        state = service._application.sessions.load('session'); interaction = state.interactions[-1]
        assert interaction.submission['registered_input']['composition'] == 'h5ad-science.v1'
        assert 'candidate_input' in interaction.submission
        assert interaction.snapshot['dialogue']['source_view']
        ordinal = 1 if independent else 0
        assert interaction.snapshot['dialogue']['source_fields'][ordinal]['fields'] == (LABEL,)
        public = next(row['dialogue'] for row in calls if row.get('utterance') == text)
        assert public['source_fields'][ordinal]['fields'] == [LABEL]
        assert public['selected_sources'][ordinal]['source'] == f'source.{ordinal}'
        assert str(tmp_path) not in json.dumps(public)
        by_id = {step.step_id:step for step in executor.plans[0].steps}
        assert by_id['reduce'].arguments['input_path'] == record.source_path
        assert by_id['reduce'].arguments['n_components'] == (2 if mode == 'continuation' else 3)
        if direct:
            assert 'roles' not in interaction.admitted
            assert interaction.admitted['inputs'] == dict(input_path=record.source_path, n_components=3)
        else:
            assert by_id['score'].arguments['reference_h5ad_path'] == record.source_path
            assert by_id['score'].arguments['label_key'] == LABEL
            assert interaction.admitted['roles'][0]['source'] == f'source.{ordinal}'
        run = service._application.run_store.load(result['run_id'])
        assert run.request.prompt == (PENDING if mode == 'continuation' else text)
        assert not state.revisions and len(executor.plans) == 1


def test_exact_column_intent_before_upload_is_retained_without_repeating_the_name(tmp_path):
    service, uploads, executor, calls, paths = fixture(tmp_path, pending=True, prefile_exact=True)
    with TestClient(create_app(service, uploads=uploads)) as client:
        assert client.post('/api/v1/sessions', json={'session_id':'session'}).status_code == 201
        first = submit(client, body(turn='pending', utterance=PENDING_EXACT))
        assert first['status'] == 'clarification' and not executor.plans, first
        origin, = service._application.sessions.load('session').interactions
        assert origin.admitted['inputs']['label_key'] == LABEL
        assert 'value' not in origin.admitted['arguments'][1] and 'source' not in origin.admitted['arguments'][1]
        selected = upload(client, paths[1].read_bytes(), upload_id='selected')
        result = submit(client, body(selected['resource_id'], turn='continue', utterance=REPLY))
        assert result['error']['code'] == 'TEST_SCIENCE_WITHHELD', result
        interaction = service._application.sessions.load('session').interactions[-1]
        assert not interaction.admitted.get('arguments') and LABEL not in REPLY
        assert interaction.admitted['inputs']['label_key'] == LABEL
        by_id = {step.step_id:step for step in executor.plans[0].steps}
        assert by_id['score'].arguments['label_key'] == LABEL
        assert by_id['score'].arguments['reference_h5ad_path'] == service.resources.load(selected['resource_id']).source_path
        assert service._application.run_store.load(result['run_id']).request.prompt == PENDING_EXACT
