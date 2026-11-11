"""HTTP result editing uses the canonical native plan; numerical work is simulated."""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from fastapi.testclient import TestClient

from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import PriorOutputRef
from agent.web.app import create_app
from helpers import wait_turn

sys.path.insert(0, str(Path(__file__).parents[1]/'native_raw'))
from test_execution_consolidation import Model


@pytest.mark.parametrize('repeated', [False, True])
def test_native_result_edit_reuses_exact_upstream_and_reopens(tmp_path, monkeypatch, repeated):
    from agent.tools import _native_process
    from agent.tools.analysis import native_matrix
    monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
    calls = []
    def simulated(self, args, stage, matrix):
        fn = args['function_name']; calls.append(fn)
        slots = []
        if fn == 'tl.spectral':
            matrix.obsm['X_spectral'] = np.ones((matrix.n_obs, 2)); slots = ['obsm.X_spectral']
        elif fn == 'pp.knn':
            matrix.obsp['distances'] = sparse.eye(matrix.n_obs, format='csr'); slots = ['obsp.distances']
        elif fn == 'tl.leiden':
            matrix.obs['leiden'] = pd.Categorical(['0'] * matrix.n_obs); slots = ['obs.leiden']
        matrix.write_h5ad(stage/'result.h5ad')
        return dict(completion='succeeded', versions={'snapatac2':'SIMULATED'}, implementation=[],
            native_output_type='NoneType', result_slots=slots,
            effective_parameters={'resolution':1.,'random_state':0, **args.get('kwargs', {})})
    monkeypatch.setattr(native_matrix.SNAPATAC2, 'native', simulated.__get__(native_matrix.SNAPATAC2))
    model = Model(umap=False); complete = model.complete
    def respond(**kw):
        value = json.loads(complete(**kw))
        if repeated and value.get('schema_version') == 4:
            steps = value['decision']['steps']
            from copy import deepcopy
            other = deepcopy(steps[-1]); other['step_id'] = 'other'
            steps.append(other)
        if repeated and 'outputs' in value:
            value['outputs'].append(dict(name='other', step_id='other', output_key='matrix_path'))
        return json.dumps(value)
    model.complete = respond
    profile = PlanningModelProfile('native-edit','scripted',model.model_id)
    app = InteractiveAgentApplication(tmp_path/'app',model_profiles=(profile,),default_profile_id=profile.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:model}))
    path = tmp_path/'counts.h5ad';ad.AnnData(sparse.eye(8, format='csr')).write_h5ad(path)
    app.create_session('session')
    result = app.submit_turn('session','first','Cluster these counts.',expected_generation=0,execution_inputs={'input_path':str(path)})
    assert result.status == 'succeeded', result.to_dict()
    sessions = app._application.sessions; before = sessions.load('session'); revision = before.revisions[-1]
    original = app._application.run_store.load(revision.run_id).to_dict()
    step = 'other' if repeated else 'clusters'
    model.complete = lambda **kw: pytest.fail('Result-scoped edit called the model')
    with TestClient(create_app(app)) as client:
        metadata = client.get(f'/api/v1/sessions/session/revisions/{revision.revision_id}/parameters')
        assert metadata.status_code == 200, metadata.text
        target = next(s for s in metadata.json()['steps'] if s['step_id'] == step)
        assert next(p for p in target['parameters'] if p['argument']=='resolution')['value'] == 1.
        calls.clear()
        response = client.post('/api/v1/sessions/session/turns',json=dict(turn_id='edit',utterance='',
            expected_generation=before.generation,rerun_revision_id=revision.revision_id,
            parameter_overrides=[dict(tool='run_snapatac2',native_function='tl.leiden',step_id=step,
                argument='kwargs',value={'resolution':.7,'random_state':17})]))
        assert response.status_code == 202, response.text
        done = wait_turn(client,'edit'); assert done['status']=='succeeded',done
        assert calls == ['tl.leiden']
        state = sessions.load('session'); new = state.revisions[-1]
        run = app._application.run_store.load(new.run_id)
        assert [s.step_id for s in run.steps] == [step]
        assert isinstance(run.plan.steps[0].arguments['input_path'], PriorOutputRef)
        assert run.steps[0].resolved_arguments['kwargs'] == {'resolution':.7,'random_state':17}
        assert new.parent_revision_id == revision.revision_id
        assert app._application.run_store.load(revision.run_id).to_dict() == original
        reopened = client.get('/api/v1/sessions/session').json()
        assert reopened['active_revision_id'] == new.revision_id
        activated = client.post('/api/v1/sessions/session/activate', json=dict(turn_id='back',
            revision_id=revision.revision_id, expected_generation=state.generation))
        assert activated.status_code == 200, activated.text
        assert sessions.load('session').active_revision_id == revision.revision_id
        assert calls == ['tl.leiden']
        assert sessions.load('session').revisions[0] == revision
