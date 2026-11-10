"""Native plan reuse and exact step scopes with explicitly simulated native work."""
import json
import pytest
from agent.application import InteractiveAgentApplication
from agent.application.structured_parameters import result_parameters
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import PriorOutputRef
from agent.schemas.orchestration import _serialize
from test_structured_interaction import setup, PROFILE


@pytest.mark.parametrize('changed', ['scale', 'again', 'both'])
def test_native_repeated_step_edits_preserve_defaults_and_lineage(tmp_path, monkeypatch, changed):
    from agent.tools import _native_process
    from agent.tools.analysis import native_matrix
    monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
    calls = []
    def simulated_native(self, args, stage, matrix):
        calls.append(args)
        matrix.write_h5ad(stage / 'result.h5ad')
        return dict(completion='succeeded', versions={'scanpy': 'SIMULATED'}, implementation=[],
            native_output_type='NoneType', result_slots=[], measurement='unchanged technical fixture',
            effective_parameters={'zero_center':True, 'max_value':None, **args.get('kwargs', {})})
    monkeypatch.setattr(native_matrix.MatrixPackage, 'native', simulated_native)
    seed, _, model, inputs = setup(tmp_path, repeated=True)
    import anndata as ad
    import numpy as np
    ad.AnnData(np.eye(4)).write_h5ad(inputs['input_path'])
    original_complete = model.complete
    def complete(**kwargs):
        if 'output_selection_schema_version' in json.loads(kwargs['prompt']):
            return json.dumps(dict(outputs=[dict(name=n, step_id=n, output_key='matrix_path') for n in ('scale','again')]))
        return original_complete(**kwargs)
    model.complete = complete
    def application():
        return InteractiveAgentApplication(seed._application.workspace_root, registry=seed._application.registry,
            model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
            planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda _:model}))
    app = application()
    first = app.submit_turn('session', 'first', 'Scale twice.', expected_generation=0, execution_inputs=inputs)
    assert first.status == 'succeeded', first.to_dict()
    before = app._application.sessions.load('session')
    revision = before.revisions[0]
    original_run = app._application.run_store.load(revision.run_id).to_dict()
    metadata = result_parameters(app._application.sessions, 'session', revision.revision_id)
    assert [s['step_id'] for s in metadata['steps']] == ['scale','again']
    assert all(next(p for p in s['parameters'] if p['argument']=='zero_center')['value'] is True for s in metadata['steps'])
    # Both Interpretation and normal planning/output selection are forbidden.
    def forbidden(**kwargs):
        raise AssertionError('Selected accepted operation called the model')
    model.complete = forbidden
    rows = [dict(tool='run_scanpy', native_function='pp.scale', step_id=changed,
                 argument='kwargs', value={'max_value': 3, 'zero_center': False})]
    if changed == 'both':
        rows[0]['step_id'] = 'scale'
        rows.append(dict(tool='run_scanpy', native_function='pp.scale', step_id='again',
                         argument='kwargs', value={'max_value':2, 'zero_center':True}))
    result = app.submit_turn('session', 'edited', '', expected_generation=1,
        rerun_revision_id=revision.revision_id, parameter_overrides=rows)
    assert result.status == 'succeeded', result.to_dict()
    reopened = application()
    state = reopened._application.sessions.load('session')
    assert state.revisions[0] == revision
    run = reopened._application.run_store.load(state.revisions[-1].run_id)
    assert [s.step_id for s in run.steps] == (['scale','again'] if changed in ('scale','both') else ['again'])
    selected = next(s for s in run.plan.steps if s.step_id==('scale' if changed=='both' else changed))
    assert _serialize(selected.arguments['kwargs']) == {'max_value':3,'zero_center':False}
    if changed == 'again':
        assert isinstance(selected.arguments['input_path'], PriorOutputRef)
        assert state.revisions[-1].outputs[1].run_id == revision.run_id
    elif changed == 'scale':
        assert 'kwargs' not in run.plan.steps[1].arguments
    else:
        assert _serialize(run.plan.steps[1].arguments['kwargs']) == {'max_value':2,'zero_center':True}
    assert reopened._application.run_store.load(revision.run_id).to_dict() == original_run
    # Duplicate transport reads the existing response; it does not execute again.
    count = len(calls)
    assert reopened.submit_turn('session', 'edited', '', expected_generation=1,
        rerun_revision_id=revision.revision_id, parameter_overrides=rows).status == 'succeeded'
    assert len(calls) == count

    from dataclasses import replace
    corrupt = _serialize(state.interactions[-1].admitted)
    corrupt['result_edit']['run_result_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='accepted-run anchor'):
        replace(state, interactions=state.interactions[:-1] + (replace(state.interactions[-1], admitted=corrupt),))
