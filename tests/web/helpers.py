"""Tiny real scientific execution with provider-independent scripted clients."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
from threading import Event
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningModelProfile, ToolRegistry, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.web.config import ScientificInputSet


class ScriptedModel:
    def __init__(self, profile):
        self.model_id = profile.model_id
        self.calls = []

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            utterance = value['utterance'].lower()
            if 'clarify' in utterance or 'stricter' in utterance:
                decision = dict(kind='clarify', reason='missing_parameter_value')
            elif utterance.startswith('inspect'):
                decision = dict(kind='execute_plan', target='inspect_scATAC')
            else:
                decision = dict(kind='answer_scientific',
                    target=dict(output='r0', subject=None), comparison=None, focus='question')
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        if 'selection_schema_version' in response_schema.get('properties', {}):
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=['processed_inspection'])))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='inspection', step_id='inspect', output_key='n_cells')]))
        if 'dialogue_schema_version' in value:
            claim = next(c['claim_id'] for c in value['evidence']['claims'] if c['field'] == 'n_cells')
            return json.dumps(dict(support='supported', paragraphs=[dict(parts=[
                dict(kind='text', text='The accepted evidence records:'), dict(kind='claim', id=claim)])]))
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[
            dict(step_id='inspect', tool='inspect_scATAC', sources=[], control_dependencies=[])])))


@dataclass
class Harness:
    service: InteractiveAgentApplication
    source: Path
    models: list
    science_calls: list
    started: Event
    release: Event

    @property
    def input_sets(self):
        return (ScientificInputSet('tiny', 'Tiny sparse matrix', {'input_path': str(self.source)}),)


def harness(tmp_path, *, blocked=False, fail_science=False, fail_factory=False, workspace=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / 'tiny.h5ad'
    ad.AnnData(sparse.csr_matrix(np.asarray([[1, 0, 1], [0, 2, 0]], dtype=np.float32)),
        obs=pd.DataFrame(index=pd.Index(['cell-a', 'cell-b'], dtype='object')),
        var=pd.DataFrame(index=pd.Index(['peak-a', 'peak-b', 'peak-c'], dtype='object'))).write_h5ad(source)
    models, science_calls = [], []
    started, release = Event(), Event()
    default = build_default_tool_registry()
    original = default.get('inspect_scATAC')

    def counted(**arguments):
        science_calls.append('inspect_scATAC')
        if blocked:
            started.set()
            assert release.wait(20), 'The test did not release scientific execution.'
        if fail_science:
            raise RuntimeError('secret-key-sentinel /server/private/input.h5ad')
        return original.function(**arguments)

    registry = ToolRegistry(tuple(replace(default.get(name), function=counted)
        if name == 'inspect_scATAC' else default.get(name) for name in default.names()))

    def factory(profile):
        if fail_factory:
            models.append('failed factory')
            raise RuntimeError('secret-key-sentinel /server/private/provider-config')
        model = ScriptedModel(profile)
        models.append(model)
        return model

    profiles = tuple(PlanningModelProfile(key, 'scripted', 'model/' + key) for key in ('alpha', 'beta'))
    service = InteractiveAgentApplication(workspace or tmp_path / 'workspace',
        model_profiles=profiles, default_profile_id='alpha', display_labels={'alpha': 'Default test model', 'beta': 'Other test model'},
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}), registry=registry)
    return Harness(service, source, models, science_calls, started, release)


def submission(turn='first', generation=0, utterance='Inspect the supplied matrix.', **changes):
    body = dict(turn_id=turn, expected_generation=generation, utterance=utterance, input_set_id='tiny')
    body.update(changes)
    return body


def wait_turn(client, turn='first', *, session='session', timeout=20, terminal_status=None):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = client.get(f'/api/v1/sessions/{session}/turns/{turn}')
        assert response.status_code == 200, response.text
        last = response.json()
        if last.get('response') is not None or terminal_status is not None and last['status'] == terminal_status:
            return last
        time.sleep(0.03)
    raise AssertionError(f'Turn did not finish: {last}')
