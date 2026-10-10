"""Both original species resolve through one catalog and the shared dialogue path.

Responses are scripted contract witnesses. The registered tiny H5AD establishes
only source identity; a recording executor withholds every scientific tool. These
tests claim resource/continuation admission, never compatible-input qualification
or a completed foundation-model execution.
"""
import json

import pytest

from agent.application.local_resources import ResourceAdmissionError, select_epizoo_resource
from agent.orchestration import build_default_tool_registry
from agent.orchestration.error_policy import classified_agent_error
from agent.orchestration.executor import ExecutionOutcome, PlanExecutor
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import ErrorCategory
from agent.schemas.orchestration import StepOutputRef, _serialize
from agent.web.config import build_interactive_application, load_web_configuration
from test_service import _tiny_h5ad


CHECKPOINT_SHA = '6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a'
AUXILIARY_PINS = {
    'human': ('8b576fa4fc60a1e2fc77607ffacff2883441def3d8cb8225776b71ad6c00e80b',
              '994d9c3e87208074e695c4c418b28d9587dd8991ad033cf33e62f96ceebc7875'),
    'mouse': ('c4c63aaae8a6f841812189bf59930014162c3f61c7a463d744be71595359171d',
              '96a80287ae085d7e9e10d05f0dd7d5b266ad86d08b91b01ccc07af6b9b01e393'),
}
OBJECTIVE = 'Analyze this dataset using EpiZoo, neighbors, Leiden and UMAP.'
TOOLS = ('epizoo_embed_cells', 'build_cell_neighbors', 'cluster_cells', 'compute_cell_umap')


def configuration(tmp_path, *, explicit_species=None):
    rows = []
    for species, (frequencies, filters) in AUXILIARY_PINS.items():
        rows.append(dict(resource_id=f'{species}-original', label=f'Original {species} resources',
            species=species, checkpoint_path='/operator/original/pretrained_EpiZoo.pth',
            checkpoint_sha256=CHECKPOINT_SHA, frequencies_sha256=frequencies,
            filter_indices_sha256=filters, qualification='Recorded original resource identities; configuration fixture.',
            default=True))
    if explicit_species is not None:
        # Explicit selection remains authoritative even if two defaults apply.
        explicit = next(row for row in rows if row['species'] == explicit_species)
        rows.append(explicit | dict(resource_id=f'{explicit_species}-explicit'))
    path = tmp_path / 'configuration.json'
    path.write_text(json.dumps(dict(workspace_root='workspace', upload_root='uploads',
        default_profile_id='scripted', model_profiles=[dict(profile_id='scripted',
            provider_id='scripted', model_id='mapping-witness')],
        input_sets=[dict(input_set_id='settings', label='Explicit device setting',
            h5ad_companion=True, execution_inputs=dict(device='cuda:0'))],
        epizoo_resources=rows)), encoding='utf-8')
    return load_web_configuration(path)


@pytest.mark.parametrize('species', ['human', 'mouse'])
def test_loaded_shared_checkpoint_catalog_selects_species_pins_and_explicit_precedence(tmp_path, species):
    config = configuration(tmp_path)
    resources = {row.species: row for row in config.epizoo_resources}
    assert resources['human'].checkpoint_sha256 == resources['mouse'].checkpoint_sha256 == CHECKPOINT_SHA
    assert resources['human'].checkpoint_path == resources['mouse'].checkpoint_path
    assert resources['human'].frequencies_sha256 != resources['mouse'].frequencies_sha256
    assert resources['human'].filter_indices_sha256 != resources['mouse'].filter_indices_sha256
    original = dict(species=species, device='cuda:0')
    values, error = select_epizoo_resource(original, None, config.epizoo_resources)
    assert error is None and values == original | resources[species].inputs()
    assert original == dict(species=species, device='cuda:0')

    explicit = configuration(tmp_path, explicit_species=species)
    values, error = select_epizoo_resource(original, None, explicit.epizoo_resources)
    assert values == original and error == 'EPIZOO_RESOURCE_AMBIGUOUS'
    selected_id = f'{species}-explicit'
    selected = next(row for row in explicit.epizoo_resources if row.resource_id == selected_id)
    values, error = select_epizoo_resource(original, selected_id, explicit.epizoo_resources)
    assert error is None and values == original | selected.inputs()
    other_species = 'mouse' if species == 'human' else 'human'
    with pytest.raises(ResourceAdmissionError) as mismatch:
        select_epizoo_resource(dict(species=other_species), selected_id, explicit.epizoo_resources)
    assert mismatch.value.code == 'EPIZOO_RESOURCE_SELECTION_INVALID'


class MappingWitness:
    model_id = 'mapping-witness'

    def __init__(self, species, calls, *, objective=OBJECTIVE, embedding_only=False):
        self.species, self.calls = species, calls
        self.objective, self.embedding_only = objective, embedding_only

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            target = 'epizoo_embed_cells' if self.embedding_only else 'compute_cell_umap'
            decision = (dict(kind='execute_plan', target=target)
                if value['utterance'] == self.objective else
                dict(kind='answer_prerequisite', pending='@species', species=self.species))
            response = dict(turn_schema_version=1, decision=decision)
        elif 'selection_schema_version' in value:
            response = dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=['embedding_analysis']))
        elif 'output_selection_schema_version' in value:
            outputs = (
                ('embeddings', 'embed', 'embedding_path'), ('neighbors', 'neighbors', 'analysis_path'),
                ('clusters', 'cluster', 'analysis_path'), ('umap', 'umap', 'analysis_path'))
            response = dict(outputs=[dict(name=name, step_id=step, output_key=key)
                for name, step, key in (outputs[:1] if self.embedding_only else outputs)])
        else:
            assert 'semantic_prompt_version' in value
            rows = [('embed', TOOLS[0], None, None), ('neighbors', TOOLS[1], 'embedding', 'embed'),
                    ('cluster', TOOLS[2], 'analysis', 'neighbors'), ('umap', TOOLS[3], 'analysis', 'cluster')]
            if self.embedding_only:
                rows = rows[:1]
            response = dict(schema_version=4, decision=dict(kind='plan', steps=[dict(
                step_id=step, tool=tool, sources=[] if target is None else
                [dict(kind='step', target=target, step=upstream)], control_dependencies=[])
                for step, tool, target, upstream in rows]))
        return json.dumps(response)

    def close(self):
        pass


class WithholdScience(PlanExecutor):
    def __init__(self, registry):
        super().__init__(registry)
        self.plans = []

    def execute(self, plan, **kwargs):
        assert self.preflight(plan).passed
        self.plans.append(plan)
        error = classified_agent_error(category=ErrorCategory.USER_INPUT_ERROR,
            code='TEST_SCIENCE_WITHHELD', exception_type='TestBoundary')
        return ExecutionOutcome((), (error,), ())


@pytest.mark.parametrize('species', ['human', 'mouse'])
def test_species_answer_resumes_original_registered_request_with_exact_default_and_no_science(tmp_path, species):
    config = configuration(tmp_path)
    registry = build_default_tool_registry()
    executor = WithholdScience(registry)
    calls = []
    factories = PlanningModelFactoryRegistry({'scripted': lambda _: MappingWitness(species, calls)})

    def application():
        return build_interactive_application(config, planning_model_factory_registry=factories,
            registry=registry, executor=executor)

    app = application()
    app.create_session('session')
    source = _tiny_h5ad(config.upload_root / 'structural-source.h5ad')
    record = app.resources.register('structural-source', source, label='Registered structural fixture',
        attribution='Exact source identity only; scientific compatibility is not asserted.')
    binding = app.resources.compose_h5ad(record.resource_id, config.input_sets[0].inputs(),
        resource_selection_error='EPIZOO_RESOURCE_REQUIRED')
    pending = app.submit_turn('session', 'initial', OBJECTIVE, expected_generation=0, registered_input=binding)
    assert pending.status == 'clarification' and pending.revision_id is None
    assert pending.response.clarification['reason'] == 'missing_species'
    state = app._application.sessions.load('session')
    original = state.interactions[0]
    original_inputs = _serialize(original.submission['execution_inputs'])
    assert original_inputs == dict(input_path=record.source_path, device='cuda:0')
    assert original.prerequisite['origin_turn_id'] == 'initial'
    assert not executor.plans and not state.revisions

    before = len(calls)
    app = application()
    assert app.turn('session', 'initial') == pending
    app.reopen_session('session')
    assert len(calls) == before
    answer = f'These are {species} cells.'
    result = app.submit_turn('session', 'answer', answer, expected_generation=0)
    assert result.status == 'failed' and result.error.code == 'TEST_SCIENCE_WITHHELD'
    assert result.revision_id is None and not result.steps
    assert len(executor.plans) == 1

    plan = executor.plans[0]
    assert tuple(step.tool_name for step in plan.steps) == TOOLS
    expected = next(row for row in config.epizoo_resources if row.species == species)
    embedded, neighbors, clustered, umap = plan.steps
    for key, value in original_inputs.items():
        assert embedded.arguments[key] == value
    assert embedded.arguments['species'] == species
    assert embedded.arguments['checkpoint_path'] == expected.checkpoint_path
    assert _serialize(embedded.arguments['expected_resource_identity']) == expected.inputs()['expected_resource_identity']
    assert neighbors.arguments['embedding_path'] == StepOutputRef('embed', 'embedding_path')
    assert neighbors.arguments['cell_ids_path'] == StepOutputRef('embed', 'cell_ids_path')
    assert clustered.arguments['analysis_path'] == StepOutputRef('neighbors', 'analysis_path')
    assert umap.arguments['analysis_path'] == StepOutputRef('cluster', 'analysis_path')
    assert 'n_neighbors' not in neighbors.arguments and 'resolution' not in clustered.arguments
    assert 'min_dist' not in umap.arguments and 'spread' not in umap.arguments
    state = app._application.sessions.load('session')
    resumed = state.interactions[-1]
    assert resumed.admitted['inputs'] == original_inputs | dict(species=species) | expected.inputs()
    assert resumed.admitted['continuation']['origin_turn_id'] == 'initial'
    assert resumed.admitted['registered_input']['resource_id'] == record.resource_id
    assert resumed.admitted['registered_input']['record_sha256'] == record.record_sha256
    selector, = [call for call in calls if 'output_selection_schema_version' in call]
    assert selector['question'] == OBJECTIVE
    assert not state.revisions and state.generation == 0
    run = app._application.run_store.load(result.run_id)
    assert not run.steps and run.errors[0].code == 'TEST_SCIENCE_WITHHELD'
    assert not run.run_verification.passed
    before = len(calls)
    assert app.submit_turn('session', 'answer', answer, expected_generation=0) == result
    assert app.turn('session', 'answer') == result and len(calls) == before
    assert len(executor.plans) == 1


def test_exact_human_species_variant_resumes_single_embedding_request_without_science(tmp_path):
    objective = 'Analyze this dataset using EpiZoo.'
    answer = 'These are human cells.'
    config = configuration(tmp_path)
    registry = build_default_tool_registry()
    executor = WithholdScience(registry)
    calls = []
    factories = PlanningModelFactoryRegistry({'scripted': lambda _: MappingWitness(
        'human', calls, objective=objective, embedding_only=True)})
    app = build_interactive_application(config, planning_model_factory_registry=factories,
        registry=registry, executor=executor)
    app.create_session('session')
    source = _tiny_h5ad(config.upload_root / 'structural-source.h5ad')
    record = app.resources.register('structural-source', source, label='Registered structural fixture',
        attribution='Exact source identity only; scientific compatibility is not asserted.')
    binding = app.resources.compose_h5ad(record.resource_id, config.input_sets[0].inputs(),
        resource_selection_error='EPIZOO_RESOURCE_REQUIRED')
    pending = app.submit_turn('session', 'initial', objective,
        expected_generation=0, registered_input=binding)
    assert pending.status == 'clarification' and pending.revision_id is None
    assert pending.response.clarification['reason'] == 'missing_species'
    state = app._application.sessions.load('session')
    original = state.interactions[0]
    original_inputs = _serialize(original.submission['execution_inputs'])
    assert original.utterance == objective
    assert original_inputs == dict(input_path=record.source_path, device='cuda:0')
    assert original.prerequisite['origin_turn_id'] == 'initial'
    assert not executor.plans and not state.revisions

    result = app.submit_turn('session', 'answer', answer, expected_generation=0)
    assert result.status == 'failed' and result.error.code == 'TEST_SCIENCE_WITHHELD'
    assert result.revision_id is None and not result.steps
    assert len(executor.plans) == 1
    embedded, = executor.plans[0].steps
    assert embedded.tool_name == 'epizoo_embed_cells' and not embedded.depends_on
    expected = next(row for row in config.epizoo_resources if row.species == 'human')
    for key, value in original_inputs.items():
        assert embedded.arguments[key] == value
    assert embedded.arguments['species'] == 'human'
    assert embedded.arguments['checkpoint_path'] == expected.checkpoint_path
    assert _serialize(embedded.arguments['expected_resource_identity']) == expected.inputs()['expected_resource_identity']
    state = app._application.sessions.load('session')
    resumed = state.interactions[-1]
    assert resumed.utterance == answer
    assert resumed.admitted['inputs'] == original_inputs | dict(species='human') | expected.inputs()
    assert resumed.admitted['continuation']['origin_turn_id'] == 'initial'
    assert resumed.admitted['registered_input']['resource_id'] == record.resource_id
    assert resumed.admitted['registered_input']['record_sha256'] == record.record_sha256
    selector, = [call for call in calls if 'output_selection_schema_version' in call]
    assert selector['question'] == objective
    assert [(row['step_id'], row['tool']) for row in selector['steps']] == [('embed', 'epizoo_embed_cells')]
    assert not state.revisions and state.generation == 0
    run = app._application.run_store.load(result.run_id)
    assert not run.steps and run.errors[0].code == 'TEST_SCIENCE_WITHHELD'
    assert not run.run_verification.passed
