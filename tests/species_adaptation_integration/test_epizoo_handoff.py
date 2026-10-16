"""Typed EpiZoo handoff checks; synthetic fixtures do not qualify biology."""
from dataclasses import asdict
import json
import pytest


def test_epizoo_attachment_decision_has_typed_handoff():
    from agent.application.epizoo_handoff import EpiZooDeclaration
    from agent.application.turn_decisions import parse_decision
    declared = EpiZooDeclaration(operation='inference', source='source.0',
        scientific_name='Danio rerio', taxonomy_id=7955, assembly='danRer11',
        matrix_semantics='binary_accessibility')
    decision = parse_decision(json.dumps(dict(turn_schema_version=1, decision=dict(kind='request_roles', target='epizoo_embed_cells',
        required_roles=[], roles=[], arguments=[], epizoo=asdict(declared)))))
    assert decision.epizoo == declared


def test_native_resource_is_available_without_manual_inputset_selection():
    from agent.application.raw_handoff import resources
    from agent.application.input_sets import ScientificInputSet
    choice = ScientificInputSet('native', 'Qualified scientific training resources',
        dict(adaptation_spec_path='/qualified/native.json', adaptation_spec_sha256='a'*64),
        h5ad_companion=True)
    assert resources((choice,), 'epizoo') == (choice,)


def wait(client, turn):
    import time
    for _ in range(600):
        value = client.get('/api/v1/sessions/test/turns/' + turn).json()
        if value.get('response') is not None and value['status'] not in ('pending', 'running', 'processing', 'submitted', 'admitted', 'interpreting'):
            return value
        time.sleep(.02)
    raise AssertionError(value)


def setup(root, choices, decisions):
    from agent.application import InteractiveAgentApplication
    from agent.application.uploads import H5ADUploadAdmission
    from agent.providers import PlanningModelFactoryRegistry
    from agent.orchestration import PlanningModelProfile
    from agent.web.app import create_app
    class Model:
        model_id = 'scripted/usi5'
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            if 'turn_schema_version' in value:
                decision = decisions(value)
                return json.dumps(dict(turn_schema_version=1, decision=decision))
            if 'selection_schema_version' in response_schema.get('properties', {}):
                return json.dumps(dict(selection_schema_version=1, decision=dict(kind='select', capability_ids=['species_adaptation'])))
            if 'output_selection_schema_version' in value:
                return json.dumps(dict(outputs=[dict(name='model', step_id='adapt', output_key='manifest_path')]))
            return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[
                dict(step_id='adopt', tool='adopt_scATAC_cell_by_features', sources=[], control_dependencies=[]),
                dict(step_id='adapt', tool='adapt_epizoo_species', sources=[dict(kind='step', target='matrix', step='adopt')], control_dependencies=[])])))
    uploads = root / 'uploads'; uploads.mkdir(parents=True, exist_ok=True)
    profile = PlanningModelProfile('scripted', 'scripted', Model.model_id)
    app = InteractiveAgentApplication(root, model_profiles=(profile,), default_profile_id='scripted',
        approved_source_roots=(uploads,), planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda _: Model()}))
    return app, create_app(app, input_sets=choices, uploads=H5ADUploadAdmission(app.resources, uploads))


def upload(client, name, content, kind='attachment'):
    result = client.put('/api/v1/sessions/test/uploads/' + name.replace('.', '-'),
        params=dict(filename=name, input_type=kind), content=content,
        headers={'content-type': 'application/octet-stream'})
    assert result.status_code == 201, result.text
    return result.json()['resource_id']


def send(client, turn, text, ids=(), generation=0):
    result = client.post('/api/v1/sessions/test/turns', json=dict(turn_id=turn,
        expected_generation=generation, utterance=text, attachment_ids=list(ids)))
    assert result.status_code == 202, result.text
    return wait(client, turn)


@pytest.fixture
def native_choice(integration, monkeypatch):
    from pathlib import Path
    from agent.tools.models.epizoo_adaptation import contract as c
    from agent.application.input_sets import ScientificInputSet
    from agent.tools.models import epizoo_inputs
    inputs, counts, _ = integration
    path = Path(inputs['adaptation_spec_path'])
    spec = json.loads(path.read_text())
    spec.update(contract_version='epizoo-adaptation-inputs.v2', execution_profile='native.v2', reference_identity_sha256=None)
    source = c.read_json(spec['source_bundle']['path'], spec['source_bundle']['sha256'])
    source['checkpoint'] = dict(sha256='c'*64)
    r = c.write_json(path.with_name('native-source.json'), source)
    spec['source_bundle'] = {k: r[k] for k in ('path', 'sha256')}
    record = c.write_json(path.with_name('native-spec.json'), spec)
    # Only the explicitly simulated source/SEAM bundle verification is stubbed.
    monkeypatch.setattr(epizoo_inputs, 'qualify_resource', lambda _: None)
    return ScientificInputSet('native', 'Simulated native resource', dict(
        adaptation_spec_path=record['path'], adaptation_spec_sha256=record['sha256']), h5ad_companion=True)


def declaration(**updates):
    from agent.application.epizoo_handoff import EpiZooDeclaration
    return EpiZooDeclaration(operation='posttraining', source='source.0',
        scientific_name='Macaca fascicularis', taxonomy_id=9541, assembly='GCF_000364345.1',
        matrix_semantics='fragment_counts', strategy='de_novo', feature_category='peak_set', **updates)


def test_missing_genome_append_restart_retains_source_and_parameters(case, native_choice, tmp_path, monkeypatch):
    from pathlib import Path
    from fastapi.testclient import TestClient
    from agent.application.epizoo_handoff import EpiZooDeclaration
    from agent.orchestration.executor import PlanExecutor
    calls = []
    def stop(self, plan, **kwargs):
        check = self.preflight(plan)
        assert check.passed, check
        calls.append(plan)
        raise RuntimeError('Explicit software acceptance stop after preflight; no simulated training.')
    monkeypatch.setattr(PlanExecutor, 'execute', stop)
    def decisions(p):
        if p['dialogue'].get('pending_prerequisite'):
            return dict(kind='answer_roles', pending='@roles', roles=[], arguments=[],
                epizoo=asdict(EpiZooDeclaration(genome_source='source.1')))
        return dict(kind='request_roles', target='adapt_epizoo_species', required_roles=[], roles=[],
            epizoo=asdict(declaration()), arguments=[dict(tool='adapt_epizoo_species', argument='max_steps',
                literal='2', evidence='2 training steps')])
    root = tmp_path / 'chat'
    app, web = setup(root, (native_choice,), decisions)
    with TestClient(web) as client:
        assert client.post('/api/v1/sessions', json={'session_id': 'test'}).status_code == 201
        source = upload(client, 'matrix.h5ad', Path(case['source_path']).read_bytes(), 'h5ad')
        first = send(client, 'first', 'Post-train EpiZoo with 2 training steps.', [source])
        assert first['status'] == 'clarification', first
        assert 'FASTA' in first['response']['text']
        assert not calls
        genome = upload(client, 'genome.fa', b'>chr1\n' + b'A'*100 + b'\n')
    app, web = setup(root, (native_choice,), decisions)
    with TestClient(web) as client:
        done = send(client, 'reply', 'Use this genome and continue.', [source, genome])
        assert calls, done
        state = app._application.sessions.load('test')
        admitted = state.interactions[-1].admitted
        assert admitted['inputs']['max_steps'] == 2
        assert admitted['inputs']['source_path'] == app.resources.load(source).source_path
        assert calls[0].steps[-1].arguments['max_steps'] == 2
        assert not state.revisions  # Deliberately stopped before science.
        from agent.application.epizoo_handoff import genomes
        from agent.application import shared_genome, reference_supplementation as refs
        from agent.application.local_resources import LocalResourceRecord
        choices = genomes(app.resources, 'test')
        assert len(choices) == 1 and not genomes(app.resources, 'another-session')
        record = LocalResourceRecord(**choices[0].inputs()['qualified_genome'])
        required = dict(species=case['species'], assembly=case['assembly'], feature_category='peak_set')
        assert refs.choices(app.resources, 'another-session', required) == ()
        assert record.session_id == 'test' and record.reference_qualification['review'] is None
        # Different peaks reuse the exact genome/index under the same private scope.
        import anndata as ad
        from types import SimpleNamespace
        data = ad.read_h5ad(case['source_path'])
        data.var_names = ['chr1:40-50', 'chr1:50-60', 'chr1:60-70']
        second = tmp_path / 'second.h5ad'; data.write_h5ad(second)
        sid = upload(client, 'second.h5ad', second.read_bytes(), 'h5ad')
        bound = shared_genome.bind(app.resources, record, app.resources.load(sid), 'test',
            SimpleNamespace(turn_id='second', admitted={'reference_requirement': required}))
        qualified = refs.check_record(app.resources, bound)
        assert qualified.genome.fasta.path == record.source_path
        assert qualified.genome.fai.path == record.source_index['path']
        original = refs.check_record(app.resources, app.resources.load(next(r.resource_id for r in
            refs._records(app.resources) if r.input_type == 'regulatory_reference' and
            'genome_resource_id' not in r.reference_qualification)))
        assert qualified.features.ordered_feature_sha256 != original.features.ordered_feature_sha256
        with pytest.raises(ValueError):
            shared_genome.bind(app.resources, record, app.resources.load(sid), 'another-session',
                SimpleNamespace(turn_id='second', admitted={'reference_requirement': required}))


def model_choices():
    from agent.application.input_sets import ScientificInputSet
    first = ScientificInputSet('one', 'Model one', dict(target_model_manifest_path='/model/one', target_model_manifest_sha256='a'*64))
    second = ScientificInputSet('two', 'Model two', dict(target_model_manifest_path='/model/two', target_model_manifest_sha256='b'*64))
    observation = dict(operation='inference', species=dict(scientific_name='Danio rerio', taxonomy_id=7955),
        assembly='danRer11', matrix_semantics='binary_accessibility', feature_membership_sha256='f'*64,
        checkpoint_sha256='c'*64)
    return first, second, {'one': observation, 'two': dict(observation, checkpoint_sha256='d'*64)}


@pytest.mark.parametrize('changed', ['assembly', 'matrix_semantics', 'species', 'vocabulary', 'checkpoint'])
def test_scientifically_incompatible_model_is_never_substituted(changed):
    from dataclasses import replace
    from agent.application.epizoo_handoff import resolve, EpiZooDeclaration
    first, second, observations = model_choices()
    d = EpiZooDeclaration(operation='inference', scientific_name='Danio rerio', taxonomy_id=7955,
        assembly='danRer11', matrix_semantics='binary_accessibility')
    source = dict(feature_membership_sha256='f'*64)
    checkpoint = None
    if changed == 'assembly': d = replace(d, assembly='another')
    if changed == 'matrix_semantics': d = replace(d, matrix_semantics='fragment_counts')
    if changed == 'species': d = replace(d, taxonomy_id=123)
    if changed == 'vocabulary': source['feature_membership_sha256'] = '0'*64
    if changed == 'checkpoint': checkpoint = 'd'*64
    chosen, missing = resolve(d, (first, second), observations, requested='resource.one', source=source, checkpoint=checkpoint)
    assert chosen is None and missing


def test_unique_choice_ambiguity_and_explicit_precedence():
    from agent.application.epizoo_handoff import resolve, EpiZooDeclaration
    first, second, observations = model_choices()
    d = EpiZooDeclaration(operation='inference', scientific_name='Danio rerio', taxonomy_id=7955,
        assembly='danRer11', matrix_semantics='binary_accessibility')
    source = dict(feature_membership_sha256='f'*64)
    assert resolve(d, (first,), {'one': observations['one']}, source=source) == (first, [])
    chosen, missing = resolve(d, (first, second), observations, source=source)
    assert chosen is None and 'Choose' in missing[0]
    assert resolve(d, (first, second), observations, source=source, requested='resource.two') == (second, [])
    assert resolve(d, (first, second), observations, source=source, requested='resource.absent')[0] is None


def test_captured_compatible_model_precedes_catalog_but_not_explicit_choice():
    from dataclasses import replace
    from agent.application.epizoo_handoff import resolve, EpiZooDeclaration
    first, second, observations = model_choices()
    first = replace(first, execution_inputs=first.inputs() | dict(captured_revision_id='a'*64))
    declared = EpiZooDeclaration(operation='inference', scientific_name='Danio rerio', taxonomy_id=7955,
        assembly='danRer11', matrix_semantics='binary_accessibility')
    source = dict(feature_membership_sha256='f'*64)
    assert resolve(declared, (first, second), observations, source=source) == (first, [])
    assert resolve(declared, (first, second), observations, source=source, requested='resource.two') == (second, [])


def test_local_target_identity_cannot_fall_back_on_failed_acceptance(tmp_path, monkeypatch):
    from agent.orchestration import target_model_binding as target
    from agent.orchestration.prior_outputs import prior_output_store
    from agent.schemas.verification_authority import AuthorityError
    from types import SimpleNamespace
    store = SimpleNamespace(root=tmp_path, load=lambda _: object())
    calls = []
    def reject(actual, run, step):
        calls.append(actual)
        raise AuthorityError('Corrupt exact local acceptance')
    monkeypatch.setattr(target, '_accepted', reject)
    monkeypatch.setenv('AGENT_EPIZOO_SOURCE_STORE', str(tmp_path/'external'))
    with prior_output_store(store), pytest.raises(AuthorityError, match='Corrupt exact'):
        target.accepted_target('/model/manifest.json', 'a'*64, 'run', 'adapt')
    assert calls == [store] and not (tmp_path/'external').exists()


def test_mapped_mode_never_fabricates_mapping(native_choice):
    from dataclasses import replace
    from agent.application.epizoo_handoff import resolve, evidence
    d = replace(declaration(), strategy='mapped_reference')
    chosen, missing = resolve(d, (native_choice,), evidence((native_choice,)))
    assert chosen is None and 'target-to-source mapping' in missing[0]


def test_downstream_zero_and_invalid_training_values_use_registry(case, native_choice):
    from types import SimpleNamespace
    from agent.application.local_resources import RegisteredInputGroup, group_role_inputs
    from agent.application.dialogue_execution import _bind_arguments
    from agent.application.turn_decisions import ScopedArgument, IntentError
    from agent.orchestration.registry import build_default_tool_registry
    registry = build_default_tool_registry()
    zero = ScopedArgument('compute_cell_umap', 'min_dist', '0', 13, 14)
    values = _bind_arguments(registry, 'Use min_dist 0.', {}, (zero,))
    assert values == {'min_dist': 0}
    from agent.application.epizoo_handoff import binding
    members = [dict(resource_id='local-'+'a'*64,record_sha256='b'*64,source_path=case['source_path'],
        source_sha256=case['source_sha256'],input_type='h5ad',label='Explicit synthetic H5AD')]
    candidates = RegisteredInputGroup(members, {}, attachment_candidates=True)
    bound = binding(candidates, declaration(), native_choice, values, dict(
        manifest_path=case['reference_manifest_path'], manifest_sha256=case['reference_manifest_sha256']))
    assert bound.execution_inputs['min_dist'] == 0
    with pytest.raises(ValueError, match='conflict'):
        binding(candidates, declaration(), native_choice, {'assembly':'another'}, dict(
            manifest_path=case['reference_manifest_path'], manifest_sha256=case['reference_manifest_sha256']))
    # Exact complete resources cannot be replaced by a different dependency.
    from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
    owner = object.__new__(LocalResourceAdmission)
    owner._registry = registry
    step = SimpleNamespace(tool_name='adopt_scATAC_cell_by_features', arguments=dict(
        source_path=case['source_path'], source_sha256=case['source_sha256'],
        reference_manifest_path='/different/reference.json', reference_manifest_sha256='c'*64))
    with pytest.raises(ResourceAdmissionError, match='exact selected inputs'):
        owner._validate_group_consumption(bound, (step,))
    for key, value in [('max_steps','0'), ('learning_rate','-1'), ('training_batch_size','5')]:
        arg = ScopedArgument('adapt_epizoo_species', key, value, len(key)+5, len(key)+5+len(value))
        with pytest.raises(IntentError):
            _bind_arguments(registry, 'Use '+key+' '+value+'.', {}, (arg,))
