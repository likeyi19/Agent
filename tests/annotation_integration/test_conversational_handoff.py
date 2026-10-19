"""USI-4 handoff contracts; fixture backend is simulated, not biological evidence."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import anndata as ad
import pytest
from fastapi.testclient import TestClient
from agent.application import raw_handoff as handoff
from agent.application.input_sets import ScientificInputSet
from agent.tools.analysis import annotation_contract as c, marker_annotation as science
from agent.tools.analysis.annotation_input import observe
from test_external_entry import (external_case, case, Model, wait_turn, wire,
    InteractiveAgentApplication, H5ADUploadAdmission, PlanningModelProfile,
    PlanningModelFactoryRegistry, create_app)

TOOL = 'annotate_scATAC_cell_types'
OBJECTIVE = 'Please annotate the cell types in my uploaded peak-count H5AD using MAESTRO and appropriate qualified human PBMC markers. Use X and an existing independent clustering. Save annotations in obs and provide the report and downloadable results.'


def choice(args, identity='reviewed'):
    return ScientificInputSet(identity, 'Reviewed human PBMC markers',
        {k: args[k] for k in ('annotation_spec_path', 'annotation_spec_sha256', 'annotation_method')})


def declared(**kwargs):
    return handoff.AnnotationDeclaration(source='source.0', context='synthetic context', **kwargs)


def argument(name, value):
    return dict(tool=TOOL, argument=name, literal=value, evidence='"'+value+'"')


class HandoffModel(Model):
    def complete(self, *, prompt, response_schema):
        p = json.loads(prompt); self.prompts.append(p)
        if 'turn_schema_version' in p:
            pending = p['dialogue'].get('pending_prerequisite')
            if pending:
                d = dict(kind='answer_roles', pending='@roles', roles=[],
                    annotation=asdict(handoff.AnnotationDeclaration(grouping_provenance='Independent simulated groups')),
                    arguments=[argument('group_column', 'supplied_groups')])
            else:
                d = dict(kind='request_roles', target=TOOL, required_roles=[], roles=[],
                    annotation=asdict(declared()), arguments=[argument('count_slot', 'X')])
            return json.dumps(dict(turn_schema_version=1, decision=d))
        if 'semantic_prompt_version' in p:
            value = wire(); sources = value['decision']['steps'][0]['sources']
            sources[:] = [r for r in sources if r['target'] != 'groups'] + [dict(target='group_column',source=dict(kind='input',input='group_column'))]
            return json.dumps(value)
        return super().complete(prompt=prompt, response_schema=response_schema)


def setup(args, root, prompts, *, choices=None):
    uploads = root/'uploads'; uploads.mkdir(parents=True, exist_ok=True)
    profile = PlanningModelProfile('scripted','scripted',Model.model_id)
    app = InteractiveAgentApplication(root, model_profiles=(profile,), default_profile_id='scripted',
        approved_source_roots=(uploads,), planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:HandoffModel(prompts)}))
    web = create_app(app, input_sets=choices if choices is not None else (choice(args),), uploads=H5ADUploadAdmission(app.resources, uploads))
    return app, web


def stage(client, args):
    assert client.post('/api/v1/sessions',json={'session_id':'upload'}).status_code == 201
    response = client.put('/api/v1/sessions/upload/uploads/matrix?filename=peaks.h5ad&input_type=h5ad',
        content=Path(args['source_path']).read_bytes(),headers={'Content-Type':'application/octet-stream'})
    assert response.status_code == 201, response.text
    return response.json()['resource_id']


def send(client, turn, text, ids=(), generation=0):
    response = client.post('/api/v1/sessions/upload/turns',json=dict(turn_id=turn,expected_generation=generation,
        utterance=text, attachment_ids=list(ids)))
    assert response.status_code == 202, response.text
    return wait_turn(client,turn)


def test_group_question_restart_continues_source_and_unique_resource(external_case,tmp_path):
    args,calls = external_case; prompts=[]; root=tmp_path/'chat'
    app,web=setup(args,root,prompts)
    with TestClient(web) as client:
        source=stage(client,args)
        first=send(client,'first',OBJECTIVE.replace('Use X','Use "X"'),[source])
        assert first['status']=='clarification', first
        assert 'supplied_groups' in first['response']['text']
        assert not calls
    app,web=setup(args,root,prompts)
    with TestClient(web) as client:
        done=send(client,'reply','Use the independent simulated obs clustering "supplied_groups".')
        assert done['status']=='succeeded',done
        assert len(calls)==2
        run=app._application.run_store.load(done['run_id'])
        assert run.request.prompt==OBJECTIVE.replace('Use X','Use "X"')
        assert run.steps[0].resolved_arguments['group_column']=='supplied_groups'
        assert run.steps[0].resolved_arguments['count_slot']=='X'
        assert run.steps[0].result['n_cells']==ad.read_h5ad(args['source_path']).n_obs
        assert app._application.sessions.load('upload').generation==1
    app,web=setup(args,root,prompts)
    with TestClient(web) as client:
        before=len(prompts)
        assert client.get('/api/v1/sessions/upload').status_code==200
        assert len(prompts)==before and len(calls)==2


def test_owner_observation_never_classifies_categorical_by_name(external_case):
    args,_=external_case
    observation=observe(args['source_path'],args['source_sha256'])
    assert 'supplied_groups' in observation['grouping_columns']
    assert observation['qualified_groupings']==[]
    a=ad.read_h5ad(args['source_path'])
    a.uns['grouping']={'supplied_groups':dict(role='clustering',provenance='Independently computed simulated clusters')}
    a.write_h5ad(args['source_path'])
    observation=observe(args['source_path'],science.sha256(args['source_path']))
    assert observation['qualified_groupings'][0]['column']=='supplied_groups'


def test_qualified_unique_group_is_automatic(external_case,tmp_path):
    args,calls=external_case
    a=ad.read_h5ad(args['source_path']);a.uns['grouping']={'supplied_groups':dict(role='clustering',provenance='Independent simulated groups')}
    a.write_h5ad(args['source_path'])
    app,web=setup(args,tmp_path/'auto',[])
    with TestClient(web) as client:
        source=stage(client,args)
        done=send(client,'first',OBJECTIVE.replace('Use X','Use "X"'),[source])
        assert done['status']=='succeeded',done
        assert len(calls)==2


def test_resource_resolution_checks_owner_context_and_pins(external_case):
    args,_=external_case; first=choice(args)
    d=replace(declared(),species='human',assembly='hg38',measurement=c.EXTERNAL_SEMANTICS)
    evidence=handoff.resource_evidence((first,),'annotation')
    assert handoff._resolve_resource(d,(first,),evidence)==([],first)
    assert handoff._resolve_resource(replace(d,context='brain'),(first,),evidence)[1] is None
    bad=replace(first,execution_inputs=first.inputs()|{'annotation_spec_sha256':'0'*64})
    assert handoff._resolve_resource(d,(bad,),handoff.resource_evidence((bad,),'annotation'))[1] is None
    assert handoff._resolve_resource(replace(d,resource='resource.absent'),(first,),evidence)[1] is None


def test_single_h5ad_browser_wire_uses_same_handoff(external_case,tmp_path):
    args,calls=external_case;prompts=[]
    app,web=setup(args,tmp_path/'single',prompts)
    with TestClient(web) as client:
        source=stage(client,args)
        response=client.post('/api/v1/sessions/upload/turns',json=dict(turn_id='first',expected_generation=0,
            utterance=OBJECTIVE.replace('Use X','Use "X"'),resource_id=source,attachment_ids=[source]))
        assert response.status_code==202,response.text
        first=wait_turn(client,'first')
        assert first['status']=='clarification',first
        assert 'supplied_groups' in first['response']['text']
        assert prompts[0]['dialogue']['annotation_sources'][0]['source']=='source.0'
        done=send(client,'reply','Use the independent simulated clustering "supplied_groups".')
        assert done['status']=='succeeded',done
        assert len(calls)==2


def alternative(args, root):
    spec=json.loads(Path(args['annotation_spec_path']).read_text())
    signatures=Path(spec['signature_resource']['path'])
    copied=root/'alternative-signatures.tsv';copied.write_bytes(signatures.read_bytes()+b'\n')
    spec['signature_resource']=dict(spec['signature_resource'],path=str(copied),sha256=science.sha256(copied))
    path=root/'alternative-resources.json';path.write_bytes(c.matrix.canonical(spec))
    return ScientificInputSet('alternative','Another qualified marker choice',dict(annotation_spec_path=str(path),
        annotation_spec_sha256=science.sha256(path),annotation_method=c.EXTERNAL_PRIMARY))


@pytest.mark.parametrize('missing_catalog',[False,True])
def test_resource_question_additive_supplement_and_explicit_selection(external_case,tmp_path,monkeypatch,missing_catalog):
    args,calls=external_case;prompts=[];root=tmp_path/'choices';first=choice(args);second=alternative(args,tmp_path)
    class ChoiceModel(HandoffModel):
        def complete(self,*,prompt,response_schema):
            p=json.loads(prompt)
            if 'turn_schema_version' in p:
                pending=p['dialogue'].get('pending_prerequisite')
                d=(dict(kind='answer_roles',pending='@roles',roles=[],arguments=[argument('count_slot','layers/raw_counts')],
                    annotation=asdict(handoff.AnnotationDeclaration(resource='resource.alternative'))) if pending else
                   dict(kind='request_roles',target=TOOL,required_roles=[],roles=[],annotation=asdict(declared(grouping_provenance='Independent simulated groups')),
                    arguments=[argument('count_slot','X'),argument('group_column','supplied_groups')]))
                return json.dumps(dict(turn_schema_version=1,decision=d))
            return super().complete(prompt=prompt,response_schema=response_schema)
    monkeypatch.setitem(HandoffModel.complete.__globals__,'HandoffModel',ChoiceModel)
    app,web=setup(args,root,prompts,choices=() if missing_catalog else (first,second))
    with TestClient(web) as client:
        source=stage(client,args)
        pending=send(client,'first','Annotate synthetic context using "X" and independent groups "supplied_groups".',[source])
        assert pending['status']=='clarification',pending
        assert not calls
        assert ('Upload a MAESTRO refgenes' if missing_catalog else 'Choose a reference') in pending['response']['text']
    app,web=setup(args,root,prompts,choices=(first,second))
    with TestClient(web) as client:
        done=send(client,'reply','Use the alternative marker resource and "layers/raw_counts".')
        assert done['status']=='succeeded',done
        assert len(calls)==2
        interaction=app._application.sessions.load('upload').interactions[-1]
        assert interaction.admitted['raw_resource_choice']=='resource.alternative'
        assert interaction.admitted['annotation_arguments']['count_slot']=='layers/raw_counts'


def test_ambiguous_qualified_group_and_explicit_precedence(external_case):
    from types import SimpleNamespace
    args,_=external_case
    a=ad.read_h5ad(args['source_path']);a.obs['other']=['second']+['first']*(a.n_obs-1)
    a.uns['grouping']={'supplied_groups':dict(role='clustering',provenance='Independent simulated grouping A'),'other':dict(role='clustering',provenance='Independent simulated grouping B')}
    a.uns['count_slot']='layers/raw_counts';a.write_h5ad(args['source_path'])
    sha=science.sha256(args['source_path']);observation=observe(args['source_path'],sha)
    binding=SimpleNamespace(members=[{'source_sha256':sha}])
    d,values,missing=handoff._annotation_requirements(binding,declared(),{'count_slot':'X'},observation)
    assert values['count_slot']=='X' and 'group_column' not in values
    assert any('Choose' in q for q in missing)
    d,values,missing=handoff._annotation_requirements(binding,declared(),{'count_slot':'X','group_column':'other'},observation)
    assert not missing and d.grouping_provenance.endswith('B')
    d,values,missing=handoff._annotation_requirements(binding,declared(),{'count_slot':'X','group_column':'absent'},observation)
    assert values['group_column']=='absent' and any('cannot be replaced' in q for q in missing)


def test_incompatible_measurements_preserve_no_execution(external_case,tmp_path):
    args,calls=external_case
    a=ad.read_h5ad(args['source_path']);a.uns['measurement']='tfidf';a.write_h5ad(args['source_path'])
    app,web=setup(args,tmp_path/'bad-counts',[])
    with TestClient(web) as client:
        source=stage(client,args)
        result=send(client,'first',OBJECTIVE.replace('Use X','Use "X"'),[source])
        assert result['status']=='clarification',result
        assert 'incompatible' in result['response']['text'] and not calls
        state=app._application.sessions.load('upload');assert not state.revisions and not state.turns


def test_pending_source_mutation_and_parameter_substitution_fail_closed(external_case,tmp_path):
    args,calls=external_case
    app,web=setup(args,tmp_path/'changed',[])
    with TestClient(web) as client:
        source=stage(client,args)
        pending=send(client,'first',OBJECTIVE.replace('Use X','Use "X"'),[source]);assert pending['status']=='clarification'
        record=app.resources.load(source);Path(record.source_path).write_bytes(b'changed')
        result=send(client,'reply','Use "supplied_groups" as the independent simulated grouping.')
        assert result['status']=='failed' and not calls,result


def test_unsupported_method_parameter_is_not_silently_ignored():
    from agent.application.turn_decisions import parse_decision
    text='Use decay 0.'
    raw=json.dumps(dict(turn_schema_version=1,decision=dict(kind='request_roles',target=TOOL,
        required_roles=[],roles=[],annotation=asdict(declared()),
        arguments=[dict(tool=TOOL,argument='decay',literal='0',evidence='0')])) )
    # The structural declaration parses; the existing Registry admission rejects
    # this unsupported scientific parameter rather than changing the method.
    decision=parse_decision(raw,utterance=text)
    from agent.application.dialogue_execution import _bind_arguments
    from agent.orchestration import build_default_tool_registry
    from agent.application.turn_decisions import IntentError
    with pytest.raises(IntentError):_bind_arguments(build_default_tool_registry(),text,{},decision.arguments)


def test_reannotation_parameter_change_navigation_and_old_revision_integrity(external_case,tmp_path,monkeypatch):
    from agent.schemas.orchestration import _serialize
    args,calls=external_case;root=tmp_path/'reannotation';prompts=[]
    a=ad.read_h5ad(args['source_path']);a.uns['grouping']={'supplied_groups':dict(role='clustering',provenance='Independent simulated groups')}
    a.write_h5ad(args['source_path'])
    class ReannotationModel(HandoffModel):
        def complete(self,*,prompt,response_schema):
            p=json.loads(prompt)
            if 'turn_schema_version' in p:
                slot='layers/raw_counts' if 'layers/raw_counts' in p['utterance'] else 'X'
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='request_roles',target=TOOL,
                    required_roles=[],roles=[],annotation=asdict(declared()),arguments=[argument('count_slot',slot)])))
            return super().complete(prompt=prompt,response_schema=response_schema)
    monkeypatch.setitem(HandoffModel.complete.__globals__,'HandoffModel',ReannotationModel)
    app,web=setup(args,root,prompts)
    with TestClient(web) as client:
        source=stage(client,args)
        first=send(client,'first','Annotate independent simulated groups using "X".',[source]);assert first['status']=='succeeded',first
        run=app._application.run_store.load(first['run_id']);before=run.to_dict()
        published=Path(run.steps[0].result['annotated_h5ad_path']);original_digest=science.sha256(published)
        second=send(client,'second','Reannotate the same source using "layers/raw_counts".',[source],generation=1)
        assert second['status']=='succeeded',second
        new=app._application.run_store.load(second['run_id'])
        assert new.steps[0].resolved_arguments['count_slot']=='layers/raw_counts'
        assert new.steps[0].resolved_arguments['source_sha256']==run.steps[0].resolved_arguments['source_sha256']
        assert app._application.run_store.load(first['run_id']).to_dict()==before
        assert science.sha256(published)==original_digest and len(calls)==4
        state=app._application.sessions.load('upload')
        assert len(state.revisions)==2 and state.revisions[1].parent_revision_id==first['revision_id']
        response=client.post('/api/v1/sessions/upload/activate',json=dict(turn_id='back',revision_id=first['revision_id'],expected_generation=2))
        assert response.status_code==200,response.text
        response=client.get('/api/v1/sessions/upload/revisions/'+second['revision_id']+'/scientific-artifacts')
        assert response.status_code==200,response.text
        artifact,=response.json()['artifacts']
        downloaded=client.get('/api/v1/sessions/upload/revisions/'+second['revision_id']+'/scientific-artifacts/'+artifact['handle'])
        assert downloaded.status_code==200 and len(calls)==4
    app,web=setup(args,root,prompts)
    with TestClient(web) as client:
        n=len(prompts);assert client.get('/api/v1/sessions/upload').status_code==200
        assert len(prompts)==n and len(calls)==4
        assert app._application.sessions.load('upload').active_revision_id==first['revision_id']


def test_group_file_supplement_uses_same_pending_task(external_case,tmp_path,monkeypatch):
    args,calls=external_case
    class GroupModel(HandoffModel):
        def complete(self,*,prompt,response_schema):
            p=json.loads(prompt)
            if 'turn_schema_version' in p:
                if p['dialogue'].get('pending_prerequisite'):
                    d=dict(kind='answer_roles',pending='@roles',roles=[],arguments=[],
                        annotation=asdict(handoff.AnnotationDeclaration(groups_source='source.1',grouping_provenance='Independent supplied simulated groups')))
                else:
                    d=dict(kind='request_roles',target=TOOL,required_roles=[],roles=[],annotation=asdict(declared()),arguments=[argument('count_slot','X')])
                return json.dumps(dict(turn_schema_version=1,decision=d))
            if 'semantic_prompt_version' in p:return json.dumps(wire())
            return super().complete(prompt=prompt,response_schema=response_schema)
    monkeypatch.setitem(HandoffModel.complete.__globals__, 'HandoffModel', GroupModel)
    app,web=setup(args,tmp_path/'group-file',[])
    with TestClient(web) as client:
        source=stage(client,args)
        first=send(client,'first','Annotate independent simulated groups using "X".',[source])
        assert first['status']=='clarification' and not calls
        response=client.put('/api/v1/sessions/upload/uploads/groups?filename=groups.tsv&input_type=attachment',
            content=Path(args['groups_path']).read_bytes(),headers={'Content-Type':'application/octet-stream'})
        assert response.status_code==201,response.text
        done=send(client,'reply','The new attachment supplies the independent ordered cell groups.',[response.json()['resource_id']])
        assert done['status']=='succeeded',done
        assert len(calls)==2
        run=app._application.run_store.load(done['run_id'])
        assert run.request.prompt=='Annotate independent simulated groups using "X".'
        assert 'groups_path' in run.steps[0].resolved_arguments and 'group_column' not in run.steps[0].resolved_arguments


def test_annotation_declarations_enforce_the_published_text_bound():
    from agent.application.turn_decisions import IntentError, parse_decision
    for name in ('species', 'assembly', 'measurement', 'context', 'grouping_provenance'):
        declaration = asdict(declared()) | {name: 'x' * 129}
        raw = json.dumps(dict(turn_schema_version=1, decision=dict(kind='request_roles',
            target=TOOL, required_roles=[], roles=[], annotation=declaration)))
        with pytest.raises(IntentError, match='invalid_decision'):
            parse_decision(raw)


def test_unoffered_source_fails_before_observation(external_case,tmp_path,monkeypatch):
    args,calls=external_case
    class UnofferedModel(HandoffModel):
        def complete(self, *, prompt, response_schema):
            value = json.loads(super().complete(prompt=prompt, response_schema=response_schema))
            if 'annotation' in value.get('decision', {}):
                value['decision']['annotation']['source'] = 'source.1'
            return json.dumps(value)
    monkeypatch.setitem(HandoffModel.complete.__globals__, 'HandoffModel', UnofferedModel)
    from agent.tools.analysis import annotation_input
    def forbidden(*args, **kwargs):
        pytest.fail('An unoffered source must be rejected before observing scientific inputs')
    monkeypatch.setattr(annotation_input, 'observe', forbidden)
    app,web=setup(args,tmp_path/'unoffered',[])
    with TestClient(web) as client:
        source=stage(client,args)
        result=send(client,'first',OBJECTIVE.replace('Use X','Use "X"'),[source])
        assert result['status']=='clarification',result
        assert result['response']['clarification']['reason']=='invalid_decision'
        assert not calls and not app._application.sessions.load('upload').revisions


def test_pending_annotation_correction_preserves_source_and_history(external_case, tmp_path, monkeypatch):
    """A prose measurement/context can be corrected without replacing the upload."""
    args, calls = external_case
    a = ad.read_h5ad(args['source_path'])
    for key in ('species', 'assembly', 'measurement', 'context'):
        a.uns.pop(key, None)
    a.write_h5ad(args['source_path'])

    class CorrectingModel(HandoffModel):
        def complete(self, *, prompt, response_schema):
            p = json.loads(prompt)
            value = json.loads(super().complete(prompt=prompt, response_schema=response_schema))
            if 'turn_schema_version' in p:
                declaration = value['decision']['annotation']
                declaration.update(species='human', assembly='hg38')
                if p['dialogue'].get('pending_prerequisite'):
                    declaration.update(measurement=c.EXTERNAL_SEMANTICS, context='synthetic context')
                else:
                    declaration.update(measurement='Cell Ranger cut-site counts', context='test tissue')
            return json.dumps(value)

    monkeypatch.setitem(HandoffModel.complete.__globals__, 'HandoffModel', CorrectingModel)
    root = tmp_path/'correction'; app, web = setup(args, root, [])
    with TestClient(web) as client:
        resource = stage(client, args)
        pending = send(client, 'first', 'Annotate using "X".', [resource])
        assert pending['status'] == 'clarification' and not calls
        before = app._application.sessions.load('upload').interactions
    # A fresh application must replay the original uncorrected question exactly.
    app, web = setup(args, root, [])
    with TestClient(web) as client:
        done = send(client, 'reply', 'These are raw Cell Ranger ATAC cut-site counts in the synthetic context; use "supplied_groups".')
        assert done['status'] == 'succeeded', done
        assert len(calls) == 2
        state = app._application.sessions.load('upload')
        assert state.interactions[:len(before)] == before
        run = app._application.run_store.load(done['run_id'])
        resolved = run.steps[0].resolved_arguments
        assert resolved['source_sha256'] == app.resources.load(resource).source_sha256
        spec = json.loads(Path(resolved['annotation_spec_path']).read_text())
        assert spec['measurement'] == c.EXTERNAL_SEMANTICS
        assert spec['context'] == 'synthetic context'
    app, web = setup(args, root, [])
    with TestClient(web) as client:
        assert client.get('/api/v1/sessions/upload').status_code == 200
        assert len(calls) == 2


def test_annotation_corrections_cannot_replace_sources_or_source_metadata():
    previous = handoff.AnnotationDeclaration(source='source.0', groups_source='source.1',
        species='human', assembly='hg38', measurement='tfidf', context='blood')
    corrected = handoff.merge(previous, handoff.AnnotationDeclaration(measurement=c.EXTERNAL_SEMANTICS),
        annotation_corrections=True)
    assert corrected.source == previous.source and corrected.groups_source == previous.groups_source
    for key in ('source', 'groups_source'):
        with pytest.raises(ValueError, match='cannot be replaced'):
            handoff.merge(previous, handoff.AnnotationDeclaration(**{key: 'source.2'}),
                annotation_corrections=True)
    # Metadata merging remains strict even after a typed correction.
    from types import SimpleNamespace
    observation = dict(source_sha256='a'*64, declarations={'measurement': 'tfidf'},
        grouping_columns=['clusters'], qualified_groupings=[])
    with pytest.raises(ValueError, match='cannot be replaced'):
        handoff._annotation_requirements(SimpleNamespace(members=[{'source_sha256': 'a'*64}]),
            replace(corrected, groups_source=None), {'count_slot':'X', 'group_column':'clusters'}, observation)


def test_annotation_schema_exposes_measurement_contract_without_relabelling():
    schema = handoff.schema([{'source':'source.0'}], [], 'annotation')
    assert c.EXTERNAL_SEMANTICS in schema['properties']['measurement']['description']
    assert schema['properties']['measurement']['anyOf'] == [
        {'type': 'string', 'maxLength': 128}, {'type': 'null'}]
    for measurement in (c.EXTERNAL_SEMANTICS, 'binary accessibility', 'fragment overlap counts', 'tfidf', None):
        value = asdict(handoff.AnnotationDeclaration(source='source.0', measurement=measurement))
        assert handoff.declaration(value, 'annotation').measurement == measurement
