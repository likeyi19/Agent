"""Private resource upload/continuation; simulated science is explicitly bounded."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import test_conversational_handoff as h
from test_conversational_handoff import external_case, case
from agent.application import raw_handoff as handoff
from agent.application.input_sets import ScientificInputSet
from agent.tools.analysis import annotation_input, marker_annotation as science


@pytest.fixture
def resources(external_case, tmp_path):
    args, calls = external_case
    spec = json.loads(Path(args['annotation_spec_path']).read_text())
    scope = spec['contig_scope']
    choice = ScientificInputSet('genome', 'Qualified human hg38 genome scope',
        dict(qc_reference_manifest_path=scope['path'], qc_reference_manifest_sha256=scope['sha256']))
    from agent.tools.data.scatac_qc_reference import load_scatac_qc_reference_bundle
    _, bundle, _ = load_scatac_qc_reference_bundle(scope['path'], expected_sha256=scope['sha256'])
    chrom = next(r.name for r in bundle.contigs if r.classification == 'primary_nuclear_qc')
    genes = tmp_path/'uploaded-genes.tsv'
    genes.write_text('bin\tname\tchrom\tstrand\ttxStart\ttxEnd\tcdsStart\tcdsEnd\texonCount\texonStarts\texonEnds\tscore\tname2\tcdsStartStat\tcdsEndStat\texonFrames\n' +
        ''.join(f'0\tT{i}\t{chrom}\t+\t1\t9\t2\t8\t1\t1,\t9,\t0\t{g}\tcmpl\tcmpl\t0,\n' for i,g in enumerate(('G','H'))))
    signature = Path(spec['signature_resource']['path'])
    return args, calls, genes, signature, choice


def uploaded(**changes):
    return handoff.AnnotationUploads(gene_source='source.1', signature_source='source.2',
        gene_provenance='Explicit simulated gene fixture', gene_release='test-v1',
        signature_provenance='Explicit simulated signature fixture', signature_release='test-v1',
        species='human', assembly='hg38', context='synthetic context', gene_namespace='gene_symbol', **changes)


def decision(resource=None, *, pending=False):
    declaration = (handoff.AnnotationDeclaration(uploaded=resource) if pending else
        h.declared(grouping_provenance='Independently supplied simulated grouping', uploaded=resource))
    return dict(kind='answer_roles' if pending else 'request_roles',
        **(dict(pending='@roles') if pending else dict(target=h.TOOL, required_roles=[])), roles=[],
        annotation=asdict(declaration), arguments=[] if pending else
        [h.argument('count_slot','X'), h.argument('group_column','supplied_groups')])


def app_for(resources, tmp_path, monkeypatch, decisions, *, choices=None):
    args, calls, genes, signatures, scope = resources
    original = h.HandoffModel
    class Model(original):
        def complete(self, *, prompt, response_schema):
            value=json.loads(prompt)
            if 'turn_schema_version' in value:
                return json.dumps(dict(turn_schema_version=1, decision=decisions.pop(0)))
            return super().complete(prompt=prompt,response_schema=response_schema)
    monkeypatch.setattr(h,'HandoffModel',Model)
    return h.setup(args,tmp_path/'web',[],choices=(scope,) if choices is None else choices)


def attach(client, path, name):
    response=client.put('/api/v1/sessions/upload/uploads/'+name+'?filename='+path.name+'&input_type=attachment',
        content=path.read_bytes(),headers={'Content-Type':'application/octet-stream'})
    assert response.status_code==201,response.text
    return response.json()['resource_id']


OBJECTIVE = 'Annotate my counts using "X" and independent "supplied_groups". Use the uploaded references with their stated provenance.'


@pytest.mark.parametrize('supplement',[False,True])
def test_uploaded_components_without_annotation_inputset(resources,tmp_path,monkeypatch,supplement):
    args,calls,genes,signatures,scope=resources
    qualification_calls=[]
    qualify=annotation_input.qualify_resources
    def counted(spec):
        qualification_calls.append(1)
        return qualify(spec)
    monkeypatch.setattr(annotation_input,'qualify_resources',counted)
    decisions=[decision(None),decision(uploaded(),pending=True)] if supplement else [decision(uploaded())]
    app,web=app_for(resources,tmp_path,monkeypatch,decisions)
    with TestClient(web) as client:
        source=h.stage(client,args)
        if supplement:
            pending=h.send(client,'first',OBJECTIVE,[source])
            assert pending['status']=='clarification',pending
            assert 'refgenes' in pending['response']['text'] and not calls
        g=attach(client,genes,'genes');s=attach(client,signatures,'signatures')
        result=h.send(client,'reply' if supplement else 'first',
            'Use the supplied human hg38 gene-symbol references with the stated synthetic provenance and release.' if supplement else OBJECTIVE,
            [g,s] if supplement else [source,g,s])
        assert result['status']=='succeeded',result
        assert len(calls)==1
        run=app._application.run_store.load(result['run_id'])
        assert run.request.prompt==OBJECTIVE
        spec=json.loads(Path(run.steps[0].resolved_arguments['annotation_spec_path']).read_text())
        assert spec['gene_resource']['path']==app.resources.load(g).source_path
        assert spec['signature_resource']['path']==app.resources.load(s).source_path
        assert spec['contig_scope']==json.loads(Path(args['annotation_spec_path']).read_text())['contig_scope']
        assert not list(app.resources._workspace.root.glob('**/operator*'))
        before=len(calls)
        assert client.get('/api/v1/sessions/upload').status_code==200
        assert len(calls)==before
        assert qualification_calls==[1]


def test_explicit_upload_overrides_catalog_marker_resource(resources,tmp_path,monkeypatch):
    args,calls,genes,signatures,scope=resources
    app,web=app_for(resources,tmp_path,monkeypatch,[decision(uploaded())],choices=(h.choice(args),))
    with TestClient(web) as client:
        source=h.stage(client,args);g=attach(client,genes,'genes');s=attach(client,signatures,'signatures')
        done=h.send(client,'first',OBJECTIVE,[source,g,s])
        assert done['status']=='succeeded',done
        step=app._application.run_store.load(done['run_id']).steps[0]
        spec=json.loads(Path(step.resolved_arguments['annotation_spec_path']).read_text())
        assert spec['signature_resource']['path']==app.resources.load(s).source_path
        assert spec['signature_resource']['source']=='Explicit simulated signature fixture'


@pytest.mark.parametrize('new_matrix,single_wire', [(False, False), (True, False), (True, True)])
def test_prior_uploads_combine_with_current_scientific_companions(resources, tmp_path, monkeypatch, new_matrix, single_wire):
    args, calls, genes, signatures, _ = resources
    selected = decision(uploaded())
    if not new_matrix:
        selected['annotation']['source'] = 'source.2'
        selected['annotation']['uploaded'].update(gene_source='source.0', signature_source='source.1')
    app, web = app_for(resources, tmp_path, monkeypatch,
        [dict(kind='clarify', reason='ambiguous_subject'), selected])
    with TestClient(web) as client:
        source = h.stage(client, args)
        g, s = attach(client, genes, 'genes'), attach(client, signatures, 'signatures')
        historical, current = ([g, s], [source]) if new_matrix else ([source], [g, s])
        first = h.send(client, 'first', 'Keep these files for my next request.', historical)
        assert first['status'] == 'clarification' and not calls
        if single_wire:
            response = client.post('/api/v1/sessions/upload/turns', json=dict(
                turn_id='annotate', expected_generation=0, utterance=OBJECTIVE,
                resource_id=source, attachment_ids=current))
            assert response.status_code == 202, response.text
            done = h.wait_turn(client, 'annotate')
        else:
            done = h.send(client, 'annotate', OBJECTIVE, current)
        assert done['status'] == 'succeeded', done
        assert len(calls) == 1  # Simulated native production; completion does not replay.
        state = app._application.sessions.load('upload')
        interaction = state.interactions[-1]
        key = 'candidate_input' if single_wire else 'available_input'
        assert [m['resource_id'] for m in interaction.submission[key]['members']] == current + historical
        assert [a['resource_id'] for a in interaction.submission['attachments']] == current
        step = app._application.run_store.load(done['run_id']).steps[0]
        assert step.resolved_arguments['source_path'] == app.resources.load(source).source_path
        spec = json.loads(Path(step.resolved_arguments['annotation_spec_path']).read_text())
        assert spec['gene_resource']['path'] == app.resources.load(g).source_path
        assert spec['signature_resource']['path'] == app.resources.load(s).source_path


def test_explicit_current_h5ad_cannot_be_replaced_by_historical_candidate(resources, tmp_path, monkeypatch):
    args, calls, _, _, _ = resources
    selected = decision(None)
    selected['annotation']['source'] = 'source.1'
    app, web = app_for(resources, tmp_path, monkeypatch,
        [dict(kind='clarify', reason='ambiguous_subject'), selected], choices=(h.choice(args),))
    with TestClient(web) as client:
        previous = h.stage(client, args)
        h.send(client, 'first', 'Keep this file.', [previous])
        response = client.put('/api/v1/sessions/upload/uploads/current?filename=current.h5ad&input_type=h5ad',
            content=Path(args['source_path']).read_bytes(), headers={'Content-Type': 'application/octet-stream'})
        assert response.status_code == 201
        current = response.json()['resource_id']
        assert current != previous
        response = client.post('/api/v1/sessions/upload/turns', json=dict(turn_id='bad',
            expected_generation=0, utterance=OBJECTIVE, resource_id=current, attachment_ids=[current]))
        assert response.status_code == 202
        done = h.wait_turn(client, 'bad')
        assert done['status'] == 'clarification' and not calls
        assert app._application.sessions.load('upload').revisions == ()


@pytest.mark.parametrize('problem',['species','assembly','context','namespace','invalid_genes','invalid_markers','missing_provenance'])
def test_invalid_explicit_upload_clarifies_without_catalog_fallback(resources,tmp_path,monkeypatch,problem):
    args,calls,genes,signatures,scope=resources
    declaration=uploaded()
    if problem=='species':declaration=replace(declaration,species='mouse')
    if problem=='assembly':declaration=replace(declaration,assembly='mm10')
    if problem=='context':declaration=replace(declaration,context=None)
    if problem=='namespace':declaration=replace(declaration,gene_namespace='Ensembl')
    if problem=='missing_provenance':declaration=replace(declaration,signature_provenance=None)
    if problem=='invalid_genes':genes.write_text('not refgenes\n')
    if problem=='invalid_markers':
        signatures=tmp_path/'bad.tsv';signatures.write_text('candidate\tgene\nA\tWRONG1\nA\tWRONG2\n')
    app,web=app_for(resources,tmp_path,monkeypatch,[decision(declaration)],choices=(h.choice(args),))
    with TestClient(web) as client:
        source=h.stage(client,args);g=attach(client,genes,'genes');s=attach(client,signatures,'signatures')
        result=h.send(client,'first',OBJECTIVE,[source,g,s])
        assert result['status']=='clarification',result
        assert result['response']['clarification']['reason']=='missing_annotation_declarations'
        state=app._application.sessions.load('upload')
        assert state.interactions[-1].prerequisite is not None and not calls and not state.revisions


def test_private_reference_cannot_be_selected_in_another_session(resources,tmp_path,monkeypatch):
    args,_,genes,_,_=resources
    app,web=app_for(resources,tmp_path,monkeypatch,[])
    with TestClient(web) as client:
        h.stage(client,args);g=attach(client,genes,'genes')
        assert client.post('/api/v1/sessions',json={'session_id':'other'}).status_code==201
        response=client.post('/api/v1/sessions/other/turns',json=dict(turn_id='steal',expected_generation=0,
            utterance='Use the other Session reference.',attachment_ids=[g]))
        assert response.status_code in (400,404,422),response.text


def test_resource_changed_during_pending_continuation_fails_closed(resources,tmp_path,monkeypatch):
    args,calls,genes,signatures,_=resources
    app,web=app_for(resources,tmp_path,monkeypatch,[decision(replace(uploaded(),context=None)),
        decision(handoff.AnnotationUploads(context='synthetic context'),pending=True)])
    with TestClient(web) as client:
        source=h.stage(client,args);g=attach(client,genes,'genes');s=attach(client,signatures,'signatures')
        result=h.send(client,'first',OBJECTIVE,[source,g,s]);assert result['status']=='clarification'
        Path(app.resources.load(s).source_path).write_text('changed')
        result=h.send(client,'reply','The reference applies to the synthetic context.')
        assert result['status']=='failed',result
        assert not calls


def test_missing_provenance_survives_restart_and_requires_no_reupload(resources,tmp_path,monkeypatch):
    args,calls,genes,signatures,scope=resources
    decisions=[decision(replace(uploaded(),signature_provenance=None)),
        decision(handoff.AnnotationUploads(signature_provenance='Explicit simulated signature fixture'),pending=True)]
    app,web=app_for(resources,tmp_path,monkeypatch,decisions)
    with TestClient(web) as client:
        source=h.stage(client,args);g=attach(client,genes,'genes');s=attach(client,signatures,'signatures')
        result=h.send(client,'first',OBJECTIVE,[source,g,s])
        assert result['status']=='clarification' and not calls
    app,web=h.setup(args,tmp_path/'web',[],choices=(scope,))
    with TestClient(web) as client:
        result=h.send(client,'reply','The signatures come from the explicit simulated signature fixture.')
        assert result['status']=='succeeded',result
        assert len(calls)==1
        run=app._application.run_store.load(result['run_id'])
        assert run.request.prompt==OBJECTIVE


def test_distinct_retained_signatures_clarify_and_invalid_choice_does_not_fallback(resources,tmp_path):
    args,_,_,_,_=resources
    first=h.choice(args);second=h.alternative(args,tmp_path)
    declared=replace(h.declared(),species='human',assembly='hg38',measurement=h.c.EXTERNAL_SEMANTICS,
        uploaded=replace(uploaded(),signature_source=None))
    evidence=handoff.resource_evidence((first,second),'annotation')
    questions,selected=handoff._resolve_resource(declared,(first,second),evidence)
    assert selected is None and 'Choose a reference' in questions[0]
    selected_declared=replace(declared,resource='resource.alternative')
    assert handoff._resolve_resource(selected_declared,(first,second),evidence)==([],second)
    assert handoff._resolve_resource(replace(declared,resource='resource.absent'),(first,second),evidence)[1] is None


def test_bad_refgenes_exons_are_not_accepted(resources):
    args,_,genes,signatures,_=resources
    base=json.loads(Path(args['annotation_spec_path']).read_text())
    members=[{},dict(source_path=str(genes),source_sha256=science.sha256(genes)),
                dict(source_path=str(signatures),source_sha256=science.sha256(signatures))]
    declared=replace(h.declared(grouping_provenance='Independent simulated groups'),species='human',assembly='hg38',
        measurement=h.c.EXTERNAL_SEMANTICS,uploaded=uploaded())
    genes.write_text(genes.read_text().replace('\t1,\t9,\t','\t9,\t1,\t'))
    members[1]['source_sha256']=science.sha256(genes)
    spec,problems=annotation_input.resource_specification(base,declared,members)
    assert not problems
    with pytest.raises(ValueError,match='transcript/exon'):
        annotation_input.qualify_resources(spec)


def test_complete_uploaded_references_need_no_qc_catalog(resources, tmp_path, monkeypatch):
    args, calls, genes, signatures, _ = resources
    app, web = app_for(resources, tmp_path, monkeypatch, [decision(uploaded())], choices=())
    with TestClient(web) as client:
        source = h.stage(client, args)
        g = attach(client, genes, 'genes'); s = attach(client, signatures, 'signatures')
        done = h.send(client, 'first', OBJECTIVE, [source, g, s])
        assert done['status'] == 'succeeded', done
        assert len(calls) == 1
        step = app._application.run_store.load(done['run_id']).steps[0]
        spec = json.loads(Path(step.resolved_arguments['annotation_spec_path']).read_text())
        assert 'contig_scope' not in spec
        assert spec['gene_resource']['path'] == app.resources.load(g).source_path
        assert spec['signature_resource']['path'] == app.resources.load(s).source_path
        assert client.get('/api/v1/sessions/upload').status_code == 200
