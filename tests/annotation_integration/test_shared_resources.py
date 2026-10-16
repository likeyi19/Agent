"""Simulated sharing policy/software witnesses; no public biology qualification."""
from dataclasses import replace
from pathlib import Path
import json
import pytest
from fastapi.testclient import TestClient
import test_uploaded_resources as u
from test_uploaded_resources import resources, external_case, case
from agent.application import shared_resources as shared, reference_supplementation as refs
from agent.application.local_resources import LocalResourceAdmission
from agent.schemas.orchestration import _serialize


def permissions(record, tmp_path, *, files=None):
    evidence = tmp_path / 'simulated-permission.txt'
    evidence.write_text('Synthetic acceptance fixtures authored for this test; sharing permitted only inside this isolated test. Not public scientific qualification.\n')
    return [dict(sha256=sha, source_url='https://example.org/simulated-fixture',
        license_url='https://example.org/simulated-fixture/permission', evidence_path=str(evidence),
        evidence_sha256=u.science.sha256(evidence), basis='Operator-reviewed simulated fixture; isolated software acceptance only.')
        for sha in sorted({r['sha256'] for r in shared.redistribution_files(files or record.reference_qualification['files'])})]


def private(resources, tmp_path, monkeypatch, decisions=None):
    args, calls, genes, signatures, scope = resources
    decisions = [u.decision(u.uploaded())] if decisions is None else decisions
    app, web = u.app_for(resources, tmp_path, monkeypatch, decisions)
    with TestClient(web) as client:
        source = u.h.stage(client, args)
        g = u.attach(client, genes, 'genes'); s = u.attach(client, signatures, 'signatures')
        done = u.h.send(client, 'first', u.OBJECTIVE, [source, g, s])
        assert done['status'] == 'succeeded', done
    rows = [r for r in refs._records(app.resources) if r.input_type == 'scientific_reference']
    assert len(rows) == 1
    return app, rows[0], done


def test_private_persistence_promotion_and_independent_user_discovery(resources, tmp_path, monkeypatch):
    decisions = [u.decision(u.uploaded()), u.decision(None), u.decision(None)]
    app, record, done = private(resources, tmp_path, monkeypatch, decisions)
    before = Path(record.source_path).read_bytes()
    assert shared.discover(app.resources, 'upload')
    assert shared.discover(app.resources, 'other') == ()
    with pytest.raises(ValueError):
        shared.check_choice(app.resources, 'other', 'resource.' + shared.input_set(record).input_set_id)
    with pytest.raises(ValueError):
        refs.promote(app.resources, record.resource_id, operator='operator', provenance='filename says public', reuse_basis='user says share')
    rows = permissions(record, tmp_path)
    promoted = refs.promote(app.resources, record.resource_id, operator='fixture operator',
        provenance='Synthetic author permission reviewed for exact bytes', reuse_basis='Isolated tests', redistribution=rows)
    assert app.resources.load(record.resource_id) == record
    assert Path(record.source_path).read_bytes() == before
    assert record.session_id == 'upload' and promoted.session_id is None
    assert shared.discover(app.resources, 'other') == (shared.input_set(promoted),)
    # New application and independent Session: no configured InputSet, no reference upload.
    app, web = u.h.setup(resources[0], tmp_path/'web', [], choices=())
    with TestClient(web) as client:
        own_source = next(r for r in app.resources.list_records(source_root=tmp_path/'web'/'uploads')
                          if r.input_type == 'h5ad' and r.session_id == 'upload')
        repeated = u.h.send(client, 'new-task', u.OBJECTIVE, [own_source.resource_id], generation=1)
        assert repeated['status'] == 'succeeded', repeated
        assert client.post('/api/v1/sessions', json={'session_id':'other'}).status_code == 201
        source = client.put('/api/v1/sessions/other/uploads/matrix?filename=counts.h5ad&input_type=h5ad',
            content=Path(resources[0]['source_path']).read_bytes(), headers={'content-type':'application/octet-stream'}).json()['resource_id']
        response = client.post('/api/v1/sessions/other/turns', json=dict(turn_id='second', expected_generation=0,
            utterance=u.OBJECTIVE, attachment_ids=[source]))
        assert response.status_code == 202, response.text
        import time
        deadline = time.monotonic() + 60
        while True:
            result = client.get('/api/v1/sessions/other/turns/second').json()
            if result['status'] in ('succeeded', 'failed', 'clarification'): break
            assert time.monotonic() < deadline
            time.sleep(.05)
        assert result['status'] == 'succeeded', result
        assert len(resources[1]) == 6
        # Neither original private record nor shared manifests are upload handles.
        response = client.post('/api/v1/sessions/other/turns', json=dict(turn_id='steal', expected_generation=1,
            utterance='Use private reference', attachment_ids=[record.resource_id]))
        assert response.status_code in (400,404,422), response.text
    reopened = LocalResourceAdmission(app.resources._workspace, approved_source_roots=app.resources._approved)
    monkeypatch.setattr(u.annotation_input, 'qualify_resources', lambda *a: pytest.fail('Discovery repeated scientific qualification'))
    assert shared.discover(reopened, 'third') == (shared.input_set(promoted),)
    assert app._application.sessions.load('other').revisions


def test_compatibility_version_private_scope_and_permission_integrity(resources, tmp_path, monkeypatch):
    app, record, _ = private(resources, tmp_path, monkeypatch)
    choice = shared.input_set(record)
    evidence = u.handoff.resource_evidence((choice,), 'annotation')
    declared = replace(u.h.declared(), species='human', assembly='hg38', measurement=u.h.c.EXTERNAL_SEMANTICS)
    assert u.handoff._resolve_resource(declared, (choice,), evidence)[1] == choice
    for key, value in [('species','mouse'), ('assembly','mm10'), ('context','unrelated tissue'), ('measurement','binary')]:
        assert u.handoff._resolve_resource(replace(declared, **{key:value}), (choice,), evidence)[1] is None
    q = _serialize(record.reference_qualification); q['profile'] = 'future-reference.v99'
    with pytest.raises(ValueError): replace(record, reference_qualification=q)
    q = _serialize(record.reference_qualification); q['scope'] = 'deployment'
    with pytest.raises(ValueError): replace(record, reference_qualification=q, session_id=None)
    rows = permissions(record, tmp_path)
    for invalid in (rows[:-1], [dict(row, sha256='a'*64) for row in rows]):
        with pytest.raises(ValueError):
            shared.promote(app.resources, record.resource_id, operator='fixture', provenance='simulated', reuse_basis='test', redistribution=invalid)
    promoted = shared.promote(app.resources, record.resource_id, operator='fixture', provenance='simulated', reuse_basis='test', redistribution=rows)
    Path(rows[0]['evidence_path']).write_text('permission evidence changed')
    with pytest.raises(ValueError): shared.check_choice(app.resources, 'other', 'resource.' + shared.input_set(promoted).input_set_id)


def test_qc_reference_reuses_existing_owner_across_task_roles(resources, tmp_path):
    args, _, _, _, scope = resources
    root = tmp_path/'catalog'; root.mkdir()
    # Existing qualified QC sources are explicitly approved in this isolated fixture.
    admission = LocalResourceAdmission(root, approved_source_roots=(tmp_path,))
    record = shared.persist(admission, scope.inputs(), session_id='one', origin_turn_id='qc', label='Simulated qualified QC reference')
    choice = shared.discover(admission, 'one')[0]
    assert choice.inputs() == scope.inputs()
    assert shared.discover(admission, 'two') == ()
    observation = u.annotation_input.resource_base(choice.inputs())
    assert observation['species'] == 'human' and observation['assembly'] == 'hg38'
    # These exact same pointers remain ordinary QC inputs; no annotation-specific copy.
    assert {'qc_reference_manifest_path','qc_reference_manifest_sha256'} <= set(choice.inputs())
    promoted = shared.promote(admission, record.resource_id, operator='fixture', provenance='simulated',
        reuse_basis='QC and annotation scope software test', redistribution=permissions(record,tmp_path))
    assert shared.discover(admission, 'two')[0].inputs() == choice.inputs()
    Path(record.source_path).write_bytes(b'corrupt')
    with pytest.raises(ValueError): shared.check_choice(admission, 'two', 'resource.' + shared.input_set(promoted).input_set_id)


def test_permission_coverage_cannot_be_bypassed_by_a_filename(tmp_path):
    data = tmp_path/'scientific_data.py'
    data.write_text('scientific input, not implementation')
    row = dict(path=str(data), sha256=u.science.sha256(data), size_bytes=data.stat().st_size)
    assert shared.redistribution_files([row]) == [row]
    with pytest.raises(ValueError):
        shared.review('operator', 'public filename', 'user assertion', [], [row])


def test_changed_genome_dependency_cannot_be_registered_as_qc(resources, tmp_path):
    from agent.tools.data.scatac_qc_reference import load_scatac_qc_reference_bundle
    from agent.tools.data._fragment_reference import load
    scope = resources[-1]
    values = scope.inputs()
    bundle = load_scatac_qc_reference_bundle(values['qc_reference_manifest_path'])[1]
    parent = load(bundle.parent_manifest.path, bundle.parent_manifest.sha256)
    Path(parent.genome.fasta.path).write_bytes(b'>changed\nACGT\n')
    root = tmp_path/'catalog'; root.mkdir()
    admission = LocalResourceAdmission(root, approved_source_roots=(tmp_path,))
    with pytest.raises(ValueError):
        shared.persist(admission, values, session_id='one', origin_turn_id='qc', label='Invalid QC dependency')
    assert not shared.discover(admission, 'one')


def test_annotation_reuse_does_not_reintroduce_upstream_fasta_or_ccre_gate(resources, tmp_path, monkeypatch):
    from agent.tools.data.scatac_qc_reference import load_scatac_qc_reference_bundle
    from agent.tools.data._fragment_reference import load
    bundle = load_scatac_qc_reference_bundle(resources[-1].inputs()['qc_reference_manifest_path'])[1]
    parent = load(bundle.parent_manifest.path, bundle.parent_manifest.sha256)
    Path(parent.genome.fasta.path).unlink()
    Path(parent.ccre.bed.path).unlink()
    app, record, _ = private(resources, tmp_path, monkeypatch)
    assert shared.check_record(app.resources, record) == shared.input_set(record).inputs()


def test_publication_uses_the_configured_attachment_root(resources, tmp_path, monkeypatch):
    app, record, _ = private(resources, tmp_path, monkeypatch)
    custom = tmp_path/'web'/'custom-attachments'; custom.mkdir()
    admission = LocalResourceAdmission(app.resources._workspace, approved_source_roots=(custom,))
    published = shared.persist(admission, shared.input_set(record).inputs(), session_id='different',
        origin_turn_id='task', label='Private reusable references')
    assert Path(published.source_path).is_relative_to(custom)
    assert shared.discover(admission, 'different') == (shared.input_set(published),)
    multi = LocalResourceAdmission(app.resources._workspace, approved_source_roots=(custom,tmp_path/'web'/'uploads'))
    spec = json.loads(Path(record.source_path).read_text())
    bound = shared.persist(multi, shared.input_set(record).inputs(), session_id='third', origin_turn_id='task',
        label='Private references', publication_source=spec['signature_resource']['path'])
    assert Path(bound.source_path).is_relative_to(tmp_path/'web'/'uploads')


def test_operator_resource_labels_do_not_claim_shared_catalog_identity():
    assert shared.check_choice(None, 'session', 'resource.qualified-human') is None
