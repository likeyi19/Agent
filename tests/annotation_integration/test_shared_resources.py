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


def test_explicit_installation_private_isolation_and_reuse(resources, tmp_path, monkeypatch):
    app, record, done = private(resources, tmp_path, monkeypatch)
    library = shared.ScientificResourceLibrary(tmp_path/'library')
    app.resources.library = library
    before = Path(record.source_path).read_bytes()
    assert shared.discover(app.resources, 'other') == ()
    with pytest.raises(ValueError): shared.install_private(app.resources, 'other', record.resource_id, shared_use=True)
    with pytest.raises(ValueError): shared.install_private(app.resources, 'upload', record.resource_id, shared_use=False)
    entry = shared.install_private(app.resources, 'upload', record.resource_id, shared_use=True)
    assert len(library.entries()) == 1
    assert Path(record.source_path).read_bytes() == before
    assert app.resources.load(record.resource_id) == record
    monkeypatch.setattr(u.annotation_input, 'qualify_resources', lambda *a: pytest.fail('Repeated scientific qualification'))
    assert shared.install_private(app.resources, 'upload', record.resource_id, shared_use=True) == entry
    reopened = LocalResourceAdmission(app.resources._workspace, approved_source_roots=app.resources._approved,
        scientific_resource_root=library.root)
    assert shared.discover(reopened, 'another') == library.choices()
    choice = library.input_set(entry)
    assert shared.check_choice(reopened, 'another', 'resource.'+choice.input_set_id) == choice.inputs()
    assert all('/src/agent/' not in f['path'] for f in entry['files'])
    # Public installation endpoints require the originating Session and explicit intent.
    from agent.web.app import create_app
    with TestClient(create_app(app)) as client:
        assert client.post('/api/v1/sessions', json={'session_id':'another'}).status_code == 201
        url='/api/v1/sessions/another/references/'+record.resource_id+'/share'
        assert client.post(url,json={'shared_use':True}).status_code == 400
        url='/api/v1/sessions/upload/references/'+record.resource_id+'/share'
        assert client.post(url,json={'shared_use':False}).status_code == 400
        assert client.post(url,json={'shared_use':True}).status_code == 200


def test_library_compatibility_ambiguity_and_corruption(resources, tmp_path, monkeypatch):
    app, record, _ = private(resources, tmp_path, monkeypatch)
    app.resources.library = shared.ScientificResourceLibrary(tmp_path/'library')
    entry = shared.install_private(app.resources, 'upload', record.resource_id, shared_use=True)
    choice = app.resources.library.input_set(entry)
    evidence = u.handoff.resource_evidence((choice,), 'annotation')
    declared = replace(u.h.declared(), species='human', assembly='hg38', measurement=u.h.c.EXTERNAL_SEMANTICS)
    assert u.handoff._resolve_resource(declared, (choice,), evidence)[1] == choice
    for key, value in [('species','mouse'), ('assembly','mm10'), ('context','unrelated tissue'), ('measurement','binary')]:
        questions, selected = u.handoff._resolve_resource(replace(declared, resource='resource.' + choice.input_set_id,
            **{key:value}), (choice,), evidence)
        assert selected is None and key in questions[0]
        if key == 'measurement':
            assert 'Replacing a reference does not change the input measurement' in questions[0]
            assert 'Supply a corrected resource' not in questions[0]
    duplicate = replace(choice, input_set_id='another', execution_inputs=dict(choice.inputs(), annotation_spec_sha256='b'*64))
    assert u.handoff._resolve_resource(declared, (choice,duplicate), dict(evidence,another=evidence[choice.input_set_id]))[1] is None
    original = app.resources.library.path.read_bytes()
    index=json.loads(original);index['resources'][0]['label']='tampered'
    app.resources.library.path.write_text(json.dumps(index))
    with pytest.raises(ValueError): app.resources.library.entries()
    app.resources.library.path.write_bytes(original)
    Path(entry['resource']['signature_resource']['path']).write_text('corrupt')
    with pytest.raises(ValueError): shared.check_choice(app.resources,'other','resource.'+choice.input_set_id)


def test_qc_reference_reuses_existing_owner_across_task_roles(resources, tmp_path, monkeypatch):
    library = shared.ScientificResourceLibrary(tmp_path/'library')
    entry=library.install('qc', resources[-1].inputs(), label='Simulated QC', version='test',
        provenance='Synthetic fixture', usage='Isolated test only')
    from agent.tools.data import scatac_qc_reference as owner
    monkeypatch.setattr(owner,'reinspect_scatac_qc_reference_bundle_sources',lambda *a:pytest.fail('Repeated QC qualification'))
    assert library.install('qc', resources[-1].inputs(), label='Simulated QC', version='test',
        provenance='Synthetic fixture', usage='Isolated test only') == entry
    assert library.input_set(entry).inputs() == resources[-1].inputs()
    library.check(entry)
    Path(entry['resource']['qc_reference_manifest_path']).write_text('corrupt')
    with pytest.raises(ValueError): library.check(entry)


def test_qualified_qc_parent_offered_to_exact_fragment_and_bam_roles(resources, tmp_path, monkeypatch):
    from agent.application import raw_handoff as handoff
    from agent.tools.data import scatac_qc_reference as owner
    library = shared.ScientificResourceLibrary(tmp_path/'library')
    entry = library.install('qc', resources[-1].inputs(), label='Simulated QC', version='test',
        provenance='Synthetic fixture', usage='Isolated test only')
    monkeypatch.setattr(owner, 'reinspect_scatac_qc_reference_bundle_sources',
        lambda *a: pytest.fail('Discovery reconstructed native science'))
    choices = library.choices()
    assert len(choices) == 3 and choices[0] == library.input_set(entry)
    fragments, = handoff.resources(choices, 'fragments')
    bam, = handoff.resources(choices, 'bam')
    assert not handoff.resources(choices, 'fastq')  # No whitelist or inferred processing context.
    assert fragments.inputs() == bam.inputs()
    bundle = owner.load_scatac_qc_reference_bundle(entry['resource']['qc_reference_manifest_path'])[1]
    assert fragments.inputs()['reference_bundle_sha256'] == bundle.parent_manifest.sha256
    declared = handoff.FragmentsDeclaration(species='human', assembly='hg38')
    evidence = handoff.resource_evidence(choices, 'fragments')
    assert handoff._resolve_resource(declared, (fragments,), evidence)[1] == fragments
    for declaration in (replace(declared, species='mouse', assembly='mm10'),
                        replace(declared, species=None, assembly='mm10')):
        assert handoff._resolve_resource(declaration, (fragments,), evidence)[1] is None
    admission = LocalResourceAdmission(tmp_path/'catalog', approved_source_roots=(tmp_path,),
        scientific_resource_root=library.root)
    for choice in (fragments, bam):
        assert shared.check_choice(admission, 'any-session', 'resource.' + choice.input_set_id) == choice.inputs()
    Path(bundle.parent_manifest.path).write_text('substituted')
    assert library.choices() == (library.input_set(entry),)
    with pytest.raises(ValueError): shared.check_choice(admission, 'any', 'resource.' + fragments.input_set_id)


def test_private_qc_parent_scope_and_loose_files_never_grant_reuse(resources, tmp_path):
    admission = LocalResourceAdmission(tmp_path/'catalog', approved_source_roots=(tmp_path,))
    assert shared.discover(admission, 'one') == ()  # Fixture files exist, but are not admitted.
    record = shared.persist(admission, resources[-1].inputs(), session_id='one', origin_turn_id='first', label='Private QC')
    choices = shared.discover(admission, 'one')
    assert len(choices) == 3 and shared.discover(admission, 'two') == ()
    for choice in choices[1:]:
        assert shared.check_choice(admission, 'one', 'resource.' + choice.input_set_id) == choice.inputs()
        with pytest.raises(ValueError): shared.check_choice(admission, 'two', 'resource.' + choice.input_set_id)
    q = _serialize(record.reference_qualification)
    from agent.tools.data.scatac_qc_reference import load_scatac_qc_reference_bundle
    bundle = load_scatac_qc_reference_bundle(q['inputs']['qc_reference_manifest_path'])[1]
    incomplete = [row for row in q['files'] if row['path'] != bundle.parent_manifest.path]
    with pytest.raises(ValueError):
        shared._raw_reference_choices(q['inputs'], 'a'*40, 'Forged closure', incomplete)


def test_experimental_uploads_cannot_enter_library(tmp_path):
    library = shared.ScientificResourceLibrary(tmp_path/'library')
    for kind in ('h5ad','bam','fastq','fragments','attachment'):
        with pytest.raises(ValueError):
            library.install(kind,{},label='Experiment',version='1',provenance='Private data',usage='task')
    assert library.entries() == ()


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


def test_annotation_qualification_survives_owner_code_updates(resources, tmp_path, monkeypatch):
    app, record, _ = private(resources, tmp_path, monkeypatch)
    q = _serialize(record.reference_qualification)
    assert all('/src/agent/' not in row['path'] for row in q['files'])
    # Model an accepted historical record with its original implementation pin.
    # Never edit real source files to simulate a software update.
    old_code = dict(path=u.annotation_input.__file__, sha256='a'*64, size_bytes=1)
    q['files'] = sorted([*q['files'], old_code], key=lambda row: row['path'])
    legacy = app.resources.register('historical-code-pin-fixture', record.source_path,
        input_type='scientific_reference', label=record.label, attribution='Simulated historical qualification',
        reference_qualification=q, session_id=record.session_id)
    monkeypatch.setattr(u.annotation_input, 'qualify_resources', lambda *a: pytest.fail('Repeated qualification'))
    assert shared.check_record(app.resources, legacy) == shared.input_set(record).inputs()
    spec = json.loads(Path(record.source_path).read_text())
    Path(spec['signature_resource']['path']).write_text('changed scientific data')
    with pytest.raises(refs.ReferenceSupplementError) as error:
        shared.check_record(app.resources, legacy)
    assert error.value.code == 'REFERENCE_SUPPLEMENT_INVALID'
    assert 'resource bytes' in error.value.message


def test_external_genome_installs_without_task_or_run(tmp_path, monkeypatch):
    library = shared.ScientificResourceLibrary(tmp_path/'library')
    fasta=tmp_path/'ref.fa'; fasta.write_text('>chr1\nACGTACGT\n')
    fai=tmp_path/'ref.fa.fai'; fai.write_text('chr1\t8\t6\t8\t9\n')
    args=dict(species=dict(scientific_name='Danio rerio',taxonomy_id=7955),assembly='test-assembly',
        label='Simulated genome',version='1',provenance='Simulated eight-base software fixture',usage='Isolated test')
    entry=library.install_genome(fasta,fai,**args)
    from agent.tools.data import chromap_reference_index as owner
    monkeypatch.setattr(owner,'verify_fasta_fai',lambda *a:pytest.fail('Repeated scientific qualification'))
    assert library.install_genome(fasta,fai,**args) == entry
    assert set(entry['resource']) == {'species','assembly','genome'}
    assert len(library.entries())==1
    library.check(entry)
    fai.write_text('chr1\t8\t7\t8\t9\n')
    with pytest.raises(ValueError): library.check(entry)


def test_invalid_native_annotation_reference_not_indexed(resources,tmp_path):
    spec=json.loads(Path(resources[0]['annotation_spec_path']).read_text())
    bad=tmp_path/'bad.tsv';bad.write_text('candidate\tgene\nUnknown\tNOT_A_GENE\n')
    spec['signature_resource']=dict(spec['signature_resource'],path=str(bad),sha256=u.science.sha256(bad))
    library=shared.ScientificResourceLibrary(tmp_path/'library')
    with pytest.raises(ValueError):
        library.install('annotation',spec,label='Invalid',version='1',provenance='Simulated',usage='test')
    assert library.entries()==()


def test_model_installation_rejects_invalid_native_checkpoint(tmp_path, monkeypatch):
    import hashlib
    import torch
    from agent.tools.analysis import epizoo_embedding as owner
    checkpoint = tmp_path/'invalid.pt'
    torch.save({'wrong_tensor': torch.zeros(1)}, checkpoint)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(owner, '_pinned_resource_paths', lambda *args: {
        key: checkpoint for key in ('checkpoint_sha256', 'frequencies_sha256', 'filter_indices_sha256')})
    resource = dict(resource_id='invalid', label='Invalid model', species='human',
        checkpoint_path=str(checkpoint), checkpoint_sha256=sha, frequencies_sha256=sha,
        filter_indices_sha256=sha, qualification='Synthetic malformed checkpoint')
    library = shared.ScientificResourceLibrary(tmp_path/'library')
    with pytest.raises(ValueError, match='missing required tensor'):
        library.install('epizoo', resource, label='Invalid model', version='test',
            provenance='Simulated fixture', usage='Isolated test only')
    assert library.entries() == ()


def test_shared_annotation_survives_upload_loss_and_restart(resources, tmp_path, monkeypatch):
    import hashlib
    app, record, result = private(resources, tmp_path, monkeypatch)
    library = shared.ScientificResourceLibrary(tmp_path/'library')
    app.resources.library = library
    original = Path(record.source_path).read_bytes()
    qualify = u.annotation_input.qualify_resources
    spec = json.loads(original)
    entry = shared.install_private(app.resources, 'upload', record.resource_id, shared_use=True)
    assert Path(record.source_path).read_bytes() == original
    assert app.resources.load(record.resource_id) == record
    assert all(Path(f['path']).is_relative_to(library.root) for f in entry['files'])
    for key in ('gene_resource', 'signature_resource'):
        old, new = Path(spec[key]['path']), Path(entry['resource'][key]['path'])
        assert old.read_bytes() == new.read_bytes()
        assert old.stat().st_ino != new.stat().st_ino
    monkeypatch.setattr(u.annotation_input, 'qualify_resources', lambda *a: pytest.fail('Repeated qualification'))
    assert shared.install_private(app.resources, 'upload', record.resource_id, shared_use=True) == entry
    # Isolated fixture only: remove every original resource pin from availability.
    for index, row in enumerate(shared.redistribution_files(_serialize(record.reference_qualification['files']))):
        path = Path(row['path'])
        if path.exists(): path.rename(path.with_name(path.name+'.unavailable'))
    restarted = LocalResourceAdmission(app.resources._workspace, approved_source_roots=app.resources._approved,
        scientific_resource_root=library.root)
    assert shared.discover(restarted, 'fresh') == library.choices()
    assert restarted.library.get(entry['id']) == entry
    selected = shared.check_choice(restarted, 'fresh', 'resource.library-'+entry['id'])
    from agent.tools.analysis.annotation_contract import specification
    durable = specification(selected['annotation_spec_path'], selected['annotation_spec_sha256'])
    assert durable['gene_resource']['sha256'] == spec['gene_resource']['sha256']
    # The actual scientific resource reader consumes only library files, including
    # relocated QC scope and its complete native parent closure.
    qualify(durable)
    from agent.tools.data import scatac_qc_reference as qc
    bundle = qc.load_scatac_qc_reference_bundle(durable['contig_scope']['path'])[1]
    qc.reinspect_scatac_qc_reference_bundle_sources(bundle)
    Path(durable['gene_resource']['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError): restarted.library.check(entry)


def test_qc_durable_closure_survives_original_source_loss(resources, tmp_path, monkeypatch):
    values = resources[-1].inputs()
    library = shared.ScientificResourceLibrary(tmp_path/'library')
    before, pins, _ = shared._qc_closure(values['qc_reference_manifest_path'], values['qc_reference_manifest_sha256'])
    args = dict(label='Simulated QC',version='fixture',provenance='Simulated',usage='test',own_files=True)
    entry = library.install('qc', values, **args)
    assert all(Path(f['path']).is_relative_to(library.root) for f in entry['files'])
    from agent.tools.data import scatac_qc_reference as qc
    original_qualifier = qc.reinspect_scatac_qc_reference_bundle_sources
    monkeypatch.setattr(qc, 'reinspect_scatac_qc_reference_bundle_sources', lambda *a: pytest.fail('Repeated qualification'))
    assert library.install('qc', values, **args) == entry
    for path in pins:
        path = Path(path)
        path.rename(path.with_name(path.name+'.unavailable'))
    reopened = shared.ScientificResourceLibrary(library.root)
    reopened.check(entry)
    after = qc.load_scatac_qc_reference_bundle(entry['resource']['qc_reference_manifest_path'])[1]
    assert after.parent_reference_identity_sha256 == before.parent_reference_identity_sha256
    assert after.tss.sha256 == before.tss.sha256 and after.lineage.sha256 == before.lineage.sha256
    original_qualifier(after)


def test_interrupted_shared_copy_never_exposes_incomplete_entry(resources, tmp_path, monkeypatch):
    app, record, _ = private(resources, tmp_path, monkeypatch)
    library = shared.ScientificResourceLibrary(tmp_path/'library');app.resources.library = library
    original = library._copy_file; calls = []
    def interrupted(path, sha):
        calls.append(path)
        if len(calls) == 2: raise OSError('Simulated interrupted component copy')
        return original(path, sha)
    monkeypatch.setattr(library, '_copy_file', interrupted)
    with pytest.raises(OSError): shared.install_private(app.resources, 'upload', record.resource_id, shared_use=True)
    assert library.entries() == () and shared.discover(app.resources, 'other') == ()
    monkeypatch.setattr(library, '_copy_file', original)
    entry = shared.install_private(app.resources, 'upload', record.resource_id, shared_use=True)
    library.check(entry)
    assert len(library.entries()) == 1


def test_new_deployment_records_are_retired_and_old_records_not_discovered(resources, tmp_path, monkeypatch):
    app, record, _ = private(resources, tmp_path, monkeypatch)
    q = _serialize(record.reference_qualification)
    q.update(scope='deployment',review=dict(operator='Fixture',provenance='Simulated old record',reuse_basis='historical'))
    with pytest.raises(ValueError, match='ScientificResourceLibrary'):
        app.resources.register('old-shared-writer',record.source_path,input_type='scientific_reference',
            label=record.label,attribution='Old deployment',reference_qualification=q)
    # Read-only historical fixture, not an alternate current publication API.
    historical = replace(record, session_id=None, reference_qualification=q)
    shared.validate_record(historical)
    monkeypatch.setattr(refs, '_records', lambda resources: (historical,))
    assert shared.discover(app.resources, 'other') == ()
    with pytest.raises(refs.ReferenceSupplementError) as rejected:
        shared.check_choice(app.resources, 'upload', 'resource.'+shared.input_set(historical).input_set_id)
    assert 'Legacy shared selection is retired' in rejected.value.message


def test_interrupted_index_publication_preserves_existing_library(resources, tmp_path, monkeypatch):
    import os
    app, record, _ = private(resources, tmp_path, monkeypatch)
    library = shared.ScientificResourceLibrary(tmp_path/'library')
    entry = library.install('qc', resources[-1].inputs(), label='QC', version='1', provenance='Simulated', usage='test')
    previous = library.path.read_bytes()
    spec = json.loads(Path(record.source_path).read_text())
    original = os.replace
    def interrupted(source, target):
        if Path(target) == library.path: raise OSError('Simulated index publication interruption')
        return original(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', interrupted)
        with pytest.raises(OSError):
            library.install('annotation',spec,label='Annotation',version='1',provenance='Simulated',usage='test',own_files=True)
    assert library.path.read_bytes() == previous
    assert library.entries() == (entry,)
    assert not list(library.root.glob('.index-*'))
    library.install('annotation',spec,label='Annotation',version='1',provenance='Simulated',usage='test',own_files=True)
    assert len(library.entries()) == 2
