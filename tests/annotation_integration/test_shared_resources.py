"""Simulated sharing policy/software witnesses; no public biology qualification."""
from dataclasses import replace
from pathlib import Path
import json
import pytest
from fastapi.testclient import TestClient
from agent.application import shared_resources as shared, reference_supplementation as refs
from agent.application.local_resources import LocalResourceAdmission
from agent.schemas.orchestration import _serialize


@pytest.fixture
def resources(annotation_scope):
    from agent.application.input_sets import ScientificInputSet
    return (ScientificInputSet('qc','Synthetic QC', dict(qc_reference_manifest_path=annotation_scope['path'],
        qc_reference_manifest_sha256=annotation_scope['sha256'])),)

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


def test_qc_projects_native_inputs_without_companion_aliases(resources, tmp_path, monkeypatch):
    from agent.application.native_raw_handoff import resource_evidence
    from agent.tools.data import scatac_qc_reference as owner
    library = shared.ScientificResourceLibrary(tmp_path/'library')
    entry = library.install('qc', resources[-1].inputs(), label='Simulated QC', version='test',
        provenance='Synthetic fixture', usage='Isolated test only')
    monkeypatch.setattr(owner,'reinspect_scatac_qc_reference_bundle_sources',lambda *a:pytest.fail('Discovery repeated science'))
    choice, = library.choices()
    assert choice == library.input_set(entry) and not choice.fragments_companion and not choice.bam_companion
    evidence = resource_evidence((choice,))[choice.input_set_id]
    assert set(evidence['components']) == {'reference','chrom_sizes','peaks','gene_anno'}
    bundle = owner.load_scatac_qc_reference_bundle(entry['resource']['qc_reference_manifest_path'])[1]
    from agent.tools.data._fragment_reference import load
    parent=load(bundle.parent_manifest.path,bundle.parent_manifest.sha256)
    assert evidence['components']['chrom_sizes']['sha256']==parent.genome.fai.sha256
    assert evidence['components']['peaks']['sha256']==parent.ccre.bed.sha256
    # Existing exact historical handles remain readable, with no new aliases.
    admission=LocalResourceAdmission(tmp_path/'catalog',approved_source_roots=(tmp_path,),scientific_resource_root=library.root)
    legacy='resource.fref-'+entry['id']
    assert shared.check_choice(admission,'any',legacy)['reference_bundle_sha256']==bundle.parent_manifest.sha256
    Path(bundle.parent_manifest.path).write_text('substituted')
    assert resource_evidence(library.choices())[choice.input_set_id] is None
    with pytest.raises(ValueError):shared.check_choice(admission,'any',legacy)


def test_private_qc_parent_scope_and_loose_files_never_grant_reuse(resources, tmp_path):
    admission = LocalResourceAdmission(tmp_path/'catalog', approved_source_roots=(tmp_path,))
    assert shared.discover(admission, 'one') == ()  # Fixture files exist, but are not admitted.
    record = shared.persist(admission, resources[-1].inputs(), session_id='one', origin_turn_id='first', label='Private QC')
    choices = shared.discover(admission, 'one')
    assert len(choices) == 1 and shared.discover(admission, 'two') == ()
    for choice in choices:
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


def test_operator_resource_labels_do_not_claim_shared_catalog_identity():
    assert shared.check_choice(None, 'session', 'resource.qualified-human') is None


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


def test_qc_durable_closure_survives_original_source_loss(resources, tmp_path, monkeypatch):
    values = resources[-1].inputs()
    library = shared.ScientificResourceLibrary(tmp_path/'library')
    before, pins = shared._qc_closure(values['qc_reference_manifest_path'], values['qc_reference_manifest_sha256'])
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
