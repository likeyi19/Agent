"""USI-3A: owner contracts, exact continuation and genuine BAM execution.

Only providers are scripted. Scientific fixtures are explicitly simulated.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent.application import raw_handoff as handoff
from agent.application.local_resources import LocalResourceAdmission
from agent.schemas.orchestration import _serialize
from agent.tools.data import scatac_reference as reference
from agent.tools.data import scatac_library_context as library
from agent.web.config import ScientificInputSet
from test_bam_chat_handoff import (ANSWER, OBJECTIVE, ChatBAMModel, backend, reusable,
                                  send, stage)
from test_bam_execution import downstream_bam_inputs, instrument


COMPLETE = handoff.BamDeclaration(source='source.0', assay='SCATAC', namespace='sample',
    species='human', assembly='hg38', barcode_policy='corrected_cb_already_corrected',
    source_profile='agent-cb-paired-atac.v2')


def other_reference(choice, root, *, species='human'):
    """Owner-built simulated alternative, not a label-based compatibility claim."""
    root.mkdir()
    _, bundle, _ = reference.load_scatac_reference_bundle(choice.inputs()['reference_bundle_path'])
    fasta = root / 'genome.fa'
    fasta.write_bytes(Path(bundle.genome.fasta.path).read_bytes().replace(b'A', b'C'))
    alternate = reference.build_scatac_reference_bundle(species=species,
        target_assembly='hg38' if species == 'human' else 'mm10', fasta_path=fasta,
        fai_path=bundle.genome.fai.path, ccre_bed_path=bundle.ccre.bed.path)
    pointer = reference.publish_scatac_reference_bundle(alternate, root / 'reference.json')
    return replace(choice, input_set_id='alternative', label='Alternative simulated genome',
        execution_inputs=choice.inputs() | dict(species=species,
            reference_bundle_path=pointer['manifest_path'], reference_bundle_sha256=pointer['manifest_sha256']))


@pytest.fixture
def choices(tmp_path):
    args = downstream_bam_inputs(tmp_path)
    first = replace(reusable(args), bam_default=False)
    return first, other_reference(first, tmp_path / 'alternative')


def resolve(choices, declared=COMPLETE, **kwargs):
    return handoff.requirements(SimpleNamespace(members=[{}]), declared, choices,
        evidence=handoff.resource_evidence(choices, 'bam'), **kwargs)


def test_owner_identity_not_default_or_duplicate_species_declaration(choices, monkeypatch):
    first, _ = choices
    values = first.inputs(); values.pop('species')
    first = replace(first, execution_inputs=values)
    def forbidden(*args, **kwargs):
        pytest.fail('Application matching must not deeply reconstruct reference science.')
    monkeypatch.setattr(reference, 'reinspect_scatac_reference_bundle_sources', forbidden)
    assert resolve((first,)) == ([], first)
    invalid = replace(first, execution_inputs=values | {'reference_bundle_sha256': '0' * 64}, bam_default=True)
    missing, selected = resolve((invalid,))
    assert selected is None and 'invalid' in ' '.join(missing)


def test_multiple_choices_explicit_preference_equivalence_and_default_policy(choices):
    first, second = choices
    missing, selected = resolve(choices)
    assert selected is None and all(s.label in ' '.join(missing) for s in choices)
    assert 'hg38' in missing[0]
    assert resolve(choices, replace(COMPLETE, resource='resource.alternative')) == ([], second)
    assert resolve((replace(first, bam_default=True), second))[1].input_set_id == first.input_set_id
    alias = replace(first, input_set_id='alias', label='Equivalent configuration alias')
    assert resolve((first, alias)) == resolve((alias, first)) == ([], alias)
    assert resolve((first, alias), replace(COMPLETE, resource='resource.synthetic')) == ([], first)
    missing, selected = resolve((first, second, alias))
    assert selected is None
    assert alias.label in missing[0] and second.label in missing[0] and first.label not in missing[0]


def test_equivalent_manifest_locators_do_not_create_a_scientific_choice(choices, tmp_path):
    first, _ = choices
    copy = tmp_path / 'same-reference.json'
    copy.write_bytes(Path(first.inputs()['reference_bundle_path']).read_bytes())
    alias = replace(first, input_set_id='alias', execution_inputs=first.inputs() | {'reference_bundle_path': str(copy)})
    assert resolve((first, alias)) == ([], alias)


def test_incompatible_singleton_and_explicit_invalid_choice_never_fall_back(choices, tmp_path):
    first, _ = choices
    mouse = other_reference(first, tmp_path / 'mouse', species='mouse')
    assert resolve((mouse,))[1] is None
    assert resolve((mouse, first))[1] == first
    missing, selected = resolve((mouse, first), replace(COMPLETE, resource='resource.alternative'))
    assert selected is None and 'conflicts' in missing[0]
    assert resolve((first,), replace(COMPLETE, resource='resource.unknown'))[1] is None
    assert resolve((first,), pinned='resource.unknown')[1] is None


def test_missing_sample_context_is_not_guessed_from_single_reference(choices):
    first, _ = choices
    for declaration in (replace(COMPLETE, species=None), replace(COMPLETE, assembly=None)):
        questions, selected = resolve((first,), declaration)
        assert questions and selected is None
    # Missing barcode history does not undo a previously admitted resource choice.
    questions, selected = resolve((first,), replace(COMPLETE, barcode_policy=None))
    assert selected == first and 'already corrected' in ' '.join(questions)


def test_reference_correction_preserves_all_other_captured_declarations():
    previous = replace(COMPLETE, resource='resource.invalid')
    corrected = handoff.merge(previous, handoff.BamDeclaration(resource='resource.valid'))
    assert corrected == replace(previous, resource='resource.valid')
    with pytest.raises(ValueError):
        handoff.merge(previous, handoff.BamDeclaration(namespace='another_library'))
    fastq = handoff.FastqDeclaration(species='human', resource='resource.invalid')
    assert handoff.merge(fastq, handoff.FastqDeclaration(resource='resource.valid')) == replace(fastq, resource='resource.valid')


def test_pending_catalog_only_allows_additions(choices):
    first, second = (s.definition() for s in choices)
    assert handoff._extend_catalog([first], [second, first]) == [second, first]
    for rows in ([], [second], [dict(first, label='changed')]):
        with pytest.raises(ValueError):
            handoff._extend_catalog([first], rows)


@pytest.mark.parametrize('mode', ['unique', 'supplement', 'multiple', 'explicit_invalid'])
def test_conversational_resource_resolution_executes_and_delivers(tmp_path, monkeypatch, mode):
    args = downstream_bam_inputs(tmp_path)
    first = replace(reusable(args), bam_default=False)
    # Species is established by the manifest and the user's sample declaration,
    # not by a second redundant InputSet species field.
    first = replace(first, execution_inputs={k: v for k, v in first.inputs().items() if k != 'species'})
    second = other_reference(first, tmp_path / 'alternative', species='mouse' if mode == 'explicit_invalid' else 'human')
    offered = () if mode == 'supplement' else (first, second) if mode in ('multiple', 'explicit_invalid') else (first,)

    class Model(ChatBAMModel):
        def complete(self, *, prompt, response_schema):
            output = json.loads(super().complete(prompt=prompt, response_schema=response_schema))
            decision = output.get('decision', {})
            if decision.get('kind') == 'answer_roles':
                utterance = json.loads(prompt)['utterance']
                if utterance == 'Use the original synthetic reference.':
                    decision['bam'] = asdict(handoff.BamDeclaration(resource='resource.synthetic'))
                elif mode == 'explicit_invalid':
                    decision['bam']['resource'] = 'resource.alternative'
            return json.dumps(output)

    registry, counts = instrument(monkeypatch)
    monkeypatch.setenv('AGENT_QC_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC', '1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG', raising=False)
    service, uploads, models, app = backend(tmp_path, offered, registry, Model)
    with TestClient(app) as client:
        ids = stage(client, Path(args['source_path']))
        question, _ = send(client, ids)
        assert question['status'] == 'clarification'
        assert not any(counts.values())
    # Fresh application while waiting; newly provisioned resources supplement the
    # same objective and exact source registrations without reupload.
    offered = (first,) if mode == 'supplement' else offered
    service, uploads, models, app = backend(tmp_path, offered, registry, Model)
    with TestClient(app) as client:
        answer = ANSWER + (' Use the alternative simulated genome.' if mode == 'explicit_invalid' else '')
        view, _ = send(client, turn='declarations', utterance=answer)
        if mode in ('multiple', 'explicit_invalid'):
            assert view['status'] == 'clarification', view
            assert not any(counts.values())
            assert ('Choose a reference' if mode == 'multiple' else 'conflicts') in view['response']['text']
            view, _ = send(client, turn='choice', utterance='Use the original synthetic reference.')
        assert view['status'] == 'succeeded', view
        state = service._application.sessions.load('session')
        admitted = _serialize(state.interactions[-1].admitted)
        assert admitted['raw_resource_choice'] == 'resource.synthetic'
        assert admitted['registered_input']['resource_choice'] == first.resource_choice()
        assert admitted['bam_arguments'] == dict(min_qc_fragment_records=0, min_tss_enrichment=0.0)
        run = service._application.run_store.load(view['run_id'])
        assert run.request.prompt == OBJECTIVE
        assert all(step.verification.passed for step in run.steps)
        matrix = run.steps[-1].result
        assert (matrix['n_cells'], matrix['n_features'], matrix['total_count']) == (1, 2, 2)
        assert counts == dict(intake=1, production=1, verification=1)
        route = '/api/v1/sessions/session/revisions/' + view['revision_id']
        artifacts = client.get(route + '/scientific-artifacts').json()['artifacts']
        assert 'matrix.h5ad' in {a['filename'] for a in artifacts}
        for artifact in artifacts:
            response = client.get(route + '/scientific-artifacts/' + artifact['handle'])
            assert response.status_code == 200
            assert hashlib.sha256(response.content).hexdigest() == artifact['sha256']
        if mode == 'unique':
            from agent.tools.data import raw_scatac
            def forbidden(*args, **kwargs):
                pytest.fail('The application must reuse the accepted exact context.')
            monkeypatch.setattr(raw_scatac, 'inspect_raw_scATAC', forbidden)
            monkeypatch.setattr(library, 'build_scatac_library_processing_context', forbidden)
            question, _ = send(client, ids, turn='repeat', generation=1)
            assert question['status'] == 'clarification', question
            repeated, _ = send(client, turn='repeat-answer', utterance=ANSWER, generation=1)
            assert repeated['status'] == 'succeeded', repeated
            second_admission = service._application.sessions.load('session').interactions[-1].admitted
            assert _serialize(second_admission['bam_context']) == admitted['bam_context']
            assert counts == dict(intake=2, production=2, verification=2)
        # Admission observations are historical, not an invitation to re-open
        # reference sources when simply reading Session history.
        before = dict(counts)
        Path(first.inputs()['reference_bundle_path']).unlink()
        assert client.get('/api/v1/sessions/session').status_code == 200
        assert counts == before
        (tmp_path / 'resource-witness.json').write_text(json.dumps(dict(
            mode=mode, selected=admitted['raw_resource_choice'], context=admitted['bam_context'],
            matrix=dict(n_cells=matrix['n_cells'], n_features=matrix['n_features'], total_count=matrix['total_count']),
            authority_schemas=[s.verification.artifact_authority['schema_version'] for s in run.steps[1:]],
            artifacts=artifacts), indent=2))


def test_explicit_reference_becoming_invalid_does_not_rebind(tmp_path, monkeypatch):
    args = downstream_bam_inputs(tmp_path)
    first = replace(reusable(args), bam_default=False)
    alternative = other_reference(first, tmp_path / 'alternative')
    class Model(ChatBAMModel):
        def complete(self, *, prompt, response_schema):
            output = json.loads(super().complete(prompt=prompt, response_schema=response_schema))
            if output.get('decision', {}).get('kind') == 'request_roles':
                output['decision']['bam']['resource'] = 'resource.synthetic'
            return json.dumps(output)
    registry, counts = instrument(monkeypatch)
    service, uploads, models, app = backend(tmp_path, (first, alternative), registry, Model)
    with TestClient(app) as client:
        ids = stage(client, Path(args['source_path']))
        assert send(client, ids)[0]['status'] == 'clarification'
        Path(first.inputs()['reference_bundle_path']).write_bytes(b'{}')
        view, _ = send(client, turn='answer', utterance=ANSWER)
        assert view['status'] == 'clarification' and 'invalid' in view['response']['text']
        admission = service._application.sessions.load('session').interactions[-1].admitted
        assert admission['raw_resource_choice'] == 'resource.synthetic'
        assert not any(counts.values())
        assert not (tmp_path / 'workspace' / 'bam_handoffs').exists()


def test_accepted_library_derivation_reuse_is_independent_of_reference_alias(tmp_path, monkeypatch):
    from agent.tools.data import raw_scatac
    args = downstream_bam_inputs(tmp_path)
    first = replace(reusable(args), bam_default=False)
    service, uploads, models, app = backend(tmp_path, (first,))
    with TestClient(app) as client:
        ids = stage(client, Path(args['source_path']))
    candidates = service.resources.compose_group(ids, attachment_candidates=True)
    binding, receipt = handoff.prepare(service.resources, 'session', candidates, COMPLETE, first, {})
    original_bytes = Path(receipt['context_path']).read_bytes()
    alias = replace(first, input_set_id='alias', label='Another reference selection')
    def forbidden(*args, **kwargs):
        pytest.fail('An admitted exact library resource must not be regenerated.')
    monkeypatch.setattr(raw_scatac, 'inspect_raw_scATAC', forbidden)
    monkeypatch.setattr(library, 'build_scatac_library_processing_context', forbidden)
    new, reused = handoff.prepare(service.resources, 'session', candidates,
        replace(COMPLETE, resource='resource.alias'), alias, {}, accepted_contexts=(receipt,))
    assert reused == receipt
    assert new.resource_choice != binding.resource_choice
    assert Path(receipt['context_path']).read_bytes() == original_bytes
    assert new.execution_inputs['source_path'] == service.resources.load(ids[0]).source_path
    Path(receipt['context_path']).write_bytes(b'{}')
    with pytest.raises(ValueError):
        handoff.prepare(service.resources, 'session', candidates, COMPLETE, first, {}, accepted_contexts=(receipt,))
