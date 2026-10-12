"""Presentation labels cannot add facts or change an already captured view."""
from dataclasses import replace

import pytest

from agent.application.dialogue_evidence import EvidenceFact
from agent.application.interactive_schemas import EvidenceView, MAX_PRESENTATION_BYTES


def evidence(labels):
    return EvidenceView('analysis', 'revision', 'clusters', 'available',
        facts=(EvidenceFact('n_cells', 'available', 32),), field_labels=labels)


def test_evidence_labels_are_detached_and_immutable_without_changing_facts():
    labels = {'n_cells': 'Cells'}
    view = evidence(labels)
    labels['n_cells'] = 'An unreviewed interpretation'
    assert view.field_labels['n_cells'] == 'Cells'
    with pytest.raises(TypeError):
        view.field_labels['n_cells'] = 'Changed'
    assert view.to_dict()['facts'][0]['value'] == 32
    assert evidence({}).facts == view.facts


@pytest.mark.parametrize('labels', [
    {'invented_biological_conclusion': 'Unsupported conclusion'},
    {'n_cells': ''}, {'n_cells': 'x' * 4097}, {'n_cells': 32},
])
def test_evidence_label_contract_rejects_unoffered_or_unbounded_metadata(labels):
    with pytest.raises((ValueError, TypeError)):
        evidence(labels)


def test_labels_share_the_existing_total_client_projection_byte_limit():
    view = evidence({'n_cells': 'Cells'})
    assert len(str(view.to_dict()).encode()) < MAX_PRESENTATION_BYTES
    # Each field is individually within its label limit; the whole projection
    # must still respect the existing result-view limit.
    fields = tuple(EvidenceFact(f'field-{index}', 'available', index) for index in range(80))
    with pytest.raises(ValueError, match='client byte limit'):
        replace(view, facts=fields, field_labels={fact.field: '界' * 4000 for fact in fields})
