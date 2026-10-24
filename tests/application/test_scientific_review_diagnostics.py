"""Private R5 observations identify bounded material without retaining its text."""
import hashlib
import json
import logging
from types import SimpleNamespace

import pytest

from agent.application import scientific_dialogue as dialogue


PREFIX = 'Scientific answer diagnostic: '
PRIVATE = '/private/patient-input.h5ad API_SECRET sensitive scientific conclusion'


def record(caplog):
    rows = [entry for entry in caplog.records
            if entry.getMessage().startswith(PREFIX)]
    assert len(rows) == 1
    return rows[0], json.loads(rows[0].getMessage()[len(PREFIX):].partition(' ')[2])


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def test_review_diagnostic_has_private_correlation_and_exact_fingerprints(caplog):
    context = ('c408ef11-a77b-4a7a-b91c-7fa1d1f255b7', PRIVATE)
    candidate = dict(support='supported', paragraphs=[dict(text=PRIVATE)])
    evidence = dict(claims=[dict(value=PRIVATE)], scope='accepted_summary')
    units = [dict(verdict='entailed', independent=True, private_reason=PRIVATE),
             dict(verdict='unsupported', independent=False, candidate=PRIVATE)]
    dialogue._diagnose('semantic_review', 'partially_reviewed_response',
        context=context, candidate=candidate, evidence=evidence, units=units, retained=(0,))
    entry, row = record(caplog)
    assert entry.scientific_answer_stage == 'semantic_review'
    assert entry.scientific_answer_reason == 'partially_reviewed_response'
    assert row['stage'] == 'semantic_review' and row['reason'] == 'partially_reviewed_response'
    for key, value in zip(('session_id', 'turn_id'), context):
        assert row[key + '_sha256'] == hashlib.sha256(value.encode()).hexdigest()
        assert value not in caplog.text
    for key, value in (('candidate', candidate), ('evidence', evidence)):
        assert row[key + '_sha256'] == hashlib.sha256(encoded(value)).hexdigest()
        assert row[key + '_bytes'] == len(encoded(value))
    assert row['review_units'] == [dict(index=0, verdict='entailed', independent=True),
                                    dict(index=1, verdict='unsupported', independent=False)]
    assert row['review_retained'] == [0]
    assert type(row['process']) is int and type(row['thread']) is int
    assert PRIVATE not in caplog.text and 'private_reason' not in caplog.text
    assert not entry.exc_info


def test_review_fingerprints_use_canonical_json(caplog):
    dialogue._diagnose('semantic_review', 'reviewed_response',
                      candidate=dict(z=1, a='λ'), evidence=dict(z=2, a='μ'))
    _, first = record(caplog)
    caplog.clear()
    dialogue._diagnose('semantic_review', 'reviewed_response',
                      candidate=dict(a='λ', z=1), evidence=dict(a='μ', z=2))
    _, second = record(caplog)
    assert first == second


def test_unknown_diagnostic_codes_are_never_logged_as_text(caplog):
    dialogue._diagnose(PRIVATE, PRIVATE, context=[PRIVATE, 't' * 4097])
    entry, row = record(caplog)
    assert row['stage'] == entry.scientific_answer_stage == 'diagnostic'
    assert row['reason'] == entry.scientific_answer_reason == 'unclassified_failure'
    assert row['session_id_sha256'] == hashlib.sha256(PRIVATE.encode()).hexdigest()
    assert 'turn_id_sha256' not in row and PRIVATE not in caplog.text


@pytest.mark.parametrize('unit_count', [24, 25])
def test_unit_observation_count_is_bounded(caplog, unit_count):
    units = [dict(verdict='entailed', independent=True, text=PRIVATE)] * unit_count
    dialogue._diagnose('semantic_review', 'reviewed_response', units=units,
                      retained=tuple(range(unit_count)))
    _, row = record(caplog)
    assert ('review_units' in row) == (unit_count == 24)
    assert ('review_retained' in row) == (unit_count == 24)
    assert len(caplog.text.encode()) < 4096
    assert PRIVATE not in caplog.text


@pytest.mark.parametrize('units,retained', [
    ([dict(verdict=PRIVATE, independent=True)], (0,)),
    ([dict(verdict='entailed', independent=1)], (0,)),
    ([PRIVATE], (0,)),
    ([dict(verdict='entailed', independent=True)], (True,)),
    ([dict(verdict='entailed', independent=True)], (1,)),
    ([dict(verdict='entailed', independent=True)], (0, 0)),
])
def test_malformed_unit_details_do_not_escape_or_imply_retention(caplog, units, retained):
    dialogue._diagnose('semantic_review_schema', 'invalid_review', units=units, retained=retained)
    _, row = record(caplog)
    assert 'review_retained' not in row
    assert PRIVATE not in caplog.text


def test_oversized_or_non_json_details_are_omitted_without_truncated_fingerprints(caplog):
    dialogue._diagnose('semantic_review_schema', 'invalid_review',
        candidate=dict(text='x' * 16385), evidence=dict(text='y' * (dialogue.MAX_TARGET_CONTEXT_BYTES + 1)))
    _, row = record(caplog)
    assert not any(key.startswith(('candidate_', 'evidence_')) for key in row)
    caplog.clear()
    dialogue._diagnose('parsing', 'malformed_provider_output',
        candidate=PRIVATE, evidence=dict(invalid=float('nan')))
    _, row = record(caplog)
    assert not any(key.startswith(('candidate_', 'evidence_')) for key in row)
    assert PRIVATE not in caplog.text


@pytest.mark.parametrize('failure', ['threshold', 'emit'])
def test_diagnostic_logger_failure_cannot_change_answer_behavior(monkeypatch, failure):
    class BrokenLogger:
        def isEnabledFor(self, level):
            assert level == logging.WARNING
            if failure == 'threshold':
                raise RuntimeError(PRIVATE)
            return True

        def warning(self, *args, **kwargs):
            raise RuntimeError(PRIVATE)

    monkeypatch.setattr(dialogue, 'logging', SimpleNamespace(
        getLogger=lambda name: BrokenLogger(), WARNING=logging.WARNING))
    assert dialogue._diagnose('provider', 'provider_failure', candidate=dict(text=PRIVATE)) is None
