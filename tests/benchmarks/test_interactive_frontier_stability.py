"""Historical evidence readers use fixed metadata, never today's LLM interfaces."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
from benchmarks.interactive import harness, run_frontier, run_frontier_stability as F2
from test_interactive_frontier_transport import valid_frontier_discovery

@pytest.fixture
def historical(tmp_path):
    # Simulated historical metadata only. No production/planning/answer replay.
    directory = tmp_path / 'f1'
    directory.mkdir()
    rows = []
    for case in run_frontier.SCHEDULE:
        rows.append(dict(case=case, status='HUMAN_REVIEW_PENDING' if case=='I08' else 'PASS',
            diagnostics=[{}] * (3 if case=='I01' else 1),
            attempt=dict(contract_success=True, admission_success=True, safety={},
                candidate=harness.candidate_identity(run_frontier.CANDIDATE),
                explanation='Original preserved explanation.', semantic_success=None)))
    offline = dict(execution_source_sha256={name:'a'*64 for name in (
        'benchmarks/interactive/run_frontier.py', 'benchmarks/interactive/openrouter.py')})
    saved = dict(phase='F1', candidate=run_frontier.CANDIDATE.manifest(),
        admission_gate_passed=True, hard_stop=None, semantic_attempts=4,
        counts=dict(provider_calls=7), results=rows, offline_validation_fingerprint=F2.fingerprint(offline))
    for name,value in {'results.json':saved, 'offline_validation.json':offline,
            'discovery.json':valid_frontier_discovery(), 'frozen_input_audit.json':{},
            'closeout.json':{}, 'independent_audit.json':{}}.items():
        write_json(directory/name,value)
    write_json(directory/'evidence_manifest.json',dict(sha256=hash_manifest(directory)))
    review=tmp_path/'review.json'
    write_json(review,dict(scenario_id='I08',candidate_id=run_frontier.CANDIDATE.candidate_id,
        historical_status='HUMAN_REVIEW_PENDING', external_human_semantic_review='PASS',
        retained_answer='Original preserved explanation.'))
    return directory,saved,offline,review

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def hash_manifest(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "evidence_manifest.json"}


def test_f1_manifest_and_external_review_are_separate_immutable_inputs(historical):
    source, saved, _, review = historical
    before = hash_manifest(source)
    returned, identity = F2.validate_f1(original=source)
    assert returned == saved
    assert identity
    disposition = F2.validate_review(review, saved)
    assert disposition["human_semantic_review"] == "PASS"
    assert hash_manifest(source) == before
    historical_answer = next(r for r in returned["results"] if r["case"] == "I08")
    assert historical_answer["status"] == "HUMAN_REVIEW_PENDING"
    assert historical_answer["attempt"]["semantic_success"] is None


def rebind_offline_validation(source, offline):
    """Model a complete historical record with its original metadata binding."""
    write_json(source / "offline_validation.json", offline)
    saved = json.loads((source / "results.json").read_text())
    saved["offline_validation_fingerprint"] = F2.fingerprint(offline)
    write_json(source / "results.json", saved)
    write_json(source / "evidence_manifest.json", {"sha256": hash_manifest(source)})
    return saved


def test_f1_historical_source_hashes_survive_later_infrastructure_with_exact_metadata_binding(historical):
    source, _, offline, _ = historical
    historical_sources = {name: str(number) * 64 for number, name in
        enumerate(offline["execution_source_sha256"], 1)}
    offline["execution_source_sha256"] = historical_sources
    saved = rebind_offline_validation(source, offline)
    before = hash_manifest(source)
    assert any(hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest for name, digest in historical_sources.items())
    returned, identity = F2.validate_f1(original=source)
    assert returned == saved
    assert identity["execution_source_sha256"] == historical_sources
    assert hash_manifest(source) == before


def test_f1_actual_absolute_source_identity_is_retained_without_current_source_equality(historical):
    source, _, offline, _ = historical
    historical_sources = {str(Path(name).resolve()): str(number) * 64 for number, name in
        enumerate(offline["execution_source_sha256"], 1)}
    offline["execution_source_sha256"] = historical_sources
    saved = rebind_offline_validation(source, offline)
    returned, identity = F2.validate_f1(original=source)
    assert returned == saved
    assert identity["execution_source_sha256"] == historical_sources


@pytest.mark.parametrize("mutation", ["missing_source", "extra_source", "duplicate_alias", "invalid_digest"])
def test_f1_source_identity_shape_remains_closed_even_with_rebound_metadata(historical, mutation):
    source, _, offline, _ = historical
    sources = offline["execution_source_sha256"]
    if mutation == "missing_source":
        sources.pop("benchmarks/interactive/openrouter.py")
    elif mutation == "extra_source":
        sources["src/agent/orchestration/semantic_prompt.py"] = "a" * 64
    elif mutation == "duplicate_alias":
        sources.pop("benchmarks/interactive/openrouter.py")
        sources[str(Path("benchmarks/interactive/run_frontier.py").resolve())] = "a" * 64
    else:
        sources["benchmarks/interactive/run_frontier.py"] = "not-a-sha256"
    rebind_offline_validation(source, offline)
    with pytest.raises(ValueError):
        F2.validate_f1(original=source)


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_f1_offline_fingerprint_binding_is_required_even_with_valid_file_manifest(historical, mutation):
    source, saved, _, _ = historical
    if mutation == "missing":
        saved.pop("offline_validation_fingerprint")
    else:
        saved["offline_validation_fingerprint"] = "a" * 64
    write_json(source / "results.json", saved)
    write_json(source / "evidence_manifest.json", {"sha256": hash_manifest(source)})
    with pytest.raises(ValueError):
        F2.validate_f1(original=source)


@pytest.mark.parametrize("mutation", ["candidate", "scenario", "verdict", "historical_status", "answer"])
def test_external_review_must_match_exact_historical_pending_answer(historical, mutation):
    _, saved, _, review = historical
    disposition = json.loads(review.read_text())
    if mutation == "candidate": disposition["candidate_id"] = "groq-gpt-oss-120b"
    elif mutation == "scenario": disposition["scenario_id"] = "I09"
    elif mutation == "verdict": disposition["external_human_semantic_review"] = "PENDING"
    elif mutation == "historical_status": disposition["historical_status"] = "PASS"
    else: disposition["retained_answer"] = "Another answer."
    write_json(review, disposition)
    with pytest.raises(ValueError):
        F2.validate_review(review, saved)


@pytest.mark.parametrize('mutation',['bytes','missing','extra','traversal'])
def test_preserved_manifest_rejects_corrupt_files_without_reconstruction(historical,mutation):
    directory,_,_,_=historical
    if mutation=='bytes': (directory/'results.json').write_text('{}')
    elif mutation=='missing': (directory/'results.json').unlink()
    elif mutation=='extra': (directory/'extra.json').write_text('{}')
    else:
        manifest=json.loads((directory/'evidence_manifest.json').read_text())
        manifest['sha256']['../outside.json']='a'*64
        write_json(directory/'evidence_manifest.json',manifest)
    with pytest.raises((ValueError,FileNotFoundError)):
        F2.validate_f1(directory)
