"""Read-only F1 evidence and review validation; frozen F2 execution retired."""
import hashlib
import json
from pathlib import Path
import re
from agent.orchestration.planning_scope import fingerprint
from . import harness, run_frontier as frontier

CANDIDATE = frontier.CANDIDATE
F1_DIRECTORY = Path('evals/frontier_comparator_f1_gpt56sol_2026-10-07')

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_f1(original=F1_DIRECTORY):
    """Check the entire retained evidence set without writing any F1 artifact."""
    directory = Path(original)
    manifest_path = directory / 'evidence_manifest.json'
    hashes = json.loads(manifest_path.read_text()).get('sha256')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('F1 needs its complete retained evidence manifest.')
    required = {'results.json', 'offline_validation.json', 'discovery.json',
                'frozen_input_audit.json', 'closeout.json', 'independent_audit.json'}
    if not required <= set(hashes):
        raise ValueError('F1 evidence manifest is incomplete.')
    actual_files = {p.name for p in directory.iterdir() if p.is_file() and p.name != '.gitignore'}
    if actual_files != set(hashes) | {'evidence_manifest.json'}:
        raise ValueError('F1 retained evidence file set changed.')
    for name, digest in hashes.items():
        if Path(name).name != name or name in ('.', '..') or _sha(directory / name) != digest:
            raise ValueError('F1 retained evidence bytes changed.')
    saved = json.loads((directory / 'results.json').read_text())
    offline = json.loads((directory / 'offline_validation.json').read_text())
    frontier.validate_discovery(json.loads((directory / 'discovery.json').read_text()))
    if (saved.get('phase') != 'F1' or saved.get('candidate') != CANDIDATE.manifest()
            or saved.get('admission_gate_passed') is not True or saved.get('hard_stop') is not None
            or saved.get('semantic_attempts') != 4
            or saved.get('counts', {}).get('provider_calls') != 7
            or [r.get('case') for r in saved.get('results', ())] != list(frontier.SCHEDULE)):
        raise ValueError('F2 requires the successful exact-candidate F1 gate and four-case screen.')
    for row in saved['results']:
        expected_status = 'HUMAN_REVIEW_PENDING' if row['case'] == 'I08' else 'PASS'
        attempt = row.get('attempt') or {}
        if (row.get('status') != expected_status or not attempt.get('contract_success')
                or attempt.get('admission_success') is not True or any(attempt.get('safety', {}).values())
                or attempt.get('candidate') != harness.candidate_identity(CANDIDATE)):
            raise ValueError('The accepted F1 attempt or zero-science identity changed.')
    if sum(len(r.get('diagnostics', ())) for r in saved['results']) != 7:
        raise ValueError('F1 completion accounting changed.')
    sources = offline.get('execution_source_sha256', {})
    expected_sources = {'benchmarks/interactive/run_frontier.py', 'benchmarks/interactive/openrouter.py'}
    source_names = {}
    if isinstance(sources, dict):
        for name in sources:
            if isinstance(name, str):
                for expected in expected_sources:
                    if name in (expected, str(Path(expected).resolve())):
                        source_names[name] = expected
    if (not isinstance(sources, dict) or len(sources) != len(expected_sources)
            or set(source_names) != set(sources) or set(source_names.values()) != expected_sources
            or any(not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
                   for digest in sources.values())
            or saved.get('offline_validation_fingerprint') != fingerprint(offline)):
        raise ValueError('Retained F1 execution metadata or its bound offline validation changed.')
    # Historical source hashes identify F1's original execution, not today's
    # infrastructure. No current semantic builder or provider is invoked to reinterpret them.
    identity = dict(directory=str(directory.resolve()), manifest_sha256=_sha(manifest_path),
                    sha256=hashes, results_sha256=_sha(directory / 'results.json'),
                    execution_source_sha256=dict(sources))
    return saved, identity

def validate_review(path, saved):
    path = Path(path)
    review = json.loads(path.read_text())
    first = next(r for r in saved['results'] if r['case'] == 'I08')
    if (review.get('scenario_id') != 'I08' or review.get('candidate_id') != CANDIDATE.candidate_id
            or review.get('external_human_semantic_review') != 'PASS'
            or review.get('historical_status') != 'HUMAN_REVIEW_PENDING'
            or ('retained_answer' in review and review['retained_answer'] != first['attempt']['explanation'])):
        raise ValueError('F1 I08 requires its separate matching external human PASS disposition.')
    return dict(path=str(path.resolve()), sha256=_sha(path), human_semantic_review='PASS',
                historical_status='HUMAN_REVIEW_PENDING', rewrites_historical_evidence=False)

def run(*args, **kwargs):
    raise RuntimeError("Historical qualification runner retired by R5; preserved reports remain readable. "
                       "Use the current interactive harness for new checks.")

def main(*args, **kwargs):
    raise RuntimeError("Historical qualification runner retired by R5; preserved reports remain readable. "
                       "Use the current interactive harness for new checks.")

if __name__ == '__main__':
    main()
