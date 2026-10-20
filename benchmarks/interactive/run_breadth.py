"""Read-only Q2 report helpers; the frozen live experiment is retired by R5."""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote, quote_plus
from .breadth_transport import _reporting_error
from .harness import candidate_identity

FAILURES = {'operational': 'OPERATIONAL_FAIL', 'contract': 'CONTRACT_FAIL',
            'semantic': 'SEMANTIC_FAIL', 'deterministic_admission': 'ADMISSION_FAIL'}

def validate_call_identity(row, scenario, candidate):
    """A completed cell needs observed calls with the same exact attempt identity."""
    attempt = row['attempt']
    calls = attempt.get('calls', ())
    count = row['provider_completions']
    if (row['status'] != status_for_attempt(attempt)
            or type(count) is not int or count <= 0 or not calls
            or count > (5 if scenario.surface == 'layer2' else 1)
            or len(row['diagnostics']) != count or count > len(calls)
            or any(call.get('candidate') != candidate_identity(candidate)
                or call.get('scenario_id') != scenario.scenario_id
                or call.get('attempt_id') != row['attempt_id'] for call in calls)):
        raise ValueError('Historical call identity or completion accounting is inconsistent.')

def _private_values(environment):
    return tuple(value for name, value in environment.items() if value and
        (name.endswith('_API_KEY') or any(label in name.upper() for label in
         ('WORKSPACE_ID', 'WORK_SPACE_ID', 'PROJECT_ID', 'ACCOUNT_ID', 'CONSUMER'))))

def _safe_json(value, private):
    # Model prose is retained only after private values are removed. Scoring precedes redaction.
    variants = {encoded for item in private for encoded in
                (item, quote(item, safe=''), quote_plus(item), json.dumps(item)[1:-1])}
    patterns = [re.compile(re.escape(item), re.IGNORECASE) for item in sorted(variants, key=len, reverse=True)]
    def clean(item):
        if isinstance(item, str):
            for pattern in patterns:
                item = pattern.sub('[REDACTED]', item)
            return item
        if isinstance(item, dict): return {clean(key): clean(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)): return [clean(value) for value in item]
        return item
    return _reporting_error(clean(value))

def status_for_attempt(attempt):
    category = attempt['failure_category']
    if category in FAILURES: return FAILURES[category]
    if attempt['contract_success'] is not True: return 'CONTRACT_FAIL'
    if attempt['human_review_required']: return 'HUMAN_REVIEW_PENDING'
    if attempt['semantic_success'] is True: return 'PASS'
    if attempt['semantic_success'] is False: return 'SEMANTIC_FAIL'
    raise ValueError('Attempt lacks a conclusive frozen-contract observation.')

def _original(original):
    if isinstance(original, dict):
        return json.loads(json.dumps(original)), None
    path = Path(original)
    if path.is_dir(): path = path / 'results.json'
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest())

def run_breadth(*args, **kwargs):
    raise RuntimeError("Historical qualification runner retired by R5; preserved reports remain readable. "
                       "Use the current interactive harness for new checks.")

def resume_breadth(*args, **kwargs):
    raise RuntimeError("Historical qualification runner retired by R5; preserved reports remain readable. "
                       "Use the current interactive harness for new checks.")

def main(*args, **kwargs):
    raise RuntimeError("Historical qualification runner retired by R5; preserved reports remain readable. "
                       "Use the current interactive harness for new checks.")

if __name__ == '__main__':
    main()
