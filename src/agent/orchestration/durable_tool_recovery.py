"""Exact opt-in publication recovery identities; no scientific retries."""
import hashlib
import json
from agent.schemas import fingerprint_plan


def execution_identity(run_id, plan, step, spec):
    value = {'run_id': run_id, 'plan_fingerprint': fingerprint_plan(plan),
             'step_id': step.step_id, 'tool': spec.name, 'policy': spec.recovery_policy_version}
    return hashlib.sha256(b'agent.durable-tool-publication.v1\0' +
        json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
