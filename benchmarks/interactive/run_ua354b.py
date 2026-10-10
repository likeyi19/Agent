"""UA3.5.4b requalification using the unchanged UA3.5.4a evaluation matrix.

All semantic responses use the ordinary real-provider interfaces. This entrypoint
always performs fresh exact-model discovery; it does not reuse catalog evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningRecoveryPolicy
from agent.orchestration.planning_recovery import SCOPED_PLANNING_RECOVERY_POLICY_VERSION

from . import run_ua354a as base
from .diagnostics import sanitize_provider_error


RECOVERY_POLICY = PlanningRecoveryPolicy(
    policy_version=SCOPED_PLANNING_RECOVERY_POLICY_VERSION,
    max_transport_retries=0, max_repairs=0, max_profile_failovers=0,
    max_primary_local_recovery_actions=0, max_total_provider_calls=2,
    max_retry_delay_seconds=0,
)


def configured_application(*args, **kwargs):
    """Configure ordinary admission with the existing explicit policy contract."""
    return InteractiveAgentApplication(*args, planning_recovery_policy=RECOVERY_POLICY, **kwargs)


class Runner(base.Runner):
    def __init__(self, output):
        super().__init__(output)
        self.report.update(kind='UA3.5.4b',
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            base_runner_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
            planning_recovery_policy=RECOVERY_POLICY.to_dict(),
            discovery_policy='fresh_exact_catalog_once',
            reused_matrix='UA3.5.4a',
            human_answer_fixture='These cells are from human.')
        self.save()

    def run(self, args):
        # Change only the reused evaluation module's constructor binding; the
        # real application, provider and scientific interfaces stay unchanged.
        with patch.object(base, 'InteractiveAgentApplication', configured_application):
            return super().run(args)

    def dependent(self, app, case, sid, utterance, field):
        if case == 'A2':
            utterance = self.report['human_answer_fixture']
        return super().dependent(app, case, sid, utterance, field)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('output', 'workspace', 'source-h5ad', 'operator-config'):
        parser.add_argument('--' + argument, required=True)
    parser.add_argument('--fixtures-json', help='Reuse the accepted tiny owner fixtures unchanged.')
    args = parser.parse_args()
    runner = Runner(args.output)
    try:
        runner.discovery()
        runner.run(args)
    except BaseException as exc:
        runner.report['hard_stop'] = dict(type=type(exc).__name__,
            diagnostic=sanitize_provider_error(exc, secrets=runner.private))
    finally:
        runner.save()
    print(json.dumps(dict(output=str(runner.output),
        model_interface_calls=len(runner.report['calls']),
        provider_completions=sum((call.get('provider') or {}).get('completion_calls', 0)
                                 for call in runner.report['calls']),
        hard_stop=runner.report['hard_stop']), ensure_ascii=False), flush=True)
    return 2 if runner.report['hard_stop'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
