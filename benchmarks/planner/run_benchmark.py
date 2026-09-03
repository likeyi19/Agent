"""Command-line runner for the M9.1 offline and explicitly opt-in live tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from agent.providers import (
    BUILTIN_PLANNING_PROVIDER_IDS,
    build_default_planning_model_factory_registry,
    build_planning_model_profile,
)

from benchmarks.planner.benchmark import (
    load_cases,
    load_replay_overrides,
    run_benchmark,
)


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASES = _ROOT / "benchmarks" / "planner" / "cases.json"
_DEFAULT_REPLAYS = (
    _ROOT / "tests" / "benchmarks" / "fixtures" / "offline_replays.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the provider-neutral LLM planner robustness benchmark."
    )
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    parser.add_argument("--replays", type=Path, default=_DEFAULT_REPLAYS)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Explicitly authorize calls to one configured planning provider.",
    )
    parser.add_argument("--provider", choices=BUILTIN_PLANNING_PROVIDER_IDS)
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def _live_model(provider: str | None, model: str | None, timeout: float):
    if provider is None or not isinstance(model, str) or not model.strip():
        raise ValueError("Live benchmark requires --provider and --model.")
    profile = build_planning_model_profile(
        provider,
        model.strip(),
        request_timeout_seconds=timeout,
    )
    planning_model = build_default_planning_model_factory_registry().create(profile)
    return profile, planning_model


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    cases = load_cases(arguments.cases)
    selected = frozenset(arguments.case_id) if arguments.case_id else None
    if arguments.live:
        profile, model = _live_model(
            arguments.provider, arguments.model, arguments.timeout
        )
        report = run_benchmark(
            cases,
            model=model,
            model_profile=profile,
            repetitions=arguments.repeat,
            selected_case_ids=selected,
        )
    else:
        if arguments.provider is not None or arguments.model is not None:
            raise ValueError("--provider and --model require the explicit --live flag.")
        report = run_benchmark(
            cases,
            replay_overrides=load_replay_overrides(arguments.replays),
            repetitions=arguments.repeat,
            selected_case_ids=selected,
        )
    rendered = json.dumps(
        report.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    if arguments.output is not None:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
