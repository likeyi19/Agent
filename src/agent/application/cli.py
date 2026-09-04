"""One-shot standard-library CLI for the research application."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from agent.orchestration import (
    DeterministicPlanner,
    RunStoreError,
)
from agent.providers import (
    BUILTIN_PLANNING_PROVIDER_IDS,
    build_default_planning_model_factory_registry,
    build_planning_model_profile,
)
from agent.schemas import AgentRequest, RunMode

from .schemas import ApplicationResult, ApplicationStatus
from .service import ApplicationServiceError, ResearchAgentApplication


_EXIT_INVALID = 2
_EXIT_RUNTIME_FAILED = 3
_EXIT_CANCELLED = 4
_EXIT_POSTPROCESSING_FAILED = 5


class _CliInputError(ValueError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent",
        description="Run durable verified single-cell epigenomic workflows.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Plan and execute one request.")
    run.add_argument("--request-id", required=True)
    run.add_argument("--request", required=True)
    run.add_argument("--workspace", required=True)
    run.add_argument("--input", dest="input_path")
    run.add_argument("--species", choices=("human", "mouse"))
    run.add_argument("--checkpoint", dest="checkpoint_path")
    run.add_argument("--device")
    run.add_argument("--inputs-json", type=Path)
    run.add_argument("--plan-only", action="store_true")
    run.add_argument(
        "--planner",
        choices=("llm", "deterministic"),
        help="New-run planning mode (default: llm).",
    )
    run.add_argument(
        "--provider",
        choices=("deterministic", *BUILTIN_PLANNING_PROVIDER_IDS),
        help="Explicit LLM provider; deterministic is a compatibility alias.",
    )
    run.add_argument("--model")
    run.add_argument(
        "--secondary-provider", choices=BUILTIN_PLANNING_PROVIDER_IDS
    )
    run.add_argument("--secondary-model")

    resume = commands.add_parser("resume", help="Resume one durable run.")
    resume.add_argument("--workspace", required=True)
    resume.add_argument("--run-id", required=True)

    cancel = commands.add_parser("cancel", help="Request cooperative cancellation.")
    cancel.add_argument("--workspace", required=True)
    cancel.add_argument("--run-id", required=True)
    return parser


def _strict_inputs(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise _CliInputError("Structured input JSON contains duplicate keys.")
            result[key] = value
        return result

    def reject_constant(_: str) -> object:
        raise _CliInputError("Structured input JSON contains a non-finite number.")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except _CliInputError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _CliInputError("Structured input JSON could not be read safely.") from exc
    if not isinstance(value, dict):
        raise _CliInputError("Structured input JSON must contain one object.")
    return value


def _planning_options(arguments: argparse.Namespace) -> dict[str, object]:
    planner_mode = arguments.planner
    provider = arguments.provider
    model = arguments.model
    secondary_provider = arguments.secondary_provider
    secondary_model = arguments.secondary_model

    deterministic_alias = provider == "deterministic"
    deterministic_mode = planner_mode == "deterministic" or (
        planner_mode is None and deterministic_alias
    )
    if deterministic_mode:
        conflicting = bool(
            (provider is not None and provider != "deterministic")
            or model is not None
            or secondary_provider is not None
            or secondary_model is not None
        )
        if conflicting:
            raise _CliInputError(
                "Deterministic planning cannot use provider/model configuration."
            )
        return {"planner": DeterministicPlanner()}

    if deterministic_alias:
        raise _CliInputError(
            "The deterministic provider alias conflicts with LLM planner mode."
        )
    if provider is None and model is None:
        if secondary_provider is not None or secondary_model is not None:
            raise _CliInputError(
                "A secondary planning profile requires a primary provider and model."
            )
        return {}
    if provider is None or not isinstance(model, str) or not model.strip():
        raise _CliInputError("LLM planning requires --provider and --model.")
    if (secondary_provider is None) != (secondary_model is None):
        raise _CliInputError(
            "Secondary planning requires both --secondary-provider and "
            "--secondary-model."
        )
    try:
        primary_profile = build_planning_model_profile(provider, model.strip())
        recovery_profile = (
            None
            if secondary_provider is None
            else build_planning_model_profile(
                secondary_provider,
                secondary_model.strip(),  # type: ignore[union-attr]
            )
        )
    except Exception as exc:
        raise _CliInputError(
            "Planning-model profile configuration is invalid."
        ) from exc
    return {
        "primary_planning_profile": primary_profile,
        "recovery_planning_profile": recovery_profile,
        "planning_model_factory_registry": (
            build_default_planning_model_factory_registry()
        ),
    }


def _merge_run_inputs(arguments: argparse.Namespace) -> dict[str, object]:
    inputs = _strict_inputs(arguments.inputs_json)
    explicit = {
        "input_path": arguments.input_path,
        "species": arguments.species,
        "checkpoint_path": arguments.checkpoint_path,
        "device": arguments.device,
    }
    for key, value in explicit.items():
        if value is None:
            continue
        if key in inputs:
            raise _CliInputError(
                "A structured input was supplied both explicitly and through JSON."
            )
        inputs[key] = value
    return inputs


def _compact_result(result: ApplicationResult) -> dict[str, object]:
    plan = result.run_result.plan
    tool_names = [] if plan is None else [step.tool_name for step in plan.steps]
    error: dict[str, object] | None = None
    if result.error is not None:
        error = result.error.to_dict()
    elif result.run_result.errors:
        authoritative = result.run_result.errors[0]
        error = {
            "code": authoritative.code,
            "message": authoritative.message,
            "stage": "RUNTIME",
        }
    return {
        "status": result.status.value,
        "run_status": result.run_status.value,
        "request_id": result.request_id,
        "run_id": result.run_id,
        "tool_names": tool_names,
        "workspace_path": result.workspace_path,
        "evidence_path": None if result.evidence is None else result.evidence.path,
        "visualization_present": result.visualization is not None,
        "report_path": None if result.report is None else result.report.path,
        "error": error,
    }


def _emit(value: object, *, stream) -> None:
    print(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True),
        file=stream,
    )


def _result_exit_code(result: ApplicationResult) -> int:
    if result.status in {ApplicationStatus.SUCCEEDED, ApplicationStatus.PLANNED}:
        return 0
    if result.status is ApplicationStatus.CANCELLED:
        return _EXIT_CANCELLED
    if result.error is not None:
        return _EXIT_POSTPROCESSING_FAILED
    return _EXIT_RUNTIME_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run":
            application = ResearchAgentApplication(
                arguments.workspace, **_planning_options(arguments)
            )
            request = AgentRequest(
                arguments.request_id,
                arguments.request,
                _merge_run_inputs(arguments),
                RunMode.PLAN_ONLY if arguments.plan_only else RunMode.EXECUTE,
            )
            result = application.run(request)
            _emit(_compact_result(result), stream=sys.stdout)
            return _result_exit_code(result)
        application = ResearchAgentApplication(arguments.workspace)
        if arguments.command == "resume":
            result = application.resume(arguments.run_id)
            _emit(_compact_result(result), stream=sys.stdout)
            return _result_exit_code(result)
        receipt = application.cancel(arguments.run_id)
        _emit(
            {
                "run_id": receipt.run_id,
                "disposition": receipt.disposition.value,
                "requested_at": receipt.requested_at,
                "terminal_status": (
                    None
                    if receipt.terminal_status is None
                    else receipt.terminal_status.value
                ),
            },
            stream=sys.stdout,
        )
        return 0
    except ApplicationServiceError as exc:
        _emit({"status": "FAILED", "error": exc.error.to_dict()}, stream=sys.stderr)
        return _EXIT_INVALID
    except _CliInputError:
        _emit(
            {
                "status": "FAILED",
                "error": {
                    "code": "CLI_INPUT_INVALID",
                    "message": "CLI input or provider configuration is invalid.",
                },
            },
            stream=sys.stderr,
        )
        return _EXIT_INVALID
    except RunStoreError:
        _emit(
            {
                "status": "FAILED",
                "error": {
                    "code": "CLI_RUNTIME_STATE_ERROR",
                    "message": "Durable runtime state operation failed.",
                },
            },
            stream=sys.stderr,
        )
        return _EXIT_RUNTIME_FAILED
    except (TypeError, ValueError):
        _emit(
            {
                "status": "FAILED",
                "error": {
                    "code": "CLI_INPUT_INVALID",
                    "message": "CLI input is invalid.",
                },
            },
            stream=sys.stderr,
        )
        return _EXIT_INVALID


__all__ = ["main"]
