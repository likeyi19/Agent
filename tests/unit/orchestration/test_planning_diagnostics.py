"""Deterministic M9.2 tests for sanitized LLM planning diagnostics."""

from __future__ import annotations

from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    FileRunStore,
    LLMPlanner,
    PlannerError,
    PlanningModelError,
    PlanningModelProfile,
    RunMode,
    RunStatus,
    ToolRegistry,
    build_default_tool_registry,
)


class DiagnosticPlanningModel:
    model_id = "diagnostic-model-v1"

    def __init__(
        self,
        response: object = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = _plan_response(_inspect_step()) if response is None else response
        self.error = error
        self.calls = 0

    def complete(self, *, prompt: str, response_schema) -> str:
        self.calls += 1
        json.loads(prompt)
        json.dumps(response_schema, allow_nan=False)
        if self.error is not None:
            raise self.error
        return self.response  # type: ignore[return-value]


def _input_binding(input_name: str) -> dict[str, object]:
    return {
        "binding_type": "input",
        "input_name": input_name,
    }


def _ref_binding(step_id: str, output_key: str) -> dict[str, object]:
    return {
        "binding_type": "ref",
        "ref_step_id": step_id,
        "ref_output_key": output_key,
    }


def _arguments(tool_name: str, **bindings: object) -> dict[str, object]:
    spec = build_default_tool_registry().get(tool_name)
    arguments: dict[str, object] = {
        name: None for name in spec.optional_arguments
    }
    arguments.update(bindings)
    return arguments


def _inspect_step(**changes: object) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "inspect",
        "tool_name": "inspect_scATAC",
        "arguments": _arguments(
            "inspect_scATAC", path=_input_binding("input_path")
        ),
        "depends_on": [],
        "description": None,
    }
    step.update(changes)
    return step


def _embed_step(**changes: object) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "embed",
        "tool_name": "epizoo_embed_cells",
        "arguments": _arguments(
            "epizoo_embed_cells",
            input_path=_input_binding("input_path"),
            output_dir=_input_binding("output_dir"),
            species=_input_binding("species"),
        ),
        "depends_on": [],
        "description": None,
    }
    step.update(changes)
    return step


def _plan_response(*steps: dict[str, object], **changes: object) -> str:
    response: dict[str, object] = {
        "schema_version": 3,
        "status": "plan",
        "steps": list(steps),
        "reason": None,
    }
    response.update(changes)
    return json.dumps(response)


def _request(
    *,
    request_id: str = "diagnostic-request",
    prompt: str = "Inspect the supplied scATAC dataset.",
    inputs: dict[str, object] | None = None,
) -> AgentRequest:
    return AgentRequest(
        request_id,
        prompt,
        inputs or {"input_path": "/synthetic/input.h5ad"},
        RunMode.PLAN_ONLY,
    )


def _guarded_registry() -> tuple[ToolRegistry, Mock]:
    source = build_default_tool_registry()
    guard = Mock(side_effect=AssertionError("diagnostic planning executed a tool"))
    return (
        ToolRegistry(
            tuple(
                replace(source.get(name), function=guard)
                for name in source.names()
            )
        ),
        guard,
    )


def _run(
    model: DiagnosticPlanningModel,
    *,
    request: AgentRequest | None = None,
    run_store: FileRunStore | None = None,
    profile: PlanningModelProfile | None = None,
    expected_calls: int | None = None,
):
    registry, guard = _guarded_registry()
    result = AgentRuntime(
        planner=LLMPlanner(model, profile=profile),
        registry=registry,
        run_store=run_store,
    ).run(request or _request())
    guard.assert_not_called()
    if expected_calls is not None:
        assert model.calls == expected_calls
    return result


def _diagnostics(result) -> list[dict[str, object]]:
    return [
        dict(event.details)
        for event in result.trace
        if "diagnostic_schema_version" in event.details
    ]


def _last_attempt_diagnostic(result) -> dict[str, object]:
    return next(
        item
        for item in reversed(_diagnostics(result))
        if item["stage"] != "recovery"
    )


def test_successful_planning_emits_complete_sanitized_diagnostics() -> None:
    model = DiagnosticPlanningModel()

    result = _run(model)
    diagnostics = _diagnostics(result)

    assert result.status is RunStatus.PLANNED
    assert [item["stage"] for item in diagnostics] == [
        "provider",
        "provider",
        "parse",
        "schema",
        "argument_binding",
        "dependency_reference",
        "candidate",
        "preflight",
        "accepted",
        "recovery",
    ]
    assert diagnostics[-2]["code"] == "FINAL_PLAN_ACCEPTED"
    assert diagnostics[-1]["code"] == "PLANNING_RECOVERY_SUMMARY"
    assert diagnostics[-2]["candidate_constructed"] is True
    assert diagnostics[-2]["candidate_preflight_passed"] is True
    assert diagnostics[-1]["total_provider_call_count"] == 1
    assert diagnostics[-1]["repair_used"] is False
    assert diagnostics[-1]["retry_used"] is False
    assert diagnostics[-1]["failover_used"] is False
    assert diagnostics[-1]["diagnostic_schema_version"] == 3
    assert diagnostics[-1]["planning_wire_schema_version"] == 3
    assert diagnostics[-1]["profile_id"] == "unprofiled"
    assert diagnostics[-1]["provider_id"] == "custom"
    assert len(diagnostics[-1]["model_identity_digest"]) == 64
    assert "diagnostic-model-v1" not in diagnostics[-1]["model_identity_digest"]
    assert len(diagnostics[-1]["catalog_fingerprint"]) == 64
    assert "inspect_scATAC" in diagnostics[-1]["offered_tool_names"]
    assert all(item["stage"] != "semantic" for item in diagnostics)


def test_profile_aware_diagnostics_preserve_identity_without_raw_model(
    tmp_path,
) -> None:
    raw_model_id = "MODEL-SECRET-9f1"
    model = DiagnosticPlanningModel()
    model.model_id = raw_model_id
    profile = PlanningModelProfile(
        profile_id="primary-planner",
        provider_id="groq",
        model_id=raw_model_id,
    )

    store = FileRunStore(tmp_path)
    result = _run(model, profile=profile, run_store=store)
    restored = store.load(result.run_id).to_run_result()
    diagnostic = _diagnostics(restored)[-1]
    rendered = json.dumps(restored.to_dict())

    assert diagnostic["diagnostic_schema_version"] == 3
    assert diagnostic["profile_id"] == "primary-planner"
    assert diagnostic["provider_id"] == "groq"
    assert len(diagnostic["model_identity_digest"]) == 64
    assert raw_model_id not in rendered


@pytest.mark.parametrize(
    "code",
    [
        "PROVIDER_AUTHENTICATION_FAILED",
        "PROVIDER_RATE_LIMITED",
        "PROVIDER_TIMEOUT",
        "PROVIDER_CONNECTION_FAILED",
        "PROVIDER_UNAVAILABLE",
        "PLANNING_PROVIDER_ERROR",
    ],
)
def test_provider_failures_are_distinct_and_transport_codes_retry_once(code: str) -> None:
    model = DiagnosticPlanningModel(
        error=PlanningModelError("raw provider body secret", code=code)
    )

    expected_calls = 2 if code in {
        "PROVIDER_RATE_LIMITED",
        "PROVIDER_TIMEOUT",
        "PROVIDER_CONNECTION_FAILED",
        "PROVIDER_UNAVAILABLE",
    } else 1
    result = _run(model, expected_calls=expected_calls)
    diagnostic = _last_attempt_diagnostic(result)

    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == code
    assert diagnostic["stage"] == "provider"
    assert diagnostic["code"] == code
    assert diagnostic["outcome"] == "failed"
    assert _diagnostics(result)[-1]["total_provider_call_count"] == expected_calls
    assert "secret" not in json.dumps(result.to_dict()).casefold()


def test_unclassified_provider_exception_is_sanitized() -> None:
    model = DiagnosticPlanningModel(
        error=RuntimeError("Authorization: Bearer API-SECRET; raw HTTP body")
    )

    result = _run(model)

    assert result.errors[0].code == "PLANNING_PROVIDER_ERROR"
    assert _last_attempt_diagnostic(result)["stage"] == "provider"
    assert "api-secret" not in json.dumps(result.to_dict()).casefold()
    assert "authorization" not in json.dumps(result.to_dict()).casefold()


def test_untrusted_provider_error_code_is_not_persisted() -> None:
    secret_code = "API-SECRET-9f1"
    result = _run(
        DiagnosticPlanningModel(
            error=PlanningModelError("raw body", code=secret_code)
        )
    )
    serialized = json.dumps(result.to_dict())

    assert result.errors[0].code == "PLANNING_PROVIDER_ERROR"
    assert secret_code not in serialized


@pytest.mark.parametrize("response", ["not json", "```json\n{}\n```"])
def test_malformed_and_markdown_json_fail_at_parse(response: str) -> None:
    result = _run(DiagnosticPlanningModel(response))
    diagnostic = _last_attempt_diagnostic(result)

    assert result.errors[0].code == "PLANNER_OUTPUT_INVALID"
    assert diagnostic["stage"] == "parse"
    assert diagnostic["reason_code"] == "malformed_json"


def test_wire_schema_invalid_output_is_distinguished_from_parse() -> None:
    result = _run(
        DiagnosticPlanningModel(
            json.dumps({"schema_version": 3, "status": "plan", "steps": []})
        )
    )
    diagnostic = _last_attempt_diagnostic(result)

    assert diagnostic["stage"] == "schema"
    assert diagnostic["reason_code"] == "wire_schema_invalid"


def test_invalid_binding_reports_safe_local_position() -> None:
    invalid = _input_binding("input_path")
    invalid["binding_type"] = "literal"
    result = _run(
        DiagnosticPlanningModel(
            _plan_response(_inspect_step(arguments={"path": invalid}))
        )
    )
    diagnostic = _last_attempt_diagnostic(result)

    assert result.errors[0].code == "PLANNER_BINDING_INVALID"
    assert diagnostic["stage"] == "argument_binding"
    assert diagnostic["step_index"] == 0
    assert diagnostic["argument_name"] == "path"
    assert diagnostic["reason_code"] == "binding_type_invalid"


def test_missing_model_requested_input_reports_name_without_value() -> None:
    result = _run(
        DiagnosticPlanningModel(
            _plan_response(
                _inspect_step(
                    arguments={"path": _input_binding("missing_input")}
                )
            )
        )
    )
    diagnostic = _last_attempt_diagnostic(result)

    assert result.errors[0].code == "MISSING_REQUIRED_INPUT"
    assert diagnostic["stage"] == "argument_binding"
    assert "input_name" not in diagnostic
    assert diagnostic["reason_code"] == "request_input_missing"


def test_unknown_tool_is_diagnosed_during_candidate_parsing() -> None:
    result = _run(
        DiagnosticPlanningModel(
            _plan_response(
                _inspect_step(tool_name="invented_tool", arguments={})
            )
        )
    )
    diagnostic = _last_attempt_diagnostic(result)

    assert result.errors[0].code == "UNKNOWN_TOOL"
    assert diagnostic["stage"] == "tool_selection"
    assert diagnostic["step_index"] == 0
    assert "tool_name" not in diagnostic
    assert diagnostic["candidate_preflight_passed"] is None


@pytest.mark.parametrize(
    ("arguments", "reason_code", "argument_name"),
    [
        ({}, "missing_tool_argument", "path"),
        (
            {
                "path": _input_binding("input_path"),
                "input_path": _input_binding("input_path"),
            },
            "unknown_tool_argument",
            "input_path",
        ),
    ],
)
def test_missing_and_unknown_tool_arguments_are_diagnosed(
    arguments: dict[str, object],
    reason_code: str,
    argument_name: str,
) -> None:
    result = _run(
        DiagnosticPlanningModel(
            _plan_response(_inspect_step(arguments=arguments))
        )
    )
    diagnostic = _last_attempt_diagnostic(result)

    assert result.errors[0].code == "INVALID_TOOL_ARGUMENTS"
    assert diagnostic["stage"] == "argument_binding"
    assert diagnostic["reason_code"] == reason_code
    assert diagnostic["argument_name"] == argument_name


def test_invalid_dependency_is_diagnosed_before_candidate_construction() -> None:
    result = _run(
        DiagnosticPlanningModel(
            _plan_response(_inspect_step(depends_on=["missing-step"]))
        )
    )
    diagnostic = _last_attempt_diagnostic(result)

    assert result.errors[0].code == "PLANNER_STRUCTURE_INVALID"
    assert diagnostic["stage"] == "dependency_reference"
    assert diagnostic["candidate_constructed"] is False
    assert diagnostic["reason_code"] == "plan_structure_invalid"
    assert not any(
        item["stage"] == "dependency_reference"
        and item["outcome"] == "succeeded"
        for item in _diagnostics(result)
    )


def test_invalid_result_field_reference_has_safe_structural_detail() -> None:
    inspect = _inspect_step()
    embed = _embed_step(
        arguments=_arguments(
            "epizoo_embed_cells",
            input_path=_ref_binding("inspect", "embedding_path"),
            output_dir=_input_binding("output_dir"),
            species=_input_binding("species"),
        ),
        depends_on=["inspect"],
    )
    result = _run(
        DiagnosticPlanningModel(_plan_response(inspect, embed)),
        request=_request(
            inputs={
                "input_path": "/synthetic/input.h5ad",
                "output_dir": "/synthetic/output",
                "species": "mouse",
            }
        ),
    )
    diagnostic = _last_attempt_diagnostic(result)

    assert result.errors[0].code == "INVALID_OUTPUT_REFERENCE"
    assert diagnostic["stage"] == "dependency_reference"
    assert diagnostic["step_index"] == 1
    assert diagnostic["argument_name"] == "input_path"
    assert diagnostic["producer_step_index"] == 0
    assert diagnostic["output_key"] == "embedding_path"


def test_explicit_unsupported_response_has_no_refusal_prose() -> None:
    refusal = "provider refusal prose with PRIVATE-TOKEN"
    response = json.dumps(
        {
            "schema_version": 3,
            "status": "unsupported",
            "steps": [],
            "reason": refusal,
        }
    )

    result = _run(DiagnosticPlanningModel(response))
    diagnostic = _last_attempt_diagnostic(result)
    serialized = json.dumps(result.to_dict())

    assert result.errors[0].code == "UNSUPPORTED_REQUEST"
    assert diagnostic["stage"] == "unsupported"
    assert diagnostic["outcome"] == "rejected"
    assert refusal not in serialized
    assert "PRIVATE-TOKEN" not in serialized

    registry, _ = _guarded_registry()
    with pytest.raises(PlannerError) as raised:
        LLMPlanner(DiagnosticPlanningModel(response)).plan(_request(), registry)
    assert refusal not in str(raised.value)


def test_diagnostics_survive_durable_success_and_failure(tmp_path) -> None:
    success_store = FileRunStore(tmp_path / "success")
    success = _run(
        DiagnosticPlanningModel(),
        request=_request(request_id="durable-success"),
        run_store=success_store,
    )
    restored_success = success_store.load(success.run_id).to_run_result()
    assert _diagnostics(restored_success)[-1]["code"] == "PLANNING_RECOVERY_SUMMARY"

    failure_store = FileRunStore(tmp_path / "failure")
    failure = _run(
        DiagnosticPlanningModel("not json"),
        request=_request(request_id="durable-failure"),
        run_store=failure_store,
    )
    restored_failure = failure_store.load(failure.run_id).to_run_result()
    assert restored_failure.errors[0].details["stage"] == "parse"
    assert _last_attempt_diagnostic(restored_failure)["stage"] == "parse"

    unsupported_store = FileRunStore(tmp_path / "unsupported")
    unsupported_response = json.dumps(
        {
            "schema_version": 3,
            "status": "unsupported",
            "steps": [],
            "reason": "unsupported provider prose",
        }
    )
    unsupported = _run(
        DiagnosticPlanningModel(unsupported_response),
        request=_request(request_id="durable-unsupported"),
        run_store=unsupported_store,
    )
    restored_unsupported = unsupported_store.load(
        unsupported.run_id
    ).to_run_result()
    assert restored_unsupported.errors[0].details["stage"] == "unsupported"
    assert _last_attempt_diagnostic(restored_unsupported)["stage"] == "unsupported"


def test_durable_diagnostics_do_not_leak_prompt_input_or_response_values(
    tmp_path,
) -> None:
    prompt_secret = "PROMPT-SECRET-9f1"
    path_secret = "/private/PATH-SECRET-9f1/input.h5ad"
    response_secret = "RESPONSE-SECRET-9f1"
    store = FileRunStore(tmp_path)
    result = _run(
        DiagnosticPlanningModel(f"not-json {response_secret}"),
        request=_request(
            request_id="durable-privacy",
            prompt=f"Inspect data {prompt_secret}",
            inputs={"input_path": path_secret},
        ),
        run_store=store,
    )
    restored = store.load(result.run_id)
    diagnostic_payload = json.dumps(
        {
            "errors": [error.to_dict() for error in restored.errors],
            "trace": [event.to_dict() for event in restored.trace],
        }
    )

    for secret in (prompt_secret, path_secret, response_secret):
        assert secret not in diagnostic_payload
    assert "Authorization" not in diagnostic_payload
    assert "Bearer" not in diagnostic_payload


def test_invented_identifier_that_looks_like_a_secret_is_not_echoed() -> None:
    identifier_secret = "API-SECRET-9f1"
    result = _run(
        DiagnosticPlanningModel(
            _plan_response(
                _inspect_step(tool_name=identifier_secret, arguments={})
            )
        )
    )
    diagnostic_payload = json.dumps(
        {
            "errors": [error.to_dict() for error in result.errors],
            "trace": [event.to_dict() for event in result.trace],
        }
    )

    assert result.errors[0].code == "UNKNOWN_TOOL"
    assert identifier_secret not in diagnostic_payload


def test_noncanonical_direct_embedding_remains_preflight_valid() -> None:
    response = _plan_response(_embed_step())
    model = DiagnosticPlanningModel(response)
    result = _run(
        model,
        request=_request(
            prompt="Create EpiZoo embeddings.",
            inputs={
                "input_path": "/synthetic/input.h5ad",
                "output_dir": "/synthetic/output",
                "species": "mouse",
            },
        ),
    )

    assert result.status is RunStatus.PLANNED
    assert result.verification is not None and result.verification.passed
    assert _last_attempt_diagnostic(result)["code"] == "FINAL_PLAN_ACCEPTED"
