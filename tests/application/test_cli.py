from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import agent.application.cli as cli_module
from agent.application.cli import main
from agent.orchestration import PlanningModelError
from agent.providers import PlanningModelFactoryRegistry


def _tiny_h5ad(path: Path) -> Path:
    ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[1, 0], [0, 1]], dtype=np.float32)),
        obs=pd.DataFrame(index=pd.Index(["cell-1", "cell-2"], dtype="object")),
        var=pd.DataFrame(index=pd.Index(["peak-1", "peak-2"], dtype="object")),
    ).write_h5ad(path)
    return path


def _run_args(
    tmp_path: Path,
    *,
    request_id: str = "cli-run",
    planner: str | None = "deterministic",
) -> list[str]:
    source = _tiny_h5ad(tmp_path / f"{request_id}.h5ad")
    arguments = [
        "run",
        "--request-id",
        request_id,
        "--request",
        "Inspect this scATAC dataset and generate a report.",
        "--workspace",
        str(tmp_path / "workspace"),
        "--input",
        str(source),
    ]
    if planner is not None:
        arguments.extend(("--planner", planner))
    return arguments


class _FixedPlanningModel:
    model_id = "custom:cli-model"

    def __init__(self, response: str | None = None) -> None:
        self.calls = 0
        self.response_schemas: list[object] = []
        self._response = _planning_response() if response is None else response

    def complete(self, *, prompt: str, response_schema: object) -> str:
        del prompt
        self.calls += 1
        self.response_schemas.append(response_schema)
        return self._response


def _planning_response() -> str:
    return json.dumps(
        {
            "schema_version": 3,
            "status": "plan",
            "steps": [
                {
                    "step_id": "inspect",
                    "tool_name": "inspect_scATAC",
                    "arguments": {
                        "path": {
                            "binding_type": "input",
                            "input_name": "input_path",
                        }
                    },
                    "depends_on": [],
                    "description": None,
                }
            ],
            "reason": None,
        }
    )


def _semantic_planning_response() -> str:
    return json.dumps(
        {
            "schema_version": 4,
            "decision": {
                "kind": "plan",
                "steps": [
                    {
                        "step_id": "inspect",
                        "tool": "inspect_scATAC",
                        "sources": [],
                        "control_dependencies": [],
                    }
                ],
            },
        }
    )


def test_cli_default_mode_is_llm_and_requires_explicit_primary_profile(
    tmp_path: Path, capsys
) -> None:
    arguments = _run_args(tmp_path, request_id="missing-profile", planner=None)

    code = main(arguments)

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == (
        "PLANNING_MODEL_PROFILE_REQUIRED"
    )
    run_state = tmp_path / "workspace" / "run_state"
    assert run_state.is_dir()
    assert tuple(run_state.iterdir()) == ()


def test_cli_default_llm_mode_uses_explicit_provider_model_without_hidden_default(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    model = _FixedPlanningModel()
    factories = PlanningModelFactoryRegistry({"openai": lambda _: model})
    monkeypatch.setattr(
        cli_module,
        "build_default_planning_model_factory_registry",
        lambda: factories,
    )
    arguments = _run_args(
        tmp_path, request_id="default-llm", planner=None
    ) + ["--provider", "openai", "--model", "configured-model", "--plan-only"]

    code = main(arguments)

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "PLANNED"
    assert model.calls == 1
    assert model.response_schemas[0]["properties"]["schema_version"][
        "enum"
    ] == (3,)


def test_cli_explicit_wire_v3_preserves_v3_planning(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    model = _FixedPlanningModel()
    factories = PlanningModelFactoryRegistry({"openai": lambda _: model})
    monkeypatch.setattr(
        cli_module,
        "build_default_planning_model_factory_registry",
        lambda: factories,
    )
    arguments = _run_args(
        tmp_path, request_id="explicit-v3", planner=None
    ) + [
        "--provider",
        "openai",
        "--model",
        "configured-model",
        "--wire-mode",
        "v3",
        "--plan-only",
    ]

    code = main(arguments)

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "PLANNED"
    assert model.calls == 1
    assert model.response_schemas[0]["properties"]["schema_version"][
        "enum"
    ] == (3,)


def test_cli_wire_v4_routes_through_application_owned_semantic_planner(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    model = _FixedPlanningModel(_semantic_planning_response())
    profiles = []

    def create(profile):
        profiles.append(profile)
        return model

    factories = PlanningModelFactoryRegistry({"groq": create})
    monkeypatch.setattr(
        cli_module,
        "build_default_planning_model_factory_registry",
        lambda: factories,
    )
    arguments = _run_args(
        tmp_path, request_id="explicit-v4", planner=None
    ) + [
        "--provider",
        "groq",
        "--model",
        "configured-model",
        "--wire-mode",
        "v4",
        "--plan-only",
    ]

    code = main(arguments)

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "PLANNED"
    assert payload["tool_names"] == ["inspect_scATAC"]
    assert model.calls == 1
    assert model.response_schemas[0]["properties"]["schema_version"][
        "enum"
    ] == (4,)
    assert len(profiles) == 1
    assert profiles[0].provider_id == "groq"
    assert profiles[0].model_id == "configured-model"


def test_cli_deterministic_provider_alias_remains_explicit(
    tmp_path: Path, capsys
) -> None:
    arguments = _run_args(
        tmp_path, request_id="deterministic-alias", planner=None
    ) + ["--provider", "deterministic", "--plan-only"]

    code = main(arguments)

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "PLANNED"
    assert payload["tool_names"] == ["inspect_scATAC"]


def test_cli_rejects_deterministic_and_llm_configuration_conflicts(
    tmp_path: Path, capsys
) -> None:
    arguments = _run_args(
        tmp_path, request_id="planner-conflict", planner="deterministic"
    ) + ["--provider", "openai", "--model", "configured-model"]

    code = main(arguments)

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "CLI_INPUT_INVALID"


@pytest.mark.parametrize("wire_mode", ("v3", "v4"))
def test_cli_rejects_wire_mode_for_deterministic_planning(
    tmp_path: Path, capsys, wire_mode: str
) -> None:
    arguments = _run_args(
        tmp_path, request_id=f"deterministic-{wire_mode}"
    ) + ["--wire-mode", wire_mode]

    code = main(arguments)

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "CLI_INPUT_INVALID"


def test_cli_rejects_wire_mode_for_deterministic_provider_alias(
    tmp_path: Path, capsys
) -> None:
    arguments = _run_args(
        tmp_path, request_id="deterministic-alias-wire", planner=None
    ) + ["--provider", "deterministic", "--wire-mode", "v4"]

    code = main(arguments)

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "CLI_INPUT_INVALID"


def test_cli_rejects_invalid_wire_mode_through_argparse(
    tmp_path: Path, capsys
) -> None:
    arguments = _run_args(
        tmp_path, request_id="invalid-wire", planner=None
    ) + ["--wire-mode", "v5"]

    with pytest.raises(SystemExit) as raised:
        main(arguments)

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--wire-mode" in captured.err
    assert "{v3,v4}" in captured.err


def test_cli_missing_credential_error_is_stable_and_never_falls_back(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    secret = "missing-credential-secret"

    def fail(_):
        raise PlanningModelError(
            secret, code="PLANNING_PROVIDER_CONFIGURATION_FAILED"
        )

    monkeypatch.setattr(
        cli_module,
        "build_default_planning_model_factory_registry",
        lambda: PlanningModelFactoryRegistry({"openai": fail}),
    )
    arguments = _run_args(
        tmp_path, request_id="missing-credential", planner=None
    ) + ["--provider", "openai", "--model", "configured-model"]

    code = main(arguments)

    captured = capsys.readouterr()
    assert code == 2
    assert secret not in captured.err
    assert json.loads(captured.err)["error"]["code"] == (
        "PLANNING_PROVIDER_CONFIGURATION_FAILED"
    )


def test_cli_optional_secondary_profile_uses_existing_final_failover(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    class TimeoutPlanningModel:
        model_id = "custom:primary-timeout"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *, prompt: str, response_schema: object) -> str:
            del prompt, response_schema
            self.calls += 1
            raise PlanningModelError(
                code="PROVIDER_TIMEOUT", retry_after_seconds=0
            )

    primary = TimeoutPlanningModel()
    secondary = _FixedPlanningModel()
    factories = PlanningModelFactoryRegistry(
        {"openai": lambda _: primary, "groq": lambda _: secondary}
    )
    monkeypatch.setattr(
        cli_module,
        "build_default_planning_model_factory_registry",
        lambda: factories,
    )
    arguments = _run_args(
        tmp_path, request_id="cli-failover", planner=None
    ) + [
        "--provider",
        "openai",
        "--model",
        "primary-model",
        "--secondary-provider",
        "groq",
        "--secondary-model",
        "secondary-model",
        "--plan-only",
    ]

    code = main(arguments)

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "PLANNED"
    assert primary.calls == 2
    assert secondary.calls == 1


def test_cli_run_prints_compact_success_json(tmp_path: Path, capsys) -> None:
    code = main(_run_args(tmp_path))

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert captured.err == ""
    assert payload["status"] == "SUCCEEDED"
    assert payload["tool_names"] == ["inspect_scATAC"]
    assert payload["visualization_present"] is False
    assert Path(payload["report_path"]).is_file()
    assert "run_result" not in payload


def test_cli_plan_only_prints_plan_without_report(tmp_path: Path, capsys) -> None:
    arguments = _run_args(tmp_path, request_id="cli-plan") + ["--plan-only"]

    code = main(arguments)

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "PLANNED"
    assert payload["tool_names"] == ["inspect_scATAC"]
    assert payload["report_path"] is None


def test_cli_resume_and_cancel_terminal_run(tmp_path: Path, capsys) -> None:
    assert main(_run_args(tmp_path, request_id="cli-lifecycle")) == 0
    first = json.loads(capsys.readouterr().out)

    assert main(
        [
            "resume",
            "--workspace",
            str(tmp_path / "workspace"),
            "--run-id",
            first["run_id"],
        ]
    ) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["status"] == "SUCCEEDED"

    assert main(
        [
            "cancel",
            "--workspace",
            str(tmp_path / "workspace"),
            "--run-id",
            first["run_id"],
        ]
    ) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["disposition"] == "ALREADY_TERMINAL"


def test_cli_rejects_invalid_json_file(tmp_path: Path, capsys) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{invalid", encoding="utf-8")
    arguments = _run_args(tmp_path, request_id="bad-json") + [
        "--inputs-json",
        str(inputs),
    ]

    code = main(arguments)

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "CLI_INPUT_INVALID"


def test_cli_rejects_reserved_output_root_from_json(tmp_path: Path, capsys) -> None:
    source = _tiny_h5ad(tmp_path / "reserved.h5ad")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps({"input_path": str(source), "output_dir": "/tmp/untrusted"}),
        encoding="utf-8",
    )

    code = main(
        [
            "run",
            "--request-id",
            "reserved",
            "--request",
            "Inspect this dataset.",
                "--workspace",
                str(tmp_path / "workspace"),
                "--planner",
                "deterministic",
                "--inputs-json",
                str(inputs),
        ]
    )

    payload = json.loads(capsys.readouterr().err)
    assert code == 2
    assert payload["error"]["code"] == "APP_REQUEST_INVALID"


def test_cli_provider_initialization_failure_is_sanitized(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    secret = "secret-provider-token"

    def fail(profile) -> object:
        assert profile.provider_id == "openai"
        assert profile.model_id == "test-model"
        raise RuntimeError(secret)

    monkeypatch.setattr(
        cli_module,
        "build_default_planning_model_factory_registry",
        lambda: PlanningModelFactoryRegistry({"openai": fail}),
    )
    arguments = _run_args(
        tmp_path, request_id="provider-failure", planner="llm"
    ) + [
        "--provider",
        "openai",
        "--model",
        "test-model",
    ]

    code = main(arguments)

    captured = capsys.readouterr()
    assert code == 2
    assert secret not in captured.err
    assert secret not in captured.out
    assert (
        json.loads(captured.err)["error"]["code"]
        == "PLANNING_PROVIDER_CONFIGURATION_FAILED"
    )


def test_cli_runtime_failure_has_nonzero_exit_and_safe_compact_error(
    tmp_path: Path, capsys
) -> None:
    code = main(
        [
            "run",
            "--request-id",
            "unsupported",
            "--request",
            "Write a poem.",
            "--workspace",
            str(tmp_path / "workspace"),
            "--planner",
            "deterministic",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 3
    assert captured.err == ""
    assert payload["status"] == "FAILED"
    assert payload["error"]["code"] == "UNSUPPORTED_REQUEST"
