from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import agent.application.cli as cli_module
from agent.application.cli import main


def _tiny_h5ad(path: Path) -> Path:
    ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[1, 0], [0, 1]], dtype=np.float32)),
        obs=pd.DataFrame(index=pd.Index(["cell-1", "cell-2"], dtype="object")),
        var=pd.DataFrame(index=pd.Index(["peak-1", "peak-2"], dtype="object")),
    ).write_h5ad(path)
    return path


def _run_args(tmp_path: Path, *, request_id: str = "cli-run") -> list[str]:
    source = _tiny_h5ad(tmp_path / f"{request_id}.h5ad")
    return [
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

    def fail(**_: object) -> object:
        raise RuntimeError(secret)

    monkeypatch.setattr(cli_module, "OpenAIPlanningModel", fail)
    arguments = _run_args(tmp_path, request_id="provider-failure") + [
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
    assert json.loads(captured.err)["error"]["code"] == "CLI_INPUT_INVALID"


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
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 3
    assert captured.err == ""
    assert payload["status"] == "FAILED"
    assert payload["error"]["code"] == "UNSUPPORTED_REQUEST"
