"""Synthetic M10.5 sources; real planning, execution, verification and composition."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.orchestration import AgentRuntime, LLMPlanner, ToolRegistry, build_default_tool_registry
from agent.schemas import AgentRequest, RunStatus


class RawModel:
    model_id = "scripted-raw-intake-v4"

    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        return json.dumps({"schema_version": 4, "decision": {"kind": "plan", "steps": [
            {"step_id": "raw", "tool": "inspect_raw_scATAC", "sources": [
                {"target": "raw_input", "source": {"kind": "input", "input": "raw_input_paths"}}
            ], "control_dependencies": []}
        ]}})


@pytest.fixture
def intake_case(raw_factory):
    def make(case="human_fastq"):
        bam = case.endswith("bam")
        species = "mouse" if case == "mouse_fastq" else "human"
        args = raw_factory("bam" if bam else "fastq", species=species,
            assembly="hg19" if case == "mismatch_bam" else None,
            malformed=case == "invalid_fastq", root=case)
        folder = Path(args["raw_input_paths"])
        readiness = "READY"
        if case == "missing_species":
            args.pop("species")
            readiness = "NEEDS_USER_INPUT"
        elif case == "missing_role":
            next(folder.glob("*R3*.fastq")).unlink()
            readiness = "NEEDS_USER_INPUT"
        elif case == "unsupported_fastq":
            args["species"] = "rat"
            readiness = "UNSUPPORTED"
        elif case == "invalid_fastq":
            readiness = "INVALID"
        elif case == "bounded_fastq":
            payload = b"".join(b"@PRIVATE_READ_%d\nACGT\n+\nIIII\n" % i for i in range(300))
            for path in folder.iterdir():
                path.write_bytes(payload)
        elif case == "mismatch_bam":
            readiness = "NEEDS_USER_INPUT"
        elif case == "unknown_bam":
            import pysam
            path = folder / "input.bam"
            with pysam.AlignmentFile(str(path), "rb") as source:
                header = source.header.to_dict()
                reads = list(source.fetch(until_eof=True))
            for sq in header["SQ"]:
                sq.pop("AS")
            with pysam.AlignmentFile(str(path), "wb", header=header) as dest:
                for read in reads:
                    dest.write(read)
            readiness = "NEEDS_USER_INPUT"
        return SimpleNamespace(args=args, readiness=readiness, folder=folder)
    return make


@pytest.fixture
def raw_context():
    default = build_default_tool_registry()
    production = Mock(wraps=default.get("inspect_raw_scATAC").function)
    forbidden = Mock(side_effect=AssertionError("Unexpected scientific production"))
    registry = ToolRegistry(tuple(replace(default.get(n),
        function=production if n == "inspect_raw_scATAC" else forbidden) for n in default.names()))
    model = RawModel()
    return SimpleNamespace(registry=registry, production=production, model=model,
                          planner=LLMPlanner(model))


@pytest.fixture
def raw_run(intake_case, raw_context):
    def make(case="human_fastq"):
        source = intake_case(case)
        request = AgentRequest("raw-report", "Inspect these raw scATAC inputs and report preprocessing readiness.", source.args)
        run = AgentRuntime(planner=raw_context.planner, registry=raw_context.registry).run(request)
        assert run.status is RunStatus.SUCCEEDED, run.errors
        assert run.steps[0].result["readiness"] == source.readiness
        return SimpleNamespace(source=source, run=run, context=raw_context)
    return make
