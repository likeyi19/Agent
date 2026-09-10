"""Natural-language Application composition; real backend requires explicit gate."""
from dataclasses import replace
import json
from pathlib import Path
import pytest

from agent.application import ResearchAgentApplication
from agent.orchestration import AgentRequest, LLMPlanner, RunMode, StepOutputRef, ToolRegistry, build_default_tool_registry
from agent.tools.data import scatac_fragments as public, fastq_fragments as private, _chromap as c
from fragments_helpers import BC, inputs_for
from test_fragments_contracts import bound, fake_runtime
from test_fragments_orchestration import configured, Model, public_args, StopStore, ProcessExit
from test_fragments_planning import wire, request
from test_fragments_evidence import mutate_fragment


def application_request(args, group=None):
    inputs = {key: value for key, value in args.items() if key != 'output_dir'}
    if group:
        inputs.pop('intake_manifest_path'); inputs.pop('intake_manifest_sha256')
        inputs.update(raw_input_paths=[path for _, path in group.files], species='human',
            raw_assay='TENX_ATAC', fastq_layout=group.layout.value)
    return AgentRequest('fragments-request', 'Prepare these verified raw scATAC FASTQs into canonical fragments.', inputs)


@pytest.mark.parametrize('upstream', [False, True])
def test_fragments_application_plan_only(tmp_path, monkeypatch, upstream):
    def forbidden(*args, **kwargs):
        pytest.fail('Scientific execution or inspection during Application PLAN_ONLY')
    from agent.tools.data import _fragments_binding, _fragments_fastq, scatac_reference, scatac_library_context, raw_scatac
    for module, name in ((public, 'resolve_execution'), (public, 'verification_runtime'),
        (private, 'prepare_fastq_fragments'), (_fragments_binding, 'preflight'),
        (_fragments_fastq, 'scan_group'), (scatac_reference, 'load_scatac_reference_bundle'),
        (scatac_library_context, 'inspect_barcode_whitelist'), (raw_scatac, '_reconstruct')):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.delenv('AGENT_CHROMAP_BIN', raising=False)
    monkeypatch.delenv('AGENT_CHROMAP_INDEX_ROOT', raising=False)
    registry = build_default_tool_registry()
    registry = ToolRegistry(tuple(replace(registry.get(name), function=forbidden) for name in registry.names()))
    model = Model(wire(upstream))
    app = ResearchAgentApplication(tmp_path / 'workspace', planner=LLMPlanner(model), registry=registry)
    req = request(upstream)
    result = app.run(replace(req, inputs={k: v for k, v in req.inputs.items() if k != 'output_dir'}))
    assert result.status.value == 'PLANNED' and not result.run_result.steps
    assert result.evidence is result.visualization is result.report is None
    step = result.run_result.plan.steps[-1]
    assert step.depends_on == (('inspect',) if upstream else ())
    if upstream:
        assert step.arguments['intake_manifest_path'] == StepOutputRef('inspect', 'manifest_path')
        assert step.arguments['intake_manifest_sha256'] == StepOutputRef('inspect', 'manifest_sha256')
    workspace = app._workspace.run_paths(result.run_id)
    assert step.arguments['output_dir'] == str(workspace.scientific)
    assert not list(workspace.scientific.iterdir())
    assert '/PRIVATE' not in model.prompt and '/PRIVATE' not in str(model.schema)


@pytest.mark.parametrize('mutation', [None, 'bgzf', 'tabix', 'whitelist'])
def test_fragments_application_resume_fresh_verification(configured, tmp_path, monkeypatch, mutation):
    args, control, _, white, _ = configured
    root = tmp_path / 'workspace'
    app = ResearchAgentApplication(root, planner=LLMPlanner(Model(wire())))
    result = app.run(application_request(args))
    assert result.status.value == 'SUCCEEDED', result
    assert result.evidence and result.report and result.visualization is None
    before = Path(result.report.path).read_bytes()
    if mutation:
        mutate_fragment(result.run_result, mutation, white)
    monkeypatch.delenv('AGENT_CHROMAP_BIN'); monkeypatch.delenv('AGENT_CHROMAP_INDEX_ROOT')
    monkeypatch.setattr(c, 'identify_backend', lambda *a, **k: pytest.fail('Aligner probed on resume'))
    resumed = ResearchAgentApplication(root).resume(result.run_id)
    if mutation:
        assert resumed.status.value == 'FAILED' and resumed.run_status.value == 'SUCCEEDED'
        assert resumed.error.code == 'APP_EVIDENCE_FAILED' and resumed.report is None
    else:
        assert resumed == result
        assert Path(resumed.report.path).read_bytes() == before
    assert control['calls'].count('chromap') == 1


@pytest.mark.parametrize('route', ['direct', 'inspect_dag', 'publication_recovery'])
def test_guarded_fragments_application_execute(tiny, reads, executables, monkeypatch, route):
    group, white = reads([(100, 200, BC, 300), (1100, 1200, BC, 1)])
    executable = executables['candidate']['path']
    inputs = inputs_for(tiny, [group], white, executable=executable)
    indexes = tiny['root'] / 'indexes'; indexes.mkdir()
    Path(inputs.index_path).parent.rename(indexes / 'accepted')
    monkeypatch.setenv('AGENT_CHROMAP_BIN', executable)
    monkeypatch.setenv('AGENT_CHROMAP_INDEX_ROOT', str(indexes))
    args = public_args(inputs, tiny['root'] / 'unused')
    original = private.run_stage; calls = []
    def counting(argv, **kwargs):
        if '--preset' in argv:
            calls.append(tuple(argv))
        return original(argv, **kwargs)
    monkeypatch.setattr(private, 'run_stage', counting)
    root = tiny['root'] / 'workspace'
    app = ResearchAgentApplication(root, planner=LLMPlanner(Model(wire(route == 'inspect_dag'))))
    req = application_request(args, group if route == 'inspect_dag' else None)
    if route == 'publication_recovery':
        app.runtime._run_store = StopStore(app.run_store, before_success=True)
        with pytest.raises(ProcessExit):
            app.run(req)
        monkeypatch.delenv('AGENT_CHROMAP_BIN'); monkeypatch.delenv('AGENT_CHROMAP_INDEX_ROOT')
        monkeypatch.setattr(c, 'identify_backend', lambda *a, **k: pytest.fail('Aligner probed on recovery'))
        app = ResearchAgentApplication(root)
        result = app.resume('fragments-request:run')
    else:
        result = app.run(req)
    assert result.status.value == result.run_status.value == 'SUCCEEDED', result
    assert result.evidence and result.report and result.visualization is None
    assert not list(app._workspace.run_paths(result.run_id).visualizations.iterdir())
    step = result.run_result.steps[-1]
    assert step.result['total_support'] == 301 and step.verification.passed
    assert step.result['artifact_schema_version'] == 2
    assert step.result['contract_version'] == 'scatac-fragments.v2'
    text = Path(result.report.path).read_text()
    assert 'scatac-fragments.v2' in text
    assert 'scatac-fragments.v1' not in text
    assert 'Total exact support: ` 301 `' in text
    assert 'Preprocessing has not been performed.' not in text
    if route == 'inspect_dag':
        assert step.resolved_arguments['intake_manifest_sha256'] == result.run_result.steps[0].result['manifest_sha256']
    before = Path(result.report.path).read_bytes()
    monkeypatch.delenv('AGENT_CHROMAP_BIN', raising=False)
    monkeypatch.delenv('AGENT_CHROMAP_INDEX_ROOT', raising=False)
    monkeypatch.setattr(c, 'identify_backend', lambda *a, **k: pytest.fail('Aligner probed on resume'))
    resumed = ResearchAgentApplication(root).resume(result.run_id)
    assert resumed == result and Path(resumed.report.path).read_bytes() == before
    assert len(calls) == 1
