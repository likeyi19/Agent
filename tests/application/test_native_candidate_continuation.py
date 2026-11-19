"""Pending native proposals are recompiled after exact choices, with science withheld."""
import json
from types import SimpleNamespace

import pytest

from agent.application import InteractiveAgentApplication
from agent.application.input_sets import ScientificInputSet
from agent.application.raw_handoff import AnnotationDeclaration, BamDeclaration, declaration_dict
from agent.orchestration import PlanningModelProfile, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import StepOutputRef
from test_human_epizoo_mapping import WithholdScience


def _proposal():
    steps = []
    for sid, function, parent, port in [
        ("extract", "pp.make_fragment_file", None, None),
        ("import", "pp.import_fragments", "extract", "fragments"),
        ("matrix", "pp.make_peak_matrix", "import", "dataset"),
        ("features", "pp.select_features", "matrix", "dataset"),
        ("spectral", "tl.spectral", "features", "dataset"),
        ("graph", "pp.knn", "spectral", "dataset"),
        ("cluster", "tl.leiden", "graph", "dataset"),
    ]:
        source = dict(kind="input", input="input_path") if parent is None else dict(
            kind="step_port", step=parent, source_port=port)
        steps.append(dict(step_id=sid, tool="run_snapatac2", control_dependencies=[], sources=[
            dict(target="dataset", source=source),
            dict(target="function_name", source=dict(kind="registered_choice", choice=function))]))
        if parent is None:
            steps[-1]["sources"].append(dict(target="input_sha256",
                source=dict(kind="input", input="input_sha256")))
    steps.append(dict(step_id="labels", tool="annotate_scATAC_cell_types", control_dependencies=[], sources=[
        dict(target=target, source=dict(kind="step_port", step=step, source_port="dataset"))
        for target, step in [("source", "matrix"), ("groups", "cluster"), ("activity_source", "import")]] + [
        dict(target="activity_method", source=dict(kind="registered_choice", choice="snapatac2.pp.make_gene_matrix"))]))
    return dict(schema_version=4, decision=dict(kind="plan", steps=steps))


@pytest.mark.parametrize("failure", [None, "candidate_source", "candidate_step", "preflight", "catalog", "generation"])
def test_human_choice_recompiles_the_captured_graph_without_interpretation_or_planning(
        tmp_path, monkeypatch, failure):
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    monkeypatch.setattr(sequence_prediction, "library", lambda: SimpleNamespace(
        entries=lambda: [dict(kind="cellmarker", id="installed")]))
    monkeypatch.setattr(cellmarker, "CellMarkerEvidence", lambda *a: SimpleNamespace(
        identity={}, metadata={"species": ["Human", "Mouse"]}))
    calls = []
    continued = False

    class Model:
        model_id = "pending-native-candidate"

        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            calls.append(value)
            if "turn_schema_version" in value:
                assert sum("turn_schema_version" in row for row in calls) == 1
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind="request_roles",
                    target="run_snapatac2", required_roles=[], roles=[],
                    bam=declaration_dict(BamDeclaration(source="source.0", assay="SCATAC",
                        species="Homo sapiens", assembly="hg19")),
                    annotation=declaration_dict(AnnotationDeclaration(species="Homo sapiens")))))
            if "semantic_prompt_version" in value:
                # Any model call after the offered Human answer is a regression.
                if continued:
                    assert failure == "catalog"
                    return json.dumps(dict(schema_version=4, decision=dict(kind="unsupported",
                        reason="Changed resource context requires a fresh scientific decision.")))
                proposal = _proposal()
                if "repair" not in value:
                    # Match the observed first attempt: the bounded repair
                    # supplies a port, then reaches the species prerequisite.
                    source = proposal["decision"]["steps"][3]["sources"][0]["source"]
                    source.pop("source_port")
                    source["kind"] = "step"
                return json.dumps(proposal)
            if "output_selection_schema_version" in value:
                return json.dumps(dict(outputs=[dict(name="annotation", step_id="labels",
                                                     output_key="annotated_h5ad_path")]))
            raise AssertionError("Unexpected model phase: " + str(list(value)))

    registry = build_default_tool_registry()
    executor = WithholdScience(registry)
    app = InteractiveAgentApplication(tmp_path / "workspace", registry=registry, executor=executor,
        model_profiles=(PlanningModelProfile("offline", "scripted", Model.model_id),),
        default_profile_id="offline", approved_source_roots=(tmp_path,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({"scripted": lambda _: Model()}))
    app.create_session("session")
    bam = tmp_path / "input.bam"
    bam.write_bytes(b"BAM\x01")
    record = app.resources.register("bam", bam, input_type="attachment", session_id="session",
                                    label="Input BAM", attribution="Offline test")
    binding = app.resources.compose_group([record.resource_id], attachment_candidates=True)
    values = dict(species="human", assembly="hg19")
    for port in ["chrom_sizes", "peaks", "gene_anno"]:
        reference = tmp_path / (port + ".reference")
        reference.write_text("Controlled " + port)
        values[port + "_path"] = str(reference)
        values[port + "_sha256"] = __import__("hashlib").sha256(reference.read_bytes()).hexdigest()
    references = (ScientificInputSet("references", "Reference components", values),)
    first = app.submit_turn("session", "ask", "Annotate this human BAM.", expected_generation=0,
        registered_input=binding, input_sets=references)
    assert first.status == "clarification", first.to_dict()
    state = app._application.sessions.load("session")
    original = state.interactions[-1]
    assert original.preparation["candidate"]["version"] == 1
    assert "native_semantic_candidate" not in original.admitted
    choice = next((c for c in first.clarification_choices if c.get("choice_identity") == "annotation-species:0"), None)
    assert choice is not None,first.to_dict()
    assert not state.turns and not executor.plans
    count = len(calls)
    planning_calls = sum("semantic_prompt_version" in row for row in calls)
    if failure in {"candidate_source", "candidate_step"}:
        # The current candidate is immutable normalized data, never reparsed
        # provider text. Corruption cannot acquire a second graph authority.
        from agent.application.session_state import AnalysisSession, SessionError
        from agent.schemas.orchestration import _serialize
        saved = _serialize(state.to_dict())
        steps = saved['interactions'][-1]['preparation']['candidate']['steps']
        if failure == 'candidate_source':
            steps[3]['sources'][0]['step_id'] = 'absent'
        else:
            steps.append(dict(step_id='invented',tool_name='invented_science',sources=[],control_dependencies=[]))
        with pytest.raises((ValueError, SessionError)):
            AnalysisSession.from_dict(saved)
        assert not executor.plans and len(calls) == count
        assert app._application.sessions.load('session') == state
        return
    if failure == "preflight":
        from agent.schemas import VerificationCheck, VerificationResult
        monkeypatch.setattr(executor, "preflight", lambda plan: VerificationResult(False, "plan", plan.plan_id,
            (VerificationCheck("blocked", False, "Independent preflight witness"),)))
    if failure == "catalog":
        references += (ScientificInputSet("additional", "Newly offered context", {}),)
    continued = True
    if failure == "generation":
        from agent.application import InteractiveBoundaryError
        with pytest.raises(InteractiveBoundaryError) as error:
            app.submit_turn("session", "human", "", expected_generation=1, input_sets=references,
                clarification_choice=dict(question_turn_id="ask", handle=choice["handle"]))
        assert error.value.error.code == "INTERACTIVE_GENERATION_CONFLICT"
        assert not executor.plans and len(calls) == count
        return
    second = app.submit_turn("session", "human", "", expected_generation=0, input_sets=references,
        clarification_choice=dict(question_turn_id="ask", handle=choice["handle"]))
    # The barcode mechanism is an independent native prerequisite. Answer it
    # exactly before this test's original reservation/preflight assertions.
    if second.status == "clarification":
        barcode = next(c for c in second.clarification_choices
                       if c.get("choice_identity") == "native-parameter.barcode_tag")
        mechanism = app.submit_turn("session", "mechanism", "", expected_generation=0, input_sets=references,
            clarification_choice=dict(question_turn_id="human", handle=barcode["handle"]))
        editor = next(c for c in mechanism.clarification_choices if c.get("action") == "parameter" and c.get("member") == "barcode_tag")
        second = app.submit_turn("session", "barcode", "", expected_generation=0, input_sets=references,
            clarification_choice=dict(question_turn_id="mechanism", handle=editor["handle"]),
            parameter_overrides=[dict(tool="run_snapatac2", native_function="pp.make_fragment_file",
                step_id="extract", argument="kwargs", value={"barcode_tag":"RG"})])
    assert sum("turn_schema_version" in row for row in calls) == 1
    assert sum("semantic_prompt_version" in row for row in calls) == planning_calls
    assert second.status == "failed", second.to_dict()  # Science is intentionally withheld.
    assert app._application.sessions.load("session").interactions[0] == original
    if failure == "preflight":
        assert second.error.code == "PLAN_PREFLIGHT_FAILED"
        assert not executor.plans
        assert len(calls) == count
    else:
        assert second.error.code == "TEST_SCIENCE_WITHHELD"
        assert len(executor.plans) == 1
        plan = executor.plans[0]
        assert plan.steps[1].arguments["input_path"] == StepOutputRef("extract", "fragments_path")
        assert plan.steps[3].arguments["input_path"] == StepOutputRef("matrix", "matrix_path")
        assert plan.steps[-1].arguments["species"] == "Human"
        assert plan.steps[-1].arguments["grouping_path"] == StepOutputRef("cluster", "matrix_path")
        assert plan.steps[-1].arguments["activity_source_path"] == StepOutputRef("import", "matrix_path")
