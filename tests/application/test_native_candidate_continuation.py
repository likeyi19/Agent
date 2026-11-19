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


@pytest.mark.parametrize('reference_answer', ['exact', 'prose'])
def test_bam_reference_and_regions_continue_with_one_population_answer(tmp_path, monkeypatch, reference_answer):
    from agent.application import preparation_state as records
    from agent.schemas import VerificationResult, VerificationCheck
    from test_preparation_owner_contracts import Model, setup
    import hashlib
    import pysam

    clarification = '确认：这份 BAM 比对到人类 UCSC hg19（GRCh37）。请使用资源库中与 hg19 匹配的人类 GENCODE 19 基因注释。'

    class Handoff(Model):
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            if 'turn_schema_version' in value:
                self.calls.append(value)
                if value['utterance'] == clarification:
                    return json.dumps(dict(turn_schema_version=1, decision=dict(kind='answer_roles',
                        pending='@roles', roles=[], arguments=[], unknown_fields=[],
                        bam=dict(declaration_dict(BamDeclaration(species='human', assembly='hg19')),
                                 resource_preference=dict(literal='人类 GENCODE 19 基因注释',
                                     evidence='请使用资源库中与 hg19 匹配的人类 GENCODE 19 基因注释。')))))
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind='request_roles',
                    target='run_snapatac2', required_roles=[], roles=[], bam=declaration_dict(
                        BamDeclaration(source='source.0', species='human', feature_source='source.1')))))
            if 'semantic_prompt_version' in value:
                self.calls.append(value)
                proposal = _proposal()
                extract, imported, matrix, *_ = proposal['decision']['steps']
                def qc(step, function, parent):
                    return dict(step_id=step, tool='run_snapatac2', control_dependencies=[], sources=[
                        dict(target='dataset', source=dict(kind='step_port', step=parent, source_port='dataset')),
                        dict(target='function_name', source=dict(kind='registered_choice', choice=function))])
                matrix['sources'][0]['source']['step'] = 'filter'
                proposal['decision']['steps'] = [extract, imported, qc('tsse', 'metrics.tsse', 'import'),
                    qc('filter', 'pp.filter_cells', 'tsse'), matrix, qc('sibling', 'pp.filter_cells', 'tsse')]
                return json.dumps(proposal)
            return super().complete(prompt=prompt, response_schema=response_schema)

    model = Handoff()
    app, executor = setup(tmp_path, model)
    bam = tmp_path / 'sample.bam'
    with pysam.AlignmentFile(bam, 'wb', header=dict(SQ=[dict(SN='chr1', LN=100)], RG=[dict(ID='cell')])) as output:
        read = pysam.AlignedSegment()
        read.query_name = 'test'
        read.query_sequence = 'A'
        read.flag = 4
        read.set_tag('RG', 'cell')
        output.write(read)
    bed = tmp_path / 'researcher.bed'
    bed.write_text('chr1\t1\t5\n')
    files = [app.resources.register(path.name, path, input_type='attachment', session_id='session',
             label=path.name, attribution='Regression fixture').resource_id for path in (bam, bed)]
    supplied = app.resources.compose_group(files, attachment_candidates=True)
    def resource(assembly, port, contents):
        path = tmp_path / (assembly + '.' + port)
        path.write_text(contents)
        label = 'Human hg19 GENCODE 19 annotation' if (assembly, port) == ('hg19', 'gene_anno') else assembly + ' ' + port
        return ScientificInputSet(assembly + '-' + port, label,
            {'species': 'human', 'assembly': assembly, port + '_path': str(path),
             port + '_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    resources = (resource('hg19', 'chrom_sizes', 'chr1\t100\n'),
                 resource('hg38', 'chrom_sizes', 'chr1\t200\n'),
                 resource('hg19', 'gene_anno', 'hg19 technical fixture'),
                 resource('hg38', 'gene_anno', 'hg38 technical fixture'))
    intercepted = []
    def preflight(plan):
        intercepted.append(plan)
        return VerificationResult(False, 'plan', plan.plan_id,
            (VerificationCheck('withheld', False, 'No native work in this test'),))
    monkeypatch.setattr(executor, 'preflight', preflight)
    first = app.submit_turn('session', 'initial', 'Compute TSS enrichment, filter and count over my BED.',
        expected_generation=0, registered_input=supplied, input_sets=resources)
    original = app._application.sessions.load('session').interactions[0]
    assert first.status == 'clarification', first.to_dict()
    calls = len(model.calls)
    def choose(turn, outcome, identity, next_turn):
        offer = next(o for o in outcome.clarification_choices if o['choice_identity'] == identity)
        return app.submit_turn('session', next_turn, '', expected_generation=0, input_sets=resources,
            clarification_choice=dict(question_turn_id=turn, handle=offer['handle']))
    tagged = choose('initial', first, 'native-parameter.barcode_tag.RG', 'barcode')
    if reference_answer == 'exact':
        selected = choose('barcode', tagged, 'resource.hg19-chrom_sizes', 'reference')
        assert len(model.calls) == calls
    else:
        selected = app.submit_turn('session', 'reference', clarification, expected_generation=0, input_sets=resources)
    state = app._application.sessions.load('session')
    current = state.interactions[-1]
    if reference_answer == 'exact':
        assert current.preparation['answers'][0]['value']['coordinates'] == {'species': 'human', 'assembly': 'hg19'}
    else:
        assert current.admitted['bam']['assembly'] == 'hg19'
        assert current.admitted['bam']['resource_preference']['literal'] == 'GENCODE 19'
    assert {r['argument'] for r in records.observation(current)['requirements']} == {'qc_population'}
    reasons = ('explicit', 'automatic') if reference_answer == 'exact' else ('automatic', 'explicit')
    assert [(r['port'], r['assembly'], r['reason']) for r in current.admitted['native_resource_bindings']
            if r['port'] in ('chrom_sizes', 'gene_anno')] == [
                ('chrom_sizes', 'hg19', reasons[0]), ('gene_anno', 'hg19', reasons[1])]
    calls = len(model.calls)
    final = choose('reference', selected, 'study_selected', 'population')
    assert final.error.code == 'PLAN_PREFLIGHT_FAILED', final.to_dict()
    steps = {step.step_id: step for step in intercepted[-1].steps}
    assert steps['extract'].arguments['kwargs']['barcode_tag'] == 'RG'
    assert steps['matrix'].arguments['peaks_path'] == str(bed)
    assert steps['tsse'].arguments['gene_anno_path'] == str(tmp_path / 'hg19.gene_anno')
    assert all(steps[step].arguments['qc_selection']['population'] == 'study_selected'
               for step in ('filter', 'sibling'))
    state = app._application.sessions.load('session')
    assert len(model.calls) == calls
    assert {i.preparation['candidate_id'] for i in state.interactions} == {original.preparation['candidate_id']}
    assert {i.preparation['task_id'] for i in state.interactions} == {original.preparation['task_id']}


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
