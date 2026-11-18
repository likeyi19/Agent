"""Executable ownership regressions for scientific preparation consolidation."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from agent.application import preparation_state as records
from agent.application.session_state import Interaction
from agent.schemas import AgentRequest
from agent.orchestration import build_default_tool_registry
from agent.orchestration.semantic_compiler import (
    SemanticPlanCandidate as Candidate, SemanticPlanStep as Step,
    SemanticRequestInputSource as Input, SemanticStepOutputSource as Output,
    SemanticRegisteredChoiceSource as Choice, analyze_candidate, AnalysisDisposition,
)


def test_compiler_answer_invalidation_is_scoped_to_consumed_source():
    registry = build_default_tool_registry()
    candidate = Candidate(tuple(step for side, name in [('left', 'reference_input_path'), ('right', 'query_input_path')] for step in (
        Step(side, 'inspect_scATAC', (Input('dataset', name),)),
        Step(side+'_cluster', 'run_episcanpy', (Output('dataset', side, 'dataset'),
            Choice('function_name', 'tl.kmeans'), Input('parameter_selection', 'parameter_selection'))))))
    analysis = analyze_candidate(AgentRequest('task', 'Cluster independently', dict(
        reference_input_path='/left.h5ad', query_input_path='/right.h5ad',
        parameter_selection='native_defaults', output_dir='/out')), candidate, registry)
    assert analysis.disposition is AnalysisDisposition.NEEDS_INPUT
    needs = [r.to_dict() for r in analysis.requirements if r.member == 'num_clusters']
    assert {r['step_id'] for r in needs} == {'left_cluster', 'right_cluster'}
    origin = Interaction('origin', 'Cluster independently', None, 0, {})
    start = records.start('s', origin, origin.utterance, candidate=candidate, contract_identity='registry')
    origin = replace(origin, preparation=start, status='clarification', admitted=dict(
        preparation_result=records.question(start, origin, needs)))
    answer = Interaction('answer', 'Use 7', None, 0, {})
    saved = records.record_answers(origin, answer, {r['requirement_id']:7 for r in needs}, provenance='interpreted_parameter')
    answer = replace(answer, preparation=saved, status='clarification', admitted=dict(
        preparation_result=records.question(saved, answer, [])))
    chain = [origin, answer]
    correction = records.record_answers(answer, Interaction('change', 'Change left source', None, 0, {}), {}, provenance='interpreted_source')
    correction['declarations'] = [dict(scope=dict(owner='annotation', sources=[dict(source_path='/left.h5ad')]),
        member='resource', old_value='old', value='new')]
    active = records.scalar_bindings(chain, correction)
    assert set(active) == {'right_cluster'}
    assert len(records.scalar_bindings(chain, saved)) == 2  # history unchanged


def test_current_hierarchy_manifest_connects_to_regions():
    registry = build_default_tool_registry()
    candidate = Candidate((Step('annotation', 'annotate_scATAC_cell_types', (Input('source','input_path'),)),
        Step('regions', 'prepare_scatac_regulatory_sequences', (Input('dataset','input_path'),
            Output('annotation','annotation','annotation')))))
    result = analyze_candidate(AgentRequest('task','Use annotation groups',dict(input_path='/source.h5ad', output_dir='/out')),candidate,registry)
    assert result.error is None
    region = next(op for op in result.operations if op.step_id == 'regions')
    assert region.arguments['annotation_manifest_path'].step_id == 'annotation'
    port = next(p for p in registry.get('prepare_scatac_regulatory_sequences').semantic_planning.consumer_ports if p.name == 'annotation')
    assert port.accepted_upstream_types == ('hierarchical_annotation.v1',)


@pytest.mark.parametrize('origin', ['var', 'nested'])
def test_positive_assembly_conflicts_are_diagnosed_without_matrix_read(tmp_path, monkeypatch, origin):
    import anndata as ad
    import numpy as np
    import pandas as pd
    import h5py
    from agent.tools.analysis import sequence_reference as owner
    from agent.orchestration.scientific_inputs import ScientificDiagnostic
    path = tmp_path/'source.h5ad'
    data = ad.AnnData(np.ones((2,3)),var=pd.DataFrame(index=['a','b','c']),uns=dict(assembly='hg38'))
    if origin == 'var': data.var['genome'] = ['hg19']*3
    else: data.uns['source'] = dict(reference='hg19')
    data.write_h5ad(path)
    monkeypatch.setattr(owner,'library',lambda:SimpleNamespace(entries=lambda:[]))
    read = h5py.Dataset.__getitem__
    def checked(item, key):
        assert not item.name.startswith('/X')
        return read(item,key)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',checked)
    with pytest.raises(ScientificDiagnostic, match='assembly'):
        owner.check_declarations(path, dict(assembly='hg38'))
    with h5py.File(path,'r') as f:
        with pytest.raises(ScientificDiagnostic): owner.metadata_declarations(f,'X',dict(assembly='hg38'),data.var)


def test_assembly_aliases_require_exact_qualified_reference_identity():
    from agent.tools.analysis.sequence_reference import reconcile_assembly
    from agent.orchestration.scientific_inputs import ScientificDiagnostic
    observed = dict(observations=[dict(values=['hg38'],complete=True),dict(values=['GRCh38'],complete=True)])
    def entry(label, sha):
        return dict(kind='genome',resource=dict(assembly=label,genome=dict(fasta=dict(sha256=sha),fai=dict(sha256='index'))))
    with pytest.raises(ScientificDiagnostic): reconcile_assembly(observed,'hg38',entries=[entry('hg38','one'),entry('GRCh38','two')])
    assert reconcile_assembly(observed,'hg38',entries=[entry('hg38','same'),entry('GRCh38','same')]) == 'hg38'
    assert reconcile_assembly(dict(observations=[dict(values=['hg38'],complete=False)])) is None


def test_annotation_conflict_has_explicit_diagnostic_and_programming_errors_escape(monkeypatch):
    from agent.application.annotation_preparation import preparer
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    from agent.orchestration.scientific_inputs import ScientificDiagnostic
    monkeypatch.setattr(sequence_prediction,'library',lambda:SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')]))
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={},metadata=dict(species=['Human'])))
    tool = build_default_tool_registry().get('annotate_scATAC_cell_types')
    with pytest.raises(ScientificDiagnostic) as error:
        preparer(None,None,{})(tool,dict(input_path='/one.h5ad',group_column='groups',grouping_path='/other.h5ad'),{})
    assert error.value.code == 'ANNOTATION_GROUP_SOURCE_CONFLICT'
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:(_ for _ in ()).throw(TypeError('programming defect')))
    with pytest.raises(TypeError,match='programming defect'):
        preparer(None,None,{})(tool,dict(input_path='/one.h5ad',group_column='groups'),{})


def test_selected_motif_assembly_refusal_becomes_invalid_application_result(tmp_path):
    import json
    import anndata as ad
    import numpy as np
    from agent.application import InteractiveAgentApplication
    from agent.orchestration import PlanningModelProfile
    from agent.providers import PlanningModelFactoryRegistry
    from test_human_epizoo_mapping import WithholdScience
    from test_response_contracts import OutcomeModel
    path = tmp_path/'conflict.h5ad'
    data = ad.AnnData(np.eye(3), uns=dict(assembly='hg38',source=dict(assembly='hg19')))
    data.write_h5ad(path)
    class Model:
        model_id='conflict'
        def complete(self, *, prompt, response_schema):
            value=json.loads(prompt)
            if 'turn_schema_version' in value:
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='execute_plan',target='run_motifmatchr')))
            if 'semantic_prompt_version' in value:
                return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='motifs',tool='run_motifmatchr',sources=[dict(target='dataset',source=dict(kind='input',input='input_path')),dict(target='function_name',source=dict(kind='registered_choice',choice='matchMotifs'))],control_dependencies=[])])))
            return OutcomeModel('Source assembly claims conflict. Scientific execution has not started.').complete(prompt=prompt,response_schema=response_schema)
        def close(self): pass
    registry=build_default_tool_registry(); executor=WithholdScience(registry)
    app=InteractiveAgentApplication(tmp_path/'workspace',registry=registry,executor=executor,
        model_profiles=(PlanningModelProfile('offline','scripted',Model.model_id),),default_profile_id='offline',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:Model()}))
    app.create_session('session')
    result=app.submit_turn('session','task','Find motif matches.',expected_generation=0,execution_inputs=dict(input_path=str(path)))
    state=app._application.sessions.load('session')
    assert result.error.code == 'SOURCE_ASSEMBLY_CONFLICT', result.to_dict()
    assert records.observation(state.interactions[-1])['status']=='INVALID'
    assert not executor.plans and not state.turns and not state.revisions


@pytest.mark.parametrize('mismatch', ['source','cells'])
def test_hierarchy_group_consumer_retains_exact_source_and_order_checks(tmp_path, monkeypatch, mismatch):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from agent.tools.analysis import scatac_regulatory_sequences as owner
    from agent.tools.analysis.hierarchical_annotation import publication
    from agent.tools.analysis import sequence_prediction as shared
    path=tmp_path/'annotated.h5ad'
    ad.AnnData(np.eye(2),obs=pd.DataFrame(dict(cell_type_final=['A','B']),index=['a','b'])).write_h5ad(path)
    monkeypatch.setattr(publication,'verify_publication',lambda *a,**kw:dict(hierarchy_path='hierarchy',hierarchy_sha256='h',annotated_h5ad_path=str(path)))
    monkeypatch.setattr(shared,'read',lambda *a:dict(source=dict(sha256='original')))
    with pytest.raises(ValueError,match='exact original|ordered cell'):
        owner._annotation_groups(dict(annotation_manifest_path='publication',annotation_manifest_sha256='p'),
            ['b','a'] if mismatch=='cells' else ['a','b'],dict(sha256='wrong' if mismatch=='source' else 'original'))


def test_gene_annotation_and_motif_owners_agree_on_same_row_qualified_aliases(tmp_path):
    from test_preparation_source_contracts import source_file, receipt
    from agent.application.native_raw_handoff import gene_annotation_component
    from agent.tools.analysis.sequence_reference import assembly_labels_match
    path = source_file(tmp_path, genomes=('hg38','GRCh38','hg38'))
    rec = receipt(tmp_path)
    identity=dict(reference=dict(path='/genome',sha256='a'*64),chrom_sizes=dict(path='/index',sha256='b'*64,ordered_sizes='c'*64))
    rec['catalog']['0']['assembly']='hg38'
    rec['catalog']['0']['components'].update(identity)
    rec['catalog']['alias']=dict(rec['catalog']['0'],assembly='GRCh38',components=identity)
    assert assembly_labels_match({'hg38','GRCh38'},rec['catalog'].values())
    assert gene_annotation_component(rec,path)[1]=='0'
    rec['catalog']['alias']['components']=dict(identity,chrom_sizes=dict(identity['chrom_sizes'],ordered_sizes='d'*64))
    assert not assembly_labels_match({'hg38','GRCh38'},rec['catalog'].values())
    from agent.orchestration.semantic_compiler import ScientificInputRequired
    with pytest.raises(ScientificInputRequired): gene_annotation_component(rec,path)
