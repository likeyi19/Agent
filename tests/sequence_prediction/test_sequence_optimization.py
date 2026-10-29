"""Simulated trajectories test contracts; native execution is separate evidence."""
import copy
import json
from pathlib import Path

import pytest

from agent.tools.analysis import sequence_prediction as shared, sequence_optimization as owner
from agent.orchestration import AgentPlan, AgentRequest, PlanStep
from agent.application import ResearchAgentApplication, OutputSelection

LABELS=['B_cell','CD14_monocyte','CD16_monocyte','CD4_T_cell','Cytotoxic_T_cell','Dendritic_cell','Natural_killer_cell']


@pytest.fixture
def native(monkeypatch):
    calls=[]
    resource=dict(family='DeepPBMC',release='simulated',conditions=LABELS,measurements=dict(accessibility=owner.MEASUREMENT),
        context_bases=2114,padding='none',training_assembly='hg38',training_domain='simulated',license='simulated',
        source='simulated',checkpoints=[],software={})
    monkeypatch.setattr(owner,'resolve_model',lambda:(resource,dict(files=[],resource=resource)))
    def run(resource,rows,effective,stage):
        calls.append(effective)
        candidates=[];trajectories=[]
        for row in rows:
            sequence=row['sequence']; current=list(sequence);changes=[[-1,'N']];predictions=[[1.]*7]
            allowed=set(range(effective['flanks'][0],2114-effective['flanks'][1]))-{i for a,b in effective['protected_ranges'] for i in range(a,b)}
            position=min(allowed)
            for i in range(effective['iterations']):
                current[position]='C' if current[position]!='C' else 'A'
                changes.append([position,current[position]])
                p=[1.]*7;p[effective['target_index']]+=float(current[position]!=sequence[position]);predictions.append(p)
            candidate=''.join(current);candidates.append(candidate)
            trajectories.append(dict(initial_sequence=sequence,designed_sequence=candidate,changes=changes,predictions=predictions))
        return dict(candidates=candidates,trajectories=trajectories,receipt=dict(completion='succeeded',optimization=effective))
    monkeypatch.setattr(owner,'native_optimize',run)
    return calls


def args(tmp_path,**kwargs):
    return dict(sequence='A'*2114,condition='B_cell',max_changes=2,edit_start=100,edit_end=110,
        protected_regions='100:105',output_dir=str(tmp_path/'output'),**kwargs)


def test_actual_distance_reversion_and_protected_bases(tmp_path,native):
    result=owner.execute(args(tmp_path))
    candidate=result['candidate_examples'][0]
    assert result['iterations']==2 and candidate['changed_bases']==0
    assert candidate['constraints_satisfied'] and result['improved_candidates']==0
    assert Path(result['sequences_path']).read_text().splitlines()[1]=='A'*2114
    assert len(native)==1


@pytest.mark.parametrize('change',[
    dict(condition='K562'),dict(comparison_conditions='B_cell'),dict(comparison_conditions='unknown'),
    dict(max_changes=0),dict(iterations=3),dict(edit_start=-1),dict(edit_end=2115),
    dict(protected_regions='0:2114'),dict(protected_regions='3'),dict(comparison_weight=-1),
])
def test_invalid_objective_and_constraints_precede_native(tmp_path,native,change):
    values=args(tmp_path);values.update(change)
    with pytest.raises(ValueError):owner.execute(values)
    assert native==[]


@pytest.mark.parametrize('sequence',['A'*500,'A'*2113,'A'*2115,'A'*2113+'N'])
def test_model_context_has_no_padding_or_truncation(tmp_path,native,sequence):
    values=args(tmp_path);values['sequence']=sequence
    with pytest.raises(ValueError,match='2114'):owner.execute(values)
    assert native==[]
    assert shared.records(dict(sequence='A'*500))[0][0]['native_zero_padding']==0


def test_native_constraint_violation_and_nonimprovement(tmp_path,native,monkeypatch):
    original=owner.native_optimize
    def corrupt(*a):
        r=original(*a);r['candidates'][0]='G'+r['candidates'][0][1:];return r
    monkeypatch.setattr(owner,'native_optimize',corrupt)
    with pytest.raises(ValueError,match='violates'):owner.execute(args(tmp_path))
    assert not (tmp_path/'output/manifest.json').exists()
    def worse(*a):
        r=original(*a)
        for p in r['trajectories'][0]['predictions'][1:]:p[0]=0
        return r
    monkeypatch.setattr(owner,'native_optimize',worse)
    values=args(tmp_path);values['max_changes']=1
    result=owner.execute(values)
    assert result['improved_candidates']==0
    assert result['candidate_examples'][0]['objective_change']<0


def test_application_publication_exact_reuse_and_restart(tmp_path,native,monkeypatch):
    from agent.orchestration.prior_outputs import bind_output
    from agent.schemas.prior_output import PriorOutputRef
    values=args(tmp_path)
    class Planner:
        def plan(self,request,registry):
            return AgentPlan('test',request.request_id,'Explicit simulated native contract',(PlanStep('design',owner.TOOL,values),))
    app=ResearchAgentApplication(tmp_path/'workspace',planner=Planner())
    app.sessions.create('design')
    selections=(OutputSelection('candidates','design','sequences_path'),OutputSelection('comparison','design','manifest_path'))
    first=app.sessions.run('design','first',AgentRequest('first','Optimize',{}),selections,expected_generation=0)
    assert first.turn('first').status=='activated', app.run_store.load('first:run').errors
    binding=bind_output(app.run_store,app.registry,first.revisions[-1].run_id,'design','sequences')
    assert binding.semantic_type=='dna_sequences.v1'
    previous=app.run_store.load('first:run').steps[0].result
    frozen=Path(previous['manifest_path']).read_bytes()
    values=dict(sequence_path=PriorOutputRef(binding,'sequences_path'),sequence_sha256=PriorOutputRef(binding,'sequences_sha256'),
        condition='CD4_T_cell',max_changes=1,output_dir=str(tmp_path/'second'))
    second=app.sessions.run('design','second',AgentRequest('second','Change target',{}),selections,
        expected_generation=1,use_active_context=True)
    assert second.turn('second').status=='activated', app.run_store.load('second:run').errors
    assert len(native)==2 and Path(previous['manifest_path']).read_bytes()==frozen
    monkeypatch.setattr(owner,'native_optimize',lambda *a:pytest.fail('native replay'))
    fresh=ResearchAgentApplication(tmp_path/'workspace',planner=Planner())
    assert fresh.resume('second:run').status.value=='SUCCEEDED'
    Path(previous['mutations_path']).write_text('corrupt')
    with pytest.raises(ValueError):shared.verify_result(args(tmp_path),previous)


def test_linear_native_comparison_objective(tmp_path,native):
    values=args(tmp_path,comparison_conditions='CD4_T_cell',comparison_weight=7)
    result=owner.execute(values)
    assert native[0]['penalty_weights']==[0.,0.,0.,7.,0.,0.,0.]
    assert result['candidate_examples'][0]['original_objective']==0


def test_direct_2114_base_typed_declaration():
    from agent.orchestration import build_default_tool_registry
    from agent.application.turn_decisions import ScopedArgument,admit_scientific_argument
    sequence='ACGT'*528+'AC';utterance='Optimize '+sequence
    spec=build_default_tool_registry().get(owner.TOOL).optional_arguments['sequence']
    declaration=ScopedArgument(owner.TOOL,'sequence',sequence,9,len(utterance))
    assert admit_scientific_argument(declaration,utterance,spec)==sequence


def test_all_design_downloads_use_shared_snapshot_privacy(tmp_path,native):
    from types import SimpleNamespace
    from agent.application import sequence_delivery
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    values=args(tmp_path);source=tmp_path/'source.fasta';source.write_text('>/private/research/source\n'+'A'*2114+'\n')
    values.pop('sequence');values.update(sequence_path=str(source),sequence_sha256=shared.file_closure([source])[0]['sha256'])
    result=owner.execute(values);manifest,_=shared.load(result['manifest_path'],result['manifest_sha256'])
    manager=MatrixDownloadPreparer()
    for pin in manifest['files']:
        name=Path(pin['path']).name
        if name not in sequence_delivery.FILES:continue
        item=SimpleNamespace(path=Path(pin['path']),managed_root=tmp_path,sha256=pin['sha256'],size_bytes=pin['size_bytes'],
            contract=owner.CONTRACT,role=name,manifest=manifest,manifest_sha256=result['manifest_sha256'])
        prepared=manager.prepare(item)
        try:
            raw=b''.join(prepared.iter_chunks());assert raw and b'/private/research/source' not in raw
            if name=='design.png':assert raw.startswith(b'\x89PNG')
        finally:prepared.close()
    assert manager.active_count==0


def test_native_figure_inventory_and_completion_relationship(tmp_path,native):
    from agent.application.dialogue_evidence import figure_inventory
    from agent.application.scientific_dialogue import _presentation_context,_figure_relationships
    from agent.application.workspace import ManagedWorkspace
    from types import SimpleNamespace
    root=tmp_path/'workspace'
    output=root/'runs'/ManagedWorkspace.run_digest('first:run')/'scientific'/'design'
    values=args(tmp_path);values['output_dir']=str(output)
    class Planner:
        def plan(self,request,registry):
            return AgentPlan('test',request.request_id,'Simulated figure contract',(PlanStep('design',owner.TOOL,values),))
    app=ResearchAgentApplication(root,planner=Planner());app.sessions.create('design')
    state=app.sessions.run('design','first',AgentRequest('first','Optimize',{}),
        (OutputSelection('candidate','design','sequences_path'),),expected_generation=0)
    assert state.turn('first').status=='activated'
    revision=state.revisions[-1];turn=state.turn('first')
    figures=figure_inventory(app.sessions,turn,revision)
    assert len(figures)==1 and figures[0]['path']==str(output/'design.png')
    locator=revision.outputs[0]
    views=[SimpleNamespace(source=SimpleNamespace(output_locator=dict(run_id=locator.run_id,
        step_id=locator.step_id,accepted_step_sha256=locator.accepted_step_sha256)))]
    figures,matching=_presentation_context(app.sessions,'design',dict(completion_run_id=turn.run_id,
        completion_delivery=dict(visualization='delivered',requested=['visualization'])),views,[])
    assert len(matching)==1
    relationship=_figure_relationships(state,figures,matching)[0]
    assert relationship['satisfies_request'] and relationship['download']['status']=='available_with_response'
    assert len(native)==1
