"""Actual compiler and scientific owners, bounded before Session reservation."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from agent.application import preparation_state as records
from agent.application.scientific_preparation import scoped_source_admission, _owner_requirements
from agent.application.annotation_preparation import preparer
from agent.application.raw_handoff import AnnotationDeclaration, declaration_dict
from agent.application.input_sets import ScientificInputSet
from agent.application.session_state import Interaction
from agent.schemas import AgentRequest
from agent.orchestration import build_default_tool_registry
from agent.orchestration.semantic_compiler import (SemanticPlanCandidate, SemanticPlanStep,
    SemanticRequestInputSource as Input, SemanticStepOutputSource as Output,
    SemanticRegisteredChoiceSource as Choice, analyze_candidate, ScientificInputRequired, AnalysisDisposition)


def test_two_source_annotation_owner_keeps_declarations_and_resource_answers_scoped(tmp_path, monkeypatch):
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    monkeypatch.setattr(sequence_prediction,'library',lambda:SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')]))
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={},metadata={'species':['Human','Mouse']}))
    import anndata as ad
    import numpy as np
    left_path=str(tmp_path/'left.h5ad');right_path=str(tmp_path/'right.h5ad')
    ad.AnnData(np.eye(3),obs=__import__('pandas').DataFrame({'groups':['a','a','b']})).write_h5ad(left_path)
    ad.AnnData(np.eye(3),obs=__import__('pandas').DataFrame({'groups':['a','a','b']}),uns={'assembly':'mm10'}).write_h5ad(right_path)
    registry=build_default_tool_registry()
    candidate=SemanticPlanCandidate(tuple(step for side,inp in [('left','reference_input_path'),('right','query_input_path')] for step in (
        SemanticPlanStep(side,'inspect_scATAC',(Input('dataset',inp),)),
        SemanticPlanStep(side+'_annotation','annotate_scATAC_cell_types',(
            Output('source',side,'dataset'),Input('group_column','group_column'),
            Choice('activity_method','episcanpy.tl.geneactivity'))))))
    request=AgentRequest('task','Annotate both sources with their declared context',dict(
        reference_input_path=left_path,query_input_path=right_path,group_column='groups',output_dir='/out'))
    resources=[ScientificInputSet(name,name,dict(species=species,assembly=assembly,
        gene_anno_path=str(tmp_path/name),gene_anno_sha256=__import__('hashlib').sha256(name.encode()).hexdigest())).definition()
        for name,species,assembly in [('human1','human','hg38'),('human2','human','hg38'),('oldhuman','human','hg19'),('mouse','mouse','mm10'),('mouse2','mouse','mm10')]]
    from pathlib import Path
    for name in ('human1','human2','oldhuman','mouse','mouse2'): (tmp_path/name).write_text(name)
    admitted=dict(annotation=declaration_dict(AnnotationDeclaration(source='source.0',species='Human',assembly='hg38')),
        annotation_sources=dict(members=[dict(source_path=left_path)]),resource_catalog=resources)
    initial=analyze_candidate(request,candidate,registry)
    assert initial.disposition is not AnalysisDisposition.INVALID,initial.error
    def resolve(admission,answers):
        needs=[];bound={}
        def callback(tool,args,selected,*,step_id):
            scoped,applies=scoped_source_admission(admission,'annotation',{},
                records.operation_sources(step_id,initial.operations,registry,lineage=True))
            try:
                values=preparer(None,None,scoped,requirement_bindings=answers.get(step_id,{}))(tool,args,selected)
                bound[step_id]=values
                return values
            except ScientificInputRequired as exc:
                needs.extend(_owner_requirements(initial.candidate_id,step_id,tool,args,exc,operations=initial.operations,registry=registry))
                return args
        result=analyze_candidate(request,candidate,registry,prepare_inputs=callback)
        assert result.disposition is not AnalysisDisposition.INVALID,result.error
        return needs,bound
    needs,_=resolve(admitted,{})
    assert len(needs)==2, (needs, initial.requirements)
    left,right=(next(r for r in needs if r['step_id']==side+'_annotation') for side in ('left','right'))
    assert {o['handle'] for o in left['offers']}=={'resource.human1','resource.human2'}
    assert 'resource.mouse' in {o['handle'] for o in right['offers']}
    origin=Interaction('origin','Source A is Human hg38',None,0,{})
    record=records.start('session',origin,origin.utterance,candidate=candidate,contract_identity='registry')
    chain=[replace(origin,preparation=record,status='clarification',admitted=dict(preparation_result=records.question(record,origin,needs)))]
    for req,identity in [(left,'resource.human1'),(right,'resource.mouse')]:
        previous=chain[-1]
        choice=next(c for c in records.choices(previous) if c['requirement_id']==req['requirement_id'] and c['choice_identity']==identity)
        turn=Interaction(identity,'',None,0,{},submission=dict(clarification_choice=dict(question_turn_id=previous.turn_id,handle=choice['handle'])))
        value=next(o for o in req['offers'] if o['handle']==identity)
        record=records.answer_question(chain,turn,values={req['requirement_id']:value})
        chain.append(replace(turn,preparation=record,status='clarification',admitted=dict(preparation_result=records.question(record,turn,needs))))
    answers=records.resource_bindings(chain,record)
    needs,bound=resolve(admitted,answers)
    assert bound['left_annotation']['gene_anno_path']==str(tmp_path/'human1')
    assert bound['left_annotation']['species']=='Human'
    assert bound['right_annotation']['gene_anno_path']==str(tmp_path/'mouse')
    assert 'species' not in bound['right_annotation']
    before=repr(chain)
    changed=dict(record,previous=chain[-1].turn_id,answers=[],declarations=[dict(member='assembly',old_value='hg38',value='hg19',
        scope=dict(owner='annotation',sources=[dict(source_path=left_path)]))])
    answers=records.resource_bindings(chain,changed)
    assert 'left_annotation' not in answers
    assert answers['right_annotation']['gene_anno_path']['handle']=='resource.mouse'
    admission=dict(admitted,annotation=dict(admitted['annotation'],assembly='hg19'))
    needs,bound=resolve(admission,answers)
    assert bound['left_annotation']['gene_anno_path']==str(tmp_path/'oldhuman')
    assert bound['right_annotation']['gene_anno_path']==str(tmp_path/'mouse')
    assert 'species' not in bound['right_annotation']
    assert repr(chain)==before


def test_raw_context_is_removed_before_other_source_annotation():
    raw=dict(bam=dict(source='source.0'),bam_sources=dict(members=[dict(source_path='/A')]),
        bam_context=dict(adapter='native-inputs.v1',assembly='hg38',component_selections={'gene_anno':'human'}),
        downstream_annotation=dict(species='Human'),raw_resource_choice='human')
    scoped,applies=scoped_source_admission(raw,'bam',{}, {'/B'})
    assert not applies and scoped=={}
