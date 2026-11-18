"""Pending BAM science keeps exact sources across real typed/textual answers."""
import json
import hashlib
from types import SimpleNamespace

import pytest

from agent.application import InteractiveAgentApplication
from agent.application.input_sets import ScientificInputSet
from agent.application.raw_handoff import AnnotationDeclaration, BamDeclaration, declaration_dict
from agent.orchestration import PlanningModelProfile, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import StepOutputRef
from test_human_epizoo_mapping import WithholdScience
from test_native_candidate_continuation import _proposal


@pytest.mark.parametrize('textual_assembly,declared_assembly', [(False,False),(True,False),(False,True)])
def test_bam_human_gtf_genome_continuation_keeps_sources_and_proposal(tmp_path, monkeypatch, textual_assembly, declared_assembly):
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    monkeypatch.setattr(sequence_prediction, 'library', lambda: SimpleNamespace(
        entries=lambda: [dict(kind='cellmarker', id='installed')]))
    monkeypatch.setattr(cellmarker, 'CellMarkerEvidence', lambda *a: SimpleNamespace(
        identity={}, metadata={'species': ['Human', 'Mouse']}))
    calls=[]
    class Model:
        model_id='reference-lifecycle'
        def complete(self, *, prompt, response_schema):
            value=json.loads(prompt);calls.append(value)
            if 'turn_schema_version' in value:
                if len([v for v in calls if 'turn_schema_version' in v]) > 1:
                    return json.dumps(dict(turn_schema_version=1, decision=dict(kind='answer_roles',
                        pending='@roles', roles=[], bam=declaration_dict(BamDeclaration(assembly='hg19')))))
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind='request_roles',
                    target='run_snapatac2', required_roles=[], roles=[],
                    bam=declaration_dict(BamDeclaration(source='source.0', assay='SCATAC', species='Homo sapiens',assembly='hg19' if declared_assembly else None)),
                    annotation=declaration_dict(AnnotationDeclaration(species='Homo sapiens')))))
            if 'semantic_prompt_version' in value:
                return json.dumps(_proposal())
            if 'output_selection_schema_version' in value:
                return json.dumps(dict(outputs=[dict(name='annotation',step_id='labels',output_key='annotated_h5ad_path')]))
            if 'dialogue_schema_version' in value or 'grounding_review_version' in value:
                from test_response_contracts import OutcomeModel
                return OutcomeModel('Scientific execution has not started; supply the required source inputs.').complete(prompt=prompt,response_schema=response_schema)
            raise AssertionError('Unexpected model phase: '+str(list(value)))
    registry=build_default_tool_registry();executor=WithholdScience(registry)
    app=InteractiveAgentApplication(tmp_path/'workspace',registry=registry,executor=executor,
        model_profiles=(PlanningModelProfile('offline','scripted',Model.model_id),),default_profile_id='offline',
        approved_source_roots=(tmp_path,),planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:Model()}))
    app.create_session('session');bam=tmp_path/'input.bam';bam.write_bytes(b'Opaque BAM registration; native science withheld')
    record=app.resources.register('bam',bam,input_type='attachment',session_id='session',label='Input BAM',attribution='Fixture')
    binding=app.resources.compose_group([record.resource_id],attachment_candidates=True)
    def reference(name,assembly,**ports):
        values=dict(species='human',assembly=assembly)
        for port in ports:
            path=tmp_path/(name+'-'+port);path.write_text(name+' '+port)
            values.update({port+'_path':str(path),port+'_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        return ScientificInputSet(name,name,values)
    references=(reference('genome','hg19',chrom_sizes=True,reference=True),reference('other-genome','hg19',chrom_sizes=True,reference=True),
        reference('genes','hg19',gene_anno=True),reference('other-genes','hg19',gene_anno=True),reference('peaks','hg19',peaks=True))
    def state():return app._application.sessions.load('session')
    def phase_count(key):return sum(key in value for value in calls)
    first=app.submit_turn('session','initial','Analyze and annotate this human BAM.',expected_generation=0,registered_input=binding,input_sets=references)
    assert first.status=='clarification',first.to_dict()
    assert state().interactions[-1].preparation.get('candidate')
    planning_count=phase_count('semantic_prompt_version')
    def choose(turn, identity):
        return next(c['handle'] for c in turn.clarification_choices if c.get('choice_identity') == identity)
    human=app.submit_turn('session','human','',expected_generation=0,input_sets=references,
        clarification_choice=dict(question_turn_id='initial',handle=choose(first,'annotation-species:0')))
    assert human.status=='clarification',human.to_dict()
    assert phase_count('turn_schema_version')==1 and phase_count('semantic_prompt_version')==planning_count
    assert state().interactions[-1].admitted['bam']['assembly']==('hg19' if declared_assembly else None)
    before=state()
    question_view=human
    question='human'
    if not declared_assembly:
        # Unknown source assembly grants no GTF compatibility, even though a
        # human reference is installed. A reference click cannot declare it.
        assert not any(c.get('choice_identity')=='resource.genes' for c in human.clarification_choices)
        text='The source assembly is hg19.' if textual_assembly else 'hg19'
        answer=app.submit_turn('session','assembly',text,expected_generation=0,input_sets=references)
        assert answer.status=='clarification',answer.to_dict()
        assert phase_count('turn_schema_version')==2 and phase_count('semantic_prompt_version')==planning_count
        current=state().interactions[-1]
        assert current.admitted['bam']['assembly']=='hg19'
        assert current.admitted['bam']['source']=='source.0'
        assert current.preparation['task_origin']=='initial'
        changes=current.preparation['declarations']
        assert len(changes)==1 and changes[0]['member']=='assembly'
        assert changes[0]['scope']['owner']=='bam'
        assert changes[0]['scope']['sources'][0]['resource_id']==record.resource_id
        from agent.application.session_state import AnalysisSession,SessionError
        from agent.schemas.orchestration import _serialize
        for key,value in [('old_value','invented'),('value','hg38'),('scope',{'owner':'annotation','sources':[]}),('evidence',{'interaction_id':'another'})]:
            forged=_serialize(state().to_dict());forged['interactions'][-1]['preparation']['declarations'][0][key]=value
            with pytest.raises(SessionError):AnalysisSession.from_dict(forged)
        question_view=answer;question='assembly'
    genes=app.submit_turn('session','genes','',expected_generation=0,input_sets=references,
        clarification_choice=dict(question_turn_id=question,handle=choose(question_view,'resource.genes')))
    assert genes.status=='clarification',genes.to_dict()
    assert phase_count('semantic_prompt_version')==planning_count
    question='genes';question_view=genes
    planning_before=phase_count('semantic_prompt_version');interpretation_before=phase_count('turn_schema_version')
    final=app.submit_turn('session','genome','',expected_generation=0,input_sets=references,
        clarification_choice=dict(question_turn_id=question,handle=choose(question_view,'resource.genome')))
    assert phase_count('turn_schema_version')==interpretation_before
    assert phase_count('semantic_prompt_version')==planning_before
    current=state().interactions[-1]
    assert current.admitted['bam']['assembly']=='hg19'
    if final.status!='failed':
        from agent.schemas.orchestration import _serialize
        pytest.fail(json.dumps(_serialize(state().interactions[-1].admitted['preparation_result']),indent=2))
    assert final.error.code=='TEST_SCIENCE_WITHHELD',final.to_dict()
    assert len({i.preparation['candidate_id'] for i in state().interactions})==1
    assert sum('candidate' in i.preparation for i in state().interactions)==1
    assert len(executor.plans)==1
    plan=executor.plans[0]
    assert plan.steps[1].arguments['chrom_sizes_path']==str(tmp_path/'genome-chrom_sizes')
    assert plan.steps[-1].arguments['gene_anno_path']==str(tmp_path/'genes-gene_anno')
    assert plan.steps[-1].arguments['grouping_path']==StepOutputRef('cluster','matrix_path')
    assert current.admitted['bam_sources']['members'][0]['resource_id']==record.resource_id
    assert current.admitted['bam_sources']['members'][0]['source_sha256']==record.source_sha256
    assert current.preparation['task_origin']=='initial'
    assert state().interactions[:len(before.interactions)]==before.interactions
    # Store replay validates the entire persisted chain independently of providers/science.
    from agent.application import ResearchAgentApplication
    assert ResearchAgentApplication(app._application.workspace_root).sessions.load('session')==state()
