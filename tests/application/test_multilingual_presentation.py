"""Language context goes to the existing response owner; no language classifier.

These scripted providers qualify ownership/transport, not model reliability.
The completed-analysis fixture performs bounded file inspection only.
"""
import json
from types import SimpleNamespace as N

import pytest

from agent.application import InteractiveAgentApplication, TurnOutcome
from agent.application.interactive_schemas import ClientError
from agent.application.responses import summarize, CONVERSATION_LANGUAGE
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
from test_scientific_completion import CompletionModel
from test_service import _tiny_h5ad, _counting_registry
from test_interactive_rich import read_only_boundary
from test_response_contracts import OutcomeModel


REQUESTS=[
    ('请说明检查结果，并给我结果文件。','已完成 inspect_scATAC 检查，结果记录了两个细胞。'),
    ('Explica la inspección y dame el archivo de resultados.','La inspección inspect_scATAC registra dos células.'),
    ('検査結果を説明し、結果ファイルもください。','inspect_scATAC の検査結果には2個の細胞が記録されています。'),
]


class LanguageModel(CompletionModel):
    def __init__(self,profile,utterance,text,files_only):
        super().__init__(profile);self.utterance=utterance;self.text=text;self.files_only=files_only;self.prose=[]
    def complete(self,*,prompt,response_schema):
        payload=json.loads(prompt)
        if 'turn_schema_version' in payload:
            raw=json.loads(super().complete(prompt=prompt,response_schema=response_schema))
            # Empty delivery was the actual genuine Scenario C witness. It now
            # defaults to conversation; only explicit files_only suppresses prose.
            raw['decision']['delivery']=['files_only'] if self.files_only else []
            return json.dumps(raw)
        if 'dialogue_schema_version' in payload:
            self.phases.append('dialogue_schema_version');self.calls.append(payload);self.prose.append(payload)
            assert payload['question']==self.utterance
            assert payload['conversation'][-1]==self.utterance
            assert CONVERSATION_LANGUAGE in payload['instructions']
            assert payload['focus']=='completion'
            claim=next(c for c in payload['evidence']['claims'] if c['field']=='n_cells')
            return json.dumps(dict(support='supported',paragraphs=[dict(text=self.text,kind='finding',citations=[claim['claim_id']],comparisons=[])]))
        return super().complete(prompt=prompt,response_schema=response_schema)


@pytest.mark.parametrize('utterance,text',REQUESTS)
@pytest.mark.parametrize('files_only',[False,True])
def test_completed_request_uses_language_context_and_saved_replay(tmp_path,utterance,text,files_only):
    if files_only:utterance+=' [files only; no explanation]'
    profile=PlanningModelProfile('offline','scripted','multilingual-offline')
    model=LanguageModel(profile,utterance,text,files_only);calls=[]
    app=InteractiveAgentApplication(tmp_path/'workspace',registry=_counting_registry(calls),
        model_profiles=(profile,),default_profile_id=profile.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:model}))
    app.create_session('session');source=_tiny_h5ad(tmp_path/'input.h5ad')
    view=app.submit_turn('session','utterance',utterance,expected_generation=0,execution_inputs={'input_path':str(source)})
    assert view.status=='succeeded',view.to_dict()
    assert calls==['inspect_scATAC']
    if files_only:
        assert view.response.text=='' and view.response.scientific is None and not model.prose
    else:
        assert view.response.text==text and 'inspect_scATAC' in view.response.text
        assert view.response.scientific['availability']=='generated'
        assert len(model.prose)==1
        assert model.phases.count('grounding_review_version')==1
    before=app._application.sessions.load('session');count=len(model.phases)
    reopened=read_only_boundary(app._application)
    assert reopened.turn('session','utterance')==view
    assert reopened.reopen_session('session').turns[-1].response==view.response
    assert reopened.recover_turn('session','utterance')==view
    assert app._application.sessions.load('session')==before and len(model.phases)==count


@pytest.mark.parametrize('utterance,text',REQUESTS)
@pytest.mark.parametrize('state_kind',['question','invalid_source','native_failure','deterministic'])
def test_outcome_language_and_exact_scientific_identities(utterance,text,state_kind):
    from agent.application.session_state import Interaction
    from agent.application.turn_decisions import Clarify
    interaction=Interaction('q',utterance,None,0,{},admitted={'tool':'annotate_scATAC_cell_types'})
    operational={
        REQUESTS[0][0]:{'question':'请补充所需输入。','invalid_source':'源文件已更改，没有接受新结果。','native_failure':'原生函数执行失败，没有接受新结果。','deterministic':''},
        REQUESTS[1][0]:{'question':'Proporciona los datos requeridos.','invalid_source':'El archivo cambió; no se aceptó un resultado nuevo.','native_failure':'La función nativa falló; no se aceptó un resultado nuevo.','deterministic':''},
        REQUESTS[2][0]:{'question':'必要な入力を指定してください。','invalid_source':'入力ファイルが変更されたため、新しい結果は受理されていません。','native_failure':'ネイティブ関数が失敗し、新しい結果は受理されていません。','deterministic':''},
    }
    model=OutcomeModel(operational[utterance][state_kind]+' cellranger_graphclust GRCh38')
    if state_kind=='question':
        outcome=TurnOutcome('clarify','clarification',Clarify('missing_parameter_value',('cellranger_graphclust',),True))
    elif state_kind=='invalid_source':
        outcome=TurnOutcome('execute','failed',error=ClientError('LOCAL_RESOURCE_INTEGRITY_INVALID','Source bytes changed.'))
    else:
        outcome=TurnOutcome('execute','failed',error=ClientError('NATIVE_FUNCTION_FAILED','Recorded native function failed.'))
    if state_kind=='deterministic':
        from dataclasses import replace
        interaction=replace(interaction,submission={'clarification_choice':{'question_turn_id':'previous','handle':'spp.captured'}})
    state=N(interactions=(interaction,),turns=())
    result=summarize(N(load=lambda _:state),'session','q',outcome,model=model)
    assert result.kind==outcome.kind and result.status==outcome.status and result.error==outcome.error
    if state_kind=='deterministic':
        assert result.text=='' and not model.calls
    else:
        assert result.text==model.text
        prompt=model.calls[0]
        assert prompt['question']==utterance and prompt['conversation']==[utterance]
        assert CONVERSATION_LANGUAGE in prompt['instructions']
        assert result.text.endswith('cellranger_graphclust GRCh38')
        assert len(model.calls)==2


def test_files_only_intent_is_explicit_and_cannot_hide_other_delivery():
    from agent.application.turn_decisions import Execute,decision_schema
    assert Execute('run_scanpy',delivery=('files_only',)).delivery==('files_only',)
    assert Execute('run_scanpy',delivery=()).delivery==()
    with pytest.raises(ValueError):Execute('run_scanpy',delivery=('files_only','explanation'))
    schema=json.dumps(decision_schema())
    assert 'files_only' in schema
