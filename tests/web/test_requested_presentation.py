"""Execute the existing browser owners with bounded DOM/transport substitutes.

No scientific/source qualification is delegated to these display tests. Owner API
and export tests independently retain those contracts.
"""
import json

import pytest
from test_static import STATIC, _javascript_value


def functions(start, end):
    source = (STATIC / 'app.js').read_text()
    return source[source.index(start):source.index(end)]


def fixture_script():
    return '''
    const state = {session:{session_id:'session', turns:[
      {turn_id:'older',revision_id:null,utterance:'First labels',response:{text:'First supplied labels'}},
      {turn_id:'latest',revision_id:null,utterance:'Other labels and data',response:{text:'Latest supplied labels'}},
      {turn_id:'unrelated',revision_id:null,utterance:'Other source',response:{text:'Unrelated'}}
    ]}, revision:{revision_id:'r',artifacts:[],scientific_artifacts:[],result_groups:[]},
      scientificFiles:new Map(), artifactPreview:false};
    const figure = (turn, revision='r', handle=turn) => ({handle,turn_id:turn,revision_id:null,
      artifact_type:'analysis_figure',figure:{manifest_sha256:handle,figure_id:handle,
        sources:[{role:'coordinates',output:{revision_id:revision}}]}});
    const cache = (turn,presentations,artifacts=[]) => state.scientificFiles.set(
      JSON.stringify(['session',turn,null]),{presentations,artifacts});
    const composition = {handle:'export',turn_id:'latest',revision_id:null,
      artifact_type:'scientific_composition', contents:{representations:[{accepted_output:{revision_id:'r'}}]}};
    const dom = {};
    const element = id => dom[id] || (dom[id]={value:'@latest', options:[],
      append(option){this.options.push(option);option.remove=()=>{this.options=this.options.filter(o=>o!==option);};},
      replaceChildren(...nodes){this.nodes=nodes;},textContent:'',hidden:false});
    const document = {createElement:tag=>({tag,append(...nodes){this.nodes=nodes;}})};
    let views=[], inventories=[], clears=0;
    const clearArtifactView=()=>{clears++;};
    const viewArtifact=(session,revision,artifact,reveal)=>views.push({session,revision,handle:artifact.handle,reveal});
    const renderArtifacts=revision=>inventories.push(revision);
    const textList=x=>x;
    '''


def run_render(body):
    code = functions('function distinctFigureArtifacts', 'function scientificArtifactPath')
    return _javascript_value('(() => {' + fixture_script() + code + body + '})()')


def test_latest_requested_output_uses_conversation_order_not_http_completion_order():
    result = run_render('''
      cache('latest',[figure('latest')],[composition]);
      cache('unrelated',[figure('unrelated','other')]);
      cache('older',[figure('older')]);
      renderResultSelection();
      return {views,inventory:inventories.at(-1),text:dom['result-facts'].nodes.map(n=>n.textContent)};
    ''')
    assert result['views'] == [dict(session='session', revision=None, handle='latest', reveal=False)]
    assert [a['handle'] for a in result['inventory']['scientific_artifacts']] == ['export']
    assert result['text'] == ['Requested: Other labels and data', 'Latest supplied labels']


def test_explicit_historical_presentation_and_manual_artifact_preview_survive_later_inventory():
    result = run_render('''
      cache('older',[figure('older')]);
      element('result-selection').value='@turn:older';
      renderResultSelection();
      cache('latest',[figure('latest')],[composition]);
      renderResultSelection();
      state.artifactPreview=true;
      element('result-selection').value='@latest';
      renderResultSelection();
      return views;
    ''')
    assert [v['handle'] for v in result] == ['older', 'older']


def test_ambiguous_requested_figures_are_explicit_choices_without_arbitrary_generic_fallback():
    result = run_render('''
      state.revision.artifacts=[{handle:'native',artifact_type:'analysis_figure',revision_id:'r',
        figure:{manifest_sha256:'native',figure_id:'native',output_names:['umap']}}];
      cache('latest',[figure('latest','r','one'),figure('latest','r','two')]);
      renderResultSelection(); return {views,inventory:inventories.at(-1)};
    ''')
    assert result['views'] == []
    assert [a['handle'] for a in result['inventory']['artifacts']] == ['one', 'two', 'native']


def test_presentation_association_requires_exact_coordinates_or_export_references():
    result = run_render('''
      cache('older',[figure('older','other')]);
      const wrongPublisher=figure('forged');
      const wrongRole=figure('latest');wrongRole.figure.sources[0].role='categories';
      const native=figure('latest');native.revision_id='r';
      cache('latest',[wrongPublisher,wrongRole,native],[{...composition,contents:{source_sha256:'r'}}]);
      return presentationChoices();
    ''')
    assert result == []


def test_saved_selection_restores_when_its_inventory_arrives_and_is_not_replaced_by_latest():
    result = run_render('''
      const select=element('result-selection');select.append({value:'@latest'});select.append({value:'@overview'});
      state.resultChoice={revisionId:'r',selection:'@turn:older'};
      cache('latest',[figure('latest')]);renderPresentationChoices();
      const waiting=select.value;
      cache('older',[figure('older')]);renderPresentationChoices();
      renderResultSelection();
      return {waiting,selection:select.value,views,options:select.options.map(o=>o.value)};
    ''')
    assert result['waiting'] == '@latest'
    assert result['selection'] == '@turn:older'
    assert result['views'][0]['handle'] == 'older'
    assert result['options'] == ['@latest', '@overview', '@turn:older', '@turn:latest']


def test_tab_conveniences_do_not_follow_another_tabs_session_or_draft():
    read = functions('function readConveniences', 'function recentSessions')
    save = functions('function saveConveniences', 'async function api')
    result = _javascript_value('(() => {' + read + save + '''
      const STORAGE_KEY='display';
      const storage=()=>({data:{},getItem(k){return this.data[k]||null;},setItem(k,v){this.data[k]=v;}});
      const first=storage(),second=storage();let sessionStorage=first;
      const state={session:{session_id:'macaque',active_revision_id:'new'},viewedRevisionId:'old',
        resultChoice:{revisionId:'old',selection:'@turn:older'}};
      const dom={utterance:{value:'Continue macaque'},'research-layout':{dataset:{panel:'results'}}};
      const element=id=>dom[id]||(dom[id]={value:''});
      saveConveniences();
      sessionStorage=second;state.session.session_id='zebrafish';element('utterance').value='Fish draft';saveConveniences();
      const fish=readConveniences();sessionStorage=first;const macaque=readConveniences();
      return {fish,macaque};
    })()''')
    assert result['fish']['sessionId'] == 'zebrafish'
    assert result['macaque']['sessionId'] == 'macaque'
    assert result['macaque']['draft'] == 'Continue macaque'
    assert result['macaque']['viewedRevisionId'] == 'old'
    assert result['macaque']['resultChoice'] == dict(revisionId='old', selection='@turn:older')
    source = (STATIC / 'app.js').read_text()
    # Shared discovery remains browser-local; active context/attachments are tab-local.
    assert 'localStorage.getItem(RECENT_SESSIONS_KEY)' in source
    assert 'localStorage.setItem(RECENT_SESSIONS_KEY' in source
    assert source.count('localStorage.') == 2
    assert 'sessionStorage.getItem(`agent.web.attachments.${sessionId}`)' in source


def test_accepted_result_refreshes_missing_explanation_once_when_saved_response_arrives():
    code = functions('function revisionToDisplay', 'async function refreshCurrentSession')
    result = _javascript_value('(() => {' + code + '''
      const state={session:{session_id:'s',active_revision_id:'r'},viewedRevisionId:'r',
        revision:{revision_id:'r',turn_id:'t',result:null},attachmentSession:'s'};
      const nodes={};const element=id=>nodes[id]||(nodes[id]={});
      const shortIdentity=x=>x,resultLabel=x=>x;
      const rememberSession=()=>{},renderHistory=()=>{},renderRevisionHistory=()=>{},
        renderRevisionContext=()=>{},renderResultNavigation=()=>{},updateControls=()=>{};
      const loads=[];const clearRevisionView=()=>{};
      const loadRevision=id=>{loads.push(id);state.revision.result={text:'Saved response'};};
      const view={session_id:'s',active_revision_id:'r',generation:1,revisions:[{revision_id:'r'}],
        turns:[{turn_id:'t',status:'finalizing',response:null}]};
      displaySession(view);
      view.turns=[{turn_id:'t',status:'succeeded',response:{text:'Saved response'}}];
      displaySession(view);displaySession(view);
      return loads;
    })()''')
    assert result == ['r']


def test_requested_figure_and_downloads_precede_long_scientific_summary():
    html = (STATIC / 'index.html').read_text()
    assert html.index('id="result-selection"') < html.index('id="artifact-viewer"')
    assert html.index('id="artifact-viewer"') < html.index('id="artifact-panel"') < html.index('id="result-facts"')
