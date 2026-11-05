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
    const document = {createElement:tag=>({tag,nodes:[],events:{},dataset:{},
      append(...nodes){this.nodes.push(...nodes);},addEventListener(event,fn){this.events[event]=fn;}})};
    const readableLabel=x=>x;
    const resultLabel=x=>x;
    const artifactPath=(session,revision,handle,turn)=>`/${session}/${revision || turn}/${handle}`;
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
    assert result['views'] == []  # Inline gallery does not select a scientific result.
    assert [a['handle'] for a in result['inventory']['scientific_artifacts']] == ['export']
    assert result['text'][:2] == ['Requested: Other labels and data', 'Latest supplied labels']


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
      return {views,clears};
    ''')
    assert result == dict(views=[], clears=2)


def test_multiple_requested_figures_are_displayed_without_arbitrary_generic_fallback():
    result = run_render('''
      state.revision.artifacts=[{handle:'native',artifact_type:'analysis_figure',revision_id:'r',
        figure:{manifest_sha256:'native',figure_id:'native',output_names:['umap']}}];
      cache('latest',[figure('latest','r','one'),figure('latest','r','two')]);
      renderResultSelection(); return {views,inventory:inventories.at(-1),
        figures:dom['result-facts'].nodes.find(n=>n.className==='result-figures').nodes
          .filter(n=>n.tag==='figure').map(n=>n.nodes.find(c=>c.tag==='img').src)};
    ''')
    assert result['views'] == []
    assert [a['handle'] for a in result['inventory']['artifacts']] == ['one', 'two', 'native']
    assert result['figures'] == ['/api/v1/session/latest/one', '/api/v1/session/latest/two']


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
      return {waiting,selection:select.value,figures:resultFigureArtifacts(null,presentationChoices()[0]).map(a=>a.handle),
        options:select.options.map(o=>o.value)};
    ''')
    assert result['waiting'] == '@latest'
    assert result['selection'] == '@turn:older'
    assert result['figures'] == ['older']
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


def test_summary_and_inline_figures_precede_download_details():
    html = (STATIC / 'index.html').read_text()
    assert html.index('id="result-selection"') < html.index('id="artifact-viewer"')
    assert html.index('id="result-facts"') < html.index('id="artifact-viewer"') < html.index('id="artifact-panel"')


def test_gallery_bound_deduplication_errors_and_exact_group_association():
    result = run_render('''
      const native=(name,output,handle=name)=>({handle,revision_id:'r',turn_id:'science',
        artifact_type:'analysis_figure',figure:{kind:'umap_leiden',sources:[],
        manifest_sha256:name,figure_id:name,output_names:[output]}});
      state.revision.artifacts=[native('unrelated','other'),native('same','chosen'),
        native('same','chosen','report-copy'),...Array.from({length:7},(_,i)=>native(`f${i}`,'chosen'))];
      const selected=resultFigureArtifacts({outputs:[{name:'chosen'}]});
      const gallery=resultFigureGallery(selected);
      const cards=gallery.nodes.filter(n=>n.tag==='figure');
      const first=cards[0],image=first.nodes.find(n=>n.tag==='img');
      const src=image.src;image.events.error();
      first.nodes.find(n=>n.tag==='button').events.click();
      return {selected:selected.map(a=>a.handle),cards:cards.length,src,hidden:image.hidden,
        error:first.nodes.find(n=>n.tag==='figcaption').textContent,
        note:gallery.nodes.at(-1).textContent,views,empty:resultFigureGallery([])};
    ''')
    assert result['selected'] == ['same'] + [f'f{i}' for i in range(7)]
    assert result['cards'] == 6 and result['empty'] is None
    assert result['src'] == '/api/v1/session/r/same'
    assert result['hidden'] and 'unavailable' in result['error']
    assert result['note'].startswith('Showing 6 of 8')
    assert result['views'] == [dict(session='session', revision='r', handle='same')]


def test_group_table_and_exact_metrics_are_visible_and_other_facts_remain_in_methods():
    values = functions('function evidenceValueNode', 'function renderEvidence')
    result = run_render(values + '''
      document.createTextNode=text=>({textContent:text});
      document.createElementOld=document.createElement;
      document.createElement=tag=>({...document.createElementOld(tag),setAttribute(){}});
      const facts=[
        {field:'n_cells',status:'available',value:569},
        {field:'group_summary',status:'available',value:[{group:'0',n_cells:203,primary_annotation:'B cells',status:'assigned'}]},
        {field:'metrics',status:'available',value:{macro_f1:0.1234567890123456}},
        {field:'unavailable_metric',status:'unavailable',value:0.99,reason:'No independent reference'},
        {field:'source_sha256',status:'available',value:'exact-sha'}];
      state.revision.result={text:'Saved scientific response'};
      state.revision.result_groups=[{handle:'a',label:'Annotation',summary_fields: ['n_cells','group_summary','metrics','unavailable_metric'],
        parameters:{},derived_from:[],
        evidence:{facts,field_labels:{},coverage:[],limitations:['Predictions are not validated identities.']}}];
      renderResultSelection();
      const nodes=dom['result-facts'].nodes;
      const section=nodes.find(n=>n.className==='scientific-result');
      return {nodes,methods:section.nodes.find(n=>n.tag==='details')};
    ''')
    text = json.dumps(result)
    assert '569' in text and '203' in text and 'B cells' in text
    assert '0.1234567890123456' in text and '0.99' not in text
    assert 'No independent reference' in text
    assert 'exact-sha' in json.dumps(result['methods'])
    # Group records and quantitative metrics are not inside a closed disclosure.
    section = next(n for n in result['nodes'] if n.get('className') == 'scientific-result')
    values = next(n for n in section['nodes'] if n['tag'] == 'dl')
    assert not any(n.get('tag') == 'details' for n in values['nodes'])
    assert result['nodes'][0]['textContent'] == 'Saved scientific response'
    assert 'not validated' in result['nodes'][1]['textContent']
