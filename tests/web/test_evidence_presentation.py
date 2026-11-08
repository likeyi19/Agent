"""Exercise ordinary evidence controls in the existing JavaScriptCore harness.

Accepted metadata comes from the existing Session/RunStore fixture. DOM and HTTP
are bounded substitutes; genuine Firefox qualification is recorded separately.
"""
import json
import sys
from pathlib import Path

import pytest
from test_static import STATIC, _javascript_value
from test_requested_presentation import fixture_script, functions

sys.path.insert(0, str(Path(__file__).parents[1] / 'application'))
from test_dialogue_evidence import accepted, forbid_work, files
from agent.application import ResearchAgentApplication, InteractiveAgentApplication
from agent.providers import PlanningModelFactoryRegistry
from rich_helpers import PROFILES


@pytest.fixture
def annotation(tmp_path, monkeypatch):
    app = ResearchAgentApplication(tmp_path / 'workspace')
    app.sessions.create('session')
    rid, _, _ = accepted(app, 'annotate_scATAC_cell_types', overrides=dict(
        n_cells=8, n_groups=2, assigned_cells=6, unassigned_cells=2, groups_omitted=0,
        group_summary=[dict(group='0', n_cells=6, primary_annotation='CD8 T', status='assigned'),
                       dict(group='1', n_cells=2, primary_annotation=None, status='insufficient_evidence')]))
    guarded = forbid_work(app, monkeypatch)
    def forbidden_provider(*args, **kwargs):
        raise AssertionError('Evidence navigation invoked a provider')
    facade = InteractiveAgentApplication(app.workspace_root, model_profiles=PROFILES,
        default_profile_id=PROFILES[0].profile_id, registry=guarded.registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden_provider}))
    before = files(app)
    views = [facade.evidence('session', rid, 'output0', fields=['group_summary']).to_dict()
             for _ in range(3)]
    assert views[0] == views[1] == views[2] and files(app) == before
    return views[0]


def run_evidence(view, body):
    fixture = fixture_script().replace('const renderArtifacts=revision=>inventories.push(revision);', '')
    fixture = fixture.replace('const textList=x=>x;', '')
    fixture = fixture.replace("const artifactPath=(session,revision,handle,turn)=>`/${session}/${revision || turn}/${handle}`;", '')
    code = (functions('class ApiError', 'function readConveniences')
            + functions('async function api', 'function sessionPath').replace('async function api', 'async function realApi')
            + functions('function sessionPath', 'function showError')
            + functions('function textList', 'function resultLabel')
            + functions('async function loadRevision', 'function scientificArtifactPath')
            + functions('function scientificArtifactPath', 'function appendScientificFiles')
            + functions('function renderArtifacts', 'async function viewArtifact')
            + functions('function evidenceValueNode', 'async function activateViewedRevision'))
    source = (STATIC / 'app.js').read_text()
    handlers = source[source.index('element("load-evidence").addEventListener'):source.index('element("model-choice").addEventListener')]
    setup = '''
    // JavaScriptCore supplies JS, while this bounded transport supplies browser query encoding.
    class URLSearchParams {
      constructor(values){this.values=Object.entries(values);}
      append(k,v){this.values.push([k,v]);}
      set(k,v){this.values=this.values.filter(([key])=>key!==k);this.append(k,v);}
      toString(){return this.values.map(([k,v])=>encodeURIComponent(k)+'='+encodeURIComponent(v)).join('&');}
    }
    const create=document.createElement;
    document.createElement=tag=>Object.assign(create(tag), {
      classList:{add(){}}, setAttribute(k,v){this[k]=v;}, style:{},
      replaceChildren(...nodes){this.nodes=nodes;}
    });
    document.createTextNode=text=>({textContent:text});
    const ids=['load-evidence','evidence-output','result-selection','detail-form','evidence-panel',
      'result-facts','evidence-content','detail-content','artifact-inventory','detail-section','detail-comparison'];
    for(const id of ids) {
      const n=element(id);n.nodes=[];n.events={};n.scrollIntoView=()=>{};
      n.addEventListener=(e,f)=>{n.events[e]=f;};
      n.append=(...nodes)=>n.nodes.push(...nodes);
      n.replaceChildren=(...nodes)=>{n.nodes=nodes;if(['result-selection','evidence-output','detail-section','detail-comparison'].includes(id)){
        n.options=nodes;n.value=nodes[0]?.value||'';}};
    }
    const calls=[], diagnostics=[], saves=[];
    let fetch;
    const console={error:(...args)=>diagnostics.push(args.map(x=>String(x)).join(' '))};
    const saveConveniences=()=>saves.push(state.resultChoice);
    const renderResultNavigation=()=>{},updateControls=()=>{};
    const snapshot=()=>JSON.stringify(state.revision);
    const flatten=n=>[n,...(n.nodes||[]).flatMap(flatten)];
    const text=id=>flatten(element(id)).map(n=>n.textContent||'').join(' ');
    const images=()=>flatten(element('result-facts')).filter(n=>n.tag==='img').map(n=>n.src);
    const downloads=()=>flatten(element('artifact-inventory')).filter(n=>n.tag==='a').map(n=>n.href);
    Object.assign(state,{epoch:1,revisionEpoch:1,evidenceEpoch:0,viewedRevisionId:'r'});
    Object.assign(state.session,{active_revision_id:'r',generation:1});
    const evidence=VIEW;
    const group={handle:'annotation',tool:'annotate_scATAC_cell_types',label:'Primary annotation',
      outputs:[{name:'output0',field:'group_summary'}],canonical_output:'output0',
      summary_fields:['group_summary'],parameters:{},derived_from:[],evidence};
    const peer={...group,handle:'peer',label:'Other accepted output',
      outputs:[{name:'peer',field:'n_cells'}],canonical_output:'peer'};
    const published=(handle,names)=>({handle,revision_id:'r',turn_id:'science',artifact_type:'analysis_figure',sha256:handle,
      figure:{kind:'annotation_composition',sources:[],step_ids:['step0'],output_names:names,manifest_sha256:handle,figure_id:handle}});
    Object.assign(state.revision,{parent_revision_id:null,is_active:true,status:'activated',turn_id:'science',run_id:'initial:run',
      retained_outputs:[],outputs:['output0','peer'],evidence_outputs:['output0','peer'],
      result_groups:[group,peer],artifacts:[published('unrelated',['peer']),published('associated',['output0'])],
      scientific_artifacts:[{handle:'h5ad',revision_id:'r',turn_id:'science',artifact_type:'scientific_annotation',
        filename:'annotated.h5ad',title:'Annotation',description:'Accepted labels',size_bytes:42}]});
    element('result-selection').value='@overview';element('evidence-output').value='output0';
    element('detail-limit').value='8';
    const acceptedRevision=state.revision;
    let api=async path=>{calls.push(path);return path.includes('/evidence?')?evidence:acceptedRevision;};
    '''.replace('VIEW', json.dumps(view))
    # The system runtime drains Promise jobs before serializing this shared result.
    script = ('(()=>{' + fixture + setup + code + handlers + '''
      const result={};(async()=>{''' + body + '''})().then(value=>{
        Object.assign(result,{completed:true},value);
      }).catch(error=>{result.error=String(error);});return result;})()''')
    result = _javascript_value(script)
    assert result.get('completed'), result
    return result


def test_show_evidence_click_renders_accepted_facts_provenance_and_associated_gallery(annotation):
    result = run_evidence(annotation, '''
      renderResultSelection();const before=snapshot(), links=downloads();
      await element('load-evidence').events.click();
      return {selection:element('result-selection').value,facts:text('result-facts'),evidence:text('evidence-content'),
        images:images(),links,afterLinks:downloads(),views,clears,unchanged:before===snapshot(),
        section:element('detail-section').value,diagnostics,calls,choice:state.resultChoice};
    ''')
    assert result['selection'] == '@field' and result['choice']['selection'] == '@field'
    assert 'CD8 T' in result['facts'] and 'Recorded null' in result['facts']
    assert 'insufficient_evidence' in result['facts']
    assert 'Verification & provenance' in result['evidence'] and 'initial:run' in result['evidence']
    assert annotation['source']['evidence_sha256'] in result['evidence']
    assert len(result['images']) == 1 and '/associated' in result['images'][0]
    assert result['links'] == result['afterLinks'] and any('h5ad' in s for s in result['links'])
    assert result['views'] == [] and result['unchanged'] and result['diagnostics'] == []
    assert result['section'] == 'annotation_rationale' and len(result['calls']) == 1


def test_evidence_navigation_reopen_and_repeated_reads_keep_preview_and_scientific_state(annotation):
    result = run_evidence(annotation, '''
      renderResultSelection();const before=snapshot();
      state.artifactPreview=true;const pinnedClears=clears;
      await element('load-evidence').events.click();await element('load-evidence').events.click();
      const fieldImages=images(),previewRetained=clears===pinnedClears;
      await loadRevision('r'); await Promise.resolve();
      const reopened=element('result-selection').value;
      element('result-selection').value='@overview';element('result-selection').events.change();
      const overview=images();element('result-selection').value='peer';element('result-selection').events.change();
      const other=images();
      return {fieldImages,previewRetained,reopened,overview,other,unchanged:before===snapshot(),diagnostics,calls};
    ''')
    assert result['previewRetained'] and result['reopened'] == '@field'
    assert len(result['fieldImages']) == 1 and '/associated' in result['fieldImages'][0]
    assert len(result['overview']) == 2 and len(result['other']) == 1 and '/unrelated' in result['other'][0], json.dumps(result)
    assert result['unchanged'] and result['diagnostics'] == []


@pytest.mark.parametrize('detail', [False, True])
@pytest.mark.parametrize('failure', ['http', 'network', 'json'])
def test_transport_failure_and_client_render_failure_have_distinct_diagnostics(annotation, detail, failure):
    result = run_evidence(annotation, '''
      const isDetail=DETAIL;const target=isDetail?'detail-content':'evidence-content';
      api=realApi;
      fetch=async()=>{
        if('FAILURE'==='network')throw new Error('offline');
        return {ok:false,status:404,json:async()=>{
          if('FAILURE'==='json')throw new Error('unreadable');
          return {error:{code:'EVIDENCE_UNAVAILABLE',message:'Accepted output not accessible'}};
        }};
      };
      await loadEvidence(isDetail);const transport=text(target),transportSummary=text('result-facts');
      fetch=async()=>({ok:true,status:200,json:async()=>evidence});
      const original=document.createElement;document.createElement=()=>{throw new ReferenceError('forced render fault');};
      await loadEvidence(isDetail);document.createElement=original;
      return {transport,transportSummary,render:text(target),summary:text('result-facts'),diagnostics};
    '''.replace('DETAIL', str(detail).lower()).replace('FAILURE', failure))
    expected = {'http': 'EVIDENCE_UNAVAILABLE: Accepted output not accessible',
                'network': 'CONNECTION_UNAVAILABLE', 'json': 'RESPONSE_UNAVAILABLE'}
    assert expected[failure] in result['transport']
    assert 'EVIDENCE_RENDER_FAILED' in result['render'] and 'was received' in result['render']
    assert 'Summary unavailable' not in result['render']
    assert len(result['diagnostics']) == 1 and 'forced render fault' in result['diagnostics'][0]
    if not detail:
        assert 'Evidence request failed' in result['transportSummary']
        assert 'was received' in result['summary']


def test_detail_read_keeps_field_gallery_and_manually_selected_artifact(annotation):
    result = run_evidence(annotation, '''
      await element('load-evidence').events.click();
      const before=images(),selected=element('result-selection').value;
      state.artifactPreview=true;const previousClears=clears;
      await element('detail-form').events.submit({preventDefault(){}});
      return {before,after:images(),selection:element('result-selection').value,selected,
        previewRetained:clears===previousClears,detail:text('detail-content'),calls};
    ''')
    assert result['before'] == result['after'] and result['selection'] == result['selected'] == '@field'
    assert result['previewRetained'] and 'CD8 T' in result['detail']
    assert 'detail_section=annotation_rationale' in result['calls'][-1]


def test_late_evidence_does_not_replace_explicit_requested_or_historical_view(annotation):
    result = run_evidence(annotation, '''
      cache('older',[figure('older')]);cache('latest',[figure('latest')]);
      let resolve;api=()=>new Promise(done=>{resolve=done;});
      const reading=element('load-evidence').events.click();
      element('result-selection').value='@turn:older';element('result-selection').events.change();
      resolve(evidence);await reading;
      return {selection:element('result-selection').value,images:images(),facts:text('result-facts'),diagnostics};
    ''')
    assert result['selection'] == '@turn:older' and '/older/' in result['images'][0]
    assert 'First supplied labels' in result['facts'] and result['diagnostics'] == []


def test_unknown_field_group_never_falls_back_to_unrelated_figures(annotation):
    result = run_evidence(annotation, '''
      api=async()=>({...evidence,output_name:'historical-only-field'});
      await element('load-evidence').events.click();return {images:images(),diagnostics};
    ''')
    assert result['images'] == [] and result['diagnostics'] == []
