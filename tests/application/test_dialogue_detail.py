"""Bounded publication access, not owner recomputation or biological qualification."""
import gzip
import hashlib
import json
from dataclasses import replace

import pytest

from agent.application.dialogue_evidence import DetailRequest
from agent.application import dialogue_evidence as de
from agent.application import scientific_dialogue as sd
from test_dialogue_evidence import app, accepted, forbid_work, files
from test_scientific_dialogue import Model, question, ask


def publication(app, *, qc=False, candidates=2, huge=False):
    root = app.workspace_root / 'published'
    root.mkdir(exist_ok=True)
    def write(name, value, raw=False):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        data = value if raw else json.dumps(value).encode()
        path.write_bytes(data)
        return dict(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    if qc:
        from agent.tools.data._barcode_qc_contract import CONTRACT, ARTIFACT
        from agent.tools.data.scatac_qc_profile import PROFILE_SHA256
        binding = write('lengths.tsv.gz', gzip.compress(''.join(f'{i}\t{i*2}\n' for i in range(1,1002)).encode()), True)
        manifest = dict(contract_version=CONTRACT, artifact_type=ARTIFACT, schema_version=1,
            science_profile_sha256=PROFILE_SHA256, histogram=dict(path='lengths.tsv.gz', **binding))
        tool = 'compute_scATAC_qc'
        values = {}
    else:
        from agent.tools.analysis.annotation_contract import CONTRACT, ARTIFACT, PROFILE_SHA256
        groups = [dict(group=g, n_cells=3, primary_annotation='B', status='assigned',
            reason='unique_positive_maximum', top_score=2.0, top_candidates=['B']) for g in ('3','7')]
        rows = [dict(group=g, candidate='B' if n == 0 else f'C{n}', score=2.0,
            matched_entries=1, positive=['MS4A1'] if not huge else ['X'*20000], negative=[],
            unobserved_marker_genes=[]) for g in ('3','7') for n in range(candidates)]
        coverage = [dict(candidate='B' if n == 0 else f'C{n}', entries=3,
            measured_entries=2, duplicate_entries=0, missing_genes=['CD19']) for n in range(candidates)]
        manifest = dict(contract_version=CONTRACT, artifact_type=ARTIFACT, schema_version=1,
            profile_sha256=PROFILE_SHA256, sidecars={
                'primary/groups.json':write('primary/groups.json',groups),
                'primary/candidate_evidence.json':write('primary/candidate_evidence.json',rows),
                'primary/signature_coverage.json':write('primary/signature_coverage.json',coverage)})
        tool = 'annotate_scATAC_cell_types'
        values = dict(group_summary=[{k:g[k] for k in ('group','n_cells','primary_annotation','status')} for g in groups], groups_omitted=0)
    binding = write('manifest.json',manifest)
    rid = accepted(app, tool, overrides=dict(values, manifest_path=str(root/'manifest.json'),
        manifest_sha256=binding['sha256'], contract_version=CONTRACT, artifact_type=ARTIFACT, artifact_schema_version=1))[0]
    return rid, root


def test_annotation_exact_detail_and_no_work(app, monkeypatch):
    rid, root = publication(app)
    guarded = forbid_work(app, monkeypatch)
    before = files(app)
    view = guarded.sessions.evidence('session',rid,'output0', detail=DetailRequest('annotation_rationale','3',1))
    assert view.status == 'available', view
    facts = {f.field:f for f in view.detail}
    assert facts['assignment.0'].value['reason'] == 'unique_positive_maximum'
    assert facts['candidate.0'].value['positive'] == ('MS4A1',)
    assert facts['candidate.records_omitted'].value == 1
    assert facts['signature_coverage.records_omitted'].value == 1
    assert facts['candidate.0'].artifact_sha256 == hashlib.sha256((root/'primary/candidate_evidence.json').read_bytes()).hexdigest()
    assert facts['candidate.0'].source_pointer == '/0'
    assert files(app) == before


@pytest.mark.parametrize('attack', ['missing','corrupt','manifest','subject','candidate','section','symlink','oversize'])
def test_detail_fails_closed(app, monkeypatch, attack):
    rid, root = publication(app)
    request = DetailRequest('annotation_rationale','3')
    path = root/'primary/candidate_evidence.json'
    if attack == 'missing': path.unlink()
    if attack == 'corrupt': path.write_bytes(path.read_bytes()+b' ')
    if attack == 'manifest': (root/'manifest.json').write_text('{}')
    if attack == 'subject': request = replace(request, subject='99')
    if attack == 'candidate': request = replace(request, candidate='unknown')
    if attack == 'section': request = replace(request, section='arbitrary/path')
    if attack == 'symlink':
        target = root/'other.json'; path.rename(target); path.symlink_to(target)
    if attack == 'oversize': monkeypatch.setattr(de,'MAX_DETAIL_BYTES',20)
    result = forbid_work(app,monkeypatch).sessions.evidence('session',rid,'output0',detail=request)
    assert result.status in ('unavailable','unsupported') and not result.detail and not result.facts


def test_oversized_record_is_explicitly_omitted(app):
    rid, _ = publication(app, huge=True)
    view = app.sessions.evidence('session',rid,'output0',detail=DetailRequest('annotation_rationale','3'))
    assert view.status == 'available'
    assert all(f.status=='omitted' and f.reason=='field_size_limit' for f in view.detail if f.field.startswith('candidate.') and f.field != 'candidate.records_omitted')


def test_non_annotation_histogram_shape(app, monkeypatch):
    rid, _ = publication(app,qc=True)
    view = forbid_work(app,monkeypatch).sessions.evidence('session',rid,'output0',detail=DetailRequest('length_histogram',limit=2))
    assert view.status=='available',view
    assert view.detail[0].value == {'length_bin':'1','fragment_records':2}
    assert view.detail[1].value == {'length_bin':'2','fragment_records':4}
    assert view.detail[-1].value==999


def test_detail_restart_subject_followup_and_comparison(app, monkeypatch):
    publication(app)
    app = forbid_work(app,monkeypatch)
    fields = ['detail.assignment.0','detail.candidate.0']
    first = ask(app,'a','Why was cluster 3 annotated this way?',Model(question(subject='3'),fields))
    assert first.status=='answered',first
    assert first.scientific.evidence_scope=='accepted_summary_and_published_detail'
    from agent.application import ResearchAgentApplication
    restarted = ResearchAgentApplication(app.workspace_root,registry=app.registry)
    second = ask(restarted,'b','What about cluster 7?',Model(question('@focus','7'),fields))
    assert second.status=='answered',second
    pair = ask(restarted,'c','How do their supporting markers differ?',Model(question('@focus',comparison=dict(output='@previous',subject=None)),fields))
    assert pair.status=='answered',pair
    assert {c.subject for c in pair.scientific.claims}=={'3','7'}
    assert all(c.source['artifact_sha256'] for c in pair.scientific.claims)
    limit = ask(restarted,'d','Does this prove distinct cell types?',Model(question('@focus'),fields,support='insufficient_evidence'))
    assert limit.scientific.support=='insufficient_evidence'


@pytest.mark.parametrize('change', ['qc_identity_sha256','selection_method','resource_qualification','selection_profile_sha256'])
def test_comparison_scope_and_semantic_mismatch(app, monkeypatch, change):
    values = dict(qc_identity_sha256='a'*64,n_selected=8)
    accepted(app,'select_scATAC_cells',arguments=dict(min_qc_fragment_records=1,min_tss_enrichment='4'),overrides=values,extra_facts={'selection_profile_sha256':'a'*64})
    other = dict(values)
    extra = {'selection_profile_sha256':'a'*64}
    if change in other or change != 'selection_profile_sha256': other[change]='different'
    else: extra[change]='b'*64
    accepted(app,'select_scATAC_cells',arguments=dict(min_qc_fragment_records=1,min_tss_enrichment='4'),name='second',overrides=other,extra_facts=extra)
    response = ask(forbid_work(app,monkeypatch),'a','Compare with the parent.',Model(question(comparison=dict(output='r1',subject=None)),['n_selected']))
    assert response.status=='unavailable'


def test_other_cross_result_comparison_requires_reviewed_scope(app):
    accepted(app,'inspect_scATAC')
    accepted(app,'inspect_scATAC',name='second')
    response=ask(app,'a','Compare with the parent.',Model(question(comparison=dict(output='r1',subject=None)),['n_cells']))
    assert response.status=='unavailable'


def test_invalid_detail_bound(app):
    for limit in (0,33,True):
        with pytest.raises(ValueError): DetailRequest('length_histogram',limit=limit)


def test_comparison_pair_is_agent_bound(app):
    values = dict(qc_identity_sha256='a'*64,n_selected=8)
    accepted(app,'select_scATAC_cells',arguments=dict(min_qc_fragment_records=1,min_tss_enrichment='4'),overrides=values)
    accepted(app,'select_scATAC_cells',arguments=dict(min_qc_fragment_records=1,min_tss_enrichment='4'),name='second',overrides=dict(values,n_selected=5))
    def reply(payload):
        evidence=payload['evidence']
        by_id={c['claim_id']:c for c in evidence['claims']}
        pair=next(p for p in evidence['comparison']['pairs'] if by_id[p['left']]['field']=='n_selected')
        assert pair['relationship']=='different'
        assert evidence['comparison']['ranking_available'] is False
        return dict(support='supported',paragraphs=[dict(parts=[dict(kind='comparison',id=pair['id'])])])
    response=ask(app,'a','Compare selected counts with the parent.',Model(question(comparison=dict(output='r1',subject=None)),response=reply))
    assert response.status=='answered',response
    assert {c.value for c in response.scientific.claims}=={5,8}
    assert 'different' in response.text


def test_no_file_discovery_and_exact_candidate_selection(app,monkeypatch):
    rid,_=publication(app,candidates=4)
    from pathlib import Path
    def forbidden(*args,**kwargs): raise AssertionError('arbitrary discovery')
    monkeypatch.setattr(Path,'glob',forbidden)
    monkeypatch.setattr(Path,'rglob',forbidden)
    view=app.sessions.evidence('session',rid,'output0',detail=DetailRequest('annotation_rationale','3',candidate='C3'))
    assert view.status=='available',view
    candidates=[f for f in view.detail if f.field=='candidate.0']
    assert len(candidates)==1 and candidates[0].value['candidate']=='C3'
    assert candidates[0].source_pointer=='/3'


def test_changed_detail_during_generation_is_rejected(app):
    _,root=publication(app)
    def reply(payload):
        c=next(c for c in payload['evidence']['claims'] if c['field']=='detail.candidate.0')
        path=root/'primary/candidate_evidence.json'
        path.write_bytes(path.read_bytes()+b' ')
        return dict(support='supported',paragraphs=[dict(parts=[dict(kind='claim',id=c['claim_id'])])])
    result=ask(app,'a','Why cluster 3?',Model(question(subject='3'),response=reply))
    assert result.status=='unavailable' and result.scientific is None
