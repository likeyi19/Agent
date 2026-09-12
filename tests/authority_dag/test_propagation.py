from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pytest

from agent.orchestration import AgentPlan, PlanStep, StepOutputRef, AgentRequest, AgentRuntime, FileRunStore
from agent.orchestration.verification_authority import accepted_authorities, load_fragment_authority
from agent.schemas.verification_authority import AuthorityError, VerifiedArtifactAuthority
from agent.tools.data.authority_context import VerificationContext
from agent.tools.data.fragments_authority_contract import CONTRACTS


class Planner:
    def __init__(self, plan): self.value = plan
    def plan(self, request, registry): return self.value


@pytest.fixture(params=['bam_fragment_production', 'external_fragment_adoption'])
def chain(request, bam_factory, source_factory, tmp_path, monkeypatch):
    from agent.tools.data import scatac_qc_reference as qr
    from agent.tools.data import scatac_reference as reference
    kind = request.param
    args = bam_factory(sq=[{'SN':n,'LN':6001} for n in ('chr2','chr1')]) if kind == 'bam_fragment_production' else source_factory()
    import pysam
    fa=tmp_path/'long-reference.fa'
    fa.write_text('>chr2\n'+'A'*6001+'\n>chr1\n'+'C'*6001+'\n')
    pysam.faidx(str(fa))
    bed=tmp_path/'ccre.bed';bed.write_text('chr2\t0\t100\nchr1\t0\t1000\n')
    parent=reference.build_scatac_reference_bundle(species='human',target_assembly='hg38',
        fasta_path=fa,fai_path=Path(str(fa)+'.fai'),ccre_bed_path=bed)
    pointer=reference.publish_scatac_reference_bundle(parent,tmp_path/'parent.json')
    args.update(reference_bundle_path=pointer['manifest_path'],reference_bundle_sha256=pointer['manifest_sha256'])
    monkeypatch.setenv('AGENT_QC_BEDTOOLS','/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS','/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC','1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG',raising=False)
    annotation = tmp_path/'annotation.tsv'
    annotation.write_text('chr2\t3000\t3100\t+\tg1\tt1\tprotein_coding\n')
    qr.build_scatac_qc_reference_bundle(parent_manifest_path=args['reference_bundle_path'],
        parent_manifest_sha256=args['reference_bundle_sha256'],annotation_path=annotation,
        annotation_source='synthetic',annotation_release='1',
        classifications=tuple(qr.QCContig(n,6001,'primary_nuclear_qc') for n in ('chr2','chr1')),
        classification_source='explicit-test',output_dir=tmp_path/'qc-reference')
    qp=tmp_path/'qc-reference/manifest.json'
    pair=lambda step,prefix:{prefix+'path':StepOutputRef(step,'manifest_path'),prefix+'sha256':StepOutputRef(step,'manifest_sha256')}
    steps=(PlanStep('fragments',CONTRACTS[kind].tool_name,args),
        PlanStep('qc','compute_scATAC_qc',pair('fragments','fragments_manifest_') | dict(
            qc_reference_manifest_path=str(qp),qc_reference_manifest_sha256=hashlib.sha256(qp.read_bytes()).hexdigest(),
            output_dir=str(tmp_path/'qc')),('fragments',)),
        PlanStep('selection','select_scATAC_cells',pair('qc','barcode_qc_manifest_') | dict(
            min_qc_fragment_records=0,min_tss_enrichment='0',output_dir=str(tmp_path/'selection')),('qc',)),
        PlanStep('matrix','build_scATAC_cell_by_ccre',pair('fragments','fragments_manifest_') | pair('selection','selected_cells_manifest_') |
            dict(reference_manifest_path=args['reference_bundle_path'],reference_manifest_sha256=args['reference_bundle_sha256'],
                 output_dir=str(tmp_path/'matrix')),('fragments','selection')))
    plan=AgentPlan('authority-plan','authority-request','Tiny producer-neutral DAG.',steps)
    return kind,args,plan,FileRunStore(tmp_path/'state')


@pytest.fixture
def completed(chain):
    kind,args,plan,store=chain
    run=AgentRuntime(planner=Planner(plan),run_store=store).run(AgentRequest('authority-request','Tiny DAG.',{}))
    assert run.status.value=='SUCCEEDED', run.to_dict()['errors']
    return kind,args,run,store


def test_producer_neutral_owned_science_and_source_access(chain, monkeypatch):
    from agent.tools.data import _external_fragment_io as io, _bam_fragment_io as bam
    from agent.tools.data import external_fragments as external, external_fragments_verifier as verifier
    from agent.tools.data import _barcode_qc_binding as binding
    kind,args,plan,store=chain
    counts=Counter(); phase=['producer']; raw=args['source_path']
    original=VerificationContext.verify
    def verified(self,kind,function,args,kwargs):
        def independent(*a,**kw):
            counts[kind]+=1
            return function(*a,**kw)
        return original(self,kind,independent,args,kwargs)
    monkeypatch.setattr(VerificationContext,'verify',verified)
    def instrument(module,name,category):
        original=getattr(module,name)
        def call(path,*a,**kw):
            if str(path)==raw:
                assert phase[0]=='producer', 'Downstream reentered raw source'
                counts[category]+=1
            return original(path,*a,**kw)
        monkeypatch.setattr(module,name,call)
    instrument(io,'encoding','bounded_encoding_inspections')
    if kind == 'bam_fragment_production':
        inspect = bam._raw_bam.inspect_bam_inputs
        def inspected(*a,**kw):
            assert phase[0]=='producer'
            counts['bounded_intake_inspections']+=1
            return inspect(*a,**kw)
        monkeypatch.setattr(bam._raw_bam,'inspect_bam_inputs',inspected)
    instrument(io,'file_sha256','full_source_hashes')
    instrument(bam,'project','bam_complete_projections')
    instrument(external,'scan_source','external_production_scans')
    instrument(verifier,'_reconstruct_source','external_independent_scans')
    original_open=Path.open
    def opened(path,*a,**kw):
        if str(path)==raw and phase[0]=='downstream':
            pytest.fail('Downstream opened a historical producer source')
        return original_open(path,*a,**kw)
    monkeypatch.setattr(Path,'open',opened)
    original_bind=binding.bind
    def bind(*a,**kw):
        phase[0]='downstream'
        return original_bind(*a,**kw)
    monkeypatch.setattr(binding,'bind',bind)
    run=AgentRuntime(planner=Planner(plan),run_store=store).run(AgentRequest('authority-request','Tiny DAG.',{}))
    assert run.status.value=='SUCCEEDED',run.to_dict()['errors']
    assert counts[kind]==counts['qc']==counts['selection']==counts['matrix']==1
    if kind=='bam_fragment_production': assert counts['bam_complete_projections']==2
    else: assert counts['external_production_scans']==counts['external_independent_scans']==1
    for step in run.steps:
        assert step.verification.artifact_authority['schema_version']==2
    handle=load_fragment_authority(store,run.run_id,'fragments')
    for other in CONTRACTS:
        if other != kind:
            with pytest.raises(AuthorityError):
                handle.validate(run.steps[0].result['manifest_path'],run.steps[0].result['manifest_sha256'],producer_kind=other)
    from agent.tools.data.cell_by_ccre_verifier import verify_cell_by_ccre
    before=dict(counts)
    with accepted_authorities(store,run.run_id):
        matrix=run.steps[-1].result
        verify_cell_by_ccre(matrix['manifest_path'],expected_sha256=matrix['manifest_sha256'],bedtools_path='/usr/bin/bedtools')
    assert dict(counts)==before
    print(kind, dict(counts))


@pytest.mark.parametrize('step_id,field', [('fragments','bgzf'),('fragments','tabix'),('qc','table'),('qc','histogram'),
    ('selection','decisions'),('selection','selected'),('matrix','matrix'),
    *[(step,k) for step in ('fragments','qc','selection','matrix') for k in ('manifest','receipt')]])
def test_published_mutations_never_inherit_authority(completed, step_id, field):
    _,_,run,store=completed
    step=next(s for s in run.steps if s.step_id==step_id)
    path=Path(step.result['manifest_path']); value=json.loads(path.read_bytes())
    if field=='manifest': target=path
    elif field=='receipt': target=(path.parent.parent if step_id in ('fragments','matrix') else path.parent)/'receipt.json'
    elif step_id=='fragments': target=path.parent/value['libraries'][0][field]['path']
    else: target=path.parent/value[field]['path']
    data=target.read_bytes(); replacement=target.with_name(target.name+'.replacement')
    replacement.write_bytes(bytes([data[0]^1])+data[1:]); replacement.replace(target)
    with pytest.raises(ValueError):
        with accepted_authorities(store,run.run_id): pass


def test_historical_source_vs_explicit_freshness(completed):
    _,args,run,store=completed
    Path(args['source_path']).write_bytes(b'changed historical source')
    with accepted_authorities(store,run.run_id): pass
    with pytest.raises(AuthorityError):
        with accepted_authorities(store,run.run_id,source_policy='current_source_freshness.v1'): pass


@pytest.mark.parametrize('field,value',[('scope','artifact_integrity_lineage.v1'),('science_profile','0'*64),
    ('verifier',{'id':'unsupported','compatibility_version':'99'}),('producer_qualification',{})])
def test_incompatible_persisted_qualification(completed,field,value):
    _,_,run,store=completed; state=store.load(run.run_id); step=state.steps[0]
    authority=VerifiedArtifactAuthority(step.verification.artifact_authority).to_dict();authority[field]=value
    changed=replace(state,steps=(replace(step,verification=replace(step.verification,artifact_authority=authority)),*state.steps[1:]))
    class InvalidStore:
        def load(self,_):return changed
    with pytest.raises(AuthorityError):
        with accepted_authorities(InvalidStore(),run.run_id):pass


@pytest.mark.parametrize('step_id', ['qc','selection','matrix'])
def test_every_downstream_authority_rejects_incompatible_bindings(completed,step_id):
    _,_,run,store=completed; state=store.load(run.run_id)
    index=next(i for i,s in enumerate(state.steps) if s.step_id==step_id)
    step=state.steps[index]
    mutations={
        'execution_identity':'0'*64, 'receipt_sha256':'0'*64,
        'scope':'presentation_validation.v1', 'science_profile':'0'*64,
        'verifier':{'id':'unknown','compatibility_version':'99'},
        'upstream':{'wrong_authority':'0'*64},
        'resources':{'wrong_reference_or_order':'0'*64},
    }
    for field,value in mutations.items():
        record=VerifiedArtifactAuthority(step.verification.artifact_authority).to_dict();record[field]=value
        modified=replace(step,verification=replace(step.verification,artifact_authority=record))
        changed=replace(state,steps=state.steps[:index]+(modified,)+state.steps[index+1:])
        class InvalidStore:
            def load(self,_):return changed
        with pytest.raises(AuthorityError):
            with accepted_authorities(InvalidStore(),run.run_id):pass


def test_freshness_audit_hashes_each_source_once_and_rechecks_mutation(completed,monkeypatch):
    from agent.tools.data import fastq_verification_authority as physical
    _,args,run,store=completed; source=args['source_path']; calls=[]
    original=physical.file_closure
    def closure(paths):
        paths=list(paths)
        calls.extend(p for p in paths if str(p)==source)
        return original(paths)
    monkeypatch.setattr(physical,'file_closure',closure)
    with accepted_authorities(store,run.run_id,source_policy='current_source_freshness.v1') as context:
        context.validate_anchors(); context.validate_anchors()
    assert len(calls)==1
    with pytest.raises(AuthorityError):
        with accepted_authorities(store,run.run_id,source_policy='current_source_freshness.v1'):
            Path(source).write_bytes(b'new source identity')


def test_resource_mutation_during_consumption_fails(completed):
    from agent.tools.data.cell_by_ccre_verifier import verify_cell_by_ccre
    _,args,run,store=completed
    path=Path(args['reference_bundle_path'])
    matrix=run.steps[-1].result
    with accepted_authorities(store,run.run_id):
        raw=path.read_bytes();path.write_bytes(raw.replace(b'hg38',b'hg39'))
        with pytest.raises(ValueError):
            verify_cell_by_ccre(matrix['manifest_path'],expected_sha256=matrix['manifest_sha256'],bedtools_path='/usr/bin/bedtools')


def test_historical_reuse_never_requires_present_source(completed):
    _,args,run,store=completed
    # Relocate only the tiny fixture source; historical qualification is retained.
    source=Path(args['source_path']);source.rename(source.with_suffix('.archived'))
    with accepted_authorities(store,run.run_id):pass
    with pytest.raises((AuthorityError,OSError)):
        with accepted_authorities(store,run.run_id,source_policy='current_source_freshness.v1'):pass


def test_new_qc_science_consumes_persisted_producer_without_raw_access(completed,monkeypatch):
    from agent.tools.data.barcode_qc_verifier import verify_barcode_qc
    _,args,run,store=completed;state=store.load(run.run_id)
    class ProducerOnlyStore:
        def load(self,_):
            return replace(state,steps=(state.steps[0], *(
                replace(step,verification=replace(step.verification,artifact_authority=None))
                for step in state.steps[1:])))
    source=Path(args['source_path']);source.rename(source.with_suffix('.archived'))
    calls=Counter();original=VerificationContext.verify
    def verify(self,kind,function,args,kwargs):
        def independent(*a,**kw):
            calls[kind]+=1
            return function(*a,**kw)
        return original(self,kind,independent,args,kwargs)
    monkeypatch.setattr(VerificationContext,'verify',verify)
    qc=run.steps[1].result
    with accepted_authorities(ProducerOnlyStore(),run.run_id):
        verify_barcode_qc(qc['manifest_path'],expected_sha256=qc['manifest_sha256'])
    assert calls=={'qc':1}


def test_unrecorded_fragments_deep_fallback_with_shared_output_root(completed,monkeypatch):
    kind,args,original_run,store=completed
    fragments=original_run.steps[0].result
    original_plan=store.load(original_run.run_id).plan
    steps=[]
    for step in original_plan.steps[1:]:
        arguments=dict(step.arguments)
        arguments['output_dir']=args['output_dir']
        if 'fragments_manifest_path' in arguments:
            arguments.update(fragments_manifest_path=fragments['manifest_path'],fragments_manifest_sha256=fragments['manifest_sha256'])
        steps.append(replace(step,arguments=arguments,depends_on=tuple(d for d in step.depends_on if d!='fragments')))
    plan=AgentPlan('fallback-plan','fallback-request','Explicit unrecorded fragments and shared output root.',tuple(steps))
    calls=Counter();original=VerificationContext.verify
    def verify(self,kind,function,args,kwargs):
        def independent(*a,**kw):
            calls[kind]+=1
            return function(*a,**kw)
        return original(self,kind,independent,args,kwargs)
    monkeypatch.setattr(VerificationContext,'verify',verify)
    run=AgentRuntime(planner=Planner(plan),run_store=store).run(AgentRequest('fallback-request','Tiny independent dependency DAG.',{}))
    assert run.status.value=='SUCCEEDED',run.to_dict()['errors']
    assert calls=={kind:1,'generic_fragments':1,'qc':1,'selection':1,'matrix':1}
    assert run.steps[0].verification.artifact_authority['upstream']['fragments']['authority_sha256'] is None
    with accepted_authorities(store,run.run_id):pass


def test_json_byte_mutation_without_requested_digest_cannot_inherit_authority(completed):
    kind,_,run,store=completed
    path=Path(run.steps[0].result['manifest_path'])
    with accepted_authorities(store,run.run_id) as context:
        path.write_bytes(path.read_bytes()+b'\n')
        with pytest.raises(AuthorityError):context.describe(kind,path,None)
        with pytest.raises(AuthorityError):context.describe('generic_fragments',path,None)
