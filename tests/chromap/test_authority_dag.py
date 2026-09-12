from pathlib import Path
import hashlib
import subprocess
from collections import Counter
import pytest

from agent.tools.data.authority_context import VerificationContext, authority_operation
from agent.tools.data import scatac_fragments as public
from test_fragments_contracts import bound, fake_runtime
from test_fragments_orchestration import configured

REAL_RUN = subprocess.run


@pytest.fixture
def dag(configured, tiny, monkeypatch):
    from agent.tools.data import scatac_qc_reference as qr
    from agent.orchestration import AgentPlan, PlanStep, StepOutputRef
    args, control, group, *_ = configured
    listing = subprocess.run
    monkeypatch.setattr(subprocess, 'run', lambda argv, **kw:
        listing(argv, **kw) if len(argv)>1 and argv[1]=='-l' else REAL_RUN(argv, **kw))
    monkeypatch.setenv('AGENT_QC_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC', '1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG', raising=False)
    annotation = tiny['root'] / 'annotation.tsv'
    annotation.write_text('chrTiny\t150\t250\t+\tg1\tt1\tprotein_coding\n'
                          'chrTiny\t2104\t2204\t+\tg2\tt2\tprotein_coding\n')
    qr.build_scatac_qc_reference_bundle(parent_manifest_path=tiny['pointer']['manifest_path'],
        parent_manifest_sha256=tiny['pointer']['manifest_sha256'], annotation_path=annotation,
        annotation_source='synthetic', annotation_release='1',
        classifications=(qr.QCContig('chrTiny',12000,'primary_nuclear_qc'),),
        classification_source='explicit-test', output_dir=tiny['root']/'qc-reference')
    qp = tiny['root'] / 'qc-reference/manifest.json'
    pair = lambda step, prefix: {prefix+'path':StepOutputRef(step,'manifest_path'),
                                 prefix+'sha256':StepOutputRef(step,'manifest_sha256')}
    steps = [PlanStep('fragments','prepare_scATAC_fragments',args),
        PlanStep('qc','compute_scATAC_qc',pair('fragments','fragments_manifest_') |
            dict(qc_reference_manifest_path=str(qp),qc_reference_manifest_sha256=hashlib.sha256(qp.read_bytes()).hexdigest(),
                 output_dir=str(tiny['root']/'qc')),('fragments',)),
        PlanStep('selection','select_scATAC_cells',pair('qc','barcode_qc_manifest_') |
            dict(min_qc_fragment_records=0,min_tss_enrichment='0',output_dir=str(tiny['root']/'selection')),('qc',)),
        PlanStep('matrix','build_scATAC_cell_by_ccre',pair('fragments','fragments_manifest_') |
            pair('selection','selected_cells_manifest_') |
            dict(reference_manifest_path=tiny['pointer']['manifest_path'],reference_manifest_sha256=tiny['pointer']['manifest_sha256'],
                 output_dir=str(tiny['root']/'matrix')),('fragments','selection'))]
    return AgentPlan('authority-plan','authority-request','Tiny scientific DAG.',tuple(steps)), group, control


def test_durable_dag_scans_owned_science_and_no_downstream_raw_reads(dag, tiny, monkeypatch):
    from agent.orchestration import AgentRuntime, AgentRequest, FileRunStore
    from test_fragments_orchestration import FixedPlanner
    from agent.tools.data.authority_context import VerificationContext
    from agent.tools.data import _fragments_fastq as raw, fastq_fragments as producer, _chromap
    plan, group, control = dag
    raw_paths = {str(p) for _,p in group.files}
    counts = Counter()
    original_scan = raw.scan_group
    def scan(*a, **kw):
        counts['structural_scans'] += 1
        return original_scan(*a, **kw)
    monkeypatch.setattr(raw,'scan_group',scan);monkeypatch.setattr(producer,'scan_group',scan)
    original_hash = _chromap.sha256
    def hashed(path):
        if str(path) in raw_paths: counts['full_source_hashes'] += 1
        return original_hash(path)
    monkeypatch.setattr(_chromap,'sha256',hashed)
    original_verify = VerificationContext.verify
    phase = ['producer']
    def verified(self, kind, function, args, kwargs):
        def independent(*a, **kw):
            counts[kind] += 1
            return function(*a, **kw)
        return original_verify(self,kind,independent,args,kwargs)
    monkeypatch.setattr(VerificationContext,'verify',verified)
    from agent.tools.data import _barcode_qc_binding as binding
    original_bind = binding.bind
    def bind(*a, **kw):
        phase[0] = 'downstream'
        return original_bind(*a, **kw)
    monkeypatch.setattr(binding, 'bind', bind)
    from agent.tools.data import _fragments_binding as producer_binding
    original_intake = producer_binding.fresh_intake
    def inspected(*a, **kw):
        assert phase[0] == 'producer'
        counts['bounded_intake_inspections'] += 1
        return original_intake(*a, **kw)
    monkeypatch.setattr(producer_binding, 'fresh_intake', inspected)
    original_open = Path.open
    def opened(path,*a,**kw):
        if str(path) in raw_paths and phase[0]=='downstream':
            pytest.fail('Downstream opened a historical raw source')
        return original_open(path,*a,**kw)
    monkeypatch.setattr(Path,'open',opened)
    store = FileRunStore(tiny['root']/'state')
    run = AgentRuntime(planner=FixedPlanner(plan),run_store=store).run(AgentRequest('authority-request','Tiny DAG.',{}))
    assert run.status.value == 'SUCCEEDED', run.to_dict()['errors']
    assert counts['structural_scans'] == 2
    assert counts['fastq_fragment_production'] == counts['qc'] == counts['selection'] == counts['matrix'] == 1
    assert run.steps[-1].result['total_count'] == 1
    assert all(s.verification.artifact_authority['schema_version']==2 for s in run.steps)
    assert control['calls'].count('chromap') == 1
    print('FASTQ source access:',dict(counts))
    from agent.orchestration.verification_authority import accepted_authorities
    from agent.tools.data.cell_by_ccre_verifier import verify_cell_by_ccre
    before = dict(counts)
    with accepted_authorities(store,run.run_id):
        verify_cell_by_ccre(run.steps[-1].result['manifest_path'],
            expected_sha256=run.steps[-1].result['manifest_sha256'],bedtools_path='/usr/bin/bedtools')
    assert dict(counts)==before
    from agent.orchestration.verification_authority import load_fragment_authority
    from agent.schemas.verification_authority import AuthorityError
    handle = load_fragment_authority(store,run.run_id,'fragments')
    for kind in ('bam_fragment_production','external_fragment_adoption'):
        with pytest.raises(AuthorityError):
            handle.validate(run.steps[0].result['manifest_path'],run.steps[0].result['manifest_sha256'],producer_kind=kind)
    from dataclasses import replace
    from agent.tools.data.barcode_qc_verifier import verify_barcode_qc
    state = store.load(run.run_id)
    class ProducerOnlyStore:
        def load(self,_):
            return replace(state,steps=(state.steps[0], *(
                replace(step,verification=replace(step.verification,artifact_authority=None))
                for step in state.steps[1:])))
    with accepted_authorities(ProducerOnlyStore(),run.run_id):
        verify_barcode_qc(run.steps[1].result['manifest_path'],expected_sha256=run.steps[1].result['manifest_sha256'])
    assert counts == Counter(before) + Counter(qc=1)
    # Exact published bytes remain bound even when no digest is requested.
    fragment_path = Path(run.steps[0].result['manifest_path'])
    with accepted_authorities(store,run.run_id) as context:
        fragment_path.write_bytes(fragment_path.read_bytes()+b'\n')
        with pytest.raises(AuthorityError):
            context.describe('fastq_fragment_production',fragment_path,None)


def test_private_publication_proof_preserves_source_scan_count(configured, monkeypatch):
    args, *_ = configured
    from agent.tools.data import _fragments_fastq as raw, fastq_fragments as producer
    calls = []
    original = raw.scan_group
    def scan(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(raw, 'scan_group', scan)
    monkeypatch.setattr(producer, 'scan_group', scan)
    with authority_operation(VerificationContext()):
        result = public.execute_fragments(args, '1' * 64)
        public.verify_public_result(args, result)
    assert len(calls) == 2


def test_fastq_application_reuses_all_scientific_authorities(dag, tiny, monkeypatch):
    from agent.application import ResearchAgentApplication
    from agent.orchestration import AgentRequest
    from test_fragments_orchestration import FixedPlanner
    plan, group, control = dag
    app = ResearchAgentApplication(tiny['root']/'application', planner=FixedPlanner(plan))
    counts = Counter()
    original = VerificationContext.verify
    def verify(self, kind, function, args, kwargs):
        def independent(*a, **kw):
            counts[kind] += 1
            return function(*a, **kw)
        return original(self, kind, independent, args, kwargs)
    monkeypatch.setattr(VerificationContext, 'verify', verify)
    result = app.run(AgentRequest('authority-request', 'Tiny FASTQ DAG.', {}))
    assert result.status.value == 'SUCCEEDED', result
    assert result.evidence and result.report and result.visualization is None
    assert result.run_result.steps[-1].result['total_count'] == 1
    assert counts == dict(fastq_fragment_production=1, generic_fragments=1, qc=1, selection=1, matrix=1)
    before = dict(counts)
    state = app.run_store.load(result.run_id)
    for _, name in group.files:
        path = Path(name)
        path.rename(path.with_suffix('.archived'))
    assert app.resume(result.run_id) == result
    assert dict(counts) == before
    assert app.run_store.load(result.run_id) == state
    assert control['calls'].count('chromap') == 1


def test_explicit_legacy_fastq_qualification(dag, tiny, monkeypatch):
    from dataclasses import replace
    from agent.orchestration import AgentRuntime, AgentRequest, FileRunStore
    from agent.orchestration.verification_authority import qualify_legacy_authorities, accepted_authorities
    from test_fragments_orchestration import FixedPlanner
    from agent.tools.data import fastq_fragments as production
    from agent.tools.data.scatac_matrix import verify_public_result
    plan, group, control = dag
    class LegacyStore(FileRunStore):
        def update(self, state, *, expected_revision):
            steps = tuple(replace(s, verification=replace(s.verification, artifact_authority=None))
                          if s.verification else s for s in state.steps)
            return super().update(replace(state, steps=steps), expected_revision=expected_revision)
    store = LegacyStore(tiny['root']/'legacy-state')
    run = AgentRuntime(planner=FixedPlanner(plan), run_store=store).run(
        AgentRequest('authority-request', 'Legacy FASTQ DAG.', {}))
    assert run.status.value == 'SUCCEEDED', run
    state = store.load(run.run_id)
    assert all(s.verification.artifact_authority is None for s in state.steps)
    assert state.steps[-1].result['total_count'] == 1
    before = store.state_path(run.run_id).read_bytes()
    counts = Counter()
    original = VerificationContext.verify
    def verify(self, kind, function, args, kwargs):
        def deep(*a, **kw):
            counts[kind] += 1
            return function(*a, **kw)
        return original(self, kind, deep, args, kwargs)
    monkeypatch.setattr(VerificationContext, 'verify', verify)
    monkeypatch.setattr(production, 'prepare_fastq_fragments',
        lambda *a, **kw: pytest.fail('FASTQ production during qualification'))
    authorities = qualify_legacy_authorities(store, run.run_id)
    assert counts == dict(fastq_fragment_production=1, generic_fragments=1, qc=1, selection=1, matrix=1)
    assert authorities['fragments'].record['producer_qualification']['scope'] == 'fastq_source_and_producer_record.v1'
    assert store.state_path(run.run_id).read_bytes() == before
    previous = dict(counts)
    for _, name in group.files:
        path = Path(name)
        path.rename(path.with_suffix('.archived'))
    assert qualify_legacy_authorities(store, run.run_id) == authorities
    with accepted_authorities(store, run.run_id):
        verify_public_result(state.steps[-1].resolved_arguments, state.steps[-1].result)
    assert dict(counts) == previous
    assert control['calls'].count('chromap') == 1
