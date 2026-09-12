"""Tiny real parsers/reconstruction; only the existing external-tool fixture is fake."""
from dataclasses import replace
import hashlib
import gzip
import json
from pathlib import Path
import subprocess
import pytest

from agent.orchestration import AgentRequest, AgentRuntime, FileRunStore, RunStatus
from agent.orchestration.verification_authority import load_fragment_authority, StoredFragmentAuthority
from agent.schemas.verification_authority import AuthorityError, VerifiedArtifactAuthority, VerificationScope
from agent.tools.data import fastq_fragments_verifier as deep
from agent.tools.data import _barcode_qc_binding as binding
from agent.tools.data.fastq_verification_authority import require_compatible
from test_fragments_contracts import bound, fake_runtime
from test_fragments_orchestration import configured, FixedPlanner, plan_for

REAL_RUN = subprocess.run


@pytest.fixture(autouse=True)
def version_one_issuance(monkeypatch):
    # Preserve every accepted .1 corruption/freshness assertion against actual
    # v1 records. New automatic v2 issuance has separate scientific-DAG tests.
    from agent.tools.data import authority_context
    monkeypatch.setattr(authority_context, 'VerificationContext', lambda: None)


@pytest.fixture
def accepted(configured, tiny, monkeypatch):
    args, control, group, white, _ = configured
    store = FileRunStore(tiny['root'] / 'authority-store')
    calls = []
    original = deep.verify_fragments
    def counted(*a, **kw):
        calls.append('deep')
        return original(*a, **kw)
    monkeypatch.setattr(deep, 'verify_fragments', counted)
    runtime = AgentRuntime(planner=FixedPlanner(plan_for(args)), run_store=store)
    run = runtime.run(AgentRequest('fragments-request', 'prepare', {}))
    assert run.status is RunStatus.SUCCEEDED, run.errors
    return store, run, calls, control, group, white


def test_authority_issued_after_deep_and_persisted_reuse(accepted):
    store, run, calls, control, *_ = accepted
    step = run.steps[0]
    assert calls and control['calls'].count('chromap') == 1
    authority = step.verification.artifact_authority
    assert authority['scope'] == VerificationScope.SCIENTIFIC.value
    assert authority == store.load(run.run_id).steps[0].verification.artifact_authority
    # Reopen persistence, proving the provenance survives the process object.
    handle = load_fragment_authority(store, run.run_id, step.step_id)
    before = len(calls)
    path, sha = step.result['manifest_path'], step.result['manifest_sha256']
    assert handle.validate(path, sha).scope is VerificationScope.INTEGRITY
    binding.qualified_fragments(path, sha, fragments_authority=handle)
    binding.qualified_fragments(path, sha, fragments_authority=handle)
    assert len(calls) == before
    binding.qualified_fragments(path, sha)
    assert len(calls) == before + 1
    with pytest.raises(AuthorityError):
        binding.qualified_fragments(path, sha, fragments_authority=authority)
    with pytest.raises(AuthorityError):
        StoredFragmentAuthority(store, run.run_id, step.step_id, 'forged')


@pytest.mark.parametrize('target', ['manifest', 'payload', 'sidecar', 'receipt', 'profile',
                                  'upstream', 'reference', 'whitelist', 'raw', 'index', 'fasta', 'ccre'])
def test_mutations_fail_closed_without_deep_fallback(accepted, tiny, target):
    store, run, calls, _, group, white = accepted
    step = run.steps[0]
    handle = load_fragment_authority(store, run.run_id, step.step_id)
    authority = step.verification.artifact_authority
    manifest = Path(step.result['manifest_path'])
    value = json.loads(manifest.read_bytes())
    paths = dict(manifest=manifest,
        payload=manifest.parent / value['libraries'][0]['bgzf']['path'],
        sidecar=manifest.parent / value['libraries'][0]['tabix']['path'],
        receipt=manifest.parent.parent / 'receipt.json', profile=manifest.parent / 'profile.json',
        upstream=Path(authority['upstream']['context']['manifest_path']),
        reference=Path(authority['upstream']['reference']['manifest_path']),
        whitelist=Path(white.resource.path), raw=Path(group.files[0][1]),
        index=Path(authority['upstream']['index']['manifest_path']), fasta=tiny['fa'], ccre=tiny['bed'])
    path = paths[target]
    data = path.read_bytes()
    # Same-size replacement, including inode replacement, never stat-only trust.
    replacement = path.with_name(path.name + '.replacement')
    replacement.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    replacement.replace(path)
    before = len(calls)
    with pytest.raises(ValueError):
        binding.qualified_fragments(manifest, step.result['manifest_sha256'], fragments_authority=handle)
    assert len(calls) == before


def test_wrong_publication_and_coordinated_metadata_forgery(accepted):
    store, run, calls, *_ = accepted
    step = run.steps[0]
    handle = load_fragment_authority(store, run.run_id, step.step_id)
    path = Path(step.result['manifest_path'])
    with pytest.raises(AuthorityError):
        handle.validate(path.with_name('other.json'), step.result['manifest_sha256'])
    # Self-consistent local digests cannot replace the accepted persistence anchor.
    value = json.loads(path.read_bytes())
    value['libraries'][0]['sum_support'] += 1
    path.write_text(json.dumps(value))
    receipt_path = path.parent.parent / 'receipt.json'
    receipt = json.loads(receipt_path.read_bytes())
    receipt['manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(AuthorityError):
        handle.validate(path, receipt['manifest_sha256'])


@pytest.mark.parametrize('field,value', [
    ('verifier', {'id': '', 'compatibility_version': '1'}),
    ('verifier', {'id': 'agent.fastq-fragments-independent', 'compatibility_version': '2'}),
    ('scope', VerificationScope.INTEGRITY.value),
    ('scope', VerificationScope.PRESENTATION.value),
    ('producer_qualification', 'generic_v2_content_only'),
    ('science_profile', '0' * 64), ('artifact_contract', 'unknown.v1'),
    ('completion', 'interrupted'),
])
def test_incompatible_authorities_fail(accepted, field, value):
    _, run, *_ = accepted
    record = VerifiedArtifactAuthority(run.steps[0].verification.artifact_authority).to_dict()
    record[field] = value
    with pytest.raises(AuthorityError):
        require_compatible(VerifiedArtifactAuthority(record))


def test_legacy_and_changed_accepted_anchor_not_adopted(accepted, monkeypatch):
    store, run, *_ = accepted
    step = run.steps[0]
    state = store.load(run.run_id)
    handle = load_fragment_authority(store, run.run_id, step.step_id)
    class ChangedStore:
        def load(self, _):
            return replace(state, steps=(replace(step, verification=replace(
                step.verification, artifact_authority=None)),))
    with pytest.raises(AuthorityError, match='legacy'):
        load_fragment_authority(ChangedStore(), run.run_id, step.step_id)
    # A previously issued handle also checks its accepted anchor again.
    monkeypatch.setattr(store, 'load', ChangedStore().load)
    with pytest.raises(AuthorityError):
        handle.validate(step.result['manifest_path'], step.result['manifest_sha256'])


def test_interrupted_verification_never_records_authority(configured, tiny, monkeypatch):
    args, *_ = configured
    store = FileRunStore(tiny['root'] / 'interrupted-store')
    from agent.tools.data import fastq_verification_authority as capture
    original = capture.capture_publication
    def stop(*a, **kw):
        original(*a, **kw)
        raise KeyboardInterrupt('verification interrupted')
    monkeypatch.setattr(capture, 'capture_publication', stop)
    runtime = AgentRuntime(planner=FixedPlanner(plan_for(args)), run_store=store)
    with pytest.raises(KeyboardInterrupt):
        runtime.run(AgentRequest('fragments-request', 'prepare', {}))
    state = store.load('fragments-request:run')
    assert all(s.verification is None or s.verification.artifact_authority is None for s in state.steps)
    with pytest.raises(AuthorityError):
        load_fragment_authority(store, state.run_id, 'fragments')


def test_qc_reconstruction_still_occurs_with_authority(accepted, tiny, monkeypatch):
    from agent.tools.data import scatac_qc_reference as qr, scatac_barcode_qc as qc
    from agent.tools.data import barcode_qc_verifier as verifier
    from agent.tools.data import _barcode_qc_contract as contract
    store, run, calls, *_ = accepted
    listing = subprocess.run
    monkeypatch.setattr(subprocess, 'run', lambda argv, **kw:
        listing(argv, **kw) if len(argv) > 1 and argv[1] == '-l' else REAL_RUN(argv, **kw))
    monkeypatch.setenv('AGENT_QC_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC', '1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG', raising=False)
    annotation = tiny['root'] / 'annotation.tsv'
    annotation.write_text('chrTiny\t3000\t3500\t+\tg1\tt1\tprotein_coding\n')
    qr.build_scatac_qc_reference_bundle(parent_manifest_path=tiny['pointer']['manifest_path'],
        parent_manifest_sha256=tiny['pointer']['manifest_sha256'], annotation_path=annotation,
        annotation_source='synthetic', annotation_release='1',
        classifications=(qr.QCContig('chrTiny', 12000, 'primary_nuclear_qc'),),
        classification_source='explicit-test', output_dir=tiny['root'] / 'qc-reference')
    qpath = tiny['root'] / 'qc-reference' / 'manifest.json'
    step = run.steps[0]
    output = qc.compute_scATAC_qc(fragments_manifest_path=step.result['manifest_path'],
        fragments_manifest_sha256=step.result['manifest_sha256'], qc_reference_manifest_path=str(qpath),
        qc_reference_manifest_sha256=hashlib.sha256(qpath.read_bytes()).hexdigest(),
        output_dir=str(tiny['root'] / 'qc-output'))
    handle = load_fragment_authority(store, run.run_id, step.step_id)
    before = len(calls)
    verifier.verify_barcode_qc(output['manifest_path'], expected_sha256=output['manifest_sha256'],
                              fragments_authority=handle)
    assert len(calls) == before
    # A rehashed QC scientific forgery must still fail independent reconstruction.
    path = Path(output['manifest_path'])
    value = json.loads(path.read_bytes())
    table = path.parent / value['table']['path']
    lines = gzip.decompress(table.read_bytes()).splitlines(keepends=True)
    fields = lines[1].decode().rstrip('\n').split('\t')
    fields[2] = str(int(fields[2]) + 1)
    lines[1] = ('\t'.join(fields) + '\n').encode()
    table.write_bytes(gzip.compress(b''.join(lines), mtime=0))
    value['table'] = contract.resource(table)
    value['identity_sha256'] = contract.manifest_identity(value)
    path.write_bytes(contract.canonical(value))
    with pytest.raises(ValueError, match='QC_SCIENTIFIC_MISMATCH'):
        verifier.verify_barcode_qc(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  fragments_authority=handle)
    assert len(calls) == before


def test_all_producers_representable_but_never_interchangeable(accepted):
    from agent.tools.data.fragments_authority_contract import CONTRACTS
    _, run, *_ = accepted
    original = VerifiedArtifactAuthority(run.steps[0].verification.artifact_authority).to_dict()
    scopes = {c.qualification_scope for c in CONTRACTS.values()}
    assert len(scopes) == 3
    for producer, contract in CONTRACTS.items():
        # In-memory contract specimens, never claimed to be issued authorities.
        record = original | dict(producer_qualification=contract.qualification,
                                 science_profile=contract.profile_sha256, verifier=contract.verifier)
        specimen = VerifiedArtifactAuthority(record)
        contract.require(specimen)
        for other, consumer in CONTRACTS.items():
            if other != producer:
                with pytest.raises(AuthorityError):
                    consumer.require(specimen)
        for change in ({'producer_qualification': {}},
                       {'science_profile': '0' * 64},
                       {'scope': VerificationScope.INTEGRITY.value},
                       {'producer_qualification': contract.qualification | {'compatibility_version': '2'}}):
            with pytest.raises(AuthorityError):
                contract.require(VerifiedArtifactAuthority(record | change))


def test_consumer_rejects_other_producer_and_insufficient_upstream(accepted, monkeypatch):
    store, run, *_ = accepted
    step = run.steps[0]
    handle = load_fragment_authority(store, run.run_id, step.step_id)
    for kind in ('bam_fragment_production', 'external_fragment_adoption'):
        with pytest.raises(AuthorityError, match='another producer'):
            handle.validate(step.result['manifest_path'], step.result['manifest_sha256'], producer_kind=kind)
    from agent.tools.data.fastq_verification_authority import check_integrity
    original = VerifiedArtifactAuthority(step.verification.artifact_authority).to_dict()
    for field in ('manifest_sha256', 'accepted_scope'):
        changed = json.loads(json.dumps(original))
        changed['upstream']['intake'][field] = '0' * 64 if field == 'manifest_sha256' else 'existence_only'
        with pytest.raises(AuthorityError):
            check_integrity(VerifiedArtifactAuthority(changed), step.resolved_arguments, step.result,
                            original['execution_identity'])


def test_mutation_during_verification_cannot_issue_authority(configured, tiny, monkeypatch):
    args, *_ = configured
    from agent.tools.data import fastq_verification_authority as capture
    original = capture.check_integrity
    def mutate(authority, *a):
        path = Path(authority.record['publication_path']).parent / 'profile.json'
        # Restore bytes: cross-verification snapshots must still reject mutation.
        data = path.read_bytes()
        path.write_bytes(data + b'changed')
        path.write_bytes(data)
        original(authority, *a)
    monkeypatch.setattr(capture, 'check_integrity', mutate)
    store = FileRunStore(tiny['root'] / 'mutation-store')
    run = AgentRuntime(planner=FixedPlanner(plan_for(args)), run_store=store).run(
        AgentRequest('fragments-request', 'prepare', {}))
    assert run.status is RunStatus.FAILED
    assert run.steps[0].verification.artifact_authority is None
    with pytest.raises(AuthorityError):
        load_fragment_authority(store, run.run_id, 'fragments')


def test_new_durable_evidence_remains_fresh_and_does_not_consume_authority(accepted, tiny, monkeypatch):
    listing = subprocess.run
    monkeypatch.setattr(subprocess, 'run', lambda argv, **kw:
        listing(argv, **kw) if len(argv) > 1 and argv[1] == '-l' else REAL_RUN(argv, **kw))
    from agent.report import build_analysis_evidence, verify_analysis_evidence
    from agent.orchestration import build_default_tool_registry
    store, run, calls, control, *_ = accepted
    before = len(calls)
    registry = build_default_tool_registry()
    evidence = build_analysis_evidence(run, tiny['root'] / 'evidence', registry=registry)
    assert len(calls) == before + 2  # step verification + fact projection
    assert verify_analysis_evidence(run, evidence, registry=registry).passed
    assert len(calls) == before + 4
    assert control['calls'].count('chromap') == 1


def test_unknown_persisted_verifier_fails_before_reuse(accepted):
    store, run, *_ = accepted
    state = store.load(run.run_id)
    step = run.steps[0]
    record = VerifiedArtifactAuthority(step.verification.artifact_authority).to_dict()
    record['verifier']['compatibility_version'] = 'unreviewed'
    changed = replace(state, steps=(replace(step, verification=replace(step.verification,
                                                                      artifact_authority=record)),))
    class UnknownStore:
        def load(self, _): return changed
    with pytest.raises(AuthorityError):
        load_fragment_authority(UnknownStore(), run.run_id, step.step_id)
