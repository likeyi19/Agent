"""Table bindings, native safety boundaries, durable acceptance and delivery."""
import hashlib
import io
from pathlib import Path

import pytest

from agent.tools.analysis import native_tables as owner


@pytest.fixture
def source(tmp_path):
    p = tmp_path / 'signal.data'; p.write_text('chr1\t0\t100\t2\nchr1\t100\t200\t3\n')
    return dict(input_path=str(p), output_dir=str(tmp_path / 'out'), function_name='bdgopt',
        kwargs={'method':'multiply', 'extra-param':2}, signal_semantics='coverage')


@pytest.mark.parametrize('text', ['chr1\t0\t100\t1\nchr1\t120\t200\t2\n',
    'chr1\t-1\t100\t2\n', 'chr1\t0\t2147483648\t2\n',
    'chr1\t0\t100\t2\textra\n'])
def test_bedgraph_does_not_silently_reinterpret_invalid_native_storage(source, text):
    Path(source['input_path']).write_text(text)
    with pytest.raises(ValueError):
        owner.bedgraph(source['input_path'])


def test_ordinary_signed_continuous_bedgraph_without_h5ad(source):
    Path(source['input_path']).write_text('chr1\t10\t100\t-2\nchr1\t100\t200\t3\n')
    assert owner.bedgraph(source['input_path']) == 2


@pytest.mark.parametrize('kwargs', [{'ofile':'/tmp/escape'}, {'outdir':'/tmp'}, {'ifile':'/etc/passwd'}, {'method':['max','FE']}, {'method':{'code':'eval'}}])
def test_native_files_and_unrepresented_multiple_outputs_not_kwargs(source, kwargs):
    with pytest.raises(ValueError):
        owner.MACS3.native_arguments(dict(source, kwargs=kwargs))


def test_statistical_scores_do_not_require_metadata_admission(source):
    args = dict(source, kwargs={})
    assert owner.MACS3.parameter_prerequisites(args) == ()
    assert owner.MACS3.native_arguments(args) == {}
    args.pop('signal_semantics')
    assert owner.MACS3.output_semantics(args) == ('bedGraph', 'MACS3_p2q_of_unspecified_scores')
    assert owner.MACS3.parameter_prerequisites(dict(args, function_name='bdgcmp')) == ('control_path',)
    assert owner.MACS3.parameter_prerequisites(dict(args, function_name='bdgcmp', control_path=source['input_path'])) == ()


def test_all_new_ports_offered_and_bind_exact_private_attachments(source, tmp_path):
    from agent.orchestration import build_default_tool_registry
    from agent.application.local_resources import LocalResourceAdmission, group_choices
    r = build_default_tool_registry(); local = LocalResourceAdmission(tmp_path / 'ws', approved_source_roots=(tmp_path,), registry=r)
    item = local.register('signal', source['input_path'], label='signal', attribution='fixture', input_type='attachment')
    group = local.compose_group([item.resource_id], attachment_candidates=True)
    roles = {(p['tool'], p['port']) for p in group_choices(group, r)['roles']}
    assert {('run_macs3','dataset'), ('run_macs3','control'), ('run_cicero','dataset')} <= roles
    for tool in ('run_macs3','run_cicero'):
        bound = local.bind_group(group, [dict(source='source.0', tool=tool, port='dataset')], verify_source=True)
        assert bound.execution_inputs['input_path'] == source['input_path']


def test_headerless_first_row_is_privacy_projected():
    from agent.application.native_delivery import prepare
    data = prepare(io.BytesIO(b'/private/chrom\t0\t10\t-2\nchr2\t0\t10\t3\n'), 'result.tsv')
    assert b'/private/' not in data and b'\t-2' in data


def test_table_acceptance_reuse_delivery_restart_and_tamper(source, tmp_path, monkeypatch):
    from agent.application import ResearchAgentApplication, OutputSelection
    from agent.orchestration import AgentPlan, AgentRequest, PlanStep
    from agent.application.workspace import ManagedWorkspace
    from agent.orchestration.prior_outputs import bind_output
    from agent.schemas.prior_output import PriorOutputRef
    from agent.application.native_delivery import accepted_artifacts
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    calls = []
    def native(self, args, stage):
        calls.append(args)
        (stage / 'result.tsv').write_text('chr1\t0\t100\t4\nchr1\t100\t200\t6\n')
        return dict(completion='succeeded', versions={'MACS3':'SIMULATED'}, implementation=[])
    monkeypatch.setattr(owner.TablePackage, 'native', native)
    root = tmp_path / 'workspace'
    source['output_dir'] = str(root / 'runs' / ManagedWorkspace.run_digest('first:run') / 'scientific')
    class Planner:
        selected = AgentPlan('first', 'first', 'Simulated native table', (PlanStep('native', owner.MACS3.TOOL, source),))
        def plan(self, request, registry): return self.selected
    planner = Planner(); app = ResearchAgentApplication(root, planner=planner); app.sessions.create('reuse')
    state = app.sessions.run('reuse', 'first', AgentRequest('first','Multiply',{}),
        (OutputSelection('table','native','table_path'),), expected_generation=0)
    assert state.turn('first').status == 'activated', app.run_store.load('first:run').errors
    binding = bind_output(app.run_store, app.registry, 'first:run','native','table')
    assert binding.semantic_type == 'native_genomic_table.v1'
    Path(source['input_path']).unlink()
    follow = dict(input_path=PriorOutputRef(binding,'table_path'),function_name='bdgopt',kwargs={'method':'add','extra-param':1},
        signal_semantics='scaled_coverage',output_dir=str(root/'runs'/ManagedWorkspace.run_digest('second:run')/'scientific'))
    planner.selected = AgentPlan('second','second','Reuse accepted native signal',(PlanStep('native',owner.MACS3.TOOL,follow),))
    state = app.sessions.run('reuse','second',AgentRequest('second','Add to accepted signal',{}),
        (OutputSelection('table','native','table_path'),),expected_generation=1)
    assert state.turn('second').status == 'activated', app.run_store.load('second:run').errors
    reopened = ResearchAgentApplication(root, planner=planner)
    assert reopened.sessions.load('reuse').to_dict() == state.to_dict() and len(calls) == 2
    artifacts = accepted_artifacts(reopened.sessions,'reuse',state.revisions[-1],state.turn('second'),None,step_id='native')
    assert len(artifacts) == 3
    for artifact in artifacts:
        prepared = MatrixDownloadPreparer().prepare(artifact)
        try:
            data = b''.join(prepared.iter_chunks())
            assert hashlib.sha256(data).hexdigest() == prepared.sha256
            assert len(data) == prepared.size_bytes and prepared.source_sha256 == artifact.sha256
        finally: prepared.close()
    table = next(a for a in artifacts if a.role == 'result.tsv'); table.path.write_text('tampered')
    with pytest.raises((ValueError, RuntimeError)):
        MatrixDownloadPreparer().prepare(table)


def test_native_interleaved_chromosomes_and_integer_spelling_preserve_values(source):
    Path(source['input_path']).write_text('chr1\t00\t100\t1\nchr2\t0\t100\t1\nchr1\t100\t200\t1\n')
    assert owner.bedgraph(source['input_path']) == 3


@pytest.mark.parametrize('score',['nan','inf','-inf'])
def test_native_nonfinite_scores_remain_native_values(source,score):
    Path(source['input_path']).write_text('chr1\t0\t100\t'+score+'\n')
    assert owner.bedgraph(source['input_path']) == 1
