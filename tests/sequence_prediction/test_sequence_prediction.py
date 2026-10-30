"""Simulated model outputs test contracts; real-weight parity is opt-in evidence."""
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path

import numpy as np
import pytest

from agent.tools.analysis import sequence_prediction as owner
from agent.orchestration import AgentPlan, PlanStep, AgentRequest, build_default_tool_registry
from agent.application import ResearchAgentApplication


def source(tmp_path, text):
    path = tmp_path / 'input.fasta'; path.write_text(text)
    return dict(sequence_path=str(path), sequence_sha256=owner.file_closure([path])[0]['sha256'])


@pytest.fixture
def native(tmp_path, monkeypatch):
    resource = dict(conditions=[f'biosample-{i}' for i in range(64)],release='simulated',family='DHS64',
        measurements=owner.MEASUREMENTS,context_bases=500,padding='left',training_assembly='hg38',
        training_domain='simulated',license='simulated',source='simulated',checkpoints=[],software={},metadata=dict(sha256='0'*64))
    monkeypatch.setattr(owner,'resolve_model',lambda:(resource,dict(resource=resource,files=[])))
    calls = []
    def predict(resource, rows, stage):
        calls.append(len(rows))
        a = np.zeros((3,2,len(rows),64),dtype=np.float32)
        a[:,0] = -1.5; a[:,1] = .25
        a[1,0] += .5; a[2,0] -= .5
        return a,dict(completion='succeeded',settings=dict(device='cpu'))
    monkeypatch.setattr(owner,'native_predict',predict)
    return calls


def test_wrapping_case_repeated_labels_and_ordinals(tmp_path):
    args = source(tmp_path, '>repeat\nacgt\nACGT\n>repeat\nTGCA\n')
    rows, pins, ref = owner.records(args)
    assert [r['original_header'] for r in rows] == ['repeat','repeat']
    assert [r['sequence_id'] for r in rows] == ['sequence-1','sequence-2']
    assert rows[0]['original_sequence'] == 'acgtACGT'
    assert rows[0]['sequence'] == 'ACGTACGT' and rows[0]['native_zero_padding'] == 492
    assert ref is None and pins[0]['sha256'] == args['sequence_sha256']


@pytest.mark.parametrize('seq',['', 'A'*501, 'ACNT', 'ACRT', 'ACUT', 'AC-G', 'AC GT'])
def test_no_truncation_or_unknown_zero_encoding(seq):
    with pytest.raises(ValueError): owner.records(dict(sequence=seq))


def test_source_integrity_and_exclusive_inline(tmp_path):
    args = source(tmp_path,'>one\nACGT\n')
    with pytest.raises(ValueError): owner.records(dict(args,sequence='ACGT'))
    Path(args['sequence_path']).write_text('>one\nTGCA\n')
    with pytest.raises(ValueError,match='changed'): owner.records(args)


def test_utterance_bases_bound_verbatim():
    from agent.application.turn_decisions import ScopedArgument, admit_scientific_argument, IntentError
    seq = 'acgt' * 125
    text = f'Predict this sequence: {seq}.'
    start = text.index(seq)
    spec = build_default_tool_registry().get(owner.TOOL).optional_arguments['sequence']
    d = ScopedArgument(owner.TOOL,'sequence',seq,start,start+len(seq))
    assert admit_scientific_argument(d,text,spec) == seq
    with pytest.raises(IntentError):
        admit_scientific_argument(ScopedArgument(owner.TOOL,'sequence',seq.upper(),start,start+len(seq)),text,spec)


def test_publication_report_recovery_no_native_replay(tmp_path,native,monkeypatch):
    args = dict(source(tmp_path,'>same\nACGT\n>same\nTGCA\n'),output_dir=str(tmp_path/'prediction'))
    class Plan:
        def plan(self,request,registry):
            return AgentPlan('test',request.request_id,'Explicit simulated test',(
                PlanStep('prediction',owner.TOOL,args),))
    app = ResearchAgentApplication(tmp_path/'workspace',planner=Plan())
    done = app.run(AgentRequest('sequence-test','Predict DNA',{}))
    assert done.status.value == 'SUCCEEDED',done.error
    assert native == [2]
    r = done.run_result.steps[0].result
    assert r['score_min'] == -1.5 and r['n_conditions'] == 64
    monkeypatch.setattr(owner,'native_predict',lambda *a:pytest.fail('native replay'))
    assert app.resume(done.run_id).status.value == 'SUCCEEDED'
    from agent.orchestration.prior_outputs import bind_output, validate_binding
    binding = bind_output(app.run_store,app.registry,done.run_id,'prediction','sequences')
    validate_binding(binding,store=app.run_store,registry=app.registry,integrity=True)
    assert binding.semantic_type == 'dna_sequences.v1'
    Path(r['scores_path']).write_text('corrupt')
    with pytest.raises(ValueError): owner.verify_result(args,r)


def test_accepted_sequence_continuation_retains_independent_prediction(tmp_path,native):
    from agent.application import OutputSelection
    from agent.orchestration.prior_outputs import bind_output
    from agent.schemas.prior_output import PriorOutputRef
    arguments=dict(sequence='ACGT',output_dir=str(tmp_path/'first'))
    class Plan:
        def plan(self, request, registry):
            return AgentPlan('p',request.request_id,'Explicit simulated continuation',(
                PlanStep('prediction',owner.TOOL,arguments),))
    app=ResearchAgentApplication(tmp_path/'workspace',planner=Plan())
    app.sessions.create('sequences')
    outputs=(OutputSelection('DNA','prediction','sequences_path'),
             OutputSelection('scores','prediction','manifest_path'))
    first=app.sessions.run('sequences','first',AgentRequest('first','Predict DNA',{}),outputs,expected_generation=0)
    binding=bind_output(app.run_store,app.registry,first.revisions[-1].run_id,'prediction','sequences')
    arguments=dict(
        sequence_path=PriorOutputRef(binding,'sequences_path'),sequence_sha256=PriorOutputRef(binding,'sequences_sha256'),
        head='binary',output_dir=str(tmp_path/'second'))
    second=app.sessions.run('sequences','second',AgentRequest('second','Reuse accepted DNA',{}),
        (OutputSelection('new scores','prediction','manifest_path'),),expected_generation=1,retain=('DNA',),
        use_active_context=True)
    assert second.turn('second').status == 'activated',app.run_store.load('second:run').errors
    assert native == [1,1]
    assert {o.name for o in second.revisions[-1].outputs} == {'new scores','DNA'}


@pytest.mark.parametrize('head,expected',[('continuous',-1.5),('binary',.25)])
def test_exact_condition_head_and_download(tmp_path,native,head,expected):
    r = owner.execute(dict(sequence='ACGT',condition='biosample-57',head=head,output_dir=str(tmp_path/'out')))
    assert r['conditions'] == ['biosample-57'] and r['score_min'] == expected
    assert r['score_examples'][0]['condition_index'] == 57
    from agent.application.sequence_delivery import prepare
    for name in ('sequences.fasta','scores.tsv','metadata.json','scores.png'):
        raw = (tmp_path/'out'/name).read_bytes()
        data = prepare(io.BytesIO(raw),name)
        assert data
    with pytest.raises(ValueError,match='exact DHS64'):
        owner.execute(dict(sequence='ACGT',condition='invented neuron',output_dir=str(tmp_path/'wrong')))
    assert len(native) == 1


def test_evaluation_correspondence_and_scale(tmp_path,native,monkeypatch):
    r = owner.execute(dict(sequence='ACGT',condition='biosample-0',output_dir=str(tmp_path/'out')))
    fields = ('sequence_id','sequence_sha256','condition','measurement','value','provenance')
    row = {k:r['score_examples'][0][k] for k in fields[:4]}
    row.update(value='-1.0',provenance='explicitly simulated observation, not experimental truth')
    path = tmp_path/'observations.tsv'
    def save():
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerow(row)
    save()
    def args(name):return dict(prediction_manifest_path=r['manifest_path'],prediction_manifest_sha256=r['manifest_sha256'],
        observations_path=str(path),observations_sha256=owner.file_closure([path])[0]['sha256'],output_dir=str(tmp_path/name))
    monkeypatch.setattr(owner,'native_predict',lambda *a:pytest.fail('evaluation repeated inference'))
    e=owner.execute_evaluation(args('eval'))
    assert e['metrics'][0]['rmse'] == .5 and e['metrics'][0]['pearson'] is None
    for incompatible in ('MPRA', 'ATAC', 'peak_counts'):
        row['measurement']=incompatible;save()
        with pytest.raises(ValueError,match='compatible DNase'):owner.execute_evaluation(args('bad'))
    row['measurement']=r['measurement'];row['sequence_id']='unmatched';save()
    with pytest.raises(ValueError,match='unique exact'):owner.execute_evaluation(args('bad2'))


def test_bed_exact_extraction_strand_and_bounds(tmp_path,monkeypatch):
    import pysam
    fasta=tmp_path/'genome.fa';fasta.write_text('>chr1\nAACCGGTTACGT\n');pysam.faidx(str(fasta))
    genome=dict(assembly='test',species=dict(taxonomy_id=9606),genome=dict(fasta=dict(path=str(fasta)),fai=dict(path=str(fasta)+'.fai')))
    entry=dict(kind='genome',resource=genome,files=owner.file_closure([fasta,str(fasta)+'.fai']))
    class Library:
        def entries(self):return (entry,)
        def check(self,e):assert e == entry;return genome
    monkeypatch.setattr(owner,'library',Library)
    args=source(tmp_path,'chr1\t0\t5\tregion\t0\t-\n')
    rows,_,ref=owner.records(dict(args,assembly='test'))
    assert rows[0]['sequence']=='CGGTT' and rows[0]['native_zero_padding']==495
    with pytest.raises(ValueError,match='explicit genome'):owner.records(args)
    with pytest.raises(ValueError,match='unique qualified'):owner.records(dict(args,assembly='other'))
    Path(args['sequence_path']).write_text('chr1\t0\t20\n')
    args['sequence_sha256']=owner.file_closure([args['sequence_path']])[0]['sha256']
    with pytest.raises(ValueError,match='out-of-bounds'):owner.records(dict(args,assembly='test'))


def test_private_labels_only_redacted_at_delivery(tmp_path,native):
    r=owner.execute(dict(source(tmp_path,'>/private/research/sample\nACGT\n'),output_dir=str(tmp_path/'out')))
    from agent.application.sequence_delivery import prepare
    for name in ('scores.tsv','sequences.fasta'):
        raw=(tmp_path/'out'/name).read_bytes(); assert b'/private/research' in raw
        data=prepare(io.BytesIO(raw),name);assert b'/private/research' not in data
    assert owner.read(tmp_path/'out/sequence-records.json')[0]['original_header']=='/private/research/sample'


def test_interruption_does_not_grant_retry(tmp_path,native,monkeypatch):
    from agent.tools._cancellation import cancellation_scope,ToolWorkCancelled
    args=dict(sequence='ACGT',output_dir=str(tmp_path/'out'))
    with cancellation_scope(lambda:True),pytest.raises(ToolWorkCancelled):owner.execute(args)
    assert not (tmp_path/'out/manifest.json').exists()
    with pytest.raises((ValueError,OSError)):owner.recover(args,{'not':'a saved execution'})


def test_binary_metrics_require_exact_labels_and_measurement(tmp_path,native):
    r=owner.execute(dict(source(tmp_path,'>a\nACGT\n>b\nTGCA\n'),head='binary',
        condition='biosample-0',output_dir=str(tmp_path/'prediction')))
    truth=tmp_path/'truth.tsv'
    truth.write_text('sequence_id\tcondition\tmeasurement\tvalue\tprovenance\n'
        'sequence-1\tbiosample-0\tDNase_peak_call\t0\tsimulated\n'
        'sequence-2\tbiosample-0\tDNase_peak_call\t1\tsimulated\n')
    args=dict(prediction_manifest_path=r['manifest_path'],prediction_manifest_sha256=r['manifest_sha256'],
        observations_path=str(truth),observations_sha256=owner.file_closure([truth])[0]['sha256'],
        output_dir=str(tmp_path/'evaluation'))
    metrics=owner.execute_evaluation(args)['metrics'][0]
    assert metrics['auroc'] == .5 and metrics['average_precision'] == .5
    assert metrics['brier'] == .3125
    assert native == [2]


def test_model_resource_change_during_execution_is_not_published(tmp_path,native,monkeypatch):
    resource,entry=owner.resolve_model()
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'simulated checkpoint')
    entry=dict(entry,files=owner.file_closure([checkpoint]))
    monkeypatch.setattr(owner,'resolve_model',lambda:(resource,entry))
    original=owner.native_predict
    def changed(*args):
        value=original(*args);checkpoint.write_bytes(b'changed checkpoint');return value
    monkeypatch.setattr(owner,'native_predict',changed)
    with pytest.raises(ValueError,match='changed during prediction'):
        owner.execute(dict(sequence='ACGT',output_dir=str(tmp_path/'out')))
    assert not (tmp_path/'out/manifest.json').exists()


def test_unavailable_condition_message_uses_captured_choices():
    from agent.application.turn_decisions import parse_decision,Clarify
    from agent.application import responses
    # Model selects the reason; exact displayed labels come from resource discovery.
    decision=parse_decision(json.dumps(dict(turn_schema_version=1,
        decision=dict(kind='clarify',reason='unavailable_model_condition'))))
    assert isinstance(decision,Clarify)
    text=responses.clarification_text(Clarify(decision.reason,('K562','CD8 T cells')))
    assert 'No substitute' in text and 'K562; CD8 T cells' in text
