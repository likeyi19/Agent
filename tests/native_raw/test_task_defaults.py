"""Task input-order adaptation and unchanged native scientific defaults.

All data are small simulated fixtures. Scientific operations execute installed
SnapATAC2; tests may inspect native results without adding production algorithms.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import pytest

from agent.tools.analysis.native_matrix import SNAPATAC2, NativeInvocationError
from agent.orchestration import AgentRequest, build_default_tool_registry
from agent.orchestration.native_parameters import input_name
from agent.orchestration.semantic_compiler import (
    SemanticPlanCandidate, SemanticPlanStep, SemanticRegisteredChoiceSource as Choice,
    SemanticRequestInputSource as Input, SemanticStepOutputSource as Output,
    build_semantic_compiler_contract, compile_semantic_plan,
)

ORDER = input_name('run_snapatac2', 'pp.import_fragments', 'fragment_order')


@pytest.fixture
def sample(tmp_path):
    rows = [(4990-i%11, 5200+i, barcode) for barcode, count in [('A',1100),('B',250),('C',50)] for i in range(count)]
    coordinate = tmp_path/'coordinates.fragments'
    coordinate.write_text(''.join(f'chr1\t{left}\t{right}\t{bc}\t1\n' for left,right,bc in sorted(rows)))
    barcode = tmp_path/'barcodes.fragments'
    barcode.write_text(''.join(f'chr1\t{left}\t{right}\t{bc}\t1\n' for left,right,bc in sorted(rows,key=lambda row:(row[2],row[0],row[1]))))
    sizes=tmp_path/'sizes';sizes.write_text('chr1\t10000\n')
    peaks=tmp_path/'peaks';peaks.write_text('chr1\t4970\t5010\nchr1\t5100\t5150\nchr1\t5200\t5400\n')
    gtf=tmp_path/'genes.gtf';gtf.write_text('chr1\tfixture\ttranscript\t5001\t6000\t.\t+\t.\tgene_id "G"; transcript_id "T"; gene_name "G"; transcript_name "T"; transcript_type "protein_coding";\n')
    return SimpleNamespace(root=tmp_path,source=coordinate,barcode=barcode,sizes=sizes,peaks=peaks,gtf=gtf)


@pytest.fixture
def native():
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION')!='1':pytest.skip('Explicit installed runtime qualification')
    assert os.environ.get('AGENT_SNAPATAC2_PYTHON')


def call(sample,label,function,source=None,**args):
    return SNAPATAC2.execute(dict(input_path=str(source or sample.source),output_dir=str(sample.root/label),
        function_name=function,**args),'task-default-test')


def task_plan(sample,kwargs=None):
    registry=build_default_tool_registry()
    values=dict(input_path=str(sample.source),chrom_sizes_path=str(sample.sizes),
        chrom_sizes_sha256=__import__('hashlib').sha256(sample.sizes.read_bytes()).hexdigest(),output_dir=str(sample.root/'plan'),**{ORDER:'unknown'})
    if kwargs is not None:values[input_name('run_snapatac2','pp.import_fragments','kwargs')]=kwargs
    request=AgentRequest('task','Analyze these fragments',values)
    candidate=SemanticPlanCandidate((
        SemanticPlanStep('import','run_snapatac2',(Choice('function_name','pp.import_fragments'),Input('dataset','input_path'),
            Input('chrom_sizes','chrom_sizes_path'))),
        SemanticPlanStep('qc','run_snapatac2',(Choice('function_name','metrics.frag_size_distr'),Output('dataset','import','dataset')))))
    return compile_semantic_plan(request,candidate,registry,build_semantic_compiler_contract(registry))


def test_existing_compiler_binds_task_order_to_import_only_without_user_kwargs(sample):
    plan=task_plan(sample)
    assert plan.steps[0].arguments['fragment_order']=='unknown'
    assert 'kwargs' not in plan.steps[0].arguments
    assert 'fragment_order' not in plan.steps[1].arguments and 'kwargs' not in plan.steps[1].arguments
    from agent.application.dialogue_execution import scientific_parameters
    option,=[p for p in scientific_parameters(build_default_tool_registry()) if p['argument']=='fragment_order']
    assert not option['required'] and 'never ask' in option['description']


def test_explicit_natural_source_order_uses_existing_choice_binding():
    from agent.application.turn_decisions import _ground_declaration,_scientific_arguments
    from agent.application.dialogue_execution import _bind_arguments
    text='These fragments are already grouped by barcode.'
    raw=dict(tool='run_snapatac2',argument='fragment_order',native_function='pp.import_fragments',
        value='barcode',literal='already grouped by barcode',evidence='already grouped by barcode')
    declaration=_scientific_arguments([_ground_declaration(raw,text)])
    values=_bind_arguments(build_default_tool_registry(),text,{},declaration)
    assert values=={ORDER:'barcode'}
    assert SNAPATAC2.native_arguments(dict(function_name='pp.import_fragments',
        input_path='/captured/source',chrom_sizes_path='/captured/sizes',fragment_order=values[ORDER]))=={'sorted_by_barcode':True}


def test_task_handoff_supplies_unknown_order_without_touching_user_kwargs(sample,monkeypatch):
    from agent.application.native_raw_handoff import reference_inputs
    from agent.application.raw_handoff import FragmentsDeclaration,BamDeclaration
    from agent.tools.data import _fragment_reference
    reference=SimpleNamespace(genome=SimpleNamespace(fai=SimpleNamespace(path=str(sample.sizes),sha256='a'*64)),
        ccre=SimpleNamespace(bed=SimpleNamespace(path=str(sample.peaks),sha256='b'*64)))
    monkeypatch.setattr(_fragment_reference,'load',lambda *a:reference)
    selected=SimpleNamespace(inputs=lambda:dict(reference_bundle_path='/captured/reference',reference_bundle_sha256='c'*64))
    external=reference_inputs(selected,FragmentsDeclaration(source='source.0',species='human',assembly='hg38'))
    assert external[ORDER]=='unknown' and not any('kwargs' in key for key in external)
    bam=reference_inputs(selected,BamDeclaration(source='source.0',species='human',assembly='hg38'))
    assert ORDER not in bam  # Native BAM extraction itself guarantees ordering.


def test_unused_reference_and_source_declarations_do_not_require_native_qc(sample):
    """Regression for the genuine import/matrix-only optional-input failure."""
    import hashlib
    from agent.application.local_resources import LocalResourceAdmission,ResourceAdmissionError
    from agent.orchestration import AgentPlan,PlanStep
    registry=build_default_tool_registry()
    def key(function,field):return input_name('run_snapatac2',function,field)
    values={ORDER:'unknown',key('pp.import_fragments','species'):'human',
        key('pp.import_fragments','assembly'):'hg38'}
    for fn,field,path in [('pp.import_fragments','chrom_sizes',sample.sizes),
                          ('pp.make_peak_matrix','peaks',sample.peaks),('metrics.tsse','gene_anno',sample.gtf)]:
        values[key(fn,field+'_path')]=str(path)
        values[key(fn,field+'_sha256')]=hashlib.sha256(path.read_bytes()).hexdigest()
    owner=LocalResourceAdmission(sample.root/'resources',approved_source_roots=(sample.root,))
    record=owner.register('source',sample.source,input_type='attachment',label='simulated fragments',attribution='test')
    binding=owner.compose_native(record.resource_id,values)
    with pytest.raises(ResourceAdmissionError) as invalid:
        owner.compose_native(record.resource_id,dict(values,output_dir='/unmanaged/output'))
    assert invalid.value.code=='LOCAL_RESOURCE_BINDING_INVALID'
    request=AgentRequest('task','Build a cCRE matrix from these human hg38 fragments',
        {**dict(binding.execution_inputs), 'output_dir':str(sample.root/'compiled')})
    candidate=SemanticPlanCandidate((
        SemanticPlanStep('import','run_snapatac2',(Choice('function_name','pp.import_fragments'),Input('dataset','input_path'),
            Input('input_sha256','input_sha256'))),
        SemanticPlanStep('matrix','run_snapatac2',(Choice('function_name','pp.make_peak_matrix'),Output('dataset','import','dataset')))))
    plan=compile_semantic_plan(request,candidate,registry,build_semantic_compiler_contract(registry))
    first,matrix=plan.steps
    assert first.arguments['species']=='human' and first.arguments['chrom_sizes_path']==str(sample.sizes)
    assert matrix.arguments['peaks_path']==str(sample.peaks) and 'species' not in matrix.arguments
    assert all('gene_anno_path' not in step.arguments for step in plan.steps)
    owner._validate_native_consumption(binding,plan.steps)
    changed=PlanStep('matrix','run_snapatac2',dict(matrix.arguments,peaks_path=str(sample.gtf)),depends_on=matrix.depends_on)
    with pytest.raises(ResourceAdmissionError) as changed_reference:
        owner._validate_native_consumption(binding,(first,changed))
    assert changed_reference.value.code=='LOCAL_RESOURCE_CONSUMPTION_MISMATCH'


@pytest.mark.parametrize('order,kw,expected',[('unknown',{},False),('barcode',{},True),('unknown',{'sorted_by_barcode':True},True),('barcode',{'sorted_by_barcode':False},False)])
def test_order_adaptation_preserves_explicit_native_parameters(sample,order,kw,expected):
    args=dict(input_path=str(sample.source),function_name='pp.import_fragments',chrom_sizes_path=str(sample.sizes),fragment_order=order,kwargs=kw)
    actual=SNAPATAC2.native_arguments(args)
    assert actual==dict(kw,sorted_by_barcode=expected)
    assert args['kwargs']==kw
    direct=dict(args);direct.pop('fragment_order')
    assert SNAPATAC2.native_arguments(direct)==kw


def test_real_task_import_uses_native_sorting_and_default_retention(native,sample):
    before=sample.source.read_bytes();plan=task_plan(sample)
    result=SNAPATAC2.execute(dict(plan.steps[0].arguments),'compiled-task')
    data=ad.read_h5ad(result['matrix_path'])
    assert data.obs_names.tolist()==['A','B'] and data.obs.n_fragment.tolist()==[1100,250]
    params=result['parameters']
    assert params['kwargs']=={} and params['agent_selected_parameters']=={'sorted_by_barcode':False}
    assert params['effective_native_parameters']['min_num_fragments']==200
    assert params['effective_native_parameters']['sorted_by_barcode'] is False
    assert params['population']['retained_barcodes']==2 and params['population']['pre_import_barcode_count'] is None
    assert sample.source.read_bytes()==before


def test_real_explicit_native_call_retains_omitted_sorting_and_depth_defaults(native,sample):
    result=call(sample,'native','pp.import_fragments',sample.barcode,chrom_sizes_path=str(sample.sizes))
    params=result['parameters']
    assert params['kwargs']=={} and 'agent_selected_parameters' not in params
    assert params['effective_native_parameters']['sorted_by_barcode'] is True
    assert params['effective_native_parameters']['min_num_fragments']==200
    assert ad.read_h5ad(result['matrix_path']).obs_names.tolist()==['A','B']
    with pytest.raises(NativeInvocationError,match='sort fragment file by barcodes'):
        call(sample,'native-unsorted','pp.import_fragments',chrom_sizes_path=str(sample.sizes))


def test_real_explicit_retention_and_order_override_preserve_all_barcodes(native,sample):
    result=call(sample,'all','pp.import_fragments',sample.barcode,chrom_sizes_path=str(sample.sizes),
        fragment_order='unknown',kwargs=dict(min_num_fragments=0,sorted_by_barcode=True))
    assert result['parameters']['kwargs']==dict(min_num_fragments=0,sorted_by_barcode=True)
    assert result['parameters']['agent_selected_parameters']=={}
    assert result['parameters']['population']['retained_barcodes']==3
    assert ad.read_h5ad(result['matrix_path']).obs_names.tolist()==['A','B','C']


def test_real_native_qc_selection_populations_and_count_defaults(native,sample):
    imported=call(sample,'import','pp.import_fragments',chrom_sizes_path=str(sample.sizes),fragment_order='unknown')
    qc=call(sample,'qc','metrics.tsse',imported['matrix_path'],gene_anno_path=str(sample.gtf))
    selected=call(sample,'selected','pp.filter_cells',qc['matrix_path'])
    assert selected['parameters']['effective_native_parameters']['min_counts']==1000
    assert selected['parameters']['effective_native_parameters']['min_tsse']==5.0
    assert selected['parameters']['population']==dict(stage='native_cell_filter',input_barcodes=2,
        selected_barcodes=0,output_barcodes=0,applied=True)
    # This simulated TSS profile fails native QC. Preserve that outcome rather
    # than lowering the default threshold to make the tiny fixture pass.
    assert ad.read_h5ad(selected['matrix_path']).obs_names.tolist()==[]
    unchanged=call(sample,'looser','pp.filter_cells',qc['matrix_path'],kwargs=dict(min_counts=1,min_tsse=0,inplace=False))
    assert unchanged['parameters']['population']==dict(stage='native_cell_filter',input_barcodes=2,
        selected_barcodes=2,output_barcodes=2,applied=False)
    default=call(sample,'paired','pp.make_peak_matrix',imported['matrix_path'],peaks_path=str(sample.peaks))
    fragments=call(sample,'overlap','pp.make_peak_matrix',imported['matrix_path'],peaks_path=str(sample.peaks),kwargs=dict(counting_strategy='fragment'))
    assert default['parameters']['effective_native_parameters']['counting_strategy']=='paired-insertion'
    assert default['measurement']=='paired_insertion_counts' and fragments['measurement']=='fragment_counts'
    a,b=(ad.read_h5ad(r['matrix_path']) for r in (default,fragments))
    assert a.obs_names.tolist()==b.obs_names.tolist()==['A','B']
    assert a.var_names.tolist()==b.var_names.tolist()==['chr1:4970-5010','chr1:5100-5150','chr1:5200-5400']
    assert a.X.toarray().tolist()!=b.X.toarray().tolist()


def test_real_native_bam_output_uses_guaranteed_barcode_order(native,sample):
    import pysam
    source=sample.root/'unsorted.bam'
    with pysam.AlignmentFile(source,'wb',header={'HD':{'VN':'1.6'},'SQ':[{'SN':'chr1','LN':10000}]}) as stream:
        for name,barcode in [('first','Z'),('second','A')]:
            for first in (True,False):
                row=pysam.AlignedSegment();row.query_name=name;row.query_sequence='A'*50
                row.flag=99 if first else 147;row.reference_id=0;row.reference_start=100 if first else 180
                row.next_reference_id=0;row.next_reference_start=180 if first else 100
                row.template_length=130 if first else -130;row.mapping_quality=60;row.cigarstring='50M';row.set_tag('CB',barcode);stream.write(row)
    output=call(sample,'bam','pp.make_fragment_file',source,kwargs=dict(barcode_tag='CB'))
    assert [line.split('\t')[3] for line in Path(output['fragments_path']).read_text().splitlines()]==['A','Z']
    imported=call(sample,'bam-import','pp.import_fragments',output['fragments_path'],chrom_sizes_path=str(sample.sizes),kwargs=dict(min_num_fragments=1))
    assert imported['parameters']['effective_native_parameters']['sorted_by_barcode'] is True
    assert 'agent_selected_parameters' not in imported['parameters']
    assert ad.read_h5ad(imported['matrix_path']).obs_names.tolist()==['A','Z']


def test_grounded_task_order_prevails_without_replacing_reference_files(sample,monkeypatch):
    from agent.application import native_raw_handoff as handoff
    from agent.application.raw_handoff import FragmentsDeclaration
    reference=input_name('run_snapatac2','pp.make_peak_matrix','peaks_path')
    monkeypatch.setattr(handoff,'reference_inputs',lambda *a,**k:{ORDER:'unknown',reference:'/captured/regions'})
    candidates=SimpleNamespace(members=[dict(resource_id='source')],attribution=lambda:{'source':'captured'})
    source=SimpleNamespace(resource_id='typed')
    owner=SimpleNamespace(load=lambda _:source,bind_raw_source=lambda *a,**k:source,
        _verify_records=lambda _:None,compose_native=lambda _,values,**k:values)
    declared=FragmentsDeclaration(source='source.0',species='human',assembly='hg38')
    values,receipt=handoff.prepare(owner,'session',candidates,declared,None,{ORDER:'barcode'})
    assert values[ORDER]=='barcode' and values[reference]=='/captured/regions'
    assert ORDER not in receipt['resources']
    with pytest.raises(ValueError,match='cannot replace selected native reference'):
        handoff.prepare(owner,'session',candidates,declared,None,{reference:'/substituted/regions'})
