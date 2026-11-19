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


@pytest.mark.parametrize('population,options,method,coordinated',[
    ('uncalled_single_sample',{},'otsu',True),
    ('study_selected',{},'native_defaults',False),
    ('depth_prefiltered',{},'native_defaults',False),
    ('multiple_samples',{},'native_defaults',False),
    ('uncalled_single_sample',{'min_counts':0,'min_tsse':None},'native_defaults',False),
    (None,{},None,False),
])
def test_selected_qc_population_coordinates_only_exact_import(population,options,method,coordinated):
    from agent.orchestration.native_parameters import coordinate_native_parameters
    from agent.orchestration.scientific_inputs import OperationPreparationContext,ScientificInputRequired
    from agent.schemas import StepOutputRef
    registry=build_default_tool_registry()
    raw='run_snapatac2'
    selected={'import':(raw,dict(function_name='pp.import_fragments',input_path='/source')),
              'unrelated':(raw,dict(function_name='pp.import_fragments',input_path='/other')),
              'qc':(raw,dict(function_name='metrics.tsse',input_path=StepOutputRef('import','matrix_path'))),
              'filter':(raw,dict(function_name='pp.filter_cells',input_path=StepOutputRef('qc','matrix_path'),kwargs=options))}
    if population:selected['filter'][1]['qc_population']=population
    effective=coordinate_native_parameters(selected,registry)
    assert effective['filter'][1]['qc_selection']['method']==method
    assert ('qc_selection' in effective['import'][1])==coordinated
    assert 'qc_selection' not in effective['unrelated'][1]
    assert 'qc_selection' not in selected['filter'][1]
    if method is None:
        context=OperationPreparationContext(registry.get(raw),'filter',registry,effective,{}, {},[],None,None,{})
        with pytest.raises(ScientificInputRequired) as exc:
            SNAPATAC2.prepare_operation(effective['filter'][1],context)
        requirement,=exc.value.requirements
        assert requirement.argument=='qc_population' and requirement.dependencies==('source:bytes',)


def test_explicit_prefilter_and_explicit_native_defaults_win():
    from agent.orchestration.native_parameters import coordinate_native_parameters
    from agent.orchestration.scientific_inputs import ScientificDiagnostic
    from agent.schemas import StepOutputRef
    registry=build_default_tool_registry()
    raw='run_snapatac2'
    selected={'import':(raw,dict(function_name='pp.import_fragments',kwargs={'min_num_fragments':17})),
              'filter':(raw,dict(function_name='pp.filter_cells',input_path=StepOutputRef('import','matrix_path'),
                                 qc_population='uncalled_single_sample'))}
    result=coordinate_native_parameters(selected,registry)
    assert result['filter'][1]['qc_selection']['population']=='depth_prefiltered'
    selected['filter'][1]['qc_thresholds']='otsu'
    with pytest.raises(ScientificDiagnostic,match='unprefiltered'):
        coordinate_native_parameters(selected,registry)
    selected['filter'][1]['qc_thresholds']='native_defaults'
    assert coordinate_native_parameters(selected,registry)['filter'][1]['qc_selection']['reason']=='explicit_native_defaults'


def test_canonical_qc_reconciles_optimistic_declaration_with_exact_prefilter(monkeypatch):
    from types import SimpleNamespace
    from agent.orchestration.scientific_inputs import OperationPreparationContext
    from agent.tools.analysis.snapatac_raw import active_qc
    registry=build_default_tool_registry()
    tool=registry.get('run_snapatac2')
    context=OperationPreparationContext(tool,'filter',registry,{}, {},{},[],None,None,{})
    source=SimpleNamespace(publication_parameters={'population':dict(stage='native_import',min_num_fragments=200)})
    monkeypatch.setattr(type(context),'existing',lambda self,value:source)
    args=dict(function_name='pp.filter_cells',input_path='/accepted',qc_population='uncalled_single_sample',
              qc_selection={'method':'otsu','reason':'qualified_uncalled_single_sample'})
    effective=SNAPATAC2.prepare_operation(args,context)
    assert active_qc(effective)=='native_defaults'
    assert effective['qc_selection']['population']=='depth_prefiltered'
    assert active_qc(dict(args,qc_thresholds='native_defaults'))=='native_defaults'


def test_qc_lineage_follows_byte_transparent_inspection_and_blocks_unlinked_otsu():
    from agent.orchestration.native_parameters import coordinate_native_parameters
    from agent.orchestration.scientific_inputs import ScientificDiagnostic
    from agent.schemas import StepOutputRef
    registry=build_default_tool_registry()
    raw='run_snapatac2'
    steps={'import':(raw,dict(function_name='pp.import_fragments',kwargs={'min_num_fragments':200})),
           'tile':(raw,dict(function_name='pp.add_tile_matrix',input_path=StepOutputRef('import','matrix_path'))),
           'inspect':('inspect_scATAC',dict(path=StepOutputRef('tile','matrix_path'))),
           'filter':(raw,dict(function_name='pp.filter_cells',qc_population='uncalled_single_sample',
                              input_path=StepOutputRef('inspect','input_path')))}
    effective=coordinate_native_parameters(steps,registry)
    assert effective['filter'][1]['qc_selection']['population']=='depth_prefiltered'
    with pytest.raises(ScientificDiagnostic,match='import-only'):
        coordinate_native_parameters({'import':(raw,dict(function_name='pp.import_fragments',
                                                         qc_thresholds='otsu'))},registry)


def test_strict_compilation_does_not_authorize_unknown_qc_population(sample):
    from agent.orchestration.semantic_compiler import SemanticPlanCompileError
    registry=build_default_tool_registry()
    request=AgentRequest('task','Filter my dataset',dict(input_path='/fixture.h5ad',output_dir=str(sample.root/'out')))
    candidate=SemanticPlanCandidate((SemanticPlanStep('filter','run_snapatac2',
        (Choice('function_name','pp.filter_cells'),Input('dataset','input_path'))),))
    with pytest.raises(SemanticPlanCompileError,match='qc_population'):
        compile_semantic_plan(request,candidate,registry,build_semantic_compiler_contract(registry))


def test_population_edit_reaches_downstream_filter_across_inspection():
    from agent.orchestration.native_parameters import coordinate_native_parameters,parameter_edit_descendants
    from agent.schemas import PlanStep,StepOutputRef
    stages=(PlanStep('first','run_snapatac2',dict(function_name='pp.filter_cells',input_path='/source',kwargs={'inplace':True})),
            PlanStep('inspect','inspect_scATAC',dict(path=StepOutputRef('first','matrix_path'))),
            PlanStep('second','run_snapatac2',dict(function_name='pp.filter_cells',
                input_path=StepOutputRef('inspect','input_path'),qc_population='uncalled_single_sample',
                qc_selection={'method':'otsu'})))
    selected={step.step_id:(step.tool_name,step.arguments) for step in stages}
    affected=parameter_edit_descendants(stages,{'first'})
    assert affected=={'first','inspect','second'}
    policy=coordinate_native_parameters(selected,build_default_tool_registry(),only=affected)
    assert policy['second'][1]['qc_selection']['population']=='study_selected'
    assert policy['second'][1]['qc_selection']['method']=='native_defaults'


def test_coordinated_otsu_uses_native_measurements_on_shared_population(native,sample):
    from agent.orchestration.native_parameters import coordinate_native_parameters
    from agent.schemas import StepOutputRef
    registry=build_default_tool_registry();raw='run_snapatac2'
    proposed={'import':(raw,dict(function_name='pp.import_fragments',fragment_order='unknown')),
              'tsse':(raw,dict(function_name='metrics.tsse',input_path=StepOutputRef('import','matrix_path'))),
              'filter':(raw,dict(function_name='pp.filter_cells',qc_population='uncalled_single_sample',
                  input_path=StepOutputRef('tsse','matrix_path'),kwargs={'min_tsse':None,'max_counts':5000}))}
    steps=coordinate_native_parameters(proposed,registry)
    imported=call(sample,'coordinated-import','pp.import_fragments',chrom_sizes_path=str(sample.sizes),
        fragment_order='unknown',qc_selection=steps['import'][1]['qc_selection'])
    assert imported['parameters']['effective_native_parameters']['min_num_fragments']==0
    assert imported['parameters']['population']['retained_barcodes']==3
    qc=call(sample,'coordinated-tsse','metrics.tsse',source=imported['matrix_path'],gene_anno_path=str(sample.gtf))
    result=call(sample,'coordinated-filter','pp.filter_cells',source=qc['matrix_path'],
        kwargs=proposed['filter'][1]['kwargs'],qc_selection=steps['filter'][1]['qc_selection'])
    record=result['parameters']
    assert record['threshold_selection']['input_barcodes']==3
    assert record['threshold_selection']['thresholds']['min_tsse']=={'method':'explicit','value':None}
    assert record['threshold_selection']['thresholds']['min_counts']['method']=='skimage.filters.threshold_otsu'
    assert record['effective_native_parameters']['max_counts']==5000


def test_qc_result_edit_uses_native_defaults_without_refitting(native,sample):
    import json
    import numpy as np
    from agent.application import InteractiveAgentApplication
    from agent.orchestration import PlanningModelProfile
    from agent.providers import PlanningModelFactoryRegistry
    source=sample.root/'source.h5ad'
    data=ad.AnnData(np.eye(4))
    data.obs['n_fragment']=[100,300,1000,1800]
    data.obs['tsse']=[1.0,4.0,6.0,10.0]
    data.write_h5ad(source)
    class Model:
        model_id='qc-result-edit'
        def complete(self,*,prompt,response_schema):
            payload=json.loads(prompt)
            if 'turn_schema_version' in payload:
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='execute_plan',target='run_snapatac2')))
            if 'output_selection_schema_version' in payload:
                return json.dumps(dict(outputs=[dict(name='filtered',step_id='filter',output_key='matrix_path')]))
            if 'semantic_prompt_version' in payload:
                return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='filter',
                    tool='run_snapatac2',control_dependencies=[],sources=[
                        dict(target='function_name',source=dict(kind='registered_choice',choice='pp.filter_cells')),
                        dict(target='dataset',source=dict(kind='input',input='input_path')),
                        dict(target='qc_population',source=dict(kind='input',input='qc_population'))])])))
            raise AssertionError('No provider-backed response is needed for scientific verification')
    model=Model()
    profile=PlanningModelProfile('qc','scripted',model.model_id)
    app=InteractiveAgentApplication(sample.root/'workspace',model_profiles=(profile,),default_profile_id='qc',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:model}))
    app.create_session('session')
    first=app.submit_turn('session','initial','Filter this established uncalled sample.',expected_generation=0,
        execution_inputs={'input_path':str(source),'qc_population':'uncalled_single_sample'})
    assert first.status=='succeeded',first.to_dict()
    revision=app._application.sessions.load('session').revisions[-1]
    previous=app._application.run_store.load(revision.run_id)
    assert 'threshold_selection' in previous.steps[0].result['parameters']
    model.complete=lambda **kwargs:pytest.fail('Result edit attempted a provider call')
    edited=app.submit_turn('session','edit','',expected_generation=1,rerun_revision_id=revision.revision_id,
        parameter_overrides=[dict(tool='run_snapatac2',native_function='pp.filter_cells',step_id='filter',
                                  argument='qc_thresholds',value='native_defaults')])
    assert edited.status=='succeeded',edited.to_dict()
    last=app._application.sessions.load('session').revisions[-1]
    parameters=app._application.run_store.load(last.run_id).steps[0].result['parameters']
    assert parameters['qc_selection']['reason']=='explicit_native_defaults'
    assert 'threshold_selection' not in parameters
    assert parameters['effective_native_parameters']['min_counts']==1000


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


def test_task_handoff_supplies_unknown_order_without_touching_user_kwargs(sample):
    from agent.application.native_raw_handoff import prepare
    from agent.application.raw_handoff import FragmentsDeclaration,BamDeclaration
    candidates=SimpleNamespace(members=[dict(resource_id='source')],attribution=lambda:{'source':'captured'})
    source=SimpleNamespace(resource_id='typed')
    owner=SimpleNamespace(load=lambda _:source,bind_raw_source=lambda *a,**k:source,
        compose_native=lambda _,values,**k:values)
    for declaration in (FragmentsDeclaration,BamDeclaration):
        external,receipt=prepare(owner,'session',candidates,declaration(source='source.0',species='human',assembly='hg38'),None,{},resource_evidence={})
        assert (external.get(ORDER)=='unknown') == (declaration is FragmentsDeclaration)
        assert not any('kwargs' in key for key in external)


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


def test_grounded_task_order_prevails_without_replacing_reference_files(sample):
    from agent.application.native_raw_handoff import prepare
    from agent.application.raw_handoff import FragmentsDeclaration
    candidates=SimpleNamespace(members=[dict(resource_id='source')],attribution=lambda:{'source':'captured'})
    source=SimpleNamespace(resource_id='typed')
    owner=SimpleNamespace(load=lambda _:source,bind_raw_source=lambda *a,**k:source,
        compose_native=lambda _,values,**k:values)
    values,receipt=prepare(owner,'session',candidates,FragmentsDeclaration(source='source.0',species='human',assembly='hg38'),None,{ORDER:'barcode'},resource_evidence={})
    assert values[ORDER]=='barcode' and receipt['resources'][ORDER]=='barcode'
