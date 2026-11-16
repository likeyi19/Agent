"""Raw tasks retain explicit downstream intent without supplying derived sources."""
from types import SimpleNamespace

import pytest

from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
from agent.application.raw_handoff import validate_arguments


@pytest.fixture(params=['fastq', 'bam', 'fragments'])
def raw_binding(request, tmp_path):
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    kind = request.param
    if kind == 'fastq':
        records = []
        for role in ('R1', 'R2', 'R3'):
            path = tmp_path / (role + '.fastq')
            path.write_bytes(b'@synthetic\nACGT\n+\nIIII\n')
            records.append(owner.register(role, path, input_type='fastq', label=role,
                attribution='Synthetic binding test only', fastq_member=dict(library_id='test',
                fastq_layout='tenx-atac-r1-r2-r3.v1', role=role, lane='001', chunk='001', compression='plain')))
        record = owner.register_fastq_collection('test', [r.resource_id for r in records],
            label='Synthetic collection', attribution='Binding test only')
        compose = owner.compose_native
    else:
        path = tmp_path / ('source.bam' if kind == 'bam' else 'source.fragments')
        path.write_bytes(b'Registered bytes are not scientific qualification')
        record = owner.register('source', path, input_type='bam' if kind == 'bam' else 'external_fragments',
            label='Synthetic source', attribution='Binding test only')
        compose = owner.compose_native
    return kind, lambda values: compose(record.resource_id, values)


def test_raw_task_accepts_explicit_analysis_parameters(raw_binding):
    kind, compose = raw_binding
    values = dict(run_snapatac2__tl_spectral__kwargs={'n_comps':30},
        run_snapatac2__pp_knn__kwargs={'n_neighbors':15}, run_snapatac2__tl_leiden__kwargs={'resolution':0.6},
        effective_genome_size=2736124898,genome_size_basis='Explicit compatible assembly estimate',
        run_scanpy__pp_pca__kwargs={'n_comps':30})
    binding = compose(values)
    assert all(binding.execution_inputs[k] == v for k, v in values.items())
    validate_arguments(tuple(SimpleNamespace(tool=tool) for tool in (
        'run_snapatac2', 'run_scanpy', 'call_scATAC_peaks')), kind)


def test_bam_request_captures_explicit_downstream_peak_calling_parameters():
    from agent.application.raw_handoff import BamDeclaration
    from agent.application.turn_decisions import RequestRoles, ScopedArgument
    declared=ScopedArgument('call_scATAC_peaks','effective_genome_size','2736124898',0,10)
    request=RequestRoles('run_snapatac2',(),arguments=(declared,),
        bam=BamDeclaration(source='source.0',assay='SCATAC',species='human',assembly='hg19'))
    assert request.arguments==(declared,)


@pytest.mark.parametrize('field,function', [('bam','pp.make_fragment_file'),('fragments','pp.import_fragments')])
def test_raw_source_scoped_parameters_use_the_exact_captured_native_input(field,function):
    from agent.application.structured_parameters import bind_plan, check_plan, input_key
    from agent.application.turn_decisions import IntentError
    from agent.orchestration import AgentPlan, PlanStep, build_default_tool_registry
    registry=build_default_tool_registry()
    row=dict(tool='run_snapatac2',argument='kwargs',native_function=function,
        port='dataset',source='source.0',value={'barcode_tag':'RG'} if field=='bam' else {'min_num_fragments':0})
    values={input_key(registry,row):row['value']}
    admitted={field:dict(source='source.0'),field+'_context':dict(adapter='native-inputs.v1'),
        field+'_sources':dict(members=[dict(source_path='/captured/raw')]),
        'registered_input':dict(composition='native-science.v1'),'roles':[]}
    plan=AgentPlan('request','plan','Exact captured raw input',(
        PlanStep('native','run_snapatac2',dict(function_name=function,input_path='/captured/raw',output_dir='/output')),))
    bound=bind_plan(plan,[row],values,registry,admitted)
    assert dict(bound.steps[0].arguments['kwargs'])==row['value']
    check_plan(bound,[row],values,registry,admitted)
    from dataclasses import replace
    for other in ('/different/raw', '/output/derived.h5ad'):
        wrong=replace(plan,steps=(replace(plan.steps[0],arguments=dict(plan.steps[0].arguments,input_path=other)),))
        with pytest.raises(IntentError):
            bind_plan(wrong,[row],values,registry,admitted)
    for altered in (dict(row,source='source.1'),dict(row,port='peaks')):
        with pytest.raises(IntentError):
            bind_plan(plan,[altered],values,registry,admitted)


def test_ordinary_bam_declaration_binds_native_and_macs_parameters_through_shared_owner():
    from dataclasses import asdict
    import json
    from agent.application.dialogue_execution import _bind_arguments
    from agent.application.raw_handoff import BamDeclaration
    from agent.application.turn_decisions import parse_decision
    from agent.orchestration.registry import build_default_tool_registry
    basis='GRCh37 的 75bp 唯一可比对序列公开估计'
    utterance=f'双端 BAM 的细胞用 RG 标签区分。MACS3 有效基因组大小采用{basis} 2736124898。'
    raw=json.dumps(dict(turn_schema_version=1,decision=dict(kind='request_roles',target='run_snapatac2',
        roles=[],required_roles=[],bam=asdict(BamDeclaration(source='source.0',assay='SCATAC',species='human',assembly='hg19')),
        annotation=None,arguments=[dict(tool='call_scATAC_peaks',argument=name,literal=text,evidence=text,native_function=None)
            for name,text in [('effective_genome_size','2736124898'),('genome_size_basis',basis)]])))
    decision=parse_decision(raw,utterance=utterance)
    values=_bind_arguments(build_default_tool_registry(),utterance,{},decision.arguments)
    assert values==dict(effective_genome_size=2736124898,genome_size_basis=basis)


def test_raw_parameter_scope_keeps_tool_function_and_operation_independent():
    from agent.application.structured_parameters import bind_plan, input_key, ScopeRequired
    from agent.application.turn_decisions import IntentError
    from agent.orchestration import AgentPlan, PlanStep, build_default_tool_registry
    registry = build_default_tool_registry()
    row = dict(tool='run_snapatac2', argument='kwargs', native_function='pp.import_fragments',
        port='dataset', source='source.0', value={'min_num_fragments':0})
    admitted = dict(fragments={'source':'source.0'}, fragments_context={'adapter':'native-inputs.v1'},
        fragments_sources={'members':[{'source_path':'/captured/raw'}]},
        registered_input={'composition':'native-science.v1'}, roles=[])
    values = {input_key(registry,row):row['value']}
    def step(name, tool, function, source='/captured/raw'):
        return PlanStep(name, tool, dict(function_name=function,input_path=source,output_dir='/out/'+name))
    chosen = step('import','run_snapatac2','pp.import_fragments')
    unrelated = (step('other-tool','run_scanpy','pp.pca'),
        step('other-function','run_snapatac2','pp.filter_cells'),
        step('other-file','run_snapatac2','pp.import_fragments','/foreign/raw'))
    plan = AgentPlan('scoped','scoped','Exact raw consumer',(chosen,*unrelated))
    bound = bind_plan(plan,[row],values,registry,admitted)
    assert dict(bound.steps[0].arguments['kwargs']) == row['value']
    assert bound.steps[1:] == unrelated
    with pytest.raises(IntentError):
        bind_plan(AgentPlan('missing','missing','No matching consumer',unrelated),[row],values,registry,admitted)
    duplicate = step('second-import','run_snapatac2','pp.import_fragments')
    ambiguous = AgentPlan('two','two','Explicit operation required',(chosen,duplicate))
    with pytest.raises(ScopeRequired):
        bind_plan(ambiguous,[row],values,registry,admitted)
    exact = bind_plan(ambiguous,[dict(row,step_id='second-import')],values,registry,admitted)
    assert exact.steps[0] == chosen and dict(exact.steps[1].arguments['kwargs']) == row['value']


@pytest.mark.parametrize('tool,argument',[('unregistered_execution','parameter'),('call_scATAC_peaks','input_path')])
def test_raw_capture_does_not_authorize_unregistered_parameters_or_dataset_paths(tool,argument):
    from agent.application.dialogue_execution import _bind_arguments
    from agent.application.turn_decisions import IntentError,ScopedArgument
    from agent.orchestration.registry import build_default_tool_registry
    with pytest.raises(IntentError,match='unsupported_intent'):
        _bind_arguments(build_default_tool_registry(),'1',{},(ScopedArgument(tool,argument,'1',0,1),))


@pytest.mark.parametrize('values', [dict(input_path='/foreign.h5ad'), dict(embedding_path='/foreign.npy'),
    dict(analysis_path='/foreign.json'), dict(cell_ids_path='/foreign.tsv'), dict(overwrite=True),
    dict(unregistered_parameter=1),dict(effective_genome_size='not numeric')])
def test_raw_task_cannot_supply_derived_sources_or_overwrite(raw_binding, values):
    _, compose = raw_binding
    with pytest.raises(ResourceAdmissionError):
        compose(values)
