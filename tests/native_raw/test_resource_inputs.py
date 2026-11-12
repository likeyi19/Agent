"""Function-specific resource binding; small technical fixtures, no biology claims."""
from dataclasses import replace
import hashlib
import json
import os
from types import SimpleNamespace

import anndata as ad
import pytest

from agent.application.input_sets import ScientificInputSet
from agent.application.native_raw_handoff import (
    bind_plan, resource_evidence, requirements, ResourcePrerequisite,
)
from agent.application.raw_handoff import FragmentsDeclaration, declaration_dict
from agent.application.shared_resources import ScientificResourceLibrary
from agent.application.turn_decisions import parse_decision, IntentError, decision_schema
from agent.orchestration import AgentPlan, PlanStep, StepOutputRef
from agent.tools.analysis.native_matrix import SNAPATAC2
from agent.tools.data import scatac_reference as ref


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def inputs(tmp_path):
    fasta=tmp_path/'genome';fasta.write_text('>chr1\n'+'A'*10000+'\n')
    fai=tmp_path/'fai';fai.write_text('chr1\t10000\t6\t10000\t10001\n')
    bed=tmp_path/'regions';bed.write_text('chr1\t4970\t5010\nchr1\t5100\t5150\nchr1\t5200\t5400\n')
    bed2=tmp_path/'other-regions';bed2.write_text('chr1\t4900\t5500\n')
    gtf=tmp_path/'genes.gtf';gtf.write_text('chr1\tfixture\ttranscript\t5001\t6000\t.\t+\t.\tgene_id "G"; transcript_id "T"; gene_name "G"; transcript_name "T"; transcript_type "protein_coding";\n')
    source=tmp_path/'fragments';source.write_text(''.join(f'chr1\t{4990-i%11}\t{5200+i}\t{bc}\t1\n' for bc,count in [('B',250),('A',1100)] for i in range(count)))
    bundle=ref.build_scatac_reference_bundle(species='human',target_assembly='hg38',fasta_path=fasta,fai_path=fai,ccre_bed_path=bed)
    pointer=ref.publish_scatac_reference_bundle(bundle,tmp_path/'reference.json')
    full=ScientificInputSet('full','Human exact cCRE release 1',dict(reference_bundle_path=pointer['manifest_path'],reference_bundle_sha256=pointer['manifest_sha256']),fragments_companion=True)
    def component(name,port,path,**kw):
        return ScientificInputSet(name,name,dict(species='human',assembly='hg38',**{port+'_path':str(path),port+'_sha256':pin(path)['sha256']}),**kw)
    return SimpleNamespace(root=tmp_path,fasta=fasta,fai=fai,bed=bed,bed2=bed2,gtf=gtf,source=source,full=full,
        genome=component('genome','chrom_sizes',fai),regions=component('regions','peaks',bed),genes=component('genes','gene_anno',gtf),component=component)


def receipt(inputs,choices,selected=None,custom=None):
    return dict(adapter='native-inputs.v1',identity='a'*56,resources={},catalog=resource_evidence(choices),
        selected=selected,custom=custom or {},species='human',assembly='hg38')


def plan(inputs,*functions):
    steps=[]
    for i,fn in enumerate(functions):
        args=dict(function_name=fn,input_path=str(inputs.source) if i==0 else StepOutputRef(str(i-1),'matrix_path'),output_dir=str(inputs.root/('out-'+str(i))))
        if fn=='pp.import_fragments':args['fragment_order']='unknown'
        steps.append(PlanStep(str(i),'run_snapatac2',args,() if i==0 else (str(i-1),)))
    return AgentPlan('plan','task','scripted method choices',tuple(steps))


@pytest.mark.parametrize('configuration',['full','genome','separate','genome_and_full'])
def test_automatic_exact_inputs_for_selected_functions(inputs,configuration):
    choices={'full':(inputs.full,), 'genome':(inputs.genome,),
        'separate':(inputs.genome,inputs.regions,inputs.genes), 'genome_and_full':(inputs.genome,inputs.full)}[configuration]
    functions=('pp.import_fragments',) if configuration=='genome' else ('pp.import_fragments','pp.make_peak_matrix')
    result=bind_plan(plan(inputs,*functions),receipt(inputs,choices))
    assert result.steps[0].arguments['chrom_sizes_path']==str(inputs.fai)
    assert result.steps[0].arguments['chrom_sizes_sha256']==pin(inputs.fai)['sha256']
    if len(result.steps)>1:assert result.steps[1].arguments['peaks_path']==str(inputs.bed)
    assert all('gene_anno_path' not in s.arguments for s in result.steps)


def test_separate_library_components_are_native_inputs_without_bundle(inputs):
    lib=ScientificResourceLibrary(inputs.root/'library')
    lib.install_genome(inputs.fasta,inputs.fai,species=ref.species_identity('human'),assembly='hg38',
        label='Human genome',version='1',provenance='technical fixture',usage='test')
    for kind,path in [('regions',inputs.bed),('gene_annotation',inputs.gtf)]:
        resource=dict(species='human',assembly='hg38',file=pin(path))
        if kind=='regions':resource['coordinate_system']='zero_based_half_open'
        lib.install(kind,resource,label=kind,version='1',provenance='technical fixture',usage='test')
    result=bind_plan(plan(inputs,'pp.import_fragments','metrics.tsse','pp.make_peak_matrix'),receipt(inputs,lib.choices()))
    assert result.steps[0].arguments['chrom_sizes_path']==str(inputs.fai)
    assert result.steps[1].arguments['gene_anno_path']==str(inputs.gtf)
    assert result.steps[2].arguments['peaks_path']==str(inputs.bed)
    assert [k for s in result.steps for k in s.arguments if k.endswith('_path') and k!='input_path']==['chrom_sizes_path','gene_anno_path','peaks_path']


def test_true_feature_ambiguity_and_default_are_distinct(inputs):
    other=inputs.component('different-release','peaks',inputs.bed2)
    choices=(inputs.full,other)
    with pytest.raises(ResourcePrerequisite,match='distinct region feature'):
        bind_plan(plan(inputs,'pp.make_peak_matrix'),receipt(inputs,choices))
    default=replace(other,fragments_companion=True,fragments_default=True)
    resolved=bind_plan(plan(inputs,'pp.make_peak_matrix'),receipt(inputs,(inputs.full,default)))
    assert resolved.steps[0].arguments['peaks_path']==str(inputs.bed2)
    # Unused ambiguous components do not block a different function.
    assert bind_plan(plan(inputs,'metrics.frag_size_distr'),receipt(inputs,choices)).steps[0].arguments==plan(inputs,'metrics.frag_size_distr').steps[0].arguments


def test_explicit_reference_and_custom_regions_are_preserved(inputs):
    other=inputs.component('different-release','peaks',inputs.bed2,fragments_companion=True,fragments_default=True)
    result=bind_plan(plan(inputs,'pp.make_peak_matrix'),receipt(inputs,(inputs.full,other),selected='full'))
    assert result.steps[0].arguments['peaks_path']==str(inputs.bed)
    result=bind_plan(plan(inputs,'pp.make_peak_matrix'),receipt(inputs,(inputs.full,),custom={'peaks':pin(inputs.bed2)}))
    assert result.steps[0].arguments['peaks_path']==str(inputs.bed2)
    # Matrix-only continuation has no chromosome-size/GTF prerequisite.
    assert set(result.steps[0].arguments)=={'input_path','output_dir','function_name','peaks_path','peaks_sha256'}


@pytest.mark.parametrize('selection',['missing','conflict'])
def test_invalid_explicit_preference_is_never_substituted(inputs,selection):
    preference=dict(literal='unknown' if selection=='missing' else 'mouse reference',start=0,end=7 if selection=='missing' else 15)
    choices=(inputs.full,)
    if selection=='conflict':choices+=(ScientificInputSet('mouse','mouse reference',dict(species='mouse',assembly='mm10',chrom_sizes_path=str(inputs.fai),chrom_sizes_sha256=pin(inputs.fai)['sha256'])),)
    d=FragmentsDeclaration(source='source.0',species='human',assembly='hg38',resource_preference=preference)
    missing,_=requirements(SimpleNamespace(members=[{}]),d,choices,resource_evidence(choices))
    assert missing and ('not available' in missing[0] if selection=='missing' else 'conflicts' in missing[0])


def test_absent_component_and_cross_assembly_are_function_scoped(inputs):
    with pytest.raises(ResourcePrerequisite,match='gene annotation'):
        bind_plan(plan(inputs,'metrics.tsse'),receipt(inputs,(inputs.genome,inputs.regions)))
    wrong=replace(inputs.genome,execution_inputs=dict(inputs.genome.inputs(),assembly='mm10'))
    with pytest.raises(ResourcePrerequisite,match='chromosome-size'):
        bind_plan(plan(inputs,'pp.import_fragments'),receipt(inputs,(wrong,)))
    p=plan(inputs,'pp.make_peak_matrix')
    substituted=replace(p,steps=(replace(p.steps[0],arguments=dict(p.steps[0].arguments,peaks_path='/etc/passwd',peaks_sha256='f'*64)),))
    with pytest.raises(ValueError,match='substitute'):bind_plan(substituted,receipt(inputs,(inputs.full,)))


def test_language_schema_has_offered_resource_selection_and_grounded_preferences():
    for field in ('fastq','bam','fragments'):
        properties=decision_schema()['$defs'][field]['properties']
        assert 'resource' in properties and 'resource_preference' in properties
    text='Prepare a cCRE matrix from these human hg38 fragments.'
    raw=dict(kind='request_roles',target='run_snapatac2',required_roles=[],roles=[],arguments=[],
        fragments=declaration_dict(FragmentsDeclaration(source='source.0',species='human',assembly='hg38')))
    parsed=parse_decision(json.dumps(dict(turn_schema_version=1,decision=raw)),utterance=text)
    assert parsed.fragments.resource is None
    raw['fragments']['resource']='resource.full'
    with pytest.raises(IntentError):parse_decision(json.dumps(dict(turn_schema_version=1,decision=raw)),utterance=text)
    raw['fragments']['resource']=None
    raw['fragments']['resource_preference']=dict(literal='Human exact cCRE release 1',evidence='Human exact cCRE release 1')
    with pytest.raises(IntentError):parse_decision(json.dumps(dict(turn_schema_version=1,decision=raw)),utterance=text)
    parsed=parse_decision(json.dumps(dict(turn_schema_version=1,decision=raw)),utterance='Use Human exact cCRE release 1.')
    assert parsed.fragments.resource_preference['literal']=='Human exact cCRE release 1'


@pytest.mark.skipif(os.environ.get('RUN_RAW_NATIVE_QUALIFICATION')!='1',reason='Explicit native runtime qualification')
def test_real_native_functions_receive_independent_exact_components(inputs):
    bound=bind_plan(plan(inputs,'pp.import_fragments','metrics.tsse','pp.make_peak_matrix'),receipt(inputs,(inputs.genome,inputs.regions,inputs.genes)))
    last=None
    for step in bound.steps:
        args=dict(step.arguments)
        if last is not None:args['input_path']=last['matrix_path']
        last=SNAPATAC2.execute(args,'component-test')
        roles=last['source_identity']['references']
        expected={'pp.import_fragments':'chrom_sizes','metrics.tsse':'gene_anno','pp.make_peak_matrix':'peaks'}[args['function_name']]
        assert roles==[dict(role=expected,sha256=pin({'chrom_sizes':inputs.fai,'gene_anno':inputs.gtf,'peaks':inputs.bed}[expected])['sha256'])]
    matrix=ad.read_h5ad(last['matrix_path'])
    assert matrix.shape==(2,3) and list(matrix.obs_names)==['A','B']


class TaskModel:
    model_id='scripted/resource-contract'
    def __init__(self,preference=None):self.calls=[];self.preference=preference;self.language_calls=[]
    def complete(self,*,prompt,response_schema):
        data=json.loads(prompt)
        if 'clarification' in data:
            self.language_calls.append(data)
            return json.dumps(dict(text=data['request']))
        self.calls.append(data)
        if 'turn_schema_version' in data:
            d=declaration_dict(FragmentsDeclaration(source='source.0',species='human',assembly='hg38'))
            if self.preference is not None:d['resource_preference']=dict(literal=self.preference,evidence=self.preference)
            return json.dumps(dict(turn_schema_version=1,decision=dict(kind='request_roles',target='run_snapatac2',required_roles=[],roles=[],arguments=[],fragments=d)))
        if 'output_selection_schema_version' in data:
            return json.dumps(dict(outputs=[dict(name='matrix',step_id='matrix',output_key='manifest_path'),dict(name='dataset',step_id='matrix',output_key='matrix_path')]))
        raise AssertionError('Unexpected provider call: '+str(data.keys()))


class TaskPlanner:
    def __init__(self):self.calls=0
    def plan(self,request,registry):
        self.calls+=1
        from agent.orchestration.semantic_compiler import (
            SemanticPlanCandidate,SemanticPlanStep,SemanticRegisteredChoiceSource as Choice,
            SemanticRequestInputSource as Input,SemanticStepOutputSource as Output,
            compile_semantic_plan,build_semantic_compiler_contract,
        )
        candidate=SemanticPlanCandidate((
            SemanticPlanStep('import','run_snapatac2',(Choice('function_name','pp.import_fragments'),Input('dataset','input_path'),Input('input_sha256','input_sha256'))),
            SemanticPlanStep('matrix','run_snapatac2',(Choice('function_name','pp.make_peak_matrix'),Output('dataset','import','dataset')))))
        return compile_semantic_plan(request,candidate,registry,build_semantic_compiler_contract(registry))


def task_application(inputs):
    from agent.application import ResearchAgentApplication
    from agent.application.local_resources import LocalResourceAdmission
    planner=TaskPlanner()
    app=ResearchAgentApplication(inputs.root/'workspace',planner=planner)
    owner=LocalResourceAdmission(app._workspace,approved_source_roots=(inputs.root,))
    record=owner.register('source',inputs.source,input_type='attachment',label='technical fragments',attribution='simulated',session_id='research')
    captured=owner.compose_group([record.resource_id],attachment_candidates=True)
    app.sessions.create('research')
    return app,planner,captured


def test_ambiguity_is_a_persistent_question_before_execution_run(inputs):
    app,planner,captured=task_application(inputs)
    model=TaskModel()
    other=inputs.component('release-2','peaks',inputs.bed2)
    choices=(inputs.full,other)
    outcome=app.sessions.respond('research','first','Prepare a cCRE matrix from these human hg38 fragments.',
        interpreter=model,submission=dict(execution_inputs=dict(captured.execution_inputs),registered_input=captured.attribution()),input_sets=choices)
    assert outcome.status=='clarification',outcome
    state=app.sessions.load('research')
    assert not state.turns and not state.revisions and not list((app.workspace_root/'run_state').glob('*.json'))
    assert planner.calls==1 and len(model.calls)==1
    from agent.application import ResearchAgentApplication
    restarted=ResearchAgentApplication(app.workspace_root)
    assert restarted.sessions.load('research')==state


@pytest.mark.skipif(os.environ.get('RUN_RAW_NATIVE_QUALIFICATION')!='1',reason='Explicit native runtime qualification')
def test_task_native_acceptance_and_saved_continuation_do_not_replan_or_reopen_resources(inputs,monkeypatch):
    app,planner,captured=task_application(inputs)
    outcome=app.sessions.respond('research','first','Prepare a cCRE matrix from these human hg38 fragments.',
        interpreter=TaskModel(),submission=dict(execution_inputs=dict(captured.execution_inputs),registered_input=captured.attribution()),
        input_sets=(inputs.genome,inputs.regions,inputs.genes))
    assert outcome.status=='activated',outcome
    state=app.sessions.load('research');assert planner.calls==1 and len(state.turns)==1 and len(state.revisions)==1
    run=app.run_store.load(state.turns[0].run_id)
    assert [s.result['function_name'] for s in run.steps]==['snapatac2.pp.import_fragments','snapatac2.pp.make_peak_matrix']
    assert [(s.result['n_cells'],s.result['n_features']) for s in run.steps]==[(2,0),(2,3)]
    monkeypatch.setattr(SNAPATAC2,'native',lambda *a:pytest.fail('Historical read repeated science'))
    for path in (inputs.source,inputs.fasta,inputs.fai,inputs.bed,inputs.gtf):path.unlink()
    from agent.application import ResearchAgentApplication
    restarted=ResearchAgentApplication(app.workspace_root)
    assert restarted.sessions.load('research')==state
    from agent.orchestration.prior_outputs import bind_output
    binding=bind_output(restarted.run_store,restarted.registry,run.run_id,'matrix','dataset')
    assert binding.semantic_type=='native_matrix_h5ad.v1'


def test_fai_layout_differences_do_not_create_native_size_ambiguity(inputs):
    library=ScientificResourceLibrary(inputs.root/'library')
    arguments=dict(species=ref.species_identity('human'),assembly='hg38',version='1',provenance='technical fixture',usage='test')
    first=library.install_genome(inputs.fasta,inputs.fai,label='unwrapped genome',**arguments)
    fasta=inputs.root/'wrapped.fa';fasta.write_text('>chr1\n'+('A'*50+'\n')*200)
    fai=inputs.root/'wrapped.fai';fai.write_text('chr1\t10000\t6\t50\t51\n')
    second=library.install_genome(fasta,fai,label='wrapped genome',**arguments)
    assert first['resource']['genome']['fai']['sha256']!=second['resource']['genome']['fai']['sha256']
    assert first['resource']['genome']['ordered_contig_sha256']==second['resource']['genome']['ordered_contig_sha256']
    evidence=[]
    bound=bind_plan(plan(inputs,'pp.import_fragments'),receipt(inputs,library.choices()),bindings=evidence)
    assert evidence[0]['reason']=='automatic'
    assert bound.steps[0].arguments['chrom_sizes_sha256'] in {pin(inputs.fai)['sha256'],pin(fai)['sha256']}
    # An explicit reference retains its exact FAI even when size-equivalent.
    selected='library-'+second['id']
    explicit=bind_plan(plan(inputs,'pp.import_fragments'),receipt(inputs,library.choices(),selected=selected))
    assert explicit.steps[0].arguments['chrom_sizes_path']==str(fai)


@pytest.mark.skipif(os.environ.get('RUN_RAW_NATIVE_QUALIFICATION')!='1',reason='Explicit native runtime qualification')
def test_explicit_answer_to_resource_question_keeps_original_source_and_objective(inputs):
    app,planner,captured=task_application(inputs)
    choices=(inputs.full,inputs.component('release-2','peaks',inputs.bed2))
    submission=dict(execution_inputs=dict(captured.execution_inputs),registered_input=captured.attribution())
    first=app.sessions.respond('research','first','Prepare a cCRE matrix from these human hg38 fragments.',interpreter=TaskModel(),submission=submission,input_sets=choices)
    assert first.status=='clarification'
    class AnswerModel(TaskModel):
        def complete(self,*,prompt,response_schema):
            data=json.loads(prompt)
            if 'turn_schema_version' in data:
                d=declaration_dict(FragmentsDeclaration())
                d['resource_preference']=dict(literal='Human exact cCRE release 1',evidence='Human exact cCRE release 1')
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='answer_roles',pending='@roles',roles=[],arguments=[],fragments=d)))
            return super().complete(prompt=prompt,response_schema=response_schema)
    second=app.sessions.respond('research','answer','Use Human exact cCRE release 1.',interpreter=AnswerModel(),input_sets=choices)
    assert second.status=='activated',second
    state=app.sessions.load('research');assert len(state.turns)==1 and len(state.revisions)==1
    current=state.interactions[-1].admitted
    assert current['fragments']['source']=='source.0'
    assert current['continuation']['origin_turn_id']=='first'
    assert current['native_resource_bindings'][-1]['reason']=='explicit'
    run=app.run_store.load(state.turns[0].run_id)
    assert run.steps[-1].resolved_arguments['peaks_sha256']==pin(inputs.bed)['sha256']


def test_future_exact_genome_installation_reuses_identity_without_relabelling(inputs):
    import shutil
    library=ScientificResourceLibrary(inputs.root/'library')
    args=dict(species=ref.species_identity('human'),assembly='hg38',version='1',provenance='technical fixture',usage='test')
    first=library.install_genome(inputs.fasta,inputs.fai,label='original',**args)
    fasta=inputs.root/'another.fa';fai=inputs.root/'another.fai'
    shutil.copyfile(inputs.fasta,fasta);shutil.copyfile(inputs.fai,fai)
    duplicate=library.install_genome(fasta,fai,label='new locator',**dict(args,provenance='same bytes supplied again'))
    assert duplicate==first and duplicate['label']=='original' and library.entries()==(first,)


def test_llm_planner_recovery_path_resolves_inputs_before_reservation(inputs):
    from agent.application import ResearchAgentApplication
    from agent.orchestration.llm_planner import LLMPlanner
    class SemanticModel(TaskModel):
        def complete(self,*,prompt,response_schema):
            data=json.loads(prompt)
            if 'semantic_prompt_version' in data:
                return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[
                    dict(step_id='import',tool='run_snapatac2',sources=[
                        dict(target='function_name',source=dict(kind='registered_choice',choice='pp.import_fragments')),
                        dict(target='dataset',source=dict(kind='input',input='input_path')),
                        dict(target='input_sha256',source=dict(kind='input',input='input_sha256'))],control_dependencies=[]),
                    dict(step_id='matrix',tool='run_snapatac2',sources=[
                        dict(target='function_name',source=dict(kind='registered_choice',choice='pp.make_peak_matrix')),
                        dict(target='dataset',source=dict(kind='step_port',step='import',source_port='dataset'))],control_dependencies=[])])))
            return super().complete(prompt=prompt,response_schema=response_schema)
    app,_,captured=task_application(inputs)
    model=SemanticModel()
    app=ResearchAgentApplication(app.workspace_root,planner=LLMPlanner(model))
    outcome=app.sessions.respond('research','first','Prepare a cCRE matrix from these human hg38 fragments.',
        interpreter=model,submission=dict(execution_inputs=dict(captured.execution_inputs),registered_input=captured.attribution()),
        input_sets=(inputs.full,inputs.component('release-2','peaks',inputs.bed2)))
    assert outcome.status=='clarification',outcome
    assert not app.sessions.load('research').turns


def test_resource_question_replays_after_catalog_order_and_wording_changes(inputs,monkeypatch):
    app,_,captured=task_application(inputs)
    other=inputs.root/'other-sizes';other.write_text('chr1\t20000\n')
    choices=(inputs.component('z-genome','chrom_sizes',inputs.fai),inputs.component('a-genome','chrom_sizes',other),inputs.regions)
    result=app.sessions.respond('research','first','Prepare a cCRE matrix from these human hg38 fragments.',
        interpreter=TaskModel(),submission=dict(execution_inputs=dict(captured.execution_inputs),registered_input=captured.attribution()),input_sets=choices)
    assert result.status=='clarification'
    state=app.sessions.load('research');before=state.interactions[0].admitted['choices']
    from agent.application import native_raw_handoff as owner
    original=owner.bind_plan
    def changed_wording(*args,**kwargs):
        try:return original(*args,**kwargs)
        except ResourcePrerequisite as exc:raise ResourcePrerequisite(['Please identify the source reference.'], options=exc.options)
    monkeypatch.setattr(owner,'bind_plan',changed_wording)
    from agent.application import ResearchAgentApplication
    reopened=ResearchAgentApplication(app.workspace_root).sessions.load('research')
    assert reopened==state and reopened.interactions[0].admitted['choices']==before
    assert not reopened.turns and not list((app.workspace_root/'run_state').glob('*.json'))


def test_user_genome_name_resolves_equivalent_configuration_aliases_per_function(inputs):
    from agent.application.native_raw_handoff import _preferred_ids
    name='Human reference release 1'
    choices=(replace(inputs.full,label=name+' with cCREs'),replace(inputs.genome,label=name+' genome only'))
    declared=FragmentsDeclaration(source='source.0',species='human',assembly='hg38',
        resource_preference=dict(literal=name,start=0,end=len(name)))
    evidence=resource_evidence(choices)
    questions,selected=requirements(SimpleNamespace(members=[{}]),declared,choices,evidence)
    assert not questions and selected is None
    captured=receipt(inputs,choices,selected=_preferred_ids(declared,evidence,selected))
    bindings=[]
    resolved=bind_plan(plan(inputs,'pp.import_fragments','pp.make_peak_matrix'),captured,bindings=bindings)
    assert resolved.steps[0].arguments['chrom_sizes_sha256']==pin(inputs.fai)['sha256']
    assert resolved.steps[1].arguments['peaks_sha256']==pin(inputs.bed)['sha256']
    assert [row['reason'] for row in bindings]==['explicit','explicit']
    alternative=inputs.component('other-regions','peaks',inputs.bed2)
    choices+=(replace(alternative,label=name+' alternate regions'),)
    evidence=resource_evidence(choices)
    questions,selected=requirements(SimpleNamespace(members=[{}]),declared,choices,evidence)
    assert not questions
    with pytest.raises(ResourcePrerequisite,match='region feature'):
        bind_plan(plan(inputs,'pp.make_peak_matrix'),receipt(inputs,choices,selected=_preferred_ids(declared,evidence,selected)))


def test_existing_configured_defaults_retain_their_declared_role_scope(inputs):
    choice=replace(inputs.genome,bam_companion=True,bam_default=True)
    assert resource_evidence((choice,),field='bam')[choice.input_set_id]['default']
    assert not resource_evidence((choice,),field='fragments')[choice.input_set_id]['default']


def test_preparation_error_uses_existing_failed_interaction_boundary(inputs,monkeypatch):
    from agent.application import dialogue_execution
    app,_,captured=task_application(inputs)
    def fail(*args):raise ValueError('simulated preparation failure')
    monkeypatch.setattr(dialogue_execution,'prepare_native_task',fail)
    with pytest.raises(ValueError,match='simulated preparation'):
        app.sessions.respond('research','first','Prepare a cCRE matrix from these human hg38 fragments.',
            interpreter=TaskModel(),submission=dict(execution_inputs=dict(captured.execution_inputs),registered_input=captured.attribution()),input_sets=(inputs.full,))
    state=app.sessions.load('research')
    assert state.interactions[0].status=='failed' and not state.turns


def test_native_dictionary_needs_no_library_and_cannot_override_explicit_reference(inputs):
    p=plan(inputs,'pp.import_fragments')
    p=replace(p,steps=(replace(p.steps[0],arguments=dict(p.steps[0].arguments,kwargs={'chrom_sizes':{'chr1':10000}})),))
    assert bind_plan(p,receipt(inputs,()))==p
    assert bind_plan(p,receipt(inputs,(inputs.full,)))==p
    with pytest.raises(ResourcePrerequisite,match='Both a native value'):
        bind_plan(p,receipt(inputs,(inputs.full,),selected='full'))


def test_corrected_alias_preference_does_not_retain_an_older_whole_resource_choice(inputs):
    choices=(replace(inputs.genome,input_set_id='genome-one',label='Source reference one'),
        replace(inputs.genome,input_set_id='genome-two',label='Source reference two'),inputs.regions,
        inputs.component('regions-two','peaks',inputs.bed2))
    app,_,captured=task_application(inputs)
    first=app.sessions.respond('research','first','Use Source reference one for a region matrix from these human hg38 fragments.',
        interpreter=TaskModel('Source reference one'),submission=dict(execution_inputs=dict(captured.execution_inputs),registered_input=captured.attribution()),input_sets=choices)
    assert first.status=='clarification'
    assert app.sessions.load('research').interactions[-1].admitted['raw_resource_choice']=='resource.genome-one'
    class AnswerModel(TaskModel):
        def complete(self,*,prompt,response_schema):
            data=json.loads(prompt)
            if 'turn_schema_version' in data:
                d=declaration_dict(FragmentsDeclaration())
                d['resource_preference']=dict(literal='Source reference',evidence='Source reference')
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='answer_roles',pending='@roles',roles=[],arguments=[],fragments=d)))
            return super().complete(prompt=prompt,response_schema=response_schema)
    second=app.sessions.respond('research','second','Use Source reference.',interpreter=AnswerModel(),input_sets=choices)
    assert second.status=='clarification'
    state=app.sessions.load('research');admitted=state.interactions[-1].admitted
    assert admitted['raw_resource_choice'] is None
    assert set(admitted['fragments_context']['selected'])=={'genome-one','genome-two'}
    assert not state.turns and not state.revisions


def test_no_resource_function_and_native_dictionary_need_no_biological_declaration(inputs):
    from agent.application.raw_handoff import BamDeclaration
    d=BamDeclaration(source='source.0')
    assert requirements(SimpleNamespace(members=[{}]),d,(),{})==([],None)
    r=dict(receipt(inputs,()),species=None,assembly=None)
    p=plan(inputs,'pp.make_fragment_file')
    assert bind_plan(p,r)==p
    p=plan(inputs,'pp.import_fragments')
    p=replace(p,steps=(replace(p.steps[0],arguments=dict(p.steps[0].arguments,kwargs={'chrom_sizes':{'chr1':10000}})),))
    assert bind_plan(p,r)==p
    # No human/mouse gate on the native source adapter.
    FragmentsDeclaration(source='source.0',species='zebrafish',assembly='danRer11')
    BamDeclaration(source='source.0',species='human',assembly='GRCh38')


def test_unknown_context_does_not_merge_known_distinct_declarations(inputs):
    other=replace(inputs.genome,input_set_id='different-frame',execution_inputs=dict(inputs.genome.inputs(),assembly='another-assembly'))
    r=dict(receipt(inputs,(inputs.genome,other)),species=None,assembly=None)
    with pytest.raises(ResourcePrerequisite,match='distinct chromosome-size'):
        bind_plan(plan(inputs,'pp.import_fragments'),r)


def test_fastq_reuses_genome_and_independent_whitelist_without_qc_or_regions(inputs,monkeypatch):
    from agent.application import ResearchAgentApplication
    from agent.application.local_resources import LocalResourceAdmission
    from agent.application.raw_handoff import FastqDeclaration,FastqMember
    from agent.application.native_raw_handoff import _mapping_inputs
    library=ScientificResourceLibrary(inputs.root/'library')
    entry=library.install_genome(inputs.fasta,inputs.fai,species=ref.species_identity('human'),assembly='hg38',
        label='Technical mapping genome',version='1',provenance='Synthetic',usage='Tests')
    white=inputs.root/'whitelist';white.write_text('A\n')
    choices=(library.input_set(entry),inputs.component('barcode-list','whitelist',white))
    d=FastqDeclaration(layout='tenx-atac-r1-r2-r3.v1',namespace='technical',species='human',
        barcode_policy='raw_sequence_whitelist',members=tuple(FastqMember('source.'+str(i),role,'001','001') for i,role in enumerate(('R1','R2','R3'))))
    assert requirements(SimpleNamespace(members=[{},{},{}]),d,choices,resource_evidence(choices,field='fastq'))==([],None)
    class Model:
        model_id='scripted/native-input-contract'
        def complete(self,*,prompt,response_schema):
            return json.dumps(dict(turn_schema_version=1,decision=dict(kind='request_roles',target='run_chromap',
                required_roles=[],roles=[],arguments=[],fastq=declaration_dict(d))))
    class Planner:
        def plan(self,request,registry):
            assert request.inputs['reference_path']==str(inputs.fasta)
            assert request.inputs['reference_sha256']==pin(inputs.fasta)['sha256']
            assert request.inputs['whitelist_path']==str(white)
            assert not any('bundle' in key or 'qc_reference' in key for key in request.inputs)
            return AgentPlan('plan',request.request_id,'Scripted mapping then requested TSSe',(
                PlanStep('map','run_chromap',dict(function_name='map',output_dir=str(Path(request.inputs['output_dir'])/'map'),
                    **{key:value for key,value in request.inputs.items() if key in ('reference_path','reference_sha256','whitelist_path','whitelist_sha256','read1_path','read2_path','barcode_path')})),
                PlanStep('tsse','run_snapatac2',dict(function_name='metrics.tsse',input_path=StepOutputRef('map','fragments_path'),output_dir=str(Path(request.inputs['output_dir'])/'tsse')),depends_on=('map',))))
    from pathlib import Path
    app=ResearchAgentApplication(inputs.root/'fastq-workspace',planner=Planner());app.sessions.create('research')
    owner=LocalResourceAdmission(app._workspace,approved_source_roots=(inputs.root,))
    records=[]
    for i in range(3):
        path=inputs.root/('read'+str(i));path.write_text('@r\nACGT\n+\nIIII\n')
        records.append(owner.register('read'+str(i),path,input_type='attachment',label='Technical read',attribution='Synthetic',session_id='research'))
    group=owner.compose_group([r.resource_id for r in records],attachment_candidates=True)
    monkeypatch.delenv('AGENT_CHROMAP_INDEX_ROOT',raising=False)
    outcome=app.sessions.respond('research','first','Map these reads and compute TSSe.',interpreter=Model(),input_sets=choices,
        submission=dict(execution_inputs={},registered_input=group.attribution()))
    assert outcome.status=='clarification',outcome
    state=app.sessions.load('research')
    assert not state.turns and not state.revisions
    a=state.interactions[0].admitted
    assert 'gene annotation' in a['choices'][0]
    assert a['fastq_context']['resources']['whitelist_sha256']==pin(white)['sha256']
    for path in (inputs.fasta,inputs.fai,white):path.unlink()
    assert ResearchAgentApplication(app._workspace.root).sessions.load('research')==state


def test_chromap_consumed_file_pin_is_checked_before_native_invocation(inputs):
    from agent.tools.data.chromap_native import execute
    from agent.schemas.verification_authority import AuthorityError
    with pytest.raises(AuthorityError,match='identity changed'):
        execute(dict(reference_path=str(inputs.fasta),reference_sha256='f'*64,function_name='index',output_dir=str(inputs.root/'unused')))
    assert not (inputs.root/'unused').exists()


def test_explicit_scientific_species_name_uses_its_recorded_field_without_alias_inference(inputs):
    from agent.application.native_raw_handoff import _context_matches
    row=dict(species={'scientific_name':'Danio rerio','taxonomy_id':7955},assembly='danRer11')
    assert _context_matches(row,dict(species='Danio rerio',assembly='danRer11'))
    assert not _context_matches(row,dict(species='guessed common alias',assembly='danRer11'))


def test_mapped_fasta_constrains_downstream_sizes_even_with_a_different_default(inputs):
    other=inputs.component('other-genome','chrom_sizes',inputs.bed2,fragments_companion=True,fragments_default=True)
    r=receipt(inputs,(inputs.full,other))
    r['resources']={'reference_sha256':pin(inputs.fasta)['sha256']}
    bound=bind_plan(plan(inputs,'pp.import_fragments'),r)
    assert bound.steps[0].arguments['chrom_sizes_sha256']==pin(inputs.fai)['sha256']


@pytest.mark.parametrize('association',['supplied','mapped'])
def test_explicit_size_reference_cannot_be_replaced_by_genome_association(inputs,association):
    other_fai=inputs.root/'explicit.fai';other_fai.write_text('chr2\t10000\n')
    requested=inputs.component('explicit-sizes','chrom_sizes',other_fai)
    captured=receipt(inputs,(inputs.full,requested),selected=requested.input_set_id,
        custom={'genome':pin(inputs.fasta)} if association=='supplied' else None)
    if association=='mapped':captured['resources']={'reference_sha256':pin(inputs.fasta)['sha256']}
    with pytest.raises(ResourcePrerequisite,match='reference.*genome'):
        bind_plan(plan(inputs,'pp.import_fragments'),captured)


def test_conflicting_explicit_sizes_remain_a_saved_question_before_science(inputs):
    from agent.application import ResearchAgentApplication
    from agent.application.local_resources import LocalResourceAdmission
    app,planner,captured=task_application(inputs)
    owner=LocalResourceAdmission(app._workspace,approved_source_roots=(inputs.root,))
    genome=owner.register('supplied-genome',inputs.fasta,input_type='attachment',label='Supplied genome',
        attribution='Technical fixture',session_id='research')
    captured=owner.compose_group([captured.members[0]['resource_id'],genome.resource_id],attachment_candidates=True)
    other_fai=inputs.root/'requested.fai';other_fai.write_text('chr2\t10000\n')
    requested=inputs.component('requested-sizes','chrom_sizes',other_fai)
    class Model(TaskModel):
        def complete(self,*,prompt,response_schema):
            result=json.loads(super().complete(prompt=prompt,response_schema=response_schema))
            result['decision']['fragments']['genome_source']='source.1'
            return json.dumps(result)
    outcome=app.sessions.respond('research','choice','Use requested-sizes with this supplied genome.',
        interpreter=Model('requested-sizes'),input_sets=(inputs.full,requested),
        submission=dict(execution_inputs={},registered_input=captured.attribution()))
    assert outcome.status=='clarification',outcome
    state=app.sessions.load('research');admitted=state.interactions[0].admitted
    assert admitted['raw_resource_choice']=='resource.requested-sizes'
    assert 'no recorded association' in admitted['choices'][0]
    assert not state.turns and not state.revisions and not list((app.workspace_root/'run_state').glob('*.json'))
    assert ResearchAgentApplication(app.workspace_root).sessions.load('research')==state


def test_raw_annotation_context_survives_reference_question_and_restart(inputs):
    from agent.application.raw_handoff import AnnotationDeclaration
    from agent.application import ResearchAgentApplication
    class CompoundModel(TaskModel):
        def complete(self, *, prompt, response_schema):
            value=json.loads(super().complete(prompt=prompt,response_schema=response_schema))
            value['decision']['annotation']=declaration_dict(AnnotationDeclaration(context='human PBMC'))
            return json.dumps(value)
    app,planner,captured=task_application(inputs)
    choices=(inputs.full,inputs.component('release-2','peaks',inputs.bed2))
    outcome=app.sessions.respond('research','first','Analyze and annotate human PBMC fragments on hg38.',
        interpreter=CompoundModel(),submission=dict(execution_inputs=dict(captured.execution_inputs),registered_input=captured.attribution()),input_sets=choices)
    assert outcome.status=='clarification',outcome
    state=app.sessions.load('research')
    admitted=state.interactions[-1].admitted
    assert admitted['downstream_annotation']['context']=='human PBMC'
    assert admitted['fragments']['source']=='source.0'
    assert state.interactions[-1].prerequisite['field']=='fragments'
    assert not state.turns and not state.revisions
    assert ResearchAgentApplication(app.workspace_root).sessions.load('research')==state
    class Continuation:
        model_id='scripted/context-continuation'
        def complete(self, *, prompt, response_schema):
            return json.dumps(dict(turn_schema_version=1,decision=dict(kind='answer_roles',pending='@roles',roles=[],arguments=[],
                fragments=declaration_dict(FragmentsDeclaration()),annotation=declaration_dict(AnnotationDeclaration(context='human blood')))))
    next_outcome=app.sessions.respond('research','second','The tissue context is human blood.',interpreter=Continuation(),input_sets=choices)
    assert next_outcome.status=='clarification',next_outcome
    new=app.sessions.load('research')
    assert new.interactions[-1].admitted['downstream_annotation']['context']=='human blood'
    assert new.interactions[0]==state.interactions[0]
    assert ResearchAgentApplication(app.workspace_root).sessions.load('research')==new


def test_malformed_continuation_keeps_exact_pending_objective_without_retry(inputs):
    from agent.application import ResearchAgentApplication
    app,planner,captured=task_application(inputs)
    choices=(inputs.full,inputs.component('release-2','peaks',inputs.bed2))
    first=app.sessions.respond('research','first','Build a matrix from human hg38 fragments.',interpreter=TaskModel(),
        submission=dict(execution_inputs=dict(captured.execution_inputs),registered_input=captured.attribution()),input_sets=choices)
    assert first.status=='clarification'
    before=app.sessions.load('research')
    class BrokenModel:
        model_id='scripted/malformed'
        def complete(self,**kwargs):return '{broken'
    failed=app.sessions.respond('research','failed','Use the first reference.',interpreter=BrokenModel(),input_sets=choices)
    assert failed.status=='failed'
    after=app.sessions.load('research')
    assert after.interactions[-1].prerequisite==before.interactions[-1].prerequisite
    assert after.interactions[0]==before.interactions[0]
    assert not after.turns and not after.revisions and planner.calls==1
    assert ResearchAgentApplication(app.workspace_root).sessions.load('research')==after


@pytest.mark.parametrize('literal', ['Human exact cCRE release 1', '使用第一版参考的原始染色体序列及配套注释'])
def test_explicit_offered_reference_can_use_a_label_or_natural_description_without_losing_sources(inputs,literal):
    from agent.application.native_raw_handoff import requirements
    from agent.application.raw_handoff import resources
    app,planner,captured=task_application(inputs)
    choices=(inputs.full,inputs.component('release-2','peaks',inputs.bed2))
    app.sessions.respond('research','first','Build a matrix from human hg38 fragments.',interpreter=TaskModel(),
        submission=dict(execution_inputs=dict(captured.execution_inputs),registered_input=captured.attribution()),input_sets=choices)
    state=app.sessions.load('research');before=state.interactions[0]
    evidence=before.admitted['raw_resource_evidence']
    declared=FragmentsDeclaration(source='source.0',species='human',assembly='hg38',resource='resource.full',
        resource_preference=dict(literal=literal,start=0,end=len(literal)))
    missing,selected=requirements(captured,declared,resources(choices,'fragments'),evidence)
    assert not missing and selected.input_set_id=='full'
    # No textual alias or source replacement was needed for the interpreted selection.
    assert app.sessions.load('research').interactions[0]==before
    assert not state.turns and not state.revisions
    invalid=replace(declared,resource='resource.not-offered')
    missing,selected=requirements(captured,invalid,resources(choices,'fragments'),evidence)
    assert missing and selected is None



def test_rejected_parameter_clarification_does_not_discard_pending_raw_task(inputs):
    app,planner,captured=task_application(inputs)
    choices=(inputs.full,inputs.component('release-2','peaks',inputs.bed2))
    app.sessions.respond('research','first','Build a matrix from human hg38 fragments.',interpreter=TaskModel(),
        submission=dict(execution_inputs=dict(captured.execution_inputs),registered_input=captured.attribution()),input_sets=choices)
    before=app.sessions.load('research')
    class AmbiguousModel:
        def complete(self,**kwargs):
            return json.dumps(dict(turn_schema_version=1,decision=dict(kind='clarify',reason='ambiguous_parameter',choices=['reference.version'],value_required=True)))
    failed=app.sessions.respond('research','failed','Use hg38.',interpreter=AmbiguousModel(),input_sets=choices)
    assert failed.status=='failed'
    after=app.sessions.load('research')
    assert after.interactions[-1].prerequisite==before.interactions[-1].prerequisite
    assert after.interactions[0]==before.interactions[0]
    assert not after.turns and not after.revisions and planner.calls==1



def test_clicked_reference_uses_exact_pending_identity_without_language_interpretation(inputs):
    from agent.application.resource_continuation import selected_answer
    from agent.application.session_state import Interaction
    from agent.application.turn_decisions import RoleAnswer
    app,planner,captured=task_application(inputs)
    choices=(inputs.full,inputs.component('release-2','peaks',inputs.bed2))
    app.sessions.respond('research','first','Build a matrix from human hg38 fragments.',interpreter=TaskModel(),
        submission=dict(execution_inputs=dict(captured.execution_inputs),registered_input=captured.attribution()),input_sets=choices)
    before=app.sessions.load('research');question=before.interactions[0];option=question.admitted['resource_options'][0]
    follow=Interaction('reply',option['label'],None,0,
        dict(prerequisite=dict(question.prerequisite,predecessor_turn_id='first')),
        submission=dict(clarification_choice=dict(question_turn_id='first',handle=option['handle'])))
    answer=selected_answer(follow,before.interactions)
    assert isinstance(answer,RoleAnswer) and answer.fragments.resource==option['handle']
    assert answer.fragments.resource_preference is None
    from agent.application.raw_handoff import merge
    previous=FragmentsDeclaration(**dict(question.admitted['fragments']))
    continued=merge(previous,answer.fragments,pending=True)
    assert continued.source==previous.source and continued.assembly==previous.assembly
    for changed in (dict(question_turn_id='other-question',handle=option['handle']),
                    dict(question_turn_id='first',handle='resource.not-offered')):
        with pytest.raises(IntentError):
            selected_answer(replace(follow,submission=dict(clarification_choice=changed)),before.interactions)
    assert selected_answer(replace(follow,submission=None,utterance='Use the reference matching my data'),before.interactions) is None
    assert app.sessions.load('research')==before and not before.turns and not before.revisions


def test_finite_choice_dispatch_also_uses_existing_resource_answer():
    from agent.application.resource_continuation import selected_answer
    from agent.application.session_state import Interaction
    from agent.application.turn_decisions import ResourceAnswer
    reference=dict(field='resources',origin_turn_id='question',binding_turn_id='question',predecessor_turn_id='question')
    question=SimpleNamespace(turn_id='question',admitted=dict(resource_options=[dict(handle='resource.exact',label='Exact offered choice')]))
    received=Interaction('answer','Exact offered choice',None,0,dict(prerequisite=reference),
        submission=dict(clarification_choice=dict(question_turn_id='question',handle='resource.exact')))
    answer=selected_answer(received,(question,))
    assert isinstance(answer,ResourceAnswer) and answer.pending=='@resources' and answer.choice=='resource.exact'


@pytest.mark.parametrize('tool,function', [('run_snapatac2','pp.make_gene_matrix'), ('run_episcanpy','tl.geneactivity')])
def test_gene_activity_uses_selected_managed_annotation_only(inputs,tool,function):
    other=inputs.root/'other.gtf';other.write_text(inputs.gtf.read_text().replace('fixture','HAVANA'))
    selected=inputs.component('selected-genes','gene_anno',other)
    p=AgentPlan('activity','activity','Native gene activity',(
        PlanStep('activity',tool,dict(function_name=function,input_path=str(inputs.source),output_dir=str(inputs.root/'activity'))),))
    bound=bind_plan(p,receipt(inputs,(inputs.genes,selected),selected='selected-genes'))
    assert bound.steps[0].arguments['gene_anno_path']==str(other)
    assert bound.steps[0].arguments['gene_anno_sha256']==pin(other)['sha256']
    assert 'chrom_sizes_path' not in bound.steps[0].arguments
    with pytest.raises(ResourcePrerequisite): bind_plan(p,receipt(inputs,(inputs.genome,)))
    with pytest.raises(ResourcePrerequisite): bind_plan(p,receipt(inputs,(inputs.genes,selected)))
