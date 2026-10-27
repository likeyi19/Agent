"""Simulated native outputs; real reference matching, publication and delivery."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import pytest
import anndata as ad
from agent.tools.analysis import annotation_corroboration as owner, marker_validation as knowledge
from agent.tools.analysis import annotation_contract as c, marker_annotation as science, scatac_annotation
from agent.application import ResearchAgentApplication, InteractiveAgentApplication, OutputSelection
from agent.orchestration import AgentPlan, PlanStep, AgentRequest
from agent.schemas import StepOutputRef
from agent.schemas.orchestration import _serialize
from test_external_entry import external_case, case


@pytest.fixture
def reference(external_case,tmp_path,monkeypatch):
    from agent.application.shared_resources import ScientificResourceLibrary
    args,_=external_case
    spec=c.specification(args['annotation_spec_path'],args['annotation_spec_sha256'])
    table=tmp_path/'reference.tsv'
    row=dict.fromkeys(knowledge.FIELDS,'None');row.update(species='Human',tissue_type='Tissue',cancer_type='Normal',
        cell_name='Type A',marker='G',Symbol='G',PMID='123',Title='Explicit synthetic reference')
    with table.open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=knowledge.FIELDS,delimiter='\t');w.writeheader();w.writerow(row)
    resource=knowledge.KnowledgeResource(str(table),science.sha256(table),str(table),science.sha256(table),
        'synthetic-fixture','simulated-v1','2026-10-27','synthetic')
    value=dict(schema=owner.RESOURCE,knowledge=asdict(resource),species='Human',context='synthetic context',
        tissues=['Tissue'],cancer_type='Normal',alignment={spec['signature_resource']['sha256']:{
            'A':[dict(external_type='Type A',relationship='broader',rationale='Explicit synthetic broad mapping')]}},
        alignment_provenance='Simulated comparison, not biological evidence')
    library=ScientificResourceLibrary(tmp_path/'library')
    entry=library.install('marker_reference',value,label='Synthetic reference',version='1',provenance='synthetic',usage='test')
    monkeypatch.setenv('AGENT_SCIENTIFIC_RESOURCE_ROOT',str(library.root))
    return value,library,entry


class Combined:
    def __init__(self,args):self.args=args
    def plan(self,request,registry):
        return AgentPlan('combined',request.request_id,'Explicit simulated annotation and database comparison',(
            PlanStep('annotation','annotate_scATAC_cell_types',self.args),
            PlanStep('reference',owner.TOOL,dict(annotation_manifest_path=StepOutputRef('annotation','manifest_path'),
                annotation_manifest_sha256=StepOutputRef('annotation','manifest_sha256'),
                output_dir=str(Path(self.args['output_dir']).parent/'reference')),depends_on=('annotation',))))


def test_combined_publication_and_recovery(external_case,reference,tmp_path,monkeypatch):
    args,calls=external_case
    app=ResearchAgentApplication(tmp_path/'app',planner=Combined(args))
    done=app.run(AgentRequest('combined','Annotate and compare with a reference database.',{}))
    assert done.status.value=='SUCCEEDED',done
    result=done.run_result.steps[-1].result
    assert result['state_counts']=={'corroborating':1,'not_applicable':1}
    assert result['n_records']==1 and len(calls)==1
    run=app.run_store.load(done.run_result.run_id)
    assert run.steps[-1].verification.artifact_authority is not None
    from agent.orchestration.verification_authority import _load_context
    from agent.tools.data.authority_context import authority_operation
    from agent.tools.analysis import external_marker_annotation
    monkeypatch.setattr(external_marker_annotation,'annotate_cell_groups',lambda **kw:pytest.fail('native replay'))
    with authority_operation(_load_context(app.run_store,run.run_id,'historical_verified_sources.v1')):
        assert owner.recover(run.steps[-1].resolved_arguments,
            run.steps[-1].verification.artifact_authority['execution_identity'])==_serialize(result)
    path=Path(result['evidence_path'])
    with path.open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
    assert rows[0]['PMID']=='123' and rows[0]['candidate']=='A' and rows[0]['relationship']=='broader'
    from agent.application.corroboration_delivery import validate_table
    manifest,_=owner.load(result['manifest_path'])
    with path.open('rb') as stream:validate_table(stream,manifest)
    path.write_text(path.read_text().replace('123','999'))
    with pytest.raises(ValueError):owner.verify_result(run.steps[-1].resolved_arguments,result)


def test_reference_mismatch_missing_and_ambiguity(external_case,reference,tmp_path,monkeypatch):
    value,library,entry=reference
    with pytest.raises(ValueError,match='unique compatible'):
        owner.resolve_reference(dict(species='mouse',context='synthetic context'))
    library.install('marker_reference',dict(value,alignment_provenance='Second independently selected mapping'),
        label='Second',version='2',provenance='synthetic',usage='test')
    with pytest.raises(ValueError,match='unique compatible'):
        owner.resolve_reference(dict(species='human',context='synthetic context'))
    monkeypatch.delenv('AGENT_SCIENTIFIC_RESOURCE_ROOT')
    with pytest.raises(ValueError,match='operator must deploy'):
        owner.resolve_reference(dict(species='human',context='synthetic context'))


@pytest.mark.parametrize('measurement,storage',[(c.EXTERNAL_SEMANTICS,'csr'),('fragment_counts','csr'),('peak_counts','dense'),('binary_accessibility','bool')])
def test_region_measurements_preserved(external_case,measurement,storage):
    args,calls=external_case
    data=ad.read_h5ad(args['source_path']);data.uns['measurement']=measurement
    if storage=='dense':data.X=data.X.toarray()  # tiny simulated fixture only
    if storage=='bool':data.X=data.X.astype(bool)
    data.write_h5ad(args['source_path']);args['source_sha256']=science.sha256(args['source_path'])
    path=Path(args['annotation_spec_path']);spec=json.loads(path.read_text())
    spec.update(method=c.REGION_PRIMARY,measurement=measurement);path.write_bytes(c.matrix.canonical(spec))
    args.update(annotation_method=c.REGION_PRIMARY,annotation_spec_sha256=science.sha256(path))
    result=scatac_annotation.execute_annotation(args)
    output=ad.read_h5ad(result['annotated_h5ad_path'])
    assert scatac_annotation._same_array(data.X,output.X)
    assert result['annotation_profile']==c.REGION_PRIMARY and len(calls)==1
    assert c.load_manifest(result['manifest_path'])['matrix_binding']['value_semantics']==measurement


@pytest.mark.parametrize('missing', [False, True], ids=['null', 'absent'])
@pytest.mark.parametrize('entry', ['public', 'durable', 'registry'])
def test_execution_requires_measurement_before_input_preparation(external_case, monkeypatch, missing, entry):
    from agent.tools.analysis import external_marker_annotation
    from agent.orchestration.registry import build_default_tool_registry
    args, calls = external_case
    path = Path(args['annotation_spec_path'])
    spec = json.loads(path.read_text())
    spec.update(method=c.REGION_PRIMARY, measurement=None)
    if missing:
        spec.pop('measurement')
    path.write_bytes(c.matrix.canonical(spec))
    args.update(annotation_method=c.REGION_PRIMARY, annotation_spec_sha256=science.sha256(path))
    if not missing:
        # Reusable resource qualification is not a completed execution declaration.
        assert c.specification(path, args['annotation_spec_sha256'])['measurement'] is None
    monkeypatch.setattr(external_marker_annotation, 'prepare',
        lambda *a, **kw: pytest.fail('Incomplete semantics reached input preparation'))
    if entry == 'durable':
        invoke = lambda: scatac_annotation.execute_annotation(args)
    else:
        function = (scatac_annotation.annotate_scATAC_cell_types if entry == 'public'
                    else build_default_tool_registry().get('annotate_scATAC_cell_types').function)
        invoke = lambda: function(**args)
    with pytest.raises(ValueError, match=None if missing else 'declared region measurement semantics'):
        invoke()
    assert not calls


def test_unknown_measurement_remains_a_conversational_prerequisite(external_case):
    from dataclasses import replace
    from types import SimpleNamespace
    from agent.application.raw_handoff import AnnotationDeclaration, _annotation_requirements
    from agent.tools.analysis.annotation_input import observe, resource_specification
    args, calls = external_case
    data = ad.read_h5ad(args['source_path'])
    data.uns.pop('measurement')
    data.write_h5ad(args['source_path'])
    sha = science.sha256(args['source_path'])
    observation = observe(args['source_path'], sha)
    binding = SimpleNamespace(members=[dict(source_sha256=sha)])
    declared = AnnotationDeclaration(source='source.0', context='synthetic context',
        grouping_provenance='Independent simulated groups')
    assert c.input_problems(None, None, None) == []
    pending, values, missing = _annotation_requirements(binding, declared,
        dict(group_column='supplied_groups'), observation)
    assert pending.measurement is None and len(missing) == 1
    assert 'measurement' in missing[0] and not calls
    completed, _, missing = _annotation_requirements(binding,
        replace(pending, measurement='fragment_counts'), values, observation)
    assert not missing
    base = c.specification(args['annotation_spec_path'], args['annotation_spec_sha256'])
    resolved, missing = resource_specification(base, completed, binding.members)
    assert not missing and resolved['measurement'] == 'fragment_counts'
    assert resolved['gene_resource'] == base['gene_resource']
    assert resolved['signature_resource'] == base['signature_resource']


def test_original_layer_with_transformed_x(external_case):
    from agent.tools.analysis.annotation_input import observe
    from agent.application.raw_handoff import AnnotationDeclaration, _annotation_requirements
    from types import SimpleNamespace
    args,calls=external_case
    data=ad.read_h5ad(args['source_path'])
    data.X=data.X.astype(float)*.125
    data.uns['measurement']='log_normalized_accessibility'
    data.uns['count_slot']='X'
    data.write_h5ad(args['source_path'])
    args['source_sha256']=science.sha256(args['source_path'])
    args['count_slot']='layers/raw_counts'
    observed=observe(args['source_path'],args['source_sha256'])
    binding=SimpleNamespace(members=[dict(source_sha256=args['source_sha256'])])
    declared=AnnotationDeclaration(source='source.0',species='human',assembly='hg38',
        measurement=c.EXTERNAL_SEMANTICS,context='synthetic context',grouping_provenance='Independent simulated groups')
    resolved,values,missing=_annotation_requirements(binding,declared,
        dict(count_slot=args['count_slot'],group_column='supplied_groups'),observed)
    assert not missing and resolved.measurement==c.EXTERNAL_SEMANTICS
    result=scatac_annotation.execute_annotation(args)
    output=ad.read_h5ad(result['annotated_h5ad_path'])
    assert scatac_annotation._same_array(data.X,output.X)
    assert scatac_annotation._same_array(data.layers['raw_counts'],output.layers['raw_counts'])
    assert len(calls)==1


def test_reference_summary_does_not_duplicate_full_records():
    from types import SimpleNamespace
    from agent.application.dialogue_evidence import detail_request, field_semantics
    from agent.orchestration.registry import build_default_tool_registry
    view=SimpleNamespace(source=SimpleNamespace(tool_name=owner.TOOL))
    assert detail_request(view,None) is None
    assert detail_request(view,'g1').section=='reference_associations'
    registry=build_default_tool_registry()
    assert 'not probabilities' in field_semantics(registry,owner.TOOL,'group_summary')['description']
    assert len(json.dumps(field_semantics(registry,owner.TOOL,'database'))) < 400


def test_web_library_scope_without_environment(reference, monkeypatch):
    from agent.application.shared_resources import scientific_resource_scope
    value, library, entry = reference
    monkeypatch.delenv('AGENT_SCIENTIFIC_RESOURCE_ROOT')
    with scientific_resource_scope(library):
        assert owner.resolve_reference(dict(species='human',context=value['context'])) == value
    with pytest.raises(ValueError, match='operator must deploy'):
        owner.resolve_reference(dict(species='human',context=value['context']))
