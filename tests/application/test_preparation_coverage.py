"""Selected-owner conditional prerequisites; no scientific execution/providers."""
from dataclasses import replace
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from agent.orchestration.scientific_inputs import ScientificInputRequired,ScientificDiagnostic
from agent.tools.analysis import sequence_prediction as sequence,sequence_optimization,sequence_generation,scatac_regulatory_sequences as regions,replicate_pseudobulk as features,chromvar_native,differential_accessibility as da
from test_preparation_planned_contracts import context


def selected(tool):return context(tool=tool)


def test_sequence_alternatives_and_qualified_conditions(tmp_path,monkeypatch):
    ctx=selected('predict_regulatory_sequences')
    with pytest.raises(ScientificInputRequired) as missing:sequence.prepare_operation({},ctx)
    assert missing.value.requirements[0].argument=='sequence_path'
    assert missing.value.requirements[0].kind=='external_input'
    with pytest.raises(ScientificDiagnostic,match='exactly one'):sequence.prepare_operation(dict(sequence='ACGT',sequence_path='/any'),ctx)
    entries=[dict(kind='sequence_predictor',resource=dict(family='DHS64',conditions=['B cell'])),dict(kind='sequence_predictor',resource=dict(family='DeepPBMC',conditions=['T cell'])),dict(kind='sequence_generator',resource=dict(family='DNA-Diffusion',conditions=['K562']))]
    monkeypatch.setattr(sequence,'library',lambda:type('Library',(),dict(entries=lambda _:entries))())
    assert sequence.prepare_operation(dict(sequence='ACGT',condition='B cell'),ctx)['condition']=='B cell'
    with pytest.raises(ScientificDiagnostic):sequence.prepare_operation(dict(sequence='ACGT',condition='invented'),ctx)
    with pytest.raises(ScientificInputRequired) as target:sequence_optimization.prepare_operation(dict(sequence='ACGT'),selected('optimize_regulatory_sequences'))
    assert target.value.requirements[0].choices==('T cell',)
    with pytest.raises(ScientificInputRequired) as generated:sequence_generation.prepare_operation({},selected('generate_regulatory_sequences'))
    assert generated.value.requirements[0].choices==('K562',)
    entries.clear()
    with pytest.raises(ScientificDiagnostic,match='configured'):sequence.prepare_operation(dict(sequence='ACGT'),ctx)


def test_region_mode_requirements_retain_publication_authority():
    ctx=selected('prepare_scatac_regulatory_sequences')
    with pytest.raises(ScientificInputRequired) as missing:regions.prepare_operation({},ctx)
    assert {r.argument for r in missing.value.requirements}=={'input_path','species','assembly','coordinate_system','matrix_semantics'}
    for method,port in [('differential_accessibility','differential_accessibility'),('accepted_candidates','regions')]:
        with pytest.raises(ScientificInputRequired) as missing:regions.prepare_operation(dict(selection_method=method),ctx)
        assert all(r.kind=='resource' and r.port==port for r in missing.value.requirements)
        assert not any(r.applicability.get('source_input_types') for r in missing.value.requirements)


def test_actual_feature_layers_coordinates_and_native_bias(tmp_path):
    path=tmp_path/'input.h5ad';data=ad.AnnData(np.eye(4),var=pd.DataFrame(dict(real_bias=[.1,.2,.3,.4]),index=['p1','p2','p3','p4']))
    data.layers['raw']=data.X.copy();data.write_h5ad(path)
    with pytest.raises(ScientificInputRequired) as missing:features.prepare_feature_space(dict(input_path=str(path),matrix_source='layer',coordinate_source='var_columns'),selected('validate_scATAC_feature_space'))
    needs={r.argument:r for r in missing.value.requirements}
    assert needs['layer_key'].choices==('raw',) and needs['feature_chrom_key'].choices==('real_bias',)
    with pytest.raises(ScientificDiagnostic):features.prepare_feature_space(dict(matrix_source='X',layer_key='raw'),selected('validate_scATAC_feature_space'))
    with pytest.raises(ScientificInputRequired) as missing:features.prepare_pseudobulk(dict(group_source='verified_annotation',group_key='predicted_label'),selected('build_replicate_pseudobulk'))
    assert missing.value.requirements[0].port=='group_annotation'
    ctx=selected('run_chromvar');args=dict(input_path=str(path),function_name='getBackgroundPeaks')
    with pytest.raises(ScientificInputRequired) as missing:chromvar_native.prepare_operation(args,ctx)
    assert missing.value.requirements[0].choices==('real_bias',)
    assert chromvar_native.prepare_operation(dict(args,bias_key='real_bias'),ctx)
    with pytest.raises(ScientificDiagnostic,match='z_scores'):chromvar_native.prepare_operation(dict(args,function_name='computeVariability'),ctx)


def test_aggregate_offers_observe_only_selected_population(tmp_path):
    path=tmp_path/'aggregate.h5ad';data=ad.AnnData(np.eye(4),obs=pd.DataFrame(dict(group=['A','A','B','B'],condition=['treated','control','x','y']),index=['a','b','c','d']));data.write_h5ad(path)
    ctx=selected('run_replicate_differential_accessibility')
    with pytest.raises(ScientificInputRequired) as missing:da.prepare_operation(dict(pseudobulk_path=str(path),group_value='A'),ctx)
    assert {r.choices for r in missing.value.requirements}=={('treated','control')}
    with pytest.raises(ScientificDiagnostic):da.prepare_operation(dict(pseudobulk_path=str(path),group_value='A',numerator_condition='x'),ctx)


def test_accepted_only_requirement_uses_exact_registry_channel():
    from types import SimpleNamespace
    from agent.schemas import PriorOutputBinding,PriorOutputRef
    from agent.orchestration.active_context import ActivePlanningContext,ActiveContextItem
    binding=PriorOutputBinding('accepted-run','da','run_replicate_differential_accessibility','differential_accessibility',
        'differential_accessibility.v1','a'*64,'b'*64,'c'*64,'accepted-file.v1')
    # A controlled context witness tests transport; real Session projection owns
    # publication authentication, independently covered by source contract tests.
    sessions=SimpleNamespace(_interaction_session_id='s',planning_context_for_interaction=lambda *a,**k:(ActivePlanningContext('s','r',1,(ActiveContextItem('ctx.0',binding),)),{}))
    ctx=replace(selected('prepare_scatac_regulatory_sequences'),sessions=sessions,interaction=SimpleNamespace(turn_id='answer'))
    with pytest.raises(ScientificInputRequired) as missing:regions.prepare_operation(dict(selection_method='differential_accessibility'),ctx)
    req,=missing.value.requirements;offer,=req.offers
    assert offer['accepted_binding']==binding.to_dict() and not offer.get('source_input')
    answered=replace(ctx,answers={'da_path':offer})
    values=regions.prepare_operation(dict(selection_method='differential_accessibility'),answered)
    assert values['da_path']==PriorOutputRef(binding,'da_path')
    forged=dict(offer,accepted_binding=dict(offer['accepted_binding'],run_id='another-run'))
    with pytest.raises(ScientificDiagnostic,match='captured'):regions.prepare_operation(dict(selection_method='differential_accessibility'),replace(ctx,answers={'da_path':forged}))


def test_observation_digest_metadata_does_not_change_acceptance_mode():
    registry=selected('run_scanpy').registry
    for tool,name,field in [('validate_scATAC_feature_space','feature_space','feature_space_sha256'),('build_replicate_pseudobulk','pseudobulk','pseudobulk_sha256')]:
        port=next(p for p in registry.get(tool).semantic_planning.producer_ports if p.name==name)
        assert port.observation_sha256_field==field and port.accepted_file_sha256_field is None


def test_pseudobulk_fields_follow_exact_feature_owner_source_without_requalification(tmp_path,monkeypatch):
    path=tmp_path/'counts.h5ad'
    data=ad.AnnData(np.eye(3,dtype='int64'),obs=pd.DataFrame(dict(donor=['a','a','b'],condition=['x','y','x'],groups=['T','B','T']),index=['a','b','c']))
    data.write_h5ad(path)
    result=features.validate_scATAC_feature_space(path,tmp_path/'features',matrix_source='X',matrix_semantics='fragment_counts',species='human',genome_assembly='hg38',coordinate_source='none')
    monkeypatch.setattr(features,'_snapshot_feature_space',lambda *a,**k:pytest.fail('Preparation must not requalify counts'))
    ctx=selected('build_replicate_pseudobulk')
    with pytest.raises(ScientificInputRequired) as missing:features.prepare_pseudobulk(dict(feature_space_path=result['feature_space_path'],group_source='raw_obs'),ctx)
    assert {r.argument for r in missing.value.requirements}=={'replicate_key','condition_key','group_key'}
    assert all(set(r.choices)=={'donor','condition','groups'} for r in missing.value.requirements)
    with path.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError):features.prepare_pseudobulk(dict(feature_space_path=result['feature_space_path'],group_source='raw_obs'),ctx)
