"""Independent small numerical/coordinate checks; fixtures are simulated biology."""
import copy
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pysam
import pytest
from scipy import sparse

from agent.tools.analysis import scatac_regulatory_sequences as owner, sequence_prediction as shared
from agent.tools.data.fastq_verification_authority import file_closure


@pytest.fixture
def source(tmp_path, monkeypatch):
    genome = tmp_path/'genome.fa'
    genome.write_text('>chr1\n'+('ACGTTGCA'*1000)+'\n')
    pysam.faidx(str(genome))
    pins = file_closure([genome, str(genome)+'.fai'])
    resource = dict(assembly='hg38',species=dict(taxonomy_id=9606,scientific_name='Homo sapiens'),
        genome=dict(fasta=pins[0],fai=pins[1]))
    monkeypatch.setattr(owner,'resolve_genome',lambda assembly,species: (resource,dict(files=pins)) if (assembly,species)==('hg38','human') else (_ for _ in ()).throw(ValueError('incompatible reference')))
    matrix = np.array([[1,0,2,0],[0,3,1,0],[1,0,0,1],[0,1,0,1]])
    data = ad.AnnData(sparse.csr_matrix(matrix),obs=pd.DataFrame({'cell_type':['B','B','T','T']},index=['c1','c2','c3','c4']),
        var=pd.DataFrame(dict(chrom=['chr1']*4,start=[2000,2500,3000,3500],end=[2100,2600,3100,3600]),index=['p1','p2','p3','p4']))
    path=tmp_path/'counts.h5ad';data.write_h5ad(path)
    args=dict(input_path=str(path),output_dir=str(tmp_path/'result'),species='human',assembly='hg38',
        coordinate_source='var_columns',coordinate_system='zero_based_half_open',matrix_semantics='peak_counts',
        group_key='cell_type',group_value='B',top_n=2)
    return args,data,matrix,genome


@pytest.mark.parametrize('storage',['csr','csc','dense','layer'])
def test_detection_and_real_context(source,storage):
    args,data,matrix,genome=source
    if storage=='csc':data.X=sparse.csc_matrix(matrix)
    elif storage=='dense':data.X=matrix
    elif storage=='layer':
        data.layers['counts']=data.X.copy();data.X=sparse.csr_matrix(matrix.shape);args['count_slot']='layers/counts'
    data.write_h5ad(args['input_path'])
    args.update(window_size=2114,comparison_group='T')
    result=owner.execute(args)
    rows=shared.read(Path(result['manifest_path']).parent/'sequence-records.json')
    assert [r['feature_id'] for r in rows]==['p3','p1']
    for row in rows:
        i=row['feature_index']
        assert row['detection_fraction']==np.count_nonzero(matrix[:2,i])/2
        assert row['fraction_difference']==(np.count_nonzero(matrix[:2,i])-np.count_nonzero(matrix[2:,i]))/2
        assert row['length']==2114
        reference=genome.read_text().splitlines()[1]
        assert row['sequence']==reference[row['window_start']:row['window_end']]
        assert row['peak_end']-row['peak_start']==100
    assert result['selection_method']=='descriptive'
    assert all('FDR' not in r for r in result['candidate_examples'])
    assert owner.verify_preparation(result['manifest_path'],expected_sha256=result['manifest_sha256'])==result


def test_explicit_names_one_based_strand(source):
    args,data,_,genome=source
    data.var.index=['chr1:2001-2100','chr1:2501-2600','chr1:3001-3100','chr1:3501-3600']
    data.var['strand']=['-']*4;data.write_h5ad(args['input_path'])
    args={k:v for k,v in args.items() if k not in ('matrix_semantics','group_key','group_value')}
    args.update(selection_method='explicit',feature_ids='chr1:3001-3100',coordinate_source='feature_names',
        coordinate_system='one_based_closed',strand_key='strand')
    result=owner.execute(args);row=shared.read(Path(result['manifest_path']).parent/'sequence-records.json')[0]
    from Bio.Seq import Seq
    assert row['peak_start']==3000 and row['peak_end']==3100 and row['strand']=='-'
    assert row['sequence']==str(Seq(genome.read_text().splitlines()[1][3000:3100]).reverse_complement())
    assert result['group_context']=={}


@pytest.mark.parametrize('package', ['pysam', 'Bio'])
def test_missing_optional_dependency_fails_at_extraction_without_publication(source, monkeypatch, package):
    import builtins
    args, data, _, genome = source
    data.var['strand'] = ['-'] * len(data.var)
    data.write_h5ad(args['input_path'])
    args['strand_key'] = 'strand'
    original = Path(args['input_path']).read_bytes()
    real_import = builtins.__import__
    def missing(name, *args, **kwargs):
        if name.split('.')[0] == package:
            raise ModuleNotFoundError(f"No module named '{package}'", name=package)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    if package == 'Bio':
        from agent.tools.analysis.sequence_reference import extract
        with pysam.FastaFile(str(genome)) as fasta:
            for strand in ('+', '.'):
                assert extract(fasta, 'chr1', 0, 8, strand) == 'ACGTTGCA'
    with pytest.raises(ModuleNotFoundError) as error:
        owner.execute(args)
    assert error.value.name == package
    assert not Path(args['output_dir']).exists()
    assert Path(args['input_path']).read_bytes() == original


@pytest.mark.parametrize('change',[
    {'assembly':'mm10'}, {'species':'mouse'}, {'coordinate_system':None}, {'coordinate_source':'none'},
    {'group_value':'missing'}, {'group_key':'absent'}, {'comparison_group':'B'}, {'min_fraction':1.1},
    {'matrix_semantics':'normalized_signal'}, {'window_size':50}, {'window_size':9000}, {'max_fdr':.05},
    {'top_n':0}, {'top_n':True}, {'count_slot':'bad'}, {'input_sha256':'0'*64}])
def test_invalid_prerequisites(source,change):
    args,_,_,_=source;args.update(change)
    with pytest.raises((ValueError,TypeError,KeyError)):owner.execute(args)
    assert not Path(args['output_dir']).exists()


@pytest.mark.parametrize('defect',['duplicate_features','fractional','negative','binary','noncoordinate'])
def test_reject_bad_features_or_measurement(source,defect):
    args,data,matrix,_=source
    if defect=='duplicate_features':data.var.index=['p1']*4
    elif defect=='noncoordinate':args['coordinate_source']='feature_names'
    elif defect=='binary':args['matrix_semantics']='binary_accessibility'
    else:
        matrix=matrix.astype(float);matrix[0,0]=.5 if defect=='fractional' else -1;data.X=sparse.csr_matrix(matrix)
    data.write_h5ad(args['input_path'])
    with pytest.raises(ValueError):owner.execute(args)


def test_identical_intervals_distinct_features_and_tamper(source):
    args,data,_,_=source
    data.var.loc['p1',['start','end']]=[3000,3100];data.write_h5ad(args['input_path'])
    result=owner.execute(args)
    assert len(result['candidate_examples'])==2
    Path(result['sequences_path']).write_text('corrupt')
    with pytest.raises(ValueError):owner.verify_preparation(result['manifest_path'],expected_sha256=result['manifest_sha256'])


def test_candidate_index_preserves_identity(source):
    args,_,_,_=source;result=owner.execute(args)
    parsed,_,_=shared.records(dict(sequence_path=result['sequences_path'],sequence_sha256=result['sequences_sha256'],sequence_index=2))
    records=shared.read(Path(result['manifest_path']).parent/'sequence-records.json')
    assert len(parsed)==1 and parsed[0]['sequence']==records[1]['sequence'] and parsed[0]['sequence_id']=='sequence-2'
    for index in (0,3,True):
        with pytest.raises(ValueError):shared.records(dict(sequence_path=result['sequences_path'],sequence_sha256=result['sequences_sha256'],sequence_index=index))


def test_registry_channels_and_publication_restart(source):
    from agent.application import ResearchAgentApplication,OutputSelection
    from agent.application.workspace import ManagedWorkspace
    from agent.orchestration import AgentPlan,AgentRequest,PlanStep
    from agent.orchestration.prior_outputs import bind_output,channel_for
    from agent.application import sequence_delivery
    args,_,_,_=source;root=Path(args['output_dir']).parent/'workspace'
    args['output_dir']=str(root/'runs'/ManagedWorkspace.run_digest('first:run')/'scientific'/'prepare')
    class Planner:
        def plan(self,request,registry):return AgentPlan('test',request.request_id,'Prepare candidates',(PlanStep('prepare',owner.TOOL,args),))
    app=ResearchAgentApplication(root,planner=Planner());app.sessions.create('regions')
    state=app.sessions.run('regions','first',AgentRequest('first','Prepare candidate DNA',{}),
        (OutputSelection('DNA','prepare','sequences_path'),OutputSelection('Regions','prepare','manifest_path')),expected_generation=0)
    assert state.turn('first').status=='activated',app.run_store.load('first:run').errors
    binding=bind_output(app.run_store,app.registry,'first:run','prepare','sequences')
    for tool in ('predict_regulatory_sequences','optimize_regulatory_sequences'):
        assert channel_for(binding,tool,'sequences',app.registry)
    reopened=ResearchAgentApplication(root,planner=Planner()).sessions.load('regions')
    assert reopened.to_dict()==state.to_dict()
    result=app.run_store.load('first:run').steps[0].result
    manifest,_=shared.load(result['manifest_path'],result['manifest_sha256'])
    for pin in manifest['files']:
        if Path(pin['path']).name in sequence_delivery.FILES:
            with open(pin['path'],'rb') as stream:assert sequence_delivery.prepare(stream,Path(pin['path']).name)
    # Accepted region continuation must survive removal of the original input,
    # retaining observed evidence without re-reading cells or redoing selection.
    from agent.schemas.prior_output import PriorOutputRef
    parent=bind_output(app.run_store,app.registry,'first:run','prepare','regions')
    Path(args['input_path']).unlink()
    args.clear();args.update(selection_method='accepted_candidates',candidate_index=2,window_size=2114,
        candidate_manifest_path=PriorOutputRef(parent,'manifest_path'),
        candidate_manifest_sha256=PriorOutputRef(parent,'manifest_sha256'),
        output_dir=str(root/'runs'/ManagedWorkspace.run_digest('expand:run')/'scientific'/'prepare'))
    expanded=app.sessions.run('regions','expand',AgentRequest('expand','Expand the second accepted candidate to real reference context',{}),
        (OutputSelection('expanded','prepare','sequences_path'),),expected_generation=1,use_active_context=True)
    assert expanded.turn('expand').status=='activated',app.run_store.load('expand:run').errors
    row=app.run_store.load('expand:run').steps[0].result['candidate_examples'][0]
    assert row['sequence_id']=='sequence-2' and row['length']==2114 and row['group_cells']==2
    assert row['feature_id']==result['candidate_examples'][1]['feature_id']


def test_metadata_conflict_and_external_storage(source,tmp_path):
    args,data,_,_=source
    data.uns['assembly']='mm10';data.write_h5ad(args['input_path'])
    with pytest.raises(ValueError,match='conflicts'):owner.execute(args)
    del data.uns['assembly'];data.write_h5ad(args['input_path'])
    import h5py
    with h5py.File(args['input_path'],'a') as h:
        del h['X'];h['X']=h5py.ExternalLink('elsewhere.h5ad','X')
    with pytest.raises(ValueError,match='Indirect'):owner.execute(args)


def test_nonhuman_accepted_source_not_human_model(source):
    from agent.tools.data.authority_context import VerificationContext,authority_operation
    args,_,_,_=source;r=owner.execute(args)
    manifest,_=shared.load(r['manifest_path'],r['manifest_sha256'])
    manifest['provenance']['reference']['species']['taxonomy_id']=10090
    ctx=VerificationContext()
    ctx.accepted['fixture']=dict(record=dict(resources=dict(manifest_identity=manifest)))
    with authority_operation(ctx),pytest.raises(ValueError,match='nonhuman'):
        shared.records(dict(sequence_path=r['sequences_path'],sequence_sha256=r['sequences_sha256']))


def test_annotation_group_binding_requires_exact_source_and_order(source,tmp_path,monkeypatch):
    from agent.tools.analysis import scatac_annotation,annotation_contract
    args,data,_,_=source;pin=owner._pin(args['input_path'])
    root=tmp_path/'annotation';(root/'primary').mkdir(parents=True)
    cells=[dict(cell_id=c,primary_annotation='B') for c in data.obs_names]
    sha=shared.write(root/'primary/cells.json',cells)
    manifest=dict(matrix_binding=dict(source_sha256=pin['sha256']),sidecars={'primary/cells.json':dict(sha256=sha)})
    msha=shared.write(root/'manifest.json',manifest)
    monkeypatch.setattr(scatac_annotation,'verify_annotation',lambda *a,**k:None)
    monkeypatch.setattr(annotation_contract,'load_manifest',lambda *a:manifest)
    a=dict(annotation_manifest_path=str(root/'manifest.json'),annotation_manifest_sha256=msha)
    labels,_=owner._annotation_groups(a,data.obs_names,pin)
    assert labels.tolist()==['B']*4
    with pytest.raises(ValueError,match='ordered cell'):owner._annotation_groups(a,list(reversed(data.obs_names)),pin)
    with pytest.raises(ValueError,match='original matrix'):owner._annotation_groups(a,data.obs_names,dict(pin,sha256='0'*64))


def test_shared_feature_name_encoding_needs_declared_coordinates(source):
    from agent.tools.analysis.replicate_pseudobulk import validate_scATAC_feature_space
    args,data,_,_=source
    data.var.index=['chr1_2000_2100','chr1_2500_2600','chr1_3000_3100','chr1_3500_3600'];data.write_h5ad(args['input_path'])
    result=validate_scATAC_feature_space(args['input_path'],args['output_dir'],matrix_source='X',matrix_semantics='fragment_counts',
        species='human',genome_assembly='hg38',coordinate_source='feature_names',coordinate_system='zero_based_half_open')
    manifest=json.loads(Path(result['feature_space_path']).read_text())
    assert manifest['coordinates']['available'] and manifest['coordinates']['source']=='feature_names'


@pytest.mark.parametrize('kind', ['h5ad', 'attachment'])
def test_captured_h5ad_does_not_offer_a_dna_source_role(kind):
    from agent.application.local_resources import RegisteredInputGroup,group_choices
    from agent.orchestration import build_default_tool_registry
    binding=RegisteredInputGroup([dict(resource_id='local-'+'1'*64,record_sha256='2'*64,
        source_sha256='3'*64,source_path='/captured/source',label='Captured source',input_type=kind)],
        {},attachment_candidates=True)
    roles=group_choices(binding,build_default_tool_registry())['roles']
    assert any(r['port']=='sequences' for r in roles) == (kind!='h5ad')
    assert any(r['tool']==owner.TOOL for r in roles)


def test_metadata_delivery_removes_private_paths_without_changing_source(source):
    import io
    from agent.application.sequence_delivery import prepare
    args,_,_,_=source;result=owner.execute(args)
    path=Path(result['manifest_path']).parent/'metadata.json';original=path.read_bytes()
    delivered=prepare(io.BytesIO(original),'metadata.json')
    assert str(Path(args['input_path']).parent).encode() in original
    assert str(Path(args['input_path']).parent).encode() not in delivered
    metadata=json.loads(delivered)
    assert metadata['source_identity']['sha256']==result['source_identity']['sha256']
    assert metadata['candidates'][0]['feature_id']==result['candidate_examples'][0]['feature_id']
    assert path.read_bytes()==original


def test_selected_accepted_candidate_keeps_id_in_consumer(source):
    from agent.tools.data.authority_context import VerificationContext,authority_operation
    args,_,_,_=source;r=owner.execute(args)
    parent,_=shared.load(r['manifest_path'],r['manifest_sha256'])
    child=owner.execute(dict(selection_method='accepted_candidates',candidate_manifest_path=r['manifest_path'],
        candidate_manifest_sha256=r['manifest_sha256'],candidate_index=2,output_dir=args['output_dir']+'-selected'))
    manifest,_=shared.load(child['manifest_path'],child['manifest_sha256'])
    ctx=VerificationContext();ctx.accepted['fixture']=dict(record=dict(resources=dict(manifest_identity=manifest)))
    with authority_operation(ctx):
        rows,_,ref=shared.records(dict(sequence_path=child['sequences_path'],sequence_sha256=child['sequences_sha256']))
    assert len(rows)==1 and rows[0]['sequence_id']=='sequence-2'
    assert ref['genome']['fasta']['sha256']==parent['provenance']['reference']['genome']['fasta']['sha256']
