"""Focused owner/authority checks; synthetic outputs are not native qualification."""
from pathlib import Path
import pytest
from agent.tools.analysis import macs3 as owner


@pytest.fixture
def native(monkeypatch):
    calls=[]
    def run(parameters,stage):
        calls.append((stage/'input.frag').read_text())
        (stage/'peaks_peaks.narrowPeak').write_text('chr1\t0\t20\tpeaks_peak_1\t50\t.\t3\t7\t5\t10\nchr1\t30\t40\tpeaks_peak_2\t40\t.\t2\t6\t4\t5\n')
        (stage/'peaks_summits.bed').write_text('chr1\t10\t11\tpeaks_peak_1\t5\nchr1\t35\t36\tpeaks_peak_2\t4\n')
        (stage/'peaks_peaks.xls').write_text('# simulated\nchr\tstart\tend\tlength\tabs_summit\tpileup\t-log10(pvalue)\tfold_enrichment\t-log10(qvalue)\tname\nchr1\t1\t20\t20\t11\t9\t7\t3\t5\tpeaks_peak_1\nchr1\t31\t40\t10\t36\t6\t6\t2\t4\tpeaks_peak_2\n')
        return dict(completion='succeeded',version='simulated',parameters=parameters)
    monkeypatch.setattr(owner,'native',run)
    return calls


def args_for(factory,tmp_path):
    args,_,_=factory()
    return {k:v for k,v in args.items() if k.startswith(('fragments_','explicit_cells_'))}|dict(effective_genome_size=90,genome_size_basis='synthetic test scope',output_dir=str(tmp_path/'peaks'))


def test_population_unit_records_and_matrix_handoff(factory,native,tmp_path):
    from agent.tools.data.fragment_feature_tool import build_scATAC_cell_by_features
    import anndata as ad
    args=args_for(factory,tmp_path);result=owner.execute(args)
    assert result['n_peaks']==2 and result['n_barcodes']==3 and result['n_fragment_records']==7
    assert result['total_source_support']==7000
    assert all(line.endswith('\t1') for line in native[0].splitlines())
    matrix=build_scATAC_cell_by_features(**{k:v for k,v in args.items() if k.startswith('fragments_')},
        reference_manifest_path=result['reference_manifest_path'],reference_manifest_sha256=result['reference_manifest_sha256'],
        explicit_cells_manifest_path=result['cells_manifest_path'],explicit_cells_manifest_sha256=result['cells_manifest_sha256'],output_dir=str(tmp_path/'matrix'))
    a=ad.read_h5ad(matrix['matrix_path']);assert list(a.obs.barcode_identifier)==['B','A','Z']
    assert list(a.var_names)==['chr1:0-20','chr1:30-40']
    assert a.X.toarray().tolist()==[[1,1],[2,0],[0,0]] and a.uns['matrix_semantics']=='fragment_counts'
    assert owner.recover(args,None)==result and len(native)==1
    Path(result['reference_manifest_path']).write_text('changed')
    with pytest.raises(ValueError):owner.recover(args,None)


@pytest.mark.parametrize('change',[dict(effective_genome_size=0),dict(effective_genome_size=True),dict(effective_genome_size=101),dict(genome_size_basis=''),dict(qvalue=0),dict(qvalue=float('nan')),dict(exclude_contigs=['chrZ']),dict(library_namespace='other')])
def test_scientific_rejections_before_native(factory,native,tmp_path,change):
    with pytest.raises(ValueError):owner.execute(args_for(factory,tmp_path)|change)
    assert not native


def test_absent_population_and_namespace(factory,native,tmp_path):
    args=args_for(factory,tmp_path);args.pop('explicit_cells_manifest_path');args.pop('explicit_cells_manifest_sha256')
    p=tmp_path/'barcodes';p.write_text('absent\n')
    with pytest.raises(ValueError,match='absent'):owner.execute(args|dict(barcodes_path=str(p),barcodes_sha256=owner.file_closure([p])[0]['sha256']))
    assert not native


def test_accepted_delivery_ports_and_restart(factory,native,tmp_path):
    from agent.application import ResearchAgentApplication,OutputSelection
    from agent.application.workspace import ManagedWorkspace
    from agent.orchestration import AgentPlan,AgentRequest,PlanStep
    from agent.application.macs3_delivery import accepted_artifacts
    from agent.application.dialogue_evidence import figure_inventory
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    from agent.orchestration.prior_outputs import bind_output
    args=args_for(factory,tmp_path);root=tmp_path/'workspace'
    args['output_dir']=str(root/'runs'/ManagedWorkspace.run_digest('first:run')/'scientific'/'peaks')
    class Planner:
        def plan(self,request,registry):return AgentPlan('test',request.request_id,'Synthetic peak integration',(PlanStep('peaks',owner.TOOL,args),))
    app=ResearchAgentApplication(root,planner=Planner());app.sessions.create('peaks')
    state=app.sessions.run('peaks','first',AgentRequest('first','Call peaks',{}),
        tuple(OutputSelection(name,'peaks',field) for name,field in [('peaks','manifest_path'),('reference','reference_manifest_path'),('cells','cells_manifest_path')]),expected_generation=0)
    assert state.turn('first').status=='activated',app.run_store.load('first:run').errors
    for port,semantic in [('peaks','scatac_peaks.v1'),('reference','regulatory_feature_reference.v1'),('cells','scatac_explicit_cells.v1')]:
        assert bind_output(app.run_store,app.registry,'first:run','peaks',port).semantic_type==semantic
    revision=state.revisions[-1];turn=state.turn('first')
    assert len(figure_inventory(app.sessions,turn,revision))==1
    items=accepted_artifacts(app.sessions,'peaks',revision,turn,None,step_id='peaks');assert len(items)==7
    for item in items:
        prepared=MatrixDownloadPreparer().prepare(item)
        try:
            import hashlib
            delivered=b''.join(prepared.iter_chunks())
            assert delivered and len(delivered)==prepared.size_bytes
            assert hashlib.sha256(delivered).hexdigest()==prepared.sha256
        finally:prepared.close()
    assert ResearchAgentApplication(root,planner=Planner()).sessions.load('peaks').to_dict()==state.to_dict()
    assert len(native)==1


def test_empty_calls_publish_no_usable_reference(factory,native,tmp_path,monkeypatch):
    original=owner.native
    def empty(p,s):
        r=original(p,s)
        for n in ('peaks_peaks.narrowPeak','peaks_summits.bed'):(s/n).write_text('')
        (s/'peaks_peaks.xls').write_text('# native empty\nchr\tstart\tend\tlength\tabs_summit\tpileup\t-log10(pvalue)\tfold_enrichment\t-log10(qvalue)\tname\n')
        return r
    monkeypatch.setattr(owner,'native',empty)
    r=owner.execute(args_for(factory,tmp_path))
    assert r['n_peaks']==0 and not r['reference_manifest_path'] and r['native_statistics']['empty']


def test_fastq_and_bam_can_use_new_feature_vocabulary(tmp_path,monkeypatch):
    from barcode_qc.test_producer_neutrality import fastq_fixture,bam_fixture
    from agent.tools.data import regulatory_feature_reference as refs,explicit_cells
    from agent.tools.data.fragment_feature_tool import build_scATAC_cell_by_features
    from agent.tools.data.scatac_fragments_v2_verifier import file_sha256
    import anndata as ad
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS','/usr/bin/bedtools')
    tiny,fastq,bc=fastq_fixture(tmp_path,monkeypatch)
    bed=tmp_path/'new-peaks.bed';bed.write_text('chrTiny\t1000\t1105\nchrTiny\t3000\t3101\n')
    ref=refs.publish_regulatory_feature_reference(refs.build_regulatory_feature_reference(species={'scientific_name':'Homo sapiens','taxonomy_id':9606},target_assembly='hg38',fasta_path=tiny['fa'],fai_path=tiny['fai'],feature_bed_path=bed,feature_category='peak_set'),tmp_path/'new-reference.json')
    cells=explicit_cells.publish_explicit_cells(cells=[('library_0',bc)],declaration='Explicit fixture cells',output_dir=tmp_path/'chosen')
    for index, producer in enumerate((fastq, bam_fixture(tiny, bc))):
        path=Path(producer['manifest_path'])
        r=build_scATAC_cell_by_features(fragments_manifest_path=str(path),fragments_manifest_sha256=file_sha256(path),explicit_cells_manifest_path=cells['manifest_path'],explicit_cells_manifest_sha256=cells['manifest_sha256'],reference_manifest_path=ref['manifest_path'],reference_manifest_sha256=ref['manifest_sha256'],output_dir=str(tmp_path/f'new-matrix-{index}'))
        a=ad.read_h5ad(r['matrix_path']);assert a.shape==(1,2) and a.X.toarray().tolist()==[[1,1]]


def test_shared_reference_species_does_not_become_import_argument():
    from agent.application.raw_handoff import _fragments_values,FragmentsDeclaration
    class Resource:
        def inputs(self):return dict(species='human',reference_bundle_path='/reference',reference_bundle_sha256='a'*64)
    declared=FragmentsDeclaration(namespace='library',species='human',assembly='hg38',source_profile='10x-atac-fragments.v1')
    values=_fragments_values(declared,Resource(),{},None)
    assert 'species' not in values and values['namespace']=='library'


def test_selected_group_requires_exact_matrix_and_order(factory,native,tmp_path):
    from agent.tools.data.fragment_feature_tool import build_scATAC_cell_by_features
    from agent.tools.analysis.pca_embedding import pca_embed_cells
    from agent.tools.analysis.embedding_analysis import build_cell_neighbors,cluster_cells
    import anndata as ad
    args=args_for(factory,tmp_path);peaks=owner.execute(args)
    matrix=build_scATAC_cell_by_features(**{k:v for k,v in args.items() if k.startswith('fragments_')},
        reference_manifest_path=peaks['reference_manifest_path'],reference_manifest_sha256=peaks['reference_manifest_sha256'],
        explicit_cells_manifest_path=peaks['cells_manifest_path'],explicit_cells_manifest_sha256=peaks['cells_manifest_sha256'],output_dir=str(tmp_path/'matrix'))
    pca=pca_embed_cells(matrix['matrix_path'],tmp_path/'pca',n_components=1)
    neighbors=build_cell_neighbors(pca['embedding_path'],pca['cell_ids_path'],tmp_path/'neighbors',n_neighbors=2)
    clusters=cluster_cells(neighbors['analysis_path'],tmp_path/'clusters')
    label=str(ad.read_h5ad(clusters['analysis_path']).obs.leiden.iloc[0])
    grouped={k:v for k,v in args.items() if not k.startswith('explicit_cells_')}|dict(
        output_dir=str(tmp_path/'group-peaks'),matrix_manifest_path=matrix['manifest_path'],matrix_manifest_sha256=matrix['manifest_sha256'],
        group_analysis_path=clusters['analysis_path'],group_analysis_sha256=owner.file_closure([clusters['analysis_path']])[0]['sha256'],group_label=label)
    result=owner.execute(grouped)
    assert result['n_barcodes']==sum(ad.read_h5ad(clusters['analysis_path']).obs.leiden.astype(str)==label)
    changed=dict(grouped,output_dir=str(tmp_path/'bad-group'),fragments_manifest_sha256='0'*64)
    with pytest.raises(ValueError):owner.grouped_population(changed,None)


def test_multiple_libraries_require_explicit_selection(tmp_path,monkeypatch,native):
    from barcode_qc.test_producer_neutrality import fastq_fixture
    _,fragments,_=fastq_fixture(tmp_path,monkeypatch,namespaces=2)
    args=dict(fragments_manifest_path=fragments['manifest_path'],fragments_manifest_sha256=fragments['manifest_sha256'],
        effective_genome_size=10000,genome_size_basis='synthetic fixture',output_dir=str(tmp_path/'ambiguous'))
    assert owner.parameter_prerequisites(args)==('library_namespace',)
    assert owner.parameter_prerequisites(args|dict(library_namespace='library_0'))==()
    with pytest.raises(ValueError,match='namespace'):owner.execute(args)
    assert not native


def test_invalid_native_coordinates_do_not_publish(factory,native,tmp_path,monkeypatch):
    original=owner.native
    def invalid(parameters,stage):
        result=original(parameters,stage)
        path=stage/'peaks_peaks.narrowPeak'
        path.write_text(path.read_text().replace('chr1\t0\t20','chr1\t0\t101'))
        return result
    monkeypatch.setattr(owner,'native',invalid)
    args=args_for(factory,tmp_path)
    with pytest.raises(ValueError,match='coordinates'):owner.execute(args)
    assert not (Path(args['output_dir'])/'manifest.json').exists()


def test_native_command_must_belong_to_pinned_runtime(tmp_path,monkeypatch):
    import json
    runtime=tmp_path/'runtime';binary=runtime/'env/bin/another-command';binary.parent.mkdir(parents=True)
    binary.write_text('not a pinned executable')
    (runtime/'runtime.json').write_text(json.dumps(dict(version='3.1.0',files=[])))
    monkeypatch.setenv('AGENT_MACS3_EXECUTABLE',str(binary))
    with pytest.raises(ValueError,match='runtime changed'):owner.native({},tmp_path)
