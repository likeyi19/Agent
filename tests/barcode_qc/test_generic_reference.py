"""Generic resource admission and shared QC/selection; no new metric engine."""
import hashlib
import json
from pathlib import Path

import pytest

from agent.tools.data import neutral_qc_reference as q, regulatory_feature_reference as ref
from agent.tools.data import primary_contigs, primary_fragment_preparation
from agent.tools.data import scatac_barcode_qc, scatac_cell_selection
from agent.tools.data import scatac_qc_reference as legacy
from agent.tools.data._external_fragment_io import resource
from agent.tools.data.neutral_fragment_import import import_primary_fragments
from agent.tools.data.authority_context import VerificationContext, authority_operation


@pytest.fixture
def generic_case(tmp_path, monkeypatch):
    monkeypatch.setenv('AGENT_QC_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC', '1')
    fa = tmp_path/'genome.fa'; fai = tmp_path/'genome.fa.fai'
    fa.write_text('>nuclear\n'+'A'*6001+'\n>organelle\n'+'A'*6001+'\n')
    fai.write_text('nuclear\t6001\t9\t6001\t6002\norganelle\t6001\t6022\t6001\t6002\n')
    bed = tmp_path/'features.bed'; bed.write_text('nuclear\t2900\t3100\nnuclear\t900\t1100\n')
    parent = ref.publish_regulatory_feature_reference(ref.build_regulatory_feature_reference(
        species={'scientific_name':'Danio rerio', 'taxonomy_id':7955}, target_assembly='synthetic-M14.10',
        fasta_path=fa, fai_path=fai, feature_bed_path=bed, feature_category='peak_set'), tmp_path/'reference.json')
    review = tmp_path/'review.txt'; review.write_text('Explicit synthetic resource review; not biological qualification.\n')
    scope = primary_contigs.publish_scope(reference_path=parent['manifest_path'], reference_sha256=parent['manifest_sha256'],
        classifications=(legacy.QCContig('nuclear',6001,'primary_nuclear_qc'), legacy.QCContig('organelle',6001,'mitochondrial')),
        classification_source=resource(review), extra_exclusions={}, output_path=tmp_path/'scope.json')
    tss = tmp_path/'tss.bed'; tss.write_text('nuclear\t3000\t3001\t+\nnuclear\t3000\t3001\t-\n')
    lineage = tmp_path/'lineage.tsv'; lineage.write_text('nuclear\t3000\t+\tg1\tt1\nnuclear\t3000\t+\tg1\tt2\nnuclear\t3000\t-\tg2\tt3\n')
    source = tmp_path/'annotation.txt'; source.write_text('Synthetic declared eligible protein-coding transcript TSSs only.\n')
    kwargs = dict(reference_path=parent['manifest_path'], reference_sha256=parent['manifest_sha256'],
        primary_scope_path=scope['path'], primary_scope_sha256=scope['sha256'], annotation_path=source,
        annotation_source='synthetic', annotation_release='M14.10', qualification_evidence_path=review,
        tss_path=tss, lineage_path=lineage, output_path=tmp_path/'qc-reference.json', synthetic=True)
    fragments_source = tmp_path/'fragments.tsv'
    fragments_source.write_text('nuclear\t1000\t1001\tA\t99\nnuclear\t2950\t3051\tA\t2\n'
        'nuclear\t2950\t3051\tB\t1\nnuclear\t4901\t5001\tA\t4\norganelle\t10\t20\tC\t2\n')
    preparation = primary_fragment_preparation.prepare_primary_fragments(source_path=str(fragments_source),
        source_sha256=resource(fragments_source)['sha256'], source_index_path=None, source_index_sha256=None,
        scope_path=scope['path'], scope_sha256=scope['sha256'], output_dir=tmp_path/'prepared')
    fragments = import_primary_fragments(preparation_path=preparation['path'], preparation_sha256=preparation['sha256'],
        reference_bundle_path=parent['manifest_path'], reference_bundle_sha256=parent['manifest_sha256'],
        namespace='lib', output_dir=tmp_path/'fragments')
    return kwargs, fragments, parent


def qc_args(case):
    kwargs, fragments, _ = case
    pointer = q.qualify_reference(**kwargs)
    return dict(fragments_manifest_path=fragments['manifest_path'], fragments_manifest_sha256=fragments['manifest_sha256'],
        qc_reference_manifest_path=pointer['manifest_path'], qc_reference_manifest_sha256=pointer['manifest_sha256'],
        output_dir=str(Path(pointer['manifest_path']).parent/'qc-output'))


def test_shared_qc_selection(generic_case, tmp_path, monkeypatch):
    args = qc_args(generic_case)
    with authority_operation(VerificationContext()):
        qc = scatac_barcode_qc.compute_scATAC_qc(**args)
        assert qc['contract_version'] == 'scatac-barcode-qc.v1'
        assert qc['n_observed_barcodes'] == 2 and qc['n_fragment_records'] == 4
        assert qc['tss_defined'] == 1
        def forbidden(*a, **kw): raise AssertionError('QC science repeated by selection')
        import agent.tools.data.barcode_qc_verifier as verifier
        monkeypatch.setattr(verifier, 'bind', forbidden)
        selected = scatac_cell_selection.select_scATAC_cells(barcode_qc_manifest_path=qc['manifest_path'],
            barcode_qc_manifest_sha256=qc['manifest_sha256'], min_qc_fragment_records=1,
            min_tss_enrichment=0, output_dir=str(tmp_path/'selected'))
        assert selected['n_selected'] == 1
        from agent.tools.data.fragment_feature_tool import build_scATAC_cell_by_features
        import anndata as ad
        monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS', '/usr/bin/bedtools')
        _, fragments, parent = generic_case
        matrix = build_scATAC_cell_by_features(fragments_manifest_path=fragments['manifest_path'],
            fragments_manifest_sha256=fragments['manifest_sha256'],
            selected_cells_manifest_path=selected['manifest_path'], selected_cells_manifest_sha256=selected['manifest_sha256'],
            reference_manifest_path=parent['manifest_path'], reference_manifest_sha256=parent['manifest_sha256'],
            output_dir=str(tmp_path/'matrix'))
        assert matrix['contract_version'] == 'scatac-cell-by-features.qc-selected.v1'
        assert ad.read_h5ad(matrix['matrix_path']).X.toarray().tolist() == [[1, 1]]


@pytest.mark.parametrize('mutation', ['outside','nonprimary','strand','duplicate','order','lineage','empty'])
def test_bad_normalization_fails_before_publication(generic_case, mutation):
    kwargs, _, _ = generic_case
    path = kwargs['lineage_path']
    if mutation == 'outside': path.write_text('nuclear\t6001\t+\tg\tt\n')
    elif mutation == 'nonprimary': path.write_text('organelle\t3000\t+\tg\tt\n')
    elif mutation == 'strand': path.write_text('nuclear\t3000\t.\tg\tt\n')
    elif mutation == 'duplicate': path.write_text('nuclear\t3000\t+\tg\tt\nnuclear\t3001\t+\tg\tt\n')
    elif mutation == 'order': path.write_text('nuclear\t3000\t-\tg2\tt2\nnuclear\t3000\t+\tg1\tt1\n')
    elif mutation == 'lineage': kwargs['tss_path'].write_text('nuclear\t3001\t3002\t+\n')
    else: kwargs['tss_path'].write_text('')
    with pytest.raises(ValueError): q.qualify_reference(**kwargs)
    assert not kwargs['output_path'].exists()


def test_production_requires_independent_catalog(generic_case):
    kwargs, _, _ = generic_case; kwargs['synthetic'] = False
    args = qc_args(generic_case)
    with pytest.raises(ValueError, match='QC_RESOURCE_UNQUALIFIED'):
        scatac_barcode_qc.compute_scATAC_qc(**args)


def test_qualified_runtime_never_reconstructs_normalization(generic_case, tmp_path, monkeypatch):
    kwargs, _, _ = generic_case; kwargs['synthetic'] = False
    args = qc_args(generic_case)
    _, bundle, _ = legacy.load_scatac_qc_reference_bundle(args['qc_reference_manifest_path'])
    entry = dict(resource_identity_sha256=bundle.resource_identity_sha256,
        parent_reference_identity_sha256=bundle.parent_reference_identity_sha256,
        annotation_sha256=bundle.annotation.resource.sha256, annotation_release=bundle.annotation.release,
        qualification_basis='synthetic-test-attestation-not-biological')
    catalog = tmp_path/'catalog.json'; catalog.write_text(json.dumps(dict(
        artifact_type='agent.qc-resource-qualification-catalog', schema_version=1, resources=[entry])))
    monkeypatch.setenv('AGENT_QC_RESOURCE_CATALOG', str(catalog))
    def forbidden(*a, **kw): raise AssertionError('normalization replay at runtime')
    monkeypatch.setattr(q, 'verify_normalized', forbidden)
    assert scatac_barcode_qc.compute_scATAC_qc(**args)['resource_qualification'] == 'operator_qualified'


@pytest.mark.parametrize('mutation', ['assembly','scope','tss','evidence'])
def test_resource_mutation_rejected(generic_case, mutation):
    args = qc_args(generic_case); kwargs, _, _ = generic_case
    if mutation in ('tss','evidence'):
        kwargs['tss_path' if mutation == 'tss' else 'qualification_evidence_path'].write_text('changed\n')
    else:
        p = Path(args['qc_reference_manifest_path']); value = json.loads(p.read_bytes())
        if mutation == 'assembly': value['assembly'] = 'other'
        else: value['primary_scope_identity_sha256'] = '1'*64
        value['resource_identity_sha256'] = q.identity(value)
        p.write_bytes(q.canonical(value)); args['qc_reference_manifest_sha256'] = resource(p)['sha256']
    with pytest.raises(ValueError): scatac_barcode_qc.compute_scATAC_qc(**args)


def test_public_selected_chain_and_authority_resume(generic_case, tmp_path, monkeypatch):
    from agent.application import ResearchAgentApplication
    from agent.orchestration import AgentRequest, LLMPlanner
    from agent.tools.data import _barcode_qc_production, _cell_selection_production, _cell_by_ccre_production
    args = qc_args(generic_case); _, _, parent = generic_case
    args.pop('output_dir')
    args.update(reference_manifest_path=parent['manifest_path'], reference_manifest_sha256=parent['manifest_sha256'],
                min_qc_fragment_records=1, min_tss_enrichment=0)
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS', '/usr/bin/bedtools')
    def inp(port, key): return dict(target=port, source=dict(kind='input', input=key))
    def upstream(port, step): return dict(target=port, source=dict(kind='step', step=step))
    steps = [
        dict(step_id='qc', tool='compute_scATAC_qc', control_dependencies=[], sources=[
            inp('fragments','fragments_manifest_path'), inp('qc_reference','qc_reference_manifest_path')]),
        dict(step_id='select', tool='select_scATAC_cells', control_dependencies=[], sources=[
            upstream('barcode_qc','qc'), inp('min_qc_fragment_records','min_qc_fragment_records'), inp('min_tss_enrichment','min_tss_enrichment')]),
        dict(step_id='matrix', tool='build_scATAC_cell_by_features', control_dependencies=[], sources=[
            inp('fragments','fragments_manifest_path'), upstream('selected_cells','select'), inp('reference','reference_manifest_path')]),
    ]
    class Model:
        model_id = 'scripted-shared-qc'
        def complete(self, *, prompt, response_schema):
            if 'selection_schema_version' in response_schema.get('properties', {}):
                return json.dumps(dict(selection_schema_version=1,decision=dict(kind='select',capability_ids=['raw_preprocessing','species_adaptation'])))
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=steps)))
    app = ResearchAgentApplication(tmp_path/'app', planner=LLMPlanner(Model()))
    run = app.run(AgentRequest('generic-qc','Compute QC, select with explicit thresholds and build the feature matrix.',args))
    assert run.status.value == 'SUCCEEDED', (run.error, run.run_result.errors)
    matrix = run.run_result.steps[-1]
    authority = matrix.verification.artifact_authority
    assert authority['schema_version'] == 2
    assert authority['upstream']['cells']['kind'] == 'selection'
    assert 'QC-selected candidates' in Path(run.report.path).read_text()
    def forbidden(*a, **kw): raise AssertionError('science during terminal resume')
    monkeypatch.setattr(_barcode_qc_production,'produce',forbidden)
    monkeypatch.setattr(_cell_selection_production,'produce',forbidden)
    monkeypatch.setattr(_cell_by_ccre_production,'construct_counts',forbidden)
    from agent.tools.data import barcode_qc_verifier, cell_selection_verifier, cell_by_ccre_verifier
    monkeypatch.setattr(barcode_qc_verifier,'bind',forbidden)
    monkeypatch.setattr(cell_selection_verifier,'canonical_thresholds',forbidden)
    monkeypatch.setattr(cell_by_ccre_verifier,'reconstruct',forbidden)
    assert ResearchAgentApplication(tmp_path/'app').resume(run.run_id).status.value == 'SUCCEEDED'
    generic_case[0]['qualification_evidence_path'].write_text('changed\n')
    assert ResearchAgentApplication(tmp_path/'app').resume(run.run_id).status.value == 'FAILED'


def test_distinct_valid_primary_scope_rejected(generic_case, tmp_path):
    kwargs, _, parent = generic_case
    other = tmp_path/'other-review.txt'; other.write_text('A different explicit synthetic review.\n')
    pointer = primary_contigs.publish_scope(reference_path=parent['manifest_path'], reference_sha256=parent['manifest_sha256'],
        classifications=(legacy.QCContig('nuclear',6001,'primary_nuclear_qc'), legacy.QCContig('organelle',6001,'mitochondrial')),
        classification_source=resource(other), extra_exclusions={}, output_path=tmp_path/'other-scope.json')
    kwargs.update(primary_scope_path=pointer['path'], primary_scope_sha256=pointer['sha256'])
    args = qc_args(generic_case)
    with pytest.raises(ValueError, match='QC_PRIMARY_SCOPE_MISMATCH'):
        scatac_barcode_qc.compute_scATAC_qc(**args)


@pytest.mark.parametrize('mode', ['missing', 'partial', 'both'])
def test_matrix_cell_choice_explicit_and_closed(mode):
    from agent.tools.data.fragment_feature_matrix import arguments
    args = dict(fragments_manifest_path='/x/fragments',fragments_manifest_sha256='a'*64,
        reference_manifest_path='/x/reference',reference_manifest_sha256='b'*64,output_dir='/x/output')
    if mode != 'missing': args['selected_cells_manifest_path'] = '/x/selected'
    if mode == 'both':
        args.update(selected_cells_manifest_sha256='c'*64,explicit_cells_manifest_path='/x/cells',explicit_cells_manifest_sha256='d'*64)
    with pytest.raises(ValueError,match='MATRIX_CELL_INPUT_REQUIRED'): arguments(args)


def test_primary_neutral_bam_uses_same_qc(generic_case, tmp_path):
    import pysam
    from agent.tools.data import neutral_bam, bam_fragment_manifest
    kwargs, _, parent = generic_case
    qc_pointer = q.qualify_reference(**kwargs)
    path = tmp_path/'source.bam'
    with pysam.AlignmentFile(path,'wb',header=dict(HD={'VN':'1.6'},SQ=[{'SN':'nuclear','LN':6001},{'SN':'organelle','LN':6001}])) as out:
        for query, start in (('flank',996), ('center',2996)):
            for flag, pos, mate, length in ((99,start,start+40,60),(147,start+40,start,-60)):
                a=pysam.AlignedSegment();a.query_name=query;a.flag=flag;a.reference_id=0;a.reference_start=pos
                a.mapping_quality=60;a.cigarstring='20M';a.next_reference_id=0;a.next_reference_start=mate
                a.template_length=length;a.query_sequence='A'*20;a.query_qualities=pysam.qualitystring_to_array('I'*20)
                a.set_tag('CB','A');out.write(a)
    _, reference, _ = ref.load_regulatory_feature_reference(parent['manifest_path'])
    spec = dict(contract_version=neutral_bam.CONTRACT,source=resource(path),source_index=None,
        source_species=reference.to_dict()['species'],source_assembly=reference.target_assembly,
        source_reference=resource(parent['manifest_path']),primary_scope=resource(kwargs['primary_scope_path']),
        library=dict(neutral_bam.LIBRARY,namespace='lib',source_library_id=None),source_history=bam_fragment_manifest.HISTORY)
    inputs=tmp_path/'bam-inputs.json';inputs.write_bytes(q.canonical(spec))
    with authority_operation(VerificationContext()):
        result=neutral_bam.prepare_neutral_bam_fragments(input_spec_path=str(inputs),input_spec_sha256=resource(inputs)['sha256'],output_dir=str(tmp_path/'bam-fragments'))
        result=scatac_barcode_qc.compute_scATAC_qc(fragments_manifest_path=result['manifest_path'],fragments_manifest_sha256=result['manifest_sha256'],
            qc_reference_manifest_path=qc_pointer['manifest_path'],qc_reference_manifest_sha256=qc_pointer['manifest_sha256'],output_dir=str(tmp_path/'bam-qc'))
        assert result['n_observed_barcodes']==1 and result['n_fragment_records']==2 and result['tss_defined']==1


def test_generic_reference_exceeds_legacy_contig_and_manifest_bounds(generic_case, tmp_path):
    kwargs, _, _ = generic_case
    fa=tmp_path/'large-dictionary.fa';fai=tmp_path/'large-dictionary.fa.fai'
    contigs=[legacy.QCContig('nuclear',6001,'primary_nuclear_qc')]
    contigs.extend(legacy.QCContig(f'unlocalized_{i:05d}',1,'other') for i in range(17000))
    offset=0
    with fa.open('w') as sequence, fai.open('w') as index:
        for c in contigs:
            header='>'+c.name+'\n';sequence.write(header+'A'*c.length+'\n');offset+=len(header)
            index.write(f'{c.name}\t{c.length}\t{offset}\t{c.length}\t{c.length+1}\n');offset+=c.length+1
    bed=tmp_path/'large-dictionary.bed';bed.write_text('nuclear\t2900\t3100\n')
    pointer=ref.publish_regulatory_feature_reference(ref.build_regulatory_feature_reference(
        species={'scientific_name':'Macaca fascicularis','taxonomy_id':9541},target_assembly='synthetic-large-dictionary',
        fasta_path=fa,fai_path=fai,feature_bed_path=bed,feature_category='regulatory_regions'),tmp_path/'large-reference.json')
    scoped=primary_contigs.publish_scope(reference_path=pointer['manifest_path'],reference_sha256=pointer['manifest_sha256'],
        classifications=contigs,classification_source=resource(kwargs['qualification_evidence_path']),extra_exclusions={},output_path=tmp_path/'large-scope.json')
    kwargs.update(reference_path=pointer['manifest_path'],reference_sha256=pointer['manifest_sha256'],
        primary_scope_path=scoped['path'],primary_scope_sha256=scoped['sha256'])
    result=q.qualify_reference(**kwargs)
    assert Path(result['manifest_path']).stat().st_size>legacy.MAX_MANIFEST_BYTES
    _, bundle, _=legacy.load_scatac_qc_reference_bundle(result['manifest_path'],expected_sha256=result['manifest_sha256'])
    assert len(bundle.contigs)==17001 and bundle.species.taxonomy_id==9541
    q.verify_integrity(bundle)


def test_rehashed_generic_resource_cannot_borrow_catalog_admission(generic_case, tmp_path, monkeypatch):
    kwargs, _, _=generic_case;kwargs['synthetic']=False
    args=qc_args(generic_case);p=Path(args['qc_reference_manifest_path']);value=json.loads(p.read_bytes())
    entry=dict(resource_identity_sha256=value['resource_identity_sha256'],
        parent_reference_identity_sha256=value['parent_reference_identity_sha256'],
        annotation_sha256=value['annotation']['resource']['sha256'],annotation_release=value['annotation']['release'],
        qualification_basis='synthetic-test-review')
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps(dict(artifact_type='agent.qc-resource-qualification-catalog',schema_version=1,resources=[entry])))
    monkeypatch.setenv('AGENT_QC_RESOURCE_CATALOG',str(catalog))
    kwargs['tss_path'].write_text('nuclear\t3100\t3101\t+\n')
    value['tss']=resource(kwargs['tss_path']);value['resource_identity_sha256']=q.identity(value)
    p.write_bytes(q.canonical(value));args['qc_reference_manifest_sha256']=resource(p)['sha256']
    with pytest.raises(ValueError,match='QC_RESOURCE_UNQUALIFIED'):scatac_barcode_qc.compute_scATAC_qc(**args)


def test_oversized_tss_rejected_before_row_scan(generic_case, monkeypatch):
    kwargs, _, _=generic_case
    monkeypatch.setattr(legacy,'MAX_SIDECAR_BYTES',1)
    def forbidden(*a, **kw): raise AssertionError('oversized TSS was scanned')
    monkeypatch.setattr(legacy,'_lines',forbidden)
    with pytest.raises(ValueError,match='QC_RESOURCE_LIMIT'):q.qualify_reference(**kwargs)


def test_tss_row_bound_fails_during_scan(generic_case, monkeypatch):
    kwargs, _, _=generic_case
    monkeypatch.setattr(legacy,'MAX_TSS_ROWS',1)
    with pytest.raises(ValueError,match='QC_RESOURCE_LIMIT'):q.qualify_reference(**kwargs)
