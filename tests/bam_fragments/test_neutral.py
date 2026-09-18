"""M14.9 hand-derived neutral BAM acceptance and closed compatibility gates."""
import gzip
import json
from pathlib import Path

import pytest

from .conftest import pair, sha
from agent.tools.data import bam_fragment_manifest as m, neutral_bam as neutral
from agent.tools.data import bam_fragments as production, bam_fragments_verifier as verifier
from agent.tools.data import scatac_bam_fragments as owner, regulatory_feature_reference as ref
from agent.tools.data import primary_contigs, scatac_reference as legacy
from agent.tools.data.scatac_qc_reference import QCContig
from agent.tools.data._external_fragment_io import resource
from agent.tools.data._fragments_common import canonical
from agent.tools.data.authority_context import VerificationContext, authority_operation
from agent.tools.data.scatac_fragments_v2_verifier import FragmentVerificationRuntime


@pytest.fixture
def neutral_factory(bam_factory):
    def make(records=None, **kwargs):
        if records is None:
            records = pair('a') + pair('duplicate',a={'flag':1123},b={'flag':1171})
            records += pair('b',a={'cb':'B-2','pos':100,'mpos':160,'tlen':80},b={'cb':'B-2','pos':160,'mpos':100,'tlen':-80})
            records += pair('excluded',a={'rid':1,'mrid':1},b={'rid':1,'mrid':1})
        old = bam_factory(records,reference_names=('nuclear','mito'),**kwargs)
        root = Path(old['output_dir']).parent
        _, base, _ = legacy.load_scatac_reference_bundle(old['reference_bundle_path'])
        bed = root/'neutral.bed'; bed.write_text('nuclear\t0\t80\nnuclear\t100\t180\n')
        pointer = ref.publish_regulatory_feature_reference(ref.build_regulatory_feature_reference(
            species={'scientific_name':'Danio rerio','taxonomy_id':7955},target_assembly='synthetic-M14.9',
            fasta_path=base.genome.fasta.path,fai_path=base.genome.fai.path,
            feature_bed_path=bed,feature_category='regulatory_regions'),root/'neutral-reference.json')
        review = root/'review.txt'; review.write_text('Explicit synthetic contig classification.')
        scope = primary_contigs.publish_scope(reference_path=pointer['manifest_path'],reference_sha256=pointer['manifest_sha256'],
            classifications=(QCContig('nuclear',1000,'primary_nuclear_qc'),QCContig('mito',1000,'mitochondrial')),
            classification_source=resource(review),extra_exclusions={},output_path=root/'scope.json')
        spec = dict(contract_version=neutral.CONTRACT,source=resource(old['source_path']),source_index=None,
            source_species={'scientific_name':'Danio rerio','taxonomy_id':7955},source_assembly='synthetic-M14.9',
            source_reference=resource(pointer['manifest_path']),primary_scope=scope,
            library=dict(neutral.LIBRARY,namespace='explicit_library',source_library_id=None),source_history=m.HISTORY)
        args = dict(input_spec_path=str(root/'inputs.json'),input_spec_sha256='',source_profile=m.NEUTRAL_PROFILE_ID,
                    output_dir=str(root/'neutral-out'))
        save(args,spec)
        return args,spec
    return make


def save(args,spec):
    Path(args['input_spec_path']).write_bytes(canonical(spec));args['input_spec_sha256']=sha(args['input_spec_path'])


def execute(args,identity=None):
    return neutral.prepare_neutral_bam_fragments(**{k:v for k,v in args.items() if k!='source_profile'},execution_identity=identity)


def contents(result):
    path=Path(result['manifest_path']);value=json.loads(path.read_bytes());entry=value['libraries'][0]
    record=m.load_record(entry['provenance']['producer_record']['path'],entry['provenance']['producer_record']['sha256'])
    return gzip.decompress((path.parent/entry['bgzf']['path']).read_bytes()).decode().splitlines(),record


def test_hand_derived_acceptance_authority_recovery_composition(neutral_factory,monkeypatch,tmp_path):
    from agent.tools.data import scientific_authority, explicit_cells, fragment_feature_matrix as matrix
    import anndata as ad
    args,spec=neutral_factory();context=VerificationContext();calls=[]
    original=verifier.reconstruct
    def counted(*a,**k): calls.append(1); return original(*a,**k)
    monkeypatch.setattr(verifier,'reconstruct',counted)
    with authority_operation(context):
        result=execute(args,'e'*64)
        rows,record=contents(result)
        assert rows==['nuclear\t14\t75\taCgT-1\t2','nuclear\t104\t175\tB-2\t1']
        assert record['qualification']['exclusions']['non_primary']==1
        assert result['eligible_pairs']==3 and result['n_templates']==4
        authority=scientific_authority.issue(context,'prepare_neutral_bam_fragments',args,result,'e'*64)
        assert authority.record['schema_version']==2
        assert authority.record['producer_qualification']['profile_id']==m.NEUTRAL_PROFILE_ID
        def forbidden(*a,**k): raise AssertionError('repeated BAM science')
        monkeypatch.setattr(production,'generate',forbidden)
        assert owner.recover_bam_fragments(args,'e'*64)==result
        monkeypatch.setattr(verifier,'reconstruct',forbidden)
        cells=explicit_cells.publish_explicit_cells(cells=[('explicit_library','B-2'),('explicit_library','aCgT-1')],
            declaration='Explicit synthetic cells, no calling.',output_dir=tmp_path/'cells')
        monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS','/usr/bin/bedtools')
        matrix_args=dict(fragments_manifest_path=result['manifest_path'],fragments_manifest_sha256=result['manifest_sha256'],
            explicit_cells_manifest_path=cells['manifest_path'],explicit_cells_manifest_sha256=cells['manifest_sha256'],
            reference_manifest_path=spec['source_reference']['path'],reference_manifest_sha256=spec['source_reference']['sha256'],
            output_dir=str(tmp_path/'matrix'))
        built=matrix.build_cell_by_features(**matrix_args)
        assert ad.read_h5ad(built['matrix_path']).X.toarray().tolist()==[[0,1],[1,0]]
        assert matrix.recover_matrix(matrix_args,None)==built
    assert calls==[1]
    # Fresh standalone recovery reconstructs once, never production.
    monkeypatch.setattr(verifier,'reconstruct',counted)
    assert neutral.recover_neutral_bam_fragments(args,'e'*64)==result
    assert calls==[1,1]
    (tmp_path/'acceptance.json').write_bytes(canonical(dict(arguments=args,result=result,
        expected_fragments=['nuclear\t14\t75\taCgT-1\t2','nuclear\t104\t175\tB-2\t1'],
        produced_fragments=rows,production_record=record,authority=authority.to_dict(),
        matrix_arguments=matrix_args,matrix_result=built,expected_matrix=[[0,1],[1,0]],
        bam_reconstructions_in_shared_operation=1,standalone_recovery_reconstructions=1)))


@pytest.mark.parametrize('field,value',[
    ('source_species',{'scientific_name':'Mus musculus','taxonomy_id':10090}),
    ('source_species',{'scientific_name':'Danio rerio','taxonomy_id':10090}),
    ('source_assembly','wrong'),('source_history',dict(m.HISTORY,coordinates='shifted')),
])
def test_explicit_source_mismatch(neutral_factory,field,value):
    args,spec=neutral_factory();spec[field]=value;save(args,spec)
    with pytest.raises(ValueError):execute(args)


@pytest.mark.parametrize('field,value',[('barcode_tag','CR'),('correction_policy','raw'),('assay','rna')])
def test_library_profile_closed(neutral_factory,field,value):
    args,spec=neutral_factory();spec['library'][field]=value;save(args,spec)
    with pytest.raises(ValueError):execute(args)


@pytest.mark.parametrize('mutation',['source','reference','fasta','fai','scope','review','inputs','bgzf','tabix','receipt'])
def test_mutation_invalidates_operation_reuse(neutral_factory,mutation):
    args,spec=neutral_factory()
    with authority_operation(VerificationContext()):
        result=execute(args)
        root=Path(result['manifest_path']).parent
        manifest=json.loads(Path(result['manifest_path']).read_bytes())
        _,reference,_=ref.load_regulatory_feature_reference(spec['source_reference']['path'])
        scope=json.loads(Path(spec['primary_scope']['path']).read_bytes())
        paths=dict(source=spec['source']['path'],reference=spec['source_reference']['path'],fasta=reference.genome.fasta.path,
            fai=reference.genome.fai.path,scope=spec['primary_scope']['path'],review=scope['classification_source']['path'],
            inputs=args['input_spec_path'],bgzf=root/manifest['libraries'][0]['bgzf']['path'],
            tabix=root/manifest['libraries'][0]['tabix']['path'],receipt=root.parent/'receipt.json')
        path=Path(paths[mutation]);path.write_bytes(path.read_bytes()+b'changed')
        with pytest.raises(ValueError):owner.recover_bam_fragments(args,None)


@pytest.mark.parametrize('records,reason',[
    (pair('secondary')+[dict(pair('secondary')[0],flag=355)],None),
    (pair('supplement')+[dict(pair('supplement')[0],flag=2147)],'split'),
    (pair('missing',a={'cb':None},b={'cb':None}),'missing_cb'),
    (pair('low',a={'mapq':29}),'mapq'),
])
def test_existing_pair_exclusions(neutral_factory,records,reason):
    args,_=neutral_factory(pair('anchor')+records);result=execute(args);rows,record=contents(result)
    assert result['eligible_pairs']==(1 if reason else 2)
    if reason: assert record['qualification']['exclusions'][reason]==1


@pytest.mark.parametrize('records',[
    pair('bad',a={'cb':'X'}),pair('bad',a={'mpos':61}),pair('bad')[:1],
    pair('bad',a={'pos':995}),pair('bad',a={'rid':1,'mrid':1,'cb':'X'},b={'rid':1,'mrid':1}),
])
def test_malformed_pairs_fail_even_non_primary(neutral_factory,records):
    args,_=neutral_factory(pair('anchor')+records)
    with pytest.raises(ValueError):execute(args)


def test_nonprimary_only_fails_empty(neutral_factory):
    args,_=neutral_factory(pair('excluded',a={'rid':1,'mrid':1},b={'rid':1,'mrid':1}))
    with pytest.raises(ValueError,match='EMPTY'):execute(args)


def test_independent_verifier_does_not_call_production(neutral_factory,tmp_path,monkeypatch):
    args,_=neutral_factory();stage=tmp_path/'stage';stage.mkdir()
    production.prepare_in_stage(args,stage,FragmentVerificationRuntime())
    def forbidden(*a,**k):raise AssertionError('production reused as verifier')
    for name in ('generate','pair','alignment','aggregate','header_check'):monkeypatch.setattr(production,name,forbidden)
    verifier.verify_bam_fragments(stage/'manifest.json',expected_sha256=sha(stage/'manifest.json'),runtime=FragmentVerificationRuntime())


@pytest.mark.parametrize('fault',['missing_start_shift','missing_end_shift','double_shift','wrong_mate',
    'wrong_cigar_end','wrong_barcode','dropped_pair','added_pair','wrong_support','endpoint_collapse',
    'include_mapq_29','exclude_mapq_30','strict_mapq','include_mapq_255'])
def test_rehashed_forged_fragments(neutral_factory,tmp_path,fault):
    from .test_verification import test_independent_verifier_rejects_forged_consistent_artifacts
    test_independent_verifier_rejects_forged_consistent_artifacts(lambda records:neutral_factory(records)[0],tmp_path,fault)


def test_production_scope_bug_caught_independently(neutral_factory,tmp_path,monkeypatch):
    args,_=neutral_factory();original=production.generate
    def faulty(binding,*a):return original(dict(binding,primary_contigs={'nuclear','mito'}),*a)
    monkeypatch.setattr(production,'generate',faulty)
    stage=tmp_path/'stage';stage.mkdir()
    production.prepare_in_stage(args,stage,FragmentVerificationRuntime())
    with pytest.raises(ValueError,match='CONSERVATION'):
        verifier.verify_bam_fragments(stage/'manifest.json',expected_sha256=sha(stage/'manifest.json'),runtime=FragmentVerificationRuntime())


def rewrite_bam(spec,header_edit=None,*,sorted_index=False):
    import pysam
    path=Path(spec['source']['path'])
    with pysam.AlignmentFile(path,'rb') as bam:
        header=bam.header.to_dict();records=list(bam)
    if header_edit:header_edit(header)
    temp=path.with_name('rewrite.bam')
    if sorted_index:records.sort(key=lambda r:(r.reference_id,r.reference_start))
    with pysam.AlignmentFile(temp,'wb',header=header) as bam:
        for record in records:bam.write(record)
    temp.replace(path);spec['source']=resource(path)
    if sorted_index:
        pysam.index(str(path));spec['source_index']=resource(str(path)+'.bai')


@pytest.mark.parametrize('change', ['correct','species','assembly','length','name','md5'])
def test_exact_header_reference_declarations(neutral_factory,change):
    import hashlib
    args,spec=neutral_factory()
    def edit(header):
        for sq in header['SQ']:
            sq.update(SP='Danio rerio',AS='synthetic-M14.9',M5=hashlib.md5(('A' if sq['SN']=='nuclear' else 'C').encode()*1000).hexdigest())
        if change=='species':header['SQ'][0]['SP']='Mus musculus'
        elif change=='assembly':header['SQ'][0]['AS']='wrong'
        elif change=='length':header['SQ'][0]['LN']=999
        elif change=='name':header['SQ'][0]['SN']='alias'
        elif change=='md5':header['SQ'][0]['M5']='0'*32
    rewrite_bam(spec,edit);save(args,spec)
    if change=='correct':assert execute(args)['eligible_pairs']==3
    else:
        with pytest.raises(ValueError,match='REFERENCE_MISMATCH|HEADER_INVALID'):execute(args)


@pytest.mark.parametrize('mode',['valid','corrupt','unrelated','mutation'])
def test_optional_index_binding(neutral_factory,mode):
    args,spec=neutral_factory();rewrite_bam(spec,sorted_index=True)
    if mode=='corrupt':
        Path(spec['source_index']['path']).write_bytes(b'wrong-index');spec['source_index']=resource(spec['source_index']['path'])
    if mode=='unrelated':
        _,other=neutral_factory(pair('only',a={'rid':1,'mrid':1},b={'rid':1,'mrid':1}))
        rewrite_bam(other,sorted_index=True);spec['source_index']=other['source_index']
    save(args,spec)
    if mode in ('corrupt','unrelated'):
        with pytest.raises(ValueError,match='INDEX_MISMATCH'):execute(args)
    else:
        with authority_operation(VerificationContext()):
            result=execute(args)
            assert owner.recover_bam_fragments(args,None)==result
            if mode=='mutation':
                path=Path(spec['source_index']['path']);path.write_bytes(path.read_bytes()+b'changed')
                with pytest.raises(ValueError):owner.recover_bam_fragments(args,None)


def test_public_matrix_ports_application_and_fresh_recovery(neutral_factory,monkeypatch,tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from fragment_feature_integration.test_public import Model,wire
    from agent.application import ResearchAgentApplication
    from agent.orchestration import AgentRequest,LLMPlanner,build_default_tool_registry
    from agent.tools.data import explicit_cells, cell_by_ccre_verifier, _cell_by_ccre_production
    import anndata as ad
    args,spec=neutral_factory();result=execute(args)
    cells=explicit_cells.publish_explicit_cells(cells=[('explicit_library','B-2'),('explicit_library','aCgT-1')],
        declaration='Explicit fixture.',output_dir=tmp_path/'cells')
    inputs=dict(fragments_manifest_path=result['manifest_path'],fragments_manifest_sha256=result['manifest_sha256'],
        explicit_cells_manifest_path=cells['manifest_path'],explicit_cells_manifest_sha256=cells['manifest_sha256'],
        reference_manifest_path=spec['source_reference']['path'],reference_manifest_sha256=spec['source_reference']['sha256'])
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS','/usr/bin/bedtools')
    assert len(build_default_tool_registry().names())==23
    run=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire()))).run(AgentRequest('neutral-bam','Build the exact fragment matrix.',inputs))
    assert run.status.value=='SUCCEEDED',run.error
    built=run.run_result.steps[0].result
    assert ad.read_h5ad(built['matrix_path']).X.toarray().tolist()==[[0,1],[1,0]]
    def forbidden(*a,**k):raise AssertionError('repeated scientific work')
    monkeypatch.setattr(verifier,'reconstruct',forbidden)
    monkeypatch.setattr(production,'generate',forbidden)
    monkeypatch.setattr(cell_by_ccre_verifier,'reconstruct',forbidden)
    monkeypatch.setattr(_cell_by_ccre_production,'construct_counts',forbidden)
    assert ResearchAgentApplication(tmp_path/'app').resume(run.run_id).status.value=='SUCCEEDED'
    path=Path(spec['source']['path']);data=path.read_bytes();path.write_bytes(data+b'changed')
    assert ResearchAgentApplication(tmp_path/'app').resume(run.run_id).status.value=='FAILED'
    path.write_bytes(data)
    assert ResearchAgentApplication(tmp_path/'app').resume(run.run_id).status.value=='SUCCEEDED'


def test_source_changes_during_transformation_reject_and_clean(neutral_factory,monkeypatch):
    args,spec=neutral_factory();original=production.generate
    def changed(*a,**k):
        result=original(*a,**k)
        path=Path(spec['source']['path']);path.write_bytes(path.read_bytes()+b'changed')
        return result
    monkeypatch.setattr(production,'generate',changed)
    with pytest.raises(ValueError):execute(args)
    assert not list(Path(args['output_dir']).glob('.bam-attempt-*'))
    assert not list(Path(args['output_dir']).glob('bam-fragments-*'))


def test_undeclared_adjacent_index_is_not_opened(neutral_factory):
    args,spec=neutral_factory()
    Path(spec['source']['path']+'.bai').write_bytes(b'not an input')
    assert execute(args)['eligible_pairs']==3


def test_target_vocabulary_cannot_broaden_primary_scope(neutral_factory,monkeypatch,tmp_path):
    from agent.tools.data import explicit_cells, fragment_feature_matrix as matrix
    args,spec=neutral_factory();result=execute(args)
    _,base,_=ref.load_regulatory_feature_reference(spec['source_reference']['path'])
    bed=tmp_path/'other.bed';bed.write_text('mito\t0\t100\n')
    other=ref.publish_regulatory_feature_reference(ref.build_regulatory_feature_reference(
        species=base.species,target_assembly=base.target_assembly,fasta_path=base.genome.fasta.path,
        fai_path=base.genome.fai.path,feature_bed_path=bed,feature_category='regulatory_regions'),tmp_path/'other.json')
    cells=explicit_cells.publish_explicit_cells(cells=[('explicit_library','B-2')],declaration='fixture',output_dir=tmp_path/'cells')
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS','/usr/bin/bedtools')
    with pytest.raises(ValueError,match='PRIMARY_CONTIG_SCOPE_INVALID'):
        matrix.build_cell_by_features(fragments_manifest_path=result['manifest_path'],fragments_manifest_sha256=result['manifest_sha256'],
            explicit_cells_manifest_path=cells['manifest_path'],explicit_cells_manifest_sha256=cells['manifest_sha256'],
            reference_manifest_path=other['manifest_path'],reference_manifest_sha256=other['manifest_sha256'],output_dir=tmp_path/'matrix')


def test_legacy_profile_identity_unchanged():
    assert m.sha_bytes(m.PROFILE_BYTES)=='79647fb6e37b89df21405765d058c20cdbe4a2e9b43df310da0c38fe9292a8e0'


def test_other_contigs_excluded_by_classification(neutral_factory):
    args,spec=neutral_factory()
    path=Path(spec['primary_scope']['path']);scope=json.loads(path.read_bytes())
    scope['contigs'][1]['classification']='other'
    scope['identity_sha256']=primary_contigs.digest({k:v for k,v in scope.items() if k!='identity_sha256'})
    path.write_bytes(canonical(scope));spec['primary_scope']=resource(path);save(args,spec)
    result=execute(args)
    assert result['eligible_pairs']==3 and contents(result)[1]['qualification']['exclusions']['non_primary']==1


def test_barcode_key_order_and_namespace(neutral_factory):
    args,_=neutral_factory(pair('a')+pair('B',a={'cb':'B-2'},b={'cb':'B-2'})+pair('A',a={'cb':'ACGT-1'},b={'cb':'ACGT-1'}))
    result=execute(args)
    assert result['namespace']=='explicit_library'
    assert contents(result)[0]==['nuclear\t14\t75\tACGT-1\t1','nuclear\t14\t75\tB-2\t1','nuclear\t14\t75\taCgT-1\t1']


@pytest.mark.parametrize('fault',['version','program_limit','md5_syntax','read_group_label'])
def test_reuses_m10_header_guards(neutral_factory,fault):
    args,spec=neutral_factory()
    def edit(header):
        if fault=='version':header['HD']['VN']='not-a-version'
        elif fault=='program_limit':header['PG']=[{'ID':str(i)} for i in range(129)]
        elif fault=='md5_syntax':header['SQ'][0]['M5']='not-an-md5'
        else:header['RG']=[{'ID':'bad\x01label'}]
    rewrite_bam(spec,edit);save(args,spec)
    with pytest.raises(ValueError,match='HEADER_INVALID'):execute(args)
