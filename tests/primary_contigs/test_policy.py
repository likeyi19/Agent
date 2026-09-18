import json
from pathlib import Path

import pytest

from agent.tools.data import primary_contigs as scope
from agent.tools.data import primary_fragment_preparation as prep
from agent.tools.data import regulatory_feature_reference as ref
from agent.tools.data.scatac_qc_reference import QCContig
from agent.tools.data._external_fragment_io import resource
from agent.tools.data._fragments_common import canonical
from agent.tools.data.neutral_fragment_import import import_primary_fragments


def setup_scope(tmp_path, *, feature='chr1', extras=None):
    fa=tmp_path/'genome.fa';fa.write_text('>chr1\n'+'A'*100+'\n>chrM\n'+'A'*100+'\n')
    fai=tmp_path/'genome.fa.fai';fai.write_text('chr1\t100\t6\t100\t101\nchrM\t100\t113\t100\t101\n')
    bed=tmp_path/'features.bed';bed.write_text(f'{feature}\t0\t20\n')
    reference=ref.publish_regulatory_feature_reference(ref.build_regulatory_feature_reference(
        species={'scientific_name':'Danio rerio','taxonomy_id':7955},target_assembly='test',
        fasta_path=fa,fai_path=fai,feature_bed_path=bed,feature_category='regulatory_regions'),tmp_path/'reference.json')
    review=tmp_path/'review.json';review.write_text('Explicit synthetic classification, not biological qualification.')
    pointer=scope.publish_scope(reference_path=reference['manifest_path'],reference_sha256=reference['manifest_sha256'],
        classifications=(QCContig('chr1',100,'primary_nuclear_qc'),QCContig('chrM',100,'mitochondrial')),
        classification_source=resource(review),extra_exclusions={'chrMT':'mitochondrial'} if extras is None else extras,
        output_path=tmp_path/'scope.json')
    return pointer,reference


def test_legacy_qc_projection_and_explicit_non_xy_species(tmp_path):
    pointer,_=setup_scope(tmp_path)
    value=scope.verify_scope(pointer['path'],pointer['sha256'])
    assert value['contigs'][0]['classification']=='primary_nuclear'
    assert value['contigs'][1]['classification']=='mitochondrial'


def test_non_primary_feature_fails_without_subsetting(tmp_path):
    with pytest.raises(ValueError):setup_scope(tmp_path,feature='chrM')
    assert not (tmp_path/'scope.json').exists()


@pytest.mark.parametrize('mutation',['order','missing','length','class','review'])
def test_scope_rehashed_forgery(tmp_path,mutation):
    pointer,_=setup_scope(tmp_path);p=Path(pointer['path']);value=json.loads(p.read_bytes())
    if mutation=='order':value['contigs'].reverse()
    elif mutation=='missing':value['contigs'].pop()
    elif mutation=='length':value['contigs'][0]['length']+=1
    elif mutation=='class':value['contigs'][0]['classification']='guessed'
    else:Path(value['classification_source']['path']).write_text('changed')
    value['identity_sha256']=scope.digest({k:v for k,v in value.items() if k!='identity_sha256'})
    p.write_bytes(canonical(value))
    with pytest.raises(ValueError):scope.verify_scope(p,resource(p)['sha256'])


def prepare(tmp_path,unknown=False):
    pointer,reference=setup_scope(tmp_path)
    source=tmp_path/'source.tsv';source.write_text('chr1\t0\t10\tB\t7\nchr1\t10\t20\tA\t2\nchrMT\t0\t10\tB\t1\n'+('mystery\t0\t10\tB\t1\n' if unknown else ''))
    result=prep.prepare_primary_fragments(source_path=str(source),source_sha256=resource(source)['sha256'],
        source_index_path=None,source_index_sha256=None,scope_path=pointer['path'],scope_sha256=pointer['sha256'],
        output_dir=tmp_path/'prepared')
    return result,reference


def test_unknown_exclusion_fails_closed(tmp_path):
    with pytest.raises(ValueError):prepare(tmp_path,True)
    assert not (tmp_path/'prepared').exists()


def test_byte_conservation_publication_and_admission(tmp_path):
    pointer,reference=prepare(tmp_path);value=prep.verify_preparation(pointer['path'],pointer['sha256'])
    assert value['summary']['retained_records']==2 and value['summary']['excluded_by_contig']=={'chrMT':1}
    result=import_primary_fragments(preparation_path=pointer['path'],preparation_sha256=pointer['sha256'],
        reference_bundle_path=reference['manifest_path'],reference_bundle_sha256=reference['manifest_sha256'],
        namespace='sample',output_dir=tmp_path/'zzz-adopted')
    assert result['n_fragment_records']==2 and result['total_support']==9
    record=json.loads((Path(result['manifest_path']).parent/'adoption.json').read_bytes())
    assert record['primary_preparation']==pointer
    Path(value['source']['path']).write_text('changed')
    with pytest.raises(ValueError):prep.verify_preparation(pointer['path'],pointer['sha256'])


def test_scoped_public_matrix_recovery_and_original_source_mutation(tmp_path,monkeypatch):
    from agent.tools.data import explicit_cells
    from agent.tools.data import _cell_by_ccre_production as production
    from agent.application import ResearchAgentApplication
    from agent.orchestration import AgentRequest,LLMPlanner
    import anndata as ad
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from fragment_feature_integration.test_public import Model,wire
    pointer,reference=prepare(tmp_path)
    result=import_primary_fragments(preparation_path=pointer['path'],preparation_sha256=pointer['sha256'],
        reference_bundle_path=reference['manifest_path'],reference_bundle_sha256=reference['manifest_sha256'],
        namespace='sample',output_dir=tmp_path/'zzz-adopted')
    cells=explicit_cells.publish_explicit_cells(cells=[('sample','B'),('sample','A')],declaration='Explicit synthetic cells',output_dir=tmp_path/'cells')
    inputs={f'{prefix}_manifest_{key}':value[f'manifest_{key}'] for prefix,value in [('fragments',result),('explicit_cells',cells),('reference',reference)] for key in ('path','sha256')}
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS','/usr/bin/bedtools')
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire())))
    run=app.run(AgentRequest('primary','Build exact primary nuclear fragment matrix.',inputs))
    assert run.status.value=='SUCCEEDED',run.error
    matrix=ad.read_h5ad(run.run_result.steps[0].result['matrix_path'])
    assert matrix.X.toarray().tolist()==[[1],[1]]  # support 7/2 does not multiply counts
    def forbidden(*a,**kw):raise AssertionError('repeated production')
    monkeypatch.setattr(production,'construct_counts',forbidden)
    assert ResearchAgentApplication(tmp_path/'app').resume(run.run_id).status.value=='SUCCEEDED'
    original=tmp_path/'source.tsv';raw=original.read_bytes();original.write_bytes(raw+b'changed')
    assert ResearchAgentApplication(tmp_path/'app').resume(run.run_id).status.value=='FAILED'
    original.write_bytes(raw)
    assert ResearchAgentApplication(tmp_path/'app').resume(run.run_id).status.value=='SUCCEEDED'


def test_rehashed_prepared_value_forgery_rejected(tmp_path):
    import subprocess
    pointer,_=prepare(tmp_path)
    value=prep.load(pointer['path'],pointer['sha256'])
    output=Path(value['prepared']['path'])
    forged=b'chr1\t0\t10\tB\t8\nchr1\t10\t20\tA\t2\n'
    output.write_bytes(subprocess.run(['/usr/bin/bgzip','-c'],input=forged,capture_output=True,check=True).stdout)
    value['prepared']=resource(output)
    Path(pointer['path']).write_bytes(canonical(value))
    with pytest.raises(ValueError):prep.verify_preparation(pointer['path'],resource(pointer['path'])['sha256'])


def test_target_feature_scope_cannot_broaden(tmp_path):
    pointer,reference=setup_scope(tmp_path)
    _,base,_=ref.load_regulatory_feature_reference(reference['manifest_path'])
    bed=tmp_path/'other.bed';bed.write_text('chrM\t0\t20\n')
    target=ref.publish_regulatory_feature_reference(ref.build_regulatory_feature_reference(
        species=base.species,target_assembly=base.target_assembly,fasta_path=base.genome.fasta.path,
        fai_path=base.genome.fai.path,feature_bed_path=bed,feature_category='regulatory_regions'),tmp_path/'other-reference.json')
    with pytest.raises(ValueError):scope.verify_scope(pointer['path'],pointer['sha256'],
        reference_path=target['manifest_path'],reference_sha256=target['manifest_sha256'])
