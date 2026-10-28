"""SWC-3 simulated software checks, never evidence of biological alignment."""
from copy import deepcopy
from pathlib import Path
import pytest
from agent.tools.models.native_posttraining import lifecycle_profile
from agent.tools.models.epizoo_adaptation import contract as c, preprocessing
from agent.tools.models.epizoo_adaptation.transfer_assessment import observe_mapping, select
from agent.tools.data.liftover_chain import qualify


def test_initialization_has_no_training_and_accepts_dense_support(matrix_factory):
    profile=lifecycle_profile(matrix_semantics='binary_accessibility',operation='initialization')
    assert profile['trainer']['max_steps']==0
    c.validate_execution(profile);c.validate_strategy(profile,'mapped_reference')
    pin=matrix_factory([[1,1,1],[0,0,0]],'binary_accessibility')
    joint,*_=preprocessing.corpus([pin],profile=profile,strategy='auto')
    assert joint.shape==(2,3)
    trained=lifecycle_profile(matrix_semantics='binary_accessibility',operation='posttraining')
    with pytest.raises(ValueError,match='SR/CCA'):preprocessing.corpus([pin],profile=trained)


@pytest.mark.parametrize('steps',[1,10,100,1000,10000])
def test_training_intervals_and_explicit_override(steps):
    p=lifecycle_profile(matrix_semantics='fragment_counts',operation='posttraining',max_steps=steps)
    assert p['trainer']['max_steps']==steps
    assert 0<=p['trainer']['warmup_steps']<steps
    assert 1<=p['trainer']['log_steps']<=p['trainer']['save_steps']<=steps
    c.validate_execution(p)
    p['trainer']['warmup_steps']+=1
    with pytest.raises(ValueError):c.validate_execution(p)


@pytest.mark.parametrize('settings',[{'max_steps':0},{'max_steps':True},{'learning_rate':float('nan')},{'training_batch_size':5}])
def test_invalid_training(settings):
    with pytest.raises(ValueError):lifecycle_profile(matrix_semantics='fragment_counts',operation='posttraining',**settings)


def test_mapping_observations_preserve_native_indices_and_denominator(tmp_path):
    lifted=tmp_path/'lifted';overlap=tmp_path/'overlap'
    lifted.write_text('chr1\t0\t10\t2\nchr1\t20\t40\t0\nchr1\t80\t90\t3\n')
    overlap.write_text('chr1\t0\t10\t2\tchr1\t0\t10\t5\t10\n'
        'chr1\t0\t10\t2\tchr1\t0\t10\t6\t10\n'
        'chr1\t20\t40\t0\tchr1\t20\t30\t5\t10\n'
        'chr1\t80\t90\t3\t.\t-1\t-1\t.\t0\n')
    stats=observe_mapping({2:5,0:5},4,8,lifted,overlap,min_overlap=1)
    assert stats['lifted_target_ccres']==3
    assert stats['usable_mapping_fraction']==.5
    assert stats['tied_best_overlap_targets']==1
    assert stats['repeated_reference_assignments']==1
    assert stats['lifted_overlap_fraction']['median']==.75
    with pytest.raises(ValueError):observe_mapping({1:5},4,8,lifted,overlap,min_overlap=1)


def candidate(identity,coverage,overlap=.8):
    return dict(identity=identity,statistics=dict(usable_mapping_fraction=coverage,
        usable_mapped_target_ccres=int(coverage*100),lifted_overlap_fraction=dict(median=overlap)))


def test_auto_policy_compares_actual_evidence_and_explicit_low_coverage():
    a=candidate('human',.3);b=candidate('mouse',.5)
    assert select('auto',[a,b])['selected_candidate']=='mouse'
    assert select('auto',[b,a])==dict(select('auto',[a,b]),candidates=[b,a])
    low=candidate('low',.01)
    assert select('auto',[low])['selected']=='de_novo'
    assert select('mapped_reference',[low])['selected']=='mapped_reference'
    assert select('auto',[candidate('poor-overlap',.8,.1)])['selected']=='de_novo'
    assert select('auto',[])['reason']=='mapping-unavailable;not-evaluated'
    assert select('de_novo',[a])['candidates']==[]
    with pytest.raises(ValueError):select('mapped_reference',[])
    with pytest.raises(ValueError):select('mapped_reference',[a,b])


@pytest.fixture
def chain(tmp_path):
    files={}
    for name,text in dict(input_fai='chrA\t100\t0\t100\t101\n',destination_fai='chrB\t200\t0\t200\t201\n',
            chain='chain 1 chrA 100 + 0 20 chrB 200 - 10 30 1\n10 0 0\n10\n').items():
        path=tmp_path/name;path.write_text(text);files[name]=c.file_record(path)
    return dict(files,input_species=dict(scientific_name='Danio rerio',taxonomy_id=7955),input_assembly='simulated-a',
        destination_species=dict(scientific_name='Homo sapiens',taxonomy_id=9606),destination_assembly='simulated-b',
        source='Explicit artificial software fixture; no biological alignment',usage='test')


def test_chain_direction_and_reusable_identity(chain):
    assert qualify(chain)['chain_count']==1
    reversed=deepcopy(chain)
    reversed['input_fai'],reversed['destination_fai']=reversed['destination_fai'],reversed['input_fai']
    with pytest.raises(ValueError,match='direction'):qualify(reversed)


@pytest.mark.parametrize('text',['','chain 1 chrA 101 + 0 20 chrB 200 + 10 30 1\n20\n',
    'chain 1 chrA 100 + 0 20 chrB 200 + 10 30 1\n19\n',
    'chain 1 chrA 100 + 0 20 chrB 200 + 10 30 1\n10 5 5\n'])
def test_chain_format_and_coordinate_rejection(chain,text):
    path=Path(chain['chain']['path']);path.write_text(text);chain['chain']=c.file_record(path)
    with pytest.raises(ValueError):qualify(chain)


def test_initialization_rejects_optimizer_requests():
    with pytest.raises(ValueError,match='optimizer'):
        lifecycle_profile(matrix_semantics='binary_accessibility',operation='initialization',max_steps=10)


def test_native_mapping_callback_observes_actual_bedtools_no_hit(tmp_path):
    """Artificial alignment, genuine native executables; no biological claim."""
    import shutil
    from epizoo.data.ccre import build_ccre_map
    lift=Path('/home/likeyi/program/agent-data/resources/runtimes/ucsc/liftOver')
    if not lift.exists() or not shutil.which('bedtools'):pytest.skip('Native mapping executables unavailable')
    chain=tmp_path/'test.chain';chain.write_text('chain 1 chrA 100 + 0 100 chrB 100 + 0 100 1\n100\n')
    target=tmp_path/'target.bed';target.write_text('chrA\t50\t60\nchrA\t10\t20\n')
    source=tmp_path/'reference.bed';source.write_text('chrB\t10\t20\n')
    evidence=[]
    def observe(mapping,n,r,lifted,overlaps):
        evidence.append(observe_mapping(mapping,n,r,lifted,overlaps,min_overlap=1))
    result=build_ccre_map(str(target),str(source),str(chain),liftover_bin=str(lift),
        bedtools_bin=shutil.which('bedtools'),evidence_callback=observe)
    assert result=={1:0}
    assert evidence[0]['lifted_target_ccres']==2 and evidence[0]['usable_mapping_fraction']==.5


def test_shared_chain_install_copies_only_assembly_relationship(chain,tmp_path):
    from agent.application.shared_resources import ScientificResourceLibrary
    library=ScientificResourceLibrary(tmp_path/'library')
    entry=library.install('liftover_chain',chain,label='Artificial alignment',version='test',
        provenance=chain['source'],usage='Simulated software test only',own_files=True)
    assert library.check(entry)==entry['resource']
    for key in ('chain','input_fai','destination_fai'):
        assert entry['resource'][key]['path']!=chain[key]['path']
        assert Path(entry['resource'][key]['path']).read_bytes()==Path(chain[key]['path']).read_bytes()
    Path(chain['chain']['path']).unlink()
    assert library.check(entry)==entry['resource']
    assert set(entry['resource'])==set(chain)  # No task features or mapping outputs.


@pytest.mark.parametrize('change',['state','species','assembly','feature_order','reference','measurement'])
def test_posttraining_rejects_wrong_parent_before_model_execution(matrix_factory,tmp_path,monkeypatch,change):
    from agent.tools.models import epizoo_posttraining as train
    from agent.tools.data.neutral_matrix_binding import reference_binding,matrix_semantics
    pin=matrix_factory()
    value=c.read_json(pin['manifest_path'])
    parent=dict(contract_version=c.LIFECYCLE_CONTRACT,
        arguments=dict(profile=dict(purpose='initialization',matrix_semantics=matrix_semantics(value))),
        reference=dict(species=value['species'],target_assembly=value['assembly'],
            reference_identity_sha256=reference_binding(value)['identity_sha256']),
        preprocessing=dict(ordered_feature_sha256=value['ordered_feature_sha256']))
    if change=='state':parent['arguments']['profile']['purpose']='posttraining'
    if change=='species':parent['reference']['species']=dict(scientific_name='Mus musculus',taxonomy_id=10090)
    if change=='assembly':parent['reference']['target_assembly']='other'
    if change=='feature_order':parent['preprocessing']['ordered_feature_sha256']='0'*64
    if change=='reference':parent['reference']['reference_identity_sha256']='0'*64
    if change=='measurement':parent['arguments']['profile']['matrix_semantics']='binary_accessibility'
    monkeypatch.setattr(train.owner,'load_manifest',lambda *a:parent)
    with pytest.raises(ValueError,match='initialized|vocabulary/reference/measurement'):
        train.expand(dict(matrix_manifest_path=pin['manifest_path'],matrix_manifest_sha256=pin['manifest_sha256'],
            target_model_manifest_path=str(tmp_path/'parent.json'),target_model_manifest_sha256='a'*64,
            output_dir=str(tmp_path/'output')))
