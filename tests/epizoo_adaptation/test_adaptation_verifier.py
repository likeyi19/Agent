"""Self-consistent rehashes must not replace deterministic input checks.

Resource loaders are mocked here; these are hostile-artifact unit tests, never
SEAM execution or checkpoint qualification.
"""
from pathlib import Path
import numpy as np
import pytest
from agent.tools.models.epizoo_adaptation import publication as p,preprocessing,contract as c


@pytest.fixture(params=['fragment_counts','binary_accessibility'])
def artifact(tmp_path,matrix_factory,profile,monkeypatch,request):
    if request.param=='binary_accessibility':
        profile=c.execution_profile(purpose='qualification',seed=0,batch_size=1,sequence_batch_size=1,
            max_steps=2,save_steps=2,log_steps=1,matrix_semantics=request.param)
        binding=matrix_factory([[1,1,0],[0,1,1]],request.param)
    else: binding=matrix_factory()
    joint,cells,manifests,ref=preprocessing.corpus([binding],profile=profile)
    root=tmp_path/'artifact';root.mkdir();(root/'training').mkdir()
    df,sentences=preprocessing.preprocess(joint)
    metadata=preprocessing.save_preprocessing(root,joint,cells,manifests,df,sentences)
    from epizoo.data.ccre import extract_dna_sequences
    seqs=extract_dna_sequences(ref.genome.fasta.path,joint.var_names.tolist(),fix_chrom_name=False,show_progress=False)
    (root/'sequences.txt').write_text('\n'.join(seqs)+'\n')
    np.save(root/'sequence_embeddings.npy',np.zeros((3,512),dtype=np.float32))
    (root/'checkpoint.pth').write_text('deliberately not a checkpoint; must not be loaded')
    (root/'training/training_log.csv').write_text('test')
    execution=dict(completed_step=2,finite_loss_steps=2,optimizer_steps=2,amp_skipped_steps=0,
        last_losses=dict(loss=1,sr_loss=1,cca_loss=0),mid_training_resume=False)
    c.write_json(root/'execution.json',execution)
    args=dict(matrices=[binding],source_bundle=dict(path=str(tmp_path/'source.json'),sha256='a'*64),
        seam_bundle=dict(path=str(tmp_path/'seam.json'),sha256='b'*64),strategy='de_novo',mapping=None,
        profile=profile,device='cuda:0',output_dir=str(tmp_path/'out'))
    value=dict(artifact_type=c.ARTIFACT,contract_version=c.CONTRACT,schema_version=1,profile_sha256=c.profile_sha256(profile),
        arguments=args,reference=ref.to_dict(),source_identity='a'*64,seam_identity='a'*64,
        preprocessing=metadata,execution=execution,files={},runtime={})
    monkeypatch.setattr(p.resources,'load_bundle',lambda **kw:dict(identity_sha256='a'*64))
    monkeypatch.setattr(p.resources,'runtime',lambda:{})
    return root,value


@pytest.mark.parametrize('change',['df','tokens','row_boundaries','cells','features','dna','embedding_shape',
                                  'embedding_nan','embedding_dtype','steps','metadata_count','profile','df_dtype','token_dtype'])
def test_rehashed_forgery_rejected(artifact,change):
    root,value=artifact
    if change=='df':np.save(root/'document_frequency.npy',np.array([1,1,1],dtype=np.int64))
    elif change=='df_dtype':np.save(root/'document_frequency.npy',np.load(root/'document_frequency.npy').astype(np.float32))
    elif change=='token_dtype':np.save(root/'sentence_tokens.npy',np.load(root/'sentence_tokens.npy').astype(np.float64))
    elif change=='tokens':np.save(root/'sentence_tokens.npy',np.array([4,5,6,4],dtype=np.int64))
    elif change=='row_boundaries':np.save(root/'sentence_indptr.npy',np.array([0,1,4],dtype=np.int64))
    elif change=='cells':(root/'cells.jsonl').write_text('{}\n')
    elif change=='features':(root/'features.txt').write_text('chr1:0-20\nchr1:30-40\nchr1:10-25\n')
    elif change=='dna':(root/'sequences.txt').write_text('N\nN\nN\n')
    elif change=='embedding_shape':np.save(root/'sequence_embeddings.npy',np.zeros((2,512),dtype=np.float32))
    elif change=='embedding_nan':np.save(root/'sequence_embeddings.npy',np.full((3,512),np.nan,dtype=np.float32))
    elif change=='embedding_dtype':np.save(root/'sequence_embeddings.npy',np.zeros((3,512),dtype=np.float64))
    elif change=='steps':
        value['execution']['completed_step']=1
        (root/'execution.json').write_bytes(c.canonical(value['execution']))
    elif change=='metadata_count':
        value['preprocessing']['n_cells']=200
        (root/'preprocessing.json').write_bytes(c.canonical(value['preprocessing']))
    else:value['arguments']['profile']['trainer']['mode']='sr'
    value['files']={str(path.relative_to(root)):{k:v for k,v in c.file_record(path).items() if k!='path'}
                   for path in root.rglob('*') if path.is_file()}
    record=c.write_json(root/'manifest.json',value)
    with pytest.raises(ValueError):p.verify_target(root/'manifest.json',expected_sha256=record['sha256'])
