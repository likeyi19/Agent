import hashlib
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from agent.tools.data import regulatory_feature_reference as r, scatac_reference as legacy


@pytest.fixture
def resources(tmp_path):
    fa = tmp_path/'genome.fa'; fa.write_text('>chr1\n'+'A'*100+'\n')
    fai = tmp_path/'genome.fa.fai'; fai.write_text('chr1\t100\t6\t100\t101\n')
    bed = tmp_path/'features.bed'; bed.write_text('chr1\t30\t40\nchr1\t0\t20\nchr1\t10\t25\n')
    return dict(species={'scientific_name':'Macaca fascicularis','taxonomy_id':9541},
        target_assembly='GCF_000364345.1', fasta_path=fa, fai_path=fai,
        feature_bed_path=bed, feature_category='peak_set',
        feature_provenance=legacy.SourceProvenance('caller_supplied', source='explicit test peak set'))


@pytest.fixture
def case(resources, tmp_path):
    bundle = r.build_regulatory_feature_reference(**resources)
    pointer = r.publish_regulatory_feature_reference(bundle, tmp_path/'reference.json')
    source = tmp_path/'source.h5ad'
    ad.AnnData(csr_matrix(np.array([[1,2,0],[2,1,0],[0,0,0]], dtype=np.int64)),
        obs=pd.DataFrame(index=['cell-z','cell-a','cell-empty']),
        var=pd.DataFrame(index=['chr1:30-40','chr1:0-20','chr1:10-25'])).write_h5ad(source)
    return dict(source_path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reference_manifest_path=pointer['manifest_path'], reference_manifest_sha256=pointer['manifest_sha256'],
        species=resources['species'], assembly=resources['target_assembly'],
        matrix_semantics='fragment_counts', output_dir=str(tmp_path/'output'))


@pytest.fixture
def integration(case,tmp_path,monkeypatch,request):
    """Mock backend science only; real publication/authority/Application path."""
    from agent.tools.models.epizoo_adaptation import publication as p,contract as c
    from agent.tools.data.authority_context import owned_verification
    from agent.tools.data import scatac_matrix_contract as m
    from pathlib import Path
    import json
    resource=tmp_path/'resource';resource.write_text('explicit unit-test resource, not model weights')
    bundles=[]
    for name in ('source','seam'):
        record=c.write_json(tmp_path/(name+'.json'),dict(files=[c.file_record(resource)],code=dict(files=[])))
        bundles.append({k:record[k] for k in ('path','sha256')})
    ref=json.loads(Path(case['reference_manifest_path']).read_text())
    binary = getattr(request, 'param', 'fragment_counts') == 'binary_accessibility'
    if binary:
        data=ad.read_h5ad(case['source_path'])
        data.X=csr_matrix(np.array([[1,1,0],[1,0,0],[0,0,1]],dtype=np.int64))
        data.write_h5ad(case['source_path'])
        case['source_sha256']=c.file_record(case['source_path'])['sha256']
        case['matrix_semantics']='binary_accessibility'
    spec=dict(contract_version='epizoo-adaptation-inputs.v1',reference_identity_sha256=ref['reference_identity_sha256'],
        species=case['species'],assembly=case['assembly'],source_bundle=bundles[0],seam_bundle=bundles[1],
        mapping=None,execution_profile='binary-de-novo-qualification.v1' if binary else 'qualification.v1')
    record=c.write_json(tmp_path/'spec.json',spec)
    inputs={k:v for k,v in case.items() if k not in ('species','assembly','output_dir')}
    inputs.update(adaptation_spec_path=record['path'],adaptation_spec_sha256=record['sha256'],strategy='de_novo')
    counts=dict(production=0,owner=0)
    monkeypatch.setattr(p.resources,'runtime',lambda:dict(mock=True))
    monkeypatch.setattr(p.resources,'code_bundle',lambda:dict(files=[]))
    def verify(path,*,expected_sha256):
        counts['owner']+=1
        value=p.load_manifest(path,expected_sha256)
        return p._summary(value,Path(path),expected_sha256)
    monkeypatch.setattr(p,'verify_target',owned_verification('epizoo_adaptation')(verify))
    def build(args,root):
        counts['production']+=1
        root.mkdir();(root/'training').mkdir()
        source=m.load_manifest_bytes(Path(args['matrices'][0]['manifest_path']).read_bytes())
        names=('checkpoint.pth','sequence_embeddings.npy','sequences.txt','document_frequency.npy','sentence_tokens.npy',
            'sentence_indptr.npy','cells.jsonl','features.txt','preprocessing.json','execution.json','training/training_log.csv')
        for name in names:(root/name).write_text('explicit mocked science payload')
        value=dict(artifact_type=c.ARTIFACT,contract_version=c.CONTRACT,schema_version=1,profile_sha256=c.profile_sha256(args['profile']),
            arguments=args,reference=ref,source_identity='a'*64,seam_identity='b'*64,
            preprocessing=dict(n_cells=source['shape'][0],n_features=source['shape'][1]),
            execution=dict(completed_step=10,optimizer_steps=7,amp_skipped_steps=3),
            files={name:{k:v for k,v in c.file_record(root/name).items() if k!='path'} for name in names},runtime=p.resources.runtime())
        record=c.write_json(root/'manifest.json',value)
        p.verify_target(root/'manifest.json',expected_sha256=record['sha256'])
        return {k:record[k] for k in ()} | dict(manifest_path=record['path'],manifest_sha256=record['sha256'])
    monkeypatch.setattr(p,'_build',build)
    return inputs,counts,resource
