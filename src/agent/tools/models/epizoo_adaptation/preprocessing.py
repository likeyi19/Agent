"""Sparse corpus composition only; EpiZoo owns every DF/ranking operation."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import anndata as ad

from agent.tools.data.cell_by_ccre_verifier import verify_cell_by_ccre
from agent.tools.data import scatac_matrix_contract as matrix, regulatory_matrix_contract as neutral
from agent.tools.data.regulatory_feature_reference import load_regulatory_feature_reference,reinspect_regulatory_feature_reference
from .contract import PROFILE,file_record,write_json


def corpus(matrix_bindings):
    if not matrix_bindings or len(matrix_bindings)>64:
        raise ValueError('One to 64 explicitly ordered matrices required.')
    arrays=[]; cells=[]; manifests=[]; identity=None; vocabulary=None; seen=set();sparse_bytes=0
    for dataset,binding in enumerate(matrix_bindings):
        matrix.shape(binding,('manifest_path','manifest_sha256'))
        key=(binding['manifest_path'],binding['manifest_sha256'])
        if key in seen: raise ValueError('A matrix cannot be included twice.')
        seen.add(key)
        verify_cell_by_ccre(binding['manifest_path'],expected_sha256=binding['manifest_sha256'])
        value=matrix.load_manifest_bytes(Path(binding['manifest_path']).read_bytes())
        if value['contract_version']!=neutral.CONTRACT or value['matrix_semantics']!=PROFILE['matrix_semantics']:
            raise ValueError('Initial adaptation qualifies exact neutral fragment_counts only.')
        sparse_bytes+=16*value['nnz']+8*(value['shape'][0]+1)
        if sparse_bytes>8*1024**3: raise ValueError('Joint sparse corpus exceeds the 8 GiB composition bound.')
        common=(value['species'],value['assembly'],value['reference'],value['ordered_feature_sha256'],value['matrix_semantics'])
        if identity is None: identity=common
        elif identity!=common: raise ValueError('Incompatible species/reference/vocabulary/value semantics for joint training.')
        data=ad.read_h5ad(Path(binding['manifest_path']).parent/value['matrix']['path'])
        if not sp.isspmatrix_csr(data.X): raise ValueError('Canonical sparse CSR required.')
        nnz=data.X.getnnz(axis=1)
        if data.n_obs==0 or np.any(nnz==0) or np.any(nnz==data.n_vars):
            raise ValueError('SR/CCA training requires positive and inaccessible features for every cell.')
        if vocabulary is None: vocabulary=data.var_names.tolist()
        elif vocabulary!=data.var_names.tolist(): raise ValueError('Column order mismatch.')
        arrays.append(data.X)
        cells.extend(dict(dataset_index=dataset,cell_id=x) for x in data.obs_names)
        manifests.append(value)
    # Stacking preserves the explicit dataset order and each exact sparse row.
    # The working AnnData index is internal; original identities remain in cells.
    joint=ad.AnnData(sp.vstack(arrays,format='csr'))
    joint.var_names=vocabulary
    ref=manifests[0]['reference']
    _,reference,_=load_regulatory_feature_reference(ref['manifest_path'],expected_sha256=ref['manifest_sha256'])
    reinspect_regulatory_feature_reference(reference)
    return joint,cells,manifests,reference


def preprocess(joint):
    from epizoo.data.processing import compute_document_frequency,compute_tfidf,generate_cell_sentences
    df=compute_document_frequency(joint.X,dtype=np.int64)
    transformed=compute_tfidf(joint,df=df,cell_number=joint.n_obs,scale_factor=10000,
                              dtype=np.float32,verbose=False)
    if not sp.issparse(transformed.X) or not np.isfinite(transformed.X.data).all():
        raise ValueError('Invalid EpiZoo sparse TF-IDF output.')
    result=generate_cell_sentences(transformed,species=None,base_offset=4)
    sentences=result.obs['cell_indices'].tolist()
    for row,sentence in enumerate(sentences):
        if sorted(sentence)!=(joint.X.indices[joint.X.indptr[row]:joint.X.indptr[row+1]]+4).tolist():
            raise ValueError('Cell-sentence support differs from exact matrix row.')
    return df,sentences


def save_preprocessing(root,joint,cells,manifests,df,sentences):
    root=Path(root)
    np.save(root/'document_frequency.npy',df,allow_pickle=False)
    np.save(root/'sentence_tokens.npy',np.asarray([x for row in sentences for x in row],dtype=np.int64),allow_pickle=False)
    np.save(root/'sentence_indptr.npy',np.concatenate(([0],np.cumsum([len(x) for x in sentences],dtype=np.int64))),allow_pickle=False)
    with (root/'cells.jsonl').open('x') as stream:
        from agent.schemas.verification_authority import canonical
        for value in cells: stream.write(canonical(value).decode()+'\n')
    (root/'features.txt').write_text('\n'.join(joint.var_names)+'\n')
    metadata=describe_preprocessing(root,joint,manifests)
    write_json(root/'preprocessing.json',metadata)
    return metadata


def describe_preprocessing(root,joint,manifests):
    return dict(profile='epizoo-target-corpus-tfidf.v1',owner='epizoo.data.processing',
        matrix_identities=[m['identity_sha256'] for m in manifests],n_cells=joint.n_obs,n_features=joint.n_vars,
        ordered_feature_sha256=manifests[0]['ordered_feature_sha256'],df_scope='joint-training-corpus',
        df_dtype='int64',tfidf_dtype='float32',scale_factor=10000,
        files={name:file_record(root/name)['sha256'] for name in ('document_frequency.npy','sentence_tokens.npy','sentence_indptr.npy','cells.jsonl','features.txt')})
