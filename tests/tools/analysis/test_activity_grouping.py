"""Exact grouping adaptation contracts using controlled accepted proof records."""
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from agent.tools.analysis.native_gene_activity import corresponding_positions, project_grouping
from agent.tools.data.authority_context import authority_operation
from agent.tools._file_identity import file_sha256
from agent.schemas.verification_authority import AuthorityError


@pytest.fixture
def lineage(tmp_path):
    original=tmp_path/'original.h5ad'
    data=ad.AnnData(np.ones((6,4)),obs=pd.DataFrame(index=[f'c{i}' for i in range(6)]))
    data.write_h5ad(original)
    activity=tmp_path/'activity.h5ad'; data.write_h5ad(activity)
    grouped=tmp_path/'grouped.h5ad'
    local=data[[4,1,3]].copy(); local.obs['leiden']=pd.Categorical(['0','1','0']);local.write_h5ad(grouped)
    proofs={}
    def proof(path,method,params,slots):
        native=dict(function_name=method,matrix_path=str(path),matrix_sha256=file_sha256(path),
            manifest_path=str(path)+'.json',manifest_sha256='a'*64,
            contract_version='native-'+method.split('.')[0]+'-matrix.v1',parameters=params,
            source_identity={'sha256':file_sha256(original)},result_slots=slots)
        proofs[str(path)]=SimpleNamespace(result_metadata=native,
            description={'manifest':{'arguments':{'input_path':str(original)}}})
        return native
    proof(activity,'episcanpy.tl.geneactivity',{},[])
    group=proof(grouped,'snapatac2.tl.leiden',{'cell_subset':{'positions':[4,1,3]}},['obs.leiden'])
    context=SimpleNamespace(proofs=proofs,validate_anchors=lambda:None)
    return SimpleNamespace(activity=activity,grouped=grouped,original=original,context=context,group=group)


def test_exact_subset_projection_preserves_activity_values(lineage):
    f=lineage; source=dict(sha256=file_sha256(f.activity))
    with authority_operation(f.context):
        matrix,record=project_grouping(ad.read_h5ad(f.activity),dict(input_path=str(f.activity),grouping_path=str(f.grouped),
            function_name='tl.rank_genes_groups',kwargs={'groupby':'leiden'}),source)
    assert record['activity_positions']==[4,1,3]
    assert matrix.obs_names.tolist()==['c4','c1','c3']
    assert matrix.obs.leiden.tolist()==['0','1','0']
    assert np.array_equal(matrix.X,ad.read_h5ad(f.activity).X[[4,1,3]])


@pytest.mark.parametrize('mutation',['foreign','changed_axis','manifest','duplicate','raw_grouping','wrong_column','subset','transpose'])
def test_equal_names_or_forged_metadata_never_establish_join(lineage,mutation):
    f=lineage
    if mutation=='foreign':
        f.context.proofs[str(f.activity)].description['manifest']['arguments']['input_path']=str(f.original)+'.other'
    elif mutation=='changed_axis':
        data=ad.read_h5ad(f.grouped); data.obs_names=['c1','c4','c3'];data.write_h5ad(f.grouped)
        f.group['matrix_sha256']=file_sha256(f.grouped)
    elif mutation=='manifest':
        f.group['source_identity']['accepted_publication']={'manifest_path':'wrong','manifest_sha256':'b'*64}
    elif mutation=='duplicate':
        f.group['parameters']['cell_subset']['positions']=[4,4,3]
    elif mutation=='raw_grouping':
        f.context.proofs.pop(str(f.grouped))
    args=dict(input_path=str(f.activity),grouping_path=str(f.grouped),function_name='tl.rank_genes_groups',
              kwargs={'groupby':'wrong' if mutation=='wrong_column' else 'leiden'})
    if mutation=='subset': args['cell_subset']={'positions':[0]}
    if mutation=='transpose': args['input_orientation']='features_by_cells'
    with authority_operation(f.context),pytest.raises(AuthorityError):
        project_grouping(ad.read_h5ad(f.activity),args,dict(sha256=file_sha256(f.activity)))
