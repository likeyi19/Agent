import hashlib
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
import pytest
from agent.tools.data import scatac_reference as ref


@pytest.fixture
def case(tmp_path):
    fa=tmp_path/'ref.fa'; fa.write_text('>chr1\n'+'A'*100+'\n')
    fai=tmp_path/'ref.fa.fai'; fai.write_text('chr1\t100\t6\t100\t101\n')
    bed=tmp_path/'ref.bed'; bed.write_text('chr1\t0\t10\nchr1\t10\t20\nchr1\t30\t40\n')
    bundle=ref.build_scatac_reference_bundle(species='human',target_assembly='hg38',fasta_path=fa,fai_path=fai,ccre_bed_path=bed)
    pointer=ref.publish_scatac_reference_bundle(bundle,tmp_path/'reference.json')
    path=tmp_path/'source.h5ad'
    a=ad.AnnData(csr_matrix(np.array([[1,2,0],[2,1,0],[0,0,0]],dtype=np.int64)),
        obs=pd.DataFrame(index=['cell-z','cell-a','cell-empty']),var=pd.DataFrame(index=['chr1:0-10','chr1:10-20','chr1:30-40']))
    a.write_h5ad(path)
    return dict(source_path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reference_manifest_path=pointer['manifest_path'],reference_manifest_sha256=pointer['manifest_sha256'],
        species='human',assembly='hg38',matrix_semantics='fragment_counts',output_dir=str(tmp_path/'output'))
