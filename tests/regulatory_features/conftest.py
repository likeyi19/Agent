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
