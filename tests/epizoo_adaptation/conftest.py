import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from agent.tools.data import regulatory_feature_reference as r,scatac_reference as old
from agent.tools.data.regulatory_matrix_adoption import adopt_cell_by_features
from agent.tools.models.epizoo_adaptation.contract import file_record,execution_profile


@pytest.fixture
def matrix_factory(tmp_path):
    fa=tmp_path/'genome.fa';fa.write_text('>chr1\n'+'ACGT'*25+'\n')
    fai=tmp_path/'genome.fa.fai';fai.write_text('chr1\t100\t6\t100\t101\n')
    bed=tmp_path/'features.bed';bed.write_text('chr1\t30\t40\nchr1\t0\t20\nchr1\t10\t25\n')
    species=dict(scientific_name='Danio rerio',taxonomy_id=7955)
    ref=r.build_regulatory_feature_reference(species=species,target_assembly='synthetic-qualification',
        fasta_path=fa,fai_path=fai,feature_bed_path=bed,feature_category='regulatory_regions',
        genome_provenance=old.SourceProvenance('caller_supplied',source='synthetic unit fixture'))
    pointer=r.publish_regulatory_feature_reference(ref,tmp_path/'reference.json')
    count=0
    def make(values=None,semantics='fragment_counts'):
        nonlocal count
        count+=1
        x=np.array([[1,2,0],[0,1,4]] if values is None else values,dtype=np.int64)
        source=tmp_path/f'input{count}.h5ad'
        ad.AnnData(csr_matrix(x),obs=pd.DataFrame(index=[f'cell{i}' for i in range(len(x))]),
            var=pd.DataFrame(index=['chr1:30-40','chr1:0-20','chr1:10-25'])).write_h5ad(source)
        result=adopt_cell_by_features(source_path=str(source),source_sha256=file_record(source)['sha256'],
            reference_manifest_path=pointer['manifest_path'],reference_manifest_sha256=pointer['manifest_sha256'],
            species=species,assembly='synthetic-qualification',matrix_semantics=semantics,output_dir=str(tmp_path/f'output{count}'))
        return {k:result[k] for k in ('manifest_path','manifest_sha256')}
    return make


@pytest.fixture
def profile():
    return execution_profile(purpose='qualification',seed=0,batch_size=1,sequence_batch_size=1,
                             max_steps=2,save_steps=2,log_steps=1)
