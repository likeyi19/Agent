"""Independent supplied-count/reference fixtures; no retired raw science executes."""
import sys,hashlib
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
sys.path.insert(0,str(Path(__file__).parents[1]/'authority_dag'))

@pytest.fixture
def matrix_source(tmp_path):
    import anndata as ad
    import numpy as np,pandas as pd
    from scipy.sparse import csr_matrix
    from agent.tools.data import scatac_reference as ref
    fa=tmp_path/'ref.fa';fa.write_text('>chr2\n'+'A'*6001+'\n>chr1\n'+'C'*6001+'\n')
    fai=tmp_path/'ref.fa.fai';fai.write_text('chr2\t6001\t6\t6001\t6002\nchr1\t6001\t6014\t6001\t6002\n')
    bed=tmp_path/'ref.bed';bed.write_text('chr2\t0\t100\nchr1\t0\t1000\n')
    bundle=ref.build_scatac_reference_bundle(species='human',target_assembly='hg38',fasta_path=fa,fai_path=fai,ccre_bed_path=bed)
    pointer=ref.publish_scatac_reference_bundle(bundle,tmp_path/'reference.json')
    path=tmp_path/'source.h5ad'
    ad.AnnData(csr_matrix(np.array([[1,0],[0,1]],dtype=np.int64)),obs=pd.DataFrame(index=['A','B']),
        var=pd.DataFrame(index=['chr2:0-100','chr1:0-1000'])).write_h5ad(path)
    return dict(source_path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_manifest_path=pointer['manifest_path'],
        reference_manifest_sha256=pointer['manifest_sha256'],species='human',assembly='hg38',matrix_semantics='fragment_counts',output_dir=str(tmp_path/'output'))

@pytest.fixture
def annotation_scope(matrix_source,tmp_path,monkeypatch):
    from agent.tools.data import scatac_qc_reference as qc
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC','1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG',raising=False)
    annotation=tmp_path/'transcripts.tsv';annotation.write_text('chr2\t2000\t2100\t+\tg1\tt1\tprotein_coding\nchr1\t2000\t2100\t+\tg2\tt2\tprotein_coding\n')
    qc.build_scatac_qc_reference_bundle(parent_manifest_path=matrix_source['reference_manifest_path'],
        parent_manifest_sha256=matrix_source['reference_manifest_sha256'],annotation_path=annotation,annotation_source='synthetic',annotation_release='1',
        classifications=tuple(qc.QCContig(n,6001,'primary_nuclear_qc') for n in ('chr2','chr1')),classification_source='explicit-test',output_dir=tmp_path/'qc-reference')
    p=tmp_path/'qc-reference/manifest.json'
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

@pytest.fixture
def completed(matrix_source,tmp_path):
    from agent.orchestration import AgentPlan,PlanStep,AgentRuntime,AgentRequest,FileRunStore
    store=FileRunStore(tmp_path/'state')
    class Planner:
        def plan(self,request,registry):
            return AgentPlan('annotation-source','authority-request','Explicit simulated supplied counts',(
                PlanStep('matrix','adopt_scATAC_cell_by_ccre',matrix_source),))
    run=AgentRuntime(planner=Planner(),run_store=store).run(AgentRequest('authority-request','Adopt supplied counts',{}))
    assert run.status.value=='SUCCEEDED',run.errors
    return 'external_matrix_adoption',matrix_source,run,store
