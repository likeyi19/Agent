"""M14.7 public Application acceptance using the unchanged M14.6 tiny fixture."""
import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import anndata as ad
import numpy as np
import pytest
from agent.application import ResearchAgentApplication
from agent.orchestration import AgentRequest, LLMPlanner
from agent.schemas.orchestration import _serialize
from agent.tools.data import cell_by_ccre_verifier as matrix_owner
from agent.tools.data import _cell_by_ccre_production as production
from agent.tools.data import external_fragments_verifier as fragments
from agent.tools.data import explicit_cells
from agent.tools.data.authority_context import owned_verification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
from fragment_features.conftest import factory
from fragment_feature_integration.test_public import Model, wire, inputs


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--resume',action='store_true')
    options=parser.parse_args();root=options.output.resolve()
    counts=dict(matrix_production=0,matrix_owner=0,fragment_owner=0,explicit_cell_owner=0)
    def counted(key,function):
        def call(*a,**kw):
            if options.resume:raise AssertionError('Fresh-process recovery repeated '+key)
            counts[key]+=1
            return function(*a,**kw)
        return call
    if not options.resume:
        root.mkdir(parents=True,exist_ok=False)
        fixture=root/'fixture';fixture.mkdir()
        with pytest.MonkeyPatch.context() as mp:
            args,_,_=factory.__wrapped__(fixture,mp)()
        write(root/'request-inputs.json',inputs(args));write(root/'semantic-plan.json',wire())
    # Explicit operator runtime, identical to the accepted fixture.
    with patch.dict('os.environ',{'AGENT_MATRIX_BEDTOOLS':'/usr/bin/bedtools'}),ExitStack() as stack:
        for module,name,key in ((production,'construct_counts','matrix_production'),
                                (matrix_owner,'reconstruct','matrix_owner'),
                                (fragments,'_reconstruct_source','fragment_owner')):
            stack.enter_context(patch.object(module,name,counted(key,getattr(module,name))))
        stack.enter_context(patch.object(explicit_cells,'verify_explicit_cells',
            owned_verification('explicit_cells')(counted('explicit_cell_owner',explicit_cells.verify_explicit_cells.__wrapped__))))
        start=time.monotonic()
        app=ResearchAgentApplication(root/'application',**({} if options.resume else dict(planner=LLMPlanner(Model(wire())))))
        if options.resume:run=app.resume('m14.7-bounded:run')
        else:run=app.run(AgentRequest('m14.7-bounded',
            'Using these zebrafish-declared external fragments, these exact barcodes and this peak set, build a cell-by-peak matrix.',
            json.loads((root/'request-inputs.json').read_text())))
        assert run.status.value=='SUCCEEDED',(run.error,run.run_result.errors)
        expected_counts={key:0 if options.resume else 1 for key in counts}
        assert counts==expected_counts,counts
        step=run.run_result.steps[0];result=_serialize(step.result)
        data=ad.read_h5ad(result['matrix_path'])
        expected=[[1,1,1,0],[0,2,2,0],[0,0,0,0]]
        np.testing.assert_array_equal(data.X.toarray(),expected)
        assert data.obs['barcode_identifier'].tolist()==['B','A','Z']
        assert data.var_names.tolist()==['chr1:30-40','chr1:0-20','chr1:10-25','chr1:90-100']
        assert run.evidence and run.report and not run.visualization
        authority=_serialize(step.verification.artifact_authority)
        assert authority['schema_version']==2 and authority['artifact_contract']=='scatac-cell-by-features.v1'
        write(root/('cold-recovery.json' if options.resume else 'accepted.json'),
            dict(status='passed',run_id=run.run_id,seconds=time.monotonic()-start,counts=counts,
                expected=expected,produced=data.X.toarray().tolist(),ordered_cells=data.obs['barcode_identifier'].tolist(),
                ordered_features=data.var_names.tolist(),result=result,authority=authority,
                evidence_path=str(run.evidence.path),report_path=str(run.report.path)))
        print('Fresh-process recovery passed' if options.resume else 'Public bounded acceptance passed', counts)


if __name__=='__main__':main()
