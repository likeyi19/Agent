"""Bounded real public-path qualification; scripted semantic provider, real science."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
from agent.application import ResearchAgentApplication
from agent.orchestration import AgentRequest,LLMPlanner
from agent.tools.models.epizoo_adaptation import contract as c,backend,publication
from agent.tools.data import cell_by_ccre_verifier as matrix_owner
from agent.tools.data.authority_context import owned_verification


def wire():
    def inp(target,key):return dict(target=target,source=dict(kind='input',input=key))
    return dict(schema_version=4,decision=dict(kind='plan',steps=[
        dict(step_id='adopt',tool='adopt_scATAC_cell_by_features',sources=[inp('source','source_path'),inp('reference','reference_manifest_path'),inp('matrix_semantics','matrix_semantics')],control_dependencies=[]),
        dict(step_id='adapt',tool='adapt_epizoo_species',sources=[dict(target='matrix',source=dict(kind='step_port',step='adopt',source_port='matrix')),
            inp('specification','adaptation_spec_path'),inp('strategy','strategy')],control_dependencies=[])]))


class Model:
    model_id='m14.3-scripted-semantic-acceptance'
    def complete(self,*,prompt,response_schema):
        if 'selection_schema_version' in response_schema.get('properties',{}):
            return json.dumps(dict(selection_schema_version=1,decision=dict(kind='select',capability_ids=['species_adaptation'])))
        return json.dumps(wire())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    opts=p.parse_args();root=opts.output.resolve()
    torch.set_num_threads(4)
    if opts.resume:
        app=ResearchAgentApplication(root/'application')
        def forbidden(*args,**kwargs):raise AssertionError('Recovery repeated scientific execution')
        with patch.object(backend,'train',forbidden),patch.object(backend,'sequences_and_embeddings',forbidden),patch.object(matrix_owner,'_verify_external',forbidden),patch.object(publication,'verify_target',owned_verification('epizoo_adaptation')(forbidden)):
            result=app.resume('m14.3-real:run')
        if result.status.value!='SUCCEEDED':raise RuntimeError(str(result))
        c.write_json(root/'cold-recovery.json',dict(status='passed',training_calls=0,seam_calls=0,matrix_owner_calls=0,adaptation_owner_calls=0,run_id=result.run_id))
        print('Fresh-process recovery passed without scientific execution.')
        return
    root.mkdir(parents=True,exist_ok=False)
    prior=Path('/home/likeyi/program/agent-acceptance/m14.2/qualification-final')
    ref=c.read_json(prior/'reference.json')
    binding=lambda path:{k:v for k,v in c.file_record(path).items() if k!='size_bytes'}
    spec=dict(contract_version='epizoo-adaptation-inputs.v1',reference_identity_sha256=ref['reference_identity_sha256'],
        species=ref['species'],assembly=ref['target_assembly'],source_bundle=binding(prior/'source-bundle.json'),
        seam_bundle=binding(prior/'seam-bundle.json'),mapping=None,execution_profile='qualification.v1')
    record=c.write_json(root/'adaptation-inputs.json',spec)
    inputs=dict(source_path=str(prior/'fixture.h5ad'),source_sha256=c.file_record(prior/'fixture.h5ad')['sha256'],
        reference_manifest_path=str(prior/'reference.json'),reference_manifest_sha256=c.file_record(prior/'reference.json')['sha256'],
        matrix_semantics='fragment_counts',adaptation_spec_path=record['path'],adaptation_spec_sha256=record['sha256'],strategy='de_novo')
    c.write_json(root/'request-inputs.json',inputs);c.write_json(root/'semantic-plan.json',wire())
    counts=dict(training=0,seam=0,matrix_owner=0,adaptation_owner=0)
    def counted(key,function):
        def call(*a,**kw):counts[key]+=1;return function(*a,**kw)
        return call
    app=ResearchAgentApplication(root/'application',planner=LLMPlanner(Model()))
    import time
    start=time.perf_counter()
    with patch.object(backend,'train',counted('training',backend.train)),patch.object(backend,'sequences_and_embeddings',counted('seam',backend.sequences_and_embeddings)),patch.object(matrix_owner,'_verify_external',counted('matrix_owner',matrix_owner._verify_external)),patch.object(publication,'verify_target',owned_verification('epizoo_adaptation')(counted('adaptation_owner',publication.verify_target.__wrapped__))):
        result=app.run(AgentRequest('m14.3-real','Post-train EpiZoo de novo on this explicit artificial zebrafish-declared matrix and reference, using the bounded qualification profile.',inputs))
    if result.status.value!='SUCCEEDED':raise RuntimeError(str(result))
    assert counts==dict(training=1,seam=1,matrix_owner=1,adaptation_owner=1),counts
    from agent.schemas.orchestration import _serialize
    c.write_json(root/'accepted.json',dict(run_id=result.run_id,status=result.status.value,counts=counts,seconds=time.perf_counter()-start,
        result=_serialize(result.run_result.steps[-1].result),evidence_path=str(result.evidence.path),report_path=str(result.report.path)))
    print('Bounded real Application qualification passed:',counts)


if __name__=='__main__':main()
