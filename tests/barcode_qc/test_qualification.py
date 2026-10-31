import hashlib
import json
from pathlib import Path
import pytest
from agent.tools.data import scatac_qc_reference as qr,scatac_barcode_qc as public
from agent.tools.data._barcode_qc_binding import resource_qualification
from agent.tools.data.scatac_qc_profile import ScATACQCError


def gtf_resource(qc_case,tmp_path):
    args,_,_,q,_=qc_case;p=tmp_path/'source.gtf'
    lines=[]
    for row in Path(q.annotation.resource.path).read_text().splitlines():
        chrom,start,end,strand,gene,tx,bio=row.split('\t')
        lines.append(f'{chrom}\tTEST\ttranscript\t{int(start)+1}\t{end}\t.\t{strand}\t.\tgene_id "{gene}"; transcript_id "{tx}"; gene_type "{bio}";\n')
    p.write_text(''.join(lines))
    bundle=qr.build_scatac_qc_reference_bundle(parent_manifest_path=q.parent_manifest.path,parent_manifest_sha256=q.parent_manifest.sha256,
        annotation_path=p,annotation_source='GENCODE',annotation_release='synthetic-qualification-fixture',parser_profile=qr.GTF_PROFILE,
        classifications=q.contigs,classification_source=q.classification_source,output_dir=tmp_path/'gtf-qc')
    path=tmp_path/'gtf-qc/manifest.json'
    args=args|dict(qc_reference_manifest_path=str(path),qc_reference_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return args,bundle


def test_declared_gtf_does_not_self_qualify(qc_case,tmp_path):
    args,q=gtf_resource(qc_case,tmp_path)
    with pytest.raises(ScATACQCError,match='UNQUALIFIED'):public.compute_scATAC_qc(**args)
    assert not Path(args['output_dir']).exists()


def test_catalog_change_between_token_and_binding_never_publishes(qc_case,tmp_path,monkeypatch):
    from agent.tools.data import _barcode_qc_binding as binding
    args,q=gtf_resource(qc_case,tmp_path)
    entry=dict(resource_identity_sha256=q.resource_identity_sha256,parent_reference_identity_sha256=q.parent_reference_identity_sha256,
        annotation_sha256=q.annotation.resource.sha256,annotation_release=q.annotation.release,qualification_basis='synthetic-test-review')
    payload=dict(artifact_type='agent.qc-resource-qualification-catalog',schema_version=1,resources=[entry])
    catalog=tmp_path/'changing-catalog.json';catalog.write_text(json.dumps(payload))
    monkeypatch.setenv('AGENT_QC_RESOURCE_CATALOG',str(catalog));original=binding.bind
    def changed(*a,**k):
        catalog.write_text(json.dumps(payload,indent=2))
        return original(*a,**k)
    monkeypatch.setattr(binding,'bind',changed)
    with pytest.raises(ScATACQCError,match='QC_BINDING_MISMATCH'):
        public.execute_barcode_qc(args,'audit-execution-identity')
    assert not Path(args['output_dir']).exists()


@pytest.mark.parametrize('mutation',[None,'identity','release','duplicate'])
def test_explicit_operator_catalog(qc_case,tmp_path,monkeypatch,mutation):
    args,q=gtf_resource(qc_case,tmp_path)
    entry=dict(resource_identity_sha256=q.resource_identity_sha256,parent_reference_identity_sha256=q.parent_reference_identity_sha256,
        annotation_sha256=q.annotation.resource.sha256,annotation_release=q.annotation.release,qualification_basis='synthetic-test-review')
    if mutation=='identity':entry['resource_identity_sha256']='1'*64
    if mutation=='release':entry['annotation_release']='wrong'
    payload=dict(artifact_type='agent.qc-resource-qualification-catalog',schema_version=1,resources=[entry,entry] if mutation=='duplicate' else [entry])
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps(payload));monkeypatch.setenv('AGENT_QC_RESOURCE_CATALOG',str(catalog))
    if mutation:
        with pytest.raises(ScATACQCError,match='UNQUALIFIED'):resource_qualification(q)
    else:
        result=public.compute_scATAC_qc(**args)
        assert result['resource_qualification']=='operator_qualified'
        catalog.write_text(json.dumps(payload,indent=2))
        with pytest.raises(ValueError,match='BINDING_MISMATCH'):public.verify_public_result(args,result)


@pytest.mark.parametrize('changed_recorded_catalog', [False, True])
def test_accepted_qc_uses_recorded_catalog_after_operator_restart(qc_case,tmp_path,monkeypatch,changed_recorded_catalog):
    from agent.application import ResearchAgentApplication
    from agent.orchestration import LLMPlanner
    from agent.orchestration.verification_authority import _load_context
    from agent.schemas.verification_authority import AuthorityError
    from tests.barcode_qc.test_integration import Model, wire, request
    args,q=gtf_resource(qc_case,tmp_path)
    catalog=tmp_path/'original-catalog.json'
    catalog.write_text(json.dumps(dict(artifact_type='agent.qc-resource-qualification-catalog',schema_version=1,
        resources=[dict(resource_identity_sha256=q.resource_identity_sha256,
            parent_reference_identity_sha256=q.parent_reference_identity_sha256,
            annotation_sha256=q.annotation.resource.sha256,annotation_release=q.annotation.release,
            qualification_basis='synthetic-test-review')])))
    monkeypatch.setenv('AGENT_QC_RESOURCE_CATALOG',str(catalog))
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(wire())))
    result=app.run(request(args))
    assert result.status.value=='SUCCEEDED',result
    before=app.run_store.load(result.run_id)
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG')
    monkeypatch.delenv('AGENT_QC_BEDTOOLS',raising=False)
    def forbidden(*a,**k): pytest.fail('Accepted context must not execute or requalify QC')
    from agent.tools.data import _barcode_qc_binding, _barcode_qc_production
    monkeypatch.setattr(_barcode_qc_binding,'backend_runtime',forbidden)
    monkeypatch.setattr(_barcode_qc_production,'produce',forbidden)
    fresh=ResearchAgentApplication(tmp_path/'app')
    if changed_recorded_catalog:
        catalog.write_text(catalog.read_text()+'\n')
        with pytest.raises(AuthorityError):
            _load_context(fresh.run_store,result.run_id,'historical_verified_sources.v1')
    else:
        context=_load_context(fresh.run_store,result.run_id,'historical_verified_sources.v1')
        context.validate_anchors()
        from agent.application.shared_resources import ScientificResourceLibrary, scientific_resource_scope
        from agent.tools.data.authority_context import authority_operation
        from agent.tools.data.barcode_qc_verifier import verify_barcode_qc
        step=next(s for s in before.steps if s.tool_name=='compute_scATAC_qc')
        with scientific_resource_scope(ScientificResourceLibrary(tmp_path/'empty-library')),authority_operation(context):
            verify_barcode_qc(step.result['manifest_path'],expected_sha256=step.result['manifest_sha256'])
        # Accepted history does not qualify a new execution without its catalog.
        with pytest.raises(ScATACQCError,match='UNQUALIFIED'):
            public._publication(args,'new-execution')
    assert fresh.run_store.load(result.run_id)==before


def test_installed_qc_owns_fresh_qualification_without_catalog(qc_case,tmp_path,monkeypatch):
    from agent.application.shared_resources import ScientificResourceLibrary, scientific_resource_scope
    from agent.tools.data import _barcode_qc_binding as binding
    args,bundle=gtf_resource(qc_case,tmp_path)
    library=ScientificResourceLibrary(tmp_path/'library')
    resource={k:args[k] for k in ('qc_reference_manifest_path','qc_reference_manifest_sha256')}
    entry=library.install('qc',resource,label='Qualified GTF fixture',version='1',
        provenance='Synthetic source qualification test',usage='Software test only')
    monkeypatch.setattr(qr,'reinspect_scatac_qc_reference_bundle_sources',
        lambda *a:pytest.fail('Installed resource qualification must not be reconstructed'))
    with scientific_resource_scope(library):
        qualification=resource_qualification(bundle)
        assert qualification == dict(mode='library_qualified',resource_id=entry['id'],
            basis='scientific_resource_library_installation')
        result=public.compute_scATAC_qc(**args)
        assert result['resource_qualification']=='library_qualified'
        public.verify_public_result(args,result)
        from agent.orchestration.barcode_qc_registry import validate_result
        from agent.orchestration.cell_selection_registry import validate_result as validate_selection
        from agent.tools.data.scatac_cell_selection import select_scATAC_cells
        validate_result(result)
        selected=select_scATAC_cells(barcode_qc_manifest_path=result['manifest_path'],
            barcode_qc_manifest_sha256=result['manifest_sha256'],output_dir=str(tmp_path/'selected'))
        validate_selection(selected)
        # Mutable unrelated legacy configuration is not a second authority.
        monkeypatch.setenv('AGENT_QC_RESOURCE_CATALOG',str(tmp_path/'absent-catalog'))
        assert resource_qualification(bundle)==qualification
        Path(bundle.tss.path).write_bytes(b'changed')
        with pytest.raises(ValueError):resource_qualification(bundle)
    # An absent installation cannot borrow eligibility from another library.
    with scientific_resource_scope(ScientificResourceLibrary(tmp_path/'empty')):
        with pytest.raises(ScATACQCError,match='UNQUALIFIED'):resource_qualification(bundle)


def test_library_removal_between_publication_and_binding_refuses(qc_case,tmp_path,monkeypatch):
    from agent.application.shared_resources import ScientificResourceLibrary, scientific_resource_scope
    from agent.tools.data import _barcode_qc_binding as binding
    args,bundle=gtf_resource(qc_case,tmp_path)
    library=ScientificResourceLibrary(tmp_path/'library')
    library.install('qc',{k:args[k] for k in ('qc_reference_manifest_path','qc_reference_manifest_sha256')},
        label='Fixture',version='1',provenance='Synthetic',usage='Test')
    original=binding.bind
    def removed(*a,**k):
        library.path.write_text(json.dumps(dict(version=1,resources=[])))
        return original(*a,**k)
    monkeypatch.setattr(binding,'bind',removed)
    with scientific_resource_scope(library),pytest.raises(ScATACQCError,match='UNQUALIFIED'):
        public.execute_barcode_qc(args,'new-execution')
    assert not Path(args['output_dir']).exists()
