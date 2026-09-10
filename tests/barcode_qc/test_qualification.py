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
