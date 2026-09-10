"""Explicit compact authority for verified barcode metrics, never selected cells."""
from pathlib import Path
from agent.tools.data.scatac_barcode_qc import BarcodeQCResult,verify_public_result

CONTRACT_FIELDS=frozenset(BarcodeQCResult.__annotations__)
FACT_FIELDS=tuple(k for k in BarcodeQCResult.__annotations__ if k not in ('status','manifest_path','manifest_sha256'))
REPORT_FIELDS=(*(k for k in FACT_FIELDS if k not in ('artifact_type','artifact_schema_version')),'tss_method','producer_authority','qc_summary')


def project_barcode_qc(arguments,result,step_id):
    value=verify_public_result(arguments,result).to_dict()
    facts={k:value[k] for k in ('tss_method','producer_authority')}
    facts['qc_summary']=value['summary']
    artifacts=[]
    for field,kind in (('table','scatac_barcode_qc_table'),('histogram','scatac_qc_length_histogram')):
        resource=value[field]
        artifacts.append(dict(producing_step_id=step_id,tool_name='compute_scATAC_qc',result_field='manifest_path',
            source_manifest_field=field,artifact_kind=kind,artifact_path=str(Path(result['manifest_path']).parent/resource['path']),
            integrity={'authoritative_digest':{'algorithm':'sha256','value':resource['sha256'],'source_manifest_field':field+'.sha256'},
                'verification_basis':('fresh_independent_barcode_qc_reconstruction','full_artifact_sha256')}))
    return facts,artifacts
