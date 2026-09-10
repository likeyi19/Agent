"""Explicit compact authority for verified QC-selected candidate cells."""
from pathlib import Path
from agent.tools.data.scatac_cell_selection import CellSelectionResult,verify_public_result

CONTRACT_FIELDS=frozenset(CellSelectionResult.__annotations__)
FACT_FIELDS=tuple(k for k in CellSelectionResult.__annotations__ if k not in ('status','manifest_path','manifest_sha256'))
REPORT_FIELDS=(*(k for k in FACT_FIELDS if k not in ('artifact_type','artifact_schema_version')),'effective_thresholds','reason_counts','qc_lineage','selection_profile_sha256')


def project_cell_selection(arguments,result,step_id):
    value=verify_public_result(arguments,result).to_dict()
    facts={k:value[k] for k in ('reason_counts','qc_lineage','selection_profile_sha256')}
    facts['effective_thresholds']=value['thresholds']
    artifacts=[]
    for field,kind in (('decisions','scatac_selection_decisions'),('selected','scatac_selected_identities')):
        resource=value[field]
        artifacts.append(dict(producing_step_id=step_id,tool_name='select_scATAC_cells',result_field='manifest_path',
            source_manifest_field=field,artifact_kind=kind,artifact_path=str(Path(result['manifest_path']).parent/resource['path']),
            integrity={'authoritative_digest':{'algorithm':'sha256','value':resource['sha256'],'source_manifest_field':field+'.sha256'},
                'verification_basis':('fresh_independent_cell_selection_reconstruction','full_artifact_sha256')}))
    return facts,artifacts
