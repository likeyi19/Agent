"""Bounded facts from the freshly independently verified M11.5 matrix."""
from agent.tools.data.scatac_matrix import CellByCCREResult, verify_public_result

CONTRACT_FIELDS=frozenset(CellByCCREResult.__annotations__)
FACT_FIELDS=tuple(k for k in CellByCCREResult.__annotations__ if k not in
    ('status','manifest_path','manifest_sha256','matrix_path'))
REPORT_FIELDS=tuple(k for k in FACT_FIELDS if k not in ('artifact_type','artifact_schema_version'))+('upstream_identities',)


def project_matrix(arguments,result,step_id):
    value=verify_public_result(arguments,result)
    return {'upstream_identities':{k:{f:p[f] for f in ('manifest_sha256','identity_sha256','contract_version')}
        for k,p in value['upstream'].items()}},[]
