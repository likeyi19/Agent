"""Read exact neutral lineage without changing a matrix's scientific contract."""
from . import regulatory_matrix_contract as external, fragment_feature_matrix_contract as fragment


def reference_binding(value):
    if value['contract_version'] == external.CONTRACT:
        external.validate(value)
        return value['reference']
    if value['contract_version'] == fragment.CONTRACT:
        fragment.validate_manifest(value)
        return value['upstream']['reference']
    raise ValueError('A qualified neutral matrix contract is required.')


def matrix_semantics(value):
    reference_binding(value)
    return value['matrix_semantics'] if value['contract_version'] == external.CONTRACT else value['profile']['matrix_semantics']
