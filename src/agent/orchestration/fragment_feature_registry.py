"""Neutral fragment-derived matrix registration; static ports contain no recipes."""
from pathlib import Path
from agent.schemas.orchestration import _serialize
from agent.tools.data import fragment_feature_tool as public, fragment_feature_matrix_contract as c
from agent.tools.data.scatac_matrix import CellByCCREResult
from .matrix_registry import MatrixArgument, classify


def validate_result(value):
    from .registry import ToolResultContractError
    from .matrix_registry import validate_result as legacy_validate
    from agent.tools.data import scatac_matrix_contract as legacy
    value = _serialize(value)
    try:
        selected_contract = c.contract_for(value)
        if (value['artifact_type'] != selected_contract.ARTIFACT
                or value['matrix_profile_id'] != selected_contract.PROFILE.profile_id
                or value['matrix_profile_sha256'] != selected_contract.PROFILE_SHA256):
            raise ValueError()
        c.validate_binding(value['species'], value['assembly'])
        legacy.shape(value['diagnostic'], c.overlap_diagnostic(0, 0))
        checked = dict(value, artifact_type=legacy.ARTIFACT, contract_version=legacy.CONTRACT,
            matrix_profile_id=legacy.PROFILE.profile_id, matrix_profile_sha256=legacy.PROFILE_SHA256,
            species='human', assembly='hg38',
            diagnostic={k.replace('any_feature', 'any_ccre'): v for k,v in value['diagnostic'].items()})
        legacy_validate(checked)
    except (ValueError, KeyError, TypeError):
        raise ToolResultContractError('Invalid fragment-derived neutral matrix result.') from None


def tool_spec():
    from .registry import (ToolSpec, DurableToolHooks, ResultContract, ArtifactSemanticKind as A,
        PlanningToolRole, PlanningSourceEligibility as S, SemanticToolSpec, SemanticConsumerPortSpec,
        SemanticProducerPortSpec, _planning_argument, _planning_result, _tool_planning,
        _semantic_member as member, _semantic_request as request,
        _semantic_request_member as input_member, _semantic_argument_port)
    required = {}; optional = {}; ports = []
    for port, prefix in (('fragments','fragments_manifest'), ('explicit_cells','explicit_cells_manifest'),
                         ('selected_cells','selected_cells_manifest'), ('reference','reference_manifest')):
        cell_port = port in ('explicit_cells', 'selected_cells')
        upstream = ('scatac_cell_selection.v1',) if port == 'selected_cells' else ()
        for suffix in ('path','sha256'):
            key = prefix+'_'+suffix
            spec = _planning_argument((str,Path) if suffix=='path' else (str,),
                'Exact supplied neutral artifact; explicit species/assembly/genome and namespace must agree.',
                source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT if upstream else S.REQUEST_INPUT_ONLY)
            (optional if cell_port else required)[key] = MatrixArgument(spec.accepted_types, planning=spec.planning)
        ports.append(SemanticConsumerPortSpec(port,
            tuple(member('manifest_'+s,prefix+'_'+s) for s in ('path','sha256')), not cell_port,
            request_sources=(request(prefix+'_path',*(input_member('manifest_'+s,prefix+'_'+s) for s in ('path','sha256'))),),
            accepted_upstream_types=upstream))
    spec = _planning_argument((str,Path),'Managed matrix publication directory.')
    required['output_dir'] = MatrixArgument(spec.accepted_types, planning=spec.planning)
    ports.append(_semantic_argument_port('output_dir',required=True))
    integers = {'artifact_schema_version','n_cells','n_features','nnz','total_count','zero_row_count'}
    return ToolSpec(name='build_scATAC_cell_by_features', function=public.build_scATAC_cell_by_features,
        required_arguments=required, optional_arguments=optional,
        result_contract=ResultContract('FragmentFeatureMatrixResult',
            {k:(int,) if k in integers else (dict,) if k in ('species','diagnostic') else (str,)
             for k in CellByCCREResult.__annotations__}, validator=validate_result,
            planning_fields={k:_planning_result('Verified fragment-derived neutral matrix authority.',
                bindable=True,artifact=A.SCATAC_CELL_BY_FEATURES) for k in ('manifest_path','manifest_sha256')}),
        exception_classifier=classify, retryable_error_codes=frozenset(),
        recovery_policy_version=public.RECOVERY_POLICY, durable_hooks=DurableToolHooks(public.execute,public.recover),
        planning=_tool_planning(PlanningToolRole.OPERATION,
            'Build a target-species cell-by-regulatory-feature matrix from producer-qualified fragments.',
            'Requires neutral fragments v2 and regulatory-feature-reference.v1. Supply exactly one complete cell pair: selected_cells (scatac_cell_selection.v1) or explicit_cells (scatac-explicit-cells.v1 override). Never both.',
            'QC-selected rows retain exact fragment/QC lineage; explicit cells assert no QC. Preserve full axes and zero-overlap rows; absent barcodes fail. No cell calling or model-readiness claim.',
            capability_ids=('species_adaptation',)),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
            SemanticProducerPortSpec('matrix','scatac_cell_by_features.v1',
                (member('manifest_path'),member('manifest_sha256'))),)))
