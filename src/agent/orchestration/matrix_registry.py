"""Reviewed M11.5 matrix ports; mechanical validation performs no scientific IO."""
from pathlib import Path
from agent.schemas import ErrorCategory, StepOutputRef
from agent.tools.data import scatac_matrix_contract as m
from agent.tools.data.scatac_matrix import (ARGUMENTS, RECOVERY_POLICY, CellByCCREResult,
    build_scATAC_cell_by_ccre, execute_matrix, recover_matrix)
from .registry import ArgumentSpec, ToolArgumentError


class MatrixArgument(ArgumentSpec):
    def validate(self, name, value):
        super().validate(name, value)
        if isinstance(value, StepOutputRef): return
        try:
            if name.endswith('_sha256'): m.sha(value)
            else: m.absolute_path(str(value))
        except ValueError:
            raise ToolArgumentError('Invalid exact matrix binding: '+name) from None


def validate_result(value):
    from .registry import ToolResultContractError
    try:
        m.shape(value, CellByCCREResult.__annotations__)
        if (value['status']!='success' or value['artifact_type']!=m.ARTIFACT
                or type(value['artifact_schema_version']) is not int or value['artifact_schema_version']!=1
                or value['contract_version']!=m.CONTRACT
                or value['matrix_semantics']!=m.PROFILE.matrix_semantics
                or value['matrix_profile_id']!=m.PROFILE.profile_id
                or value['matrix_profile_sha256']!=m.PROFILE_SHA256
                or (value['species'],value['assembly']) not in (('human','hg38'),('mouse','mm10'))): raise ValueError()
        for k in ('manifest_path','matrix_path'): m.absolute_path(value[k])
        if Path(value['matrix_path'])!=Path(value['manifest_path']).parent/'matrix.h5ad': raise ValueError()
        for k in value:
            if k.endswith('_sha256'): m.sha(value[k])
        n=m.integer(value['n_cells']); f=m.integer(value['n_features'],1)
        z=m.integer(value['zero_row_count'],0,n); nz=m.integer(value['nnz']); total=m.integer(value['total_count'])
        if (not n-z<=nz<=(n-z)*f or total<nz or (nz==0)!=(total==0)
                or value['readiness']!=('matrix_available' if n else 'no_selected_cells')): raise ValueError()
        d=value['diagnostic']; m.shape(d,m.overlap_diagnostic(0,0))
        expected=m.overlap_diagnostic(d['n_selected_fragment_records_total'],d['n_selected_fragment_records_overlapping_any_ccre'])
        if m.canonical(expected)!=m.canonical(d): raise ValueError()
        records=d['n_selected_fragment_records_total']; hits=d['n_selected_fragment_records_overlapping_any_ccre']
        if records<n or (n==0 and records!=0) or not hits<=total<=hits*f: raise ValueError()
    except (ValueError,TypeError,KeyError):
        raise ToolResultContractError('Invalid cell-by-cCRE result.') from None


def classify(exception):
    from .registry import ErrorClassification
    from .cell_selection_registry import classify as upstream
    code=getattr(exception,'code','')
    if code.startswith('MATRIX_'):
        category=ErrorCategory.USER_INPUT_ERROR
        if any(s in code for s in ('MISMATCH','RECOVERY','CHANGED','VERIFICATION')): category=ErrorCategory.VERIFICATION_ERROR
        elif any(s in code for s in ('LIMIT','CONFLICT','BACKEND','OVERFLOW')): category=ErrorCategory.RESOURCE_ERROR
        return ErrorClassification(category,code)
    return upstream(exception)


def matrix_tool_spec():
    from .registry import (ToolSpec,DurableToolHooks,ResultContract,ArtifactSemanticKind as A,
        PlanningToolRole,PlanningSourceEligibility as S,SemanticToolSpec,SemanticConsumerPortSpec,
        SemanticProducerPortSpec,_planning_argument,_planning_result,_tool_planning,
        _semantic_member as member,_semantic_request as request,_semantic_request_member as input_member,_semantic_argument_port)
    required={}; ports=[]
    for port,prefix,kind,upstream in (
            ('fragments','fragments_manifest',A.SCATAC_FRAGMENTS,('scatac_fragments.v2',)),
            ('selected_cells','selected_cells_manifest',A.SCATAC_CELL_SELECTION,('scatac_cell_selection.v1',)),
            ('reference','reference_manifest',A.SCATAC_REFERENCE_BUNDLE,())):
        for suffix in ('path','sha256'):
            key=prefix+'_'+suffix
            spec=_planning_argument((str,Path) if suffix=='path' else (str,),
                'Exact verified authority; all three inputs must share reference and fragment lineage.',
                source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT if upstream else S.REQUEST_INPUT_ONLY,artifacts=(kind,))
            required[key]=MatrixArgument(spec.accepted_types,planning=spec.planning)
        ports.append(SemanticConsumerPortSpec(port,tuple(member('manifest_'+s,prefix+'_'+s) for s in ('path','sha256')),True,
            request_sources=(request(prefix+'_path',*(input_member('manifest_'+s,prefix+'_'+s) for s in ('path','sha256'))),),
            accepted_upstream_types=upstream))
    spec=_planning_argument((str,Path),'Managed matrix publication directory.')
    required['output_dir']=MatrixArgument(spec.accepted_types,planning=spec.planning)
    ports.append(_semantic_argument_port('output_dir',required=True))
    integers={'artifact_schema_version','n_cells','n_features','nnz','total_count','zero_row_count'}
    return ToolSpec(name='build_scATAC_cell_by_ccre',function=build_scATAC_cell_by_ccre,required_arguments=required,optional_arguments={},
        result_contract=ResultContract('CellByCCREResult',
            {k:(int,) if k in integers else (dict,) if k=='diagnostic' else (str,) for k in CellByCCREResult.__annotations__},validator=validate_result,
            planning_fields={k:_planning_result('Verified full ordered canonical fragment-record matrix.',bindable=True,
                artifact=A.RAW_SCATAC if k=='matrix_path' else A.SCATAC_CELL_BY_CCRE) for k in ('manifest_path','manifest_sha256','matrix_path')}),
        exception_classifier=classify,retryable_error_codes=frozenset(),recovery_policy_version=RECOVERY_POLICY,
        durable_hooks=DurableToolHooks(execute_matrix,recover_matrix),
        planning=_tool_planning(PlanningToolRole.OPERATION,'Build full ordered cell-by-cCRE canonical fragment-record counts.',
            'Exact verified fragments v2, QC-selected cells and shared reference. Count each overlapping canonical record once per cCRE; support does not weight counts.',
            'Preserve selected row order, full reference columns and zeros. Empty selection succeeds. No filtering, cell calling or model inference.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
            SemanticProducerPortSpec('matrix','scatac_cell_by_ccre.v1',(member('manifest_path'),member('manifest_sha256'))),
            SemanticProducerPortSpec('dataset','scatac_matrix_h5ad.v1',(member('value','matrix_path'),)),)))
