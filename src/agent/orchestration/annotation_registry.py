"""Narrow annotation ports; explicit specification supplies all biological choices."""
from pathlib import Path
from agent.tools.analysis import annotation_contract as c, scatac_annotation as public
from .matrix_registry import MatrixArgument


def validate_result(value):
    from .registry import ToolResultContractError
    try:
        c.matrix.shape(value, public.CellTypeAnnotationResult.__annotations__)
        if (value['status'] != 'success' or value['artifact_type'] != c.ARTIFACT or
                type(value['artifact_schema_version']) is not int or value['artifact_schema_version'] != 1 or
                value['contract_version'] != c.CONTRACT or value['annotation_profile'] != c.science.PROFILE or
                value['validation_state'] != 'not_assessed' or
                (value['species'],value['assembly']) not in (('human','hg38'),('mouse','mm10'))):
            raise ValueError()
        for key in value:
            if key.endswith('_sha256'): c.matrix.sha(value[key])
        for key in ('manifest_path','annotated_h5ad_path'): c.matrix.absolute_path(value[key])
        if Path(value['annotated_h5ad_path']) != Path(value['manifest_path']).parent/'annotated.h5ad':
            raise ValueError()
        c.science.identifier(value['context'])
        n = c.matrix.integer(value['n_cells'],1)
        g = c.matrix.integer(value['n_groups'],2,n)
        assigned = c.matrix.integer(value['assigned_cells'],0,n)
        if c.matrix.integer(value['unassigned_cells'],0,n) != n-assigned: raise ValueError()
        c.matrix.integer(value['ambiguous_cells'],0,n-assigned)
        if value['groups_omitted'] != max(0,g-50) or len(value['group_summary']) != min(50,g): raise ValueError()
        for row in value['group_summary']:
            c.matrix.shape(row,('group','n_cells','primary_annotation','status'))
            c.science.identifier(row['group']); c.matrix.integer(row['n_cells'],1,n)
            if row['status'] not in ('assigned','unresolved','insufficient_evidence','ambiguous'): raise ValueError()
            if row['status']=='assigned': c.science.identifier(row['primary_annotation'])
            elif row['primary_annotation'] is not None: raise ValueError()
    except (ValueError,KeyError,TypeError):
        raise ToolResultContractError('Invalid cell-type annotation result.') from None


def classify(exception):
    from .registry import ErrorClassification
    from agent.schemas import ErrorCategory
    from agent.schemas.verification_authority import AuthorityError
    if isinstance(exception, AuthorityError):
        return ErrorClassification(ErrorCategory.VERIFICATION_ERROR,'ANNOTATION_AUTHORITY_INVALID')
    if isinstance(exception,(ValueError,FileNotFoundError)):
        return ErrorClassification(ErrorCategory.USER_INPUT_ERROR,'ANNOTATION_PREREQUISITE_INVALID')
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR,'ANNOTATION_EXECUTION_FAILED')


def annotation_tool_spec():
    from .registry import (ToolSpec,DurableToolHooks,ResultContract,ArtifactSemanticKind as A,
        PlanningToolRole,PlanningSourceEligibility as S,SemanticToolSpec,SemanticConsumerPortSpec,
        SemanticProducerPortSpec,_planning_argument,_planning_result,_tool_planning,
        _semantic_member as member,_semantic_request as request,_semantic_request_member as input_member,_semantic_argument_port)
    required = {}
    for key in c.ARGUMENTS:
        spec = _planning_argument((str,Path) if key.endswith('_path') or key=='output_dir' else (str,),
            'Explicit matrix/specification identities.',
            source=S.REQUEST_INPUT_ONLY)
        required[key] = MatrixArgument(spec.accepted_types,planning=spec.planning)
    ports = []
    for name,prefix in (('matrix','matrix_manifest'),('specification','annotation_spec')):
        ports.append(SemanticConsumerPortSpec(name,tuple(member(k,prefix+'_'+k) for k in ('path','sha256')),True,
            request_sources=(request(prefix+'_path',*(input_member(k,prefix+'_'+k) for k in ('path','sha256'))),)))
    ports.append(_semantic_argument_port('output_dir',required=True))
    ints = {'artifact_schema_version','n_cells','n_groups','assigned_cells','unassigned_cells','ambiguous_cells','groups_omitted'}
    return ToolSpec(name='annotate_scATAC_cell_types',function=public.annotate_scATAC_cell_types,
        required_arguments=required,optional_arguments={},
        result_contract=ResultContract('CellTypeAnnotationResult',
            {k:(int,) if k in ints else (list,) if k=='group_summary' else (str,) for k in public.CellTypeAnnotationResult.__annotations__},
            validator=validate_result, planning_fields={k:_planning_result('Primary labels; unchanged matrix.',
                bindable=True,artifact=A.RAW_SCATAC if k=='annotated_h5ad_path' else A.SCATAC_CELL_TYPE_ANNOTATION)
                for k in ('manifest_path','manifest_sha256','annotated_h5ad_path')}),
        exception_classifier=classify,retryable_error_codes=frozenset(),recovery_policy_version=c.RECOVERY_POLICY,
        durable_hooks=DurableToolHooks(public.execute_annotation,public.recover_annotation),
        planning=_tool_planning(PlanningToolRole.OPERATION,'Annotate scATAC groups.',
            'Accepted canonical fragment-record matrix; scatac-annotation-inputs.v1 pins run/step, exact group TSV, species/assembly, context and genes/signatures.',
            'MAESTRO annotation; retain unresolved/ambiguous cells. No clustering, inferred biology, external counts, peaks or required validation.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
            SemanticProducerPortSpec('annotation','scatac_cell_type_annotation.v1',(member('manifest_path'),member('manifest_sha256'))),
            SemanticProducerPortSpec('dataset','scatac_matrix_h5ad.v1',(member('value','annotated_h5ad_path'),)),)))
