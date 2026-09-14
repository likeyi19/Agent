"""Thin exact external matrix tool metadata; no scientific IO in planning."""
from pathlib import Path
from .registry import ArgumentSpec, ToolArgumentError
from .matrix_registry import classify
from agent.schemas import StepOutputRef
from agent.tools.data import scatac_matrix_adoption as public, external_matrix_contract as e, scatac_matrix_contract as m


class AdoptionArgument(ArgumentSpec):
    def validate(self,name,value):
        super().validate(name,value)
        if isinstance(value,StepOutputRef): return
        try:
            if name.endswith('_sha256'): m.sha(value)
            elif name.endswith('_path') or name=='output_dir': m.absolute_path(str(value))
            elif value not in {'species':('human','mouse'),'assembly':('hg38','mm10'),'matrix_semantics':e.SEMANTICS}[name]: raise ValueError()
        except (ValueError,KeyError,TypeError): raise ToolArgumentError('Invalid external matrix binding: '+name) from None


def validate_result(value):
    from .registry import ToolResultContractError
    try:
        m.shape(value,public.AdoptedCellByCCREResult.__annotations__)
        if (value['status']!='success' or value['artifact_type']!=m.ARTIFACT
                or type(value['artifact_schema_version']) is not int or value['artifact_schema_version']!=1
                or value['contract_version']!=e.CONTRACT or value['matrix_profile_id']!=e.PROFILE
                or value['matrix_profile_sha256']!=e.PROFILE_SHA256 or value['matrix_semantics'] not in e.SEMANTICS
                or (value['species'],value['assembly']) not in (('human','hg38'),('mouse','mm10'))): raise ValueError()
        for key in value:
            if key.endswith('_sha256'): m.sha(value[key])
        for key in ('manifest_path','matrix_path'): m.absolute_path(value[key])
        if Path(value['matrix_path'])!=Path(value['manifest_path']).parent/'matrix.h5ad': raise ValueError()
        n,p=m.integer(value['n_cells']),m.integer(value['n_features'],1)
        z,nz,t=m.integer(value['zero_row_count'],0,n),m.integer(value['nnz']),m.integer(value['total_count'])
        if (not n-z<=nz<=(n-z)*p or t<nz or (nz==0)!=(t==0)
                or value['matrix_semantics']=='binary_accessibility' and nz!=t
                or value['logical_matrix_sha256']!=value['source_logical_matrix_sha256']
                or value['readiness']!=('matrix_available' if n else 'no_source_cells')): raise ValueError()
    except (ValueError,TypeError,KeyError): raise ToolResultContractError('Invalid external cell-by-cCRE result.') from None


def matrix_adoption_tool_spec():
    from .registry import (ToolSpec,DurableToolHooks,ResultContract,ArtifactSemanticKind as A,
        PlanningToolRole,PlanningSourceEligibility as S,SemanticToolSpec,SemanticConsumerPortSpec,
        SemanticProducerPortSpec,_planning_argument,_planning_result,_tool_planning,
        _semantic_member as member,_semantic_request as request,_semantic_request_member as input_member,_semantic_argument_port)
    required={}
    for key in public.ARGUMENTS:
        description = ('Declared external value meaning; fragment_counts, insertion_counts or binary_accessibility. No historical counting proof.'
                       if key=='matrix_semantics' else 'Exact external matrix adoption binding; no reordering, conversion or inferred lineage.')
        spec=_planning_argument((str,Path) if key.endswith('_path') or key=='output_dir' else (str,),description,
                               source=S.REQUEST_INPUT_ONLY)
        required[key]=AdoptionArgument(spec.accepted_types,planning=spec.planning,choices={'species':('human','mouse'),'assembly':('hg38','mm10'),'matrix_semantics':e.SEMANTICS}.get(key,()))
    ports=[]
    for port,prefix in (('source','source'),('reference','reference_manifest')):
        ports.append(SemanticConsumerPortSpec(port,tuple(member(s,prefix+'_'+s) for s in ('path','sha256')),True,
            request_sources=(request(prefix+'_path',*(input_member(s,prefix+'_'+s) for s in ('path','sha256'))),)))
    ports.extend(_semantic_argument_port(k,required=True) for k in ('species','assembly','matrix_semantics','output_dir'))
    ints={'artifact_schema_version','n_cells','n_features','nnz','total_count','zero_row_count'}
    return ToolSpec(name='adopt_scATAC_cell_by_ccre',function=public.adopt_scATAC_cell_by_ccre,required_arguments=required,optional_arguments={},
        result_contract=ResultContract('AdoptedCellByCCREResult',{k:(int,) if k in ints else (str,) for k in public.AdoptedCellByCCREResult.__annotations__},
            validator=validate_result,planning_fields={k:_planning_result('Verified external canonical matrix with exact source conservation.',bindable=True,
                artifact=A.RAW_SCATAC if k=='matrix_path' else A.SCATAC_CELL_BY_CCRE) for k in ('manifest_path','manifest_sha256','matrix_path')}),
        exception_classifier=classify,retryable_error_codes=frozenset(),recovery_policy_version=public.RECOVERY_POLICY,
        durable_hooks=DurableToolHooks(public.execute_matrix,public.recover_matrix),
        planning=_tool_planning(PlanningToolRole.OPERATION,'Adopt exact external canonical cell-by-cCRE H5AD.',
            'Requires full ordered reference vocabulary, unique source cells, canonical int64 CSR and explicit value semantics.',
            'Conserves external values and identities. No peak projection, fragment/QC/selection lineage or model-readiness claim.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
            SemanticProducerPortSpec('matrix','scatac_cell_by_ccre.external.v1',(member('manifest_path'),member('manifest_sha256'))),
            SemanticProducerPortSpec('dataset','scatac_matrix_h5ad.v1',(member('value','matrix_path'),)),)))
