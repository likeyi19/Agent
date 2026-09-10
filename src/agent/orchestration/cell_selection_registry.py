"""Reviewed explicit selection ports and IO-free threshold contracts."""
from pathlib import Path
from agent.schemas import ErrorCategory, StepOutputRef
from agent.tools.data import _cell_selection_contract as m, _barcode_qc_contract as qc, scatac_fragments_v2 as v2
from agent.tools.data.scatac_selection_profile import integer, rational, METHOD
from agent.tools.data.scatac_cell_selection import (CellSelectionResult, select_scATAC_cells,
    execute_cell_selection, recover_cell_selection)
from .registry import ArgumentSpec, ToolArgumentError


class ThresholdArgument(ArgumentSpec):
    def validate(self, name, value):
        super().validate(name,value)
        if isinstance(value,StepOutputRef): raise ToolArgumentError('Selection thresholds require explicit request values.')
        if value is None: return
        try:
            (rational if name in ('min_tss_enrichment','max_nucleosome_signal') else integer)(value)
        except ValueError: raise ToolArgumentError('Invalid exact selection threshold: '+name) from None


def validate_result(value):
    from .registry import ToolResultContractError
    try:
        v2.shape(value,CellSelectionResult.__annotations__)
        if (value['status']!='success' or value['artifact_type']!=m.ARTIFACT
            or type(value['artifact_schema_version']) is not int or value['artifact_schema_version']!=1
            or value['contract_version']!=m.CONTRACT or value['selection_method']!=METHOD
            or value['cell_call_method']!='none' or value['cell_call_state']!='not_assessed'
            or value['resource_qualification'] not in ('synthetic_only','operator_qualified')): raise ValueError()
        v2.absolute_path(value['manifest_path'])
        for key in ('manifest_sha256','ordered_selected_sha256','qc_identity_sha256'): v2.sha(value[key])
        for key in ('n_observed_barcodes','n_selected','n_rejected'):
            if type(value[key]) is not int or not 0<=value[key]<=qc.MAX_BARCODES: raise ValueError()
        if (value['n_observed_barcodes']<1 or value['n_selected']+value['n_rejected']!=value['n_observed_barcodes']
            or value['readiness']!=('selected_candidates_available' if value['n_selected'] else 'no_selected_cells')): raise ValueError()
    except (ValueError,TypeError,KeyError): raise ToolResultContractError('Invalid cell selection result.') from None


def classify(exception):
    from .registry import ErrorClassification
    from .barcode_qc_registry import classify as upstream
    code=getattr(exception,'code','')
    if code.startswith('SELECTION_'):
        category=ErrorCategory.USER_INPUT_ERROR
        if any(s in code for s in ('MISMATCH','RECOVERY','CHANGED')): category=ErrorCategory.VERIFICATION_ERROR
        elif any(s in code for s in ('LIMIT','CONFLICT')): category=ErrorCategory.RESOURCE_ERROR
        return ErrorClassification(category,code)
    return upstream(exception)


def cell_selection_tool_spec():
    from .registry import (ToolSpec,DurableToolHooks,ResultContract,ArtifactSemanticKind as A,
        PlanningToolRole,PlanningSourceEligibility as S,SemanticToolSpec,SemanticConsumerPortSpec,
        SemanticProducerPortSpec,_planning_argument,_planning_result,_tool_planning,
        _semantic_member as member,_semantic_request as request,_semantic_request_member as input_member,_semantic_argument_port)
    required={key:_planning_argument(types,'Exact verified barcode QC; preserves source and qualification lineage.',
        source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT,artifacts=(A.SCATAC_BARCODE_QC,))
        for key,types in (('barcode_qc_manifest_path',(str,Path)),('barcode_qc_manifest_sha256',(str,)))}
    required['output_dir']=_planning_argument((str,Path),'Managed selection output directory.')
    ports=[SemanticConsumerPortSpec('barcode_qc',(member('manifest_path','barcode_qc_manifest_path'),member('manifest_sha256','barcode_qc_manifest_sha256')),True,
        request_sources=(request('barcode_qc_manifest_path',input_member('manifest_path','barcode_qc_manifest_path'),input_member('manifest_sha256','barcode_qc_manifest_sha256')),),
        accepted_upstream_types=('scatac_barcode_qc.v1',)),_semantic_argument_port('output_dir',required=True)]
    optional={}
    for key in (*m.REQUIRED,*m.OPTIONAL):
        is_required=key in m.REQUIRED
        types=(int,str) if key in ('min_tss_enrichment','max_nucleosome_signal') else (int,)
        spec=_planning_argument(types if is_required else (*types,type(None)),
            'Explicit inclusive threshold; nonnegative integer or exact decimal/fraction string for ratios; no floats. Optional omission/null disables. Integers/reduced ratios <=1e18; decimals <=18 places.',
            scientific_parameter=True)
        (required if is_required else optional)[key]=ThresholdArgument(spec.accepted_types,planning=spec.planning)
        ports.append(_semantic_argument_port(key,required=is_required))
    integers={'artifact_schema_version','n_observed_barcodes','n_selected','n_rejected'}
    return ToolSpec(name='select_scATAC_cells',function=select_scATAC_cells,required_arguments=required,optional_arguments=optional,
        result_contract=ResultContract('CellSelectionResult',{k:(int,) if k in integers else (str,) for k in CellSelectionResult.__annotations__},validator=validate_result,
            planning_fields={k:_planning_result('Verified QC-selected candidate cells.',bindable=True,artifact=A.SCATAC_CELL_SELECTION) for k in ('manifest_path','manifest_sha256')}),
        exception_classifier=classify,retryable_error_codes=frozenset(),recovery_policy_version=m.POLICY,
        durable_hooks=DurableToolHooks(execute_cell_selection,recover_cell_selection),
        planning=_tool_planning(PlanningToolRole.OPERATION,'Select candidate cells using explicit QC thresholds.',
            'Required minimum QC-record depth and TSS enrichment; no inferred thresholds. Optional minimum L+R, maximum QC depth and nucleosome signal.',
            'Inclusive exact comparisons; undefined TSS fails; undefined nucleosome fails only with enabled maximum. Retain all failures; empty selection succeeds.',
            'Statistical cell calling not assessed. No doublets, FRiP, matrix or inference. Metrics-only QC remains valid.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
            SemanticProducerPortSpec('selected_cells','scatac_cell_selection.v1',(member('manifest_path'),member('manifest_sha256'))),)))
