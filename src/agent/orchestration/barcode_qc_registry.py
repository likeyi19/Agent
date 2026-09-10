"""Reviewed semantic interfaces for producer-neutral per-barcode QC."""
from pathlib import Path
from agent.schemas import ErrorCategory
from agent.tools.data import _barcode_qc_contract as m
from agent.tools.data.scatac_barcode_qc import BarcodeQCResult,compute_scATAC_qc,execute_barcode_qc,recover_barcode_qc


def validate_result(value):
    from .registry import ToolResultContractError
    from agent.tools.data import scatac_fragments_v2 as v2
    from agent.tools.data.scatac_qc_profile import PROFILE_SHA256
    try:
        v2.shape(value,BarcodeQCResult.__annotations__)
        if (value['status']!='success' or value['artifact_type']!=m.ARTIFACT
            or type(value['artifact_schema_version']) is not int or value['artifact_schema_version']!=1
            or value['contract_version']!=m.CONTRACT or value['science_profile_sha256']!=PROFILE_SHA256
            or value['resource_qualification'] not in ('synthetic_only','operator_qualified')): raise ValueError()
        v2.absolute_path(value['manifest_path'])
        for k in ('manifest_sha256','qc_resource_identity_sha256'): v2.sha(value[k])
        for k in ('n_observed_barcodes','n_fragment_records','n_qc_fragment_records','tss_defined','tss_undefined'): m.unsigned(value[k])
        if (not 1<=value['n_observed_barcodes']<=m.MAX_BARCODES or not value['n_observed_barcodes']<=value['n_fragment_records']<=m.MAX_RECORDS
            or value['n_qc_fragment_records']>value['n_fragment_records']
            or value['tss_defined']+value['tss_undefined']!=value['n_observed_barcodes']): raise ValueError()
    except (ValueError,TypeError,KeyError): raise ToolResultContractError('Invalid barcode QC result.') from None


def classify(exception):
    from .registry import ErrorClassification,_classify_tool_exception
    code=getattr(exception,'code','')
    if code.startswith('QC_'):
        category=ErrorCategory.USER_INPUT_ERROR
        if any(s in code for s in ('UNQUALIFIED','BACKEND','BEDTOOLS')): category=ErrorCategory.ENVIRONMENT_ERROR
        elif any(s in code for s in ('MISMATCH','CHANGED','RECOVERY')): category=ErrorCategory.VERIFICATION_ERROR
        elif any(s in code for s in ('LIMIT','CONFLICT')): category=ErrorCategory.RESOURCE_ERROR
        return ErrorClassification(category,code)
    return _classify_tool_exception(exception)


def barcode_qc_tool_spec():
    from .registry import (ToolSpec,DurableToolHooks,ResultContract,ArtifactSemanticKind as A,
        PlanningToolRole,PlanningSourceEligibility as S,SemanticToolSpec,SemanticConsumerPortSpec,
        SemanticProducerPortSpec,_planning_argument,_planning_result,_tool_planning,
        _semantic_member as member,_semantic_request as request,_semantic_request_member as input_member,_semantic_argument_port)
    arguments={};ports=[]
    for port,prefix,kind,upstream,description in (
        ('fragments','fragments_manifest',A.SCATAC_FRAGMENTS,('scatac_fragments.v2',),'Exact v2 fragments; fresh producer qualification required.'),
        ('qc_reference','qc_reference_manifest',A.SCATAC_QC_REFERENCE,(),'Exact QC reference; same parent identity; independently qualified for production.')):
        path,sha=prefix+'_path',prefix+'_sha256'
        for key,types in ((path,(str,Path)),(sha,(str,))):
            arguments[key]=_planning_argument(types,description,source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT if upstream else S.REQUEST_INPUT_ONLY,artifacts=(kind,))
        ports.append(SemanticConsumerPortSpec(port,(member('manifest_path',path),member('manifest_sha256',sha)),True,
            request_sources=(request(path,input_member('manifest_path',path),input_member('manifest_sha256',sha)),),accepted_upstream_types=upstream))
    arguments['output_dir']=_planning_argument((str,Path),'Managed QC output directory.')
    ports.append(_semantic_argument_port('output_dir',required=True))
    integers={'artifact_schema_version','n_observed_barcodes','n_fragment_records','n_qc_fragment_records','tss_defined','tss_undefined'}
    return ToolSpec(name='compute_scATAC_qc',function=compute_scATAC_qc,required_arguments=arguments,optional_arguments={},
        result_contract=ResultContract('BarcodeQCResult',{k:(int,) if k in integers else (str,) for k in BarcodeQCResult.__annotations__},validator=validate_result,
            planning_fields={k:_planning_result('Verified barcode QC.',bindable=True,artifact=A.SCATAC_BARCODE_QC) for k in ('manifest_path','manifest_sha256')}),
        exception_classifier=classify,retryable_error_codes=frozenset(),recovery_policy_version=m.POLICY,
        durable_hooks=DurableToolHooks(execute_barcode_qc,recover_barcode_qc),
        planning=_tool_planning(PlanningToolRole.OPERATION,'Compute verified QC for every observed namespace/barcode.',
            'Fixed record depth, QC-contig depth, TSS incidence, nucleosome counts; no support weights.',
            'Fragments and QC reference must have identical parent identity and manifest hash.',
            'No calling, selection, FRiP, cCRE matrix or EpiZoo. Resource and backend are qualified at execution.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
            SemanticProducerPortSpec('barcode_qc','scatac_barcode_qc.v1',(member('manifest_path'),member('manifest_sha256'))),)))
