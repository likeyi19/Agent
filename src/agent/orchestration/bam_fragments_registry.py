"""Reviewed interface authority for the single executable BAM profile."""
from pathlib import Path
from agent.schemas import ErrorCategory
from agent.tools.data import bam_fragment_manifest as m, scatac_fragments_v2 as v2
from agent.tools.data.scatac_bam_fragments import (BamFragmentsResult, RECOVERY_POLICY,
    prepare_scATAC_bam_fragments, execute_bam_fragments, recover_bam_fragments)


def classify_bam_fragments(exception):
    from .registry import ErrorClassification, _classify_tool_exception
    code=getattr(exception,'code','')
    if code.startswith(('BAM_FRAGMENTS_','BAM_','FRAGMENTS_','EXTERNAL_FRAGMENTS_','REFERENCE_','LIBRARY_')):
        category=ErrorCategory.USER_INPUT_ERROR
        if any(s in code for s in ('RUNTIME','TOOLCHAIN','BACKEND')): category=ErrorCategory.ENVIRONMENT_ERROR
        elif any(s in code for s in ('MISMATCH','CHANGED','VERIFICATION','RECOVERY')): category=ErrorCategory.VERIFICATION_ERROR
        elif 'CONFLICT' in code: category=ErrorCategory.RESOURCE_ERROR
        elif code.endswith(('_SORT_FAILED','_PACKAGING_FAILED')): category=ErrorCategory.TOOL_EXECUTION_ERROR
        return ErrorClassification(category,code)
    return _classify_tool_exception(exception)


def validate_bam_result(value):
    from .registry import ToolResultContractError
    try:
        v2.shape(value,BamFragmentsResult.__annotations__)
        if (value['status']!='success' or value['artifact_type']!=v2.ARTIFACT_TYPE
                or type(value['artifact_schema_version']) is not int or value['artifact_schema_version']!=2
                or value['contract_version']!=v2.CONTRACT_VERSION or value['source_profile']!=m.PROFILE_ID
                or value['species'] not in ('human','mouse')
                or value['assembly']!={'human':'hg38','mouse':'mm10'}[value['species']]
                or value['strand_mode']!='absent'): raise ValueError()
        v2.absolute_path(value['manifest_path']); v2.sha(value['manifest_sha256']); v2.token(value['namespace'])
        for k in ('n_libraries','n_fragment_records','total_support','eligible_pairs','n_templates'): v2.number(value[k])
        if (value['n_libraries']!=1 or not value['n_fragment_records']<=value['total_support']==value['eligible_pairs']<=value['n_templates']): raise ValueError()
    except (ValueError,TypeError,KeyError): raise ToolResultContractError('Invalid compact BAM fragments result.') from None


def bam_fragments_tool_spec():
    from .registry import (ToolSpec,DurableToolHooks,ResultContract,ArtifactSemanticKind as A,
        PlanningToolRole,PlanningSourceEligibility as S,SemanticToolSpec,SemanticConsumerPortSpec,
        SemanticProducerPortSpec,_planning_argument,_planning_result,_tool_planning,
        _semantic_member as member,_semantic_request as request,_semantic_request_member as input_member,_semantic_argument_port)
    arguments={}; ports=[]
    for port,path,sha,kind,upstream,description in (
        ('intake','intake_manifest_path','intake_manifest_sha256',A.RAW_SCATAC_INTAKE_MANIFEST,('raw_scatac_intake_manifest.v1',),'Verified M10 BAM intake; bounded observations do not replace full execution qualification.'),
        ('library_context','library_context_path','library_context_sha256',A.SCATAC_LIBRARY_CONTEXT,(),'One BAM group/library/namespace; explicit CB corrected_identifier/already_corrected; no raw correction.'),
        ('reference','reference_bundle_path','reference_bundle_sha256',A.SCATAC_REFERENCE_BUNDLE,(),'Explicit human/hg38 or mouse/mm10; exact source assembly and FAI names/lengths; no alias repair.'),
        ('source','source_path','source_sha256',None,(),'Exact selected intake BAM and complete encoded SHA-256; no BAI/CSI required.')):
        for key,types in ((path,(str,Path)),(sha,(str,))):
            arguments[key]=_planning_argument(types,description,source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT if upstream else S.REQUEST_INPUT_ONLY,
                artifacts=(kind,) if kind is not None else ())
        ports.append(SemanticConsumerPortSpec(port,(member('manifest_path',path),member('manifest_sha256',sha)),True,
            request_sources=(request(path,input_member('manifest_path',path),input_member('manifest_sha256',sha)),),accepted_upstream_types=upstream))
    arguments['source_profile']=_planning_argument((str,),
        'Explicit declaration of unshifted paired ATAC BAM with duplicate pairs retained. Source history cannot be inferred from tags, filenames or coordinates.',
        choices=(m.PROFILE_ID,),scientific_parameter=True)
    arguments['output_dir']=_planning_argument((str,Path),'Agent-managed BAM fragment output directory.')
    ports.extend(_semantic_argument_port(k,required=True) for k in ('source_profile','output_dir'))
    integers={'artifact_schema_version','n_libraries','n_fragment_records','total_support','eligible_pairs','n_templates'}
    return ToolSpec(name='prepare_scATAC_bam_fragments',function=prepare_scATAC_bam_fragments,required_arguments=arguments,optional_arguments={},
        result_contract=ResultContract('BamFragmentsResult',{k:(int,) if k in integers else (str,) for k in BamFragmentsResult.__annotations__},
            validator=validate_bam_result,planning_fields={k:_planning_result('Verified strand-absent BAM fragments v2; exact namespace/barcode/endpoint read-pair support.',bindable=True,artifact=A.SCATAC_FRAGMENTS) for k in ('manifest_path','manifest_sha256')}),
        exception_classifier=classify_bam_fragments,retryable_error_codes=frozenset(),recovery_policy_version=RECOVERY_POLICY,
        durable_hooks=DurableToolHooks(execute_bam_fragments,recover_bam_fragments),
        planning=_tool_planning(PlanningToolRole.OPERATION,'Prepare qualified corrected-CB BAM into independently verified fragments v2.',
            'One source/group/library/namespace; paired inward primary alignments. Both MAPQ>=30 excluding 255 before aggregation; exact +4/-5 once; duplicate-marked pairs retained.',
            'All exact FAI contigs; no mitochondrial filter, barcode correction, cell calling, QC, matrix or model inference. Agent-defined policy, not Cell Ranger reproduction.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
            SemanticProducerPortSpec('fragments','scatac_fragments.v2',(member('manifest_path'),member('manifest_sha256'))),)))
