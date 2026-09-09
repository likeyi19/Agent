"""Reviewed registry metadata for the single FASTQ fragments capability."""
from pathlib import Path
import re

from agent.schemas import ErrorCategory
from agent.tools.data.scatac_fragments import (ScATACFragmentsResult, RECOVERY_POLICY,
    prepare_scATAC_fragments, execute_fragments, recover_fragments)


def classify_fragments_exception(exception):
    from .registry import ErrorClassification, _classify_tool_exception
    code = getattr(exception, 'code', None)
    if isinstance(code, str) and code.startswith(('CHROMAP_', 'FRAGMENTS_')):
        if code in ('CHROMAP_SOURCE_CHANGED', 'CHROMAP_REFERENCE_CHANGED',
                    'CHROMAP_WHITELIST_MISMATCH', 'CHROMAP_FASTA_FAI_MISMATCH',
                    'CHROMAP_INDEX_BINDING_MISMATCH', 'CHROMAP_INDEX_IDENTITY_MISMATCH'):
            category = ErrorCategory.VERIFICATION_ERROR
        elif code in ('CHROMAP_INPUT_INVALID', 'CHROMAP_SOURCE_INVALID',
                      'CHROMAP_WHITELIST_REQUIRED', 'CHROMAP_BARCODE_LENGTH_UNSUPPORTED',
                      'CHROMAP_GROUP_INVALID', 'CHROMAP_LAYOUT_UNSUPPORTED', 'CHROMAP_ROLE_INVALID'):
            category = ErrorCategory.USER_INPUT_ERROR
        elif code in ('CHROMAP_OUTPUT_CONFLICT', 'CHROMAP_INDEX_OUTPUT_CONFLICT'):
            category = ErrorCategory.RESOURCE_ERROR
        elif code.startswith('CHROMAP_') and code not in ('CHROMAP_EXECUTION_FAILED','CHROMAP_OUTPUT_INCOMPLETE'):
            category = ErrorCategory.ENVIRONMENT_ERROR
        elif code == 'FRAGMENTS_TOOLCHAIN_MISMATCH':
            category = ErrorCategory.ENVIRONMENT_ERROR
        elif code in ('CHROMAP_EXECUTION_FAILED','CHROMAP_OUTPUT_INCOMPLETE','FRAGMENTS_CANONICALIZATION_FAILED'):
            category = ErrorCategory.TOOL_EXECUTION_ERROR
        elif code == 'FRAGMENTS_ARTIFACT_CONFLICT':
            category = ErrorCategory.RESOURCE_ERROR
        elif any(s in code for s in ('MISMATCH','CHANGED','VERIFICATION','RECOVERY')):
            category = ErrorCategory.VERIFICATION_ERROR
        else:
            category = ErrorCategory.USER_INPUT_ERROR
        return ErrorClassification(category, code)
    return _classify_tool_exception(exception)


def validate_fragments_result(result):
    from .registry import ToolResultContractError
    if (set(result) != set(ScATACFragmentsResult.__annotations__)
            or result['status'] != 'success' or result['artifact_type'] != 'agent.scatac-fragments'
            or result['artifact_schema_version'] != 1 or result['contract_version'] != 'scatac-fragments.v1'
            or result['species'] not in ('human','mouse')
            or result['assembly'] != {'human':'hg38','mouse':'mm10'}[result['species']]
            or re.fullmatch('[0-9a-f]{64}', result['manifest_sha256']) is None
            or not Path(result['manifest_path']).is_absolute()
            or any(ord(ch)<32 for ch in result['manifest_path'])
            or not 1 <= result['n_libraries'] <= 4096
            or not result['n_libraries'] <= result['n_fragment_records'] <= result['total_support'] <= 2**128-1):
        raise ToolResultContractError('Invalid compact fragments result.')


def fragments_tool_spec():
    from .registry import (ToolSpec, DurableToolHooks, ResultContract, ArtifactSemanticKind as A,
        PlanningToolRole, PlanningSourceEligibility as S, SemanticToolSpec,
        SemanticConsumerPortSpec, SemanticProducerPortSpec, _planning_argument,
        _planning_result, _tool_planning, _semantic_member as member,
        _semantic_request as request, _semantic_request_member as input_member,
        _semantic_argument_port)
    arguments = {}
    ports = []
    for port, path, sha, kind, upstream in (
        ('intake','intake_manifest_path','intake_manifest_sha256',A.RAW_SCATAC_INTAKE_MANIFEST,('raw_scatac_intake_manifest.v1',)),
        ('library_context','library_context_path','library_context_sha256',A.SCATAC_LIBRARY_CONTEXT,()),
        ('reference','reference_bundle_path','reference_bundle_sha256',A.SCATAC_REFERENCE_BUNDLE,()),
    ):
        for name, types in ((path,(str,Path)),(sha,(str,))):
            arguments[name] = _planning_argument(types, 'Authoritative compatible '+port+' artifact identity.',
                source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT if upstream else S.REQUEST_INPUT_ONLY,
                artifacts=(kind,))
        ports.append(SemanticConsumerPortSpec(port,
            (member('manifest_path',path),member('manifest_sha256',sha)),True,
            request_sources=(request(path,input_member('manifest_path',path),input_member('manifest_sha256',sha)),),
            accepted_upstream_types=upstream))
    arguments['output_dir'] = _planning_argument((str,Path),'Agent-managed scientific output directory.')
    ports.append(_semantic_argument_port('output_dir',required=True))
    return ToolSpec(name='prepare_scATAC_fragments',function=prepare_scATAC_fragments,
        required_arguments=arguments,optional_arguments={},
        result_contract=ResultContract('ScATACFragmentsResult',
            {key:(int,) if key in ('artifact_schema_version','n_libraries','n_fragment_records','total_support') else (str,)
             for key in ScATACFragmentsResult.__annotations__},validator=validate_fragments_result,
            planning_fields={key:_planning_result('Canonical scATAC fragments manifest preserving every library namespace.',
                bindable=True,artifact=A.SCATAC_FRAGMENTS) for key in ('manifest_path','manifest_sha256')}),
        exception_classifier=classify_fragments_exception,retryable_error_codes=frozenset(),
        recovery_policy_version=RECOVERY_POLICY,
        durable_hooks=DurableToolHooks(execute_fragments,recover_fragments),
        planning=_tool_planning(PlanningToolRole.OPERATION,
            'Prepare verified raw scATAC FASTQ sequencing into canonical fragments using repository-owned preprocessing policy.',
            'Requires compatible intake, library-processing context, and reference artifacts. FASTQ only; no cell calling, QC, or cCRE matrix construction.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
            SemanticProducerPortSpec('fragments','scatac_fragments.v1',
                (member('manifest_path'),member('manifest_sha256'))),)))
