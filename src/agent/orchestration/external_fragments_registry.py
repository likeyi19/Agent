"""Reviewed registry authority for one external-fragment adoption capability."""
from pathlib import Path

from agent.schemas import ErrorCategory
from agent.tools.data import external_fragment_manifest as m, scatac_fragments_v2 as v2
from agent.tools.data.scatac_fragment_import import (ExternalFragmentsResult, RECOVERY_POLICY,
    import_scATAC_fragments, execute_external_fragments, recover_external_fragments)


def classify_external_fragments(exception):
    from .registry import ErrorClassification, _classify_tool_exception
    code = getattr(exception, 'code', '')
    if code.startswith(('EXTERNAL_FRAGMENTS_', 'FRAGMENTS_V2_', 'REFERENCE_')):
        if any(s in code for s in ('MISMATCH', 'CHANGED', 'VERIFICATION')):
            category = ErrorCategory.VERIFICATION_ERROR
        elif 'CONFLICT' in code or 'UNAVAILABLE' in code:
            category = ErrorCategory.RESOURCE_ERROR
        elif code.endswith(('_SORT_FAILED', '_PACKAGING_FAILED')):
            category = ErrorCategory.TOOL_EXECUTION_ERROR
        else:
            category = ErrorCategory.USER_INPUT_ERROR
        return ErrorClassification(category, code)
    if code == 'FRAGMENTS_TOOLCHAIN_MISMATCH':
        return ErrorClassification(ErrorCategory.ENVIRONMENT_ERROR, code)
    return _classify_tool_exception(exception)


def validate_external_result(value):
    from .registry import ToolResultContractError
    try:
        v2.shape(value, ExternalFragmentsResult.__annotations__)
        if (value['status'] != 'success' or value['artifact_type'] != v2.ARTIFACT_TYPE
                or type(value['artifact_schema_version']) is not int or value['artifact_schema_version'] != 2
                or value['contract_version'] != v2.CONTRACT_VERSION or value['source_profile'] != m.PROFILE_ID
                or value['source_encoding'] not in ('plain', 'gzip', 'bgzf')
                or value['species'] not in ('human', 'mouse')
                or value['assembly'] != {'human': 'hg38', 'mouse': 'mm10'}[value['species']]
                or value['strand_mode'] not in ('absent', 'present')):
            raise ValueError()
        v2.absolute_path(value['manifest_path']); v2.sha(value['manifest_sha256']); v2.token(value['namespace'])
        v2.number(value['n_libraries'], high=1); v2.number(value['n_fragment_records']); v2.number(value['total_support'])
        if value['n_fragment_records'] > value['total_support']:
            raise ValueError()
    except (ValueError, TypeError, KeyError):
        raise ToolResultContractError('Invalid compact external-adoption result.') from None


def external_fragments_tool_spec():
    from .registry import (ToolSpec, DurableToolHooks, ResultContract, ArtifactSemanticKind as A,
        PlanningToolRole, SemanticToolSpec, SemanticConsumerPortSpec, SemanticProducerPortSpec,
        _planning_argument, _planning_result, _tool_planning, _semantic_member as member,
        _semantic_request as request, _semantic_request_member as input_member, _semantic_argument_port)
    required = {}; optional = {}; ports = []
    for port, path, sha, mandatory, description in (
        ('source', 'source_path', 'source_sha256', True, 'External fragment file with its full encoded SHA-256; explicit source semantics required.'),
        ('reference', 'reference_bundle_path', 'reference_bundle_sha256', True, 'Explicit reference bundle; exact species/assembly and contigs, no aliases or conversion.'),
        ('source_index', 'source_index_path', 'source_index_sha256', False, 'Optional explicit source TBI and SHA-256 pair; BGZF sources only. Omission means no source index claim.'),
    ):
        target = required if mandatory else optional
        target[path] = _planning_argument((str, Path) if mandatory else (str, Path, type(None)), description)
        target[sha] = _planning_argument((str,) if mandatory else (str, type(None)), 'Exact byte identity paired with the selected resource.')
        ports.append(SemanticConsumerPortSpec(port, (member('path', path), member('sha256', sha)), mandatory,
            request_sources=(request(path, input_member('path', path), input_member('sha256', sha)),)))
    required['source_profile'] = _planning_argument((str,),
        'Explicit declaration of reviewed 10x ATAC/ARC fragment semantics, already adjusted intervals and corrected identifiers. Never infer producer from names, headers or column count.',
        choices=(m.PROFILE_ID,), scientific_parameter=True)
    required['namespace'] = _planning_argument((str,), 'Explicit processing namespace for this whole source; preserve barcode suffixes/case, no GEM-group splitting.', scientific_parameter=True)
    required['output_dir'] = _planning_argument((str, Path), 'Agent-managed adoption output directory.')
    optional['source_selection'] = _planning_argument((str,), 'Declared historical export coverage; default unknown. Does not select/filter records or establish called cells.',
        choices=tuple(m.SELECTION), scientific_parameter=True)
    ports.extend(_semantic_argument_port(n, required=n in required) for n in ('source_profile', 'namespace', 'output_dir', 'source_selection'))
    integers = {'artifact_schema_version', 'n_libraries', 'n_fragment_records', 'total_support'}
    return ToolSpec(name='import_scATAC_fragments', function=import_scATAC_fragments,
        required_arguments=required, optional_arguments=optional,
        result_contract=ResultContract('ExternalFragmentsResult',
            {k: (int,) if k in integers else (str,) for k in ExternalFragmentsResult.__annotations__},
            validator=validate_external_result, planning_fields={k: _planning_result(
                'Verified producer-neutral fragments preserving source records, exact support and optional strand.',
                bindable=True, artifact=A.SCATAC_FRAGMENTS) for k in ('manifest_path', 'manifest_sha256')}),
        exception_classifier=classify_external_fragments, retryable_error_codes=frozenset(),
        recovery_policy_version=RECOVERY_POLICY,
        durable_hooks=DurableToolHooks(execute_external_fragments, recover_external_fragments),
        planning=_tool_planning(PlanningToolRole.OPERATION,
            'Adopt explicitly declared external ATAC fragments into verified canonical fragments v2.',
            'Accept consistent five/six columns, plain/gzip/BGZF and bounded leading comments. Source profile, reference and namespace are explicit.',
            'Preserve coordinates, read-pair support and strand; reject duplicate keys. No alignment, second shift, correction, filtering, cell calling or matrix construction. Historical processing remains declared.'),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports), producer_ports=(
            SemanticProducerPortSpec('fragments', 'scatac_fragments.v2', (member('manifest_path'), member('manifest_sha256'))),)))
