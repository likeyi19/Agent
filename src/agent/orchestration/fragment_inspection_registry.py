"""Reference-independent inspection through the existing Registry and roles."""
from pathlib import Path
from agent.tools.data.fragment_inspection import inspect_scATAC_fragments, POLICY
from .external_fragments_registry import classify_external_fragments

FIELDS = {
    'source_path': (str,), 'source_sha256': (str,), 'source_index_sha256': (str, type(None)),
    'source_size_bytes': (int,), 'encoding': (str,), 'columns': (int,), 'record_limit': (int,),
    'n_records_observed': (int,), 'sum_support_observed': (int,),
    'n_distinct_barcodes_observed': (int,), 'n_contigs_observed': (int,),
    'reached_eof': (bool,), 'index_status': (str,), 'limitations': (str,),
}


def tool_spec():
    from .registry import (ToolSpec, ResultContract, ArtifactSemanticKind as A,
        PlanningToolRole, SemanticToolSpec, SemanticConsumerPortSpec, SemanticProducerPortSpec,
        _planning_argument, _planning_result, _tool_planning, _semantic_member as member,
        _semantic_request as request, _semantic_request_member as input_member)
    required, optional, ports = {}, {}, []
    for port, prefix, mandatory, description in (
        ('source', 'source', True, 'The exact fragments file to inspect structurally; no species, reference or processing declaration required.'),
        ('source_index', 'source_index', False, 'Optional exact TBI attachment. Checks only its header and byte identity, not source/index correspondence.')):
        target = required if mandatory else optional
        target[prefix + '_path'] = _planning_argument((str, Path) if mandatory else (str, Path, type(None)), description,
            artifacts=(A.RAW_SCATAC_SEQUENCING,))
        target[prefix + '_sha256'] = _planning_argument((str,) if mandatory else (str, type(None)), 'Exact registered byte identity.')
        ports.append(SemanticConsumerPortSpec(port,
            (member('path', prefix + '_path'), member('sha256', prefix + '_sha256')), mandatory,
            request_sources=(request(prefix + '_path', input_member('path', prefix + '_path'),
                input_member('sha256', prefix + '_sha256')),)))
    return ToolSpec(name='inspect_scATAC_fragments', function=inspect_scATAC_fragments,
        required_arguments=required, optional_arguments=optional,
        result_contract=ResultContract('FragmentStructureInspection', FIELDS,
            planning_fields={k: _planning_result(description, bindable=k == 'source_sha256')
                for k, description in {
                    'source_sha256': 'Full encoded byte identity of the inspected source; not reference or processing qualification.',
                    'source_index_sha256': 'Full encoded identity of an optional index; not proof of pairing.',
                    'n_records_observed': 'Number of records in the inspected prefix, never an estimated full-file count.',
                    'sum_support_observed': 'Sum of supplied support values in the inspected prefix; processing semantics remain unqualified.',
                    'n_distinct_barcodes_observed': 'Distinct barcode strings in the inspected prefix, not called cells.',
                    'n_contigs_observed': 'Distinct contig names in the inspected prefix; neither assembly nor contig bounds is established.',
                    'reached_eof': 'True only when sequential inspection reached EOF; false means all observations remain prefix-scoped.',
                    'record_limit': 'Maximum number of records inspected; not a biological threshold.',
                    'index_status': 'Optional TBI header status; source/index correspondence is not verified.',
                    'limitations': 'Explicit scientific scope and non-qualification of this structural inspection.',
                }.items()}),
        exception_classifier=classify_external_fragments, recovery_policy_version=POLICY,
        planning=_tool_planning(PlanningToolRole.INSPECTION,
            'Inspect basic structure of plain/gzip/BGZF fragments without references or preprocessing declarations.',
            'Read at most 10,000 records. Report observed record/support/barcode/contig counts and coverage; barcode strings are not cells.',
            'Does not qualify assembly, processing semantics, index correspondence or QC readiness and cannot feed downstream preprocessing.',
            capability_ids=('raw_preprocessing',)),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports), producer_ports=(
            SemanticProducerPortSpec('inspection', 'fragment_structure_inspection.v1', (member('source_sha256'),)),)))
