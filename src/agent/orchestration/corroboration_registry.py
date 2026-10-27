"""Reference comparison ports over existing accepted annotation identities."""
from pathlib import Path
from agent.tools.analysis import annotation_corroboration as owner
from .annotation_registry import classify


def tool_spec():
    from .registry import (ToolSpec, ResultContract, DurableToolHooks, ScientificAuthorityHooks,
        ArtifactSemanticKind as A, PlanningToolRole, SemanticToolSpec, SemanticConsumerPortSpec,
        SemanticProducerPortSpec, _planning_argument, _planning_result, _tool_planning,
        _semantic_member as member, _semantic_argument_port)
    required = {k:_planning_argument((str,) if k.endswith('_sha256') else (str,Path),description,
        artifacts=(A.SCATAC_CELL_TYPE_ANNOTATION,) if k.startswith('annotation_') else ()) for k,description in (
        ('annotation_manifest_path','Exact accepted native MAESTRO annotation publication; reuse it without recomputation.'),
        ('annotation_manifest_sha256','Exact annotation publication digest.'),('output_dir','Destination for separate reference evidence.'))}
    optional = {k:_planning_argument((str,) if k.endswith('_sha256') else (str,Path),
        'Optional exact CellMarker reference specification, distinct from MAESTRO gene/signature annotation_spec. Omit when no matching reference_spec request input is offered: the owner selects the unique qualified species/context match from the operator library.')
        for k in ('reference_spec_path','reference_spec_sha256')}
    return ToolSpec(name=owner.TOOL,function=owner.corroborate_cell_type_annotation,
        required_arguments=required,optional_arguments=optional,
        result_contract=ResultContract('CellTypeCorroborationResult',owner.RESULT_FIELDS,
            planning_fields={k:_planning_result('Separate qualified literature corroboration; primary labels unchanged.',bindable=True,
                artifact=A.SCATAC_CELL_TYPE_CORROBORATION) for k in ('manifest_path','manifest_sha256','evidence_path')}),
        durable_hooks=DurableToolHooks(owner.execute,owner.recover),exception_classifier=classify,
        recovery_policy_version=owner.POLICY,
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND,producer_ports=('corroboration',),
            describe=owner.describe,publication_record=owner.publication_record,reuse_result=owner.reuse_result,
            proof_metadata=owner.proof_metadata),
        planning=_tool_planning(PlanningToolRole.OPERATION,
            'Corroborate an existing cell-type annotation against a qualified local CellMarker database snapshot.',
            'Use after annotate_scATAC_cell_types in a combined request, or select an exact prior annotation for later database review. Native annotation must not be recomputed for evidence review. Reference context is checked by the owner; operator-installed compatible resources need no user resource IDs.',
            'Separate literature marker associations, not experimental ground truth or subtype validation. Preserve native labels and scores; no fuzzy terminology or cross-species inference. Do not run unless the researcher requests database/reference corroboration.',capability_ids=('marker_annotation',)),
        semantic_planning=SemanticToolSpec(consumer_ports=(
            SemanticConsumerPortSpec('annotation',(member('manifest_path','annotation_manifest_path'),member('manifest_sha256','annotation_manifest_sha256')),True,
                accepted_upstream_types=('scatac_cell_type_annotation.v1',)),
            _semantic_argument_port('output_dir',required=True),
            _semantic_argument_port('reference_spec_path',required=False),
            _semantic_argument_port('reference_spec_sha256',required=False)),
            producer_ports=(SemanticProducerPortSpec('corroboration','cell_type_corroboration.v1',
                (member('manifest_path'),member('manifest_sha256')),lineage_from_port='annotation'),)))
