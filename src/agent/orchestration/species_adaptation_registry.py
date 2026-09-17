"""Two public tools; fixed mechanics stay outside semantic planning."""
from pathlib import Path
from agent.schemas.orchestration import _serialize
from .matrix_registry import MatrixArgument, classify as matrix_classify
from .registry import ArgumentSpec
from agent.tools.data import neutral_matrix_tool as adoption
from agent.tools.models import species_adaptation as adaptation


def classify(exception):
    from .registry import ErrorClassification
    from agent.schemas import ErrorCategory
    from agent.schemas.verification_authority import AuthorityError
    if isinstance(exception, AuthorityError):
        return ErrorClassification(ErrorCategory.VERIFICATION_ERROR,'SPECIES_AUTHORITY_INVALID')
    if getattr(exception,'code','').startswith('MATRIX_'):
        return matrix_classify(exception)
    if isinstance(exception,(ValueError,FileNotFoundError)):
        return ErrorClassification(ErrorCategory.USER_INPUT_ERROR,'SPECIES_PREREQUISITE_INVALID')
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR,'SPECIES_EXECUTION_FAILED')


def validate_adoption(value):
    value = _serialize(value)
    from .registry import ToolResultContractError
    from agent.tools.data import scatac_matrix_adoption as old, regulatory_matrix_contract as c
    from .matrix_adoption_registry import validate_result
    from copy import deepcopy
    try:
        if (value['artifact_type'] != c.ARTIFACT or value['contract_version'] != c.CONTRACT or
            value['matrix_profile_id'] != c.PROFILE or value['matrix_profile_sha256'] != c.PROFILE_SHA256):
            raise ValueError()
        c.validate_binding(value['species'],value['assembly'])
        # Reuse the complete numeric/conservation result checks, retaining the
        # neutral contract above; no conversion of scientific inputs occurs.
        from agent.tools.data import external_matrix_contract as legacy
        checked=deepcopy(value)
        checked.update(artifact_type=legacy.ARTIFACT,contract_version=legacy.CONTRACT,
            matrix_profile_id=legacy.PROFILE,matrix_profile_sha256=legacy.PROFILE_SHA256,
            species='human',assembly='hg38')
        validate_result(checked)
    except (ValueError,KeyError,TypeError):
        raise ToolResultContractError('Invalid neutral matrix adoption result.') from None


def validate_adaptation(value):
    value = _serialize(value)
    from .registry import ToolResultContractError
    from agent.tools.models.epizoo_adaptation import contract as c
    try:
        c.checks.shape(value, adaptation.RESULT_FIELDS)
        if (value['status']!='success' or value['artifact_type']!=c.ARTIFACT or
            value['contract_version']!=c.CONTRACT or value['profile_sha256']!=c.PROFILE_SHA256 or
            value['purpose'] not in ('qualification','production') or
            value['strategy'] not in ('de_novo','mapped_reference')): raise ValueError()
        from agent.tools.data.regulatory_matrix_contract import validate_binding
        validate_binding(value['species'],value['assembly'])
        for key,val in value.items():
            if key.endswith('_sha256') and not (key=='mapping_identity_sha256' and val is None): c.checks.sha(val)
        for key in ('manifest_path','checkpoint_path'): c.checks.absolute_path(value[key])
        if Path(value['checkpoint_path']) != Path(value['manifest_path']).parent/'checkpoint.pth': raise ValueError()
        for key in ('n_cells','n_features','completed_step','optimizer_steps'): c.checks.integer(value[key],1)
        c.checks.integer(value['amp_skipped_steps'])
        if value['completed_step'] != value['optimizer_steps']+value['amp_skipped_steps']: raise ValueError()
        if (value['mapping_identity_sha256'] is None) != (value['strategy']=='de_novo'): raise ValueError()
    except (ValueError,KeyError,TypeError):
        raise ToolResultContractError('Invalid species adaptation result.') from None


def tool_specs():
    from .registry import (ToolSpec,DurableToolHooks,ResultContract,ArtifactSemanticKind as A,
        PlanningToolRole,PlanningSourceEligibility as S,SemanticToolSpec,SemanticConsumerPortSpec,
        SemanticProducerPortSpec,_planning_argument,_planning_result,_tool_planning,
        _semantic_member as member,_semantic_request as request,_semantic_request_member as input_member,_semantic_argument_port)
    from agent.tools.data.scatac_matrix_adoption import AdoptedCellByCCREResult
    specs=[]
    for name,public,result_fields,validator,bindings,choices in (
        ('adopt_scATAC_cell_by_features',adoption,tuple(AdoptedCellByCCREResult.__annotations__),validate_adoption,
         (('source','source',()),('reference','reference_manifest',())),
         {'matrix_semantics':('fragment_counts','insertion_counts','binary_accessibility')}),
        ('adapt_epizoo_species',adaptation,adaptation.RESULT_FIELDS,validate_adaptation,
         (('matrix','matrix_manifest',('scatac_cell_by_features.external.v1',)),('specification','adaptation_spec',())),
         {'strategy':('de_novo','mapped_reference')})):
        required={};ports=[]
        for key in public.ARGUMENTS:
            spec=_planning_argument((str,Path) if key.endswith('_path') or key=='output_dir' else (str,),
                'Explicit supplied binding; no inferred resources or strategy fallback.',
                source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT if key.startswith('matrix_manifest') else S.REQUEST_INPUT_ONLY,
                choices=choices.get(key,()))
            required[key]=(ArgumentSpec if key in choices else MatrixArgument)(spec.accepted_types,planning=spec.planning,choices=choices.get(key,()))
        for port,prefix,upstream in bindings:
            members=('manifest_path','manifest_sha256') if upstream else ('path','sha256')
            ports.append(SemanticConsumerPortSpec(port,tuple(member(m,prefix+'_'+s) for m,s in zip(members,('path','sha256'))),True,
                request_sources=(request(prefix+'_path',*(input_member(m,prefix+'_'+s) for m,s in zip(members,('path','sha256')))),),
                accepted_upstream_types=upstream))
        ports.extend(_semantic_argument_port(k,required=True) for k in (*choices,'output_dir'))
        is_adoption=public is adoption
        ints={'artifact_schema_version','n_cells','n_features','nnz','total_count','zero_row_count','completed_step','optimizer_steps','amp_skipped_steps'}
        fields={k:(int,) if k in ints else (dict,) if k=='species' else (str,type(None)) if k=='mapping_identity_sha256' else (str,) for k in result_fields}
        specs.append(ToolSpec(name=name,function=getattr(public,name),required_arguments=required,optional_arguments={},
            result_contract=ResultContract('NeutralMatrixResult' if is_adoption else 'SpeciesAdaptationResult',fields,
                validator=validator,planning_fields={k:_planning_result('Verified exact artifact binding.',bindable=True,
                    artifact=A.SCATAC_CELL_BY_FEATURES if is_adoption else A.EPIZOO_TARGET_MODEL) for k in ('manifest_path','manifest_sha256')}),
            exception_classifier=classify,retryable_error_codes=frozenset(),recovery_policy_version=public.RECOVERY_POLICY,
            durable_hooks=DurableToolHooks(public.execute,public.recover),
            planning=_tool_planning(PlanningToolRole.OPERATION,
                'Adopt exact new-species cell-by-regulatory-feature H5AD.' if is_adoption else 'Post-train pretrained EpiZoo for an explicit target species.',
                'Complete ordered neutral reference/genome; int64 CSR; declared value semantics. Peaks are exact features, never projected to canonical cCREs.' if is_adoption else
                'Qualified neutral fragment_counts matrix; epizoo-adaptation-inputs.v1 pins exact reference, source/SEAM bundles and qualification or production-candidate profile.',
                'No inferred raw/QC lineage, curated-cCRE or model-readiness claim.' if is_adoption else
                'Explicit de_novo or mapped_reference; mapped requires supplied chain/reference resources. No fallback, biological quality claim or target-model inference.',
                capability_ids=('species_adaptation',)),
            semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
                SemanticProducerPortSpec('matrix' if is_adoption else 'model',
                    'scatac_cell_by_features.external.v1' if is_adoption else 'epizoo_target_model.v1',
                    (member('manifest_path'),member('manifest_sha256'))),))))
    return tuple(specs)
