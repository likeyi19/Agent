"""General annotation comparison through existing scientific lifecycle owners."""
from pathlib import Path
from agent.tools.analysis import cell_label_comparison as owner


def tool_spec():
    from .registry import (ToolSpec, ResultContract, DurableToolHooks, ScientificAuthorityHooks,
        ArtifactSemanticKind as A, PlanningToolRole, PlanningSourceEligibility as S, SemanticToolSpec,
        SemanticConsumerPortSpec as Consumer, SemanticProducerPortSpec as Producer, _planning_argument as arg,
        _planning_result, _tool_planning, _semantic_member as member, _semantic_request as request,
        _semantic_request_member as rm, _semantic_argument_port as scalar)
    from .native_matrix_registry import classify
    from .native_delivery import delivery_spec
    required = dict(output_dir=arg((str,Path), 'Managed fixed-label comparison destination.'))
    ports = []
    for role in ('prediction','reference'):
        key = role+'_path'
        required[key] = arg((str,Path), 'Fixed '+role+' H5AD obs or TSV/CSV label table; feature axes need not match.',
            source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT, artifacts=(A.RAW_SCATAC,A.CELL_ANNOTATION))
        ports.append(Consumer(role+'_dataset', (member('dataset',key),), True,
            request_sources=(request(key,rm('dataset',key)),),
            accepted_upstream_types=('scatac_matrix_h5ad.v1','native_matrix_h5ad.v1')))
        required[role+'_label_key'] = arg((str,), 'Exact '+role+' categorical label column; native MAESTRO uses primary_annotation.',
            scientific_parameter=True, conversational_exact_string=True, observation_source=role+'_dataset')
    optional = {}
    for role in ('prediction','reference'):
        for name, description in (('cell_key','Cell identifier column; omit only for H5AD obs_names.'),
            ('sample_key','Sample/library identity column. Both sources must identify the same physical libraries; unrelated sample barcodes never match.')):
            optional[role+'_'+name] = arg((str,), description, scientific_parameter=True,
                conversational_exact_string=True, observation_source=role+'_dataset')
        optional[role+'_sha256'] = arg((str,), 'Optional exact captured source checksum.')
    optional['shared_library_id'] = arg((str,), 'Explicit user declaration that BOTH label sources describe the same physical sample/library, with this exact identity. Required when no pair of library columns is selected. Never infer from filenames, equal cell counts or barcodes.',
        scientific_parameter=True, conversational_exact_string=True)
    for side in ('prediction', 'reference'):
        optional[side+'_mapping'] = arg((dict,),
            'Explicit '+side+' original-label to common-label string mapping. Only user-specified one-to-one or many-to-one rules; unspecified labels stay unchanged. Resolve exact original category names; ask for missing/unknown/conflicting correspondence. Never infer equivalence, split categories, map missing/unassigned values or change cell identities.',
            scientific_parameter=True, conversational_json_object=True, conversational_string_mapping=True)
    optional['umap_path'] = arg((str,Path), 'Optional already accepted UMAP on the exact prediction source; reuse coordinates for predicted/reference/agreement views, never refit.',
        source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT, artifacts=(A.UMAP_ANALYSIS,))
    optional['annotation_manifest_path'] = arg((str,Path), 'Accepted native MAESTRO publication authenticating the prediction file and original-source ancestry when reusing UMAP from original counts.',source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT,artifacts=(A.SCATAC_CELL_TYPE_ANNOTATION,))
    optional['annotation_manifest_sha256'] = arg((str,), 'Exact accepted native annotation manifest checksum.')
    ports.append(Consumer('annotation', (member('manifest_path','annotation_manifest_path'), member('manifest_sha256','annotation_manifest_sha256')),False,accepted_upstream_types=('scatac_cell_type_annotation.v1',)))
    optional['umap_sha256'] = arg((str,), 'Exact accepted UMAP checksum.')
    ports.append(Consumer('coordinates', (member('analysis','umap_path'),), False,
        accepted_upstream_types=('umap_analysis.v1',)))
    ports.extend(scalar(k, required=k in required) for k in (*required,*optional) if k not in ('prediction_path','reference_path','umap_path','annotation_manifest_path','annotation_manifest_sha256'))
    pub = owner.publication
    return ToolSpec(name=owner.TOOL,function=owner.compare_cell_annotations,required_arguments=required,optional_arguments=optional,
        result_contract=ResultContract('CellLabelComparisonResult', owner.RESULT_FIELDS,
            planning_fields={k:_planning_result('Fixed label comparison, coverage and aligned labels.',bindable=True,
                artifact=A.ANNOTATION_EVALUATION_REPORT) for k in ('manifest_path','manifest_sha256','comparison_path','aligned_path')}),
        durable_hooks=DurableToolHooks(owner.execute,owner.recover),parameter_prerequisites=owner.parameter_prerequisites,exception_classifier=classify,recovery_policy_version=owner.POLICY,
        scientific_delivery=delivery_spec(owner,'scientific_native_result'),
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND,producer_ports=('comparison',),
            describe=pub.describe,publication_record=pub.publication_record,reuse_result=pub.reuse_result,
            proof_metadata=pub.proof_metadata,independent_active_publications=True),
        planning=_tool_planning(PlanningToolRole.EVALUATION,
            'Compare fixed native MAESTRO or supplied cell labels with independently uploaded H5AD/TSV/CSV reference annotations.',
            'Use exact (library, cell identifier) correspondence; ask for missing identity/columns. Feature axes and row order may differ. Strict comparison is default. prediction_mapping/reference_mapping require explicit user rules, never inferred equivalence. A mapped continuation reuses exact original sources, columns and library declaration, publishing a new result. No measurement comparison, training or biological truth claim.',
            'Report matched/unmatched and missing coverage, accuracy, macro-F1, balanced accuracy, per-class results and descriptive ARI/NMI. Reuse a compatible accepted UMAP through coordinates for label overlays. For native MAESTRO predictions bind its accepted annotation publication too, to authenticate original-source ancestry. Do not compute another UMAP for comparison.',
            'A previous comparison publication is not a label dataset. Recomparison must bind the original prediction/reference label files: select their exact captured upload roles, or the compatible accepted label artifacts from their original results. context=current alone does not supply those inputs. Preserve original columns and the library declaration using accepted parameters. A missing-mapping request still needs these original source roles before its parameter continuation.',
            capability_ids=('reference_annotation',)),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(Producer('comparison',owner.CONTRACT,
            (member('path','manifest_path'),member('sha256','manifest_sha256'))),)))
