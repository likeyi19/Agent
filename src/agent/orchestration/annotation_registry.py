"""Single production iterative annotation operation and its native authority."""
from pathlib import Path
from collections.abc import Mapping
from agent.tools.analysis import iterative_annotation as coordinator, hierarchical_annotation as owner
from .native_matrix_registry import classify


def annotation_tool_spec():
    from .registry import (ToolSpec, ResultContract, DurableToolHooks, ScientificAuthorityHooks,
        ArtifactSemanticKind as A, PlanningToolRole, PlanningSourceEligibility as S, SemanticToolSpec,
        SemanticConsumerPortSpec as Consumer, SemanticProducerPortSpec as Producer,
        _planning_argument as arg, _planning_result, _tool_planning, _semantic_member as member,
        _semantic_request as request, _semantic_request_member as rm, _semantic_argument_port as scalar)
    required = dict(
        input_path=arg((str, Path), 'Original peak accessibility H5AD; retain all scientific slots.',
            artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        cellmarker_id=arg((str,), 'Installed CellMarker 3.0 evidence selected by the scientific resource library.'),
        output_dir=arg((str, Path), 'Managed hierarchical publication destination.'))
    optional = dict(grouping_path=arg((str, Path), 'Exact accepted native initial clustering with original source ancestry.',
            artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        group_column=arg((str,), 'Explicit original source obs column to reuse at level 1; user-supplied grouping, without validation of biological identity or derivation.',
            scientific_parameter=True, observation_source='source'),
        input_sha256=arg((str,), 'Exact original file identity.'),
        activity_path=arg((str, Path), 'Compatible accepted Gene Activity; reused without reconstruction.', source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        activity_source_path=arg((str, Path), 'Accepted native activity construction source: fragment/insertion H5AD for default SnapATAC2, selected peak X for use_x=True or epiScanpy.', source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        gene_anno_path=arg((str, Path), 'Native gene annotation reference.', source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        gene_anno_sha256=arg((str,), 'Exact gene annotation reference identity.'),
        cellmarker_identity=arg((Mapping,), 'Captured installed CellMarker resource identity.'))
    parameters = dict(
        activity_method=((str,), 'Registered native Gene Activity construction when accepted activity is absent.'),
        activity_kwargs=((dict,), 'Exact native Gene Activity kwargs.'),
        activity_count_slot=((str,), 'Selected native X or layer for Gene Activity construction.'),
        maximum_levels=((int,), 'Maximum hierarchy depth, including initial level 1; default five.'),
        local_kwargs=((dict,), 'Exact local SnapATAC2 parameters by native function.'),
        differential_kwargs=((dict,), 'Native differential-test parameters; populations are derived from accepted membership.'),
        parameter_selection=((str,), 'Bounded ASW for unspecified local resolution, or explicitly requested native_defaults.'),
        species=((str,), 'Declared species for exact CellMarker evidence matching.'),
        marker_alpha=((float, int), 'Marker adjusted-p interpretation threshold; default 0.05.'),
        ora_alpha=((float, int), 'BH-corrected hypergeometric ORA threshold; default 0.05.'),
        adjusted_p_threshold=((float, int), 'Differential peak adjusted-p threshold; default 0.01.'),
        minimum_peak_features=((int,), 'Minimum distinct qualifying peaks; default five.'),
        marker_context=((dict,), 'Explicit CellMarker context restriction.'),
        identity_kind=((str,), 'Exact gene identity kind; default name.'),
        identity_column=((str,), 'Explicit Gene Activity feature identity column.'),
        operation_overrides=((dict,), 'Result-scoped branch/operation edits captured by the existing parameter editor.'))
    for name, (types, description) in parameters.items():
        options = dict(scientific_parameter=True)
        if types == (dict,): options['conversational_json_object'] = True
        elif types == (str,): options['conversational_exact_string'] = True
        if name == 'activity_method':
            options.update(choices=('snapatac2.pp.make_gene_matrix', 'episcanpy.tl.geneactivity'), planner_choice=True, conversational_choice=True)
            options.pop('conversational_exact_string', None)
        optional[name] = arg(types, description, **options)
    optional['previous_annotation_path'] = arg((str, Path), 'Exact selected accepted annotation for dependency-checked result editing.', source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT)
    ports = [Consumer('source', (member('dataset','input_path'),), True,
        request_sources=(request('input_path', rm('dataset','input_path')),),
        accepted_upstream_types=('raw_scatac_dataset.v1','scatac_matrix_h5ad.v1','hierarchical_annotation_h5ad.v1','native_matrix_h5ad.v1'))]
    for name in ('grouping', 'activity', 'activity_source'):
        ports.append(Consumer('groups' if name=='grouping' else name, (member('dataset',name+'_path'),), False,
            request_sources=(request(name+'_path',rm('dataset',name+'_path')),),
            accepted_upstream_types=('native_matrix_h5ad.v1',)))
    ports.append(Consumer('gene_anno', (member('path','gene_anno_path'),member('sha256','gene_anno_sha256')), False,
        request_sources=(request('gene_anno_path',rm('path','gene_anno_path'),rm('sha256','gene_anno_sha256')),)))
    ports.append(Consumer('previous_annotation', (member('manifest_path','previous_annotation_path'),), False, accepted_upstream_types=('hierarchical_annotation.v1',)))
    excluded = {'previous_annotation_path','input_path','grouping_path','activity_path','activity_source_path','gene_anno_path','gene_anno_sha256'}
    ports.extend(scalar(k, required=k in required) for k in (*required,*optional) if k not in excluded)
    from .native_delivery import delivery_spec
    pub = owner.publication
    return ToolSpec(owner.TOOL, coordinator.annotate_scATAC_cell_types, required, optional,
        ResultContract('HierarchicalAnnotation', owner.RESULT_FIELDS,
            planning_fields={k:_planning_result('Accepted hierarchical annotation and complete evidence.',
                bindable=True,artifact=A.RAW_SCATAC if k=='annotated_h5ad_path' else A.SCATAC_CELL_TYPE_ANNOTATION)
                for k in owner.RESULT_FIELDS if k.endswith(('_path','_sha256'))}),
        classify, recovery_policy_version='hierarchical-annotation-v1',
        durable_hooks=DurableToolHooks(coordinator.coordinate, owner.recover),
        scientific_authority=ScientificAuthorityHooks(owner.KIND, ('dataset','annotation'), pub.describe,
            pub.publication_record, pub.reuse_result, pub.proof_metadata),
        planning=_tool_planning(PlanningToolRole.OPERATION, 'Automatically annotate scATAC cell types hierarchically.',
            'Select original peak accessibility. An explicit source group_column reuses that obs grouping directly at level 1; do not plan initial clustering for it. Otherwise reuse exact accepted native grouping. Reuse compatible Gene Activity. '
            'If clustering is missing, plan the necessary registered native representation, neighbors and clustering steps. '
            'For Gene Activity choose an applicable registered native construction and gene annotation reference.',
            'Default SnapATAC2 uses '
            'actual accepted fragment/insertion storage; use_x=True uses the selected X; epiScanpy uses peak X. Never infer cross-file '
            'correspondence from names or shapes. Gene-only inputs can annotate the current level but cannot subdivide in peak space.',
            'Deterministic iteration uses native Scanpy Wilcoxon/BH, CellMarker exact matching, SciPy hypergeometric ORA/BH, '
            'local peak-space SnapATAC2 feature selection/Spectral/KNN/Leiden with bounded ASW and native child-versus-rest differential peaks. '
            'Default maximum five levels. No per-iteration model planning.',
            'UMAP is optional: reuse accepted coordinates without refitting. '
            'Labels are computational candidates, not biological validation.', capability_ids=('marker_annotation',)),
        application_inputs=frozenset({'cellmarker_id','cellmarker_identity','operation_overrides','previous_annotation_path',
            'species','grouping_path','gene_anno_path','gene_anno_sha256'}),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
            Producer('annotation','hierarchical_annotation.v1',(member('manifest_path'),member('manifest_sha256'))),
            Producer('dataset','hierarchical_annotation_h5ad.v1',(member('dataset','annotated_h5ad_path'),member('sha256','annotated_h5ad_sha256'))))),
        scientific_delivery=delivery_spec(owner, 'scientific_annotation', h5ad_roles=('annotated.h5ad',)),
        annotation_children=coordinator.CHILDREN)
