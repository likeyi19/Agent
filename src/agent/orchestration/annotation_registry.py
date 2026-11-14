"""Single production iterative annotation operation and its native authority."""
from pathlib import Path
from collections.abc import Mapping
from agent.tools.analysis import iterative_annotation as coordinator, hierarchical_annotation as owner
from .native_matrix_registry import classify as native_classify


def classify(exception):
    from .annotation_runs import AnnotationChildError
    from .registry import ErrorClassification
    if isinstance(exception, AnnotationChildError):
        error = exception.error
        return ErrorClassification(error.category, error.code, error.details.get('public_diagnostic'))
    return native_classify(exception)


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
    optional = dict(grouping_path=arg((str, Path), 'Optional distinct accepted native initial clustering. Omit groups when reusing an original obs group_column; inspection is not a clustering result.',
            artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        group_column=arg((str,), 'Explicit original source obs column to reuse at level 1; user-supplied grouping, without validation of biological identity or derivation.',
            scientific_parameter=True, observation_source='source'),
        input_sha256=arg((str,), 'Exact original file identity.'),
        activity_path=arg((str, Path), 'Optional genuine accepted Gene Activity H5AD, constructed by a native gene-activity operation. Inspection of accessibility does not construct Gene Activity. Omit activity to construct it inside annotation using activity_method.', source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        activity_source_path=arg((str, Path), 'Optional distinct accepted source for native Gene Activity construction. Omit activity_source when constructing from the annotation source or its selected grouping: use_x=True selects its X for SnapATAC2; epiScanpy uses its peak X. Default SnapATAC2 requires accepted fragment/insertion storage.', source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
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
            'Port source is the original accessibility dataset. With original obs group_column, omit groups. '
            'Omit activity unless genuine constructed Gene Activity is available. For construction from original peak X, '
            'choose activity_method and its applicable activity_kwargs; omit activity_source. The coordinator performs the native construction.',
            'inspect_scATAC.dataset is an unchanged accessibility handoff, not Gene Activity, native grouping or a distinct activity construction source. '
            'Optional UMAP prerequisites consume that inspection dataset through their own dataset ports, without changing annotation input roles.',
            'SnapATAC2 use_x=False requires '
            'actual accepted fragment/insertion storage; use_x=True uses the selected X; epiScanpy uses peak X. Never infer cross-file '
            'correspondence from names or shapes. The intended input is cell-by-peak accessibility. Native input legality and failures remain visible; no Agent peak-space eligibility certificate is required.',
            'Choose a Gene Activity construction supported by the offered inputs. When only the original peak-count X is available, '
            'epiScanpy tl.geneactivity can construct activity directly. Do not choose SnapATAC2 use_x=False and request missing fragments '
            'when a registered construction already supports the supplied X. Counts described as cut sites or insertions in X are not '
            'native fragment/insertion storage. An explicit use_x=True setting enables that SnapATAC2 X route.',
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
