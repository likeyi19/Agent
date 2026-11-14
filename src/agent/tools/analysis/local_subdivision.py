"""ICTA-3C: read accepted native evidence for one local subdivision attempt.

IDclust-inspired retention, with SnapATAC2 logistic differential testing, not
IDclust reproduction. No execution, recursion, publication or label inference.
"""
from copy import deepcopy
import math

import h5py
from anndata.io import read_elem

from agent.orchestration.prior_outputs import validate_binding
from agent.schemas.orchestration import _serialize, StepOutputRef
from agent.schemas.prior_output import PriorOutputBinding, PriorOutputRef
from agent.schemas.verification_authority import AuthorityError
from agent.tools._file_identity import snapshots, unchanged, file_closure
from .cell_result import describe

LOCAL_METHODS = ('snapatac2.pp.select_features', 'snapatac2.tl.spectral',
                 'snapatac2.pp.knn', 'snapatac2.tl.leiden')


def _read(binding, store, registry):
    step = validate_binding(binding, store=store, registry=registry, integrity=True)
    return step, _serialize(step.result)


def _metadata(native, *slots):
    before = snapshots([native['matrix_path']])
    with h5py.File(native['matrix_path'], 'r') as handle:
        values = [read_elem(handle[slot]) for slot in slots]
    unchanged(before)
    return values


def _follows(step, native, previous):
    """Existing accepted source identities, never shape or barcode coincidence."""
    source = native['source_identity']
    if (step.resolved_arguments.get('input_path') != previous['matrix_path']
            or source['sha256'] != previous['matrix_sha256']
            or source.get('accepted_publication') != {
                k: previous[k] for k in ('manifest_path', 'manifest_sha256')}):
        raise AuthorityError('Subdivision evidence does not consume the exact accepted source.')


def _unchanged_population(native):
    parameters = native['parameters']
    if ('cell_subset' in parameters or 'input_orientation' in parameters
            or parameters.get('count_slot', 'X') != 'X'):
        raise AuthorityError('Evidence changed the selected local population or representation.')


def _positions(values, names):
    """Interpret recorded native selectors only; do not prepare or gate calls."""
    if all(type(v) is bool for v in values):
        return [i for i, selected in enumerate(values) if selected]
    if all(isinstance(v, str) for v in values):
        if len(set(names)) != len(names):
            raise AuthorityError('Barcode selectors do not identify unique local positions.')
        return [names.index(v) for v in values]
    return list(values)


def _failed_step(locator, source_binding, source, function, store):
    run = store.load(locator['run_id'])
    step = next(s for s in run.steps if s.step_id == locator['step_id'])
    planned = next(s for s in run.plan.steps if s.step_id == step.step_id)
    reference = planned.arguments.get('input_path')
    exact = reference == PriorOutputRef(source_binding, 'matrix_path') or (
        run.run_id == source_binding.run_id and reference == StepOutputRef(source_binding.step_id, 'matrix_path'))
    args = _serialize(step.resolved_arguments)
    if (not exact or step.status.value != 'FAILED' or step.tool_name != 'run_snapatac2'
            or args.get('function_name') != function or args.get('input_path') != source['matrix_path']):
        raise AuthorityError('Failure locator does not identify this bound native attempt.')
    status = 'native_execution_failure' if step.error.code == 'NATIVE_FUNCTION_REJECTED' else 'execution_failure'
    return status, dict(locator=deepcopy(locator), run_status=run.lifecycle_status.value, step=step.to_dict())


def differential_kwargs(local_binding, child, *, store, registry):
    """Derive child/rest positions from the actual accepted local Leiden output."""
    _, native = _read(local_binding, store, registry)
    obs, = _metadata(native, 'obs')
    return _comparison(native, child, obs)


def _comparison(native, child, obs):
    descriptor = describe('run_snapatac2', native)
    if native['function_name'] != LOCAL_METHODS[-1] or not descriptor or not descriptor['grouping']:
        raise ValueError('Select an accepted native local Leiden grouping.')
    labels = obs[descriptor['grouping']].tolist()
    if child == '-1':
        raise ValueError('SnapATAC2 -1 denotes unassigned cells, not a candidate child.')
    if child not in labels:
        raise ValueError('Child is absent from the accepted local grouping.')
    return dict(cell_group1=[i for i, label in enumerate(labels) if label == child],
                cell_group2=[i for i, label in enumerate(labels) if label != child], direction='positive')


def testable_features(matrix, cells, names, features=None):
    """ICTA-only preparation for installed SnapATAC2's sparse binary response.

    SnapATAC2 sets every stored CSC value to one, including explicitly stored
    zeros. Preserve that response exactly; do not alter X or its library sizes.
    """
    selected = _feature_positions(names, features)
    testable, provenance = _testable_response(matrix, cells)
    return [i for i in selected if testable[i]], dict(provenance,
        available_feature_count=len(names), selected_feature_count=len(selected),
        testable_feature_count=sum(bool(testable[i]) for i in selected))


def _feature_positions(names, features):
    from numbers import Integral
    selected = list(range(len(names))) if features is None else _positions(list(features), names)
    if any(isinstance(i, bool) or not isinstance(i, Integral) or not -len(names) <= i < len(names) for i in selected):
        raise ValueError('Differential features must select existing feature positions.')
    return [int(i) for i in selected]


def _testable_response(matrix, cells):
    import numpy as np
    from scipy import sparse
    response = sparse.csc_matrix(matrix[cells, :], copy=True)
    response.data = np.ones(response.data.shape)
    response.sum_duplicates()
    detected = np.diff(response.indptr)
    # Duplicate sparse entries can produce nonbinary native responses. They are
    # not silently repaired or excluded by this binary-constant correction.
    ones = np.asarray((response == 1).sum(axis=0)).ravel()
    constant_zero = detected == 0
    constant_one = ones == len(cells)
    return ~(constant_zero | constant_one), dict(constant_zero_feature_count=int(constant_zero.sum()),
        constant_one_feature_count=int(constant_one.sum()),
        response='Installed SnapATAC2 CSC stored values replaced by one on comparison cells',
        multiple_testing='Native BH over features passing native min_pct and min_log_fc after this selection; not the original feature universe')


def prepare_differential(local_binding, child, kwargs, *, store, registry):
    """Exact accepted child/rest population; only the native features argument changes."""
    return differential_preparer(local_binding, store=store, registry=registry)(child, kwargs)


def differential_preparer(local_binding, *, store, registry):
    """Prepare one exact parent axis for its child-versus-all-rest comparisons.

    Every comparison partitions this entire axis. Binary constant-feature counts
    are invariant to that partition's row ordering. Feature selectors and other
    kwargs remain per operation, including explicit result-edit overrides.
    """
    _, native = _read(local_binding, store, registry)
    before = snapshots([native['matrix_path']])
    obs, var, matrix = _metadata(native, 'obs', 'var', 'X')
    testable, facts = _testable_response(matrix, list(range(len(obs))))
    names = var.index.tolist()
    def prepare(child, kwargs):
        nonlocal before
        try:
            unchanged(before)
        except AuthorityError:
            # A timestamp/inode change invalidates reuse, not scientific input
            # legality. Recheck exact bytes; identical replacements remain valid.
            if file_closure([native['matrix_path']])[0]['sha256'] != native['matrix_sha256']:
                raise AuthorityError('Accepted differential source changed')
            before = snapshots([native['matrix_path']])
        comparison = _comparison(native, child, obs)
        selected = _feature_positions(names, kwargs.get('features'))
        features = [i for i in selected if testable[i]]
        provenance = dict(facts, available_feature_count=len(names), selected_feature_count=len(selected),
            testable_feature_count=len(features), source_binding=local_binding.to_dict(),
            source_sha256=native['matrix_sha256'], comparison=comparison,
            cell_ids=obs.index[comparison['cell_group1'] + comparison['cell_group2']].tolist())
        return dict(kwargs, **comparison, features=features), provenance
    return prepare


def _qualifying(rows, alpha):
    # Positive is a direction, not another effect-size cutoff. This matters when
    # a researcher explicitly allows zero/negative native min_log_fc values.
    return sorted({r['feature name'] for r in rows
                   if r['log2(fold_change)'] is not None and r['log2(fold_change)'] > 0
                   and r['adjusted p-value'] is not None
                   and math.isfinite(r['adjusted p-value']) and r['adjusted p-value'] < alpha})


def _decide(children, parent_positions, alpha, minimum):
    """Pure interpretation of already bound evidence; no native admission rule."""
    for child in children:
        if child['evidence_status'] != 'accepted':
            child.update(retained=None, qualifying_peak_count=None,
                         reason='Differential evidence unavailable; support was not assessed.')
            continue
        peaks = _qualifying(child['differential']['rows'], alpha)
        retained = len(peaks) >= minimum
        child.update(qualifying_peaks=peaks, qualifying_peak_count=len(peaks), retained=retained,
                     reason='Meets selected differential-peak rule.' if retained else
                            'Fewer distinct child-enriched peaks than the selected minimum.')
    kept = {p for c in children if c.get('retained') for p in c['source_positions']}
    residual = [p for p in parent_positions if p not in kept]
    if any(c['evidence_status'] == 'native_execution_failure' for c in children):
        outcome, reason = 'native_execution_failure', 'Native differential execution failed; no biological stopping conclusion.'
    elif any(c['evidence_status'] == 'execution_failure' for c in children):
        outcome, reason = 'execution_failure', 'Execution did not supply differential evidence; no biological stopping conclusion.'
    elif any(c['evidence_status'] != 'accepted' for c in children):
        outcome, reason = 'missing_input', 'Required differential evidence is unavailable; no biological stopping conclusion.'
    elif not kept:
        outcome, reason = 'no_supported_subdivision', 'No candidate child meets the selected differential-peak rule; parent terminal for this attempt.'
    elif residual:
        outcome, reason = 'partial_supported_subdivision', 'Retain supported children; unsupported cells remain in the residual parent without reclustering.'
    else:
        outcome, reason = 'supported_subdivision', 'Retain supported children; future exploration belongs to the caller.'
    return outcome, reason, residual


def assess_subdivision(parent_binding, local_bindings=(), differentials=None, *, store, registry,
                       adjusted_p_threshold=0.01, minimum_peak_features=5,
                       annotation=None, annotation_bindings=(), budget_reason=None, failed_local=None, source_grouping=None):
    """Return ordinary data for one attempt, never dispatch scientific work.

    ``parent_binding`` selects an accepted clustering retaining peak X. The four
    ordered ``local_bindings`` are the native route in LOCAL_METHODS; its first
    step's ICTA-2 cell_subset owns the exact parent membership. Differential keys
    are actual local labels, values accepted dataset bindings or failed Run/step
    locators {run_id, step_id}. Omitted evidence is explicitly unassessed.

    Optional ICTA-3B evidence is preserved whole. Its accepted construction/marker
    chain must descend from this local clustering, with identical ordered cells.
    External Gene Activity without accepted ancestry cannot establish that join.
    A supplied budget reason is recorded separately, without enforcing depth.
    """
    if (type(adjusted_p_threshold) not in (int, float) or not math.isfinite(adjusted_p_threshold)
            or not 0 <= adjusted_p_threshold <= 1):
        raise ValueError('Adjusted-p interpretation threshold must lie in [0, 1].')
    if type(minimum_peak_features) is not int or minimum_peak_features < 1:
        raise ValueError('Minimum distinct peak interpretation count must be a positive integer.')
    if source_grouping is None:
        _, parent = _read(parent_binding, store, registry)
        parent_record = dict(binding=parent_binding.to_dict(), native_result=parent)
        descriptor = describe(parent_binding.tool_name, parent)
    else:
        if parent_binding is not None:
            raise ValueError('Select either native grouping or the original source column')
        parent = source_grouping
        parent_record = dict(binding=None, source_grouping=source_grouping)
        descriptor = dict(preserves_counts=True, grouping=source_grouping['groupby'])
    result = dict(schema='local-subdivision.v1', parent=parent_record,
        local_steps=[], children=[], parent_membership=None, residual_parent=None, annotation=None,
        interpretation=dict(adjusted_p_threshold=adjusted_p_threshold, minimum_peak_features=minimum_peak_features,
            rule='distinct native features with positive log2 fold change and adjusted p strictly below threshold',
            method='IDclust-inspired retention using SnapATAC2 tl.diff_test logistic regression',
            references=['10.1093/nargab/lqae174', '10.1038/s41586-023-06824-9'],
            limitation='Exploratory within-data evidence, not independent biological validation or proof of a new cell type.'),
        execution_budget=dict(default_maximum_levels=5, supplied_reason=deepcopy(budget_reason), enforced=False))
    if not local_bindings and failed_local is None:
        result.update(outcome='missing_input', reason='Accepted local native clustering evidence is unavailable.')
        return result
    if len(local_bindings) != len(LOCAL_METHODS) and not (failed_local is not None and len(local_bindings) < 4):
        raise ValueError('Supply the four accepted native local steps; this component executes none.')
    parent_obs, = _metadata(parent if source_grouping is None else {'matrix_path': parent['source']['path']}, 'obs')
    previous = parent
    for i, (binding, method) in enumerate(zip(local_bindings, LOCAL_METHODS)):
        step, native = _read(binding, store, registry)
        if binding.tool_name != 'run_snapatac2' or native['function_name'] != method:
            raise ValueError('Local evidence differs from the selected SnapATAC2 route.')
        if i == 0 and source_grouping is not None:
            pin = source_grouping['source']
            if step.resolved_arguments.get('input_path') != pin['path'] or native['source_identity']['sha256'] != pin['sha256']:
                raise AuthorityError('Local selection does not consume the exact supplied grouping source')
        else:
            _follows(step, native, previous)
        written = ('var.selected', 'obsm.X_spectral', 'obsp.distances')
        if i < 3 and written[i] not in native['result_slots']:
            raise AuthorityError('Local evidence retained an inherited slot instead of computing it.')
        if i == 0:
            selection = native['parameters'].get('cell_subset')
            if (not selection or selection.get('groupby') != descriptor['grouping']
                    or not selection.get('complete_group_membership')
                    or len(selection.get('groups', [])) != 1
                    or selection.get('position_axis') != 'obs'):
                raise AuthorityError('Local selection must identify one complete accepted parent group.')
            positions = selection['positions']
            expected = [j for j, label in enumerate(parent_obs[descriptor['grouping']])
                        if label in selection['groups']]
            if len(positions) != len(set(positions)) or set(positions) != set(expected):
                raise AuthorityError('Local parent membership has losses, duplications or substitutions.')
            if native['parameters'].get('count_slot', 'X') != 'X':
                raise AuthorityError('Local subdivision must retain the selected original peak X.')
            result['parent_membership'] = deepcopy(selection)
        else:
            _unchanged_population(native)
        if i >= 2:
            graphs = native['parameters'].get('graph_inputs', [])
            spectral = result['local_steps'][1]['native_result']
            if not any(g['function'] == LOCAL_METHODS[2] and g['representation'] == 'X_spectral'
                       and g['graph'] == 'distances' and g['source_sha256'] == spectral['matrix_sha256']
                       for g in graphs):
                raise AuthorityError('Local graph lacks the accepted local Spectral source.')
        result['local_steps'].append(dict(binding=binding.to_dict(), native_result=native))
        previous = native
    if failed_local is not None:
        if len(local_bindings) >= 4:
            raise ValueError('A failed local step must follow the supplied successful local prefix.')
        status, failure = _failed_step(failed_local, local_bindings[-1] if local_bindings else parent_binding,
                                      previous, LOCAL_METHODS[len(local_bindings)].removeprefix('snapatac2.'), store)
        result.update(outcome=status, failure=failure, reason='Local native execution did not complete; no biological stopping conclusion.')
        return result
    local = previous
    obs, var = _metadata(local, 'obs', 'var')
    if obs.index.tolist() != selection['cell_ids']:
        raise AuthorityError('Local ordered cells differ from the accepted subset.')
    local_descriptor = describe('run_snapatac2', local)
    if not local_descriptor['grouping']:
        raise ValueError('Local Leiden returned no newly written grouping.')
    labels = obs[local_descriptor['grouping']]
    if labels.isna().any():
        result.update(outcome='missing_input', reason='Local grouping contains unassigned cells; complete candidate membership is unavailable.')
        return result
    # Installed Leiden uses -1 for native-unassigned observations. These remain
    # in every child-versus-rest comparison and in residual, never a named child.
    unassigned = [i for i, label in enumerate(labels) if label == '-1']
    result['unassigned_local_positions'] = unassigned
    groups = [g for g in dict.fromkeys(labels.tolist()) if g != '-1']
    differentials = {} if differentials is None else differentials
    if set(differentials) - set(groups):
        raise AuthorityError('Differential evidence names a child outside this local result.')
    if annotation is not None:
        if not annotation_bindings or annotation['provenance']['binding'] != annotation_bindings[-1].to_dict():
            raise AuthorityError('Annotation must retain its exact accepted marker binding.')
        previous = local
        for binding in annotation_bindings:
            step, native = _read(binding, store, registry)
            _follows(step, native, previous)
            _unchanged_population(native)
            annotated_obs, = _metadata(native, 'obs')
            grouping = local_descriptor['grouping']
            if (not annotated_obs.index.equals(obs.index) or grouping not in annotated_obs
                    or annotated_obs[grouping].tolist() != obs[grouping].tolist()):
                raise AuthorityError('Annotation changed the bound local population or grouping.')
            previous = native
        if (annotation_bindings[-1].tool_name != 'run_scanpy'
                or previous['function_name'] != 'scanpy.tl.rank_genes_groups'
                or annotation['provenance']['native_result'] != previous):
            raise AuthorityError('Annotation native provenance differs from the accepted publication.')
        result['annotation'] = deepcopy(annotation)
    if not groups or (len(groups) == 1 and not unassigned):
        result.update(outcome='no_candidate_subdivision', reason=(
                          'Native Leiden left all cells unassigned; no candidate subdivision was produced.' if not groups else
                          'Local clustering produced only one population; no proposed split to retain.'),
                      residual_parent=deepcopy(selection))
        return result
    for group in groups:
        local_positions = [i for i, label in enumerate(labels) if label == group]
        rest = [i for i, label in enumerate(labels) if label != group]
        child = dict(cluster=group, local_positions=local_positions,
                     source_positions=[positions[i] for i in local_positions],
                     cell_ids=obs.index[local_positions].tolist(), evidence_status='missing_input',
                     annotation_cluster=None, exploratory_unnamed=True)
        if annotation is not None:
            matched = [c for c in annotation['clusters'] if c['cluster'] == group
                       and c['groupby'] == local_descriptor['grouping']]
            if len(matched) == 1:
                child['annotation_cluster'] = deepcopy(matched[0])
                child['exploratory_unnamed'] = matched[0]['status'] != 'assigned'
        evidence = differentials.get(group)
        if isinstance(evidence, PriorOutputBinding):
            step, native = _read(evidence, store, registry)
            _follows(step, native, local)
            _unchanged_population(native)
            if evidence.tool_name != 'run_snapatac2' or native['function_name'] != 'snapatac2.tl.diff_test':
                raise ValueError('Differential evidence must be accepted native SnapATAC2 tl.diff_test.')
            effective = native['parameters']['effective_native_parameters']
            if (effective['direction'] != 'positive'
                    or sorted(_positions(effective['cell_group1'], obs.index.tolist())) != local_positions
                    or sorted(_positions(effective['cell_group2'], obs.index.tolist())) != rest):
                raise AuthorityError('Differential comparison is not the actual child versus remaining parent.')
            returned, = _metadata(native, 'uns/agent_native_return/value')
            rows = returned.to_dict(orient='records')
            if any(r['feature name'] not in var.index for r in rows):
                raise AuthorityError('Differential feature is absent from the selected peak axis.')
            for row in rows:
                for key in ('log2(fold_change)', 'p-value', 'adjusted p-value'):
                    if not math.isfinite(row[key]): row[key] = None
            child.update(evidence_status='accepted', differential=dict(binding=evidence.to_dict(),
                native_result=native, rows=rows, method='SnapATAC2 tl.diff_test logistic-regression likelihood-ratio test; native BH adjustment'))
        elif evidence is not None:
            status, failure = _failed_step(evidence, local_bindings[-1], local, 'tl.diff_test', store)
            args = failure['step']['resolved_arguments']
            effective = args.get('kwargs', {})
            if (effective.get('direction') != 'positive'
                    or sorted(_positions(effective.get('cell_group1', []), obs.index.tolist())) != local_positions
                    or sorted(_positions(effective.get('cell_group2', []), obs.index.tolist())) != rest):
                raise AuthorityError('Failure locator does not identify this child differential attempt.')
            child.update(evidence_status=status, failure=failure)
        result['children'].append(child)
    outcome, reason, residual = _decide(result['children'], positions, adjusted_p_threshold, minimum_peak_features)
    result.update(outcome=outcome, reason=reason, residual_parent=dict(source_sha256=selection['source_sha256'],
        positions=residual, cell_ids=parent_obs.index[residual].tolist(), position_axis='obs'))
    return result
