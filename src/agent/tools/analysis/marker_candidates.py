"""Internal current-round evidence from accepted Scanpy markers and CellMarker.

No execution, publication, label mutation or subdivision. The caller submits the
ordinary native step, then supplies its accepted binding. All returned data use
the existing JSON serialization; the accepted H5AD and static index own originals.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from importlib.metadata import version
import math

import h5py
from anndata._io.specs import read_elem
from scipy.stats import false_discovery_control, hypergeom

from agent.orchestration.prior_outputs import validate_binding
from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError
from agent.tools._file_identity import file_closure, snapshots, unchanged

CONTEXT = frozenset(('tissue_class', 'tissue_type', 'disease_category', 'disease'))


def marker_kwargs(groupby):
    """Approved defaults for an ordinary registered Scanpy invocation on X."""
    return dict(groupby=groupby, method='wilcoxon', corr_method='benjamini-hochberg',
                reference='rest', pts=True, use_raw=False, n_genes=None)


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _interpret(universe, positive, candidates, alpha):
    """One cluster's complete test family; SciPy owns both statistics."""
    universe, positive = set(universe), set(positive) & set(universe)
    tests, uncovered = [], []
    for candidate in candidates:
        genes = set(candidate['genes']) & universe
        if not genes:
            uncovered.append(candidate['candidate'])
            continue
        overlap = sorted(genes & positive)
        tests.append(dict(candidate=candidate['candidate'], N=len(universe), K=len(genes),
            n=len(positive), k=len(overlap), matched_genes=overlap,
            pvalue=float(hypergeom.sf(len(overlap) - 1, len(universe), len(genes), len(positive)))))
    adjusted = false_discovery_control([t['pvalue'] for t in tests], method='bh') if tests else []
    for test, q in zip(tests, adjusted):
        test.update(pvalue_adj=float(q), significant=bool(q <= alpha))
    significant = [t for t in tests if t['significant']]
    best = min((t['pvalue_adj'] for t in significant), default=None)
    preferred = [t['candidate'] for t in significant if t['pvalue_adj'] == best]
    status = ('insufficient_evidence' if not universe or not tests else
              'unresolved' if not preferred else 'assigned' if len(preferred) == 1 else 'ambiguous')
    return dict(status=status, preferred=preferred, significant=[t['candidate'] for t in significant],
        tests=tests, untestable_candidates=uncovered, testing_family_size=len(tests),
        positive_genes=sorted(positive))


def _reference(lookup, universe, kind, species, context):
    """Union distinct input identities, never rows, publications or algorithms."""
    candidates = {(r['species'], r['cell_name']): dict(candidate=r, genes=set(),
        categories=set(), fields=set(), methods=set(), contexts=set(), literature=set(),
        evidence_items=0, original_rows=0) for r in lookup.candidates(species=species, filters=context)
        if r['cell_name'] != ''}
    query = dict(names=universe if kind == 'name' else [],
                 gene_ids=universe if kind == 'gene_id' else [], species=species, filters=context)
    for page in lookup.query(**query):
        for item in page['items']:
            fields = item['fields']
            candidate = candidates.get((fields['species'], fields['cell_name']))
            if candidate is None:
                continue
            # One source association supplies one representation, not an alias
            # equivalence claim. Keep every original match in lazy evidence.
            matched = {m['field']: m['value'] for m in item['matches']}
            selected = (matched.get('symbol') or matched.get('marker')
                        if kind == 'name' else matched.get('gene_id'))
            candidate['genes'].add(selected)
            candidate['categories'].add(fields['marker_source'])
            candidate['fields'].update(m['field'] for m in item['matches'])
            if fields['method_details']:
                candidate['methods'].add(fields['method_details'])
            candidate['contexts'].add(tuple(fields[k] for k in sorted(CONTEXT)))
            literature = tuple(fields[k] for k in ('pmid', 'title', 'journal', 'year'))
            if any(literature):
                candidate['literature'].add(literature)
            candidate['evidence_items'] += 1
            candidate['original_rows'] += item['multiplicity']
    result = []
    for row in candidates.values():
        result.append(dict(candidate=row['candidate'], genes=sorted(row['genes']),
            coverage='covered' if row['genes'] else 'no_markers_in_tested_background',
            evidence=dict(source_categories=sorted(row['categories']), matched_fields=sorted(row['fields']),
                method_details=sorted(row['methods']), context_records=len(row['contexts']),
                literature_records=len(row['literature']), distinct_records=row['evidence_items'],
                original_rows=row['original_rows'],
                locator=dict(species=row['candidate']['species'],
                    filters=dict(context, cell_name=row['candidate']['cell_name'])))))
    return result, query


def candidate_evidence(result, candidate, lookup, *, page_size=128):
    """Read original fields, exact field matches and source-row references lazily.

    Candidate results carry no raw-record payload. This exact pinned query keeps
    every tissue/disease/literature/method record available without a hit cap.
    """
    if result['reference']['resource'] != lookup.identity:
        raise AuthorityError('Candidate evidence belongs to a different CellMarker resource.')
    row = next(r for r in result['candidates'] if r['candidate'] == candidate)
    query = dict(result['reference']['query'], **row['evidence']['locator'])
    yield from lookup.query(**query, page_size=page_size)


def annotate_markers(binding, *, store, registry, evidence=None, species=None, context=None,
                     identity_kind='name', identity_column=None, marker_alpha=0.05, ora_alpha=0.05):
    """Interpret one exact accepted marker publication; never execute native work.

    Identity is one explicitly selected exact name/ID per feature, defaulting to
    var_names. No alias inference or cross-namespace union. Repeated identities
    count once; any positive tested feature with that identity selects the gene.
    Ambiguous feature-name-to-ID associations are excluded locally.
    Missing reference/identities and incomplete marker tables are evidence outcomes,
    while altered accepted bytes/anchors remain genuine integrity errors.
    """
    if identity_kind not in ('name', 'gene_id'):
        raise ValueError('Identity kind must be name or gene_id.')
    context = dict(context or {})
    if not set(context) <= CONTEXT or any(not isinstance(v, str) for v in context.values()):
        raise ValueError('Optional context uses exact tissue/disease fields only.')
    if species is not None and not isinstance(species, str):
        raise ValueError('Species must be an exact reference string or omitted.')
    for alpha in (marker_alpha, ora_alpha):
        if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('Evidence thresholds must lie in [0, 1].')
    step = validate_binding(binding, store=store, registry=registry, integrity=True)
    native = _serialize(step.result)
    if binding.tool_name != 'run_scanpy' or native['function_name'] != 'scanpy.tl.rank_genes_groups':
        raise ValueError('Select an accepted registered Scanpy rank_genes_groups result.')
    before = snapshots([native['matrix_path']])
    parameters = native['parameters']
    effective = parameters['effective_native_parameters']
    key = parameters['kwargs'].get('key_added') or 'rank_genes_groups'
    with h5py.File(native['matrix_path'], 'r') as handle:
        var = read_elem(handle['var'])
        ranked = read_elem(handle['uns'][key])
    unchanged(before)
    features = var.index.tolist()
    values = (features if identity_column is None else
              var[identity_column].tolist() if identity_column in var else [None] * len(features))
    identities = defaultdict(set)
    for feature, value in zip(features, values):
        identities[feature].add(value if isinstance(value, str) and value != '' else None)
    mapping = {name: next(iter(ids)) if len(ids) == 1 else None for name, ids in identities.items()}
    exclusions = {name: ('conflicting identities for the same native feature name'
                        if len(identities[name]) != 1 else 'no nonempty exact gene identity')
                  for name, identity in mapping.items() if identity is None}
    universe = sorted({v for v in mapping.values() if v is not None})
    reasons = []
    if effective.get('method') != 'wilcoxon' or effective.get('corr_method') != 'benjamini-hochberg':
        reasons.append('Selected native statistics are not Wilcoxon with BH correction.')
    if ranked['params'].get('use_raw') != False or effective.get('layer') is not None:
        reasons.append('Selected native statistics do not describe the selected Gene Activity X.')
    if 'gene_activity' not in parameters:
        reasons.append('Selected publication has no Gene Activity declaration or accepted handoff.')
    if not universe:
        reasons.append('No comparable tested gene identities are available.')
    groups = ranked['names'].dtype.names
    if any(Counter(ranked['names'][g].tolist()) != Counter(features) for g in groups):
        reasons.append('Selected marker table does not retain the full tested feature set.')
    result = dict(schema='marker-candidates.v1',
        provenance=dict(binding=binding.to_dict(), native_result=native,
            evidence_implementation=file_closure([__file__]), scipy_version=version('scipy')),
        interpretation=dict(marker_alpha=marker_alpha, ora_alpha=ora_alpha,
            positive_rule='native BH p-value <= marker_alpha and signed Wilcoxon score > 0',
            ora='scipy.stats.hypergeom.sf(k - 1, N, K, n)', correction='Benjamini-Hochberg per cluster, all K > 0 candidates',
            candidate_identity='exact (species, cell_name)', identity_kind=identity_kind,
            identity_column=identity_column, duplicate_rule='distinct literal gene identity; any positive tested feature',
            reference_identity_rule='one identity per source association: matched symbol, otherwise matched marker; exact gene_id for ID queries',
            scope='current-round computational candidates; no biological certainty or subdivision decision'),
        tested_features=[dict(feature=f, identity=v if isinstance(v, str) and v != '' else None)
                         for f, v in zip(features, values)],
        universe=universe, excluded_features=[f for f in features if mapping[f] is None],
        identity_exclusions=exclusions,
        reference=None, candidates=[], clusters=[])
    if evidence is None:
        reasons.append('CellMarker reference evidence is unavailable.')
    if evidence is not None:
        # No new identity system: each query uses exactly one declared namespace.
        candidates, query = _reference(evidence, universe, identity_kind, species, context)
        result['reference'] = dict(resource=deepcopy(evidence.identity), query=query)
        result['candidates'] = candidates
    for group in groups:
        markers, positive = [], set()
        for i, feature in enumerate(ranked['names'][group]):
            score, p, q = (_number(ranked[k][group][i]) for k in ('scores', 'pvals', 'pvals_adj'))
            identity = mapping.get(feature)
            selected = score is not None and score > 0 and q is not None and q <= marker_alpha
            if selected and identity is not None:
                positive.add(identity)
            fractions = {}
            for slot in ('pts', 'pts_rest'):
                if slot in ranked and group in ranked[slot]:
                    value = ranked[slot].loc[feature, group]
                    # Repeated feature names have no rank-to-position mapping in
                    # Scanpy. Retain all native fractions, without inventing one.
                    fractions[slot] = ([_number(v) for v in value.tolist()] if hasattr(value, 'tolist')
                                       and getattr(value, 'ndim', 0) else _number(value))
            markers.append(dict(feature=str(feature), identity=identity, exclusion_reason=exclusions.get(feature), score=score, pvalue=p,
                pvalue_adj=q, positive=bool(selected), detection_fractions=fractions))
        group_reasons = list(reasons)
        if not any(m['identity'] is not None and m['score'] is not None and m['pvalue_adj'] is not None for m in markers):
            group_reasons.append('Comparable genes have no finite score and adjusted p-value pair in the native statistics.')
        outcome = _interpret(universe, positive, result['candidates'], ora_alpha) if not group_reasons else dict(
            status='insufficient_evidence', preferred=[], significant=[], tests=[],
            untestable_candidates=[], testing_family_size=0, positive_genes=sorted(positive))
        if outcome['status'] == 'insufficient_evidence' and not group_reasons:
            group_reasons.append('No candidate has marker coverage in the tested background.')
        outcome.update(cluster=group, groupby=ranked['params']['groupby'],
                       comparison=ranked['params']['reference'], markers=markers, reasons=group_reasons)
        result['clusters'].append(outcome)
    return result
