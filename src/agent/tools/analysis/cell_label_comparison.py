"""Fixed cell-label comparison on explicitly declared library/cell identities.

Feature matrices are not compared or loaded. Reference labels never train, rename
or otherwise change predictions. Legacy transfer-confidence evaluation remains
in annotation_evaluation with its original publication contract.
"""
from collections import Counter
from collections.abc import Mapping
import csv
from pathlib import Path

import h5py
from anndata.io import read_elem
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score,
    normalized_mutual_info_score, precision_recall_fscore_support)

from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools.data.authority_context import current
from agent.tools.data.fastq_verification_authority import file_closure
from .native_publication import NativePublication
from .sequence_prediction import arguments, read, write

TOOL = 'compare_cell_annotations'
KIND = 'cell_label_comparison'
CONTRACT = 'cell-label-comparison.v1'
POLICY = 'compare-cell-annotations-v1'
HISTORICAL_PROFILE = digest(dict(contract=CONTRACT, identity='explicit_library_and_exact_cell_id',
    labels='exact_no_mapping', unassigned='included_in_accuracy_and_recall',
    macro_f1='union_of_observed_biological_labels', partitions='jointly_labeled_only'))
PROFILE = digest(dict(contract=CONTRACT, base=HISTORICAL_PROFILE,
    labels='explicit_side_scoped_simultaneous_string_mapping.v1', unspecified='unchanged',
    originals='immutable_with_original_metrics', missing='never_mapped'))
VERIFIER = dict(id='agent.cell-label-comparison', compatibility_version='1')
LIMITATIONS = ('Reference annotations are supplied research evidence, not infallible biological truth. '
    'Library identity is explicitly declared or supplied in selected metadata columns, not independently verified. '
    'Exact cell identifiers are joined without barcode rewriting, positional matching or label mapping. '
    'Accuracy, macro-F1 and balanced accuracy use matched cells with a reference label; missing predictions count as errors. '
    'Macro-F1 averages the union of observed biological labels; balanced accuracy averages recall over supported reference classes. '
    'ARI/NMI use only jointly labeled matched cells and measure partitions, not biological label correctness. '
    'UMAP is descriptive; marker evidence used in annotation is not independent validation.')
FACT_FIELDS = ('n_prediction_cells', 'n_reference_cells', 'matched_cells', 'prediction_only_cells',
    'reference_only_cells', 'matched_fraction_prediction', 'matched_fraction_reference',
    'evaluated_cells', 'reference_label_missing_cells', 'unassigned_prediction_cells',
    'evaluation_coverage', 'assigned_accuracy', 'accuracy', 'macro_f1', 'balanced_accuracy',
    'ari', 'nmi', 'partition_cells', 'identity_basis', 'label_policy', 'per_class',
    'classes_omitted', 'composition', 'composition_omitted', 'umap_available', 'software_versions', 'limitations')
EXTENDED_FACT_FIELDS = ('vocabulary', 'mapping', 'original_metrics', 'metric_denominators')
RESULT_FIELDS = dict(status=(str,), contract_version=(str,), manifest_path=(str,), manifest_sha256=(str,),
    comparison_path=(str,), comparison_sha256=(str,), aligned_path=(str,), aligned_sha256=(str,))
for _k in FACT_FIELDS:
    RESULT_FIELDS[_k] = ((int,) if _k in ('n_prediction_cells','n_reference_cells','matched_cells',
        'prediction_only_cells','reference_only_cells','evaluated_cells','reference_label_missing_cells',
        'unassigned_prediction_cells','partition_cells','classes_omitted','composition_omitted') else
        (float, type(None)) if _k in ('assigned_accuracy','accuracy','macro_f1','balanced_accuracy','ari','nmi') else
        (float,) if _k in ('matched_fraction_prediction','matched_fraction_reference','evaluation_coverage') else
        (list,) if _k in ('per_class','composition') else (dict,) if _k=='software_versions' else
        (bool,) if _k=='umap_available' else (str,))
DELIVERY_FILES = {
    'comparison.json': ('Comparison metrics, confusion counts, coverage and method', 'application/json'),
    'aligned-labels.tsv': ('Exact library/cell join, original labels, optional mapped labels and missing coverage', 'text/tab-separated-values'),
    'per-class.tsv': ('Per-class precision, recall, F1 and supported counts', 'text/tab-separated-values'),
    'composition.tsv': ('Full prediction and reference label composition', 'text/tab-separated-values'),
    'overlay.json': ('UMAP overlay availability; coordinates and aligned labels when available', 'application/json'),
}
MAX_CELLS = 200_000
MAX_CLASSES = 256


def _text(value, what):
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 512 or any(not c.isprintable() for c in value):
        raise ValueError(what + ' must contain exact nonempty printable text (at most 512 characters)')
    return value


def _pin(path, sha=None):
    pin = file_closure([path])[0]
    if sha is not None and pin['sha256'] != sha:
        raise AuthorityError('Selected comparison source bytes changed')
    return pin


def _labels(values):
    return [None if pd.isna(v) or v == '' else _text(v, 'Cell label') for v in values]


def table_fields(stream):
    """Bounded header-name discovery, no row values or scientific qualification."""
    header=stream.readline(16385)
    if isinstance(header,bytes):
        header=header.decode("utf-8")
    if len(header.encode("utf-8"))>16384 or not header.endswith("\n"):
        raise ValueError("Label table header must be a bounded complete line")
    delimiter="\t" if "\t" in header else ","
    names=next(csv.reader([header],delimiter=delimiter))
    if not 2<=len(names)<=64 or len(set(names))!=len(names):
        raise ValueError("Label table requires 2–64 distinct column names")
    return tuple(_text(name,"Table column") for name in names),delimiter


def parameter_prerequisites(inputs):
    if inputs.get("shared_library_id") or inputs.get("prediction_sample_key") and inputs.get("reference_sample_key"):
        return ()
    if inputs.get("prediction_sample_key"):
        return ("reference_sample_key",)
    if inputs.get("reference_sample_key"):
        return ("prediction_sample_key",)
    return ("shared_library_id",)


def observations(path, label_key, cell_key=None, sample_key=None):
    """Read only obs, never X/var/embeddings, including zero-variable H5AD."""
    label_key = _text(label_key, 'Label column')
    path = Path(path)
    if h5py.is_hdf5(path):
        from .peak_h5ad import inspect
        with h5py.File(path, 'r') as handle:
            inspect(handle, observation_fields=())
            group = handle['obs']; index_key = group.attrs['_index']
            selected = tuple(dict.fromkeys(k for k in (index_key, label_key, cell_key, sample_key) if k is not None))
            if any(k not in group for k in selected):
                raise ValueError('Selected observation column is absent')
            inspect(handle, observation_fields=selected, max_logical_bytes=128 * 1024**2)
            index = read_elem(group[index_key])
            if not 1 <= len(index) <= MAX_CELLS:
                raise ValueError('Label observations exceed the 200000-cell bound')
            columns = {k:read_elem(group[k]) for k in selected if k != index_key}
            if (getattr(index, 'ndim', None) != 1 or any(
                    getattr(value, 'ndim', None) != 1 or len(value) != len(index)
                    for value in columns.values())):
                raise ValueError('Selected H5AD observations require one value per cell; scalar broadcasting is forbidden')
            obs = pd.DataFrame(columns, index=index)
    else:
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError('Label table exceeds 64 MiB')
        with path.open(newline='') as stream:
            names,delimiter = table_fields(stream)
            stream.seek(0)
            obs = pd.read_csv(stream, sep=delimiter, dtype=str, keep_default_na=False, nrows=MAX_CELLS+1)
        # Pandas can reinterpret surplus leading fields as an index rather than
        # reporting a malformed row. A label table never has an implicit index:
        # cell identifiers are selected explicitly, and source fields must not shift.
        if not isinstance(obs.index, pd.RangeIndex):
            raise ValueError('Label table row width exceeds its header; implicit index inference is forbidden')
        if cell_key is None:
            raise ValueError('A TSV/CSV requires an explicit cell identifier column')
    if not 1 <= len(obs) <= MAX_CELLS or not obs.columns.is_unique:
        raise ValueError('Label observations require 1–200000 rows and unique columns')
    for key in (label_key, cell_key, sample_key):
        if key is not None and key not in obs:
            raise ValueError('Selected observation column is absent: ' + str(key))
    ids = [_text(v, 'Cell identifier') for v in (obs.index.tolist() if cell_key is None else obs[cell_key].tolist())]
    samples = None if sample_key is None else [_text(v, 'Library identifier') for v in obs[sample_key].tolist()]
    return ids, samples, _labels(obs[label_key].tolist()), tuple(obs.index.tolist())


def _table(path, rows, fields):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def _annotation_parent(args, prediction_pin):
    if not args.get("annotation_manifest_path"):
        return None, []
    from . import scatac_annotation, annotation_contract
    mp = _pin(args["annotation_manifest_path"],args["annotation_manifest_sha256"])
    scatac_annotation.verify_annotation(mp["path"],expected_sha256=mp["sha256"])
    manifest = annotation_contract.load_manifest(mp["path"],mp["sha256"])
    if manifest["sidecars"]["annotated.h5ad"]["sha256"] != prediction_pin["sha256"]:
        raise ValueError("Selected annotation publication does not identify the exact prediction file")
    binding=manifest["matrix_binding"]
    return binding.get("source_sha256",binding.get("matrix_sha256")), [mp]


def _overlay(args, prediction_pin, ids, labels, reference, matched, annotation_parent, category_order):
    if 'umap_path' not in args:
        if 'umap_sha256' in args:
            raise ValueError('UMAP checksum requires the selected UMAP file')
        return dict(available=False), []
    pin = _pin(args['umap_path'], args.get('umap_sha256'))
    from .categorical import coordinates
    from .embedding_analysis import PROVENANCE_KEY, _read_analysis, _validate_neighbors_artifact, _validate_umap_labels
    coordinate_artifact = _read_analysis(Path(pin['path']))
    _validate_neighbors_artifact(coordinate_artifact, expected_stages=frozenset({'umap'}))
    _validate_umap_labels(coordinate_artifact)
    cells, xy = coordinates(pin['path'], pin['sha256'])
    with h5py.File(pin['path'], 'r') as handle:
        provenance = read_elem(handle['uns'][PROVENANCE_KEY])
    parent = provenance.get('embedding', {}).get('source_sha256')
    # Copied H5AD metadata alone is not accepted native annotation authority.
    # Authenticate the original-source relationship through its publication.
    parents = {prediction_pin['sha256']}
    if annotation_parent is not None:
        parents.add(annotation_parent)
    if parent not in parents or tuple(cells) != tuple(ids):
        raise ValueError('UMAP requires exact original-source ancestry and ordered prediction cells; equal counts/barcodes do not qualify')
    return dict(available=True, cell_ids=list(cells), coordinates=xy.tolist(),
        predicted=labels, reference=reference, matched=matched, category_order=category_order,
        source_sha256=pin['sha256'], correspondence='exact_original_source_and_ordered_cells'), [pin]


def _metric_data(predicted, truth):
    """The same metric definitions for exact and explicitly mapped partitions."""
    evaluated = [i for i, r in enumerate(truth) if r is not None]
    classes = sorted({truth[k] for k in evaluated} | {predicted[k] for k in evaluated if predicted[k] is not None})
    if len(classes) > MAX_CLASSES:
        raise ValueError('Comparison exceeds the 256-class presentation bound')
    codes = {c:i for i,c in enumerate(classes)}
    y = np.asarray([codes[truth[k]] for k in evaluated], dtype=int)
    p = np.asarray([codes.get(predicted[k], -1) for k in evaluated], dtype=int)
    assigned = p >= 0
    per_class = []
    confusion = np.zeros((len(classes), len(classes)+1), dtype=np.int64)
    if len(y):
        precision, recall, f1, support = precision_recall_fscore_support(y, p, labels=list(range(len(classes))), zero_division=0)
        for i, c in enumerate(classes):
            per_class.append(dict(label=c, precision=float(precision[i]), recall=float(recall[i]),
                f1=float(f1[i]), reference_cells=int(support[i]), predicted_cells=int((p==i).sum()),
                correct_cells=int(((p==i)&(y==i)).sum())))
        np.add.at(confusion, (y, np.where(assigned, p, len(classes))), 1)
        accuracy = float(accuracy_score(y, p))
        macro = float(f1_score(y, p, labels=list(range(len(classes))), average='macro', zero_division=0))
        balanced = float(np.mean(recall[support>0]))
    else:
        accuracy = macro = balanced = None
    partition = int(assigned.sum())
    identifiable = partition > 1 and len(set(y[assigned])) > 1 and len(set(p[assigned])) > 1
    metrics = dict(evaluated_cells=len(y), unassigned_prediction_cells=int((~assigned).sum()),
        accuracy=accuracy, macro_f1=macro, balanced_accuracy=balanced,
        assigned_accuracy=float(accuracy_score(y[assigned],p[assigned])) if partition else None,
        partition_cells=partition, ari=float(adjusted_rand_score(y[assigned],p[assigned])) if identifiable else None,
        nmi=float(normalized_mutual_info_score(y[assigned],p[assigned],average_method='arithmetic')) if identifiable else None)
    return metrics, per_class, dict(rows=classes, columns=classes+[None], counts=confusion.tolist())


def _mapping(values, rules, side):
    if not isinstance(rules, Mapping) or not 1 <= len(rules) <= 32:
        raise ValueError(side + ' mapping requires 1–32 explicit original-label to common-label rules')
    vocabulary = {v for v in values if v is not None}
    for source, target in rules.items():
        _text(source, 'Original mapping label'); _text(target, 'Common mapping label')
        if source.casefold() in {'unassigned', 'unknown', 'unlabeled', 'unlabelled'} or target.casefold() in {'unassigned', 'unknown', 'unlabeled', 'unlabelled'}:
            raise ValueError('Missing/unassigned categories cannot be mapped to or from a biological class')
        if source not in vocabulary:
            raise ValueError(side + ' mapping label does not exist: ' + source + '; supply an exact original label')
    # Simultaneous substitution, never recursive/chained or cell-dependent.
    mapped = [rules.get(v, v) if v is not None else None for v in values]
    labeled = sum(v is not None for v in values)
    covered = sum(v in rules for v in values if v is not None)
    return mapped, dict(rules=[dict(original_label=k, common_label=v) for k,v in rules.items()],
        rule_source='explicit_request_parameter', total_cells=len(values), labeled_cells=labeled,
        missing_cells=len(values)-labeled, rule_covered_cells=covered,
        changed_cells=sum(a != b for a,b in zip(values, mapped)),
        rule_coverage_labeled=covered/labeled if labeled else None,
        unmapped_categories=sorted(vocabulary-set(rules)), unmapped_labeled_cells=labeled-covered,
        mapped_vocabulary=sorted({v for v in mapped if v is not None}))


def _vocabulary(predicted, truth):
    p, r = {v for v in predicted if v is not None}, {v for v in truth if v is not None}
    if len(p | r) > MAX_CLASSES:
        raise ValueError('Comparison exceeds the 256-class bound across full sources')
    return dict(prediction=sorted(p), reference=sorted(r), prediction_only_names=sorted(p-r),
        reference_only_names=sorted(r-p), category_names_differ=p != r)


def _bounded_categories(record):
    return {**record, **{k:v[:50] for k,v in record.items() if isinstance(v,list)},
        'categories_omitted':{k:max(0,len(v)-50) for k,v in record.items() if isinstance(v,list)}}


def compare(args):
    sources = [_pin(args['prediction_path'], args.get('prediction_sha256')),
               _pin(args['reference_path'], args.get('reference_sha256'))]
    if ('annotation_manifest_path' in args) != ('annotation_manifest_sha256' in args):
        raise ValueError('An annotation publication requires its exact path and checksum together')
    shared = args.get('shared_library_id')
    pk, rk = args.get('prediction_sample_key'), args.get('reference_sample_key')
    if shared is not None:
        _text(shared, 'Explicit shared library identity')
        if pk is not None or rk is not None:
            raise ValueError('Select shared_library_id OR both library columns')
    elif pk is None or rk is None:
        raise ValueError('Declare the same physical sample/library with shared_library_id, or select both sample/library columns; bare barcodes do not establish correspondence')
    left_ids, left_samples, predicted, axis = observations(sources[0]['path'], args['prediction_label_key'], args.get('prediction_cell_key'), pk)
    right_ids, right_samples, truth, _ = observations(sources[1]['path'], args['reference_label_key'], args.get('reference_cell_key'), rk)
    left = list(zip(left_samples or [shared]*len(left_ids), left_ids))
    right = list(zip(right_samples or [shared]*len(right_ids), right_ids))
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError('Duplicate (library, cell identifier) keys make correspondence ambiguous')
    li, ri = {k:i for i,k in enumerate(left)}, {k:i for i,k in enumerate(right)}
    matches = [k for k in left if k in ri]
    if not matches:
        raise ValueError('No cells match exact declared library/cell identities')
    original_predicted, original_truth = predicted, truth
    vocabulary = _vocabulary(predicted, truth)
    original_metrics, original_per_class, original_confusion = _metric_data(
        [predicted[li[k]] for k in matches], [truth[ri[k]] for k in matches])
    mapping = {}
    for side, values in (('prediction', predicted), ('reference', truth)):
        rules = args.get(side+'_mapping')
        if rules is not None:
            mapped, coverage = _mapping(values, rules, side)
            mapping[side] = coverage
            if side == 'prediction': predicted = mapped
            else: truth = mapped
        else:
            labeled = sum(v is not None for v in values)
            mapping[side] = dict(rules=[], rule_source='none', total_cells=len(values),
                labeled_cells=labeled, missing_cells=len(values)-labeled, rule_covered_cells=0,
                changed_cells=0, rule_coverage_labeled=0.0 if labeled else None,
                unmapped_categories=sorted({v for v in values if v is not None}),
                unmapped_labeled_cells=labeled, mapped_vocabulary=sorted({v for v in values if v is not None}))
    harmonized = any(side+'_mapping' in args for side in ('prediction','reference'))
    mapping.update(applied=harmonized, unspecified_labels='unchanged', substitution='simultaneous_once',
        coverage_population='full_source_each_side; metric denominators are separately recorded')
    reference = [truth[ri[k]] if k in ri else None for k in left]
    matched = [k in ri for k in left]
    aligned = []
    for k in left + [k for k in right if k not in li]:
        p, r = predicted[li[k]] if k in li else None, truth[ri[k]] if k in ri else None
        aligned.append(dict(library_id=k[0], cell_id=k[1], prediction_present=k in li, reference_present=k in ri,
            predicted_label=p, reference_label=r, evaluated=k in li and k in ri and r is not None,
            agreement=None if p is None or r is None or k not in li or k not in ri else p==r))
        if harmonized:
            # Keep original download column meanings; mapped values are additional.
            row = aligned[-1]
            row['mapped_predicted_label'], row['mapped_reference_label'] = p, r
            row['mapped_agreement'] = row['agreement']
            op = original_predicted[li[k]] if k in li else None
            ot = original_truth[ri[k]] if k in ri else None
            row.update(predicted_label=op, reference_label=ot,
                agreement=None if op is None or ot is None or k not in li or k not in ri else op==ot)
    metrics, per_class, confusion = (_metric_data([predicted[li[k]] for k in matches], [truth[ri[k]] for k in matches])
        if harmonized else (original_metrics, original_per_class, original_confusion))
    composition = []
    for name, values in (('prediction', predicted), ('reference', truth)):
        for label, count in sorted(Counter(values).items(), key=lambda x:(x[0] is None,x[0] or '')):
            composition.append(dict(source=name, label=label, n_cells=count, fraction=count/len(values)))
    category_order=sorted({x for x in predicted+truth if x is not None})
    if len(category_order)>MAX_CLASSES:
        raise ValueError('Comparison exceeds the 256-class bound across full sources')
    annotation_parent, annotation_sources = _annotation_parent(args,sources[0])
    overlay, extra_sources = _overlay(args, sources[0], axis, predicted, reference, matched, annotation_parent, category_order)
    if harmonized:
        overlay['label_policy'] = 'explicit_harmonized_labels'
    sources += annotation_sources + extra_sources
    summary = dict(status='success', contract_version=CONTRACT,
        n_prediction_cells=len(left), n_reference_cells=len(right), matched_cells=len(matches),
        prediction_only_cells=len(left)-len(matches), reference_only_cells=len(right)-len(matches),
        matched_fraction_prediction=len(matches)/len(left), matched_fraction_reference=len(matches)/len(right),
        reference_label_missing_cells=len(matches)-metrics['evaluated_cells'],
        evaluation_coverage=metrics['evaluated_cells']/len(left), **metrics,
        identity_basis='declared_shared_library' if shared is not None else 'selected_library_columns',
        label_policy='explicit_harmonized_labels' if harmonized else 'exact_labels_no_mapping', per_class=per_class[:50], classes_omitted=max(0,len(per_class)-50),
        composition=composition[:50], composition_omitted=max(0,len(composition)-50), umap_available=overlay['available'],
        software_versions={'scikit_learn':sklearn_version}, limitations=(LIMITATIONS if not harmonized else
            LIMITATIONS.replace(' or label mapping', '') + ' Explicit user mappings rename/merge categories without validating biological equivalence. '
            'Metrics describe harmonized labels; ARI/NMI describe mapped partitions, not original clustering. '
            'Original exact-label metrics and original labels are preserved separately.'))
    denominators = dict(accuracy_cells=metrics['evaluated_cells'], assigned_accuracy_cells=metrics['partition_cells'],
        partition_cells=metrics['partition_cells'], macro_f1_classes=len(per_class),
        balanced_accuracy_reference_classes=sum(c['reference_cells']>0 for c in per_class),
        original_macro_f1_classes=len({original_truth[ri[k]] for k in matches if original_truth[ri[k]] is not None}
            | {original_predicted[li[k]] for k in matches if original_truth[ri[k]] is not None and original_predicted[li[k]] is not None}),
        original_balanced_accuracy_reference_classes=sum(c['reference_cells']>0 for c in original_per_class),
        reference_missing_excluded_cells=summary['reference_label_missing_cells'],
        missing_prediction_policy='included_as_errors', zero_division=0,
        partition_scope='mapped_partitions' if harmonized else 'original_label_partitions')
    summary.update(vocabulary=_bounded_categories(vocabulary),
        mapping={**mapping, **{side:_bounded_categories(mapping[side]) for side in ('prediction','reference')}},
        original_metrics=dict(label_policy='exact_labels_no_mapping', **original_metrics), metric_denominators=denominators)
    report = dict(schema=CONTRACT, summary=summary, per_class=per_class, composition=composition,
        confusion=confusion, vocabulary=vocabulary, mapping=mapping,
        methods=dict(identity_basis=summary['identity_basis'], label_policy=summary['label_policy'],
            sample_declaration=shared, prediction_sample_key=pk, reference_sample_key=rk,
            prediction_label_key=args['prediction_label_key'], reference_label_key=args['reference_label_key']),
        sources=sources)
    if harmonized:
        report['original_comparison'] = dict(metrics=original_metrics, per_class=original_per_class,
            confusion=original_confusion, label_policy='exact_labels_no_mapping')
    if file_closure(r['path'] for r in sources) != sorted({r['path']:r for r in sources}.values(),key=lambda r:r['path']):
        raise AuthorityError('Comparison source changed while reading')
    return summary, report, aligned, overlay, sources


def execute(values, execution_identity=None):
    args = arguments(values)
    root = Path(args['output_dir']); root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Comparison destination must be empty; saved publication recovery cannot rerun science')
    summary, report, rows, overlay, sources = compare(args)
    write(root/'comparison.json', report)
    write(root/'overlay.json', overlay)
    _table(root/'aligned-labels.tsv', rows, tuple(rows[0]))
    _table(root/'per-class.tsv', report['per_class'], ('label','precision','recall','f1','reference_cells','predicted_cells','correct_cells'))
    _table(root/'composition.tsv', report['composition'], ('source','label','n_cells','fraction'))
    value = dict(contract=CONTRACT, profile=PROFILE, arguments=args, execution_identity=execution_identity,
        result=summary, files=file_closure(root/name for name in DELIVERY_FILES),
        sources=sorted({r['path']:r for r in sources}.values(),key=lambda r:r['path']), native_completion='succeeded')
    path = root/'manifest.json'; sha = write(path,value)
    if current() is not None:
        current().register_execution(KIND, root, execution_identity)
    return verify_publication(path, expected_sha256=sha)


def compare_cell_annotations(prediction_path, reference_path, prediction_label_key, reference_label_key,
    output_dir, shared_library_id=None, prediction_sample_key=None, reference_sample_key=None,
    prediction_cell_key=None, reference_cell_key=None, prediction_sha256=None, reference_sha256=None,
    umap_path=None, umap_sha256=None, annotation_manifest_path=None, annotation_manifest_sha256=None,
    prediction_mapping=None, reference_mapping=None):
    return execute({k:v for k,v in locals().items() if v is not None})


publication = NativePublication(tool=TOOL, kind=KIND, contract=CONTRACT, profile=PROFILE,
    verifier=VERIFIER, files=tuple(DELIVERY_FILES), arguments=arguments,
    outputs={'comparison':'comparison.json', 'aligned':'aligned-labels.tsv'}, historical_profiles=(HISTORICAL_PROFILE,))
load, result = publication.load, publication.result
verify_publication, verify_result = publication.verify_publication, publication.verify_result
recover = publication.recover
