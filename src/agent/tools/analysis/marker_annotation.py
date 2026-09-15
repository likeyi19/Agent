"""Private M13 scientific component; no orchestration or final identity assignment."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from importlib.metadata import version
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

from agent.tools.data import scatac_matrix_contract as matrix_contract
from agent.tools.data._ordered_identity import ordered_identity_sha256

PROFILE = 'maestro-enhanced-marker-candidates.v1'
REVISION = '74f10babf4125b98ef005df4186b14ab02e13bfd'
SOURCES = {
    'MAESTRO/scATAC_Genescore.py': '9925cd2c6a24ae23da3bb018ff8735b537a38bd213adedc22f6948871eca62fe',
    'R/FindAllMarkersMAESTRO.R': '3696fbbabb07e939deec55a1b48848df0e77b2e88ac952de82273964dd6e5a3e',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_file(path, expected):
    if len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f'Content identity mismatch: {path}')
    return Path(path).resolve()


def identifier(value):
    if not isinstance(value, str) or not value or value.strip() != value or any(ord(c) < 32 for c in value):
        raise ValueError('Nonempty exact single-line identity required')
    return value


@dataclass(frozen=True)
class Resource:
    path: str
    sha256: str
    source: str
    revision: str
    species: str
    assembly: str
    context: str
    gene_namespace: str
    normalization: str
    semantics: str

    def check(self, species, assembly, semantics, normalization):
        for value in asdict(self).values():
            identifier(value)
        if (self.species, self.assembly, self.semantics, self.normalization) != (species, assembly, semantics, normalization):
            raise ValueError('Incompatible resource declaration')
        return checked_file(self.path, self.sha256)


@dataclass(frozen=True)
class MatrixInput:
    manifest_path: str
    manifest_sha256: str
    matrix_sha256: str
    ordered_cells_sha256: str
    ordered_features_sha256: str
    species: str
    assembly: str
    value_semantics: str


@dataclass(frozen=True)
class Runtime:
    upstream_dir: str
    rscript: str
    r_library: str


def exact_groups(cells, rows):
    """Rows are explicit (cell_id, group) pairs, already in canonical order."""
    ordered_identity_sha256(cells)
    if any(len(r) != 2 for r in rows) or [r[0] for r in rows] != list(cells):
        raise ValueError('Exact ordered cell-to-group mapping required')
    for _, group in rows:
        identifier(group)
    return list(dict.fromkeys(group for _, group in rows))


def load_matrix(binding):
    path = checked_file(binding.manifest_path, binding.manifest_sha256)
    m = matrix_contract.load_manifest_bytes(path.read_bytes())
    if binding.value_semantics != 'canonical-fragment-record-overlap-counts.v1' or m['contract_version'] != matrix_contract.CONTRACT:
        raise ValueError('Unsupported matrix-value semantics; only canonical fragment-record counts qualify')
    if (m['species'], m['assembly'], m['ordered_feature_sha256']) != (binding.species, binding.assembly, binding.ordered_features_sha256):
        raise ValueError('Matrix species, assembly or feature identity mismatch')
    p = checked_file(path.parent / m['matrix']['path'], binding.matrix_sha256)
    if binding.matrix_sha256 != m['matrix']['sha256'] or p.stat().st_size != m['matrix']['size_bytes']:
        raise ValueError('Matrix manifest byte binding mismatch')
    a = ad.read_h5ad(p)
    cells, features = a.obs_names.tolist(), a.var_names.tolist()
    if ordered_identity_sha256(cells) != binding.ordered_cells_sha256 or ordered_identity_sha256(features) != binding.ordered_features_sha256:
        raise ValueError('Exact ordered matrix axes required')
    if list(a.shape) != m['shape'] or matrix_contract.csr_identity(a.X) != {k: m[k] for k in ('logical_matrix_sha256', 'nnz', 'total_count', 'zero_row_count')}:
        raise ValueError('Canonical matrix logical identity mismatch')
    if not len(cells):
        raise ValueError('Annotation requires nonempty cells')
    rows = [(i, a.obs.iloc[i]['namespace'], a.obs.iloc[i]['barcode_identifier'], c) for i, c in enumerate(cells)]
    if a.obs['matrix_row_index'].tolist() != list(range(len(cells))):
        raise ValueError('Canonical matrix row index mismatch')
    if matrix_contract.selected_axis_identity(rows)[1] != m['ordered_selected_sha256']:
        raise ValueError('Canonical selected-cell identity mismatch')
    return a, p


def read_signatures(path):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames != ['candidate', 'gene']:
            raise ValueError('Signature TSV requires candidate,gene columns')
        signatures = {}
        for row in reader:
            if None in row or None in row.values():
                raise ValueError('Malformed signature row')
            candidate, gene = identifier(row['candidate']), identifier(row['gene']).upper()
            signatures.setdefault(candidate, []).append(gene)
    if not signatures or any(len(v) < 2 for v in signatures.values()):
        raise ValueError('Each signature requires at least two entries for log2 denominator')
    return signatures


def score_candidates(cells, groups, genes, markers, signatures):
    """Qualified list-weighted sum(logFC)/log2(entries); no biological acceptance."""
    order = exact_groups(cells, groups)
    gene_set = {identifier(g).upper() for g in genes}
    if len(gene_set) != len(genes):
        raise ValueError('Duplicate or case-colliding RP genes')
    if not signatures or any(len(gs) < 2 for gs in signatures.values()):
        raise ValueError('Invalid signature denominator')
    effects = {g: {} for g in order}
    for group, gene, effect in markers:
        gene = identifier(gene).upper()
        if group not in effects or gene not in gene_set or gene in effects[group] or not math.isfinite(effect):
            raise ValueError('Invalid or duplicate marker identity/value')
        effects[group][gene] = float(effect)
    coverage = []
    for candidate, gs in signatures.items():
        identifier(candidate)
        if any(identifier(g) != g.upper() for g in gs):
            raise ValueError('Signature genes must use declared uppercase normalization')
        coverage.append(dict(candidate=candidate, entries=len(gs), duplicate_entries=len(gs)-len(set(gs)),
                             measured_entries=sum(g in gene_set for g in gs), missing_genes=sorted(set(gs)-gene_set)))
    evidence, assignments = [], []
    for group in order:
        scores, counts = {}, {}
        for candidate, gs in signatures.items():
            matched = [g for g in gs if g in effects[group]]
            score = sum(effects[group][g] for g in matched) / math.log2(len(gs))
            if not math.isfinite(score):
                raise ValueError('Nonfinite candidate score')
            scores[candidate], counts[candidate] = score, len(matched)
            evidence.append(dict(group=group, candidate=candidate, score=score, matched_entries=len(matched),
                                 positive=[g for g in matched if effects[group][g] > 0],
                                 negative=[g for g in matched if effects[group][g] < 0],
                                 unobserved_marker_genes=sorted((set(gs) & gene_set) - effects[group].keys())))
        top = max(scores.values())
        winners = [c for c, s in scores.items() if s == top]
        status = ('insufficient_evidence' if not any(counts.values()) else 'unresolved' if top <= 0
                  else 'ambiguous' if len(winners) != 1 else 'candidate')
        assignments.append(dict(group=group, n_cells=sum(g == group for _, g in groups),
                                candidate=winners[0] if status == 'candidate' else None,
                                accepted_identity=None, status=status, top_score=top,
                                top_candidates=winners, validation_state='not_assessed',
                                reason={'candidate':'unique_positive_maximum_only', 'ambiguous':'exact_positive_top_tie',
                                        'unresolved':'nonpositive_maximum', 'insufficient_evidence':'no_matched_signature_entries'}[status]))
    lookup = {a['group']: a for a in assignments}
    projection = [dict(cell_id=c, group=g, candidate=lookup[g]['candidate'], status=lookup[g]['status'],
                       accepted_identity=None) for c, g in groups]
    return dict(groups=assignments, cells=projection, candidate_evidence=evidence, signature_coverage=coverage)


def enhanced_rp(x, features, cells, gene_path, source):
    """Invoke pinned original functions; adapt only sparse input and writer IO."""
    checked_file(source, SOURCES['MAESTRO/scATAC_Genescore.py'])
    tree = ast.parse(Path(source).read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    captured = {}
    def capture(path, matrix, genes, barcodes, **kwargs):
        captured.update(matrix=matrix, genes=list(genes), cells=list(barcodes))
    ns = dict(np=np, pd=pd, sp_sparse=sparse, write_10X_h5=capture)
    exec(compile(tree, str(source), 'exec'), ns)
    converted = []
    for feature in features:
        chrom, interval = feature.rsplit(':', 1)
        start, end = interval.split('-')
        if not chrom or int(start) < 0 or int(end) <= int(start):
            raise ValueError('Invalid canonical region')
        encoded = f'{chrom}_{start}_{end}'
        if encoded.rsplit('_', 2) != [chrom, start, end]:
            raise ValueError('Nonreversible region encoding')
        converted.append(encoded.encode())
    binary = x.copy()
    binary.data = np.ones(binary.nnz, dtype=np.float64)
    ns['calculate_RP_score'](binary.T.tocsr(), converted, cells, str(gene_path), 10000, 'unused', 'Enhanced')
    rp, genes = captured['matrix'].T.tocsr(), captured['genes']
    if captured['cells'] != cells or rp.shape != (len(cells), len(genes)) or not np.isfinite(rp.data).all() or (rp.data < 0).any():
        raise ValueError('Invalid upstream RP output')
    ordered_identity_sha256(genes)
    for gene in genes:
        identifier(gene)
    if len({g.upper() for g in genes}) != len(genes) or (np.asarray(rp.sum(axis=1)).ravel() <= 0).any():
        raise ValueError('Unusable RP gene identities or zero-evidence cells')
    rp.eliminate_zeros()
    return rp, genes


def annotate_cell_groups(*, matrix: MatrixInput, groups_path: str, groups_sha256: str,
                         grouping_provenance: str, gene_resource: Resource, signature_resource: Resource,
                         runtime: Runtime, profile: str, output_dir: str):
    """Accept owner-verified canonical input; publish new private scientific sidecars.

    This checks current bytes/logical identity, not upstream fragment reconstruction.
    Runtime/source paths are operator inputs, never planner-supplied execution.
    """
    if profile != PROFILE:
        raise ValueError('Unsupported explicit scientific profile')
    identifier(grouping_provenance)
    gene_path = gene_resource.check(matrix.species, matrix.assembly, 'maestro-refgenes-transcripts-exons.v1', 'none')
    signature_path = signature_resource.check(matrix.species, matrix.assembly, 'candidate-gene-list.v1', 'uppercase')
    if gene_resource.gene_namespace != signature_resource.gene_namespace:
        raise ValueError('Gene namespace mismatch')
    group_path = checked_file(groups_path, groups_sha256)
    groups_frame = pd.read_csv(group_path, sep='\t', dtype=str, keep_default_na=False)
    if groups_frame.columns.tolist() != ['cell_id', 'group']:
        raise ValueError('Group TSV requires cell_id,group columns')
    groups = list(groups_frame.itertuples(index=False, name=None))
    a, source_matrix = load_matrix(matrix)
    cells = a.obs_names.tolist()
    if len(exact_groups(cells, groups)) < 2:
        raise ValueError('One-versus-rest markers require at least two explicit groups')
    signatures = read_signatures(signature_path)
    upstream = Path(runtime.upstream_dir).resolve()
    for name, digest in SOURCES.items():
        checked_file(upstream / name, digest)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValueError('Output must be new; overwrite is unsupported')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.marker-annotation-', dir=destination.parent))
    try:
        rp, genes = enhanced_rp(a.X, a.var_names.tolist(), cells, gene_path, upstream / 'MAESTRO/scATAC_Genescore.py')
        sparse.save_npz(stage / 'rp.npz', rp)
        mmwrite(str(stage / 'rp-genes-by-cells.mtx'), rp.T, precision=17)
        (stage / 'genes.tsv').write_text('\n'.join(genes)+'\n')
        (stage / 'cells.tsv').write_text('\n'.join(cells)+'\n')
        groups_frame.to_csv(stage / 'groups.tsv', sep='\t', index=False, quoting=csv.QUOTE_NONE)
        script = Path(__file__).parent / 'r/maestro_markers_v1.R'
        env = dict(os.environ, R_LIBS_USER=runtime.r_library, R_ENVIRON_USER='/dev/null', R_PROFILE_USER='/dev/null')
        with (stage / 'marker-runtime.txt').open('w') as log:
            subprocess.run([runtime.rscript, '--vanilla', str(script), str(stage), str(upstream)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        f = pd.read_csv(stage / 'markers-signature-input.tsv', sep='\t', dtype={'cluster':str, 'gene':str}, keep_default_na=False, quoting=csv.QUOTE_NONE)
        result = score_candidates(cells, groups, genes, list(f[['cluster','gene','avg_logFC']].itertuples(index=False, name=None)), signatures)
        for key, rows in result.items():
            (stage / f'{key}.json').write_text(json.dumps(rows, sort_keys=True, allow_nan=False)+'\n')
        # Recheck all supplied immutable dependencies before publishing.
        for name, digest in SOURCES.items():
            checked_file(upstream / name, digest)
        for path, digest in [(source_matrix, matrix.matrix_sha256), (matrix.manifest_path, matrix.manifest_sha256),
                             (group_path, groups_sha256), (gene_path, gene_resource.sha256), (signature_path, signature_resource.sha256)]:
            checked_file(path, digest)
        (stage / 'rp-genes-by-cells.mtx').unlink()
        provenance = dict(schema='agent.group-marker-candidates.v1', profile=profile, method=dict(private_input='sparse X>0 float64', rp='MAESTRO Enhanced', decay=10000, normalization='log1p(RP/cell_sum*10000)', markers='Presto 1.1.0 one-versus-rest, nthreads=1', raw_p_lt=.01, abs_logfc_gt=.25, pct_either_gt=.1, adjusted_p_lt=1e-5, score='sum signed matched effects / log2(original signature entry count)', duplicates='preserve list weighting', missing='report; never biological negatives'), matrix=asdict(matrix),
                          groups=dict(path=str(group_path), sha256=groups_sha256, provenance=grouping_provenance),
                          gene_resource=asdict(gene_resource), signature_resource=asdict(signature_resource),
                          runtime=asdict(runtime), upstream_revision=REVISION, upstream_license='GPL-3.0-or-later',
                          upstream_sources=SOURCES, adapter_sha256=sha256(__file__), r_adapter_sha256=sha256(script),
                          rscript_sha256=sha256(runtime.rscript), python_versions={k:version(k) for k in ('numpy','scipy','pandas','anndata')},
                          sidecars={p.name:sha256(p) for p in sorted(stage.iterdir())}, accepted_identity_policy='none')
        (stage / 'result.json').write_text(json.dumps(provenance, sort_keys=True, allow_nan=False)+'\n')
        if destination.exists():
            raise ValueError('Output appeared during computation')
        stage.rename(destination)
        return provenance
    finally:
        if stage.exists():
            shutil.rmtree(stage)
