# Native scATAC QC and dimensionality reduction

For a SnapATAC2 workflow, use the existing `run_snapatac2` interface:
prepare a native tile or region matrix when needed, `pp.select_features`,
`tl.spectral`, `pp.knn`, and `tl.leiden`. `tl.umap` is a separate downstream
operation when requested; clustering does not require UMAP. Processed peak-count H5AD
starts with feature selection; it does not require raw import, cell calling,
a genome declaration, or a model resource. Native functions own numerical input
legality and adaptive behavior. Except for the bounded clustering selection below,
omitted parameters retain the installed native defaults, including feature selection, Spectral's requested 30 components and
its native small-matrix adjustment, cosine similarity and eigenvalue weighting.

In this spectral workflow, KNN and UMAP use `X_spectral`; Leiden consumes the native graph. The same accepted
representation and count-preserving outputs can supply later clustering, figures,
downloads and native annotation groups through existing result handoffs. Exact
source identities and ordered cells establish correspondence. EpiZoo retains its separate model inference and exact resource requirements.
Its published compact H5AD supplies `obsm.X_epizoo` to compatible native
representation consumers. For SnapATAC2, bind `use_rep=X_epizoo` on KNN/UMAP;
for Scanpy/epiScanpy, bind it on neighbors and reuse that graph downstream. This H5AD has no count matrix. The native KD-tree
requires float64: a float32 representation is widened losslessly during that
call and restored unchanged before publication; the adaptation is recorded. Native annotation methods and reference
requirements are unchanged.

## Raw barcode QC

For an ordinary **uncalled, single-sample** fragment population, planning selects
`qc_thresholds=otsu` on `pp.import_fragments` and `pp.filter_cells`, with native
`metrics.tsse` between them. Import then uses `min_num_fragments=0`, retaining all
observed barcodes without a depth prefilter, including those below the former
native default of 200. Native chromosome handling and explicit whitelist
restrictions still apply; no separate source-barcode census is claimed.
SnapATAC2 still owns import, sorting, QC measurements, filtering and counting.
Explicit import retention always prevails and its truncation remains visible.

Agent passes the following sample-derived scalars directly to native
`pp.filter_cells`:

- `min_counts`: apply `skimage.filters.threshold_otsu` to log10 of finite native
  `obs.n_fragment >= 100`, then exponentiate the threshold.
- `min_tsse`: apply the same established implementation to finite native
  `obs.tsse >= 1`, on its original linear scale.

Each fit uses its own eligible population and 5,000 histogram bins. These fitting
floors and transformations follow [pycisTopic's published QC implementation](https://github.com/aertslab/pycisTopic/blob/main/src/pycisTopic/qc.py)
and the sample-level Otsu approach in the [PUMATAC protocol benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC11180611/).
The established [scikit-image Otsu implementation](https://scikit-image.org/docs/0.25.x/api/skimage.filters.html#skimage.filters.threshold_otsu)
owns threshold calculation; Agent adds no optimizer or new calling algorithm.
An empty fitting population requires explicit researcher thresholds. Constant
populations retain scikit-image's constant threshold behavior. Nonfinite
observations are excluded from fitting and counted in the recorded method details.

SnapATAC2 `n_fragment` excludes mitochondrial fragments and is **not** pycisTopic's
unique fragments in peaks. Native `tsse` retains SnapATAC2's calculation. Exact
measurement equivalence to pycisTopic/PUMATAC is not claimed. Native filtering
uses inclusive minima (`>=`); pycisTopic's final filtering uses strict `>`.
No FRiP threshold or second caller is silently added. Effective values, fitting
population sizes, import retention, selected populations, package versions and
source/implementation identities accompany each publication.

Older accepted imports retain their original populations and thresholds. An Otsu
fit after a depth prefilter cannot be described as an untruncated sample fit;
use the original fragments when a fresh uncalled population is needed. Barcode
correction or a whitelist alone does not establish called-cell identities.

Explicit minima, including `null` to disable one minimum, take precedence
independently; explicit upper bounds remain native kwargs. Already called cells
and processed H5AD identities are retained unless the researcher requests
filtering. Do not derive one pooled threshold for multiple samples merely because
they share a file. Sample separation is a researcher/input prerequisite; Agent
adds no sample inference or optimizer.

`pp.call_cells` exposes SnapATAC2's native OrdMag caller for explicit use, with
an explicit `use_rep` observation field. It is not stacked with Otsu by default.
An explicit native-function invocation that omits `qc_thresholds` retains native
import/filter defaults. This distinction preserves the native interface while
making the ordinary uncalled workflow's method visible.

## Native method selection and scientific prerequisites

AI Research uses LLM planning with deterministic binding of explicit values.
Scanpy, epiScanpy and SnapATAC2 native functions remain selectable by their actual
capabilities and researcher intent, without package priority. Scanpy `pp.pca` and
epiScanpy `tl.pca` are available alongside other supported reductions. The dedicated
Agent PCA, `build_cell_neighbors`, `cluster_cells` and `compute_cell_umap`
implementations, obsolete publications and historical parameter readers are removed.
Disposable development records impose no backward compatibility requirement.

Scanpy and epiScanpy `pp.neighbors` produce compatible Scanpy neighbors consumed
by their Leiden and UMAP calls. Scanpy PAGA additionally needs existing groups;
plan the selected native neighbors and clustering prerequisites when absent.
DPT needs neighbors, diffusion maps (`run_scanpy` / `tl.diffmap` when absent), and
supplied root-cell information in `uns.iroot`. The planner must not invent a root.
SnapATAC2 Leiden consumes the graph from `pp.knn`; its UMAP consumes a selected
representation independently. Package graphs are not interchangeable.

EpiZoo's compact H5AD keeps `X=None`, ordered cells and `obsm.X_epizoo`.
Representation consumers select this exact field through their existing native
parameters: the shared structured `use_rep` binding applies only where the selected
native function exposes `use_rep`, including Scanpy/epiScanpy neighbors and SnapATAC2 KNN/UMAP. Native kwargs take precedence.
Scanpy UMAP uses the selected neighbors graph and has no `use_rep` argument.
No count matrix is reconstructed and no implicit reduction is added to embeddings.

Result parameter edits bind the exact Revision, function and step. Changing Leiden
resolution reruns that step and affected descendants, reusing unaffected accepted
upstream outputs. Explicit researcher values prevail; native packages own omitted
defaults and numerical legality. Publications record the actual function, parameters,
source identity and completion; current native readers serve evidence and downloads.

## Bounded ASW clustering parameter selection

After the LLM selects a native method, AI Research automatically evaluates an
omitted eligible parameter inside the existing isolated native execution:

| Native functions | Parameter | Candidates |
| --- | --- | --- |
| SnapATAC2/Scanpy/epiScanpy `tl.leiden`, epiScanpy `tl.louvain` | `resolution` | 0.1, 0.25, 0.5, 1, 1.5, 2 |
| epiScanpy `tl.kmeans`, `tl.hc` | `num_clusters` | 2, 3, 4, 6, 8, 10, restricted to values smaller than the cell count |

All six use one scan and installed `sklearn.metrics.silhouette_score`. Each
candidate changes only that parameter, calls the selected native function on its
full input population, and reuses the compatible accepted graph. The highest
valid ASW wins; exact ties select the first (smallest) candidate. At most 2,000
cells are sampled once with NumPy seed 0, with the same sample for every candidate.
Native clustering seeds/defaults remain unchanged. Winning native labels are
retained without another clustering calculation.

Explicit researcher kwargs and result edits bypass scanning. An explicit
`parameter_selection=native_defaults` also bypasses it; epiScanpy kmeans/hc have
no cluster-count default and request that value through the existing parameter
question. Other native parameters retain their actual defaults. Native Functions
remains unchanged, zero-provider and unoptimized.

Graph construction records the actual representation, dimensions, metric,
ordered-cell digest and graph/representation digests in existing native receipts.
Only the exact verified upstream publication carries these associations forward.
Scanpy/epiScanpy connectivities and SnapATAC2 distances are never interchanged.
K-means and hierarchical clustering use all of `X_pca`, as required by the
installed epiScanpy APIs. UMAP is not a substitute for the clustering input.
No PCA, Spectral, EpiZoo, KNN or UMAP calculation is repeated by the scan.

One cluster, invalid label counts, unassigned labels in the fixed sample, or an
unusable representation yield undefined scores, never invented numbers. If no
candidate is valid, the native default is retained and selection is reported as
unavailable. For epiScanpy kmeans/hc, unavailable ASW with no count/default leaves
the attempted Run failed and retains its candidate diagnostics. The Interaction
then asks for function-scoped `kwargs.num_clusters` through the existing parameter
question. Its explicit answer runs the same prepared task with that count, without
ASW or provider calls, preserving exact accepted upstream references. No clustering
result or Revision is accepted from the failed attempt. Genuine native errors
remain failures. Unqualified external graph provenance,
restricted subclustering and non-vtraag Louvain flavors currently use the default
fallback; explicit native calls remain available.

The selected value, origin, candidate scores, sample details and native identities
are recorded in ordinary result parameters, receipts and reports. Adjust parameters
shows the actual selected value. Editing another setting preserves it, including
on affected downstream clustering steps; unchanged upstream results remain exact
accepted references. Native Leiden, Louvain, K-means, hierarchical and DBSCAN
written grouping slots are recognized by the existing display, figures, annotation,
evaluation and download owners. Inherited observation columns confer no authority.

ASW measures separation in the selected representation. It neither identifies
biologically correct cell types nor proves a universal optimum. See the
[scikit-learn definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
and the [selection acceptance report](../../../agent-data/acceptance/scientific-clustering-parameter-selection-2026-11-11/REPORT.md).

The [consolidation report](../../../agent-data/acceptance/scientific-execution-consolidation-2026-11-11/REPORT.md)
records the audit, bounded native evidence and remaining limitations.
