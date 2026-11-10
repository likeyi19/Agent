# Native scATAC QC and dimensionality reduction

Ordinary compatible scATAC analysis uses the existing `run_snapatac2` interface:
prepare a native tile or region matrix when needed, `pp.select_features`,
`tl.spectral`, `pp.knn`, `tl.leiden`, and `tl.umap`. Processed peak-count H5AD
starts with feature selection; it does not require raw import, cell calling,
a genome declaration, or a model resource. Native functions own numerical input
legality and adaptive behavior. Omitted parameters retain the installed native
defaults, including feature selection, Spectral's requested 30 components and
its native small-matrix adjustment, cosine similarity and eigenvalue weighting.

KNN and UMAP use `X_spectral`; Leiden consumes the native graph. The same accepted
representation and count-preserving outputs can supply later clustering, figures,
downloads and native annotation groups through existing result handoffs. Exact
source identities and ordered cells establish correspondence. EpiZoo remains a
separate model-based capability. Native annotation methods and reference
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

## PCA history and optional methods

The dedicated `pca_embed_cells` execution and TF-IDF/PCA planning path are retired.
Accepted historical PCA v1/v2 results retain their original descriptors, evidence,
reports, exact downloads and source lineage through read-only readers. A loose
historical receipt cannot create scientific authority; fitting and independent
PCA reconstruction are removed. Explicit PCA remains `run_scanpy` / `pp.pca`
or `run_episcanpy` / `tl.pca` through the existing native-function interface.
No implicit PCA preprocessing or automatic fallback from EpiZoo is introduced.

The Stage 1 report records bounded technical qualification and Stage 2 readiness;
biological validity and unrestricted provider behavior require separate evidence.
