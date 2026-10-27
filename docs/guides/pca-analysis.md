# PCA representations for scATAC exploration

`pca_embed_cells` uses sparse-safe centered Scanpy PCA with ARPACK. Numerical
eligibility alone does not establish that an input representation is biologically
appropriate. Choose the representation explicitly and review depth effects,
feature definitions, sample structure and upstream QC before interpreting clusters.

The optional `preprocessing` parameter has two choices:

- `none` (default): use the supplied `X` unchanged, preserving the established
  behavior for already prepared measurements and historical analyses.
- `tfidf_l2`: require finite nonnegative integer counts or binary accessibility,
  with a positive total in every cell. Apply the installed scikit-learn
  `TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)`
  before centered PCA. For feature j, IDF is
  `log((1 + number_of_cells) / (1 + cells_with_nonzero_feature_j)) + 1`.
  Multiply counts by IDF, then normalize each weighted cell vector to unit L2
  length. Explicit sparse zeros do not count as observations. Zero-frequency
  features remain present. No cells/features are dropped or automatically aligned.

This is an exploratory TF-IDF/PCA representation, not Signac LSI, batch correction,
cell calling, reference qualification, or evidence of optimal biological separation.
L2 normalization removes simple per-cell multiplicative scale; it does not remove
all sequencing-depth or technical effects. Do not apply the count transform to
log-normalized or otherwise processed measurements by relabelling them as counts.
The original file, measurements and ordered axes remain unchanged; the compact
embedding publication records the transformation. An integrated export retains
original `X` and adds selected embeddings, coordinates and labels.

The method uses [scikit-learn's documented TF-IDF implementation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html).
[Signac's ATAC workflow](https://stuartlab.org/signac/reference/runtfidf) documents
different TF-IDF variants; equivalence to those variants or their uncentered SVD
is not claimed.

The Registry offers this choice through the existing conversational parameter
owner. A researcher can request “TF-IDF with L2 cell normalization before centered
PCA”; the interpreter binds that phrase to `tfidf_l2`. There is no language-based
scientific transform or silent default change. The numerical source and eigenpair
checks operate on the chosen representation. Unchanged raw-X publications retain
v1 identities; transformed publications use embedding/publication v2 and a distinct
scientific profile. Readers do not reinterpret accepted v1 evidence.

RPD-1 acceptance, including real PBMC analysis, explicit graph reuse, original-data
exports and provider limitations, is recorded in
[the managed report](../../../agent-data/acceptance/researcher-pilot-demo/rpd-1/REPORT.md).
