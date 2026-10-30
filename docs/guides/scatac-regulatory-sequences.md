# scATAC regions and regulatory DNA

`prepare_scatac_regulatory_sequences` connects ordinary region-level scATAC H5AD
and accepted replicate-aware DA to reference DNA. It is independent of DNA model
prediction, optimization and generation. Original datasets and accepted results
are never modified.

Attach a peak-count H5AD and describe the species, assembly, actual matrix
measurement and coordinate convention. For example: “For this human PBMC dataset
on hg38, use peak counts in X and the supplied B-cell group. Select 2 regions by
the fraction of cells with nonzero counts, extract their reference DNA, and give
me the FASTA and evidence table.” Use digits for numeric choices. Supplied labels
and accepted annotations are not experimental validation of cell identities.

Three input routes share one scientific operation:

- **Descriptive:** fractions of cells with a nonzero value in the selected group,
  optionally minus the fraction in a distinct comparison group. Raw peak,
  fragment or insertion counts and exact binary accessibility are supported.
  There is no inferential test, p-value, FDR or biological-replicate requirement.
  Default ranking is descending fraction (or difference, then fraction), with
  ties in original feature order. At least one group cell must be accessible.
- **Explicit:** named source features, in the requested order. This requires
  neither annotation, replication nor DA. IDs must uniquely identify features;
  distinct IDs may describe identical intervals.
- **Accepted DA:** tested features in the actual accepted group and exact
  numerator/denominator contrast. Defaults are FDR ≤ 0.05 and a strictly positive
  directional logFC; denominator selection reverses the effect direction.
  Rank by ascending FDR, descending directional effect, then source order.
  A treatment contrast is not automatically a cell-type comparison. edgeR is
  not rerun. Exact pseudobulk and feature-space lineage supplies coordinates;
  older coordinate-free results can follow their pinned original H5AD when its
  encoding and coordinate convention are explicitly declared.

The default candidate count is 10, with up to 1,000 per preparation. An optional
maximum peak length filters the candidate pool before taking the requested count.
Thresholds are never silently relaxed. Descriptive matrix access is chunked,
including dense, CSR, CSC and explicitly selected `layers/name` storage. No
complete single-cell matrix is densified. Unassigned cells remain on the original
axis and do not match a named group.

Coordinates may come from explicit `.var` chromosome/start/end columns or
`chrom:start-end` / `chrom_start_end` feature identifiers. Column defaults, when
that encoding is selected, are `chrom`, `start`, `end`. Neither names nor column
spelling establish zero-based half-open versus one-based closed coordinates;
declare the convention. Conflicting recorded source metadata is rejected.
Published BED coordinates always use zero-based half-open intervals.

The ScientificResourceLibrary supplies a unique qualified matching genome. No
contig aliases, assembly substitution, liftOver, boundary clipping, synthetic
padding or unknown-base replacement occurs. Optional `+`, `-`, `.` source strands
are respected; absent strand uses reference-plus orientation without claiming
biological strand specificity. Reference intervals containing unresolved bases
require a different scientifically justified candidate.

Exact peaks are the default. Explicit wider windows use
`midpoint = floor((start + end) / 2)` and
`window_start = midpoint - floor(window_size / 2)`. The window must contain the
whole source peak and remain within its contig. A midpoint is not a measured
summit. Source peaks and extracted windows are retained separately. A genuine
2,114-base reference window can provide DeepPBMC input without invented flanks.

Downloads contain FASTA, source-peak BED, extraction-window BED, observed-evidence
TSV and versioned method/reference metadata. Exact private publications retain
source hashes, ordered feature identity, group or contrast context, candidate
statistics, reference identity, bases and sequence checksums. The conversation
receives a bounded preview of up to 12 candidates; files retain the complete set.

Select an accepted region result to expand a particular candidate later. The
`accepted_candidates` route uses its pinned region/evidence publication and the
same reference, without reopening the original matrix or repeating selection.
Existing prediction and optimization tools can consume the exact DNA port, with
an optional 1-based `sequence_index` for a single candidate. Explicit historical
result selection, revision parentage, restart and private downloads retain their
existing owners.

[DHS64](regulatory-sequence-prediction.md) accepts up to 500 real bases and predicts
human DNase measurements, not ATAC ground truth or enhancer activity.
[DeepPBMC](regulatory-sequence-optimization.md) requires exactly 2,114 real bases
and an exact supported human PBMC target. A supplied cluster or annotation does
not automatically map to a model target. Known accepted nonhuman genomic ancestry
is rejected at these human model boundaries. Independent reference extraction can
still be valid for another species.

[DNA-Diffusion](regulatory-sequence-generation.md) remains independent synthetic
generation with its own exact conditions; generated DNA has no inferred genomic
location. Predictions and optimized scores are not experimental validation.
Existing internal academic licensing and runtime restrictions remain in force.

The [RSD-4 acceptance report](../../../agent-data/acceptance/regulatory-sequence-design/rsd-4/REPORT.md)
separates simulated checks, genuine native tools, live provider/browser journeys,
observed failures and the final bounded readiness recommendation.
