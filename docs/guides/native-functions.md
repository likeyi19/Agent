# Native scientific functions

STE integrates reviewed functions of installed scientific packages. A function
invocation is an independent capability; it does not imply a complete research
workflow. Existing motif accessibility, peak calling and raw processing remain
separate operations. Package order and implementation language confer no priority.

## Supported interfaces

The existing `run_motifmatchr` supports `matchMotifs` with native matches output;
see [native motif invocation](native-motif-invocation.md).

`run_chromvar(input_path, output_dir, function_name, kwargs=...)` supports these
installed chromVAR 1.32.0 functions through one reviewed dispatch:

| Function | Input | Native arguments exposed | Result |
| --- | --- | --- | --- |
| `computeExpectations` | Ordinary H5AD count matrix | `norm`; optional `group_key` binds an exact `obs` column to native `group` | One numeric expectation per original feature |
| `getFragmentsPerPeak` | Same count representation | No kwargs | One supplied-measurement total per original feature |
| `getFragmentsPerSample` | Same count representation | No kwargs | One supplied-measurement total per original cell |
| `computeVariability` | H5AD with chromVAR deviations in X and separately identified `z_scores` | `bootstrap_error`, `bootstrap_samples`, `bootstrap_quantiles`, `na.rm`; optional explicit `seed` | Native variability, chi-square p-values, BH adjustment and optional bootstrap bounds |

For example: “Run chromVAR::computeExpectations on this H5AD using norm=false.
The measurement in X is insertion_counts.” Parameters can be grounded prose or
`name=value` assignments; explicit JSON is also supported. Omitted kwargs are not
added to the R call. The recorded parameters distinguish supplied values from
native defaults. Invalid scientific values reach the native package unchanged;
unknown functions, input objects, paths and executable expressions are not native
kwargs. No retry changes the requested parameters or substitutes a function.

Count functions accept dense, CSR and CSC data, X or an explicitly chosen layer.
The selected matrix requires its actual raw measurement: peak counts, fragment
overlap counts, insertion counts or binary accessibility. X metadata can supply
that declaration; a different layer needs its own declaration. They need no
coordinates, genome, motif collection, QC, clustering or positive-count filtering.
The native function names do not turn insertion counts into independent fragments.

`computeExpectations` preserves the installed implementation, including its
normalization/group behavior. In this release, `norm=TRUE` sums the per-cell
fractions before optional group averaging; do not assume the returned vector
sums to one. Group labels come from an exact selected column, never biological
inference. `norm=TRUE` uses a native dense denominator, so the adapter rejects
more than 16 million denominator elements without changing the parameter.

Variability uses the exact source motif and cell order. Native objects are
constructed locally from the stored assays and names; no uploaded R object or
serialization is needed. The accepted motif-accessibility `scores` port binds
its precise H5AD through existing schema-2 scientific authority. Compatible
uploaded scores must explicitly identify deviations and z-score semantics;
that metadata alone does not verify their scientific origin. Motif accessions
identify rows; repeated TF names remain separate labels. The function's native
statistics describe the supplied cells, not replicate-level condition tests or
TF binding. Undefined numbers remain explicit in the table. No motifs,
backgrounds or deviations are recomputed by this invocation.

## Runtime and result delivery

The existing isolated chromVAR runtime and `AGENT_CHROMVAR_RSCRIPT` override are
reused. No environment or resource installation is performed by the runner.
Native calls run in a bounded process group with cooperative cancellation and a
two-hour wall-time limit. Count exchange stays sparse and uses the existing
2 GiB stored-matrix bound; H5AD storage checks and exact double-count eligibility
remain. Bootstrap resamples have a 10,000 execution cap. Native validity constraints
remain with R. An explicit seed controls R/BiocParallel; execution uses SerialParam.
Omitting the seed does not promise repeatability.

Downloads include `result.tsv`, `cells.tsv`, `features.tsv`, `metadata.json` and
`native-receipt.json`. Numeric results use 17 significant digits. Ordered indices
and source labels remain alongside the values. The original source and accepted
publication are immutable. The download owner omits private metadata and may
replace unsafe labels with an ordered-index placeholder. Therefore delivered
bytes have their own checksum and size; they must not be confused with original
scientific source hashes. Native binary files use exact private snapshots;
JSON/TSV projections keep one snapshot and reserve only measured file growth
before writing, under the existing transfer capacity and 1 GiB delivery cap.
Reopening accepted results and preparing downloads do not rerun science.

## Engineering ownership

`ToolRegistry` is the sole executable allowlist. Each compatible package owns
its reviewed functions, scientific input construction, parameters, native call
and genuine result representation. Existing Interpreter members, typed parameter
binding, Semantic Compiler, RunStore, Session and Revision owners remain in place.
There is no arbitrary R evaluation, Python import/callable dispatch, shell command
interface, object store, environment manager or second result database.

`ScientificDeliverySpec` declares the owner's legitimate files and presentation
metadata in the existing Registry. Shared `application/native_delivery.py` reads
exact accepted publications and checks authority, managed membership and anchors.
The existing matrix download preparer owns private snapshots, capacity admission,
privacy projection and delivered-byte metadata. Native figures reuse the same
accepted figure path. Historical motif, motifmatchr and MACS3 types/handles remain.
Their superseded package-specific delivery modules are removed.

`tools/analysis/native_publication.py` extracts the existing motifmatchr completion,
file-integrity and durable-recovery mechanics for compatible native publications.
It performs no second scientific calculation. `tools/_native_process.py` shares
subprocess waiting, process-group cancellation and timeout handling across the
R calls and MACS3. Scientific algorithms and mature workflow-specific validation
remain in their original owners. `sequence_reference.py` owns the shared genomic
coordinate/declaration and genome/motif resolution helpers; count materialization
belongs to the existing H5AD adapter. motifmatchr no longer imports chromVAR.

The [STE-NFI-1A report](../../../agent-data/acceptance/scientific-tool-ecosystem/ste-nfi-1a/REPORT.md)
separates direct native parity, simulated regression, scripted accepted reuse,
genuine Firefox/provider execution and remaining conversational limitations. It
also records prompt/schema costs and the package-neutral future integration matrix.
This initial interface work does not integrate every API of any package or start
STE-NFI-1B/1C/1D.
