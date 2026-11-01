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

## STE-NFI-1C matrix package batch

Three additional package interfaces share bounded AnnData adaptation and native
completion/publication: `run_signac`, `run_episcanpy`, and `run_snapatac2`.
They take `input_path`, `output_dir`, `function_name`, optional `kwargs`,
`input_sha256`, `count_slot`, and `matrix_semantics`. Functions are fixed reviewed
public APIs, not arbitrary imported callables. Each call runs in its package's
isolated runtime; source files and shared environments are never mutated.

| Package/version | Function names | Native result represented in H5AD |
| --- | --- | --- |
| Signac 1.16.0 | `RunTFIDF`, `BinarizeCounts` | Transformed or binary `X`; sparse features × cells are passed to R, then returned to cells × features |
| Signac 1.16.0 | `FindTopFeatures` | Native count/percentile statistics in `var.Signac_count` / `var.Signac_percentile`; `var.Signac_native_rank` preserves native descending-count order while H5AD retains original feature order |
| epiScanpy 0.4.0 | `pp.binarize`, `pp.log1p` | Binary or log-transformed `X`; `copy` is forwarded faithfully on the task-local object |
| epiScanpy 0.4.0 | `pp.tfidf` | Native TF-IDF layer, default `tf-idf`, with original `X` retained |
| epiScanpy 0.4.0 | `pp.lsi` | Native `obsm.X_lsi`, consuming the specified existing TF-IDF layer |
| SnapATAC2 2.9.0 | `pp.select_features` | `var.selected`; native in-place mode also records `var.count`. The matrix is not subset |
| SnapATAC2 2.9.0 | `tl.spectral` | Native `obsm.X_spectral` and `uns.spectral_eigenvalue`; `X` stays unchanged |

The actual installed APIs own default values and numerical validity. Omitted
kwargs stay omitted. Signac's matrix `FindTopFeatures` method accepts but ignores
`min.cutoff`; it does not perform the assay method's variable-feature selection.
Its native rank is mapped by exact feature identity, never assumed to match row
order. Signac's four TF-IDF methods differ from sklearn TF-IDF. In epiScanpy 0.4.0,
`pp.tfidf` defaults to `norm='l1'` despite the inconsistent docstring. Native LSI's
missing-layer behavior is not repaired: supply a compatible existing layer or
explicitly request TF-IDF as a separate operation. SnapATAC2 spectral's default
`features='selected'` requires that actual field; explicit `features=null` uses
all features. Native return-array modes are mapped to their named slots.

These methods consume finite numeric matrix measurements, without a genome
or reference prerequisite. `pp.log1p` also accepts signed transformed values;
its installed native implementation judges the logarithm domain and base.
The other reviewed functions retain nonnegative admission. This does not claim
that fractional measurements are integer counts or that all methods are suitable
for every declared measurement. Exact unique cell/feature IDs are required. H5AD
storage bounds apply; matrices are never expanded from sparse to dense by the
adapter. `count_slot` selects X or one exact layer and preserves its declared
measurement; an independently selected layer needs its own declaration. Original
layers, obs/var, obsm/varm, obsp/varp, raw and ordinary serialized uns metadata
are retained through AnnData's own copy/read/write operations. Bounded HDF5
storage inspection excludes external links, unsafe filters and unsupported
objects. Native-written slots replace existing values; retained reductions and
annotations are not recomputed or asserted to describe the new X. Agent updates
measurement and invocation provenance explicitly. Privacy projection remains a
separate download responsibility.
Source-file identity remains pinned separately from transformed-result identity.
These preserved annotations do not acquire biological validation.

The supported kwargs are exposed by Registry metadata and the package dispatch
in `tools/analysis/native_matrix.py`. Precomputed Signac IDF vectors, SnapATAC2
whitelist/blacklist files and feature-weight arrays, serialized Seurat/ArchR/CDS/
Cistopic objects, plotting APIs and arbitrary package exports are not supported
by these interfaces. They require their own exact typed adapters. No method is
substituted. The two-hour native process limit, bounded iteration/dimension
arguments, and 16-million-element limit for quadratic non-cosine spectral calls
are execution limits, not scientific recommendations. Nonfinite native matrix
results fail publication rather than being repaired. Native errors retain private
diagnostics and never authorize automatic retries.

Reviewed ordinary kwargs retain their JSON types and values, including invalid
native values, within an 8-KiB/depth-four bound. Arrays or objects do not authorize
scientific inputs: spectral masks must name an existing var field, and external
files, feature-weight/IDF vectors and executable callbacks need separate typed
adapters. No dynamic import or code evaluation is available. `pp.log1p` exposes
native `chunked`/`chunk_size`, spectral exposes `num_threads` (cap 32, also its
installed default), and `BinarizeCounts` accepts native `verbose` (ignored by its
matrix method). Omitted options remain native defaults.

Downloads use Registry-declared `ScientificDeliverySpec` and the existing shared
H5AD privacy exporter: `result.h5ad`, `metadata.json`, `native-receipt.json`.
Accepted `native_matrix_h5ad.v1` ports permit exact cross-package reuse. A new
native result is not automatically a canonical raw count matrix or an Agent
embedding bundle. Preserve the appropriate measurement and output slot when
using it elsewhere. Reopening and downloading never rerun native computation.

Finalization adapter profiles preserve ordinary AnnData contents and are
independent of catalog membership. Each publication still pins the exact native
function, parameters, runtime and implementation. The three exact earlier
profiles remain read-only; accepted authority and delivery retain their original
profile identities. They cannot create new publications under an old profile.

### Extending compatible functions

The function/member table and descriptions are declarations; the fixed callable
map invokes the installed implementation. For an ordinary AnnData function that
changes X and returns None or AnnData, extension requires its catalog row/member
descriptions, one fixed callable entry, result-slot/measurement declarations and
parity tests. An annotation-only function also joins the unchanged-X check.
There is no new ToolSpec, source binder, lifecycle, evidence projector or download
handler. A returned array/tuple needs an explicit slot mapping, as feature masks
and spectral results already do. Catalog growth no longer changes adapter identity.

Installed-API review identifies epiScanpy `pp.normalize_total` in-place use as
compatible with this family, including its optional obs/layer outputs. Its wrapper
discards Scanpy's dictionary return in non-inplace mode, which must not be invented
by an adapter. SnapATAC2 `tl.umap` fits the existing embedding slots, with a small
return-array mapping; `pp.knn` needs an explicit graph slot/return mapping. These
are extension candidates, not registered capabilities.

Signac `RunSVD.default` accepts the matrix exchange but returns Seurat `DimReduc`:
embeddings, loadings, standard deviations and miscellaneous components need an
accurate thin result adapter. `AverageCounts` requires Seurat metadata/identities;
`RegionStats` needs a qualified genome; `corSparse` produces a potentially dense
square/cross correlation matrix with different axes. They cannot truthfully share
the existing X-transform contract unchanged. Fragment imports/exports, plot saves
and file-path mutations need explicit managed file authorization. Filtering or
aggregation that changes axes needs its own lineage contract. No function quota
or package priority follows from these distinctions.

### Operator installation

Install once outside Sessions. Python 3.11 venvs and the exact dependency locks
are separate because epiScanpy requires an older compatible AnnData/numpy stack.
From the repository root, using the existing operator Python/Conda executables:

```bash
python3.11 -m venv ../agent-data/resources/runtimes/episcanpy-0.4.0/env
../agent-data/resources/runtimes/episcanpy-0.4.0/env/bin/pip install -r requirements/native/episcanpy-0.4.0.txt
python3.11 -m venv ../agent-data/resources/runtimes/native-matrix-1/env
../agent-data/resources/runtimes/native-matrix-1/env/bin/pip install -r requirements/native/snapatac2-2.9.0.txt
conda create -p ../agent-data/resources/runtimes/signac-1/env --file requirements/native/signac-1.16.0-linux-64.txt
```

The Signac explicit lock is Linux x86-64 specific. Python wheels and package
compilation retain their platform requirements; this is CPU qualification, with
no RTX-specific implementation. Defaults resolve beside the configured resource
library. Operator overrides are `AGENT_SIGNAC_RSCRIPT`, `AGENT_EPISCANPY_PYTHON`,
and `AGENT_SNAPATAC2_PYTHON`. The Python executable must identify a venv with its
`pyvenv.cfg`; deployment is not a researcher Session responsibility. Signac and
SnapATAC2 retain MIT notices, epiScanpy retains BSD-3-Clause, and dependencies
retain their own licenses; these locks do not redistribute package code.

The [STE-NFI-1C report](../../../agent-data/acceptance/scientific-tool-ecosystem/ste-nfi-1c/REPORT.md)
separates direct native parity, genuine Firefox/provider acceptance, simulated
binding checks, actual downloads and unresolved language/schema limits. This is
partial ecosystem coverage, not complete integration of any package API.
