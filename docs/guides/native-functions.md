# Native scientific functions

The current STE-NFI-1H inventory is **45 Registry tools, 74 package-qualified
native functions, 183 scientific-parameter offers and 10 scientific packages**. The
[native research-object guide](native-research-objects.md) covers genuine ArchR
Arrow/project snapshots and monaLisa sequence-bin enrichment. pycisTopic remains
license-blocked and regulatory topic modeling remains uncovered. Earlier milestone
sections below retain their historical inventories and acceptance scope.
Centralized STE-NFI integration is **CLOSED**; 1H is **COMPLETE within bounded
scope / CONDITIONAL GO for controlled researcher use**. These are reviewed
portions of native APIs, not complete upstream coverage or universal biological
validation. Future additions remain open to legitimate researcher demand;
no next milestone is initiated by this closeout.

STE integrates reviewed functions of installed scientific packages. A function
invocation is an independent capability; it does not imply a complete research
workflow. Existing motif accessibility, peak calling and raw processing remain
separate operations. Package order and implementation language confer no priority.

## Supported interfaces

The existing `run_motifmatchr` supports `matchMotifs` with native matches, scores or positions output;
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
Measurements are optional supplied provenance, never native admission. Signed,
fractional and processed values reach the native function unchanged. A selected
layer does not inherit X measurement metadata. They need no
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
its precise H5AD through existing schema-2 scientific authority. Compatible uploaded scores bind X to the native deviations assay and
`layers/z_scores` to the native z assay by this explicit invocation. No semantics
certificate is required; this assignment does not verify their upstream origin. Motif accessions
identify rows; repeated TF names remain separate labels. The function's native
statistics describe the supplied cells, not replicate-level condition tests or
TF binding. Undefined numbers remain explicit in the table. No motifs,
backgrounds or deviations are recomputed by this invocation.

## Runtime and result delivery

The existing isolated chromVAR runtime and `AGENT_CHROMVAR_RSCRIPT` override are
reused. No environment or resource installation is performed by the runner.
Native calls run in a bounded process group with cooperative cancellation and a
two-hour wall-time limit. Matrix exchange stays sparse and uses the existing
2 GiB stored-matrix bound; H5AD storage safety and exact integer-to-double
conversion remain. Bootstrap resamples have a 10,000 execution cap. Native validity constraints
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

These methods pass numeric measurements to their installed implementation,
without a genome, reference or measurement declaration prerequisite. Signed,
fractional and nonfinite numerical suitability belongs to the native function. This does not claim
that fractional measurements are integer counts or that all methods are suitable
for every declared measurement. Exact ordered cell/feature labels and source identities are retained; native
functions decide whether duplicate or empty axes are suitable. H5AD
storage bounds apply; matrices are never expanded from sparse to dense by the
adapter. `count_slot` selects X or one exact layer. Optional declarations retain their
actual slot; missing measurement metadata remains unspecified. Original
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
whitelist/blacklist files and feature-weight arrays, arbitrary serialized Seurat/ArchR/CDS/
Cistopic objects, plotting APIs and arbitrary package exports are not supported
by these interfaces. They require their own exact typed adapters. No method is
substituted. The two-hour native process limit, bounded iteration/dimension
arguments, and 16-million-element limit for quadratic non-cosine spectral calls
are execution limits, not scientific recommendations. Native nonfinite results are retained honestly, without repair or a claim of
numerical usability. Native errors retain private
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

STE-NFI-1D implements `pp.normalize_total`, `tl.umap` and `pp.knn` with the
result mappings below. The installed epiScanpy wrapper discards Scanpy's
dictionary return in non-inplace mode; the adapter preserves that behavior.

Signac `RunSVD.default` accepts the matrix exchange but returns Seurat `DimReduc`:
STE-NFI-1D maps its embeddings, loadings, standard deviations and miscellaneous
components through the thin adapter below. `AverageCounts` requires Seurat metadata/identities;
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


## STE-NFI-1D function families

Eleven additional functions extend three matrix interfaces and introduce two
package interfaces, `run_macs3` and `run_cicero`. That milestone had 41 Registry tools;
native functions are choices within their package, not separate workflows.
The [acceptance report](../../../agent-data/acceptance/scientific-tool-ecosystem/ste-nfi-1d/REPORT.md)
owns the package-neutral coverage matrix and distinguishes native parity, simulated
regressions, real browser execution, continuation and explanation quality.

| Interface | Added functions | Actual inputs and result |
| --- | --- | --- |
| `run_episcanpy` | `pp.normalize_total` | X and optionally named layers; normalized X/layers and optional obs normalization factors. `inplace=false` leaves the object unchanged and returns native None. |
| `run_signac` | `RunSVD` | Selected numeric matrix passed as features × cells; DimReduc embeddings become `obsm.X_lsi`, loadings `varm.LSI`, stdev/key/assay/misc `uns.Signac_svd`. This does not apply TF-IDF. |
| `run_snapatac2` | `tl.umap`, `pp.knn`, `tl.leiden` | UMAP and kNN consume an existing `obsm` embedding (native default `X_spectral`); kNN publishes `obsp.distances`; Leiden consumes that graph. Returned-array modes map to the same named slots. Small native Leiden groups may be `-1`. |
| `run_macs3` | `bdgopt`, `bdgcmp`, `bdgpeakcall`, `bdgbroadcall` | Ordinary bedGraph tracks; native bedGraph, narrowPeak or gappedPeak bytes, respectively. No fragment pipeline or H5AD prerequisite. |
| `run_cicero` | `generate_ccans`, `find_overlapping_ccans` | Peak1/Peak2/coaccess connection TSV → Peak/CCAN membership TSV; membership TSV → CCAN1/CCAN2 overlap TSV. No estimation or invented CDS. |

STE-NFI-1E removes duplicate numerical input screening and measurement
prerequisites from these direct calls. Exact ordered axes and unchanged retained
X are still checked, treating unchanged NaNs as unchanged. AnnData files with
no X can reach functions consuming an existing embedding or graph. Native
missing-object and numerical failures are preserved.

`run_macs3(input_path, output_dir, function_name, kwargs=..., signal_semantics=...)`
uses the existing MACS3 3.1.0 executable. `bdgcmp` additionally binds `control_path`,
with optional `control_semantics` and `assembly` declarations. Optional exact
source checksums are supported. bedGraph uses zero-based half-open intervals;
MACS3's native interval-end representation requires continuous, sorted,
nonoverlapping intervals within each chromosome to avoid silently filling gaps.
A nonzero initial start retains native zero-baseline behavior. Finite scores must fit native float32 and coordinates int32. Explicit native
NaN/Inf scores are forwarded and retained; no usability is inferred. No reference qualification is inferred.
Omitting the `bdgopt` method invokes native `p2q`, which interprets scores as
negative log10 p-values. Poisson/likelihood methods interpret treatment pileup
and expected control counts. These scientific meanings are guidance, not an
Agent metadata approval stage. No score or assembly declaration is required. One comparison method
is supported per invocation; several differently named outputs require a further
result adapter. The argv list has no shell interpolation or user output paths.

`run_cicero(input_path, output_dir, function_name, kwargs=..., seed=...)` invokes
Cicero 1.28.3 in its isolated R environment. Numeric connection weights reach the native implementation;
peak identifiers are retained exactly. Unconvertible text fails faithful numeric
conversion rather than silently becoming NA. `generate_ccans` exposes
`coaccess_cutoff_override`, `tolerance_digits`; optional seed controls native R
RNG, while omission preserves it. `find_overlapping_ccans` exposes `min_overlap`;
its native chr_start_end parsing uses one-based closed ranges and compares each
CCAN's bounding span, rather than requiring individual member peaks to overlap. This does not
convert arbitrary genomic conventions or establish assembly correspondence.
These two calls use ordinary data frames, never uploaded serialized R objects.

For MACS3, kwargs are native CLI option names: `method`, `extra-param`,
`scaling-factor`, `pseudocount`, `cutoff`, `min-length`, `max-gap`,
`cutoff-peak`, `cutoff-link`, `lvl1-max-gap`, `lvl2-max-gap`, `no-trackline`,
`verbose`, each only on its actual subcommand. Matrix kwargs and installed
signatures remain in the Registry/native dispatch and pinned receipts. Explicit
values retain JSON types; omitted options are not synthesized. Invalid native
parameters are not silently corrected. Execution limits remain separate from
scientific defaults: table inputs/outputs 256 MiB, five million bedGraph records,
ordinary scalar vectors at most 16 members, shared two-hour process timeout.

Table downloads reuse `ScientificDeliverySpec`: `result.tsv` contains the native
format named in metadata, plus `metadata.json` and `native-receipt.json`. The
suffix does not imply every result has a header. External tools that choose a
parser by extension may require a local rename to `.bedGraph`, `.narrowPeak` or
`.gappedPeak`, according to `result_format`; renaming does not convert data.
The shared delivery declaration retains fixed publication filenames, avoiding
changes to accepted file identities. Privacy projection handles the
first row of headerless tables too. Original and delivered hashes remain distinct.
Accepted `native_genomic_table.v1` ports pin exact accepted bytes; a consumer must
still accept their actual format and declared measurement. Table compatibility
never follows from a filename alone. The existing native matrix port similarly
permits transformed H5AD reuse without treating it as original counts.

New packages need no historical profile. Existing exact profiles remain read-only;
unknown/mismatched profiles remain invalid. Function catalog growth does not
invalidate acceptance, while actual adaptation changes still need compatibility
review. No new execution, retry, source-binding or delivery owner is introduced.

### Additional runtime installation

The new Cicero environment is isolated; existing scientific runtimes remain intact:

```bash
conda create -p ../agent-data/resources/runtimes/cicero-1/env --file requirements/native/cicero-1.28.3-linux-64.txt
```

The explicit Linux x86-64 lock pins R 4.5.3, Cicero 1.28.3, monocle 2.38.0,
igraph 2.3.4 and GenomicRanges 1.62.1. The default location is beside the configured
resource library; `AGENT_CICERO_RSCRIPT` is the existing-style operator override.
MACS3 uses `AGENT_MACS3_EXECUTABLE` or its existing pinned runtime. These are
operator installations, never per-Session package installation. Cicero is MIT;
its dependency licenses remain separately applicable. ArchR and pycisTopic are
not installed/integrated by this unit: valid project/Arrow and topic-model object
adapters are absent. pycisTopic's upstream academic non-commercial terms must be
reviewed for the intended deployment; no unrestricted distribution is claimed.

### Catalog maintenance budget

STE-NFI-1C's four detailed planning prompt+schema fixtures were 75,904 / 77,657 /
78,344 / 83,894 bytes. STE-NFI-1D measures 80,058 / 81,853 / 82,503 / 88,085 bytes
for inspection, downstream, label transfer and paired DA. The maintenance test
budget is explicitly revised from 80 to 96 KiB, leaving 10,219 bytes above the
largest fixture. This reflects reviewed function/parameter and two package-port
additions; it is not a provider API limit. No capability, source, port or choice
was removed. Existing lossless schema/domain tests remain. Actual official
OpenAI table journeys completed with the expanded catalog; this does not qualify
every request size or provider. The report retains individual package ablations,
Interpretation measurements, live diagnostics and historical failures.
Registry metadata now allows a bounded union of 128 offered kwargs names for a
package; each submitted structured argument still has its existing 32-member
bound. These are distinct bounds, not permission for arbitrary scientific objects.


## STE-NFI-1E: native validation and scientific object interoperability

There are **43 Registry tools**. Ten additional native functions are implemented;
this is selected API coverage, not full package support.

| Interface / installed version | New functions | Actual representation |
| --- | --- | --- |
| `run_episcanpy` / 0.4.0 | `tl.pca`, `pp.neighbors`, `tl.dpt`, `tl.tsne`, `tl.umap` | Native AnnData PCA, graph, pseudotime and embedding slots; original ordered axes remain |
| `run_scanpy` / 1.9.8 | `tl.diffmap`, `pp.scale` | Existing graph → diffusion coordinates/eigenvalues; native scaling of selected X, layer or obsm |
| `run_cicero_cds` / 1.28.3 | `make_cicero_cds`, `estimate_distance_parameter`, `run_cicero` | Genuine CellDataSet construction, native overlapping aggregates, parameter estimates and co-accessibility connections |

The installed epiScanpy `tl.diffmap` wrapper passes its `copy` positional argument
as Scanpy's `neighbors_key` and fails on an ordinary valid graph. It is not
advertised as an executable function. `run_scanpy` explicitly invokes Scanpy's
own public API; it never substitutes for an epiScanpy request. Scanpy uses the
existing locked epiScanpy runtime (BSD-3-Clause), with optional
`AGENT_SCANPY_PYTHON`; no environment was changed. Runtime executable overrides
work without requiring a resource-library catalog for independent matrix calls.

Cicero aggregation takes an ordinary H5AD, `reduced_coordinates_key` naming an
actual `obsm` matrix, optional `count_slot`, explicit native kwargs and optional
R seed. Exact ordered cell/region identities accompany the sparse matrix and
coordinates. Requested summary fields come from actual `obs`. Native Cicero
owns k, normalization, numerical validity, region-name parsing and aggregation.
No embedding, genome length or annotation is invented. Its native dense
aggregation and membership allocations are limited to 16 million elements.

The resulting `cds.zip` contains only `exprs.mtx` and `object.json`, representing
actual exprs, pData, fData and dense/sparse storage for the supported Cicero calls.
A genuine native CellDataSet is reconstructed from these data; no RDS/pickle,
code, arbitrary archive paths or general fitted-object deserialization is used.
This is a focused exchange, not full Monocle serialization. Native observation
and region axes, size factors, metadata column types and native names are retained.
`return_agg_info=true` additionally publishes the native aggregation table;
its native `agg_` membership labels and `agg` pData names are not silently renamed.
Omission retains native false, with an empty header-only aggregation artifact.

`run_cicero` and `estimate_distance_parameter` take that ordinary exchange or an
exact accepted `cds` port. `run_cicero` requires an authorized two-column
chromosome-length file through `genomic_coords_path`. The file is optional for
`estimate_distance_parameter`: omission preserves installed Cicero's
`cicero::human.hg19.genome` default, whose exact 93-row table and digest are recorded
in the native receipt and parameter evidence. This is a human hg19 package default,
not an inferred or qualified assembly for the supplied data. An explicit file is
forwarded unchanged instead. Accepted objects are reused without reaggregation.
Results include the retained CDS,
connection/parameter TSV, optional aggregation TSV, metadata and receipt through
the existing shared publication and private download owners. An empty connection
table from `make_cicero_cds` means that it did not estimate connections.

The data-only exchange uses `format="cicero-cds-exchange.v1"` and
`storage="dense"` or `"sparse"` in `object.json`. `observations.ids` and
`regions.ids` contain ordered strings; their `columns` maps contain typed
`values` arrays (`numeric`, `integer`, `logical`, `character`, or `factor`, with
factor `levels`/`ordered`). Numeric strings retain R's nonfinite values and
17-digit precision. `exprs.mtx` is coordinate real/integer MatrixMarket with
regions as rows and observations as columns; its dimensions must match those
axes. Dense native storage is reconstructed only within the allocation bound.
Source identities belong to the pinned publication, not invented object fields.
The existing upload-role owner exposes `run_cicero_cds.dataset`,
`run_cicero_cds.genomic_coords` and `run_scanpy.dataset`; no separate intake is
needed for these ordinary files.

Native scientific arguments and defaults remain authoritative. Agent retains
bounded JSON, reviewed callable/argument names, protected file/object ports,
managed copies, process cancellation/timeout, exact source identity and
publication integrity. HDF5 indirect links/unsafe filters remain forbidden.
Integer-to-R-double conversion checks actual round-trip fidelity: exactly
representable large integers are admitted. Sparse centered Scanpy scaling would
densify the whole source and is therefore an explicit conversion/resource limit;
choose native `zero_center=false` yourself or supply a dense representation.
The adapter never changes this option automatically.

MACS3 interleaved chromosomes and leading-zero integer spellings now pass when
within-chromosome intervals are continuous. The gap check is retained: an actual
installed native probe changes `[0,10),[20,30)` into `[0,10),[10,30)`, filling the
gap. Retaining this narrow check prevents silent coordinate reinterpretation.

Historical accepted profile identities and displayed text remain unchanged.
The current matrix profile accepts its two old read-only profiles, and chromVAR
retains its old read-only profile plus a bounded historical declaration reader.
New matrix/chromVAR prerequisite callbacks are removed, not bypassed. The detailed
final planning fixtures measure 83,270 / 85,108 / 85,729 / 91,359 bytes,
within the unchanged 96 KiB maintenance budget. A separate Interpretation fixture
measures 105,894 prompt bytes plus 97,085 compact schema bytes; the 96 KiB detailed
planning budget is not an Interpretation or provider limit.

The [STE-NFI-1E report](../../../agent-data/acceptance/scientific-tool-ecosystem/ste-nfi-1e/REPORT.md)
owns the cleanup audit, installed-package comparisons, exact Web scope, historical
checks, failures and limitations. ArchR and pycisTopic are not claimed integrated;
their actual project/runtime and licensing limits are recorded there.

STE-NFI-1E binarization records `native_binarized_supplied_measurement`, not
canonical raw binary accessibility: a transformed input does not acquire raw
measurement provenance simply because the native output contains zeros and ones.
Python matrix calls also admit duplicate labels and empty unused feature axes,
preserving ordered source positions. R TSV exchange retains delimiter-safety
checks; CellDataSet construction retains the native data-frame identity rules.
Cicero overlap calls honor the native first-two-column convention and permit
extra ordinary columns; native required-column and numerical checks are not
duplicated by Agent.

## STE-NFI-1F: native API coverage at that milestone

At 1F the catalog had **66 distinct package-qualified native functions**, an increase
of **31**, through the same **43 Registry tools**. No additional motifmatchr output
mode is exposed. Earlier sections describe their original qualification scope;
this section records the 1F additions. The 1G section below supersedes its deferred-input limits.

| Installed package | Added functions | Representation |
| --- | --- | --- |
| Scanpy 1.9.8 | `pp.calculate_qc_metrics`, `pp.normalize_total`, `pp.log1p`, `pp.highly_variable_genes`, `pp.pca`, `pp.neighbors`, `tl.leiden`, `tl.umap`, `tl.tsne`, `tl.paga`, `tl.rank_genes_groups`, `tl.score_genes`, `tl.dpt` | Native AnnData annotations, transformed measurements, reductions, graphs and statistics; independent returns retained separately |
| epiScanpy 0.4.0 | `tl.leiden`, `tl.louvain`, `tl.kmeans`, `tl.hc`, `tl.rank_features` | Native graph/PCA clustering and structured feature rankings |
| SnapATAC2 2.9.0 | `tl.dbscan`, `pp.mnc_correct`, `tl.marker_regions`, `tl.diff_test` | Native labels/corrected embeddings; independent region lists and statistical table |
| chromVAR 1.32.0 | `getTotalFragments`, `getSampleCorrelation`, `getSampleDistance`, `differentialDeviations`, `differentialVariability` | Scalar, ordered cell-by-cell table, or ordered feature statistics |
| Signac 1.16.0 | `corSparse` | Native Y omitted: cell-by-cell correlation/covariance in `obsp.Signac_corSparse` |
| MACS3 3.1.0 | `cmbreps`, `pileup`, `filterdup` | Two authorized bedGraph inputs, native read pileup, or native BED6/MACS BEDPE output |

Use the existing package interface with the exact function and native kwargs, for
example `run_scanpy`, `function_name="pp.pca"`,
`kwargs={"n_comps":8,"random_state":7}`. For graph construction explicitly select
`use_rep="X_pca"` when that is the intended input. Agent inserts no normalization,
filtering, reduction or graph. Native internal defaults and representation
selection still apply. Only reviewed kwargs are supported, not arbitrary callbacks,
input objects or file destinations. Structured JSON remains bounded to 128 entries
per vector and the existing request-size limits.

`copy` retains returned AnnData. Non-inplace returns are stored in
`uns.agent_native_return`, with their native type and actual data; they do not
replace X or pretend to mutate scientific slots. QC returns retain their cell and
feature tables. Scanpy normalization returns retain X/layer arrays and normalization
factors separately. Variable-feature return tables preserve interval categories as
codes plus exact left/right bounds, closure, order and original column order.
Native feature rankings retain AnnData record-array encodings. Native function and
column names containing “genes” do not establish gene annotation for supplied peaks.

Native variable-feature subsetting records exact source positions in the receipt
and H5AD provenance, including duplicate feature labels. Retained X is checked
against those source positions; an axis change is not admitted for an unrelated
function. The matrix adapter has a new representation profile; prior profiles remain
read-only with their exact historical interpretation and download authority. Catalog
additions alone do not change profiles. Source bytes and accepted history remain
immutable. Nonfinite native statistics remain nonfinite. Reuploaded downloads can
be analyzed and downloaded again: the shared export owner refreshes only its
reserved delivery marker in the derivative, pinning the exact accepted source
bytes. Earlier export metadata never grants scientific authority. Embedding-only
native results may omit X. Safe record-array statistics retain their data;
private string fields still pass through the existing privacy projection.

chromVAR sample/differential functions use supplied deviations X and `z_scores`,
without inventing scores or backgrounds. Differential calls take `group_key` from
actual obs. Correlation/distance use deviations with z-score variability selection;
`differentialDeviations` tests deviations and `differentialVariability` tests
z scores. Sample tables use `source_index`/`source_id` rows and `cell_0`, `cell_1`,
… columns whose order is exactly `cells.tsv`. Native `dist` is expanded to its
ordered square matrix. `getTotalFragments` returns one scalar `value`. These totals
retain supplied measurement semantics; neither totals nor deviation comparisons
become independent-molecule counts or replicate-level accessibility inference.
Signac `corSparse` receives features-by-cells and omits Y; a second matrix is not
silently created. Its dense native values use lossless sparse storage in H5AD.
Sample similarity allocations are bounded to 16 million elements.

MACS3 `cmbreps` currently takes exactly two tracks through the existing dataset and
control ports; `fisher` expects native -log10 p-values. `pileup` and `filterdup`
receive actual read/alignment files with explicitly requested format where needed.
MACS BEDPE is the package's three-column fragment representation. Output file bytes
remain pinned; downloads use the existing privacy projection, retaining bare `+`/`-`
strand fields. Genome-size aliases and other omitted settings are native defaults,
not inferred references. No second peak-calling workflow is introduced.

The isolated epiScanpy/Scanpy environment adds pinned `igraph==0.11.9`,
`leidenalg==0.10.2`, `louvain==0.8.2` and `texttable==1.7.0`; the existing lock records
them. Other installed packages were not upgraded. The original scientific package
versions remain unchanged. These optional graph dependencies retain their own
licenses, including GPL terms for igraph/Leiden/Louvain.

### Reviewed limits at 1F

The following records the 1F boundary; 1G updates are listed below.

- SnapATAC2 `tl.kmeans` is not exposed: float64 reaches an unimplemented Rust path;
  float32 is rejected by the installed binding. `tl.hdbscan` and `pp.harmony` need
  absent optional dependencies. No method or dtype is substituted. Fragment-backed
  peak/gene matrix operations require genuine native fragment storage and exact
  genomic relationships; a generic peak-count matrix does not supply them.
- epiScanpy `tl.diffmap` remains unexposed because of the documented upstream wrapper
  defect. `tl.rank_features` retains its own method overrides and errors; it is not
  a transparent alias for arbitrary Scanpy ranking settings. `pp.variability_features`
  also creates a native plot and ignores its `show` flag; no incomplete plot-result
  adapter is advertised here.
- chromVAR `getSampleDepths` needs alignment-file ports, not counts; background
  selection needs actual supplied bias. Those inputs are not fabricated.
- motifmatchr scores/positions are inspected but remain unsupported. Scores contain
  distinct match/score/count assays; positions require native genomic ranges and
  motif associations. The existing boolean-match contract is not reused for them.
- Cicero `make_sparse_matrix` is not exported in 1.28.3. Gene activity requires an
  original accessibility CDS with actual `fData$gene`, exact connection-table input,
  and separately named normalization metadata. The aggregated CDS exchange does not
  establish those meanings; no new Cicero function is claimed.
- Signac `Jaccard` requires two explicit matrices; no two-source matrix port is
  introduced. Seurat/ChromatinAssay operations require their actual object content.
  `FitMeanVar` and `PearsonResidualVar` are absent from 1.16.0.
- MACS3 `bdgdiff` and more-than-two-track `cmbreps` remain unsupported: four-role
  signal binding/multiple outputs or a bounded multi-track port are not implemented.
- ArchR and pycisTopic are not installed or executable. Real Arrow/project ownership
  and native topic/model state remain necessary. pycisTopic's upstream internal
  academic-use and redistribution restrictions do not authorize general hosted use.

The detailed planning fixtures measure **83,991 / 85,829 / 86,450 / 92,080 bytes**,
within the unchanged **96 KiB** maintenance budget. Interpretation is separately
**116,381 prompt + 99,015 schema = 215,396 bytes** for the same baseline fixture;
most growth is the scientific-parameter catalog. These are maintenance measurements,
not universal provider limits.

The [STE-NFI-1F report](../../../agent-data/acceptance/scientific-tool-ecosystem/ste-nfi-1f/REPORT.md)
records actual native parity, historical recovery, Firefox/OpenAI acceptance and
provider failures. Coverage does not imply full-package, biological or universal
provider qualification.


## STE-NFI-1G: native input families and interaction

The catalog contains **69 distinct package-qualified native functions** across
**eight packages**, still **43 Registry tools**. Three functions are new:
`MACS3 bdgdiff`, `chromVAR::getBackgroundPeaks`, and `Signac::Jaccard`.
`matchMotifs` gains two output modes; `corSparse` gains its two-matrix form.
These modes are not additional function counts.

| Interface | Actual inputs | Native output |
| --- | --- | --- |
| motifmatchr `matchMotifs(out="scores")` | Existing ordered peaks, qualified genome and motifs | matches, maximum scores and match counts, exact native R result |
| motifmatchr `matchMotifs(out="positions")` | Same genomic inputs | Native genomic hits with motif identity, 1-based closed ranges, strand and score; exact GRangesList RDS |
| MACS3 `bdgdiff` | Four actual continuous bedGraphs: treatment 1, control 1, treatment 2, control 2 | Three native BED5 files: condition 1 unique, condition 2 unique and common regions |
| chromVAR `getBackgroundPeaks` | H5AD numeric counts and actual numeric per-feature `var.bias`, or an explicit `bias_key` | Native 1-based feature indices per background iteration, preserving feature order |
| Signac `Jaccard` / two-source `corSparse` | Two H5ADs with identical ordered feature identities; independent cell axes and matrix slots | Rectangular cross-cell similarities, correlations or covariance |

For MACS3, the existing dataset/control ports bind treatment/control for condition 1;
`treatment2` and `control2` bind condition 2. Use unscaled pileup and lambda tracks,
not SPMR. `cutoff`, `min-len`, `max-gap`, `d1`/`depth1`, `d2`/`depth2` and `verbose`
are forwarded unchanged. Output destinations are managed fixed names, never CLI
kwargs. The primary table is condition 1 unique; `condition2.tsv` and `common.tsv`
remain separate downloadable files, even when a native table has zero regions.
This is native likelihood-ratio signal comparison, not replicate-aware edgeR DA.
`cmbreps` still accepts two tracks, not an arbitrary path list.

Background sampling forwards `niterations`, `w`, `bs` and an optional explicit R
seed. No GC estimate or bias is invented. Results reference the exact ordered
`features.tsv`; indices remain 1-based and may repeat. The result has a 16-million
element execution bound. The native package retains numerical validation/defaults.
`getSampleDepths` remains unavailable: its real BAM/BED file-list, pairing and
read-group semantics are not supplied by this count-matrix adapter.

Signac `second` binds an exact independent H5AD. `second_count_slot` selects its
own X/layer; it does not inherit the first matrix's measurement declaration.
No feature alignment, automatic binarization or Seurat object is fabricated.
Jaccard receives cells × features; corSparse receives features × cells and native
Y. `cov=true` selects native covariance. Rectangular values and both cell axes are
stored in `uns.Signac_pairwise`; single-source corSparse remains
`obsp.Signac_corSparse`. Pairwise output is bounded to 16 million elements.

A scientific task may compose multiple native calls. When no function is fixed by
the user, the existing Planner may select a reviewed `function_name` from Registry.
Explicit calls can scope grounded declarations to their exact native function,
including separate kwargs for several calls of one package. The existing compiler
binds these scalar sources and the typed admission owner checks exact consumers;
there is no keyword router or second executor. Multiple instances of the *same*
function with conflicting settings remain ambiguous; no new step-memory mechanism
is introduced. Specify `use_rep="X_pca"` to establish actual embedding consumption;
merely retaining X_pca does not establish that it was used.

The Interpreter now uses one fixed typed proposal schema. Tool, native-function,
argument, source and port names are proposed canonical identities, not schema enums
or execution authority. Existing admission validates each identity against Registry
and captured context, grounds user values in exact utterance spans, checks function
and source scopes, and protects explicit declarations through planning and compilation.
Native packages retain numerical validity and omitted defaults.

One Registry-derived capability projection exposes all **43 tools, 69 native
functions and 164 parameter offers**. Every native function's supported kwarg names
remain visible for task discovery. Exact registered API names expand their actual
kwarg descriptions; explicit package/tool names, pending prerequisites and active
accepted results expand relevant package detail. This changes descriptive detail
only: it neither selects an operation nor adds a routing model call. The existing
Planner receives complete method details and composes scientific dependencies.

The combinatorial Interpreter schema builder, enum factoring/compactor and shared
parameter alias projection are removed. There is one production protocol and no
old-schema fallback. Existing persisted declaration readers retain real historical
wire meanings; they do not select another generation protocol. Accepted parameter
evidence uses the same Registry-owned function-qualified input keys as admission,
so independent calls of one package retain their exact user attribution.

The section-13 matched official-API fixture measured **101,263 prompt bytes + 32,340 schema
bytes**, **182 enum occurrences** and **26,457 input tokens**. The preceding design
measured 102,429 + 88,181 bytes, 682 enums and 44,910 input tokens. Synthetic additions
of tools, functions, parameters and independent source fields change projected
context but leave the schema unchanged. Context still grows with real information;
this is not an unlimited-context claim. Planning's separate complete-catalog
projection and existing maintenance bounds remain unchanged.
Section 14's clarified choice/source instructions bring the matched prompt to
**102,211 bytes**; the fixed schema and complete inventory remain unchanged.

The [1G report](../../../agent-data/acceptance/scientific-tool-ecosystem/ste-nfi-1g/REPORT.md),
sections 13–14, owns the current architecture and evidence. A bounded official OpenAI
request using the complete production protocol passed interpretation, admission,
planning, compilation and output selection with science intentionally withheld.
A new genuine Firefox natural-language task then completed native Scanpy PCA and
an actual checked H5AD download. A parameter-evidence reader omission initially
withheld the explanation; it is corrected and the accepted summary reads after
restart, while the saved historical reply stays exact. A second explicit neighbors
continuation returned a complete model response but failed admission before science.
Its exact generated decision was not persisted, so the precise rejected association
is not established. Those historical limitations are retained in section 13.

Section 14 adds genuine Firefox/official OpenAI task composition of PCA, neighbors,
Leiden and UMAP, exact accepted-PCA continuation, a neighbors parameter change
without PCA rerun, and a complete reviewed comparison answer. A new motif positions
result is accepted with six checked browser downloads. Missing dataset input asks
for the file without creating a Run. These are genuine provider-driven executions,
separate from the earlier scripted plans and native parity. Task completion prose
is partially reviewed; native summaries can remain technical. No automatic PNG is
claimed for generic native Scanpy UMAP or motif positions.

Explicit function selections use the reviewed choice value; `native_function`
scopes parameters but cannot replace that value. A source/port declaration asserts
direct consumption of that original file, not ancestry through a derived matrix.
Compiler automatic binding considers only the selected function's qualified scalar
sources; unspecified kwargs remain omitted. Prerequisite advice and parameter
continuation share a temporary projection for one unambiguous native function.
The captured request stays exact; multiple function scopes wait for planning and
scientific input validation. Historical questions retain their saved meanings.

Fourteen bounded submissions observed 30 provider calls, including two connection
failures; 28 completions exposed usage. Restart preserves accepted outputs and
displayed history. The bounded recommendation is CONDITIONAL GO for demonstrated
controlled research use, not every function, provider response or biological claim.

The earlier empty/incomplete responses and diagnostic-only reduced schema remain
historical evidence in sections 11–12. The OpenAI adapter retains safe HTTP/status/
usage diagnostics and distinguishes missing/null/zero usage. Provider settings,
credentials, retry policy and scientific algorithms are unchanged. No unrestricted
provider, scientific or biological qualification is implied.

Cicero gene activity still needs an original accessibility CDS with actual gene
annotation and exact connections, then separate normalization inputs. SnapATAC2
fragment import needs its native fragment storage (including noncanonical sparse
indices), genome ordering and cell-filter semantics, not an ordinary H5AD adapter.
At 1G, ArchR Arrow/Project and pycisTopic objects were not executable integrations.
1H adds the focused ArchR contracts linked above; pycisTopic and broad Seurat
adaptation remain unsupported. No deferred API is advertised as callable.

For the complete 1H catalog, the largest maintained detailed-planning fixture is
103,379 bytes (64,358 prompt + 39,021 schema). The maintenance budget is 112 KiB,
up from 96 KiB, retaining about 11 KiB headroom. This growth represents two real
scientific object/enrichment tools and their source/parameter contracts. The fixed
Interpreter schema stays exactly 32,340 bytes; no alternate protocol, missing
capabilities or provider-specific workaround is introduced. The budget is not an
API limit or a claim about provider latency.
