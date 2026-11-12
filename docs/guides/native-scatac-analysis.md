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

## Native Gene Activity and exact populations (ICTA-2)

AI Research exposes two native constructors through the existing matrix adapters.
They publish independent cells × genes/transcripts H5AD files through the ordinary
`dataset` port (`native_matrix_h5ad.v1`), with exact source bytes and ordered cells.
Neither constructor normalizes, binarizes or relabels the supplied measurement.

- `run_snapatac2(function_name='pp.make_gene_matrix')` consumes the accepted native
  fragment-backed H5AD and a compatible managed `gene_anno_path` / checksum GTF or
  GFF. Fragment input first uses the existing native import. Defaults are
  `id_type='gene'`, upstream 2,000 bp, downstream 0, whole gene body, and
  `counting_strategy='paired-insertion'`. Native fragment/body, insertion/end-point
  and paired-insertion counts remain distinct. Explicit `use_x=True` aggregates
  existing region values and records their original measurement. `inplace=True`
  mutates only the task-local copy; otherwise the adapter binds native `file` to
  the managed H5AD destination. The installed `backend` argument is supported.
  Arbitrary output-file kwargs are not accepted. Keep the accepted fragment source:
  a new native gene matrix need not contain fragment storage. Native feature names
  and attributes are preserved; absent gene IDs are not reconstructed from names.
- `run_episcanpy(function_name='tl.geneactivity')` consumes ordinary peak H5AD and
  a compatible managed GTF on the same `gene_anno` port. Native `var_names` parsing
  accepts `chr:start-end` and `chr_start_end`. It sums the supplied peak values
  overlapping native gene domains, using upstream 5,000 bp by default. Binary
  accessibility therefore produces sums of accessible features, not fragment
  counts. The installed default `annotation='HAVANA'` filters GTF source records;
  explicit `annotation=None` accepts all sources. `feature_type` is passed unchanged.
  The native coordinate conventions and chromosome names are not corrected.
  Only overlapping features are returned; duplicate symbols are preserved.
  Missing gene/transcript IDs, unnamed indexes and native duplicate names are retained
  as returned; they are not repaired or biologically qualified.

The installed epiScanpy 0.4.0 implementation creates dense intermediates and output.
It uses the existing isolated memory/time/cancellation boundary, with no alternative
Agent calculation or added size cap. Empty overlap can fail inside native
concatenation. `key_added`, `layer_name` and `copy` do not change its actual new-X
return behavior. `raw=True` selects `.raw.X` even when a layer was selected as X.
The native implementation indexes that matrix using current `var_names` positions.
Different raw/current feature axes do not cause an Agent refusal: the receipt records
ordered equality or inequality and the lack of independently verified correspondence.
Actual missing-raw or out-of-range failures remain native failures. Independent raw
measurement history stays unknown.

All three routes expose `parameters.gene_activity.interface='gene-activity.v1'` in the
existing native result. This compact description identifies X, actual feature type,
ordered cell/feature and feature-attribute digests, native feature index name, original selected slot and
measurement, declarations, known source history and inherited metadata slots.
The enclosing result supplies construction method, effective native parameters,
software versions, annotation checksum, source checksum and accepted publication
identity when available. The same description accompanies the H5AD in
`uns.agent_gene_activity`; that field alone grants no scientific authority.
NativePublication and accepted Run/step bindings remain authoritative. Copied graphs,
embeddings and metadata are explicitly source-derived, not new gene-space analyses.
Unknown transformation history remains unknown. A retained construction description
is not a claim that a later transformed X is still raw activity: consult the current
`matrix_semantics` and exact native invocation/accepted ancestry.

The third route is direct existing Gene Activity/Gene Score reuse. Pass
`gene_activity=True` to the requested existing `run_scanpy`, `run_episcanpy` or
SnapATAC2 **matrix** consumer. `count_slot='X'` (default) or an explicit layer
selects the unchanged native input values. No annotation file, peaks, Agent-specific
uns field, separate gene IDs, species or normalization declaration is required.
Supplied `matrix_semantics` and source metadata retain their meaning; absent history
stays unknown. This declaration identifies the selected input, not an independently
verified construction. An uploaded `uns.agent_gene_activity` is source data only.
Exact accepted source proofs carry the recorded construction through subsequent X
consumers; they do not identify an arbitrary different layer as activity.

For example, explicitly logarithmized Gene Activity with `obs['group']` can go
straight to `run_scanpy(function_name='tl.rank_genes_groups', gene_activity=True,
kwargs={'groupby': 'group', 'method': 't-test', 'use_raw': False})`. The same native
Scanpy path consumes both constructor outputs. In the tiny qualification fixtures,
`pp.log1p` was explicitly requested before marker testing; the adapter never inserts
it. Native comparison results and warnings are retained. Gene Activity is not
measured RNA expression and no universal marker cutoff or cell identity is inferred.

Optional `cell_subset` is an internal input adaptation on those existing matrix
consumers. For an annotation-owned cluster selection, the existing result binding
supplies an accepted source with its grouping; the internal invocation supplies only
`cell_subset={'groupby': 'leiden', 'groups': ['0']}`. The adapter derives the complete
actual membership in source order. Researchers need not enter positions, cell IDs,
checksums or arbitrary population labels. Registry metadata does not expose a
conversational JSON cell-selection flow. This prepares ICTA-1 child invocations;
biological branch decisions and hierarchy validation remain future work.

The existing source binding and NativePublication retain accepted Run/step/publication
identity. The adapter records the actual source checksum, ordered positions, cell
identities and source-axis digests in the published execution parameters. Explicit low-level
positions are also supported; supplied checksum/ID assertions must agree with that
source. When grouping is supplied alongside positions, every selected cell must
belong to the claimed groups. Complete membership is recorded descriptively, not
required for every native operation. No `population` label is required.

AnnData performs `a[positions, :].copy()`; the requested native operation consumes
that exact selection and publishes its ordinary H5AD. Repeated positions retain
repeated observations and native warnings, without deduplication or renaming. This
does not qualify duplicate cells for future biological branches. Categorical readers
preserve repeated positions; association within the same pinned artifact is positional.
Cross-artifact readers that have only names cannot establish an unambiguous join for
repeated IDs. The adapter and native publication/recovery preserve them faithfully.

Original accessibility and activity sources must have an established exact cell
relationship. In the execution fixture, accepted clustering retains original
accessibility X, and its native activity construction retains the same ordered cells
and grouping. The existing accepted-result bindings feed both ICTA-1 child Runs;
the same group intent derives matching memberships with each source's own checksum,
positions and distinct feature axis. No equal-size or barcode-based automatic
correspondence is inferred. Features, X/layers and source semantics remain separate.
Source files and accepted upstream publications remain unchanged.

Sliced global graphs and coordinates remain inherited slots. Their accepted graph
provenance is not carried as newly computed local graph provenance. Request native
feature selection/reduction/neighbors/clustering explicitly for a local analysis;
these transformations and ASW retain their existing owners. Explicit requests to use
inherited slots still reach native functions. No minimum population threshold is
added. Native numerical feasibility and failures remain authoritative.

Default AnnData orientation is cells by features. Only explicit
`input_orientation='features_by_cells'` invokes native AnnData transpose before slot
and population selection; positions then refer to original var. Shape never selects
orientation. This is not a general conversion interface. Raw SnapATAC2 constructors
retain their existing raw input owner; these optional adaptations apply to matrix
consumers.

The ICTA-2B report records genuine registered marker calls, accepted PriorOutput
reuse, source-bound subsets, Session restart and H5AD delivery for all three routes.
Tiny synthetic fixtures qualify mechanics, not biological annotation accuracy or
production-scale performance. ICTA-1 child lifecycle, direct Native Functions,
MAESTRO annotation and UI remain unchanged. The ICTA-3B and ICTA-3C sections below
describe current internal evidence interpretation; hierarchical orchestration
remains a later phase.

### Shared native invocation boundary (ICTA-2 engineering closeout)

Registered native kwargs accept data-only strict JSON without arbitrary string,
collection, nesting or payload-size caps in the matrix adapter. Callable and kwarg
names remain registered; executable objects and nonfinite JSON literals remain
unsupported. The existing isolated process controls memory and time. Sparse Scanpy
`pp.scale(zero_center=True)` now reaches that installed function unchanged; native
Scanpy owns any densification and the actual result or resource failure.

Optional supplied declarations do not veto a general matrix invocation. The exact
manifest retains supplied arguments; source identity records source declarations
and supplied declarations separately. Existing source species/assembly and X
measurement retain their stored meaning; no metadata disagreement establishes
reference compatibility. Requested native transformations still describe their
actual output measurement. Dedicated coordinate/reference qualification retains
its existing contract and is not inferred from these declarations.

The shared HDF5 inspector accepts ordinary complex numeric arrays and flat record
arrays with more than 128 fields. Native publication, recovery and privacy export
retain their exact values. External/virtual storage, unreviewed filters and unsafe
encodings remain rejected; host-side storage/privacy budgets remain enforced.
Native Mapping returns retain string keys without coercion; ambiguous nonstring
keys and unsupported serialized objects fail with a technical publication error.
No general Python object transport or scientific repair is introduced.

### Current-round marker candidates (ICTA-3B, internal)

`agent.tools.analysis.marker_candidates` consumes an exact accepted `run_scanpy`
`tl.rank_genes_groups` dataset binding through the existing RunStore/Registry
integrity checks. It does not execute science or register a new tool. A future
caller submits the ordinary native step using `marker_kwargs(actual_groupby)`:
Wilcoxon, native BH correction, reference `rest`, detection fractions enabled,
`use_raw=False`, and `n_genes=None` (all tested features). Other parameters retain
installed defaults. An already accepted suitable result is consumed directly.
SnapATAC2 activity, epiScanpy activity and declared existing Gene Activity/Gene
Score use the same calculation. No normalization or reconstruction is implicit.

`annotate_markers(binding, store=..., registry=..., evidence=CellMarkerEvidence(...),
species=..., context=...)` returns ordinary serializable evidence. The native
publication retains the actual selected X measurement, source and construction
history, cells, grouping, parameters, Run/Step and publication identities. Marker
records retain original feature names, signed scores, raw/adjusted p-values and
available detection fractions. Nonfinite values are represented as missing in
this JSON projection; the original H5AD remains authoritative and unchanged.

The default gene identity is an exact nonempty feature name. A caller may explicitly
select a feature column and either `identity_kind='name'` (literal marker/symbol
lookup) or `'gene_id'` (literal database ID lookup). Numeric IDs are not converted;
`12501.0` and `12501` remain different. There is no alias or ontology inference.
Repeated literal identities count once, and a gene is positive if any corresponding
tested feature has positive score and native adjusted p-value at most 0.05.
If duplicate ranked feature names cannot identify a unique selected feature-column
value, only those names are excluded, with their original statistics and exclusion
reasons retained. ORA continues on all remaining comparable genes without a
minimum coverage requirement. Full ranked-feature coverage is
required for this evidence calculation, never as a native execution gate.

The universe includes all distinct comparable tested genes, including unmatched
ones. Candidate identity is exact `(species, cell_name)`; context can constrain
only explicitly supplied tissue/disease fields. All source categories contribute
together. Each candidate uses a set of distinct matched input identities; source
rows and multiple algorithms never multiply genes. Within each source association,
name lookup selects its nonempty symbol if that exact symbol matched the universe,
otherwise its exact matched marker; ID lookup selects its exact matched gene ID.
All original field matches remain in lazy evidence. This is a source-field
representation choice, not verified alias mapping or reconciliation across records.
For each cluster, SciPy
`hypergeom.sf(k-1, N, K, n)` and `false_discovery_control(method='bh')` evaluate
all candidates with `K > 0`, including zero-overlap candidates. Candidates with
`K=0` are recorded as uncovered, not biological negatives. Marker and enrichment
thresholds are separate evidence-interpretation arguments, both defaulting to 0.05.

All significant candidates are retained. The unique smallest adjusted enrichment
p-value yields `assigned`; an exact minimum tie yields `ambiguous`. Valid analysis
without significant enrichment is `unresolved`; unavailable identities or
reference coverage yield `insufficient_evidence`. These are current-round
computational outcomes, not probabilities of biological correctness.

Candidate summaries contain source categories, matched fields, method details,
record/context/literature counts and an exact pinned evidence locator. Entirely
blank literature tuples contribute zero; partial literature fields remain counted.
Empty method strings are not named methods. Original blanks remain in lazy evidence.
`candidate_evidence(result, candidate, lookup)` streams every original matching
record, including tissue/disease/literature fields, missing values and original
source-row references. No raw-row payload or candidate cap is imposed on ordinary
results. Integrity failures remain errors, not scientific uncertainty.

Multiple significant candidates do not trigger subclustering. MAESTRO, Native
Functions, public planning/execution and Session/Revision owners are unchanged.
ICTA-3C now supplies the internal subdivision interpretation described below;
ICTA-4 orchestration remains unimplemented. See the
[ICTA-3B report](../../../agent-data/acceptance/iterative-cell-type-annotation/icta-3b/REPORT.md)
for genuine native fixtures, exact provenance, preservation and bounded validation.

### Local subdivision evidence (ICTA-3C, internal)

`agent.tools.analysis.local_subdivision.assess_subdivision` interprets one attempted
local subdivision. It accepts the parent clustering binding, the four accepted
local bindings (`pp.select_features → tl.spectral → pp.knn → tl.leiden`) and
child-keyed accepted `tl.diff_test` bindings. It executes no native work. Existing
Registry, Runtime, native workers, publication and exact PriorOutput owners retain
all execution and integrity responsibilities; existing ASW and explicit parameter
precedence are unchanged.

The selected parent clustering must retain the original peak X. Existing
source-bound peak declarations or accepted `pp.make_peak_matrix` ancestry identify
that space independently of measurement units. For example, an input recording
`matrix_semantics='binary_accessibility'` or `'fragment_counts'` can carry the
explicit existing `matrix_semantics='peak_accessibility'` invocation declaration.
The native owner retains the original measurement and the supplied declaration
separately, tied to the exact source bytes. No new field or measurement whitelist
is required. A counting unit alone does not establish peaks. Gene constructors,
tile construction and raw fragments retain their actual native identity even if
a later declaration calls them peaks; filenames, shapes and matching cells do not
establish peak space. Accepted count-preserving ancestry carries a peak declaration
without requiring SnapATAC2-specific construction history.
The first local step uses ICTA-2 `cell_subset={"groupby": actual_parent_key,
"groups": [actual_parent_label]}`. The adapter derives positions, cell IDs and
source hashes. Complete parent membership and actual local-written feature,
Spectral and KNN slots are checked at evidence reading. These are correspondence
checks, not new native execution admission conditions. Separate representations
without accepted ancestry require the caller's existing exact handoff; this
component never constructs cross-file cell alignment.

`differential_kwargs(local_binding, child, store=..., registry=...)` derives actual
local child and remaining-parent positions and `direction='positive'`. The caller
uses ordinary registered `run_snapatac2(function_name='tl.diff_test')`. SnapATAC2
2.9.0 performs its logistic-regression likelihood-ratio test and native BH adjustment.
Omitted parameters retain native defaults, including `min_log_fc=0.25`,
`min_pct=0.05`, `features=None` and `covariates=None`. The installed covariates option
raises its genuine native error. Full native returns remain in the accepted H5AD;
the decision preserves the publication binding, complete table projection and
actual effective parameters. Nonfinite projected numbers become null while
original native bytes remain unchanged.

Default interpretation retains a child with at least five distinct actual peak
features having positive native log2 fold change and native adjusted p-value
**strictly below 0.01**. Positive sign identifies enrichment, without another
fold-change magnitude cutoff. Repeated native rows for a feature count once.
`adjusted_p_threshold` and `minimum_peak_features` can reinterpret the same accepted
statistics without executing science. They are initial interpretation defaults,
not universal biological truths or native admission rules. No cell-count,
population-fraction, feature-detection or stability gate is added.

A single candidate yields `no_candidate_subdivision`. Multiple candidates yield
`no_supported_subdivision`, `partial_supported_subdivision` or
`supported_subdivision`. Unsupported children remain explored native clustering
results, with their cells retained exactly in the residual parent. Supported
children plus residual cover the original selected positions once, preserving
source order. Unassessed evidence has `retained=None`, not a biological negative.
Native Leiden's `-1` observations remain unassigned residual cells and participate
in each child's rest comparison; they are not treated as another candidate child.
Failed Run/step locators preserve actual native or infrastructure execution errors;
`failed_local` can identify an unsuccessful next step after an accepted local
prefix. Missing peak or differential input stays `missing_input`. Integrity errors
propagate instead of becoming biological uncertainty. No result claims homogeneity.

Optional full ICTA-3B evidence is carried unchanged with an accepted
`annotation_bindings` construction/marker chain descending from the local
clustering. Peak and gene axes remain distinct, and the chain must retain exact
ordered cells and grouping. Names, ambiguity and numbers of significant CellMarker
candidates do not affect retention. Supported unnamed children remain exploratory.
A supplied `budget_reason` is separate from scientific stopping; the recorded
default maximum is five levels, but no depth enforcement or scheduler exists here.

This is an [IDclust-inspired](https://doi.org/10.1093/nargab/lqae174) decision strategy
using SnapATAC2, not a reproduction of IDclust's pseudobulk edgeR or its optimizer.
The [mouse-brain atlas](https://doi.org/10.1038/s41586-023-06824-9) supplies the local
SnapATAC2 clustering basis. Significant peaks are exploratory within-data evidence,
not independent validation or proof of a new cell type. ICTA-4 owns future
orchestration, result-scoped exposure, iteration and budget enforcement; none is
implemented here. See the [ICTA-3C report](../../../agent-data/acceptance/iterative-cell-type-annotation/icta-3c/REPORT.md)
for genuine native outcomes, independent checks and preservation evidence.

### ICTA-4A internal iterative annotation and hierarchical H5AD

ICTA-4A implements the internal orchestration previously deferred above. It does
not change the production `annotate_scATAC_cell_types` Registry entry, which remains
MAESTRO-backed until ICTA-4B. `iterative_annotation.coordinate` is a durable hook;
only the opt-in test harness registers it. Native Functions has no implicit
coordinator, provider call or annotation workflow.

The caller captures an original H5AD, an accepted native grouping through
`PriorOutputRef`, an accepted compatible Gene Activity result or one selected
ICTA-2 construction route, and the installed CellMarker resource identity.
Initial clustering remains the existing research planner's responsibility. The
coordinator does not choose another method or plan a missing global graph. It
uses `annotation_invocation().invoke` for every native operation and the same
Runtime, Executor, RunStore, source bindings and native publication owners.

Activity is constructed once when needed, with the captured native method,
reference, slot and kwargs. Optional `activity_source_path` is an exact accepted
`PriorOutputRef` selected independently of `grouping_path`: default SnapATAC2
`pp.make_gene_matrix` needs actual fragment/insertion storage, while explicit
`use_x=True` retains its native X-based semantics. Omission preserves the existing
grouping-input route, including epiScanpy peak-count construction. There is no
storage inference, source substitution or automatic change to `use_x`; native
functions judge their supplied inputs. Accepted `make_peak_matrix` ancestry
retains the verified ordered cell axis back to the selected fragment source, so
activity and grouping can meet at that exact source without barcode inference.
Existing accepted activity is reused without implicit
normalization or logarithm. Registered Scanpy markers accept an optional internal
`grouping_path` binding: the native input adapter derives exact activity positions
from common accepted source ancestry and recorded subsets, then projects the
selected clustering column. Cell IDs are checked after this positional join;
matching names, shape or uploaded `uns` cannot establish it. The projection records
its grouping publication, checksum and actual activity positions. It already
selects the complete current population; additional subset or transpose adaptation
must occur upstream. This handoff neither computes groups nor rebuilds activity.
Independently supplied activity with no exact common ancestry needs an established
source association before it can be paired with a separate peak result.

Each round uses the existing `marker_kwargs` defaults and `annotate_markers`:
Scanpy Wilcoxon/BH, positive signed markers with adjusted p ≤ 0.05, all static
CellMarker categories without weighting, SciPy hypergeometric ORA/BH with adjusted
p ≤ 0.05, unique best significant candidate, and exact ties kept ambiguous. Labels
are current-round computational evidence, not independently established cell types.

Branches are scheduled breadth first, in first-appearance order on the accepted
cell axis. Their global IDs hash the parent branch and actual native group value.
Each eligible branch performs the existing local select_features → spectral → knn
→ leiden sequence, with bounded ASW for omitted resolution and explicit parameters
prevailing. Its genuine positive child-versus-parent-rest differential Runs feed
`assess_subdivision`. Only supported children become branches. Unsupported and
native `-1` cells remain in the residual parent, without residual reclustering.
Naming, ambiguity and the number of significant labels do not control recursion.

`maximum_levels` defaults to **5**, counting the initial annotation as level 1.
This is a configurable execution budget. Branches can stop independently for no
candidate, no support, partial/full retention, missing original peak space,
native-unassigned membership or budget exhaustion. Gene-only accepted data still
receive current-round annotation and a factual missing-peak limitation. Native
failure, cancellation and interruption propagate through the existing lifecycle;
they do not become biological stopping evidence or a completed publication.

The existing reservation/outcome trace is the only durable progress owner.
Explicit continuation reconstructs the tree from accepted children and repeats
only deterministic evidence interpretation. Accepted native Runs are verified and
reused. Failed/unknown child execution remains failed/unknown: continuation does
not authorize scientific retry or alter parameters. Successfully accepted sibling
work remains intact. The ICTA-1 completion gate requires every reserved child to
have an exact successful outcome before accepting/recovering a parent publication.

`hierarchical_annotation` publishes one managed `annotated.h5ad` and three evidence
files (`hierarchy.json`, `annotations.json`, `subdivisions.json`) through
`NativePublication`. The manifest pins exact sources, resources, native evidence,
implementation files and output checksums. The H5AD copies the original encoded
scientific slots, preserves every ordered observation, X, layers, raw and existing
scientific metadata, and adds these observation columns:

- `cell_type_level_N`: the assigned current-round label at that level; genuinely
  missing for unnamed/ambiguous branches and for deeper levels never reached.
- `annotation_branch_level_N`: the global branch identity independently of labels.
- `cell_type_final`: the deepest assigned label available along the ancestry.
- `annotation_status`: the current leaf's assigned, ambiguous, unresolved,
  insufficient-evidence or unassigned status.
- `annotation_branch_id` and `annotation_depth`: the actual current leaf and level.

An unnamed child can therefore retain a broad parent label in `cell_type_final`
while its own label is missing and its current status remains unresolved. That is
not a confirmed subtype. `uns["agent_hierarchical_annotation"]` contains an ordinary
AnnData dictionary with the level budget, relative evidence filenames and a branch
DataFrame indexed by global branch identity. Each row records its parent (empty
for roots), level, cell count, residual cell count and stopping reason. It contains
no native execution records or private absolute file locations. The complete
hierarchy and marker/ORA/differential evidence remain in the pinned JSON sidecars
and native child publications, without truncation. Verification compares every
portable branch row with the complete hierarchy. Uploaded hierarchy metadata is never execution
authority. Existing observation columns with these output names are not silently
overwritten. Verification checks the hierarchy partition, exact tested child and
residual memberships, original slots, observation projection and file identities.

The shared H5AD inspector reads complete UTF-8 strings with h5py, including
compound-record fields and attributes. It applies no per-string byte limit or
per-attribute size/count limit. Decoded dataset and attribute payloads contribute
to the existing configured aggregate resource budget. Storage-layout, link,
compression-filter and privacy checks remain in force, including private text
appearing late in a long value.

The [ICTA-4A report](../../../agent-data/acceptance/iterative-cell-type-annotation/icta-4a/REPORT.md)
records bounded synthetic/native qualification and limitations. ICTA-4B must wire
this owner to the single existing public operation, connect existing result-scoped
parameter binding and SRV-1 delivery, and fully retire MAESTRO-specific production
science, preparation, contracts, reports, figures and downloads. No MAESTRO fallback
or dual production annotation route is intended. Web/browser and biological
qualification remain outside ICTA-4A.
