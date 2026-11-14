# Iterative scATAC cell-type annotation

`annotate_scATAC_cell_types` is the single AI Research annotation operation. It
uses the ICTA-4A deterministic coordinator and hierarchical NativePublication
owner. The former MAESTRO execution, preparation, contracts, RP projections and
delivery implementation are deleted. Native reference-to-query label transfer
remains a separate supported operation. Native Functions keeps direct structured
calls and never implicitly starts this coordinator or a provider.

## Inputs and planning

Ask the Agent to annotate the selected scATAC cells. The ordinary AI Research
planner selects the original accessibility source. An explicitly supplied original
`obs` column (`group_column`) is reused directly at level one, without an initial
Leiden execution or synthetic clustering publication. Multiple plausible source
columns are offered by their observed names through the existing clickable choice;
an explicit column needs no confirmation. The selected grouping is recorded as
`user_supplied`: its biological identity and derivation are not validated, and no
`uns.grouping` record is required. Exact accepted native grouping reuse remains
available. If groups are absent, ordinary planning selects registered native
clustering prerequisites. An explicit reuse-only request never permits recomputing
initial groups. Subsequent branches always compute the approved native local
clustering and differential evidence; they do not reuse the original groups as
local subdivisions. Equal filenames, dimensions or cell names do not establish source
correspondence. An unresolved required identity uses the existing scoped question.

The ScientificResourceLibrary selects installed CellMarker 3.0 evidence and pins
its identity. Biological species declarations are matched to the pinned reference's
actual species value by unique `casefold()` equality. The captured researcher
declaration remains unchanged; evidence records the declaration and resolved
reference value. Unknown or ambiguous species produce a diagnostic, rather than
an empty biological evidence result. There is no species synonym inference.
Compatible accepted Gene Activity is reused. Otherwise, the chosen
native construction receives its actual source and gene annotation reference:

- Default SnapATAC2 `pp.make_gene_matrix` receives accepted fragment/insertion storage.
- Explicit `use_x=True` retains the chosen X and supplied parameters.
- epiScanpy `tl.geneactivity` uses its appropriate peak input.

Inspection passes through the original accessibility file; it is neither a
Gene Activity result nor an independent grouping input. Planning supplies a distinct
activity or grouping port only when that scientific input is actually selected.

Native packages own input legality. Ordinary scATAC H5ADs need no Agent peak-space
declaration or feature certificate to reach local native analysis. Every eligible
branch below the depth budget proceeds through the registered native route.
The intended input is cell-by-peak accessibility; this does not qualify gene or
tile features as differential peaks or broaden the peak-specific retention rule.
Incompatible inputs and native failures remain visible failures.

## Scientific execution

The coordinator uses accepted native Scanpy Wilcoxon markers with BH correction,
exact CellMarker gene matching, and SciPy hypergeometric ORA with BH correction.
Each eligible branch uses original peak space with native SnapATAC2 feature
selection, Spectral, KNN and Leiden. The existing bounded ASW scan applies to
unspecified resolution; explicit researcher values prevail. Native `tl.diff_test`
compares candidate children with parent-rest, and ICTA-3C evidence decides retention.
There is no LLM planning call inside the iteration.
Identical CellMarker background queries reuse one reference projection within
that coordinator invocation. The exact gene universe, namespace, species and
filters must agree; resource registration and file snapshots are checked again
on every read. Rounds receive independent records. Native marker publications,
statistics and their accepted bindings are still checked separately.

ICTA differential preparation excludes constant zero/one responses on the exact
accepted child-versus-rest cells, intersecting any selected features. Installed
SnapATAC2 2.9 constructs this response by replacing sparse stored values with one
(including explicitly stored zeros). Original X, feature axes, library sizes and
native parameters are retained. Native BH adjustment covers the hypotheses that
pass native filtering within that testable selection; it does not cover all
original peaks. Evidence records both feature counts and the actual native-tested
count. An empty testable selection retains the native empty result and explicitly
records insufficient differential evidence. Direct Native Functions calls keep
their original inputs and parameters without this ICTA preparation.

Independent branches execute in bounded parallel waves at the same level;
aggregation and hierarchy order remain deterministic breadth first. The existing
native process owner shares OS memory reservations across Agent processes and
Native Functions workloads. Each operation retains its own systemd/cgroup memory
cap and zero swap; the sum of reserved caps cannot exceed the aggregate budget.
Insufficient current capacity waits with cancellation checkpoints. Abandoned
reservations are reclaimed only after their native subtree is stopped.

`AGENT_NATIVE_MEMORY_BYTES` remains the per-operation limit. Optional
`AGENT_NATIVE_TOTAL_MEMORY_BYTES` can lower the host-derived aggregate budget;
headroom is the larger of 2 GiB and 10% of effective host/ancestor memory.
`AGENT_NATIVE_CPU_COUNT` defaults to at most four available CPUs per operation;
capacity accounting respects CPU affinity and ancestor quotas, and worker thread
environments are bounded. CPU quota is effective only when the host delegates
the CPU controller. Execution diagnostics distinguish memory limits, sampled
cgroup use, process RSS, native timings and resource waiting.

Default depth is five, including initial level one. Current-level marker and ORA
thresholds remain 0.05; subdivision retains the reviewed adjusted-p threshold 0.01
and five qualifying peak features. Configured values, actual ASW selections and
native execution identities are retained. Native failures are failed work, never
successful biological stopping reasons.

## Results and presentation

The accepted H5AD preserves original X, layers, raw, cells, features and scientific
metadata. It adds `cell_type_level_N`, `annotation_branch_level_N`,
`cell_type_final`, `annotation_status`, `annotation_branch_id` and
`annotation_depth`. Unreached levels remain missing. Broad ancestor labels may
remain as final labels when a finer round has insufficient or ambiguous evidence;
this is distinguished from a newly assigned finer label. Cluster IDs never become
biological labels. The compact `uns` hierarchy contains no copied native results.

Summary facts show actual reached levels, branch and terminal-population counts,
evidence statuses, inherited-label cells and bounded stopping outcomes. Complete
hierarchy, markers/ORA and differential evidence remain in `hierarchy.json`,
`annotations.json` and `subdivisions.json`.

New `annotations.json` uses `annotation-evidence.v1`, with `rounds` and
`shared_backgrounds`. A round's integer `background` selects its complete tested
features, gene universe, reference query and candidates in that same file. Only
byte-equivalent backgrounds share a block. Each `marker-candidates.v2` round keeps
cluster markers as explicit `columns` and ordered `rows`. Zip each row with its
column names to recover the complete record, including native statistics and
repeated-name detection-fraction arrays. No child publication is needed to
interpret these tables. Accepted earlier row/table evidence remains readable and
is never rewritten; rankings and ORA semantics remain unchanged.

Subdivision differential rows use the same explicit column-table layout, retaining
all native fields. Within native records, repeated effective parameter arrays may
use `effective-parameters.from-kwargs.v1`: `from_kwargs` names arrays retained in
the same record's `kwargs`, and `values` contains the remaining effective settings.
Reconstruct effective parameters by copying `values` and adding each named array
from `kwargs`. Complete source identities, native Runs, requested/effective values,
memberships and statistics remain in each downloadable sidecar.

For example, ordinary Python can read the shared background and a marker table:

```python
round_evidence = document['rounds']['initial']
background = document['shared_backgrounds'][round_evidence['background']]
table = round_evidence['clusters'][0]['markers']
markers = [dict(zip(table['columns'], row)) for row in table['rows']]
```

The existing scientific gallery provides composition, branch relationships,
representative-marker dot plots and heatmaps. Values come from accepted Gene
Activity X, with clearly labeled display-only `log1p`. This is accessibility-derived
activity, not measured RNA expression. Marker selection follows accepted positive
native marker order: at most two per displayed branch, twelve distinct genes and
thirty branches. The hierarchy view shows at most sixty branches; omissions are
explicit. No positive evidence means no invented marker plot.
The gallery and deferred marker overlay share the same positional feature reader.
Repeated selected names without an unambiguous native column are explicitly
omitted from presentation; scientific values are neither aggregated nor renamed.
Composition shows final labels separately from current-round evidence status,
including the count of cells inheriting an ancestor label. Inheritance does not
turn ambiguous or unresolved evidence into a confident current-round assignment.

Qualified accepted global UMAP can be colored by final annotation or reached levels
(up to five displayed levels), and by at most six representative markers. The
annotation-level control only selects existing presentation artifacts; it creates
no scientific Run. Before planning, missing compatible coordinates prompt the
fixed English **Generate UMAP?** choice. **Generate UMAP** requests registered
native representation prerequisites and UMAP in the annotation plan, retaining
the supplied initial grouping. SnapATAC2 UMAP consumes its selected representation
and fits its own neighbors; Scanpy and epiScanpy use their compatible native neighbor
graphs. A visualization request alone does not require a new Leiden grouping.
**Skip UMAP** continues annotation without the
optional computation. These clicks use captured choices without another model
interpretation call. Reporting never fits coordinates; a cropped global plot is
not a local UMAP. Exact cell ancestry can survive native representation transforms
without granting those transformed matrices original count authority. The
ICTA-4C-1 report records the current browser qualification
and any incomplete integration boundaries.

Hierarchical JSON evidence has an explicit 512 MiB bound per sidecar, including
verification, figure projection and detailed rationale reads. Full tested-gene
tables can exceed the generic 64 MiB metadata limit; original checksums, source
authority, duplicate-key and finite-value checks remain required.

Downloads & reports exposes the primary hierarchical H5AD, all three evidence
sidecars, source-bound figures and the technical report. Download privacy and
integrity use the shared native delivery owner. Reports record native operations,
parameters, producing identities and interpretation limits without rerunning science.
If transfer capacity is busy, let an active download finish and retry the file;
the existing transfer and memory limits remain in effect.

## Progress, editing and continuation

The existing annotation progress trace supplies current level, branch, native
function, actual completed/running child steps and known pending branches. Future
branch work has no invented fixed total. Cancellation and recovery retain Runtime,
RunStore and accepted-child ownership.
Completed annotations show their recorded completed child count. An accepted
continuation is identified from its existing trace; old interruption messages
remain historical and are not rewritten as an uninterrupted execution.

Adjust parameters projects actual recorded child operations, including exact branch
and operation identities for repeated native calls. A selected edit creates a new
Run and preserves the old result. Matching accepted upstream operations can be
reused; changed operations and dependent work get new native producing identities.
Owned marker method/comparison/correction and differential population bindings are
not editable. ORA/differential interpretation settings and depth belong to the
coordinator. Failed-work exact continuation remains distinct from result editing.

Conversational responses retain the existing user-input language policy. Fixed UI,
plots, H5AD fields and technical reports use English. No translation subsystem is
added.

## Qualification boundary

The [ICTA-4B report](../../../agent-data/acceptance/iterative-cell-type-annotation/icta-4b/REPORT.md)
owns exact execution, regression, preservation and remaining acceptance limits.
Small synthetic native fixtures establish mechanics, not biological validation or
production-scale performance. The [ICTA-4C-1 report](../../../agent-data/acceptance/iterative-cell-type-annotation/icta-4c/phase1-h5ad-browser/REPORT.md)
records the ongoing genuine H5AD/browser acceptance and its incomplete boundaries.
External scoring and large PBMC/GLUE benchmarking remain outside that acceptance.
The [ICTA-4C-1 optimization report](../../../agent-data/acceptance/iterative-cell-type-annotation/icta-4c/phase1-h5ad-browser/optimization-2026-11-14/REPORT.md)
records real accepted-evidence performance comparisons, complete table/privacy
equivalence and isolated figure qualification. Fresh Generate UMAP browser
acceptance remains pending.
The [final pre-Generate-UMAP optimization report](../../../agent-data/acceptance/iterative-cell-type-annotation/icta-4c/phase1-h5ad-browser/final-optimization-2026-11-14/REPORT.md)
records further invocation-local read/preparation reuse, self-contained sidecar
deduplication, figure cleanup and preserved-science checks. Generate UMAP remains
unstarted by that pass.
The [engineering Git checkpoint](../../../agent-data/acceptance/iterative-cell-type-annotation/icta-4c/phase1-h5ad-browser/engineering-git-closeout-2026-11-14/REPORT.md)
records the cumulative review, scoped checks and authorized commit/push.
ICTA-4C-1 remains **CONDITIONAL**: fresh Generate UMAP Firefox acceptance is
still outstanding.
