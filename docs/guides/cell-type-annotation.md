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
its identity. Compatible accepted Gene Activity is reused. Otherwise, the chosen
native construction receives its actual source and gene annotation reference:

- Default SnapATAC2 `pp.make_gene_matrix` receives accepted fragment/insertion storage.
- Explicit `use_x=True` retains the chosen X and supplied parameters.
- epiScanpy `tl.geneactivity` uses its appropriate peak input.

Native packages own scientific input legality. No extra genome whitelist, alias
mapping or inferred gene-identity equivalence is introduced. Gene-only inputs may
support current-level annotation but cannot establish peak-space subdivision.

## Scientific execution

The coordinator uses accepted native Scanpy Wilcoxon markers with BH correction,
exact CellMarker gene matching, and SciPy hypergeometric ORA with BH correction.
Each eligible branch uses original peak space with native SnapATAC2 feature
selection, Spectral, KNN and Leiden. The existing bounded ASW scan applies to
unspecified resolution; explicit researcher values prevail. Native `tl.diff_test`
compares candidate children with parent-rest, and ICTA-3C evidence decides retention.
There is no LLM planning call inside the iteration.

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

The existing scientific gallery provides composition, branch relationships,
representative-marker dot plots and heatmaps. Values come from accepted Gene
Activity X, with clearly labeled display-only `log1p`. This is accessibility-derived
activity, not measured RNA expression. Marker selection follows accepted positive
native marker order: at most two per displayed branch, twelve distinct genes and
thirty branches. The hierarchy view shows at most sixty branches; omissions are
explicit. No positive evidence means no invented marker plot.

Qualified accepted global UMAP can be colored by final annotation or reached levels
(up to five displayed levels), and by at most six representative markers. The
annotation-level control only selects existing presentation artifacts; it creates
no scientific Run. A compatible UMAP must already exist. No coordinates are
fabricated or fitted by reporting, and a cropped global plot is not a local UMAP.

Downloads & reports exposes the primary hierarchical H5AD, all three evidence
sidecars, source-bound figures and the technical report. Download privacy and
integrity use the shared native delivery owner. Reports record native operations,
parameters, producing identities and interpretation limits without rerunning science.

## Progress, editing and continuation

The existing annotation progress trace supplies current level, branch, native
function, actual completed/running child steps and known pending branches. Future
branch work has no invented fixed total. Cancellation and recovery retain Runtime,
RunStore and accepted-child ownership.

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
production-scale performance. ICTA-4C browser qualification, paid evaluations and
large PBMC/GLUE benchmarking are outside this phase.
