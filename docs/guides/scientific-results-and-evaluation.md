# Scientific results and evaluation

## Accepted native handoff and delivery

The SRV-2C replacement derives a transient cell-result view from accepted RunStore
records, publication contracts, native function identity, actual written slots and
exact dependencies. Native Leiden groups and native 2-D UMAP coordinates can be
reused across turns. Custom supported keys retain their names; independent returns
and inherited fields are not newly computed slots. A loose H5AD with similarly
named fields has no accepted computational authority. Dedicated analysis and
historical figure identities retain their meanings.

The existing renderer can display native Leiden UMAP and later compatible
annotation colors using the same exact coordinates. Source ancestry and ordered
cells must agree. Composition and marker/RP panels use actual annotation evidence;
unassigned labels remain missing categories. Rendering creates neither a synthetic
analysis result nor another scientific Run.

`MatrixDownloadPreparer` owns finite source, destination, decoded-storage,
decoded-chunk, temporary-storage and concurrent-transfer limits. Defaults are
1 GiB source/destination, 2 GiB decoded storage, 16 MiB decoded chunks, 8 GiB
aggregate temporary reservation and two transfers. Operators may configure
`AGENT_DELIVERY_SOURCE_BYTES`, `AGENT_DELIVERY_DELIVERED_BYTES`,
`AGENT_DELIVERY_LOGICAL_BYTES`, `AGENT_DELIVERY_CHUNK_BYTES`,
`AGENT_DELIVERY_TEMPORARY_BYTES` and `AGENT_DELIVERY_TRANSFERS` with finite positive
integers. Checkpoint-specific limits remain separate. Configuration does not alter
scientific admission or annotation capacity.

The same private snapshot, streaming, privacy export and checksum owners serve
native matrices and other supported scientific exports. Both source and delivered
checksums remain explicit; a privacy export may change encoded bytes while
preserving exact scientific values and axes. Capacity is held until the transfer
closes and released on failure. SRV-2C exercised the accepted 1,126,032,034-byte
native H5AD using a 2 GiB source/destination policy, with exact count/axis/coordinate
comparison. This is local engineering evidence, not a new Firefox acceptance.

The [implementation report](../../../agent-data/acceptance/scientific-results/srv-2c/implementation-2026-11-07/REPORT.md)
records the original handoff acceptance. The subsequent
[execution-safety report](../../../agent-data/acceptance/scientific-results/srv-2c/execution-safety-2026-11-07/REPORT.md)
qualifies aggregate Python/R containment and supersedes that earlier blocker.
The [engineering closeout](../../../agent-data/acceptance/scientific-results/srv-2c/closeout-2026-11-08/REPORT.md)
records final review, commit/push and the conditions for a separately authorized
Gate 1 continuation. Containment does not guarantee full-data annotation success.


SRV shifts development from broad tool integration to useful, interpretable,
reproducible research workflows. The Agent paper should evaluate workflow
reliability, researcher effort and scientific utility, while attributing native
algorithms and foundation models to their original authors. EpiZoo, EpiAgent and
other packages have equal architectural status; actual scientific applicability
and available registered operations determine use.

The three long-term research objectives are to:

1. Lower the barrier to single-cell epigenomics analysis through natural language.
2. Automate labor-intensive scientific research workflows.
3. Support new scientific discoveries.

The Agent is the research subject; native packages and foundation models supply
scientifically appropriate methods without privileged architectural status.

## Active roadmap

| Phase | Scope and exit criteria |
| --- | --- |
| SRV-1 — Normal scientific results | Improve necessary visualizations, evidence, readability, general scientific comparisons, reporting, methods and downloads within the Agent product. Exit each bounded increment with exact source/figure binding, focused tests and a genuine Web result review. Figureless results remain valid. |
| SRV-2 — External end-to-end evaluation | The research team evaluates complete ordinary Agent workflows against real published original data, processed matrices and reference annotations. Report scientific agreement, execution outcomes and limitations. No benchmark execution mode or production subsystem; a successful tool call alone does not pass. |
| SRV-3 — Advanced research | Future, demand-driven representative experiments: EpiAgent cCRE knockout, perturbation or mapping; EpiZoo-Cancer variant/LoA, sequence attribution or cross-species regulation. Require native scientific parity before exposing a coherent workflow; no blanket paper-experiment integration. |
| SRV-4 — Independent datasets | Evaluate and support independent compatible datasets beyond original examples, retaining model/assay/reference restrictions. Exclude paper/filename/path/dataset special cases. |
| SRV-5 — Paper evidence | Future consolidation of reproducible results, fair baselines, biological metrics, effort/cost, negative outcomes, limitations and publication figures. Accumulate evidence throughout earlier work. |

SRV-1 and SRV-2 can proceed concurrently. SRV-1A is the first bounded presentation
increment; it does not complete all necessary marker, QC, comparison or advanced
figures. SRV-2A prepares the first practical published-study evaluation;
SRV-3/4/5 implementation is not started. The [SRV-1A closeout](../../../agent-data/acceptance/scientific-results/srv-1a/REPORT.md)
and [SRV-2A specification](../../../agent-data/acceptance/scientific-results/srv-1a/EVALUATION.md)
own actual evidence, readiness and remaining blockers.

## Product comparisons and external evaluation

Normal research conversations may start with FASTQ/BAM/fragments preprocessing or
an existing matrix, continue through supported QC, representations, clustering and
annotation, and modify parameters or reuse accepted results in later turns.
Researchers may then upload a reference H5AD or labels and request a scientifically
compatible comparison, metrics, figures, interpretation and downloads. Such
source-aware comparison operations and their presentation belong to SRV-1.

Current support includes compatible clustering evaluation, legacy label-transfer
confidence evaluation, bounded categorical reads and SRV-1B's general
`compare_cell_annotations`. The latter compares fixed native MAESTRO or supplied
H5AD labels with an independently uploaded H5AD or headered TSV/CSV. It reads only
selected observation fields; different feature axes and row orders are allowed.
H5AD defaults to `obs_names` for cell identifiers; tables require an explicit cell
column. Biological correspondence is never inferred from filenames or bare barcodes.
Selected H5AD fields must contain one value per cell. Tables cannot infer an index
from surplus fields; quoted labels and omitted trailing labels retain their ordinary
CSV meanings, with omitted labels counted as missing.

Select both sample/library identity columns, or explicitly declare that both files
describe the same physical library with `shared_library_id`. Matching uses the
exact `(library, cell identifier)` key; duplicate keys and absence of shared keys
are rejected. Library declarations are not independently verified. There is no
barcode rewriting, inferred label alias correction, optimal class mapping or prediction
update. Source roles and captured H5AD observation/table header names bind fields;
these name offers do not qualify a source scientifically.

The result reports matched/unmatched counts, both matching fractions, missing
reference coverage and the evaluated fraction of the prediction population.
Accuracy, macro-F1 and balanced accuracy include missing predictions as errors on
matched cells with a reference label. Macro-F1 averages the union of observed
biological classes; balanced accuracy averages recall over supported reference
classes. Assigned accuracy uses only jointly labeled cells. Descriptive ARI/NMI
also use jointly labeled cells and are unavailable for noninformative partitions.
Per-class results and full-source composition remain separately downloadable.
Reference annotations are research evidence, not infallible biological truth.

Strict original-label comparison remains the default. When vocabularies differ,
the Agent can explain the actual categories and name differences, then ask for
explicit correspondences. Disagreement between individual cells within the same
vocabulary is not a reason to request a mapping. Researchers can reply naturally:

> Map CD4 Naive T and CD4 Memory T in my predictions to CD4 T, and map CD4 T cell
> in the reference to CD4 T. Leave the other labels unchanged.

The existing operation accepts optional `prediction_mapping` and
`reference_mapping` dictionaries internally; researchers need not write JSON.
Both original and replacement strings are grounded in the user's words. Rules
must name actual categories on the selected side. A missing correspondence uses
the ordinary parameter clarification; ambiguous or conflicting instructions need
clarification. Unknown original categories and invalid rules cannot execute.
Simple renaming and many-to-one merging are supported, with at most 32 rules per
side. Rules apply simultaneously once: targets do not trigger another rule.
Unspecified labels and all cells remain present. Missing labels and the explicit
unassigned/unknown/unlabeled/unlabelled sentinels cannot become biological classes.
Category splitting, synonym inference, ontology matching and agreement-optimized
mapping are not provided.

Recomparison reads the original prediction/reference label datasets through their
existing upload roles or accepted label artifacts, preserving columns and physical
library declarations. The comparison report itself is not a label dataset. No
re-upload is needed when those exact files remain available in the Session.

A mapped comparison is a new accepted publication. Source files, original labels,
the earlier strict comparison and its displayed response remain unchanged. The
comparison JSON includes explicit side-specific rules and their request-parameter
origin, full-source mapped/unmapped categories and coverage, original exact-label
metrics, harmonized metrics and actual cell/class denominators. Bounded evidence
names omitted categories; full vocabularies remain in the JSON. Original label
columns in the aligned TSV retain their meanings; mapped labels and agreement are
additional columns. Existing tables, confusion plots, coordinate overlays and
downloads describe the recorded label policy, with mapped figures identified as
harmonized views. Mapped ARI/NMI describe the merged partitions, not the original
clustering. A researcher-specified equivalence is a comparison convention, not
independent biological validation.

This bounded extension is recorded under
[SRV-2C Phase 0](../../../agent-data/acceptance/scientific-results/srv-2c/phase-0/REPORT.md).
The dated multi-turn addendum and final architecture review qualify genuine
strict-result and reference-side to two-sided many-to-one continuation with the original files. Retained observation columns
keep their exact accepted registered sources when those sources remain offered;
explicit current source selections take precedence, including distinct sources for
repeated operations. The decision is bounded CONDITIONAL GO for controlled use.
A historical strict-result request was refused despite complete inputs; a later
single confirmation with the same wording succeeded. This is not universal
first-attempt reliability. Scientific factual review is unchanged.
It does not start the full published-data SRV-2C evaluation or close SRV-1.

Scientific explanations summarize newly executed accepted results by default;
retained results remain available for explicit historical reading or comparison.
Their facts are kept separately attributed even when values or field definitions
are equal. A mapped publication already records its original metrics, so an
ordinary completion does not need to repeat an entire retained strict result.

The separate [SRV-1/R5 context report](../../../agent-data/acceptance/scientific-results/srv-1-r5-context/REPORT.md)
records removal of the former hard 32 KiB evidence rejection. That size is now a
preferred organization budget: optional automatic detail and unsolicited delivery
descriptions can be omitted with disclosure. Requested detail, figure provenance,
selected export contents and necessary accepted summaries may exceed it. Repeated
field definitions share their existing reviewed meanings; source facts and citations
are not merged. Each target uses the existing 128-field owner allowance, with room
for requested detail, rather than sharing 64 slots across all selected results.
Existing source/fact/view limits remain. The complete prompt, response schema and
16 KiB answer reserve have a 1 MiB operational resource ceiling. This is an Agent
resource policy, not a model token-window estimate. Model capacities and tokenizer
ratios are not invented; actual provider limits use ordinary request/error handling.
Provider rejection, resource-budget refusal and factual-review withholding remain
distinct, while accepted science and saved historical responses are preserved.

An optional accepted UMAP supplies the same coordinates for prediction, reference
and agreement panels. Missing/unmatched reference coverage is gray. Exact
original-source ancestry and ordered prediction cells are required. For native
MAESTRO predictions, bind its accepted annotation publication as well as its H5AD
to authenticate that relationship; copied `uns` metadata alone is insufficient.
The reference needs neither an embedding nor the same features. No UMAP is refit
by comparison or rendering. Without coordinates, comparison still produces
confusion/composition figures; overlay metadata states that coordinates are absent.

The existing Registry, native publication, scientific authority, evidence, figure,
privacy and download owners deliver these results. Tables are limited to 64 MiB,
selected H5AD observations to 128 MiB, sources to 200,000 cells and the full label
union to 256 classes. Preview tables/legends/confusion are bounded with explicit
omission/coverage; complete comparison files retain their labels and counts.
Legacy evaluation/confidence artifacts keep their original contracts.

SRV-2 is organized outside the Agent: select a published study with original
sequencing inputs and a corresponding processed peak matrix with published
cell-type annotations, run the ordinary Agent without reference answers or a
benchmark prompt, preserve its outputs, then evaluate them externally. Published
annotations are reference annotations, not infallible truth. No Multiome,
RNA-only label set, compulsory native-script baseline or dedicated blinded
infrastructure is required for the first experiment.

Keep reference labels out of the initial analysis and parameter selection. A later
reference upload is a separate normal comparison request, never permission to
retroactively change the preserved predictions. External scripts may use the same
valid metric computations without creating another accepted Agent result or
sharing evaluation authority. SRV-2 adds no Planner, Registry, Executor, workflow
template, storage or scientific-result schema. More elaborate baselines,
generalization and efficiency studies remain optional later SRV-5 work.

## Result experience

The existing Web result selector shows the exact saved response, applicable
scientific limitations, published figures, concise operation summaries and
quantitative tables. Annotation tables expose native group names, cell counts,
candidate labels and assignment status. Literature corroboration shows native
scores and recorded assessment categories; expanded methods retain marker records
and mapping rationale. These scores are not probabilities or independent proof
of cell identity. Group omission counters remain visible. Full scientific files
remain downloadable even when a preview is bounded.

The gallery displays up to six distinct, source-bound figures from the selected
result or requested presentation. Further figures remain in Downloads & reports.
Showing multiple figures does not select a new scientific result. Duplicate report
copies use their manifest and figure identities. Historical previews and requested
output choices retain their existing associations. An unavailable PNG is reported
as a preview failure; it does not trigger computation. Full-size figure inspection,
provenance and downloads use the existing artifact endpoints.

The presentation does not add missing scientific steps. An accepted UMAP is needed
for a UMAP figure. Recoloring requires an exact qualified cell correspondence.
Missing UMAP, gene activity, QC observations or compatible reference labels cannot
become figures or accuracy estimates by requesting a report. MAESTRO RP and ArchR gene
scores are accessibility-derived activity estimates, not measured expression;
marker evidence used in annotation is dependent evidence. Existing native motif,
peak and sequence figures retain their recorded meanings.

Methods, effective parameters, full accepted facts, technical reports, source
identities and downloads remain accessible in disclosures. No stored report,
scientific artifact, historical displayed response or acceptance identity is
rewritten by the presentation. Download requests are not local delivery receipts.

## Annotation figures and remaining coverage

New native annotation result displays include group/cell-type composition and,
when positive native marker evidence exists, an RP dot plot and per-cell activity
distributions. These use the pinned sparse per-cell MAESTRO RP matrix, exact genes,
cells, groups and candidate/marker sidecars. No annotation, RP fit or marker test is
rerun. Selection prefers positive markers supporting the native assigned candidate,
then positive native group markers in published order: up to two per group and
twelve distinct genes. Dot color is `log1p(mean raw RP)`; size is fraction `RP > 0`.
Distributions show per-cell `log1p(raw RP)` for up to four selected genes and twenty
groups. Dot previews retain up to forty recorded groups. Limits are explicit.
Raw RP is sensitive to library depth; it is accessibility-derived activity, not
measured RNA expression. These markers are dependent annotation evidence.

A source-compatible accepted UMAP in the same Run also receives native cell-type
colors automatically. Explicit later accepted-result overlays retain the existing
Interaction composer. A missing UMAP never causes hidden fitting. New annotation
figure projections have an explicit version marker; saved historical bundles keep
their original figure sets and displayed prose.

Key annotation/comparison counts and metrics precede the gallery; supporting
composition/per-class tables and methods follow it. Downloads remain in their
existing disclosure. The [SRV-1B report](../../../agent-data/acceptance/scientific-results/srv-1b/REPORT.md)
separates native data/UMAP evidence, simulated references, focused tests and official
OpenAI Firefox acceptance. Generated explanatory prose remains bounded: the primary
comparison's explanation was withheld by the existing semantic review, while its
accepted tables, figures and downloads remained available. No paid prose retry was
made. Production's official OpenAI default is unchanged.

SRV-1 remains open. Barcode QC already publishes length histograms and per-barcode
metrics, but automatic QC distributions/scatter and selection-retention figures
remain gaps. Replicate DA retains native statistics and exports but lacks dedicated
volcano/MA and replicate diagnostics. Native chromVAR accessibility patterns and
MACS3 peak distributions already exist; motif-hit context, sequence logos, genomic
peak context and cross-result motif/peak figures remain separate work. Sequence
prediction/design/generation retain their existing model-specific score, edit and
composition figures; attribution, cross-model calibration and richer observed-
measurement diagnostics are not added here. Native package computational results
do not automatically expose every upstream plotting API. No external evaluation
or advanced EpiAgent/EpiZoo experiment is started by SRV-1B.

## Owners and scientific principles

Reuse Registry planning, the Executor, scientific owners, ScientificResourceLibrary,
Session/RunStore/Revision acceptance, R5 evidence/review and Web result/artifact
owners. Scientific task requests and explicit native-function requests remain one
Agent. Presentation composes accepted evidence after execution; it grants no new
scientific authority and never reruns algorithms to improve appearance.

Use native algorithms, actual measurement units, exact cells/features/references
and effective parameters. Never silently substitute methods, infer missing
processing history or fabricate confidence, significance or biological conclusions.
Keep sparse matrices sparse. Preserve accepted history and protected data. Remove
redundant presentation restrictions within the affected boundary, without adding a
parallel report engine, result catalog, storage system or speculative framework.

Each increment needs bounded completion criteria, focused verification and clearly
classified genuine, simulated and historical evidence. External evaluation
labels stay out of the initial planning, annotation and parameter selection;
preserve the completed results before introducing a comparison reference.
Missing prerequisites and failed or inconclusive analyses remain part of the
evaluation record.
