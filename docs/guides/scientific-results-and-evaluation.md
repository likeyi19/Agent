# Scientific results and evaluation

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

Current support includes compatible clustering evaluation, label-transfer
annotation evaluation, and bounded categorical counts/contingency reads. The
annotation evaluator does not currently consume native MAESTRO publications or
publish balanced accuracy. Arbitrary uploaded-H5AD comparisons are not universally
implemented. Valid matched-cell bindings and compatible measurements remain
necessary. Unsupported comparisons must be explained; future extensions should
be general scientific capabilities, not paper-specific benchmark implementations.

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
