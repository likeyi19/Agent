# Cell-type annotation and reference evidence

Researchers can upload scATAC H5AD and request cell-type annotation, optionally
with database corroboration, through the ordinary conversation. MAESTRO produces
candidate annotations; CellMarker supplies separate literature associations.
Neither native scores nor reference matches are calibrated biological confidence.

## Inputs and native method

`annotate_scATAC_cell_types` reuses pinned MAESTRO Enhanced regulatory potential
(RP), native group-marker calculation and signature scoring. Its direct region
method `maestro-enhanced-region-counts.v2` accepts explicitly identified raw peak
counts, fragment-overlap counts and binary accessibility for human/hg38 or
mouse/mm10 with compatible transcript/exon and signature resources. The original
human Cell Ranger cut-site method and accepted canonical-matrix method retain
their identities and historical readers. This extension does not reinterpret old
publications or binarize supplied fragment counts.

The [MAESTRO implementation](https://raw.githubusercontent.com/liulab-dfci/MAESTRO/74f10babf4125b98ef005df4186b14ab02e13bfd/MAESTRO/scATAC_Genescore.py)
weights region measurements by transcript/exon and distance relationships.
[MAESTRO documentation](https://liulab-dfci.github.io/MAESTRO/) describes raw and
binary peak input. Numerical admissibility alone does not establish the best
biological representation, reference relevance or cell identity.

Required scientific information is the measurement history, species/assembly,
exact zero-based `chrom:start-end` feature identities, compatible references, and
independent cell groups. Groups may come from an explicitly selected `obs` column,
exact ordered TSV, or an accepted PCA-derived Leiden result whose recorded ancestry
identifies the exact original source bytes. Names, equal cell counts and categorical
values do not establish lineage or independent clustering. Explicit `uns.grouping`
role/provenance metadata can qualify a unique grouping automatically.

The default count slot is `X`. `layers/name` selects an original measurement layer
without changing transformed `X`. New source observations scope global measurement
metadata to `uns.count_slot` (default `X`); another selected layer needs its own
explicit measurement declaration. Historical observations remain readable with
their original interpretation. No integer-valued matrix is automatically labelled
raw counts. `raw.X` with an independent feature axis and standalone gene-activity
matrices do not have a direct annotation execution route in this integration;
provide a suitable original region-count slot and exact coordinates. No automatic
feature mapping, alias correction, normalization reversal or classifier is added.

Pending conversational measurement declarations may remain unknown while Agent
asks for clarification. A resolved execution specification must declare a supported
measurement before input preparation; nullable resource metadata alone does not
authorize scientific execution. The [owner-contract closeout](../../../agent-data/acceptance/scientific-workflow-completion/swc-2/finalization/owner-contract-closeout/REPORT.md)
records this admission correction and its focused verification.

Storage support includes canonical CSR/CSC and bounded dense arrays (at most
512 MiB converted to sparse). Source H5AD is limited to 1 GiB encoded/2 GiB decoded,
100,000 cells, 2,000,000 unique features and 100,000,000 nonzeros. Browser upload
limits are separately operator-configurable. Complete sparse matrices are never
densified. Native zero-weight and unmatched-contig participation is reported.

## Optional CellMarker corroboration

`corroborate_cell_type_annotation` is the 26th registered tool. It consumes an exact
accepted annotation manifest and checksum, reading native groups, genes and marker
sidecars through existing authority. It never reruns MAESTRO, rewrites candidate
labels, or changes annotation acceptance. Annotation-only requests remain separate.
Combined requests can connect the annotation output directly to corroboration;
later requests select the accepted annotation through ordinary planning context.

The tool uses a pinned local CellMarker snapshot, not a live online query. Its
existing comparison owner checks exact species/tissue/cancer context and
unambiguous gene symbols. Signature-bound, explicitly curated terminology links
record reference terms, broad/equivalent/narrower or competing relationships and
mapping rationale. Otherwise only exact case/whitespace-normalized term equality
is used; there is no fuzzy alias or cross-species mapping. Unmapped terms remain
unmapped. The SWC-2 deployed mapping covers the entire pinned 28-category immune
signature vocabulary, not just the six observed PBMC predictions; it is an explicit
curated resource mapping, not an independently expert-validated ontology.

- `corroborating`: positive markers associated with a mapped equivalent/broader term.
- `mixed`: positive nonexclusive associations with a mapped competing lineage;
  this does not establish a contradiction or exclusive lineage membership.
- `insufficient_evidence`: no positive equivalent/broader mapped association.
- `unmapped`: no qualified terminology correspondence.
- `not_applicable`: the native group has no assigned candidate.

Broad CD8-lineage associations do not prove exhaustion. RNA/protein literature
markers do not directly validate scATAC accessibility or provide matched ground
truth. No evidence is not disproof. The exact native score remains a native score.

The accepted publication includes per-group summaries and complete source-bound
JSON/TSV evidence with source row, gene/effect, term, mapping, tissue, PMID, title,
release and table checksum. Normal result views offer the annotated original H5AD
and reference TSV. Privacy checks belong to delivery; source artifacts remain
unchanged. Whole-result answers use bounded summary examples; group-specific
questions can read published rows. Omission counts do not imply evidence absence.

## Operator deployment

Install a `marker_reference` in the existing `ScientificResourceLibrary`, using
schema `cell-marker-reference.v1`, a pinned `KnowledgeResource`, exact context and
signature-bound alignment. Existing owners check source/table closure and all
mapped terms. The tool can also receive an explicit reference specification path
and SHA-256. An omitted specification resolves only a unique matching library
entry; missing or ambiguous references fail closed with an actionable error.
There is no automatic online resource acquisition or typed upload continuation
for missing CellMarker resources yet.

The Web operator's `scientific_resource_root` is also the scientific execution
library. It is scoped to each operation, so separate deployments do not share a
process-global reference selection. Standalone tool/CLI callers may continue to
set `AGENT_SCIENTIFIC_RESOURCE_ROOT`. The qualified SWC-2 deployment uses
`resources/library-swc2`; production defaults are unchanged. Existing
`AGENT_ANNOTATION_MAESTRO_ROOT`, `AGENT_ANNOTATION_RSCRIPT` and
`AGENT_ANNOTATION_R_LIBRARY` native runtime requirements remain.

CellMarker 2.0 snapshot dates, hashes, use restrictions and the 28-label mapping
are recorded in the [SWC-2 evidence](../../../agent-data/acceptance/scientific-workflow-completion/swc-2/REPORT.md).
No database redistribution license is inferred from public availability; downloads
contain selected evidence records, not the complete database. Human PBMC is the
genuine deployed/tested reference context; other species/tissues require their
own qualified resources and are not biologically qualified by this acceptance.

## Completing grouping prerequisites

A researcher may request annotation and reference evidence without naming tools,
parameters, grouping columns or reference IDs. New source observations (interface
version 4) distinguish a missing computed grouping from missing biological
information. Species, assembly, actual measurement history and tissue context
still require source evidence or a meaningful user declaration. Older captured
observations and displayed clarifications retain their exact historical meaning.

A unique explicitly qualified supplied grouping is reused. An explicitly selected
accepted Leiden result retains the existing exact-source TSV projection. When no
qualified grouping exists, the ordinary Planner can compose existing inspection,
representation, neighbor and clustering operations and bind the clustered artifact
directly to annotation. This is not a fixed annotation pipeline. A request to use
existing groups only is preserved as `grouping_strategy=reuse_only` and does not
authorize computing replacements. Multiple qualified groupings still require an
unambiguous selection. UMAP is not required to annotate groups.

The Registry now permits the Planner to select the documented PCA preprocessing
vocabulary when the user has not supplied a choice. Declared raw region counts or
binary accessibility can use the existing `tfidf_l2` route; explicitly supplied
choices take precedence and other settings retain native defaults. This is a
recorded method selection, not an inferred user instruction or a new global PCA
default. TF-IDF/PCA remains exploratory and is not Signac LSI or biological proof.

The annotation owner accepts exact clustered-analysis path/checksum ports, verifies
its PCA original-file identity and exact ordered cells, and retains the clustering
artifact in the annotation publication's integrity closure. Original measurements
remain in the original count slot. Legacy embedding results without exact source
identity cannot establish this correspondence. The existing annotation publication
and native completion verification remain authoritative; no new grouping tool,
workflow engine or scientific result catalog is added.

The [SWC-2 finalization report](../../../agent-data/acceptance/scientific-workflow-completion/swc-2/FINALIZATION-REPORT.md)
records genuine Firefox/OpenRouter processing of an unclustered 569-cell human
PBMC fragment-count H5AD: one native six-operation analysis, 13 groups, 553 assigned
cells, 16 unassigned cells and 233 CellMarker records. Actual downloads preserve
counts and axes; later evidence answers reuse acceptance without scientific reruns.
Development-time planning and presentation failures remain recorded, so this is
bounded workflow qualification, not a guarantee of first-attempt provider success.
